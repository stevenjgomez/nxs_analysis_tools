import os
import pytest
import numpy as np
from nexusformat.nexus import NXdata, NXfield
import matplotlib
matplotlib.use('Agg')
import builtins
builtins.display = lambda *args, **kwargs: None

from nxs_analysis_tools.pairdistribution import (
    Symmetrizer,
    symmetrizer,
    _rotate_plane_affine,
)


def make_synthetic_nxdata(shape, extents, names=('h', 'k', 'l'), signal_name='counts'):
    """
    Helper to construct synthetic 2D or 3D NXdata with specified shapes and physical extents.
    """
    ndim = len(shape)
    axes = []
    coords = []
    for i in range(ndim):
        n = shape[i]
        low, high = extents[i]
        c = np.linspace(low, high, n)
        coords.append(c)
        axes.append(NXfield(c, name=names[i]))
    
    # Meshgrid to easily place peaks or signals
    arr = np.zeros(shape, dtype=float)
    return NXdata(NXfield(arr, name=signal_name), tuple(axes))


def add_gaussian_peak(nxdata, center, sigma, intensity=100.0):
    """Add a Gaussian peak at physical coordinates `center`."""
    arr = nxdata.nxsignal.nxdata
    coords = [ax.nxdata for ax in nxdata.nxaxes]
    if nxdata.ndim == 2:
        Q0, Q1 = np.meshgrid(coords[0], coords[1], indexing='ij')
        dist_sq = (Q0 - center[0])**2 + (Q1 - center[1])**2
        peak = intensity * np.exp(-dist_sq / (2 * sigma**2))
        nxdata.nxsignal.nxdata = arr + peak
    elif nxdata.ndim == 3:
        Q0, Q1, Q2 = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
        dist_sq = (Q0 - center[0])**2 + (Q1 - center[1])**2 + (Q2 - center[2])**2
        peak = intensity * np.exp(-dist_sq / (2 * sigma**2))
        nxdata.nxsignal.nxdata = arr + peak


class TestSymmetrizerAffine:
    """Test the single-pass affine rotation engine directly."""

    def test_rotate_plane_affine_hexagonal_60deg(self):
        # Hexagonal grid: peak at (1.0, 0.0) rotates by 60 deg to (0.0, 1.0)
        nxdata = make_synthetic_nxdata((61, 61), ((-3.0, 3.0), (-3.0, 3.0)), names=('h', 'k'))
        add_gaussian_peak(nxdata, (1.0, 0.0), sigma=0.15, intensity=100.0)

        q1 = nxdata.nxaxes[0].nxdata
        q2 = nxdata.nxaxes[1].nxdata
        dq1 = (q1[-1] - q1[0]) / (len(q1) - 1)
        dq2 = (q2[-1] - q2[0]) / (len(q2) - 1)
        c = np.array([-q1[0] / dq1, -q2[0] / dq2])

        rot = _rotate_plane_affine(
            nxdata.nxsignal.nxdata,
            lattice_angle=60.0,
            rotation_angle=60.0,
            dq1=dq1,
            dq2=dq2,
            origin=c,
            aspect=1.0,
            order=3
        )

        Q0, Q1 = np.meshgrid(q1, q2, indexing='ij')
        total = rot.sum()
        assert total > 0
        h_cent = (Q0 * rot).sum() / total
        k_cent = (Q1 * rot).sum() / total

        assert np.isclose(h_cent, 0.0, atol=0.03)
        assert np.isclose(k_cent, 1.0, atol=0.03)


class TestSymmetrizerPermutations:
    """
    Test suite covering the 6 primary test cases from the architectural plan:
    - Equal vs unequal shapes
    - Symmetric vs asymmetric coordinate bounds
    - Equal vs unequal step sizes
    - Hexagonal, Tetragonal, Trigonal, Orthorhombic symmetries
    - Different layer axes (L, Qh, Qk)
    """

    def test_case1_standard_isotropic_hexagonal(self):
        # Shape: (51, 51, 41), Extents: [-2, 2], [-2, 2], [-2, 2], Axis: 2 (l)
        data = make_synthetic_nxdata((51, 51, 41), ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)), names=('h', 'k', 'l'))
        # Put peak at (h=1.0, k=0.0, l=1.0)
        add_gaussian_peak(data, (1.0, 0.0, 1.0), sigma=0.12, intensity=100.0)

        sym = Symmetrizer(data, symmetry='hexagonal', layer_axis=2)
        res = sym.symmetrize(method='average', parallel=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()

        # In hexagonal 6-fold with +L / -L pairing:
        # A peak at (1, 0, 1) should exist at (1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)
        # on both l = +1.0 and l = -1.0 slices!
        l_coords = res.nxaxes[2].nxdata
        idx_pos_l = np.argmin(np.abs(l_coords - 1.0))
        idx_neg_l = np.argmin(np.abs(l_coords - (-1.0)))

        slice_pos = res.nxsignal.nxdata[:, :, idx_pos_l]
        slice_neg = res.nxsignal.nxdata[:, :, idx_neg_l]

        # Slices at +l and -l should be non-zero and identical
        assert np.allclose(slice_pos, slice_neg, atol=1e-5)
        assert slice_pos.max() > 5.0

    def test_case2_unequal_rectangular_tetragonal_axis0(self):
        # Shape: (31, 51, 41), Extents: [-2, 2], [-2, 2], [-2, 2], Stacking axis: 0 (Ql)
        data = make_synthetic_nxdata((31, 51, 41), ((-1.5, 1.5), (-2.0, 2.0), (-2.0, 2.0)), names=('ql', 'qh', 'qk'))
        # Put peak at (ql=0.5, qh=1.0, qk=0.0)
        add_gaussian_peak(data, (0.5, 1.0, 0.0), sigma=0.12, intensity=80.0)

        sym = Symmetrizer(data, symmetry='tetragonal', layer_axis=0)
        res = sym.symmetrize(method='average', parallel=False)

        assert res.shape == data.shape
        # Check that 4-fold in-plane symmetry appeared on qh, qk
        ql_coords = res.nxaxes[0].nxdata
        idx_ql = np.argmin(np.abs(ql_coords - 0.5))
        s = res.nxsignal.nxdata[idx_ql, :, :]

        # Peak was at (1, 0), so it should also appear at (0, 1), (-1, 0), (0, -1)
        qh_coords = res.nxaxes[1].nxdata
        qk_coords = res.nxaxes[2].nxdata
        val_1_0 = s[np.argmin(np.abs(qh_coords - 1.0)), np.argmin(np.abs(qk_coords - 0.0))]
        val_0_1 = s[np.argmin(np.abs(qh_coords - 0.0)), np.argmin(np.abs(qk_coords - 1.0))]
        assert val_1_0 > 5.0
        assert val_0_1 > 5.0

    def test_case3_asymmetric_coordinate_bounds(self):
        # Equal shapes, but coordinate extents not centered around 0
        # Extents: h in [-1, 2], k in [-1, 3], l in [-2, 2]
        data = make_synthetic_nxdata((45, 55, 35), ((-1.0, 2.0), (-1.0, 3.0), (-2.0, 2.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 0.0, 1.0), sigma=0.12, intensity=90.0)

        sym = Symmetrizer(data, symmetry='hexagonal', layer_axis='l')
        res = sym.symmetrize(method='average', parallel=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()

    def test_case4_unequal_shape_asymmetric_extents_orthorhombic(self):
        # Shape: (35, 45, 31), Extents: [-1.5, 2.5], [-2.0, 2.0], [-1.0, 2.0]
        # Layer axis: 1 ('k')
        data = make_synthetic_nxdata((35, 45, 31), ((-1.5, 2.5), (-2.0, 2.0), (-1.0, 2.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 1.0, 0.0), sigma=0.15, intensity=100.0)

        sym = Symmetrizer(data, symmetry='orthorhombic', layer_axis='k')
        res = sym.symmetrize(method='average', parallel=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()

    def test_case5_trigonal_3fold(self):
        # Trigonal (3-fold, 120 deg skew)
        data = make_synthetic_nxdata((41, 41, 25), ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 0.0, 0.5), sigma=0.12, intensity=100.0)

        sym = Symmetrizer(data, symmetry='trigonal', layer_axis=2)
        res = sym.symmetrize(method='average', parallel=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()

    def test_case6_missing_negative_layers(self):
        # L in [0.0, 3.0] with no negative counterpart layers
        data = make_synthetic_nxdata((41, 41, 21), ((-2.0, 2.0), (-2.0, 2.0), (0.0, 3.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 0.0, 1.5), sigma=0.12, intensity=100.0)

        sym = Symmetrizer(data, symmetry='hexagonal', layer_axis=2)
        res = sym.symmetrize(method='average', parallel=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()
        # Verify L=1.5 slice was symmetrized despite having no -1.5 partner
        l_coords = res.nxaxes[2].nxdata
        idx_l = np.argmin(np.abs(l_coords - 1.5))
        assert res.nxsignal.nxdata[:, :, idx_l].max() > 5.0


class TestSymmetrizerFeatures:
    """Test 2D slicing, parallel execution equivalence, subclass compatibility, and warnings."""

    def test_2d_symmetrizer(self):
        data_2d = make_synthetic_nxdata((45, 45), ((-2.0, 2.0), (-2.0, 2.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (1.0, 0.0), sigma=0.12, intensity=100.0)

        sym = Symmetrizer(data_2d, symmetry='tetragonal')
        res = sym.symmetrize(method='average')

        assert res.shape == (45, 45)
        h = res.nxaxes[0].nxdata
        k = res.nxaxes[1].nxdata
        # 4-fold: peak at (1, 0) should also be at (0, 1), (-1, 0), (0, -1)
        assert res.nxsignal.nxdata[np.argmin(np.abs(h - 0.0)), np.argmin(np.abs(k - 1.0))] > 5.0

    def test_symmetrize_slice_method(self):
        data = make_synthetic_nxdata((35, 35, 25), ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 0.0, 0.5), sigma=0.12, intensity=100.0)

        sym = Symmetrizer(data, symmetry='hexagonal', layer_axis='l')
        slice_res = sym.symmetrize_slice(0.5, method='average')

        assert slice_res.ndim == 2
        assert slice_res.shape == (35, 35)

    def test_parallel_vs_serial_identity(self):
        data = make_synthetic_nxdata((31, 31, 15), ((-1.5, 1.5), (-1.5, 1.5), (-1.0, 1.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (0.8, 0.0, 0.5), sigma=0.15, intensity=100.0)

        sym = Symmetrizer(data, symmetry='hexagonal', layer_axis=2)
        res_serial = sym.symmetrize(method='average', parallel=False)
        res_parallel = sym.symmetrize(method='average', parallel=True, num_workers=2)

        assert np.allclose(res_serial.nxsignal.nxdata, res_parallel.nxsignal.nxdata, atol=1e-10)

    def test_default_method_is_average(self):
        data_2d = make_synthetic_nxdata((25, 25), ((-1.0, 1.0), (-1.0, 1.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (0.5, 0.5), sigma=0.2, intensity=20.0)
        sym = Symmetrizer(data_2d, symmetry='tetragonal')
        res = sym.symmetrize()
        res_average = sym.symmetrize(method='average')
        assert np.allclose(res.nxsignal.nxdata, res_average.nxsignal.nxdata)

    def test_subclasses_removed(self):
        """Verify Symmetrizer2D and Symmetrizer3D are removed in v0.2.0."""
        import nxs_analysis_tools.pairdistribution as pd
        assert not hasattr(pd, "Symmetrizer2D")
        assert not hasattr(pd, "Symmetrizer3D")

    def test_positive_values_warning_and_override(self):
        # Create dataset with negative values
        data_2d = make_synthetic_nxdata((25, 25), ((-1.0, 1.0), (-1.0, 1.0)), names=('h', 'k'))
        data_2d.nxsignal.nxdata[:, :] = -5.0
        add_gaussian_peak(data_2d, (0.5, 0.5), sigma=0.2, intensity=20.0)

        sym = Symmetrizer(data_2d, symmetry='tetragonal')

        # positive_values=True (default) should warn and clip
        with pytest.warns(UserWarning, match="Negative values found in symmetrized dataset and clipped to zero"):
            res_clipped = sym.symmetrize(method='average', positive_values=True)
        assert (res_clipped.nxsignal.nxdata >= 0.0).all()

        # positive_values=False should not warn and should preserve negative values
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res_raw = sym.symmetrize(method='average', positive_values=False)
        assert (res_raw.nxsignal.nxdata < 0.0).any()

    def test_wedge_rotations_and_reconstruction(self):
        # 45-degree wedge with mirror=True -> rotations must be 4
        s_wedge = Symmetrizer(theta_min=0, theta_max=45, mirror=True, mirror_axis=0)
        assert s_wedge.rotations == 4

        # 45-degree wedge with mirror=False -> rotations must be 8
        s_nomirror = Symmetrizer(theta_min=0, theta_max=45, mirror=False)
        assert s_nomirror.rotations == 8

        # 90-degree wedge with mirror=False -> rotations must be 4
        s_90 = Symmetrizer(theta_min=45, theta_max=135, mirror=False)
        assert s_90.rotations == 4

        # Symmetrize synthetic 4-fold data
        data_2d = make_synthetic_nxdata((45, 45), ((-2.0, 2.0), (-2.0, 2.0)), names=('h', 'k'))
        data_2d.nxsignal.nxdata[:, :] = 1.0
        add_gaussian_peak(data_2d, (1.0, 0.0), sigma=0.15, intensity=50.0)
        add_gaussian_peak(data_2d, (0.0, 1.0), sigma=0.15, intensity=50.0)

        res = s_wedge.symmetrize_2d(data_2d, method='wedge')
        assert res.shape == (45, 45)
        # Symmetrized result must be fully reconstructed without zero-gap sectors
        assert (res.nxsignal.nxdata > 0).all()

        # Test method on 2D wedge should return (2, 2) subplots
        fig, axesarr = s_wedge.test(data_2d)
        assert axesarr.shape == (2, 2)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSymmetrizerAutoExpand:
    """Test auto_expand option to automatically extend coordinate bounds."""

    def test_auto_expand_2d_hexagonal(self):
        # Hexagonal symmetry: peak at (1.5, 1.5) on grid [-2.0, 2.0] x [-2.0, 2.0]
        # Under 6-fold rotation, (1.5, 1.5) rotates to (-3.0, 1.5)
        # Without auto_expand, corners or outer rotated points get cut off.
        data_2d = make_synthetic_nxdata((41, 41), ((-2.0, 2.0), (-2.0, 2.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (1.5, 1.5), sigma=0.15, intensity=100.0)

        sym = Symmetrizer(data_2d, symmetry='hexagonal')
        res_no_expand = sym.symmetrize_2d(auto_expand=False)
        assert res_no_expand.shape == (41, 41)
        assert np.isclose(float(res_no_expand.nxaxes[0][0]), -2.0)
        assert np.isclose(float(res_no_expand.nxaxes[0][-1]), 2.0)

        res_expand = sym.symmetrize_2d(auto_expand=True)
        # Bounds should extend to [-4.0, 4.0]
        assert res_expand.shape == (81, 81)
        assert np.isclose(float(res_expand.nxaxes[0][0]), -4.0)
        assert np.isclose(float(res_expand.nxaxes[0][-1]), 4.0)
        assert np.isclose(float(res_expand.nxaxes[1][0]), -4.0)
        assert np.isclose(float(res_expand.nxaxes[1][-1]), 4.0)
        # Signal sum in expanded result should be strictly greater than unexpanded
        assert res_expand.nxsignal.nxdata.sum() > res_no_expand.nxsignal.nxdata.sum()

    def test_auto_expand_2d_rectangular_tetragonal(self):
        # Rectangular grid: h in [-1.0, 1.0], k in [-3.0, 3.0]
        # Under 4-fold, h must extend to [-3.0, 3.0]
        data_2d = make_synthetic_nxdata((21, 61), ((-1.0, 1.0), (-3.0, 3.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (0.0, 2.5), sigma=0.15, intensity=80.0)

        sym = Symmetrizer(data_2d, symmetry='tetragonal', auto_expand=True)
        res = sym.symmetrize_2d()
        assert np.isclose(float(res.nxaxes[0][0]), -3.0)
        assert np.isclose(float(res.nxaxes[0][-1]), 3.0)
        assert np.isclose(float(res.nxaxes[1][0]), -3.0)
        assert np.isclose(float(res.nxaxes[1][-1]), 3.0)
        # Check 4-fold peak rotated from (0, 2.5) to (2.5, 0) and (-2.5, 0)
        h_vals = res.nxaxes[0].nxdata
        k_vals = res.nxaxes[1].nxdata
        val_pos = res.nxsignal.nxdata[np.argmin(np.abs(h_vals - 2.5)), np.argmin(np.abs(k_vals - 0.0))]
        val_neg = res.nxsignal.nxdata[np.argmin(np.abs(h_vals - (-2.5))), np.argmin(np.abs(k_vals - 0.0))]
        assert val_pos > 5.0
        assert val_neg > 5.0

    def test_auto_expand_raw_numpy_array(self):
        arr = np.zeros((21, 21), dtype=float)
        arr[15, 15] = 100.0
        sym = Symmetrizer(symmetry='hexagonal', auto_expand=True)
        res = sym.symmetrize_2d(arr)
        assert isinstance(res, np.ndarray)
        assert res.shape == (41, 41)

    def test_auto_expand_alias_kwargs(self):
        data_2d = make_synthetic_nxdata((21, 21), ((-1.0, 1.0), (-1.0, 1.0)), names=('h', 'k'))
        sym = Symmetrizer(data_2d, symmetry='hexagonal')
        res_alias = sym.symmetrize(extend_bounds=True)
        assert res_alias.shape == (41, 41)

    def test_functional_symmetrizer(self):
        data_2d = make_synthetic_nxdata((21, 21), ((-1.0, 1.0), (-1.0, 1.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (0.5, 0.0), sigma=0.1, intensity=50.0)
        res = symmetrizer(data_2d, symmetry='tetragonal', auto_expand=True)
        assert res.shape == (21, 21)

    def test_auto_expand_3d_average(self):
        data_3d = make_synthetic_nxdata((11, 21, 21), ((-0.5, 0.5), (-1.0, 1.0), (-1.0, 1.0)), names=('l', 'h', 'k'))
        add_gaussian_peak(data_3d, (0.2, 0.8, 0.8), sigma=0.1, intensity=50.0)
        sym = Symmetrizer(data_3d, symmetry='hexagonal', layer_axis='l', auto_expand=True)
        res = sym.symmetrize(method='average')
        assert res.shape == (11, 41, 41)
        assert np.isclose(float(res.nxaxes[1][0]), -2.0)
        assert np.isclose(float(res.nxaxes[1][-1]), 2.0)
        assert np.isclose(float(res.nxaxes[2][0]), -2.0)
        assert np.isclose(float(res.nxaxes[2][-1]), 2.0)

    def test_auto_expand_on_test_slice_nxs(self):
        path = "scratch/auto_expand_symmetrizer/test_slice.nxs"
        if not os.path.exists(path):
            pytest.skip("scratch/auto_expand_symmetrizer/test_slice.nxs not present")
        import nexusformat.nexus as nx
        nxdata = nx.nxload(path)
        data = nxdata.entry.data

        sym = Symmetrizer(data, symmetry='hexagonal', layer_axis='Ql')
        sl_unexpanded = sym.symmetrize_slice(coord=0.0, auto_expand=False)
        sl_expanded = sym.symmetrize_slice(coord=0.0, auto_expand=True)

        assert sl_unexpanded.shape == (1101, 1101)
        assert sl_expanded.shape == (2201, 2201)
        assert np.isclose(float(sl_expanded.nxaxes[0][0]), -22.0)
        assert np.isclose(float(sl_expanded.nxaxes[0][-1]), 22.0)
        assert np.isclose(float(sl_expanded.nxaxes[1][0]), -22.0)
        assert np.isclose(float(sl_expanded.nxaxes[1][-1]), 22.0)
        assert sl_expanded.nxsignal.nxdata.sum() > sl_unexpanded.nxsignal.nxdata.sum()
