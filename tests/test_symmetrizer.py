import os
import pytest
import numpy as np
from nexusformat.nexus import NXdata, NXfield
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import builtins
builtins.display = lambda *args, **kwargs: None

from nxs_analysis_tools import symmetrize as top_level_symmetrize
from nxs_analysis_tools.pairdistribution import (
    symmetrize,
    _rotate_plane_affine,
    _resolve_symmetry_params,
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


class TestSymmetrizePermutations:
    """
    Test suite covering the 6 primary test cases:
    - Equal vs unequal shapes
    - Symmetric vs asymmetric coordinate bounds
    - Equal vs unequal step sizes
    - Hexagonal, Tetragonal, Trigonal, Orthorhombic symmetries
    - Different layer axes (L, Qh, Qk)
    """

    def test_case1_standard_isotropic_hexagonal(self):
        data = make_synthetic_nxdata((51, 51, 41), ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 0.0, 1.0), sigma=0.12, intensity=100.0)

        res = symmetrize(data, symmetry='hexagonal', layer_axis=2, method='average', parallel=False, auto_expand=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()

        l_coords = res.nxaxes[2].nxdata
        idx_pos_l = np.argmin(np.abs(l_coords - 1.0))
        idx_neg_l = np.argmin(np.abs(l_coords - (-1.0)))

        slice_pos = res.nxsignal.nxdata[:, :, idx_pos_l]
        slice_neg = res.nxsignal.nxdata[:, :, idx_neg_l]

        assert np.allclose(slice_pos, slice_neg, atol=1e-5)
        assert slice_pos.max() > 5.0

    def test_case2_unequal_rectangular_tetragonal_axis0(self):
        data = make_synthetic_nxdata((31, 51, 41), ((-1.5, 1.5), (-2.0, 2.0), (-2.0, 2.0)), names=('ql', 'qh', 'qk'))
        add_gaussian_peak(data, (0.5, 1.0, 0.0), sigma=0.12, intensity=80.0)

        res = symmetrize(data, symmetry='tetragonal', layer_axis=0, method='average', parallel=False, auto_expand=False)

        assert res.shape == data.shape
        ql_coords = res.nxaxes[0].nxdata
        idx_ql = np.argmin(np.abs(ql_coords - 0.5))
        s = res.nxsignal.nxdata[idx_ql, :, :]

        qh_coords = res.nxaxes[1].nxdata
        qk_coords = res.nxaxes[2].nxdata
        val_1_0 = s[np.argmin(np.abs(qh_coords - 1.0)), np.argmin(np.abs(qk_coords - 0.0))]
        val_0_1 = s[np.argmin(np.abs(qh_coords - 0.0)), np.argmin(np.abs(qk_coords - 1.0))]
        assert val_1_0 > 5.0
        assert val_0_1 > 5.0

    def test_case3_asymmetric_coordinate_bounds(self):
        data = make_synthetic_nxdata((45, 55, 35), ((-1.0, 2.0), (-1.0, 3.0), (-2.0, 2.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 0.0, 1.0), sigma=0.12, intensity=90.0)

        res = symmetrize(data, symmetry='hexagonal', layer_axis='l', method='average', parallel=False, auto_expand=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()

    def test_case4_unequal_shape_asymmetric_extents_orthorhombic(self):
        data = make_synthetic_nxdata((35, 45, 31), ((-1.5, 2.5), (-2.0, 2.0), (-1.0, 2.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 1.0, 0.0), sigma=0.15, intensity=100.0)

        res = symmetrize(data, symmetry='orthorhombic', layer_axis='k', method='average', parallel=False, auto_expand=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()

    def test_case5_trigonal_3fold(self):
        data = make_synthetic_nxdata((41, 41, 25), ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 0.0, 0.5), sigma=0.12, intensity=100.0)

        res = symmetrize(data, symmetry='trigonal', layer_axis=2, method='average', parallel=False, auto_expand=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()

    def test_case6_missing_negative_layers(self):
        data = make_synthetic_nxdata((41, 41, 21), ((-2.0, 2.0), (-2.0, 2.0), (0.0, 3.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (1.0, 0.0, 1.5), sigma=0.12, intensity=100.0)

        res = symmetrize(data, symmetry='hexagonal', layer_axis=2, method='average', parallel=False, auto_expand=False)

        assert res.shape == data.shape
        assert not np.isnan(res.nxsignal.nxdata).any()
        l_coords = res.nxaxes[2].nxdata
        idx_l = np.argmin(np.abs(l_coords - 1.5))
        assert res.nxsignal.nxdata[:, :, idx_l].max() > 5.0


class TestSymmetrizeFeatures:
    """Test 2D slicing, parallel execution equivalence, legacy class removal, and warnings."""

    def test_top_level_import(self):
        assert top_level_symmetrize is symmetrize

    def test_2d_symmetrize(self):
        data_2d = make_synthetic_nxdata((45, 45), ((-2.0, 2.0), (-2.0, 2.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (1.0, 0.0), sigma=0.12, intensity=100.0)

        res = symmetrize(data_2d, symmetry='tetragonal', method='average')

        assert res.shape == (45, 45)
        h = res.nxaxes[0].nxdata
        k = res.nxaxes[1].nxdata
        assert res.nxsignal.nxdata[np.argmin(np.abs(h - 0.0)), np.argmin(np.abs(k - 1.0))] > 5.0

    def test_parallel_vs_serial_identity(self):
        data = make_synthetic_nxdata((31, 31, 15), ((-1.5, 1.5), (-1.5, 1.5), (-1.0, 1.0)), names=('h', 'k', 'l'))
        add_gaussian_peak(data, (0.8, 0.0, 0.5), sigma=0.15, intensity=100.0)

        res_serial = symmetrize(data, symmetry='hexagonal', layer_axis=2, method='average', parallel=False)
        res_parallel = symmetrize(data, symmetry='hexagonal', layer_axis=2, method='average', parallel=True, num_workers=2)

        assert np.allclose(res_serial.nxsignal.nxdata, res_parallel.nxsignal.nxdata, atol=1e-10)

    def test_default_method_is_average(self):
        data_2d = make_synthetic_nxdata((25, 25), ((-1.0, 1.0), (-1.0, 1.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (0.5, 0.5), sigma=0.2, intensity=20.0)
        res = symmetrize(data_2d, symmetry='tetragonal')
        res_average = symmetrize(data_2d, symmetry='tetragonal', method='average')
        assert np.allclose(res.nxsignal.nxdata, res_average.nxsignal.nxdata)

    def test_classes_removed(self):
        """Verify Symmetrizer, Symmetrizer2D, and Symmetrizer3D are removed in v0.2.0."""
        import nxs_analysis_tools.pairdistribution as pd
        assert not hasattr(pd, "Symmetrizer")
        assert not hasattr(pd, "Symmetrizer2D")
        assert not hasattr(pd, "Symmetrizer3D")

    def test_positive_values_warning_and_override(self):
        data_2d = make_synthetic_nxdata((25, 25), ((-1.0, 1.0), (-1.0, 1.0)), names=('h', 'k'))
        data_2d.nxsignal.nxdata[:, :] = -5.0
        add_gaussian_peak(data_2d, (0.5, 0.5), sigma=0.2, intensity=20.0)

        with pytest.warns(UserWarning, match="Negative values found in symmetrized dataset and clipped to zero"):
            res_clipped = symmetrize(data_2d, symmetry='tetragonal', method='average', positive_values=True, auto_expand=False)
        assert (res_clipped.nxsignal.nxdata >= 0.0).all()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res_raw = symmetrize(data_2d, symmetry='tetragonal', method='average', positive_values=False, auto_expand=False)
        assert (res_raw.nxsignal.nxdata < 0.0).any()

    def test_wedge_rotations_and_reconstruction(self):
        # 45-degree wedge with mirror=True -> rotations must be 4
        _, _, _, _, _, rot_wedge = _resolve_symmetry_params(theta_min=0, theta_max=45, mirror=True, mirror_axis=0)
        assert rot_wedge == 4

        # 45-degree wedge with mirror=False -> rotations must be 8
        _, _, _, _, _, rot_nomirror = _resolve_symmetry_params(theta_min=0, theta_max=45, mirror=False)
        assert rot_nomirror == 8

        # 90-degree wedge with mirror=False -> rotations must be 4
        _, _, _, _, _, rot_90 = _resolve_symmetry_params(theta_min=45, theta_max=135, mirror=False)
        assert rot_90 == 4

        # Symmetrize synthetic 4-fold data
        data_2d = make_synthetic_nxdata((45, 45), ((-2.0, 2.0), (-2.0, 2.0)), names=('h', 'k'))
        data_2d.nxsignal.nxdata[:, :] = 1.0
        add_gaussian_peak(data_2d, (1.0, 0.0), sigma=0.15, intensity=50.0)
        add_gaussian_peak(data_2d, (0.0, 1.0), sigma=0.15, intensity=50.0)

        res = symmetrize(data_2d, method='wedge', theta_min=0, theta_max=45, mirror=True, mirror_axis=0, auto_expand=False)
        assert res.shape == (45, 45)
        assert (res.nxsignal.nxdata > 0).all()

        # test=True on 2D wedge should render 4 axes (2x2) and return NXdata
        plt.close('all')
        res_test = symmetrize(data_2d, method='wedge', theta_min=0, theta_max=45, mirror=True, mirror_axis=0, auto_expand=False, test=True)
        assert isinstance(res_test, NXdata)
        assert res_test.shape == (45, 45)
        plt.close('all')

    def test_diagnostic_test_flag_3d(self):
        data_3d = make_synthetic_nxdata((25, 25, 11), ((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)), names=('h', 'k', 'l'))
        add_gaussian_peak(data_3d, (0.5, 0.0, 0.0), sigma=0.15, intensity=50.0)
        plt.close('all')
        res_slice = symmetrize(data_3d, symmetry='tetragonal', test=True, slice_coord=0.0)
        assert isinstance(res_slice, NXdata)
        assert res_slice.ndim == 2
        assert res_slice.shape == (25, 25)
        plt.close('all')


class TestSymmetrizeAutoExpand:
    """Test auto_expand option (default True) to automatically extend coordinate bounds."""

    def test_auto_expand_default_is_true_2d_hexagonal(self):
        # Hexagonal symmetry: peak at (1.5, 1.5) on grid [-2.0, 2.0] x [-2.0, 2.0]
        # Under 6-fold rotation, (1.5, 1.5) rotates to (-3.0, 1.5)
        data_2d = make_synthetic_nxdata((41, 41), ((-2.0, 2.0), (-2.0, 2.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (1.5, 1.5), sigma=0.15, intensity=100.0)

        res_no_expand = symmetrize(data_2d, symmetry='hexagonal', auto_expand=False)
        assert res_no_expand.shape == (41, 41)
        assert np.isclose(float(res_no_expand.nxaxes[0][0]), -2.0)
        assert np.isclose(float(res_no_expand.nxaxes[0][-1]), 2.0)

        # Default auto_expand=True
        res_expand = symmetrize(data_2d, symmetry='hexagonal')
        assert res_expand.shape == (81, 81)
        assert np.isclose(float(res_expand.nxaxes[0][0]), -4.0)
        assert np.isclose(float(res_expand.nxaxes[0][-1]), 4.0)
        assert np.isclose(float(res_expand.nxaxes[1][0]), -4.0)
        assert np.isclose(float(res_expand.nxaxes[1][-1]), 4.0)
        assert res_expand.nxsignal.nxdata.sum() > res_no_expand.nxsignal.nxdata.sum()

    def test_auto_expand_never_truncates_smaller_non_empty_bounds(self):
        """
        Verify that when non-empty data occupies a region strictly smaller than
        the original data array bounds, auto_expand=True preserves the original
        data limits and never truncates the array.
        """
        data_2d = make_synthetic_nxdata((61, 61), ((-3.0, 3.0), (-3.0, 3.0)), names=('h', 'k'))
        # Only populate a small central region [-0.5, 0.5] x [-0.5, 0.5] (rest is exact 0.0)
        arr = data_2d.nxsignal.nxdata
        arr[25:36, 25:36] = 10.0
        arr[32, 30] = 50.0

        res = symmetrize(data_2d, symmetry='hexagonal', auto_expand=True)
        assert res.shape == (61, 61)
        assert np.isclose(float(res.nxaxes[0][0]), -3.0)
        assert np.isclose(float(res.nxaxes[0][-1]), 3.0)
        assert np.isclose(float(res.nxaxes[1][0]), -3.0)
        assert np.isclose(float(res.nxaxes[1][-1]), 3.0)

    def test_auto_expand_2d_rectangular_tetragonal(self):
        data_2d = make_synthetic_nxdata((21, 61), ((-1.0, 1.0), (-3.0, 3.0)), names=('h', 'k'))
        add_gaussian_peak(data_2d, (0.0, 2.5), sigma=0.15, intensity=80.0)

        res = symmetrize(data_2d, symmetry='tetragonal')
        assert np.isclose(float(res.nxaxes[0][0]), -3.0)
        assert np.isclose(float(res.nxaxes[0][-1]), 3.0)
        assert np.isclose(float(res.nxaxes[1][0]), -3.0)
        assert np.isclose(float(res.nxaxes[1][-1]), 3.0)
        h_vals = res.nxaxes[0].nxdata
        k_vals = res.nxaxes[1].nxdata
        val_pos = res.nxsignal.nxdata[np.argmin(np.abs(h_vals - 2.5)), np.argmin(np.abs(k_vals - 0.0))]
        val_neg = res.nxsignal.nxdata[np.argmin(np.abs(h_vals - (-2.5))), np.argmin(np.abs(k_vals - 0.0))]
        assert val_pos > 5.0
        assert val_neg > 5.0

    def test_auto_expand_raw_numpy_array(self):
        arr = np.zeros((21, 21), dtype=float)
        arr[20, 20] = 100.0
        res = symmetrize(arr, symmetry='hexagonal')
        assert isinstance(res, np.ndarray)
        assert res.shape == (41, 41)

    def test_auto_expand_3d_average(self):
        data_3d = make_synthetic_nxdata((11, 21, 21), ((-0.5, 0.5), (-1.0, 1.0), (-1.0, 1.0)), names=('l', 'h', 'k'))
        add_gaussian_peak(data_3d, (0.2, 0.8, 0.8), sigma=0.1, intensity=50.0)
        res = symmetrize(data_3d, symmetry='hexagonal', layer_axis='l', method='average')
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
        slice_2d = data[0.0, :, :]

        sl_unexpanded = symmetrize(slice_2d, symmetry='hexagonal', auto_expand=False)
        sl_expanded = symmetrize(slice_2d, symmetry='hexagonal')

        assert sl_unexpanded.shape == (1101, 1101)
        assert sl_expanded.shape == (1145, 1145)
        assert np.isclose(float(sl_expanded.nxaxes[0][0]), -11.44)
        assert np.isclose(float(sl_expanded.nxaxes[0][-1]), 11.44)
        assert np.isclose(float(sl_expanded.nxaxes[1][0]), -11.44)
        assert np.isclose(float(sl_expanded.nxaxes[1][-1]), 11.44)
        assert sl_expanded.nxsignal.nxdata.sum() > sl_unexpanded.nxsignal.nxdata.sum()
