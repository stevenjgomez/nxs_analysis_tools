import pytest
import numpy as np
from nexusformat.nexus import NXdata, NXfield
from nxs_analysis_tools.pairdistribution import generate_gaussian


@pytest.fixture
def lattice_params():
    return (4.0, 4.0, 6.0, 90.0, 90.0, 90.0)


def test_generate_gaussian_default_center(lattice_params):
    """Test that default center=None does not raise TypeError and peaks at (0, 0, 0)."""
    h = np.linspace(-1, 1, 11)
    k = np.linspace(-1, 1, 11)
    l = np.linspace(-1, 1, 11)

    # Calling with default center=None (was raising TypeError prior to fix)
    g = generate_gaussian(h, k, l, amp=2.5, stddev=0.8, lattice_params=lattice_params)
    assert g.shape == (11, 11, 11)
    assert np.isclose(g[5, 5, 5], 2.5)
    assert g[5, 5, 5] == np.max(g)


def test_generate_gaussian_plain_arrays_and_custom_center(lattice_params):
    """Test plain numpy arrays with custom center."""
    h = np.linspace(-2, 2, 21)
    k = np.linspace(-3, 3, 31)
    l = np.linspace(-4, 4, 41)

    center = (1.0, -1.0, 2.0)
    g = generate_gaussian(h, k, l, amp=5.0, stddev=1.2, lattice_params=lattice_params, center=center)
    assert g.shape == (21, 31, 41)

    # Peak should occur at h=1.0 (index 15), k=-1.0 (index 10), l=2.0 (index 30)
    h_idx = np.argmin(np.abs(h - 1.0))
    k_idx = np.argmin(np.abs(k - (-1.0)))
    l_idx = np.argmin(np.abs(l - 2.0))
    assert np.isclose(g[h_idx, k_idx, l_idx], 5.0)


def test_generate_gaussian_nxdata_lkh_axis_order(lattice_params):
    """Test generate_gaussian with NXdata in [Ql, Qk, Qh] order (use_nxlink=True convention)."""
    l = NXfield(np.linspace(-3, 3, 31), name='Ql')
    k = NXfield(np.linspace(-2, 2, 21), name='Qk')
    h = NXfield(np.linspace(-1, 1, 11), name='Qh')
    data = NXdata(NXfield(np.zeros((31, 21, 11)), name='counts'), (l, k, h))

    # Pass axes directly as fields
    g_fields = generate_gaussian(
        data.nxaxes[0], data.nxaxes[1], data.nxaxes[2],
        amp=3.0, stddev=0.7, lattice_params=lattice_params
    )
    assert g_fields.shape == (31, 21, 11)
    assert g_fields.shape == data.nxsignal.shape

    # Pass NXdata object directly
    g_nxdata = generate_gaussian(data, amp=3.0, stddev=0.7, lattice_params=lattice_params)
    assert g_nxdata.shape == (31, 21, 11)
    assert np.allclose(g_fields, g_nxdata)

    # Verify subtraction works with no broadcasting error
    diff = data.nxsignal.nxdata - g_nxdata
    assert diff.shape == (31, 21, 11)


def test_generate_gaussian_physics_alignment(lattice_params):
    """Verify that physical coordinates align identically between HKL and LKH representations."""
    h_vals = np.linspace(-1, 1, 11)
    k_vals = np.linspace(-2, 2, 21)
    l_vals = np.linspace(-3, 3, 31)

    # In HKL order (shape: 11, 21, 31)
    h_field = NXfield(h_vals, name='Qh')
    k_field = NXfield(k_vals, name='Qk')
    l_field = NXfield(l_vals, name='Ql')
    data_hkl = NXdata(NXfield(np.zeros((11, 21, 31)), name='counts'), (h_field, k_field, l_field))
    g_hkl = generate_gaussian(data_hkl, amp=2.0, stddev=1.0, lattice_params=lattice_params)

    # In LKH order (shape: 31, 21, 11)
    data_lkh = NXdata(NXfield(np.zeros((31, 21, 11)), name='counts'), (l_field, k_field, h_field))
    g_lkh = generate_gaussian(data_lkh, amp=2.0, stddev=1.0, lattice_params=lattice_params)

    # Verify at multiple coordinate points that g_hkl[h, k, l] == g_lkh[l, k, h]
    for hi, ki, li in [(2, 5, 8), (5, 10, 15), (8, 14, 25)]:
        assert np.isclose(g_hkl[hi, ki, li], g_lkh[li, ki, hi])


def test_generate_gaussian_return_nxdata(lattice_params):
    """Test return_nxdata=True wrapping in NXdata with metadata preserved."""
    l = NXfield(np.linspace(-3, 3, 31), name='Ql')
    k = NXfield(np.linspace(-2, 2, 21), name='Qk')
    h = NXfield(np.linspace(-1, 1, 11), name='Qh')
    data = NXdata(NXfield(np.zeros((31, 21, 11)), name='counts'), (l, k, h))

    g_obj = generate_gaussian(data, amp=1.0, stddev=1.0, lattice_params=lattice_params, return_nxdata=True)
    assert isinstance(g_obj, NXdata)
    assert g_obj.nxsignal.shape == (31, 21, 11)
    assert [ax.nxname for ax in g_obj.nxaxes] == ['Ql', 'Qk', 'Qh']


def test_generate_gaussian_dict_center(lattice_params):
    """Test specifying center as a dictionary with coordinate keys."""
    h = np.linspace(-1, 1, 11)
    k = np.linspace(-1, 1, 11)
    l = np.linspace(-1, 1, 11)
    center_dict = {'H': 0.4, 'K': -0.2, 'L': 0.6}
    center_tuple = (0.4, -0.2, 0.6)

    g_dict = generate_gaussian(h, k, l, amp=1.0, stddev=1.0, lattice_params=lattice_params, center=center_dict)
    g_tuple = generate_gaussian(h, k, l, amp=1.0, stddev=1.0, lattice_params=lattice_params, center=center_tuple)
    assert np.allclose(g_dict, g_tuple)


def test_generate_gaussian_errors(lattice_params):
    """Test error handling in generate_gaussian."""
    h = np.linspace(-1, 1, 10)
    k = np.linspace(-1, 1, 10)

    # Missing L axis when input is not NXdata
    with pytest.raises(ValueError, match="H, K, and L coordinate axes must all be provided"):
        generate_gaussian(h, k, amp=1.0, stddev=1.0, lattice_params=lattice_params)

    # Missing lattice_params
    with pytest.raises(ValueError, match="lattice_params.*must be provided"):
        generate_gaussian(h, k, h, amp=1.0, stddev=1.0)

    # Non-3D NXdata
    data_2d = NXdata(NXfield(np.zeros((10, 10)), name='counts'), (NXfield(h, name='Qh'), NXfield(k, name='Qk')))
    with pytest.raises(ValueError, match="requires a 3-dimensional NXdata"):
        generate_gaussian(data_2d, amp=1.0, stddev=1.0, lattice_params=lattice_params)
