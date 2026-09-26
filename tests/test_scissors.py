import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import builtins
builtins.display = lambda *args, **kwargs: None

from nexusformat.nexus import NXdata, NXfield
from nxs_analysis_tools.datareduction import Scissors
from nxs_analysis_tools.chess import TempDependence

@pytest.fixture
def sample_3d_uniform_volume():
    # 3D dataset with a 1D peak along Qh, and uniform intensity along Qk and Ql
    qh = np.linspace(-1, 1, 21)
    qk = np.linspace(-1, 1, 21)
    ql = np.linspace(-1, 1, 21)
    
    # Peak along Qh
    peak_1d = np.exp(-0.5 * (qh / 0.2)**2)
    # Broadcast across 3D
    v = peak_1d[:, None, None] * np.ones((len(qh), len(qk), len(ql)))
    
    x_field = NXfield(qh, name='Qh')
    y_field = NXfield(qk, name='Qk')
    z_field = NXfield(ql, name='Ql')
    v_field = NXfield(v, name='counts')
    return NXdata(v_field, (x_field, y_field, z_field))

def test_cut_data_unnormalized_scales_with_window(sample_3d_uniform_volume):
    sc = Scissors(data=sample_3d_uniform_volume, center=(0, 0, 0))
    
    # Smaller integration window
    cut_small = sc.cut_data(window=(1.0, 0.1, 0.1), normalize=False)
    # Larger integration window
    cut_large = sc.cut_data(window=(1.0, 0.3, 0.3), normalize=False)
    
    # Raw sum should be larger for larger window
    assert np.max(cut_large.nxsignal.nxdata) > np.max(cut_small.nxsignal.nxdata)

def test_cut_data_normalized_is_invariant_to_window(sample_3d_uniform_volume):
    sc = Scissors(data=sample_3d_uniform_volume, center=(0, 0, 0))
    
    # When normalized, average intensity per bin should be identical regardless of window size
    cut_small = sc.cut_data(window=(1.0, 0.1, 0.1), normalize=True)
    cut_large = sc.cut_data(window=(1.0, 0.3, 0.3), normalize=True)
    
    np.testing.assert_allclose(cut_small.nxsignal.nxdata, cut_large.nxsignal.nxdata, rtol=1e-5)
    assert cut_small.attrs.get('normalized') is True

def test_cut_data_normalized_with_nans(sample_3d_uniform_volume):
    data_with_nans = sample_3d_uniform_volume
    raw = np.copy(data_with_nans.nxsignal.nxdata)
    # Put NaNs in half of the transverse bins
    raw[:, 11:, :] = np.nan
    data_with_nans.counts = raw
    
    sc = Scissors(data=data_with_nans, center=(0, 0, 0))
    cut_norm = sc.cut_data(window=(1.0, 0.2, 0.2), normalize=True, empty_bins='nan')
    
    # Center of peak should still be ~1.0 despite NaNs in the window
    center_idx = len(cut_norm.nxsignal.nxdata) // 2
    assert np.isclose(cut_norm.nxsignal.nxdata[center_idx], 1.0, atol=1e-3)

def test_cut_data_empty_region_yields_zero():
    # Volume with all zeros or NaNs
    qh = np.linspace(-1, 1, 11)
    qk = np.linspace(-1, 1, 11)
    ql = np.linspace(-1, 1, 11)
    v = np.zeros((11, 11, 11))
    
    data = NXdata(NXfield(v, name='counts'),
                  (NXfield(qh, name='Qh'), NXfield(qk, name='Qk'), NXfield(ql, name='Ql')))
    sc = Scissors(data=data, center=(0, 0, 0))
    cut = sc.cut_data(window=(0.5, 0.2, 0.2), normalize=True, empty_bins='both')
    
    assert np.all(cut.nxsignal.nxdata == 0.0)

def test_temp_dependence_cut_data_propagation(sample_3d_uniform_volume):
    from nxs_analysis_tools.fitting import LinecutModel
    td = TempDependence()
    td.temperatures = ['15']
    td.datasets['15'] = sample_3d_uniform_volume
    td.scissors['15'] = Scissors(data=sample_3d_uniform_volume, center=(0, 0, 0), window=(1.0, 0.2, 0.2))
    td.linecutmodels['15'] = LinecutModel()
    
    linecuts = td.cut_data(center=(0, 0, 0), window=(1.0, 0.2, 0.2), normalize=True)
    assert linecuts['15'].attrs.get('normalized') is True

def test_cubic_l_rods_normalization():
    from nxs_analysis_tools.datasets import cubic_l_rods
    from nxs_analysis_tools.datareduction import load_data
    data = load_data(cubic_l_rods(), print_tree=False)
    sc = Scissors(data=data, center=(0, 0, 0))
    # Cut across H (axis 0) with two different window sizes along L (axis 2)
    cut_narrow = sc.cut_data(window=(1.5, 0.1, 0.2), normalize=True)
    cut_wide = sc.cut_data(window=(1.5, 0.1, 0.8), normalize=True)
    np.testing.assert_allclose(cut_narrow.nxsignal.nxdata, cut_wide.nxsignal.nxdata, rtol=1e-3)


def test_scissors_highlight_integration_window_removed(sample_3d_uniform_volume):
    sc = Scissors(data=sample_3d_uniform_volume)
    assert not hasattr(sc, 'highlight_integration_window')


def test_rotate_data_2d_removed():
    import nxs_analysis_tools.datareduction as dr
    assert not hasattr(dr, 'rotate_data_2D')

