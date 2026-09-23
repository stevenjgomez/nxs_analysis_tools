import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import builtins
builtins.display = lambda *args, **kwargs: None

from nexusformat.nexus import NXdata, NXfield
from nxs_analysis_tools.chess import TempDependence
from nxs_analysis_tools.datareduction import Scissors

@pytest.fixture
def sample_3d_nxdata():
    x = NXfield(np.linspace(-2, 2, 21), name='Qh')
    y = NXfield(np.linspace(-2, 2, 21), name='Qk')
    z = NXfield(np.linspace(-2, 2, 21), name='Ql')
    v = NXfield(np.ones((21, 21, 21)), name='counts')
    return NXdata(v, (x, y, z))

@pytest.fixture
def temp_dependence_instance(sample_3d_nxdata):
    td = TempDependence()
    td.temperatures = ['15']
    td.datasets['15'] = sample_3d_nxdata
    sc = Scissors(data=sample_3d_nxdata, center=(0.0, 0.0, 0.0), window=(0.2, 0.2, 0.2))
    td.scissors['15'] = sc
    return td

def test_chess_plot_integration_window_width_height(temp_dependence_instance):
    td = temp_dependence_instance
    plots = td.plot_integration_window(temperature='15', width=1.0, height=0.6)
    
    # Check that width and height appropriately set axis limits on the principal cross-sections
    for p in plots:
        ax = p.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert np.isclose(xlim[0], -0.5)
        assert np.isclose(xlim[1], 0.5)
        assert np.isclose(ylim[0], -0.3)
        assert np.isclose(ylim[1], 0.3)

def test_chess_plot_integration_window_zooms_when_show_highlight_false(temp_dependence_instance):
    td = temp_dependence_instance
    plots = td.plot_integration_window(temperature='15', show_highlight=False, width=1.0, height=0.6)
    
    for p in plots:
        ax = p.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert np.isclose(xlim[0], -0.5)
        assert np.isclose(xlim[1], 0.5)
        assert np.isclose(ylim[0], -0.3)
        assert np.isclose(ylim[1], 0.3)

def test_chess_highlight_integration_window_width_height_deprecated(temp_dependence_instance):
    td = temp_dependence_instance
    with pytest.deprecated_call():
        plots = td.highlight_integration_window(temperature='15', width=1.0, height=0.6)
    
    for p in plots:
        ax = p.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert np.isclose(xlim[0], -0.5)
        assert np.isclose(xlim[1], 0.5)
        assert np.isclose(ylim[0], -0.3)
        assert np.isclose(ylim[1], 0.3)

def test_chess_plot_integration_window_defaults_temperature_warning(temp_dependence_instance):
    td = temp_dependence_instance
    with pytest.warns(UserWarning, match="No temperature specified. Defaulting to temperature 15 K."):
        plots = td.plot_integration_window(width=1.0, height=0.6)
    assert len(plots) == 3

def test_chess_plot_integration_window_numeric_temperature(temp_dependence_instance):
    td = temp_dependence_instance
    plots = td.plot_integration_window(temperature=15, width=1.0, height=0.6)
    assert len(plots) == 3
