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


@pytest.fixture
def multi_temp_dependence():
    td = TempDependence()
    td.temperatures = ['15', '20', '300']
    for t in td.temperatures:
        qh = NXfield(np.linspace(0, 1, 5), name='Qh')
        qk = NXfield(np.linspace(-2, 2, 6), name='Qk')
        ql = NXfield(np.linspace(-3, 3, 7), name='Ql')
        sig = NXfield(np.full((5, 6, 7), float(t)), name='counts')
        td.datasets[t] = NXdata(sig, (qh, qk, ql))
    return td


def test_to_xtec_basic_int_temperatures(multi_temp_dependence):
    td = multi_temp_dependence
    data = td.to_xtec()

    assert isinstance(data, NXdata)
    assert td.xtec_data is data
    assert data.nxsignal.shape == (3, 5, 6, 7)
    assert [ax.nxname for ax in data.nxaxes] == ['Te', 'Qh', 'Qk', 'Ql']

    # Temperature axis checks (must be integer and not string)
    te_vals = data['Te'].nxvalue
    assert np.issubdtype(te_vals.dtype, np.integer)
    assert not np.issubdtype(te_vals.dtype, np.str_)
    assert np.array_equal(te_vals, [15, 20, 300])

    # Spatial coordinates
    assert np.allclose(data['Qh'].nxvalue, np.linspace(0, 1, 5))
    assert np.allclose(data['Qk'].nxvalue, np.linspace(-2, 2, 6))
    assert np.allclose(data['Ql'].nxvalue, np.linspace(-3, 3, 7))


def test_to_xtec_float_temperatures():
    td = TempDependence()
    td.temperatures = ['15', '25p5', '300']
    for t in td.temperatures:
        qh = NXfield(np.linspace(0, 1, 5), name='Qh')
        qk = NXfield(np.linspace(-2, 2, 6), name='Qk')
        ql = NXfield(np.linspace(-3, 3, 7), name='Ql')
        t_val = float(t.replace('p', '.'))
        sig = NXfield(np.full((5, 6, 7), t_val), name='counts')
        td.datasets[t] = NXdata(sig, (qh, qk, ql))

    data = td.to_xtec()
    te_vals = data['Te'].nxvalue
    assert np.issubdtype(te_vals.dtype, np.floating)
    assert not np.issubdtype(te_vals.dtype, np.str_)
    assert np.allclose(te_vals, [15.0, 25.5, 300.0])


def test_to_xtec_save_and_load(multi_temp_dependence, tmp_path):
    from nexusformat.nexus import nxload
    td = multi_temp_dependence
    filepath = tmp_path / "test_xtec.nxs"

    ret_data = td.to_xtec(filepath=filepath)
    assert filepath.exists()
    assert ret_data is td.xtec_data

    # Load and test slicing as expected in XTEC-GPU tutorial
    loaded_file = nxload(str(filepath), 'r')
    assert 'entry/data' in loaded_file

    data = loaded_file['entry/data'][:, 0.0:0.5, -1.0:1.0, -2.0:2.0]
    i_signal = data.nxsignal.nxvalue
    qh = data['Qh'].nxvalue
    qk = data['Qk'].nxvalue
    ql = data['Ql'].nxvalue
    temp = data['Te'].nxvalue

    assert i_signal.shape == (3, 3, 4, 5)
    assert np.array_equal(temp, [15, 20, 300])
    assert np.allclose(qh, [0.0, 0.25, 0.5])


def test_to_xtec_custom_axis_name_and_units(multi_temp_dependence):
    td = multi_temp_dependence
    data = td.to_xtec(temp_axis_name='temperature', temp_units='Celsius')

    assert [ax.nxname for ax in data.nxaxes] == ['temperature', 'Qh', 'Qk', 'Ql']
    assert 'temperature' in data
    assert data['temperature'].attrs.get('units') == 'Celsius'


def test_to_xtec_temperature_filtering(multi_temp_dependence):
    td = multi_temp_dependence
    # Filter with integer 15 and string '300'
    data = td.to_xtec(temperatures=[15, '300'])

    assert data.nxsignal.shape == (2, 5, 6, 7)
    assert np.array_equal(data['Te'].nxvalue, [15, 300])


def test_to_xtec_overwrite_protection(multi_temp_dependence, tmp_path):
    td = multi_temp_dependence
    filepath = tmp_path / "exists.nxs"
    filepath.write_text("existing content")

    with pytest.raises(FileExistsError, match="already exists"):
        td.to_xtec(filepath=filepath, overwrite=False)


def test_to_xtec_validation_errors(multi_temp_dependence):
    # Empty datasets
    empty_td = TempDependence()
    with pytest.raises(ValueError, match="No datasets found"):
        empty_td.to_xtec()

    # Non-numeric temperature key
    td = multi_temp_dependence
    td.datasets['bad'] = td.datasets['15']
    with pytest.raises(ValueError, match="Could not parse temperature 'bad' as a numeric value"):
        td.to_xtec(temperatures=['15', 'bad'])

    # Shape mismatch with valid numeric temperature
    mismatched = NXdata(NXfield(np.ones((2, 2, 2)), name='counts'),
                        (NXfield([0, 1], name='Qh'), NXfield([0, 1], name='Qk'), NXfield([0, 1], name='Ql')))
    td.datasets['40'] = mismatched
    with pytest.raises(ValueError, match="does not match expected shape"):
        td.to_xtec(temperatures=['15', '40'])

    # Key error for missing temperature
    with pytest.raises(KeyError, match="not found in datasets"):
        td.to_xtec(temperatures=['999'])


