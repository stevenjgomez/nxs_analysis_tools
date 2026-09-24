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


def test_find_temperatures_decimal_and_p_format(tmp_path):
    (tmp_path / "sample_15p5.nxs").touch()
    (tmp_path / "sample_20.nxs").touch()
    (tmp_path / "sample_25.5.nxs").touch()

    td = TempDependence(str(tmp_path))
    td.find_temperatures()

    # Must be numeric values sorted numerically, no 'p'
    assert td.temperatures == [15.5, 20, 25.5]


def test_find_temperatures_legacy_chess_format(tmp_path):
    # Setup directories for legacy CHESS format
    for folder in ["15p5", "20", "25.5", "300"]:
        d = tmp_path / folder
        d.mkdir()
        (d / "data_hkli.nxs").touch()

    # Add non-temperature or non-nxs folders to ensure they are ignored
    other = tmp_path / "other_folder"
    other.mkdir()
    (other / "data.nxs").touch()

    no_nxs = tmp_path / "99"
    no_nxs.mkdir()
    (no_nxs / "readme.txt").touch()

    td = TempDependence(str(tmp_path))
    td.find_temperatures()

    assert td.temperatures == [15.5, 20, 25.5, 300]

    # Test filtering with logical operators
    below_150 = [T for T in td.temperatures if T < 150]
    assert below_150 == [15.5, 20, 25.5]


def test_plot_linecuts_heatmap_float_temperatures():
    td = TempDependence()
    td.temperatures = ['15.5', '20.5', '25.5']
    for t in td.temperatures:
        x = NXfield(np.linspace(0, 1, 10), name='Qh')
        sig = NXfield(np.ones(10), name='counts')
        td.linecuts[t] = NXdata(sig, (x,))

    p = td.plot_linecuts_heatmap()
    # Ensure heatmap coordinates on y-axis are floats preserving decimal precision
    # The meshgrid y values correspond to the temperatures
    y_coords = p.axes.get_children()[0].get_coordinates()[:, :, 1]
    # Check that y bounds cover the float range [15.5, 25.5]
    assert np.isclose(y_coords.min(), 15.5) or y_coords.min() <= 15.5
    assert np.isclose(y_coords.max(), 25.5) or y_coords.max() >= 25.5


def test_plot_order_parameter_float_temperatures():
    from unittest.mock import MagicMock
    td = TempDependence()
    td.temperatures = ['15.5', '20.5', '25.5']
    for t in td.temperatures:
        m = MagicMock()
        m.modelresult.params = {'peakheight': MagicMock(value=float(t) * 2.0)}
        td.linecutmodels[t] = m

    fig, ax = td.plot_order_parameter()
    line = ax.get_lines()[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    assert np.allclose(xdata, [15.5, 20.5, 25.5])
    assert np.issubdtype(xdata.dtype, np.floating)
    assert np.allclose(ydata, [31.0, 41.0, 51.0])


def test_plot_integration_window_float_resolution(sample_3d_nxdata):
    td = TempDependence()
    td.temperatures = ['15.5']
    td.datasets['15.5'] = sample_3d_nxdata
    td.scissors['15.5'] = Scissors(data=sample_3d_nxdata, center=(0.0, 0.0, 0.0), window=(0.2, 0.2, 0.2))

    # Float numeric lookup
    plots1 = td.plot_integration_window(temperature=15.5)
    assert len(plots1) == 3

    # String lookup
    plots2 = td.plot_integration_window(temperature='15.5')
    assert len(plots2) == 3

    # Legacy 'p' lookup
    plots3 = td.plot_integration_window(temperature='15p5')
    assert len(plots3) == 3


def test_temp_dict_indexing():
    from nxs_analysis_tools.chess import TempDict
    d = TempDict()
    d['15'] = 'val_15'
    d['25p5'] = 'val_25.5'
    d[300] = 'val_300'

    # Numeric and string retrieval
    assert d[15] == 'val_15'
    assert d['15'] == 'val_15'
    assert d[15.0] == 'val_15'
    assert d[25.5] == 'val_25.5'
    assert d['25.5'] == 'val_25.5'
    assert d['25p5'] == 'val_25.5'
    assert d[300] == 'val_300'
    assert d['300'] == 'val_300'

    # Containment
    assert 15 in d
    assert '15' in d
    assert 25.5 in d
    assert '25.5' in d
    assert '25p5' in d
    assert 999 not in d

    # .get() and .pop()
    assert d.get(15) == 'val_15'
    assert d.get('25.5') == 'val_25.5'
    assert d.get(999, 'default') == 'default'
    assert d.pop('15') == 'val_15'
    assert 15 not in d


def test_temp_dependence_numeric_indexing(sample_3d_nxdata):
    td = TempDependence()
    # Insert with integer
    td.datasets[15] = sample_3d_nxdata
    td.scissors[15] = Scissors(data=sample_3d_nxdata, center=(0.0, 0.0, 0.0), window=(0.2, 0.2, 0.2))

    # Access with string or int
    assert td.datasets[15] is sample_3d_nxdata
    assert td.datasets['15'] is sample_3d_nxdata
    assert td.scissors[15] is not None
    assert td.scissors['15'] is not None


def test_set_temperatures_numeric():
    td = TempDependence()
    td.set_temperatures(['15', '20.5', '25p5', 300])
    assert td.temperatures == [15, 20.5, 25.5, 300]
    assert all(isinstance(t, (int, float)) for t in td.temperatures)


def test_temp_dependence_initialize_connects_data(sample_3d_nxdata):
    td = TempDependence()
    td.temperatures = [15, 20]
    td.datasets[15] = sample_3d_nxdata
    td.datasets[20] = sample_3d_nxdata
    td.initialize()

    assert td.scissors[15].data is sample_3d_nxdata
    assert td.scissors[20].data is sample_3d_nxdata
    assert td.scissors['15'].data is sample_3d_nxdata
    assert td.scissors['20'].data is sample_3d_nxdata


def test_load_transforms_flexible_temperature_formats(tmp_path, sample_3d_nxdata, monkeypatch):
    # Setup files with mixed 'p', decimal, and integer formats
    (tmp_path / "sample_15p5.nxs").touch()
    (tmp_path / "sample_20.nxs").touch()
    (tmp_path / "sample_25.5.nxs").touch()
    (tmp_path / "sample_300.nxs").touch()

    monkeypatch.setattr("nxs_analysis_tools.chess.load_transform", lambda path, **kwargs: sample_3d_nxdata)

    # 1. temperatures with 'p' notation string and standard decimal float
    td = TempDependence(str(tmp_path))
    td.load_transforms(temperatures=['15p5', 25.5], print_tree=False)
    assert td.temperatures == [15.5, 25.5]
    assert 15.5 in td.datasets
    assert 25.5 in td.datasets
    assert '15p5' in td.datasets

    # 2. temperatures with decimal string and int, plus exclude_temperatures with 'p' string
    td2 = TempDependence(str(tmp_path))
    td2.load_transforms(temperatures=['15.5', 20, '25p5'], exclude_temperatures='15p5', print_tree=False)
    assert td2.temperatures == [20, 25.5]

    # 3. exclude_temperatures as a list of mixed formats (decimal string and float)
    td3 = TempDependence(str(tmp_path))
    td3.load_transforms(exclude_temperatures=['25.5', 15.5], print_tree=False)
    assert td3.temperatures == [20, 300]

    # 4. Deprecation warning when using temperatures_list
    td4 = TempDependence(str(tmp_path))
    with pytest.deprecated_call(match="`temperatures_list` is deprecated"):
        td4.load_transforms(temperatures_list=[20, 300], print_tree=False)
    assert td4.temperatures == [20, 300]


def test_load_datasets_flexible_temperature_formats(tmp_path, sample_3d_nxdata, monkeypatch):
    # Setup directories with mixed 'p', decimal, and integer formats
    for folder in ["15p5", "20", "25.5", "300"]:
        d = tmp_path / folder
        d.mkdir()
        (d / "data_hkli.nxs").touch()

    monkeypatch.setattr("nxs_analysis_tools.chess.load_data", lambda path, print_tree=True: sample_3d_nxdata)

    # 1. temperatures with decimal float and decimal string matching '15p5' folder
    td = TempDependence(str(tmp_path))
    td.load_datasets(temperatures=[15.5, '25.5'], print_tree=False)
    assert td.temperatures == [15.5, 25.5]
    assert 15.5 in td.datasets
    assert '15p5' in td.datasets

    # 2. temperatures with 'p' string, excluding single int
    td2 = TempDependence(str(tmp_path))
    td2.load_datasets(temperatures=['15p5', 20, 300], exclude_temperatures=20, print_tree=False)
    assert td2.temperatures == [15.5, 300]

    # 3. exclude_temperatures with 'p' string and decimal string
    td3 = TempDependence(str(tmp_path))
    td3.load_datasets(exclude_temperatures=['15p5', '25.5'], print_tree=False)
    assert td3.temperatures == [20, 300]

    # 4. Deprecation warning when using temperatures_list
    td4 = TempDependence(str(tmp_path))
    with pytest.deprecated_call(match="`temperatures_list` is deprecated"):
        td4.load_datasets(temperatures_list=[20, 300], print_tree=False)
    assert td4.temperatures == [20, 300]






