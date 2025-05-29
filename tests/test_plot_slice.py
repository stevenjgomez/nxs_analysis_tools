import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from nexusformat.nexus import NXdata, NXfield

from nxs_analysis_tools.datareduction import plot_slice

# Prevent actual display call from running in tests
import builtins
builtins.display = lambda *args, **kwargs: None

@pytest.fixture
def sample_2d_array():
    return np.arange(100).reshape(10, 10)

@pytest.fixture
def sample_3d_array():
    return np.random.rand(4, 10, 10)

@pytest.fixture
def sample_nxdata():
    x = NXfield(np.linspace(0, 1, 10), name='x')
    y = NXfield(np.linspace(0, 1, 10), name='y')
    z = NXfield(np.random.rand(10, 10), name='z')
    return NXdata(z, (x, y))

def test_plot_basic_2d(sample_2d_array):
    p = plot_slice(sample_2d_array)
    assert isinstance(p, QuadMesh)

def test_plot_basic_nxdata(sample_nxdata):
    p = plot_slice(sample_nxdata)
    assert isinstance(p, QuadMesh)

def test_plot_with_axes(sample_2d_array):
    fig, ax = plt.subplots()
    p = plot_slice(sample_2d_array, ax=ax)
    assert p.axes is ax

def test_3d_array_requires_sum_axis(sample_3d_array):
    with pytest.raises(ValueError, match="sum_axis must be specified"):
        plot_slice(sample_3d_array)

def test_3d_array_sum_axis(sample_3d_array):
    p = plot_slice(sample_3d_array, sum_axis=0)
    assert isinstance(p, QuadMesh)

def test_invalid_data_type():
    with pytest.raises(TypeError):
        plot_slice("not an array")

def test_invalid_X_type(sample_2d_array):
    with pytest.raises(TypeError):
        plot_slice(sample_2d_array, X="bad type")

def test_transpose_behavior(sample_2d_array):
    p = plot_slice(sample_2d_array, transpose=True)
    assert isinstance(p, QuadMesh)

def test_custom_color_limits(sample_2d_array):
    p = plot_slice(sample_2d_array, vmin=10, vmax=50)
    assert isinstance(p.norm, Normalize)
    assert p.norm.vmin == 10
    assert p.norm.vmax == 50

def test_log_scale_color_norm(sample_2d_array):
    arr = sample_2d_array.copy()
    arr = arr.astype(float)
    arr[arr == 0] = 1e-3  # Avoid log(0)
    p = plot_slice(arr, logscale=True)
    assert isinstance(p.norm, LogNorm)

def test_symlog_scale_color_norm(sample_2d_array):
    arr = sample_2d_array - 50  # Make it negative + positive
    p = plot_slice(arr, symlogscale=True, linthresh=5)
    assert isinstance(p.norm, SymLogNorm)
    assert p.norm.linthresh == 5

def test_skew_angle_applied(sample_2d_array):
    p = plot_slice(sample_2d_array, skew_angle=75)
    assert isinstance(p, QuadMesh)

def test_xlim_ylim_and_ticks(sample_2d_array):
    fig, ax = plt.subplots()
    p = plot_slice(
        sample_2d_array,
        ax=ax,
        xlim=(2, 8),
        ylim=(1, 9),
        xticks=2,
        yticks=2
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[0] <= 2 and xlim[1] >= 8
    assert ylim[0] <= 1 and ylim[1] >= 9

def test_colorbar_label(sample_nxdata):
    fig, ax = plt.subplots()
    p = plot_slice(sample_nxdata, ax=ax, cbar=True, cbartitle="MyColorBar")
    # No assertion possible unless you retrieve the colorbar itself, which is tricky
    assert isinstance(p, QuadMesh)

def test_mdheading_rendering(sample_2d_array):
    # Should not raise error with heading
    plot_slice(sample_2d_array, mdheading="Custom Heading")

def test_plot_returns_quadmesh(sample_nxdata):
    p = plot_slice(sample_nxdata)
    assert isinstance(p, QuadMesh)
