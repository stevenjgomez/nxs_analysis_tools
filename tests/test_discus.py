import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import builtins
builtins.display = lambda *args, **kwargs: None

import h5py
from nexusformat.nexus import NXdata
from nxs_analysis_tools import load_discus_nxs, plot_slice


def create_mock_discus_nxs(filepath, shape, lower_limits, step_vecs, step_sizes=None):
    """
    Helper to create a synthetic DISCUS HDF5 file matching the DISCUS schema.
    """
    with h5py.File(str(filepath), 'w') as f:
        f.create_dataset('PROGRAM', data=np.bytes_('DISCUS60'))
        f.create_dataset('format', data=np.bytes_('Yell 1.0'))
        f.create_dataset('is_direct', data=np.int8(0))
        f.create_dataset('lower_limits', data=np.array(lower_limits, dtype=np.float64))
        if step_sizes is not None:
            f.create_dataset('step_sizes', data=np.array(step_sizes, dtype=np.float64))
        else:
            f.create_dataset('step_sizes', data=np.array([np.max(np.abs(v)) for v in step_vecs], dtype=np.float64))

        if step_vecs is not None and len(step_vecs) >= 2:
            f.create_dataset('step_sizes_abs', data=np.array(step_vecs[0], dtype=np.float64))
            f.create_dataset('step_sizes_ord', data=np.array(step_vecs[1], dtype=np.float64))
            if len(step_vecs) >= 3:
                f.create_dataset('step_sizes_top', data=np.array(step_vecs[2], dtype=np.float64))

        data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        f.create_dataset('data', data=data)
        f.create_dataset('unit_cell', data=np.array([5.0, 5.0, 10.0, 90.0, 90.0, 90.0], dtype=np.float64))

    return filepath


def test_load_discus_nxs_2d_hl_plane(tmp_path):
    """Test loading a 2D HL plane (K is fixed) from a 3D DISCUS file with trailing singleton."""
    p = tmp_path / "test_hl.nxs"
    # Shape: (21, 31, 1). Abs is H (step 0.2), Ord is L (step 0.1), Top is K (fixed at 3.0)
    create_mock_discus_nxs(
        filepath=p,
        shape=(21, 31, 1),
        lower_limits=[-2.0, 3.0, -1.5],
        step_vecs=[[0.2, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]],
    )

    data = load_discus_nxs(str(p))
    assert isinstance(data, NXdata)
    assert data.ndim == 2
    assert data.shape == (21, 31)
    assert data.axes == ['Qh', 'Ql']
    assert data.counts.shape == (21, 31)
    assert np.isclose(data.Qh[0], -2.0)
    assert np.isclose(data.Qh[-1], -2.0 + 20 * 0.2)
    assert np.isclose(data.Ql[0], -1.5)
    assert np.isclose(data.Ql[-1], -1.5 + 30 * 0.1)
    # Check fixed coordinate Qk
    assert 'Qk' in data
    assert np.isclose(float(data.Qk.nxdata), 3.0)

    # Verify that plot_slice handles the resulting NXdata
    qm = plot_slice(data)
    assert qm is not None


def test_load_discus_nxs_2d_hk_plane(tmp_path):
    """Test loading a 2D HK plane (L is fixed)."""
    p = tmp_path / "test_hk.nxs"
    create_mock_discus_nxs(
        filepath=p,
        shape=(15, 25, 1),
        lower_limits=[-1.0, -2.0, 0.0],
        step_vecs=[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]],
    )

    data = load_discus_nxs(str(p))
    assert data.ndim == 2
    assert data.shape == (15, 25)
    assert data.axes == ['Qh', 'Qk']
    assert 'Ql' in data
    assert np.isclose(float(data.Ql.nxdata), 0.0)


def test_load_discus_nxs_2d_kl_plane(tmp_path):
    """Test loading a 2D KL plane (H is fixed)."""
    p = tmp_path / "test_kl.nxs"
    create_mock_discus_nxs(
        filepath=p,
        shape=(10, 20, 1),
        lower_limits=[1.0, -1.0, -2.0],
        step_vecs=[[0.0, 0.2, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]],
    )

    data = load_discus_nxs(str(p))
    assert data.ndim == 2
    assert data.shape == (10, 20)
    assert data.axes == ['Qk', 'Ql']
    assert 'Qh' in data
    assert np.isclose(float(data.Qh.nxdata), 1.0)


def test_load_discus_nxs_3d_volume(tmp_path):
    """Test loading a full 3D DISCUS volume."""
    p = tmp_path / "test_3d.nxs"
    create_mock_discus_nxs(
        filepath=p,
        shape=(10, 12, 14),
        lower_limits=[-1.0, -2.0, -3.0],
        step_vecs=[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]],
    )

    data = load_discus_nxs(str(p))
    assert data.ndim == 3
    assert data.shape == (10, 12, 14)
    assert data.axes == ['Qh', 'Qk', 'Ql']
    assert data.counts.shape == (10, 12, 14)
    assert np.isclose(data.Qh[0], -1.0)


def test_load_discus_nxs_already_squeezed_2d(tmp_path):
    """Test loading a 2D DISCUS file that was already saved with 2D shape."""
    p = tmp_path / "test_2d_squeezed.nxs"
    create_mock_discus_nxs(
        filepath=p,
        shape=(15, 20),
        lower_limits=[-1.0, 0.0, -2.0],
        step_vecs=[[0.1, 0.0, 0.0], [0.0, 0.0, 0.2]],
        step_sizes=[0.1, 0.2],
    )

    data = load_discus_nxs(str(p))
    assert data.ndim == 2
    assert data.shape == (15, 20)
    assert data.axes == ['Qh', 'Ql']


def test_load_discus_nxs_fallback_no_step_vecs(tmp_path):
    """Test fallback when step_sizes_abs/ord/top are absent."""
    p = tmp_path / "test_fallback.nxs"
    with h5py.File(str(p), 'w') as f:
        f.create_dataset('lower_limits', data=np.array([-2.0, 0.0, -3.0], dtype=np.float64))
        f.create_dataset('step_sizes', data=np.array([0.2, 0.0, 0.1], dtype=np.float64))
        f.create_dataset('data', data=np.zeros((10, 15, 1), dtype=np.float64))

    data = load_discus_nxs(str(p))
    assert data.ndim == 2
    assert data.shape == (10, 15)
    assert data.axes == ['Qh', 'Ql']




def test_load_discus_nxs_format_nexus_2d(tmp_path):
    """Test loading a DISCUS file written with 'form nexus' (entry/data/Sq with Qh, Ql)."""
    p = tmp_path / "test_nexus_2d.nxs"
    with h5py.File(str(p), 'w') as f:
        entry = f.create_group('entry')
        data_grp = entry.create_group('data')
        data_grp.attrs['signal'] = 'Sq'
        data_grp.attrs['axes'] = 'Qh:Ql'
        data_grp.create_dataset('Sq', data=np.ones((20, 30), dtype=np.float64))
        data_grp.create_dataset('Qh', data=np.linspace(-1.0, 1.0, 20))
        data_grp.create_dataset('Ql', data=np.linspace(-3.0, 3.0, 30))

    data = load_discus_nxs(str(p))
    assert data.ndim == 2
    assert data.shape == (20, 30)
    assert data.axes == ['Qh', 'Ql']
    assert data.signal == 'counts'
    assert np.isclose(data.Qh[0], -1.0)
    assert np.isclose(data.Ql[-1], 3.0)


def test_load_discus_nxs_format_nexus_3d(tmp_path):
    """Test loading a 3D DISCUS file written with 'form nexus'."""
    p = tmp_path / "test_nexus_3d.nxs"
    with h5py.File(str(p), 'w') as f:
        entry = f.create_group('entry')
        data_grp = entry.create_group('data')
        data_grp.attrs['signal'] = 'Sq'
        data_grp.attrs['axes'] = 'Qh:Qk:Ql'
        data_grp.create_dataset('Sq', data=np.ones((10, 12, 14), dtype=np.float64))
        data_grp.create_dataset('Qh', data=np.linspace(-1.0, 1.0, 10))
        data_grp.create_dataset('Qk', data=np.linspace(-2.0, 2.0, 12))
        data_grp.create_dataset('Ql', data=np.linspace(-3.0, 3.0, 14))

    data = load_discus_nxs(str(p))
    assert data.ndim == 3
    assert data.shape == (10, 12, 14)
    assert data.axes == ['Qh', 'Qk', 'Ql']
    assert np.isclose(data.Qh[0], -1.0)
    assert np.isclose(data.Qk[0], -2.0)
    assert np.isclose(data.Ql[0], -3.0)


def test_load_discus_nxs_off_by_one_counts_extra(tmp_path):
    """Test off-by-one where counts array has 1 extra slice along a dimension."""
    p = tmp_path / "test_off_by_one_counts.nxs"
    with h5py.File(str(p), 'w') as f:
        entry = f.create_group('entry')
        data_grp = entry.create_group('data')
        data_grp.attrs['signal'] = 'Sq'
        data_grp.attrs['axes'] = 'Qh:Qk'
        # Sq has shape (21, 30), but Qh has 20 points (off by 1)
        data_grp.create_dataset('Sq', data=np.zeros((21, 30), dtype=np.float64))
        data_grp.create_dataset('Qh', data=np.linspace(-2.0, 2.0, 20))
        data_grp.create_dataset('Qk', data=np.linspace(-3.0, 3.0, 30))

    with pytest.warns(UserWarning, match="off-by-one"):
        data = load_discus_nxs(str(p))

    assert data.ndim == 2
    assert data.shape == (20, 30)
    assert data.axes == ['Qh', 'Qk']
    assert len(data.Qh) == 20
    assert len(data.Qk) == 30


def test_load_discus_nxs_off_by_one_axis_extra(tmp_path):
    """Test off-by-one where axis has 1 extra point compared to counts array."""
    p = tmp_path / "test_off_by_one_axis.nxs"
    with h5py.File(str(p), 'w') as f:
        entry = f.create_group('entry')
        data_grp = entry.create_group('data')
        data_grp.attrs['signal'] = 'Sq'
        data_grp.attrs['axes'] = 'Qh:Qk'
        # Sq has shape (20, 30), but Qk has 31 points (off by 1)
        data_grp.create_dataset('Sq', data=np.zeros((20, 30), dtype=np.float64))
        data_grp.create_dataset('Qh', data=np.linspace(-2.0, 2.0, 20))
        data_grp.create_dataset('Qk', data=np.linspace(-3.0, 3.0, 31))

    with pytest.warns(UserWarning, match="off-by-one"):
        data = load_discus_nxs(str(p))

    assert data.ndim == 2
    assert data.shape == (20, 30)
    assert data.axes == ['Qh', 'Qk']
    assert len(data.Qh) == 20
    assert len(data.Qk) == 30

