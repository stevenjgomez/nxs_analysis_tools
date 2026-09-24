---
name: nexus-chess-formats
description: >-
  NeXus and HDF5 data structures, CHESS beamline directory conventions, lazy loading mechanics with NXlink/NXfield, temperature key normalization, and XTEC data integration for nxs-analysis-tools.
---

# NeXus & CHESS Data Formats (`nexus-chess-formats`)

This skill documents domain knowledge, file structure conventions, lazy loading behaviors, and testing patterns for NeXus data in `nxs-analysis-tools`.

---

## 1. Directory Structures & Format Detection

`nxs-analysis-tools` primarily works with two distinct beamline data formats. **NXRefine and legacy CHESS formats are never mixed in the same sample directory.**

### A. NXRefine Format
- **Directory Layout**:
  ```text
  sample_directory/
  ├── cubic_15.nxs          <-- Top-level wrapper NeXus file
  ├── cubic_25.nxs
  ├── 15/
  │   └── transform.nxs     <-- Raw reciprocal space volume
  └── 25/
      └── transform.nxs
  ```
- **NeXus Schema**:
  - Top-level wrapper (`cubic_15.nxs`) contains an `NXentry` named `entry` and an `NXdata` group named `transform`.
  - The signal dataset `transform.data` is an `NXlink` pointing to `<temperature>/transform.nxs` at target `/entry/data/v`.
  - Axes are ordered `[Ql, Qk, Qh]` (C-contiguous reciprocal space coordinates).
- **Detection Pattern**: Files matching `*_(\d+(?:[p.]\d+)?)\.nxs` directly in `sample_directory`.

### B. Legacy CHESS Format
- **Directory Layout**:
  ```text
  sample_directory/
  ├── 15/
  │   └── data_hkli.nxs     <-- Data file ending with 'hkli.nxs' (or custom)
  ├── 25/
  │   └── data_hkli.nxs
  ```
- **NeXus Schema**:
  - The file contains `entry/data` with signal `counts` and axes `H`, `K`, `L`.
- **Detection Pattern**: Subdirectories named `<temperature>` containing `.nxs` files matching the specified `file_ending` (default `'hkli.nxs'`).

---

## 2. Lazy Loading Mechanics in `nexusformat`

Understanding `nexusformat`'s memory model is essential to avoiding out-of-memory errors when processing multi-temperature 3D volumes.

### Why `nxload` is Lazy
When `nexusformat.nexus.nxload(path)` is called, the file structure is traversed, but **underlying array data is not loaded into RAM**:
- An `NXfield` or `NXlinkfield` backed by an HDF5 dataset initializes with `_value = None`.
- Accessing `.nxdata` forces an eager read of the entire multidimensional array into memory.
- Slicing with `data[slice_obj]` reads **only the requested hyperslab** directly from disk via HDF5 chunking, keeping the rest of the array unloaded.

### Avoiding Eager Loads in NXRefine
- When loading NXRefine data, always use `use_nxlink=True` (default in `load_transform` and `load_datasets`).
- Setting `use_nxlink=False` calls `.nxdata.transpose(2, 1, 0)`, which pulls the entire 1–2 GB volume per temperature into memory upfront.
- With `use_nxlink=True`, `root.entry.transform` retains the `NXlink` pointer, using $O(1)$ memory until sliced (e.g., via `Scissors.cut_data()`).

### Legacy CHESS Datasets Are Already Lazy
- Datasets loaded via `load_data()` return `g.entry.data`.
- Because `load_data()` never accesses `.nxdata`, legacy datasets are natively lazy and never load full 3D arrays into RAM upfront.

---

## 3. Temperature Key Conventions & Normalization

Beamline files encode temperatures in three equivalent notations:
1. **Integer**: `15`, `300`
2. **Float**: `15.5`, `25.0`
3. **Filesystem 'p' Notation**: `'15p5'` (used in filenames to avoid decimal points)

### Normalization Pattern
Use `_normalize_temperature(val)` from `chess.py`:
- Converts `'15p5'` $\rightarrow$ `15.5`
- Converts `'15.0'` $\rightarrow$ `15` (whole floats to int)
- Converts `15.5` $\rightarrow$ `15.5`

### Transparent Access via `TempDict`
`TempDependence.datasets` and related containers use `TempDict`:
- Any temperature can be accessed using `int`, `float`, or `str` format interchangeably:
  ```python
  td.datasets[15.5] == td.datasets['15.5'] == td.datasets['15p5']
  ```
- `td.temperatures` is a sorted list of numeric keys (`int` if whole, `float` otherwise).
- Heatmap and order parameter plots preserve floating point precision (never cast temperature keys to `int` when plotting).

---

## 4. Multi-Dataset Stacking & XTEC Export

`TempDependence.to_xtec()` compiles 3D temperature datasets into a 4D `NXdata` structure:
- **Axis 0**: Temperature (`Te`, units `K`).
- **Axes 1, 2, 3**: Spatial reciprocal space axes (`Qh`, `Qk`, `Ql`).
- When constructing stacked NeXus objects, always copy attributes (`units`, `long_name`, `angles`) from the underlying fields.

---

## 5. Synthetic Testing Fixtures

When writing unit tests in `tests/`, use synthetic NeXus structures to avoid network downloads or large binary dependencies:

```python
from nexusformat.nexus import NXroot, NXentry, NXdata, NXfield, NXlink, nxsave
import numpy as np

def create_synthetic_legacy_file(filepath):
    root = NXroot()
    root['entry'] = NXentry()
    h = NXfield(np.linspace(0, 1, 10), name='H')
    k = NXfield(np.linspace(0, 1, 10), name='K')
    l = NXfield(np.linspace(0, 1, 10), name='L')
    counts = NXfield(np.ones((10, 10, 10), dtype=np.float32), name='counts')
    root['entry']['data'] = NXdata(counts, (h, k, l))
    nxsave(filepath, root)

def create_synthetic_nxrefine_structure(tmp_path, temp=15):
    # Raw transform file
    subfolder = tmp_path / str(temp)
    subfolder.mkdir(parents=True, exist_ok=True)
    raw_path = subfolder / 'transform.nxs'
    
    raw_root = NXroot()
    raw_root['entry'] = NXentry()
    raw_root['entry']['data'] = NXdata(NXfield(np.ones((10, 10, 10), dtype=np.float32), name='v'))
    nxsave(str(raw_path), raw_root)
    
    # Top-level wrapper file
    wrapper_path = tmp_path / f"sample_{temp}.nxs"
    w = NXroot()
    w['entry'] = NXentry()
    w['entry']['transform'] = NXdata(
        NXlink(name='data', target='/entry/data/v', file=f"{temp}/transform.nxs"),
        (NXfield(np.arange(10), name='Ql'), NXfield(np.arange(10), name='Qk'), NXfield(np.arange(10), name='Qh'))
    )
    nxsave(str(wrapper_path), w)
    return wrapper_path
```
