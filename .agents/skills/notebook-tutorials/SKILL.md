---
name: notebook-tutorials
description: >-
  Procedures for safely creating, updating, and verifying Jupyter notebook tutorials (.ipynb) in docs/source/examples/ for nxs-analysis-tools. Use when editing tutorials, adding new examples, or fixing broken documentation notebooks.
---

# Notebook Tutorials Workflow (`notebook-tutorials`)

All tutorials in `docs/source/examples/` are rendered and executed by Sphinx using `myst-nb`. Because `nb_execution_raise_on_error = True` in `docs/source/conf.py`, any failing cell in any notebook causes the documentation build to fail.

## 1. Safe Notebook Editing with `nbformat`

Never use direct text replacement or regular expressions to edit `.ipynb` files. Raw text edits can easily invalidate the JSON schema, corrupt cell IDs, or invalidate output payloads.

Instead, write a quick Python script (e.g. in `scratch/`) using `nbformat`:

```python
import nbformat as nbf

nb_path = "docs/source/examples/using_scissors.ipynb"
nb = nbf.read(nb_path, as_version=4)

# Example: iterate and update a specific code cell
for cell in nb.cells:
    if cell.cell_type == "code" and "old_function_call(" in cell.source:
        cell.source = cell.source.replace("old_function_call(", "new_function_call(")

# Example: append a new explanation and code cell
md_cell = nbf.v4.new_markdown_cell("### Applying the New Mask")
code_cell = nbf.v4.new_code_cell("scissors.apply_mask(custom_mask)")
nb.cells.extend([md_cell, code_cell])

nbf.write(nb, nb_path)
```

## 2. Creating New Examples

When asked to generate a new tutorial:
1. **Choose an Example Name**: Use snake_case matching existing tutorials (e.g., `using_delta_pdf_3d.ipynb`).
2. **Use Built-in Pooch Datasets**: Do not assume large local files exist. Import datasets from `nxs_analysis_tools.datasets`:
   - `vacancies()` and `vacanciesfft()`
   - `hexagonal(temperatures=[15, 300])`
   - `cubic(temperatures=[15, 300])`
   - `orthorhombic(temperatures=[15, 100, 300])`
   - `cubic_l_rods()`
3. **Draft the Notebook**: Build standard v4 notebook structure with introductory markdown cells, step-by-step code blocks, and visual plots using `plot_slice`.
4. **Register in Sphinx Index**:
   Add the tutorial filename (without `.ipynb` extension) to `docs/source/examples/index.md` under the `{toctree}` directive.

## 3. NXdata Slicing: Integer vs. Float Indices

`nexusformat.nexus.NXdata` objects interpret integer and float indices differently:
- **Integer index** (e.g. `data[:, 0, :]`): selects the **0th array index** along that axis (the first entry in the underlying array, which may correspond to the minimum coordinate such as $K = -2.0$).
- **Float index** (e.g. `data[:, 0.0, :]`): selects by **physical coordinate value** along the axis (locating the bin closest to physical coordinate $K = 0.0$).

Always use float literals (e.g., `0.0`, `1.5`) when intending to slice by physical reciprocal space coordinates.

## 4. Temperature Parameter Conventions & Loading

When writing or updating tutorial examples that load temperature series:
- **Use `load_datasets()`**: Prefer the unified `sample.load_datasets()` method over deprecated `load_transforms()`.
- **Use Numeric Keys**: Prefer numeric values in temperature lists:
  ```python
  sample.load_datasets(temperatures=[15, 300])
  ```
  Note that strings (e.g. `'15.5'`, `'15p5'`) are also supported.
- **Filtering with Logical Operators**: Showcase discovering available temperatures and filtering with list comprehensions:
  ```python
  sample.find_temperatures()
  temps_below_150 = [T for T in sample.temperatures if T < 150]
  sample.load_datasets(temperatures=temps_below_150)
  ```
- **Parameter Name**: Always use `temperatures` (the parameter `temperatures_list` is deprecated).

## 5. Verifying Notebooks Locally

### Headless Display Configuration
When executing notebooks programmatically or in test scripts, configure Matplotlib for headless operation to prevent blocking GUI backend errors:
```python
import matplotlib
matplotlib.use('Agg')
import builtins
builtins.display = lambda *args, **kwargs: None
```

### Option A: Pytest Suite (Fastest & Most Informative)
Run the automated notebook test suite using `pytest`:
```bash
# Test all tutorial notebooks
pytest tests/test_notebooks.py

# Test a specific tutorial notebook
pytest tests/test_notebooks.py -k using_scissors
```
If any cell fails, `pytest` will output the exact failing cell, traceback, and standard output.

### Option B: Sphinx & myst-nb Build
To ensure the notebook renders and builds cleanly in Sphinx documentation:
```bash
sphinx-build -b html docs/source _build/html
```
To force Sphinx to re-execute all notebooks without using cache:
```bash
sphinx-build -E -b html docs/source _build/html
```
If any cell raises an unhandled exception, `sphinx-build` will output the exact traceback and cell number. Fix the exception before committing.

