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
3. **Draft the Notebook**: Build standard v4 notebook structure with introductory markdown cells, step-by-step code blocks, and visual plots using `plot_slice`.
4. **Register in Sphinx Index**:
   Add the tutorial filename (without `.ipynb` extension) to `docs/source/examples/index.md` under the `{toctree}` directive.

## 3. Verifying Notebooks Locally

To ensure the notebook runs cleanly through Sphinx and `myst-nb`:
```bash
sphinx-build -b html docs/source _build/html
```
If any cell raises an unhandled exception, `sphinx-build` will output the exact traceback and cell number. Fix the exception before committing.
