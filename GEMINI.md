# Development Rules for nxs-analysis-tools

## 1. Context & Codebase Inspection
- **Do NOT read the entire codebase** into context.
- Always consult the ReadTheDocs documentation first (online at `https://nxs-analysis-tools.readthedocs.io/en/stable/` or local `docs/source/api.rst`).
- Identify the target module (`datareduction`, `chess`, `fitting`, `pairdistribution`) and inspect only the specific function, class, or lines needed for the task.

## 2. Git & Branching Strategy
- Base branch is `main`.
- Naming conventions:
  - Bugfixes: `bugfix/issue-<id>-<description>` or `bugfix/<description>`
  - Features: `feature/<description>`
  - Documentation: `docs/<description>`
- Never commit directly to `main`.
- Changes must be proposed via a Pull Request into `main` using GitHub CLI (`gh pr create`).
- **Commit Attribution**: All AI-assisted commits must include the following trailer in the commit message body:
  ```git
  Co-authored-by: Antigravity <noreply@google.com>
  ```

## 3. Testing Standards
- All bugfixes and new features must be accompanied by unit tests in `tests/`.
- Run tests using `pytest`.
- Use synthetic `NXdata` (with `numpy` arrays and `nexusformat.nexus` classes: `NXdata`, `NXfield`, `NXentry`, `NXroot`) or built-in pooch datasets (`nxs_analysis_tools.datasets`).
- For graphical/plotting functions, enforce headless mode:
  ```python
  import matplotlib
  matplotlib.use('Agg')
  import builtins
  builtins.display = lambda *args, **kwargs: None
  ```
- Regression test rule: For any bugfix, write a test that fails before the fix is applied and passes afterward.

## 4. Documentation & Jupyter Notebooks
- Keep docstrings in NumPy format (`numpydoc`).
- Preserve existing comments and docstrings.
- Documentation uses Sphinx with `myst-nb` (`nb_execution_mode = 'cache'`, `nb_execution_raise_on_error = True`).
- All notebooks in `docs/source/examples/` must execute without errors.
- Never edit `.ipynb` files with raw string replacement or regex. Always use `nbformat` in a script or helper to preserve notebook schema integrity.
- When adding a new notebook, register it in `docs/source/examples/index.md`.

## 5. Typing & Stubs
- When changing public function signatures, run:
  ```bash
  stubgen src/nxs_analysis_tools
  ```
