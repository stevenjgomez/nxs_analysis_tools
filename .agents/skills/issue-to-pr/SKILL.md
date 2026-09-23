---
name: issue-to-pr
description: >-
  Automates the complete workflow from a GitHub issue or feature request to an opened Pull Request for nxs-analysis-tools. Use when instructed to "Fix issue #...", "Implement feature ...", or when asked to prepare a PR.
---

# Issue to Pull Request Workflow (`issue-to-pr`)

Follow this step-by-step procedure to resolve an issue or implement a feature and submit a Pull Request.

## Step 1: Ingest Requirements
- If an issue number is specified (e.g., `#101`), retrieve the issue information using the GitHub CLI:
  ```bash
  gh issue view <issue_number> --json number,title,body,comments
  ```
- If `gh` is not yet authenticated or the user provided the description directly in the prompt, parse the title and requirements from the user's message.

## Step 2: Create a Dedicated Branch
- Ensure working directory is clean and up to date with `main`:
  ```bash
  git checkout main
  git pull origin main
  ```
- Create and switch to a descriptive branch based on conventions:
  - For bugfixes: `git checkout -b bugfix/issue-<id>-<short_desc>`
  - For features: `git checkout -b feature/<feature_name>`

## Step 3: Targeted API & Documentation Review
- Do **not** sweep or read the whole codebase.
- Consult `docs/source/api.rst` or the ReadTheDocs documentation (`https://nxs-analysis-tools.readthedocs.io/en/stable/`) to identify which submodule owns the behavior:
  - `datareduction.py`: Slicing, 2D plotting (`plot_slice`), masking, cuts, line cuts, scissors.
  - `chess.py`: CHESS beamline data loading, temperature dependence series.
  - `fitting.py`: Peak and linecut fitting models (`lmfit` integrations).
  - `pairdistribution.py`: Delta-PDF transformations, FFT, symmetrization.
  - `datasets.py`: Pooch remote data loaders.
- View only the specific class or function in `src/nxs_analysis_tools/` using targeted line ranges.

## Step 4: Write Unit Tests First
- Add a test in `tests/test_<module>.py`:
  - For bugfixes: write a test that reproduces the bug and fails before the fix.
  - For features: write unit tests checking expected return types, argument handling, and edge cases.
- Use synthetic `NXdata` or `nxs_analysis_tools.datasets` data loaders.
- Ensure Matplotlib is run headlessly (`builtins.display = lambda *args, **kwargs: None`).
- Run `pytest` to confirm test behavior.

## Step 5: Implement Fix or Feature
- Edit the target module surgically, keeping comments, typing, and `numpydoc` docstrings intact.
- Re-run `pytest` to confirm the test passes and existing tests are not broken.

## Step 6: Synchronize Docs, Tutorials & Stubs
- If public signatures or behavior changed:
  - Update `docs/source/api.rst` or docstrings if needed.
  - If a tutorial in `docs/source/examples/` uses the changed API, update it using `nbformat`.
  - Run `stubgen src/nxs_analysis_tools` to update typing stubs if applicable.

## Step 7: Push Branch and Open PR
- Stage modified files and commit with a standard semantic commit message and the Co-authored-by trailer:
  ```bash
  git add src/ tests/ docs/
  git commit -m "fix(<module>): <description> (fixes #<issue_number>)

Co-authored-by: Antigravity <noreply@google.com>"
  ```
- Push the branch to remote:
  ```bash
  git push -u origin <branch_name>
  ```
- Create the Pull Request:
  ```bash
  gh pr create --title "<Type>: <Summary>" --body "Closes #<issue_number>. <Description of changes and verification steps>." --base main
  ```
- Report the PR URL and a concise summary back to the user.
