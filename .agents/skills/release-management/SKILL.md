---
name: release-management
description: >-
  Procedures for managing tags and releases for nxs-analysis-tools. Use when asked to tag a new version, create an alpha/rc pre-release, or publish a GitHub release.
---

# Release Management Workflow (`release-management`)

This document defines the versioning, tagging, and release policies for `nxs-analysis-tools`.

## 1. Versioning Conventions (PEP 440 & SemVer)
The package uses `setuptools_scm` to dynamically determine version numbers from git tags.
- **Stable Releases**: `vX.Y.Z` (e.g. `v0.1.15`, `v0.1.16`).
- **Pre-releases (Alpha / RC)**: `vX.Y.ZaN` or `vX.Y.ZrcN` (e.g. `v0.1.16a0`, `v0.1.16rc1`).

## 2. Pre-Releases vs. Official GitHub Releases Policy
- **Alpha & RC Tags**:
  - Generated as annotated git tags (e.g. `git tag -a v0.1.16a0 -m "Release v0.1.16a0"`).
  - Pushed to GitHub (`git push origin v0.1.16a0`), which automatically triggers the `Upload Python Package` GitHub Action workflow to build and publish wheels to PyPI.
  - **Do NOT create a formal GitHub Release** on GitHub for alpha or RC tags. This keeps the public GitHub Releases tab and sidebar clean and focused only on stable milestones.
- **Stable Releases**:
  - Tagged on `main` after all PRs and checks have merged and verified.
  - Published as an official **GitHub Release** via `gh release create vX.Y.Z` with release notes and full changelog comparison link.

## 3. Pre-Release Workflow (Alpha / RC)
1. Verify all PRs targeting the pre-release are merged into `main`.
2. Update local `main`:
   ```bash
   git checkout main
   git pull origin main
   ```
3. Run the full test suite and verify notebooks:
   ```bash
   pytest
   pytest tests/test_notebooks.py
   ```
4. Create the annotated tag and push:
   ```bash
   git tag -a v0.1.16a0 -m "Release v0.1.16a0"
   git push origin v0.1.16a0
   ```
5. Monitor the GitHub Action run to ensure publishing to PyPI succeeds.
6. Collaborators can now install via:
   ```bash
   pip install nxs-analysis-tools==0.1.16a0
   ```
   *(Note: `pip install` without `--pre` or exact version string ignores alpha/RC tags by default).*

## 4. Stable Release Workflow
1. Ensure all pre-release testing and documentation fixes are merged into `main`.
2. Update local `main`:
   ```bash
   git checkout main
   git pull origin main
   ```
3. Create the stable tag:
   ```bash
   git tag -a v0.1.16 -m "Release v0.1.16"
   git push origin v0.1.16
   ```
4. **Inspect Commits & Generate Release Notes File**:
   - Query all commits since the previous stable release tag:
     ```bash
     PREV_TAG=$(git tag -l "v[0-9]*.[0-9]*.[0-9]*" --sort=-v:refname | sed -n 2p)
     git log ${PREV_TAG}..HEAD --oneline
     ```
   - Generate a markdown release notes file `scratch/release_notes_<tag>.md` using the project's standard structure:
     ```bash
     cat << 'EOF' > scratch/release_notes_v0.1.16.md
     <High-level summary of the release>

     ## What's Changed
     * **Feature / Category**:
       * Detail bullet points describing changes (#PR or commit reference)
     * **Bugfixes & Maintenance**:
       * Detail bullet points describing fixes

     **Full Changelog**: https://github.com/stevenjgomez/nxs_analysis_tools/compare/<prev_tag>...<new_tag>
     EOF
     ```
     *(Note: Alternatively, `gh release create <tag> --generate-notes` can be used to pull GitHub PR titles automatically).*

5. **Publish the Official GitHub Release**:
   ```bash
   gh release create v0.1.16 --title "v0.1.16" --notes-file scratch/release_notes_v0.1.16.md
   ```

