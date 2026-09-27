---
name: discus-simulation
description: >-
  Running DISCUS simulations, generating crystal structures and supercells, performing Fourier transforms for diffuse scattering (2D reciprocal space slices and 3D volumes), exporting to NeXus/HDF5 formats, and integrating with nxs-analysis-tools.
---

# DISCUS Simulation & Operation Guide (`discus-simulation`)

This skill documents how to operate the DISCUS suite (part of the Diffuse software package by R. B. Neder and T. Proffen) on this machine, create and run simulation macros, generate reciprocal space scattering maps (2D planes and 3D volumes), and import the outputs into `nxs-analysis-tools`.

---

## 1. DISCUS Installation & Executables

On this machine:
- **Primary Suite Binary**:
  ```bash
  $HOME/bin/discus_suite
  ```
  *(Alternative build path: `/Users/stevengomezalvarado/DIFFUSE_INSTALL/develop/DiffuseBuild/suite/prog/discus_suite`)*
- **User Scripts & Installations**:
  - `/Users/stevengomezalvarado/DIFFUSE_INSTALL/`
- **Manuals**:
  - `/Users/stevengomezalvarado/DIFFUSE_INSTALL/develop/DiffuseCode/Manual/discus_man.pdf`

### Running DISCUS Non-Interactively
`discus_suite` starts at the `SUITE` interactive command prompt. To run scripts or batch simulations without hanging:
```bash
cat << 'EOF' | $HOME/bin/discus_suite
discus
@my_macro.mac
exit
exit
EOF
```
Or pipe command sequences:
```bash
printf "discus\n@my_macro.mac\nexit\nexit\n" | $HOME/bin/discus_suite
```

---

## 2. Common DISCUS Workflows

### A. Creating or Reading Crystal Structures
DISCUS supports `.cell`, `.stru`, `.cif`, and `.pdb` formats:
```text
discus
  read
    cell my_cell.cell, 20, 20, 20   # Expands unit cell into a 20x20x20 supercell
```
Example `.cell` file format:
```text
title my_crystal
spcgr P1
cell 5.4949, 5.4949, 9.3085, 90.0, 90.0, 120.0
atoms x, y, z, Biso, Property, MoleNo, MoleAt, Occ
V  0.0, 0.5, 0.5, 0.0003, 1, 0, 0, 1.0
CS 0.0, 0.0, 0.0, 0.0003, 1, 0, 0, 1.0
```

### B. Calculating Fourier Transforms (Diffuse Scattering)
The `fourier` menu calculates reciprocal space scattering:
```text
fourier
  xray                              # Radiation: 'xray', 'neutron', or 'electron'
  ll -6.0, 3.0, -10.0               # Lower left corner (h, k, l)
  lr  6.0, 3.0, -10.0               # Lower right corner (h, k, l)
  ul -6.0, 3.0,  10.0               # Upper left corner (h, k, l)
  tl -6.0, 3.0,  10.0               # Top left corner (only needed for 3D)

  na 251                            # Number of grid points along abscissa
  no 251                            # Number of grid points along ordinate
  nt 1                              # Number of grid points along top (1 for 2D, >1 for 3D)

  abs h                             # Physical axis for abscissa (h, k, or l)
  ord l                             # Physical axis for ordinate (h, k, or l)
  top k                             # Physical axis for top direction

  temp use                          # Include thermal displacement parameters
  disp off                          # Exclude anomalous dispersion if not needed

  # Averaging / Lots:
  set aver, 50                      # Subtract average structure scattering (e.g. 50% sample)
  lots box, 10, 10, 10, 50, yes     # Lot calculation over 50 random sub-boxes of 10x10x10

  run
exit
```

### C. Exporting to NeXus / HDF5 Format
To export the scattering intensities to an HDF5 `.nxs` file readable by `nxs_analysis_tools.load_discus_nxs`:
```text
output
  outfile my_simulation.nxs
  value inte                        # Intensity (or 'ampl', 'phas', etc.)
  form hdf5                         # Format must be 'hdf5'
  run
exit
```

---

## 3. DISCUS NeXus File Structure

When DISCUS exports HDF5 files (`form hdf5`), it writes the following datasets to the root group:
- `data`: 3D array of double-precision intensities with dimensions:
  `shape == (n_abs, n_ord, n_top)`.
  - For 3D volumes: `(N_h, N_k, N_l)` (or permuted according to `abs`, `ord`, `top`).
  - For 2D slices: One dimension has size 1 (e.g. `(251, 251, 1)` for an $HL$ plane).
- `lower_limits`: Array of 3 floats `[H_min, K_min, L_min]`. Always indexed $[H, K, L]$.
- `step_sizes`: Array of 3 floats `[step_abs, step_ord, step_top]`.
- `step_sizes_abs`: 3D vector `[dH, dK, dL]` for the abscissa (dimension 0).
- `step_sizes_ord`: 3D vector `[dH, dK, dL]` for the ordinate (dimension 1).
- `step_sizes_top`: 3D vector `[dH, dK, dL]` for the top axis (dimension 2).
- `unit_cell`: Lattice parameters `[a, b, c, alpha, beta, gamma]`.
- `is_direct`: `0` for reciprocal space scattering, `1` for direct space (PDF / 3D-PDF).
- `PROGRAM`: `'DISCUS60'`.
- `format`: `'Yell 1.0'`.

---

## 4. Integration with `nxs-analysis-tools`

Use [`load_discus_nxs(path)`](file:///Users/stevengomezalvarado/nxs_analysis_tools/src/nxs_analysis_tools/datareduction.py) to import DISCUS `.nxs` files:
```python
from nxs_analysis_tools import load_discus_nxs, plot_slice

# 2D plane (e.g., HL plane with fixed K)
data_2d = load_discus_nxs("mmc_output_adjusted.nxs")
# Result is 2D NXdata with axes ('H', 'L') and fixed coordinate K stored as an attribute/field
plot_slice(data_2d)

# 3D volume
data_3d = load_discus_nxs("mmc_output_adjusted_3D.nxs")
# Result is 3D NXdata with axes ('H', 'K', 'L')
plot_slice(data_3d[:, 0.0, :])  # Slice HL plane at K=0.0
```
