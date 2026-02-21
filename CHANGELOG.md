# Changelog

All notable changes to the LIFE E2E analytical model are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] — 2026-02-20

### Added
- **Module 1** (`m1_fiber_coupling.py`): Fiber coupling analysis — top-hat and
  Gaussian illumination, Maréchal approximation, Zernike aberrations.
- **Module 2** (`m2_throughput_chain.py`): Surface-by-surface throughput chain
  across ~20 optical elements of the modified Mach–Zehnder combiner.
- **Module 3** (`m3_null_error_propagation.py`): Analytical null depth budget
  from nine physical error sources (OPD, intensity, polarisation, WFE, BS
  chromaticity, pointing, shear).
- **Module 4** (`m4_surface_sensitivity.py`): Per-surface WFE sensitivity
  ranking and tolerance analysis.
- **Monte Carlo** (`monte_carlo.py`): 10⁵-realisation end-to-end integration
  combining Modules 1–4 with full surface catalogue.
- **Supporting libraries**: `material_properties.py` (optical constants for Au,
  CaF₂, ZnSe, KBr, Si:As, AR coatings, fiber), `fiber_modes.py` (V-parameter,
  MFR, overlap integrals, spatial filtering).
- Regression test suite (33 tests across all modules).
- Demo Jupyter notebook (`notebooks/demo_quick.ipynb`).
- GitHub Actions CI (Python 3.10, 3.11, 3.12).
- Zenodo DOI: [10.5281/zenodo.18716470](https://doi.org/10.5281/zenodo.18716470).

### Reference
Companion paper: V. Huarcaya, "Analytical Throughput, Null Depth, and Surface
Tolerance Budget for the LIFE Nulling Interferometer Combiner," *Journal: TBD* (2026, in preparation).
