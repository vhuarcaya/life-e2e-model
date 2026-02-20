# LIFE E2E Nulling Interferometer — Analytical Beam Propagation Model

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

End-to-end analytical wavefront propagation model for the [Large Interferometer For Exoplanets (LIFE)](https://www.life-space-mission.com) nulling combiner.
This code traces photon throughput and null depth through every optical surface of the LIFE modified Mach–Zehnder (MMZ) beam combiner across the 6–16 µm mid-infrared science band.

**Companion paper:**  
V. Huarcaya, "Analytical Throughput, Null Depth, and Surface Tolerance Budget for the LIFE Nulling Interferometer Combiner," *Astronomy & Astrophysics* (2026, in preparation).

---

## Scientific Context

LIFE is a proposed ESA large-class space mission consisting of five formation-flying spacecraft at L2 that would use nulling interferometry to detect and characterise the thermal emission of temperate exoplanets around Sun-like stars.
The LIFE concept study ([Quanz et al. 2022](https://doi.org/10.1051/0004-6361/202140366); [Glauser et al. 2024](https://doi.org/10.1117/12.3019318)) identified the need for a full end-to-end wavefront propagation model to assess diffraction and wavefront error propagation through the instrument optical train.

This codebase provides that model — a **Phase I warm-bench, pre-optimisation analytical baseline** that assumes:

- 300 K surface quality specifications (no cryogenic improvement)
- No adaptive nulling correction
- Statistically independent surface errors (worst-case)

The model establishes quantitative performance floors that the NICE (Nulling Interferometry Cryogenic Experiment) cryogenic testbed at ETH Zürich will improve upon.

### Key Findings

- **Photon conversion efficiency:** 7.0–7.7% across 6–16 µm (within the 3.5–10% design range)
- **Surface WFE dominates null depth** — not beamsplitter chromaticity — due to quartic σ⁴/λ⁴ fiber-filtered scaling accumulated across ~20 optical surfaces
- **Technology gap:** 140× at 6 µm, 6–20× at 10 µm (warm-bench), requiring surface improvements of only 2.1–3.4× thanks to the same quartic scaling
- **Validated** against NICE warm-bench testbed measurements (Birbacher et al. 2026)

---

## Module Architecture

The model consists of **4 analysis modules**, **2 supporting libraries**, and **1 integration engine** (~8,300 lines total):

```
material_properties.py          fiber_modes.py
   (optical constants)          (Gaussian modes)
        │                            │
        ├────────────┬────────────────┤
        │            │                │
        ▼            ▼                ▼
  m2_throughput   m3_null_error    m1_fiber_coupling
    _chain.py      _propagation.py    .py
                     │
                     ▼
               m4_surface
                _sensitivity.py

  material_properties ──┬──► monte_carlo.py
  m3_null_error_prop ───┘    (10⁵-realisation MC)
```

### Supporting Libraries

| Module | Lines | Description |
|--------|------:|-------------|
| `material_properties.py` | 1,143 | Canonical library: Au reflectivity (Ordal/Palik), CaF₂ & ZnSe Sellmeier models, AR/BS coatings, Si:As BIB detector QE. 30 public functions. |
| `fiber_modes.py` | 882 | Single-mode fiber optics: V-parameter, Gaussian mode field radius w_f(λ), coupling integrals for Gaussian and top-hat illumination. |

### Analysis Modules

| Module | Lines | Paper Section | Description |
|--------|------:|:---:|-------------|
| `m1_fiber_coupling.py` | 1,475 | §3.1 | Fiber coupling efficiency vs beam profile, Zernike aberration sensitivity, optimal β parameter. |
| `m2_throughput_chain.py` | 1,194 | §3.2 | Surface-by-surface throughput waterfall for 24-element LIFE optical train. NICE testbed validation. |
| `m3_null_error_propagation.py` | 1,368 | §3.3 | Null depth error budget: OPD, intensity mismatch, beamsplitter chromaticity, polarisation. Per-polarisation exact computation. |
| `m4_surface_sensitivity.py` | 1,076 | §3.4 | Surface-by-surface WFE sensitivity ranking with Maréchal coupling and common-mode rejection. 20-surface catalogue. |

### Integration Engine

| Module | Lines | Paper Section | Description |
|--------|------:|:---:|-------------|
| `monte_carlo.py` | 1,169 | §3.5 | Full end-to-end 10⁵-realisation Monte Carlo combining Modules 1–4. Null depth distributions, CDF analysis, technology gap assessment. |

### Wavelength Conventions

Two conventions coexist by design, with documented boundary wrappers:

- **µm group** (libraries + MC): `material_properties`, `fiber_modes`, `m2_throughput_chain`, `monte_carlo`
- **SI metres group** (physics modules): `m1_fiber_coupling`, `m3_null_error_propagation`, `m4_surface_sensitivity`

---

## Installation

### Requirements

- Python ≥ 3.10
- NumPy, SciPy, Matplotlib (no other dependencies)

```bash
git clone https://github.com/vhuarcaya/life-e2e-model.git
cd life-e2e-model
pip install -r requirements.txt
```

### Running the Modules

Each module is self-contained and can be run independently:

```bash
# Run individual analysis modules
python src/life_e2e/m1_fiber_coupling.py
python src/life_e2e/m2_throughput_chain.py
python src/life_e2e/m3_null_error_propagation.py
python src/life_e2e/m4_surface_sensitivity.py

# Run the full Monte Carlo (10⁵ realisations, ~2 min)
python src/life_e2e/monte_carlo.py
```

Each module generates its paper figures in the working directory and prints summary tables to stdout.

---

## Reproducing Paper Results

To regenerate all figures and tables from the paper, run all modules in sequence:

```bash
python src/life_e2e/m1_fiber_coupling.py
python src/life_e2e/m2_throughput_chain.py
python src/life_e2e/m3_null_error_propagation.py
python src/life_e2e/m4_surface_sensitivity.py
python src/life_e2e/monte_carlo.py
```

Key numerical results to verify:

| Quantity | Value | Module |
|----------|-------|--------|
| Top-hat coupling η₀ | 81.45% | M1 |
| PCE at 10 µm | ~7.1% | M2 |
| BS chromatic null (6 µm) | 2.0 × 10⁻⁵ | M3 |
| MC mean null (6 µm) | 1.4 × 10⁻³ | MC |
| MC mean null (10 µm) | 1.9 × 10⁻⁴ | MC |
| Technology gap (6 µm) | 140× | MC |

---

## Repository Structure

```
life-e2e-model/
├── README.md
├── LICENSE                        # GPL-3.0
├── CITATION.cff                   # Machine-readable citation
├── requirements.txt
├── .gitignore
├── src/
│   └── life_e2e/
│       ├── __init__.py
│       ├── material_properties.py     # Optical constants library
│       ├── fiber_modes.py             # Fiber mode library
│       ├── m1_fiber_coupling.py       # Module 1: Coupling
│       ├── m2_throughput_chain.py     # Module 2: Throughput
│       ├── m3_null_error_propagation.py  # Module 3: Null depth
│       ├── m4_surface_sensitivity.py  # Module 4: Surface WFE
│       └── monte_carlo.py            # Monte Carlo integration
├── figures/                       # Generated paper figures
├── tests/
└── docs/
```

---

## Citation

If you use this code in your research, please cite both the paper and the software:

```bibtex
@article{huarcaya2026_life_e2e,
  author  = {Huarcaya, Victor},
  title   = {Analytical Throughput, Null Depth, and Surface Tolerance Budget
             for the {LIFE} Nulling Interferometer Combiner},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  note    = {in preparation}
}

@software{huarcaya2026_life_e2e_code,
  author    = {Huarcaya, Victor},
  title     = {{LIFE} E2E Nulling Interferometer Analytical Model},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/vhuarcaya/life-e2e-model},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

---

## Related Work

This model builds on and connects to:

- **LIFE mission:** [life-space-mission.com](https://www.life-space-mission.com) — Quanz et al. (2022), Dannert et al. (2022), Glauser et al. (2024)
- **NICE testbed:** Nulling Interferometry Cryogenic Experiment at ETH Zürich — Gheorghe et al. (2020), Birbacher et al. (2026)
- **LIFEsim:** LIFE detection yield simulator — Dannert et al. (2022), available at [github.com/fdannert/LIFEsim](https://github.com/fdannert/LIFEsim)
- **Nulling theory:** Bracewell (1978), Mennesson & Mariotti (1997), Angel & Woolf (1997)
- **Kernel nulling:** Hansen et al. (2022), Guyon (2013)

---

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

---

## Contact

**Victor Huarcaya**  
Physikalisches Institut, University of Bern  
Sidlerstrasse 5, CH-3012 Bern, Switzerland  
[victor.huarcaya@unibe.ch](mailto:victor.huarcaya@unibe.ch)
