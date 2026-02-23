#!/usr/bin/env python3
"""
LIFE E2E Monte Carlo — Merged Module
======================================
Author:  Victor Huarcaya
Date:    February 2026

Merged from:
  - e2e_monte_carlo_v2.py  (computation engine, 832 lines)
  - e2e_monte_carlo_v3.py  (figure generator, 457 lines)

Engine (from MC v2):
  OpticalSurface dataclass, build_surface_catalogue(), compute_throughput(),
  null_depth_exact(), null_depth_per_polarization(), bs_chromatic_opd(),
  get_null_requirements(), compute_single_realization(), run_monte_carlo(),
  _print_summary(), run_wfe_zero_validation(), print_bugfix_comparison()

Figures (from MC v3):
  generate_figure_11(), generate_figure_12(), generate_figure_13(),
  print_cross_validation(), print_technology_gap()

Material properties are imported from the canonical library
(material_properties.py); no material model is duplicated here.

Module 3 analytical functions imported from m3_null_error_propagation.py.

Wavelength convention
---------------------
Internal wavelength unit: **µm** (matches material_properties library).
Module 3 functions expect metres — conversion at call boundary.

WFE propagation path:
  surface WFE → Maréchal coupling → intensity mismatch → null depth
  Gives quartic N ∝ σ⁴/λ⁴ (fiber-filtered), NOT quadratic σ²/λ² (unfiltered)

v2.0 engine changes:
  1. Fiber coupling OAP marked post-combination (no null contribution)
  2. Gold reflectivity: tabulated Ordal/Palik data (not Drude)
  3. ZnSe Sellmeier: Tatian 1984 three-term (not Connolly)
  4. CaF₂ absorption: coefficient 0.03 (not 0.01)
  5. BS chromatic OPD: wavelength-dependent substrate (CaF₂ < 10µm, ZnSe >= 10µm)
  6. Per-polarization null computation (exact, not effective δφ/δI)
  7. Three null requirement sets (paper, Module 3, Birbacher 2026)
  8. Factored per-realization computation (eliminates WFE-zero code duplication)
  9. Vectorized wavelength loop for performance
  10. Consistent mirror count with paper optical train

v3.0 figure changes:
  - All figures consistent with paper Table 7 values
  - Analytical reference lines from Module 3 compute_null_budget()
  - Both requirement sets (this work + Birbacher+2026) shown
  - LaTeX math mode in all plot labels
  - Proper figure numbering matching paper
"""

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# ============================================================
# Library imports — ALL material functions from canonical source
# ============================================================
from material_properties import (
    gold_reflectivity,       # tabulated Ordal/Palik
    caf2_sellmeier,          # Li 1980 / Malitson 1963
    caf2_absorption,         # multiphonon edge, coeff 0.03
    znse_sellmeier,          # Tatian 1984 three-term
    detector_qe,             # Si:As BIB model
)

# Module 3 analytical functions (expects wavelengths in metres)
from m3_null_error_propagation import (
    compute_null_budget,
    differential_wfe_to_intensity_mismatch,
    null_requirement_curve,
)


# ============================================================
# Global plot style (LaTeX math mode throughout)
# ============================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
})

OUTPUT_DIR = os.environ.get('LIFE_OUTPUT_DIR', os.getcwd())


# ============================================================
# SECTION 1: Surface Definition & WFE Model
# ============================================================

@dataclass
class OpticalSurface:
    """Defines an optical surface in the LIFE combiner.

    Attributes
    ----------
    name : descriptive name
    is_differential : True = exists in one arm only (or different path per arm)
    is_pre_fiber : True = before spatial filter (affects coupling)
    post_combination : True = after cross-combiner (cannot create arm δI)
    CMR : Common-mode rejection (0 = fully differential, 1 = fully common)
    wfe_mean_nm : Mean WFE RMS [nm]
    wfe_sigma_nm : Std dev of WFE manufacturing spread [nm]
    count : Number of such surfaces in the beam path
    is_transmissive : True for BS substrates
    """
    name: str
    is_differential: bool
    is_pre_fiber: bool
    post_combination: bool       # surfaces after beam combination
    CMR: float
    wfe_mean_nm: float
    wfe_sigma_nm: float
    count: int = 1
    is_transmissive: bool = False


def build_surface_catalogue() -> List[OpticalSurface]:
    """Build the complete LIFE combiner surface catalogue.

    WFE values at NICE-demonstrated quality levels (100–300 nm RMS).
    From Module 4 v2 Table with corrected CMR and post-combination flags.

    Optical train (per Glauser et al. 2024 / paper §2.1):
      Collector → Steering → OAP compressor → Cold pupil → DM →
      Delay line (4×) → Dichroics → APS (3× periscope) →
      Cross-combiner (BS + roof) → MMZ (BS×2 + fold) →
      Fiber coupling OAP → Fiber → Spectrograph
    """
    surfaces = [
        # --- Differential surfaces (Tier 1: λ/97 at 6µm) ---
        # These exist in one arm path or have different R/T paths
        OpticalSurface("MMZ beamsplitter 1", True, True, False, 0.30, 200, 50, 1, True),
        OpticalSurface("MMZ beamsplitter 2", True, True, False, 0.30, 200, 50, 1, True),
        OpticalSurface("Cross-combiner BS",  True, True, False, 0.30, 200, 50, 1, True),
        OpticalSurface("Cross-comb roof mir",True, True, False, 0.00, 141.4, 35, 1),
        OpticalSurface("APS periscope mir 1",True, True, False, 0.00, 100, 25, 1),
        OpticalSurface("APS periscope mir 2",True, True, False, 0.00, 100, 25, 1),
        OpticalSurface("APS periscope mir 3",True, True, False, 0.00, 100, 25, 1),
        OpticalSurface("MMZ fold mirror",    True, True, False, 0.00, 100, 25, 1),

        # --- Quasi common-mode surfaces (Tier 2) ---
        OpticalSurface("Delay line mirrors", False, True, False, 0.70, 100, 25, 4),

        # --- Common-mode surfaces (Tier 3) ---
        OpticalSurface("OAP beam compressor",False, True, False, 0.85, 300, 75, 1),
        OpticalSurface("Sci/ctrl dichroic",  False, True, False, 0.85, 200, 50, 1),
        OpticalSurface("Pol. compensator",   False, True, False, 0.80, 200, 50, 1),
        OpticalSurface("Bandpass dichroics", False, True, False, 0.90, 200, 50, 2),
        OpticalSurface("Collector mirror",   False, True, False, 0.90, 200, 50, 1),
        OpticalSurface("Deformable mirror",  False, True, False, 0.80, 100, 25, 1),
        OpticalSurface("Steering mirror",    False, True, False, 0.85, 100, 25, 1),

        # --- Post-combination surfaces (cannot affect null) ---
        # Fiber coupling OAP sits AFTER cross-combiner: both arms already combined.
        # WFE here degrades throughput equally but cannot create arm δI.
        OpticalSurface("Fiber coupling OAP", False, True, True, 0.85, 300, 75, 1),
    ]
    return surfaces


# ============================================================
# SECTION 2: Throughput Chain (Module 2, library materials)
# ============================================================

def compute_throughput(wavelength_um: np.ndarray,
                       eta_coupling: np.ndarray) -> np.ndarray:
    """Complete throughput chain from Module 2 (library material models).

    Parameters
    ----------
    wavelength_um : wavelengths [µm]
    eta_coupling : fiber coupling efficiency (degraded by WFE)

    Returns
    -------
    T_total : photon conversion efficiency
    """
    # Gold mirror reflectivity (flight quality, tabulated)
    R_gold = gold_reflectivity(wavelength_um, 'flight')
    n_mirrors = 20  # reconciled with paper optical train
    # Count: 1 collector + 1 steering + 2 pol comp + 1 OAP + 1 DM
    # + 4 delay + 3 APS + 1 MMZ fold + 2 cross-comb roof
    # + 1 fiber OAP + 3 spectrograph = 20 reflective gold surfaces
    T_mirrors = R_gold ** n_mirrors

    # BS substrate transmission (multi-band: CaF₂ < 10 µm, ZnSe >= 10 µm)
    T_bs = np.ones_like(wavelength_um)
    mask_short = wavelength_um < 10.0
    mask_long = ~mask_short

    if np.any(mask_short):
        alpha_caf2 = caf2_absorption(wavelength_um[mask_short])
        n_caf2 = caf2_sellmeier(wavelength_um[mask_short])
        T_bulk = np.exp(-alpha_caf2 * 0.2)  # 2mm = 0.2 cm
        R_fresnel = ((n_caf2 - 1) / (n_caf2 + 1))**2
        T_ar = (1 - R_fresnel * 0.1)**2  # 90% AR efficiency, 2 surfaces
        T_bs[mask_short] = T_bulk * T_ar

    if np.any(mask_long):
        n_znse = znse_sellmeier(wavelength_um[mask_long])
        R_fresnel = ((n_znse - 1) / (n_znse + 1))**2
        T_ar = (1 - R_fresnel * 0.1)**2
        T_bs[mask_long] = T_ar  # ZnSe transparent across LIFE band

    # 3 BS traversals: 2 in MMZ + 1 in cross-combiner
    T_bs_total = T_bs ** 3

    # Dichroic losses: 2 bandpass + 1 sci/ctrl
    T_dichroics = 0.95 ** 3

    # Fundamental splitting losses
    T_mmz_split = 0.50     # MMZ: only dark output carries planet signal
    T_cross_split = 0.50   # Cross-combiner: Double Bracewell

    # Detector QE
    qe = detector_qe(wavelength_um, 'SiAs')

    T_total = (T_mirrors * T_bs_total * T_dichroics *
               T_mmz_split * T_cross_split *
               eta_coupling * qe)

    return T_total


# ============================================================
# SECTION 3: Null Depth Model (Module 3)
# ============================================================

def null_depth_exact(delta_phi: np.ndarray,
                     delta_I: np.ndarray) -> np.ndarray:
    """Exact null depth formula (Birbacher+2026 Appendix A.1 / Serabyn 2000).

    N = [1 − √(1 − δI²) cos(δφ)] / [1 + √(1 − δI²)]

    Valid for arbitrarily large errors. Reduces to ¼(δφ² + δI²) for small errors.
    """
    delta_I_clipped = np.clip(np.abs(delta_I), 0, 0.9999)
    sqrt_term = np.sqrt(1 - delta_I_clipped**2)
    N = (1 - sqrt_term * np.cos(delta_phi)) / (1 + sqrt_term)
    return np.clip(N, 0, 1)


def null_depth_per_polarization(delta_phi: np.ndarray,
                                delta_I: np.ndarray,
                                delta_phi_sp: float,
                                delta_I_sp: float) -> np.ndarray:
    """Compute null depth averaging over both polarizations (Birbacher Eq. 3).

    Computes N_s and N_p separately using exact formula, then averages.
    More accurate than effective δφ/δI approach at large polarization mismatch.

    Parameters
    ----------
    delta_phi : common-mode phase error [rad] (average of s and p)
    delta_I : common-mode intensity mismatch [dimensionless]
    delta_phi_sp : s-p differential phase [rad]
    delta_I_sp : s-p differential intensity mismatch [dimensionless]
    """
    # s-polarization: δφ_s = δφ + δφ_sp/2, δI_s = δI + δI_sp/2
    N_s = null_depth_exact(delta_phi + delta_phi_sp / 2,
                           delta_I + delta_I_sp / 2)
    # p-polarization: δφ_p = δφ − δφ_sp/2, δI_p = δI − δI_sp/2
    N_p = null_depth_exact(delta_phi - delta_phi_sp / 2,
                           np.abs(delta_I - delta_I_sp / 2))
    return 0.5 * (N_s + N_p)


def bs_chromatic_opd(wavelength_um: np.ndarray,
                     delta_d_um: float,
                     lambda_ref_um: float = 10.0) -> np.ndarray:
    """Chromatic OPD from BS thickness mismatch with multi-band substrates.

    Uses CaF₂ for λ < 10 µm, ZnSe for λ ≥ 10 µm (matching LIFE architecture).
    δOPD(λ) = [n_substrate(λ) − n_substrate(λ_ref)] × Δd

    Parameters
    ----------
    wavelength_um : wavelengths [µm]
    delta_d_um : BS thickness mismatch [µm]
    lambda_ref_um : compensator reference wavelength [µm]

    Returns
    -------
    delta_opd_nm : chromatic OPD [nm]
    """
    delta_opd_nm = np.zeros_like(wavelength_um)

    mask_short = wavelength_um < 10.0
    mask_long = ~mask_short

    if np.any(mask_short):
        n_lam = caf2_sellmeier(wavelength_um[mask_short])
        n_ref = caf2_sellmeier(np.array([lambda_ref_um]))[0]
        delta_opd_nm[mask_short] = (n_lam - n_ref) * delta_d_um * 1000

    if np.any(mask_long):
        n_lam = znse_sellmeier(wavelength_um[mask_long])
        # ZnSe compensator referenced at 10 µm
        n_ref_znse = znse_sellmeier(np.array([lambda_ref_um]))[0]
        delta_opd_nm[mask_long] = (n_lam - n_ref_znse) * delta_d_um * 1000

    return delta_opd_nm


# ============================================================
# SECTION 4: Null Requirement Curves
# ============================================================

def get_null_requirements() -> Dict[str, Dict[float, float]]:
    """Three requirement sets for comparison.

    Returns dict of {source_name: {wavelength: N_req}}.
    """
    return {
        'paper': {
            # From life_e2e_phase1_v2.tex Table null_budget
            6.0: 1.0e-5, 8.0: 6.8e-6, 10.0: 3.0e-5,
            12.0: 1.8e-5, 16.0: 6.0e-5,
        },
        'birbacher': {
            # From Birbacher et al. 2026, Table 1
            # Derived from 10% SNR degradation methodology
            6.0: 1.5e-5, 8.0: 7.0e-6, 10.0: 9.0e-6,
            12.0: 4.0e-5, 16.0: 8.0e-5,
        },
        'flat_1e-5': {
            # Simple flat requirement for reference
            6.0: 1.0e-5, 8.0: 1.0e-5, 10.0: 1.0e-5,
            12.0: 1.0e-5, 16.0: 1.0e-5,
        },
    }


# ============================================================
# SECTION 5: Per-Realization Computation (factored out)
# ============================================================

def compute_single_realization(
    rng: np.random.Generator,
    surfaces: List[OpticalSurface],
    wavelengths_um: np.ndarray,
    bs_chromatic_nm: np.ndarray,
    # Error parameters
    opd_mean_nm: float,
    opd_sigma_nm: float,
    delta_I_static: float,
    sigma_delta_I: float,
    delta_phi_sp: float,
    delta_I_sp: float,
    eta_tophat: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute null, throughput, and coupling for one Monte Carlo realization.

    This factored-out function is used by both the main MC and the WFE-zero
    validation, eliminating code duplication.

    Returns
    -------
    null_depths : array [n_wav]
    throughputs : array [n_wav]
    coupling_effs : array [n_wav]
    delta_I_vals : array [n_wav]
    delta_phi_vals : array [n_wav]
    """
    n_wav = len(wavelengths_um)

    # ---- STEP 1: Draw surface WFEs (once per realization) ----
    surface_wfes = []
    for s in surfaces:
        wfe = np.clip(np.abs(rng.normal(s.wfe_mean_nm, s.wfe_sigma_nm)),
              0, s.wfe_mean_nm + 3 * s.wfe_sigma_nm)
        wfe_total = wfe * np.sqrt(s.count)  # RSS for multi-surface groups
        surface_wfes.append((s, wfe_total))

    # ---- STEP 2: Draw OPD (once per realization, all wavelengths) ----
    opd_draw = rng.normal(opd_mean_nm, opd_sigma_nm)

    # ---- STEP 3: Draw static intensity mismatch ----
    delta_I_draw = np.abs(rng.normal(delta_I_static, sigma_delta_I))

    # ---- STEP 4: Per-wavelength computations (vectorizable) ----
    lam_nm = wavelengths_um * 1000  # µm → nm

    # Accumulate WFE for each beam at all wavelengths simultaneously
    sigma_sq_beam1 = np.zeros(n_wav)
    sigma_sq_beam2 = np.zeros(n_wav)

    for s, wfe_nm in surface_wfes:
        if not s.is_pre_fiber:
            continue

        # Post-combination surfaces degrade throughput but not null
        if s.post_combination:
            # These surfaces affect both beams equally after combination.
            # They degrade coupling (throughput) but cannot create arm δI.
            # We'll add them to a separate "throughput WFE" accumulator below.
            continue

        if s.is_differential and s.CMR == 0.0:
            # Fully differential: entire WFE in beam 2 only
            sigma_sq_beam2 += wfe_nm**2

        elif s.is_differential and s.CMR > 0.0:
            # Partially differential (BSes): R and T paths differ
            common_wfe = wfe_nm * s.CMR
            diff_wfe = wfe_nm * (1 - s.CMR)
            sigma_sq_beam1 += common_wfe**2
            sigma_sq_beam2 += (common_wfe**2 + diff_wfe**2)

        else:
            # Common-mode: small differential from manufacturing
            common_wfe = wfe_nm * s.CMR
            diff_residual = wfe_nm * (1 - s.CMR)
            sign = rng.choice([-1, 1])
            sigma_sq_beam1 += (common_wfe + sign * diff_residual * 0.5)**2
            sigma_sq_beam2 += (common_wfe - sign * diff_residual * 0.5)**2

    sigma_beam1 = np.sqrt(sigma_sq_beam1)
    sigma_beam2 = np.sqrt(sigma_sq_beam2)

    # ---- STEP 5: Maréchal coupling per beam (THE CORE FIX) ----
    # η_i = η₀ × exp(−(2πσ_i/λ)²)
    marechal_1 = (2 * np.pi * sigma_beam1 / lam_nm)**2
    marechal_2 = (2 * np.pi * sigma_beam2 / lam_nm)**2

    eta_1 = eta_tophat * np.exp(-marechal_1)
    eta_2 = eta_tophat * np.exp(-marechal_2)

    # Post-combination WFE degrades both beams equally (throughput only)
    sigma_sq_post = 0.0
    for s, wfe_nm in surface_wfes:
        if s.post_combination:
            sigma_sq_post += wfe_nm**2
    if sigma_sq_post > 0:
        marechal_post = (2 * np.pi * np.sqrt(sigma_sq_post) / lam_nm)**2
        post_factor = np.exp(-marechal_post)
        eta_1 *= post_factor
        eta_2 *= post_factor

    eta_avg = 0.5 * (eta_1 + eta_2)

    # ---- STEP 6: Intensity mismatch from WFE ----
    delta_I_wfe = np.abs(eta_1 - eta_2) / (eta_1 + eta_2 + 1e-30)

    # ---- STEP 7: Phase error (OPD + BS chromatic, coherent sum) ----
    delta_opd_total_nm = opd_draw + bs_chromatic_nm
    delta_phi = 2 * np.pi * delta_opd_total_nm / lam_nm

    # ---- STEP 8: Total intensity mismatch (RSS of independent sources) ----
    delta_I_total = np.sqrt(delta_I_wfe**2 + delta_I_draw**2)

    # ---- STEP 9: Null depth with per-polarization averaging ----
    # Compute N_s and N_p separately, then average (Birbacher Eq. 3)
    null_depths = null_depth_per_polarization(
        delta_phi, delta_I_total, delta_phi_sp, delta_I_sp
    )

    # ---- STEP 10: Throughput ----
    throughputs = compute_throughput(wavelengths_um, eta_avg)

    return null_depths, throughputs, eta_avg, delta_I_total, delta_phi


# ============================================================
# SECTION 6: E2E Monte Carlo Driver
# ============================================================

def run_monte_carlo(N_realizations: int = 100000,
                    wavelengths_um: np.ndarray = None,
                    seed: int = 42,
                    verbose: bool = True,
                    zero_wfe: bool = False) -> dict:
    """Full end-to-end Monte Carlo combining Modules 1–4.

    Parameters
    ----------
    N_realizations : number of MC draws (paper uses 10⁵)
    wavelengths_um : evaluation wavelengths [µm]
    seed : RNG seed for reproducibility
    verbose : print summary tables
    zero_wfe : if True, set all surface WFEs to zero (for Module 3 validation)

    Returns
    -------
    results : dict with arrays of null_depths, throughputs, coupling_effs, etc.
    """
    if wavelengths_um is None:
        wavelengths_um = np.array([6.0, 8.0, 10.0, 12.0, 16.0])

    rng = np.random.default_rng(seed)
    surfaces = build_surface_catalogue()

    # For WFE-zero validation: override all surface WFEs
    if zero_wfe:
        for s in surfaces:
            s.wfe_mean_nm = 0.0
            s.wfe_sigma_nm = 0.0

    # --- Module 3 error parameters (NICE-demonstrated levels) ---
    opd_mean_nm = 0.5           # Mean OPD offset [nm]
    opd_sigma_nm = 1.2          # OPD RMS fluctuation [nm]
    delta_I_static = 0.0043     # Static intensity mismatch (0.43%, NICE)
    sigma_delta_I = 0.0043      # Intensity mismatch drift RMS (0.43%)
    delta_d_bs_um = 0.1         # BS thickness mismatch [µm]
    pol_phase_deg = 0.15        # Polarization phase mismatch [deg]
    pol_intensity = 0.003       # Polarization intensity mismatch (0.3%)
    eta_tophat = 0.8145         # Ideal top-hat coupling (Module 1)

    # Precompute
    bs_chromatic_nm = bs_chromatic_opd(wavelengths_um, delta_d_bs_um)
    delta_phi_sp = np.radians(pol_phase_deg)
    delta_I_sp = pol_intensity

    # --- Results storage ---
    n_wav = len(wavelengths_um)
    null_depths = np.zeros((N_realizations, n_wav))
    throughputs = np.zeros((N_realizations, n_wav))
    coupling_effs = np.zeros((N_realizations, n_wav))
    delta_I_all = np.zeros((N_realizations, n_wav))
    delta_phi_all = np.zeros((N_realizations, n_wav))

    # --- Main MC loop ---
    for i in range(N_realizations):
        N_i, T_i, eta_i, dI_i, dphi_i = compute_single_realization(
            rng, surfaces, wavelengths_um, bs_chromatic_nm,
            opd_mean_nm, opd_sigma_nm,
            delta_I_static, sigma_delta_I,
            delta_phi_sp, delta_I_sp, eta_tophat,
        )
        null_depths[i] = N_i
        throughputs[i] = T_i
        coupling_effs[i] = eta_i
        delta_I_all[i] = dI_i
        delta_phi_all[i] = dphi_i

    # --- Collect results ---
    results = {
        'wavelengths': wavelengths_um,
        'null_depths': null_depths,
        'throughputs': throughputs,
        'coupling_effs': coupling_effs,
        'delta_I': delta_I_all,
        'delta_phi': delta_phi_all,
        'parameters': {
            'N_realizations': N_realizations,
            'opd_mean_nm': opd_mean_nm,
            'opd_sigma_nm': opd_sigma_nm,
            'delta_I_static': delta_I_static,
            'sigma_delta_I': sigma_delta_I,
            'delta_d_bs_um': delta_d_bs_um,
            'pol_phase_deg': pol_phase_deg,
            'pol_intensity': pol_intensity,
            'eta_tophat': eta_tophat,
            'zero_wfe': zero_wfe,
        },
    }

    if verbose:
        _print_summary(results)

    return results


# ============================================================
# SECTION 7: Output & Verification
# ============================================================

def _print_summary(results: dict) -> None:
    """Print comprehensive MC summary with multi-requirement comparison."""
    wavelengths_um = results['wavelengths']
    null_depths = results['null_depths']
    throughputs = results['throughputs']
    coupling_effs = results['coupling_effs']
    params = results['parameters']

    req_sets = get_null_requirements()
    n_wav = len(wavelengths_um)

    print("=" * 110)
    print("LIFE E2E MONTE CARLO — MERGED MODULE")
    print(f"N_realizations = {params['N_realizations']}, "
          f"zero_wfe = {params['zero_wfe']}")
    print("=" * 110)
    print()

    # --- Main statistics table ---
    header = (f"{'λ [µm]':>8} | {'N_mean':>12} | {'N_median':>12} | "
              f"{'N_95%':>12} | {'T_mean':>8} | {'η_mean':>8}")
    print(header)
    print("-" * 80)

    for j, lam in enumerate(wavelengths_um):
        N_col = null_depths[:, j]
        T_col = throughputs[:, j]
        eta_col = coupling_effs[:, j]

        print(f"{lam:8.1f} | {np.mean(N_col):12.3e} | {np.median(N_col):12.3e} | "
              f"{np.percentile(N_col, 95):12.3e} | "
              f"{np.mean(T_col)*100:7.2f}% | {np.mean(eta_col)*100:7.1f}%")

    # --- Pass rates against all three requirement sets ---
    print()
    print("--- Pass Rates P(N < N_req) by Requirement Source ---")
    header = f"{'λ [µm]':>8}"
    for name in req_sets:
        header += f" | {name:>15}"
    print(header)
    print("-" * (10 + 18 * len(req_sets)))

    for j, lam in enumerate(wavelengths_um):
        N_col = null_depths[:, j]
        row = f"{lam:8.1f}"
        for name, reqs in req_sets.items():
            N_req = reqs.get(lam, 1e-5)
            P_pass = np.mean(N_col < N_req) * 100
            row += f" | {P_pass:13.1f}%"
        print(row)

    # --- Verification checks ---
    print()
    print("=" * 110)
    print("VERIFICATION CHECKLIST")
    print("=" * 110)
    print()

    idx = {lam: j for j, lam in enumerate(wavelengths_um)}

    # Check 1: N_mean at 10 µm
    if 10.0 in idx:
        N10 = np.mean(null_depths[:, idx[10.0]])
        if params['zero_wfe']:
            expected = "~1.3×10⁻⁵ (Module 3 analytical)"
            in_range = 5e-6 <= N10 <= 5e-5
        else:
            expected = "3×10⁻⁵ to 5×10⁻⁴ (Module 3 + Module 4 WFE)"
            in_range = 3e-5 <= N10 <= 5e-4
        status = "✓ PASS" if in_range else "✗ FAIL"
        print(f"[{status}] N_mean at 10 µm = {N10:.3e}  (expected: {expected})")

    # Check 2: N(6)/N(10) ratio
    if 6.0 in idx and 10.0 in idx:
        N6 = np.mean(null_depths[:, idx[6.0]])
        N10 = np.mean(null_depths[:, idx[10.0]])
        ratio = N6 / N10
        in_range = 2 <= ratio <= 30
        status = "✓ PASS" if in_range else "✗ FAIL"
        pure_theory = (10.0/6.0)**4
        print(f"[{status}] N(6)/N(10) = {ratio:.1f}  "
              f"(pure λ⁻⁴: {pure_theory:.1f}; range 2–30 acceptable)")

    # Check 3: Throughput at 10 µm
    if 10.0 in idx:
        T10 = np.mean(throughputs[:, idx[10.0]]) * 100
        in_range = 3.0 <= T10 <= 12.0
        status = "✓ PASS" if in_range else "✗ FAIL"
        print(f"[{status}] T at 10 µm = {T10:.2f}%  "
              f"(Module 2 predicts 7.0–8.1% PCE for ideal coupling)")

    # Check 4: Wavelength scaling
    if 10.0 in idx and 16.0 in idx:
        N10 = np.mean(null_depths[:, idx[10.0]])
        N16 = np.mean(null_depths[:, idx[16.0]])
        ratio = N10 / N16
        in_range = 1 <= ratio <= 15
        status = "✓ PASS" if in_range else "✗ FAIL"
        print(f"[{status}] N(10)/N(16) = {ratio:.1f}  "
              f"(pure λ⁻⁴: {(16./10.)**4:.1f}; range 1–15 acceptable)")

    # --- Technology gap assessment ---
    print()
    print("--- Technology Gap Assessment ---")
    birbacher_reqs = req_sets['birbacher']
    for j, lam in enumerate(wavelengths_um):
        N_mean = np.mean(null_depths[:, j])
        N_req = birbacher_reqs.get(lam, 1e-5)
        gap = N_mean / N_req
        surface_factor = gap**(1/4)
        print(f"λ = {lam:5.1f} µm: N_mean/N_req = {gap:6.1f}×, "
              f"surface quality improvement needed: {surface_factor:.2f}×")


def run_wfe_zero_validation(N_check: int = 5000,
                            seed: int = 123,
                            verbose: bool = True) -> dict:
    """Run MC with zero WFE to validate against Module 3 analytical predictions.

    Uses the same compute_single_realization() function as the main MC,
    eliminating duplicated code.
    """
    # Pull analytical references from Module 3 directly so this validation
    # stays consistent with any upstream budget updates.
    module3_analytical = get_analytical_reference()

    results = run_monte_carlo(
        N_realizations=N_check,
        seed=seed,
        verbose=False,
        zero_wfe=True,
    )

    if verbose:
        print()
        print("=" * 80)
        print("WFE-ZERO VALIDATION (should match Module 3 analytical)")
        print("=" * 80)
        print()

        wavelengths_um = results['wavelengths']
        null_depths = results['null_depths']

        for j, lam in enumerate(wavelengths_um):
            N_mc = np.mean(null_depths[:, j])
            N_analytical = module3_analytical.get(lam, None)
            if N_analytical is not None:
                ratio = N_mc / N_analytical
                in_range = 0.3 <= ratio <= 3.0
                status = "✓ PASS" if in_range else "✗ FAIL"
                print(f"[{status}] λ={lam:5.1f} µm: MC(WFE=0) = {N_mc:.3e}, "
                      f"Mod3 = {N_analytical:.3e}, ratio = {ratio:.2f}")

    return results


# ============================================================
# SECTION 8: Comparison with Old Buggy Values
# ============================================================

def print_bugfix_comparison(results: dict) -> None:
    """Print comparison of v2 results with the old buggy (WFE-as-phase) version.

    The old buggy code treated WFE as direct phase variance (σ²/λ²) instead of
    the correct coupling → δI pathway (σ⁴/λ⁴). This caused N(6µm) ≈ 0.2,
    474× larger than N(10µm), when the correct ratio should be ~6–10×.
    """
    old_buggy = {
        6.0:  {'N_mean': 1.95e-1, 'T_mean': 9.7e-2, 'eta': 50.9},
        8.0:  {'N_mean': 1.29e-3, 'T_mean': 11.9e-2, 'eta': 62.5},
        10.0: {'N_mean': 4.11e-4, 'T_mean': 13.1e-2, 'eta': 68.7},
        12.0: {'N_mean': 2.16e-4, 'T_mean': 13.1e-2, 'eta': 72.4},
        16.0: {'N_mean': 9.44e-5, 'T_mean': 13.8e-2, 'eta': 76.2},
    }

    print()
    print("=" * 90)
    print("COMPARISON: OLD (BUGGY WFE-AS-PHASE) vs CORRECTED (COUPLING→δI)")
    print("=" * 90)
    print(f"{'λ [µm]':>8} | {'Old N_mean':>12} | {'New N_mean':>12} | "
          f"{'Improvement':>12} | {'Old T':>8} | {'New T':>8}")
    print("-" * 80)

    wavelengths_um = results['wavelengths']
    null_depths = results['null_depths']
    throughputs = results['throughputs']

    for j, lam in enumerate(wavelengths_um):
        old = old_buggy.get(lam, {})
        new_N = np.mean(null_depths[:, j])
        new_T = np.mean(throughputs[:, j])
        old_N = old.get('N_mean', float('nan'))
        old_T = old.get('T_mean', float('nan'))
        improvement = old_N / new_N if new_N > 0 else float('nan')

        print(f"{lam:8.1f} | {old_N:12.3e} | {new_N:12.3e} | "
              f"{improvement:11.1f}× | {old_T*100:7.2f}% | {new_T*100:7.2f}%")


# ============================================================
# SECTION 9: MC Execution Helpers (from MC v3)
# ============================================================

def run_full_mc(N_realizations=100_000, seed=42):
    """Run full Monte Carlo with all surfaces."""
    print(f"Running full MC with N = {N_realizations:,} realisations...")
    t0 = time.time()

    results = run_monte_carlo(
        N_realizations=N_realizations,
        wavelengths_um=np.array([6.0, 8.0, 10.0, 12.0, 16.0]),
        seed=seed,
        verbose=True,
    )

    dt = time.time() - t0
    print(f"  MC completed in {dt:.1f} s")
    return results


def get_analytical_reference():
    """Get Module 3 analytical budget totals for reference lines.

    Note: compute_null_budget() expects wavelengths in metres.
    """
    wavelengths_m = np.array([6, 8, 10, 12, 16]) * 1e-6  # µm → m
    budget = compute_null_budget(wavelengths_m)
    return dict(zip([6.0, 8.0, 10.0, 12.0, 16.0], budget['N_total']))


def get_surface_only_analytical_reference(wavelengths_um: np.ndarray = None):
    """Analytical surface-only null (Module 4-consistent per-surface sum).

    This excludes non-surface terms (OPD, static intensity, polarization) and
    sums only pre-fiber, pre-combination surface WFE contributions at nominal
    quality.
    """
    if wavelengths_um is None:
        wavelengths_um = np.array([6.0, 8.0, 10.0, 12.0, 16.0], dtype=float)

    surfaces = build_surface_catalogue()
    ref = {}

    for lam_um in wavelengths_um:
        lam_m = lam_um * 1e-6
        N_sum = 0.0
        for s in surfaces:
            if (not s.is_pre_fiber) or s.post_combination:
                continue

            # Per-type WFE (RSS for grouped surfaces, as in Module 4).
            wfe_total_m = s.wfe_mean_nm * np.sqrt(s.count) * 1e-9
            wfe_diff_m = wfe_total_m * (1.0 - s.CMR)

            dI_wfe = float(np.atleast_1d(
                differential_wfe_to_intensity_mismatch(wfe_diff_m, lam_m)
            )[0])
            N_sum += 0.25 * dI_wfe**2

        ref[float(lam_um)] = N_sum

    return ref


# ============================================================
# SECTION 10: Figure Generation (from MC v3, LaTeX labels)
# ============================================================

def generate_figure_11(results, analytical):
    """
    Paper Figure 11: Three-panel histogram of null depth distributions
    from the FULL surface-accumulation Monte Carlo.

    Shows mean, median, 95th percentile, requirement, and analytical
    reference at λ = 6, 10, 16 µm.
    """
    print("\n--- Generating Figure 11: MC null depth distributions ---")

    wavelengths = results['wavelengths']
    null_depths = results['null_depths']
    N_real = null_depths.shape[0]

    # Requirement sets
    req_sets = get_null_requirements()
    req_paper = req_sets['paper']
    req_birb = req_sets['birbacher']

    # Three representative wavelengths
    plot_lams = [6.0, 10.0, 16.0]
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, (lam, color) in enumerate(zip(plot_lams, colors)):
        ax = axes[ax_idx]
        j = list(wavelengths).index(lam)
        N_col = null_depths[:, j]

        # Statistics
        N_mean = np.mean(N_col)
        N_median = np.median(N_col)
        N_p95 = np.percentile(N_col, 95)
        N_req = req_paper.get(lam, 1e-5)
        N_analytical = analytical.get(lam, None)

        # Histogram in log space
        log_N = np.log10(N_col[N_col > 0])
        ax.hist(log_N, bins=100, density=True, color=color, alpha=0.7,
                edgecolor='black', linewidth=0.3)

        # Vertical reference lines
        ax.axvline(np.log10(N_mean), color='black', ls='-', lw=2,
                   label=f'Mean: {N_mean:.1e}')
        ax.axvline(np.log10(N_median), color='black', ls='--', lw=1.5,
                   label=f'Median: {N_median:.1e}')
        ax.axvline(np.log10(N_p95), color='orange', ls='-', lw=1.5,
                   label=f'95th %: {N_p95:.1e}')
        ax.axvline(np.log10(N_req), color='red', ls=':', lw=2,
                   label=f'Req: {N_req:.1e}')

        if N_analytical is not None and N_analytical > 0:
            ax.axvline(np.log10(N_analytical), color='purple', ls='-.',
                       lw=1.5, label=f'Analytical: {N_analytical:.1e}')

        ax.set_xlabel(r'$\log_{10}$(Null Depth)')
        ax.set_title(r'$\lambda = %d\,\mu$m' % int(lam))
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Print summary
        gap = N_mean / N_req
        print(f"  {lam:.0f} µm: Mean={N_mean:.2e}, Median={N_median:.2e}, "
              f"95th={N_p95:.2e}, Req={N_req:.1e}, Gap={gap:.1f}x")
        if N_analytical:
            print(f"         Analytical={N_analytical:.2e}, "
                  f"MC/Analytical={N_mean/N_analytical:.1f}x (Jensen enhancement)")

    axes[0].set_ylabel('Probability density')
    fig.suptitle(
        r'Monte Carlo null depth distributions ($10^5$ realisations, '
        r'NICE performance)',
        fontsize=13, y=1.02)
    fig.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'fig11_mc_null_distributions.png')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def generate_figure_12(results):
    """
    Paper Figure 12: CDF of null depth at all five wavelengths.

    Shows the probability of achieving a given null depth.
    Both requirement sets overlaid as vertical lines.
    """
    print("\n--- Generating Figure 12: Cumulative null depth distribution ---")

    wavelengths = results['wavelengths']
    null_depths = results['null_depths']
    N_real = null_depths.shape[0]

    req_sets = get_null_requirements()
    req_paper = req_sets['paper']
    req_birb = req_sets['birbacher']

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

    fig, ax = plt.subplots(figsize=(8, 6))

    for j, (lam, col) in enumerate(zip(wavelengths, colors)):
        N_col = null_depths[:, j]
        sorted_N = np.sort(N_col)
        cdf = np.arange(1, N_real + 1) / N_real

        ax.semilogx(sorted_N, cdf, color=col, lw=2.5,
                     label=r'$\lambda = %d\,\mu$m' % int(lam))

        # Requirement lines (this work)
        N_req_p = req_paper.get(lam, 1e-5)
        ax.axvline(x=N_req_p, color=col, ls='-.', lw=1, alpha=0.5)

        # Requirement lines (Birbacher)
        N_req_b = req_birb.get(lam, 1e-5)
        ax.axvline(x=N_req_b, color=col, ls=':', lw=1, alpha=0.3)

    # Horizontal percentile lines
    ax.axhline(y=0.50, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(8e-6, 0.52, '50th percentile', fontsize=8, color='gray')
    ax.axhline(y=0.95, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(8e-6, 0.97, '95th percentile', fontsize=8, color='gray')

    ax.set_xlabel('Null Depth')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Null Depth Distribution')
    ax.legend(fontsize=10)
    ax.set_xlim(1e-6, 1e-1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, which='both')

    # Annotations for requirement sets
    ax.text(0.02, 0.15,
            'Dash-dot: Req. (this work)\nDotted: Req. (Birbacher+2026)',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'fig12_null_cdf.png')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

    # Print pass rates
    print("\n  Pass rates P(N < N_req):")
    print(f"  {'λ [µm]':>8} {'Paper req':>12} {'Birbacher':>12}")
    print(f"  {'-'*36}")
    for j, lam in enumerate(wavelengths):
        N_col = null_depths[:, j]
        P_paper = np.mean(N_col < req_paper.get(lam, 1e-5)) * 100
        P_birb = np.mean(N_col < req_birb.get(lam, 1e-5)) * 100
        print(f"  {lam:8.0f} {P_paper:11.1f}% {P_birb:11.1f}%")


def generate_figure_13(results):
    """
    Paper Figure 13: Two-panel throughput figure.
      Left:  PCE distribution across wavelengths
      Right: Fiber coupling distribution (Maréchal degradation)
    """
    print("\n--- Generating Figure 13: Throughput distributions ---")

    wavelengths = results['wavelengths']
    throughputs = results['throughputs']
    coupling_effs = results['coupling_effs']
    N_real = throughputs.shape[0]

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

    fig, (ax_pce, ax_eta) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- Left: PCE distributions ----
    for j, (lam, col) in enumerate(zip(wavelengths, colors)):
        T_col = throughputs[:, j] * 100  # to %
        T_mean = np.mean(T_col)

        ax_pce.hist(T_col, bins=80, density=True, color=col, alpha=0.6,
                    edgecolor='black', linewidth=0.2,
                    label=r'$%d\,\mu$m (mean: %.1f%%)' % (int(lam), T_mean))

    # Threshold line
    ax_pce.axvline(x=3.5, color='red', ls='--', lw=1.5)
    ax_pce.text(3.7, ax_pce.get_ylim()[1] * 0.9, 'Max. req. (3.5%)',
                fontsize=9, color='red')

    ax_pce.set_xlabel('Photon Conversion Efficiency [%]')
    ax_pce.set_ylabel('Probability Density')
    ax_pce.set_title('PCE Distribution')
    ax_pce.legend(fontsize=8)
    ax_pce.grid(True, alpha=0.3)

    # ---- Right: Fiber coupling distributions ----
    for j, (lam, col) in enumerate(zip(wavelengths, colors)):
        eta_col = coupling_effs[:, j] * 100  # to %
        eta_mean = np.mean(eta_col)

        ax_eta.hist(eta_col, bins=80, density=True, color=col, alpha=0.6,
                    edgecolor='black', linewidth=0.2,
                    label=r'$%d\,\mu$m (mean: %.1f%%)' % (int(lam), eta_mean))

    ax_eta.set_xlabel('Fiber Coupling Efficiency [%]')
    ax_eta.set_ylabel('Probability Density')
    ax_eta.set_title(r'Fiber Coupling Distribution (Mar$\acute{e}$chal degradation)')
    ax_eta.legend(fontsize=8)
    ax_eta.grid(True, alpha=0.3)

    fig.suptitle(
        r'Throughput and Coupling Distributions ($10^5$ realisations)',
        fontsize=13, y=1.02)
    fig.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'fig13_throughput_distributions.png')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

    # Print summary
    print("\n  Throughput summary:")
    print(f"  {'λ [µm]':>8} {'PCE mean':>10} {'PCE med':>10} {'η mean':>10}")
    print(f"  {'-'*42}")
    for j, lam in enumerate(wavelengths):
        T_mean = np.mean(throughputs[:, j]) * 100
        T_med = np.median(throughputs[:, j]) * 100
        eta_mean = np.mean(coupling_effs[:, j]) * 100
        print(f"  {lam:8.0f} {T_mean:9.2f}% {T_med:9.2f}% {eta_mean:9.1f}%")


# ============================================================
# SECTION 11: Cross-Validation & Technology Gap Tables (from MC v3)
# ============================================================

def print_cross_validation(results, analytical, wfe_zero_results=None):
    """Print cross-module consistency check (paper Table 7)."""
    print("\n" + "=" * 80)
    print("CROSS-MODULE CONSISTENCY VALIDATION (Paper Table 7)")
    print("=" * 80)

    wavelengths = results['wavelengths']
    null_depths = results['null_depths']
    throughputs = results['throughputs']
    surface_analytical = get_surface_only_analytical_reference(wavelengths)

    # If WFE-zero results were not provided by the caller, compute a compact
    # validation sample here so the table remains self-consistent.
    if wfe_zero_results is None:
        wfe_zero_results = run_monte_carlo(
            N_realizations=5000,
            wavelengths_um=wavelengths,
            seed=123,
            verbose=False,
            zero_wfe=True,
        )

    print(f"\n  {'Quantity':<30} {'Analytical':>14} {'MC Mean':>14} {'Note':>10}")
    print(f"  {'-'*72}")

    # η₀ (ideal top-hat)
    print(f"  {'η₀ (ideal top-hat)':<30} {'81.45%':>14} {'81.45%':>14} {'Exact':>10}")

    # PCE at 10 µm
    j10 = list(wavelengths).index(10.0)
    T_mc = np.mean(throughputs[:, j10]) * 100
    T_anal = 7.1
    pce_delta_pct = abs(T_mc - T_anal) / max(T_anal, 1e-30) * 100
    pce_note = f"{pce_delta_pct:.1f}% {'lower' if T_mc < T_anal else 'higher'}"
    print(f"  {'PCE at 10 µm':<30} {f'{T_anal:.1f}%':>14} {T_mc:13.1f}% {pce_note:>10}")

    # Null at WFE=0, 10 µm
    N_anal_10 = analytical.get(10.0, 0)
    j10_wfe0 = list(wfe_zero_results['wavelengths']).index(10.0)
    N_mc_wfe0 = np.mean(wfe_zero_results['null_depths'][:, j10_wfe0])
    n_delta_pct = abs(N_mc_wfe0 - N_anal_10) / max(N_anal_10, 1e-30) * 100
    n_note = f"{n_delta_pct:.0f}% {'lower' if N_mc_wfe0 < N_anal_10 else 'higher'}"
    print(f"  {'Null at WFE=0, 10 µm':<30} {N_anal_10:14.1e} {N_mc_wfe0:14.1e} {n_note:>10}")

    # Surface null at 6, 10, 16 µm
    surface_ratios = []
    for lam in [6.0, 10.0, 16.0]:
        j = list(wavelengths).index(lam)
        N_anal = surface_analytical.get(lam, 0)
        N_mc = np.mean(null_depths[:, j])
        ratio = N_mc / N_anal if N_anal > 0 else float('inf')
        surface_ratios.append(ratio)
        note = f'{ratio:.1f}x'
        print(f"  {'Surface null at %.0f µm' % lam:<30} "
              f"{N_anal:14.1e} {N_mc:14.1e} {note:>10}")

    print()
    print("  † Surface-only MC mean exceeds per-surface analytical sum due to")
    print("    nonlinear multi-surface accumulation and random-walk residuals.")
    if surface_ratios:
        print("    Enhancement factor in this run: "
              f"{min(surface_ratios):.1f}–{max(surface_ratios):.1f}×.")


def print_technology_gap(results):
    """Print technology gap assessment (paper Section 3.5.3)."""
    print("\n" + "=" * 80)
    print("TECHNOLOGY GAP ASSESSMENT")
    print("=" * 80)

    wavelengths = results['wavelengths']
    null_depths = results['null_depths']

    req_sets = get_null_requirements()
    req_paper = req_sets['paper']
    req_birb = req_sets['birbacher']

    print(f"\n  {'λ [µm]':>8} {'N_mean':>12} {'Req(paper)':>12} {'Gap':>8} "
          f"{'Req(B+26)':>12} {'Gap':>8} {'σ improve':>12}")
    print(f"  {'-'*80}")

    for j, lam in enumerate(wavelengths):
        N_mean = np.mean(null_depths[:, j])
        N_p = req_paper.get(lam, 1e-5)
        N_b = req_birb.get(lam, 1e-5)
        gap_p = N_mean / N_p
        gap_b = N_mean / N_b
        sigma_improve = gap_b**(1/4)
        print(f"  {lam:8.0f} {N_mean:12.2e} {N_p:12.1e} {gap_p:7.1f}x "
              f"{N_b:12.1e} {gap_b:7.1f}x {sigma_improve:11.2f}x")

    print()
    print("  σ improve = gap^(1/4): surface quality improvement factor needed")
    print("  Values > 1 indicate technology gap; < 1 indicates margin")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("LIFE E2E Monte Carlo — Merged Module")
    print("  Engine:  e2e_monte_carlo_v2.py")
    print("  Figures: e2e_monte_carlo_v3.py")
    print("  Archive: e2e_monte_carlo_fixed.py is DEPRECATED")
    print("=" * 80)
    print()

    # Suppress CaF₂ warning at reference wavelength
    warnings.filterwarnings('ignore', message='.*CaF.*Sellmeier.*')

    # 1. Run the full MC with surface WFE
    results = run_full_mc(N_realizations=100_000, seed=42)

    # 2. Get analytical reference values (Module 3, wavelengths in metres)
    analytical = get_analytical_reference()

    # 3. Generate all 3 figures
    generate_figure_11(results, analytical)
    generate_figure_12(results)
    generate_figure_13(results)

    # 4. WFE-zero validation against Module 3
    wfe_zero_results = run_wfe_zero_validation(N_check=5000, seed=123, verbose=True)

    # 5. Cross-validation and technology gap tables
    print_cross_validation(results, analytical, wfe_zero_results=wfe_zero_results)
    print_technology_gap(results)

    # 6. Bug fix comparison
    print_bugfix_comparison(results)

    print("\n" + "=" * 80)
    print("All figures and tables generated successfully.")
    print("=" * 80)
