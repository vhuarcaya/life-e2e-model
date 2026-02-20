"""
LIFE End-to-End Wavefront Propagation Study -- Module 4: Surface Sensitivity Ranking
=====================================================================================

Author:  Victor Huarcaya (University of Bern)
Version: 3.0  (codebase reorganisation Phase A, Step 5)
Date:    2026-02-14

Purpose:
    For each of the ~20 optical surfaces in the LIFE combiner, compute the
    sensitivity of throughput (dT/dWFE) and null depth (dN/dWFE) to wavefront
    error at that surface. Rank all surfaces to determine which need lam/100
    quality vs which can tolerate lam/20.

    The key insight: a surface's sensitivity depends on THREE factors:
      1. Its POSITION in the chain (pre-combination vs post-combination
         vs post-fiber)
      2. Whether it is COMMON (both arms) or DIFFERENTIAL (one arm only)
      3. Its impact on COUPLING (throughput) vs NULL (via intensity mismatch)

UNIT CONVENTION
===============
**Module 4 works entirely in SI METRES for wavelength.**
No unit conversion is needed -- the imported null_requirement_curve()
also accepts metres.

Physics references:
    - Module 1: Zernike coupling sensitivity (Marechal + FFT overlap)
    - Module 3: Null depth = 1/4(d_phi^2 + dI^2), with dI from
      differential coupling
    - Birbacher+2026: NICE error budget Table 2
    - Ruilier & Cassaing (2001): fiber coupling formalism

    v3.0 changes vs v2.0:
      - null_requirement_curve() imported from m3_null_error_propagation
        (removed embedded copy)
      - Unicode replaced with LaTeX/ASCII in print/plot labels

    v2.0 changes from v1.0:
      1. FIXED quality requirement inversion: (4N)^{1/4} -> (16N)^{1/4}
         v1 used wrong small-sigma approximation N ~ 1/4 x^4 (4x too large);
         correct is N ~ x^4/16 where x = 2*pi*sigma/lam.  All differential
         surface specs were ~41% too tight (lam/137 -> lam/97).
      2. FIXED CMR handling: v1 ignored common-mode rejection for surfaces
         flagged is_differential=True.  Now ALL surfaces use the unified
         formula  wfe_diff = wfe_total * (1 - CMR).  This corrects BS
         surfaces from 200 nm -> 140 nm differential WFE.
      3. ADDED post_combination flag: distinguishes pre-fiber surfaces that
         are AFTER beam combination (fiber coupling OAP, bandpass dichroics
         in detection chain) from those that are before combination.
         Post-combination surfaces cannot create differential intensity;
         their null contribution is zero regardless of CMR.
      4. FIXED docstring approximation: N ~ x^4/16, not 1/4 x^4.
      5. Removed unused scipy.special.j1 import.

Key results (v2.0):
    - APS periscope mirrors are the #1 most sensitive surfaces
      (differential, pre-fiber)
    - MMZ beamsplitters are #2 (differential R/T imbalance + chromatic phase)
    - Cross-combiner roof mirrors #3 (differential, two surfaces)
    - Common-mode surfaces (collector, OAP, delay line) contribute via
      manufacturing tolerance (1 - CMR fraction)
    - Post-combination surfaces (fiber coupling OAP) affect only throughput,
      not null
    - Post-fiber surfaces (spectrograph, detector) affect only throughput,
      not null
    - lam/97 at 6 um required for APS mirrors; lam/20 adequate for
      spectrograph
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# =============================================================================
# Import null requirement curve from Module 3 (canonical source)
# =============================================================================
from m3_null_error_propagation import null_requirement_curve


# ============================================================================
# LIFE mission parameters (shared with Modules 1-3)
# ============================================================================

LAMBDA_REF = 10.0e-6       # [m]  reference wavelength
D_BEAM = 20.0e-3           # [m]  collimated beam diameter


# ============================================================================
# 1. Optical surface catalogue with sensitivity metadata
# ============================================================================

def define_surface_catalogue():
    """
    Define all optical surfaces with metadata relevant to sensitivity ranking.

    Each surface has:
      - name: component name
      - section: Receiving / Correcting / Nulling / Cross-combiner / Detection
      - count: number of surfaces (reflections/transmissions) in the path
      - surface_type: 'mirror', 'BS', 'dichroic', 'fiber', 'detector', 'mask'
      - is_differential: True if surface is in only ONE arm (creates dI directly)
                         False if in BOTH arms (common-mode WFE partially cancels)
      - pre_fiber: True if before the spatial filter (affects coupling AND
                   potentially null)
                   False if after the spatial filter (affects only throughput)
      - post_combination: True if surface is AFTER beam combination
                          (cannot create differential intensity;
                          null contribution = 0)
                          False if surface is before combination
                          (can affect null)
      - common_mode_rejection: fraction of WFE that cancels in null
                               (0 = none, 1 = all)
                               For purely differential (APS, MMZ fold): 0.0
                               For BS (R/T asymmetry): 0.3
                               For quasi-common (manufacturing tolerance):
                               ~0.85-0.90
      - wfe_quality_waves: typical surface quality (waves RMS at 10 um)
      - notes: physical description

    The catalogue follows the optical train from the LIFE SPIE 2024 paper
    (Fig. 3):
      Collector -> Steering -> Pol. comp -> OAP -> DM -> Delay line ->
      Bandpass dichroics -> Science/ctrl dichroic -> APS -> MMZ ->
      Cross-combiner -> Fiber coupling OAP -> Fiber -> Spectrograph ->
      Detector

    v2.0 changes:
      - Added post_combination flag for fiber coupling OAP
      - Moved bandpass dichroics to Correcting section (before APS/MMZ)
      - Cold pupil mask omitted (aperture stop, zero WFE contribution)
    """

    surfaces = [
        # ===== RECEIVING OPTICS (per beam, in both arms -> common mode) =====
        {
            'name': 'Collector mirror',
            'section': 'Receiving',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.90,  # 90% common, 10% differential from manufacturing
            'wfe_quality_waves': 0.02,      # lam/50 at 10 um
            'notes': '2m spherical, gold. Identical for all 4 collectors.'
        },
        {
            'name': 'Steering mirror',
            'section': 'Receiving',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.85,
            'wfe_quality_waves': 0.01,      # lam/100 (flat mirror, easier)
            'notes': 'Flat fold, tip/tilt actuated. Per beam.'
        },
        {
            'name': 'Polarization compensator',
            'section': 'Receiving',
            'count': 2,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.80,
            'wfe_quality_waves': 0.01,
            'notes': 'Two fold mirrors. Per beam. Similar but not identical paths.'
        },

        # ===== CORRECTING OPTICS (per beam, common mode) =====
        {
            'name': 'OAP beam compressor',
            'section': 'Correcting',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.85,
            'wfe_quality_waves': 0.03,      # lam/33 -- harder for off-axis parabola
            'notes': 'Off-axis parabola, compresses to ~20mm. Most demanding figure.'
        },
        {
            'name': 'Deformable mirror',
            'section': 'Correcting',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.80,
            'wfe_quality_waves': 0.01,      # lam/100 -- actuated, should be very good
            'notes': 'Tip/tilt + higher-order correction. Active surface.'
        },
        {
            'name': 'Delay line mirrors',
            'section': 'Correcting',
            'count': 4,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.70,  # more bounces -> more differential risk
            'wfe_quality_waves': 0.01,      # lam/100 flat mirrors
            'notes': 'Trombone retro-reflector, 4 reflections.'
        },
        {
            'name': 'Bandpass dichroics',
            'section': 'Correcting',           # v2: moved from Detection (before MMZ)
            'count': 2,
            'surface_type': 'dichroic',
            'is_differential': False,
            'pre_fiber': True,
            'post_combination': False,         # before APS/MMZ -> pre-combination
            'common_mode_rejection': 0.90,
            'wfe_quality_waves': 0.02,
            'notes': 'Split 6-16 um into sub-bands. Per beam, before nulling.'
        },
        {
            'name': 'Science/control dichroic',
            'section': 'Correcting',
            'count': 1,
            'surface_type': 'dichroic',
            'is_differential': False,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.85,
            'wfe_quality_waves': 0.02,
            'notes': 'Splits <4 um to control, >4 um to science.'
        },

        # ===== NULLING OPTICS -- HERE IS WHERE DIFFERENTIAL MATTERS =====
        {
            'name': 'APS periscope mirror 1',
            'section': 'Nulling',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': True,         # ONLY in one arm!
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.0,    # Purely differential
            'wfe_quality_waves': 0.01,
            'notes': 'First mirror of 3-mirror periscope. In ONE beam only.'
        },
        {
            'name': 'APS periscope mirror 2',
            'section': 'Nulling',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': True,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.0,
            'wfe_quality_waves': 0.01,
            'notes': 'Second mirror of periscope (roof). In ONE beam only.'
        },
        {
            'name': 'APS periscope mirror 3',
            'section': 'Nulling',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': True,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.0,
            'wfe_quality_waves': 0.01,
            'notes': 'Third mirror of periscope. In ONE beam only.'
        },
        {
            'name': 'MMZ beamsplitter 1',
            'section': 'Nulling',
            'count': 1,
            'surface_type': 'BS',
            'is_differential': True,         # R vs T create asymmetric WFE
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.3,    # Partially common but R != T paths
            'wfe_quality_waves': 0.02,
            'notes': '50/50 CaF2 BS. Beam 1 reflects, Beam 2 transmits.'
        },
        {
            'name': 'MMZ beamsplitter 2',
            'section': 'Nulling',
            'count': 1,
            'surface_type': 'BS',
            'is_differential': True,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.3,
            'wfe_quality_waves': 0.02,
            'notes': 'Second BS of MMZ. Complementary R/T to BS1.'
        },
        {
            'name': 'MMZ fold mirror',
            'section': 'Nulling',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': True,         # Only in one arm of MMZ
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.0,
            'wfe_quality_waves': 0.01,
            'notes': 'Internal MMZ correction mirror in one arm.'
        },

        # ===== CROSS-COMBINER (after first null, before fiber) =====
        {
            'name': 'Cross-combiner BS',
            'section': 'Cross-combiner',
            'count': 1,
            'surface_type': 'BS',
            'is_differential': True,         # Combines two different null channels
            'pre_fiber': True,
            'post_combination': False,       # This IS the combination stage
            'common_mode_rejection': 0.3,
            'wfe_quality_waves': 0.02,
            'notes': '50/50 BS combining dark outputs A and B.'
        },
        {
            'name': 'Cross-combiner roof mirrors',
            'section': 'Cross-combiner',
            'count': 2,
            'surface_type': 'mirror',
            'is_differential': True,
            'pre_fiber': True,
            'post_combination': False,
            'common_mode_rejection': 0.0,
            'wfe_quality_waves': 0.01,
            'notes': 'Roof mirror pair for pi/2 phase. In one path only.'
        },

        # ===== POST-COMBINATION, PRE-FIBER (throughput only, NOT null) =====
        {
            'name': 'Fiber coupling OAP',
            'section': 'Detection',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': True,               # before fiber
            'post_combination': True,        # v2: AFTER combination -> no null impact
            'common_mode_rejection': 1.0,    # effectively infinite CMR
            'wfe_quality_waves': 0.03,       # OAP is harder to figure
            'notes': 'Post-combination. Focuses combined beam onto fiber. '
                     'Affects throughput only.'
        },

        # ===== POST-FIBER (affect throughput only) =====
        {
            'name': 'Spectrograph collimator',
            'section': 'Detection',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': False,
            'post_combination': True,
            'common_mode_rejection': 1.0,    # irrelevant
            'wfe_quality_waves': 0.05,       # lam/20 is fine
            'notes': 'Post-fiber. Affects only throughput, not null.'
        },
        {
            'name': 'Spectrograph disperser',
            'section': 'Detection',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': False,
            'post_combination': True,
            'common_mode_rejection': 1.0,
            'wfe_quality_waves': 0.05,
            'notes': 'Grating or prism reflection. Post-fiber.'
        },
        {
            'name': 'Spectrograph camera',
            'section': 'Detection',
            'count': 1,
            'surface_type': 'mirror',
            'is_differential': False,
            'pre_fiber': False,
            'post_combination': True,
            'common_mode_rejection': 1.0,
            'wfe_quality_waves': 0.05,
            'notes': 'Camera mirror. Post-fiber.'
        },
    ]

    return surfaces


# ============================================================================
# 2. Coupling sensitivity: WFE -> delta_eta (from Module 1)
# ============================================================================

def coupling_sensitivity(wfe_rms, wavelength=LAMBDA_REF):
    """
    Coupling efficiency ratio eta/eta_0 for given WFE (Marechal approx.).

    eta/eta_0 ~ exp(-(2*pi * sigma_WFE / lam)^2)

    Parameters
    ----------
    wfe_rms : float or array [m] -- RMS wavefront error
    wavelength : float [m]

    Returns
    -------
    eta_ratio : float or array -- coupling ratio [0, 1]
    """
    return np.exp(-(2.0 * np.pi * wfe_rms / wavelength)**2)


def coupling_sensitivity_derivative(wfe_rms, wavelength=LAMBDA_REF):
    """
    Derivative of coupling with respect to WFE:

    d(eta/eta_0)/d(sigma) = -2(2*pi/lam)^2 * sigma * exp(-(2*pi*sigma/lam)^2)

    Returns d_eta/d_sigma [1/m], the coupling change per unit WFE increase.
    """
    x = (2.0 * np.pi / wavelength)**2
    return -2.0 * x * wfe_rms * np.exp(-x * wfe_rms**2)


# ============================================================================
# 3. Null depth sensitivity: WFE -> delta_N (from Module 3)
# ============================================================================

def null_from_differential_wfe(wfe_diff_rms, wavelength=LAMBDA_REF):
    """
    Null depth contribution from differential WFE between two arms.

    The differential WFE creates an intensity mismatch:
      dI = |eta_1 - eta_2| / (eta_1 + eta_2)

    For one arm with WFE = sigma and the other with WFE = 0:
      eta_1 = exp(-(2*pi*sigma/lam)^2),  eta_2 = 1
      dI = (1 - eta_1) / (1 + eta_1)

    Null depth: N = 1/4 * dI^2

    Small-sigma expansion (for reference):
      eta ~ 1 - x^2  where x = 2*pi*sigma/lam
      dI ~ x^2/2
      N ~ x^4/16 = (2*pi*sigma/lam)^4 / 16        [v2: corrected from 1/4 x^4]

    NOTE: The code uses the EXACT formula, not the approximation.
    """
    eta = coupling_sensitivity(wfe_diff_rms, wavelength)
    dI = np.abs(1.0 - eta) / (1.0 + eta)
    return 0.25 * dI**2


def null_sensitivity_derivative(wfe_diff_rms, wavelength=LAMBDA_REF):
    """
    Derivative of null depth with respect to differential WFE:

    dN/d_sigma = dN/d(dI) * d(dI)/d_eta * d_eta/d_sigma

    Returns dN/d_sigma [1/m].
    """
    eta = coupling_sensitivity(wfe_diff_rms, wavelength)
    # d(dI)/d_eta = -2/(1+eta)^2 (for eta_1 = eta, eta_2 = 1)
    dI = (1.0 - eta) / (1.0 + eta)
    ddI_deta = -2.0 / (1.0 + eta)**2
    # d_eta/d_sigma
    x = (2.0 * np.pi / wavelength)**2
    deta_dsigma = -2.0 * x * wfe_diff_rms * eta
    # dN/d(dI) = 1/2 * dI
    dN_ddI = 0.5 * dI
    return dN_ddI * ddI_deta * deta_dsigma


# ============================================================================
# 4. Compute sensitivity for each surface
# ============================================================================

def compute_surface_sensitivities(surfaces, wavelengths=None):
    """
    For each surface, compute:
      1. Throughput sensitivity: delta_T from WFE at that surface
      2. Null depth sensitivity: delta_N from WFE at that surface
      3. Combined figure of merit: prioritized ranking

    The key physics:
      - Pre-combination, pre-fiber surfaces affect coupling AND potentially null
      - Post-combination, pre-fiber surfaces affect only coupling (not null)
      - Post-fiber surfaces affect only throughput (via vignetting, not coupling)
      - Differential surfaces (one arm only) directly degrade null depth
      - Common-mode surfaces (both arms) partially cancel in null

    v2.0 changes from v1.0:
      - Unified CMR treatment: wfe_diff = wfe_total * (1 - CMR) for ALL
        surfaces (v1 ignored CMR when is_differential=True)
      - Post-combination flag: surfaces after beam combination get N = 0
      - Quality requirement inversion: uses correct N ~ x^4/16 approximation
        (v1 used N ~ 1/4 x^4, making specs ~41% too tight)

    Parameters
    ----------
    surfaces : list of dicts -- from define_surface_catalogue()
    wavelengths : array [m] -- if None, uses [6, 8, 10, 12, 16] um

    Returns
    -------
    results : list of dicts, one per surface, with sensitivity data
    """
    if wavelengths is None:
        wavelengths = np.array([6e-6, 8e-6, 10e-6, 12e-6, 16e-6])

    results = []

    for surf in surfaces:
        n = surf['count']
        cmr = surf['common_mode_rejection']
        is_diff = surf['is_differential']
        pre_fiber = surf['pre_fiber']
        post_comb = surf.get('post_combination', False)
        quality = surf['wfe_quality_waves'] * LAMBDA_REF  # convert to meters

        # Total WFE from this component: RSS of n surfaces
        wfe_total = quality * np.sqrt(n)

        # ----------------------------------------------------------------
        # v2: Unified CMR treatment for ALL surfaces
        # ----------------------------------------------------------------
        # For purely differential (CMR=0): wfe_diff = wfe_total
        # For BS (CMR=0.3): wfe_diff = 0.70 x wfe_total
        # For common-mode (CMR=0.85): wfe_diff = 0.15 x wfe_total
        # For post-combination: N = 0 regardless (enforced below)
        wfe_differential = wfe_total * (1.0 - cmr)

        # Compute sensitivities at each wavelength
        coupling_loss = {}
        null_contribution = {}
        coupling_deriv = {}
        null_deriv = {}

        for lam in wavelengths:
            lam_key = f"{lam*1e6:.0f}"

            if pre_fiber:
                # Coupling loss from WFE (affects throughput)
                eta_ratio = coupling_sensitivity(wfe_total, lam)
                coupling_loss[lam_key] = 1.0 - eta_ratio
                coupling_deriv[lam_key] = coupling_sensitivity_derivative(
                    wfe_total, lam)

                # Null depth from differential WFE
                # Post-combination surfaces: N = 0 (cannot create dI)
                if post_comb:
                    null_contribution[lam_key] = 0.0
                    null_deriv[lam_key] = 0.0
                else:
                    N_wfe = null_from_differential_wfe(wfe_differential, lam)
                    null_contribution[lam_key] = N_wfe
                    null_deriv[lam_key] = null_sensitivity_derivative(
                        wfe_differential, lam)
            else:
                # Post-fiber: minor throughput loss from geometric vignetting
                # Approximation: ~0.1% per lam/20 surface for post-fiber optics
                coupling_loss[lam_key] = 0.001 * (quality / (LAMBDA_REF / 20))
                coupling_deriv[lam_key] = 0.0
                null_contribution[lam_key] = 0.0
                null_deriv[lam_key] = 0.0

        # Figure of merit: null depth contribution at 6 um (most stringent)
        null_at_6 = null_contribution.get('6', 0.0)
        null_at_ref = null_contribution.get('10', 0.0)
        fom = null_at_6

        # ----------------------------------------------------------------
        # v2: Corrected quality requirement inversion
        # ----------------------------------------------------------------
        # From exact: N = 1/4 * dI^2, with dI = (1-eta)/(1+eta)
        # Small-sigma:  N ~ x^4/16  where x = 2*pi*sigma/lam
        #               [v2: was 1/4 x^4 in v1]
        # Invert:   sigma = (lam/(2*pi)) * (16N)^{1/4}
        #           [v2: was (4N)^{1/4}]
        #
        # For surfaces with CMR > 0, the physical WFE can be larger since
        # only (1-CMR) fraction is differential:
        #   sigma_phys = sigma_diff / (1 - CMR)

        N_target = 1e-5     # requirement at 6 um
        lam_strict = 6e-6   # most stringent wavelength
        N_budget_surfaces = 9  # number of surfaces sharing the budget

        if pre_fiber and not post_comb:
            N_alloc = N_target / N_budget_surfaces
            # Maximum differential WFE for N_alloc
            # v2: (16 * N_alloc)^{1/4}  [corrected from (4 * N_alloc)^{1/4}]
            sigma_diff_max = (lam_strict / (2 * np.pi)) * (16 * N_alloc)**0.25
            # Physical WFE accounting for CMR
            cmr_eff = max(1.0 - cmr, 0.01)  # avoid division by zero
            sigma_phys_max = sigma_diff_max / cmr_eff
            quality_req = sigma_phys_max / lam_strict  # in waves at 6 um
        else:
            # Post-combination or post-fiber: lam/20 is adequate
            sigma_phys_max = lam_strict / 20
            quality_req = 1.0 / 20

        # Express as lam/X
        quality_spec = 1.0 / quality_req if quality_req > 0 else 20

        results.append({
            'name': surf['name'],
            'section': surf['section'],
            'count': n,
            'is_differential': is_diff,
            'pre_fiber': pre_fiber,
            'post_combination': post_comb,
            'cmr': cmr,
            'wfe_total_nm': wfe_total * 1e9,
            'wfe_differential_nm': wfe_differential * 1e9,
            'coupling_loss': coupling_loss,
            'null_contribution': null_contribution,
            'coupling_deriv': coupling_deriv,
            'null_deriv': null_deriv,
            'fom': fom,
            'quality_req_lambda_over': quality_spec,
            'quality_current_waves': surf['wfe_quality_waves'],
            'notes': surf['notes'],
        })

    # Sort by figure of merit (highest = most sensitive)
    results.sort(key=lambda x: x['fom'], reverse=True)

    # Add rank
    for i, r in enumerate(results):
        r['rank'] = i + 1

    return results


# ============================================================================
# 5. WFE tolerance curves per surface category
# ============================================================================

def compute_tolerance_curves(wavelength=6e-6):
    """
    Compute null depth vs WFE for differential and common-mode surfaces.

    Returns arrays for plotting tolerance curves.
    """
    wfe_range = np.linspace(1e-9, 500e-9, 500)  # 1 to 500 nm

    # Purely differential (APS, MMZ fold)
    N_diff = null_from_differential_wfe(wfe_range, wavelength)

    # Partially common-mode (CMR = 0.85, like OAP)
    N_cmr85 = null_from_differential_wfe(wfe_range * 0.15, wavelength)

    # Partially common-mode (CMR = 0.70, like delay line)
    N_cmr70 = null_from_differential_wfe(wfe_range * 0.30, wavelength)

    # BS (CMR = 0.30, R/T asymmetry)
    N_cmr30 = null_from_differential_wfe(wfe_range * 0.70, wavelength)

    return wfe_range, N_diff, N_cmr85, N_cmr70, N_cmr30


# ============================================================================
# 6. Main analysis and figure generation
# ============================================================================

def run_full_analysis():
    """
    Run the complete Module 4 analysis and produce publication-quality figures.

    Produces 4 key figures:
      Fig 13: Surface sensitivity ranking (tornado chart)
      Fig 14: Null depth vs WFE tolerance curves by surface category
      Fig 15: Wavelength-dependent sensitivity for top surfaces
      Fig 16: Required surface quality specification (lam/X) for each surface
    """

    print("=" * 70)
    print("LIFE E2E Module 4: Surface Sensitivity Ranking (v3.0)")
    print("=" * 70)

    surfaces = define_surface_catalogue()
    wavelengths = np.array([6e-6, 8e-6, 10e-6, 12e-6, 16e-6])

    results = compute_surface_sensitivities(surfaces, wavelengths)

    # ========================================================================
    # Print ranked table
    # ========================================================================
    print("\n--- Surface Sensitivity Ranking (by null contribution at 6 um) ---")
    print(f"{'Rank':>4s}  {'Surface':>30s}  {'Diff?':>5s}  {'Pre-C':>5s}  "
          f"{'CMR':>5s}  {'WFE_diff':>8s}  {'dN@6um':>10s}  {'dN@10um':>10s}  "
          f"{'d_eta@10um':>10s}  {'Req':>8s}")
    print(f"{'':>4s}  {'':>30s}  {'':>5s}  {'':>5s}  "
          f"{'':>5s}  {'[nm]':>8s}  {'':>10s}  {'':>10s}  "
          f"{'[%]':>10s}  {'[lam/X]':>8s}")
    print("-" * 120)

    for r in results:
        diff_str = "YES" if r['is_differential'] else "no"
        # v2: "Pre-C" = pre-combination (not just pre-fiber)
        if r['post_combination']:
            prec_str = "no"
        elif r['pre_fiber']:
            prec_str = "YES"
        else:
            prec_str = "no"
        null_6 = r['null_contribution'].get('6', 0.0)
        null_10 = r['null_contribution'].get('10', 0.0)
        coup_10 = r['coupling_loss'].get('10', 0.0) * 100
        req = r['quality_req_lambda_over']
        req_str = f"lam/{req:.0f}" if req < 1000 else "relaxed"

        print(f"{r['rank']:>4d}  {r['name']:>30s}  {diff_str:>5s}  {prec_str:>5s}  "
              f"{r['cmr']:>5.2f}  {r['wfe_differential_nm']:>8.1f}  "
              f"{null_6:>10.1e}  {null_10:>10.1e}  {coup_10:>10.2f}  {req_str:>8s}")

    # ========================================================================
    # Figure 13: Surface sensitivity ranking (tornado chart)
    # ========================================================================
    print("\n--- Figure 13: Surface sensitivity ranking ---")

    fig13, (ax13a, ax13b) = plt.subplots(1, 2, figsize=(15, 9))

    # Panel A: Null depth contribution at 6 um (most stringent)
    names = [r['name'] for r in results]
    null_6_vals = [r['null_contribution'].get('6', 0.0) for r in results]
    # Color: red = differential, blue = common, gray = post-combination/post-fiber
    colors_null = []
    for r in results:
        if r['post_combination'] or not r['pre_fiber']:
            colors_null.append('#999999')
        elif r['is_differential']:
            colors_null.append('#d62728')
        else:
            colors_null.append('#1f77b4')
    edge_colors = ['black' if (r['pre_fiber'] and not r['post_combination'])
                   else 'gray' for r in results]

    y_pos = np.arange(len(names))

    # Replace zeros with tiny value for log scale
    null_6_plot = [max(v, 1e-15) for v in null_6_vals]

    ax13a.barh(y_pos, null_6_plot, color=colors_null, edgecolor=edge_colors,
              linewidth=1.5, height=0.7)
    ax13a.set_yticks(y_pos)
    ax13a.set_yticklabels(names, fontsize=9)
    ax13a.set_xlabel(r'Null depth contribution at 6 $\mu$m', fontsize=12)
    ax13a.set_title(r'Null depth sensitivity ($\lambda$ = 6 $\mu$m)',
                    fontsize=13)
    ax13a.set_xscale('log')
    ax13a.set_xlim(1e-14, 1e-3)
    ax13a.grid(True, alpha=0.3, which='both', axis='x')
    ax13a.invert_yaxis()

    # Legend
    diff_patch = mpatches.Patch(color='#d62728', label='Differential (pre-comb)')
    comm_patch = mpatches.Patch(color='#1f77b4', label='Common-mode (pre-comb)')
    post_patch = mpatches.Patch(color='#999999',
                                label='Post-combination / post-fiber')
    ax13a.legend(handles=[diff_patch, comm_patch, post_patch,
                          plt.Line2D([0], [0], color='red', ls='--', lw=2,
                                     label=r'Req: $10^{-5}$'),
                          plt.Line2D([0], [0], color='orange', ls=':', lw=1.5,
                                     label=r'Budget: $10^{-5}$/9')],
                fontsize=8, loc='lower right')
    ax13a.axvline(x=1e-5, color='red', ls='--', lw=2)
    ax13a.axvline(x=1e-5/9, color='orange', ls=':', lw=1.5)

    # Panel B: Coupling loss (throughput impact)
    coup_10_vals = [r['coupling_loss'].get('10', 0.0) * 100 for r in results]

    ax13b.barh(y_pos, coup_10_vals, color=colors_null, edgecolor=edge_colors,
              linewidth=1.5, height=0.7)
    ax13b.set_yticks(y_pos)
    ax13b.set_yticklabels(names, fontsize=9)
    ax13b.set_xlabel(r'Coupling loss at 10 $\mu$m [%]', fontsize=12)
    ax13b.set_title(r'Throughput sensitivity ($\lambda$ = 10 $\mu$m)',
                    fontsize=13)
    ax13b.grid(True, alpha=0.3, axis='x')
    ax13b.invert_yaxis()

    fig13.tight_layout()
    fig13.savefig('fig13_surface_ranking.png', dpi=200,
                 bbox_inches='tight')
    print("  Saved: fig13_surface_ranking.png")

    # ========================================================================
    # Figure 14: Null depth vs WFE tolerance curves
    # ========================================================================
    print("\n--- Figure 14: Null depth vs WFE tolerance curves ---")

    fig14, (ax14a, ax14b) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: at 6 um (most stringent)
    wfe, N_diff, N_cmr85, N_cmr70, N_cmr30 = compute_tolerance_curves(6e-6)

    ax14a.semilogy(wfe * 1e9, N_diff, 'r-', lw=2.5,
                   label='Differential (APS, MMZ fold)')
    ax14a.semilogy(wfe * 1e9, N_cmr30, 'm-', lw=2,
                   label='BS (CMR=0.30)')
    ax14a.semilogy(wfe * 1e9, N_cmr70, 'b--', lw=2,
                   label='Delay line (CMR=0.70)')
    ax14a.semilogy(wfe * 1e9, N_cmr85, 'g-.', lw=2,
                   label='Common-mode (CMR=0.85)')

    ax14a.axhline(y=1e-5, color='red', ls=':', lw=2, alpha=0.7)
    ax14a.text(350, 1.3e-5, r'N = $10^{-5}$ req', fontsize=10, color='red')
    ax14a.axhline(y=1e-5/9, color='orange', ls=':', lw=1.5, alpha=0.7)
    ax14a.text(350, 1.5e-6, 'Budget/9', fontsize=9, color='orange')

    for quality, wfe_nm, col in [(r'$\lambda$/100', 60, 'green'),
                                   (r'$\lambda$/50', 120, 'orange'),
                                   (r'$\lambda$/20', 300, 'red')]:
        ax14a.axvline(x=wfe_nm, color=col, ls='--', alpha=0.5)
        ax14a.text(wfe_nm + 5, 5e-4, quality, fontsize=9,
                   color=col, rotation=90)

    ax14a.set_xlabel('Surface WFE RMS [nm]', fontsize=13)
    ax14a.set_ylabel('Null depth contribution', fontsize=13)
    ax14a.set_title(r'Null sensitivity at $\lambda$ = 6 $\mu$m', fontsize=13)
    ax14a.legend(fontsize=9, loc='lower right')
    ax14a.set_ylim(1e-14, 1e-2)
    ax14a.set_xlim(0, 500)
    ax14a.grid(True, alpha=0.3, which='both')

    # Panel B: at 10 um (reference)
    wfe, N_diff, N_cmr85, N_cmr70, N_cmr30 = compute_tolerance_curves(10e-6)

    ax14b.semilogy(wfe * 1e9, N_diff, 'r-', lw=2.5,
                   label='Differential (APS, MMZ fold)')
    ax14b.semilogy(wfe * 1e9, N_cmr30, 'm-', lw=2,
                   label='BS (CMR=0.30)')
    ax14b.semilogy(wfe * 1e9, N_cmr70, 'b--', lw=2,
                   label='Delay line (CMR=0.70)')
    ax14b.semilogy(wfe * 1e9, N_cmr85, 'g-.', lw=2,
                   label='Common-mode (CMR=0.85)')

    ax14b.axhline(y=3e-5, color='red', ls=':', lw=2, alpha=0.7)
    ax14b.text(350, 3.5e-5, r'N = $3 \times 10^{-5}$ req',
               fontsize=10, color='red')

    for quality, wfe_nm, col in [(r'$\lambda$/100', 100, 'green'),
                                   (r'$\lambda$/50', 200, 'orange'),
                                   (r'$\lambda$/20', 500, 'red')]:
        if wfe_nm <= 500:
            ax14b.axvline(x=wfe_nm, color=col, ls='--', alpha=0.5)
            ax14b.text(wfe_nm + 5, 5e-4, quality, fontsize=9,
                       color=col, rotation=90)

    ax14b.set_xlabel('Surface WFE RMS [nm]', fontsize=13)
    ax14b.set_ylabel('Null depth contribution', fontsize=13)
    ax14b.set_title(r'Null sensitivity at $\lambda$ = 10 $\mu$m', fontsize=13)
    ax14b.legend(fontsize=9, loc='lower right')
    ax14b.set_ylim(1e-14, 1e-2)
    ax14b.set_xlim(0, 500)
    ax14b.grid(True, alpha=0.3, which='both')

    fig14.tight_layout()
    fig14.savefig('fig14_wfe_tolerance.png', dpi=200,
                 bbox_inches='tight')
    print("  Saved: fig14_wfe_tolerance.png")

    # ========================================================================
    # Figure 15: Wavelength-dependent sensitivity for top surfaces
    # ========================================================================
    print("\n--- Figure 15: Wavelength-dependent null sensitivity ---")

    fig15, ax15 = plt.subplots(1, 1, figsize=(10, 6))

    lam_fine = np.linspace(4e-6, 18.5e-6, 200)
    lam_fine_um = lam_fine * 1e6

    # Top 6 surfaces from ranking (with non-zero null contribution)
    top_surfaces = [r for r in results
                    if r['null_contribution'].get('10', 0) > 0][:6]

    colors_15 = ['#d62728', '#ff7f0e', '#2ca02c',
                 '#1f77b4', '#9467bd', '#8c564b']
    linestyles = ['-', '-', '--', '-', '-.', ':']

    for i, r in enumerate(top_surfaces):
        # Recompute at fine wavelength grid
        wfe_diff = r['wfe_differential_nm'] * 1e-9  # back to meters
        N_vs_lam = null_from_differential_wfe(wfe_diff, lam_fine)

        label = f"#{r['rank']} {r['name']}"
        if r['is_differential']:
            label += " (diff)"
        ax15.semilogy(lam_fine_um, N_vs_lam, color=colors_15[i],
                     ls=linestyles[i], lw=2, label=label)

    # Requirement curve
    N_req = null_requirement_curve(lam_fine)
    ax15.semilogy(lam_fine_um, N_req, 'k:', lw=2.5, label='Requirement')

    # Budget allocation (1/9 of requirement)
    ax15.semilogy(lam_fine_um, N_req / 9, 'k--', lw=1, alpha=0.5,
                 label='Budget alloc. (req/9)')

    ax15.axvspan(6, 16, alpha=0.06, color='green')
    ax15.set_xlabel(r'Wavelength [$\mu$m]', fontsize=13)
    ax15.set_ylabel('Null depth contribution', fontsize=13)
    ax15.set_title(
        'Wavelength-dependent null sensitivity -- top 6 surfaces',
        fontsize=13)
    ax15.legend(fontsize=9, loc='upper right')
    ax15.set_ylim(1e-9, 1e-2)
    ax15.set_xlim(4, 18.5)
    ax15.grid(True, alpha=0.3, which='both')

    fig15.tight_layout()
    fig15.savefig('fig15_wavelength_sensitivity.png', dpi=200,
                 bbox_inches='tight')
    print("  Saved: fig15_wavelength_sensitivity.png")

    # ========================================================================
    # Figure 16: Required surface quality specification
    # ========================================================================
    print("\n--- Figure 16: Required surface quality specifications ---")

    fig16, ax16 = plt.subplots(1, 1, figsize=(12, 8))

    # Sort by section order for this plot
    section_order = {'Nulling': 0, 'Cross-combiner': 1, 'Receiving': 2,
                     'Correcting': 3, 'Detection': 4}
    results_by_section = sorted(results,
                                 key=lambda x: (section_order.get(
                                     x['section'], 5),
                                                -x['fom']))

    names_16 = [r['name'] for r in results_by_section]
    req_values = [r['quality_req_lambda_over'] for r in results_by_section]
    current_values = [1.0 / r['quality_current_waves']
                      if r['quality_current_waves'] > 0
                      else 20 for r in results_by_section]
    sections = [r['section'] for r in results_by_section]
    is_diff = [r['is_differential'] for r in results_by_section]
    pre_f = [r['pre_fiber'] for r in results_by_section]

    y_pos = np.arange(len(names_16))

    # Color by section
    section_colors = {
        'Nulling': '#d62728',
        'Cross-combiner': '#ff7f0e',
        'Receiving': '#2ca02c',
        'Correcting': '#1f77b4',
        'Detection': '#9467bd'
    }
    bar_colors = [section_colors.get(s, 'gray') for s in sections]

    # Required quality (filled bars)
    bars_req = ax16.barh(y_pos - 0.2, req_values, height=0.35,
                         color=bar_colors, alpha=0.8, edgecolor='black',
                         linewidth=1, label=r'Required $\lambda$/X')

    # Current quality (outline bars)
    bars_cur = ax16.barh(y_pos + 0.2, current_values, height=0.35,
                         color='none', edgecolor=bar_colors, linewidth=2.5,
                         linestyle='--', label=r'Current $\lambda$/X')

    # Add markers for differential vs common
    for i, (d, p) in enumerate(zip(is_diff, pre_f)):
        marker = 'D' if d else 'o'
        color = 'red' if d else 'blue'
        ax16.plot(max(req_values[i], current_values[i]) * 1.15, y_pos[i],
                 marker=marker, color=color, ms=8,
                 mfc=color if d else 'none', mew=1.5)

    # Reference lines
    for quality, val, col in [(r'$\lambda$/100', 100, 'green'),
                                (r'$\lambda$/50', 50, 'orange'),
                                (r'$\lambda$/20', 20, 'red')]:
        ax16.axvline(x=val, color=col, ls=':', alpha=0.5)
        ax16.text(val, -1, quality, fontsize=10, color=col, ha='center')

    ax16.set_yticks(y_pos)
    ax16.set_yticklabels(names_16, fontsize=9)
    ax16.set_xlabel(r'Surface quality specification ($\lambda$/X at 6 $\mu$m)',
                    fontsize=13)
    ax16.set_title('Required vs Current Surface Quality (v3.0)', fontsize=13)
    ax16.set_xscale('log')
    ax16.set_xlim(5, 5000)
    ax16.grid(True, alpha=0.3, which='both', axis='x')
    ax16.invert_yaxis()

    # Legend
    handles = [mpatches.Patch(color=c, label=s, alpha=0.8)
               for s, c in section_colors.items()]
    handles.append(plt.Line2D([0], [0], marker='D', color='red', ls='', ms=8,
                              label='Differential'))
    handles.append(plt.Line2D([0], [0], marker='o', color='blue', ls='', ms=8,
                              mfc='none', label='Common-mode'))
    ax16.legend(handles=handles, fontsize=8, loc='lower right', ncol=2)

    fig16.tight_layout()
    fig16.savefig('fig16_quality_specs.png', dpi=200,
                 bbox_inches='tight')
    print("  Saved: fig16_quality_specs.png")

    # ========================================================================
    # Summary: Tiered quality specification table
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Surface Quality Specification Tiers (v3.0)")
    print("=" * 70)

    # v2: tier classification based on pre-combination null-contributing surfaces
    tier1 = [r for r in results if r['quality_req_lambda_over'] >= 80
             and r['pre_fiber'] and not r['post_combination']]
    tier2 = [r for r in results if 30 <= r['quality_req_lambda_over'] < 80
             and r['pre_fiber'] and not r['post_combination']]
    tier3 = [r for r in results if r['quality_req_lambda_over'] < 30
             or not r['pre_fiber'] or r['post_combination']]

    print("\nTIER 1 -- lam/80 or better (critical for null depth):")
    for r in tier1:
        print(f"  {r['name']:>35s}  req: lam/{r['quality_req_lambda_over']:.0f}  "
              f"WFE_diff: {r['wfe_differential_nm']:.0f} nm  "
              f"[{'differential' if r['is_differential'] else 'common-mode'}]")

    print("\nTIER 2 -- lam/30 to lam/80 (important for null budget):")
    for r in tier2:
        print(f"  {r['name']:>35s}  req: lam/{r['quality_req_lambda_over']:.0f}  "
              f"WFE_diff: {r['wfe_differential_nm']:.0f} nm  "
              f"[{'differential' if r['is_differential'] else 'common-mode'}]")

    print("\nTIER 3 -- lam/20 or relaxed (non-critical):")
    for r in tier3:
        reason = 'post-fiber' if not r['pre_fiber'] else (
            'post-combination' if r['post_combination'] else 'common-mode')
        print(f"  {r['name']:>35s}  req: lam/{r['quality_req_lambda_over']:.0f}  "
              f"[{reason}]")

    # Cross-check: total null from all surfaces at 6 um
    print("\n--- Cross-check: Total null from all surface WFEs ---")
    for lam_key in ['6', '10', '16']:
        N_total = sum(r['null_contribution'].get(lam_key, 0.0) for r in results)
        req_val = null_requirement_curve(float(lam_key) * 1e-6)
        print(f"  lam = {lam_key} um: Sum(N) = {N_total:.2e}  "
              f"(req = {req_val:.2e}, "
              f"fraction = {N_total/req_val*100:.0f}%)")

    # v2: Verification -- plug corrected spec back into exact formula
    print("\n--- Verification: spec -> null depth round-trip ---")
    lam_v = 6e-6
    for r in results[:8]:  # top 8
        spec = r['quality_req_lambda_over']
        sigma_phys = lam_v / spec
        cmr_eff = max(1.0 - r['cmr'], 0.01)
        sigma_diff = sigma_phys * cmr_eff
        N_check = null_from_differential_wfe(sigma_diff, lam_v)
        N_alloc = 1e-5 / 9
        print(f"  {r['name']:>30s}: lam/{spec:.0f} -> sigma_diff = "
              f"{sigma_diff*1e9:.1f} nm "
              f"-> N = {N_check:.2e}  (target: {N_alloc:.2e}, "
              f"ratio: {N_check/N_alloc:.3f})")

    plt.close('all')
    print("\nAll figures saved. Module 4 v3.0 complete.")

    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    results = run_full_analysis()
