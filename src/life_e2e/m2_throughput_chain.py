"""
LIFE End-to-End Wavefront Propagation Study -- Module 2: Throughput Chain
=========================================================================

Author:  Victor Huarcaya
Date:    2025-02
Version: 3.0 (February 2026 -- codebase reorganisation, Phase A Step 3)

Models the complete surface-by-surface throughput budget for the LIFE
nulling interferometer combiner, from collector mirror to detector.

Key features:
  - Wavelength-resolved throughput T(lam) across the full 4--18.5 um band
  - Gold mirror reflectivity from tabulated Ordal+1983/Palik data (v2 fix)
  - CaF2 beamsplitter substrate (Sellmeier + corrected absorption, v2 fix)
  - ZnSe beamsplitter substrate (Tatian 1984 Sellmeier)
  - Dielectric AR coating model
  - Data-driven throughput computed from optical train definition
  - Fiber coupling from fiber_modes library (top-hat / Gaussian, Marcuse MFD)
  - Detector QE models (Si:As BIB, MCT)
  - NICE testbed section-by-section validation vs Birbacher+2026 Table 3
  - Cumulative throughput waterfall analysis (7-bar layout, v2)

Reference architecture: Glauser et al. 2024 (SPIE 13095 130951D) Fig. 3
Validation:  Birbacher et al. 2026 (A&A, arXiv:2602.02279) Table 3

Changelog v3.0:
  - All material functions now imported from material_properties.py
    and fiber_modes.py (codebase reorganisation Phase A).
  - No locally-defined material models remain.
  - API compatibility wrappers bridge library vs legacy signatures.

Changelog v2.0:
  - Gold R: Drude model -> tabulated Ordal+1983/Palik (n,k) data.
    Fixes ~0.4% per-surface overestimate (R 99.1% -> 98.6% at 10 um),
    compounding to ~6% throughput error over 14 reflections.
  - CaF2 absorption: coefficient 0.01 -> 0.03 in exponential model.
    Fixes 3x underestimate at 10 um (alpha: 0.9 -> 2.7 cm^-1).
    CaF2 BS transmission at 10 um: 83.5% -> 58.3% (2 mm substrate).
  - Waterfall figure: 6 bars -> 7 bars. "Fiber coupling" separated from
    "Detection" to make the 18.5% top-hat mode-mismatch penalty visible.
  - Flight correction reduced: 0.3% -> 0.15% (Drude bias removed).

Changelog v1.1:
  - ZnSe: Marple (1964) two-term Sellmeier replaces single Cauchy term
  - CaF2: refactored -- single _caf2_bulk_absorption() used everywhere
  - MMZ throughput: clear separation of material/coating/splitting losses
  - Throughput engine: data-driven, iterates over train definition
  - NICE: section-by-section validation table matching Birbacher Table 3
  - Fiber coupling: standalone Marcuse-based formula (Module 1 compatible)
  - Documented omitted Glauser Fig. 3 components (PIAA, intensity ctrl, etc.)
  - Single vs dual NICE output distinction clarified

Produces:
  - Fig 5: Wavelength-resolved material properties (R_Au, T_CaF2, QE)
  - Fig 6: Cumulative throughput waterfall diagram (7 bars)
  - Fig 7: Total throughput T(lam) -- LIFE vs NICE comparison
  - Fig 8: Throughput sensitivity to component count & quality

IMPORTANT NOTES ON SCOPE:
=========================
This module computes the *material-limited* throughput -- the best-case
PCE assuming perfect alignment and zero wavefront error on all surfaces.
The additional coupling degradation from real WFE (the Marechal factor
exp(-(2*pi*sigma/lam)^2) per surface) is handled in Module 4 (Surface
Sensitivity).  Together, Modules 2+4 give the complete throughput budget.

Components omitted from the Glauser+2024 Fig. 3 baseline:
  - PIAA optics (would boost coupling from 81.5% toward ~100%)
  - Intensity control actuators (negligible throughput impact)
  - Adaptive nuller dispersion prisms (not yet designed in detail)
  - Polarization compensator in correcting section (second set)
These omissions are conservative: PIAA would improve throughput.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Import ALL material functions from canonical libraries
# =============================================================================
from material_properties import (
    # Gold
    gold_reflectivity,
    # CaF2
    caf2_sellmeier,
    caf2_absorption,
    caf2_transmission as _lib_caf2_transmission,
    # ZnSe
    znse_sellmeier,
    znse_absorption,
    znse_transmission as _lib_znse_transmission,
    # Detector
    detector_qe as _lib_detector_qe,
    # AR coating
    ar_coating_efficiency,
    # BS coating
    beamsplitter_coating_absorption,
    # Dichroic
    dichroic_transmission,
)
from fiber_modes import (
    coupling_tophat_analytical,
)

# =============================================================================
# Publication-quality plot settings
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
})

# =============================================================================
# Physical Constants
# =============================================================================
c_light = 2.998e8       # m/s
h_planck = 6.626e-34    # J s
k_boltz = 1.381e-23     # J/K
e_charge = 1.602e-19    # C

# =============================================================================
# Wavelength Grid
# =============================================================================
lam_um = np.linspace(3.5, 20.0, 500)   # um, extended range
lam_m = lam_um * 1e-6                   # m

# LIFE science band boundaries
LIFE_BAND = (6.0, 16.0)   # um, requirement
LIFE_GOAL = (4.0, 18.5)   # um, goal


# =============================================================================
# API COMPATIBILITY WRAPPERS
# =============================================================================
# The canonical libraries use (ar_coated: bool, ar_efficiency: float)
# while Module 2 v2 used a single ar_efficiency parameter (0.0 = bare).
# These thin wrappers bridge the two interfaces so the optical train
# definitions and physics functions below remain readable.
# =============================================================================

def caf2_transmission(lam_um, thickness_mm=2.0, ar_efficiency=0.0):
    """CaF2 substrate single-pass transmission (Module 2 API).

    Parameters
    ----------
    lam_um : array
        Wavelength in um.
    thickness_mm : float
        Substrate thickness in mm.
    ar_efficiency : float
        AR coating efficiency. 0.0 = bare, >0 = coated.
    """
    ar_coated = (ar_efficiency > 0.0)
    return _lib_caf2_transmission(lam_um, thickness_mm,
                                  ar_coated=ar_coated,
                                  ar_efficiency=ar_efficiency)


def znse_transmission(lam_um, thickness_mm=2.0, ar_efficiency=0.0):
    """ZnSe substrate single-pass transmission (Module 2 API).

    Parameters
    ----------
    lam_um : array
        Wavelength in um.
    thickness_mm : float
        Substrate thickness in mm.
    ar_efficiency : float
        AR coating efficiency. 0.0 = bare, >0 = coated.
    """
    ar_coated = (ar_efficiency > 0.0)
    return _lib_znse_transmission(lam_um, thickness_mm,
                                  ar_coated=ar_coated,
                                  ar_efficiency=ar_efficiency)


def detector_qe(lam_um, detector_type='SiAs_BIB'):
    """Detector QE with Module 2 legacy type names.

    Maps 'MCT_13um' -> 'HgCdTe' and 'requirement' -> flat 60%.
    All other types are passed through to the library.
    """
    if detector_type == 'MCT_13um':
        return _lib_detector_qe(lam_um, 'HgCdTe')
    elif detector_type == 'requirement':
        arr = np.atleast_1d(np.asarray(lam_um, dtype=float))
        return 0.60 * np.ones_like(arr)
    else:
        return _lib_detector_qe(lam_um, detector_type)


def fiber_coupling_tophat(lam_um, beta_opt=1.1209):
    """Top-hat beam coupling into single-mode fiber.

    eta = 2*(1 - exp(-beta^2))^2 / beta^2  (Ruilier & Cassaing 2001)
    eta_max = 81.45% at beta_opt = 1.1209.

    Wavelength-independent when the coupling lens f-ratio is
    re-optimised per sub-band (w_f proportional to lam).
    """
    eta = float(np.asarray(coupling_tophat_analytical(beta_opt).item()).flat[0])
    return eta * np.ones_like(np.atleast_1d(np.asarray(lam_um, dtype=float)))


def fiber_coupling_gaussian():
    """Gaussian -> Gaussian coupling (NICE testbed: matched beams).

    Theoretical maximum: 100% when perfectly mode-matched.
    NICE measured 63% -- limited by non-optimal alignment
    (Birbacher+2026 Sec. 5.4).
    """
    return 1.0


def dichroic_throughput(lam_um, cutoff_um=4.0, transition_width=0.3,
                        passband_T=0.95):
    """Dichroic mirror transmission (passband for lam > cutoff).

    Thin wrapper around library dichroic_transmission().
    """
    return dichroic_transmission(lam_um, cutoff_um=cutoff_um,
                                 transition_width=transition_width,
                                 passband_T=passband_T)


# =============================================================================
# MODULE 2-SPECIFIC PHYSICS: Multi-band BS substrate
# =============================================================================

def multiband_bs_substrate(lam_um, thickness_mm=2.0, ar_efficiency=0.90):
    """Best-of-both BS substrate transmission at each wavelength.

    LIFE's multi-band approach:
      lam < ~10 um:  CaF2  (low n ~ 1.4, minimal Fresnel, but absorbs > 10 um)
      lam >= ~10 um: ZnSe  (n ~ 2.4, needs AR, but transparent to 20 um)

    In practice, dichroics split the band before the MMZ; each sub-band has
    its own optimised BS substrate.
    """
    T_caf2 = caf2_transmission(lam_um, thickness_mm, ar_efficiency)
    T_znse = znse_transmission(lam_um, thickness_mm, ar_efficiency)
    return np.maximum(T_caf2, T_znse)


# =============================================================================
# MMZ BEAM COMBINER -- clean throughput bookkeeping
# =============================================================================

def mmz_material_throughput(lam_um, bs_substrate='multiband',
                            bs_thickness_mm=2.0, ar_efficiency=0.90,
                            coating_absorption=0.015,
                            mirror_quality='flight'):
    """Material losses in the MMZ beam combiner (excluding 50/50 splitting).

    Physics: each beam traverses the MMZ on a symmetric path --
    one reflection off BS1 coating, one transmission through BS2 substrate
    and coating, plus one internal fold mirror reflection.  Both arms
    see identical paths (1R + 1T), which is why the MMZ maintains
    balanced outputs for deep nulling.

    Loss sources (per beam, one pass through MMZ):
      1. BS substrate absorption   -- 1 pass through CaF2 or ZnSe
      2. BS coating absorption     -- 2 encounters (1 R-side + 1 T-side)
      3. Internal fold mirror      -- 1 gold reflection

    The 50/50 splitting is a *fundamental* loss applied separately.

    Parameters
    ----------
    bs_substrate : str
        'caf2', 'znse', or 'multiband' (best of each per wavelength).
    coating_absorption : float
        Fractional power absorption per BS coating encounter.
        Typical: 1-2% for broadband mid-IR dielectric coatings.
    mirror_quality : str
        Gold mirror quality for internal fold mirror(s).

    Returns
    -------
    T_material : array
        Material-limited throughput (multiply by 0.5 for splitting).
    components : dict
        Individual component throughputs for diagnostics.
    """
    # BS substrate transmission (one pass)
    if bs_substrate == 'caf2':
        T_substrate = caf2_transmission(lam_um, bs_thickness_mm, ar_efficiency)
    elif bs_substrate == 'znse':
        T_substrate = znse_transmission(lam_um, bs_thickness_mm, ar_efficiency)
    else:
        T_substrate = multiband_bs_substrate(lam_um, bs_thickness_mm,
                                             ar_efficiency)

    # BS coating absorption (2 encounters: 1 reflection + 1 transmission)
    T_coating = (1.0 - coating_absorption)**2 * np.ones_like(lam_um)

    # Internal fold mirror(s)
    R_gold = gold_reflectivity(lam_um, quality=mirror_quality)
    T_fold = R_gold.copy()

    T_material = T_substrate * T_coating * T_fold

    components = {
        'substrate': T_substrate,
        'coating': T_coating,
        'fold_mirror': T_fold,
    }
    return T_material, components


# =============================================================================
# OPTICAL TRAIN DEFINITION (data-driven)
# =============================================================================

def define_LIFE_optical_train(mirror_quality='flight', beam_profile='tophat',
                              n_delay_bounces=4,
                              include_cross_combiner=True):
    """Define the LIFE combiner optical train as a list of components.

    Each component is a dict with:
      - name:     human-readable label
      - section:  functional group (Receiving, Correcting, Nulling, etc.)
      - T_func:   callable(lam_um) -> throughput array
      - notes:    documentation string

    The throughput engine iterates over this list, making the train
    definition the single source of truth.

    Architecture reference: Glauser+2024 Fig. 3, Table 1 of paper.
    """
    def R_gold(lam):
        return gold_reflectivity(lam, quality=mirror_quality)

    train = []

    # ===== SECTION 1: RECEIVING OPTICS (per beam, x4) =====
    train.append({
        'name': 'Collector mirror',
        'section': 'Receiving',
        'T_func': lambda lam: R_gold(lam),
        'notes': '2m spherical, gold-coated'
    })
    train.append({
        'name': 'Steering mirror',
        'section': 'Receiving',
        'T_func': lambda lam: R_gold(lam),
        'notes': 'Flat fold, tip/tilt actuated'
    })
    train.append({
        'name': 'Pol. compensator (x2)',
        'section': 'Receiving',
        'T_func': lambda lam: R_gold(lam)**2,
        'notes': 'Two fold mirrors cancelling geometric polarization'
    })

    # ===== SECTION 2: CORRECTING OPTICS (per beam, x4) =====
    train.append({
        'name': 'OAP compressor',
        'section': 'Correcting',
        'T_func': lambda lam: R_gold(lam),
        'notes': 'Off-axis parabola, 2m -> 20mm beam'
    })
    train.append({
        'name': 'Cold pupil mask',
        'section': 'Correcting',
        'T_func': lambda lam: 0.99 * np.ones_like(lam),
        'notes': 'Conjugated with collector, ~99% from diffraction'
    })
    train.append({
        'name': 'DM',
        'section': 'Correcting',
        'T_func': lambda lam: R_gold(lam),
        'notes': 'Deformable mirror, gold-coated'
    })
    train.append({
        'name': f'Delay line (x{n_delay_bounces})',
        'section': 'Correcting',
        'T_func': lambda lam, n=n_delay_bounces: R_gold(lam)**n,
        'notes': f'Trombone retro-reflector, {n_delay_bounces} reflections'
    })
    train.append({
        'name': 'Sci/ctrl dichroic',
        'section': 'Correcting',
        'T_func': lambda lam: dichroic_throughput(lam, cutoff_um=4.0),
        'notes': 'Splits <4um to control, >4um to science. ~95% in passband.'
    })

    # ===== SECTION 3: NULLING (APS + MMZ) =====
    train.append({
        'name': 'Periscope APS (x3)',
        'section': 'Nulling',
        'T_func': lambda lam: R_gold(lam)**3,
        'notes': '3-mirror periscope for achromatic pi (one beam per pair)'
    })
    train.append({
        'name': 'MMZ material losses',
        'section': 'Nulling',
        'T_func': lambda lam: mmz_material_throughput(
            lam, mirror_quality=mirror_quality)[0],
        'notes': '1 substrate pass + 2 coating encounters + 1 fold mirror'
    })
    train.append({
        'name': 'MMZ 50/50 splitting',
        'section': 'Nulling',
        'T_func': lambda lam: 0.50 * np.ones_like(lam),
        'notes': 'Fundamental: planet light splits between outputs'
    })

    # ===== SECTION 4: CROSS-COMBINER =====
    if include_cross_combiner:
        train.append({
            'name': 'Cross-comb. BS losses',
            'section': 'Cross-combiner',
            'T_func': lambda lam: (
                multiband_bs_substrate(lam, 2.0, 0.90)
                * (1.0 - 0.015)  # 1 coating encounter
            ) * np.ones_like(lam),
            'notes': 'BS substrate + single coating absorption'
        })
        train.append({
            'name': 'Cross-comb. split',
            'section': 'Cross-combiner',
            'T_func': lambda lam: 0.50 * np.ones_like(lam),
            'notes': 'Fundamental: 2nd Bracewell stage 50/50'
        })
        train.append({
            'name': 'Roof mirror (x2)',
            'section': 'Cross-combiner',
            'T_func': lambda lam: R_gold(lam)**2,
            'notes': 'Angled roof (2 reflections)'
        })

    # ===== SECTION 5: FIBER COUPLING =====
    # v2.0: separated from Detection to make the mode-mismatch penalty
    # visible as its own bar in the waterfall (7 sections, not 6).
    # This is where top-hat and Gaussian first diverge.
    train.append({
        'name': 'Bandpass dichroics (x2)',
        'section': 'Fiber coupling',
        'T_func': lambda lam: 0.95**2 * np.ones_like(lam),
        'notes': 'Split 6-16um into ~3 sub-bands. 2 surfaces per path.'
    })
    train.append({
        'name': 'Coupling OAP',
        'section': 'Fiber coupling',
        'T_func': lambda lam: R_gold(lam),
        'notes': 'Focuses beam onto fiber entrance'
    })

    if beam_profile == 'tophat':
        train.append({
            'name': 'Fiber coupling (mode)',
            'section': 'Fiber coupling',
            'T_func': lambda lam: fiber_coupling_tophat(lam),
            'notes': 'Top-hat -> Gaussian: 81.5% (Ruilier & Cassaing 2001)'
        })
    else:
        train.append({
            'name': 'Fiber coupling (mode)',
            'section': 'Fiber coupling',
            'T_func': lambda lam: np.ones_like(lam),
            'notes': 'Gaussian -> Gaussian: 100% when matched'
        })

    train.append({
        'name': 'Fiber Fresnel',
        'section': 'Fiber coupling',
        'T_func': lambda lam: 0.995 * np.ones_like(lam),
        'notes': 'AR-coated entrance facet (~0.5% loss)'
    })
    train.append({
        'name': 'Fiber propagation',
        'section': 'Fiber coupling',
        'T_func': lambda lam: 0.98 * np.ones_like(lam),
        'notes': 'Short length, ~0.1 dB per sub-band fiber'
    })

    # ===== SECTION 6: DETECTION =====
    train.append({
        'name': 'Spectrograph (x3)',
        'section': 'Detection',
        'T_func': lambda lam: R_gold(lam)**3,
        'notes': 'Collimator + grating/prism + camera'
    })
    train.append({
        'name': 'Detector QE',
        'section': 'Detection',
        'T_func': lambda lam: detector_qe(lam, 'SiAs_BIB'),
        'notes': 'Si:As BIB baseline. QE ~60% at 10 um.'
    })

    return train


# =============================================================================
# DATA-DRIVEN THROUGHPUT ENGINE
# =============================================================================

def compute_throughput(lam_um, train, verbose=True, ref_wavelength=10.0):
    """Compute wavelength-resolved throughput by iterating over a train definition.

    This is the single throughput engine used for both LIFE and NICE.
    The train definition (from define_LIFE_optical_train or
    define_NICE_optical_train) is the sole source of truth.

    Parameters
    ----------
    lam_um : array
        Wavelength array in um.
    train : list of dicts
        Optical train from define_*_optical_train().
    verbose : bool
        Print component-by-component budget.
    ref_wavelength : float
        Reference wavelength (um) for summary print.

    Returns
    -------
    result : dict
        'T_total'     -- total throughput array T(lam)
        'components'  -- list of (name, T_array) tuples
        'sections'    -- dict of section-level cumulative throughput
        'train'       -- the input train definition
    """
    T_running = np.ones_like(lam_um)
    components = []
    section_entry = {}    # T at entry to each section
    current_section = None

    for comp in train:
        name = comp['name']
        section = comp['section']

        # Record section entry point
        if section != current_section:
            section_entry[section] = T_running.copy()
            current_section = section

        T_comp = comp['T_func'](lam_um)
        T_running *= T_comp
        components.append((name, T_comp))

    # Build section-level throughputs
    sections = {}
    section_names = list(dict.fromkeys(c['section'] for c in train))
    for i, sec in enumerate(section_names):
        T_entry = section_entry[sec]
        # Find the exit point: entry of next section, or final
        if i + 1 < len(section_names):
            T_exit = section_entry[section_names[i + 1]]
        else:
            T_exit = T_running
        sections[sec] = T_exit / np.maximum(T_entry, 1e-30)

    if verbose:
        idx = np.argmin(np.abs(lam_um - ref_wavelength))
        print("=" * 65)
        print(f"Throughput Budget at lam = {ref_wavelength} um")
        print("=" * 65)
        print(f"{'Component':<30} {'T(lam)':<10} {'Cumulative':<10}")
        print("-" * 50)
        T_cum = 1.0
        for name, T_arr in components:
            T_val = T_arr[idx]
            T_cum *= T_val
            print(f"  {name:<28} {T_val:8.4f}   {T_cum:8.5f}")
        print("-" * 50)
        print(f"  {'TOTAL':<28} {'':8s}   {T_running[idx]:8.5f}")
        print(f"  {'TOTAL (%)':<28} {'':8s}   {T_running[idx]*100:7.3f}%")
        print("=" * 65)

    return {
        'T_total': T_running,
        'components': components,
        'sections': sections,
        'train': train,
    }


def compute_throughput_LIFE(lam_um, beam_profile='tophat',
                            mirror_quality='flight',
                            include_cross_combiner=True,
                            n_delay_bounces=4,
                            verbose=True):
    """Convenience wrapper: build LIFE train and compute throughput."""
    train = define_LIFE_optical_train(
        mirror_quality=mirror_quality,
        beam_profile=beam_profile,
        n_delay_bounces=n_delay_bounces,
        include_cross_combiner=include_cross_combiner,
    )
    return compute_throughput(lam_um, train, verbose=verbose,
                              ref_wavelength=10.0)


# =============================================================================
# NICE TESTBED MODEL + SECTION-BY-SECTION VALIDATION
# =============================================================================

def define_NICE_optical_train(mirror_quality='flight'):
    """NICE testbed optical train (Birbacher+2026 Fig. 3, Table 3).

    NICE is a single-Bracewell nuller with Gaussian input (no collector),
    operating at 4.0 um (QCL source).

    Architecture:
      Control section -> APS -> MMZ -> Spatial filter (-> Detector)

    Measured throughput at 4.0 um (Birbacher+2026 Table 3):
      Control section:    90 +/- 2%
      APS:                95 +/- 2%
      Beam combiner:      40 +/- 2%  (includes 50/50 splitting)
      Spatial filter:     63 +/- 2%
      Total:              22 +/- 1%

    Note: 22% is for a SINGLE MMZ output. With both outputs, the
    requirement is 17% per output x 2 = 34%.
    """
    def R_gold(lam):
        return gold_reflectivity(lam, quality=mirror_quality)

    train = []

    # Control section: ~4 steering mirrors + alignment losses
    # Modeled as 4 gold reflections x alignment/scattering factor
    train.append({
        'name': 'Control section',
        'section': 'Control',
        'T_func': lambda lam: R_gold(lam)**4 * 0.95,
        'notes': '~4 mirrors + scattering/alignment. Measured: 90+/-2%.'
    })

    # APS: 3-mirror periscope
    train.append({
        'name': 'APS periscope',
        'section': 'APS',
        'T_func': lambda lam: R_gold(lam)**3,
        'notes': '3 gold reflections. Measured: 95+/-2%.'
    })

    # MMZ beam combiner: 2 CaF2 BS + 2 fold mirrors + 50/50 splitting
    # Note: NICE uses the same CaF2 BS design as LIFE baseline
    train.append({
        'name': 'MMZ material',
        'section': 'Combiner',
        'T_func': lambda lam: (
            caf2_transmission(lam, thickness_mm=2.0, ar_efficiency=0.85)
            * (1.0 - 0.015)**2      # 2 coating encounters
            * R_gold(lam)**2          # 2 fold mirrors
        ),
        'notes': 'BS substrate + coating + 2 fold mirrors'
    })
    train.append({
        'name': 'MMZ splitting',
        'section': 'Combiner',
        'T_func': lambda lam: 0.50 * np.ones_like(lam),
        'notes': '50/50 splitting (single output implemented)'
    })

    # Spatial filter: Gaussian coupling -> InF3 fiber
    # Measured 63% includes: coupling alignment + Fresnel + propagation
    train.append({
        'name': 'Spatial filter',
        'section': 'Spatial filter',
        'T_func': lambda lam: 0.63 * np.ones_like(lam),
        'notes': 'Measured 63+/-2% (alignment-limited, not optimised)'
    })

    return train


def validate_NICE(lam_um, verbose=True):
    """Compute NICE throughput and compare section-by-section against
    Birbacher+2026 Table 3 measurements at lam = 4.0 um.

    Returns
    -------
    result : dict with throughput arrays
    validation : dict with section-by-section comparison
    """
    train = define_NICE_optical_train()
    result = compute_throughput(lam_um, train, verbose=False,
                                ref_wavelength=4.0)

    idx4 = np.argmin(np.abs(lam_um - 4.0))

    # Measured values from Birbacher Table 3
    measured = {
        'Control':       (0.90, 0.02),
        'APS':           (0.95, 0.02),
        'Combiner':      (0.40, 0.02),  # includes 50/50 splitting
        'Spatial filter': (0.63, 0.02),
    }

    # Model predictions per section
    modeled = {}
    T_cum_model = 1.0
    for section_name, T_section_arr in result['sections'].items():
        T_val = T_section_arr[idx4]
        modeled[section_name] = T_val
        T_cum_model *= T_val

    if verbose:
        print("\n" + "=" * 72)
        print("NICE Validation vs Birbacher+2026 Table 3  (lam = 4.0 um)")
        print("=" * 72)
        print(f"{'Section':<20} {'Measured':>10} {'Model':>10} {'Delta':>10}  "
              f"{'Status'}")
        print("-" * 72)

        T_meas_total = 1.0
        T_model_total = 1.0
        for sec in measured:
            m_val, m_err = measured[sec]
            T_meas_total *= m_val
            if sec in modeled:
                p_val = modeled[sec]
                T_model_total *= p_val
                delta = (p_val - m_val) / m_val * 100
                within = abs(p_val - m_val) <= 2 * m_err
                status = "OK" if within else "WARN"
                print(f"  {sec:<18} {m_val*100:7.1f}+/-{m_err*100:.0f}%"
                      f"   {p_val*100:7.1f}%   {delta:+6.1f}%  {status}")

        T_total_model = result['T_total'][idx4]
        print("-" * 72)
        print(f"  {'Total':<18} {T_meas_total*100:7.1f}%"
              f"     {T_total_model*100:7.1f}%"
              f"   {(T_total_model - T_meas_total)/T_meas_total*100:+6.1f}%")
        print(f"\n  Note: 22% is for single output; dual-output req. = 34%")
        print(f"  Spatial filter uses measured value (not predictive)")
        print("=" * 72)

    return result, measured


# =============================================================================
# SECTION-LEVEL THROUGHPUT (for waterfall)
# =============================================================================

def compute_section_throughputs(lam_um, beam_profile='tophat',
                                mirror_quality='flight'):
    """Compute throughput grouped by functional section for waterfall diagram.

    Uses the data-driven train definition.
    """
    train = define_LIFE_optical_train(
        mirror_quality=mirror_quality,
        beam_profile=beam_profile,
        include_cross_combiner=True,
    )
    result = compute_throughput(lam_um, train, verbose=False)
    return result['sections']


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def make_fig5_material_properties():
    """Fig 5: Wavelength-resolved material properties.

    Panel A: Gold mirror reflectivity (3 quality grades)
    Panel B: BS substrate transmission (CaF2 vs ZnSe vs multi-band)
    Panel C: Detector QE models
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # --- Panel A: Gold reflectivity ---
    ax = axes[0]
    for quality, ls, label in [('ideal', '-', 'Pristine gold'),
                                ('flight', '--', 'Flight (protected)'),
                                ('aged', ':', 'Aged coating')]:
        R = gold_reflectivity(lam_um, quality)
        ax.plot(lam_um, R * 100, ls=ls, label=label)

    ax.axhspan(98, 100, alpha=0.1, color='green')
    ax.set_ylabel('Reflectivity (%)')
    ax.set_ylim(96, 100)
    ax.legend(loc='lower right')
    ax.set_title('(a) Gold Mirror Reflectivity (Tabulated, Ordal+1983/Palik)')
    ax.axvspan(*LIFE_BAND, alpha=0.08, color='blue')

    # --- Panel B: BS substrate transmission ---
    ax = axes[1]
    T_caf2_bare = caf2_transmission(lam_um, 2.0, ar_efficiency=0.0)
    T_caf2_ar = caf2_transmission(lam_um, 2.0, ar_efficiency=0.90)
    T_znse_ar = znse_transmission(lam_um, 2.0, ar_efficiency=0.90)
    T_multi = multiband_bs_substrate(lam_um, 2.0, 0.90)

    ax.plot(lam_um, T_caf2_ar * 100, '-',
            label=r'CaF$_2$ + AR ($n \approx 1.4$)', color='C0')
    ax.plot(lam_um, T_znse_ar * 100, '-',
            label=r'ZnSe + AR ($n \approx 2.4$, Tatian)', color='C1')
    ax.plot(lam_um, T_multi * 100, '--', label='Multi-band (best of each)',
            color='black', lw=2.5, alpha=0.7)
    ax.plot(lam_um, T_caf2_bare * 100, ':',
            label=r'CaF$_2$ bare (no AR)', color='C3', alpha=0.5)

    ax.axvline(10.0, color='red', alpha=0.3, ls='--')
    ax.text(10.2, 40, r'CaF$_2$ $\to$ ZnSe' + '\ntransition',
            fontsize=8, color='red', alpha=0.6)
    ax.set_ylabel('Transmission (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_title('(b) Beamsplitter Substrate (2 mm, single pass)')
    ax.axvspan(*LIFE_BAND, alpha=0.08, color='blue')

    # --- Panel C: Detector QE ---
    ax = axes[2]
    for det, ls, label in [('SiAs_BIB', '-', 'Si:As BIB (MIRI-type)'),
                           ('MCT_13um', '--',
                            r'MCT ($\lambda_{\rm cut}$=13 $\mu$m)'),
                           ('requirement', ':', 'NICE requirement (60%)')]:
        qe = detector_qe(lam_um, det)
        ax.plot(lam_um, qe * 100, ls=ls, label=label)

    ax.set_ylabel('Quantum Efficiency (%)')
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylim(0, 85)
    ax.legend(loc='upper right')
    ax.set_title('(c) Detector Quantum Efficiency')
    ax.axvspan(*LIFE_BAND, alpha=0.08, color='blue')

    for ax in axes:
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    fig.tight_layout()
    fig.savefig('fig5_material_properties.png')
    plt.close()
    print("[OK] Fig 5 saved: material properties")


def make_fig6_waterfall():
    """Fig 6: Cumulative throughput waterfall at lam = 10 um.

    v2.0: Now shows 7 bars (Fiber coupling separated from Detection).
    The Fiber coupling bar is where top-hat and Gaussian first diverge,
    making the 18.5% mode-mismatch penalty visually obvious.
    """
    lam_ref = 10.0
    idx = np.argmin(np.abs(lam_um - lam_ref))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Top-hat (LIFE baseline)
    res_th = compute_throughput_LIFE(lam_um, 'tophat', verbose=False)
    # Gaussian (for comparison)
    res_g = compute_throughput_LIFE(lam_um, 'gaussian', verbose=False)

    # Build cumulative arrays from components
    def cumulative_from_components(components, idx):
        cum = [1.0]
        for _, T_arr in components:
            cum.append(cum[-1] * T_arr[idx])
        return cum

    cum_th = cumulative_from_components(res_th['components'], idx)
    cum_g = cumulative_from_components(res_g['components'], idx)

    # Use section-level labels for cleaner plot
    sections_th = res_th['sections']
    section_names = list(sections_th.keys())
    T_sec_th = [sections_th[s][idx] for s in section_names]

    # Section-level cumulative
    cum_sec_th = [1.0]
    for t in T_sec_th:
        cum_sec_th.append(cum_sec_th[-1] * t)

    sections_g = res_g['sections']
    T_sec_g = [sections_g[s][idx] for s in section_names]
    cum_sec_g = [1.0]
    for t in T_sec_g:
        cum_sec_g.append(cum_sec_g[-1] * t)

    x = np.arange(len(section_names) + 1)
    labels = ['Input\n(100%)'] + [s.replace(' ', '\n') for s in section_names]
    width = 0.30

    bars_th = ax.bar(x - width / 2, [c * 100 for c in cum_sec_th], width,
                     color='steelblue', alpha=0.8, edgecolor='navy',
                     label='Top-hat beam (LIFE)')
    bars_g = ax.bar(x + width / 2, [c * 100 for c in cum_sec_g], width,
                    color='coral', alpha=0.8, edgecolor='darkred',
                    label='Gaussian beam (NICE-like)')

    for bar_group in [bars_th, bars_g]:
        for bar in bar_group:
            height = bar.get_height()
            if height > 2:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom',
                        fontsize=6.5)

    ax.axhline(5.0, color='green', ls='--', alpha=0.5, lw=1.5)
    ax.text(len(section_names) + 0.3, 5.5, 'LIFE req.\n(~5% PCE)',
            fontsize=8, color='green')
    ax.axhline(20.0, color='orange', ls='--', alpha=0.5, lw=1.5)
    ax.text(len(section_names) + 0.3, 20.5, 'NICE req.\n(20% PCE)',
            fontsize=8, color='darkorange')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, ha='center')
    ax.set_ylabel('Cumulative Throughput (%)')
    ax.set_title(r'Throughput Cascade at $\lambda$ = '
                 f'{lam_ref} $\\mu$m -- LIFE Full Chain (7 sections)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.0)

    fig.tight_layout()
    fig.savefig('fig6_throughput_waterfall.png')
    plt.close()
    print("[OK] Fig 6 saved: throughput waterfall")


def make_fig7_throughput_vs_wavelength():
    """Fig 7: End-to-end PCE vs wavelength.

    Compares:
      - LIFE full (top-hat, double Bracewell)
      - LIFE full (Gaussian)
      - LIFE single Bracewell (top-hat)
      - LIFE optics only (no detector)
      - CaF2-only (showing the disaster at long lam)
      - NICE measured point (22% at 4 um)
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    res_life_th = compute_throughput_LIFE(lam_um, 'tophat',
                                          include_cross_combiner=True,
                                          verbose=False)
    res_life_g = compute_throughput_LIFE(lam_um, 'gaussian',
                                         include_cross_combiner=True,
                                         verbose=False)
    res_sb_th = compute_throughput_LIFE(lam_um, 'tophat',
                                        include_cross_combiner=False,
                                        verbose=False)

    # Optics-only (remove detector QE)
    qe = detector_qe(lam_um, 'SiAs_BIB')
    T_optics_only = res_life_th['T_total'] / np.maximum(qe, 0.01)

    # CaF2-only throughput (replace multiband with CaF2)
    T_caf2_bs = caf2_transmission(lam_um, 2.0, 0.90)
    T_multi_bs = multiband_bs_substrate(lam_um, 2.0, 0.90)
    ratio_bs = T_caf2_bs / np.maximum(T_multi_bs, 1e-20)
    T_caf2_only = res_life_th['T_total'] * ratio_bs**3

    ax = axes[0]
    ax.semilogy(lam_um, res_life_th['T_total'] * 100, '-', color='C0',
                lw=2.0, label='LIFE full (top-hat, multi-band BS)')
    ax.semilogy(lam_um, res_life_g['T_total'] * 100, '--', color='C1',
                lw=1.5, label='LIFE full (Gaussian, multi-band BS)')
    ax.semilogy(lam_um, res_sb_th['T_total'] * 100, ':', color='C2',
                lw=1.5, label='LIFE single Bracewell (top-hat)')
    ax.semilogy(lam_um, T_optics_only * 100, '-.', color='C4', lw=1.2,
                label='LIFE optics only (no detector)')
    ax.semilogy(lam_um, np.maximum(T_caf2_only * 100, 0.001), '-',
                color='gray', lw=1.2, alpha=0.6,
                label=r'LIFE (CaF$_2$ only -- no ZnSe)')

    # NICE measured point
    ax.plot(4.0, 22.0, 's', color='red', ms=10, zorder=5,
            label='NICE measured: 22% (single output)')

    ax.fill_between([3.5, 20], 3.5, 10.0, alpha=0.06, color='green')
    ax.text(11.0, 5.0, 'LIFE PCE range\n(3.5--10%)', fontsize=9,
            color='green', ha='center', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.7, edgecolor='green', linewidth=0.5))
    ax.axvspan(*LIFE_BAND, alpha=0.06, color='blue')
    ax.set_ylabel('Photon Conversion Efficiency (%)')
    ax.set_ylim(0.5, 100)
    ax.set_title('End-to-End Throughput: LIFE Combiner')
    ax.legend(loc='upper right', fontsize=8)

    # Bottom panel: top-hat / Gaussian ratio
    ax = axes[1]
    ratio = res_life_th['T_total'] / np.maximum(res_life_g['T_total'], 1e-10)
    ax.plot(lam_um, ratio, '-', color='purple', lw=2)
    ax.axhline(0.815, color='gray', ls='--', alpha=0.5)
    ax.text(18, 0.82, '81.5% (Module 1)', fontsize=8, color='gray')
    ax.set_ylabel('Top-hat / Gaussian\nthroughput ratio')
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylim(0.75, 0.90)
    ax.axvspan(*LIFE_BAND, alpha=0.06, color='blue')

    fig.tight_layout()
    fig.savefig('fig7_throughput_vs_wavelength.png')
    plt.close()
    print("[OK] Fig 7 saved: throughput vs wavelength")


def make_fig8_sensitivity():
    """Fig 8: Throughput sensitivity to design choices.

    Panel A: Delay line bounces (2/4/6/8)
    Panel B: Mirror quality (ideal/flight/aged)
    Panel C: Total mirror count
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # --- Panel A: Delay line bounces ---
    ax = axes[0]
    for n_dl in [2, 4, 6, 8]:
        res = compute_throughput_LIFE(lam_um, n_delay_bounces=n_dl,
                                      verbose=False)
        ax.plot(lam_um, res['T_total'] * 100, label=f'{n_dl} bounces')
    ax.axvspan(*LIFE_BAND, alpha=0.06, color='blue')
    ax.axhline(5.0, color='green', ls='--', alpha=0.4)
    ax.set_ylabel('PCE (%)')
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_title('(a) Delay Line Bounces')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 12)

    # --- Panel B: Mirror quality ---
    ax = axes[1]
    for qual, ls in [('ideal', '-'), ('flight', '--'), ('aged', ':')]:
        res = compute_throughput_LIFE(lam_um, mirror_quality=qual,
                                      verbose=False)
        ax.plot(lam_um, res['T_total'] * 100, ls=ls, label=f'{qual}')
    ax.axvspan(*LIFE_BAND, alpha=0.06, color='blue')
    ax.axhline(5.0, color='green', ls='--', alpha=0.4)
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_title('(b) Mirror Coating Quality')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 12)

    # --- Panel C: Total mirror count ---
    ax = axes[2]
    R_flight = gold_reflectivity(lam_um, 'flight')
    idx10 = np.argmin(np.abs(lam_um - 10.0))

    n_mirrors_range = np.arange(10, 35)
    T_at_10um = []
    for n_m in n_mirrors_range:
        T_mirrors = R_flight[idx10]**n_m
        # Fixed losses: splitting(x2), BS material, fiber, dichroics, mask, det
        T_fixed = (0.50 * 0.50 * 0.97 * 0.97 * 0.815
                   * 0.95**2 * 0.99 * 0.995 * 0.98 * 0.60)
        T_at_10um.append(T_mirrors * T_fixed * 100)

    ax.plot(n_mirrors_range, T_at_10um, 'o-', color='C0', ms=4)

    # Baseline mirror count
    n_baseline = (4 + 6 + 3 + 1 + 2 + 1 + 3)
    # recv+corr+APS+MMZfold+roof+OAP+spectro
    ax.axvline(n_baseline, color='red', ls='--', alpha=0.5)
    ax.text(n_baseline + 0.5, 8, f'Baseline\n({n_baseline} mirrors)',
            fontsize=8, color='red')
    ax.axhline(5.0, color='green', ls='--', alpha=0.4)
    ax.set_xlabel('Total gold mirror reflections')
    ax.set_ylabel(r'PCE at 10 $\mu$m (%)')
    ax.set_title('(c) Mirror Count Sensitivity')
    ax.set_ylim(0, 15)

    fig.tight_layout()
    fig.savefig('fig8_throughput_sensitivity.png')
    plt.close()
    print("[OK] Fig 8 saved: throughput sensitivity")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def generate_summary_table():
    """Comprehensive throughput summary at key wavelengths."""
    wavelengths = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

    print("\n" + "=" * 80)
    print("THROUGHPUT SUMMARY TABLE -- LIFE End-to-End")
    print("=" * 80)
    print(f"{'Quantity':<40} ", end='')
    for w in wavelengths:
        print(f"{w:>7.0f}um", end='')
    print()
    print("-" * 80)

    R = gold_reflectivity(lam_um, 'flight')

    rows = [
        ('Gold reflectivity (per surface)', R),
        ('CaF2+AR (single pass, AR=0.90)',
         caf2_transmission(lam_um, 2.0, 0.90)),
        ('ZnSe+AR (single pass, AR=0.90, Tatian)',
         znse_transmission(lam_um, 2.0, 0.90)),
        ('Fiber coupling (top-hat)', fiber_coupling_tophat(lam_um)),
        ('Detector QE (Si:As BIB)', detector_qe(lam_um, 'SiAs_BIB')),
    ]

    for name, arr in rows:
        print(f"  {name:<38} ", end='')
        for w in wavelengths:
            idx = np.argmin(np.abs(lam_um - w))
            print(f"  {arr[idx]*100:5.1f}%", end='')
        print()

    print("-" * 80)

    configs = [
        ('LIFE full (top-hat, DB)', 'tophat', True),
        ('LIFE full (Gaussian, DB)', 'gaussian', True),
        ('LIFE single Bracewell (top-hat)', 'tophat', False),
    ]

    for label, bp, cc in configs:
        res = compute_throughput_LIFE(lam_um, beam_profile=bp,
                                      include_cross_combiner=cc,
                                      verbose=False)
        print(f"  {label:<38} ", end='')
        for w in wavelengths:
            idx = np.argmin(np.abs(lam_um - w))
            print(f"  {res['T_total'][idx]*100:5.2f}%", end='')
        print()

    print("=" * 80)

    # Key findings
    idx10 = np.argmin(np.abs(lam_um - 10.0))
    res_th = compute_throughput_LIFE(lam_um, 'tophat', verbose=False)
    res_g = compute_throughput_LIFE(lam_um, 'gaussian', verbose=False)

    print("\nKEY FINDINGS:")
    print(f"  LIFE PCE at 10 um (top-hat):  {res_th['T_total'][idx10]*100:.2f}%")
    print(f"  LIFE PCE at 10 um (Gaussian): {res_g['T_total'][idx10]*100:.2f}%")
    print(f"  Top-hat penalty factor:       "
          f"{res_th['T_total'][idx10]/res_g['T_total'][idx10]:.3f}")
    print(f"  LIFE requirement range:       3.5--10%")

    # Dominant losses
    print("\nDOMINANT LOSS SOURCES at 10 um:")
    losses = []
    for name, T_arr in res_th['components']:
        loss = (1.0 - T_arr[idx10]) * 100
        losses.append((loss, name))
    losses.sort(reverse=True)
    for loss, name in losses[:8]:
        print(f"  {name:<30} {loss:5.1f}% loss")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("LIFE End-to-End Throughput Chain -- Module 2 v3.0")
    print("=" * 55)

    # 1. Full LIFE throughput (top-hat)
    print("\n--- LIFE (top-hat beam) ---")
    res_life = compute_throughput_LIFE(lam_um, beam_profile='tophat',
                                       mirror_quality='flight',
                                       include_cross_combiner=True,
                                       n_delay_bounces=4)

    # 2. NICE validation (section-by-section)
    print("\n--- NICE Validation ---")
    res_nice, nice_meas = validate_NICE(lam_um)

    # 3. Summary table
    generate_summary_table()

    # 4. Figures
    print("\nGenerating publication figures...")
    make_fig5_material_properties()
    make_fig6_waterfall()
    make_fig7_throughput_vs_wavelength()
    make_fig8_sensitivity()

    print("\n[OK] Module 2 v3.0 complete. All figures saved.")
