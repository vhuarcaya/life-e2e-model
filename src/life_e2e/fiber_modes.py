"""
LIFE End-to-End Instrument Model — Fiber Modes Library
========================================================

Canonical source of single-mode fiber mode calculations for every
module in the LIFE end-to-end study.  All analysis modules and the
Monte Carlo engine import from here.

Author
------
Victor Huarcaya

Wavelength convention
---------------------
**All public functions accept wavelengths in micrometres.**
Modules that work in SI metres (e.g. Module 1, Module 3) must convert
at the boundary: ``lam_um = lam_m * 1e6``.

Why the fiber matters
---------------------
The single-mode fiber is the *critical* element in the LIFE architecture:

1. **Spatial filter** -- converts arbitrary wavefronts into a clean
   Gaussian output regardless of input aberrations.
2. **Coupling efficiency** -- defines the fraction of collected
   starlight / planet light that enters the fundamental mode.
3. **Null boundary** -- creates a sharp boundary between null-critical
   and null-irrelevant optics (Module 4 finding).
4. **Error scaling** -- converts WFE from quadratic (sigma^2/lam^2,
   infeasible) to quartic (sigma^4/lam^4, achievable) null contribution.

Sections
--------
=====  ============================================================
S      Contents
=====  ============================================================
1      V-parameter, single-mode cutoff, number of modes
2      Mode field radius: Marcuse (default) + power-law (legacy)
3      Mode field profiles: Gaussian, exact LP01, top-hat, Airy
4      Overlap integral (generic)
5      Coupling: top-hat analytical (Ruilier), Gaussian-to-Gaussian
6      Spatial filter rejection + transfer function
7      Optimal focal length (Gaussian + top-hat)
8      FIBER_PARAMS dict and convenience functions
9      Figure generation (``make_fiber_figures``, behind __main__)
=====  ============================================================

Key references
--------------
- Ruilier & Cassaing (2001) JOSA A -- fiber coupling of Airy pattern
- Marcuse (1978) JOSA 68, 103 -- Gaussian mode field approximation
- Shaklan & Roddier (1988) -- fiber coupling fundamentals
- Mennesson, Ollivier & Ruilier (2002) -- fiber-linked interferometry
- Birbacher et al. (2026) arXiv:2602.02279 -- NICE testbed results

Version history
---------------
3.0  2026-01-14  Clean rewrite for codebase reorganisation (Phase A).
                 Consolidates functions from fiber_modes_v2, Module 1
                 (e2e_fiber_coupling_v1_1), and Module 3 (null error
                 propagation).  Unified interface, um throughout.
2.0  2025-08-11  Fixed coupling_tophat_analytical (2x), FIBER_PARAMS
                 (Delta-n), coupling_gaussian tilt exponent.
1.0  2025-02-10  Initial version.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j0, j1

# -- Beam constants (LIFE combiner default) --------------------------
D_BEAM: float = 20.0e-3  # [m] collimated beam diameter in combiner

# ======================================================================
# FIBER_PARAMS  (v2-corrected Delta-n values)
# ======================================================================

FIBER_PARAMS: dict = {
    'InF3': {
        'n_core': 1.50,
        'n_clad': 1.48,
        'a_core_um': 4.5,
        'range_um': (3.0, 5.5),
        'notes': 'NICE testbed baseline, Delta-n=0.02, SM above 2.9 um',
    },
    'chalcogenide': {
        'n_core': 2.80,
        'n_clad': 2.77,
        'a_core_um': 5.0,
        'range_um': (5.5, 12.0),
        'notes': 'As2Se3 SM fiber, Delta-n=0.03, NA=0.41, SM above 5.3 um',
    },
    'silver_halide': {
        'n_core': 2.15,
        'n_clad': 2.135,
        'a_core_um': 17.0,
        'range_um': (12.0, 20.0),
        'notes': 'AgClBr SM fiber, Delta-n=0.015, NA=0.25, SM above 11.3 um',
    },
}
"""Fiber parameters for the three LIFE wavelength sub-bands.

v2 corrections: chalcogenide Delta-n reduced from 0.20 to 0.03;
silver_halide Delta-n reduced from 0.05 to 0.015.  Old values gave
V = 4-10 (severely multimode).
"""


# ======================================================================
# S1  V-PARAMETER AND SINGLE-MODE CONDITION
# ======================================================================

def v_parameter(lam_um: ArrayLike,
                n_core: float,
                n_clad: float,
                a_core_um: float) -> NDArray:
    """Fiber V-parameter (normalised frequency).

    V = (2 pi a / lam) * NA = (2 pi a / lam) * sqrt(n_core^2 - n_clad^2)

    The fiber is single-mode when V < 2.405 (first zero of J0).

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    n_core : float
        Core refractive index.
    n_clad : float
        Cladding refractive index.
    a_core_um : float
        Core radius [um].

    Returns
    -------
    V : ndarray
        V-parameter (dimensionless).
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    NA = np.sqrt(n_core**2 - n_clad**2)
    return 2.0 * np.pi * a_core_um / lam_um * NA


def single_mode_cutoff(n_core: float,
                       n_clad: float,
                       a_core_um: float) -> float:
    """Single-mode cutoff wavelength lam_c [um].

    Below this wavelength the fiber supports higher-order modes.

    lam_c = (2 pi a * NA) / 2.405
    """
    NA = np.sqrt(n_core**2 - n_clad**2)
    return 2.0 * np.pi * a_core_um * NA / 2.405


def number_of_modes(lam_um: ArrayLike,
                    n_core: float,
                    n_clad: float,
                    a_core_um: float) -> NDArray:
    """Approximate number of guided modes.

    For step-index fiber: N_modes ~ V^2 / 2  (for V >> 1).
    For V < 2.405: exactly 1 mode (LP01).
    """
    V = v_parameter(lam_um, n_core, n_clad, a_core_um)
    return np.where(V < 2.405, 1, np.round(V**2 / 2).astype(int))


# ======================================================================
# S2  MODE FIELD RADIUS
# ======================================================================

def mode_field_radius(lam_um: ArrayLike,
                      n_core: float,
                      n_clad: float,
                      a_core_um: float) -> NDArray:
    """Mode field radius w_f of the LP01 mode (Marcuse approximation).

    w_f / a = 0.65 + 1.619 V^(-3/2) + 2.879 V^(-6)

    Valid for 0.8 < V < 2.5 (single-mode regime).  This is the
    **default** MFR function for the entire LIFE study.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    n_core, n_clad : float
        Core / cladding refractive indices.
    a_core_um : float
        Core radius [um].

    Returns
    -------
    w_f_um : ndarray
        Mode field radius [um].

    Notes
    -----
    The MFR is always larger than the core radius and grows with
    wavelength (as V decreases toward cutoff).  For LIFE at 10 um
    with chalcogenide fiber: V ~ 1.8, w_f ~ 8.5 um ~ 1.4 * a_core.
    """
    V = v_parameter(lam_um, n_core, n_clad, a_core_um)
    V = np.maximum(V, 0.5)  # avoid division by zero
    ratio = 0.65 + 1.619 * V**(-1.5) + 2.879 * V**(-6)
    return a_core_um * ratio


def mode_field_radius_wavelength(lam_um: ArrayLike,
                                 fiber_type: str = 'chalcogenide') -> NDArray:
    """Convenience: MFR as a function of wavelength for a named fiber type.

    Uses parameters from ``FIBER_PARAMS``.
    """
    p = FIBER_PARAMS[fiber_type]
    return mode_field_radius(lam_um, p['n_core'], p['n_clad'],
                             p['a_core_um'])


def mode_field_radius_linear(lam_um: ArrayLike,
                             w_f_ref_um: float = 6.0,
                             lam_ref_um: float = 4.0) -> NDArray:
    """Simplified linear scaling of MFR with wavelength (legacy).

    w_f(lam) ~ w_f_ref * (lam / lam_ref)

    This approximation is used in Modules 1-4 and is valid when the
    V-parameter changes slowly with wavelength (true in the SM window).
    Prefer ``mode_field_radius()`` for quantitative work.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    w_f_ref_um : float
        Reference MFR [um] at lam_ref_um.
    lam_ref_um : float
        Reference wavelength [um].

    Returns
    -------
    w_f_um : ndarray
        Mode field radius [um].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    return w_f_ref_um * lam_um / lam_ref_um


# ======================================================================
# S3  MODE FIELD PROFILES
# ======================================================================

def gaussian_mode(r: ArrayLike, w_f: float) -> NDArray:
    """Gaussian approximation of the LP01 mode field amplitude.

    E(r) = sqrt(2/pi) / w_f * exp(-r^2 / w_f^2)

    Normalised so that integral |E|^2 2 pi r dr = 1.

    Parameters
    ----------
    r : array
        Radial coordinate [same units as w_f].
    w_f : float
        Mode field radius.

    Returns
    -------
    E : ndarray
        Field amplitude.
    """
    r = np.asarray(r, dtype=float)
    return np.sqrt(2.0 / np.pi) / w_f * np.exp(-r**2 / w_f**2)


def exact_lp01_mode(r: ArrayLike,
                    a_core: float,
                    V: float,
                    n_core: float,
                    n_clad: float) -> NDArray:
    """Exact LP01 mode field (Bessel / modified Bessel).

    Inside core (r < a):  E ~ J0(U r/a)
    In cladding (r > a):  E ~ K0(W r/a)

    Uses approximate eigenvalue for visualisation purposes.

    Parameters
    ----------
    r : array
        Radial coordinate [same units as a_core].
    a_core : float
        Core radius.
    V : float
        V-parameter.
    n_core, n_clad : float
        Refractive indices.

    Returns
    -------
    E : ndarray
        Normalised field amplitude.
    """
    from scipy.special import k0 as K0

    r = np.asarray(r, dtype=float)

    # Approximate U, W from V
    U = 2.405 * (1.0 - np.exp(-0.5 * V))
    W = np.sqrt(max(V**2 - U**2, 0.01))

    E = np.where(
        r <= a_core,
        j0(U * r / a_core) / j0(0),
        j0(U) / K0(W) * K0(W * r / a_core)
    )

    # Normalise
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    norm = np.sqrt(2.0 * np.pi * np.sum(E**2 * r * dr))
    if norm > 0:
        E = E / norm

    return E


def tophat_field(r: ArrayLike, R_beam: float) -> NDArray:
    """Top-hat (uniform circular) beam field amplitude.

    E(r) = 1/sqrt(pi R^2) for r < R, 0 otherwise.
    Normalised so integral |E|^2 2 pi r dr = 1.
    """
    r = np.asarray(r, dtype=float)
    return np.where(r <= R_beam,
                    1.0 / np.sqrt(np.pi * R_beam**2), 0.0)


def airy_field(r: ArrayLike,
               wavelength: float,
               f_lens: float,
               D_beam: float) -> NDArray:
    """Airy pattern field amplitude at the fiber plane.

    E(r) ~ 2 J1(pi D r / (lam f)) / (pi D r / (lam f))

    This is what a top-hat beam becomes after focusing.

    Parameters
    ----------
    r : array
        Radial position [m].
    wavelength : float
        Wavelength [m].
    f_lens : float
        Coupling lens focal length [m].
    D_beam : float
        Beam diameter [m].

    Returns
    -------
    E : ndarray
        Normalised field amplitude.
    """
    r = np.asarray(r, dtype=float)
    x = np.pi * D_beam * r / (wavelength * f_lens)
    x = np.where(x == 0, 1e-30, x)
    E = 2.0 * j1(x) / x

    dr = r[1] - r[0] if len(r) > 1 else 1.0
    norm = np.sqrt(2.0 * np.pi * np.sum(E**2 * r * dr))
    if norm > 0:
        E = E / norm

    return E


# ======================================================================
# S4  OVERLAP INTEGRAL (GENERIC)
# ======================================================================

def overlap_integral(E_in: ArrayLike,
                     E_fiber: ArrayLike,
                     r: ArrayLike,
                     dr: float | None = None) -> float:
    """Power coupling efficiency via overlap integral.

    eta = |integral E_in * E_fiber^* * 2 pi r dr|^2
          / (integral |E_in|^2 2 pi r dr * integral |E_fiber|^2 2 pi r dr)

    If both fields are already power-normalised the denominator is 1.

    Parameters
    ----------
    E_in : array
        Input field amplitude (real or complex).
    E_fiber : array
        Fiber mode field amplitude (real).
    r : array
        Radial coordinate grid.
    dr : float or None
        Radial step; inferred from r if None.

    Returns
    -------
    eta : float
        Coupling efficiency [0, 1].
    """
    r = np.asarray(r, dtype=float)
    E_in = np.asarray(E_in)
    E_fiber = np.asarray(E_fiber)
    if dr is None:
        dr = r[1] - r[0]

    overlap = 2.0 * np.pi * np.sum(E_in * np.conj(E_fiber) * r * dr)
    P_in = 2.0 * np.pi * np.sum(np.abs(E_in)**2 * r * dr)
    P_fib = 2.0 * np.pi * np.sum(np.abs(E_fiber)**2 * r * dr)

    if P_in * P_fib == 0:
        return 0.0

    return float(np.abs(overlap)**2 / (P_in * P_fib))


# ======================================================================
# S5  COUPLING EFFICIENCY
# ======================================================================

def coupling_tophat_analytical(beta: ArrayLike) -> NDArray:
    """Analytical coupling of a top-hat beam into a Gaussian fiber mode.

    eta(beta) = 2 (1 - exp(-beta^2))^2 / beta^2

    Derived from pupil-plane overlap (Ruilier & Cassaing 2001):

    * Back-propagated Gaussian waist: w_back = lam f / (pi w_f)
    * Define beta = (D/2) / w_back = pi w_f D / (2 lam f)
    * Maximum: eta = 81.45 % at beta_opt = 1.1209.

    Parameters
    ----------
    beta : float or array
        Dimensionless coupling parameter.

    Returns
    -------
    eta : ndarray
        Coupling efficiency [0, 1].

    Notes
    -----
    v2 fix: v1 had [2(1-e^{-beta^2})]^2 / beta^2 = 4(...)^2/beta^2,
    giving eta_max = 163 %.  The correct formula has the factor 2 *outside*
    the square: 2(...)^2/beta^2.
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    return 2.0 * (1.0 - np.exp(-beta**2))**2 / beta**2


def coupling_gaussian_to_gaussian(w_in: ArrayLike,
                                  w_fiber: ArrayLike,
                                  offset: ArrayLike = 0.0,
                                  tilt_rad: ArrayLike = 0.0,
                                  wavelength: float | None = None) -> NDArray:
    """Gaussian-to-Gaussian coupling with misalignment.

    eta = (2 w_in w_f / (w_in^2 + w_f^2))^2
          * exp(-2 dx^2 / (w_in^2 + w_f^2))
          * exp(-pi^2 w_eff^2 theta^2 / lam^2)

    where w_eff^2 = 2 w_in^2 w_f^2 / (w_in^2 + w_f^2).

    For matched beams (w_in = w_f, offset = 0, tilt = 0): eta = 100 %.

    Parameters
    ----------
    w_in : float or array
        Input beam waist (unit-agnostic; must match w_fiber).
    w_fiber : float or array
        Fiber mode field radius.
    offset : float or array
        Lateral misalignment (same units as w_in).
    tilt_rad : float or array
        Angular misalignment [rad].
    wavelength : float or None
        Wavelength (same units as w_in); needed if tilt_rad > 0.

    Returns
    -------
    eta : ndarray
        Coupling efficiency [0, 1].

    Notes
    -----
    v2 fix: tilt exponent corrected.  v1 had exp(-pi^2 w^2 theta^2 / (4 lam^2))
    for matched beams; correct is exp(-pi^2 w^2 theta^2 / (2 lam^2)) at
    amplitude level, squared to exp(-pi^2 w^2 theta^2 / lam^2) at power level.
    """
    w_in = np.asarray(w_in, dtype=float)
    w_fiber = np.asarray(w_fiber, dtype=float)
    offset = np.asarray(offset, dtype=float)

    # Mode mismatch
    w2_sum = w_in**2 + w_fiber**2
    eta_mode = (2.0 * w_in * w_fiber / w2_sum)**2

    # Lateral offset
    eta_offset = np.exp(-2.0 * offset**2 / w2_sum)

    # Tilt
    if np.any(np.asarray(tilt_rad) > 0) and wavelength is not None:
        tilt_rad = np.asarray(tilt_rad, dtype=float)
        w2_eff = 2.0 * w_in**2 * w_fiber**2 / w2_sum
        eta_tilt = np.exp(-np.pi**2 * w2_eff * tilt_rad**2
                          / wavelength**2)
    else:
        eta_tilt = 1.0

    return eta_mode * eta_offset * eta_tilt


# ======================================================================
# S6  SPATIAL FILTERING
# ======================================================================

def spatial_filter_rejection(n_zernike: int,
                             m_zernike: int,
                             amplitude_waves: float,
                             V: float = 1.8) -> float:
    """Spatial filter rejection of a specific Zernike aberration.

    The SMF transmits only the LP01 mode.  Any wavefront error that
    does not project onto the fundamental mode is rejected.

    Parameters
    ----------
    n_zernike : int
        Radial order.
    m_zernike : int
        Azimuthal order.
    amplitude_waves : float
        Aberration amplitude [waves RMS].
    V : float
        Fiber V-parameter.

    Returns
    -------
    coupling_loss_fraction : float
        Fractional coupling reduction [0, 1].

    Notes
    -----
    Higher-order Zernike modes with |m| > 0 couple less efficiently
    to the azimuthally symmetric LP01 mode.  Modes with m != 0 are
    partially rejected by the fiber's circular symmetry.
    """
    sigma_rad = 2.0 * np.pi * amplitude_waves
    coupling_loss = 1.0 - np.exp(-sigma_rad**2)

    if m_zernike == 0:
        rejection_factor = 1.0
    elif abs(m_zernike) == 1:
        rejection_factor = 0.8
    else:
        rejection_factor = max(0.3, 1.0 - 0.15 * abs(m_zernike))

    return coupling_loss * rejection_factor


def spatial_filter_transfer_function(spatial_freq: ArrayLike,
                                     w_f: float,
                                     wavelength: float = 0.0) -> NDArray:
    """Spatial filtering transfer function of a single-mode fiber.

    The Gaussian LP01 mode acts as a low-pass filter in the spatial
    frequency domain with cutoff f_c = 1 / (pi w_f).

    Parameters
    ----------
    spatial_freq : array
        Spatial frequency [cycles / (same unit as w_f)].
    w_f : float
        Mode field radius.
    wavelength : float
        Included for interface generality; not used for Gaussian mode.

    Returns
    -------
    H : ndarray
        Transfer function magnitude [0, 1].
    """
    spatial_freq = np.asarray(spatial_freq, dtype=float)
    f_c = 1.0 / (np.pi * w_f)
    return np.exp(-(spatial_freq / f_c)**2)


# ======================================================================
# S7  OPTIMAL COUPLING CONFIGURATION
# ======================================================================

def optimal_focal_length(D_beam: float,
                         w_f: float,
                         wavelength: float,
                         beam_type: str = 'tophat') -> float:
    """Compute the optimal coupling lens focal length.

    For top-hat beam: f_opt = pi D w_f / (2 beta_opt lam)
    For Gaussian:     f_opt = pi w0 w_f / lam  (mode-matching condition)

    Parameters
    ----------
    D_beam : float
        Beam diameter [m]  (or half-beam for Gaussian: w0 = D/2).
    w_f : float
        Fiber MFR [m].
    wavelength : float
        Wavelength [m].
    beam_type : {'tophat', 'gaussian'}
        Input beam profile.

    Returns
    -------
    f_opt : float
        Optimal focal length [m].
    """
    if beam_type == 'tophat':
        beta_opt = 1.1209
        return np.pi * D_beam * w_f / (2.0 * beta_opt * wavelength)
    else:
        w0 = D_beam / 2.0
        return np.pi * w0 * w_f / wavelength


def optimal_coupling_summary(wavelength_um: float = 10.0,
                             fiber_type: str = 'chalcogenide') -> None:
    """Print a summary of optimal coupling parameters."""
    p = FIBER_PARAMS[fiber_type]
    lam_arr = np.array([wavelength_um])
    w_f = float(mode_field_radius(lam_arr, p['n_core'], p['n_clad'],
                                  p['a_core_um'])[0])
    V_val = float(v_parameter(lam_arr, p['n_core'], p['n_clad'],
                              p['a_core_um'])[0])
    lam_c = single_mode_cutoff(p['n_core'], p['n_clad'], p['a_core_um'])

    w_f_m = w_f * 1e-6
    lam_m = wavelength_um * 1e-6
    f_opt = optimal_focal_length(D_BEAM, w_f_m, lam_m, 'tophat')

    print(f"\n--- Optimal coupling at lam = {wavelength_um} um"
          f" ({fiber_type}) ---")
    print(f"  V-parameter:        {V_val:.3f} (single-mode: V < 2.405)")
    print(f"  SM cutoff:          lam_c = {lam_c:.1f} um")
    print(f"  Mode field radius:  w_f = {w_f:.2f} um")
    print(f"  Optimal f_lens:     f = {f_opt*1e3:.1f} mm (top-hat)")
    print(f"  Max coupling (TH):  eta = 81.45%")
    print(f"  f/#:                {f_opt/D_BEAM:.1f}")


# ======================================================================
# S9  FIGURE GENERATION
# ======================================================================

def make_fiber_figures() -> None:
    """Generate publication-quality fiber mode analysis figures.

    Produces three figure files:

    * **fig20** -- V-parameter and MFR vs wavelength for all fiber types.
    * **fig21** -- Mode field profiles and coupling vs wavelength.
    * **fig22** -- eta(beta) curve + Gaussian coupling sensitivity.

    Also prints fiber parameter summary tables.
    """
    import matplotlib.pyplot as plt

    print("\n--- Generating fiber mode figures ---")

    colors = {'InF3': 'blue', 'chalcogenide': 'red', 'silver_halide': 'green'}

    # -- Fig 20: V-parameter and MFR ---------------------------------
    fig20, (ax20a, ax20b) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ftype, params in FIBER_PARAMS.items():
        lam_range = np.linspace(params['range_um'][0],
                                params['range_um'][1], 200)
        V_arr = v_parameter(lam_range, params['n_core'],
                            params['n_clad'], params['a_core_um'])
        w_f_arr = mode_field_radius(lam_range, params['n_core'],
                                    params['n_clad'], params['a_core_um'])
        color = colors[ftype]
        ax20a.plot(lam_range, V_arr, color=color, lw=2.5, label=ftype)
        ax20b.plot(lam_range, w_f_arr, color=color, lw=2.5, label=ftype)

    ax20a.axhline(y=2.405, color='black', ls='--', lw=1.5, alpha=0.7)
    ax20a.text(3.5, 2.55, 'V = 2.405 (SM cutoff)', fontsize=10)
    ax20a.axvspan(6, 16, alpha=0.08, color='green')
    ax20a.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax20a.set_ylabel('V-parameter', fontsize=12)
    ax20a.set_title('Fiber V-parameter', fontsize=13)
    ax20a.legend(fontsize=10)
    ax20a.set_xlim(2, 22)
    ax20a.set_ylim(0, 4)
    ax20a.grid(True, alpha=0.3)

    lam_lin = np.linspace(4, 20, 200)
    w_lin = mode_field_radius_linear(lam_lin, 6.0, 4.0)
    ax20b.plot(lam_lin, w_lin, 'k:', lw=1.5, alpha=0.5,
               label='Linear approx')
    ax20b.axvspan(6, 16, alpha=0.08, color='green')
    ax20b.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax20b.set_ylabel(r'Mode field radius [$\mu$m]', fontsize=12)
    ax20b.set_title('Mode Field Radius (Marcuse)', fontsize=13)
    ax20b.legend(fontsize=10)
    ax20b.set_xlim(2, 22)
    ax20b.grid(True, alpha=0.3)

    fig20.tight_layout()
    fig20.savefig('fig20_V_parameter_MFR.png', dpi=200,
                  bbox_inches='tight')
    print("  Saved: fig20_V_parameter_MFR.png")

    # -- Fig 21: Mode field profiles and coupling --------------------
    fig21, (ax21a, ax21b) = plt.subplots(1, 2, figsize=(13, 5.5))

    params_c = FIBER_PARAMS['chalcogenide']
    w_f = float(mode_field_radius(np.array([10.0]), params_c['n_core'],
                                  params_c['n_clad'],
                                  params_c['a_core_um'])[0])
    a_core = params_c['a_core_um']
    r = np.linspace(0, 4 * a_core, 500)  # um

    E_gauss = gaussian_mode(r, w_f)
    lam_m = 10.0e-6
    f_opt = optimal_focal_length(D_BEAM, w_f * 1e-6, lam_m, 'tophat')
    E_airy = airy_field(r * 1e-6, lam_m, f_opt, D_BEAM)

    ax21a.plot(r, E_gauss / np.max(E_gauss), 'b-', lw=2.5,
               label=f'LP01 Gaussian (w_f={w_f:.1f} um)')
    ax21a.plot(r, np.abs(E_airy) / np.max(np.abs(E_airy)), 'r--', lw=2,
               label='Airy (focused top-hat)')
    ax21a.axvline(x=a_core, color='gray', ls=':', alpha=0.5)
    ax21a.text(a_core + 0.3, 0.95, f'a={a_core} um', fontsize=9,
               color='gray')
    ax21a.set_xlabel(r'Radial position [$\mu$m]', fontsize=12)
    ax21a.set_ylabel('Normalised field amplitude', fontsize=12)
    ax21a.set_title(r'Mode Profiles at $\lambda$=10 $\mu$m (chalcogenide)',
                    fontsize=13)
    ax21a.legend(fontsize=10)
    ax21a.set_xlim(0, 4 * a_core)
    ax21a.grid(True, alpha=0.3)

    # Coupling vs wavelength
    lam_plot = np.linspace(4, 20, 200)
    for ftype, params in FIBER_PARAMS.items():
        lam_valid = lam_plot[(lam_plot >= params['range_um'][0]) &
                             (lam_plot <= params['range_um'][1])]
        # At optimal f per wavelength: coupling = 81.45 %
        ax21b.plot(lam_valid, [0.8145] * len(lam_valid),
                   color=colors[ftype], lw=2.5, label=ftype)

    # Fixed-f coupling degradation
    lam_sweep = np.linspace(4, 20, 200)
    wf_10 = mode_field_radius(10.0, params_c['n_core'],
                              params_c['n_clad'],
                              params_c['a_core_um']) * 1e-6
    f_fixed = optimal_focal_length(D_BEAM, wf_10, 10e-6, 'tophat')

    eta_fixed = []
    for lv in lam_sweep:
        wf = mode_field_radius(lv, params_c['n_core'],
                               params_c['n_clad'],
                               params_c['a_core_um']) * 1e-6
        beta = np.pi * D_BEAM * wf / (2.0 * f_fixed * lv * 1e-6)
        if beta > 0:
            eta = float(coupling_tophat_analytical(np.atleast_1d(beta))[0])
        else:
            eta = 0.0
        eta_fixed.append(eta)

    ax21b.plot(lam_sweep, eta_fixed, 'k--', lw=1.5, alpha=0.7,
               label='Fixed f (opt @ 10 um)')
    ax21b.axvspan(6, 16, alpha=0.08, color='green')
    ax21b.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax21b.set_ylabel('Coupling efficiency', fontsize=12)
    ax21b.set_title('Fiber Coupling Efficiency (top-hat input)', fontsize=13)
    ax21b.legend(fontsize=9, loc='lower left')
    ax21b.set_xlim(4, 20)
    ax21b.set_ylim(0, 1.0)
    ax21b.grid(True, alpha=0.3)

    fig21.tight_layout()
    fig21.savefig('fig21_mode_profiles.png', dpi=200,
                  bbox_inches='tight')
    print("  Saved: fig21_mode_profiles.png")

    # -- Fig 22: eta(beta) + misalignment ----------------------------
    fig22, (ax22a, ax22b) = plt.subplots(1, 2, figsize=(13, 5.5))

    beta = np.linspace(0.1, 4.0, 500)
    eta = coupling_tophat_analytical(beta)
    ax22a.plot(beta, eta * 100, 'b-', lw=2.5)
    ax22a.axvline(x=1.1209, color='red', ls='--', lw=1.5)
    ax22a.axhline(y=81.45, color='red', ls=':', lw=1)
    ax22a.text(1.25, 82.5, r'$\beta_\mathrm{opt}$ = 1.121',
               fontsize=10, color='red')
    ax22a.text(0.3, 82.5, r'$\eta_\mathrm{max}$ = 81.45%',
               fontsize=10, color='red')
    ax22a.set_xlabel(r'$\beta = \pi D w_f / (2\lambda f)$', fontsize=12)
    ax22a.set_ylabel('Coupling efficiency [%]', fontsize=12)
    ax22a.set_title('Top-hat -> SMF Coupling (Ruilier formula)',
                    fontsize=13)
    ax22a.set_xlim(0, 4)
    ax22a.set_ylim(0, 100)
    ax22a.grid(True, alpha=0.3)

    # Misalignment sensitivity
    offsets = np.linspace(0, 5, 200)
    w = 1.0  # normalised
    eta_lat = coupling_gaussian_to_gaussian(w, w, offset=offsets * w)

    w_in_range = np.linspace(0.5, 2.0, 200)
    eta_mismatch = coupling_gaussian_to_gaussian(
        w_in_range, np.ones_like(w_in_range))

    ax22b.plot(offsets, eta_lat * 100, 'b-', lw=2.5,
               label='Lateral offset [w_f units]')
    ax22b.plot(w_in_range, eta_mismatch * 100, 'r--', lw=2,
               label='Mode mismatch (w_in/w_f)')
    ax22b.axhline(y=99, color='gray', ls=':', alpha=0.5)
    ax22b.text(3.5, 99.5, '99%', fontsize=9, color='gray')
    ax22b.set_xlabel('Misalignment parameter', fontsize=12)
    ax22b.set_ylabel('Coupling efficiency [%]', fontsize=12)
    ax22b.set_title('Gaussian Coupling Sensitivity', fontsize=13)
    ax22b.legend(fontsize=10)
    ax22b.set_xlim(0, max(offsets[-1], w_in_range[-1]))
    ax22b.set_ylim(0, 105)
    ax22b.grid(True, alpha=0.3)

    fig22.tight_layout()
    fig22.savefig('fig22_coupling_sensitivity.png', dpi=200,
                  bbox_inches='tight')
    print("  Saved: fig22_coupling_sensitivity.png")

    plt.close('all')

    # -- Summary tables ----------------------------------------------
    print("\n--- Fiber Parameters Summary ---")
    print(f"{'Fiber':>15s}  {'n_core':>6s}  {'n_clad':>6s}  "
          f"{'a [um]':>7s}  {'lam_c [um]':>10s}  "
          f"{'V@10um':>7s}  {'w_f@10um':>9s}")
    print("-" * 70)
    for ftype, p in FIBER_PARAMS.items():
        lam_c = single_mode_cutoff(p['n_core'], p['n_clad'],
                                   p['a_core_um'])
        V10 = float(v_parameter(np.array([10.0]), p['n_core'],
                                p['n_clad'], p['a_core_um'])[0])
        wf10 = float(mode_field_radius(np.array([10.0]), p['n_core'],
                                       p['n_clad'], p['a_core_um'])[0])
        print(f"{ftype:>15s}  {p['n_core']:>6.2f}  {p['n_clad']:>6.2f}  "
              f"{p['a_core_um']:>7.1f}  {lam_c:>10.1f}  "
              f"{V10:>7.2f}  {wf10:>9.2f}")

    for lam_val in [4.0, 6.0, 10.0, 16.0]:
        ftype = ('InF3' if lam_val < 5.5
                 else ('chalcogenide' if lam_val < 12
                       else 'silver_halide'))
        optimal_coupling_summary(lam_val, ftype)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("LIFE Fiber Modes Library -- Summary Figures")
    print("=" * 70)
    make_fiber_figures()
    print("\nAll figures saved.  Fiber modes library ready.")