"""
LIFE End-to-End Wavefront Propagation Study -- Module 3: Null Depth Error Propagation
======================================================================================

Author:  Victor Huarcaya (University of Bern)
Version: 3.0  (codebase reorganisation Phase A, Step 4)
Date:    2026-02-14

Purpose:
    Compute the mean null depth <N>(lam) across the LIFE science band by
    propagating all error terms through the Birbacher et al. (2026) null
    depth equation (arXiv:2602.02279, Eqs. 2-3 and Appendix A):

      <N> = 1/4(  <d_phi>^2 + sigma^2_dphi + 1/4 <d_phi_sp>^2
                + <dI>^2 + sigma^2_dI + 1/4 <dI_sp>^2  )

    Nine physical error sources are mapped onto the six algebraic slots:

      Phase (<d_phi>^2):  mean OPD offset + BS chromatic (summed COHERENTLY)
      Phase (sigma^2_dphi):  OPD RMS jitter
      Phase (d_phi_sp):  s-p polarisation phase split from imperfect APS
      Intensity (<dI>^2):  static mismatch + WFE-driven coupling + pointing
      Intensity (sigma^2_dI):  fluctuating intensity mismatch
      Intensity (dI_sp):  s-p polarisation amplitude split

UNIT CONVENTION
===============
**Module 3 works entirely in SI METRES for wavelength.**
The canonical material/fiber libraries (material_properties.py,
fiber_modes.py) accept wavelengths in *micrometres*.  Thin private
wrappers (_caf2_n, _znse_n, _fiber_w_f) perform the m -> um
conversion at the call site so that all Module 3 physics functions
remain in metres.

    v3.0 changes vs v2.1:
      - Removed local caf2_refractive_index(), znse_refractive_index(),
        fiber_mode_radius().  Now imported from canonical libraries via
        unit-converting wrappers.
      - Unicode replaced with LaTeX/ASCII in print/plot labels.

    v2.1 changes vs v2.0:
      - Multi-band BS chromatic: CaF2 for lam < 10 um, ZnSe for lam >= 10 um
        (was CaF2 only, extrapolating beyond 9.7 um validity limit)
      - Default bs_material changed from 'caf2' to 'multiband' throughout
      - WFE function: docstring corrected to describe actual asymmetric model;
        dead symmetric-attempt code removed
      - ZnSe reference corrected: Tatian (1984) not Connolly (1979)

    v2.0 changes vs v1.0:
      - Coherent (not quadrature) addition of mean OPD + BS chromatic phases
      - Symmetric WFE differential: both arms +/- sigma_diff/sqrt(2) instead
        of (sigma, 0)
      - Removed ad-hoc 5 %/octave chromatic factor on polarisation phase
      - Complete NICE validation with all four non-polarisation terms
      - CaF2 Sellmeier validity guard (lam < 9.7 um)
      - Consistent defaults (NICE-demonstrated values everywhere)
      - Explicit cross-term tracking in budget output

Physics references:
    [B26]  Birbacher et al. 2026, arXiv:2602.02279 -- Eqs. 2-4, Appendix A
    [B24]  Birbacher et al. 2024, SPIE 13095.130951H -- Tables 1-2
    [S00]  Serabyn 2000, SPIE 4006, 328 -- original null-depth decomposition
    [RC01] Ruilier & Cassaing 2001, JOSA A 18, 143 -- fibre coupling
    [L80]  Li 1980, JPCRD 9, 161 -- CaF2 Sellmeier coefficients
    [T84]  Tatian 1984, Appl. Opt. 23, 4477 -- ZnSe Sellmeier
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt

# =============================================================================
# Import canonical material / fiber functions (wavelength in MICROMETRES)
# =============================================================================
from material_properties import (
    caf2_sellmeier,
    znse_sellmeier,
)
from fiber_modes import (
    mode_field_radius_linear,
)


# ============================================================================
# Physical constants and LIFE mission parameters
# ============================================================================

# LIFE science band
LAMBDA_MIN: float = 6.0e-6         # [m]
LAMBDA_MAX: float = 16.0e-6        # [m]
LAMBDA_GOAL_MAX: float = 18.5e-6   # [m]  extended goal band edge
LAMBDA_REF: float = 10.0e-6        # [m]  compensator reference wavelength

# NICE testbed wavelengths
LAMBDA_NICE_A: float = 3.85e-6     # [m]  Laser A
LAMBDA_NICE_B: float = 4.5e-6      # [m]  Laser B
LAMBDA_NICE_NULL: float = 4.7e-6   # [m]  best null measurement wavelength

# Beam parameters
D_BEAM: float = 20.0e-3            # [m]  compressed beam diameter in combiner

# Fibre parameters (from Module 1)
MFD_REF: float = 12.0e-6           # [m]  mode-field diameter at lam_MFD_REF
W_F_REF: float = MFD_REF / 2       # [m]  mode-field radius
LAMBDA_MFD_REF: float = 4.0e-6     # [m]  reference wavelength for MFD


# ============================================================================
# UNIT-BOUNDARY WRAPPERS  (metres <-> micrometres)
# ============================================================================
# The canonical libraries accept wavelength in MICROMETRES.
# Module 3 works in METRES.  These private wrappers convert at the
# call site so that all downstream physics stays in SI.
# ============================================================================

def _caf2_n(wavelength_m: ArrayLike) -> NDArray:
    """CaF2 refractive index.  Wavelength in metres.

    Delegates to material_properties.caf2_sellmeier(lam_um).
    Preserves the Sellmeier validity guard: CaF2 coefficients are from
    Malitson (1963), valid 0.15-9.7 um.  Beyond 9.7 um the multiphonon
    absorption edge makes the Sellmeier unphysical; a warning is issued
    and values are extrapolated.
    """
    lam_um = np.atleast_1d(np.asarray(wavelength_m, dtype=float)) * 1e6

    if np.any(lam_um > 9.7):
        warnings.warn(
            "CaF2 Sellmeier evaluated beyond 9.7 um validity limit. "
            "Values are extrapolated and may be unphysical.",
            stacklevel=2,
        )

    return caf2_sellmeier(lam_um)


def _znse_n(wavelength_m: ArrayLike) -> NDArray:
    """ZnSe refractive index.  Wavelength in metres.

    Delegates to material_properties.znse_sellmeier(lam_um).
    Tatian (1984) three-term Sellmeier, valid 0.6-22 um.
    """
    lam_um = np.atleast_1d(np.asarray(wavelength_m, dtype=float)) * 1e6
    return znse_sellmeier(lam_um)


def _fiber_w_f(wavelength_m: ArrayLike,
               w_f_ref_m: float = W_F_REF,
               lam_ref_m: float = LAMBDA_MFD_REF) -> NDArray:
    """Mode-field radius of single-mode fibre (linear lam-scaling).

    Wavelength in metres.  Returns w_f in metres.

    Delegates to fiber_modes.mode_field_radius_linear(lam_um).
    Approximate: w_f proportional to lam for step-index fibres near cutoff.
    More accurate Marcuse (1978) formula lives in the fibre modes library.
    """
    lam_um = np.atleast_1d(np.asarray(wavelength_m, dtype=float)) * 1e6
    w_f_ref_um = w_f_ref_m * 1e6
    lam_ref_um = lam_ref_m * 1e6
    # Returns um, convert back to m
    return mode_field_radius_linear(lam_um, w_f_ref_um, lam_ref_um) * 1e-6


# ============================================================================
# 1. Null depth requirement curve  [B26 Table 2]
# ============================================================================

def null_requirement_curve(wavelengths: ArrayLike) -> NDArray:
    """Wavelength-dependent null depth requirement for LIFE / NICE.

    The requirement is driven by astrophysical background levels (zodiacal
    dust, stellar leakage).  Most stringent near 8 um; relaxes at both band
    edges.

    Anchor points:
        4 um -> 2.0e-5   [B26 Table 2]
        6 um -> 1.0e-5   [Paper Table 5]
        8 um -> 6.8e-6   [B26 Table 2, most stringent]
       10 um -> 3.0e-5   [Paper Table 5]
       12 um -> 1.8e-5   [B26 Table 2]
       16 um -> 6.0e-5   [Paper Table 5]
       18 um -> 1.0e-4   [B26 Table 2]

    Interpolated in log-space between anchors.
    """
    lam_um = np.atleast_1d(np.asarray(wavelengths, dtype=float)) * 1e6

    lam_anchors = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 18.0])
    N_anchors = np.array([2.0e-5, 1.0e-5, 6.8e-6, 3.0e-5,
                          1.8e-5, 6.0e-5, 1.0e-4])

    log_N = np.interp(lam_um, lam_anchors, np.log10(N_anchors))
    result = 10.0 ** log_N
    return float(result.flat[0]) if np.ndim(wavelengths) == 0 else result


def nice_error_budget_table2() -> dict:
    """Return the NICE equal-weight error budget from [B26] Table 2."""
    lam_um = np.array([4.0, 8.0, 12.0, 18.0])
    return {
        'lambda_um': lam_um,
        'N_required': np.array([2.0e-5, 6.8e-6, 1.8e-5, 1.0e-4]),
        # Static errors
        'dphi_mean_nm': np.array([2.3, 2.7, 6.7, 24.0]),
        'dI_mean_pct': np.array([0.37, 0.21, 0.35, 0.83]),
        'dI_sp_pct': np.array([0.74, 0.43, 0.70, 1.7]),
        'dphi_sp_nm': np.array([4.7, 5.4, 13.0, 48.0]),
        # Dynamic errors
        'sigma_dphi_nm': np.array([2.3, 2.7, 6.7, 24.0]),
        'sigma_dI_pct': np.array([0.37, 0.21, 0.35, 0.83]),
        'sigma_pointing_urad': np.array([16, 25, 47, 110]),
        'sigma_shear_um': np.array([204, 155, 199, 307]),
        'sigma_higher_pct': np.array([0.18, 0.11, 0.17, 0.42]),
    }


# ============================================================================
# 2. Null depth formulae  [B26 Eq. 2, Appendix A.1]
# ============================================================================

def null_depth_birbacher(
    dphi_mean: ArrayLike,
    sigma_dphi: ArrayLike,
    dphi_sp: ArrayLike,
    dI_mean: ArrayLike,
    sigma_dI: ArrayLike,
    dI_sp: ArrayLike,
) -> NDArray:
    r"""Mean null depth from [B26] Eq. 2 (both polarisation modes).

    .. math::
        \langle N \rangle = \tfrac14\bigl(
            \langle\delta\varphi\rangle^2
          + \sigma_{\delta\varphi}^2
          + \tfrac14 \langle\delta\varphi_{sp}\rangle^2
          + \langle\delta I\rangle^2
          + \sigma_{\delta I}^2
          + \tfrac14 \langle\delta I_{sp}\rangle^2
        \bigr)

    Parameters
    ----------
    dphi_mean  : mean phase error [rad]
    sigma_dphi : phase-error RMS (temporal) [rad]
    dphi_sp    : s-p differential phase error [rad]
    dI_mean    : mean intensity mismatch [fractional, 0-1]
    sigma_dI   : intensity-mismatch RMS (temporal) [fractional]
    dI_sp      : s-p differential intensity mismatch [fractional]
    """
    return 0.25 * (
        np.asarray(dphi_mean) ** 2
        + np.asarray(sigma_dphi) ** 2
        + 0.25 * np.asarray(dphi_sp) ** 2
        + np.asarray(dI_mean) ** 2
        + np.asarray(sigma_dI) ** 2
        + 0.25 * np.asarray(dI_sp) ** 2
    )


def null_depth_simple(dphi: ArrayLike, dI: ArrayLike) -> NDArray:
    r"""Simplified null depth for a single polarisation mode.

    .. math::  N \approx \tfrac14(\delta\varphi^2 + \delta I^2)

    [B26] Appendix A.1, second-order Taylor expansion.
    """
    return 0.25 * (np.asarray(dphi) ** 2 + np.asarray(dI) ** 2)


def null_depth_exact(dphi: ArrayLike, dI: ArrayLike) -> NDArray:
    r"""Exact null depth without Taylor approximation.

    .. math::
        N = \frac{1 - \sqrt{1 - \delta I^2}\,\cos\delta\varphi}
                 {1 + \sqrt{1 - \delta I^2}}

    [B26] Appendix A.1 / [S00].  Valid for arbitrary d_phi and dI in [0, 1).
    """
    dI_safe = np.clip(np.abs(np.asarray(dI, dtype=float)), 0.0, 1.0 - 1e-12)
    sqrt_term = np.sqrt(1.0 - dI_safe ** 2)
    return (1.0 - sqrt_term * np.cos(dphi)) / (1.0 + sqrt_term)


# ============================================================================
# 3. Chromatic OPD <-> phase conversion
# ============================================================================

def opd_to_phase(opd: ArrayLike, wavelength: ArrayLike) -> NDArray:
    r"""Convert OPD [m] to phase error [rad].

    .. math::  \delta\varphi(\lambda) = 2\pi\,\delta\mathrm{OPD}\,/\,\lambda

    The same physical OPD gives a larger phase error at shorter wavelengths.
    """
    return 2.0 * np.pi * np.asarray(opd) / np.asarray(wavelength)


def phase_to_opd(phase: ArrayLike, wavelength: ArrayLike) -> NDArray:
    """Convert phase error [rad] back to OPD [m]."""
    return np.asarray(phase) * np.asarray(wavelength) / (2.0 * np.pi)


# ============================================================================
# 4. Beamsplitter thickness mismatch -> chromatic phase error
# ============================================================================

def bs_thickness_chromatic_opd(
    wavelength: ArrayLike,
    delta_d: float,
    material: str = 'multiband',
) -> NDArray:
    r"""Residual chromatic OPD from differential beamsplitter thickness.

    One arm traverses extra glass thickness Delta_d.  The compensator
    corrects at lam_ref, leaving a residual:

    .. math::
        \delta\mathrm{OPD}_\mathrm{res}(\lambda)
        = \bigl[n(\lambda) - n(\lambda_\mathrm{ref})\bigr]\,\Delta d

    v2.1: Added ``'multiband'`` option matching the LIFE architecture:
    CaF2 for lam < 10 um, ZnSe for lam >= 10 um.  This avoids
    extrapolating CaF2 beyond its 9.7 um Sellmeier validity limit and
    uses ZnSe's well-characterised flat dispersion for the long-wavelength
    sub-band.

    Parameters
    ----------
    wavelength : [m]
    delta_d    : thickness mismatch [m]
    material   : ``'caf2'``, ``'znse'``, or ``'multiband'`` (default)
    """
    n_func = {'caf2': _caf2_n,
              'znse': _znse_n}

    if material == 'multiband':
        lam = np.atleast_1d(np.asarray(wavelength, dtype=float))

        if lam.size == 1:
            # Scalar wavelength: pick material directly so that delta_d
            # (which may be an array, e.g. in tolerance sweeps) broadcasts
            # freely without shape conflicts.
            if lam[0] < LAMBDA_REF:
                n_lam = float(_caf2_n(lam)[0])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    n_ref = float(_caf2_n(np.array([LAMBDA_REF]))[0])
            else:
                n_lam = float(_znse_n(lam)[0])
                n_ref = float(_znse_n(np.array([LAMBDA_REF]))[0])
            return (n_lam - n_ref) * np.asarray(delta_d)

        # Array wavelength path (delta_d is typically scalar here)
        result = np.empty_like(lam)
        # CaF2 for short-wavelength sub-band (lam < 10 um)
        mask_short = lam < LAMBDA_REF
        if np.any(mask_short):
            n_short = _caf2_n(lam[mask_short])
            # Reference n at 10 um is just beyond the formal 9.7 um limit
            # but the Sellmeier is still accurate here (resonance at 34.6 um)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                n_ref_short = float(np.atleast_1d(
                    _caf2_n(LAMBDA_REF))[0])
            result[mask_short] = (n_short - n_ref_short) * delta_d
        # ZnSe for long-wavelength sub-band (lam >= 10 um)
        mask_long = ~mask_short
        if np.any(mask_long):
            n_long = _znse_n(lam[mask_long])
            n_ref_long = float(np.atleast_1d(_znse_n(LAMBDA_REF))[0])
            result[mask_long] = (n_long - n_ref_long) * delta_d
        return result

    if material not in n_func:
        raise ValueError(f"Unknown material: {material!r}. "
                         f"Choose from {{'caf2', 'znse', 'multiband'}}.")

    n_lam = n_func[material](wavelength)
    n_ref = n_func[material](LAMBDA_REF)
    return (n_lam - n_ref) * delta_d


def bs_chromatic_phase_error(
    wavelength: ArrayLike,
    delta_d: float,
    material: str = 'multiband',
) -> NDArray:
    r"""Phase error from BS thickness mismatch.

    .. math::
        \delta\varphi_\mathrm{BS}(\lambda)
        = \frac{2\pi}{\lambda}\bigl[n(\lambda)
          - n(\lambda_\mathrm{ref})\bigr]\,\Delta d

    Parameters
    ----------
    wavelength : [m]
    delta_d    : thickness mismatch [m]
    material   : ``'caf2'``, ``'znse'``, or ``'multiband'`` (default)
    """
    opd = bs_thickness_chromatic_opd(wavelength, delta_d, material)
    return opd_to_phase(opd, wavelength)


# ============================================================================
# 5. Polarisation error: s-p phase and intensity split
# ============================================================================

def polarization_phase_error(
    wavelength: ArrayLike,
    mismatch_angle_deg: float = 0.15,
) -> NDArray:
    r"""s-p differential phase error from imperfect APS.

    The periscope APS provides a geometric pi phase shift that is achromatic.
    Imperfect alignment introduces a small polarisation rotation mismatch
    alpha, producing  d_phi_sp ~ 2*alpha  [B26 Sec. 3.2].

    For a geometric (reflection-based) APS the s-p phase split is
    wavelength-independent -- the gold Fresnel coefficient variation across
    6-16 um is < 1 %, negligible compared to other uncertainties.

    Parameters
    ----------
    wavelength          : [m]  (used only for array shape; value ignored)
    mismatch_angle_deg  : polarisation rotation mismatch [degrees]
    """
    alpha_rad = np.radians(mismatch_angle_deg)
    dphi_sp = 2.0 * alpha_rad
    return dphi_sp * np.ones_like(np.atleast_1d(np.asarray(wavelength)))


def polarization_intensity_mismatch(
    wavelength: ArrayLike,
    mismatch_angle_deg: float = 0.15,
) -> NDArray:
    r"""s-p intensity mismatch from polarisation rotation error.

    For small rotation alpha the projected intensity goes as cos^2(alpha),
    so the fractional mismatch between s and p channels is

    .. math::  \delta I_{sp} \approx 2\,\alpha

    Wavelength-independent for a geometric APS.
    """
    alpha_rad = np.radians(mismatch_angle_deg)
    return 2.0 * alpha_rad * np.ones_like(
        np.atleast_1d(np.asarray(wavelength))
    )


# ============================================================================
# 6. Wavefront error -> differential coupling -> intensity mismatch
# ============================================================================

def wfe_to_coupling_ratio(
    wfe_rms: ArrayLike,
    wavelength: ArrayLike,
) -> NDArray:
    r"""Marechal coupling degradation factor eta/eta_0  [RC01].

    .. math::
        \frac{\eta}{\eta_0}
        = \exp\!\Bigl[-\Bigl(\frac{2\pi\,\sigma}{\lambda}\Bigr)^2\Bigr]

    Parameters
    ----------
    wfe_rms    : RMS wavefront error [m]
    wavelength : [m]

    Returns
    -------
    Coupling ratio eta/eta_0 in (0, 1].
    """
    return np.exp(-(2.0 * np.pi * np.asarray(wfe_rms)
                    / np.asarray(wavelength)) ** 2)


def differential_wfe_to_intensity_mismatch(
    wfe_diff_rms: ArrayLike,
    wavelength: ArrayLike,
) -> NDArray:
    r"""Intensity mismatch from *differential* WFE between two arms.

    The differential WFE sigma_diff represents the total wavefront error
    difference between the two interferometric arms after removing the
    common-mode component.  We model arm 1 as having the full
    differential WFE and arm 2 as perfect (common-path already removed):

    .. math::
        \eta_1 = \eta_0\,\exp\!\bigl[-\bigl(2\pi\,\sigma_\text{diff}
                 /\lambda\bigr)^2\bigr],
        \qquad \eta_2 = \eta_0

    The intensity mismatch is then  dI = |eta_1 - eta_2| / (eta_1 + eta_2).

    Note: a symmetric split (+/- sigma/sqrt(2) per arm) gives dI = 0
    identically under the Marechal approximation because exp(-x^2) is
    even in sigma.  Non-zero mismatch from symmetric WFE requires the
    full overlap integral with Zernike-specific coupling (odd aberrations
    break the symmetry).  For this analytical budget the asymmetric model
    is the standard conservative treatment, consistent with how Module 7
    (MC) interprets the differential component.

    Parameters
    ----------
    wfe_diff_rms : differential WFE RMS between the two arms [m]
    wavelength   : [m]
    """
    eta_1 = wfe_to_coupling_ratio(wfe_diff_rms, wavelength)
    eta_2 = np.ones_like(eta_1)   # perfect arm (common-path already removed)

    denom = eta_1 + eta_2
    denom = np.where(denom > 0, denom, 1.0)  # guard against zero
    return np.abs(eta_1 - eta_2) / denom


def pointing_to_intensity_mismatch(
    sigma_pointing: float,
    wavelength: ArrayLike,
    w0: float = D_BEAM / (2 * np.sqrt(2)),
) -> NDArray:
    r"""Intensity mismatch from differential pointing error.

    For statistically independent Gaussian pointing errors with RMS
    sigma_pointing per beam, the mean absolute differential coupling gives:

    .. math::
        \langle|\delta I|\rangle
        \approx \sqrt{\frac{2}{\pi}}\,
                \Bigl(\frac{\pi\,w_0\,\sigma_\alpha}{\lambda}\Bigr)^2

    Parameters
    ----------
    sigma_pointing : RMS pointing error per beam [rad]
    wavelength     : [m]
    w0             : effective beam waist [m]  (D/(2*sqrt(2)) for top-hat)
    """
    x = (np.pi * w0 * sigma_pointing / np.asarray(wavelength)) ** 2
    return np.sqrt(2.0 / np.pi) * x


def shear_to_intensity_mismatch(
    sigma_shear: float,
    w0: float = D_BEAM / 2,
) -> float:
    r"""Intensity mismatch from lateral beam shear.

    .. math::  \delta I \approx (\sigma_\mathrm{shear} / w_0)^2

    Approximately wavelength-independent because the focused-spot scale
    and fibre-mode radius both scale with lam, cancelling out.
    """
    return (sigma_shear / w0) ** 2


# ============================================================================
# 7. Complete null depth budget: all error terms combined
# ============================================================================

def compute_null_budget(
    wavelengths: ArrayLike,
    opd_mean: float = 0.5e-9,
    opd_rms: float = 1.2e-9,
    dI_mean: float = 0.43e-2,
    dI_rms: float = 0.43e-2,
    pol_angle_deg: float = 0.15,
    bs_delta_d: float = 0.1e-6,
    bs_material: str = 'multiband',
    wfe_diff_rms: float = 50e-9,
    pointing_rms: float = 10e-6,
    shear_rms: float = 0.17e-6,
) -> dict:
    r"""Compute the full null depth budget across wavelength.

    Maps nine physical error sources onto the six slots of [B26] Eq. 2:

    **Phase terms** (-> <d_phi>^2, sigma^2_dphi, 1/4 d_phi_sp^2)
      1. Mean OPD offset   }
      2. BS chromatic disp. } -> coherently summed into <d_phi>_total,
                                  then squared
      3. OPD RMS jitter     -> sigma^2_dphi
      4. s-p phase split    -> 1/4 d_phi_sp^2

    **Intensity terms** (-> <dI>^2, sigma^2_dI, 1/4 dI_sp^2)
      5. Static mean mismatch }
      6. WFE-driven coupling  } -> RSS-combined into <dI>_total
      7. Pointing coupling    }
      8. Fluctuating mismatch -> sigma^2_dI
      9. s-p intensity split  -> 1/4 dI_sp^2

    Parameters
    ----------
    wavelengths   : [m]
    opd_mean      : mean OPD offset from null [m]         (default 0.5 nm)
    opd_rms       : OPD RMS fluctuation [m]               (default 1.2 nm, NICE)
    dI_mean       : mean intensity mismatch [fractional]   (default 0.43 %, NICE)
    dI_rms        : intensity mismatch RMS [fractional]    (default 0.43 %, NICE)
    pol_angle_deg : polarisation rotation mismatch [deg]   (default 0.15 deg)
    bs_delta_d    : BS glass thickness mismatch [m]        (default 0.1 um)
    bs_material   : ``'caf2'``, ``'znse'``, or ``'multiband'`` (default)
    wfe_diff_rms  : differential WFE between arms [m]      (default 50 nm)
    pointing_rms  : pointing error RMS [rad]               (default 10 urad)
    shear_rms     : lateral shear RMS [m]                  (default 0.17 um)

    Returns
    -------
    dict with wavelength-resolved arrays for every term and the total.
    """
    lam = np.atleast_1d(np.asarray(wavelengths, dtype=float))

    # ------------------------------------------------------------------
    # Phase error terms
    # ------------------------------------------------------------------

    # 1+2. Mean OPD offset + BS chromatic dispersion  (COHERENT sum)
    dphi_opd_mean = opd_to_phase(opd_mean, lam)
    dphi_chromatic = bs_chromatic_phase_error(lam, bs_delta_d, bs_material)
    dphi_systematic = dphi_opd_mean + dphi_chromatic   # coherent addition
    N_phase_systematic = 0.25 * dphi_systematic ** 2

    # For diagnostic breakdown, also store individual + cross term:
    N_opd_mean_only = 0.25 * dphi_opd_mean ** 2
    N_chromatic_only = 0.25 * dphi_chromatic ** 2
    N_cross_term = 0.25 * 2.0 * dphi_opd_mean * dphi_chromatic
    # Verify: N_phase_systematic = N_opd_mean + N_chromatic + N_cross_term

    # 3. OPD RMS jitter
    sigma_dphi = opd_to_phase(opd_rms, lam)
    N_opd_rms = 0.25 * sigma_dphi ** 2

    # 4. Polarisation phase (s-p split)
    dphi_sp = polarization_phase_error(lam, pol_angle_deg)
    N_pol_phase = 0.25 * 0.25 * dphi_sp ** 2    # factor 1/4 from [B26] Eq. 3

    # ------------------------------------------------------------------
    # Intensity mismatch terms
    # ------------------------------------------------------------------

    # 5. Static intensity mismatch (from source / BS splitting ratio)
    dI_static = dI_mean

    # 6. WFE-driven coupling mismatch
    dI_wfe = differential_wfe_to_intensity_mismatch(wfe_diff_rms, lam)

    # 7. Pointing-driven coupling mismatch
    dI_pointing = pointing_to_intensity_mismatch(pointing_rms, lam)

    # Combined mean intensity mismatch (RSS of independent sources)
    # RSS is valid because dI contributions are independent and
    # the null ~ dI^2, so 1/4 * sum(dI_i^2) = sum(1/4 * dI_i^2)
    N_dI_mean_static = 0.25 * dI_static ** 2 * np.ones_like(lam)
    N_wfe = 0.25 * dI_wfe ** 2
    N_pointing = 0.25 * dI_pointing ** 2

    # 8. Fluctuating intensity mismatch
    N_dI_rms = 0.25 * dI_rms ** 2 * np.ones_like(lam)

    # 9. Polarisation intensity mismatch
    dI_sp = polarization_intensity_mismatch(lam, pol_angle_deg)
    N_pol_intensity = 0.25 * 0.25 * dI_sp ** 2

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------
    N_total = (N_phase_systematic + N_opd_rms
               + N_pol_phase
               + N_dI_mean_static + N_wfe + N_pointing
               + N_dI_rms
               + N_pol_intensity)

    return {
        'wavelength': lam,
        'N_total': N_total,
        # Phase diagnostic breakdown
        'N_phase_systematic': N_phase_systematic,
        'N_opd_mean': N_opd_mean_only,
        'N_chromatic': N_chromatic_only,
        'N_cross_term': N_cross_term,
        'N_opd_rms': N_opd_rms,
        'N_pol_phase': N_pol_phase,
        # Intensity diagnostic breakdown
        'N_dI_mean': N_dI_mean_static,
        'N_dI_rms': N_dI_rms,
        'N_wfe': N_wfe,
        'N_pointing': N_pointing,
        'N_pol_intensity': N_pol_intensity,
    }


# ============================================================================
# 8. Monte Carlo null depth simulation
# ============================================================================

def monte_carlo_null(
    wavelength: float,
    N_realizations: int = 100_000,
    opd_mean: float = 0.5e-9,
    opd_rms: float = 1.2e-9,
    dI_mean: float = 0.43e-2,
    dI_rms: float = 0.43e-2,
    pol_angle_deg: float = 0.15,
    bs_delta_d: float = 0.1e-6,
    bs_material: str = 'multiband',
    wfe_diff_rms: float = 50e-9,
    pointing_rms: float = 10e-6,
    seed: int = 42,
) -> tuple[NDArray, dict]:
    """Monte Carlo simulation of null depth at a single wavelength.

    Draws all error sources from their distributions and evaluates the
    EXACT (non-Taylor) null depth formula for each realisation, correctly
    combining s and p polarisation channels.

    Returns
    -------
    N_samples : array of shape (N_realizations,)
    stats     : dict with mean, median, std, p95, p99, min, max
    """
    rng = np.random.default_rng(seed)

    # --- Phase: OPD samples (Gaussian) + deterministic BS chromatic ---
    opd_samples = rng.normal(opd_mean, opd_rms, N_realizations)
    dphi_opd = opd_to_phase(opd_samples, wavelength)

    # BS chromatic (deterministic at given lam)
    dphi_chrom = float(np.atleast_1d(
        bs_chromatic_phase_error(wavelength, bs_delta_d, bs_material)
    )[0])

    # Total phase per realisation (coherent sum -- same as the Birbacher
    # Appendix A treatment: all phase offsets add before entering |E1-E2|^2)
    dphi_total = dphi_opd + dphi_chrom

    # --- Polarisation: s-p phase split (static) ---
    dphi_sp = float(np.atleast_1d(
        polarization_phase_error(wavelength, pol_angle_deg)
    )[0])

    # Phase for each polarisation:
    #   s-pol -> d_phi_total + d_phi_sp/2
    #   p-pol -> d_phi_total - d_phi_sp/2
    dphi_s = dphi_total + dphi_sp / 2
    dphi_p = dphi_total - dphi_sp / 2

    # --- Intensity mismatch ---
    # Systematic (deterministic) components
    dI_wfe = float(np.atleast_1d(
        differential_wfe_to_intensity_mismatch(wfe_diff_rms, wavelength)
    )[0])
    dI_point = float(np.atleast_1d(
        pointing_to_intensity_mismatch(pointing_rms, wavelength)
    )[0])

    # RSS of systematic intensity mismatch sources
    dI_systematic = np.sqrt(dI_mean ** 2 + dI_wfe ** 2 + dI_point ** 2)

    # Sample fluctuating component, add to systematic (always positive)
    dI_samples = np.abs(
        rng.normal(dI_systematic, dI_rms, N_realizations)
    )

    # s-p intensity split (static)
    dI_sp = float(np.atleast_1d(
        polarization_intensity_mismatch(wavelength, pol_angle_deg)
    )[0])

    dI_s = np.abs(dI_samples + dI_sp / 2)
    dI_p = np.abs(dI_samples - dI_sp / 2)

    # --- Exact null for each polarisation ---
    N_s = null_depth_exact(dphi_s, dI_s)
    N_p = null_depth_exact(dphi_p, dI_p)

    # Combined: average of s and p  [B26] Appendix A.2
    N_samples = 0.5 * (N_s + N_p)

    return N_samples, {
        'mean': float(np.mean(N_samples)),
        'median': float(np.median(N_samples)),
        'std': float(np.std(N_samples)),
        'p95': float(np.percentile(N_samples, 95)),
        'p99': float(np.percentile(N_samples, 99)),
        'min': float(np.min(N_samples)),
        'max': float(np.max(N_samples)),
    }


# ============================================================================
# 9. NICE validation model
# ============================================================================

def nice_null_model(
    wavelength: float = LAMBDA_NICE_NULL,
    opd_mean: float = 0.3e-9,
    opd_rms: float = 1.2e-9,
    dI_mean: float = 0.43e-2,
    dI_rms: float = 0.0,
) -> dict:
    r"""Predict the NICE null depth at the measurement wavelength.

    NICE measured: N = 7.17 x 10^-6 at 4.7 um  [B26 Sec. 4.1]
    with sigma_OPD = 1.2 nm, dI_mean = 0.43 %  [B26 Table 4]

    The NICE measurement used a narrowband laser -> BS chromatic term ~ 0.
    A linear polariser at the output -> polarisation terms ~ 0.

    Birbacher Eq. 9 predicts 5.27 x 10^-6 using only sigma_OPD and dI_mean.
    We include all four non-polarisation terms:
      <N> ~ 1/4(<d_phi>^2 + sigma^2_dphi + <dI>^2 + sigma^2_dI)

    Note on sigma_dI:  [B26] Table 4 reports sigma_I = 1.4 % as the
    long-term intensity fluctuation RMS (laser source A).  However, this
    represents drift over the full measurement campaign, not the
    instantaneous stability during the best null measurement.  Including
    it in the 1/4*sigma^2_dI term would predict N ~ 5x10^-5, far above
    the measured 7.17x10^-6.  Birbacher's own model (Eq. 9) omits
    sigma_dI, so we default to dI_rms=0.  Pass dI_rms > 0 to explore
    its impact on null floor scatter.
    """
    # Phase contributions
    dphi_mean = opd_to_phase(opd_mean, wavelength)
    sigma_dphi = opd_to_phase(opd_rms, wavelength)
    N_opd_mean = 0.25 * dphi_mean ** 2
    N_opd_rms = 0.25 * sigma_dphi ** 2

    # Intensity contributions
    N_dI_mean = 0.25 * dI_mean ** 2
    N_dI_rms = 0.25 * dI_rms ** 2

    N_total = N_opd_mean + N_opd_rms + N_dI_mean + N_dI_rms

    return {
        'N_total': N_total,
        'N_opd_mean': N_opd_mean,
        'N_opd_rms': N_opd_rms,
        'N_dI_mean': N_dI_mean,
        'N_dI_rms': N_dI_rms,
        'N_measured': 7.17e-6,
    }


# ============================================================================
# 10. Analytical-vs-MC cross-validation helper
# ============================================================================

def cross_validate(
    wavelength: float,
    N_mc: int = 100_000,
    **kwargs,
) -> dict:
    """Compare analytical budget vs Monte Carlo at a single wavelength.

    Returns percentage difference and both results for inspection.
    """
    budget = compute_null_budget(np.array([wavelength]), **kwargs)
    N_analytical = float(budget['N_total'][0])

    _, mc_stats = monte_carlo_null(wavelength, N_realizations=N_mc, **kwargs)
    N_mc_mean = mc_stats['mean']

    return {
        'wavelength': wavelength,
        'N_analytical': N_analytical,
        'N_mc_mean': N_mc_mean,
        'diff_pct': (N_mc_mean - N_analytical) / N_analytical * 100,
        'mc_stats': mc_stats,
    }


# ============================================================================
# 11. Main analysis and figure generation
# ============================================================================

def run_full_analysis() -> dict:
    """Run the complete Module 3 analysis and produce publication figures.

    Produces:
      Fig 9:  Null depth error budget vs wavelength (all terms)
      Fig 10: Error budget breakdown at key wavelengths (bar chart)
      Fig 11: Monte Carlo null depth distributions at 6, 10, 16 um
      Fig 12: OPD and BS-thickness tolerance curves
    """

    print("=" * 70)
    print("LIFE E2E Module 3: Null Depth Error Propagation  (v3.0)")
    print("=" * 70)

    # Wavelength grid
    wavelengths = np.linspace(4e-6, 18.5e-6, 300)
    lam_um = wavelengths * 1e6

    # ==================================================================
    # NICE validation
    # ==================================================================
    print("\n--- NICE Validation ---")
    nice = nice_null_model()
    print(f"  NICE at 4.7 um:")
    print(f"    N_opd_mean    = {nice['N_opd_mean']:.2e}")
    print(f"    N_opd_rms     = {nice['N_opd_rms']:.2e}")
    print(f"    N_dI_mean     = {nice['N_dI_mean']:.2e}")
    print(f"    N_dI_rms      = {nice['N_dI_rms']:.2e}")
    print(f"    Model total   = {nice['N_total']:.2e}")
    print(f"    Measured       = {nice['N_measured']:.2e}")
    ratio = nice['N_total'] / nice['N_measured']
    print(f"    Ratio (model/meas) = {ratio:.2f}")

    # ==================================================================
    # Baseline error budget: NICE-demonstrated performance
    # ==================================================================
    print("\n--- Baseline Error Budget (NICE-demonstrated) ---")

    budget_nice = compute_null_budget(
        wavelengths,
        opd_mean=0.5e-9,
        opd_rms=1.2e-9,
        dI_mean=0.43e-2,
        dI_rms=0.43e-2,
        pol_angle_deg=0.15,
        bs_delta_d=0.1e-6,
        bs_material='multiband',
        wfe_diff_rms=50e-9,
        pointing_rms=10e-6,
        shear_rms=0.17e-6,
    )

    # LIFE-requirement performance (stricter)
    budget_life = compute_null_budget(
        wavelengths,
        opd_mean=0.3e-9,
        opd_rms=0.8e-9,
        dI_mean=0.26e-2,
        dI_rms=0.02e-2,
        pol_angle_deg=0.15,
        bs_delta_d=0.05e-6,
        bs_material='multiband',
        wfe_diff_rms=30e-9,
        pointing_rms=5e-6,
        shear_rms=0.1e-6,
    )

    # Print summary at key wavelengths
    for lam_target in [6e-6, 8e-6, 10e-6, 12e-6, 16e-6]:
        idx = np.argmin(np.abs(wavelengths - lam_target))
        lam_label = f"{lam_target * 1e6:.0f} um"
        N_req = float(null_requirement_curve(lam_target))
        N_nice = budget_nice['N_total'][idx]
        N_life = budget_life['N_total'][idx]
        s_nice = "PASS" if N_nice < N_req else "FAIL"
        s_life = "PASS" if N_life < N_req else "FAIL"
        cross = budget_nice['N_cross_term'][idx]
        print(f"  lam={lam_label}:  Req={N_req:.1e}  "
              f"NICE->{N_nice:.1e}[{s_nice}]  "
              f"LIFE->{N_life:.1e}[{s_life}]  "
              f"cross={cross:+.1e}")

    # ==================================================================
    # Analytical-vs-MC cross-validation
    # ==================================================================
    print("\n--- Analytical vs MC Cross-Validation ---")
    for lam_target in [6e-6, 10e-6, 16e-6]:
        cv = cross_validate(
            lam_target,
            opd_mean=0.5e-9, opd_rms=1.2e-9,
            dI_mean=0.43e-2, dI_rms=0.43e-2,
            pol_angle_deg=0.15,
            bs_delta_d=0.1e-6, wfe_diff_rms=50e-9,
            pointing_rms=10e-6,
        )
        print(f"  lam={lam_target*1e6:.0f} um: "
              f"Analytical={cv['N_analytical']:.2e}  "
              f"MC mean={cv['N_mc_mean']:.2e}  "
              f"Delta={cv['diff_pct']:+.1f}%")

    # ==================================================================
    # Figure 9: Null depth error budget vs wavelength
    # ==================================================================
    print("\n--- Figure 9: Null depth error budget vs wavelength ---")

    fig9, (ax9a, ax9b) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Panel A: NICE-demonstrated performance ---
    b = budget_nice
    ax9a.semilogy(lam_um, b['N_opd_mean'], 'b-', lw=1.5,
                  label='OPD mean offset')
    ax9a.semilogy(lam_um, b['N_opd_rms'], 'b--', lw=2.5,
                  label=r'OPD RMS ($\sigma$=1.2 nm)')
    ax9a.semilogy(lam_um, np.maximum(np.abs(b['N_chromatic']), 1e-20),
                  'c-', lw=1.5,
                  label=r'BS chromatic ($\Delta d$=0.1 $\mu$m)')
    ax9a.semilogy(lam_um, b['N_phase_systematic'], 'b:', lw=2,
                  label='OPD mean + BS chromatic (coherent)')
    ax9a.semilogy(lam_um, b['N_dI_mean'], 'r-', lw=2,
                  label='Intensity mismatch mean')
    ax9a.semilogy(lam_um, b['N_dI_rms'], 'r--', lw=2,
                  label='Intensity mismatch RMS')
    ax9a.semilogy(lam_um, b['N_pol_phase'], 'g-', lw=1.5,
                  label='Polarisation phase')
    ax9a.semilogy(lam_um, b['N_pol_intensity'], 'g--', lw=1.5,
                  label='Polarisation intensity')
    ax9a.semilogy(lam_um, b['N_wfe'], 'm-', lw=1.5,
                  label='Differential WFE')
    ax9a.semilogy(lam_um, b['N_pointing'], 'm--', lw=1.5,
                  label=r'Pointing $\to$ coupling')
    ax9a.semilogy(lam_um, b['N_total'], 'k-', lw=3,
                  label='Total null depth')

    # Requirement curve
    N_req = null_requirement_curve(wavelengths)
    ax9a.semilogy(lam_um, N_req, 'k:', lw=2, label='Requirement')

    # NICE measured point
    ax9a.plot(4.7, 7.17e-6, 'r*', ms=15, zorder=10, label='NICE measured')

    # LIFE science band
    ax9a.axvspan(6, 16, alpha=0.06, color='green')

    ax9a.set_ylabel('Null depth contribution', fontsize=13)
    ax9a.set_title(
        'Null error budget -- NICE-demonstrated performance', fontsize=13)
    ax9a.legend(fontsize=8, ncol=2, loc='upper right')
    ax9a.set_ylim(1e-10, 1e-3)
    ax9a.set_xlim(4, 18.5)
    ax9a.grid(True, alpha=0.3, which='both')
    ax9a.text(6.3, 2e-4, 'LIFE science band', fontsize=10,
              color='green', alpha=0.7)

    # --- Panel B: LIFE-requirement performance ---
    b = budget_life
    ax9b.semilogy(lam_um, b['N_opd_mean'], 'b-', lw=1.5,
                  label='OPD mean offset')
    ax9b.semilogy(lam_um, b['N_opd_rms'], 'b--', lw=2.5,
                  label=r'OPD RMS ($\sigma$=0.8 nm)')
    ax9b.semilogy(lam_um, np.maximum(np.abs(b['N_chromatic']), 1e-20),
                  'c-', lw=1.5,
                  label=r'BS chromatic ($\Delta d$=0.05 $\mu$m)')
    ax9b.semilogy(lam_um, b['N_phase_systematic'], 'b:', lw=2,
                  label='OPD mean + BS chromatic (coherent)')
    ax9b.semilogy(lam_um, b['N_dI_mean'], 'r-', lw=2,
                  label='Intensity mismatch mean')
    ax9b.semilogy(lam_um, b['N_dI_rms'], 'r--', lw=2,
                  label='Intensity mismatch RMS')
    ax9b.semilogy(lam_um, b['N_pol_phase'], 'g-', lw=1.5,
                  label='Polarisation phase')
    ax9b.semilogy(lam_um, b['N_pol_intensity'], 'g--', lw=1.5,
                  label='Polarisation intensity')
    ax9b.semilogy(lam_um, b['N_wfe'], 'm-', lw=1.5,
                  label='Differential WFE')
    ax9b.semilogy(lam_um, b['N_pointing'], 'm--', lw=1.5,
                  label=r'Pointing $\to$ coupling')
    ax9b.semilogy(lam_um, b['N_total'], 'k-', lw=3,
                  label='Total null depth')

    ax9b.semilogy(lam_um, N_req, 'k:', lw=2, label='Requirement')
    ax9b.plot(4.7, 7.17e-6, 'r*', ms=15, zorder=10, label='NICE measured')
    ax9b.axvspan(6, 16, alpha=0.06, color='green')

    ax9b.set_xlabel(r'Wavelength [$\mu$m]', fontsize=13)
    ax9b.set_ylabel('Null depth contribution', fontsize=13)
    ax9b.set_title('Null error budget -- LIFE requirements met', fontsize=13)
    ax9b.legend(fontsize=7, ncol=2, loc='upper right')
    ax9b.set_ylim(1e-10, 1e-3)
    ax9b.grid(True, alpha=0.3, which='both')

    fig9.tight_layout()
    fig9.savefig('fig9_null_error_budget.png', dpi=200,
                 bbox_inches='tight')
    plt.close()
    print("  Saved: fig9_null_error_budget.png")

    # ==================================================================
    # Figure 10: Error budget breakdown at key wavelengths (bar chart)
    # ==================================================================
    print("\n--- Figure 10: Error budget breakdown at key wavelengths ---")

    fig10, axes10 = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    key_wavelengths = [6e-6, 10e-6, 16e-6]
    key_labels = [r'6 $\mu$m' + '\n(most stringent OPD)',
                  r'10 $\mu$m' + '\n(reference)',
                  r'16 $\mu$m' + '\n(band edge)']

    term_names = ['Phase\nsystematic', 'OPD\nRMS',
                  'dI\nmean', 'dI\nRMS',
                  'Pol.\nphase', 'Pol.\nintensity',
                  'Diff.\nWFE', 'Pointing\ncoupling']
    term_keys = ['N_phase_systematic', 'N_opd_rms',
                 'N_dI_mean', 'N_dI_rms',
                 'N_pol_phase', 'N_pol_intensity',
                 'N_wfe', 'N_pointing']
    colors = ['#1f77b4', '#1f77b4',
              '#d62728', '#d62728',
              '#2ca02c', '#2ca02c',
              '#9467bd', '#9467bd']
    hatches = ['', '//',
               '', '//',
               '', '//',
               '', '//']

    for ax_idx, (lam_target, label) in enumerate(
            zip(key_wavelengths, key_labels)):
        ax = axes10[ax_idx]
        idx = np.argmin(np.abs(wavelengths - lam_target))

        values = [budget_nice[k][idx] for k in term_keys]
        N_total = budget_nice['N_total'][idx]
        N_req_val = float(null_requirement_curve(lam_target))

        bars = ax.bar(range(len(values)), values, color=colors,
                      edgecolor='black', linewidth=0.5)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.axhline(y=N_req_val, color='red', ls='--', lw=2,
                   label=f'Requirement: {N_req_val:.1e}')
        ax.axhline(y=N_total, color='black', ls='-', lw=2,
                   label=f'Total: {N_total:.1e}')

        ax.set_xticks(range(len(term_names)))
        ax.set_xticklabels(term_names, fontsize=7, rotation=45, ha='right')
        ax.set_yscale('log')
        ax.set_ylim(1e-11, 1e-3)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both', axis='y')

    axes10[0].set_ylabel('Null depth contribution', fontsize=12)
    fig10.suptitle('Error budget breakdown (NICE-demonstrated performance)',
                   fontsize=14, y=1.02)
    fig10.tight_layout()
    fig10.savefig('fig10_error_breakdown.png', dpi=200,
                  bbox_inches='tight')
    plt.close()
    print("  Saved: fig10_error_breakdown.png")

    # ==================================================================
    # Figure 11: Monte Carlo null depth distributions
    # ==================================================================
    print("\n--- Figure 11: Monte Carlo null depth distributions ---")

    fig11, axes11 = plt.subplots(1, 3, figsize=(14, 5))

    mc_wavelengths = [6e-6, 10e-6, 16e-6]
    mc_labels = [r'$\lambda$ = 6 $\mu$m',
                 r'$\lambda$ = 10 $\mu$m',
                 r'$\lambda$ = 16 $\mu$m']
    mc_colors = ['#1f77b4', '#2ca02c', '#d62728']

    for ax_idx, (lam, label, color) in enumerate(
            zip(mc_wavelengths, mc_labels, mc_colors)):
        ax = axes11[ax_idx]

        N_samples, stats = monte_carlo_null(
            lam, N_realizations=100_000,
            opd_mean=0.5e-9, opd_rms=1.2e-9,
            dI_mean=0.43e-2, dI_rms=0.43e-2,
            pol_angle_deg=0.15,
            bs_delta_d=0.1e-6, bs_material='multiband',
            wfe_diff_rms=50e-9, pointing_rms=10e-6,
        )

        log_N = np.log10(N_samples[N_samples > 0])
        ax.hist(log_N, bins=100, density=True, color=color, alpha=0.7,
                edgecolor='black', linewidth=0.3)

        ax.axvline(np.log10(stats['mean']), color='black', ls='-', lw=2,
                   label=f"Mean: {stats['mean']:.1e}")
        ax.axvline(np.log10(stats['median']), color='black', ls='--', lw=1.5,
                   label=f"Median: {stats['median']:.1e}")
        ax.axvline(np.log10(stats['p95']), color='orange', ls='-', lw=1.5,
                   label=f"95th %: {stats['p95']:.1e}")

        N_req_val = float(null_requirement_curve(lam))
        ax.axvline(np.log10(N_req_val), color='red', ls=':', lw=2,
                   label=f"Req: {N_req_val:.1e}")

        budget_at_lam = compute_null_budget(
            np.array([lam]),
            opd_mean=0.5e-9, opd_rms=1.2e-9,
            dI_mean=0.43e-2, dI_rms=0.43e-2,
            pol_angle_deg=0.15,
            bs_delta_d=0.1e-6, wfe_diff_rms=50e-9,
            pointing_rms=10e-6,
        )
        N_analytical = budget_at_lam['N_total'][0]
        ax.axvline(np.log10(N_analytical), color='purple', ls='-.',
                   lw=1.5, label=f"Analytical: {N_analytical:.1e}")

        ax.set_xlabel(r'$\log_{10}$(Null Depth)', fontsize=11)
        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

        print(f"  {label}: Mean={stats['mean']:.2e}, "
              f"Analytical={N_analytical:.2e}, "
              f"Ratio={stats['mean']/N_analytical:.2f}")

    axes11[0].set_ylabel('Probability density', fontsize=12)
    fig11.suptitle(
        r'Monte Carlo null depth distributions (10$^5$ realisations, '
        'NICE performance)', fontsize=13, y=1.02)
    fig11.tight_layout()
    fig11.savefig('fig11_monte_carlo_null.png', dpi=200,
                  bbox_inches='tight')
    plt.close()
    print("  Saved: fig11_monte_carlo_null.png")

    # ==================================================================
    # Figure 12: OPD and BS tolerance curves
    # ==================================================================
    print("\n--- Figure 12: OPD tolerance curves ---")

    fig12, (ax12a, ax12b) = plt.subplots(1, 2, figsize=(13, 6))

    # --- Panel A: Null depth vs OPD RMS ---
    opd_range = np.linspace(0, 5e-9, 200)

    test_wavelengths = [4e-6, 6e-6, 8e-6, 10e-6, 12e-6, 16e-6]
    colors_12 = ['#440154', '#3b528b', '#21918c',
                 '#5ec962', '#fde725', '#d62728']

    for lam, col in zip(test_wavelengths, colors_12):
        dphi = opd_to_phase(opd_range, lam)
        N_opd = 0.25 * dphi ** 2
        ax12a.semilogy(opd_range * 1e9, N_opd, '-', color=col, lw=2,
                       label=r'{:.0f} $\mu$m'.format(lam * 1e6))

    ax12a.axhline(y=1e-5, color='red', ls=':', alpha=0.6)
    ax12a.text(4.1, 1.2e-5, r'N = $10^{-5}$', fontsize=9, color='red')
    ax12a.axhline(y=3e-5, color='orange', ls=':', alpha=0.6)
    ax12a.text(4.1, 3.5e-5, r'N = $3 \times 10^{-5}$',
               fontsize=9, color='orange')

    ax12a.axvline(x=1.2, color='gray', ls='--', alpha=0.6)
    ax12a.text(1.3, 3e-4, 'NICE\nmeasured\n(1.2 nm)',
               fontsize=9, color='gray')

    ax12a.set_xlabel('OPD RMS [nm]', fontsize=13)
    ax12a.set_ylabel('Null depth (OPD term only)', fontsize=13)
    ax12a.set_title('Null depth vs OPD stability', fontsize=13)
    ax12a.legend(fontsize=10, title='Wavelength')
    ax12a.set_ylim(1e-8, 1e-2)
    ax12a.set_xlim(0, 5)
    ax12a.grid(True, alpha=0.3, which='both')

    # --- Panel B: BS thickness mismatch tolerance ---
    delta_d_range = np.linspace(0, 1e-6, 200)

    for lam, col in zip([6e-6, 8e-6, 10e-6, 12e-6, 16e-6], colors_12[1:]):
        dphi_bs = bs_chromatic_phase_error(lam, delta_d_range, 'multiband')
        N_bs = 0.25 * dphi_bs ** 2
        N_bs = np.maximum(N_bs, 1e-15)
        ax12b.semilogy(delta_d_range * 1e6, N_bs, '-', color=col, lw=2,
                       label=r'{:.0f} $\mu$m'.format(lam * 1e6))

    ax12b.axhline(y=1e-5, color='red', ls=':', alpha=0.6)
    ax12b.text(0.82, 1.2e-5, r'N = $10^{-5}$', fontsize=9, color='red')

    ax12b.axvline(x=0.2, color='gray', ls='--', alpha=0.6)
    ax12b.text(0.22, 3e-4, r'NICE req' + '\n' + r'(0.2 $\mu$m)',
               fontsize=9, color='gray')

    ax12b.set_xlabel(r'BS thickness mismatch $\Delta d$ [$\mu$m]',
                     fontsize=13)
    ax12b.set_ylabel('Null depth (chromatic term only)', fontsize=13)
    ax12b.set_title('Null depth vs BS thickness mismatch', fontsize=13)
    ax12b.legend(fontsize=10, title='Wavelength')
    ax12b.set_ylim(1e-12, 1e-2)
    ax12b.set_xlim(0, 1.0)
    ax12b.grid(True, alpha=0.3, which='both')

    fig12.tight_layout()
    fig12.savefig('fig12_opd_tolerance.png', dpi=200,
                  bbox_inches='tight')
    plt.close()
    print("  Saved: fig12_opd_tolerance.png")

    # ==================================================================
    # Summary Table
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Null Depth Budget at Key Wavelengths")
    print("=" * 70)
    hdr = f"{'':>22s}"
    for l in [6, 8, 10, 12, 16]:
        hdr += f"  {l:>2d} um    "
    print(hdr)
    print("-" * 80)

    rows = [
        ('Requirement', None),
        ('Phase systematic*', 'N_phase_systematic'),
        ('  (OPD mean only)', 'N_opd_mean'),
        ('  (BS chrom only)', 'N_chromatic'),
        ('  (cross term)', 'N_cross_term'),
        ('OPD RMS', 'N_opd_rms'),
        ('dI mean', 'N_dI_mean'),
        ('dI RMS', 'N_dI_rms'),
        ('Pol. phase', 'N_pol_phase'),
        ('Pol. intensity', 'N_pol_intensity'),
        ('Diff. WFE', 'N_wfe'),
        ('Pointing', 'N_pointing'),
        ('TOTAL', 'N_total'),
    ]

    for name, key in rows:
        row = f"{name:>22s}"
        for lam_target in [6e-6, 8e-6, 10e-6, 12e-6, 16e-6]:
            if key is None:
                val = float(null_requirement_curve(lam_target))
            else:
                idx = np.argmin(np.abs(wavelengths - lam_target))
                val = budget_nice[key][idx]
            row += f"  {val:>10.1e}"
        print(row)

    print("-" * 80)
    print("* Phase systematic = (OPD mean + BS chromatic) coherently summed")
    print("\nUsing NICE-demonstrated performance: "
          "sigma_OPD=1.2 nm, dI=0.43%, pol=0.15 deg, Delta_d_BS=0.1 um")

    # Dominant error identification
    print("\n--- Dominant Error Terms ---")
    for lam_target in [6e-6, 10e-6, 16e-6]:
        idx = np.argmin(np.abs(wavelengths - lam_target))
        N_total = budget_nice['N_total'][idx]

        fracs = {}
        for name, key in [('Phase systematic', 'N_phase_systematic'),
                          ('OPD RMS', 'N_opd_rms'),
                          ('dI mean', 'N_dI_mean'),
                          ('dI RMS', 'N_dI_rms'),
                          ('Pol. phase', 'N_pol_phase'),
                          ('Pol. intensity', 'N_pol_intensity'),
                          ('Diff. WFE', 'N_wfe'),
                          ('Pointing', 'N_pointing')]:
            fracs[name] = budget_nice[key][idx] / N_total * 100

        sorted_fracs = sorted(fracs.items(), key=lambda x: x[1],
                              reverse=True)
        top3 = sorted_fracs[:3]
        print(f"  lam={lam_target * 1e6:.0f} um: "
              f"{top3[0][0]} ({top3[0][1]:.0f}%), "
              f"{top3[1][0]} ({top3[1][1]:.0f}%), "
              f"{top3[2][0]} ({top3[2][1]:.0f}%)")

    plt.close('all')
    print("\nAll figures saved. Module 3 v3.0 complete.")

    return {
        'budget_nice': budget_nice,
        'budget_life': budget_life,
        'nice_validation': nice,
    }


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    results = run_full_analysis()
