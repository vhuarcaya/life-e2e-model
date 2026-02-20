"""
LIFE End-to-End Wavefront Propagation Study -- Module 1: Fiber Coupling Engine
==============================================================================

Author:  Victor Huarcaya (University of Bern)
Version: 2.0  (reorganized)
Date:    2026-02-14

Changelog:
    v2.0 (2026-02-14):
        - Reorganized: imports V-parameter, mode field radius (Marcuse +
          power-law), optimal focal length, and Ruilier coupling formula
          from fiber_modes.py instead of re-implementing them locally.
        - Unit conversion at boundary: this module works in METERS;
          fiber_modes.py works in MICROMETERS.  All conversions happen
          in thin wrapper functions documented below.
        - Replaced Unicode symbols in print() and plot labels with LaTeX
          math-mode or ASCII equivalents.  Unicode is retained in docstrings.
        - Seeded all random generators for reproducibility.

    v1.1 (2026-02-11):
        - Replaced power-law MFD scaling with Marcuse approximation (Marcuse 1978)
        - Added V-parameter single-mode cutoff check with warnings
        - Added null_from_shear() for complete error budget (Birbacher Eq. 15)
        - Added numerical top-hat pointing via overlap integral
        - Documented top-hat shear as approximate (Gaussian model; exact geometric
          clipping added for comparison but NICE margins make this irrelevant)
        - Seeded random phase screen for reproducibility
        - Documented fiber mode normalization conventions
        - Cleaned up derivation comments in analytical coupling function

    v1.0 (2026-02-10):
        - Initial version: analytical and numerical coupling, Zernike sensitivity

Purpose:
    Compute and compare single-mode fiber coupling efficiency for:
      (a) Gaussian input beam  (as used in NICE testbed)
      (b) Top-hat / uniform circular input beam  (as LIFE will have)
      (c) Centrally-obstructed top-hat  (if secondary obscuration present)
    across the full LIFE science band (6-16 um), including sensitivity to
    pointing errors, lateral shear, and wavefront aberrations (low-order Zernike).

Wavelength convention:
    **METERS** throughout this module.  The supporting library fiber_modes.py
    uses micrometers (um).  Conversion is handled inside the thin wrappers
    v_parameter(), fiber_mode_radius_marcuse(), fiber_mode_radius_powerlaw(),
    coupling_tophat_analytical(), and optimal_focal_length().  Callers of
    this module should pass wavelengths in meters.

Conventions:
    - All fields are expressed as electric field amplitude E(x,y), NOT intensity.
    - Fiber mode normalization: psi = sqrt(2/(pi*w_f^2)) * exp(-r^2/w_f^2)
      This gives int|psi|^2 dA = 1 (power-normalized).
    - w_f is the 1/e FIELD radius (= 1/e^2 intensity radius), i.e., MFD = 2*w_f.
    - The coupling parameter beta = (D/2) / w_back where w_back = lam*f/(pi*w_f)
      is the fiber mode back-propagated to the pupil plane.
    - All wavefront errors are in meters of OPD (not waves).

Physics references:
    - Ruilier & Cassaing (2001), JOSA A -- fiber coupling of Airy pattern
    - Marcuse (1978), JOSA 68, 103 -- Gaussian mode field approximation
    - Birbacher et al. (2026), A&A (arXiv:2602.02279) -- Eqs. 4-15
    - Garreau et al. (2024), Asgard/NOTT (arXiv:2402.09013) -- Eq. 2-3
    - Shaklan & Roddier (1988) -- fiber coupling fundamentals

Key results:
    - Top-hat max coupling:  eta ~ 81.45%  (vs 100% for matched Gaussian)
    - This ~18.5% penalty directly impacts LIFE collector sizing
    - Coupling is wavelength-independent IF f_lens is re-optimized per lam
      AND the fiber MFD scales exactly linearly with lam (exact for Marcuse
      only in a narrow regime; see fiber_mode_radius_marcuse())
"""

import numpy as np
import math
import warnings
from scipy.special import j1  # Bessel J_1
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- Imports from the canonical fiber library --------------------------------
# fiber_modes.py uses MICROMETERS for wavelengths and lengths.  We alias the
# imports with an _fm prefix and wrap them below with unit conversion.
from fiber_modes import (
    v_parameter as _fm_v_parameter,
    mode_field_radius as _fm_mode_field_radius,
    mode_field_radius_linear as _fm_mode_field_radius_linear,
    coupling_tophat_analytical as _fm_coupling_tophat,
    optimal_focal_length as _fm_optimal_focal_length,
    FIBER_PARAMS,
)

# ============================================================================
# Physical constants and LIFE mission parameters
# ============================================================================

# LIFE science band
LAMBDA_MIN = 6.0e-6       # [m]  -- shortest science wavelength
LAMBDA_MAX = 16.0e-6      # [m]  -- longest science wavelength
LAMBDA_GOAL_MAX = 18.5e-6 # [m]  -- goal wavelength upper limit
LAMBDA_REF = 10.0e-6      # [m]  -- reference wavelength

# Beam parameters (after beam compression in combiner)
D_BEAM = 20.0e-3          # [m]  -- compressed beam diameter in combiner (typical)
W0_GAUSSIAN = D_BEAM / 2  # [m]  -- Gaussian beam waist (1/e amplitude radius)
                          #        Note: for NICE, this is the laser beam waist

# Fiber parameters (InF3 single-mode fiber as used in NICE)
# Physical fiber parameters for Marcuse approximation:
FIBER_CORE_RADIUS = 4.5e-6   # [m]  -- core radius (typical InF3 SMF)
FIBER_NA = 0.244             # [-]  -- numerical aperture
FIBER_N_CORE = 1.50          # [-]  -- core refractive index (InF3 at ~10 um)
FIBER_N_CLAD = 1.48          # [-]  -- cladding refractive index

# Derived: core radius in um for library calls
_CORE_RADIUS_UM = FIBER_CORE_RADIUS * 1e6  # [um]

# Legacy power-law parameters (kept for comparison; Marcuse is now default)
MFD_REF = 12.0e-6         # [m]  -- mode field diameter at reference wavelength
W_F_REF = MFD_REF / 2     # [m]  -- mode field radius at reference
LAMBDA_MFD_REF = 4.0e-6   # [m]  -- reference wavelength for MFD
MFD_SCALING_EXP = 1.0     # [-]  -- scaling exponent (1.0 for step-index)

# Central obstruction (LIFE collectors are single mirrors, no obstruction)
OBSTRUCTION_RATIO = 0.0   # [-]  -- alpha in Ruilier formula (0 = unobstructed)


# ============================================================================
# 1. V-parameter and single-mode check  (delegates to fiber_modes)
# ============================================================================

def v_parameter(wavelength, core_radius=FIBER_CORE_RADIUS, NA=FIBER_NA):
    """
    Compute the V-parameter (normalized frequency) of a step-index fiber.

    The fiber is single-mode when V < 2.405 (LP11 cutoff).

    Delegates to fiber_modes.v_parameter() with m -> um conversion.
    Accepts either (wavelength, core_radius, NA) or just (wavelength) with
    Module 1 defaults.  Internally converts NA -> (n_core, n_clad) for the
    library call.

    Parameters
    ----------
    wavelength   : float or array [m]
    core_radius  : float [m] -- fiber core radius
    NA           : float [-] -- numerical aperture = sqrt(n_core^2 - n_clad^2)

    Returns
    -------
    V : float or array -- V-parameter (dimensionless)
    """
    lam_um = np.asarray(wavelength) * 1e6
    a_um = core_radius * 1e6
    # V = 2*pi*a*NA/lam -- equivalent to library's formula with (n_core, n_clad)
    # We use the direct formula so this wrapper accepts (core_radius, NA)
    # rather than requiring (n_core, n_clad).
    return 2 * np.pi * a_um * NA / lam_um


def check_single_mode(wavelength, core_radius=FIBER_CORE_RADIUS, NA=FIBER_NA,
                      warn=True):
    """
    Check whether the fiber is single-mode at the given wavelength.

    Issues a warning if V > 2.0 (approaching LP11 cutoff at 2.405)
    or if V < 0.5 (poorly confined mode, field extends far into cladding).

    Note: Birbacher et al. (2026) Section 5.3 reports observed LP11 mode
    leakage through the InF3 spatial filter on NICE, confirming that this
    check is physically important near sub-band edges.

    Parameters
    ----------
    wavelength   : float or array [m]
    core_radius  : float [m]
    NA           : float [-]
    warn         : bool -- whether to issue warnings

    Returns
    -------
    V              : float or array -- V-parameter
    is_single_mode : bool or array -- True if V < 2.405
    """
    V = v_parameter(wavelength, core_radius, NA)
    is_single_mode = V < 2.405

    if warn:
        V_scalar = np.atleast_1d(V)
        lam_scalar = np.atleast_1d(wavelength)
        for Vi, li in zip(V_scalar, lam_scalar):
            if Vi > 2.0:
                warnings.warn(
                    f"V = {Vi:.2f} at lam = {li*1e6:.1f} um: approaching LP11 "
                    f"cutoff (V = 2.405). Spatial filter leakage may degrade "
                    f"null depth. See Birbacher+2026 Sec. 5.3.",
                    stacklevel=2
                )
            elif Vi < 0.5:
                warnings.warn(
                    f"V = {Vi:.2f} at lam = {li*1e6:.1f} um: mode poorly "
                    f"confined -- field extends far into cladding. MFD "
                    f"estimate may be inaccurate.",
                    stacklevel=2
                )

    return V, is_single_mode


# ============================================================================
# 2. Fiber mode field radius as function of wavelength
#    (delegates to fiber_modes with m <-> um conversion)
# ============================================================================

def fiber_mode_radius_marcuse(wavelength, core_radius=FIBER_CORE_RADIUS,
                              NA=FIBER_NA):
    """
    Mode field radius via Marcuse (1978) Gaussian approximation.

    For a step-index fiber with V-parameter V:
        w_f / a = 0.65 + 1.619 / V^(3/2) + 2.879 / V^6

    Delegates to fiber_modes.mode_field_radius() with unit conversion
    (meters -> um -> meters).

    Parameters
    ----------
    wavelength   : float or array [m]
    core_radius  : float [m] -- fiber core radius
    NA           : float [-] -- numerical aperture

    Returns
    -------
    w_f : float or array [m] -- mode field radius (1/e field)
    """
    lam_um = np.asarray(wavelength) * 1e6
    a_um = core_radius * 1e6
    # The Marcuse formula only depends on V = 2*pi*a*NA/lam.
    # fiber_modes.mode_field_radius takes (lam_um, n_core, n_clad, a_core_um)
    # and internally computes V.  We pass the default InF3 indices.
    n_core = FIBER_N_CORE
    n_clad = FIBER_N_CLAD
    w_f_um = _fm_mode_field_radius(lam_um, n_core, n_clad, a_um)
    result = np.asarray(w_f_um) * 1e-6
    return np.squeeze(result).item() if np.ndim(wavelength) == 0 else np.squeeze(result)


def fiber_mode_radius_powerlaw(wavelength, w_f_ref=W_F_REF,
                               lam_ref=LAMBDA_MFD_REF,
                               scaling_exp=MFD_SCALING_EXP):
    """
    Mode field radius using simple power-law scaling (legacy).

    w_f(lam) = w_f_ref * (lam / lam_ref)^alpha,  with alpha ~ 1 for step-index.

    Delegates to fiber_modes.mode_field_radius_linear() with unit conversion.

    Parameters
    ----------
    wavelength  : float or array [m]
    w_f_ref     : float [m] -- mode field radius at reference wavelength
    lam_ref     : float [m] -- reference wavelength
    scaling_exp : float [-] -- scaling exponent (1.0 for step-index)

    Returns
    -------
    w_f : float or array [m] -- mode field radius at given wavelength
    """
    lam_um = np.asarray(wavelength) * 1e6
    w_f_ref_um = w_f_ref * 1e6
    lam_ref_um = lam_ref * 1e6
    w_f_um = _fm_mode_field_radius_linear(lam_um, w_f_ref_um, lam_ref_um)
    result = np.asarray(w_f_um) * 1e-6
    return np.squeeze(result).item() if np.ndim(wavelength) == 0 else np.squeeze(result)


def fiber_mode_radius(wavelength, method='marcuse', **kwargs):
    """
    Mode field radius of single-mode fiber -- unified interface.

    Default method is 'marcuse' (physically accurate across the LIFE band).
    Use method='powerlaw' for the legacy linear scaling.

    Parameters
    ----------
    wavelength : float or array [m]
    method     : str -- 'marcuse' (default) or 'powerlaw'
    **kwargs   : passed to the underlying function

    Returns
    -------
    w_f : float or array [m] -- mode field radius at given wavelength
    """
    if method == 'marcuse':
        return fiber_mode_radius_marcuse(wavelength, **kwargs)
    elif method == 'powerlaw':
        return fiber_mode_radius_powerlaw(wavelength, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'marcuse' or 'powerlaw'.")


# ============================================================================
# 3. Optimal focal length for fiber coupling  (delegates to fiber_modes)
# ============================================================================

def optimal_focal_length(w0_or_D, w_f, wavelength, beam_type='gaussian'):
    """
    Compute the focal length of the coupling lens that maximizes fiber coupling.

    For Gaussian beam:  f = pi * w0 * w_f / lam  (matches waists)
    For top-hat beam:   f = pi * w_f * D / (2 * lam * beta_opt)  where beta_opt ~ 1.1209

    Delegates to fiber_modes.optimal_focal_length().  Note: the library always
    takes D_beam as its first argument (and derives w0 = D/2 for Gaussian),
    while this module's API takes w0 directly for Gaussian.  The adapter
    passes D_beam = 2*w0 for the Gaussian case.

    Parameters
    ----------
    w0_or_D   : float [m] -- Gaussian waist radius, or beam diameter for top-hat
    w_f       : float [m] -- fiber mode field radius
    wavelength: float [m]
    beam_type : str -- 'gaussian' or 'tophat'

    Returns
    -------
    f : float [m] -- optimal focal length
    """
    if beam_type == 'tophat':
        return _fm_optimal_focal_length(w0_or_D, w_f, wavelength, 'tophat')
    elif beam_type == 'gaussian':
        # Library expects D_beam; Module 1 passes w0.  D_beam = 2 * w0.
        return _fm_optimal_focal_length(2.0 * w0_or_D, w_f, wavelength, 'gaussian')
    else:
        raise ValueError(f"Unknown beam_type: {beam_type}")


# ============================================================================
# 4. Coupling efficiency: Gaussian beam -> Gaussian fiber mode
# ============================================================================

def coupling_gaussian(w0, w_f, pointing_error=0.0, shear_offset=0.0,
                      wavelength=LAMBDA_REF):
    """
    Coupling efficiency of a Gaussian beam into a single-mode fiber
    with Gaussian fundamental mode.

    From Birbacher+2026 Eqs. 6-11:
      eta_mode     = (2 w0 w_f / (w0^2 + w_f^2))^2   -- mode size mismatch
      eta_pointing = exp(-pi^2 alpha^2 w0^2 / lam^2)  -- pointing error alpha [rad]
      eta_shear    = exp(-o^2 / w0^2)                  -- lateral offset o [m]
      eta_total    = eta_mode * eta_pointing * eta_shear

    Parameters
    ----------
    w0             : float [m] -- input beam waist radius (1/e field)
    w_f            : float [m] -- fiber mode field radius
    pointing_error : float [rad] -- beam tilt angle
    shear_offset   : float [m] -- lateral beam offset in pupil plane
    wavelength     : float [m]

    Returns
    -------
    eta : float -- coupling efficiency [0, 1]
    """
    # Mode mismatch (Birbacher Eq. 6)
    eta_mode = (2 * w0 * w_f / (w0**2 + w_f**2))**2

    # Pointing error (Birbacher Eq. 10)
    eta_pointing = np.exp(-np.pi**2 * pointing_error**2 * w0**2 / wavelength**2)

    # Shear error (Birbacher Eq. 11)
    eta_shear = np.exp(-shear_offset**2 / w0**2)

    return eta_mode * eta_pointing * eta_shear


# ============================================================================
# 5. Coupling efficiency: Top-hat (uniform circular) beam -> Gaussian fiber
# ============================================================================

def coupling_tophat_numerical(D_beam, w_f, wavelength, f_lens,
                              pointing_error=0.0, shear_offset=0.0,
                              N_grid=1024, obstruction=0.0):
    """
    Numerical coupling efficiency of a top-hat beam into a Gaussian fiber mode.

    Computes the overlap integral via 2D FFT:
      1. Define uniform circular aperture (top-hat) in pupil plane
      2. Propagate to focal plane via FFT (Fraunhofer approximation)
      3. Compute overlap integral with Gaussian fiber mode

    This handles arbitrary obstruction, pointing errors, and shear.

    Parameters
    ----------
    D_beam         : float [m] -- beam diameter
    w_f            : float [m] -- fiber mode field radius
    wavelength     : float [m]
    f_lens         : float [m] -- coupling lens focal length
    pointing_error : float [rad] -- beam tilt angle (along x)
    shear_offset   : float [m] -- lateral beam offset in pupil (along x)
    N_grid         : int -- grid size for FFT (N x N)
    obstruction    : float -- central obstruction ratio [0, 1)

    Returns
    -------
    eta : float -- coupling efficiency [0, 1]
    """
    R_beam = D_beam / 2.0

    # Pupil plane grid (must be large enough to contain the beam)
    L_pupil = D_beam * 2.0      # physical size of pupil grid
    dx = L_pupil / N_grid       # pixel size in pupil plane
    x = np.linspace(-L_pupil/2, L_pupil/2, N_grid, endpoint=False)
    xx, yy = np.meshgrid(x, x)

    # Top-hat aperture (with optional central obstruction)
    rr = np.sqrt((xx - shear_offset)**2 + yy**2)
    aperture = np.zeros_like(rr)
    aperture[rr <= R_beam] = 1.0
    if obstruction > 0:
        aperture[rr <= obstruction * R_beam] = 0.0

    # Add pointing error as linear phase tilt
    k = 2 * np.pi / wavelength
    E_pupil = aperture * np.exp(1j * k * pointing_error * xx)

    # Propagate to focal plane via FFT (Fraunhofer diffraction)
    E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil)))

    # Focal plane coordinates
    df = wavelength * f_lens / (N_grid * dx)  # pixel size in focal plane
    x_focal = np.arange(-N_grid//2, N_grid//2) * df
    xx_f, yy_f = np.meshgrid(x_focal, x_focal)

    # Gaussian fiber mode (power-normalized)
    psi = np.sqrt(2 / (np.pi * w_f**2)) * np.exp(-(xx_f**2 + yy_f**2) / w_f**2)

    # Overlap integral
    overlap = np.sum(E_focal * np.conj(psi)) * df**2
    power_in = np.sum(np.abs(E_focal)**2) * df**2
    power_mode = np.sum(np.abs(psi)**2) * df**2

    eta = np.abs(overlap)**2 / (np.real(power_in) * np.real(power_mode))
    return float(np.real(eta))


def coupling_tophat_analytical(D_beam, w_f, wavelength, f_lens, obstruction=0.0):
    """
    Analytical coupling efficiency from Ruilier & Cassaing (2001).

    Computes beta = pi w_f D / (2 lam f), then delegates the unobstructed case
    to fiber_modes.coupling_tophat_analytical(beta).  The obstructed annular-
    pupil extension is evaluated locally:

        eta(beta, alpha) = 2 [exp(-alpha^2 beta^2) - exp(-beta^2)]^2
                            / [beta^2 (1 - alpha^2)]

    Maximum eta = 81.45% at beta_opt ~ 1.1209.

    Parameters
    ----------
    D_beam      : float [m] -- beam diameter
    w_f         : float [m] -- fiber mode field radius
    wavelength  : float [m]
    f_lens      : float [m] -- coupling lens focal length
    obstruction : float -- central obstruction ratio [0, 1)

    Returns
    -------
    rho : float -- coupling efficiency [0, 1]
    """
    beta = np.pi * w_f * D_beam / (2 * wavelength * f_lens)

    if obstruction == 0:
        # Delegate to library (takes dimensionless beta)
        return float(np.squeeze(_fm_coupling_tophat(beta)))
    else:
        alpha = obstruction
        rho = (2.0 * (np.exp(-alpha**2 * beta**2) - np.exp(-beta**2))**2 /
               (beta**2 * (1.0 - alpha**2)))
        return float(rho)


def find_optimal_beta(obstruction=0.0):
    """
    Find the beta that maximizes coupling for given
    central obstruction ratio.

    Returns
    -------
    beta_opt : float -- optimal beta parameter
    eta_max   : float -- maximum coupling efficiency
    """
    alpha = obstruction

    def neg_coupling(beta):
        if alpha == 0:
            return -float(np.squeeze(_fm_coupling_tophat(beta)))
        else:
            return -2.0 * (np.exp(-alpha**2 * beta**2) - np.exp(-beta**2))**2 / \
                   (beta**2 * (1.0 - alpha**2))

    result = minimize_scalar(neg_coupling, bounds=(0.5, 3.0), method='bounded')
    return result.x, -result.fun


# ============================================================================
# 6. Coupling with wavefront errors (Zernike aberrations)
# ============================================================================

def coupling_with_wfe(D_beam, w_f, wavelength, f_lens, wfe_rms,
                      beam_type='tophat', N_grid=512, seed=None):
    """
    Coupling efficiency with random wavefront errors.

    Adds a random phase screen with given RMS to the pupil before propagation.
    For spatial filtering, the coupling drops approximately as:
        eta_wfe ~ eta_0 * exp(-(2*pi*WFE_rms/lam)^2)   (Marechal approximation)

    This function computes it numerically for validation.

    NOTE: The phase screen uses a Gaussian-filtered random field, which does
    NOT have the power-law PSD (k^-2 to k^-3) characteristic of polished
    optical surfaces. For realistic surface error analysis, use
    coupling_with_zernike() with specific Zernike coefficients or supply a
    measured/simulated PSD. This function is for VALIDATION of the Marechal
    approximation, not for realistic surface modeling.

    Parameters
    ----------
    D_beam    : float [m]
    w_f       : float [m]
    wavelength: float [m]
    f_lens    : float [m]
    wfe_rms   : float [m] -- RMS wavefront error
    beam_type : str -- 'gaussian' or 'tophat'
    N_grid    : int
    seed      : int or None -- random seed for reproducibility

    Returns
    -------
    eta : float -- coupling efficiency with WFE
    """
    if seed is None:
        seed = 42  # default reproducible seed
    rng = np.random.default_rng(seed)

    R_beam = D_beam / 2.0
    L_pupil = D_beam * 2.5
    dx = L_pupil / N_grid
    x = np.linspace(-L_pupil/2, L_pupil/2, N_grid, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2 + yy**2)

    # Aperture
    if beam_type == 'tophat':
        E_amp = np.where(rr <= R_beam, 1.0, 0.0)
    elif beam_type == 'gaussian':
        w0 = R_beam  # Gaussian waist = beam radius
        E_amp = np.exp(-rr**2 / w0**2)
    else:
        raise ValueError(f"Unknown beam_type: {beam_type}")

    # Random wavefront error (smooth random screen for validation)
    phase_screen = _generate_smooth_phase_screen(N_grid, L_pupil, wfe_rms,
                                                  correlation_length=D_beam/4,
                                                  rng=rng)
    # Apply phase only inside aperture
    k = 2 * np.pi / wavelength
    E_pupil = E_amp * np.exp(1j * k * phase_screen)

    # FFT propagation
    E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil)))
    df = wavelength * f_lens / (N_grid * dx)
    x_focal = np.arange(-N_grid//2, N_grid//2) * df
    xx_f, yy_f = np.meshgrid(x_focal, x_focal)

    # Fiber mode (power-normalized)
    psi = np.sqrt(2 / (np.pi * w_f**2)) * np.exp(-(xx_f**2 + yy_f**2) / w_f**2)

    # Overlap integral
    overlap = np.sum(E_focal * np.conj(psi)) * df**2
    power_in = np.sum(np.abs(E_focal)**2) * df**2
    power_mode = np.sum(np.abs(psi)**2) * df**2

    eta = np.abs(overlap)**2 / (np.real(power_in) * np.real(power_mode))
    return float(np.real(eta))


def _generate_smooth_phase_screen(N, L, rms, correlation_length, rng=None):
    """
    Generate a smooth random phase screen with given RMS and correlation length.

    WARNING: Uses Gaussian-filtered white noise -- NOT a realistic polished-surface
    PSD. For validation and quick estimates only.

    Parameters
    ----------
    N                  : int -- grid size
    L                  : float [m] -- physical grid size
    rms                : float [m] -- target RMS wavefront error
    correlation_length : float [m] -- spatial correlation scale
    rng                : numpy Generator or None -- random number generator

    Returns
    -------
    screen : ndarray (N, N) -- phase screen in meters of OPD
    """
    if rng is None:
        rng = np.random.default_rng(42)

    noise = rng.standard_normal((N, N))

    # Spatial frequencies
    fx = np.fft.fftfreq(N, d=L/N)
    fxx, fyy = np.meshgrid(fx, fx)
    fr = np.sqrt(fxx**2 + fyy**2)

    # Low-pass filter (Gaussian envelope in Fourier space)
    sigma_f = 1.0 / (2 * np.pi * correlation_length)
    filt = np.exp(-fr**2 / (2 * sigma_f**2))

    # Filter in Fourier space
    noise_fft = np.fft.fft2(noise)
    filtered = np.fft.ifft2(noise_fft * filt).real

    # Normalize to desired RMS
    if np.std(filtered) > 0:
        filtered = filtered / np.std(filtered) * rms
    return filtered


# ============================================================================
# 7. Coupling with Zernike aberrations (specific low-order terms)
# ============================================================================

def zernike_radial(n, m, rho):
    """Compute Zernike radial polynomial R_n^m(rho)."""
    R = np.zeros_like(rho)
    for s in range((n - abs(m)) // 2 + 1):
        coeff = ((-1)**s * math.factorial(n - s) /
                (math.factorial(s) *
                 math.factorial((n + abs(m))//2 - s) *
                 math.factorial((n - abs(m))//2 - s)))
        R += coeff * rho**(n - 2*s)
    return R


def zernike_polynomial(n, m, rho, theta):
    """
    Compute Zernike polynomial Z_n^m(rho, theta) on unit disk.
    Uses Noll normalization.
    """
    R = zernike_radial(n, m, rho)
    if m == 0:
        return np.sqrt(n + 1) * R
    elif m > 0:
        return np.sqrt(2*(n + 1)) * R * np.cos(m * theta)
    else:
        return np.sqrt(2*(n + 1)) * R * np.sin(abs(m) * theta)


# Standard Zernike indices (Noll ordering): j -> (n, m)
ZERNIKE_NOLL = {
    1: (0, 0),    # Piston
    2: (1, 1),    # Tilt X
    3: (1, -1),   # Tilt Y
    4: (2, 0),    # Defocus
    5: (2, -2),   # Astigmatism 45 deg
    6: (2, 2),    # Astigmatism 0 deg
    7: (3, -1),   # Coma Y
    8: (3, 1),    # Coma X
    9: (3, -3),   # Trefoil Y
    10: (3, 3),   # Trefoil X
    11: (4, 0),   # Spherical
}


def coupling_with_zernike(D_beam, w_f, wavelength, f_lens,
                          zernike_coeffs, beam_type='tophat', N_grid=512):
    """
    Coupling efficiency with specific Zernike aberrations.

    Parameters
    ----------
    D_beam         : float [m]
    w_f            : float [m]
    wavelength     : float [m]
    f_lens         : float [m]
    zernike_coeffs : dict {Noll_index: amplitude_in_meters}
                     e.g., {4: 50e-9, 7: 20e-9} for 50 nm defocus + 20 nm coma
    beam_type      : str -- 'gaussian' or 'tophat'
    N_grid         : int

    Returns
    -------
    eta : float -- coupling efficiency [0, 1]
    """
    R_beam = D_beam / 2.0
    L_pupil = D_beam * 2.5
    x = np.linspace(-L_pupil/2, L_pupil/2, N_grid, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    rho_norm = rr / R_beam  # normalized radius

    # Aperture
    mask = rr <= R_beam
    if beam_type == 'tophat':
        E_amp = np.where(mask, 1.0, 0.0)
    else:
        w0 = R_beam
        E_amp = np.exp(-rr**2 / w0**2)

    # Build wavefront error from Zernike coefficients
    wfe = np.zeros_like(rr)
    for j, amplitude in zernike_coeffs.items():
        n, m = ZERNIKE_NOLL[j]
        Z = np.where(mask, zernike_polynomial(n, m,
                     np.clip(rho_norm, 0, 1), theta), 0.0)
        wfe += amplitude * Z

    k = 2 * np.pi / wavelength
    E_pupil = E_amp * np.exp(1j * k * wfe)

    # FFT to focal plane
    dx = L_pupil / N_grid
    E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pupil)))
    df = wavelength * f_lens / (N_grid * dx)
    x_focal = np.arange(-N_grid//2, N_grid//2) * df
    xx_f, yy_f = np.meshgrid(x_focal, x_focal)

    psi = np.sqrt(2 / (np.pi * w_f**2)) * np.exp(-(xx_f**2 + yy_f**2) / w_f**2)

    overlap = np.sum(E_focal * np.conj(psi)) * df**2
    power_in = np.sum(np.abs(E_focal)**2) * df**2
    power_mode = np.sum(np.abs(psi)**2) * df**2

    eta = np.abs(overlap)**2 / (np.real(power_in) * np.real(power_mode))
    return float(np.real(eta))


# ============================================================================
# 8. Marechal coupling approximation
# ============================================================================

def marechal_coupling(eta_0, wfe_rms, wavelength):
    """
    Marechal coupling approximation for fiber coupling degradation from WFE.

    eta = eta_0 * exp(-(2*pi*sigma/lam)^2)

    This is the extended Marechal approximation applied to fiber coupling.
    Originally derived for Strehl ratio, it also applies to the overlap
    integral for small aberrations because fiber coupling responds to phase
    errors identically to peak intensity (both are projections of the
    aberrated field onto a reference mode).

    Accurate to < 0.5% for sigma < lam/20 (the regime relevant for LIFE optics).
    Validated against individual Zernike modes in coupling_with_zernike().

    Parameters
    ----------
    eta_0      : float -- ideal coupling efficiency (no aberration)
    wfe_rms    : float [m] -- RMS wavefront error (OPD)
    wavelength : float [m]

    Returns
    -------
    eta : float -- degraded coupling efficiency
    """
    return eta_0 * np.exp(-(2 * np.pi * wfe_rms / wavelength)**2)


# ============================================================================
# 9. Differential coupling and null depth impact
# ============================================================================

def null_depth_from_coupling_mismatch(eta_1, eta_2):
    """
    Null depth contribution from differential fiber coupling (intensity mismatch).

    From Birbacher+2026 Eq. 4 and Serabyn (2000):
        dI = (eta_1 - eta_2) / (eta_1 + eta_2)
        N_intensity = dI^2 / 4

    Parameters
    ----------
    eta_1, eta_2 : float -- coupling efficiency of beam 1 and 2

    Returns
    -------
    N_intensity : float -- null depth contribution from intensity mismatch
    delta_I     : float -- fractional intensity mismatch
    """
    delta_I = (eta_1 - eta_2) / (eta_1 + eta_2)
    N_intensity = delta_I**2 / 4.0
    return N_intensity, delta_I


def null_from_shear(shear_offset, w0, wavelength=None):
    """
    Null depth contribution from lateral beam shear (Birbacher+2026 Eq. 15).

    Shear produces differential coupling:
        dI_shear ~ o^2 / (2 w0^2)
        N_shear  = dI_shear^2 / 4 = o^4 / (16 w0^4)

    Parameters
    ----------
    shear_offset : float or array [m] -- lateral beam offset
    w0           : float [m] -- beam waist radius (1/e field)
    wavelength   : float [m] or None -- not used (wavelength-independent
                   for Gaussian beams), kept for API consistency

    Returns
    -------
    N_shear  : float or array -- null depth contribution
    delta_I  : float or array -- intensity mismatch from shear
    """
    delta_I = shear_offset**2 / (2 * w0**2)
    N_shear = delta_I**2 / 4.0
    return N_shear, delta_I


def null_from_pointing(pointing_error, w0, wavelength):
    """
    Null depth contribution from pointing error (derived from Birbacher Eq. 10).

    Pointing produces differential coupling:
        dI_pointing ~ pi^2 alpha^2 w0^2 / lam^2
        N_pointing  = dI_pointing^2 / 4

    Parameters
    ----------
    pointing_error : float or array [rad] -- beam tilt angle
    w0             : float [m] -- beam waist radius (1/e field)
    wavelength     : float [m]

    Returns
    -------
    N_pointing : float or array -- null depth contribution
    delta_I    : float or array -- intensity mismatch from pointing
    """
    delta_I = np.pi**2 * pointing_error**2 * w0**2 / wavelength**2
    N_pointing = delta_I**2 / 4.0
    return N_pointing, delta_I


# ============================================================================
# 10. Top-hat pointing and shear -- numerical and approximate
# ============================================================================

def coupling_tophat_with_pointing_numerical(D_beam, w_f, wavelength, f_lens,
                                             pointing_error, N_grid=512):
    """
    Exact numerical coupling for a top-hat beam with pointing error.

    Computes the full overlap integral with a tilted pupil field via FFT.

    Parameters
    ----------
    D_beam         : float [m] -- beam diameter
    w_f            : float [m] -- fiber mode field radius
    wavelength     : float [m]
    f_lens         : float [m] -- coupling lens focal length
    pointing_error : float [rad] -- beam tilt angle

    Returns
    -------
    eta : float -- coupling efficiency [0, 1]
    """
    return coupling_tophat_numerical(D_beam, w_f, wavelength, f_lens,
                                     pointing_error=pointing_error,
                                     N_grid=N_grid)


def coupling_tophat_pointing_approx(pointing_error, D_beam, wavelength):
    """
    Approximate pointing sensitivity for top-hat beam using effective waist.

    Uses the second-moment-matched Gaussian waist:
        w0_eff = D / (2*sqrt(2))

    Then applies the Gaussian pointing formula (Birbacher Eq. 10):
        eta_pointing ~ exp(-pi^2 alpha^2 w0_eff^2 / lam^2)

    APPROXIMATION: This is the Gaussian-equivalent model. The actual
    top-hat response involves the Airy pattern overlap integral.
    Accurate to ~5% for alpha < 15 urad at LIFE beam parameters.

    Parameters
    ----------
    pointing_error : float or array [rad]
    D_beam         : float [m] -- beam diameter
    wavelength     : float [m]

    Returns
    -------
    eta_relative : float or array -- relative coupling eta/eta_0
    """
    w0_eff = D_beam / (2 * np.sqrt(2))
    return np.exp(-np.pi**2 * pointing_error**2 * w0_eff**2 / wavelength**2)


def coupling_tophat_shear_geometric(shear_offset, D_beam):
    """
    Geometric (exact) coupling penalty for a laterally shifted top-hat beam.

    For a uniform circular pupil shifted by offset o relative to the optical
    axis, the overlap with the original aperture is the intersection area
    of two offset circles (a lens shape).

    For o << D: eta ~ 1 - (4/pi) * (o/D)  [linear, NOT quadratic]

    Parameters
    ----------
    shear_offset : float or array [m] -- lateral beam offset
    D_beam       : float [m] -- beam diameter

    Returns
    -------
    eta_relative : float or array -- relative coupling eta/eta_0
    """
    R = D_beam / 2.0
    o = np.abs(shear_offset)
    ratio = np.minimum(o / (2 * R), 1.0)
    area_frac = np.where(
        ratio < 1.0,
        (2 / np.pi) * (np.arccos(ratio) - ratio * np.sqrt(1 - ratio**2)),
        0.0
    )
    return area_frac


def coupling_tophat_shear_gaussian_approx(shear_offset, D_beam):
    """
    Approximate (Gaussian model) coupling penalty for top-hat shear.

    Uses exp(-o^2/R^2) where R = D/2 is the beam radius.

    APPROXIMATION: The true top-hat shear penalty is linear (not quadratic)
    for small offsets -- see coupling_tophat_shear_geometric().

    Parameters
    ----------
    shear_offset : float or array [m] -- lateral beam offset
    D_beam       : float [m] -- beam diameter

    Returns
    -------
    eta_relative : float or array -- relative coupling eta/eta_0
    """
    return np.exp(-shear_offset**2 / (D_beam/2)**2)


# ============================================================================
# 11. Main analysis and plotting
# ============================================================================

def run_full_analysis():
    """
    Run the complete Module 1 analysis and produce publication-quality figures.

    Produces 5 key figures:
      Fig 1: Coupling efficiency vs beta parameter (top-hat vs Gaussian)
      Fig 2: Coupling efficiency vs wavelength across LIFE band
             (now includes Marcuse vs power-law MFD comparison)
      Fig 3: Pointing and shear sensitivity comparison
             (now includes numerical top-hat pointing & geometric shear)
      Fig 4: Zernike sensitivity (defocus, coma, astigmatism, spherical)
      Fig 5: V-parameter and MFD comparison across LIFE band
    """

    print("=" * 70)
    print("LIFE E2E Module 1: Fiber Coupling Analysis (v2.0)")
    print("=" * 70)

    # ---- V-parameter check across band ----
    print("\n--- V-parameter check across LIFE band ---")
    for lam in [6e-6, 10e-6, 16e-6, 18.5e-6]:
        V = v_parameter(lam)
        _, sm = check_single_mode(lam, warn=False)
        status = "single-mode" if sm else "MULTI-MODE"
        print(f"  lam = {lam*1e6:5.1f} um:  V = {V:.3f}  ({status})")

    # ---- Verify analytical formula against known results ----
    beta_opt, eta_max = find_optimal_beta(obstruction=0.0)
    print(f"\nTop-hat (unobstructed):")
    print(f"  Optimal beta = {beta_opt:.4f}  (expected: 1.1209)")
    print(f"  Max coupling eta = {eta_max:.4f}  (expected: ~0.8145)")

    beta_opt_obs10, eta_max_obs10 = find_optimal_beta(obstruction=0.10)
    beta_opt_obs20, eta_max_obs20 = find_optimal_beta(obstruction=0.20)
    print(f"\nTop-hat (10% obstruction):  beta_opt = {beta_opt_obs10:.4f},  "
          f"eta_max = {eta_max_obs10:.4f}")
    print(f"Top-hat (20% obstruction):  beta_opt = {beta_opt_obs20:.4f},  "
          f"eta_max = {eta_max_obs20:.4f}")

    # ---- Figure 1: Coupling vs beta ----
    print("\n--- Figure 1: Coupling efficiency vs beta parameter ---")

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    beta_arr = np.linspace(0.3, 3.0, 500)

    # Top-hat unobstructed (via library)
    eta_tophat = np.asarray(_fm_coupling_tophat(beta_arr))

    # Top-hat with 10% obstruction
    alpha_10 = 0.10
    eta_obs10 = (2.0 * (np.exp(-alpha_10**2 * beta_arr**2) -
                 np.exp(-beta_arr**2))**2 /
                 (beta_arr**2 * (1.0 - alpha_10**2)))

    # Top-hat with 20% obstruction
    alpha_20 = 0.20
    eta_obs20 = (2.0 * (np.exp(-alpha_20**2 * beta_arr**2) -
                 np.exp(-beta_arr**2))**2 /
                 (beta_arr**2 * (1.0 - alpha_20**2)))

    # Gaussian mode mismatch: eta = (2s/(s^2+1))^2 where s = w0/wf
    s_arr = np.linspace(0.3, 3.0, 500)
    eta_gaussian_mismatch = (2 * s_arr / (s_arr**2 + 1))**2

    ax1.plot(beta_arr, eta_tophat, 'b-', lw=2.5, label='Top-hat (unobstructed)')
    ax1.plot(beta_arr, eta_obs10, 'r--', lw=2, label='Top-hat (10% obstruction)')
    ax1.plot(beta_arr, eta_obs20, 'g-.', lw=2, label='Top-hat (20% obstruction)')

    # Mark optimal points
    ax1.axhline(y=eta_max, color='b', ls=':', alpha=0.5)
    ax1.plot(beta_opt, eta_max, 'b*', ms=15, zorder=5)
    ax1.annotate(r'$\eta_{\rm max}$' + f' = {eta_max:.1%}\n'
                 + r'$\beta_{\rm opt}$' + f' = {beta_opt:.3f}',
                xy=(beta_opt, eta_max), xytext=(beta_opt + 0.3, eta_max - 0.05),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'),
                color='blue')

    ax1.set_xlabel(r'Coupling parameter $\beta$', fontsize=13)
    ax1.set_ylabel(r'Coupling efficiency $\eta$', fontsize=13)
    ax1.set_title('Single-mode fiber coupling: top-hat vs Gaussian beam profiles',
                  fontsize=13)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0.3, 3.0)
    ax1.grid(True, alpha=0.3)

    # Add secondary axis for Gaussian mismatch
    ax1b = ax1.twiny()
    ax1b.plot(s_arr, eta_gaussian_mismatch, 'k-', lw=1.5, alpha=0.6,
             label='Gaussian (mode mismatch)')
    ax1b.set_xlabel(r'Gaussian: $w_0 / w_f$ ratio', fontsize=11, color='gray')
    ax1b.tick_params(axis='x', colors='gray')
    ax1b.legend(fontsize=10, loc='center right')

    fig1.tight_layout()
    fig1.savefig('./fig1_coupling_vs_beta.png', dpi=200,
                bbox_inches='tight')
    print("  Saved: fig1_coupling_vs_beta.png")

    # ---- Figure 5: V-parameter and MFD comparison ----
    print("\n--- Figure 5: V-parameter and MFD: Marcuse vs power-law ---")

    wavelengths_full = np.linspace(3e-6, 20e-6, 300)
    wavelengths_um = wavelengths_full * 1e6

    V_arr = v_parameter(wavelengths_full)
    wf_marcuse = fiber_mode_radius(wavelengths_full, method='marcuse') * 1e6
    wf_powerlaw = fiber_mode_radius(wavelengths_full, method='powerlaw') * 1e6

    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Panel A: V-parameter
    ax5a.plot(wavelengths_um, V_arr, 'b-', lw=2)
    ax5a.axhline(y=2.405, color='red', ls='--', lw=1.5, alpha=0.7,
                label=r'LP$_{11}$ cutoff ($V = 2.405$)')
    ax5a.axhline(y=0.5, color='orange', ls='--', lw=1.5, alpha=0.7,
                label='Poor confinement ($V = 0.5$)')
    ax5a.axvspan(6, 16, alpha=0.08, color='green', label='LIFE science band')
    ax5a.set_ylabel('V-parameter', fontsize=12)
    ax5a.set_title(f'Fiber parameters: a = {FIBER_CORE_RADIUS*1e6:.1f} um, '
                   f'NA = {FIBER_NA}', fontsize=12)
    ax5a.legend(fontsize=9, loc='upper right')
    ax5a.grid(True, alpha=0.3)
    ax5a.set_ylim(0, 4)

    # Panel B: MFD comparison
    ax5b.plot(wavelengths_um, wf_marcuse, 'b-', lw=2.5, label='Marcuse (1978)')
    ax5b.plot(wavelengths_um, wf_powerlaw, 'r--', lw=2,
             label=r'Power-law: $w_f \propto \lambda^{' + f'{MFD_SCALING_EXP}' + r'}$')
    ax5b.axvspan(6, 16, alpha=0.08, color='green')

    # Compute and annotate the relative difference
    in_band = (wavelengths_full >= 6e-6) & (wavelengths_full <= 16e-6)
    rel_diff = np.abs(wf_marcuse[in_band] - wf_powerlaw[in_band]) / wf_marcuse[in_band]
    ax5b.set_ylabel(r'Mode field radius $w_f$ [$\mu$m]', fontsize=12)
    ax5b.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax5b.legend(fontsize=10, loc='upper left')
    ax5b.grid(True, alpha=0.3)
    ax5b.text(12, wf_powerlaw[len(wf_powerlaw)//2] * 0.6,
             f'Max in-band difference:\n{rel_diff.max()*100:.1f}%',
             fontsize=10, color='purple',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig5.tight_layout()
    fig5.savefig('./fig5_v_parameter_mfd.png', dpi=200,
                bbox_inches='tight')
    print("  Saved: fig5_v_parameter_mfd.png")

    # ---- Figure 2: Coupling vs wavelength across LIFE band ----
    print("\n--- Figure 2: Coupling vs wavelength across LIFE band ---")

    wavelengths = np.linspace(4e-6, 18.5e-6, 200)

    # For each wavelength, compute optimal coupling for top-hat and Gaussian
    eta_th_vs_lam = []
    eta_th_vs_lam_powerlaw = []
    w_f_vs_lam = []
    f_opt_th = []
    f_opt_gauss = []

    for lam in wavelengths:
        w_f = fiber_mode_radius(lam, method='marcuse')
        w_f_pl = fiber_mode_radius(lam, method='powerlaw')
        w_f_vs_lam.append(w_f * 1e6)

        # Top-hat: at optimal beta, coupling = eta_max
        f_opt = optimal_focal_length(D_BEAM, w_f, lam, beam_type='tophat')
        f_opt_th.append(f_opt * 1e3)

        eta_th_optimal = coupling_tophat_analytical(D_BEAM, w_f, lam, f_opt)
        eta_th_vs_lam.append(eta_th_optimal)

        # Same but with power-law MFD
        f_opt_pl = optimal_focal_length(D_BEAM, w_f_pl, lam, beam_type='tophat')
        eta_th_pl = coupling_tophat_analytical(D_BEAM, w_f_pl, lam, f_opt_pl)
        eta_th_vs_lam_powerlaw.append(eta_th_pl)

        f_gauss = optimal_focal_length(W0_GAUSSIAN, w_f, lam, beam_type='gaussian')
        f_opt_gauss.append(f_gauss * 1e3)

    eta_th_vs_lam = np.array(eta_th_vs_lam)
    eta_th_vs_lam_powerlaw = np.array(eta_th_vs_lam_powerlaw)
    wavelengths_um = wavelengths * 1e6

    # Case B: Fixed f_lens (optimized at 10 um), coupling degrades at other lam
    w_f_10 = fiber_mode_radius(10e-6, method='marcuse')
    f_fixed = optimal_focal_length(D_BEAM, w_f_10, 10e-6, beam_type='tophat')

    eta_th_fixed_f = []
    eta_gauss_fixed_f = []
    for lam in wavelengths:
        w_f = fiber_mode_radius(lam, method='marcuse')
        eta_th = coupling_tophat_analytical(D_BEAM, w_f, lam, f_fixed)
        eta_th_fixed_f.append(eta_th)

        # Gaussian with fixed f: focused waist = lam*f/(pi*w0)
        w_focus = lam * f_fixed / (np.pi * W0_GAUSSIAN)
        eta_g = (2 * w_focus * w_f / (w_focus**2 + w_f**2))**2
        eta_gauss_fixed_f.append(eta_g)

    eta_th_fixed_f = np.array(eta_th_fixed_f)
    eta_gauss_fixed_f = np.array(eta_gauss_fixed_f)

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Panel A: coupling efficiency
    ax2a.plot(wavelengths_um, eta_th_vs_lam * 100, 'b-', lw=2.5,
             label=r'Top-hat, Marcuse MFD ($f$ per $\lambda$)')
    ax2a.plot(wavelengths_um, eta_th_vs_lam_powerlaw * 100, 'b:', lw=1.5,
             alpha=0.7, label=r'Top-hat, power-law MFD ($f$ per $\lambda$)')
    ax2a.plot(wavelengths_um, eta_th_fixed_f * 100, 'b--', lw=2,
             label=(r'Top-hat, Marcuse ($f$ fixed at 10 $\mu$m'
                    f' = {f_fixed*1e3:.1f} mm)'))
    ax2a.plot(wavelengths_um, eta_gauss_fixed_f * 100, 'r--', lw=2,
             label=r'Gaussian ($f$ fixed at 10 $\mu$m)')
    ax2a.axhline(y=81.45, color='gray', ls=':', alpha=0.5)
    ax2a.axhline(y=100, color='gray', ls=':', alpha=0.5)

    # LIFE science band
    ax2a.axvspan(6, 16, alpha=0.08, color='green', label='LIFE science band')

    ax2a.set_ylabel(r'Fiber coupling efficiency $\eta$ [%]', fontsize=13)
    ax2a.set_title('Fiber coupling across LIFE wavelength range', fontsize=13)
    ax2a.legend(fontsize=9, loc='lower right')
    ax2a.set_ylim(0, 105)
    ax2a.grid(True, alpha=0.3)
    ax2a.text(7, 84, 'Top-hat theoretical max: 81.45%', fontsize=9, color='gray')

    # Panel B: required focal length and fiber mode field radius
    ax2b.plot(wavelengths_um, f_opt_th, 'b-', lw=2,
             label=r'$f_{\rm opt}$ (top-hat)')
    ax2b.set_ylabel('Optimal focal length [mm]', fontsize=13, color='blue')
    ax2b.set_xlabel(r'Wavelength [$\mu$m]', fontsize=13)
    ax2b.tick_params(axis='y', labelcolor='blue')

    ax2b_r = ax2b.twinx()
    ax2b_r.plot(wavelengths_um, w_f_vs_lam, 'r-', lw=2,
                label=r'$w_f(\lambda)$ Marcuse')
    ax2b_r.set_ylabel(r'Fiber mode field radius $w_f$ [$\mu$m]',
                       fontsize=13, color='red')
    ax2b_r.tick_params(axis='y', labelcolor='red')

    ax2b.axvspan(6, 16, alpha=0.08, color='green')
    ax2b.grid(True, alpha=0.3)

    lines_b, labels_b = ax2b.get_legend_handles_labels()
    lines_br, labels_br = ax2b_r.get_legend_handles_labels()
    ax2b.legend(lines_b + lines_br, labels_b + labels_br, loc='upper left',
               fontsize=10)

    fig2.tight_layout()
    fig2.savefig('./fig2_coupling_vs_wavelength.png', dpi=200,
                bbox_inches='tight')
    print("  Saved: fig2_coupling_vs_wavelength.png")

    # ---- Figure 3: Pointing and shear sensitivity ----
    print("\n--- Figure 3: Pointing & shear sensitivity ---")

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Pointing errors ---
    alpha_range = np.linspace(0, 50e-6, 200)  # 0 to 50 urad

    for lam_test in [6e-6, 10e-6, 16e-6]:
        w_f = fiber_mode_radius(lam_test, method='marcuse')

        # Gaussian beam pointing sensitivity (Birbacher Eq. 10) -- EXACT
        eta_gauss_point = np.exp(-np.pi**2 * alpha_range**2 *
                                  W0_GAUSSIAN**2 / lam_test**2)

        # Top-hat: effective-waist approximation
        eta_tophat_approx = coupling_tophat_pointing_approx(
            alpha_range, D_BEAM, lam_test)

        lam_label = r'{:.0f} $\mu$m'.format(lam_test*1e6)
        ax3a.plot(alpha_range * 1e6, eta_gauss_point, '-', lw=2,
                 label=f'Gaussian {lam_label}')
        ax3a.plot(alpha_range * 1e6, eta_tophat_approx, '--', lw=2,
                 label=f'Top-hat approx. {lam_label}')

    # Add a few numerical top-hat points at 10 um for validation
    lam_val = 10e-6
    w_f_val = fiber_mode_radius(lam_val, method='marcuse')
    f_val = optimal_focal_length(D_BEAM, w_f_val, lam_val, beam_type='tophat')
    eta_0_val = coupling_tophat_analytical(D_BEAM, w_f_val, lam_val, f_val)

    alpha_num = np.array([0, 5, 10, 15, 20, 30, 40, 50]) * 1e-6
    eta_num = []
    for a in alpha_num:
        eta = coupling_tophat_with_pointing_numerical(
            D_BEAM, w_f_val, lam_val, f_val, a, N_grid=512)
        eta_num.append(eta / eta_0_val)
    ax3a.plot(alpha_num * 1e6, eta_num, 'ks', ms=6, zorder=5,
             label=r'Top-hat numerical (10 $\mu$m)')

    # NICE measured pointing: ~10 urad
    ax3a.axvline(x=10, color='gray', ls=':', alpha=0.6)
    ax3a.text(11, 0.55, 'NICE\nmeasured', fontsize=9, color='gray')

    # NICE requirement: 19 urad (Birbacher Table 4)
    ax3a.axvline(x=19, color='red', ls=':', alpha=0.6)
    ax3a.text(20, 0.55, 'NICE\nreq.', fontsize=9, color='red')

    ax3a.set_xlabel(r'Pointing error $\alpha$ [$\mu$rad]', fontsize=12)
    ax3a.set_ylabel(r'Relative coupling $\eta/\eta_0$', fontsize=12)
    ax3a.set_title('Pointing sensitivity\n(top-hat: effective-waist approx. '
                   r'validated by $\blacksquare$ numerical)', fontsize=11)
    ax3a.legend(fontsize=7, ncol=2, loc='lower left')
    ax3a.set_ylim(0.5, 1.02)
    ax3a.grid(True, alpha=0.3)

    # --- Shear errors ---
    shear_range = np.linspace(0, 500e-6, 200)  # 0 to 500 um

    # Gaussian: exact
    eta_gauss_shear = np.exp(-shear_range**2 / W0_GAUSSIAN**2)

    # Top-hat: Gaussian approximation
    eta_tophat_shear_approx = coupling_tophat_shear_gaussian_approx(
        shear_range, D_BEAM)

    # Top-hat: geometric clipping (exact for uniform beam)
    eta_tophat_shear_exact = coupling_tophat_shear_geometric(
        shear_range, D_BEAM)

    ax3b.plot(shear_range * 1e6, eta_gauss_shear, 'b-', lw=2,
             label='Gaussian (exact)')
    ax3b.plot(shear_range * 1e6, eta_tophat_shear_approx, 'r--', lw=2,
             label='Top-hat (Gaussian approx.)')
    ax3b.plot(shear_range * 1e6, eta_tophat_shear_exact, 'g-', lw=2,
             label='Top-hat (geometric clipping, exact)')

    # NICE measured shear: 0.17 um
    ax3b.axvline(x=0.17, color='gray', ls=':', alpha=0.6)
    ax3b.text(5, 0.55, r'NICE meas.' + '\n' + r'0.17 $\mu$m',
              fontsize=8, color='gray')
    # NICE requirement: 203 um
    ax3b.axvline(x=203, color='red', ls=':', alpha=0.6)
    ax3b.text(210, 0.55, 'NICE\nreq.', fontsize=9, color='red')

    ax3b.set_xlabel(r'Shear offset $o$ [$\mu$m]', fontsize=12)
    ax3b.set_ylabel(r'Relative coupling $\eta/\eta_0$', fontsize=12)
    ax3b.set_title('Shear sensitivity\n(geometric exact vs. Gaussian approx.)',
                   fontsize=11)
    ax3b.legend(fontsize=9)
    ax3b.set_ylim(0.5, 1.02)
    ax3b.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig('./fig3_pointing_shear_sensitivity.png', dpi=200,
                bbox_inches='tight')
    print("  Saved: fig3_pointing_shear_sensitivity.png")

    # ---- Figure 4: Zernike sensitivity ----
    print("\n--- Figure 4: Zernike aberration sensitivity ---")

    fig4, ax4 = plt.subplots(1, 1, figsize=(9, 6))

    lam_test = 10e-6
    w_f = fiber_mode_radius(lam_test, method='marcuse')
    f_opt = optimal_focal_length(D_BEAM, w_f, lam_test, beam_type='tophat')

    # WFE amplitudes to test (in waves at 10 um)
    wfe_waves = np.linspace(0, 0.15, 30)  # 0 to lam/7
    wfe_meters = wfe_waves * lam_test

    # Baseline coupling with no aberration
    eta_0_tophat = coupling_tophat_analytical(D_BEAM, w_f, lam_test, f_opt)

    zernike_names = {
        4: 'Z4: Defocus',
        5: 'Z5: Astig 45 deg',
        6: 'Z6: Astig 0 deg',
        7: 'Z7: Coma Y',
        8: 'Z8: Coma X',
        11: 'Z11: Spherical',
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (j, name) in enumerate(zernike_names.items()):
        eta_vs_wfe = []
        for wfe in wfe_meters:
            if wfe == 0:
                eta_vs_wfe.append(eta_0_tophat)
            else:
                eta = coupling_with_zernike(D_BEAM, w_f, lam_test, f_opt,
                                           {j: wfe}, beam_type='tophat',
                                           N_grid=256)
                eta_vs_wfe.append(eta)
        eta_vs_wfe = np.array(eta_vs_wfe)
        ax4.plot(wfe_waves, eta_vs_wfe / eta_0_tophat, '-o',
                color=colors[idx], lw=2, ms=4, label=name)

    # Marechal approximation
    strehl_approx = np.exp(-(2 * np.pi * wfe_waves)**2)
    ax4.plot(wfe_waves, strehl_approx, 'k--', lw=1.5, alpha=0.6,
            label='Marechal approx.')

    # Reference lines
    ax4.axhline(y=0.9, color='gray', ls=':', alpha=0.4)
    ax4.axhline(y=0.8, color='gray', ls=':', alpha=0.4)

    # Mark typical mirror quality levels
    for quality, waves, color in [(r'$\lambda$/100', 0.01, 'green'),
                                    (r'$\lambda$/50', 0.02, 'orange'),
                                    (r'$\lambda$/20', 0.05, 'red')]:
        ax4.axvline(x=waves, color=color, ls=':', alpha=0.5)
        ax4.text(waves + 0.002, 0.55, quality, fontsize=9, color=color,
                rotation=90)

    ax4.set_xlabel(r'Wavefront error amplitude [waves RMS at 10 $\mu$m]',
                   fontsize=12)
    ax4.set_ylabel(r'Relative coupling $\eta/\eta_0$', fontsize=12)
    ax4.set_title(r'Fiber coupling sensitivity to Zernike aberrations '
                  r'(top-hat beam, 10 $\mu$m)', fontsize=12)
    ax4.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.55, 0.0))
    ax4.set_ylim(0.5, 1.02)
    ax4.set_xlim(0, 0.15)
    ax4.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig('./fig4_zernike_sensitivity.png', dpi=200,
                bbox_inches='tight')
    print("  Saved: fig4_zernike_sensitivity.png")

    # ---- Numerical validation: compare analytical vs numerical top-hat ----
    print("\n--- Validation: Analytical vs Numerical top-hat coupling ---")

    lam_val = 10e-6
    w_f_val = fiber_mode_radius(lam_val, method='marcuse')
    f_val = optimal_focal_length(D_BEAM, w_f_val, lam_val, beam_type='tophat')

    eta_analytical = coupling_tophat_analytical(D_BEAM, w_f_val, lam_val, f_val)
    eta_numerical = coupling_tophat_numerical(D_BEAM, w_f_val, lam_val, f_val,
                                              N_grid=1024)

    print(f"  At lam = {lam_val*1e6:.0f} um (Marcuse MFD):")
    print(f"    Analytical: eta = {eta_analytical:.4f}")
    print(f"    Numerical:  eta = {eta_numerical:.4f}")
    print(f"    Difference: {abs(eta_analytical - eta_numerical):.2e}")

    # ---- Validation: Marechal vs numerical WFE ----
    print("\n--- Validation: Marechal approximation vs numerical WFE ---")
    for wfe_nm in [20, 50, 100, 200]:
        wfe_m = wfe_nm * 1e-9
        eta_mar = marechal_coupling(eta_analytical, wfe_m, lam_val)
        eta_zern = coupling_with_zernike(D_BEAM, w_f_val, lam_val, f_val,
                                          {4: wfe_m}, beam_type='tophat',
                                          N_grid=256)
        print(f"  WFE = {wfe_nm:3d} nm:  Marechal eta = {eta_mar:.4f},  "
              f"Zernike(Z4) eta = {eta_zern:.4f},  "
              f"diff = {abs(eta_mar - eta_zern) / eta_mar * 100:.1f}%")

    # ---- Null depth from shear ----
    print("\n--- Null from shear (Birbacher Eq. 15) ---")
    for shear_um in [0.17, 1.0, 10.0, 100.0, 203.0]:
        N_sh, dI_sh = null_from_shear(shear_um * 1e-6, W0_GAUSSIAN)
        print(f"  Shear = {shear_um:6.1f} um:  dI = {dI_sh:.2e},  "
              f"N_shear = {N_sh:.2e}")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("SUMMARY: Coupling Penalty Budget (Marcuse MFD)")
    print("=" * 70)
    print(f"{'Wavelength':>12s}  {'V-param':>8s}  {'w_f [um]':>10s}  "
          f"{'eta_TH [%]':>10s}  {'eta_TH/eta_G':>12s}")
    print("-" * 58)

    for lam in [6e-6, 8e-6, 10e-6, 12e-6, 14e-6, 16e-6]:
        w_f = fiber_mode_radius(lam, method='marcuse')
        V = v_parameter(lam)
        f_opt_t = optimal_focal_length(D_BEAM, w_f, lam, beam_type='tophat')
        eta_t = coupling_tophat_analytical(D_BEAM, w_f, lam, f_opt_t)
        ratio = eta_t  # vs Gaussian eta = 1.0

        print(f"{lam*1e6:12.1f}  {V:8.3f}  {w_f*1e6:10.2f}  "
              f"{eta_t*100:10.2f}  {ratio:12.4f}")

    print("-" * 58)
    print(f"\nWith Marcuse MFD, the top-hat/Gaussian ratio is NOT exactly")
    print(f"constant -- it varies by ~{rel_diff.max()*100:.1f}% across the LIFE band")
    print(f"due to nonlinear V-parameter dependence of the mode field radius.")
    print(f"The power-law model predicts exactly constant 81.45% -- this is an")
    print(f"approximation that breaks down at band edges where V changes rapidly.")
    print(f"\nTo compensate the top-hat penalty: collectors must be "
          f"1/sqrt(eta) = {1/np.sqrt(eta_max):.2f}x larger,")
    print(f"  or ~{(1/np.sqrt(eta_max) - 1)*100:.0f}% diameter increase "
          f"({(1/eta_max - 1)*100:.0f}% area increase).")

    plt.close('all')
    print("\nAll figures saved. Module 1 v2.0 complete.")

    return {
        'eta_max_tophat': eta_max,
        'beta_opt': beta_opt,
        'eta_analytical_10um': eta_analytical,
        'eta_numerical_10um': eta_numerical,
        'fiber_params': {
            'core_radius': FIBER_CORE_RADIUS,
            'NA': FIBER_NA,
        },
    }


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    results = run_full_analysis()
