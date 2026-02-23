"""
LIFE End-to-End Instrument Model — Material Properties Library
================================================================

Canonical source of optical material properties for every module in the
LIFE end-to-end study. 

Author
------
Victor Huarcaya

Wavelength convention
---------------------
**All public functions accept and return wavelengths in micrometres.**
Modules that work in SI metres (e.g. Module 3) must convert at the
boundary: ``lam_um = lam_m * 1e6``.

Materials covered
-----------------
============  ======  =================================================
Group         Section Functions
============  ======  =================================================
Gold (Au)     1       Tabulated Ordal/Palik R (normal inc, **default**)
                      Drude complex n-hat (for AOI / Fresnel calcs)
                      Fresnel R at arbitrary angle + polarisation
CaF2          2       Sellmeier n, dn/dlambda, group index, absorption,
                      transmission, chromatic OPD
ZnSe          3       Tatian 1984 Sellmeier n, absorption, transmission,
                      chromatic OPD
KBr           4       Sellmeier n, transmission
Fibers        5       Attenuation (InF3, chalcogenide, AgClBr, HC-PCF),
                      propagation transmission, Fresnel facet loss
Detectors     6       Si:As BIB, HgCdTe (MCT), Si:Sb BIB QE models
AR coatings   7       Broadband / narrowband efficiency vs lambda
BS coatings   8       Dielectric / metallic absorption model
Dichroics     9       Sigmoid transmission model
Thermal       10      Planck radiance, thermal background photon rate
Utilities     11      Material comparison table, best-substrate selector
Figures       12      ``make_material_figures()`` (behind __main__)
============  ======  =================================================

Key references
--------------
- Ordal et al. (1983)  Appl. Opt. 22, 1099  -- Au optical constants
- Palik (1998)         Handbook of Optical Constants of Solids
- Rakic et al. (1998)  Appl. Opt. 37, 5271  -- Au Drude parameters
- Malitson (1963)      Appl. Opt. 2, 1103   -- CaF2 Sellmeier
- Tatian (1984)        Appl. Opt. 23, 4477  -- ZnSe Sellmeier
- Li (1976/1980)       Handbook values       -- KBr Sellmeier
- Glauser et al. (2024) SPIE 13092          -- LIFE baseline design
- Birbacher et al. (2026) arXiv:2602.02279  -- NICE testbed results

Version history
---------------
3.0  2026-01-14  Clean rewrite for codebase reorganisation (Phase A).
                 Consolidates duplicates from MC v2, Throughput, Null
                 Error, and previous library v2.  Gold now carries both
                 tabulated R and Drude n-hat.  14-point MC v2 R table
                 preserved alongside (n,k) Fresnel computation.
2.0  2026-08-12  Fixed ZnSe (Tatian 1984), gold (tabulated), CaF2 alpha.
1.0  2025-02-10  Initial version.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# -- Physical constants --------------------------------------------------
C_LIGHT: float = 2.99792458e8     # speed of light  [m/s]
H_PLANCK: float = 6.62607015e-34  # Planck constant  [J s]
K_BOLTZMANN: float = 1.380649e-23 # Boltzmann constant [J/K]

# ======================================================================
# S1  GOLD MIRROR REFLECTIVITY
# ======================================================================

# -- 14-point (n, k) table from Ordal+1983 / Palik --------------------
# Used by gold_reflectivity() for normal-incidence Fresnel R.
# Log-interpolated for smoothness across decades.

_AU_NK_TABLE = np.array([
    # lam [um]   n         k
    [  1.0,    0.26,     6.97],
    [  2.0,    0.54,    16.50],
    [  3.0,    1.23,    21.80],
    [  4.0,    2.25,    27.50],
    [  5.0,    3.50,    33.50],
    [  6.0,    5.00,    39.50],
    [  8.0,    8.50,    52.00],
    [ 10.0,   12.24,    65.30],
    [ 12.0,   16.70,    78.70],
    [ 14.0,   21.00,    91.50],
    [ 16.0,   25.90,   104.00],
    [ 18.0,   31.00,   117.00],
    [ 20.0,   36.50,   130.00],
    [ 25.0,   50.00,   165.00],
])

# -- 14-point pre-computed R table from MC v2 (Ordal+1983 pristine) ----
# Direct reflectivity values; useful for fast Monte Carlo look-ups
# where (n,k)->Fresnel is not required.
_AU_R_TABLE_UM = np.array([
    1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    8.0, 10.0, 12.0, 14.0, 18.0, 25.0,
])
_AU_R_TABLE_R = np.array([
    0.9810, 0.9850, 0.9870, 0.9875, 0.9878, 0.9878, 0.9875, 0.9878,
    0.9882, 0.9890, 0.9898, 0.9905, 0.9912, 0.9925,
])


def gold_reflectivity(lam_um: ArrayLike,
                      quality: str = 'flight') -> NDArray:
    """Normal-incidence gold mirror reflectivity from tabulated (n, k).

    Computes Fresnel power reflectivity from Ordal+1983 / Palik (n, k)
    data, log-interpolated over 1-25 um.  This replaces the Drude model
    which overestimates R by 0.3-0.7 % across the LIFE band.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    quality : {'ideal', 'flight', 'aged'}
        ``'ideal'``  -- pristine evaporated gold (lab best).
        ``'flight'`` -- space-qualified protected gold, -0.3 % overcoat loss
                       (**default**, LIFE baseline).
        ``'aged'``   -- degraded coating, -0.8 % contamination + overcoat.

    Returns
    -------
    R : ndarray
        Power reflectivity [0, 1].

    Notes
    -----
    Typical flight-quality values:

    ====  =======
    lam   R
    ====  =======
    6 um  ~98.5 %
    10 um ~98.7 %
    16 um ~98.8 %
    ====  =======

    Over 14 mirror reflections the cumulative throughput is R^14 ~ 83 %.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))

    lam_tab = _AU_NK_TABLE[:, 0]
    n_tab = _AU_NK_TABLE[:, 1]
    k_tab = _AU_NK_TABLE[:, 2]

    # Log-interpolate n and k for smooth behaviour
    n_interp = np.exp(np.interp(np.log(lam_um), np.log(lam_tab),
                                np.log(n_tab)))
    k_interp = np.exp(np.interp(np.log(lam_um), np.log(lam_tab),
                                np.log(k_tab)))

    R = ((n_interp - 1)**2 + k_interp**2) / \
        ((n_interp + 1)**2 + k_interp**2)

    # Quality degradation (additive loss)
    _correction = {'ideal': 0.0, 'flight': 0.003, 'aged': 0.008}
    R = R - _correction.get(quality, 0.003)

    return np.clip(R, 0.0, 1.0)


def gold_reflectivity_fast(lam_um: ArrayLike,
                           quality: str = 'flight') -> NDArray:
    """Fast gold reflectivity from the MC v2 pre-computed R table.

    Linear interpolation of 14 directly-measured R values (Ordal+1983).
    Faster than ``gold_reflectivity`` but less accurate at intermediate
    wavelengths because it skips the (n, k) -> Fresnel step.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    quality : {'pristine', 'flight', 'aged'}
        ``'pristine'`` -- lab-best (table values).
        ``'flight'``   -- x0.997 multiplicative correction.
        ``'aged'``     -- x0.994 multiplicative correction.

    Returns
    -------
    R : ndarray
        Power reflectivity [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    R = np.interp(lam_um, _AU_R_TABLE_UM, _AU_R_TABLE_R)

    _mult = {'pristine': 1.0, 'flight': 0.997, 'aged': 0.994}
    R *= _mult.get(quality, 0.997)

    return np.clip(R, 0.0, 1.0)


def gold_refractive_index(lam_um: ArrayLike) -> NDArray:
    """Complex refractive index of gold from the Drude free-electron model.

    Retained for angle-of-incidence calculations where the *relative*
    s-p splitting is accurate even though absolute R is ~0.3-0.7 % high.
    For normal-incidence power reflectivity, use ``gold_reflectivity()``.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].

    Returns
    -------
    n_complex : ndarray (complex)
        Complex refractive index  n + ik.

    Notes
    -----
    Drude parameters from Rakic et al. (1998):

    =====  ================
    w_p    1.37e16 rad/s   (plasma frequency)
    gamma  4.05e13 rad/s   (collision rate)
    =====  ================

    Valid for lam > 1 um where interband transitions are negligible.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))

    omega_p = 1.37e16   # rad/s
    gamma = 4.05e13     # rad/s
    omega = 2.0 * np.pi * C_LIGHT / (lam_um * 1e-6)

    eps_real = 1.0 - omega_p**2 / (omega**2 + gamma**2)
    eps_imag = omega_p**2 * gamma / (omega * (omega**2 + gamma**2))
    eps = eps_real + 1j * eps_imag

    return np.sqrt(eps)


def gold_reflectivity_aoi(lam_um: ArrayLike,
                          angle_deg: float,
                          polarization: str = 'unpolarized',
                          quality: str = 'flight') -> NDArray:
    """Gold reflectivity at arbitrary angle of incidence (Fresnel + Drude).

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    angle_deg : float
        Angle of incidence [degrees].
    polarization : {'s', 'p', 'unpolarized'}
        Polarisation state.
    quality : {'ideal', 'flight', 'aged'}
        Surface quality (additive R correction).

    Returns
    -------
    R : ndarray
        Power reflectivity [0, 1].

    Notes
    -----
    For gold in mid-IR at theta < 45 deg, |R_s - R_p| < 0.5 %.
    At 45 deg: R_s ~ 99.5 %, R_p ~ 98.0 %.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    theta_i = np.radians(angle_deg)

    n2 = gold_refractive_index(lam_um)
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    cos_t = np.sqrt(1.0 - (sin_i / n2)**2)

    r_s = (cos_i - n2 * cos_t) / (cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - cos_t) / (n2 * cos_i + cos_t)

    R_s = np.abs(r_s)**2
    R_p = np.abs(r_p)**2

    _correction = {'ideal': 0.0, 'flight': 0.003, 'aged': 0.008}
    corr = _correction.get(quality, 0.003)

    if polarization == 's':
        return np.clip(R_s - corr, 0.0, 1.0)
    elif polarization == 'p':
        return np.clip(R_p - corr, 0.0, 1.0)
    else:
        return np.clip(0.5 * (R_s + R_p) - corr, 0.0, 1.0)


# ======================================================================
# S2  CaF2  (CALCIUM FLUORIDE) -- BS substrate baseline
# ======================================================================

def caf2_sellmeier(lam_um: ArrayLike) -> NDArray:
    """CaF2 refractive index from the Malitson (1963) Sellmeier equation.

    Three-term form with C values as resonance wavelengths (C-squared form):

        n^2 - 1 = sum_i  B_i lam^2 / (lam^2 - C_i^2)

    Valid 0.15-12 um.  Extrapolation beyond 12 um is monotonic but
    should be treated with caution past the 9.7 um multi-phonon edge.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].

    Returns
    -------
    n : ndarray
        Refractive index (real).

    Notes
    -----
    ===  =============  ============  ======================
    i    B_i            C_i [um]      Physical origin
    ===  =============  ============  ======================
    1    0.5675888      0.050264      UV electronic resonance
    2    0.4710914      0.100391      UV electronic resonance
    3    3.8484723      34.649        IR phonon at 34.6 um
    ===  =============  ============  ======================

    Denominators (lam^2 - C^2) are equivalent to the pre-squared
    constants used in MC v2 (0.00252643, 0.01007833, 1200.556).
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    lam2 = lam_um**2

    B1, C1 = 0.5675888, 0.050263605   # um
    B2, C2 = 0.4710914, 0.1003909     # um
    B3, C3 = 3.8484723, 34.649040     # um

    n_sq = 1.0 + (B1 * lam2 / (lam2 - C1**2)
                 + B2 * lam2 / (lam2 - C2**2)
                 + B3 * lam2 / (lam2 - C3**2))
    return np.sqrt(np.maximum(n_sq, 1.0))


def caf2_dn_dlambda(lam_um: ArrayLike) -> NDArray:
    """CaF2 chromatic dispersion dn/dlambda [um^-1] (numerical derivative).

    Critical for estimating chromatic OPD from BS thickness mismatch.
    Negative throughout mid-IR (normal dispersion).

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    dlam = 0.001  # um step
    return (caf2_sellmeier(lam_um + dlam) -
            caf2_sellmeier(lam_um - dlam)) / (2.0 * dlam)


def caf2_group_index(lam_um: ArrayLike) -> NDArray:
    """CaF2 group index n_g = n - lam (dn/dlam).

    Relevant for broadband pulse propagation and coherence length.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    n = caf2_sellmeier(lam_um)
    dn = caf2_dn_dlambda(lam_um)
    return n - lam_um * dn


def caf2_absorption(lam_um: ArrayLike) -> NDArray:
    """CaF2 bulk absorption coefficient [cm^-1].

    Near-zero for lam < 7.5 um; rises exponentially from multi-phonon
    absorption onset.  Practical transparency limit ~10 um.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].

    Returns
    -------
    alpha : ndarray
        Absorption coefficient [cm^-1], capped at 100.

    Notes
    -----
    Coefficient 0.03 calibrated to ISP Optics / Optovac datasheets:

    =====  ============
    lam    alpha
    =====  ============
    8 um   ~0.08 cm^-1   (>99 % T through 2 mm)
    10 um  ~2.7  cm^-1   (~58 % T through 2 mm)
    12 um  ~99   cm^-1   (opaque)
    =====  ============
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    alpha = np.zeros_like(lam_um)
    mask = lam_um > 7.5
    alpha[mask] = 0.03 * np.exp(1.8 * (lam_um[mask] - 7.5))
    return np.minimum(alpha, 100.0)


def caf2_transmission(lam_um: ArrayLike,
                      thickness_mm: float = 2.0,
                      ar_coated: bool = False,
                      ar_efficiency: float = 0.95) -> NDArray:
    """CaF2 substrate single-pass transmission (Fresnel + bulk absorption).

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    thickness_mm : float
        Substrate thickness [mm].
    ar_coated : bool
        Apply AR coating model to both surfaces.
    ar_efficiency : float
        Fraction of Fresnel loss eliminated by coating [0, 1].

    Returns
    -------
    T : ndarray
        Power transmission [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    n = caf2_sellmeier(lam_um)

    R_fresnel = ((n - 1.0) / (n + 1.0))**2
    if ar_coated:
        R_fresnel *= (1.0 - ar_efficiency)
    T_fresnel = (1.0 - R_fresnel)**2  # two surfaces

    d_cm = thickness_mm * 0.1
    T_bulk = np.exp(-caf2_absorption(lam_um) * d_cm)

    return T_fresnel * T_bulk


def caf2_chromatic_opd(lam_um: ArrayLike,
                       delta_d_um: float,
                       lam_ref_um: float = 10.0) -> NDArray:
    """Chromatic OPD from CaF2 BS thickness mismatch.

        delta_OPD(lam) = [n(lam) - n(lam_ref)] * Delta_d

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    delta_d_um : float
        Thickness mismatch [um].
    lam_ref_um : float
        Reference wavelength where compensator is tuned [um].

    Returns
    -------
    dopd_m : ndarray
        Chromatic OPD [m].
    """
    n = caf2_sellmeier(lam_um)
    n_ref = float(caf2_sellmeier(np.atleast_1d(lam_ref_um))[0])
    return (n - n_ref) * delta_d_um * 1e-6  # um -> m


# ======================================================================
# S3  ZnSe  (ZINC SELENIDE) -- alternative BS substrate
# ======================================================================

def znse_sellmeier(lam_um: ArrayLike) -> NDArray:
    """ZnSe refractive index from Tatian (1984) three-term Sellmeier.

    Valid 0.6-22 um.  Transparent across the full LIFE band.
    High refractive index (n ~ 2.4) -> larger Fresnel losses, but
    excellent transparency to ~20 um.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].

    Returns
    -------
    n : ndarray
        Refractive index.

    Notes
    -----
    ===  ==========  ============
    i    B_i         C_i [um]
    ===  ==========  ============
    1    4.45813     0.200859
    2    0.467216    0.391880
    3    2.89566     47.1362
    ===  ==========  ============

    Replaces the old 2-term Connolly (1979) form that had 3-5 % error.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    lam2 = lam_um**2

    B1, C1 = 4.45813, 0.200859
    B2, C2 = 0.467216, 0.391880
    B3, C3 = 2.89566, 47.1362

    n_sq = 1.0 + (B1 * lam2 / (lam2 - C1**2)
                 + B2 * lam2 / (lam2 - C2**2)
                 + B3 * lam2 / (lam2 - C3**2))
    return np.sqrt(np.maximum(n_sq, 1.0))


def znse_absorption(lam_um: ArrayLike) -> NDArray:
    """ZnSe bulk absorption coefficient [cm^-1].

    Very low absorption 2-16 um (< 0.001 cm^-1).
    Onset near 20 um from multi-phonon absorption.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    alpha = np.zeros_like(lam_um)
    mask = lam_um > 16.0
    alpha[mask] = 0.1 * np.exp(1.5 * (lam_um[mask] - 16.0))
    return alpha


def znse_transmission(lam_um: ArrayLike,
                      thickness_mm: float = 2.0,
                      ar_coated: bool = False,
                      ar_efficiency: float = 0.90) -> NDArray:
    """ZnSe substrate single-pass transmission.

    Without AR coating: Fresnel loss ~ 17 % per surface (n ~ 2.4),
    giving T ~ 69 % for two surfaces.  AR coating is essential.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    thickness_mm : float
        Substrate thickness [mm].
    ar_coated : bool
        Apply AR coating model.
    ar_efficiency : float
        Fraction of Fresnel loss eliminated [0, 1].

    Returns
    -------
    T : ndarray
        Power transmission [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    n = znse_sellmeier(lam_um)

    R_fresnel = ((n - 1.0) / (n + 1.0))**2
    if ar_coated:
        R_fresnel *= (1.0 - ar_efficiency)
    T_fresnel = (1.0 - R_fresnel)**2

    d_cm = thickness_mm * 0.1
    T_bulk = np.exp(-znse_absorption(lam_um) * d_cm)

    return T_fresnel * T_bulk


def znse_chromatic_opd(lam_um: ArrayLike,
                       delta_d_um: float,
                       lam_ref_um: float = 10.0) -> NDArray:
    """Chromatic OPD from ZnSe BS thickness mismatch [m].

    Same physics as ``caf2_chromatic_opd`` but with ZnSe dispersion.
    """
    n = znse_sellmeier(lam_um)
    n_ref = float(znse_sellmeier(np.atleast_1d(lam_ref_um))[0])
    return (n - n_ref) * delta_d_um * 1e-6


# ======================================================================
# S4  KBr  (POTASSIUM BROMIDE) -- window material
# ======================================================================

def kbr_refractive_index(lam_um: ArrayLike) -> NDArray:
    """KBr refractive index (Li 1976 Sellmeier, adjusted).

    Valid 0.2-30 um.  Very low dispersion in mid-IR (n ~ 1.54).
    KBr is hygroscopic and requires a dry environment.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    lam2 = lam_um**2
    n_sq = (1.0 + 1.3284 * lam2 / (lam2 - 0.01662)
                + 0.2513 * lam2 / (lam2 - 625.0))
    return np.sqrt(np.maximum(n_sq, 1.0))


def kbr_transmission(lam_um: ArrayLike,
                     thickness_mm: float = 5.0) -> NDArray:
    """KBr window transmission.  Transparent 0.2-25 um.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    thickness_mm : float
        Window thickness [mm].

    Returns
    -------
    T : ndarray
        Power transmission [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    n = kbr_refractive_index(lam_um)
    R_fresnel = ((n - 1.0) / (n + 1.0))**2
    T_fresnel = (1.0 - R_fresnel)**2

    alpha = np.zeros_like(lam_um)
    mask = lam_um > 22.0
    alpha[mask] = 0.5 * np.exp(2.0 * (lam_um[mask] - 22.0))
    T_bulk = np.exp(-alpha * thickness_mm * 0.1)

    return T_fresnel * T_bulk


# ======================================================================
# S5  FIBER MATERIALS -- attenuation models
# ======================================================================

def fiber_attenuation(lam_um: ArrayLike,
                      fiber_type: str = 'InF3') -> NDArray:
    """Single-mode fiber attenuation coefficient [dB/m].

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    fiber_type : {'InF3', 'chalcogenide', 'silver_halide', 'hollow_core'}
        ``'InF3'``          -- indium fluoride (ZBLAN family), best 2-5.5 um.
        ``'chalcogenide'``  -- As2Se3/As2S3, broadband 2-12 um.
        ``'silver_halide'`` -- AgClBr, full LIFE band 3-20 um.
        ``'hollow_core'``   -- hollow-core PCF, narrow 4-12 um.

    Returns
    -------
    alpha_dBm : ndarray
        Attenuation [dB/m], capped at 50.

    Notes
    -----
    For the NICE testbed (InF3, ~0.3 m): ~0.1 dB/m at 4 um.
    Short fibers (< 1 m) -> propagation loss is a minor contributor.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))

    if fiber_type == 'InF3':
        alpha = np.where(lam_um < 5.0,
                         0.05 + 0.01 * (lam_um - 4.0)**2,
                         0.05 + 2.0 * (lam_um - 5.0)**2)
    elif fiber_type == 'chalcogenide':
        alpha = 0.1 + 0.01 * (lam_um - 6.0)**2
        alpha = np.where(lam_um > 10.0,
                         alpha + 0.5 * (lam_um - 10.0)**2, alpha)
    elif fiber_type == 'silver_halide':
        alpha = 0.5 + 0.003 * (lam_um - 10.0)**2
    elif fiber_type == 'hollow_core':
        alpha = 0.3 + 0.1 * (lam_um - 8.0)**2
    else:
        alpha = 0.5 * np.ones_like(lam_um)

    return np.minimum(alpha, 50.0)


def fiber_transmission(lam_um: ArrayLike,
                       fiber_type: str = 'InF3',
                       length_m: float = 0.5) -> NDArray:
    """Fiber propagation transmission (excludes coupling and Fresnel).

    T = 10^(-alpha * L / 10)  where alpha is in dB/m.
    """
    alpha = fiber_attenuation(lam_um, fiber_type)
    return 10.0**(-alpha * length_m / 10.0)


def fiber_fresnel_loss(lam_um: ArrayLike,
                       n_core: float = 1.5,
                       ar_coated: bool = False) -> NDArray:
    """Fresnel reflection loss at fiber entrance facet.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um]  (used only for array shape).
    n_core : float
        Core refractive index (1.5 for fluoride, 2.7 for chalcogenide).
    ar_coated : bool
        AR-coated tip (reduces Fresnel by ~90 %).

    Returns
    -------
    T : ndarray
        Transmission through entrance facet [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    R = ((n_core - 1.0) / (n_core + 1.0))**2
    if ar_coated:
        R *= 0.1
    return (1.0 - R) * np.ones_like(lam_um)


# ======================================================================
# S6  DETECTOR QUANTUM EFFICIENCY
# ======================================================================

def detector_qe(lam_um: ArrayLike,
                detector_type: str = 'SiAs_BIB') -> NDArray:
    """Detector quantum efficiency model.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    detector_type : {'SiAs_BIB', 'SiAs', 'HgCdTe', 'SiSb_BIB'}
        ``'SiAs_BIB'`` or ``'SiAs'`` -- Si:As blocked impurity band
        (LIFE baseline, JWST MIRI heritage).  Peak QE ~60-65 % near 15 um.
        Requires cooling to ~7 K.
        ``'HgCdTe'``   -- mercury cadmium telluride (long-wave cutoff ~14 um).
        ``'SiSb_BIB'`` -- Si:Sb BIB (broader response, cutoff ~40 um).

    Returns
    -------
    QE : ndarray
        Quantum efficiency [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))

    if detector_type in ('SiAs_BIB', 'SiAs'):
        # Si:As BIB: onset ~2 um, flat 5-25 um, cutoff ~28 um
        QE = np.where(
            lam_um < 3.0,
            0.1 * (lam_um / 3.0)**2,
            np.where(lam_um < 25.0,
                     0.60 + 0.05 * np.exp(-((lam_um - 15.0) / 8.0)**2),
                     0.60 * np.exp(-((lam_um - 25.0) / 3.0)**2)))
        QE *= (1.0 - 0.02 * np.abs(lam_um - 15.0) / 15.0)

    elif detector_type == 'HgCdTe':
        lam_cutoff = 14.0
        QE = np.where(lam_um < lam_cutoff,
                      0.70 * (1.0 - np.exp(-2.0 * (lam_cutoff - lam_um))),
                      0.01)
        QE = np.where(lam_um < 2.0, 0.1 * (lam_um / 2.0), QE)

    elif detector_type == 'SiSb_BIB':
        QE = np.where(lam_um < 35.0, 0.50,
                      0.50 * np.exp(-((lam_um - 35.0) / 5.0)**2))
    else:
        QE = 0.50 * np.ones_like(lam_um)

    return np.clip(QE, 0.0, 1.0)


# ======================================================================
# S7  AR COATING MODELS
# ======================================================================

def ar_coating_efficiency(lam_um: ArrayLike,
                          design_lam_um: float = 10.0,
                          bandwidth: str = 'broadband') -> NDArray:
    """Anti-reflection coating efficiency vs wavelength.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    design_lam_um : float
        Centre wavelength of AR design [um].
    bandwidth : {'narrowband', 'broadband'}
        ``'narrowband'`` -- single-layer quarter-wave.
        ``'broadband'``  -- multi-layer, ~95 % over 6-16 um.

    Returns
    -------
    efficiency : ndarray
        Fraction of Fresnel loss eliminated [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))

    if bandwidth == 'narrowband':
        delta = np.abs(lam_um - design_lam_um) / design_lam_um
        efficiency = np.cos(np.minimum(delta * np.pi / 0.4, np.pi / 2))**2
    else:
        center = (6.0 + 16.0) / 2.0   # 11 um
        halfwidth = 7.0
        x = (lam_um - center) / halfwidth
        efficiency = 0.95 * np.exp(-2.0 * x**4)  # super-Gaussian
        efficiency = np.maximum(efficiency, 0.5)

    return np.clip(efficiency, 0.0, 1.0)


# ======================================================================
# S8  BEAMSPLITTER COATING ABSORPTION
# ======================================================================

def beamsplitter_coating_absorption(lam_um: ArrayLike,
                                    coating_type: str = 'dielectric') -> NDArray:
    """Absorption loss in BS 50/50 coating.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    coating_type : {'dielectric', 'metallic'}
        ``'dielectric'`` -- ~1.5 % base, slight rise at band edges.
        ``'metallic'``   -- ~4 % (Cr/Ni), more achromatic splitting.

    Returns
    -------
    absorption : ndarray
        Fractional absorption per pass [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))

    if coating_type == 'dielectric':
        absorption = 0.015 + 0.005 * np.exp(-((lam_um - 10.0) / 6.0)**2)
    elif coating_type == 'metallic':
        absorption = 0.04 * np.ones_like(lam_um)
    else:
        absorption = 0.02 * np.ones_like(lam_um)

    return absorption


# ======================================================================
# S9  DICHROIC MIRROR TRANSMISSION
# ======================================================================

def dichroic_transmission(lam_um: ArrayLike,
                          cutoff_um: float = 4.0,
                          transition_width: float = 0.3,
                          passband_T: float = 0.95) -> NDArray:
    """Dichroic mirror transmission (sigmoid profile).

    Transmits wavelengths above cutoff, reflects below.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    cutoff_um : float
        Transition centre wavelength [um].
    transition_width : float
        Sigmoid scale parameter [um].
    passband_T : float
        In-band peak transmission.

    Returns
    -------
    T : ndarray
        Power transmission [0, 1].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    x = (lam_um - cutoff_um) / transition_width
    T = passband_T / (1.0 + np.exp(-x))
    return np.maximum(T, 0.001)


# ======================================================================
# S10  THERMAL EMISSION MODELS
# ======================================================================

def planck_spectral_radiance(lam_um: ArrayLike,
                             T_K: float) -> NDArray:
    """Planck spectral radiance B_lam [W m^-2 sr^-1 um^-1].

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    T_K : float
        Temperature [K].

    Returns
    -------
    B : ndarray
        Spectral radiance [W/m^2/sr/um].
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    lam_m = lam_um * 1e-6

    x = H_PLANCK * C_LIGHT / (lam_m * K_BOLTZMANN * T_K)
    x = np.minimum(x, 500.0)  # prevent overflow

    B = 2.0 * H_PLANCK * C_LIGHT**2 / (lam_m**5 * (np.exp(x) - 1.0))
    return B * 1e-6  # per-m -> per-um


def thermal_background_photon_rate(lam_um: ArrayLike,
                                   T_K: float,
                                   emissivity: float,
                                   A_collect: float,
                                   Omega_beam: float,
                                   delta_lam_um=None) -> NDArray:
    """Thermal photon rate from an optical surface at temperature *T_K*.

    Parameters
    ----------
    lam_um : float or array
        Wavelength [um].
    T_K : float
        Surface temperature [K].
    emissivity : float
        Surface emissivity (= 1 - R for opaque surfaces).
    A_collect : float
        Collecting area [m^2].
    Omega_beam : float
        Beam solid angle [sr].
    delta_lam_um : float or array or None
        Spectral bandwidth [um].  Default: 0.1 * lambda.

    Returns
    -------
    photon_rate : ndarray
        Photons s^-1 per spectral bin.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    if delta_lam_um is None:
        delta_lam_um = 0.1 * lam_um

    B = planck_spectral_radiance(lam_um, T_K)
    power = emissivity * B * A_collect * Omega_beam * delta_lam_um

    E_photon = H_PLANCK * C_LIGHT / (lam_um * 1e-6)
    return power / E_photon


# ======================================================================
# S11  COMPARISON AND SUMMARY UTILITIES
# ======================================================================

def material_comparison_table(lam_um: float = 10.0) -> None:
    """Print a comparison table of all BS substrates at a given wavelength."""
    lam = np.atleast_1d(lam_um)

    print(f"\n--- Material properties comparison at lambda = {lam_um} um ---")
    print(f"{'Material':>15s}  {'n':>6s}  {'alpha [cm-1]':>12s}  "
          f"{'T(2mm,bare)':>12s}  {'T(2mm,AR)':>10s}")
    print("-" * 65)

    materials = [
        ('CaF2', caf2_sellmeier, caf2_absorption, caf2_transmission),
        ('ZnSe', znse_sellmeier, znse_absorption, znse_transmission),
    ]
    for name, n_func, a_func, t_func in materials:
        n = float(n_func(lam)[0])
        a = float(a_func(lam)[0])
        T_bare = float(t_func(lam, 2.0, False)[0])
        T_ar = float(t_func(lam, 2.0, True)[0])
        print(f"{name:>15s}  {n:>6.3f}  {a:>12.3f}  "
              f"{T_bare:>12.3f}  {T_ar:>10.3f}")

    n_kbr = float(kbr_refractive_index(lam)[0])
    T_kbr = float(kbr_transmission(lam, 2.0)[0])
    print(f"{'KBr':>15s}  {n_kbr:>6.3f}  {'~0':>12s}  "
          f"{T_kbr:>12.3f}  {'N/A':>10s}")

    R_au = float(gold_reflectivity(lam, 'flight')[0])
    print(f"{'Gold':>15s}  {'--':>6s}  {'--':>12s}  "
          f"{'R=' + f'{R_au:.4f}':>12s}  {'--':>10s}")


def best_substrate(lam_um: ArrayLike,
                   thickness_mm: float = 2.0):
    """Return the best BS substrate material at each wavelength.

    CaF2 preferred for lam < ~9 um (lower n -> lower Fresnel loss).
    ZnSe preferred for lam > ~9 um (transparent across full LIFE band).

    Returns
    -------
    best : ndarray of str
        ``'CaF2'`` or ``'ZnSe'`` at each wavelength.
    T_best : ndarray
        Transmission of the better substrate.
    """
    lam_um = np.atleast_1d(np.asarray(lam_um, dtype=float))
    T_caf2 = caf2_transmission(lam_um, thickness_mm, True, 0.95)
    T_znse = znse_transmission(lam_um, thickness_mm, True, 0.90)

    best = np.where(T_caf2 >= T_znse, 'CaF2', 'ZnSe')
    T_best = np.maximum(T_caf2, T_znse)
    return best, T_best


# ======================================================================
# S12  FIGURE GENERATION
# ======================================================================

def make_material_figures() -> None:
    """Generate publication-quality material-properties overview figures.

    Produces three figure files:

    * **fig17** -- Refractive indices and dispersion (CaF2, ZnSe, KBr).
    * **fig18** -- Substrate transmission + fiber attenuation.
    * **fig19** -- Gold reflectivity + detector QE.

    Also prints ``material_comparison_table`` at 6, 10, 16 um.
    """
    import matplotlib.pyplot as plt

    lam = np.linspace(1.0, 25.0, 500)

    # -- Fig 17: Refractive indices ----------------------------------
    fig17, (ax17a, ax17b) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax17a.plot(lam, caf2_sellmeier(lam), 'b-', lw=2.5, label=r'CaF$_2$')
    ax17a.plot(lam, znse_sellmeier(lam), 'r-', lw=2.5, label='ZnSe')
    ax17a.plot(lam, kbr_refractive_index(lam), 'g--', lw=2, label='KBr')
    ax17a.axvspan(6, 16, alpha=0.08, color='green', label='LIFE band')
    ax17a.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax17a.set_ylabel('Refractive index $n$', fontsize=12)
    ax17a.set_title('Refractive Index of BS Substrates', fontsize=13)
    ax17a.legend(fontsize=10)
    ax17a.set_xlim(1, 25)
    ax17a.grid(True, alpha=0.3)

    ax17b.plot(lam, caf2_dn_dlambda(lam), 'b-', lw=2.5,
               label=r'CaF$_2$ $dn/d\lambda$')
    ax17b.plot(lam, np.gradient(znse_sellmeier(lam), lam), 'r-', lw=2.5,
               label=r'ZnSe $dn/d\lambda$')
    ax17b.axvspan(6, 16, alpha=0.08, color='green')
    ax17b.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax17b.set_ylabel(r'$dn/d\lambda$ [$\mu$m$^{-1}$]', fontsize=12)
    ax17b.set_title('Chromatic Dispersion (drives BS OPD error)', fontsize=13)
    ax17b.legend(fontsize=10)
    ax17b.set_xlim(1, 25)
    ax17b.grid(True, alpha=0.3)

    fig17.tight_layout()
    fig17.savefig('fig17_refractive_indices.png', dpi=200,
                  bbox_inches='tight')
    print("  Saved: fig17_refractive_indices.png")

    # -- Fig 18: Substrate transmission + fiber attenuation ----------
    fig18, (ax18a, ax18b) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax18a.plot(lam, caf2_transmission(lam, 2.0, True) * 100, 'b-', lw=2.5,
               label=r'CaF$_2$ (2mm, AR)')
    ax18a.plot(lam, znse_transmission(lam, 2.0, True) * 100, 'r-', lw=2.5,
               label='ZnSe (2mm, AR)')
    ax18a.plot(lam, kbr_transmission(lam, 2.0) * 100, 'g--', lw=2,
               label='KBr (2mm, bare)')
    _, T_best = best_substrate(lam, 2.0)
    ax18a.plot(lam, T_best * 100, 'k:', lw=2, alpha=0.5,
               label='Best substrate')
    ax18a.axvspan(6, 16, alpha=0.08, color='green')
    ax18a.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax18a.set_ylabel('Transmission [%]', fontsize=12)
    ax18a.set_title('BS Substrate Transmission (2mm, AR-coated)', fontsize=13)
    ax18a.legend(fontsize=9)
    ax18a.set_xlim(1, 25)
    ax18a.set_ylim(0, 100)
    ax18a.grid(True, alpha=0.3)

    lam_f = np.linspace(2, 22, 400)
    for ftype, color, ls in [('InF3', 'blue', '-'),
                              ('chalcogenide', 'red', '-'),
                              ('silver_halide', 'green', '--'),
                              ('hollow_core', 'purple', '-.')]:
        att = fiber_attenuation(lam_f, ftype)
        ax18b.semilogy(lam_f, att, color=color, ls=ls, lw=2, label=ftype)
    ax18b.axvspan(6, 16, alpha=0.08, color='green')
    ax18b.axhline(y=1.0, color='gray', ls=':', alpha=0.5)
    ax18b.text(20, 1.2, '1 dB/m', fontsize=9, color='gray')
    ax18b.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax18b.set_ylabel('Attenuation [dB/m]', fontsize=12)
    ax18b.set_title('Single-Mode Fiber Attenuation', fontsize=13)
    ax18b.legend(fontsize=9)
    ax18b.set_xlim(2, 22)
    ax18b.set_ylim(0.01, 100)
    ax18b.grid(True, alpha=0.3, which='both')

    fig18.tight_layout()
    fig18.savefig('fig18_transmission_fiber.png', dpi=200,
                  bbox_inches='tight')
    print("  Saved: fig18_transmission_fiber.png")

    # -- Fig 19: Gold reflectivity + detector QE ---------------------
    fig19, (ax19a, ax19b) = plt.subplots(1, 2, figsize=(13, 5.5))

    for qual, color, ls in [('ideal', 'gold', '-'),
                             ('flight', 'orange', '-'),
                             ('aged', 'brown', '--')]:
        R = gold_reflectivity(lam, qual) * 100
        ax19a.plot(lam, R, color=color, ls=ls, lw=2.5,
                   label=f'Gold ({qual})')
    ax19a.axvspan(6, 16, alpha=0.08, color='green')
    ax19a.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax19a.set_ylabel('Reflectivity [%]', fontsize=12)
    ax19a.set_title('Gold Mirror Reflectivity', fontsize=13)
    ax19a.legend(fontsize=10, loc='lower right')
    ax19a.set_xlim(1, 25)
    ax19a.set_ylim(95, 100)
    ax19a.grid(True, alpha=0.3)

    for det, color, ls in [('SiAs_BIB', 'blue', '-'),
                            ('HgCdTe', 'red', '-'),
                            ('SiSb_BIB', 'green', '--')]:
        QE = detector_qe(lam, det) * 100
        ax19b.plot(lam, QE, color=color, ls=ls, lw=2.5, label=det)
    ax19b.axvspan(6, 16, alpha=0.08, color='green')
    ax19b.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
    ax19b.set_ylabel('Quantum Efficiency [%]', fontsize=12)
    ax19b.set_title('Detector Quantum Efficiency', fontsize=13)
    ax19b.legend(fontsize=10)
    ax19b.set_xlim(1, 25)
    ax19b.set_ylim(0, 100)
    ax19b.grid(True, alpha=0.3)

    fig19.tight_layout()
    fig19.savefig('fig19_gold_detector.png', dpi=200, bbox_inches='tight')
    print("  Saved: fig19_gold_detector.png")

    plt.close('all')

    for lam_val in [6.0, 10.0, 16.0]:
        material_comparison_table(lam_val)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("LIFE Material Properties Library -- Summary Figures")
    print("=" * 70)
    make_material_figures()
    print("\nAll figures saved.  Material properties library ready.")
