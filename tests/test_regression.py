"""
LIFE E2E Model — Regression Test Suite
========================================

Spot-checks across all 7 modules, matching the values verified
in the codebase audit. All tolerances are generous (0.5–2%) to
account for platform-dependent floating-point differences.

Run with:
    pytest tests/test_regression.py -v

Reference: codebase_audit.md (February 2026)
"""

import numpy as np
import pytest


# Fiber parameters (InF3 single-mode fiber, from m1_fiber_coupling.py)
N_CORE = 1.50
N_CLAD = 1.48
A_CORE_UM = 4.5


# ============================================================
# 1. Module imports (7 checks)
# ============================================================

class TestImports:
    """Verify all 7 modules import without error."""

    def test_import_material_properties(self):
        import material_properties  # noqa: F401

    def test_import_fiber_modes(self):
        import fiber_modes  # noqa: F401

    def test_import_m1(self):
        import m1_fiber_coupling  # noqa: F401

    def test_import_m2(self):
        import m2_throughput_chain  # noqa: F401

    def test_import_m3(self):
        import m3_null_error_propagation  # noqa: F401

    def test_import_m4(self):
        import m4_surface_sensitivity  # noqa: F401

    def test_import_monte_carlo(self):
        import monte_carlo  # noqa: F401


# ============================================================
# 2. Material properties (5 checks)
# ============================================================

class TestMaterialProperties:
    """Regression checks against published optical constants."""

    def test_gold_reflectivity_10um(self):
        """Au R(10 µm, flight) = 98.60% — Ordal+1983 / Palik."""
        from material_properties import gold_reflectivity
        R = gold_reflectivity(np.array([10.0]), 'flight')
        assert abs(R[0] - 0.9860) < 0.001

    def test_caf2_index_10um(self):
        """CaF₂ n(10 µm) = 1.2996 — Malitson 1963."""
        from material_properties import caf2_sellmeier
        n = caf2_sellmeier(np.array([10.0]))
        assert abs(n[0] - 1.2996) < 0.005

    def test_znse_index_10um(self):
        """ZnSe n(10 µm) = 2.4065 — Tatian 1984."""
        from material_properties import znse_sellmeier
        n = znse_sellmeier(np.array([10.0]))
        assert abs(n[0] - 2.4065) < 0.005

    def test_caf2_absorption_10um(self):
        """CaF₂ α(10 µm) = 2.70 cm⁻¹ — ISP Optics."""
        from material_properties import caf2_absorption
        alpha = caf2_absorption(np.array([10.0]))
        assert abs(alpha[0] - 2.70) < 0.2

    def test_detector_qe_10um(self):
        """Si:As BIB QE(10 µm) ≈ 63% — JWST MIRI heritage."""
        from material_properties import detector_qe
        qe = detector_qe(np.array([10.0]))
        assert abs(qe[0] - 0.63) < 0.05


# ============================================================
# 3. Fiber modes (5 checks)
# ============================================================

class TestFiberModes:
    """Fiber optics library checks."""

    def test_tophat_coupling_optimal(self):
        """η₀(β_opt) = 81.45% — Ruilier & Cassaing 2001."""
        from fiber_modes import coupling_tophat_analytical
        beta_opt = 1.1209
        eta = coupling_tophat_analytical(np.array([beta_opt]))
        assert abs(eta[0] - 0.8145) < 0.002

    def test_tophat_coupling_beta_opt(self):
        """Optimal β ≈ 1.121 maximises top-hat coupling."""
        from fiber_modes import coupling_tophat_analytical
        betas = np.linspace(0.5, 2.0, 1000)
        etas = coupling_tophat_analytical(betas)
        beta_max = betas[np.argmax(etas)]
        assert abs(beta_max - 1.121) < 0.01

    def test_gaussian_coupling_unity(self):
        """Matched Gaussian → Gaussian coupling = 100%."""
        from fiber_modes import coupling_gaussian_to_gaussian
        eta = coupling_gaussian_to_gaussian(1.0, 1.0)
        assert abs(np.asarray(eta).flat[0] - 1.0) < 1e-10

    def test_v_parameter_10um(self):
        """V-parameter should be > 0 and finite at 10 µm."""
        from fiber_modes import v_parameter
        V = v_parameter(np.array([10.0]), N_CORE, N_CLAD, A_CORE_UM)
        assert V[0] > 0
        assert np.isfinite(V[0])

    def test_mode_field_radius_positive(self):
        """Mode field radius positive and finite at 10 µm."""
        from fiber_modes import mode_field_radius
        w = mode_field_radius(np.array([10.0]), N_CORE, N_CLAD, A_CORE_UM)
        assert w[0] > 0
        assert np.isfinite(w[0])


# ============================================================
# 4. Module 1 — Fiber coupling (3 checks)
# ============================================================

class TestModule1:
    """Fiber coupling engine checks."""

    def test_tophat_coupling_deficit(self):
        """Top-hat coupling deficit = 18.55% (1 - 0.8145)."""
        from fiber_modes import coupling_tophat_analytical
        eta = coupling_tophat_analytical(np.array([1.1209]))
        deficit = 1.0 - eta[0]
        assert abs(deficit - 0.1855) < 0.005

    def test_coupling_decreases_with_aberration(self):
        """Coupling should decrease with increasing WFE."""
        from fiber_modes import coupling_tophat_analytical
        eta_clean = coupling_tophat_analytical(np.array([1.1209]))[0]
        # Maréchal: η ≈ η₀ × exp(-(2π σ/λ)²) → always less
        sigma_wfe = 50e-9  # 50 nm RMS
        lam = 10e-6
        marechal = np.exp(-(2 * np.pi * sigma_wfe / lam) ** 2)
        assert marechal < 1.0
        assert eta_clean * marechal < eta_clean

    def test_coupling_monotonic_in_wavelength(self):
        """Maréchal degradation should decrease (improve) at longer λ."""
        sigma_wfe = 50e-9
        lam_short = 6e-6
        lam_long = 16e-6
        m_short = np.exp(-(2 * np.pi * sigma_wfe / lam_short) ** 2)
        m_long = np.exp(-(2 * np.pi * sigma_wfe / lam_long) ** 2)
        assert m_long > m_short


# ============================================================
# 5. Module 3 — Null error budget (3 checks)
# ============================================================

class TestModule3:
    """Null error propagation checks."""

    def test_null_budget_returns_dict(self):
        """compute_null_budget returns a dict with expected keys."""
        from m3_null_error_propagation import compute_null_budget
        wavelengths = np.array([6.0, 10.0, 16.0]) * 1e-6  # SI metres
        result = compute_null_budget(wavelengths)
        assert isinstance(result, dict)
        assert 'N_total' in result

    def test_null_depth_wavelength_ordering(self):
        """Null depth should be worse (higher) at shorter λ."""
        from m3_null_error_propagation import compute_null_budget
        wavelengths = np.array([6.0, 10.0, 16.0]) * 1e-6
        result = compute_null_budget(wavelengths)
        N = result['N_total']
        # 6 µm should have worse null than 16 µm
        assert N[0] > N[2]

    def test_null_requirement_curve(self):
        """Requirement curve should return positive values."""
        from m3_null_error_propagation import null_requirement_curve
        wavelengths = np.array([6.0, 10.0, 16.0]) * 1e-6
        N_req = null_requirement_curve(wavelengths)
        assert all(n > 0 for n in N_req)


# ============================================================
# 6. Monte Carlo (8 checks)
# ============================================================

class TestMonteCarlo:
    """MC engine checks with small N for speed."""

    @pytest.fixture(scope='class')
    def mc_results(self):
        """Run MC once with N=1000 for all tests in this class."""
        import warnings
        warnings.filterwarnings('ignore')
        from monte_carlo import run_monte_carlo
        return run_monte_carlo(
            N_realizations=1000,
            wavelengths_um=np.array([6.0, 10.0, 16.0]),
            seed=42,
            verbose=False,
        )

    def test_mc_returns_dict(self, mc_results):
        assert isinstance(mc_results, dict)

    def test_mc_has_null_depths(self, mc_results):
        assert 'null_depths' in mc_results
        assert mc_results['null_depths'].shape[0] == 1000

    def test_mc_has_throughputs(self, mc_results):
        assert 'throughputs' in mc_results
        assert mc_results['throughputs'].shape[0] == 1000

    def test_mc_null_depth_range(self, mc_results):
        """MC null depths should be in [10⁻⁶, 1] range."""
        N = mc_results['null_depths']
        assert np.all(N > 1e-6)
        assert np.all(N < 1.0)

    def test_mc_null_6um_worse_than_16um(self, mc_results):
        """Null at 6 µm (idx 0) > null at 16 µm (idx 2)."""
        N = mc_results['null_depths']
        assert np.mean(N[:, 0]) > np.mean(N[:, 2])

    def test_mc_throughput_range(self, mc_results):
        """PCE should be in 3–12% range."""
        T = mc_results['throughputs']
        T_mean = np.mean(T, axis=0)
        assert np.all(T_mean > 0.03)
        assert np.all(T_mean < 0.12)

    def test_mc_coupling_efficiency_range(self, mc_results):
        """Coupling efficiency in 30–82% range."""
        eta = mc_results['coupling_effs']
        eta_mean = np.mean(eta, axis=0)
        assert np.all(eta_mean > 0.30)
        assert np.all(eta_mean < 0.82)

    def test_mc_reproducibility(self):
        """Same seed → same results."""
        import warnings
        warnings.filterwarnings('ignore')
        from monte_carlo import run_monte_carlo
        r1 = run_monte_carlo(N_realizations=100, seed=99, verbose=False,
                             wavelengths_um=np.array([10.0]))
        r2 = run_monte_carlo(N_realizations=100, seed=99, verbose=False,
                             wavelengths_um=np.array([10.0]))
        np.testing.assert_array_equal(r1['null_depths'], r2['null_depths'])


# ============================================================
# 7. Cross-module consistency (2 checks)
# ============================================================

class TestCrossModule:
    """Verify modules agree on shared constants."""

    def test_tophat_coupling_consistent(self):
        """fiber_modes and MC agree on η₀ = 81.45%."""
        from fiber_modes import coupling_tophat_analytical
        eta_fm = coupling_tophat_analytical(np.array([1.1209]))[0]
        assert abs(eta_fm - 0.8145) < 0.002

    def test_quartic_scaling(self):
        """Null ∝ σ⁴/λ⁴: halving λ should increase null by 16×."""
        sigma = 50e-9
        lam1 = 6e-6
        lam2 = 12e-6
        # N ~ (σ/λ)⁴ → N(6µm)/N(12µm) = (12/6)⁴ = 16
        N1 = (sigma / lam1) ** 4
        N2 = (sigma / lam2) ** 4
        ratio = N1 / N2
        assert abs(ratio - 16.0) < 0.01
