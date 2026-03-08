#!/usr/bin/env python3
"""
Reproduce headline results from Huarcaya (2026), Table 7.

Quick verification that the codebase reproduces the paper's key numbers:
  - Analytical null budget from Module 3
  - MC mean null depth and technology gap at 6, 10, 16 µm
  - Photon conversion efficiency at 10 µm

Run time:  ~10 s (10³ realisations) or ~2 min (10⁵ for paper-grade).

Usage
-----
    python examples/reproduce_table7.py              # quick (N=1000)
    python examples/reproduce_table7.py --full       # paper-grade (N=100000)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# -- Adjust path if running from the examples/ directory -------------------
# Works with both flat layout and src/life_e2e/ package layout.
try:
    from life_e2e.monte_carlo import (
        run_monte_carlo,
        get_analytical_reference,
        get_null_requirements,
    )
    from life_e2e.m3_null_error_propagation import compute_null_budget
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'life_e2e'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from monte_carlo import (
        run_monte_carlo,
        get_analytical_reference,
        get_null_requirements,
    )
    from m3_null_error_propagation import compute_null_budget


# -- Paper reference values (Table 7) ------------------------------------
# These are the values a reader should recover with N=100000, seed=42.
PAPER_VALUES = {
    #  λ [µm]: (MC mean null, gap vs Birbacher req)
    6.0:  (1.4e-3,  140.0),
    10.0: (1.9e-4,   21.0),
    16.0: (2.5e-5,    0.3),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='Run 10⁵ realisations (paper-grade, ~2 min)')
    args = parser.parse_args()

    N = 100_000 if args.full else 1_000
    wavelengths = np.array([6.0, 10.0, 16.0])

    print("=" * 70)
    print("LIFE E2E — Reproduce Table 7 headline results")
    print(f"  Realisations: {N:,}  (use --full for paper-grade 10⁵)")
    print(f"  Wavelengths:  {wavelengths} µm")
    print("=" * 70)

    # -- 1. Analytical null budget (Module 3, instant) --------------------
    print("\n--- Module 3: Analytical null budget ---")
    analytical = get_analytical_reference()
    for lam in wavelengths:
        N_anal = analytical[lam]
        print(f"  λ = {lam:5.1f} µm:  N_analytical = {N_anal:.2e}")

    # -- 2. Monte Carlo ---------------------------------------------------
    print(f"\n--- Monte Carlo: {N:,} realisations ---")
    t0 = time.time()
    results = run_monte_carlo(
        N_realizations=N,
        wavelengths_um=wavelengths,
        seed=42,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f} s")

    # -- 3. Technology gap ------------------------------------------------
    req = get_null_requirements()['birbacher']
    null_depths = results['null_depths']
    throughputs = results['throughputs']

    print("\n--- Results vs Paper Table 7 ---")
    print(f"  {'λ [µm]':>8}  {'MC mean':>11}  {'Paper':>11}  "
          f"{'Gap':>7}  {'Paper gap':>10}  {'PCE [%]':>8}")
    print(f"  {'-' * 65}")

    all_ok = True
    for j, lam in enumerate(wavelengths):
        N_mean = np.mean(null_depths[:, j])
        N_req = req.get(lam, 1e-5)
        gap = N_mean / N_req

        paper_null, paper_gap = PAPER_VALUES[lam]
        pce = np.mean(throughputs[:, j]) * 100

        # Tolerance: 50% for N=1000 (statistical), 5% for N=100000
        tol = 0.05 if N >= 100_000 else 0.50
        ok_null = abs(N_mean - paper_null) / paper_null < tol
        ok_gap = abs(gap - paper_gap) / max(paper_gap, 0.1) < tol

        status = "OK" if (ok_null and ok_gap) else "CHECK"
        if status == "CHECK":
            all_ok = False

        print(f"  {lam:8.1f}  {N_mean:11.2e}  {paper_null:11.1e}  "
              f"{gap:6.1f}×  {paper_gap:9.1f}×  {pce:7.1f}%  [{status}]")

    print()
    if all_ok:
        print("  ✓ All values within expected tolerance.")
    elif N < 100_000:
        print("  Note: N=1,000 has high statistical variance.")
        print("  Re-run with --full for paper-grade agreement.")
    else:
        print("  ✗ Some values outside 5% tolerance — investigate.")

    print()


if __name__ == '__main__':
    main()
