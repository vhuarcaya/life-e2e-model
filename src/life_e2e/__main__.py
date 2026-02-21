"""
LIFE E2E Analytical Model — Command-line Entry Point
=====================================================

Run the full analysis pipeline with:

    python -m life_e2e [--quick] [--output-dir DIR]

Options:
    --quick        Run Monte Carlo with N=1,000 (~5 s) instead of 10⁵ (~2 min)
    --output-dir   Directory for figures (default: ./figures)

This reproduces all key results and figures from the companion paper:
V. Huarcaya, "Analytical Throughput, Null Depth, and Surface Tolerance Budget
for the LIFE Nulling Interferometer Combiner," A&A (2026).
"""

import argparse
import os
import sys
import time
import warnings

# Add package directory to sys.path so bare imports (from m1_fiber_coupling ...)
# work the same as when running modules standalone.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


def main():
    parser = argparse.ArgumentParser(
        prog='python -m life_e2e',
        description='LIFE E2E nulling interferometer — analytical beam propagation model',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick run: N=1,000 MC realisations (~5 s) instead of 10^5 (~2 min)',
    )
    parser.add_argument(
        '--output-dir', default='./figures',
        help='Output directory for figures (default: ./figures)',
    )
    args = parser.parse_args()

    N_MC = 1_000 if args.quick else 100_000
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Set output directory for modules that respect LIFE_OUTPUT_DIR
    os.environ['LIFE_OUTPUT_DIR'] = output_dir

    # Save original cwd and switch to output dir (for modules using './')
    original_cwd = os.getcwd()
    os.chdir(output_dir)

    # Suppress CaF₂ extrapolation warnings (expected beyond 9.7 µm)
    warnings.filterwarnings('ignore', message='.*CaF.*Sellmeier.*')

    # Use non-interactive backend
    import matplotlib
    matplotlib.use('Agg')

    print('=' * 70)
    print('  LIFE E2E Nulling Interferometer — Analytical Model')
    print('  DOI: 10.5281/zenodo.18716470')
    print('=' * 70)
    print(f'  Monte Carlo realisations : {N_MC:,}')
    print(f'  Output directory         : {output_dir}')
    print('=' * 70)
    print()

    t_start = time.time()

    # ── Module 1: Fiber Coupling ──────────────────────────────────────
    print('[1/4] Module 1 — Fiber coupling analysis...')
    t0 = time.time()
    from m1_fiber_coupling import run_full_analysis as run_m1
    run_m1()
    print(f'      Done ({time.time() - t0:.1f} s, figures saved)\n')

    # ── Module 2: Throughput Chain ────────────────────────────────────
    print('[2/4] Module 2 — Throughput chain...')
    t0 = time.time()
    from m2_throughput_chain import (make_fig6_waterfall,
                                     make_fig7_throughput_vs_wavelength)
    make_fig6_waterfall()
    make_fig7_throughput_vs_wavelength()
    print(f'      Done ({time.time() - t0:.1f} s, figures saved)\n')

    # ── Module 3: Null Error Budget ───────────────────────────────────
    print('[3/4] Module 3 — Null error propagation...')
    t0 = time.time()
    from m3_null_error_propagation import run_full_analysis as run_m3
    m3_results = run_m3()
    print(f'      Done ({time.time() - t0:.1f} s)\n')

    # ── Monte Carlo ───────────────────────────────────────────────────
    print(f'[4/4] Monte Carlo — {N_MC:,} realisations...')
    t0 = time.time()
    from monte_carlo import run_monte_carlo
    import numpy as np

    results = run_monte_carlo(
        N_realizations=N_MC,
        wavelengths_um=np.array([6.0, 8.0, 10.0, 12.0, 16.0]),
        seed=42,
        verbose=True,
    )
    print(f'      Done ({time.time() - t0:.1f} s, figures saved)\n')

    # ── Summary ───────────────────────────────────────────────────────
    dt_total = time.time() - t_start
    os.chdir(original_cwd)

    n_figures = len([f for f in os.listdir(output_dir) if f.endswith('.png')])

    print('=' * 70)
    print(f'  Complete! {n_figures} figures saved to {output_dir}')
    print(f'  Total time: {dt_total:.1f} s')
    print('=' * 70)


if __name__ == '__main__':
    main()
