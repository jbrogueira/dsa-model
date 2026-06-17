"""
Backend-equivalence validation: NumPy vs JAX.

Runs the baseline transition kernel (cohort solve + Monte-Carlo simulation +
aggregation + government budget) under both backends and compares the aggregate
paths.  This is the only backend-dependent computation in the fiscal pipeline;
the fiscal layer on top (debt path, NFA residual, bisection) is pure NumPy
arithmetic and is backend-independent, so equivalence here implies equivalence
of the production figures.

Why this is not a machine-precision check
-----------------------------------------
The two backends use different PRNGs (NumPy MT19937 vs JAX ThreeFry) at the
same seed.  The *solved policy functions* match to ~1e-14, but the *simulated
aggregates* differ by Monte-Carlo sampling noise, which falls as 1/sqrt(n_sim).
`simulate_transition` exposes no seed argument (seed fixed at 42 internally), so
we cannot average over seeds.  Instead we run at two n_sim values and check
whether the backend gap *shrinks* with n_sim:

  - gap shrinks ~1/sqrt(n_sim)  -> pure sampling noise -> backends equivalent
  - gap flat and above ~1e-6    -> systematic solver discrepancy -> investigate

This is a quick "test run": one baseline solve per backend per n_sim level, not
the full 26-call fiscal pipeline.  Use --quick to also shrink the solve grids
(n_a, n_y, n_alpha) for a fast smoke test before committing to a GPU run.

Usage
-----
    # On the Linux GPU box (JAX targets CUDA):
    source ~/venvs/jax-arm/bin/activate     # built by setup_jax.sh (jax[cuda12])
    python validate_backends.py --config calibration_input_GR.json --n-sim 1000

    # Fast smoke test (shrunk grids):
    python validate_backends.py --quick --n-sim 400

Confirm the JAX device printed at startup says gpu/cuda.  If it says cpu, check
that JAX_PLATFORM_NAME is not pinned to cpu in the environment.
"""
import os
import platform
# Mirror run_fiscal_figures: avoid the Metal backend on macOS (no float64).
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse
import json
import time

import numpy as np

from calibrate import build_olg_transition


# Macro paths to compare (keys returned by simulate_transition).
MACRO_KEYS = ['r', 'w', 'K', 'L', 'Y', 'C', 'K_g', 'K_domestic', 'NFA']
# Budget lines to compare (keys returned by compute_government_budget_path).
BUDGET_KEYS = ['total_revenue', 'total_spending', 'primary_deficit']


def _path_reldiff(a, b):
    """Path-level relative difference: max_t|a-b| / (max_t|a| + tiny).

    Uses the path scale (not pointwise) so paths that cross zero (NFA,
    primary_deficit) do not blow up the ratio.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    abs_diff = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(a))) + 1e-30
    return abs_diff, abs_diff / scale


def _run_baseline(config_data, backend, n_sim):
    """Build the economy on `backend` and run the baseline transition + budget.

    Returns (macro_dict, budget_dict, elapsed_seconds, device_str).
    """
    economy, paths, T_tr = build_olg_transition(config_data, backend=backend)

    r_path = paths['r_path']
    tax_paths = {k: paths[k] for k in
                 ['tau_c_path', 'tau_l_path', 'tau_p_path', 'tau_k_path',
                  'pension_replacement_path']}

    # I_g must be a LEVEL when eta_g != 0 (I_g->K_g->Y simultaneity forbids the
    # ratio form).  With delta_g=0/K_g const this is the no-net-investment path,
    # matching run_fiscal_figures' warmup call.
    prod = config_data.get('production', {})
    eta_g = float(prod.get('eta_g', 0.0))
    I_g_level = np.full(T_tr, prod.get('delta_g', 0.0) * prod.get('K_g', 0.0))

    # Spending shares of Y(t): backend-independent budget arithmetic, but include
    # them so the primary-deficit comparison is meaningful.
    share_kwargs = dict(
        G_over_Y=paths.get('G_over_Y', 0.13),
        defense_over_Y=paths.get('defense_over_Y', 0.0),
        other_net_over_Y=paths.get('other_net_spending_over_Y', 0.0),
    )
    if eta_g == 0.0:
        # ratio form is allowed; use it (matches run_fiscal_figures config mode)
        share_kwargs['I_g_over_Y'] = paths.get('I_g_over_Y', 0.03)
        ig_kwargs = {}
    else:
        ig_kwargs = dict(I_g_path=I_g_level)

    device = backend
    if backend == 'jax':
        try:
            import jax
            device = ', '.join(str(d) for d in jax.devices())
        except Exception as exc:  # pragma: no cover
            device = f'jax (device query failed: {exc})'

    t0 = time.perf_counter()
    macro = economy.simulate_transition(
        r_path=r_path, n_sim=n_sim, verbose=False,
        **tax_paths, **share_kwargs, **ig_kwargs,
    )
    budget = economy.compute_government_budget_path(n_sim=n_sim, verbose=False)
    elapsed = time.perf_counter() - t0
    return macro, budget, elapsed, device


def _compare(macro_np, budget_np, macro_jx, budget_jx):
    """Return (rows, reldiff_by_path).

    rows = [(name, absdiff, reldiff)]; reldiff_by_path maps name -> reldiff.
    """
    rows = []
    by_path = {}
    for key in MACRO_KEYS:
        if key in macro_np and key in macro_jx:
            ad, rd = _path_reldiff(macro_np[key], macro_jx[key])
            rows.append((key, ad, rd))
            by_path[key] = rd
    for key in BUDGET_KEYS:
        if key in budget_np and key in budget_jx:
            ad, rd = _path_reldiff(budget_np[key], budget_jx[key])
            rows.append((f'budget.{key}', ad, rd))
            by_path[f'budget.{key}'] = rd
    return rows, by_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='calibration_input_GR.json')
    ap.add_argument('--n-sim', type=int, default=1000,
                    help='base Monte-Carlo panel size (default 1000)')
    ap.add_argument('--scale', type=int, default=4,
                    help='second run uses n_sim*scale for the scaling test; '
                         'set 1 to skip (default 4)')
    ap.add_argument('--quick', action='store_true',
                    help='shrink solve grids (n_a, n_y, n_alpha) for a fast '
                         'smoke test')
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='reldiff above which a flat (non-shrinking) gap is '
                         'flagged systematic (default 1e-6)')
    args = ap.parse_args()

    with open(args.config) as f:
        config_data = json.load(f)

    if args.quick:
        m = config_data.setdefault('model', {})
        before = {k: m.get(k) for k in ('n_a', 'n_y', 'n_alpha')}
        m['n_a'] = min(int(m.get('n_a', 100)), 30)
        m['n_y'] = min(int(m.get('n_y', 5)), 3)
        m['n_alpha'] = min(int(m.get('n_alpha', 5)), 2)
        print(f"[quick] grids shrunk: {before} -> "
              f"{{'n_a': {m['n_a']}, 'n_y': {m['n_y']}, 'n_alpha': {m['n_alpha']}}}")

    n_levels = [args.n_sim]
    if args.scale and args.scale > 1:
        n_levels.append(args.n_sim * args.scale)

    print("=" * 70)
    print("Backend equivalence: NumPy vs JAX  (baseline transition kernel)")
    print("=" * 70)

    reldiff_by_n = {}   # n_sim -> {path: reldiff}
    for n_sim in n_levels:
        print(f"\n--- n_sim = {n_sim} ---")
        macro_np, budget_np, t_np, dev_np = _run_baseline(config_data, 'numpy', n_sim)
        print(f"  numpy: {t_np:6.1f}s  [{dev_np}]")
        macro_jx, budget_jx, t_jx, dev_jx = _run_baseline(config_data, 'jax', n_sim)
        print(f"  jax:   {t_jx:6.1f}s  [{dev_jx}]")

        rows, by_path = _compare(macro_np, budget_np, macro_jx, budget_jx)
        reldiff_by_n[n_sim] = by_path
        print(f"  {'path':<22}{'max|Δ|':>14}{'rel diff':>14}")
        for name, ad, rd in rows:
            print(f"  {name:<22}{ad:>14.3e}{rd:>14.3e}")
        print(f"  {'OVERALL max rel diff':<22}{'':>14}"
              f"{max(by_path.values()):>14.3e}")

    # ---- Verdict (per path) ----------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if len(n_levels) < 2:
        o = max(reldiff_by_n[n_levels[0]].values())
        print(f"Single n_sim={n_levels[0]}: overall rel diff = {o:.3e}")
        print("Re-run with --scale 4 to separate Monte-Carlo noise from a "
              "systematic solver discrepancy.")
        return

    n0, n1 = n_levels
    expected = (n0 / n1) ** 0.5  # MC noise should fall ~ sqrt(n0/n1)
    flagged = []
    print(f"per-path shrink test (expect ratio <= {expected:.3f} for MC noise; "
          f"tol={args.tol:.0e}):")
    print(f"  {'path':<22}{'rel@'+str(n0):>12}{'rel@'+str(n1):>12}{'ratio':>10}")
    for name in reldiff_by_n[n0]:
        r0 = reldiff_by_n[n0][name]
        r1 = reldiff_by_n[n1].get(name, r0)
        ratio = r1 / r0 if r0 > 0 else 0.0
        ok = (r1 < args.tol) or (ratio <= expected * 1.8)
        print(f"  {name:<22}{r0:>12.3e}{r1:>12.3e}{ratio:>10.3f}"
              f"{'' if ok else '   <-- FLAG'}")
        if not ok:
            flagged.append(name)

    print()
    if not flagged:
        print("PASS: every path is below tol or shrinks ~1/sqrt(n_sim) -> "
              "differences are Monte-Carlo sampling, not a solver discrepancy. "
              "JAX results are trustworthy.")
    else:
        print(f"FLAG: {len(flagged)} path(s) do NOT shrink with n_sim: "
              f"{', '.join(flagged)}. Possible systematic solver discrepancy; "
              "inspect before trusting JAX counterfactuals.")


if __name__ == '__main__':
    main()
