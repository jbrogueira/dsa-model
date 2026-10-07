"""
Normalize A_tfp so BASE-YEAR output equals a target level (default 1), at
fixed theta (_derived.theta), and with --pin-tau-y set the output-tax rate
fiscal.tau_y so that the base-year primary balance equals
fiscal.primary_balance_target_over_Y at the same time.

The two are solved together: tau_y enters the firm's conditions, so it moves
the wage and with it hours, the tax bases and output; A_tfp moves output and
with it every ratio. Each evaluation is one base-year cross-section at a
trial (A_tfp, tau_y). The rate is updated by a secant on the primary-balance
residual (first step at a slope of 0.7: one unit of the rate raises revenue by
one unit of output and lowers the labour-income bases by about a third of
that), A_tfp by the elasticity step and then a secant on |Y - 1|. The
primary balance is the full one of compute_fiscal_ratios: the household-side
balance plus the output tax and the transfer from abroad, less G, defence,
education, the lump-sum transfer and public investment at the level the
transition spends, (delta_g + Gamma_0 - 1) K_g.

The variable named "Y_ss" throughout this script is DETRENDED per-capita
output, Y_t/(Z_t N_t), in the base-year equilibrium: one lifecycle problem at
constant detrended prices, aggregated over the measured 2023 cross-section.
It is not a steady state. With trend growth on there is no stationary level of
output -- levels grow at Gamma_t - 1 = (1+g)(1+n_t) - 1 and per-capita terms
at g -- and with a population still in transition the detrended aggregate is
not constant either; it settles only once demography does. Normalizing the
base-year value to 1 is a units choice.

With public capital on (eta_g != 0), the K_g level in the config is a K_g/Y
target only if base-year output is normalized: Y_ss = 1 makes K_g = K_g/Y
by construction.
Y_ss is endogenous (Y_ss = (Y/L)_ss * L_ss, with L_ss from the household block)
and hours respond to the wage level, so Y_ss is NOT proportional to A_tfp —
this is a genuine 1-D root-find, not a closed-form rescaling.  (With log
consumption the intratemporal FOC is invariant to a proportional scaling of w
and c, so the elasticity step below is near-exact; the remaining curvature
comes from the level objects that do not scale.)

Each evaluation rebuilds equilibrium prices (w, K/L from the firm FOC at the
trial A_tfp) and re-solves + re-simulates the stationary lifecycle via
run_model_moments — the same stationary solve pin_baseline_closure.py uses.
Root-find: elasticity-based first step (Y ~ A_tfp^{1/(1-alpha)} holds only
approximately), then secant steps with a bisection safeguard once a bracket
exists.

After convergence the script prints the SMM target moments at the new A_tfp
(model vs target at fixed theta) so the need for an SMM re-run can be judged.

Usage:
    python normalize_A_tfp.py [--backend jax|numpy] [--config FILE]
                              [--target 1.0] [--tol 1e-4] [--max-iter 12]
                              [--write]

--write stores the solved A_tfp into production.A_tfp of the config and, with
--pin-tau-y, the solved rate into fiscal.tau_y.
"""
import os
import sys
import platform
import json
import copy
import argparse
import dataclasses
import tempfile
import time
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

from calibrate import (load_config, run_model_moments, _compute_ss_aggregates,
                       theta_from_config)

DEFAULT_CONFIG = 'calibration_input_GR.json'

p = argparse.ArgumentParser()
p.add_argument('--backend', default='jax', choices=['jax', 'numpy'])
p.add_argument('--config', default=DEFAULT_CONFIG)
p.add_argument('--target', type=float, default=1.0, help='target base-year output')
p.add_argument('--tol', type=float, default=1e-4,
               help='convergence tolerance on |Y - target|')
p.add_argument('--max-iter', type=int, default=12)
p.add_argument('--write', action='store_true',
               help='write the solved A_tfp into production.A_tfp')
p.add_argument('--pin-tau-y', action='store_true',
               help='also solve fiscal.tau_y for the base-year primary balance target')
p.add_argument('--tol-pb', type=float, default=1e-4,
               help='convergence tolerance on |primary balance - target|')
args = p.parse_args()

with open(args.config) as f:
    raw0 = json.load(f)

theta_dict = raw0.get('_derived', {}).get('theta')
if theta_dict is None:
    sys.exit("No _derived.theta in config — run calibration first.")

alpha = raw0.get('production', {}).get('alpha', 0.33)
A_start = raw0.get('production', {}).get('A_tfp', 1.0)
tau_start = float(raw0.get('fiscal', {}).get('tau_y', 0.0) or 0.0)
pb_target = float(raw0.get('fiscal', {}).get('primary_balance_target_over_Y', 0.0195))

_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
_tmp.close()


def full_primary_balance(ratios, raw):
    """The base-year primary balance over output as the transition books it:
    compute_fiscal_ratios' full balance with public investment at the level
    the transition spends, (delta_g + Gamma_0 - 1) K_g, over this output."""
    from calibrate import _demography_path
    prod, ext, fisc = raw['production'], raw['external_params'], raw.get('fiscal', {})
    n0 = 0.0
    demog = _demography_path(raw)
    if demog is not None:
        d = np.load(demog)
        yrs = list(np.asarray(d['pop_years'], dtype=int))
        n0 = float(np.asarray(d['n_path'])[yrs.index(
            int(raw['transition'].get('current_year', int(d['base_year']))))])
    Gamma0 = (1.0 + float(ext.get('trend_growth', 0.0))) * (1.0 + n0)
    I_g_level = (float(prod.get('delta_g', 0.05)) + Gamma0 - 1.0) * float(prod.get('K_g', 0.0))
    return (ratios['primary_balance_full_over_Y']
            + float(fisc.get('I_g_over_Y', 0.0)) - I_g_level / float(ratios['Y']))


def eval_ss(A_tfp, tau_y=None):
    """Base-year cross-section at trial (A_tfp, tau_y). Returns
    (Y_ss, pb_full, m_model, spec)."""
    from calibrate import compute_fiscal_ratios
    raw = copy.deepcopy(raw0)
    raw['production']['A_tfp'] = float(A_tfp)
    if tau_y is not None:
        raw.setdefault('fiscal', {})['tau_y'] = float(tau_y)
    with open(_tmp.name, 'w') as f:
        json.dump(raw, f)
    loaded = load_config(_tmp.name)
    spec = dataclasses.replace(loaded['spec'], backend=args.backend)
    # theta_from_config fills any parameter the last SMM did not fit from its
    # initial value and says so; indexing theta_dict directly raised KeyError
    # whenever calibration.params gained an entry before the SMM had run.
    theta = theta_from_config(raw, spec, verbose=False)
    m_model, panels = run_model_moments(theta, spec, return_panels=True)
    agg = _compute_ss_aggregates(panels, spec)
    pb = None
    if args.pin_tau_y:
        ratios = compute_fiscal_ratios(panels, spec, loaded['config_data'])
        if 'error' in ratios:
            sys.exit(f"compute_fiscal_ratios failed: {ratios['error']}")
        pb = full_primary_balance(ratios, loaded['config_data'])
    return agg['Y'], pb, m_model, spec


print(f"config={args.config}, backend={args.backend}, target Y_ss={args.target}")
print("theta:", {k: float(v) for k, v in theta_dict.items()})
prod0 = raw0.get('production', {})
print(f"production: A_tfp={A_start}, K_g={prod0.get('K_g')}, "
      f"eta_g={prod0.get('eta_g')}, delta_g={prod0.get('delta_g')}, alpha={alpha}")

t0 = time.time()
history = []          # (A, f) pairs, f = Y - target
tau_hist = []         # (tau_y, pb - target) pairs
best = None           # (score, A, tau, Y, pb, m, spec)


def pb_resid(pb):
    return 0.0 if pb is None else pb - pb_target


def record(A, tau, Y, pb, m, spec):
    global best
    f = Y - args.target
    h = pb_resid(pb)
    history.append((A, f))
    tau_hist.append((tau, h))
    # Both residuals count, each relative to its tolerance.
    score = max(abs(f) / args.tol, abs(h) / args.tol_pb)
    if best is None or score < best[0]:
        best = (score, A, tau, Y, pb, m, spec)
    pb_txt = f"  pb={pb:+.6f}  pb_resid={h:+.2e}" if pb is not None else ""
    print(f"  iter {len(history):2d}: A_tfp={A:.8f}  tau_y={tau:.6f}  Y_ss={Y:.6f}  "
          f"resid={f:+.2e}{pb_txt}  [{time.time()-t0:.0f}s]", flush=True)
    return f, h


def next_tau(tau, h):
    """Secant step on the primary-balance residual; 0.7 a priori."""
    if not args.pin_tau_y:
        return tau
    slope = 0.7
    if len(tau_hist) >= 2:
        (t1, h1), (t2, h2) = tau_hist[-2], tau_hist[-1]
        if t2 != t1 and h2 != h1:
            s_est = (h2 - h1) / (t2 - t1)
            if 0.2 <= s_est <= 2.0:
                slope = s_est
    return tau - h / slope


def done(f, h):
    return abs(f) <= args.tol and abs(h) <= args.tol_pb


print("\nsolving the base-year cross-section at the current A_tfp"
      + (" and tau_y ..." if args.pin_tau_y else " ..."), flush=True)
tau = tau_start if args.pin_tau_y else None
Y, pb, m, spec = eval_ss(A_start, tau)
f, h = record(A_start, tau_start, Y, pb, m, spec)

if not done(f, h):
    # Elasticity-based first step: if Y ~ c*A^{1/(1-alpha)}, the exact fix is
    # A1 = A0*(target/Y0)^{1-alpha}. Endogenous hours make this approximate.
    A_next = A_start * (args.target / Y) ** (1.0 - alpha)
    tau_next = next_tau(tau_start, h)
    while not done(f, h) and len(history) < args.max_iter:
        Y, pb, m, spec = eval_ss(A_next, tau_next if args.pin_tau_y else None)
        f, h = record(A_next, tau_next, Y, pb, m, spec)
        if done(f, h):
            break
        # Secant step from the two most recent points
        (A1, f1), (A2, f2) = history[-2], history[-1]
        if f2 != f1:
            A_sec = A2 - f2 * (A2 - A1) / (f2 - f1)
        else:
            A_sec = A2 * (args.target / (f2 + args.target)) ** (1.0 - alpha)
        # Bisection safeguard: if a sign-change bracket exists and the secant
        # step leaves it, bisect instead.
        pos = [(a, ff) for a, ff in history if ff > 0]
        neg = [(a, ff) for a, ff in history if ff < 0]
        if pos and neg:
            lo = max(a for a, ff in history if ff < 0)
            hi = min(a for a, ff in history if ff > 0)
            if lo > hi:
                lo, hi = hi, lo
            if not (lo < A_sec < hi):
                A_sec = 0.5 * (lo + hi)
        A_next = max(A_sec, 1e-6)
        tau_next = next_tau(tau_next, h)

_, A_star, tau_star, Y_star, pb_star, m_star, spec_star = best
converged = done(Y_star - args.target, pb_resid(pb_star))
pb_txt = (f"   tau_y = {tau_star:.6f}   pb = {pb_star:+.6f} (target {pb_target:+.4f})"
          if args.pin_tau_y else "")
print(f"\n{'CONVERGED' if converged else 'NOT CONVERGED (best point reported)'}: "
      f"A_tfp = {A_star:.8f}   Y_ss = {Y_star:.6f}{pb_txt}   "
      f"(target {args.target}, tol {args.tol:g}, {len(history)} evals, "
      f"{time.time()-t0:.0f}s)")

print("\nSMM target moments at solved A_tfp (fixed theta):")
print(f"  {'moment':<28s} {'target':>10s} {'model':>10s} {'dev %':>8s}")
for mom, mv in zip(spec_star.moments, m_star):
    dev = 100.0 * (mv - mom.value) / mom.value if mom.value != 0 else float('nan')
    print(f"  {mom.compute_key:<28s} {mom.value:>10.4f} {float(mv):>10.4f} {dev:>+8.2f}")

os.unlink(_tmp.name)

if args.write:
    if not converged:
        sys.exit("\nRefusing to --write: root-find did not converge.")
    with open(args.config) as f:
        raw_disk = json.load(f)
    raw_disk.setdefault('production', {})['A_tfp'] = round(float(A_star), 8)
    if args.pin_tau_y:
        raw_disk.setdefault('fiscal', {})['tau_y'] = round(float(tau_star), 6)
    with open(args.config, 'w') as f:
        json.dump(raw_disk, f, indent=2)
    print(f"\nWrote A_tfp={A_star:.8f}"
          + (f" and fiscal.tau_y={tau_star:.6f}" if args.pin_tau_y else "")
          + f" to {args.config}")

print("\nDONE")
