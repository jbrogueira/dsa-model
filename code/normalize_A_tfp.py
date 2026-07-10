"""
Normalize A_tfp so the INITIAL-STEADY-STATE output equals a target level
(default Y_ss = 1), at fixed theta (_derived.theta).

With public capital on (eta_g != 0), the K_g level in the config is a K_g/Y
target only if Y_ss is normalized: Y_ss = 1 makes K_g = K_g/Y by construction.
Y_ss is endogenous (Y_ss = (Y/L)_ss * L_ss, with L_ss from the household block)
and hours respond to the wage level, so Y_ss is NOT proportional to A_tfp —
this is a genuine 1-D root-find, not a closed-form rescaling.

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

--write stores the solved A_tfp into production.A_tfp of the config. Re-run
pin_baseline_closure.py --write afterwards: the closure constant was pinned at
the old A_tfp and shifts with the output level.
"""
import os, sys, platform, json, copy, argparse, dataclasses, tempfile, time
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

from calibrate import load_config, run_model_moments, _compute_ss_aggregates

DEFAULT_CONFIG = 'calibration_input_GR.json'

p = argparse.ArgumentParser()
p.add_argument('--backend', default='jax', choices=['jax', 'numpy'])
p.add_argument('--config', default=DEFAULT_CONFIG)
p.add_argument('--target', type=float, default=1.0, help='target initial-SS Y')
p.add_argument('--tol', type=float, default=1e-4,
               help='convergence tolerance on |Y - target|')
p.add_argument('--max-iter', type=int, default=12)
p.add_argument('--write', action='store_true',
               help='write the solved A_tfp into production.A_tfp')
args = p.parse_args()

with open(args.config) as f:
    raw0 = json.load(f)

theta_dict = raw0.get('_derived', {}).get('theta')
if theta_dict is None:
    sys.exit("No _derived.theta in config — run calibration first.")

alpha = raw0.get('production', {}).get('alpha', 0.33)
A_start = raw0.get('production', {}).get('A_tfp', 1.0)

_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
_tmp.close()


def eval_ss(A_tfp):
    """Stationary solve at trial A_tfp. Returns (Y_ss, m_model, spec)."""
    raw = copy.deepcopy(raw0)
    raw['production']['A_tfp'] = float(A_tfp)
    with open(_tmp.name, 'w') as f:
        json.dump(raw, f)
    loaded = load_config(_tmp.name)
    spec = dataclasses.replace(loaded['spec'], backend=args.backend)
    theta = np.array([theta_dict[pp.name] for pp in spec.params])
    m_model, panels = run_model_moments(theta, spec, return_panels=True)
    agg = _compute_ss_aggregates(panels, spec)
    return agg['Y'], m_model, spec


print(f"config={args.config}, backend={args.backend}, target Y_ss={args.target}")
print("theta:", {k: float(v) for k, v in theta_dict.items()})
prod0 = raw0.get('production', {})
print(f"production: A_tfp={A_start}, K_g={prod0.get('K_g')}, "
      f"eta_g={prod0.get('eta_g')}, delta_g={prod0.get('delta_g')}, alpha={alpha}")

t0 = time.time()
history = []          # (A, f) pairs, f = Y - target
best = None           # (|f|, A, Y, m, spec)


def record(A, Y, m, spec):
    global best
    f = Y - args.target
    history.append((A, f))
    if best is None or abs(f) < best[0]:
        best = (abs(f), A, Y, m, spec)
    print(f"  iter {len(history):2d}: A_tfp={A:.8f}  Y_ss={Y:.6f}  "
          f"resid={f:+.2e}  [{time.time()-t0:.0f}s]", flush=True)
    return f


print("\nsolving stationary lifecycle at current A_tfp ...", flush=True)
Y, m, spec = eval_ss(A_start)
f = record(A_start, Y, m, spec)

if abs(f) > args.tol:
    # Elasticity-based first step: if Y ~ c*A^{1/(1-alpha)}, the exact fix is
    # A1 = A0*(target/Y0)^{1-alpha}. Endogenous hours make this approximate.
    A_next = A_start * (args.target / Y) ** (1.0 - alpha)
    while abs(f) > args.tol and len(history) < args.max_iter:
        Y, m, spec = eval_ss(A_next)
        f = record(A_next, Y, m, spec)
        if abs(f) <= args.tol:
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

_, A_star, Y_star, m_star, spec_star = best
converged = abs(Y_star - args.target) <= args.tol
print(f"\n{'CONVERGED' if converged else 'NOT CONVERGED (best point reported)'}: "
      f"A_tfp = {A_star:.8f}   Y_ss = {Y_star:.6f}   "
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
    with open(args.config, 'w') as f:
        json.dump(raw_disk, f, indent=2)
    print(f"\nWrote A_tfp={A_star:.8f} to {args.config} (production block)")
    print("Re-run pin_baseline_closure.py --write to re-pin the fiscal closure.")

print("\nDONE")
