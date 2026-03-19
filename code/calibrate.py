"""
SMM calibration infrastructure for the OLG lifecycle model.

Calibrates income process, labor market, and preference parameters via
Simulated Method of Moments using standalone LifecycleModelPerfectForesight
instances (partial equilibrium, fixed prices).

Usage:
    python calibrate.py --n-sim 10000 --maxiter 500 --seed 42
    python calibrate.py --config targets.json --output results.json
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import minimize

from lifecycle_perfect_foresight import LifecycleConfig, LifecycleModelPerfectForesight


# ---------------------------------------------------------------------------
# 1. SimPanel — named wrapper for the 21-tuple simulation output
# ---------------------------------------------------------------------------

class SimPanel(NamedTuple):
    a_sim: np.ndarray           # (T, n_sim) assets
    c_sim: np.ndarray           # (T, n_sim) consumption
    y_sim: np.ndarray           # (T, n_sim) raw income state value
    h_sim: np.ndarray           # (T, n_sim) health productivity
    h_idx_sim: np.ndarray       # (T, n_sim) health state index
    effective_y_sim: np.ndarray # (T, n_sim) effective earnings (after employment/health)
    employed_sim: np.ndarray    # (T, n_sim) bool
    ui_sim: np.ndarray          # (T, n_sim) UI benefits
    m_sim: np.ndarray           # (T, n_sim) medical costs
    oop_m_sim: np.ndarray       # (T, n_sim) out-of-pocket medical
    gov_m_sim: np.ndarray       # (T, n_sim) government medical
    tax_c_sim: np.ndarray       # (T, n_sim) consumption tax
    tax_l_sim: np.ndarray       # (T, n_sim) labor tax
    tax_p_sim: np.ndarray       # (T, n_sim) payroll tax
    tax_k_sim: np.ndarray       # (T, n_sim) capital tax
    avg_earnings_sim: np.ndarray  # (T, n_sim)
    pension_sim: np.ndarray     # (T, n_sim)
    retired_sim: np.ndarray     # (T, n_sim) bool
    l_sim: np.ndarray           # (T, n_sim) labor hours
    alive_sim: np.ndarray       # (T, n_sim) bool
    bequest_sim: np.ndarray     # (T, n_sim)


def wrap_sim_output(tup):
    """Convert the raw 21-tuple from model.simulate() to a SimPanel."""
    return SimPanel(*tup)


# ---------------------------------------------------------------------------
# 2. Moment computation functions
# ---------------------------------------------------------------------------

def compute_gini(x, weights=None):
    """Gini coefficient of array *x*. Supports optional sample weights."""
    x = np.asarray(x, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        # Remove entries with zero weight
        mask = weights > 0
        x, weights = x[mask], weights[mask]
    if len(x) == 0:
        return 0.0

    # Sort by value
    if weights is None:
        xs = np.sort(x)
        n = len(xs)
        idx = np.arange(1, n + 1)
        return (2.0 * np.sum(idx * xs) / (n * np.sum(xs)) - (n + 1) / n)
    else:
        order = np.argsort(x)
        xs = x[order]
        ws = weights[order]
        cum_w = np.cumsum(ws)
        total_w = cum_w[-1]
        cum_xw = np.cumsum(xs * ws)
        total_xw = cum_xw[-1]
        if total_xw == 0:
            return 0.0
        # Weighted Gini
        return 1.0 - 2.0 * np.sum(ws * cum_xw) / (total_w * total_xw) + 1.0 / total_w


def compute_earnings_variance_by_age(effective_y_sim, employed_sim, alive_sim,
                                     retirement_age):
    """Variance of log earnings at each working age, excluding unemployed/dead.

    Returns array of shape (retirement_age,). Ages with fewer than 2 employed
    alive agents get NaN.
    """
    T = effective_y_sim.shape[0]
    n_ages = min(retirement_age, T)
    var_by_age = np.full(n_ages, np.nan)
    for t in range(n_ages):
        mask = alive_sim[t] & employed_sim[t] & (effective_y_sim[t] > 0)
        if np.sum(mask) >= 2:
            log_earn = np.log(effective_y_sim[t, mask])
            var_by_age[t] = np.var(log_earn)
    return var_by_age


def compute_wealth_gini(a_sim, alive_sim, ages=None):
    """Gini of assets among alive agents. Optionally restrict to *ages* (list)."""
    if ages is None:
        mask = alive_sim.astype(bool)
        vals = a_sim[mask]
    else:
        vals = []
        for t in ages:
            if t < a_sim.shape[0]:
                m = alive_sim[t].astype(bool)
                vals.append(a_sim[t, m])
        vals = np.concatenate(vals) if vals else np.array([])
    if len(vals) == 0:
        return 0.0
    return compute_gini(vals)


def compute_zero_wealth_fraction(a_sim, alive_sim, threshold=0.0):
    """Fraction of alive agents with assets <= threshold."""
    mask = alive_sim.astype(bool)
    vals = a_sim[mask]
    if len(vals) == 0:
        return 0.0
    return np.mean(vals <= threshold)


def compute_wealth_to_income_by_age(a_sim, effective_y_sim, alive_sim,
                                    employed_sim, retirement_age):
    """Median wealth-to-income ratio by working age.

    Returns array of shape (retirement_age,). Ages with no employed alive agents
    get NaN.
    """
    T = a_sim.shape[0]
    n_ages = min(retirement_age, T)
    ratio_by_age = np.full(n_ages, np.nan)
    for t in range(n_ages):
        mask = alive_sim[t] & employed_sim[t] & (effective_y_sim[t] > 0)
        if np.sum(mask) > 0:
            ratio_by_age[t] = np.median(a_sim[t, mask] / effective_y_sim[t, mask])
    return ratio_by_age


def compute_unemployment_rate(employed_sim, retired_sim, alive_sim):
    """Fraction unemployed among alive non-retired agents."""
    mask = alive_sim.astype(bool) & ~retired_sim.astype(bool)
    if np.sum(mask) == 0:
        return 0.0
    return 1.0 - np.mean(employed_sim[mask])


def compute_health_distribution_by_age(h_idx_sim, alive_sim, n_h):
    """Health state shares by age. Returns (T, n_h) array."""
    T = h_idx_sim.shape[0]
    dist = np.zeros((T, n_h))
    for t in range(T):
        mask = alive_sim[t].astype(bool)
        n_alive = np.sum(mask)
        if n_alive > 0:
            for h in range(n_h):
                dist[t, h] = np.sum(h_idx_sim[t, mask] == h) / n_alive
    return dist


def compute_average_hours(l_sim, employed_sim, alive_sim, retired_sim):
    """Mean hours among alive, employed, non-retired agents."""
    mask = alive_sim.astype(bool) & employed_sim.astype(bool) & ~retired_sim.astype(bool)
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(l_sim[mask])


def compute_consumption_gini(c_sim, alive_sim):
    """Consumption Gini pooled across all ages."""
    mask = alive_sim.astype(bool)
    vals = c_sim[mask]
    if len(vals) == 0:
        return 0.0
    return compute_gini(vals)


# ---------------------------------------------------------------------------
# 3. Parameter mapping
# ---------------------------------------------------------------------------

@dataclass
class CalibrationParam:
    """One calibration parameter with bounds."""
    name: str          # e.g. 'rho_y'
    path: str          # e.g. 'edu_params.*.rho_y' or 'job_finding_rate'
    lower: float       # lower bound
    upper: float       # upper bound
    initial: float     # starting guess


def apply_params(config, params, theta):
    """Apply parameter vector *theta* to *config*, returning a new LifecycleConfig.

    Does not mutate the original config. Handles:
    - 'edu_params.*.field' — sets field for ALL education types
    - 'edu_params.low.field' — sets field for one type
    - 'field' — sets top-level LifecycleConfig field
    """
    # Deep copy edu_params so we don't mutate the original
    new_edu = copy.deepcopy(config.edu_params)
    changes = {}

    for p, val in zip(params, theta):
        parts = p.path.split('.')
        if parts[0] == 'edu_params':
            edu_key = parts[1]  # '*' or a specific type
            field_name = parts[2]
            if edu_key == '*':
                for edu_type in new_edu:
                    new_edu[edu_type][field_name] = val
            else:
                new_edu[edu_key][field_name] = val
        else:
            changes[p.path] = val

    changes['edu_params'] = new_edu
    return config._replace(**changes)


def theta_to_unbounded(theta, params):
    """Logit transform: [lower, upper] -> R."""
    x = np.empty(len(theta))
    for i, (val, p) in enumerate(zip(theta, params)):
        # Clamp to avoid log(0)
        t = (val - p.lower) / (p.upper - p.lower)
        t = np.clip(t, 1e-12, 1.0 - 1e-12)
        x[i] = np.log(t / (1.0 - t))
    return x


def unbounded_to_theta(x, params):
    """Inverse sigmoid: R -> [lower, upper]."""
    theta = np.empty(len(x))
    for i, (xi, p) in enumerate(zip(x, params)):
        s = 1.0 / (1.0 + np.exp(-xi))
        theta[i] = p.lower + s * (p.upper - p.lower)
    return theta


# ---------------------------------------------------------------------------
# 4. Target moments
# ---------------------------------------------------------------------------

@dataclass
class TargetMoment:
    """One empirical target moment."""
    name: str              # identifier
    value: float           # empirical value
    weight: float = 1.0    # diagonal weight (e.g. 1/variance)
    compute_key: str = ''  # key into moment computation dispatch


# ---------------------------------------------------------------------------
# 5. CalibrationSpec — groups everything
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSpec:
    """Full specification for an SMM calibration."""
    params: list          # list of CalibrationParam
    moments: list         # list of TargetMoment
    education_shares: dict = field(default_factory=lambda: {
        'low': 0.3, 'medium': 0.5, 'high': 0.2
    })
    base_config: LifecycleConfig = field(default_factory=LifecycleConfig)
    n_sim: int = 10_000
    seed: int = 42
    r: float = 0.03
    w: float = 1.0


# ---------------------------------------------------------------------------
# 6. Core calibration loop
# ---------------------------------------------------------------------------

# Dispatch table: compute_key -> function(panels, spec) -> float
# panels is dict[edu_type -> SimPanel], spec is CalibrationSpec

def _moment_wealth_gini(panels, spec):
    """Wealth Gini pooled across education types."""
    all_a, all_alive = [], []
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        n = panel.a_sim.shape[1]
        w = np.full(np.sum(panel.alive_sim.astype(bool)), share / n)
        all_a.append(panel.a_sim[panel.alive_sim.astype(bool)])
        all_alive.append(w)
    vals = np.concatenate(all_a)
    weights = np.concatenate(all_alive)
    return compute_gini(vals, weights)


def _moment_zero_wealth_fraction(panels, spec):
    """Zero-wealth fraction pooled."""
    total, zero = 0.0, 0.0
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        mask = panel.alive_sim.astype(bool)
        n_alive = np.sum(mask)
        n_zero = np.sum(panel.a_sim[mask] <= 0.0)
        total += share * n_alive
        zero += share * n_zero
    return zero / total if total > 0 else 0.0


def _moment_earnings_var_slope(panels, spec):
    """Slope of variance of log earnings by age (pooled across types)."""
    # Average across education types
    ret_age = spec.base_config.retirement_age
    pooled = np.zeros(ret_age)
    total_w = 0.0
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        v = compute_earnings_variance_by_age(
            panel.effective_y_sim, panel.employed_sim,
            panel.alive_sim, ret_age)
        valid = ~np.isnan(v)
        pooled[valid] += share * v[valid]
        total_w += share
    pooled /= total_w
    # Slope via OLS on non-NaN entries
    valid = ~np.isnan(pooled) & (pooled > 0)
    if np.sum(valid) < 2:
        return 0.0
    ages = np.where(valid)[0]
    vals = pooled[valid]
    slope = np.polyfit(ages, vals, 1)[0]
    return slope


def _moment_earnings_var_mean(panels, spec):
    """Mean variance of log earnings across working ages (pooled)."""
    ret_age = spec.base_config.retirement_age
    pooled = np.zeros(ret_age)
    total_w = 0.0
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        v = compute_earnings_variance_by_age(
            panel.effective_y_sim, panel.employed_sim,
            panel.alive_sim, ret_age)
        valid = ~np.isnan(v)
        pooled[valid] += share * v[valid]
        total_w += share
    pooled /= total_w
    valid = ~np.isnan(pooled) & (pooled > 0)
    if np.sum(valid) == 0:
        return 0.0
    return np.mean(pooled[valid])


def _moment_unemployment_rate(panels, spec):
    """Unemployment rate pooled."""
    total, unemp = 0.0, 0.0
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        mask = panel.alive_sim.astype(bool) & ~panel.retired_sim.astype(bool)
        n = np.sum(mask)
        n_unemp = np.sum(~panel.employed_sim[mask])
        total += share * n
        unemp += share * n_unemp
    return unemp / total if total > 0 else 0.0


def _moment_average_hours(panels, spec):
    """Average hours pooled across education types."""
    total, hours_sum = 0.0, 0.0
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        mask = (panel.alive_sim.astype(bool) &
                panel.employed_sim.astype(bool) &
                ~panel.retired_sim.astype(bool))
        n = np.sum(mask)
        total += share * n
        hours_sum += share * np.sum(panel.l_sim[mask])
    return hours_sum / total if total > 0 else 0.0


def _moment_consumption_gini(panels, spec):
    """Consumption Gini pooled."""
    all_c = []
    for edu, panel in panels.items():
        mask = panel.alive_sim.astype(bool)
        all_c.append(panel.c_sim[mask])
    vals = np.concatenate(all_c)
    if len(vals) == 0:
        return 0.0
    return compute_gini(vals)


MOMENT_DISPATCH = {
    'wealth_gini': _moment_wealth_gini,
    'zero_wealth_fraction': _moment_zero_wealth_fraction,
    'earnings_var_slope': _moment_earnings_var_slope,
    'earnings_var_mean': _moment_earnings_var_mean,
    'unemployment_rate': _moment_unemployment_rate,
    'average_hours': _moment_average_hours,
    'consumption_gini': _moment_consumption_gini,
}


def run_model_moments(theta, spec):
    """Solve + simulate for each education type, compute target moments.

    Returns 1D array of model moments in the same order as spec.moments.
    """
    config = apply_params(spec.base_config, spec.params, theta)

    # Build per-education-type panels
    panels = {}
    for edu_type in spec.education_shares:
        cfg = config._replace(
            education_type=edu_type,
            r_path=np.full(config.T, spec.r),
            w_path=np.full(config.T, spec.w),
        )
        model = LifecycleModelPerfectForesight(cfg, verbose=False)
        model.solve(verbose=False)
        raw = model.simulate(n_sim=spec.n_sim, seed=spec.seed)
        panels[edu_type] = wrap_sim_output(raw)

    # Compute each target moment
    m_model = np.empty(len(spec.moments))
    for i, mom in enumerate(spec.moments):
        fn = MOMENT_DISPATCH.get(mom.compute_key)
        if fn is None:
            raise ValueError(f"Unknown compute_key: {mom.compute_key!r}")
        m_model[i] = fn(panels, spec)
    return m_model


def smm_objective(x_unbounded, spec):
    """SMM objective: weighted distance between data and model moments.

    Returns scalar Q = (m_data - m_model)' W (m_data - m_model) where W is
    diagonal with weights from spec.moments.
    """
    theta = unbounded_to_theta(x_unbounded, spec.params)
    m_model = run_model_moments(theta, spec)
    m_data = np.array([m.value for m in spec.moments])
    w = np.array([m.weight for m in spec.moments])
    diff = m_data - m_model
    return float(diff @ np.diag(w) @ diff)


def calibrate(spec, maxiter=500, tol=1e-6, verbose=True):
    """Run SMM calibration via Nelder-Mead.

    Returns dict with keys: theta, objective, model_moments, data_moments,
    convergence, history, elapsed_seconds.
    """
    theta0 = np.array([p.initial for p in spec.params])
    x0 = theta_to_unbounded(theta0, spec.params)

    history = []
    _last_obj = [None]  # mutable container for closure

    def objective_wrapper(x_unbounded, spec):
        val = smm_objective(x_unbounded, spec)
        _last_obj[0] = val
        return val

    def callback(xk):
        theta_k = unbounded_to_theta(xk, spec.params)
        obj = _last_obj[0]
        history.append({'theta': theta_k.tolist(), 'objective': obj})
        if verbose and len(history) % 10 == 0:
            param_str = ', '.join(
                f'{p.name}={v:.6f}' for p, v in zip(spec.params, theta_k))
            print(f"  iter {len(history):4d}  obj={obj:.8f}  {param_str}")

    t0 = time.time()
    if verbose:
        print(f"Starting SMM calibration: {len(spec.params)} params, "
              f"{len(spec.moments)} moments, n_sim={spec.n_sim}")

    result = minimize(
        objective_wrapper,
        x0,
        args=(spec,),
        method='Nelder-Mead',
        callback=callback,
        options={
            'maxiter': maxiter,
            'xatol': tol,
            'fatol': tol,
            'adaptive': True,
        },
    )

    elapsed = time.time() - t0
    theta_opt = unbounded_to_theta(result.x, spec.params)
    m_model = run_model_moments(theta_opt, spec)
    m_data = np.array([m.value for m in spec.moments])

    if verbose:
        print(f"\nCalibration finished in {elapsed:.1f}s "
              f"({result.nit} iterations, {result.nfev} evaluations)")

    return {
        'theta': theta_opt,
        'objective': result.fun,
        'model_moments': m_model,
        'data_moments': m_data,
        'convergence': result.success,
        'message': result.message,
        'history': history,
        'elapsed_seconds': elapsed,
    }


# ---------------------------------------------------------------------------
# 7. Diagnostics
# ---------------------------------------------------------------------------

def print_calibration_results(result, spec):
    """Print a summary table of calibration results."""
    print("\n" + "=" * 72)
    print("CALIBRATION RESULTS")
    print("=" * 72)

    print(f"\nObjective: {result['objective']:.8f}")
    print(f"Converged: {result['convergence']}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")

    print(f"\n{'Parameter':<25} {'Initial':>10} {'Calibrated':>12} "
          f"{'Lower':>8} {'Upper':>8}")
    print("-" * 72)
    for p, v in zip(spec.params, result['theta']):
        print(f"{p.name:<25} {p.initial:>10.6f} {v:>12.6f} "
              f"{p.lower:>8.4f} {p.upper:>8.4f}")

    print(f"\n{'Moment':<25} {'Data':>10} {'Model':>10} {'% Dev':>8} "
          f"{'Weight':>8}")
    print("-" * 72)
    for m, mv, dv in zip(spec.moments, result['model_moments'],
                         result['data_moments']):
        pct = 100 * (mv - dv) / abs(dv) if abs(dv) > 1e-12 else 0.0
        print(f"{m.name:<25} {m.value:>10.4f} {mv:>10.4f} "
              f"{pct:>7.2f}% {m.weight:>8.2f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 8. Default calibration specification
# ---------------------------------------------------------------------------

def default_spec(n_sim=10_000, seed=42):
    """Build a default CalibrationSpec for quick testing."""
    params = [
        CalibrationParam('rho_y', 'edu_params.*.rho_y', 0.80, 0.995, 0.97),
        CalibrationParam('sigma_y', 'edu_params.*.sigma_y', 0.005, 0.10, 0.03),
        CalibrationParam('job_finding_rate', 'job_finding_rate', 0.1, 0.9, 0.5),
    ]
    moments = [
        TargetMoment('earnings_var_slope', 0.005, 1.0, 'earnings_var_slope'),
        TargetMoment('earnings_var_mean', 0.10, 1.0, 'earnings_var_mean'),
        TargetMoment('wealth_gini', 0.80, 1.0, 'wealth_gini'),
        TargetMoment('unemployment_rate', 0.06, 1.0, 'unemployment_rate'),
    ]
    base_config = LifecycleConfig(
        T=60, retirement_age=45, n_a=100, n_y=5, n_h=1,
        beta=0.96, gamma=2.0, r_default=0.03, w_default=1.0,
    )
    return CalibrationSpec(
        params=params,
        moments=moments,
        base_config=base_config,
        n_sim=n_sim,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SMM calibration for OLG lifecycle model')
    parser.add_argument('--n-sim', type=int, default=10_000)
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--test', action='store_true',
                        help='Use tiny model for quick smoke test')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON file with target moments and initial guesses')
    parser.add_argument('--output', type=str, default=None,
                        help='JSON file to save results')
    args = parser.parse_args()

    if args.test:
        base_config = LifecycleConfig(
            T=10, retirement_age=7, n_a=15, n_y=3, n_h=1,
            beta=0.96, gamma=2.0,
        )
        spec = CalibrationSpec(
            params=[
                CalibrationParam('sigma_y', 'edu_params.*.sigma_y', 0.01, 0.10, 0.03),
            ],
            moments=[
                TargetMoment('wealth_gini', 0.50, 1.0, 'wealth_gini'),
            ],
            base_config=base_config,
            education_shares={'medium': 1.0},
            n_sim=min(args.n_sim, 200),
            seed=args.seed,
        )
        args.maxiter = min(args.maxiter, 10)
    elif args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        params = [CalibrationParam(**p) for p in cfg['params']]
        moments = [TargetMoment(**m) for m in cfg['moments']]
        base_kwargs = cfg.get('base_config', {})
        base_config = LifecycleConfig(**base_kwargs)
        spec = CalibrationSpec(
            params=params,
            moments=moments,
            base_config=base_config,
            education_shares=cfg.get('education_shares',
                                     {'low': 0.3, 'medium': 0.5, 'high': 0.2}),
            n_sim=args.n_sim,
            seed=args.seed,
        )
    else:
        spec = default_spec(n_sim=args.n_sim, seed=args.seed)

    result = calibrate(spec, maxiter=args.maxiter, tol=args.tol)
    print_calibration_results(result, spec)

    if args.output:
        out = {
            'theta': result['theta'].tolist(),
            'objective': result['objective'],
            'model_moments': result['model_moments'].tolist(),
            'data_moments': result['data_moments'].tolist(),
            'convergence': bool(result['convergence']),
            'message': result['message'],
            'elapsed_seconds': result['elapsed_seconds'],
            'params': [{'name': p.name, 'path': p.path, 'lower': p.lower,
                         'upper': p.upper, 'initial': p.initial}
                        for p in spec.params],
            'moments': [{'name': m.name, 'value': m.value, 'weight': m.weight,
                          'compute_key': m.compute_key}
                         for m in spec.moments],
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
