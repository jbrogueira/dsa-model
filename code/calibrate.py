"""
SMM calibration infrastructure for the OLG lifecycle model.

Calibrates income process, labor market, and preference parameters via
Simulated Method of Moments using standalone LifecycleModelPerfectForesight
instances (partial equilibrium, fixed prices).

Country-specific data is read from a JSON input file (see calibration_input_GR.json
for the schema). No country-specific logic lives in this module.

Usage:
    python calibrate.py --config calibration_input_GR.json
    python calibrate.py --config calibration_input_GR.json --n-sim 20000
    python calibrate.py --test  # tiny smoke test
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import minimize, differential_evolution

from lifecycle_perfect_foresight import LifecycleConfig, LifecycleModelPerfectForesight

try:
    from lifecycle_jax import LifecycleModelJAX
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


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
        # Weighted Gini (Lerman & Yitzhaki 1989)
        return 1.0 - 2.0 * np.sum(ws * cum_xw) / (total_w * total_xw)


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
    age_weights: Optional[np.ndarray] = None  # (T,) stationary age distribution
    backend: str = 'numpy'  # 'numpy' or 'jax'
    production: dict = field(default_factory=lambda: {
        'alpha': 0.33, 'delta': 0.07, 'A_tfp': 1.0,
        'K_g': 0.0, 'eta_g': 0.0, 'K_over_L': None,
    })


# ---------------------------------------------------------------------------
# 6. Core calibration loop
# ---------------------------------------------------------------------------

# Dispatch table: compute_key -> function(panels, spec) -> float
# panels is dict[edu_type -> SimPanel], spec is CalibrationSpec


def _agent_weights(panel, spec, edu, mask=None):
    """Per-(t,i) weights for a single education type, incorporating age weights.

    *mask* is (T, n_sim) bool selecting which entries to include.
    Returns (values_mask, weights) where values_mask indexes the flat panel and
    weights has the same length as values_mask.sum().
    """
    T, n_sim = panel.a_sim.shape
    alive = panel.alive_sim.astype(bool)
    if mask is not None:
        alive = alive & mask
    share = spec.education_shares[edu]
    # Age weights: omega(t) per period, spread equally across n_sim agents
    if spec.age_weights is not None:
        aw = spec.age_weights[:T]
    else:
        aw = np.ones(T) / T
    # Build (T, n_sim) weight array, then mask
    w_grid = (share * aw[:, None] / n_sim) * np.ones((1, n_sim))
    return alive, w_grid[alive]


def _pool_weighted(panels, spec, field, mask_fn=None):
    """Pool a SimPanel field across education types with age + education weights.

    *field*: attribute name on SimPanel (e.g. 'a_sim').
    *mask_fn*: optional callable(panel) -> (T, n_sim) bool mask.
    Returns (values, weights) arrays.
    """
    all_vals, all_w = [], []
    for edu, panel in panels.items():
        mask = mask_fn(panel) if mask_fn else None
        alive, w = _agent_weights(panel, spec, edu, mask)
        all_vals.append(getattr(panel, field)[alive])
        all_w.append(w)
    return np.concatenate(all_vals), np.concatenate(all_w)


def _moment_wealth_gini(panels, spec):
    """Wealth Gini pooled across education types (age-weighted)."""
    vals, weights = _pool_weighted(panels, spec, 'a_sim')
    return compute_gini(vals, weights)


def _moment_zero_wealth_fraction(panels, spec):
    """Zero-wealth fraction pooled (age-weighted)."""
    vals, weights = _pool_weighted(panels, spec, 'a_sim')
    if len(vals) == 0:
        return 0.0
    return float(np.sum(weights[vals <= 0.0]) / np.sum(weights))


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
    """Unemployment rate pooled (age-weighted)."""
    def _working(p):
        return p.alive_sim.astype(bool) & ~p.retired_sim.astype(bool)
    # Employed among working-age alive
    emp_vals, emp_w = _pool_weighted(panels, spec, 'employed_sim', _working)
    if len(emp_vals) == 0:
        return 0.0
    return 1.0 - float(np.sum(emp_vals * emp_w) / np.sum(emp_w))


def _moment_average_hours(panels, spec):
    """Average hours pooled (age-weighted)."""
    def _employed(p):
        return (p.alive_sim.astype(bool) &
                p.employed_sim.astype(bool) &
                ~p.retired_sim.astype(bool))
    vals, weights = _pool_weighted(panels, spec, 'l_sim', _employed)
    if len(vals) == 0:
        return 0.0
    return float(np.sum(vals * weights) / np.sum(weights))


def _moment_consumption_gini(panels, spec):
    """Consumption Gini pooled (age-weighted)."""
    vals, weights = _pool_weighted(panels, spec, 'c_sim')
    if len(vals) == 0:
        return 0.0
    return compute_gini(vals, weights)


def _moment_income_gini(panels, spec):
    """Gini of total income (earnings + pensions + UI), age-weighted."""
    all_inc, all_w = [], []
    for edu, panel in panels.items():
        alive, w = _agent_weights(panel, spec, edu)
        income = (panel.effective_y_sim[alive] +
                  panel.pension_sim[alive] +
                  panel.ui_sim[alive])
        all_inc.append(income)
        all_w.append(w)
    return compute_gini(np.concatenate(all_inc), np.concatenate(all_w))


def _moment_earnings_gini(panels, spec):
    """Gini of effective earnings among employed non-retired, age-weighted."""
    def _employed(p):
        return (p.alive_sim.astype(bool) &
                p.employed_sim.astype(bool) &
                ~p.retired_sim.astype(bool))
    vals, weights = _pool_weighted(panels, spec, 'effective_y_sim', _employed)
    return compute_gini(vals, weights)


def _moment_mean_assets(panels, spec):
    """Age-weighted mean assets among alive agents."""
    vals, weights = _pool_weighted(panels, spec, 'a_sim')
    if len(vals) == 0:
        return 0.0
    return float(np.sum(vals * weights) / np.sum(weights))


def _moment_median_wealth_to_income(panels, spec):
    """Pooled median wealth-to-income among employed non-retired."""
    def _employed_pos(p):
        return (p.alive_sim.astype(bool) &
                p.employed_sim.astype(bool) &
                ~p.retired_sim.astype(bool) &
                (p.effective_y_sim > 0))
    a_vals, _ = _pool_weighted(panels, spec, 'a_sim', _employed_pos)
    y_vals, _ = _pool_weighted(panels, spec, 'effective_y_sim', _employed_pos)
    if len(a_vals) == 0:
        return 0.0
    return float(np.median(a_vals / y_vals))


def _moment_p90_p10_income(panels, spec):
    """P90/P10 ratio of total income among alive agents."""
    all_inc = []
    for edu, panel in panels.items():
        alive = panel.alive_sim.astype(bool)
        income = (panel.effective_y_sim[alive] +
                  panel.pension_sim[alive] +
                  panel.ui_sim[alive])
        all_inc.append(income)
    vals = np.concatenate(all_inc)
    vals = vals[vals > 0]
    if len(vals) < 10:
        return 0.0
    p90, p10 = np.percentile(vals, 90), np.percentile(vals, 10)
    return p90 / p10 if p10 > 0 else 0.0


def _moment_mean_consumption(panels, spec):
    """Age-weighted mean consumption among alive agents."""
    vals, weights = _pool_weighted(panels, spec, 'c_sim')
    if len(vals) == 0:
        return 0.0
    return float(np.sum(vals * weights) / np.sum(weights))


def _compute_ss_aggregates(panels, spec):
    """Age-weighted steady-state aggregates pooled across education types.

    Same convention as compute_fiscal_ratios: per-period cross-sectional means
    among alive, weighted by education share and stationary age weights, summed
    over ages. Returns dict with keys for income components, taxes, transfers,
    plus L, K_domestic, Y derived from production primitives in spec.production.
    """
    T = spec.base_config.T
    aw = spec.age_weights if spec.age_weights is not None else np.ones(T) / T

    keys = ['labor_income', 'consumption', 'assets', 'pension', 'ui',
            'oop_health', 'gov_health', 'tax_c', 'tax_l', 'tax_p', 'tax_k']
    agg = {k: 0.0 for k in keys}
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        alive = panel.alive_sim.astype(bool)
        for t in range(T):
            a_t = alive[t]
            if not np.any(a_t):
                continue
            wt = share * aw[t]
            agg['labor_income'] += wt * float(np.mean(panel.effective_y_sim[t, a_t]))
            agg['consumption']  += wt * float(np.mean(panel.c_sim[t, a_t]))
            agg['assets']       += wt * float(np.mean(panel.a_sim[t, a_t]))
            agg['pension']      += wt * float(np.mean(panel.pension_sim[t, a_t]))
            agg['ui']           += wt * float(np.mean(panel.ui_sim[t, a_t]))
            agg['oop_health']   += wt * float(np.mean(panel.oop_m_sim[t, a_t]))
            agg['gov_health']   += wt * float(np.mean(panel.gov_m_sim[t, a_t]))
            agg['tax_c']        += wt * float(np.mean(panel.tax_c_sim[t, a_t]))
            agg['tax_l']        += wt * float(np.mean(panel.tax_l_sim[t, a_t]))
            agg['tax_p']        += wt * float(np.mean(panel.tax_p_sim[t, a_t]))
            agg['tax_k']        += wt * float(np.mean(panel.tax_k_sim[t, a_t]))

    prod = spec.production or {}
    alpha = prod.get('alpha', 0.33)
    A_tfp = prod.get('A_tfp', 1.0)
    K_g = prod.get('K_g', 0.0)
    eta_g = prod.get('eta_g', 0.0)
    K_g_factor = K_g ** eta_g if (K_g > 0 and eta_g > 0) else 1.0

    L = agg['labor_income'] / spec.w if spec.w > 0 else 0.0
    K_over_L = prod.get('K_over_L') or 0.0
    K_domestic = K_over_L * L
    Y = A_tfp * K_g_factor * K_domestic ** alpha * L ** (1.0 - alpha) if L > 0 else 0.0

    out = dict(agg)
    out['L'] = L
    out['K_domestic'] = K_domestic
    out['Y'] = Y
    return out


def _moment_A_over_Y(panels, spec):
    """Aggregate household assets divided by SS output."""
    agg = _compute_ss_aggregates(panels, spec)
    return agg['assets'] / agg['Y'] if agg['Y'] > 0 else 0.0


def _moment_K_over_Y(panels, spec):
    """Domestic capital (firm-FOC pinned) divided by SS output."""
    agg = _compute_ss_aggregates(panels, spec)
    return agg['K_domestic'] / agg['Y'] if agg['Y'] > 0 else 0.0


def _moment_C_over_Y(panels, spec):
    """Aggregate consumption divided by SS output."""
    agg = _compute_ss_aggregates(panels, spec)
    return agg['consumption'] / agg['Y'] if agg['Y'] > 0 else 0.0


def _moment_labor_share(panels, spec):
    """Labor share w*L/Y. In Cobb-Douglas this pins mechanically to 1-alpha."""
    agg = _compute_ss_aggregates(panels, spec)
    if agg['Y'] <= 0:
        return 0.0
    return spec.w * agg['L'] / agg['Y']


def _moment_tax_revenue_over_Y(panels, spec):
    """Total tax revenue (c + l + p + k) divided by SS output."""
    agg = _compute_ss_aggregates(panels, spec)
    if agg['Y'] <= 0:
        return 0.0
    return (agg['tax_c'] + agg['tax_l'] + agg['tax_p'] + agg['tax_k']) / agg['Y']


def _moment_pensions_over_Y(panels, spec):
    """Aggregate pension expenditure / SS output."""
    agg = _compute_ss_aggregates(panels, spec)
    return agg['pension'] / agg['Y'] if agg['Y'] > 0 else 0.0


def _moment_ui_over_Y(panels, spec):
    """Aggregate UI expenditure / SS output."""
    agg = _compute_ss_aggregates(panels, spec)
    return agg['ui'] / agg['Y'] if agg['Y'] > 0 else 0.0


def _moment_health_gov_over_Y(panels, spec):
    """Government share of health expenditure / SS output."""
    agg = _compute_ss_aggregates(panels, spec)
    return agg['gov_health'] / agg['Y'] if agg['Y'] > 0 else 0.0


MOMENT_DISPATCH = {
    'wealth_gini': _moment_wealth_gini,
    'zero_wealth_fraction': _moment_zero_wealth_fraction,
    'earnings_var_slope': _moment_earnings_var_slope,
    'earnings_var_mean': _moment_earnings_var_mean,
    'unemployment_rate': _moment_unemployment_rate,
    'average_hours': _moment_average_hours,
    'consumption_gini': _moment_consumption_gini,
    'income_gini': _moment_income_gini,
    'earnings_gini': _moment_earnings_gini,
    'mean_assets': _moment_mean_assets,
    'median_wealth_to_income': _moment_median_wealth_to_income,
    'p90_p10_income': _moment_p90_p10_income,
    'mean_consumption': _moment_mean_consumption,
    'A_over_Y': _moment_A_over_Y,
    'K_over_Y': _moment_K_over_Y,
    'C_over_Y': _moment_C_over_Y,
    'labor_share': _moment_labor_share,
    'tax_revenue_over_Y': _moment_tax_revenue_over_Y,
    'pensions_over_Y': _moment_pensions_over_Y,
    'ui_over_Y': _moment_ui_over_Y,
    'health_over_Y': _moment_health_gov_over_Y,
}


def run_model_moments(theta, spec, return_panels=False):
    """Solve + simulate for each education type, compute target moments.

    Returns 1D array of model moments in the same order as spec.moments.
    If *return_panels* is True, returns (m_model, panels) tuple.
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
        cls = LifecycleModelJAX if (spec.backend == 'jax' and _JAX_AVAILABLE) else LifecycleModelPerfectForesight
        model = cls(cfg, verbose=False)
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
    if return_panels:
        return m_model, panels
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


def smm_objective_bounded(theta, spec):
    """SMM objective in original bounded parameter space (for global optimizers)."""
    m_model = run_model_moments(theta, spec)
    m_data = np.array([m.value for m in spec.moments])
    w = np.array([m.weight for m in spec.moments])
    diff = m_data - m_model
    return float(diff @ np.diag(w) @ diff)


def calibrate(spec, maxiter=500, tol=1e-6, verbose=True, method='Nelder-Mead'):
    """Run SMM calibration.

    method: 'Nelder-Mead' (default), 'differential_evolution'.
    With 'differential_evolution', a global search is run first, then polished
    with Nelder-Mead starting from the DE optimum.

    Returns dict with keys: theta, objective, model_moments, data_moments,
    convergence, history, elapsed_seconds.
    """
    t0 = time.time()
    if verbose:
        print(f"Starting SMM calibration [{method}]: {len(spec.params)} params, "
              f"{len(spec.moments)} moments, n_sim={spec.n_sim}")

    history = []
    _iter = [0]

    if method == 'differential_evolution':
        bounds = [(p.lower, p.upper) for p in spec.params]

        def de_callback(xk, convergence):
            _iter[0] += 1
            obj = smm_objective_bounded(xk, spec)
            history.append({'theta': xk.tolist(), 'objective': obj})
            if verbose and _iter[0] % 5 == 0:
                param_str = ', '.join(
                    f'{p.name}={v:.6f}' for p, v in zip(spec.params, xk))
                print(f"  DE gen {_iter[0]:4d}  obj={obj:.8f}  {param_str}")

        de_result = differential_evolution(
            smm_objective_bounded,
            bounds,
            args=(spec,),
            maxiter=maxiter,
            tol=tol,
            seed=spec.seed,
            workers=1,
            polish=True,
            callback=de_callback,
            popsize=5,
            mutation=(0.5, 1.5),
            recombination=0.9,
            init='latinhypercube',
        )
        theta_opt = de_result.x
        # Polish with Nelder-Mead from DE optimum
        if verbose:
            print(f"\nDE finished (obj={de_result.fun:.8f}). Polishing with Nelder-Mead...")
        x0_polish = theta_to_unbounded(theta_opt, spec.params)
        _last_obj = [None]

        def polish_obj(x, spec):
            val = smm_objective(x, spec)
            _last_obj[0] = val
            return val

        def polish_cb(xk):
            tk = unbounded_to_theta(xk, spec.params)
            obj = _last_obj[0]
            history.append({'theta': tk.tolist(), 'objective': obj})
            if verbose and len(history) % 10 == 0:
                param_str = ', '.join(
                    f'{p.name}={v:.6f}' for p, v in zip(spec.params, tk))
                print(f"  NM iter {len(history):4d}  obj={obj:.8f}  {param_str}")

        nm_result = minimize(
            polish_obj, x0_polish, args=(spec,), method='Nelder-Mead',
            callback=polish_cb,
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
        )
        theta_opt = unbounded_to_theta(nm_result.x, spec.params)
        final_obj = nm_result.fun
        converged = de_result.success or nm_result.success
        message = f"DE: {de_result.message} | NM: {nm_result.message}"

    else:
        # Nelder-Mead on logit-transformed parameters
        theta0 = np.array([p.initial for p in spec.params])
        x0 = theta_to_unbounded(theta0, spec.params)
        _last_obj = [None]

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
        theta_opt = unbounded_to_theta(result.x, spec.params)
        final_obj = result.fun
        converged = result.success
        message = result.message

    elapsed = time.time() - t0
    m_model, panels = run_model_moments(theta_opt, spec, return_panels=True)
    m_data = np.array([m.value for m in spec.moments])

    if verbose:
        print(f"\nCalibration finished in {elapsed:.1f}s")

    return {
        'theta': theta_opt,
        'objective': final_obj,
        'model_moments': m_model,
        'data_moments': m_data,
        'convergence': converged,
        'message': message,
        'history': history,
        'elapsed_seconds': elapsed,
        'panels': panels,
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
# 8. JSON config loader
# ---------------------------------------------------------------------------

# Maps JSON external_params keys to LifecycleConfig field names where they differ.
_PARAM_FIELD_MAP = {
    'tau_c': 'tau_c_default',
    'tau_l': 'tau_l_default',
    'tau_p': 'tau_p_default',
    'tau_k': 'tau_k_default',
}


def compute_equilibrium_prices(config_data):
    """Derive w and K/L from firm FOC given exogenous r and production params.

    In SOE: r is exogenous. Firm FOC for capital pins K/L, then w follows.
    With public capital: Y = A_tfp * K_g^eta_g * K_dom^alpha * L^(1-alpha).
    """
    r = config_data['prices']['r']
    prod = config_data.get('production', {})
    alpha = prod.get('alpha', 0.33)
    delta = prod.get('delta', 0.07)
    A_tfp = prod.get('A_tfp', 1.0)
    K_g = prod.get('K_g', 0.0)
    eta_g = prod.get('eta_g', 0.0)

    K_g_factor = K_g ** eta_g if (K_g > 0 and eta_g > 0) else 1.0

    # FOC for K: r + delta = alpha * A_tfp * K_g^eta_g * (K/L)^(alpha-1)
    K_over_L = ((r + delta) / (alpha * A_tfp * K_g_factor)) ** (1.0 / (alpha - 1.0))
    # FOC for L: w = (1-alpha) * A_tfp * K_g^eta_g * (K/L)^alpha
    w = (1.0 - alpha) * A_tfp * K_g_factor * K_over_L ** alpha
    Y_over_L = A_tfp * K_g_factor * K_over_L ** alpha

    return {
        'w': w,
        'K_over_L': K_over_L,
        'Y_over_L': Y_over_L,
        'K_g_factor': K_g_factor,
    }


def compute_age_weights(T, pop_growth=0.0, survival_probs=None):
    """Stationary cross-section age weights: omega(t) = (1+g)^{-t} * S(t).

    S(t) = cumulative survival to age t. Returns normalised weights summing to 1.
    """
    omega = np.ones(T)
    # Population growth discounting
    if pop_growth != 0.0:
        for t in range(T):
            omega[t] = (1.0 + pop_growth) ** (-t)
    # Cumulative survival
    if survival_probs is not None:
        surv = np.asarray(survival_probs).ravel()
        cum_surv = 1.0
        for t in range(T):
            omega[t] *= cum_surv
            if t < len(surv):
                cum_surv *= surv[t]
    # Normalise
    omega /= omega.sum()
    return omega


def load_config(path):
    """Load a calibration input JSON and build a CalibrationSpec.

    Derives w from firm FOC (not from JSON). Computes stationary age weights.
    Returns dict with 'spec', 'config_data', 'eq_prices', 'age_weights'.
    """
    with open(path) as f:
        raw = json.load(f)

    base_config, eq_prices = build_lifecycle_config(raw)
    r = raw['prices']['r']
    w = eq_prices['w']
    raw['_derived'] = {'w': w, 'K_over_L': eq_prices.get('K_over_L', 0),
                       'Y_over_L': eq_prices.get('Y_over_L', 0)}

    # Age weights
    T = raw['model']['T']
    pop_growth = raw.get('external_params', {}).get('pop_growth', 0.0)
    surv = np.array(raw['survival_probs']) if raw.get('survival_probs') else None
    age_weights = compute_age_weights(T, pop_growth, surv)

    # CalibrationSpec
    params = [CalibrationParam(**p) for p in raw['calibration']['params']]
    moments = [TargetMoment(**m) for m in raw['calibration']['targets']]

    sim = raw.get('simulation', {})
    prod = raw.get('production', {})
    production = {
        'alpha': prod.get('alpha', 0.33),
        'delta': prod.get('delta', 0.07),
        'A_tfp': prod.get('A_tfp', 1.0),
        'K_g': prod.get('K_g', 0.0),
        'eta_g': prod.get('eta_g', 0.0),
        'K_over_L': eq_prices.get('K_over_L'),
    }
    spec = CalibrationSpec(
        params=params,
        moments=moments,
        education_shares=raw['education_shares'],
        base_config=base_config,
        n_sim=sim.get('n_sim', 10_000),
        seed=sim.get('seed', 42),
        r=r,
        w=w,
        age_weights=age_weights,
        backend=sim.get('backend', 'numpy'),
        production=production,
    )
    return {'spec': spec, 'config_data': raw, 'eq_prices': eq_prices,
            'age_weights': age_weights}


def build_lifecycle_config(raw, w=None):
    """Build a LifecycleConfig from parsed JSON dict.

    If *w* is None, derives it from firm FOC. Returns (config, eq_prices).
    """
    if w is None:
        eq_prices = compute_equilibrium_prices(raw)
        w = eq_prices['w']
    else:
        eq_prices = {'w': w}
    r = raw['prices']['r']

    # Edu params with defaults for calibrated fields
    edu_params = {}
    for edu_type, edu_data in raw['edu_params'].items():
        edu_params[edu_type] = dict(edu_data)
    for p in raw.get('calibration', {}).get('params', []):
        parts = p['path'].split('.')
        if parts[0] == 'edu_params':
            field_name = parts[2]
            if parts[1] == '*':
                for et in edu_params:
                    edu_params[et].setdefault(field_name, p['initial'])
            else:
                edu_params[parts[1]].setdefault(field_name, p['initial'])
    # Ensure rho_y and sigma_y have defaults even without calibration section
    for et in edu_params:
        edu_params[et].setdefault('rho_y', 0.95)
        edu_params[et].setdefault('sigma_y', 0.10)

    kwargs = {}
    T = raw['model']['T']
    for k, v in raw['model'].items():
        kwargs[k] = v
    for k, v in raw['external_params'].items():
        config_key = _PARAM_FIELD_MAP.get(k, k)
        if config_key == 'pop_growth':
            continue
        kwargs[config_key] = v
    kwargs['edu_params'] = edu_params
    kwargs['r_path'] = np.full(T, r)
    kwargs['w_path'] = np.full(T, w)
    kwargs['r_default'] = r
    kwargs['w_default'] = w
    if raw.get('survival_probs') is not None:
        kwargs['survival_probs'] = np.array(raw['survival_probs'])
    if raw.get('m_age_profile') is not None:
        kwargs['m_age_profile'] = np.array(raw['m_age_profile'])
    if raw.get('wage_age_profile') is not None:
        kwargs['wage_age_profile'] = np.array(raw['wage_age_profile'])
    # Pension avg weight
    paw = raw.get('pension_avg_weight')
    if paw is not None:
        kwargs['pension_avg_weight'] = paw
    else:
        ret_age = raw['model'].get('retirement_age', 40)
        rho_init = 0.95
        for p in raw.get('calibration', {}).get('params', []):
            if p['name'] == 'rho_y':
                rho_init = p['initial']
                break
        kwargs['pension_avg_weight'] = (1 - rho_init ** ret_age) / (ret_age * (1 - rho_init))

    return LifecycleConfig(**kwargs), eq_prices


def build_olg_transition(config_data, backend='numpy'):
    """Build an OLGTransition and transition paths from parsed JSON dict.

    Returns (economy, paths) where paths is a dict with r_path, tau paths,
    pension_replacement_path, G_path, I_g_path, B_path, etc.
    """
    from olg_transition import OLGTransition

    lifecycle_config, eq_prices = build_lifecycle_config(config_data)
    w = eq_prices['w']
    r = config_data['prices']['r']
    prod = config_data.get('production', {})
    trans = config_data.get('transition', {})
    ext = config_data.get('external_params', {})
    T_tr = trans.get('T_transition', 60)

    # Build OLGTransition
    economy = OLGTransition(
        lifecycle_config=lifecycle_config,
        alpha=prod.get('alpha', 0.33),
        delta=prod.get('delta', 0.07),
        A=prod.get('A_tfp', 1.0),
        eta_g=prod.get('eta_g', 0.0),
        K_g_initial=prod.get('K_g', 0.0),
        delta_g=prod.get('delta_g', 0.05),
        economy_type='soe',
        r_star=r,
        pop_growth=ext.get('pop_growth', 0.0),
        birth_year=trans.get('birth_year', 1960),
        current_year=trans.get('current_year', 2020),
        education_shares=config_data.get('education_shares'),
        backend=backend,
        jax_sim_chunk_size=trans.get('jax_chunk_size', 10) if backend == 'jax' else None,
        sim_agent_batch_size=trans.get('sim_agent_batch_size', 10_000),
    )

    # Build transition paths
    r_i = trans.get('r_initial', r)
    r_f = trans.get('r_final', r)
    r_decay = trans.get('r_decay', 5)
    t = np.arange(T_tr)
    r_path = r_f + (r_i - r_f) * np.exp(-t / r_decay) if r_i != r_f else np.full(T_tr, r_i)

    paths = {
        'r_path': r_path,
        'tau_c_path': np.full(T_tr, ext.get('tau_c', 0.20)),
        'tau_l_path': np.full(T_tr, ext.get('tau_l', 0.10)),
        'tau_p_path': np.full(T_tr, ext.get('tau_p', 0.20)),
        'tau_k_path': np.full(T_tr, ext.get('tau_k', 0.20)),
        'pension_replacement_path': np.full(T_tr, ext.get('pension_replacement_default', 0.50)),
    }

    # G and I_g paths (constant at data ratios × steady-state Y, will be rescaled after first sim)
    fiscal = config_data.get('fiscal', {})
    paths['G_over_Y'] = fiscal.get('G_over_Y', 0.13)
    paths['I_g_over_Y'] = fiscal.get('I_g_over_Y', 0.03)
    paths['B_over_Y'] = fiscal.get('B_over_Y', 0.0)

    return economy, paths, T_tr


# ---------------------------------------------------------------------------
# 9. Untargeted moments & report generation
# ---------------------------------------------------------------------------

def compute_untargeted_moments(panels, spec):
    """Compute all moments in MOMENT_DISPATCH not already targeted."""
    targeted_keys = {m.compute_key for m in spec.moments}
    out = {}
    for key, fn in MOMENT_DISPATCH.items():
        if key not in targeted_keys:
            try:
                out[key] = fn(panels, spec)
            except Exception:
                out[key] = float('nan')
    return out


def compute_fiscal_ratios(panels, spec, config_data):
    """Compute government budget components as shares of Y.

    Uses age-weighted aggregation from the simulation panels and
    production function parameters from config_data.
    """
    T = spec.base_config.T
    ret_age = spec.base_config.retirement_age
    n_sim = spec.n_sim
    r = spec.r
    w = spec.w

    # Age weights (T,) — stationary cross-section
    aw = spec.age_weights if spec.age_weights is not None else np.ones(T) / T

    # --- Aggregate per-period means across education types ---
    # For each variable, compute age-weighted cross-sectional mean
    agg = {k: 0.0 for k in ['labor_income', 'consumption', 'assets',
                              'pension', 'ui', 'oop_health', 'gov_health',
                              'tax_c', 'tax_l', 'tax_p', 'tax_k',
                              'transfer']}
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        alive = panel.alive_sim.astype(bool)
        for t in range(T):
            a_t = alive[t]
            n_alive = np.sum(a_t)
            if n_alive == 0:
                continue
            wt = share * aw[t]
            # Means at age t among alive
            agg['labor_income'] += wt * np.mean(panel.effective_y_sim[t, a_t])
            agg['consumption'] += wt * np.mean(panel.c_sim[t, a_t])
            agg['assets'] += wt * np.mean(panel.a_sim[t, a_t])
            agg['pension'] += wt * np.mean(panel.pension_sim[t, a_t])
            agg['ui'] += wt * np.mean(panel.ui_sim[t, a_t])
            agg['oop_health'] += wt * np.mean(panel.oop_m_sim[t, a_t])
            agg['gov_health'] += wt * np.mean(panel.gov_m_sim[t, a_t])
            agg['tax_c'] += wt * np.mean(panel.tax_c_sim[t, a_t])
            agg['tax_l'] += wt * np.mean(panel.tax_l_sim[t, a_t])
            agg['tax_p'] += wt * np.mean(panel.tax_p_sim[t, a_t])
            agg['tax_k'] += wt * np.mean(panel.tax_k_sim[t, a_t])

    # --- Production side ---
    prod = config_data.get('production', {})
    alpha = prod.get('alpha', 0.33)
    A_tfp = prod.get('A_tfp', 1.0)
    K_g = prod.get('K_g', 0.0)
    eta_g = prod.get('eta_g', 0.0)
    K_g_factor = K_g ** eta_g if (K_g > 0 and eta_g > 0) else 1.0

    # L = aggregate effective labor = labor_income / w (since w * L = total labor income)
    L = agg['labor_income'] / w if w > 0 else 0.0
    K_over_L = config_data.get('_derived', {}).get('K_over_L', 0.0)
    K_domestic = K_over_L * L
    Y = A_tfp * K_g_factor * K_domestic ** alpha * L ** (1.0 - alpha) if L > 0 else 0.0

    if Y <= 0:
        return {'error': 'Y <= 0, cannot compute ratios'}

    # --- Fiscal ratios ---
    fiscal_data = config_data.get('fiscal', {})
    B_over_Y = fiscal_data.get('B_over_Y', 0.0)

    tax_revenue = agg['tax_c'] + agg['tax_l'] + agg['tax_p'] + agg['tax_k']
    expenditure = (agg['pension'] + agg['ui'] + agg['gov_health'] +
                   r * B_over_Y * Y)  # interest on debt

    ratios = {
        'Y': Y,
        'C_over_Y': agg['consumption'] / Y,
        'K_over_Y': K_domestic / Y,
        'L': L,
        'w': w,
        'tax_revenue_over_Y': tax_revenue / Y,
        'tax_c_over_Y': agg['tax_c'] / Y,
        'tax_l_over_Y': agg['tax_l'] / Y,
        'tax_p_over_Y': agg['tax_p'] / Y,
        'tax_k_over_Y': agg['tax_k'] / Y,
        'pensions_over_Y': agg['pension'] / Y,
        'ui_over_Y': agg['ui'] / Y,
        'health_over_Y': agg['gov_health'] / Y,
        'interest_over_Y': r * B_over_Y,
        'primary_balance_over_Y': (tax_revenue - expenditure + r * B_over_Y * Y) / Y,
        'total_balance_over_Y': (tax_revenue - expenditure) / Y,
    }

    # Compare to data if available
    comparisons = {}
    for key in ratios:
        data_key = key
        if data_key in fiscal_data:
            comparisons[key] = {
                'model': ratios[key],
                'data': fiscal_data[data_key],
            }
    ratios['_comparisons'] = comparisons

    return ratios


def generate_report(result, spec, config_data, output_dir='output/calibration'):
    """Write a concise calibration report as markdown. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    country = config_data.get('country', 'XX')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'calibration_{country}_{ts}.md')

    lines = []

    def _add(s=''):
        lines.append(s)

    # Header
    _add(f'# Calibration — {country}')
    _add(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if config_data.get('description'):
        _add(f'\n{config_data["description"]}')
    _add()

    # Summary
    m_data = result['data_moments']
    m_model = result['model_moments']
    abs_pct = np.mean([abs(100 * (mv - dv) / dv) if abs(dv) > 1e-12 else 0.0
                       for mv, dv in zip(m_model, m_data)])
    _add('## Summary')
    _add('| | |')
    _add('|---|---|')
    _add(f'| Objective (Q) | {result["objective"]:.6e} |')
    _add(f'| Mean abs % dev (targeted) | {abs_pct:.1f}% |')
    _add(f'| Converged | {result["convergence"]} |')
    _add(f'| Elapsed | {result["elapsed_seconds"]:.1f}s |')
    _add(f'| n_sim | {spec.n_sim:,} |')
    _add(f'| Seed | {spec.seed} |')
    _add()

    # Derived prices
    derived = config_data.get('_derived', {})
    if derived:
        _add('## Prices (derived from firm FOC)')
        _add('| | |')
        _add('|---|---|')
        _add(f'| r (exogenous) | {spec.r:.4f} |')
        _add(f'| w (from FOC) | {derived.get("w", spec.w):.4f} |')
        _add(f'| K/L | {derived.get("K_over_L", 0):.4f} |')
        _add(f'| Y/L | {derived.get("Y_over_L", 0):.4f} |')
        _add()

    # External parameters
    _add('## External Parameters')
    _add('| Parameter | Value |')
    _add('|---|---|')
    model_keys = ['T', 'retirement_age', 'n_a', 'n_y', 'beta', 'gamma']
    for k in model_keys:
        v = config_data['model'].get(k)
        if v is not None:
            _add(f'| {k} | {v} |')
    _add(f'| r | {config_data["prices"]["r"]} |')
    _add(f'| w (derived) | {spec.w:.4f} |')
    for k, v in config_data['external_params'].items():
        _add(f'| {k} | {v} |')
    _add()

    # Education params
    edu = config_data['edu_params']
    all_fields = sorted({f for ep in edu.values() for f in ep})
    edu_types = sorted(edu.keys())
    _add('### Education')
    header = '| Field | ' + ' | '.join(edu_types) + ' |'
    _add(header)
    _add('|---' * (len(edu_types) + 1) + '|')
    for f in all_fields:
        row = f'| {f} |'
        for et in edu_types:
            row += f' {edu[et].get(f, "")} |'
        _add(row)
    _add(f'\nShares: {config_data["education_shares"]}')
    _add()

    # Calibrated parameters
    _add('## Calibrated Parameters')
    _add('| Parameter | Initial | Final | Lower | Upper | Near bound? |')
    _add('|---|---|---|---|---|---|')
    for p, v in zip(spec.params, result['theta']):
        rng = p.upper - p.lower
        near = ''
        if (v - p.lower) < 0.05 * rng:
            near = 'lower'
        elif (p.upper - v) < 0.05 * rng:
            near = 'upper'
        _add(f'| {p.name} | {p.initial:.6f} | {v:.6f} | {p.lower:.4f} | '
             f'{p.upper:.4f} | {near} |')
    _add()

    # Targeted moments
    _add('## Targeted Moments')
    _add('| Moment | Data | Model | % Dev | Weight |')
    _add('|---|---|---|---|---|')
    for m, mv, dv in zip(spec.moments, m_model, m_data):
        pct = 100 * (mv - dv) / abs(dv) if abs(dv) > 1e-12 else 0.0
        _add(f'| {m.name} | {m.value:.4f} | {mv:.4f} | {pct:+.1f}% | {m.weight:.2f} |')
    _add()

    # Untargeted moments
    untargeted_model = result.get('untargeted_moments', {})
    untargeted_data = config_data.get('untargeted', {})
    if untargeted_model:
        _add('## Untargeted Moments')
        _add('| Moment | Model | Data | % Dev |')
        _add('|---|---|---|---|')
        for key in sorted(untargeted_model):
            mv = untargeted_model[key]
            dv = untargeted_data.get(key)
            if dv is not None and abs(dv) > 1e-12:
                pct = f'{100 * (mv - dv) / abs(dv):+.1f}%'
                _add(f'| {key} | {mv:.4f} | {dv:.4f} | {pct} |')
            else:
                dv_str = '—' if dv is None else f'{dv}'
                _add(f'| {key} | {mv:.4f} | {dv_str} | |')
        _add()

    # Fiscal ratios
    fiscal = result.get('fiscal_ratios', {})
    if fiscal and 'error' not in fiscal:
        fiscal_data = config_data.get('fiscal', {})
        _add('## Fiscal Ratios (model vs data, share of Y)')
        _add('| Ratio | Model | Data | Dev |')
        _add('|---|---|---|---|')
        display_keys = [
            'C_over_Y', 'K_over_Y', 'tax_revenue_over_Y',
            'tax_c_over_Y', 'tax_l_over_Y', 'tax_p_over_Y', 'tax_k_over_Y',
            'pensions_over_Y', 'ui_over_Y', 'health_over_Y',
            'interest_over_Y', 'primary_balance_over_Y', 'total_balance_over_Y',
        ]
        for key in display_keys:
            mv = fiscal.get(key)
            if mv is None:
                continue
            dv = fiscal_data.get(key)
            if dv is not None and abs(dv) > 1e-12:
                dev = f'{mv - dv:+.3f}'
                _add(f'| {key} | {mv:.3f} | {dv:.3f} | {dev} |')
            else:
                _add(f'| {key} | {mv:.3f} | — | |')
        _add(f'\nY (model units) = {fiscal.get("Y", 0):.4f}, '
             f'w = {fiscal.get("w", 0):.4f}')
        _add()

    # Convergence history (subsample)
    history = result.get('history', [])
    if history:
        _add('## Convergence')
        param_names = [p.name for p in spec.params]
        header = '| Iter | Objective | ' + ' | '.join(param_names) + ' |'
        _add(header)
        _add('|---' * (len(param_names) + 2) + '|')
        # Show first, every 10th, and last
        indices = sorted(set([0] + list(range(9, len(history), 10)) +
                             [len(history) - 1]))
        for i in indices:
            h = history[i]
            vals = ' | '.join(f'{v:.6f}' for v in h['theta'])
            _add(f'| {i + 1} | {h["objective"]:.6e} | {vals} |')
        _add()

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


# ---------------------------------------------------------------------------
# 10. Default calibration specification
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
# 11. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SMM calibration for OLG lifecycle model')
    parser.add_argument('--n-sim', type=int, default=None,
                        help='Override n_sim from config')
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed from config')
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--test', action='store_true',
                        help='Use tiny model for quick smoke test')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON calibration input file')
    parser.add_argument('--output', type=str, default=None,
                        help='JSON file to save raw results')
    parser.add_argument('--report-dir', type=str, default='output/calibration',
                        help='Directory for markdown reports')
    parser.add_argument('--backend', type=str, default=None,
                        choices=['numpy', 'jax'],
                        help='Override backend (numpy or jax)')
    parser.add_argument('--method', type=str, default='Nelder-Mead',
                        choices=['Nelder-Mead', 'differential_evolution'],
                        help='Optimization method')
    args = parser.parse_args()

    config_data = None

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
            n_sim=min(args.n_sim or 200, 200),
            seed=args.seed or 42,
        )
        args.maxiter = min(args.maxiter, 10)
    elif args.config:
        loaded = load_config(args.config)
        spec = loaded['spec']
        config_data = loaded['config_data']
        # CLI overrides — use dataclasses.replace-style rebuild
        overrides = {}
        if args.n_sim is not None:
            overrides['n_sim'] = args.n_sim
        if args.seed is not None:
            overrides['seed'] = args.seed
        if args.backend is not None:
            overrides['backend'] = args.backend
        if overrides:
            spec = replace(spec, **overrides)
    else:
        spec = default_spec(
            n_sim=args.n_sim or 10_000,
            seed=args.seed or 42)

    result = calibrate(spec, maxiter=args.maxiter, tol=args.tol, method=args.method)
    print_calibration_results(result, spec)

    # Compute untargeted moments and fiscal ratios
    panels = result.get('panels', {})
    if panels:
        result['untargeted_moments'] = compute_untargeted_moments(panels, spec)
        if config_data is not None:
            result['fiscal_ratios'] = compute_fiscal_ratios(
                panels, spec, config_data)

    if config_data is not None:
        report_path = generate_report(result, spec, config_data, args.report_dir)
        print(f"\nReport: {report_path}")

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
        if 'untargeted_moments' in result:
            out['untargeted_moments'] = result['untargeted_moments']
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
