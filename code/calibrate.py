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
# 1. SimPanel — named wrapper for the 22-tuple simulation output
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
    alpha_idx_sim: np.ndarray   # (T, n_sim) int — Phase 8 permanent FE grid index (constant across t)
    transfer_sim: np.ndarray    # (T, n_sim) means-tested transfer received (consumption floor top-up)
    # Exact cross-sections only (exact_panel_to_simpanel): a column is a state
    # of the grid and weight_sim its mass of households. None: one household
    # per column.
    weight_sim: Optional[np.ndarray] = None


def wrap_sim_output(tup):
    """Convert the raw 23-tuple from model.simulate() to a SimPanel.

    Older emitters: a 21-tuple lacks alpha_idx_sim (Phase 8) and transfer_sim
    (2026-10-02); a 22-tuple lacks transfer_sim. Both are padded with zeros,
    which is n_alpha=1 and no transfer floor.
    """
    tup = tuple(tup)
    T_sim, n_sim = tup[0].shape
    if len(tup) == 21:
        tup = tup + (np.zeros((T_sim, n_sim), dtype=np.int32),)
    if len(tup) == 22:
        tup = tup + (np.zeros((T_sim, n_sim)),)
    return SimPanel(*tup)


def exact_panel_to_simpanel(panel, mass):
    """SimPanel from the (panel, mass) of exact_panel() / cross_section_exact().

    One column per state of the grid, weighted by the mass of households on
    it; alive_sim marks the states with positive mass. The moment functions
    read weight_sim, so they take population moments from it where they take
    sample moments from a simulated panel.
    """
    fields = list(panel)
    mass = np.asarray(mass, dtype=float)
    fields[SimPanel._fields.index('alive_sim')] = mass > 0.0
    return SimPanel(*fields, weight_sim=mass)


def _alive_mean(panel, field, t, alive_t):
    """Mean of a panel field among the columns alive at age t."""
    x = getattr(panel, field)[t, alive_t]
    if panel.weight_sim is None:
        return float(np.mean(x))
    w = panel.weight_sim[t, alive_t]
    return float(np.sum(w * x) / np.sum(w))


def _quantile(values, q, weights=None):
    """q-th quantile (q in [0, 1]). With weights, the smallest value whose
    cumulative weight share reaches q: the population quantile of a discrete
    distribution, which is what the sample quantile of a growing simulated
    panel settles on."""
    if weights is None:
        return float(np.percentile(values, 100.0 * q))
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights, dtype=float)[order]
    cum = np.cumsum(w) / np.sum(w)
    return float(v[min(int(np.searchsorted(cum, q - 1e-12)), len(v) - 1)])


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
        # Weighted Gini: one minus twice the area under the Lorenz curve, the
        # area taken by the trapezoid rule over the sorted sample, so each
        # observation's step contributes w_i times the mean of the cumulative
        # income share before and after it. Until 2026-10-02 only the share
        # after it entered (a right-endpoint sum), which understated every
        # weighted Gini: 0.067 instead of 0.267 for the values 1..5 with equal
        # weights, and a negative value on skewed weights.
        cum_xw_prev = np.concatenate([[0.0], cum_xw[:-1]])
        return 1.0 - np.sum(ws * (cum_xw + cum_xw_prev)) / (total_w * total_xw)


def compute_earnings_variance_by_age(effective_y_sim, employed_sim, alive_sim,
                                     retirement_age, weights=None):
    """Variance of log earnings at each working age, excluding unemployed/dead.

    Returns array of shape (retirement_age,). Ages with fewer than 2 employed
    alive agents get NaN. *weights* (T, n) gives the mass of each column
    (exact cross-sections); None weights the columns equally.
    """
    T = effective_y_sim.shape[0]
    n_ages = min(retirement_age, T)
    var_by_age = np.full(n_ages, np.nan)
    for t in range(n_ages):
        mask = alive_sim[t] & employed_sim[t] & (effective_y_sim[t] > 0)
        if np.sum(mask) >= 2:
            log_earn = np.log(effective_y_sim[t, mask])
            if weights is None:
                var_by_age[t] = np.var(log_earn)
            else:
                w = weights[t, mask] / np.sum(weights[t, mask])
                mean = np.sum(w * log_earn)
                var_by_age[t] = np.sum(w * (log_earn - mean) ** 2)
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
    weight: float = 1.0    # diagonal weight; the live configs use 1/target**2,
                           # which makes the objective equal-weighted squared
                           # RELATIVE deviations, not inverse variances
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
    cohort_survival: Optional[np.ndarray] = None  # (T, T) row j = cohort aged 25+j
    n_sim_cohorts: int = 2000   # agents per cohort when cohort_survival is set
    # 'simulation': moments from simulated panels (n_sim / n_sim_cohorts draws);
    # 'exact': from the distribution over states carried forward by age (no draws)
    aggregation: str = 'simulation'
    # Retirement age and career-average pension weight of the cohort aged 25+j
    # in the base year, (T,) each; None keeps base_config's for every cohort.
    cohort_retirement_age: Optional[np.ndarray] = None
    cohort_pension_avg_weight: Optional[np.ndarray] = None
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
    # omega(t) is spread across the agents ALIVE at t, not across n_sim. aw is
    # already the living share of each age, so dividing by n_sim would apply
    # survival a second time and tilt these moments toward the young: the
    # weights at age t would sum to aw[t]*S_t instead of aw[t]. With a mask the
    # denominator stays the alive count, so a subgroup's weight is its share of
    # the living at that age -- which is what a cross-sectional average means.
    if panel.weight_sim is None:
        n_alive = np.maximum(panel.alive_sim.astype(bool).sum(axis=1), 1)
        w_grid = share * aw[:, None] / n_alive[:, None] * np.ones((1, n_sim))
    else:
        # Exact cross-section: a column's weight among the living at its age
        # is its share of their mass.
        alive_mass = np.where(panel.alive_sim.astype(bool), panel.weight_sim, 0.0).sum(axis=1)
        w_grid = share * aw[:, None] * panel.weight_sim / np.maximum(alive_mass, 1e-300)[:, None]
    return alive, w_grid[alive]


def _column_mass(panel, mask):
    """Mass of the masked columns for statistics that pool households without
    age or education weights: None for a simulated panel (every household
    counts once), the state masses for an exact cross-section."""
    return None if panel.weight_sim is None else panel.weight_sim[mask]


def _pooled(values, masses):
    """Concatenate per-education values and their masses (None if unweighted)."""
    vals = np.concatenate(values) if values else np.empty(0)
    if any(m is None for m in masses):
        return vals, None
    return vals, np.concatenate(masses)


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
            panel.alive_sim, ret_age, weights=panel.weight_sim)
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
            panel.alive_sim, ret_age, weights=panel.weight_sim)
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
    """Gini of total income (earnings + UI + pensions), age-weighted.

    effective_y_sim already equals wage_income + ui_sim from the simulation,
    so adding ui_sim again double-counts UI. The correct total income for
    Gini purposes is effective_y_sim (employed wage OR UI) + pension_sim.
    """
    all_inc, all_w = [], []
    for edu, panel in panels.items():
        alive, w = _agent_weights(panel, spec, edu)
        income = panel.effective_y_sim[alive] + panel.pension_sim[alive]
        all_inc.append(income)
        all_w.append(w)
    return compute_gini(np.concatenate(all_inc), np.concatenate(all_w))


def _disposable_income(panel):
    """Individual disposable income: gross less income tax and contributions.

    EU-SILC's disposable income is gross income net of income tax and social
    contributions. Here effective_y_sim is the wage when employed and UI
    otherwise, pension_sim the pension; tax_l_sim is the income tax on
    whichever of these applies, and tax_p_sim the contribution, levied on
    wages only. Consumption and capital taxes are outside the concept.
    """
    return (panel.effective_y_sim + panel.pension_sim
            - panel.tax_l_sim - panel.tax_p_sim)


def _moment_disposable_income_gini(panels, spec):
    """Gini of individual disposable income, age-weighted.

    Brings the model onto the data's definition as far as a model without
    households allows. Eurostat ilc_di12 measures *equivalised household*
    disposable income: netting taxes closes the gross-to-disposable half of
    the gap, while household pooling and equivalisation cannot be reproduced
    here, and both would lower the statistic further. Read it as an upper
    bound on the comparable number, not as a like-for-like match.
    """
    all_inc, all_w = [], []
    for edu, panel in panels.items():
        alive, w = _agent_weights(panel, spec, edu)
        all_inc.append(_disposable_income(panel)[alive])
        all_w.append(w)
    return compute_gini(np.concatenate(all_inc), np.concatenate(all_w))


def _moment_disposable_p90_p10(panels, spec):
    """P90/P10 of individual disposable income among alive agents.

    Same caveat as _moment_disposable_income_gini: the data counterpart
    (Eurostat ilc_di01) is equivalised household disposable income.
    """
    all_inc, all_w = [], []
    for edu, panel in panels.items():
        alive = panel.alive_sim.astype(bool)
        all_inc.append(_disposable_income(panel)[alive])
        all_w.append(_column_mass(panel, alive))
    vals, mass = _pooled(all_inc, all_w)
    keep = vals > 0
    vals, mass = vals[keep], (mass[keep] if mass is not None else None)
    if len(vals) < 10:
        return 0.0
    p90, p10 = _quantile(vals, 0.9, mass), _quantile(vals, 0.1, mass)
    return float(p90 / p10) if p10 > 0 else 0.0


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
    _, mass = _pooled([], [_column_mass(p, _employed_pos(p)) for p in panels.values()])
    if mass is None:
        return float(np.median(a_vals / y_vals))
    return _quantile(a_vals / y_vals, 0.5, mass)


def _moment_p90_p10_income(panels, spec):
    """P90/P10 ratio of total income (earnings + UI + pensions) among alive agents.

    effective_y_sim already includes ui_sim (wage_income + ui_sim from the
    simulation), so adding ui_sim again would double-count. Total income for
    the ratio is effective_y_sim + pension_sim.
    """
    all_inc, all_w = [], []
    for edu, panel in panels.items():
        alive = panel.alive_sim.astype(bool)
        income = panel.effective_y_sim[alive] + panel.pension_sim[alive]
        all_inc.append(income)
        all_w.append(_column_mass(panel, alive))
    vals, mass = _pooled(all_inc, all_w)
    keep = vals > 0
    vals, mass = vals[keep], (mass[keep] if mass is not None else None)
    if len(vals) < 10:
        return 0.0
    p90, p10 = _quantile(vals, 0.9, mass), _quantile(vals, 0.1, mass)
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
            'oop_health', 'gov_health', 'tax_c', 'tax_l', 'tax_p', 'tax_k',
            'bequest', 'transfer']
    agg = {k: 0.0 for k in keys}
    for edu, panel in panels.items():
        share = spec.education_shares[edu]
        alive = panel.alive_sim.astype(bool)
        for t in range(T):
            a_t = alive[t]
            if not np.any(a_t):
                continue
            wt = share * aw[t]
            agg['labor_income'] += wt * _alive_mean(panel, 'effective_y_sim', t, a_t)
            agg['consumption']  += wt * _alive_mean(panel, 'c_sim', t, a_t)
            agg['assets']       += wt * _alive_mean(panel, 'a_sim', t, a_t)
            agg['pension']      += wt * _alive_mean(panel, 'pension_sim', t, a_t)
            agg['ui']           += wt * _alive_mean(panel, 'ui_sim', t, a_t)
            agg['oop_health']   += wt * _alive_mean(panel, 'oop_m_sim', t, a_t)
            agg['gov_health']   += wt * _alive_mean(panel, 'gov_m_sim', t, a_t)
            agg['tax_c']        += wt * _alive_mean(panel, 'tax_c_sim', t, a_t)
            agg['tax_l']        += wt * _alive_mean(panel, 'tax_l_sim', t, a_t)
            agg['tax_p']        += wt * _alive_mean(panel, 'tax_p_sim', t, a_t)
            agg['tax_k']        += wt * _alive_mean(panel, 'tax_k_sim', t, a_t)
            # Accidental bequests of those who die at age t, per person alive
            # at t (bequest_sim is nonzero only for the dying, who are alive
            # at the start of the period).
            agg['bequest']      += wt * _alive_mean(panel, 'bequest_sim', t, a_t)
            agg['transfer']     += wt * _alive_mean(panel, 'transfer_sim', t, a_t)

    prod = spec.production or {}
    alpha = prod.get('alpha', 0.33)
    A_tfp = prod.get('A_tfp', 1.0)
    K_g = prod.get('K_g', 0.0)
    eta_g = prod.get('eta_g', 0.0)
    K_g_factor = K_g ** eta_g if (K_g > 0 and eta_g > 0) else 1.0

    # Labour input in efficiency units: wage income / w. labor_income is the
    # panel's effective_y_sim, which carries UI as well, so UI is netted out --
    # a transfer is not an efficiency unit of labour. Until 2026-10-02 it was
    # left in, overstating L, K and Y by UI/(wL).
    L = (agg['labor_income'] - agg['ui']) / spec.w if spec.w > 0 else 0.0
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


def _moment_tax_p_over_Y(panels, spec):
    """Payroll / social-security-contribution revenue / SS output (SSC/GDP)."""
    agg = _compute_ss_aggregates(panels, spec)
    return agg['tax_p'] / agg['Y'] if agg['Y'] > 0 else 0.0


MOMENT_DISPATCH = {
    'wealth_gini': _moment_wealth_gini,
    'zero_wealth_fraction': _moment_zero_wealth_fraction,
    'earnings_var_slope': _moment_earnings_var_slope,
    'earnings_var_mean': _moment_earnings_var_mean,
    'unemployment_rate': _moment_unemployment_rate,
    'average_hours': _moment_average_hours,
    'consumption_gini': _moment_consumption_gini,
    'income_gini': _moment_income_gini,
    'disposable_income_gini': _moment_disposable_income_gini,
    'disposable_p90_p10': _moment_disposable_p90_p10,
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
    'health_gov_over_Y': _moment_health_gov_over_Y,
    'tax_p_over_Y': _moment_tax_p_over_Y,
}


def theta_from_config(raw, spec, verbose=True):
    """Parameter vector from _derived.theta, falling back to each param's initial.

    _derived.theta holds only the parameters the last SMM run fitted, so adding a
    parameter to calibration.params breaks every caller that indexes it directly
    until a calibration has run. This fills the gaps from `initial` and says
    which, rather than raising a KeyError far from the cause.
    """
    th = raw.get('_derived', {}).get('theta', {})
    out, missing = [], []
    for p in spec.params:
        if p.name in th:
            out.append(float(th[p.name]))
        else:
            out.append(float(p.initial))
            missing.append(p.name)
    if missing and verbose:
        print(f"  [theta_from_config] not in _derived.theta, using initial: "
              f"{', '.join(missing)} (run the SMM to fit them)")
    return np.array(out, dtype=float)


def base_year_cross_section(theta, spec, cfg=None, n_sim=None, seed=None,
                            survival=None, seed_per_cohort=True, verbose=False,
                            batched=True, chunk_size=None):
    """Panels whose row j is the cohort aged 25+j in the base year, at age j.

    The transition's t=0 cross-section mixes sixty cohorts, each having solved
    its own lifecycle problem against its own survival diagonal; they differ by
    up to 0.47 in the probability of reaching 84. A single stationary solve
    cannot represent that, which is why the calibration's moments and the
    transition's t=0 disagree.

    Each cohort is solved over the full horizon -- backward induction at age j
    needs every later age, so the solve cannot be truncated -- but simulated
    only to age j, which is all the base year observes. That is 1,830 cohort
    periods instead of 3,600.

    *survival* overrides the schedules: a (T, n_h) vector is used for every
    cohort, a (T, T) array is one schedule per cohort. Otherwise they come from
    the spec, or from the config.

    Each cohort draws its own shocks (*seed_per_cohort*), so the ages are
    statistically independent -- unlike the single-solve panel, where one
    lifecycle serves every age and the ages are perfectly correlated. Pass
    False to reuse one seed, which is what makes the result reproduce the
    single-solve panel exactly and is the regression test for this plumbing.

    On the JAX backend, *batched* solves and simulates an education group's
    cohorts in one call vectorised over their survival schedules
    (LifecycleModelJAX.cross_section_batched), with *chunk_size* cohorts per
    simulation call; False runs them one at a time. The JAX simulation runs
    every cohort to the last age either way. The NumPy backend always runs
    them one at a time.
    """
    config = apply_params(spec.base_config, spec.params, theta)
    T = config.T
    S = survival
    if S is None:
        S = spec.cohort_survival
    if S is None and cfg is not None:
        S = base_year_cohort_survival(cfg, T)
    if S is None:
        raise ValueError('no cohort survival schedules: configure '
                         'transition.demography_file')
    n_sim = spec.n_sim_cohorts if n_sim is None else n_sim
    S = np.asarray(S, dtype=float)
    # A (T, n_h) argument is one schedule for every cohort; a (T, T) one is a
    # schedule per cohort. With n_h == T the two are ambiguous, so the caller's
    # (T, T) per-cohort form wins only when it cannot be a single vector.
    shared = survival is not None and S.shape == (T, config.n_h) and config.n_h != T
    cls = (LifecycleModelJAX if (spec.backend == 'jax' and _JAX_AVAILABLE)
           else LifecycleModelPerfectForesight)
    seed = spec.seed if seed is None else seed
    exact = spec.aggregation == 'exact'

    # Retirement age and pension weight of each cohort (the base config's when
    # the spec carries none).
    if spec.cohort_retirement_age is not None:
        ret = [(int(J), float(lam)) for J, lam in zip(spec.cohort_retirement_age,
                                                      spec.cohort_pension_avg_weight)]
    else:
        ret = [(int(config.retirement_age), float(config.pension_avg_weight))] * T

    def cohort_config(edu_type, surv, j):
        return config._replace(
            education_type=edu_type,
            survival_probs=surv,
            r_path=np.full(T, spec.r),
            w_path=np.full(T, spec.w),
            retirement_age=ret[j][0],
            pension_avg_weight=ret[j][1],
        )

    panels = {}
    for edu_type in spec.education_shares:
        if batched and cls is LifecycleModelJAX:
            # One batched call per retirement group: the retirement age is a
            # static argument of the JAX solve and the pension weight a scalar
            # shared by the batch.
            groups = {}
            for j in range(T):
                groups.setdefault(ret[j], []).append(j)
            out = mass = None
            for js in groups.values():
                surv_stack = np.stack([S if shared else S[j].reshape(T, config.n_h)
                                       for j in js])
                model = cls(cohort_config(edu_type, surv_stack[0], js[0]), verbose=False)
                if exact:
                    part, part_mass = model.cross_section_exact(
                        surv_stack, rows=js, chunk_size=chunk_size)
                    if mass is None:
                        mass = np.zeros((T, part_mass.shape[1]))
                    mass[js] = part_mass
                else:
                    seeds = [seed + j if seed_per_cohort else seed for j in js]
                    part = model.cross_section_batched(surv_stack, seeds, n_sim,
                                                       chunk_size=chunk_size, rows=js)
                if out is None:
                    out = [np.zeros((T, x.shape[1]), dtype=x.dtype) for x in part]
                for o, x in zip(out, part):
                    o[js] = x
            panels[edu_type] = (exact_panel_to_simpanel(tuple(out), mass) if exact
                                else wrap_sim_output(tuple(out)))
            if verbose:
                print(f'    {edu_type}: {T} cohorts in {len(groups)} retirement '
                      f'groups (batched)', flush=True)
            continue
        rows = None
        for j in range(T):
            surv_j = S if shared else S[j].reshape(T, config.n_h)
            cfg_j = cohort_config(edu_type, surv_j, j)
            model = cls(cfg_j, verbose=False)
            model.solve(verbose=False)
            if exact:
                # The cohort's cross-section over states at age j, as row j.
                panel = exact_panel_to_simpanel(*model.exact_panel(rows=[j]))
                at = 0
            else:
                panel = wrap_sim_output(model.simulate(
                    T_sim=j + 1, n_sim=n_sim,
                    seed=seed + j if seed_per_cohort else seed))
                at = j
            fields = [f for f in panel._fields if getattr(panel, f) is not None]
            if rows is None:
                n_col = np.asarray(panel.a_sim).shape[1]
                rows = {f: np.zeros((T, n_col), dtype=np.asarray(getattr(panel, f)).dtype)
                        for f in fields}
            for f in fields:
                rows[f][j] = np.asarray(getattr(panel, f))[at]
            if verbose and (j + 1) % 10 == 0:
                print(f'    {edu_type}: cohort {j + 1}/{T}', flush=True)
        panels[edu_type] = SimPanel(**rows)
    return panels


def run_model_moments(theta, spec, return_panels=False):
    """Solve + simulate for each education type, compute target moments.

    Returns 1D array of model moments in the same order as spec.moments.
    If *return_panels* is True, returns (m_model, panels) tuple.
    """
    if spec.cohort_survival is not None:
        # One household per birth cohort, each facing the mortality its birth
        # year lived, so the base-year cross-section is built the way the
        # transition's t=0 is rather than from one representative lifecycle.
        panels = base_year_cross_section(theta, spec)
        m_model = np.empty(len(spec.moments))
        for i, mom in enumerate(spec.moments):
            fn = MOMENT_DISPATCH.get(mom.compute_key)
            if fn is None:
                raise ValueError(f"Unknown compute_key: {mom.compute_key!r}")
            m_model[i] = fn(panels, spec)
        return (m_model, panels) if return_panels else m_model

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
        if spec.aggregation == 'exact':
            panels[edu_type] = exact_panel_to_simpanel(*model.exact_panel())
        else:
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


# Forward-difference step for the 'least_squares' Jacobian, in the logit-
# transformed parameters: about 1% of nu and 0.5% of the replacement rate at
# the 2026-10 calibration, where the moments respond smoothly.
LSQ_JAC_STEP = 0.02


def calibrate(spec, maxiter=500, tol=1e-6, verbose=True, method='Nelder-Mead'):
    """Run SMM calibration.

    method: 'Nelder-Mead' (default), 'differential_evolution', 'least_squares'.
    With 'differential_evolution', a global search is run first, then polished
    with Nelder-Mead starting from the DE optimum. 'least_squares' minimises the
    same objective as a sum of squared residuals sqrt(w)(m_model - m_data) on
    the logit-transformed parameters (scipy trust-region reflective), with a
    forward-difference Jacobian at a fixed step LSQ_JAC_STEP: the simulated
    moments are flat at very small steps (asset choices are on a grid), so an
    adaptive step would read derivatives off that flatness.

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

    elif method == 'least_squares':
        from scipy.optimize import least_squares
        theta0 = np.array([p.initial for p in spec.params])
        x0 = theta_to_unbounded(theta0, spec.params)
        m_data = np.array([m.value for m in spec.moments])
        sqrt_w = np.sqrt(np.array([m.weight for m in spec.moments]))
        _cache = {}

        def residuals(x):
            key = tuple(np.asarray(x, dtype=float))
            if key not in _cache:
                theta = unbounded_to_theta(np.asarray(x, dtype=float), spec.params)
                _cache[key] = sqrt_w * (run_model_moments(theta, spec) - m_data)
                obj = float(_cache[key] @ _cache[key])
                history.append({'theta': theta.tolist(), 'objective': obj})
                if verbose:
                    param_str = ', '.join(
                        f'{p.name}={v:.6f}' for p, v in zip(spec.params, theta))
                    print(f"  eval {len(history):4d}  obj={obj:.10f}  {param_str}",
                          flush=True)
            return _cache[key]

        def jacobian(x):
            r0 = residuals(x)
            J = np.empty((len(r0), len(x)))
            for i in range(len(x)):
                xh = np.array(x, dtype=float)
                xh[i] += LSQ_JAC_STEP
                J[:, i] = (residuals(xh) - r0) / LSQ_JAC_STEP
            return J

        result = least_squares(residuals, x0, jac=jacobian, method='trf',
                               xtol=tol, ftol=tol, gtol=tol, max_nfev=maxiter)
        theta_opt = unbounded_to_theta(result.x, spec.params)
        final_obj = float(result.fun @ result.fun)
        converged = bool(result.success)
        message = result.message

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


def _demography_path(raw):
    """Absolute path to the configured demographic sidecar, or None."""
    rel = raw.get('transition', {}).get('demography_file')
    if not rel:
        return None
    path = rel if os.path.isabs(rel) else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
    if not os.path.exists(path):
        print(f"  [load_config] demography_file not found: {path}")
        return None
    return path


def base_year_cohort_survival(raw, T):
    """Survival schedules of the cohorts alive in the base year, by model age.

    Returns (T, T) with row j the schedule the cohort aged 25+j in the base
    year faces over its whole life: `px[base - j + a, a]` at model age a, so
    historical life tables for the ages it has already passed and projected
    ones for those ahead. Row 0 is the cohort entering at t = 0.

    This is the object the base-year equilibrium needs in order to face the
    same mortality as the transition's t = 0 cross-section. A single vector
    cannot: the sixty cohorts differ by up to 0.47 in the probability of
    reaching 84, which moves hours and pensions/Y by a few percent.

    None when no demographic path is configured.
    """
    path = _demography_path(raw)
    if path is None:
        return None
    d = np.load(path)
    years = np.asarray(d['years'], dtype=int)
    px = np.asarray(d['px'], dtype=float)
    base = int(d['base_year'])
    if px.shape[1] != T:
        print(f"  [base_year_cohort_survival] demography age dim {px.shape[1]} "
              f"!= model T {T}; skipping cohort survival.")
        return None
    lo, hi = int(years[0]), int(years[-1])
    out = np.empty((T, T))
    for j in range(T):
        entry = base - j                      # year this cohort reached age 25
        if entry < lo:
            raise ValueError(
                f"demography starts in {lo} but the cohort aged {25 + j} in "
                f"{base} entered in {entry}")
        rows = np.searchsorted(years, np.clip(entry + np.arange(T), lo, hi))
        out[j] = px[rows, np.arange(T)]
    return out


def base_year_age_weights(raw, T):
    """Share of the living population at each model age in the base year.

    The calibration aggregates cross-sectional means taken among the alive, so
    the weight on an age is that age's share of the living population. That is
    a different object from the transition's weights, which are cohort sizes
    at entry because its means run over all agents with the dead holding zero.

    Uses the measured base-year cross-section when a demographic path is
    configured, and the stationary approximation otherwise.
    """
    path = _demography_path(raw)
    if path is not None:
        cs = np.asarray(np.load(path)['cross_section_base'], dtype=float)
        if len(cs) == T:
            return cs / cs.sum()
        print(f"  [load_config] demography age dim {len(cs)} != model T {T}; "
              f"using the stationary weights.")
    pop_growth = raw.get('external_params', {}).get('pop_growth', 0.0)
    surv = np.array(raw['survival_probs']) if raw.get('survival_probs') else None
    return compute_age_weights(T, pop_growth, surv)


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
    # Preserve any existing _derived block (e.g. theta from a previous SMM run)
    # rather than overwriting it with only the firm-FOC-derived prices.
    raw.setdefault('_derived', {})
    raw['_derived'].update({'w': w, 'K_over_L': eq_prices.get('K_over_L', 0),
                            'Y_over_L': eq_prices.get('Y_over_L', 0)})

    # Age weights
    T = raw['model']['T']
    age_weights = base_year_age_weights(raw, T)

    # Cohort-consistent base year: one household per birth cohort alive in the
    # base year, each on its own survival schedule. Off by default, because it
    # makes every SMM evaluation an order of magnitude dearer.
    cohort_survival = None
    if raw.get('calibration', {}).get('base_year_cohorts', False):
        cohort_survival = base_year_cohort_survival(raw, T)
        if cohort_survival is None:
            raise ValueError(
                'calibration.base_year_cohorts is set but the cohort survival '
                'schedules are unavailable; check transition.demography_file')

    cohort_J_R = cohort_lam = None
    if cohort_survival is not None:
        cohort_J_R, cohort_lam = base_year_cohort_retirement(raw, T)

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
        cohort_survival=cohort_survival,
        n_sim_cohorts=sim.get('n_sim_cohorts', 2000),
        aggregation=sim.get('aggregation', 'simulation'),
        cohort_retirement_age=cohort_J_R,
        cohort_pension_avg_weight=cohort_lam,
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
    # Ensure rho_y, sigma_y, and sigma_alpha have defaults even without calibration section
    for et in edu_params:
        edu_params[et].setdefault('rho_y', 0.95)
        edu_params[et].setdefault('sigma_y', 0.10)
        edu_params[et].setdefault('sigma_alpha', 0.0)  # FE off by default

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

    # Apply calibrated theta from _derived.theta if present. This closes the loop
    # between calibrate.py (which writes _derived.theta after SMM) and downstream
    # consumers (build_olg_transition, run_fiscal_figures.py): post-SMM runs use
    # the calibrated parameters automatically, without manually editing model.nu /
    # model.beta. Param paths come from raw['calibration']['params'].
    derived_theta = raw.get('_derived', {}).get('theta', {})
    if derived_theta:
        param_paths = {p['name']: p['path']
                       for p in raw.get('calibration', {}).get('params', [])}
        for name, value in derived_theta.items():
            path = param_paths.get(name, name)
            parts = path.split('.')
            if parts[0] == 'edu_params' and len(parts) == 3:
                field_name = parts[2]
                if parts[1] == '*':
                    for et in edu_params:
                        edu_params[et][field_name] = value
                else:
                    edu_params[parts[1]][field_name] = value
            else:
                kwargs[path] = value
    if raw.get('survival_probs') is not None:
        kwargs['survival_probs'] = np.array(raw['survival_probs'])
    if raw.get('m_age_profile') is not None:
        kwargs['m_age_profile'] = np.array(raw['m_age_profile'])
    if raw.get('wage_age_profile') is not None:
        kwargs['wage_age_profile'] = np.array(raw['wage_age_profile'])
    kwargs['pension_avg_weight'] = pension_avg_weight_for(
        raw, raw['model'].get('retirement_age', 40))

    return LifecycleConfig(**kwargs), eq_prices


def pension_avg_weight_for(raw, ret_age):
    """Weight on the last income state in the career-average pension base.

    The config's pension_avg_weight if set; otherwise
    lambda = (1 - rho^J_R) / (J_R (1 - rho)), the weight that makes the
    last-state base match the career average of an AR(1) with persistence rho
    over J_R working years, so it depends on the retirement age.
    """
    paw = raw.get('pension_avg_weight')
    if paw is not None:
        return paw
    rho = 0.95
    for p in raw.get('calibration', {}).get('params', []):
        if p['name'] == 'rho_y':
            rho = p['initial']
            break
    return (1 - rho ** ret_age) / (ret_age * (1 - rho))


def _sidecar_path(raw, key):
    """Absolute path of transition.<key>, or None when unset or missing."""
    rel = raw.get('transition', {}).get(key)
    if not rel:
        return None
    path = rel if os.path.isabs(rel) else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
    if not os.path.exists(path):
        print(f"  [load_config] {key} not found: {path}")
        return None
    return path


def cohort_retirement_table(raw):
    """{entry_year: (J_R, pension_avg_weight)} from transition.retirement_age_file.

    entry_year is the year the cohort enters at model age 0 (real age 25); J_R
    is its first period of retirement (build_retirement_age_GR.py). None when
    the file is not configured, in which case model.retirement_age applies to
    every cohort.
    """
    path = _sidecar_path(raw, 'retirement_age_file')
    if path is None:
        return None
    d = np.load(path)
    T = raw['model']['T']
    table = {}
    for k, J in zip(d['entry_years'].tolist(), d['J_R'].tolist()):
        if not 0 < J < T:
            raise ValueError(f'retirement index {J} for entry year {k} outside 1..{T - 1}')
        table[int(k)] = (int(J), pension_avg_weight_for(raw, int(J)))
    base = int(raw.get('transition', {}).get('current_year', 2023))
    J_model = raw['model'].get('retirement_age')
    # The cohort that turns 25 + J_model in the base year retires in it; its
    # retirement age is the base-year statutory age, which model.retirement_age
    # must equal for the base-year moments that read base_config.retirement_age.
    k_base = base - J_model
    if J_model is not None and k_base in table and table[k_base][0] != J_model:
        raise ValueError(f'model.retirement_age = {J_model} but the cohort retiring in '
                         f'{base} retires at index {table[k_base][0]} in the table')
    return table


def base_year_cohort_retirement(raw, T):
    """(J_R, pension_avg_weight) arrays, (T,) each, for the cohort aged 25+j
    in the base year, or (None, None) without a retirement table."""
    table = cohort_retirement_table(raw)
    if table is None:
        return None, None
    base = int(raw.get('transition', {}).get('current_year', 2023))
    years = sorted(table)
    J, lam = [], []
    for j in range(T):
        k = int(np.clip(base - j, years[0], years[-1]))
        J.append(table[k][0])
        lam.append(table[k][1])
    return np.array(J, dtype=int), np.array(lam, dtype=float)


def build_olg_transition(config_data, backend='numpy'):
    """Build an OLGTransition and transition paths from parsed JSON dict.

    Returns (economy, paths, T_tr) where paths is a dict with r_path, tau paths,
    pension_replacement_path, G_path, I_g_path, B_path, etc.
    """
    from olg_transition import OLGTransition

    lifecycle_config, eq_prices = build_lifecycle_config(config_data)
    r = config_data['prices']['r']
    prod = config_data.get('production', {})
    trans = config_data.get('transition', {})
    ext = config_data.get('external_params', {})
    T_tr = trans.get('T_transition', 60)

    # Demographic path: entering-cohort sizes, the population growth rate by
    # year, and a survival table that runs to the end of the transition. When
    # present it supersedes survival_data_file, whose historical table stops at
    # the base year and would hold mortality fixed over the whole transition.
    demography = None
    survival_table = None
    demog_path = _demography_path(config_data)
    if demog_path is not None:
        _d = np.load(demog_path)
        if _d['px'].shape[1] == lifecycle_config.T:
            demography = {k: _d[k] for k in ('entrant_years', 'entrants',
                                             'pop_years', 'n_path')}
            survival_table = (_d['years'], _d['px'])
        else:
            print(f"  [build_olg_transition] demography age dim {_d['px'].shape[1]} "
                  f"!= model T {lifecycle_config.T}; skipping.")

    surv_file = trans.get('survival_data_file')        # opt-in via config key
    if survival_table is None and surv_file:
        surv_path = surv_file if os.path.isabs(surv_file) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), surv_file)
        if os.path.exists(surv_path):
            _sd = np.load(surv_path)
            _yrs, _px = _sd['years'], _sd['px']
            if _px.shape[1] == lifecycle_config.T:     # only if age dim matches model T
                survival_table = (_yrs, _px)
            else:
                print(f"  [build_olg_transition] survival_data_file age dim {_px.shape[1]} "
                      f"!= model T {lifecycle_config.T}; skipping data survival.")
        else:
            print(f"  [build_olg_transition] survival_data_file not found: {surv_path}")

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
        r_B=config_data['prices'].get('r_B'),
        pop_growth=ext.get('pop_growth', 0.0),
        birth_year=trans.get('birth_year', 1960),
        current_year=trans.get('current_year', 2020),
        survival_table=survival_table,
        demography=demography,
        cohort_retirement=cohort_retirement_table(config_data),
        aggregation=trans.get('aggregation', 'simulation'),
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

    # Calibrated values written to _derived.theta (e.g. tau_p) override the
    # external_params disk values for the transition paths, so post-SMM runs use
    # the calibrated tax rates. theta keys are param names mapped via the field
    # map (e.g. JSON param 'tau_p' -> field 'tau_p_default'); here we key by the
    # external_params name directly.
    derived_theta = config_data.get('_derived', {}).get('theta', {})
    def _tax(name, default):
        return derived_theta.get(name, ext.get(name, default))
    paths = {
        'r_path': r_path,
        'tau_c_path': np.full(T_tr, _tax('tau_c', 0.20)),
        'tau_l_path': np.full(T_tr, _tax('tau_l', 0.10)),
        'tau_p_path': np.full(T_tr, _tax('tau_p', 0.20)),
        'tau_k_path': np.full(T_tr, _tax('tau_k', 0.20)),
        'pension_replacement_path': np.full(T_tr, _tax('pension_replacement_default', 0.50)),
    }

    # G and I_g paths (constant at data ratios × steady-state Y, will be rescaled after first sim)
    fiscal = config_data.get('fiscal', {})
    paths['G_over_Y'] = fiscal.get('G_over_Y', 0.13)
    paths['I_g_over_Y'] = fiscal.get('I_g_over_Y', 0.03)
    paths['defense_over_Y'] = fiscal.get('defense_over_Y', 0.0)
    paths['other_net_spending_over_Y'] = fiscal.get('other_net_spending_over_Y', 0.0)
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
    r = spec.r
    w = spec.w

    # Age weights (T,) — the measured base-year living cross-section, not a
    # stationary one; see base_year_age_weights
    aw = spec.age_weights if spec.age_weights is not None else np.ones(T) / T

    # --- Aggregate per-period means across education types ---
    # For each variable, compute age-weighted cross-sectional mean
    agg = {k: 0.0 for k in ['labor_income', 'consumption', 'assets',
                              'pension', 'ui', 'oop_health', 'gov_health',
                              'tax_c', 'tax_l', 'tax_p', 'tax_k', 'bequest',
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
            agg['labor_income'] += wt * _alive_mean(panel, 'effective_y_sim', t, a_t)
            agg['consumption'] += wt * _alive_mean(panel, 'c_sim', t, a_t)
            agg['assets'] += wt * _alive_mean(panel, 'a_sim', t, a_t)
            agg['pension'] += wt * _alive_mean(panel, 'pension_sim', t, a_t)
            agg['ui'] += wt * _alive_mean(panel, 'ui_sim', t, a_t)
            agg['oop_health'] += wt * _alive_mean(panel, 'oop_m_sim', t, a_t)
            agg['gov_health'] += wt * _alive_mean(panel, 'gov_m_sim', t, a_t)
            agg['tax_c'] += wt * _alive_mean(panel, 'tax_c_sim', t, a_t)
            agg['tax_l'] += wt * _alive_mean(panel, 'tax_l_sim', t, a_t)
            agg['tax_p'] += wt * _alive_mean(panel, 'tax_p_sim', t, a_t)
            agg['tax_k'] += wt * _alive_mean(panel, 'tax_k_sim', t, a_t)
            agg['bequest'] += wt * _alive_mean(panel, 'bequest_sim', t, a_t)
            agg['transfer'] += wt * _alive_mean(panel, 'transfer_sim', t, a_t)

    # --- Production side ---
    prod = config_data.get('production', {})
    alpha = prod.get('alpha', 0.33)
    A_tfp = prod.get('A_tfp', 1.0)
    K_g = prod.get('K_g', 0.0)
    eta_g = prod.get('eta_g', 0.0)
    K_g_factor = K_g ** eta_g if (K_g > 0 and eta_g > 0) else 1.0

    # L = labour in efficiency units = wage income / w. labor_income is
    # effective_y_sim = wage income + UI, so UI is netted out (same convention
    # as _compute_ss_aggregates and simulate_transition since 2026-10-02).
    L = (agg['labor_income'] - agg['ui']) / w if w > 0 else 0.0
    K_over_L = config_data.get('_derived', {}).get('K_over_L', 0.0)
    K_domestic = K_over_L * L
    Y = A_tfp * K_g_factor * K_domestic ** alpha * L ** (1.0 - alpha) if L > 0 else 0.0

    if Y <= 0:
        return {'error': 'Y <= 0, cannot compute ratios'}

    # --- Fiscal ratios ---
    fiscal_data = config_data.get('fiscal', {})
    B_over_Y = fiscal_data.get('B_over_Y', 0.0)
    # Sovereign debt service uses r_B (separate from the firm FOC return r).
    # Default to r if r_B not specified — preserves prior behavior.
    r_B = config_data.get('prices', {}).get('r_B', r)

    tax_revenue = agg['tax_c'] + agg['tax_l'] + agg['tax_p'] + agg['tax_k']
    # Bequest tax: the same line the transition's budget books
    # (olg_transition.compute_government_budget), so the base-year primary
    # balance that pins the closure sees the same revenue the transition does.
    tau_beq = float(getattr(spec.base_config, 'tau_beq', 0.0))
    bequest_tax = tau_beq * agg['bequest']
    # Means-tested transfers (consumption floor) are an outlay, booked here as
    # in the transition's budget (olg_transition.compute_government_budget).
    expenditure = (agg['pension'] + agg['ui'] + agg['gov_health'] + agg['transfer'] +
                   r_B * B_over_Y * Y)  # interest on debt at sovereign rate

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
        'health_gov_over_Y': agg['gov_health'] / Y,
        'health_oop_over_Y': agg['oop_health'] / Y,
        'health_total_over_Y': (agg['gov_health'] + agg['oop_health']) / Y,
        'interest_over_Y': r_B * B_over_Y,
        'bequests_over_Y': agg['bequest'] / Y,
        'bequest_tax_over_Y': bequest_tax / Y,
        'transfers_over_Y': agg['transfer'] / Y,
        'primary_balance_over_Y': (tax_revenue + bequest_tax - expenditure + r_B * B_over_Y * Y) / Y,
        'total_balance_over_Y': (tax_revenue + bequest_tax - expenditure) / Y,
    }

    # --- Full primary balance & baseline closure, pinned at the initial SS ---
    # primary_balance_over_Y above is the household-side balance
    # (tax_revenue + bequest_tax - pension - ui - gov_health - transfers)/Y,
    # interest cancelled out.
    # The transition's primary balance also nets out the discretionary spending
    # lines (G, I_g, defense) and the other-net closure residual, and excludes
    # interest — so add those here for a like-for-like full SS primary balance.
    G_over_Y       = fiscal_data.get('G_over_Y', 0.0)
    I_g_over_Y     = fiscal_data.get('I_g_over_Y', 0.0)
    defense_over_Y = fiscal_data.get('defense_over_Y', 0.0)
    other_over_Y   = fiscal_data.get('other_net_spending_over_Y', 0.0)
    discretionary  = G_over_Y + I_g_over_Y + defense_over_Y
    pb_house       = ratios['primary_balance_over_Y']
    ratios['primary_balance_full_over_Y'] = pb_house - discretionary - other_over_Y
    # Closure that pins the FULL SS primary surplus (other=0) to the data target.
    # other_net_spending is a structural SS constant; the transition takes it as
    # given, so its t=0 primary balance need not equal the target exactly.
    target = fiscal_data.get('primary_balance_target_over_Y', 0.0195)
    ratios['closure_other_over_Y'] = pb_house - discretionary - target

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
            'pensions_over_Y', 'ui_over_Y',
            'health_gov_over_Y', 'health_oop_over_Y', 'health_total_over_Y',
            'interest_over_Y', 'primary_balance_over_Y', 'total_balance_over_Y',
            'primary_balance_full_over_Y', 'closure_other_over_Y',
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
                        choices=['Nelder-Mead', 'differential_evolution', 'least_squares'],
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

    # Write calibrated theta back into the input JSON's _derived.theta block
    # so downstream consumers (build_olg_transition, run_fiscal_figures.py)
    # pick up the post-SMM values automatically. Re-read the file from disk
    # to preserve formatting and ignore the in-memory derived-prices fields.
    if args.config and result.get('convergence', False):
        with open(args.config) as f:
            raw_disk = json.load(f)
        raw_disk.setdefault('_derived', {})
        raw_disk['_derived']['theta'] = {
            p.name: float(v) for p, v in zip(spec.params, result['theta'])
        }
        raw_disk['_derived']['theta_metadata'] = {
            'calibration_date': datetime.now().isoformat(timespec='seconds'),
            'source_report': report_path if config_data is not None else None,
        }
        with open(args.config, 'w') as f:
            json.dump(raw_disk, f, indent=2)
        print(f"Calibrated theta written to {args.config}._derived.theta")

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
