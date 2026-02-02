"""
JAX-accelerated lifecycle model with perfect foresight.

Port of the computational core from lifecycle_perfect_foresight.py.
Uses vectorized operations and XLA compilation for massive speedup.
The NumPy implementation remains as reference/fallback.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from functools import partial
from scipy.linalg import eig

from lifecycle_perfect_foresight import LifecycleConfig, LifecycleModelPerfectForesight

# Enable float64 for numerical equivalence with NumPy reference
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def utility_jax(c, gamma):
    """CRRA utility, pure JAX function."""
    return jnp.where(
        gamma == 1.0,
        jnp.log(jnp.maximum(c, 1e-10)),
        (jnp.maximum(c, 1e-10) ** (1.0 - gamma)) / (1.0 - gamma),
    )


def _hsv_tax(income, tax_kappa, tax_eta):
    """HSV progressive tax: T(y) = y - κ·y^(1-η), non-negative."""
    tax = income - tax_kappa * jnp.maximum(income, 1e-10) ** (1 - tax_eta)
    return jnp.maximum(tax, 0.0)


def compute_budget_jax(
    a_grid, y_grid, h_grid, m_grid,
    P_y, P_h_t,
    r_t, w_t, w_at_retirement,
    tau_l_t, tau_p_t, tau_k_t,
    pension_replacement_t,
    ui_replacement_rate, kappa,
    is_retired,
    pension_min_floor=0.0,
    tax_progressive=False,
    tax_kappa_hsv=0.8,
    tax_eta=0.15,
    transfer_floor=0.0,
    child_cost_t=0.0,
    education_subsidy_rate=0.0,
    in_schooling=False,
):
    """
    Vectorised budget for ALL (n_a, n_y, n_h, n_y_last) states.

    Returns
    -------
    budget : array, shape (n_a, n_y, n_h, n_y)
    """
    n_a = a_grid.shape[0]
    n_y = y_grid.shape[0]
    n_h = h_grid.shape[0]

    # Broadcast grids: a(n_a,1,1,1), y(1,n_y,1,1), h(1,1,n_h,1), y_last(1,1,1,n_y)
    a = a_grid[:, None, None, None]           # (n_a,1,1,1)
    y = y_grid[None, :, None, None]           # (1,n_y,1,1)
    h = h_grid[None, None, :, None]           # (1,1,n_h,1)
    y_last = y_grid[None, None, None, :]      # (1,1,1,n_y)
    m = m_grid[None, None, :, None]           # (1,1,n_h,1)  — already age-indexed slice

    # --- Retired branch ---
    pension = pension_replacement_t * w_at_retirement * y_last       # (1,1,1,n_y)
    # Feature #11: minimum pension floor
    pension = jnp.maximum(pension, pension_min_floor)
    # Feature #14: progressive or flat tax
    retired_income_tax = jnp.where(
        tax_progressive,
        _hsv_tax(pension, tax_kappa_hsv, tax_eta),
        tau_l_t * pension,
    )
    retired_after_tax_labor = pension - retired_income_tax            # (1,1,1,n_y)

    # --- Working branch ---
    ui_benefit = ui_replacement_rate * w_t * y_last                   # (1,1,1,n_y)
    is_unemployed = (y == 0.0)                                        # (1,n_y,1,1)

    gross_wage_income = w_t * y * h                                   # (1,n_y,n_h,1)
    ui_term = jnp.where(is_unemployed, ui_benefit, 0.0)              # broadcast → (1,n_y,n_h,n_y)
    gross_labor_income = gross_wage_income + ui_term                  # (1,n_y,n_h,n_y)
    payroll_tax = tau_p_t * gross_wage_income                         # on wages only
    taxable_income = gross_labor_income - payroll_tax
    # Feature #14: progressive or flat income tax
    income_tax = jnp.where(
        tax_progressive,
        _hsv_tax(taxable_income, tax_kappa_hsv, tax_eta),
        tau_l_t * taxable_income,
    )
    working_after_tax_labor = gross_labor_income - payroll_tax - income_tax

    after_tax_labor = jnp.where(is_retired, retired_after_tax_labor, working_after_tax_labor)

    # Capital income (same for both branches)
    gross_capital_income = r_t * a
    capital_income_tax = tau_k_t * gross_capital_income
    after_tax_capital = gross_capital_income - capital_income_tax

    # Out-of-pocket health (m_grid is already age-indexed)
    oop_health = (1.0 - kappa) * m

    budget = a + after_tax_capital + after_tax_labor - oop_health     # (n_a, n_y, n_h, n_y)

    # Feature #4: schooling child costs
    net_child_cost = (1.0 - education_subsidy_rate) * child_cost_t
    budget = jnp.where(in_schooling, budget - net_child_cost, budget)

    # Feature #15: means-tested transfers (consumption floor)
    transfer = jnp.maximum(0.0, transfer_floor - budget)
    budget = budget + transfer

    return budget


# ---------------------------------------------------------------------------
# Vectorised single-period solve
# ---------------------------------------------------------------------------

def solve_period_jax(V_next, period_params, model_params):
    """
    Solve a single non-terminal period via grid search.

    Parameters
    ----------
    V_next : array (n_a, n_y, n_h, n_y)
        Continuation value from t+1.
    period_params : dict-like tuple
        (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t, pension_replacement_t,
         P_h_t, P_y_t, is_retired, survival_t, child_cost_t, in_schooling_t)
    model_params : dict-like tuple
        (a_grid, y_grid, h_grid, m_grid, P_y, w_at_retirement,
         ui_replacement_rate, kappa, beta, gamma,
         pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
         transfer_floor, education_subsidy_rate, P_y_age_health)

    Returns
    -------
    (V_t, a_pol_t, c_pol_t) each shape (n_a, n_y, n_h, n_y)
    """
    (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
     pension_replacement_t, P_h_t, P_y_t, is_retired,
     survival_t, child_cost_t, in_schooling_t) = period_params

    (a_grid, y_grid, h_grid, m_grid, P_y, w_at_retirement,
     ui_replacement_rate, kappa, beta, gamma,
     pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
     transfer_floor, education_subsidy_rate, P_y_age_health) = model_params

    n_a = a_grid.shape[0]
    n_y = y_grid.shape[0]
    n_h = h_grid.shape[0]

    # 1. Budget for all states: (n_a, n_y, n_h, n_y)
    budget = compute_budget_jax(
        a_grid, y_grid, h_grid, m_grid,
        P_y, P_h_t,
        r_t, w_t, w_at_retirement,
        tau_l_t, tau_p_t, tau_k_t,
        pension_replacement_t,
        ui_replacement_rate, kappa,
        is_retired,
        pension_min_floor=pension_min_floor,
        tax_progressive=tax_progressive,
        tax_kappa_hsv=tax_kappa_hsv,
        tax_eta=tax_eta,
        transfer_floor=transfer_floor,
        child_cost_t=child_cost_t,
        education_subsidy_rate=education_subsidy_rate,
        in_schooling=in_schooling_t,
    )

    # 2. Consumption candidates: (n_a, n_y, n_h, n_y, n_a_next)
    a_next = a_grid[None, None, None, None, :]
    c_all = (budget[..., None] - a_next) / (1.0 + tau_c_t)

    # 3. Expected continuation value
    # Contract over h': EV_h[a', y', i_h, y_last] = sum_{h'} P_h[i_h, h'] * V[a', y', h', y_last]
    EV_h = jnp.einsum('jk,aykl->ayjl', P_h_t, V_next)  # (n_a, n_y, n_h, n_y)

    # EV_h[a, yn, h, yl] -> transpose to [yl, a, h, yn]
    EV_h_t = jnp.transpose(EV_h, (3, 0, 2, 1))  # (n_y, n_a, n_h, n_y)

    # Feature #3+#5: P_y can be (n_y, n_y) constant or (n_h, n_y, n_y) age/health-dependent
    # P_y_t is the per-period slice: (n_h, n_y, n_y) when age-dependent, else same as P_y (n_y, n_y)
    # Working EV: EV_working[yl, a, h] = sum_yn P_y_eff[yl, yn, h] * EV_h_t[yl, a, h, yn]
    # When P_y is constant (2D): P_y[yl, yn], broadcast over h
    # When P_y_t is (n_h, n_y, n_y): P_y_t[h, yl, yn]
    EV_working_3d = jnp.where(
        P_y_age_health,
        # 4D case: P_y_t shape (n_h, n_y, n_y). For each (yl, a, h):
        # EV = sum_yn P_y_t[h, yl, yn] * EV_h_t[yl, a, h, yn]
        jnp.einsum('bij,iabj->iab', P_y_t, EV_h_t),  # (n_y, n_a, n_h)
        # 2D case: P_y[yl, yn]
        jnp.einsum('ij,iabj->iab', P_y, EV_h_t),     # (n_y, n_a, n_h)
    )

    EV_working = jnp.transpose(EV_working_3d, (1, 0, 2))  # (n_a, n_y, n_h)
    EV_working = EV_working[..., None] * jnp.ones((1, 1, 1, n_y))

    # Retired: V_next[a', 0, h', 0], contract over h' only
    V_next_ret = V_next[:, 0, :, 0]  # (n_a, n_h)
    EV_retired_2d = jnp.einsum('jk,ak->aj', P_h_t, V_next_ret)  # (n_a, n_h)
    EV_retired = EV_retired_2d[:, None, :, None] * jnp.ones((1, n_y, 1, n_y))

    EV = jnp.where(is_retired, EV_retired, EV_working)

    # Feature #2: survival risk — multiply EV by survival probability
    # survival_t shape: (n_h,) — broadcast to (1, 1, n_h, 1)
    survival_broadcast = survival_t[None, None, :, None]
    EV = EV * survival_broadcast

    # 4. Grid search
    EV_for_search = jnp.transpose(EV, (1, 2, 3, 0))  # (n_y, n_h, n_y, n_a_next)
    EV_for_search = EV_for_search[None, ...]

    u_all = utility_jax(c_all, gamma)
    val_all = u_all + beta * EV_for_search

    val_all = jnp.where(c_all > 0, val_all, -jnp.inf)

    best_a_idx = jnp.argmax(val_all, axis=-1)
    best_val = jnp.max(val_all, axis=-1)
    best_c = jnp.take_along_axis(c_all, best_a_idx[..., None], axis=-1)[..., 0]

    # 5. Fallback
    c_fallback = jnp.maximum(budget / (1.0 + tau_c_t), 1e-10)
    u_fallback = utility_jax(c_fallback, gamma)

    V_t = jnp.where(jnp.isfinite(best_val), best_val, u_fallback)
    a_pol_t = jnp.where(jnp.isfinite(best_val), best_a_idx, 0).astype(jnp.int32)
    c_pol_t = jnp.where(jnp.isfinite(best_val), best_c, c_fallback)

    return V_t, a_pol_t, c_pol_t


def _solve_terminal_period_jax(
    a_grid, y_grid, h_grid, m_grid,
    P_y, P_h_T, w_at_retirement,
    r_T, w_T, tau_c_T, tau_l_T, tau_p_T, tau_k_T,
    pension_replacement_T,
    ui_replacement_rate, kappa,
    is_retired_T, gamma,
    pension_min_floor=0.0,
    tax_progressive=False,
    tax_kappa_hsv=0.8,
    tax_eta=0.15,
    transfer_floor=0.0,
    child_cost_T=0.0,
    education_subsidy_rate=0.0,
    in_schooling_T=False,
):
    """Solve terminal period: consume everything, a'=0."""
    budget = compute_budget_jax(
        a_grid, y_grid, h_grid, m_grid,
        P_y, P_h_T,
        r_T, w_T, w_at_retirement,
        tau_l_T, tau_p_T, tau_k_T,
        pension_replacement_T,
        ui_replacement_rate, kappa,
        is_retired_T,
        pension_min_floor=pension_min_floor,
        tax_progressive=tax_progressive,
        tax_kappa_hsv=tax_kappa_hsv,
        tax_eta=tax_eta,
        transfer_floor=transfer_floor,
        child_cost_t=child_cost_T,
        education_subsidy_rate=education_subsidy_rate,
        in_schooling=in_schooling_T,
    )
    c = jnp.maximum(budget / (1.0 + tau_c_T), 1e-10)
    V = utility_jax(c, gamma)
    a_pol = jnp.zeros_like(V, dtype=jnp.int32)
    return V, a_pol, c


def solve_lifecycle_jax(
    a_grid, y_grid, h_grid, m_grid,
    P_y, P_h,                       # P_h: (T, n_h, n_h)
    w_at_retirement,
    r_path, w_path,
    tau_c_path, tau_l_path, tau_p_path, tau_k_path,
    pension_replacement_path,
    ui_replacement_rate, kappa,
    beta, gamma,
    T, retirement_age,
    pension_min_floor=0.0,
    tax_progressive=False,
    tax_kappa_hsv=0.8,
    tax_eta=0.15,
    transfer_floor=0.0,
    education_subsidy_rate=0.0,
    child_cost_profile=None,
    schooling_years=0,
    survival_probs=None,
    P_y_by_age_health=None,
):
    """
    Full backward induction using jax.lax.scan.

    Returns
    -------
    V : (T, n_a, n_y, n_h, n_y)
    a_policy : (T, n_a, n_y, n_h, n_y), int32
    c_policy : (T, n_a, n_y, n_h, n_y)
    """
    n_a = a_grid.shape[0]
    n_y = y_grid.shape[0]
    n_h = h_grid.shape[0]

    P_y_age_health = P_y_by_age_health is not None

    # Build survival array: (T, n_h), default all 1.0
    if survival_probs is None:
        survival_arr = jnp.ones((T, n_h))
    else:
        survival_arr = survival_probs  # (T, n_h)

    # Build child cost array
    if child_cost_profile is None:
        child_costs = jnp.zeros(T)
    else:
        child_costs = child_cost_profile

    # P_y for age-health case: (T, n_h, n_y, n_y). Else P_y is (n_y, n_y).
    # For the scan, we need a per-period P_y_t. When constant, we replicate.
    if P_y_age_health:
        P_y_scan = P_y_by_age_health  # (T, n_h, n_y, n_y)
    else:
        # Tile to (T, n_h, n_y, n_y) so scan indexing works uniformly
        P_y_scan = jnp.tile(P_y[None, None, :, :], (T, n_h, 1, 1))

    # m_grid can be (T, n_h) or (n_h,). Ensure (T, n_h) for per-period indexing.
    if m_grid.ndim == 1:
        m_grid_path = jnp.tile(m_grid[None, :], (T, 1))  # (T, n_h)
    else:
        m_grid_path = m_grid  # Already (T, n_h)

    # Use a dummy m_grid_base (n_h,) for model_params (not used directly in budget)
    m_grid_base = m_grid_path[0]  # (n_h,)

    model_params = (a_grid, y_grid, h_grid, m_grid_base, P_y, w_at_retirement,
                    ui_replacement_rate, kappa, beta, gamma,
                    pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
                    transfer_floor, education_subsidy_rate, P_y_age_health)

    # Terminal period
    is_retired_T = (T - 1) >= retirement_age

    V_T, a_pol_T, c_pol_T = _solve_terminal_period_jax(
        a_grid, y_grid, h_grid, m_grid_path[T - 1],
        P_y, P_h[T - 1], w_at_retirement,
        r_path[T - 1], w_path[T - 1],
        tau_c_path[T - 1], tau_l_path[T - 1], tau_p_path[T - 1], tau_k_path[T - 1],
        pension_replacement_path[T - 1],
        ui_replacement_rate, kappa, is_retired_T, gamma,
        pension_min_floor=pension_min_floor,
        tax_progressive=tax_progressive,
        tax_kappa_hsv=tax_kappa_hsv,
        tax_eta=tax_eta,
        transfer_floor=transfer_floor,
        child_cost_T=child_costs[T - 1],
        education_subsidy_rate=education_subsidy_rate,
        in_schooling_T=(T - 1) < schooling_years,
    )

    # Stack period params for t = T-2 ... 0 (reversed)
    ts = jnp.arange(T - 2, -1, -1)  # [T-2, T-3, ..., 0]

    period_params_stack = (
        r_path[ts],
        w_path[ts],
        tau_c_path[ts],
        tau_l_path[ts],
        tau_p_path[ts],
        tau_k_path[ts],
        pension_replacement_path[ts],
        P_h[ts],                        # (T-1, n_h, n_h)
        P_y_scan[ts],                   # (T-1, n_h, n_y, n_y)
        (ts >= retirement_age),         # is_retired for each t
        survival_arr[ts],               # (T-1, n_h)
        child_costs[ts],                # (T-1,)
        (ts < schooling_years),         # in_schooling for each t
        m_grid_path[ts],                # (T-1, n_h) — per-period medical costs
    )

    def scan_fn(V_next, period_params_slice):
        (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
         pension_replacement_t, P_h_t, P_y_t, is_retired,
         survival_t, child_cost_t, in_schooling_t, m_grid_t) = period_params_slice

        # Override m_grid in model_params with per-period slice
        model_params_t = model_params[:3] + (m_grid_t,) + model_params[4:]

        V_t, a_pol_t, c_pol_t = solve_period_jax(
            V_next,
            (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
             pension_replacement_t, P_h_t, P_y_t, is_retired,
             survival_t, child_cost_t, in_schooling_t),
            model_params_t,
        )
        return V_t, (V_t, a_pol_t, c_pol_t)

    V_final, (V_scan, a_pol_scan, c_pol_scan) = lax.scan(
        scan_fn, V_T, period_params_stack
    )

    # Reverse to natural order and append terminal period
    V_scan = V_scan[::-1]
    a_pol_scan = a_pol_scan[::-1]
    c_pol_scan = c_pol_scan[::-1]

    V = jnp.concatenate([V_scan, V_T[None]], axis=0)
    a_policy = jnp.concatenate([a_pol_scan, a_pol_T[None]], axis=0)
    c_policy = jnp.concatenate([c_pol_scan, c_pol_T[None]], axis=0)

    return V, a_policy, c_policy


# JIT-compile with static args for shapes and scalar params
_solve_lifecycle_jax_jit = jax.jit(
    solve_lifecycle_jax,
    static_argnames=('T', 'retirement_age', 'tax_progressive', 'schooling_years'),
)

# Batched solve: vmap over cohorts with shared grids/transitions.
# Per-cohort inputs (in_axes=0): w_at_retirement, r/w/tax/pension paths.
# Shared inputs (in_axes=None): grids, P_y, P_h, scalars + new feature params.
_solve_lifecycle_jax_batched = jax.jit(
    jax.vmap(solve_lifecycle_jax, in_axes=(
        None, None, None, None,  # a_grid, y_grid, h_grid, m_grid
        None, None,              # P_y, P_h
        0,                       # w_at_retirement
        0, 0,                    # r_path, w_path
        0, 0, 0, 0,             # tau_c/l/p/k_path
        0,                       # pension_replacement_path
        None, None,              # ui_replacement_rate, kappa
        None, None,              # beta, gamma
        None, None,              # T, retirement_age
        None, None,              # pension_min_floor, tax_progressive
        None, None,              # tax_kappa_hsv, tax_eta
        None, None,              # transfer_floor, education_subsidy_rate
        None, None,              # child_cost_profile, schooling_years
        None, None,              # survival_probs, P_y_by_age_health
    )),
    static_argnames=('T', 'retirement_age', 'tax_progressive', 'schooling_years'),
)


# ---------------------------------------------------------------------------
# Phase 2: JAX simulation
# ---------------------------------------------------------------------------

def _agent_step_jax(carry, t_data, a_policy, c_policy,
                    a_grid, y_grid, h_grid, m_grid,
                    P_y, P_h,
                    w_path, w_at_retirement,
                    tau_c_path, tau_l_path, tau_p_path, tau_k_path,
                    r_path, pension_replacement_path,
                    ui_replacement_rate, kappa,
                    retirement_age, T, current_age,
                    pension_min_floor=0.0,
                    tax_progressive=False,
                    tax_kappa_hsv=0.8,
                    tax_eta=0.15,
                    P_y_age_health=False,
                    P_y_4d=None):
    """
    Single time-step for one agent.

    carry: (i_a, i_y, i_h, i_y_last, avg_earnings, n_earnings_years)
    t_data: (t_sim_idx, u_y, u_h)
    """
    i_a, i_y, i_h, i_y_last, avg_earnings, n_earnings_years = carry
    t_sim_idx, u_y, u_h = t_data

    lifecycle_age = current_age + t_sim_idx
    is_retired = lifecycle_age >= retirement_age
    is_last_step = (t_sim_idx == (T - current_age - 1))

    # Look up policy
    a_pol_val = a_policy[lifecycle_age, i_a, i_y, i_h, i_y_last]
    c_pol_val = c_policy[lifecycle_age, i_a, i_y, i_h, i_y_last]

    # Current state values
    a_val = a_grid[i_a]
    y_val = y_grid[i_y]
    h_val = h_grid[i_h]
    y_last_val = y_grid[i_y_last]

    # Pension with floor
    pension_replacement = pension_replacement_path[lifecycle_age]
    pension_raw = pension_replacement * w_at_retirement * y_last_val
    pension_with_floor = jnp.maximum(pension_raw, pension_min_floor)
    pension = jnp.where(is_retired, pension_with_floor, 0.0)

    # UI
    ui = jnp.where(
        (~is_retired) & (i_y == 0),
        ui_replacement_rate * w_path[lifecycle_age] * y_last_val,
        0.0,
    )

    # Employment
    employed = (~is_retired) & (i_y > 0)

    # Effective income
    wage_income = w_path[lifecycle_age] * y_val * h_val
    effective_y = jnp.where(is_retired, 0.0, wage_income + ui)

    # Health expenditure — m_grid is (T, n_h) now
    m_val = m_grid[lifecycle_age, i_h]
    oop_m = (1.0 - kappa) * m_val
    gov_m = kappa * m_val

    # Taxes
    r_t = r_path[lifecycle_age]
    tax_c = tau_c_path[lifecycle_age] * c_pol_val

    # Payroll tax on wages only
    tax_p = jnp.where(is_retired, 0.0, tau_p_path[lifecycle_age] * wage_income)

    # Labor income tax (progressive or flat)
    taxable_retired = pension
    taxable_working = effective_y - tax_p
    tax_l = jnp.where(
        is_retired,
        jnp.where(tax_progressive,
                   _hsv_tax(taxable_retired, tax_kappa_hsv, tax_eta),
                   tau_l_path[lifecycle_age] * taxable_retired),
        jnp.where(tax_progressive,
                   _hsv_tax(taxable_working, tax_kappa_hsv, tax_eta),
                   tau_l_path[lifecycle_age] * taxable_working),
    )

    # Capital income tax
    gross_capital = r_t * a_val
    tax_k = tau_k_path[lifecycle_age] * gross_capital

    # Update average earnings (working years only)
    new_n_years = jnp.where(is_retired, n_earnings_years, n_earnings_years + 1)
    new_avg_earnings = jnp.where(
        is_retired,
        avg_earnings,
        jnp.where(
            new_n_years > 0,
            (avg_earnings * n_earnings_years + wage_income) / new_n_years,
            wage_income,
        ),
    )

    # --- State transitions ---
    new_i_a = jnp.where(is_last_step, i_a, a_pol_val)

    # Next income state — handle age/health-dependent P_y
    P_y_row = jnp.where(
        P_y_age_health,
        P_y_4d[lifecycle_age, i_h, i_y, :],
        P_y[i_y, :],
    )
    cum_P_y = jnp.cumsum(P_y_row)
    new_i_y_draw = jnp.searchsorted(cum_P_y, u_y)
    new_i_y_draw = jnp.clip(new_i_y_draw, 0, P_y.shape[-1] - 1)
    new_i_y = jnp.where(is_retired, 0, new_i_y_draw)
    new_i_y_last = jnp.where(is_retired, i_y_last, i_y)

    # Next health state
    cum_P_h = jnp.cumsum(P_h[lifecycle_age, i_h, :])
    new_i_h = jnp.searchsorted(cum_P_h, u_h)
    new_i_h = jnp.clip(new_i_h, 0, P_h.shape[2] - 1)

    new_carry = (new_i_a.astype(jnp.int32), new_i_y.astype(jnp.int32),
                 new_i_h.astype(jnp.int32), new_i_y_last.astype(jnp.int32),
                 new_avg_earnings, new_n_years)

    step_out = (
        a_val, c_pol_val, y_val, h_val, i_h,
        effective_y, employed, ui,
        m_val, oop_m, gov_m,
        tax_c, tax_l, tax_p, tax_k,
        new_avg_earnings, pension, is_retired,
    )

    return new_carry, step_out


def simulate_lifecycle_jax(
    a_policy, c_policy,
    a_grid, y_grid, h_grid, m_grid,
    P_y, P_h,
    w_path, w_at_retirement,
    tau_c_path, tau_l_path, tau_p_path, tau_k_path,
    r_path, pension_replacement_path,
    ui_replacement_rate, kappa,
    retirement_age, T, current_age,
    n_sim, key,
    initial_i_a, initial_i_y, initial_i_h, initial_i_y_last,
    initial_avg_earnings, initial_n_earnings_years,
    pension_min_floor=0.0,
    tax_progressive=False,
    tax_kappa_hsv=0.8,
    tax_eta=0.15,
    P_y_age_health=False,
    P_y_4d=None,
):
    """
    Simulate lifecycle paths for n_sim agents using vmap + lax.scan.

    Returns tuple of 18 arrays, each shape (T_sim, n_sim).
    """
    T_sim = T - current_age

    # Pre-generate random draws
    key1, key2 = jax.random.split(key)
    u_y_all = jax.random.uniform(key1, shape=(T_sim, n_sim))
    u_h_all = jax.random.uniform(key2, shape=(T_sim, n_sim))
    t_indices = jnp.arange(T_sim)

    # Dummy P_y_4d if not provided (for JAX tracing)
    if P_y_4d is None:
        n_y = P_y.shape[0]
        n_h = P_h.shape[1]
        P_y_4d = jnp.zeros((T, n_h, n_y, n_y))

    step_fn = partial(
        _agent_step_jax,
        a_policy=a_policy, c_policy=c_policy,
        a_grid=a_grid, y_grid=y_grid, h_grid=h_grid, m_grid=m_grid,
        P_y=P_y, P_h=P_h,
        w_path=w_path, w_at_retirement=w_at_retirement,
        tau_c_path=tau_c_path, tau_l_path=tau_l_path,
        tau_p_path=tau_p_path, tau_k_path=tau_k_path,
        r_path=r_path, pension_replacement_path=pension_replacement_path,
        ui_replacement_rate=ui_replacement_rate, kappa=kappa,
        retirement_age=retirement_age, T=T, current_age=current_age,
        pension_min_floor=pension_min_floor,
        tax_progressive=tax_progressive,
        tax_kappa_hsv=tax_kappa_hsv,
        tax_eta=tax_eta,
        P_y_age_health=P_y_age_health,
        P_y_4d=P_y_4d,
    )

    def simulate_one(init_state, u_y_seq, u_h_seq):
        """Scan over T_sim steps for one agent."""
        xs = (t_indices, u_y_seq, u_h_seq)
        _, outputs = lax.scan(step_fn, init_state, xs)
        return outputs

    # vmap across n_sim agents
    init_states = (initial_i_a, initial_i_y, initial_i_h, initial_i_y_last,
                   initial_avg_earnings, initial_n_earnings_years)

    # u_y_all, u_h_all are (T_sim, n_sim) — we vmap over axis 1 (agents)
    all_outputs = jax.vmap(
        simulate_one,
        in_axes=(0, 1, 1),  # init_state per-agent, draws per-agent
    )(init_states, u_y_all, u_h_all)

    # all_outputs is a tuple of 18 arrays, each (n_sim, T_sim) from vmap
    # Transpose each to (T_sim, n_sim) to match NumPy convention
    result = tuple(out.T if out.ndim == 2 else out.T for out in all_outputs)

    return result


_simulate_lifecycle_jax_jit = jax.jit(
    simulate_lifecycle_jax,
    static_argnames=('retirement_age', 'T', 'current_age', 'n_sim',
                     'tax_progressive', 'P_y_age_health'),
)

# Batched simulation: vmap over cohorts with shared grids/transitions.
_simulate_lifecycle_jax_batched = jax.jit(
    jax.vmap(simulate_lifecycle_jax, in_axes=(
        0, 0,                    # a_policy, c_policy
        None, None, None, None,  # a_grid, y_grid, h_grid, m_grid
        None, None,              # P_y, P_h
        0, 0,                    # w_path, w_at_retirement
        0, 0, 0, 0,             # tau_c/l/p/k_path
        0, 0,                    # r_path, pension_replacement_path
        None, None,              # ui_replacement_rate, kappa
        None, None, None,        # retirement_age, T, current_age
        None, 0,                 # n_sim, key
        0, 0, 0, 0,             # initial_i_a/y/h/y_last
        0, 0,                    # initial_avg_earnings, initial_n_earnings_years
        None, None,              # pension_min_floor, tax_progressive
        None, None,              # tax_kappa_hsv, tax_eta
        None, None,              # P_y_age_health, P_y_4d
    )),
    static_argnames=('retirement_age', 'T', 'current_age', 'n_sim',
                     'tax_progressive', 'P_y_age_health'),
)


# ---------------------------------------------------------------------------
# Wrapper class: same interface as LifecycleModelPerfectForesight
# ---------------------------------------------------------------------------

class LifecycleModelJAX:
    """
    JAX-accelerated lifecycle model.

    Same public interface as LifecycleModelPerfectForesight:
        __init__(config, verbose)
        solve(verbose)
        simulate(T_sim, n_sim, seed)
    """

    def __init__(self, config: LifecycleConfig, verbose: bool = True):
        # Build a NumPy reference model for grids, income process, etc.
        self._np_model = LifecycleModelPerfectForesight(config, verbose=verbose)
        self.config = config
        self.verbose = verbose

        # Copy frequently used attributes
        self.T = self._np_model.T
        self.beta = self._np_model.beta
        self.gamma = self._np_model.gamma
        self.n_a = self._np_model.n_a
        self.n_y = self._np_model.n_y
        self.n_h = self._np_model.n_h
        self.current_age = self._np_model.current_age
        self.retirement_age = self._np_model.retirement_age
        self.ui_replacement_rate = self._np_model.ui_replacement_rate
        self.kappa = self._np_model.kappa

        # Convert grids and processes to JAX arrays
        self.a_grid = jnp.array(self._np_model.a_grid)
        self.y_grid = jnp.array(self._np_model.y_grid)
        self.h_grid = jnp.array(self._np_model.h_grid)
        self.m_grid = jnp.array(self._np_model.m_grid)  # Now (T, n_h)
        self.P_y = jnp.array(self._np_model.P_y)  # (n_y, n_y) or (T, n_h, n_y, n_y)
        self.P_h = jnp.array(self._np_model.P_h)
        self.w_at_retirement = float(self._np_model.w_at_retirement)

        # New feature parameters
        self.pension_min_floor = float(config.pension_min_floor)
        self.tax_progressive = bool(config.tax_progressive)
        self.tax_kappa_hsv = float(config.tax_kappa)
        self.tax_eta = float(config.tax_eta)
        self.transfer_floor = float(config.transfer_floor)
        self.education_subsidy_rate = float(config.education_subsidy_rate)
        self.schooling_years = int(config.schooling_years)
        self.child_cost_profile = jnp.array(config.child_cost_profile)
        self.P_y_age_health = self._np_model.P_y_age_health

        # Survival probabilities
        if config.survival_probs is not None:
            self.survival_probs = jnp.array(config.survival_probs)
        else:
            self.survival_probs = None

        # P_y for age/health-dependent case
        if self.P_y_age_health:
            self.P_y_4d = self.P_y  # Already (T, n_h, n_y, n_y)
            # Keep a 2D P_y for the constant path (initial period, health 0)
            self.P_y_2d = self.P_y[0, 0]
        else:
            self.P_y_4d = None
            self.P_y_2d = self.P_y

        # Paths
        self.r_path = jnp.array(self._np_model.r_path)
        self.w_path = jnp.array(self._np_model.w_path)
        self.tau_c_path = jnp.array(self._np_model.tau_c_path)
        self.tau_l_path = jnp.array(self._np_model.tau_l_path)
        self.tau_p_path = jnp.array(self._np_model.tau_p_path)
        self.tau_k_path = jnp.array(self._np_model.tau_k_path)
        self.pension_replacement_path = jnp.array(self._np_model.pension_replacement_path)

        # Placeholders for results
        self.V = None
        self.a_policy = None
        self.c_policy = None

    def solve(self, verbose=False, **kwargs):
        """Solve lifecycle via JAX backward induction."""
        if verbose:
            print(f"Solving lifecycle model (JAX) for {self.config.education_type} education...")

        # m_grid is (T, n_h) — pass the per-period m_grid slice via the scan
        # For terminal period in solve, m_grid needs to be indexed.
        # The solve_lifecycle_jax function handles per-period indexing internally
        # BUT compute_budget_jax expects m_grid as a 1D (n_h,) slice.
        # We need to pass the full (T, n_h) and index per-period in the scan.
        # Actually, compute_budget_jax receives m_grid and broadcasts.
        # For the terminal period, we pass m_grid[T-1].
        # For the scan, we pass m_grid[ts] per period.
        # The simplest approach: pass per-period m_grid slices in the period_params_stack.
        # But the current API passes m_grid as a model param (shared across periods).
        # We need to modify: pass m_grid_path (T, n_h) and index per period.

        # For now, pass the full m_grid to solve_lifecycle_jax and let it handle per-period indexing.
        # But compute_budget_jax expects a 1D slice... We need to restructure.
        # Actually let's just pass the per-period slices through the scan params.

        V, a_policy, c_policy = _solve_lifecycle_jax_jit(
            self.a_grid, self.y_grid, self.h_grid, self.m_grid,
            self.P_y_2d if not self.P_y_age_health else self.P_y_2d,
            self.P_h,
            self.w_at_retirement,
            self.r_path, self.w_path,
            self.tau_c_path, self.tau_l_path, self.tau_p_path, self.tau_k_path,
            self.pension_replacement_path,
            self.ui_replacement_rate, self.kappa,
            self.beta, self.gamma,
            self.T, self.retirement_age,
            pension_min_floor=self.pension_min_floor,
            tax_progressive=self.tax_progressive,
            tax_kappa_hsv=self.tax_kappa_hsv,
            tax_eta=self.tax_eta,
            transfer_floor=self.transfer_floor,
            education_subsidy_rate=self.education_subsidy_rate,
            child_cost_profile=self.child_cost_profile,
            schooling_years=self.schooling_years,
            survival_probs=self.survival_probs,
            P_y_by_age_health=self.P_y_4d if self.P_y_age_health else None,
        )

        # Store as numpy for compatibility with downstream code
        self.V = np.asarray(V)
        self.a_policy = np.asarray(a_policy)
        self.c_policy = np.asarray(c_policy)

        if verbose:
            print("Done!")

    def simulate(self, T_sim=None, n_sim=10000, seed=42, **kwargs):
        """
        Simulate lifecycle paths.

        Returns same 18-tuple as LifecycleModelPerfectForesight.simulate().
        """
        if self.V is None:
            raise RuntimeError("Must call solve() before simulate().")

        if T_sim is None:
            T_sim = self.T - self.current_age

        key = jax.random.PRNGKey(seed)

        # Initial conditions (same logic as NumPy model)
        edu_unemployment_rate = self.config.edu_params[self.config.education_type]['unemployment_rate']

        if edu_unemployment_rate < 1e-10:
            n_employed = self.n_y - 1
            # Uniform over employed states
            key, subkey = jax.random.split(key)
            initial_i_y = jax.random.choice(subkey, jnp.arange(1, self.n_y), shape=(n_sim,)).astype(jnp.int32)
        else:
            # Stationary distribution — use 2D P_y (handles 4D case)
            eigenvalues, eigenvectors = eig(np.asarray(self.P_y_2d).T)
            stationary_idx = np.argmax(eigenvalues.real)
            stationary = eigenvectors[:, stationary_idx].real
            stationary = stationary / stationary.sum()
            key, subkey = jax.random.split(key)
            initial_i_y = jax.random.choice(subkey, self.n_y, shape=(n_sim,),
                                            p=jnp.array(stationary)).astype(jnp.int32)

        initial_i_y_last = initial_i_y.copy()
        initial_i_h = jnp.zeros(n_sim, dtype=jnp.int32)

        if self.config.initial_assets is not None:
            i_a_initial = int(jnp.argmin(jnp.abs(self.a_grid - self.config.initial_assets)))
            initial_i_a = jnp.full(n_sim, i_a_initial, dtype=jnp.int32)
        else:
            initial_i_a = jnp.zeros(n_sim, dtype=jnp.int32)

        if self.config.initial_avg_earnings is not None:
            initial_avg_earnings = jnp.ones(n_sim) * self.config.initial_avg_earnings
            initial_n_years = jnp.full(n_sim, self.current_age, dtype=jnp.float64)
        else:
            initial_avg_earnings = jnp.zeros(n_sim)
            initial_n_years = jnp.zeros(n_sim, dtype=jnp.float64)

        # Convert policies to JAX for simulation
        a_policy_jax = jnp.array(self.a_policy)
        c_policy_jax = jnp.array(self.c_policy)

        key, subkey = jax.random.split(key)

        # Build P_y_4d dummy for JAX tracing if not age/health-dependent
        P_y_4d_sim = self.P_y_4d if self.P_y_age_health else None

        result = _simulate_lifecycle_jax_jit(
            a_policy_jax, c_policy_jax,
            self.a_grid, self.y_grid, self.h_grid, self.m_grid,
            self.P_y_2d, self.P_h,
            self.w_path, self.w_at_retirement,
            self.tau_c_path, self.tau_l_path, self.tau_p_path, self.tau_k_path,
            self.r_path, self.pension_replacement_path,
            self.ui_replacement_rate, self.kappa,
            self.retirement_age, self.T, self.current_age,
            n_sim, subkey,
            initial_i_a, initial_i_y, initial_i_h, initial_i_y_last,
            initial_avg_earnings, initial_n_years,
            pension_min_floor=self.pension_min_floor,
            tax_progressive=self.tax_progressive,
            tax_kappa_hsv=self.tax_kappa_hsv,
            tax_eta=self.tax_eta,
            P_y_age_health=self.P_y_age_health,
            P_y_4d=P_y_4d_sim,
        )

        # Convert all outputs to numpy arrays
        return tuple(np.asarray(x) for x in result)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    if "--test" in sys.argv:
        print("=" * 70)
        print("JAX LIFECYCLE MODEL — CROSS-VALIDATION TEST")
        print("=" * 70)

        config = LifecycleConfig(
            T=10,
            beta=0.96,
            gamma=2.0,
            current_age=0,
            retirement_age=8,
            pension_replacement_default=0.40,
            education_type='medium',
            n_a=50,
            n_y=2,
            n_h=1,
            m_good=0.0,
        )

        # NumPy reference
        print("\n--- NumPy solve ---")
        t0 = time.time()
        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)
        t_np = time.time() - t0
        print(f"  Time: {t_np:.3f}s")

        # JAX solve
        print("\n--- JAX solve ---")
        t0 = time.time()
        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)
        t_jax = time.time() - t0
        print(f"  Time: {t_jax:.3f}s  (includes JIT compilation)")

        # Compare V
        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        print(f"\n  Max |V_numpy - V_jax| = {V_diff:.2e}")

        # Compare policies
        a_match = np.all(np_model.a_policy == jax_model.a_policy)
        print(f"  Asset policies identical: {a_match}")
        if not a_match:
            n_diff = np.sum(np_model.a_policy != jax_model.a_policy)
            n_total = np_model.a_policy.size
            print(f"  Differing entries: {n_diff}/{n_total} ({100*n_diff/n_total:.2f}%)")

        c_diff = np.max(np.abs(np_model.c_policy - jax_model.c_policy))
        print(f"  Max |c_numpy - c_jax| = {c_diff:.2e}")

        # Second JIT call (warm)
        print("\n--- JAX solve (warm JIT) ---")
        t0 = time.time()
        jax_model2 = LifecycleModelJAX(config, verbose=False)
        jax_model2.solve(verbose=False)
        t_jax2 = time.time() - t0
        print(f"  Time: {t_jax2:.3f}s")

        # Simulation comparison
        print("\n--- Simulation comparison ---")
        np_results = np_model.simulate(n_sim=5000, seed=42)
        jax_results = jax_model.simulate(n_sim=5000, seed=42)

        print(f"  NumPy mean assets: {np.mean(np_results[0]):.4f}")
        print(f"  JAX   mean assets: {np.mean(jax_results[0]):.4f}")
        print(f"  NumPy mean consumption: {np.mean(np_results[1]):.4f}")
        print(f"  JAX   mean consumption: {np.mean(jax_results[1]):.4f}")

        print("\n" + "=" * 70)
        if V_diff < 1e-6:
            print("PASS: Value functions match within 1e-6")
        else:
            print(f"WARN: Value functions differ by {V_diff:.2e}")
        print("=" * 70)
    else:
        print("Usage: python lifecycle_jax.py --test")
