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


def compute_budget_jax(
    a_grid, y_grid, h_grid, m_grid,
    P_y, P_h_t,
    r_t, w_t, w_at_retirement,
    tau_l_t, tau_p_t, tau_k_t,
    pension_replacement_t,
    ui_replacement_rate, kappa,
    is_retired,
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
    m = m_grid[None, None, :, None]           # (1,1,n_h,1)

    # --- Retired branch ---
    pension = pension_replacement_t * w_at_retirement * y_last       # (1,1,1,n_y)
    retired_income_tax = tau_l_t * pension
    retired_after_tax_labor = pension - retired_income_tax            # (1,1,1,n_y)

    # --- Working branch ---
    # i_y==0 means unemployed => UI benefit based on y_last
    ui_benefit = ui_replacement_rate * w_t * y_last                   # (1,1,1,n_y)
    # For employed (i_y>0), UI is 0 — we mask with is_unemployed later
    is_unemployed = (y == 0.0)                                        # (1,n_y,1,1)

    gross_wage_income = w_t * y * h                                   # (1,n_y,n_h,1)
    # UI only if unemployed
    ui_term = jnp.where(is_unemployed, ui_benefit, 0.0)              # broadcast → (1,n_y,n_h,n_y)
    gross_labor_income = gross_wage_income + ui_term                  # (1,n_y,n_h,n_y)
    payroll_tax = tau_p_t * gross_wage_income                         # on wages only
    income_tax = tau_l_t * (gross_labor_income - payroll_tax)
    working_after_tax_labor = gross_labor_income - payroll_tax - income_tax

    after_tax_labor = jnp.where(is_retired, retired_after_tax_labor, working_after_tax_labor)

    # Capital income (same for both branches)
    gross_capital_income = r_t * a
    capital_income_tax = tau_k_t * gross_capital_income
    after_tax_capital = gross_capital_income - capital_income_tax

    # Out-of-pocket health
    oop_health = (1.0 - kappa) * m

    budget = a + after_tax_capital + after_tax_labor - oop_health     # (n_a, n_y, n_h, n_y)
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
        (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t, pension_replacement_t, P_h_t, is_retired)
    model_params : dict-like tuple
        (a_grid, y_grid, h_grid, m_grid, P_y, w_at_retirement,
         ui_replacement_rate, kappa, beta, gamma)

    Returns
    -------
    (V_t, a_pol_t, c_pol_t) each shape (n_a, n_y, n_h, n_y)
    """
    (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
     pension_replacement_t, P_h_t, is_retired) = period_params

    (a_grid, y_grid, h_grid, m_grid, P_y, w_at_retirement,
     ui_replacement_rate, kappa, beta, gamma) = model_params

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
    )

    # 2. Consumption candidates: (n_a, n_y, n_h, n_y, n_a_next)
    a_next = a_grid[None, None, None, None, :]                        # (1,1,1,1,n_a)
    c_all = (budget[..., None] - a_next) / (1.0 + tau_c_t)           # (n_a,n_y,n_h,n_y,n_a)

    # 3. Expected continuation value
    # Working: EV[i_a_next, i_y, i_h, i_y_last]
    #   = sum_{y'} P_y[i_y, y'] * sum_{h'} P_h[i_h, h'] * V_next[i_a_next, y', h', i_y]
    # Note: for working, the last dim of V_next is indexed by *current* i_y (becomes i_y_last next period)

    # Working EV: first contract over h'
    # V_next shape: (n_a, n_y, n_h, n_y)
    # P_h_t shape: (n_h, n_h)
    # Contract h: EV_h[a', y', i_h, y_last] = sum_{h'} P_h[i_h, h'] * V[a', y', h', y_last]
    EV_h = jnp.einsum('jk,aykl->ayjl', P_h_t, V_next)  # (n_a, n_y, n_h, n_y)

    # For working: V_next[a', y', h', i_y] — last dim is current i_y
    # EV_working[a', i_y, i_h, i_y_last] = sum_{y'} P_y[i_y, y'] * EV_h[a', y', i_h, i_y]
    # But we need i_y_last to pass through, and the pension dimension uses i_y as last-income.
    # Actually in working case, next period's i_y_last = current i_y.
    # So V_next is accessed as V_next[a', y', h', i_y].
    # EV_h contracted over h' gives us: contracted[a', y', i_h, i_y] = sum_h' P_h[i_h,h'] V[a',y',h',i_y]
    # Then contract over y': EV[a', i_y, i_h] = sum_y' P_y[i_y, y'] * contracted[a', y', i_h, i_y]

    # EV_h has shape (n_a, n_y_next, n_h_current, n_y_last_in_Vnext)
    # For working, n_y_last_in_Vnext = current i_y
    # We want EV_working[a', i_y, i_h, i_y_last_current] but i_y_last_current is irrelevant
    # for EV computation (it only affects budget, not continuation).
    # So EV_working[a', i_y, i_h] = sum_y' P_y[i_y, y'] * EV_h[a', y', i_h, i_y]

    # Use gather for the i_y indexing in last dim of EV_h
    # EV_h shape: (n_a, n_y, n_h, n_y)  — dims are (a', y', h_curr, y_last_of_Vnext)
    # We need: for each i_y, EV_h[a', y', i_h, i_y] then sum over y' with P_y[i_y, y']
    # This is: sum_y' P_y[i_y, y'] * EV_h[a', y', i_h, i_y]
    # = einsum('ij, ajki -> aki') but we need to be careful with dims

    # Let's reshape more carefully.
    # EV_h[a, y_next, h_cur, yl] : (n_a, n_y, n_h, n_y)
    # For working case, yl = current i_y. We want for each current i_y:
    #   EV_w[a, i_y, h_cur] = sum_{y_next} P_y[i_y, y_next] * EV_h[a, y_next, h_cur, i_y]

    # Rearrange: create EV_h_T[a, yl, h_cur, y_next] by transposing last two dims of certain axes
    # Actually EV_h is already (a, y_next, h_cur, yl).
    # We want: for each yl, dot P_y[yl, :] with EV_h[:, :, :, yl] over the y_next axis.
    # EV_working[a, yl, h] = sum_yn P_y[yl, yn] * EV_h[a, yn, h, yl]
    # = einsum('iy, ayhi -> ahi') but with an extra yl dim...

    # Let's just do it explicitly:
    # EV_h[a, yn, h, yl] -> transpose to [yl, a, h, yn]
    EV_h_t = jnp.transpose(EV_h, (3, 0, 2, 1))  # (n_y, n_a, n_h, n_y)  = (yl, a, h, yn)
    # P_y[yl, yn]
    # EV_working[yl, a, h] = sum_yn P_y[yl, yn] * EV_h_t[yl, a, h, yn]
    EV_working_3d = jnp.einsum('ij,iabj->iab', P_y, EV_h_t)  # (n_y, n_a, n_h)
    # = (i_y, a', h_cur)
    # Broadcast to (n_a_next, n_y, n_h, n_y_last): last dim is i_y_last (irrelevant for EV)
    EV_working = jnp.transpose(EV_working_3d, (1, 0, 2))  # (n_a, n_y, n_h)
    EV_working = EV_working[..., None] * jnp.ones((1, 1, 1, n_y))  # broadcast i_y_last

    # Retired: V_next[a', 0, h', 0], contract over h' only
    # V_next_retired[a'] = sum_h' P_h[i_h, h'] * V_next[a', 0, h', 0]
    V_next_ret = V_next[:, 0, :, 0]  # (n_a, n_h)  — retired uses y=0, y_last=0
    EV_retired_2d = jnp.einsum('jk,ak->aj', P_h_t, V_next_ret)  # (n_a, n_h)
    # Broadcast to (n_a, n_y, n_h, n_y)
    EV_retired = EV_retired_2d[:, None, :, None] * jnp.ones((1, n_y, 1, n_y))

    EV = jnp.where(is_retired, EV_retired, EV_working)  # (n_a, n_y, n_h, n_y)

    # 4. Grid search: val_all = u(c) + beta*EV
    #    c_all: (n_a, n_y, n_h, n_y, n_a_next)
    #    EV:    (n_a_next,) — need to index by the n_a_next dim
    #    But EV[i_a_next, ...] corresponds to the *next-period* asset state.
    #    So EV should be broadcast along the last axis (n_a_next dimension of c_all).
    #    EV is (n_a, n_y, n_h, n_y) representing EV for each a_next choice.
    #    Wait — EV is indexed by a_next (first dim). We need:
    #    val_all[i_a, i_y, i_h, i_y_last, i_a_next] = u(c_all[...]) + beta * EV[i_a_next, i_y, i_h, i_y_last]

    # EV needs to be (1, n_y, n_h, n_y, n_a) where the last axis is a_next
    # Currently EV is (n_a_next, n_y, n_h, n_y). Transpose the a_next to last:
    EV_for_search = jnp.transpose(EV, (1, 2, 3, 0))  # (n_y, n_h, n_y, n_a_next)
    EV_for_search = EV_for_search[None, ...]           # (1, n_y, n_h, n_y, n_a_next)

    u_all = utility_jax(c_all, gamma)                  # (n_a, n_y, n_h, n_y, n_a)
    val_all = u_all + beta * EV_for_search             # (n_a, n_y, n_h, n_y, n_a)

    # Mask infeasible (c <= 0)
    val_all = jnp.where(c_all > 0, val_all, -jnp.inf)

    # Argmax over last axis (a_next)
    best_a_idx = jnp.argmax(val_all, axis=-1)          # (n_a, n_y, n_h, n_y)
    best_val = jnp.max(val_all, axis=-1)               # (n_a, n_y, n_h, n_y)
    best_c = jnp.take_along_axis(c_all, best_a_idx[..., None], axis=-1)[..., 0]

    # 5. Fallback for states where no feasible choice exists
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

    model_params = (a_grid, y_grid, h_grid, m_grid, P_y, w_at_retirement,
                    ui_replacement_rate, kappa, beta, gamma)

    # Terminal period
    is_retired_T = (T - 1) >= retirement_age
    V_T, a_pol_T, c_pol_T = _solve_terminal_period_jax(
        a_grid, y_grid, h_grid, m_grid,
        P_y, P_h[T - 1], w_at_retirement,
        r_path[T - 1], w_path[T - 1],
        tau_c_path[T - 1], tau_l_path[T - 1], tau_p_path[T - 1], tau_k_path[T - 1],
        pension_replacement_path[T - 1],
        ui_replacement_rate, kappa, is_retired_T, gamma,
    )

    # Stack period params for t = T-2 ... 0 (reversed)
    # We iterate backwards: the scan processes t = T-2, T-3, ..., 0
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
        (ts >= retirement_age),         # is_retired for each t
    )

    def scan_fn(V_next, period_params_slice):
        (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
         pension_replacement_t, P_h_t, is_retired) = period_params_slice

        V_t, a_pol_t, c_pol_t = solve_period_jax(
            V_next,
            (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
             pension_replacement_t, P_h_t, is_retired),
            model_params,
        )
        return V_t, (V_t, a_pol_t, c_pol_t)

    V_final, (V_scan, a_pol_scan, c_pol_scan) = lax.scan(
        scan_fn, V_T, period_params_stack
    )
    # V_scan shape: (T-1, n_a, n_y, n_h, n_y) — in reversed time order

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
    static_argnames=('T', 'retirement_age'),
)

# Batched solve: vmap over cohorts with shared grids/transitions.
# Per-cohort inputs (in_axes=0): w_at_retirement, r/w/tax/pension paths.
# Shared inputs (in_axes=None): grids, P_y, P_h, scalars.
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
    )),
    static_argnames=('T', 'retirement_age'),
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
                    retirement_age, T, current_age):
    """
    Single time-step for one agent.

    carry: (i_a, i_y, i_h, i_y_last, avg_earnings, n_earnings_years)
    t_data: (t_sim_idx, u_y, u_h)  — t_sim_idx is 0-based simulation step, u_y/u_h are uniform draws
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

    # Pension
    pension_replacement = pension_replacement_path[lifecycle_age]
    pension = jnp.where(is_retired,
                        pension_replacement * w_at_retirement * y_last_val,
                        0.0)

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

    # Health expenditure
    m_val = m_grid[i_h]
    oop_m = (1.0 - kappa) * m_val
    gov_m = kappa * m_val

    # Taxes
    r_t = r_path[lifecycle_age]
    tax_c = tau_c_path[lifecycle_age] * c_pol_val

    # Payroll tax on wages only
    tax_p = jnp.where(is_retired, 0.0, tau_p_path[lifecycle_age] * wage_income)

    # Labor income tax
    tax_l = jnp.where(
        is_retired,
        tau_l_path[lifecycle_age] * pension,
        tau_l_path[lifecycle_age] * (effective_y - tax_p),
    )

    # Capital income tax
    gross_capital = r_t * a_val
    tax_k = tau_k_path[lifecycle_age] * gross_capital

    # Update average earnings (working years only)
    gross_labor_income_for_avg = wage_income  # w * y * h
    new_n_years = jnp.where(is_retired, n_earnings_years, n_earnings_years + 1)
    new_avg_earnings = jnp.where(
        is_retired,
        avg_earnings,
        jnp.where(
            new_n_years > 0,
            (avg_earnings * n_earnings_years + gross_labor_income_for_avg) / new_n_years,
            gross_labor_income_for_avg,
        ),
    )

    # --- State transitions ---
    # Next asset state (keep same dtype as carry input)
    new_i_a = jnp.where(is_last_step, i_a, a_pol_val)

    # Next income state (only if working)
    cum_P_y = jnp.cumsum(P_y[i_y, :])
    new_i_y_draw = jnp.searchsorted(cum_P_y, u_y)
    new_i_y_draw = jnp.clip(new_i_y_draw, 0, P_y.shape[1] - 1)
    # If retired, i_y stays 0 and i_y_last is frozen
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
        a_val,           # 0: assets
        c_pol_val,       # 1: consumption
        y_val,           # 2: income level
        h_val,           # 3: health productivity
        i_h,             # 4: health index
        effective_y,     # 5: effective income
        employed,        # 6: employed flag
        ui,              # 7: UI benefit
        m_val,           # 8: medical expenditure
        oop_m,           # 9: out-of-pocket medical
        gov_m,           # 10: gov medical spending
        tax_c,           # 11: consumption tax
        tax_l,           # 12: labor income tax
        tax_p,           # 13: payroll tax
        tax_k,           # 14: capital income tax
        new_avg_earnings,  # 15: avg earnings
        pension,         # 16: pension
        is_retired,      # 17: retired flag
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
):
    """
    Simulate lifecycle paths for n_sim agents using vmap + lax.scan.

    Returns tuple of 18 arrays, each shape (T_sim, n_sim).
    """
    T_sim = T - current_age

    # Pre-generate random draws: (T_sim, n_sim, 2)
    key1, key2 = jax.random.split(key)
    u_y_all = jax.random.uniform(key1, shape=(T_sim, n_sim))
    u_h_all = jax.random.uniform(key2, shape=(T_sim, n_sim))
    t_indices = jnp.arange(T_sim)

    # Build per-agent scan function using partial
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
    static_argnames=('retirement_age', 'T', 'current_age', 'n_sim'),
)

# Batched simulation: vmap over cohorts with shared grids/transitions.
# Per-cohort inputs (in_axes=0): policies, paths, key, initial states.
# Shared inputs (in_axes=None): grids, P_y, P_h, scalars.
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
        0, 0, 0, 0,             # initial_i_a, initial_i_y, initial_i_h, initial_i_y_last
        0, 0,                    # initial_avg_earnings, initial_n_earnings_years
    )),
    static_argnames=('retirement_age', 'T', 'current_age', 'n_sim'),
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
        self.m_grid = jnp.array(self._np_model.m_grid)
        self.P_y = jnp.array(self._np_model.P_y)
        self.P_h = jnp.array(self._np_model.P_h)
        self.w_at_retirement = float(self._np_model.w_at_retirement)

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

        V, a_policy, c_policy = _solve_lifecycle_jax_jit(
            self.a_grid, self.y_grid, self.h_grid, self.m_grid,
            self.P_y, self.P_h,
            self.w_at_retirement,
            self.r_path, self.w_path,
            self.tau_c_path, self.tau_l_path, self.tau_p_path, self.tau_k_path,
            self.pension_replacement_path,
            self.ui_replacement_rate, self.kappa,
            self.beta, self.gamma,
            self.T, self.retirement_age,
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
            # Stationary distribution
            eigenvalues, eigenvectors = eig(np.asarray(self.P_y).T)
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

        result = _simulate_lifecycle_jax_jit(
            a_policy_jax, c_policy_jax,
            self.a_grid, self.y_grid, self.h_grid, self.m_grid,
            self.P_y, self.P_h,
            self.w_path, self.w_at_retirement,
            self.tau_c_path, self.tau_l_path, self.tau_p_path, self.tau_k_path,
            self.r_path, self.pension_replacement_path,
            self.ui_replacement_rate, self.kappa,
            self.retirement_age, self.T, self.current_age,
            n_sim, subkey,
            initial_i_a, initial_i_y, initial_i_h, initial_i_y_last,
            initial_avg_earnings, initial_n_years,
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
