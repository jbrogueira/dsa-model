"""
JAX-accelerated lifecycle model with perfect foresight.

Port of the computational core from lifecycle_perfect_foresight.py.
Uses vectorized operations and XLA compilation for massive speedup.
The NumPy implementation remains as reference/fallback.
"""

import os
import platform

# On macOS ARM, JAX defaults to the experimental Metal backend, which does not
# support float64 and breaks the lifecycle solver. Force CPU before any jax
# import; setdefault preserves an explicit user override.
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from functools import partial
from scipy.linalg import eig

from lifecycle_perfect_foresight import (LifecycleConfig, LifecycleModelPerfectForesight,
                                         encoded_pension_floor)

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


def solve_labor_hours_jax(c, net_wage, nu, phi, gamma):
    """FOC: l* = (c^{-gamma} * net_wage / nu)^{1/phi}, clamped to [0, 1]."""
    l = (jnp.maximum(c, 1e-10) ** (-gamma) * jnp.maximum(net_wage, 0.0) / nu) ** (1.0 / phi)
    return jnp.clip(l, 0.0, 1.0)  # time endowment is 1


def solve_labor_robust_jax(c_guess, mw, nu, phi, gamma, tau_c_t, n_iters=12):
    """Robust labor-hours solve, consistent with the budget (1+τ_c)·c = resources.

    Intratemporal FOC (working, employed):
        ν·l^φ = c^{-γ} · MW / (1+τ_c),   c(l) = c_guess + MW·(l-1)/(1+τ_c),
    where MW is the marginal after-tax labor income per unit of l (effective wage
    × the after-tax wedge; see callers). The residual
        G(l) = ν·l^φ·(1+τ_c) − c(l)^{-γ}·MW
    is monotonically increasing on the feasible region c(l)>0, so the root is
    unique. Solved by safeguarded Newton–bisection (rtsafe), bracketed to
    [l_lo, l_hi] with l_lo = max(0, 1 − c_guess·(1+τ_c)/MW) (the l where c(l)=0)
    and l_hi found by doubling from 1 until G(l_hi) > 0: hours are not capped at
    the time endowment (the cap at 1 was removed 2026-10-02), so the condition
    holds with equality at every interior solution.
    Newton steps are taken only when they stay inside the bracket, else a
    bisection step — robust against the consumption-floor region that traps a
    plain Newton iteration started from an infeasible guess. Branchless
    (where-selects) for vmap/XLA. Returns l >= 0; 0 when MW≤0 (no productive
    labor, e.g. unemployment).
    """
    onetc = 1.0 + tau_c_t
    mws = jnp.maximum(mw, 1e-12)
    lo = jnp.maximum(0.0, 1.0 - c_guess * onetc / mws)
    hi = jnp.ones_like(lo)
    for _ in range(6):
        c_hi = jnp.maximum(c_guess + mw * (hi - 1.0) / onetc, 1e-12)
        G_hi = nu * hi ** phi * onetc - c_hi ** (-gamma) * mw
        hi = jnp.where(G_hi < 0.0, 2.0 * hi, hi)
    l = 0.5 * (lo + hi)
    for _ in range(n_iters):
        c_l = jnp.maximum(c_guess + mw * (l - 1.0) / onetc, 1e-12)
        G = nu * jnp.maximum(l, 0.0) ** phi * onetc - c_l ** (-gamma) * mw
        Gp = (nu * phi * jnp.maximum(l, 1e-12) ** (phi - 1.0) * onetc
              + gamma * c_l ** (-gamma - 1.0) * mw * (mw / onetc))
        lo = jnp.where(G < 0.0, l, lo)
        hi = jnp.where(G > 0.0, l, hi)
        l_newton = l - G / Gp
        # Projected Newton: clip the step into the bracket [lo, hi] (rather than
        # falling back to bisection). Interior states get full Newton speed;
        # corner states (l*→1 or l*→l_lo) snap to the bound in ~2 steps instead
        # of crawling at the bisection rate. Bisection-midpoint fallback only
        # when the Newton step is non-finite (Gp→0).
        l = jnp.where(jnp.isfinite(l_newton),
                      jnp.clip(l_newton, lo, hi), 0.5 * (lo + hi))
    return jnp.where(mw > 1e-12, jnp.maximum(l, 0.0), 0.0)


# Largest hours solve_labor_robust_jax can return: its upper bracket after six
# doublings from 1. The log-utility solve below caps hours at the same value.
_HOURS_CAP = 64.0
_HOURS_TABLE_SIZE = 2048


def hours_table_log_utility(nu, phi):
    """Nodes (z, l) of the hours rule under log utility, z ascending.

    With γ = 1 and c(l) = c_guess + MW·(l−1)/(1+τ_c), the intratemporal FOC
    ν·l^φ·(1+τ_c) = MW/c(l) reduces to
        ν·l^φ·(z + l) = 1,   z = c_guess·(1+τ_c)/MW − 1,
    so hours depend on the state and on a' only through the scalar z. The
    inverse is closed form, z(l) = 1/(ν·l^φ) − l, strictly decreasing in l; the
    table evaluates it on a log-spaced grid of l from _HOURS_CAP down to 1e-10.
    """
    l = jnp.exp(jnp.linspace(jnp.log(_HOURS_CAP), jnp.log(1e-10), _HOURS_TABLE_SIZE))
    z = 1.0 / (nu * l ** phi) - l
    return z, l


def solve_labor_log_jax(c_guess, mw, nu, phi, tau_c_t, hours_table, n_newton=3):
    """Labor hours under log utility (γ = 1); same root as solve_labor_robust_jax.

    Solves ν·l^φ·(z + l) = 1 (see hours_table_log_utility) by linear
    interpolation in the table followed by n_newton Newton steps. Adjacent
    table nodes are 1.3% apart in l, so the interpolated start is within ~1e-5
    of the root and three steps reach machine precision. Hours are capped at
    _HOURS_CAP, the value solve_labor_robust_jax returns when the root lies
    above its bracket. Returns 0 when MW ≤ 0, as solve_labor_robust_jax does.
    """
    z_tab, l_tab = hours_table
    onetc = 1.0 + tau_c_t
    mws = jnp.maximum(mw, 1e-12)
    z = c_guess * onetc / mws - 1.0
    i = jnp.clip(jnp.searchsorted(z_tab, z) - 1, 0, _HOURS_TABLE_SIZE - 2)
    z_lo, z_hi = z_tab[i], z_tab[i + 1]
    weight = jnp.clip((z - z_lo) / (z_hi - z_lo), 0.0, 1.0)
    l = l_tab[i] + weight * (l_tab[i + 1] - l_tab[i])
    for _ in range(n_newton):
        p = l ** (phi - 1.0)
        G = nu * p * l * (z + l) - 1.0
        Gp = nu * p * (phi * (z + l) + l)
        l = l - G / Gp
    l = jnp.where(z > z_tab[0], l, _HOURS_CAP)
    return jnp.where(mw > 1e-12, l, 0.0)


def labor_disutility_jax(l, nu, phi):
    """nu * l^(1+phi) / (1+phi)"""
    return nu * l ** (1 + phi) / (1 + phi)


def _pension_floor_jax(floor, replacement):
    """Floor in a period with the given replacement rate: `floor` itself when
    non-negative, -floor times the rate when negative (the indexed floor, see
    lifecycle_perfect_foresight.encoded_pension_floor)."""
    return jnp.where(floor < 0.0, -floor * replacement, floor)


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
    labor_hours=1.0,
    bequest_lumpsum=0.0,
    kappa_wage_t=1.0,
    kappa_wage_ret=None,
    pension_avg_weight=1.0,
    mean_kappa_working=1.0,
    mean_y_employed=1.0,
    alpha_mult=1.0,
):
    """
    Vectorised budget for ALL (n_a, n_y, n_h, n_y_last) states.

    Returns
    -------
    budget : array, shape (n_a, n_y, n_h, n_y)
    """
    # Broadcast grids: a(n_a,1,1,1), y(1,n_y,1,1), h(1,1,n_h,1), y_last(1,1,1,n_y)
    a = a_grid[:, None, None, None]           # (n_a,1,1,1)
    y = y_grid[None, :, None, None]           # (1,n_y,1,1)
    h = h_grid[None, None, :, None]           # (1,1,n_h,1)
    y_last = y_grid[None, None, None, :]      # (1,1,1,n_y)
    m = m_grid[None, None, :, None]           # (1,1,n_h,1)  — already age-indexed slice

    # --- Retired branch ---
    # Career-average pension approximation, scaled by permanent FE multiplier
    lam = pension_avg_weight
    # The pension is fixed at retirement, so its own-income component is valued at
    # the wage multiplier of the LAST WORKING age, kappa(retirement_age - 1), not at
    # the retiree's current age, where kappa is 1.0 and the base is understated. The
    # NumPy solve (lifecycle_perfect_foresight.py) and this module's own simulate
    # step both use kappa(retirement_age - 1). Until 2026-10-01 this kernel received
    # kappa_wage_t, the current age's value, while its comment claimed otherwise --
    # so the JAX solve priced a poorer retirement than the one it then paid.
    kappa_ret = kappa_wage_t if kappa_wage_ret is None else kappa_wage_ret
    pension_base = lam * kappa_ret * y_last + (1 - lam) * mean_kappa_working * mean_y_employed
    pension = pension_replacement_t * w_at_retirement * pension_base * alpha_mult  # (1,1,1,n_y)
    # Feature #11: minimum pension floor (flat amount, not scaled by alpha)
    pension = jnp.maximum(pension, _pension_floor_jax(pension_min_floor, pension_replacement_t))
    # Feature #14: progressive or flat tax
    retired_income_tax = jnp.where(
        tax_progressive,
        _hsv_tax(pension, tax_kappa_hsv, tax_eta),
        tau_l_t * pension,
    )
    retired_after_tax_labor = pension - retired_income_tax            # (1,1,1,n_y)

    # --- Working branch ---
    ui_benefit = ui_replacement_rate * w_t * kappa_wage_t * y_last * alpha_mult    # (1,1,1,n_y)
    is_unemployed = (y == 0.0)                                        # (1,n_y,1,1)

    gross_wage_income = w_t * kappa_wage_t * y * h * labor_hours * alpha_mult       # (1,n_y,n_h,1)
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

    # After-tax bequest lump-sum transfer at age 0 (scalar, non-zero only when caller sets it)
    budget = budget + bequest_lumpsum

    # Feature #4: schooling child costs
    net_child_cost = (1.0 - education_subsidy_rate) * child_cost_t
    budget = jnp.where(in_schooling, budget - net_child_cost, budget)

    # Feature #15: means-tested transfers (consumption floor). Gated on the floor
    # being positive, as the NumPy solve is: at transfer_floor == 0 the
    # unconditional form reduces to max(0, -budget), which hands free resources
    # to any state with a negative budget and silently made the two backends
    # disagree wherever that happened. Infeasibility belongs to the consumption
    # clamp, not to a transfer nobody legislated.
    transfer = jnp.where(transfer_floor > 0.0,
                         jnp.maximum(0.0, transfer_floor - budget), 0.0)
    budget = budget + transfer

    return budget


# ---------------------------------------------------------------------------
# Vectorised single-period solve
# ---------------------------------------------------------------------------

def solve_period_jax(V_next, period_params, model_params, alpha_mult=1.0,
                     hours_table=None):
    """
    Solve a single non-terminal period via grid search.

    Parameters
    ----------
    V_next : array (n_a, n_y, n_h, n_y)
        Continuation value from t+1.
    period_params : dict-like tuple
        (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t, pension_replacement_t,
         P_h_t, P_y_t, is_retired, survival_t, child_cost_t, in_schooling_t,
         bequest_lumpsum_t, kappa_wage_t, kappa_wage_ret)
    model_params : dict-like tuple
        (a_grid, y_grid, h_grid, m_grid, P_y, w_at_retirement,
         ui_replacement_rate, kappa, beta, gamma,
         pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
         transfer_floor, education_subsidy_rate, P_y_age_health,
         labor_supply, nu, phi, trend_growth,
         pension_avg_weight, mean_kappa_working, mean_y_employed)
    alpha_mult : scalar float, default 1.0
        Phase 8 permanent productivity FE multiplier (= exp(alpha_grid[k]) for
        the alpha being solved). Multiplies wage income, UI benefit, and
        pension wage component.
    hours_table : (z, l) arrays from hours_table_log_utility(nu, phi), or None
        to build them here. Used only when gamma == 1 and labor_supply is on.

    Returns
    -------
    (V_t, a_pol_t, c_pol_t, l_pol_t) each shape (n_a, n_y, n_h, n_y)
    """
    (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
     pension_replacement_t, P_h_t, P_y_t, is_retired,
     survival_t, child_cost_t, in_schooling_t,
     bequest_lumpsum_t, kappa_wage_t, kappa_wage_ret) = period_params

    (a_grid, y_grid, h_grid, m_grid, P_y, w_at_retirement,
     ui_replacement_rate, kappa, beta, gamma,
     pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
     transfer_floor, education_subsidy_rate, P_y_age_health,
     labor_supply, nu, phi, trend_growth,
     pension_avg_weight, mean_kappa_working, mean_y_employed) = model_params

    n_a = a_grid.shape[0]
    n_y = y_grid.shape[0]
    n_h = h_grid.shape[0]

    # 1. Budget for all states with l=1: (n_a, n_y, n_h, n_y)
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
        bequest_lumpsum=bequest_lumpsum_t,
        kappa_wage_t=kappa_wage_t,
        kappa_wage_ret=kappa_wage_ret,
        pension_avg_weight=pension_avg_weight,
        mean_kappa_working=mean_kappa_working,
        mean_y_employed=mean_y_employed,
        alpha_mult=alpha_mult,
    )

    # 2. Consumption candidates: (n_a, n_y, n_h, n_y, n_a_next).
    # In detrended units a unit of next-period assets costs (1+g) today.
    a_next = (1.0 + trend_growth) * a_grid[None, None, None, None, :]
    c_all = (budget[..., None] - a_next) / (1.0 + tau_c_t)

    # 2b. Labor supply FOC (when labor_supply=True).
    # Marginal after-tax labor income per unit of l, consistent with the budget:
    # income tax falls on income net of payroll, so the wedge is multiplicative
    #   flat:        MW = effective_wage·(1−τ_p)(1−τ_l)
    #   progressive: MW ≈ effective_wage·(1−τ_p)   (HSV marginal handled separately)
    #
    # Hours are solved on the y_last = 0 slice, (n_a, n_y, n_h, n_a_next), and
    # broadcast over y_last: the budget of an employed working-age state does
    # not depend on y_last (it enters only UI and the pension), and the hours
    # of the unemployed and the retired are not used (set to 1 below).
    if labor_supply:
        y_4d = y_grid[None, :, None, None]         # (1, n_y, 1, 1)
        h_4d = h_grid[None, None, :, None]         # (1, 1, n_h, 1)
        effective_wage = w_t * kappa_wage_t * y_4d * h_4d * alpha_mult  # (1, n_y, n_h, 1)
        wedge = jnp.where(tax_progressive, 1.0 - tau_p_t, (1.0 - tau_p_t) * (1.0 - tau_l_t))
        mw_4d = effective_wage * wedge             # (1, n_y, n_h, 1)

        if hours_table is None:
            hours_table = hours_table_log_utility(nu, phi)
        is_unemployed_4d = (y_4d == 0.0)           # (1, n_y, 1, 1)
        l_star = lax.cond(
            gamma == 1.0,
            lambda c: solve_labor_log_jax(c, mw_4d, nu, phi, tau_c_t, hours_table),
            lambda c: solve_labor_robust_jax(c, mw_4d, nu, phi, gamma, tau_c_t),
            c_all[:, :, :, 0, :],
        )
        l_star = jnp.where(is_unemployed_4d | is_retired, 1.0, l_star)

        # Budget adjustment for l≠1 uses the same marginal after-tax wage MW.
        delta_budget = (mw_4d * (l_star - 1.0))[:, :, :, None, :]
        c_all = (budget[..., None] + delta_budget - a_next) / (1.0 + tau_c_t)
        l_all = jnp.broadcast_to(l_star[:, :, :, None, :], c_all.shape)

        # Labor disutility — only working-age EMPLOYED agents work (retired and
        # unemployed bear no labor disutility; l_star is held at 1.0 for them
        # only to zero delta_budget, not because they work).
        work_employed = (~is_retired) & (y_4d > 0.0)
        v_labor = jnp.where(work_employed,
                            labor_disutility_jax(l_star, nu, phi), 0.0)[:, :, :, None, :]
    else:
        l_all = jnp.ones_like(c_all)
        v_labor = 0.0

    # 3. Expected continuation value
    EV_h = jnp.einsum('jk,aykl->ayjl', P_h_t, V_next)
    EV_h_t = jnp.transpose(EV_h, (3, 0, 2, 1))

    EV_working_3d = jnp.where(
        P_y_age_health,
        jnp.einsum('bij,iabj->iab', P_y_t, EV_h_t),
        jnp.einsum('ij,iabj->iab', P_y, EV_h_t),
    )

    EV_working = jnp.transpose(EV_working_3d, (1, 0, 2))
    EV_working = jnp.broadcast_to(EV_working[..., None], (n_a, n_y, n_h, n_y))

    # Retired continuation: the pension depends on y_last, frozen at
    # retirement, so V_next is read at each state's own y_last (last axis).
    # Until 2026-10-02 it was read at y_last index 0 for every retiree.
    V_next_ret = V_next[:, 0, :, :]                                   # (n_a, n_h_next, n_y_last)
    EV_retired_3d = jnp.einsum('jk,akl->ajl', P_h_t, V_next_ret)       # (n_a, n_h, n_y_last)
    EV_retired = jnp.broadcast_to(EV_retired_3d[:, None, :, :], (n_a, n_y, n_h, n_y))

    EV = jnp.where(is_retired, EV_retired, EV_working)

    survival_broadcast = survival_t[None, None, :, None]
    EV = EV * survival_broadcast

    # 4. Grid search
    EV_for_search = jnp.transpose(EV, (1, 2, 3, 0))

    # utility_jax on one branch: only the log or the CRRA form is evaluated.
    u_all = lax.cond(
        gamma == 1.0,
        jnp.log,
        lambda c: c ** (1.0 - gamma) / (1.0 - gamma),
        jnp.maximum(c_all, 1e-10),
    ) - v_labor
    val_all = u_all + beta * EV_for_search

    val_all = jnp.where(c_all > 0, val_all, -jnp.inf)

    best_a_idx = jnp.argmax(val_all, axis=-1)
    best_val = jnp.max(val_all, axis=-1)
    best_c = jnp.take_along_axis(c_all, best_a_idx[..., None], axis=-1)[..., 0]
    best_l = jnp.take_along_axis(l_all, best_a_idx[..., None], axis=-1)[..., 0]

    # 5. Fallback
    c_fallback = jnp.maximum(budget / (1.0 + tau_c_t), 1e-10)
    u_fallback = utility_jax(c_fallback, gamma)

    V_t = jnp.where(jnp.isfinite(best_val), best_val, u_fallback)
    a_pol_t = jnp.where(jnp.isfinite(best_val), best_a_idx, 0).astype(jnp.int32)
    c_pol_t = jnp.where(jnp.isfinite(best_val), best_c, c_fallback)
    l_pol_t = jnp.where(jnp.isfinite(best_val), best_l, 1.0)

    return V_t, a_pol_t, c_pol_t, l_pol_t


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
    labor_supply=False,
    nu=1.0,
    phi=2.0,
    kappa_wage_T=1.0,
    kappa_wage_ret=None,
    pension_avg_weight=1.0,
    mean_kappa_working=1.0,
    mean_y_employed=1.0,
    alpha_mult=1.0,
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
        kappa_wage_t=kappa_wage_T,
        kappa_wage_ret=kappa_wage_ret,
        pension_avg_weight=pension_avg_weight,
        mean_kappa_working=mean_kappa_working,
        mean_y_employed=mean_y_employed,
        alpha_mult=alpha_mult,
    )
    c = jnp.maximum(budget / (1.0 + tau_c_T), 1e-10)

    # Labor supply FOC at terminal period (a'=0). Same corrected marginal
    # after-tax wage MW and (1+τ_c) wedge as the non-terminal solve.
    y_4d = y_grid[None, :, None, None]       # (1, n_y, 1, 1)
    h_4d = h_grid[None, None, :, None]       # (1, 1, n_h, 1)
    effective_wage = w_T * kappa_wage_T * y_4d * h_4d * alpha_mult
    wedge = jnp.where(tax_progressive, 1.0 - tau_p_T, (1.0 - tau_p_T) * (1.0 - tau_l_T))
    mw_4d = effective_wage * wedge

    is_unemployed_4d = (y_4d == 0.0)
    l_star = solve_labor_robust_jax(c, mw_4d, nu, phi, gamma, tau_c_T)
    l_star = jnp.where(is_unemployed_4d | is_retired_T, 1.0, l_star)
    l_pol = jnp.where(labor_supply, l_star, 1.0)

    # Final budget and consumption with converged labor hours
    delta_budget = jnp.where(labor_supply, mw_4d * (l_pol - 1.0), 0.0)
    c = jnp.maximum((budget + delta_budget) / (1.0 + tau_c_T), 1e-10)

    work_employed = (~is_retired_T) & (y_4d > 0.0)
    v_labor = jnp.where(labor_supply & work_employed,
                        labor_disutility_jax(l_pol, nu, phi), 0.0)
    V = utility_jax(c, gamma) - v_labor
    a_pol = jnp.zeros_like(V, dtype=jnp.int32)
    return V, a_pol, c, l_pol


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
    labor_supply=False,
    nu=1.0,
    phi=2.0,
    trend_growth=0.0,
    bequest_lumpsum=0.0,
    wage_age_profile=None,
    pension_avg_weight=1.0,
    mean_kappa_working=1.0,
    mean_y_employed=1.0,
    alpha_mult=1.0,
):
    """
    Full backward induction using jax.lax.scan.

    Returns
    -------
    V : (T, n_a, n_y, n_h, n_y)
    a_policy : (T, n_a, n_y, n_h, n_y), int32
    c_policy : (T, n_a, n_y, n_h, n_y)
    l_policy : (T, n_a, n_y, n_h, n_y)
    """
    n_h = h_grid.shape[0]

    # Default wage_age_profile to ones
    if wage_age_profile is None:
        wage_age_profile = jnp.ones(T)

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
    if P_y_age_health:
        P_y_scan = P_y_by_age_health
    else:
        P_y_scan = jnp.tile(P_y[None, None, :, :], (T, n_h, 1, 1))

    # m_grid can be (T, n_h) or (n_h,). Ensure (T, n_h) for per-period indexing.
    if m_grid.ndim == 1:
        m_grid_path = jnp.tile(m_grid[None, :], (T, 1))
    else:
        m_grid_path = m_grid

    m_grid_base = m_grid_path[0]

    model_params = (a_grid, y_grid, h_grid, m_grid_base, P_y, w_at_retirement,
                    ui_replacement_rate, kappa, beta, gamma,
                    pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
                    transfer_floor, education_subsidy_rate, P_y_age_health,
                    labor_supply, nu, phi, trend_growth,
                    pension_avg_weight, mean_kappa_working, mean_y_employed)

    # Terminal period
    is_retired_T = (T - 1) >= retirement_age

    V_T, a_pol_T, c_pol_T, l_pol_T = _solve_terminal_period_jax(
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
        labor_supply=labor_supply,
        nu=nu,
        phi=phi,
        kappa_wage_T=wage_age_profile[T - 1],
        kappa_wage_ret=wage_age_profile[retirement_age - 1],
        pension_avg_weight=pension_avg_weight,
        mean_kappa_working=mean_kappa_working,
        mean_y_employed=mean_y_employed,
        alpha_mult=alpha_mult,
    )

    # Stack period params for t = T-2 ... 0 (reversed)
    ts = jnp.arange(T - 2, -1, -1)

    # Bequest lump-sum: non-zero only at age 0 (the last step of the backward scan)
    bequest_at_age = jnp.where(ts == 0, bequest_lumpsum, 0.0)

    period_params_stack = (
        r_path[ts],
        w_path[ts],
        tau_c_path[ts],
        tau_l_path[ts],
        tau_p_path[ts],
        tau_k_path[ts],
        pension_replacement_path[ts],
        P_h[ts],
        P_y_scan[ts],
        (ts >= retirement_age),
        survival_arr[ts],
        child_costs[ts],
        (ts < schooling_years),
        m_grid_path[ts],
        bequest_at_age,
        wage_age_profile[ts],
        # kappa at the last working age: constant over the lifecycle, stacked so
        # the scan carries it alongside the per-period values.
        jnp.full(ts.shape, wage_age_profile[retirement_age - 1]),
    )

    # The hours table depends on (nu, phi) only: built once, used at every age.
    hours_table = hours_table_log_utility(nu, phi) if labor_supply else None

    def scan_fn(V_next, period_params_slice):
        (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
         pension_replacement_t, P_h_t, P_y_t, is_retired,
         survival_t, child_cost_t, in_schooling_t, m_grid_t,
         bequest_t, kappa_wage_t, kappa_wage_ret) = period_params_slice

        model_params_t = model_params[:3] + (m_grid_t,) + model_params[4:]

        V_t, a_pol_t, c_pol_t, l_pol_t = solve_period_jax(
            V_next,
            (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t,
             pension_replacement_t, P_h_t, P_y_t, is_retired,
             survival_t, child_cost_t, in_schooling_t, bequest_t, kappa_wage_t,
             kappa_wage_ret),
            model_params_t,
            alpha_mult=alpha_mult,
            hours_table=hours_table,
        )
        return V_t, (V_t, a_pol_t, c_pol_t, l_pol_t)

    V_final, (V_scan, a_pol_scan, c_pol_scan, l_pol_scan) = lax.scan(
        scan_fn, V_T, period_params_stack
    )

    # Reverse to natural order and append terminal period
    V_scan = V_scan[::-1]
    a_pol_scan = a_pol_scan[::-1]
    c_pol_scan = c_pol_scan[::-1]
    l_pol_scan = l_pol_scan[::-1]

    V = jnp.concatenate([V_scan, V_T[None]], axis=0)
    a_policy = jnp.concatenate([a_pol_scan, a_pol_T[None]], axis=0)
    c_policy = jnp.concatenate([c_pol_scan, c_pol_T[None]], axis=0)
    l_policy = jnp.concatenate([l_pol_scan, l_pol_T[None]], axis=0)

    return V, a_policy, c_policy, l_policy


# JIT-compile with static args for shapes and scalar params
_solve_lifecycle_jax_jit = jax.jit(
    solve_lifecycle_jax,
    static_argnames=('T', 'retirement_age', 'tax_progressive', 'schooling_years',
                     'labor_supply'),
)

# Batched solve: vmap over cohorts with shared grids/transitions.
# Per-cohort inputs (in_axes=0): w_at_retirement, r/w/tax/pension paths.
# Shared inputs (in_axes=None): grids, P_y, P_h, scalars + new feature params.
_SOLVE_IN_AXES = (
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
        0, None,                 # survival_probs (per-cohort), P_y_by_age_health
        None, None, None,        # labor_supply, nu, phi
        None,                    # trend_growth (shared scalar)
        0,                       # bequest_lumpsum (per-cohort scalar)
        None,                    # wage_age_profile (shared)
        None, None, None,        # pension_avg_weight, mean_kappa_working, mean_y_employed
        None,                    # alpha_mult (shared across cohorts within one solve sweep)
)
_solve_lifecycle_jax_batched = jax.jit(
    jax.vmap(solve_lifecycle_jax, in_axes=_SOLVE_IN_AXES),
    static_argnames=('T', 'retirement_age', 'tax_progressive', 'schooling_years',
                     'labor_supply'),
)


# ---------------------------------------------------------------------------
# Phase 2: JAX simulation
# ---------------------------------------------------------------------------

def _state_outcomes_jax(i_a, i_y, i_h, i_y_last, lifecycle_age,
                        a_policy, c_policy, l_policy,
                        a_grid, y_grid, h_grid, m_grid,
                        w_path, w_at_retirement,
                        tau_c_path, tau_l_path, tau_p_path, tau_k_path,
                        r_path, pension_replacement_path,
                        ui_replacement_rate, kappa,
                        retirement_age, current_age,
                        pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
                        wage_age_profile,
                        pension_avg_weight, mean_kappa_working, mean_y_employed,
                        alpha_idx, alpha_mult,
                        transfer_floor, bequest_lumpsum):
    """Period outcomes of a living household in state (i_a, i_y, i_h, i_y_last)
    with fixed effect alpha_idx at lifecycle_age.

    Everything here is a function of the state and the age. The simulation
    evaluates it at each agent's drawn state (_agent_step_jax); the exact
    aggregation evaluates it at every state of the grid (exact_age_means_jax).
    """
    is_retired = lifecycle_age >= retirement_age

    # Look up policy on the agent's alpha slice
    a_pol_val = a_policy[alpha_idx, lifecycle_age, i_a, i_y, i_h, i_y_last]
    c_pol_val = c_policy[alpha_idx, lifecycle_age, i_a, i_y, i_h, i_y_last]
    l_pol_val = l_policy[alpha_idx, lifecycle_age, i_a, i_y, i_h, i_y_last]

    # Current state values
    a_val = a_grid[i_a]
    y_val = y_grid[i_y]
    h_val = h_grid[i_h]
    y_last_val = y_grid[i_y_last]

    # Wage age profile
    kappa_wage_t = wage_age_profile[lifecycle_age]

    # Pension with career-average approximation, scaled by permanent FE multiplier
    pension_replacement = pension_replacement_path[lifecycle_age]
    kappa_wage_ret = wage_age_profile[retirement_age - 1]
    pension_base = (pension_avg_weight * kappa_wage_ret * y_last_val
                    + (1 - pension_avg_weight) * mean_kappa_working * mean_y_employed)
    pension_raw = pension_replacement * w_at_retirement * pension_base * alpha_mult
    pension_with_floor = jnp.maximum(
        pension_raw, _pension_floor_jax(pension_min_floor, pension_replacement))
    pension = jnp.where(is_retired, pension_with_floor, 0.0)

    # UI (scales with permanent FE)
    ui = jnp.where(
        (~is_retired) & (i_y == 0),
        ui_replacement_rate * w_path[lifecycle_age] * kappa_wage_t * y_last_val * alpha_mult,
        0.0,
    )

    # Employment
    employed = (~is_retired) & (i_y > 0)

    # Effective income (scales with permanent FE)
    wage_income = w_path[lifecycle_age] * kappa_wage_t * y_val * h_val * l_pol_val * alpha_mult
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

    # Means-tested transfer, as compute_budget_jax grants it in the solve:
    # resources at one hour of work (before the hours adjustment), plus the
    # bequest receipt at entry, against the floor. Child costs are not
    # replicated here (schooling is off whenever a floor is on; the wrapper
    # refuses the combination).
    wage_l1 = w_path[lifecycle_age] * kappa_wage_t * y_val * h_val * alpha_mult
    gross_l1 = jnp.where(is_retired, 0.0, wage_l1 + ui)
    payroll_l1 = jnp.where(is_retired, 0.0, tau_p_path[lifecycle_age] * wage_l1)
    taxable_l1 = gross_l1 - payroll_l1
    inc_tax_l1 = jnp.where(tax_progressive,
                           _hsv_tax(taxable_l1, tax_kappa_hsv, tax_eta),
                           tau_l_path[lifecycle_age] * taxable_l1)
    after_tax_labor_l1 = jnp.where(is_retired, pension - tax_l,
                                   gross_l1 - payroll_l1 - inc_tax_l1)
    budget_pre = (a_val + gross_capital - tax_k + after_tax_labor_l1 - oop_m
                  + jnp.where(lifecycle_age == current_age, bequest_lumpsum, 0.0))
    transfer = jnp.where(transfer_floor > 0.0,
                         jnp.maximum(0.0, transfer_floor - budget_pre), 0.0)

    return dict(
        a_pol_val=a_pol_val, c_pol_val=c_pol_val, l_pol_val=l_pol_val,
        a_val=a_val, y_val=y_val, h_val=h_val,
        is_retired=is_retired, employed=employed,
        pension=pension, ui=ui, wage_income=wage_income, effective_y=effective_y,
        m_val=m_val, oop_m=oop_m, gov_m=gov_m,
        tax_c=tax_c, tax_l=tax_l, tax_p=tax_p, tax_k=tax_k,
        transfer=transfer,
    )



def _agent_step_jax(carry, t_data, a_policy, c_policy, l_policy,
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
                    P_y_4d=None,
                    survival_probs=None,
                    wage_age_profile=None,
                    pension_avg_weight=1.0,
                    mean_kappa_working=1.0,
                    mean_y_employed=1.0,
                    alpha_idx=0,
                    alpha_mult=1.0,
                    trend_growth=0.0,
                    transfer_floor=0.0,
                    bequest_lumpsum=0.0):
    """
    Single time-step for one agent.

    carry: (i_a, i_y, i_h, i_y_last, avg_earnings, n_earnings_years, alive)
    t_data: (t_sim_idx, u_y, u_h, u_alive)

    Phase 8: a_policy/c_policy/l_policy are 6-D, indexed
    [n_alpha, T, n_a, n_y, n_h, n_y]. The agent's permanent FE is captured by the
    closed-over `alpha_idx` (slice into the leading axis) and `alpha_mult`
    (= exp(alpha_grid[alpha_idx]), multiplies wage, UI, and pension wage component).
    """
    i_a, i_y, i_h, i_y_last, avg_earnings, n_earnings_years, alive = carry
    t_sim_idx, u_y, u_h, u_alive = t_data

    lifecycle_age = current_age + t_sim_idx
    is_last_step = (t_sim_idx == (T - current_age - 1))

    s = _state_outcomes_jax(
        i_a, i_y, i_h, i_y_last, lifecycle_age,
        a_policy, c_policy, l_policy,
        a_grid, y_grid, h_grid, m_grid,
        w_path, w_at_retirement,
        tau_c_path, tau_l_path, tau_p_path, tau_k_path,
        r_path, pension_replacement_path,
        ui_replacement_rate, kappa,
        retirement_age, current_age,
        pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
        wage_age_profile,
        pension_avg_weight, mean_kappa_working, mean_y_employed,
        alpha_idx, alpha_mult,
        transfer_floor, bequest_lumpsum,
    )
    a_pol_val, c_pol_val, l_pol_val = s['a_pol_val'], s['c_pol_val'], s['l_pol_val']
    a_val, y_val, h_val = s['a_val'], s['y_val'], s['h_val']
    is_retired, employed = s['is_retired'], s['employed']
    pension, ui = s['pension'], s['ui']
    wage_income, effective_y = s['wage_income'], s['effective_y']
    m_val, oop_m, gov_m = s['m_val'], s['oop_m'], s['gov_m']
    tax_c, tax_l, tax_p, tax_k = s['tax_c'], s['tax_l'], s['tax_p'], s['tax_k']
    transfer = s['transfer']

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

    # --- Mortality ---
    # Survival draw uses current-period state (age t, h_t), not next-period.
    # The bequest is the wealth the household carried out of the period,
    # (1+g) a', in current detrended units (zero at the terminal age, where
    # a' = 0).
    surv_t = survival_probs[lifecycle_age, i_h]
    dies = alive & (u_alive > surv_t)
    bequest_this_period = jnp.where(dies, (1.0 + trend_growth) * a_grid[a_pol_val], 0.0)
    new_alive = alive & ~dies

    new_carry = (new_i_a.astype(jnp.int32), new_i_y.astype(jnp.int32),
                 new_i_h.astype(jnp.int32), new_i_y_last.astype(jnp.int32),
                 new_avg_earnings, new_n_years,
                 new_alive)

    step_out = (
        jnp.where(alive, a_val, 0.0),
        jnp.where(alive, c_pol_val, 0.0),
        jnp.where(alive, y_val, 0.0),
        jnp.where(alive, h_val, 0.0),
        jnp.where(alive, i_h, jnp.int32(0)),
        jnp.where(alive, effective_y, 0.0),
        jnp.where(alive, employed, False),
        jnp.where(alive, ui, 0.0),
        jnp.where(alive, m_val, 0.0),
        jnp.where(alive, oop_m, 0.0),
        jnp.where(alive, gov_m, 0.0),
        jnp.where(alive, tax_c, 0.0),
        jnp.where(alive, tax_l, 0.0),
        jnp.where(alive, tax_p, 0.0),
        jnp.where(alive, tax_k, 0.0),
        jnp.where(alive, new_avg_earnings, 0.0),
        jnp.where(alive, pension, 0.0),
        jnp.where(alive, is_retired, False),
        jnp.where(alive & employed, l_pol_val, 0.0),  # retired and unemployed supply l=0
        alive,
        bequest_this_period,
        jnp.where(alive, transfer, 0.0),
    )

    return new_carry, step_out


def simulate_lifecycle_jax(
    a_policy, c_policy, l_policy,
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
    survival_probs=None,
    wage_age_profile=None,
    pension_avg_weight=1.0,
    mean_kappa_working=1.0,
    mean_y_employed=1.0,
    alpha_idx_sim=None,
    alpha_mult_sim=None,
    trend_growth=0.0,
    transfer_floor=0.0,
    bequest_lumpsum=0.0,
):
    """
    Simulate lifecycle paths for n_sim agents using vmap + lax.scan.

    Phase 8: a_policy/c_policy/l_policy are 6-D, indexed
    [n_alpha, T, n_a, n_y, n_h, n_y]. alpha_idx_sim (n_sim,) gives each
    agent's permanent fixed-effect grid index; alpha_mult_sim (n_sim,) is
    exp(alpha_grid[alpha_idx_sim]). When n_alpha=1 the leading axis of the
    policies is a singleton, alpha_idx_sim is all zeros, and alpha_mult_sim
    is all ones — recovering pre-Phase-8 behavior exactly.

    Returns tuple of 23 arrays, each shape (T_sim, n_sim): the 21 panel
    arrays, alpha_idx_sim broadcast to the panel shape, and transfer_sim.
    """
    T_sim = T - current_age

    # Pre-generate random draws
    key1, key2, key3 = jax.random.split(key, 3)
    u_y_all = jax.random.uniform(key1, shape=(T_sim, n_sim))
    u_h_all = jax.random.uniform(key2, shape=(T_sim, n_sim))
    u_alive_all = jax.random.uniform(key3, shape=(T_sim, n_sim))
    t_indices = jnp.arange(T_sim)

    # Default wage_age_profile to ones
    if wage_age_profile is None:
        wage_age_profile = jnp.ones(T)

    # Dummy P_y_4d if not provided (for JAX tracing)
    if P_y_4d is None:
        n_y = P_y.shape[0]
        n_h = P_h.shape[1]
        P_y_4d = jnp.zeros((T, n_h, n_y, n_y))

    # Survival probs: always an array (ones = no mortality)
    n_h = P_h.shape[1]
    if survival_probs is None:
        survival_probs_arr = jnp.ones((T, n_h))
    else:
        survival_probs_arr = jnp.asarray(survival_probs)

    # Phase 8: per-agent FE arrays. Default to all-zero index / unit multiplier
    # (n_alpha=1 case), which makes the 6-D policy lookup degenerate to the
    # original 5-D lookup at index 0.
    if alpha_idx_sim is None:
        alpha_idx_sim = jnp.zeros(n_sim, dtype=jnp.int32)
    if alpha_mult_sim is None:
        alpha_mult_sim = jnp.ones(n_sim)

    initial_alive = jnp.ones(n_sim, dtype=jnp.bool_)

    def simulate_one(init_state, alpha_idx_self, alpha_mult_self,
                     u_y_seq, u_h_seq, u_alive_seq):
        """Scan over T_sim steps for one agent. alpha_idx/alpha_mult are
        per-agent constants captured into the step closure."""
        step_fn = partial(
            _agent_step_jax,
            a_policy=a_policy, c_policy=c_policy, l_policy=l_policy,
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
            survival_probs=survival_probs_arr,
            wage_age_profile=wage_age_profile,
            pension_avg_weight=pension_avg_weight,
            mean_kappa_working=mean_kappa_working,
            mean_y_employed=mean_y_employed,
            alpha_idx=alpha_idx_self,
            alpha_mult=alpha_mult_self,
            trend_growth=trend_growth,
            transfer_floor=transfer_floor,
            bequest_lumpsum=bequest_lumpsum,
        )
        xs = (t_indices, u_y_seq, u_h_seq, u_alive_seq)
        _, outputs = lax.scan(step_fn, init_state, xs)
        return outputs

    # vmap across n_sim agents
    # init_states: tuple of (n_sim,) arrays — axis 0
    # alpha_idx_sim, alpha_mult_sim: (n_sim,) — axis 0
    # u_y_all, u_h_all, u_alive_all: (T_sim, n_sim) — axis 1
    init_states = (initial_i_a, initial_i_y, initial_i_h, initial_i_y_last,
                   initial_avg_earnings, initial_n_earnings_years, initial_alive)

    all_outputs = jax.vmap(
        simulate_one,
        in_axes=(0, 0, 0, 1, 1, 1),
    )(init_states, alpha_idx_sim, alpha_mult_sim, u_y_all, u_h_all, u_alive_all)

    # all_outputs is a tuple of 22 arrays, each (n_sim, T_sim) from vmap:
    # the 21 panel arrays and transfer_sim. Transpose to (T_sim, n_sim).
    outs = tuple(out.T for out in all_outputs)

    # Phase 8: alpha_idx_panel broadcast to (T_sim, n_sim) sits at index 21,
    # transfer_sim at 22, matching the NumPy backend's 23-tuple.
    alpha_idx_panel = jnp.broadcast_to(alpha_idx_sim[None, :], (T_sim, n_sim)).astype(jnp.int32)
    return outs[:21] + (alpha_idx_panel,) + (outs[21],)


_simulate_lifecycle_jax_jit = jax.jit(
    simulate_lifecycle_jax,
    static_argnames=('retirement_age', 'T', 'current_age', 'n_sim',
                     'tax_progressive', 'P_y_age_health'),
)

# Batched simulation: vmap over cohorts with shared grids/transitions.
_SIMULATE_IN_AXES = (
        0, 0, 0,                 # a_policy, c_policy, l_policy
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
        0,                       # survival_probs (per-cohort)
        None,                    # wage_age_profile (shared)
        None, None, None,        # pension_avg_weight, mean_kappa_working, mean_y_employed
        0, 0,                    # alpha_idx_sim, alpha_mult_sim (per-cohort: each cohort has its own draw)
        None,                    # trend_growth (shared scalar; must be passed positionally)
        None,                    # transfer_floor (shared scalar)
        0,                       # bequest_lumpsum (per-cohort scalar)
)
_simulate_lifecycle_jax_batched = jax.jit(
    jax.vmap(simulate_lifecycle_jax, in_axes=_SIMULATE_IN_AXES),
    static_argnames=('retirement_age', 'T', 'current_age', 'n_sim',
                     'tax_progressive', 'P_y_age_health'),
)


# ---------------------------------------------------------------------------
# Exact aggregation: the distribution over states, carried forward by age
# ---------------------------------------------------------------------------

def exact_age_means_jax(
    a_policy, c_policy, l_policy,
    a_grid, y_grid, h_grid, m_grid,
    P_y, P_h,
    w_path, w_at_retirement,
    tau_c_path, tau_l_path, tau_p_path, tau_k_path,
    r_path, pension_replacement_path,
    ui_replacement_rate, kappa,
    retirement_age, T, current_age,
    initial_dist, alpha_grid,
    pension_min_floor=0.0,
    tax_progressive=False,
    tax_kappa_hsv=0.8,
    tax_eta=0.15,
    P_y_age_health=False,
    P_y_4d=None,
    survival_probs=None,
    wage_age_profile=None,
    pension_avg_weight=1.0,
    mean_kappa_working=1.0,
    mean_y_employed=1.0,
    trend_growth=0.0,
    transfer_floor=0.0,
    bequest_lumpsum=0.0,
    return_dist=False,
    panel_rows=None,
):
    """
    Per-age population means of the panel variables, without simulation.

    The distribution of households over (alpha, a, y, h, y_last) is carried
    forward from initial_dist: the decision rule moves mass between asset
    nodes (a' is a node, so nothing is interpolated), P_y and P_h move it
    between income and health states, and survival scales it. Each mean is the
    mass-weighted sum over the grid of the same per-state outcomes the
    simulation records (_state_outcomes_jax), the dead counting as zero as
    they do in a panel mean.

    initial_dist : (n_alpha, n_a, n_y, n_h, n_y), mass at age current_age,
        summing to one.
    alpha_grid : (n_alpha,) log fixed effects.

    Returns means, (T_sim, 23): column i is the mean of element i of the
    simulate_lifecycle_jax panel. Column 15 (average past earnings) is NaN: it
    depends on the household's history, not on its state. With return_dist
    also returns the mass at the start of each age,
    (T_sim, n_alpha, n_a, n_y, n_h, n_y).

    With panel_rows, an integer array (R,) of age indices, returns
    (means, panel, mass): panel is (R, 23, n_states), the value of each panel
    element at every state of the grid at those ages (booleans and indices as
    floats), and mass is (R, n_states), the mass on each state. Together they
    are a cross-section in which a state stands for a household and its mass
    is the household's weight; the bequest element is its expected value,
    (1 - survival) * (1+g) * a'.
    """
    T_sim = T - current_age
    n_alpha = a_policy.shape[0]
    n_a, n_y, n_h = a_grid.shape[0], y_grid.shape[0], h_grid.shape[0]

    if wage_age_profile is None:
        wage_age_profile = jnp.ones(T)
    if P_y_4d is None:
        P_y_4d = jnp.zeros((T, n_h, n_y, n_y))
    survival = jnp.ones((T, n_h)) if survival_probs is None else jnp.asarray(survival_probs)

    # State indices on the grid, each of shape (n_alpha, n_a, n_y, n_h, n_y)
    K, A, Y, H, YL = jnp.meshgrid(jnp.arange(n_alpha), jnp.arange(n_a), jnp.arange(n_y),
                                  jnp.arange(n_h), jnp.arange(n_y), indexing='ij')
    alpha_mult = jnp.exp(alpha_grid)[K]
    mean_alpha_idx = jnp.sum(initial_dist * K)

    def outcomes_at(age):
        def one_state(i_a, i_y, i_h, i_y_last, alpha_idx, mult):
            return _state_outcomes_jax(
                i_a, i_y, i_h, i_y_last, age,
                a_policy, c_policy, l_policy,
                a_grid, y_grid, h_grid, m_grid,
                w_path, w_at_retirement,
                tau_c_path, tau_l_path, tau_p_path, tau_k_path,
                r_path, pension_replacement_path,
                ui_replacement_rate, kappa,
                retirement_age, current_age,
                pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
                wage_age_profile,
                pension_avg_weight, mean_kappa_working, mean_y_employed,
                alpha_idx, mult,
                transfer_floor, bequest_lumpsum,
            )
        flat = jax.vmap(one_state)(A.ravel(), Y.ravel(), H.ravel(), YL.ravel(),
                                   K.ravel(), alpha_mult.ravel())
        return {name: x.reshape(K.shape) for name, x in flat.items()}

    def columns_at(age):
        """The 23 panel elements at every state (None where not a function of
        the state), with the decision rule and survival used to move the mass."""
        s = outcomes_at(age)
        surv = survival[age][H]
        # Accidental bequest of those who die at this age: the wealth carried
        # out of the period, (1+g) a'.
        bequest = (1.0 - surv) * (1.0 + trend_growth) * a_grid[s['a_pol_val']]
        columns = (
            s['a_val'], s['c_pol_val'], s['y_val'], s['h_val'], H,
            s['effective_y'], s['employed'], s['ui'],
            s['m_val'], s['oop_m'], s['gov_m'],
            s['tax_c'], s['tax_l'], s['tax_p'], s['tax_k'],
            None,                                               # avg_earnings
            s['pension'], s['is_retired'],
            jnp.where(s['employed'], s['l_pol_val'], 0.0),      # hours supplied
            jnp.ones(K.shape),                                  # alive
            bequest,
            None,                                               # alpha index
            s['transfer'],
        )
        return s, surv, columns

    keep_dist = return_dist or panel_rows is not None

    def step(mu, t):
        age = current_age + t
        s, surv, columns = columns_at(age)
        means = jnp.stack([
            jnp.sum(mu * x) if x is not None
            else (jnp.nan if i == 15 else mean_alpha_idx)
            for i, x in enumerate(columns)])

        # Next age. Survivors move to their chosen asset node; then
        #   working: y' ~ P_y(age, h)[y, .], y_last' = y
        #   retired: y' = 0,                 y_last' = y_last
        # and h' ~ P_h(age)[h, .].
        moved = jnp.zeros_like(mu).at[K, s['a_pol_val'], Y, H, YL].add(mu * surv)
        P_y_age = jnp.where(P_y_age_health, P_y_4d[age],
                            jnp.broadcast_to(P_y, (n_h, n_y, n_y)))
        P_h_age = P_h[age]
        working = jnp.einsum('kayh,hyz,hg->kazgy', moved.sum(axis=4), P_y_age, P_h_age)
        retired = jnp.zeros_like(mu).at[:, :, 0, :, :].set(
            jnp.einsum('kayhl,hg->kagl', moved, P_h_age))
        mu_next = jnp.where(age >= retirement_age, retired, working)
        return mu_next, ((means, mu) if keep_dist else means)

    _, out = lax.scan(step, initial_dist, jnp.arange(T_sim))
    if panel_rows is None:
        return out

    means, dists = out

    def panel_row(r):
        _, _, columns = columns_at(current_age + r)
        return jnp.stack([
            jnp.asarray(x, dtype=jnp.float64).reshape(-1) if x is not None
            else (jnp.full(K.size, jnp.nan) if i == 15
                  else K.reshape(-1).astype(jnp.float64))
            for i, x in enumerate(columns)])

    panel = jax.vmap(panel_row)(panel_rows)
    mass = dists[panel_rows].reshape(panel_rows.shape[0], -1)
    return means, panel, mass


_EXACT_STATIC = ('retirement_age', 'T', 'current_age', 'tax_progressive',
                 'P_y_age_health', 'return_dist')
_exact_age_means_jax_jit = jax.jit(exact_age_means_jax, static_argnames=_EXACT_STATIC)

# Batched over cohorts that share the grids, the initial distribution and the
# retirement age.
_EXACT_IN_AXES = (
        0, 0, 0,                 # a_policy, c_policy, l_policy
        None, None, None, None,  # a_grid, y_grid, h_grid, m_grid
        None, None,              # P_y, P_h
        0, 0,                    # w_path, w_at_retirement
        0, 0, 0, 0,             # tau_c/l/p/k_path
        0, 0,                    # r_path, pension_replacement_path
        None, None,              # ui_replacement_rate, kappa
        None, None, None,        # retirement_age, T, current_age
        None, None,              # initial_dist, alpha_grid
        None, None,              # pension_min_floor, tax_progressive
        None, None,              # tax_kappa_hsv, tax_eta
        None, None,              # P_y_age_health, P_y_4d
        0,                       # survival_probs (per-cohort)
        None,                    # wage_age_profile (shared)
        None, None, None,        # pension_avg_weight, mean_kappa_working, mean_y_employed
        None,                    # trend_growth
        None,                    # transfer_floor
        0,                       # bequest_lumpsum (per-cohort scalar)
        None,                    # return_dist
)
_exact_age_means_jax_batched = jax.jit(
    jax.vmap(exact_age_means_jax, in_axes=_EXACT_IN_AXES),
    static_argnames=_EXACT_STATIC,
)


# Integer and boolean elements of the panel; exact_age_means_jax returns every
# element as a float.
_PANEL_INT_FIELDS = (4, 21)
_PANEL_BOOL_FIELDS = (6, 17, 19)


def _exact_panel_to_numpy(panel, mass):
    """(R, 23, n_states) panel and (R, n_states) mass -> the 23-tuple layout of
    simulate(), each element (R, n_states), and the mass."""
    panel = np.asarray(panel)
    fields = []
    for i in range(panel.shape[1]):
        x = panel[:, i]
        if i in _PANEL_INT_FIELDS:
            x = np.rint(x).astype(np.int32)
        elif i in _PANEL_BOOL_FIELDS:
            x = x > 0.5
        fields.append(x)
    return tuple(fields), np.asarray(mass)


def _cross_section_exact(surv, alpha_mults, rows, initial_dist, alpha_grid,
                         a_grid, y_grid, h_grid, m_grid, P_y_2d, P_h, P_y_4d,
                         w_at_retirement, paths,
                         ui_replacement_rate, kappa, beta, gamma,
                         pension_min_floor, tax_kappa_hsv, tax_eta,
                         transfer_floor, education_subsidy_rate,
                         child_cost_profile, nu, phi, trend_growth,
                         wage_age_profile, pension_avg_weight,
                         mean_kappa_working, mean_y_employed, bequest_lumpsum,
                         T, retirement_age, current_age, tax_progressive,
                         schooling_years, labor_supply, P_y_age_health, n_alpha, chunk):
    """Body of LifecycleModelJAX.cross_section_exact, compiled as one call.

    Solves every cohort once per fixed-effect node, as _cross_section does,
    then carries each cohort's distribution over states forward and keeps the
    state-level panel row rows[c] of cohort c with the mass on each state.
    """
    C = surv.shape[0]
    # A (T,) path is shared by the cohorts; a (C, T) one has a row per cohort.
    bc = lambda x: x if jnp.ndim(x) == 2 else jnp.broadcast_to(x, (C,) + jnp.shape(x))
    r_c, w_c, tc_c, tl_c, tp_c, tk_c, pen_c = (bc(x) for x in paths)
    w_ret_c = bc(jnp.asarray(w_at_retirement, dtype=jnp.float64))

    solve = jax.vmap(solve_lifecycle_jax, in_axes=_SOLVE_IN_AXES)

    def solve_node(alpha_mult):
        _, a_b, c_b, l_b = solve(
            a_grid, y_grid, h_grid, m_grid, P_y_2d, P_h,
            w_ret_c, r_c, w_c, tc_c, tl_c, tp_c, tk_c, pen_c,
            ui_replacement_rate, kappa, beta, gamma,
            T, retirement_age,
            pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
            transfer_floor, education_subsidy_rate,
            child_cost_profile, schooling_years,
            surv, P_y_4d,
            labor_supply, nu, phi, trend_growth,
            jnp.zeros(C),
            wage_age_profile,
            pension_avg_weight, mean_kappa_working, mean_y_employed,
            alpha_mult,
        )
        return a_b, c_b, l_b

    nodes = [solve_node(alpha_mults[k]) for k in range(n_alpha)]
    a_pol, c_pol, l_pol = (jnp.stack([nd[f] for nd in nodes], axis=1) for f in range(3))

    exact = jax.vmap(exact_age_means_jax, in_axes=_EXACT_IN_AXES + (0,))
    beq_c = jnp.full(chunk, bequest_lumpsum)
    panels, masses = [], []
    for start in range(0, C, chunk):
        stop = min(start + chunk, C)
        # Pad the last chunk to the chunk length by repeating its last cohort.
        idx = np.concatenate([np.arange(start, stop),
                              np.full(chunk - (stop - start), stop - 1)])
        take = lambda x: x[idx]
        _, panel, mass = exact(
            take(a_pol), take(c_pol), take(l_pol),
            a_grid, y_grid, h_grid, m_grid,
            P_y_2d, P_h,
            take(w_c), take(w_ret_c),
            take(tc_c), take(tl_c), take(tp_c), take(tk_c),
            take(r_c), take(pen_c),
            ui_replacement_rate, kappa,
            retirement_age, T, current_age,
            initial_dist, alpha_grid,
            pension_min_floor, tax_progressive,
            tax_kappa_hsv, tax_eta,
            P_y_age_health, P_y_4d,
            take(surv),
            wage_age_profile,
            pension_avg_weight, mean_kappa_working, mean_y_employed,
            trend_growth,
            transfer_floor,
            beq_c,
            False,
            take(rows)[:, None],
        )
        n = stop - start
        panels.append(panel[:n, 0])
        masses.append(mass[:n, 0])
    return jnp.concatenate(panels), jnp.concatenate(masses)


_cross_section_exact_jit = jax.jit(
    _cross_section_exact,
    static_argnames=('T', 'retirement_age', 'current_age', 'tax_progressive',
                     'schooling_years', 'labor_supply', 'P_y_age_health',
                     'n_alpha', 'chunk'),
)


def _draw_sim_inputs(keys, stationary, asset_dist, a_grid, alpha_probs, alpha_grid,
                     i_a_fixed, avg_earnings0, n_years0,
                     n_sim, n_y, n_alpha, y_mode, a_mode, earnings_mode):
    """LifecycleModelJAX._simulate_inputs for a stack of PRNG keys, in one call.

    Same draws in the same key order as the per-seed method, vmapped over
    keys. y_mode: 'employed' (uniform over employed states) or 'stationary';
    a_mode: 'dist', 'fixed' or 'zero'; earnings_mode: 'given' or 'zero'.
    """
    def draw(key):
        key, subkey = jax.random.split(key)
        if y_mode == 'employed':
            i_y = jax.random.choice(subkey, jnp.arange(1, n_y), shape=(n_sim,)).astype(jnp.int32)
        else:
            i_y = jax.random.choice(subkey, n_y, shape=(n_sim,), p=stationary).astype(jnp.int32)
        i_h = jnp.zeros(n_sim, dtype=jnp.int32)
        if a_mode == 'dist':
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, shape=(n_sim,), minval=0, maxval=asset_dist.shape[0])
            sampled = asset_dist[idx]
            i_a = jnp.argmin(jnp.abs(a_grid[None, :] - sampled[:, None]), axis=1).astype(jnp.int32)
        elif a_mode == 'fixed':
            i_a = jnp.full(n_sim, i_a_fixed, dtype=jnp.int32)
        else:
            i_a = jnp.zeros(n_sim, dtype=jnp.int32)
        if earnings_mode == 'given':
            avg = jnp.ones(n_sim) * avg_earnings0
            n_years = jnp.full(n_sim, n_years0, dtype=jnp.float64)
        else:
            avg = jnp.zeros(n_sim)
            n_years = jnp.zeros(n_sim, dtype=jnp.float64)
        if n_alpha > 1:
            key, subkey = jax.random.split(key)
            alpha_idx = jax.random.choice(
                subkey, n_alpha, shape=(n_sim,), p=alpha_probs).astype(jnp.int32)
        else:
            alpha_idx = jnp.zeros(n_sim, dtype=jnp.int32)
        alpha_mult = jnp.exp(alpha_grid[alpha_idx])
        key, subkey = jax.random.split(key)
        return (i_a, i_y, i_h, i_y, avg, n_years, alpha_idx, alpha_mult, subkey)

    return jax.vmap(draw)(keys)


_draw_sim_inputs_jit = jax.jit(
    _draw_sim_inputs,
    static_argnames=('n_sim', 'n_y', 'n_alpha', 'y_mode', 'a_mode', 'earnings_mode'),
)


def _cross_section(surv, sim_inputs, alpha_mults, rows,
                   a_grid, y_grid, h_grid, m_grid, P_y_2d, P_h, P_y_4d,
                   w_at_retirement, paths,
                   ui_replacement_rate, kappa, beta, gamma,
                   pension_min_floor, tax_kappa_hsv, tax_eta,
                   transfer_floor, education_subsidy_rate,
                   child_cost_profile, nu, phi, trend_growth,
                   wage_age_profile, pension_avg_weight,
                   mean_kappa_working, mean_y_employed, bequest_lumpsum,
                   T, retirement_age, current_age, n_sim, tax_progressive,
                   schooling_years, labor_supply, P_y_age_health, n_alpha, chunk):
    """Body of LifecycleModelJAX.cross_section_batched, compiled as one call.

    Solves every cohort once per fixed-effect node (vmapped over the survival
    schedules), simulates them in chunks of *chunk* cohorts, and keeps panel row
    rows[c] of cohort c. Every call below is the same function the per-cohort solve() and
    simulate() call, vmapped over cohorts.
    """
    C = surv.shape[0]
    # A (T,) path is shared by the cohorts; a (C, T) one has a row per cohort.
    bc = lambda x: x if jnp.ndim(x) == 2 else jnp.broadcast_to(x, (C,) + jnp.shape(x))
    r_c, w_c, tc_c, tl_c, tp_c, tk_c, pen_c = (bc(x) for x in paths)
    w_ret_c = bc(jnp.asarray(w_at_retirement, dtype=jnp.float64))

    solve = jax.vmap(solve_lifecycle_jax, in_axes=_SOLVE_IN_AXES)

    def solve_node(alpha_mult):
        # The bequest lump sum is 0 in the solve, as in solve(); it enters
        # only the simulated budget.
        _, a_b, c_b, l_b = solve(
            a_grid, y_grid, h_grid, m_grid, P_y_2d, P_h,
            w_ret_c, r_c, w_c, tc_c, tl_c, tp_c, tk_c, pen_c,
            ui_replacement_rate, kappa, beta, gamma,
            T, retirement_age,
            pension_min_floor, tax_progressive, tax_kappa_hsv, tax_eta,
            transfer_floor, education_subsidy_rate,
            child_cost_profile, schooling_years,
            surv, P_y_4d,
            labor_supply, nu, phi, trend_growth,
            jnp.zeros(C),
            wage_age_profile,
            pension_avg_weight, mean_kappa_working, mean_y_employed,
            alpha_mult,
        )
        return a_b, c_b, l_b

    # One vmapped sweep per fixed-effect node, as in solve(). Vectorising over
    # the nodes as well measured no faster on an H200 (the sweep is
    # throughput-bound) and holds n_alpha times the policies at once.
    nodes = [solve_node(alpha_mults[k]) for k in range(n_alpha)]
    # (C, n_alpha, T, n_a, n_y, n_h, n_y), the per-alpha layout simulate() reads
    a_pol, c_pol, l_pol = (jnp.stack([nd[f] for nd in nodes], axis=1) for f in range(3))
    (i_a, i_y, i_h, i_y_last, avg_earn, n_years,
     alpha_idx_sim, alpha_mult_sim, keys) = sim_inputs

    simulate = jax.vmap(simulate_lifecycle_jax, in_axes=_SIMULATE_IN_AXES)
    beq_c = jnp.full(chunk, bequest_lumpsum)
    pieces = []
    for start in range(0, C, chunk):
        stop = min(start + chunk, C)
        # Pad the last chunk to the chunk length by repeating its last cohort.
        idx = np.concatenate([np.arange(start, stop),
                              np.full(chunk - (stop - start), stop - 1)])
        take = lambda x: x[idx]
        out = simulate(
            take(a_pol), take(c_pol), take(l_pol),
            a_grid, y_grid, h_grid, m_grid,
            P_y_2d, P_h,
            take(w_c), take(w_ret_c),
            take(tc_c), take(tl_c), take(tp_c), take(tk_c),
            take(r_c), take(pen_c),
            ui_replacement_rate, kappa,
            retirement_age, T, current_age,
            n_sim, take(keys),
            take(i_a), take(i_y), take(i_h), take(i_y_last),
            take(avg_earn), take(n_years),
            pension_min_floor, tax_progressive,
            tax_kappa_hsv, tax_eta,
            P_y_age_health, P_y_4d,
            take(surv),
            wage_age_profile,
            pension_avg_weight, mean_kappa_working, mean_y_employed,
            take(alpha_idx_sim), take(alpha_mult_sim),
            trend_growth,
            transfer_floor,
            beq_c,
        )
        # Each output is (chunk, T_sim, n_sim); keep row rows[c] of cohort c.
        local = np.arange(stop - start)
        pieces.append([x[local, rows[start:stop]] for x in out])
    return tuple(jnp.concatenate([p[f] for p in pieces]) for f in range(len(pieces[0])))


_cross_section_jit = jax.jit(
    _cross_section,
    static_argnames=('T', 'retirement_age', 'current_age', 'n_sim', 'tax_progressive',
                     'schooling_years', 'labor_supply', 'P_y_age_health',
                     'n_alpha', 'chunk'),
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

        # Labor supply parameters
        self.labor_supply = bool(config.labor_supply)
        self.nu = float(config.nu)
        self.phi = float(config.phi)
        self.trend_growth = float(config.trend_growth)

        # New feature parameters
        self.pension_min_floor = float(encoded_pension_floor(config))
        self.tax_progressive = bool(config.tax_progressive)
        self.tax_kappa_hsv = float(config.tax_kappa)
        self.tax_eta = float(config.tax_eta)
        self.transfer_floor = float(config.transfer_floor)
        self.bequest_lumpsum = float(config.bequest_lumpsum)
        self.education_subsidy_rate = float(config.education_subsidy_rate)
        self.schooling_years = int(config.schooling_years)
        self.child_cost_profile = jnp.array(config.child_cost_profile)
        self.wage_age_profile = jnp.array(self._np_model.wage_age_profile)
        self.pension_avg_weight = float(self._np_model.pension_avg_weight)
        self.mean_kappa_working = float(self._np_model.mean_kappa_working)
        self.mean_y_employed = float(self._np_model.mean_y_employed)
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

        # Phase 8: permanent productivity fixed-effect grid (mirrors NumPy reference)
        self.n_alpha = int(self._np_model.n_alpha)
        self.alpha_grid = jnp.array(self._np_model.alpha_grid)
        self.alpha_probs = jnp.array(self._np_model.alpha_probs)
        self._alpha_mult = 1.0

        # Placeholders for results
        self.V = None
        self.a_policy = None
        self.c_policy = None
        self.l_policy = None
        self.V_alpha = None
        self.a_policy_alpha = None
        self.c_policy_alpha = None
        self.l_policy_alpha = None

    def solve(self, verbose=False, **kwargs):
        """Solve lifecycle via JAX backward induction.

        With n_alpha > 1, repeats the JAX solve once per alpha grid point
        (passing alpha_mult = exp(alpha_grid[k]) as a runtime scalar) and
        stores per-alpha policies in self.{V,a,c,l}_policy_alpha. The
        scalar self.{V,a,c,l}_policy attributes are aliased to the alpha=0
        slice so existing callers continue to work unchanged.
        """
        if verbose:
            print(f"Solving lifecycle model (JAX) for {self.config.education_type} education...")
            if self.n_alpha > 1:
                print(f"  Looping over n_alpha = {self.n_alpha} permanent FE grid points")

        V_list, a_list, c_list, l_list = [], [], [], []
        for alpha_idx in range(self.n_alpha):
            alpha_mult = float(np.exp(np.asarray(self.alpha_grid)[alpha_idx]))
            if verbose and self.n_alpha > 1:
                print(f"  alpha[{alpha_idx}] = {float(self.alpha_grid[alpha_idx]):+.4f}  "
                      f"(exp = {alpha_mult:.4f})")
            V, a_policy, c_policy, l_policy = _solve_lifecycle_jax_jit(
                self.a_grid, self.y_grid, self.h_grid, self.m_grid,
                self.P_y_2d,
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
                labor_supply=self.labor_supply,
                nu=self.nu,
                phi=self.phi,
                trend_growth=self.trend_growth,
                wage_age_profile=self.wage_age_profile,
                pension_avg_weight=self.pension_avg_weight,
                mean_kappa_working=self.mean_kappa_working,
                mean_y_employed=self.mean_y_employed,
                alpha_mult=alpha_mult,
            )
            V_list.append(np.asarray(V))
            a_list.append(np.asarray(a_policy))
            c_list.append(np.asarray(c_policy))
            l_list.append(np.asarray(l_policy))

        # Stack per-alpha policies on a leading axis (n_alpha, T, ...)
        self.V_alpha = np.stack(V_list, axis=0)
        self.a_policy_alpha = np.stack(a_list, axis=0)
        self.c_policy_alpha = np.stack(c_list, axis=0)
        self.l_policy_alpha = np.stack(l_list, axis=0)

        # Backward-compat aliases pointing at alpha=0
        self.V = self.V_alpha[0]
        self.a_policy = self.a_policy_alpha[0]
        self.c_policy = self.c_policy_alpha[0]
        self.l_policy = self.l_policy_alpha[0]

        if verbose:
            print("Done!")

    def _simulate_inputs(self, n_sim, seed):
        """Initial states, fixed-effect draws and the simulation key for one seed.

        Shared by simulate() and cross_section_batched(), so both draw the
        same shocks from the same seed.
        """
        key = jax.random.PRNGKey(seed)

        # Initial conditions (same logic as NumPy model)
        edu_unemployment_rate = self.config.edu_params[self.config.education_type]['unemployment_rate']

        if edu_unemployment_rate < 1e-10:
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

        initial_i_y_last = initial_i_y  # JAX arrays are immutable; no copy needed
        initial_i_h = jnp.zeros(n_sim, dtype=jnp.int32)

        if self.config.initial_asset_distribution is not None:
            dist = np.asarray(self.config.initial_asset_distribution)
            key, subkey = jax.random.split(key)
            sample_idx = jax.random.randint(subkey, shape=(n_sim,), minval=0, maxval=len(dist))
            sampled = jnp.array(dist)[sample_idx]
            initial_i_a = jnp.array(
                [int(jnp.argmin(jnp.abs(self.a_grid - float(v)))) for v in np.asarray(sampled)],
                dtype=jnp.int32,
            )
        elif self.config.initial_assets is not None:
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

        # Phase 8: per-agent permanent FE draw. Use the 6-D policies stored as
        # self.{a,c,l}_policy_alpha (always present after solve()). With
        # n_alpha=1 the leading axis is a singleton and the draws collapse to
        # zeros / ones.
        if self.n_alpha > 1:
            key, subkey = jax.random.split(key)
            alpha_idx_sim = jax.random.choice(
                subkey, self.n_alpha, shape=(n_sim,), p=self.alpha_probs
            ).astype(jnp.int32)
        else:
            alpha_idx_sim = jnp.zeros(n_sim, dtype=jnp.int32)
        alpha_mult_sim = jnp.exp(self.alpha_grid[alpha_idx_sim])

        key, subkey = jax.random.split(key)
        return (initial_i_a, initial_i_y, initial_i_h, initial_i_y_last,
                initial_avg_earnings, initial_n_years,
                alpha_idx_sim, alpha_mult_sim, subkey)

    def _simulate_inputs_batched(self, n_sim, seeds):
        """_simulate_inputs for every seed in one compiled call.

        Each output gains a leading seed axis. The per-seed method issues
        about thirty-five small device operations per seed; on a GPU their
        dispatch, not the arithmetic, is what costs.
        """
        edu_u = self.config.edu_params[self.config.education_type]['unemployment_rate']
        if edu_u < 1e-10:
            y_mode, stationary = 'employed', jnp.zeros(self.n_y)
        else:
            eigenvalues, eigenvectors = eig(np.asarray(self.P_y_2d).T)
            st = eigenvectors[:, np.argmax(eigenvalues.real)].real
            y_mode, stationary = 'stationary', jnp.array(st / st.sum())
        i_a_fixed = 0
        asset_dist = jnp.zeros(1)
        if self.config.initial_asset_distribution is not None:
            a_mode = 'dist'
            asset_dist = jnp.array(np.asarray(self.config.initial_asset_distribution))
        elif self.config.initial_assets is not None:
            a_mode = 'fixed'
            i_a_fixed = int(jnp.argmin(jnp.abs(self.a_grid - self.config.initial_assets)))
        else:
            a_mode = 'zero'
        if self.config.initial_avg_earnings is not None:
            earnings_mode, avg0 = 'given', float(self.config.initial_avg_earnings)
        else:
            earnings_mode, avg0 = 'zero', 0.0
        keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, dtype=jnp.int64))
        return _draw_sim_inputs_jit(
            keys, stationary, asset_dist, self.a_grid, self.alpha_probs, self.alpha_grid,
            i_a_fixed, avg0, float(self.current_age),
            n_sim=int(n_sim), n_y=int(self.n_y), n_alpha=int(self.n_alpha),
            y_mode=y_mode, a_mode=a_mode, earnings_mode=earnings_mode)

    def simulate(self, T_sim=None, n_sim=10000, seed=42, **kwargs):
        """
        Simulate lifecycle paths.

        Returns the 22-tuple matching LifecycleModelPerfectForesight.simulate().
        Phase 8: each agent draws alpha_idx from self.alpha_probs at t=0; the
        6-D per-alpha policies (self.{a,c,l}_policy_alpha) and the implied
        per-agent multiplier alpha_mult = exp(alpha_grid[alpha_idx]) flow into
        simulate_lifecycle_jax. With n_alpha=1, alpha_idx is all zero,
        alpha_mult is all one, and behavior matches pre-Phase-8 exactly.
        """
        if self.a_policy_alpha is None:
            raise RuntimeError("Must call solve() before simulate().")
        if (float(self.transfer_floor) > 0.0
                and int(getattr(self.config, 'schooling_years', 0) or 0) > 0):
            raise NotImplementedError(
                "the simulated transfer_sim replicates the solve's budget without "
                "child costs; a positive transfer_floor with schooling_years > 0 "
                "would record the wrong transfer")

        if T_sim is None:
            T_sim = self.T - self.current_age

        (initial_i_a, initial_i_y, initial_i_h, initial_i_y_last,
         initial_avg_earnings, initial_n_years,
         alpha_idx_sim, alpha_mult_sim, subkey) = self._simulate_inputs(n_sim, seed)

        # Convert per-alpha policies to JAX (shape (n_alpha, T, n_a, n_y, n_h, n_y))
        a_policy_jax = jnp.array(self.a_policy_alpha)
        c_policy_jax = jnp.array(self.c_policy_alpha)
        l_policy_jax = jnp.array(self.l_policy_alpha)

        # Build P_y_4d dummy for JAX tracing if not age/health-dependent
        P_y_4d_sim = self.P_y_4d if self.P_y_age_health else None

        result = _simulate_lifecycle_jax_jit(
            a_policy_jax, c_policy_jax, l_policy_jax,
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
            survival_probs=self.survival_probs,
            wage_age_profile=self.wage_age_profile,
            pension_avg_weight=self.pension_avg_weight,
            mean_kappa_working=self.mean_kappa_working,
            mean_y_employed=self.mean_y_employed,
            alpha_idx_sim=alpha_idx_sim,
            alpha_mult_sim=alpha_mult_sim,
            trend_growth=self.trend_growth,
            transfer_floor=float(self.transfer_floor),
            bequest_lumpsum=float(self.bequest_lumpsum),
        )

        # Convert all outputs to numpy arrays (23-tuple: panel, alpha_idx_panel, transfer_sim)
        return tuple(np.asarray(x) for x in result)

    def exact_age_means(self, T_sim=None, return_dist=False):
        """Per-age population means of the panel variables, without simulation.

        Same interface and conventions as
        LifecycleModelPerfectForesight.exact_age_means(): (T_sim, 23), column i
        the mean of element i of the simulate() panel. See exact_age_means_jax.
        """
        if self.a_policy_alpha is None:
            raise RuntimeError("Must call solve() before exact_age_means().")
        if (float(self.transfer_floor) > 0.0
                and int(getattr(self.config, 'schooling_years', 0) or 0) > 0):
            raise NotImplementedError(
                "the recorded transfer replicates the solve's budget without "
                "child costs; a positive transfer_floor with schooling_years > 0 "
                "would record the wrong transfer")
        if T_sim is not None and T_sim != self.T - self.current_age:
            raise ValueError("exact_age_means covers ages current_age..T-1")

        out = _exact_age_means_jax_jit(
            jnp.array(self.a_policy_alpha), jnp.array(self.c_policy_alpha),
            jnp.array(self.l_policy_alpha),
            self.a_grid, self.y_grid, self.h_grid, self.m_grid,
            self.P_y_2d, self.P_h,
            self.w_path, self.w_at_retirement,
            self.tau_c_path, self.tau_l_path, self.tau_p_path, self.tau_k_path,
            self.r_path, self.pension_replacement_path,
            self.ui_replacement_rate, self.kappa,
            self.retirement_age, self.T, self.current_age,
            jnp.array(self._np_model._initial_distribution()), self.alpha_grid,
            pension_min_floor=self.pension_min_floor,
            tax_progressive=self.tax_progressive,
            tax_kappa_hsv=self.tax_kappa_hsv,
            tax_eta=self.tax_eta,
            P_y_age_health=self.P_y_age_health,
            P_y_4d=self.P_y_4d if self.P_y_age_health else None,
            survival_probs=self.survival_probs,
            wage_age_profile=self.wage_age_profile,
            pension_avg_weight=self.pension_avg_weight,
            mean_kappa_working=self.mean_kappa_working,
            mean_y_employed=self.mean_y_employed,
            trend_growth=self.trend_growth,
            transfer_floor=float(self.transfer_floor),
            bequest_lumpsum=float(self.bequest_lumpsum),
            return_dist=bool(return_dist),
        )
        if return_dist:
            return np.asarray(out[0]), np.asarray(out[1])
        return np.asarray(out)

    def exact_panel(self, rows=None):
        """State-level cross-sections at the ages in *rows* (default: all).

        Returns (panel, mass): panel is the 23-tuple layout of simulate() with
        one column per state of the grid, each element (len(rows), n_states);
        mass is the mass of households on each state at that age. A column with
        zero mass is a state nobody is in. See exact_age_means_jax.
        """
        if self.a_policy_alpha is None:
            raise RuntimeError("Must call solve() before exact_panel().")
        T_sim = self.T - self.current_age
        rows = np.arange(T_sim) if rows is None else np.asarray(rows, dtype=int)
        _, panel, mass = _exact_age_means_jax_jit(
            jnp.array(self.a_policy_alpha), jnp.array(self.c_policy_alpha),
            jnp.array(self.l_policy_alpha),
            self.a_grid, self.y_grid, self.h_grid, self.m_grid,
            self.P_y_2d, self.P_h,
            self.w_path, self.w_at_retirement,
            self.tau_c_path, self.tau_l_path, self.tau_p_path, self.tau_k_path,
            self.r_path, self.pension_replacement_path,
            self.ui_replacement_rate, self.kappa,
            self.retirement_age, self.T, self.current_age,
            jnp.array(self._np_model._initial_distribution()), self.alpha_grid,
            pension_min_floor=self.pension_min_floor,
            tax_progressive=self.tax_progressive,
            tax_kappa_hsv=self.tax_kappa_hsv,
            tax_eta=self.tax_eta,
            P_y_age_health=self.P_y_age_health,
            P_y_4d=self.P_y_4d if self.P_y_age_health else None,
            survival_probs=self.survival_probs,
            wage_age_profile=self.wage_age_profile,
            pension_avg_weight=self.pension_avg_weight,
            mean_kappa_working=self.mean_kappa_working,
            mean_y_employed=self.mean_y_employed,
            trend_growth=self.trend_growth,
            transfer_floor=float(self.transfer_floor),
            bequest_lumpsum=float(self.bequest_lumpsum),
            panel_rows=jnp.asarray(rows),
        )
        return _exact_panel_to_numpy(panel, mass)

    def _pension_stack(self, pension_stack, C):
        """Replacement-rate paths of the C cohorts of a cross-section: this
        model's path for all of them, or one (T,) row per cohort."""
        if pension_stack is None:
            return self.pension_replacement_path
        pen = jnp.asarray(pension_stack, dtype=jnp.float64)
        if pen.shape != (C, int(self.T)):
            raise ValueError(f'pension_stack has shape {pen.shape}, expected {(C, int(self.T))}')
        return pen

    def cross_section_exact(self, survival_stack, rows=None, chunk_size=None,
                            pension_stack=None):
        """One state-level panel row per cohort, for cohorts that differ only
        in survival and, with pension_stack, in their replacement-rate path:
        the exact counterpart of cross_section_batched().

        Cohort c solves this model's lifecycle problem under survival_stack[c]
        and contributes its cross-section over states at age index rows[c]
        (default rows[c] = c). Returns (panel, mass) as exact_panel() does,
        each element (C, n_states).
        """
        surv = jnp.asarray(survival_stack, dtype=jnp.float64)
        C = surv.shape[0]
        rows = np.arange(C) if rows is None else np.asarray(rows, dtype=int)
        if rows.shape != (C,) or rows.min() < 0 or rows.max() >= self.T - self.current_age:
            raise ValueError(f'rows {rows} do not index the panel\'s '
                             f'{self.T - self.current_age} rows for {C} cohorts')
        if (float(self.transfer_floor) > 0.0
                and int(getattr(self.config, 'schooling_years', 0) or 0) > 0):
            raise NotImplementedError(
                "the recorded transfer replicates the solve's budget without "
                "child costs; a positive transfer_floor with schooling_years > 0 "
                "would record the wrong transfer")
        chunk = C if chunk_size is None else max(1, min(int(chunk_size), C))
        alpha_mults = np.exp(np.asarray(self.alpha_grid))
        panel, mass = _cross_section_exact_jit(
            surv, jnp.asarray(alpha_mults), jnp.asarray(rows),
            jnp.array(self._np_model._initial_distribution()), self.alpha_grid,
            self.a_grid, self.y_grid, self.h_grid, self.m_grid,
            self.P_y_2d, self.P_h, self.P_y_4d if self.P_y_age_health else None,
            self.w_at_retirement,
            (self.r_path, self.w_path, self.tau_c_path, self.tau_l_path,
             self.tau_p_path, self.tau_k_path, self._pension_stack(pension_stack, C)),
            self.ui_replacement_rate, self.kappa, self.beta, self.gamma,
            self.pension_min_floor, self.tax_kappa_hsv, self.tax_eta,
            self.transfer_floor, self.education_subsidy_rate,
            self.child_cost_profile, self.nu, self.phi, self.trend_growth,
            self.wage_age_profile, self.pension_avg_weight,
            self.mean_kappa_working, self.mean_y_employed,
            float(self.bequest_lumpsum),
            T=int(self.T), retirement_age=int(self.retirement_age),
            current_age=int(self.current_age),
            tax_progressive=bool(self.tax_progressive),
            schooling_years=int(self.schooling_years),
            labor_supply=bool(self.labor_supply),
            P_y_age_health=bool(self.P_y_age_health),
            n_alpha=int(self.n_alpha), chunk=chunk,
        )
        return _exact_panel_to_numpy(panel, mass)

    def cross_section_batched(self, survival_stack, seeds, n_sim, chunk_size=None,
                              rows=None, pension_stack=None):
        """One panel row per cohort, for cohorts that differ only in survival
        and, with pension_stack, in their replacement-rate path.

        Cohort c solves this model's lifecycle problem under survival_stack[c]
        (T, n_h), simulates n_sim agents from seeds[c] and contributes row
        rows[c] of its panel (default rows[c] = c, the cohort aged c); the
        result is the 23-tuple of simulate() with shape (C, n_sim) per field. Equivalent to C separate solve() + simulate()
        calls on copies of this model, run as two compiled calls: the initial
        draws, then the solve sweeps, simulation and row selection.

        chunk_size bounds how many cohorts are simulated at once: the
        simulation holds C x T x n_sim values per output before the row is
        taken. None simulates every cohort at once.
        """
        surv = jnp.asarray(survival_stack, dtype=jnp.float64)
        C = surv.shape[0]
        if len(seeds) != C:
            raise ValueError(f'{len(seeds)} seeds for {C} survival schedules')
        rows = np.arange(C) if rows is None else np.asarray(rows, dtype=int)
        if rows.shape != (C,) or rows.min() < 0 or rows.max() >= self.T - self.current_age:
            raise ValueError(f'rows {rows} do not index the panel\'s '
                             f'{self.T - self.current_age} rows for {C} cohorts')
        if (float(self.transfer_floor) > 0.0
                and int(getattr(self.config, 'schooling_years', 0) or 0) > 0):
            raise NotImplementedError(
                "the simulated transfer_sim replicates the solve's budget without "
                "child costs; a positive transfer_floor with schooling_years > 0 "
                "would record the wrong transfer")
        sim_inputs = self._simulate_inputs_batched(n_sim, seeds)
        chunk = C if chunk_size is None else max(1, min(int(chunk_size), C))
        alpha_mults = np.exp(np.asarray(self.alpha_grid))
        out = _cross_section_jit(
            surv, sim_inputs, jnp.asarray(alpha_mults), jnp.asarray(rows),
            self.a_grid, self.y_grid, self.h_grid, self.m_grid,
            self.P_y_2d, self.P_h, self.P_y_4d if self.P_y_age_health else None,
            self.w_at_retirement,
            (self.r_path, self.w_path, self.tau_c_path, self.tau_l_path,
             self.tau_p_path, self.tau_k_path, self._pension_stack(pension_stack, C)),
            self.ui_replacement_rate, self.kappa, self.beta, self.gamma,
            self.pension_min_floor, self.tax_kappa_hsv, self.tax_eta,
            self.transfer_floor, self.education_subsidy_rate,
            self.child_cost_profile, self.nu, self.phi, self.trend_growth,
            self.wage_age_profile, self.pension_avg_weight,
            self.mean_kappa_working, self.mean_y_employed,
            float(self.bequest_lumpsum),
            T=int(self.T), retirement_age=int(self.retirement_age),
            current_age=int(self.current_age), n_sim=int(n_sim),
            tax_progressive=bool(self.tax_progressive),
            schooling_years=int(self.schooling_years),
            labor_supply=bool(self.labor_supply),
            P_y_age_health=bool(self.P_y_age_health),
            n_alpha=int(self.n_alpha), chunk=chunk,
        )
        return tuple(np.asarray(x) for x in out)


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
