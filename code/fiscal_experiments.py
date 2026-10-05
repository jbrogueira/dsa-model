"""
Fiscal Scenario Framework for OLG Transition Model
====================================================

Computes budget-balanced (or debt-financed) transition paths under exogenous
policy shocks.  Three experiment types:

  Type A — Debt-financed    : one simulate_transition() call; B_path is the residual.
  Type B — Tax/transfer-financed : root-find on a scalar Δτ until the chosen balance
             condition is satisfied, starting from Δτ = 0.
  Type C — NFA/CA-constrained : outer NFA/CA feasibility check + inner bisection
             on shock scale (Mode I) or tax rate (Mode II).

Usage
-----
    from fiscal_experiments import FiscalScenario, run_fiscal_scenario, compare_scenarios

    scenario = FiscalScenario(
        name='spending_shock',
        delta_G_path=np.ones(T) * 0.02,
        financing='tau_l',
        balance_condition='terminal_debt_gdp',
        target_debt_gdp=0.60,
    )

    base_paths = dict(r_path=r_path, tau_l_path=tau_l_base, ...)
    result = run_fiscal_scenario(olg, scenario, base_paths, n_sim=500, verbose=False)
    fig = compare_scenarios(result_base, result)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRE_TP_KEYS = ('r_path', 'w_path', 'tau_l_path', 'tau_c_path', 'tau_p_path',
                'tau_k_path', 'pension_replacement_path')


def _nice_ylim(ax, min_span=0.01):
    """Enforce a minimum y-axis span so tiny MC noise doesn't dominate the plot.

    The minimum span is max(min_span, 0.1 × order-of-magnitude of |midpoint|),
    which gives at least two readable decimal places at any scale.
    """
    ylo, yhi = ax.get_ylim()
    ymid = (ylo + yhi) / 2
    if abs(ymid) > 1e-10:
        scale_span = 10 ** np.floor(np.log10(abs(ymid))) * 0.1
    else:
        scale_span = min_span
    span = max(min_span, scale_span)
    if (yhi - ylo) < span:
        ax.set_ylim(ymid - span / 2, ymid + span / 2)


def _build_pre_transition_paths(olg, base_paths):
    """Build pre_transition_paths dict from base_paths + lifecycle_config."""
    pre_tp = {k: base_paths.get(k) for k in _PRE_TP_KEYS}
    pre_tp['transfer_floor'] = getattr(olg.lifecycle_config, 'transfer_floor', 0.0)
    return pre_tp


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FiscalScenario:
    """Specifies an exogenous policy shock, a financing instrument, and a
    budget balance condition.

    All ``delta_*`` arrays are *additive* changes relative to the base paths
    and must have length ``T_transition`` when provided.
    """
    name: str = 'unnamed'

    # ── Policy shocks (additive, relative to base) ──────────────────────────
    delta_G_path:        Optional[np.ndarray] = None   # Δ government consumption
    delta_I_g_path:      Optional[np.ndarray] = None   # Δ public investment
    delta_tau_l_path:    Optional[np.ndarray] = None   # Δ labor income tax
    delta_tau_c_path:    Optional[np.ndarray] = None   # Δ consumption tax
    delta_tau_k_path:    Optional[np.ndarray] = None   # Δ capital income tax
    delta_tau_p_path:    Optional[np.ndarray] = None   # Δ payroll tax
    delta_pension_path:  Optional[np.ndarray] = None   # Δ pension replacement rate

    # ── Financing instrument ─────────────────────────────────────────────────
    # 'debt'          : B_path is the residual (no iteration on agent problems)
    # 'tau_l' | 'tau_c' | 'tau_k' | 'tau_p' : that tax rate adjusts
    # 'transfer_floor': means-tested transfer floor adjusts (scalar uniform shift)
    financing: str = 'debt'

    # ── Adjustment profile ───────────────────────────────────────────────────
    # The balancing instrument enters as: Δinstrument_t = Δ * psi_t
    # None → uniform: psi_t = 1 for all t
    adjustment_profile: Optional[np.ndarray] = None

    # ── Budget balance condition ─────────────────────────────────────────────
    # 'terminal_debt_gdp'    : B_T / Y_T = target_debt_gdp
    # 'terminal_nfa_gdp'     : NFA_T / Y_T = target_nfa_gdp  (full NFA = A - K_dom - B)
    #                          Terminal external-balance target, analogue of
    #                          terminal_debt_gdp; interior NFA path is free.
    # 'terminal_flow_balance': PD[T-1]/Y[T-1] = (G - 1 - r_terminal) * target_debt_gdp,
    #                          with G = (1+g)(1+n) read from the OLG object
    #                          Ensures the terminal period is a fiscal rest point.
    #                          For r > g and target = 0: PD[T-1] ≈ 0.
    # 'pv_balance'           : Σ_t δ^t * PrimaryDeficit_t = 0
    # 'period_balance'       : minimise max_t |PrimaryDeficit_t| (bounded scalar min)
    balance_condition: str = 'terminal_debt_gdp'
    target_debt_gdp: float = 0.0
    target_nfa_gdp: float = 0.0        # used only by 'terminal_nfa_gdp'
    discount_rate: float = 0.04        # used only by 'pv_balance'

    # ── Initial debt ─────────────────────────────────────────────────────────
    B_initial: float = 0.0

    # ── External constraint (optional) ───────────────────────────────────────
    nfa_limit: Optional[float] = None  # per-period NFA floor: NFA_t >= -nfa_limit
    ca_limit:  Optional[float] = None  # per-period CA floor: CA_t >= -ca_limit

    # ── Bequest handling ─────────────────────────────────────────────────────
    recompute_bequests: bool = False

    # ── Post-target extension ─────────────────────────────────────────────────
    # When n_post > 0, the simulation runs for T_transition + n_post periods
    # with all paths frozen at their terminal values.  The balance condition is
    # still evaluated at T_transition; the extra periods show convergence to the
    # long-run rest point.
    n_post: int = 0


@dataclass
class FiscalScenarioResult:
    """Output of run_fiscal_scenario()."""
    scenario: FiscalScenario

    # Macroeconomic paths (dicts with keys r, w, K, L, Y, and optionally K_g, NFA)
    base_macro:  dict
    cf_macro:    dict

    # Fiscal paths (dicts of np.ndarray, length T_transition)
    base_budget: dict
    cf_budget:   dict

    # Debt paths (length T_transition + 1)
    B_path:     np.ndarray
    B_gdp_path: np.ndarray

    # NFA/CA paths (length T_transition), populated in SOE mode
    NFA_path: Optional[np.ndarray]
    CA_path:  Optional[np.ndarray]

    # Adjustment found by bisection
    adjustment_scalar: float
    adjustment_path:   np.ndarray
    adjustment_label:  str

    # Solver diagnostics
    converged:        bool
    n_iterations:     int
    residual_history: list

    # Terminal convergence diagnostics
    # drift: relative period-on-period change for each stock at t = T-1
    # terminal_converged: True if all drifts < convergence tolerance
    terminal_drift:      dict = field(default_factory=dict)
    terminal_converged:  bool = True

    # Period at which the balance condition was evaluated (< total simulation
    # length when n_post > 0; None means balance was at the last period).
    T_balance: Optional[int] = None


# ---------------------------------------------------------------------------
# Adjustment profile helpers
# ---------------------------------------------------------------------------

def uniform_profile(T: int) -> np.ndarray:
    """psi_t = 1 for all t. Δ is a permanent level shift."""
    return np.ones(T)


def linear_phase_in(T: int, n_ramp: int) -> np.ndarray:
    """psi_t ramps linearly from 0 to 1 over n_ramp periods, then stays at 1."""
    if n_ramp <= 0:
        return np.ones(T)
    n_ramp = min(n_ramp, T)
    ramp = np.linspace(0, 1, n_ramp + 1)[1:]
    tail = np.ones(T - n_ramp)
    return np.concatenate([ramp, tail])


def back_loaded(T: int, n_delay: int) -> np.ndarray:
    """psi_t = 0 for first n_delay periods, then 1."""
    n_delay = max(0, min(n_delay, T))
    return np.concatenate([np.zeros(n_delay), np.ones(T - n_delay)])


def exponential_convergence(T: int, half_life: float) -> np.ndarray:
    """psi_t = 1 - 0.5^(t / half_life). Asymptotically approaches 1."""
    t = np.arange(T)
    return 1.0 - 0.5 ** (t / half_life)


# ---------------------------------------------------------------------------
# Path extension utility
# ---------------------------------------------------------------------------

def _extend_base_paths(base_paths: dict, n_post: int) -> dict:
    """Return base_paths with every 1-D float array extended by n_post periods.

    The last value of each array is repeated, so prices and policies are frozen
    at their terminal levels for the post-target extension window.
    Non-array values (scalars, strings, nested dicts) are passed through unchanged.
    """
    if n_post <= 0:
        return base_paths
    extended = {}
    for key, val in base_paths.items():
        if val is None or isinstance(val, (str, dict)):
            extended[key] = val
            continue
        try:
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 1 and len(arr) > 0:
                extended[key] = np.concatenate([arr, np.full(n_post, arr[-1])])
            else:
                extended[key] = val
        except (TypeError, ValueError):
            extended[key] = val
    return extended


# ---------------------------------------------------------------------------
# Debt accumulation utility
# ---------------------------------------------------------------------------

def _growth_seq(growth_factor, n: int) -> np.ndarray:
    """Gamma_t for n periods. A scalar is broadcast; a short path is held."""
    G = np.atleast_1d(np.asarray(growth_factor, dtype=float))
    if G.size == 1:
        return np.full(n, float(G[0]))
    if G.size >= n:
        return G[:n].astype(float)
    return np.concatenate([G, np.full(n - G.size, G[-1])])


def _olg_growth(olg, n: int) -> np.ndarray:
    """Gamma_t over n periods from an economy, demographic path included."""
    fn = getattr(olg, 'growth_factors', None)
    if fn is not None:
        return _growth_seq(fn(n), n)
    return _growth_seq(getattr(olg, 'growth_factor', 1.0), n)


def compute_debt_path(primary_deficit_path: np.ndarray,
                      r_B_path: np.ndarray,
                      B_initial: float = 0.0,
                      growth_factor=1.0) -> np.ndarray:
    """Forward-recursive sovereign-debt accumulation.

    B[0] = B_initial
    B[t+1] = [(1 + r_B[t]) * B[t] + PrimaryDeficit[t]] / Gamma_t

    Debt is a per-capita detrended stock, so the whole right-hand side — the
    stock carried forward and the period-t deficit alike — is divided by
    Gamma_t = (1+g)(1+n_t), passed either as a scalar or as a path. The
    default 1.0 reproduces the no-growth recursion exactly; callers in the
    transition pass the economy's Gamma_t.

    The sovereign rate r_B is used, not the capital return r — these can differ
    in SOE mode (sovereign spread).  Callers should pass base_paths['r_B_path'],
    which `run_fiscal_scenario` populates from olg.r_B (with capital-r fallback).

    Returns B_path of length T_transition + 1.
    """
    T = len(primary_deficit_path)
    B = np.zeros(T + 1)
    B[0] = B_initial
    G = _growth_seq(growth_factor, T)
    for t in range(T):
        B[t + 1] = ((1.0 + float(r_B_path[t])) * B[t]
                    + float(primary_deficit_path[t])) / G[t]
    return B


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _balance_residual(budget_path: dict,
                      Y_path: np.ndarray,
                      B_path: np.ndarray,
                      scenario: FiscalScenario,
                      r_terminal: float = 0.04,
                      T_balance: Optional[int] = None,
                      NFA_partial: Optional[np.ndarray] = None,
                      growth_factor=1.0) -> float:
    """Compute the scalar balance residual for the chosen balance_condition.

    Returns a value whose *sign* tells bisection which direction to adjust:
      > 0  → too much deficit  (raise taxes / cut spending more)
      < 0  → surplus           (cut taxes / raise spending)
    For 'period_balance' returns max_t |PD_t| (always >= 0, use minimization).

    Parameters
    ----------
    r_terminal : float
        Terminal interest rate; used only by 'terminal_flow_balance'.
    T_balance : int, optional
        Period at which to evaluate the balance condition.  Defaults to the
        full simulation length.  Set to T_transition when n_post > 0 so the
        condition is evaluated at the original horizon, not the extended one.
    """
    PD = budget_path['primary_deficit']
    T_full = len(PD)
    T_bal  = T_balance if T_balance is not None else T_full
    cond = scenario.balance_condition

    if cond == 'terminal_debt_gdp':
        # Date the stock and the flow together. B_path has length T_full+1 and
        # B_path[t] is the end-of-period-t stock, so the stock contemporaneous
        # with Y_path[T_bal-1] is B_path[T_bal-1]. Until 2026-10-01 this used
        # B_path[T_bal] over Y_path[T_bal-1], one period apart, while
        # terminal_nfa_gdp below paired them correctly -- so the two balance
        # conditions targeted differently dated ratios.
        B_T = B_path[T_bal - 1]
        Y_T = float(Y_path[T_bal - 1])
        return float(B_T / Y_T - scenario.target_debt_gdp)

    elif cond == 'terminal_nfa_gdp':
        # Full NFA = (A - K_domestic) - B.  NFA_partial is the simulation's
        # A - K_domestic (length T_full); B_path[t] is the end-of-period-t stock.
        # Evaluate at the balance horizon end (T_bal-1), paired with Y[T_bal-1].
        if NFA_partial is None:
            raise ValueError("terminal_nfa_gdp requires NFA_partial (A - K_domestic).")
        NFA_T = float(np.asarray(NFA_partial)[T_bal - 1] - B_path[T_bal - 1])
        Y_T   = float(Y_path[T_bal - 1])
        # >0 when NFA below target → raise τ_l (lowers B, raises NFA).
        return float(scenario.target_nfa_gdp - NFA_T / Y_T)

    elif cond == 'terminal_flow_balance':
        # Fiscal rest point of B' = [(1+r_B)B + PD]/G:
        #   PD[T_bal-1]/Y[T_bal-1] = (G - 1 - r_B) * target_debt_gdp
        G    = float(_growth_seq(growth_factor, T_bal)[T_bal - 1])
        PD_T = float(PD[T_bal - 1])
        Y_T  = float(Y_path[T_bal - 1])
        stability_rhs = (G - 1.0 - r_terminal) * scenario.target_debt_gdp
        return float(PD_T / Y_T - stability_rhs)

    elif cond == 'pv_balance':
        delta = scenario.discount_rate
        t_arr = np.arange(T_bal)
        weights = (1.0 / (1.0 + delta)) ** t_arr
        return float(np.dot(weights, PD[:T_bal]))

    elif cond == 'period_balance':
        return float(np.max(np.abs(PD[:T_bal])))

    else:
        raise ValueError(f"Unknown balance_condition: {scenario.balance_condition!r}")


def _check_terminal_convergence(cf_macro: dict,
                                cf_budget: dict,
                                olg,
                                tol: float = 0.005,
                                B_path=None) -> tuple:
    """Check that key stocks have stopped moving in the last period.

    Computes relative period-on-period change |x[T-1] - x[T-2]| / |x[T-2]|
    for macro stocks (K, K_g, L, Y, C, A) and fiscal flows
    (total_revenue, total_spending, S_pens).

    NFA IS tracked, at tol_slow (see below). In an SOE it equals A - K - B, so it drifts whenever B
    is still accumulating even if real variables have converged.
    primary_deficit is excluded: it can be near zero, making relative changes
    numerically meaningless.

    NFA and S_pens are tracked separately with a looser threshold (tol_slow)
    because pension funds and the external position converge slowly.

    Also checks whether K_g has reached its new steady state:
    K_g_ss = I_g_terminal / (delta_g + Gamma_T - 1).  A large K_g_ss_gap means T_transition
    is too short for the public-capital block to have converged.

    Parameters
    ----------
    cf_macro  : counterfactual macro dict
    cf_budget : counterfactual budget dict
    olg       : OLGTransition instance (for delta_g, _active_I_g_path)
    tol       : relative-change threshold for core variables (default 0.5%)

    Returns
    -------
    drift : dict mapping variable name → relative change at terminal period
    all_converged : bool, True if every drift < its threshold
    """
    tol_slow = 4 * tol   # looser tolerance for slow-converging stocks (NFA, S_pens)
    drift = {}
    thresholds = {}

    # ── Core macro stocks ─────────────────────────────────────────────────────
    for key in ['K', 'K_g', 'L', 'Y', 'C', 'A']:
        arr = cf_macro.get(key)
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            if len(arr) >= 2:
                drift[key] = float(abs(arr[-1] - arr[-2]) / (abs(arr[-2]) + 1e-10))
                thresholds[key] = tol

    # ── NFA: tracked but with looser tolerance ─────────────────────────────────
    nfa = cf_macro.get('NFA')
    if nfa is not None:
        nfa = np.asarray(nfa, dtype=float)
        if len(nfa) >= 2:
            drift['NFA'] = float(abs(nfa[-1] - nfa[-2]) / (abs(nfa[-2]) + 1e-10))
            thresholds['NFA'] = tol_slow

    # ── Fiscal flows (revenue and spending, not deficit) ──────────────────────
    for key in ['total_revenue', 'total_spending']:
        arr = cf_budget.get(key)
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            if len(arr) >= 2:
                drift[key] = float(abs(arr[-1] - arr[-2]) / (abs(arr[-2]) + 1e-10))
                thresholds[key] = tol

    # ── Sovereign debt: no rest point when r_B > Gamma_T - 1 ───────────────────
    # Tracked at the slow tolerance and reported, not imposed: with the
    # balancing item set at the base year the baseline debt ratio drifts, and
    # a run can be 'converged' in every other stock while B/Y is still moving.
    # Until 2026-10-02 B was not in this list at all.
    if B_path is not None:
        Bp = np.asarray(B_path, dtype=float)
        if len(Bp) >= 2:
            drift['B'] = float(abs(Bp[-1] - Bp[-2]) / (abs(Bp[-2]) + 1e-10))
            thresholds['B'] = tol_slow

    # ── Pension fund: slow-converging ──────────────────────────────────────────
    S = cf_budget.get('S_pens')
    if S is not None:
        S = np.asarray(S, dtype=float)
        if len(S) >= 2:
            drift['S_pens'] = float(abs(S[-1] - S[-2]) / (abs(S[-2]) + 1e-10))
            thresholds['S_pens'] = tol_slow

    # ── K_g steady-state gap: |K_g[T-1] - I_g_terminal/(delta_g + G - 1)| ───
    K_g = cf_macro.get('K_g')
    if K_g is not None and olg is not None:
        delta_g  = float(getattr(olg, 'delta_g', 0.0))
        # The rest point is terminal, so it takes Gamma_T.
        G_growth = float(_olg_growth(olg, len(np.asarray(K_g)))[-1])
        I_g_path = getattr(olg, '_active_I_g_path', None)
        # Stationary public capital solves K_g = [(1-delta_g)K_g + I_g]/G.
        denom = delta_g + G_growth - 1.0
        if denom > 0 and I_g_path is not None:
            I_g_T  = float(np.asarray(I_g_path)[-1])
            K_g_ss = I_g_T / denom
            if K_g_ss > 0:
                K_g_T = float(np.asarray(K_g)[-1])
                drift['K_g_ss_gap'] = float(abs(K_g_T - K_g_ss) / K_g_ss)
                thresholds['K_g_ss_gap'] = tol

    all_converged = all(drift[k] < thresholds[k] for k in drift)
    return drift, all_converged


def _get_psi(scenario: FiscalScenario, T: int) -> np.ndarray:
    """Return the adjustment profile psi_t (length T)."""
    if scenario.adjustment_profile is None:
        return np.ones(T)
    psi = np.asarray(scenario.adjustment_profile, dtype=float)
    if len(psi) != T:
        raise ValueError(
            f"adjustment_profile has length {len(psi)}, expected {T}"
        )
    return psi


def _apply_shock(scenario: FiscalScenario,
                 base_paths: dict,
                 T: int,
                 instrument_delta: float = 0.0,
                 shock_scale: float = 1.0) -> dict:
    """Build counterfactual policy paths.

    Parameters
    ----------
    scenario : FiscalScenario
    base_paths : dict
        Must contain 'r_path'.  Optionally: 'tau_c_path', 'tau_l_path',
        'tau_p_path', 'tau_k_path', 'pension_replacement_path',
        'G_path', 'I_g_path'.  Missing keys default to None (zero paths).
    T : int
        T_transition.
    instrument_delta : float
        Scalar Δ for the financing instrument (found by bisection).
    shock_scale : float
        Scale factor applied to ALL delta_* shock paths (for Mode I NFA
        bisection where the question is "how much of the shock is feasible").

    Returns
    -------
    dict with keys: 'tau_c', 'tau_l', 'tau_p', 'tau_k',
                    'pension_replacement', 'G_path', 'I_g_path'
                    plus 'transfer_floor_delta' (scalar) when financing='transfer_floor'.
    """
    def _base(key, default=None):
        v = base_paths.get(key, default)
        if v is None:
            return None
        arr = np.array(v, dtype=float)
        if len(arr) < T:
            arr = np.concatenate([arr, np.full(T - len(arr), arr[-1])])
        return arr[:T]

    def _shock(arr):
        if arr is None:
            return np.zeros(T)
        a = np.asarray(arr, dtype=float)
        if len(a) < T:
            a = np.concatenate([a, np.full(T - len(a), a[-1])])
        return a[:T] * shock_scale

    psi = _get_psi(scenario, T)

    # Tax paths
    tau_l   = _base('tau_l_path')
    tau_c   = _base('tau_c_path')
    tau_k   = _base('tau_k_path')
    tau_p   = _base('tau_p_path')
    pension = _base('pension_replacement_path')
    G       = _base('G_path')
    I_g     = _base('I_g_path')
    defense = _base('defense_spending_path')
    other   = _base('other_net_spending_path')

    cf = {}

    # Apply exogenous shocks
    cf['tau_l']   = (tau_l   if tau_l   is not None else np.zeros(T)) + _shock(scenario.delta_tau_l_path)
    cf['tau_c']   = (tau_c   if tau_c   is not None else np.zeros(T)) + _shock(scenario.delta_tau_c_path)
    cf['tau_k']   = (tau_k   if tau_k   is not None else np.zeros(T)) + _shock(scenario.delta_tau_k_path)
    cf['tau_p']   = (tau_p   if tau_p   is not None else np.zeros(T)) + _shock(scenario.delta_tau_p_path)
    cf['pension'] = (pension if pension is not None else np.full(T, 0.4)) + _shock(scenario.delta_pension_path)

    # GDP-share spending mode: when base_paths carries *_over_Y ratios, the
    # spending lines (and their shocks) are shares of Y(t); the budget multiplies
    # by each run's realized Y_path, so levels move with output and the SS ratios
    # are preserved. delta_G_path / delta_I_g_path are then ratio deltas.
    # The I_g line is per-line: with eta_g != 0 the caller passes I_g as a level
    # path (no 'I_g_over_Y' key) since a GDP-share I_g would need an I_g↔K_g↔Y
    # fixed point; delta_I_g_path is then a level delta while the other lines
    # stay ratios.
    ratio_mode = any(k in base_paths for k in
                     ('G_over_Y', 'I_g_over_Y', 'defense_over_Y', 'other_net_over_Y'))
    if ratio_mode:
        def _rbase(key):
            v = base_paths.get(key)
            if v is None:
                return np.zeros(T)
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            if len(arr) < T:
                arr = np.concatenate([arr, np.full(T - len(arr), arr[-1])])
            return arr[:T]
        cf['G_over_Y']         = _rbase('G_over_Y')   + _shock(scenario.delta_G_path)
        if 'I_g_over_Y' in base_paths:
            cf['I_g_over_Y'] = _rbase('I_g_over_Y') + _shock(scenario.delta_I_g_path)
            cf['I_g_path']   = None
        else:
            cf['I_g_over_Y'] = None
            cf['I_g_path']   = (I_g if I_g is not None else np.zeros(T)) \
                               + _shock(scenario.delta_I_g_path)
        cf['defense_over_Y']   = _rbase('defense_over_Y')      # baseline line, no shock
        cf['other_net_over_Y'] = _rbase('other_net_over_Y')    # baseline closure, no shock
        cf['G_path'] = None
        cf['defense_spending_path'] = cf['other_net_spending_path'] = None
    else:
        cf['G_path']  = (G       if G       is not None else np.zeros(T)) + _shock(scenario.delta_G_path)
        cf['I_g_path'] = (I_g    if I_g     is not None else np.zeros(T)) + _shock(scenario.delta_I_g_path)
        # Defense and other-net spending are baseline closure lines: no shock; pass
        # through unchanged (None → object-level fallback / zero in the budget).
        cf['defense_spending_path']     = defense
        cf['other_net_spending_path']   = other

    # Apply financing instrument adjustment
    fin = scenario.financing
    if fin == 'tau_l':
        cf['tau_l']   = cf['tau_l']   + instrument_delta * psi
    elif fin == 'tau_c':
        cf['tau_c']   = cf['tau_c']   + instrument_delta * psi
    elif fin == 'tau_k':
        cf['tau_k']   = cf['tau_k']   + instrument_delta * psi
    elif fin == 'tau_p':
        cf['tau_p']   = cf['tau_p']   + instrument_delta * psi
    elif fin == 'pension_replacement':
        cf['pension'] = cf['pension'] + instrument_delta * psi
    elif fin == 'transfer_floor':
        # NOTE: this financing mode cannot currently be run. A positive
        # transfer_floor tops up household budgets but is not an outlay in
        # compute_government_budget, so the transfer is paid by nobody and the
        # government budget it is supposed to balance omits it. The household
        # constructor refuses a positive floor for that reason. Booking it needs
        # the simulation to carry the per-agent transfer so it can be aggregated.
        cf['transfer_floor_delta'] = instrument_delta  # scalar uniform shift
    elif fin == 'debt':
        pass  # no adjustment to any instrument
    else:
        raise ValueError(f"Unknown financing instrument: {fin!r}")

    return cf


def _run_one_simulation(olg, base_paths: dict, cf: dict,
                        n_sim: int, verbose: bool,
                        recompute_bequests: bool,
                        pre_transition_paths: Optional[dict] = None) -> tuple:
    """Run simulate_transition + compute_government_budget_path for cf paths.

    Returns (macro_result, budget_path).
    """
    r_path = np.asarray(base_paths['r_path'], dtype=float)

    tf_delta = cf.get('transfer_floor_delta', 0.0)
    orig_tf  = getattr(olg.lifecycle_config, 'transfer_floor', 0.0)
    transfer_floor = orig_tf + float(tf_delta) if tf_delta != 0.0 else None

    macro = olg.simulate_transition(
        r_path=r_path,
        tau_c_path=cf.get('tau_c'),
        tau_l_path=cf.get('tau_l'),
        tau_p_path=cf.get('tau_p'),
        tau_k_path=cf.get('tau_k'),
        pension_replacement_path=cf.get('pension'),
        I_g_path=cf.get('I_g_path'),
        govt_spending_path=cf.get('G_path'),
        defense_spending_path=cf.get('defense_spending_path'),
        other_net_spending_path=cf.get('other_net_spending_path'),
        G_over_Y=cf.get('G_over_Y'),
        I_g_over_Y=cf.get('I_g_over_Y'),
        defense_over_Y=cf.get('defense_over_Y'),
        other_net_over_Y=cf.get('other_net_over_Y'),
        transfer_floor=transfer_floor,
        n_sim=n_sim,
        verbose=verbose,
        recompute_bequests=recompute_bequests,
        pre_transition_paths=pre_transition_paths,
    )
    budget = olg.compute_government_budget_path(n_sim=n_sim, verbose=verbose)
    return macro, budget


def _nfa_ca_paths(macro: dict, growth_factor=1.0) -> tuple:
    """Return (NFA_path, CA_path) from macro dict; both None if NFA absent.

    In per-capita detrended units CA_t = Gamma_t * NFA_{t+1} - NFA_t; the last
    period repeats the terminal stock, which on a balanced path gives
    CA = (Gamma_T - 1) * NFA[-1].  A scalar 1.0 reproduces the no-growth
    difference exactly.
    """
    NFA = macro.get('NFA')
    if NFA is None:
        return None, None
    NFA = np.asarray(NFA, dtype=float)
    G = _growth_seq(growth_factor, len(NFA))
    CA = np.empty_like(NFA)
    CA[:-1] = G[:-1] * NFA[1:] - NFA[:-1]
    CA[-1] = (G[-1] - 1.0) * NFA[-1]
    return NFA, CA


def _correct_base_macro_nfa(base_macro: dict, base_budget: dict,
                            scenario: FiscalScenario,
                            r_B_path: np.ndarray, T_total: int,
                            growth_factor=1.0) -> dict:
    """Return a copy of base_macro with NFA converted from the partial
    (A - K_domestic) that simulate_transition returns to the full
    NFA = A - K_domestic - B_base.

    Mirrors the cf_macro correction so that a result's base_macro and cf_macro
    NFA are on the same (full) definition.  No-op (returns the input) when NFA
    is absent.  Does not mutate the input dict.
    """
    if base_macro.get('NFA') is None:
        return base_macro
    B_base = compute_debt_path(base_budget['primary_deficit'], r_B_path,
                               B_initial=scenario.B_initial,
                               growth_factor=growth_factor)
    out = dict(base_macro)
    out['NFA'] = np.asarray(base_macro['NFA']) - B_base[:T_total]
    return out


def _check_nfa_violation(NFA: Optional[np.ndarray],
                         CA: Optional[np.ndarray],
                         scenario: FiscalScenario,
                         NFA_base: Optional[np.ndarray] = None,
                         CA_base: Optional[np.ndarray] = None) -> bool:
    """True if any NFA or CA constraint is violated.

    When a baseline path is supplied, the floor tracks it as a band:
    NFA_t >= NFA_base_t - nfa_limit (and likewise for CA).  With NFA_base=None
    the floor is the absolute level -nfa_limit (original behaviour).
    nfa_limit/ca_limit are the band half-width; 0.0 enforces exact tracking.
    """
    if NFA is not None and scenario.nfa_limit is not None:
        floor = (-scenario.nfa_limit if NFA_base is None
                 else np.asarray(NFA_base) - scenario.nfa_limit)
        if np.any(np.asarray(NFA) < floor):
            return True
    if CA is not None and scenario.ca_limit is not None:
        floor = (-scenario.ca_limit if CA_base is None
                 else np.asarray(CA_base) - scenario.ca_limit)
        if np.any(np.asarray(CA) < floor):
            return True
    return False


# ---------------------------------------------------------------------------
# Type A — Debt-financed (no iteration)
# ---------------------------------------------------------------------------

def run_debt_financed(olg, scenario: FiscalScenario, base_paths: dict,
                      base_macro: dict, base_budget: dict,
                      n_sim: int = 500, verbose: bool = False) -> FiscalScenarioResult:
    """Debt-financed experiment: one simulate_transition() call.

    The entire shock is applied.  Debt accumulates as the accounting residual.
    When scenario.n_post > 0, all paths are extended by n_post periods (terminal
    values frozen) so the simulation shows post-target dynamics.
    """

    T_base  = len(np.asarray(base_paths['r_path'], dtype=float))  # original unextended T
    n_post  = scenario.n_post
    T_total = T_base + n_post
    # Per-capita detrended stocks lose Gamma_t = (1+g)(1+n_t) per period.
    G_growth = _olg_growth(olg, T_total)

    ext_paths = _extend_base_paths(base_paths, n_post)
    r_B_path = np.asarray(ext_paths['r_B_path'], dtype=float)
    pre_tp = base_paths.get('_pre_transition_paths') or _build_pre_transition_paths(olg, base_paths)

    cf = _apply_shock(scenario, ext_paths, T_total, instrument_delta=0.0, shock_scale=1.0)
    cf_macro, cf_budget = _run_one_simulation(
        olg, ext_paths, cf, n_sim=n_sim, verbose=verbose,
        recompute_bequests=scenario.recompute_bequests,
        pre_transition_paths=pre_tp,
    )

    B_path = compute_debt_path(
        cf_budget['primary_deficit'], r_B_path, B_initial=scenario.B_initial,
        growth_factor=G_growth
    )
    Y_path = np.asarray(cf_macro['Y'], dtype=float)
    B_gdp  = B_path[:-1] / Y_path  # length T_total

    # Correct NFA: NFA = A - K_domestic - B  (simulate_transition returns A - K_domestic)
    if cf_macro.get('NFA') is not None:
        cf_macro = dict(cf_macro)
        cf_macro['NFA'] = np.asarray(cf_macro['NFA']) - B_path[:T_total]
    NFA, CA = _nfa_ca_paths(cf_macro, G_growth)
    # Same correction on the baseline so base_macro/cf_macro NFA are comparable.
    base_macro = _correct_base_macro_nfa(base_macro, base_budget, scenario,
                                         r_B_path, T_total, G_growth)

    t_drift, t_conv = _check_terminal_convergence(cf_macro, cf_budget, olg, B_path=B_path)
    t_balance = T_base if n_post > 0 else None

    return FiscalScenarioResult(
        scenario=scenario,
        base_macro=base_macro,
        cf_macro=cf_macro,
        base_budget=base_budget,
        cf_budget=cf_budget,
        B_path=B_path,
        B_gdp_path=np.append(B_gdp, B_path[-1] / Y_path[-1]),
        NFA_path=NFA,
        CA_path=CA,
        adjustment_scalar=0.0,
        adjustment_path=np.zeros(T_total),
        adjustment_label='none (debt-financed)',
        converged=True,
        n_iterations=1,
        residual_history=[],
        terminal_drift=t_drift,
        terminal_converged=t_conv,
        T_balance=t_balance,
    )


# ---------------------------------------------------------------------------
# Type B — Tax/transfer-financed (scalar bisection)
# ---------------------------------------------------------------------------

def run_tax_financed(olg, scenario: FiscalScenario, base_paths: dict,
                     base_macro: dict, base_budget: dict,
                     n_sim: int = 500, verbose: bool = False,
                     tol: float = 1e-4,
                     max_iter: int = 40,
                     Delta_lo: float = -0.5,
                     Delta_hi: float = 0.5,
                     Delta_init: float = 0.0,
                     Delta_step: float = 0.02) -> FiscalScenarioResult:
    """Tax/transfer-financed experiment via scalar bisection.

    Finds the scalar Δ such that:
      instrument_path_cf = instrument_path_base
                          + shock_path (from scenario)
                          + Δ * psi_t

    satisfies the balance_condition.

    Parameters
    ----------
    Delta_lo, Delta_hi : float
        Bounds of the search.  They are the bracket only if stepping from
        Delta_init finds no sign change inside them; Delta_hi is then expanded
        geometrically if needed.
    Delta_init : float
        First Δ evaluated.  0 is the debt-financed path of the same shock.
    Delta_step : float
        Size of the first step away from Delta_init.
    """

    T_base  = len(np.asarray(base_paths['r_path'], dtype=float))  # original unextended T
    n_post  = scenario.n_post
    T_total = T_base + n_post
    # Per-capita detrended stocks lose Gamma_t = (1+g)(1+n_t) per period.
    G_growth = _olg_growth(olg, T_total)

    ext_paths = _extend_base_paths(base_paths, n_post)
    r_B_path = np.asarray(ext_paths['r_B_path'], dtype=float)
    psi = _get_psi(scenario, T_total)
    cond = scenario.balance_condition
    residual_history = []
    pre_tp = base_paths.get('_pre_transition_paths') or _build_pre_transition_paths(olg, base_paths)

    # r_terminal is the sovereign rate at the balance-condition period (used by
    # terminal_flow_balance: PD/Y = (g − r_B)·target_debt_gdp).
    r_terminal = float(np.asarray(base_paths['r_B_path'], dtype=float)[-1])

    # Period at which the balance condition is evaluated
    T_bal = T_base if n_post > 0 else None

    def _simulate_and_residual(Delta):
        cf = _apply_shock(scenario, ext_paths, T_total,
                          instrument_delta=Delta, shock_scale=1.0)
        macro, budget = _run_one_simulation(
            olg, ext_paths, cf, n_sim=n_sim, verbose=False,
            recompute_bequests=scenario.recompute_bequests,
            pre_transition_paths=pre_tp,
        )
        # Get Y_path from last simulate_transition result (stored on olg)
        Y = np.asarray(olg.Y_path, dtype=float)
        B = compute_debt_path(budget['primary_deficit'], r_B_path,
                               B_initial=scenario.B_initial,
                               growth_factor=G_growth)
        # NFA_partial (A - K_domestic) for the 'terminal_nfa_gdp' condition.
        NFA_partial = macro.get('NFA')
        return (_balance_residual(budget, Y, B, scenario, r_terminal, T_bal, NFA_partial,
                                  growth_factor=G_growth),
                budget, B, Y)

    # Mutable container to hold the last evaluated (budget, B, Y) tuple
    _last_result = [None]  # [0] = (budget, B, Y) or None

    def _simulate_and_residual_cached(Delta):
        res, budget, B, Y = _simulate_and_residual(Delta)
        _last_result[0] = (budget, B, Y)
        return res, budget, B, Y

    # For 'period_balance', use bounded minimization
    if cond == 'period_balance':
        eval_count = [0]

        def _objective(Delta):
            eval_count[0] += 1
            res, budget, B, Y = _simulate_and_residual_cached(Delta)
            residual_history.append(res)
            return res  # minimise max_t |PD_t|

        opt = minimize_scalar(
            _objective,
            bounds=(Delta_lo, Delta_hi),
            method='bounded',
            options={'xatol': tol, 'maxiter': max_iter},
        )
        Delta_star = float(opt.x)
        converged  = opt.success
        n_iters    = eval_count[0]
        # Reuse last cached result if available; otherwise run once more
        if _last_result[0] is not None:
            cf_budget, B_path, Y_path = _last_result[0]
        else:
            _, cf_budget, B_path, Y_path = _simulate_and_residual_cached(Delta_star)

    else:
        # Bisection for 'terminal_debt_gdp' and 'pv_balance'
        converged  = False
        cf_budget = B_path = Y_path = None

        # ── Bracket the root from Delta_init ─────────────────────────────────
        # Evaluate at Delta_init, step by Delta_step in the direction that
        # lowers the residual (a higher tax or lower spending when it is
        # positive), then extrapolate by the secant until two evaluations have
        # residuals of opposite sign. Each extrapolation is at most four times
        # the previous step and stays inside [Delta_lo, Delta_hi].
        raises_deficit = scenario.financing in ('pension_replacement', 'transfer_floor')
        D_a = float(np.clip(Delta_init, Delta_lo, Delta_hi))
        res_a, last_budget, last_B, last_Y = _simulate_and_residual_cached(D_a)
        residual_history.append(res_a)
        D_b, res_b = D_a, res_a
        if abs(res_a) < tol:
            Delta_star = D_a
            cf_budget, B_path, Y_path = last_budget, last_B, last_Y
            converged = True
        else:
            direction = (1.0 if res_a > 0 else -1.0) * (-1.0 if raises_deficit else 1.0)
            D_b = float(np.clip(D_a + direction * abs(Delta_step), Delta_lo, Delta_hi))
            if D_b == D_a:
                D_b = float(np.clip(D_a - direction * abs(Delta_step), Delta_lo, Delta_hi))
            for _ in range(8):
                res_b, last_budget, last_B, last_Y = _simulate_and_residual_cached(D_b)
                residual_history.append(res_b)
                if abs(res_b) < tol:
                    Delta_star = D_b
                    cf_budget, B_path, Y_path = last_budget, last_B, last_Y
                    converged = True
                    break
                if np.sign(res_b) != np.sign(res_a):
                    break
                step = D_b - D_a
                if res_b != res_a:
                    move = float(np.clip(-res_b * step / (res_b - res_a),
                                         -4.0 * abs(step), 4.0 * abs(step)))
                else:
                    move = 2.0 * step
                D_new = float(np.clip(D_b + move, Delta_lo, Delta_hi))
                if D_new == D_b:
                    break
                D_a, res_a = D_b, res_b
                D_b = D_new

        if not converged and np.sign(res_a) != np.sign(res_b):
            if D_a < D_b:
                Delta_lo, res_lo, Delta_hi, res_hi = D_a, res_a, D_b, res_b
            else:
                Delta_lo, res_lo, Delta_hi, res_hi = D_b, res_b, D_a, res_a
        elif not converged:
            # No sign change found from Delta_init: bracket with the bounds.
            res_lo, _, _, _ = _simulate_and_residual_cached(Delta_lo)
            residual_history.append(res_lo)
            res_hi, _, _, _ = _simulate_and_residual_cached(Delta_hi)
            residual_history.append(res_hi)

            # Expand hi until we have opposite signs
            _expand = 0
            while np.sign(res_lo) == np.sign(res_hi) and _expand < 10:
                Delta_hi *= 2.0
                res_hi, _, _, _ = _simulate_and_residual_cached(Delta_hi)
                residual_history.append(res_hi)
                _expand += 1

        n_iters    = len(residual_history)
        if not converged:
            Delta_star = Delta_lo  # fallback

        # Illinois (modified regula falsi): keep the verified opposite-sign
        # bracket but step via secant interpolation instead of the midpoint,
        # ~halving the iterations vs pure bisection on a smooth residual. The
        # retained-endpoint residual is halved after two same-side moves to
        # prevent the classic regula-falsi stall. Falls back to the bisection
        # midpoint when the secant step would leave the bracket (e.g. a
        # non-bracketing interval after a failed expansion).
        # Without a sign change the interval can shrink onto a point that is
        # not a root; only a bracketing interval narrower than tol counts.
        bracketed = (not converged) and np.sign(res_lo) != np.sign(res_hi)
        _side = 0
        for _ in range(0 if converged else max_iter):
            denom = res_hi - res_lo
            if denom != 0.0:
                Delta_mid = (Delta_lo * res_hi - Delta_hi * res_lo) / denom
            else:
                Delta_mid = 0.5 * (Delta_lo + Delta_hi)
            _lo_b, _hi_b = min(Delta_lo, Delta_hi), max(Delta_lo, Delta_hi)
            if not (_lo_b < Delta_mid < _hi_b):
                Delta_mid = 0.5 * (Delta_lo + Delta_hi)
            res_mid, last_budget, last_B, last_Y = _simulate_and_residual_cached(Delta_mid)
            residual_history.append(res_mid)
            n_iters += 1
            if abs(res_mid) < tol or (bracketed and (Delta_hi - Delta_lo) < tol):
                Delta_star = Delta_mid
                cf_budget, B_path, Y_path = last_budget, last_B, last_Y
                converged  = True
                break
            if not bracketed and (Delta_hi - Delta_lo) < tol:
                Delta_star = Delta_mid
                cf_budget, B_path, Y_path = last_budget, last_B, last_Y
                break
            if np.sign(res_mid) == np.sign(res_lo):
                Delta_lo, res_lo = Delta_mid, res_mid
                if _side == +1:
                    res_hi *= 0.5
                _side = +1
            else:
                Delta_hi, res_hi = Delta_mid, res_mid
                if _side == -1:
                    res_lo *= 0.5
                _side = -1
            Delta_star = Delta_mid

        # If bisection did not converge within loop (fallback), run final simulation
        if not converged or cf_budget is None:
            _, cf_budget, B_path, Y_path = _simulate_and_residual_cached(Delta_star)

    # Reconstruct full macro from last run (already stored on olg)
    cf_macro = {
        'r': olg.r_path, 'w': olg.w_path,
        'K': olg.K_path, 'A': olg.K_path,   # K = A = household wealth
        'L': olg.L_path, 'Y': olg.Y_path, 'C': olg.C_path,
    }
    if olg.K_domestic_path is not None:
        cf_macro['K_domestic'] = olg.K_domestic_path
    if olg.K_g_path is not None:
        cf_macro['K_g'] = olg.K_g_path
    if olg.NFA_path is not None:
        cf_macro['NFA'] = olg.NFA_path

    # Correct NFA: NFA = A - K_domestic - B  (simulate_transition returns A - K_domestic)
    if cf_macro.get('NFA') is not None:
        cf_macro['NFA'] = np.asarray(cf_macro['NFA']) - B_path[:T_total]
    # Same correction on the baseline so base_macro/cf_macro NFA are comparable.
    base_macro = _correct_base_macro_nfa(base_macro, base_budget, scenario,
                                         r_B_path, T_total, G_growth)

    B_gdp = B_path[:-1] / Y_path
    NFA, CA = _nfa_ca_paths(cf_macro, G_growth)
    adj_path = Delta_star * psi

    t_drift, t_conv = _check_terminal_convergence(cf_macro, cf_budget, olg, B_path=B_path)

    return FiscalScenarioResult(
        scenario=scenario,
        base_macro=base_macro,
        cf_macro=cf_macro,
        base_budget=base_budget,
        cf_budget=cf_budget,
        B_path=B_path,
        B_gdp_path=np.append(B_gdp, B_path[-1] / Y_path[-1]),
        NFA_path=NFA,
        CA_path=CA,
        adjustment_scalar=Delta_star,
        adjustment_path=adj_path,
        adjustment_label=f'Δ{scenario.financing}',
        converged=converged,
        n_iterations=n_iters,
        residual_history=residual_history,
        terminal_drift=t_drift,
        terminal_converged=t_conv,
        T_balance=T_bal,
    )


# ---------------------------------------------------------------------------
# Type C — NFA/CA-constrained
# ---------------------------------------------------------------------------

def run_nfa_constrained(olg, scenario: FiscalScenario, base_paths: dict,
                        base_macro: dict, base_budget: dict,
                        n_sim: int = 500, verbose: bool = False,
                        tol: float = 1e-4,
                        max_iter: int = 40) -> FiscalScenarioResult:
    """NFA/CA-constrained experiment.

    Step 1: Run debt-financed with full shock; check NFA/CA constraints.
    Step 2a (no violation): return the debt-financed result.
    Step 2b-Mode I (financing='debt'): bisect on shock scale Δ_shock.
    Step 2b-Mode II (financing != 'debt'): full shock + bisect on Δτ for NFA.
    """

    T_base  = len(np.asarray(base_paths['r_path'], dtype=float))  # original unextended T
    n_post  = scenario.n_post
    T_total = T_base + n_post
    # Per-capita detrended stocks lose Gamma_t = (1+g)(1+n_t) per period.
    G_growth = _olg_growth(olg, T_total)

    ext_paths = _extend_base_paths(base_paths, n_post)
    r_B_path = np.asarray(ext_paths['r_B_path'], dtype=float)
    residual_history = []
    pre_tp = base_paths.get('_pre_transition_paths') or _build_pre_transition_paths(olg, base_paths)

    # ── Baseline NFA/CA band reference ───────────────────────────────────────
    # base_macro['NFA'] is the partial A - K_domestic; subtract the baseline debt
    # path to get full NFA, matching the per-iteration correction below.  The
    # constraint is then a band around this path (see _check_nfa_violation).
    NFA_base = CA_base = None
    if base_macro.get('NFA') is not None:
        B_base = compute_debt_path(base_budget['primary_deficit'], r_B_path,
                                   B_initial=scenario.B_initial,
                                   growth_factor=G_growth)
        _base_full = dict(base_macro)
        _base_full['NFA'] = np.asarray(base_macro['NFA']) - B_base[:T_total]
        NFA_base, CA_base = _nfa_ca_paths(_base_full, G_growth)

    def _full_nfa_ca(macro, budget):
        """Full NFA/CA (net of this iteration's debt) from raw simulation macro."""
        if macro.get('NFA') is None:
            return None, None
        B = compute_debt_path(budget['primary_deficit'], r_B_path,
                              B_initial=scenario.B_initial,
                              growth_factor=G_growth)
        m = dict(macro)
        m['NFA'] = np.asarray(macro['NFA']) - B[:T_total]
        return _nfa_ca_paths(m, G_growth)

    # ── Step 1: Debt-financed, full shock ────────────────────────────────────
    full_debt_result = run_debt_financed(
        olg, scenario, base_paths, base_macro, base_budget, n_sim=n_sim, verbose=False
    )
    NFA_full, CA_full = full_debt_result.NFA_path, full_debt_result.CA_path
    violation = _check_nfa_violation(NFA_full, CA_full, scenario, NFA_base, CA_base)

    # ── Step 2a: No violation ────────────────────────────────────────────────
    if not violation:
        return full_debt_result

    # ── Step 2b: Violation exists ────────────────────────────────────────────
    if scenario.financing == 'debt':
        # Mode I: trim the shock scale (bisect on Δ_shock ∈ [0, 1])
        label = 'shock scale Δ'

        def _feasible(shock_scale):
            cf = _apply_shock(scenario, ext_paths, T_total,
                              instrument_delta=0.0, shock_scale=shock_scale)
            macro, budget = _run_one_simulation(
                olg, ext_paths, cf, n_sim=n_sim, verbose=False,
                recompute_bequests=scenario.recompute_bequests,
                pre_transition_paths=pre_tp,
            )
            NFA, CA = _full_nfa_ca(macro, budget)
            return not _check_nfa_violation(NFA, CA, scenario, NFA_base, CA_base), macro, budget

        lo, hi = 0.0, 1.0
        converged = False
        Delta_star = 0.0
        cf_macro_star = cf_budget_star = None

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            ok, macro_mid, budget_mid = _feasible(mid)
            residual_history.append(mid)
            if hi - lo < tol:
                Delta_star = mid
                cf_macro_star, cf_budget_star = macro_mid, budget_mid
                converged = True
                break
            if ok:
                lo, Delta_star = mid, mid
                cf_macro_star, cf_budget_star = macro_mid, budget_mid
            else:
                hi = mid

        if cf_macro_star is None:
            cf_macro_star, cf_budget_star = full_debt_result.cf_macro, full_debt_result.cf_budget
        psi = _get_psi(scenario, T_total)
        adj_path = Delta_star * psi

    else:
        # Mode II: full shock + bisect on Δτ for NFA/CA feasibility
        label = f'Δ{scenario.financing} (NFA-constrained)'

        def _nfa_ok_at(Delta):
            cf = _apply_shock(scenario, ext_paths, T_total,
                              instrument_delta=Delta, shock_scale=1.0)
            macro, budget = _run_one_simulation(
                olg, ext_paths, cf, n_sim=n_sim, verbose=False,
                recompute_bequests=scenario.recompute_bequests,
                pre_transition_paths=pre_tp,
            )
            NFA, CA = _full_nfa_ca(macro, budget)
            return not _check_nfa_violation(NFA, CA, scenario, NFA_base, CA_base), macro, budget

        # Search for a Δτ that eliminates the violation
        lo, hi = 0.0, 0.5
        # Expand hi until feasible
        ok_hi, _, _ = _nfa_ok_at(hi)
        _exp = 0
        while not ok_hi and _exp < 10:
            hi *= 2.0
            ok_hi, _, _ = _nfa_ok_at(hi)
            _exp += 1

        converged = False
        Delta_star = 0.0
        cf_macro_star = cf_budget_star = None

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            ok_mid, macro_mid, budget_mid = _nfa_ok_at(mid)
            residual_history.append(mid)
            if hi - lo < tol:
                Delta_star = mid
                cf_macro_star, cf_budget_star = macro_mid, budget_mid
                converged = True
                break
            if ok_mid:
                hi = mid  # feasible at smaller adjustment
            else:
                lo, Delta_star = mid, mid
                cf_macro_star, cf_budget_star = macro_mid, budget_mid

        if cf_macro_star is None:
            cf_macro_star, cf_budget_star = full_debt_result.cf_macro, full_debt_result.cf_budget
        psi = _get_psi(scenario, T_total)
        adj_path = Delta_star * psi

    B_path = compute_debt_path(
        cf_budget_star['primary_deficit'], r_B_path, B_initial=scenario.B_initial,
        growth_factor=G_growth
    )
    Y_path = np.asarray(cf_macro_star['Y'], dtype=float)
    B_gdp  = B_path[:-1] / Y_path

    # Correct NFA: NFA = A - K_domestic - B  (simulate_transition returns A - K_domestic)
    if cf_macro_star.get('NFA') is not None:
        cf_macro_star = dict(cf_macro_star)
        cf_macro_star['NFA'] = np.asarray(cf_macro_star['NFA']) - B_path[:T_total]
    NFA, CA = _nfa_ca_paths(cf_macro_star, G_growth)
    # Same correction on the baseline so base_macro/cf_macro NFA are comparable.
    base_macro = _correct_base_macro_nfa(base_macro, base_budget, scenario,
                                         r_B_path, T_total, G_growth)

    t_drift, t_conv = _check_terminal_convergence(cf_macro_star, cf_budget_star, olg, B_path=B_path)
    t_balance = T_base if n_post > 0 else None

    return FiscalScenarioResult(
        scenario=scenario,
        base_macro=base_macro,
        cf_macro=cf_macro_star,
        base_budget=base_budget,
        cf_budget=cf_budget_star,
        B_path=B_path,
        B_gdp_path=np.append(B_gdp, B_path[-1] / Y_path[-1]),
        NFA_path=NFA,
        CA_path=CA,
        adjustment_scalar=Delta_star,
        adjustment_path=adj_path,
        adjustment_label=label,
        converged=converged,
        n_iterations=len(residual_history),
        residual_history=residual_history,
        terminal_drift=t_drift,
        terminal_converged=t_conv,
        T_balance=t_balance,
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def _with_default_paths(olg, base_paths: dict) -> dict:
    """Shallow copy of base_paths with G_path / I_g_path / r_B_path filled in."""
    base_paths = dict(base_paths)  # shallow copy; don't mutate caller's dict
    if 'G_path' not in base_paths:
        base_paths['G_path'] = olg.govt_spending_path
    if 'I_g_path' not in base_paths:
        base_paths['I_g_path'] = olg.I_g_path

    # Sovereign rate path: broadcast scalar r_B, else fall back to capital r_path.
    # Used by compute_debt_path (B law of motion) and by _balance_residual's
    # terminal_flow_balance condition.  Adding it here lets all three financing
    # branches (debt / tax / NFA) pick it up uniformly.
    if 'r_B_path' not in base_paths:
        _r_path_arr = np.asarray(base_paths['r_path'], dtype=float)
        base_paths['r_B_path'] = (np.full(len(_r_path_arr), float(olg.r_B))
                                  if olg.r_B is not None else _r_path_arr)
    return base_paths


def _baseline_settings(n_post, n_sim, recompute_bequests) -> dict:
    return dict(n_post=int(n_post), n_sim=int(n_sim),
                recompute_bequests=bool(recompute_bequests))


def run_baseline(olg, base_paths: dict, n_post: int = 0, n_sim: int = 500,
                 recompute_bequests: bool = False, verbose: bool = False) -> dict:
    """Run the no-shock baseline once and attach its results to a copy of base_paths.

    The returned dict carries 'base_macro', 'base_budget', 'w_path' and
    '_pre_transition_paths'. Passing it to run_fiscal_scenario() for scenarios
    with the same n_post, n_sim and recompute_bequests reuses this baseline
    instead of re-running it for every scenario.
    """
    base_paths = _with_default_paths(olg, base_paths)
    if '_pre_transition_paths' in base_paths:
        pre_tp = base_paths['_pre_transition_paths']
    else:
        pre_tp = _build_pre_transition_paths(olg, base_paths)
        base_paths['_pre_transition_paths'] = pre_tp

    # Extend baseline paths if the scenarios request post-target dynamics.
    ext_base = _extend_base_paths(base_paths, n_post)
    T = len(np.asarray(ext_base['r_path']))
    base_cf = _apply_shock(
        FiscalScenario(name='_baseline'),  # zero shock
        ext_base, T, instrument_delta=0.0, shock_scale=0.0,
    )
    # Baseline simulation: no shock, so MIT stitching is a no-op (counterfactual
    # paths = baseline paths). Skip pre_transition_paths to avoid 177 redundant
    # NumPy baseline solves. The solved models are then cached for counterfactual runs.
    base_macro, base_budget = _run_one_simulation(
        olg, ext_base, base_cf, n_sim=n_sim, verbose=verbose,
        recompute_bequests=recompute_bequests,
        pre_transition_paths=None,
    )
    # Capture baseline wage path so that MIT-shock stitching in counterfactual
    # runs uses the correct (baseline) wages, not the counterfactual ones.
    # Critical when I_g shocks change K_g → w.
    # Store only the first T_TR periods (before any post-target extension) so
    # that _extend_base_paths can safely extend it later.
    T_orig = len(np.asarray(base_paths['r_path']))
    base_paths['w_path'] = np.array(olg.w_path[:T_orig])

    # Pre-populate the MIT baseline cache from the baseline run's solutions.
    # Counterfactual runs need baseline policy functions for stitching
    # pre-transition ages. The baseline run just solved these exact models —
    # reuse them instead of re-solving 177 NumPy models sequentially.
    if olg.birth_cohort_solutions is not None:
        for edu_type, models_dict in olg.birth_cohort_solutions.items():
            for bp, model in models_dict.items():
                if bp < 0:
                    olg._mit_baseline_cache[(edu_type, bp)] = model
        # Set the pre_tp id so the cache is not invalidated
        olg._mit_pre_tp_id = id(pre_tp)

    base_paths['base_macro'] = base_macro
    base_paths['base_budget'] = base_budget
    base_paths['_baseline_settings'] = _baseline_settings(n_post, n_sim, recompute_bequests)
    return base_paths


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_fiscal_scenario(olg, scenario: FiscalScenario, base_paths: dict,
                        n_sim: int = 500, verbose: bool = False,
                        bisect_tol: float = 1e-4,
                        bisect_max_iter: int = 40,
                        bisect_init: float = 0.0,
                        bisect_step: float = 0.02) -> FiscalScenarioResult:
    """Run a fiscal scenario against a baseline.

    Parameters
    ----------
    olg : OLGTransition
        A fully-configured OLG model (backend, lifecycle_config, etc.).
        Its object-level attributes (govt_spending_path, I_g_path) are
        temporarily mutated during counterfactual simulations and then restored.
    scenario : FiscalScenario
    base_paths : dict
        Must contain 'r_path'.  Should contain the same policy paths that were
        used in the baseline simulation: 'tau_l_path', 'tau_c_path', 'tau_p_path',
        'tau_k_path', 'pension_replacement_path'.
        If 'G_path' / 'I_g_path' are omitted, the current olg.govt_spending_path /
        olg.I_g_path values are used.
        Optionally contains 'base_macro' and 'base_budget' (pre-computed output
        of simulate_transition and compute_government_budget_path).  If absent,
        the baseline simulation is run once here.  Pass the dict returned by
        run_baseline() to share one baseline run across scenarios.
    n_sim : int
        Number of Monte Carlo agents.
    verbose : bool
    bisect_tol, bisect_max_iter : convergence settings for tax-financed runs.
    bisect_init, bisect_step : first Δ evaluated by a tax-financed run and the
        size of its first step (see run_tax_financed).

    Returns
    -------
    FiscalScenarioResult
    """
    # ── Run (or retrieve) baseline ───────────────────────────────────────────
    base_paths = _with_default_paths(olg, base_paths)
    settings = _baseline_settings(scenario.n_post, n_sim, scenario.recompute_bequests)
    have_baseline = ('base_macro' in base_paths and 'base_budget' in base_paths
                     # a baseline from run_baseline() under other settings is not this one
                     and base_paths.get('_baseline_settings', settings) == settings)
    if have_baseline:
        # Reuse pre_transition_paths if already in base_paths (avoids cache
        # invalidation across scenarios sharing the same baseline).
        if '_pre_transition_paths' not in base_paths:
            base_paths['_pre_transition_paths'] = _build_pre_transition_paths(olg, base_paths)
    else:
        if verbose:
            print(f"[run_fiscal_scenario] Running baseline for scenario '{scenario.name}'...")
        base_paths = run_baseline(olg, base_paths, n_post=scenario.n_post, n_sim=n_sim,
                                  recompute_bequests=scenario.recompute_bequests,
                                  verbose=verbose)
    base_macro  = base_paths['base_macro']
    base_budget = base_paths['base_budget']

    # ── Dispatch ─────────────────────────────────────────────────────────────
    kwargs = dict(n_sim=n_sim, verbose=verbose)

    nfa_constrained = (scenario.nfa_limit is not None or scenario.ca_limit is not None)

    if nfa_constrained:
        return run_nfa_constrained(olg, scenario, base_paths,
                                   base_macro, base_budget,
                                   tol=bisect_tol, max_iter=bisect_max_iter,
                                   **kwargs)
    elif scenario.financing == 'debt':
        return run_debt_financed(olg, scenario, base_paths,
                                 base_macro, base_budget, **kwargs)
    else:
        return run_tax_financed(olg, scenario, base_paths,
                                base_macro, base_budget,
                                tol=bisect_tol, max_iter=bisect_max_iter,
                                Delta_init=bisect_init, Delta_step=bisect_step,
                                **kwargs)


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def compare_scenarios(
    base: FiscalScenarioResult,
    *counterfactuals: FiscalScenarioResult,
    variables: Optional[list] = None,
    var_labels: Optional[dict] = None,
    title: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
    output_dir: str = 'output',
    T_balance: Optional[int] = None,
) -> plt.Figure:
    """Side-by-side line plots comparing base vs. counterfactual paths.

    Parameters
    ----------
    variables : list of str, optional
        Keys to plot.  Each key is looked up in cf_macro, cf_budget, B_gdp_path,
        NFA_path.  Defaults to ['Y', 'primary_deficit', 'B_gdp_path', 'NFA'].
    var_labels : dict, optional
        Mapping from variable key to display label, e.g. {'B_gdp_path': 'B/Y'}.
        Keys not present fall back to the raw key string.
    title : str, optional
        Figure suptitle.  Defaults to 'Fiscal Scenario Comparison'.
    """
    import os
    if variables is None:
        variables = ['Y', 'primary_deficit', 'B_gdp_path', 'NFA']
    if var_labels is None:
        var_labels = {}

    n_vars = len(variables)
    ncols  = min(3, n_vars)
    nrows  = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    all_axes  = axes.flat

    T = len(base.cf_macro['Y'])
    periods = np.arange(T)

    # Auto-detect T_balance from counterfactuals if not supplied
    if T_balance is None:
        for r in counterfactuals:
            if r.T_balance is not None:
                T_balance = r.T_balance
                break

    # Colour cycle
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_results = [base] + list(counterfactuals)
    labels      = [r.scenario.name for r in all_results]

    def _get_series(result: FiscalScenarioResult, key: str) -> Optional[np.ndarray]:
        if key == 'B_gdp_path':
            arr = result.B_gdp_path
            return arr[:T] if arr is not None else None
        if key == 'NFA_gdp':
            if result.NFA_path is None:
                return None
            Y = np.asarray(result.cf_macro['Y'], dtype=float)
            n = min(len(result.NFA_path), len(Y))
            return np.asarray(result.NFA_path[:n]) / Y[:n]
        if key.endswith('_gdp'):
            # Generic "<base>/Y" ratio for any macro or budget line (e.g.
            # A_gdp, primary_deficit_gdp, tax_l_gdp, pension_gdp, ui_gdp).
            base_key = key[:-len('_gdp')]
            src = (result.cf_macro.get(base_key)
                   if base_key in result.cf_macro else result.cf_budget.get(base_key))
            if src is None:
                return None
            src = np.asarray(src, dtype=float)
            Y = np.asarray(result.cf_macro['Y'], dtype=float)
            n = min(len(src), len(Y))
            return src[:n] / Y[:n]
        if key in result.cf_macro:
            return np.asarray(result.cf_macro[key])
        if key in result.cf_budget:
            return np.asarray(result.cf_budget[key])
        if key == 'NFA' and result.NFA_path is not None:
            return result.NFA_path
        if key == 'CA' and result.CA_path is not None:
            return result.CA_path
        return None

    for ax, key in zip(all_axes, variables):
        for i, (result, label) in enumerate(zip(all_results, labels)):
            series = _get_series(result, key)
            if series is None:
                continue
            ax.plot(periods, series[:T], label=label,
                    color=colours[i % len(colours)], linewidth=1.8,
                    marker='o' if i == 0 else None, markersize=3)
        ax.set_title(var_labels.get(key, key), fontweight='bold')
        ax.set_xlabel('Period')
        ax.set_xticks(periods[::5])
        if T_balance is not None:
            ax.axvline(T_balance, color='grey', linewidth=1.0, linestyle='--',
                       label=f'T_balance={T_balance}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _nice_ylim(ax)

    # Hide unused axes
    for ax in list(all_axes)[n_vars:]:
        ax.set_visible(False)

    plt.suptitle(title or 'Fiscal Scenario Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            names = '_vs_'.join(labels)[:60]
            filename = f'fiscal_comparison_{names}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        if not False:  # always print path
            print(f"Comparison plot saved to: {filepath}")

    return fig


def fiscal_multiplier(
    base: FiscalScenarioResult,
    cf: FiscalScenarioResult,
    shock_variable: str = 'G',
    output_variable: str = 'Y',
    growth_factors=None,
) -> np.ndarray:
    """Compute per-period ΔOutput_t / ΔShock_t.

    *growth_factors* is Gamma_t, scalar or path. Supply it for the cumulative
    statistic to be a ratio of aggregate flows; without it the paths are summed
    untrended, which over-weights early periods.

    Returns array of length T_transition.  Where ΔShock_t = 0 the multiplier
    is set to NaN.

    The cumulative multiplier Σ_t ΔY_t / Σ_t ΔG_t is printed.
    """
    def _get(result: FiscalScenarioResult, key: str) -> np.ndarray:
        if key in result.cf_macro:
            return np.asarray(result.cf_macro[key], dtype=float)
        if key in result.cf_budget:
            return np.asarray(result.cf_budget[key], dtype=float)
        raise KeyError(f"'{key}' not found in macro or budget paths")

    Y_base = _get(base, output_variable)
    Y_cf   = _get(cf,   output_variable)
    delta_Y = Y_cf - Y_base

    G_base = _get(base, shock_variable)
    G_cf   = _get(cf,   shock_variable)
    delta_G = G_cf - G_base

    with np.errstate(divide='ignore', invalid='ignore'):
        mult = np.where(delta_G != 0.0, delta_Y / delta_G, np.nan)

    # Cumulative multiplier. The paths are per-capita detrended, so summing them
    # directly would weight early periods far too heavily: an aggregate flow at
    # date t carries prod_{s<t} Gamma_s, which reaches about 9.6 over a 200-period
    # horizon. Re-trend before summing so the statistic is a ratio of aggregate
    # cumulative flows, which is what a cumulative multiplier means.
    gam = growth_factors
    if gam is None:
        scale = np.ones(len(delta_Y))
    else:
        g_seq = _growth_seq(gam, len(delta_Y))
        scale = np.concatenate([[1.0], np.cumprod(g_seq[:-1])])
    total_dG = np.sum(scale * delta_G)
    if total_dG != 0.0:
        cum = np.sum(scale * delta_Y) / total_dG
        flat = (np.sum(delta_Y) / np.sum(delta_G)
                if np.sum(delta_G) != 0.0 else float('nan'))
        print(f"Cumulative multiplier Σ(Δ{output_variable}) / Σ(Δ{shock_variable}) "
              f"= {cum:.4f}   (untrended, as reported before 2026-10-01: {flat:.4f})")

    return mult


def debt_fan_chart(
    scenarios: list,
    labels: list,
    save: bool = True,
    filename: Optional[str] = None,
    output_dir: str = 'output',
) -> plt.Figure:
    """Plot B/Y paths for multiple scenarios — standard DSA output."""
    import os
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (result, label) in enumerate(zip(scenarios, labels)):
        T = len(result.cf_macro['Y'])
        B_gdp = result.B_gdp_path[:T]
        ax.plot(np.arange(T), B_gdp * 100.0,
                label=label, color=colours[i % len(colours)], linewidth=2,
                marker='o' if i == 0 else None, markersize=3)

    ax.axhline(y=60.0, color='black', linestyle='--', alpha=0.4, label='60% reference')
    ax.set_xlabel('Period')
    ax.set_ylabel('Debt / GDP (%)')
    ax.set_title('Debt Sustainability Analysis — Fan Chart', fontweight='bold')
    ax.set_xticks(np.arange(0, T, 5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = 'debt_fan_chart.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Debt fan chart saved to: {filepath}")

    return fig
