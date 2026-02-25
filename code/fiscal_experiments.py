"""
Fiscal Scenario Framework for OLG Transition Model
====================================================

Computes budget-balanced (or debt-financed) transition paths under exogenous
policy shocks.  Three experiment types:

  Type A — Debt-financed    : one simulate_transition() call; B_path is the residual.
  Type B — Tax/transfer-financed : bisect on a scalar Δτ until the chosen balance
             condition is satisfied (~10–15 simulate calls).
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

import copy
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize_scalar


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
    # 'terminal_debt_gdp' : B_T / Y_T = target_debt_gdp
    # 'pv_balance'        : Σ_t δ^t * PrimaryDeficit_t = 0
    # 'period_balance'    : minimise max_t |PrimaryDeficit_t| (bounded scalar min)
    balance_condition: str = 'terminal_debt_gdp'
    target_debt_gdp: float = 0.0
    discount_rate: float = 0.04        # used only by 'pv_balance'

    # ── Initial debt ─────────────────────────────────────────────────────────
    B_initial: float = 0.0

    # ── External constraint (optional) ───────────────────────────────────────
    nfa_limit: Optional[float] = None  # per-period NFA floor: NFA_t >= -nfa_limit
    ca_limit:  Optional[float] = None  # per-period CA floor: CA_t >= -ca_limit

    # ── Bequest handling ─────────────────────────────────────────────────────
    recompute_bequests: bool = False


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
# Debt accumulation utility
# ---------------------------------------------------------------------------

def compute_debt_path(primary_deficit_path: np.ndarray,
                      r_path: np.ndarray,
                      B_initial: float = 0.0) -> np.ndarray:
    """Forward-recursive debt accumulation.

    B[0] = B_initial
    B[t+1] = (1 + r[t]) * B[t] + PrimaryDeficit[t]

    Returns B_path of length T_transition + 1.
    """
    T = len(primary_deficit_path)
    B = np.zeros(T + 1)
    B[0] = B_initial
    for t in range(T):
        B[t + 1] = (1.0 + float(r_path[t])) * B[t] + float(primary_deficit_path[t])
    return B


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _balance_residual(budget_path: dict,
                      Y_path: np.ndarray,
                      B_path: np.ndarray,
                      scenario: FiscalScenario) -> float:
    """Compute the scalar balance residual for the chosen balance_condition.

    Returns a value whose *sign* tells bisection which direction to adjust:
      > 0  → too much deficit  (raise taxes / cut spending more)
      < 0  → surplus           (cut taxes / raise spending)
    For 'period_balance' returns max_t |PD_t| (always >= 0, use minimization).
    """
    PD = budget_path['primary_deficit']
    T = len(PD)
    cond = scenario.balance_condition

    if cond == 'terminal_debt_gdp':
        # B_path has length T+1; index T is the end-of-horizon level
        B_T = B_path[T]
        Y_T = float(Y_path[T - 1])
        residual = B_T / Y_T - scenario.target_debt_gdp
        return float(residual)

    elif cond == 'pv_balance':
        delta = scenario.discount_rate
        t_arr = np.arange(T)
        weights = (1.0 / (1.0 + delta)) ** t_arr
        return float(np.dot(weights, PD))

    elif cond == 'period_balance':
        return float(np.max(np.abs(PD)))

    else:
        raise ValueError(f"Unknown balance_condition: {scenario.balance_condition!r}")


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
        return np.array(v, dtype=float)

    def _shock(arr):
        if arr is None:
            return np.zeros(T)
        return np.asarray(arr, dtype=float) * shock_scale

    psi = _get_psi(scenario, T)

    # Tax paths
    tau_l   = _base('tau_l_path')
    tau_c   = _base('tau_c_path')
    tau_k   = _base('tau_k_path')
    tau_p   = _base('tau_p_path')
    pension = _base('pension_replacement_path')
    G       = _base('G_path')
    I_g     = _base('I_g_path')

    cf = {}

    # Apply exogenous shocks
    cf['tau_l']   = (tau_l   if tau_l   is not None else np.zeros(T)) + _shock(scenario.delta_tau_l_path)
    cf['tau_c']   = (tau_c   if tau_c   is not None else np.zeros(T)) + _shock(scenario.delta_tau_c_path)
    cf['tau_k']   = (tau_k   if tau_k   is not None else np.zeros(T)) + _shock(scenario.delta_tau_k_path)
    cf['tau_p']   = (tau_p   if tau_p   is not None else np.zeros(T)) + _shock(scenario.delta_tau_p_path)
    cf['pension'] = (pension if pension is not None else np.full(T, 0.4)) + _shock(scenario.delta_pension_path)
    cf['G_path']  = (G       if G       is not None else np.zeros(T)) + _shock(scenario.delta_G_path)
    cf['I_g_path'] = (I_g    if I_g     is not None else np.zeros(T)) + _shock(scenario.delta_I_g_path)

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
        cf['transfer_floor_delta'] = instrument_delta  # scalar uniform shift
    elif fin == 'debt':
        pass  # no adjustment to any instrument
    else:
        raise ValueError(f"Unknown financing instrument: {fin!r}")

    return cf


def _run_one_simulation(olg, base_paths: dict, cf: dict,
                        n_sim: int, verbose: bool,
                        recompute_bequests: bool) -> tuple:
    """Run simulate_transition + compute_government_budget_path for cf paths.

    Temporarily mutates olg.govt_spending_path and olg.I_g_path to apply
    G / I_g shocks, restoring them with try/finally.

    Returns (macro_result, budget_path).
    """
    # Save original object-level paths that can't be passed via simulate_transition
    orig_G   = olg.govt_spending_path
    orig_I_g = olg.I_g_path
    orig_tf  = getattr(olg.lifecycle_config, 'transfer_floor', 0.0)

    try:
        # Apply G and I_g shocks by temporarily mutating object state
        G_cf   = cf.get('G_path')
        I_g_cf = cf.get('I_g_path')

        if G_cf is not None:
            olg.govt_spending_path = np.asarray(G_cf, dtype=float)
        if I_g_cf is not None:
            olg.I_g_path = np.asarray(I_g_cf, dtype=float)

        # Apply transfer_floor adjustment (scalar)
        tf_delta = cf.get('transfer_floor_delta', 0.0)
        if tf_delta != 0.0:
            olg.lifecycle_config.transfer_floor = orig_tf + float(tf_delta)

        r_path = np.asarray(base_paths['r_path'], dtype=float)

        macro = olg.simulate_transition(
            r_path=r_path,
            tau_c_path=cf.get('tau_c'),
            tau_l_path=cf.get('tau_l'),
            tau_p_path=cf.get('tau_p'),
            tau_k_path=cf.get('tau_k'),
            pension_replacement_path=cf.get('pension'),
            n_sim=n_sim,
            verbose=verbose,
            recompute_bequests=recompute_bequests,
        )
        budget = olg.compute_government_budget_path(n_sim=n_sim, verbose=verbose)

    finally:
        olg.govt_spending_path = orig_G
        olg.I_g_path           = orig_I_g
        olg.lifecycle_config.transfer_floor = orig_tf

    return macro, budget


def _nfa_ca_paths(macro: dict) -> tuple:
    """Return (NFA_path, CA_path) from macro dict; both None if NFA absent."""
    NFA = macro.get('NFA')
    if NFA is None:
        return None, None
    NFA = np.asarray(NFA)
    # CA_t = NFA_{t+1} - NFA_t; last period repeats last value
    CA = np.diff(NFA, append=NFA[-1])
    return NFA, CA


def _check_nfa_violation(NFA: Optional[np.ndarray],
                         CA: Optional[np.ndarray],
                         scenario: FiscalScenario) -> bool:
    """True if any NFA or CA constraint is violated."""
    if NFA is not None and scenario.nfa_limit is not None:
        if np.any(NFA < -scenario.nfa_limit):
            return True
    if CA is not None and scenario.ca_limit is not None:
        if np.any(CA < -scenario.ca_limit):
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
    """
    T = int(olg.T_transition)
    r_path = np.asarray(base_paths['r_path'], dtype=float)

    cf = _apply_shock(scenario, base_paths, T, instrument_delta=0.0, shock_scale=1.0)
    cf_macro, cf_budget = _run_one_simulation(
        olg, base_paths, cf, n_sim=n_sim, verbose=verbose,
        recompute_bequests=scenario.recompute_bequests,
    )

    B_path = compute_debt_path(
        cf_budget['primary_deficit'], r_path, B_initial=scenario.B_initial
    )
    Y_path = np.asarray(cf_macro['Y'], dtype=float)
    B_gdp  = B_path[:-1] / Y_path  # length T

    NFA, CA = _nfa_ca_paths(cf_macro)

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
        adjustment_path=np.zeros(T),
        adjustment_label='none (debt-financed)',
        converged=True,
        n_iterations=1,
        residual_history=[],
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
                     Delta_hi: float = 0.5) -> FiscalScenarioResult:
    """Tax/transfer-financed experiment via scalar bisection.

    Finds the scalar Δ such that:
      instrument_path_cf = instrument_path_base
                          + shock_path (from scenario)
                          + Δ * psi_t

    satisfies the balance_condition.

    Parameters
    ----------
    Delta_lo, Delta_hi : float
        Initial bisection bracket.  Expanded geometrically if needed.
    """
    T = int(olg.T_transition)
    r_path = np.asarray(base_paths['r_path'], dtype=float)
    psi = _get_psi(scenario, T)
    cond = scenario.balance_condition
    residual_history = []

    def _simulate_and_residual(Delta):
        cf = _apply_shock(scenario, base_paths, T,
                          instrument_delta=Delta, shock_scale=1.0)
        _, budget = _run_one_simulation(
            olg, base_paths, cf, n_sim=n_sim, verbose=False,
            recompute_bequests=scenario.recompute_bequests,
        )
        # Get Y_path from last simulate_transition result (stored on olg)
        Y = np.asarray(olg.Y_path, dtype=float)
        B = compute_debt_path(budget['primary_deficit'], r_path,
                               B_initial=scenario.B_initial)
        return _balance_residual(budget, Y, B, scenario), budget, B, Y

    # For 'period_balance', use bounded minimization
    if cond == 'period_balance':
        eval_count = [0]

        def _objective(Delta):
            eval_count[0] += 1
            res, budget, B, Y = _simulate_and_residual(Delta)
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
        _, cf_budget, B_path, Y_path = _simulate_and_residual(Delta_star)

    else:
        # Bisection for 'terminal_debt_gdp' and 'pv_balance'
        # ── Verify / expand bracket ──────────────────────────────────────────
        res_lo, _, _, _ = _simulate_and_residual(Delta_lo)
        residual_history.append(res_lo)
        res_hi, _, _, _ = _simulate_and_residual(Delta_hi)
        residual_history.append(res_hi)

        # Expand hi until we have opposite signs
        _expand = 0
        while np.sign(res_lo) == np.sign(res_hi) and _expand < 10:
            Delta_hi *= 2.0
            res_hi, _, _, _ = _simulate_and_residual(Delta_hi)
            residual_history.append(res_hi)
            _expand += 1

        converged  = False
        n_iters    = len(residual_history)
        Delta_star = Delta_lo  # fallback

        for _ in range(max_iter):
            Delta_mid = 0.5 * (Delta_lo + Delta_hi)
            res_mid, _, _, _ = _simulate_and_residual(Delta_mid)
            residual_history.append(res_mid)
            n_iters += 1
            if abs(res_mid) < tol or (Delta_hi - Delta_lo) < tol:
                Delta_star = Delta_mid
                converged  = True
                break
            if np.sign(res_mid) == np.sign(res_lo):
                Delta_lo, res_lo = Delta_mid, res_mid
            else:
                Delta_hi, res_hi = Delta_mid, res_mid
            Delta_star = Delta_mid

        # Final simulation at converged Δ
        _, cf_budget, B_path, Y_path = _simulate_and_residual(Delta_star)

    # Reconstruct full macro from last run (already stored on olg)
    cf_macro = {
        'r': olg.r_path, 'w': olg.w_path,
        'K': olg.K_path, 'L': olg.L_path, 'Y': olg.Y_path,
    }
    if olg.K_g_path is not None:
        cf_macro['K_g'] = olg.K_g_path
    if olg.NFA_path is not None:
        cf_macro['NFA'] = olg.NFA_path

    B_gdp = B_path[:-1] / Y_path
    NFA, CA = _nfa_ca_paths(cf_macro)
    adj_path = Delta_star * psi

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
    T = int(olg.T_transition)
    r_path = np.asarray(base_paths['r_path'], dtype=float)
    residual_history = []

    # ── Step 1: Debt-financed, full shock ────────────────────────────────────
    full_debt_result = run_debt_financed(
        olg, scenario, base_paths, base_macro, base_budget, n_sim=n_sim, verbose=False
    )
    NFA_full, CA_full = full_debt_result.NFA_path, full_debt_result.CA_path
    violation = _check_nfa_violation(NFA_full, CA_full, scenario)

    # ── Step 2a: No violation ────────────────────────────────────────────────
    if not violation:
        return full_debt_result

    # ── Step 2b: Violation exists ────────────────────────────────────────────
    if scenario.financing == 'debt':
        # Mode I: trim the shock scale (bisect on Δ_shock ∈ [0, 1])
        label = 'shock scale Δ'

        def _feasible(shock_scale):
            cf = _apply_shock(scenario, base_paths, T,
                              instrument_delta=0.0, shock_scale=shock_scale)
            macro, budget = _run_one_simulation(
                olg, base_paths, cf, n_sim=n_sim, verbose=False,
                recompute_bequests=scenario.recompute_bequests,
            )
            NFA, CA = _nfa_ca_paths(macro)
            return not _check_nfa_violation(NFA, CA, scenario), macro, budget

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
        psi = _get_psi(scenario, T)
        adj_path = Delta_star * psi

    else:
        # Mode II: full shock + bisect on Δτ for NFA/CA feasibility
        label = f'Δ{scenario.financing} (NFA-constrained)'

        def _nfa_ok_at(Delta):
            cf = _apply_shock(scenario, base_paths, T,
                              instrument_delta=Delta, shock_scale=1.0)
            macro, budget = _run_one_simulation(
                olg, base_paths, cf, n_sim=n_sim, verbose=False,
                recompute_bequests=scenario.recompute_bequests,
            )
            NFA, CA = _nfa_ca_paths(macro)
            return not _check_nfa_violation(NFA, CA, scenario), macro, budget

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
        psi = _get_psi(scenario, T)
        adj_path = Delta_star * psi

    B_path = compute_debt_path(
        cf_budget_star['primary_deficit'], r_path, B_initial=scenario.B_initial
    )
    Y_path = np.asarray(cf_macro_star['Y'], dtype=float)
    B_gdp  = B_path[:-1] / Y_path
    NFA, CA = _nfa_ca_paths(cf_macro_star)

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
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_fiscal_scenario(olg, scenario: FiscalScenario, base_paths: dict,
                        n_sim: int = 500, verbose: bool = False,
                        bisect_tol: float = 1e-4,
                        bisect_max_iter: int = 40) -> FiscalScenarioResult:
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
        the baseline simulation is run once here.
    n_sim : int
        Number of Monte Carlo agents.
    verbose : bool
    bisect_tol, bisect_max_iter : convergence settings for tax-financed runs.

    Returns
    -------
    FiscalScenarioResult
    """
    # ── Fill in G_path / I_g_path defaults from olg object ──────────────────
    base_paths = dict(base_paths)  # shallow copy; don't mutate caller's dict
    if 'G_path' not in base_paths:
        base_paths['G_path'] = olg.govt_spending_path
    if 'I_g_path' not in base_paths:
        base_paths['I_g_path'] = olg.I_g_path

    # ── Run (or retrieve) baseline ───────────────────────────────────────────
    if 'base_macro' in base_paths and 'base_budget' in base_paths:
        base_macro  = base_paths['base_macro']
        base_budget = base_paths['base_budget']
    else:
        if verbose:
            print(f"[run_fiscal_scenario] Running baseline for scenario '{scenario.name}'...")
        T = len(np.asarray(base_paths['r_path']))
        base_cf = _apply_shock(
            FiscalScenario(name='_baseline'),  # zero shock
            base_paths, T, instrument_delta=0.0, shock_scale=0.0,
        )
        base_macro, base_budget = _run_one_simulation(
            olg, base_paths, base_cf, n_sim=n_sim, verbose=verbose,
            recompute_bequests=scenario.recompute_bequests,
        )

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
                                **kwargs)


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def compare_scenarios(
    base: FiscalScenarioResult,
    *counterfactuals: FiscalScenarioResult,
    variables: Optional[list] = None,
    save: bool = True,
    filename: Optional[str] = None,
    output_dir: str = 'output',
) -> plt.Figure:
    """Side-by-side line plots comparing base vs. counterfactual paths.

    Parameters
    ----------
    variables : list of str, optional
        Keys to plot.  Each key is looked up in cf_macro, cf_budget, B_gdp_path,
        NFA_path.  Defaults to ['Y', 'primary_deficit', 'B_gdp_path', 'NFA'].
    """
    import os
    if variables is None:
        variables = ['Y', 'primary_deficit', 'B_gdp_path', 'NFA']

    n_vars = len(variables)
    ncols  = min(3, n_vars)
    nrows  = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    all_axes  = axes.flat

    T = len(base.cf_macro['Y'])
    periods = np.arange(T)

    # Colour cycle
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_results = [base] + list(counterfactuals)
    labels      = [r.scenario.name for r in all_results]

    def _get_series(result: FiscalScenarioResult, key: str) -> Optional[np.ndarray]:
        if key == 'B_gdp_path':
            arr = result.B_gdp_path
            return arr[:T] if arr is not None else None
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
                    color=colours[i % len(colours)], linewidth=1.8)
        ax.set_title(key, fontweight='bold')
        ax.set_xlabel('Period')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in list(all_axes)[n_vars:]:
        ax.set_visible(False)

    plt.suptitle('Fiscal Scenario Comparison', fontsize=14, fontweight='bold')
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
) -> np.ndarray:
    """Compute per-period ΔOutput_t / ΔShock_t.

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

    total_dG = np.sum(delta_G)
    if total_dG != 0.0:
        cum = np.sum(delta_Y) / total_dG
        print(f"Cumulative multiplier Σ(Δ{output_variable}) / Σ(Δ{shock_variable}) = {cum:.4f}")

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
                label=label, color=colours[i % len(colours)], linewidth=2)

    ax.axhline(y=60.0, color='black', linestyle='--', alpha=0.4, label='60% reference')
    ax.set_xlabel('Period')
    ax.set_ylabel('Debt / GDP (%)')
    ax.set_title('Debt Sustainability Analysis — Fan Chart', fontweight='bold')
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
