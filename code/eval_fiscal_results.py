#!/usr/bin/env python3
"""
eval_fiscal_results.py — Post-simulation validation for fiscal_results.json

Checks numerical identities (FAIL tier) and theory predictions (WARN tier)
implied by the OLG-SOE model and MIT shock assumptions.

Usage:
    python eval_fiscal_results.py --input output/fiscal_test/fiscal_results.json
    python eval_fiscal_results.py --input fiscal_results.json --config calibration_input_GR.json

Exit codes:
    0 — no FAILs (WARNs are allowed)
    1 — at least one FAIL
"""

import argparse
import json
import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

IDENTITY_TOL  = 1e-4   # hard: accounting identities must hold to this
FOC_TOL       = 1e-3   # soft: firm FOCs (aggregation introduces small errors)
RATIO_TOL     = 0.20   # calibration ratio checks: 20% relative tolerance
NEUTRAL_REL   = 0.02   # debt-financed neutrality: 2% of mean Y
BISECT_TOL    = 1e-2   # bisection target tolerance
WEALTH_LO     = 0.5    # A/Y lower bound (short test models can be < 1)
WEALTH_HI     = 15.0   # A/Y upper bound


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name:      str
    scenario:  str
    tier:      str   # 'FAIL' | 'WARN' | 'INFO'
    status:    str   # 'PASS' | 'FAIL' | 'WARN' | 'SKIP'
    violation: float
    message:   str


def _pass(name, scenario, tier):
    return CheckResult(name, scenario, tier, 'PASS', 0.0, '')


def _fail(name, scenario, tier, violation, msg):
    status = 'FAIL' if tier == 'FAIL' else 'WARN'
    return CheckResult(name, scenario, tier, status, violation, msg)


def _skip(name, scenario, tier, reason=''):
    return CheckResult(name, scenario, tier, 'SKIP', 0.0, reason)


# ---------------------------------------------------------------------------
# Helper: safely get array
# ---------------------------------------------------------------------------

def _arr(d, key):
    v = d.get(key)
    return np.asarray(v, dtype=float) if v is not None else None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def chk_terminal_converged(exp_data, scenario):
    # WARN tier: terminal convergence is a model quality diagnostic, not an
    # accounting identity.  Short T_transition will naturally show drift.
    if not exp_data.get('terminal_converged', True):
        drift = exp_data.get('terminal_drift', {})
        bad = {k: v for k, v in drift.items() if v >= 0.005}
        return _fail('terminal_converged', scenario, 'WARN',
                     max(bad.values()) if bad else 1.0,
                     f"Not converged at T: { {k: f'{v:.1%}' for k, v in bad.items()} }")
    return _pass('terminal_converged', scenario, 'WARN')


def chk_kg_ss_gap(exp_data, params, scenario):
    drift = exp_data.get('terminal_drift', {})
    gap = drift.get('K_g_ss_gap')
    if gap is None:
        return _skip('kg_ss_gap', scenario, 'WARN', 'no K_g or delta_g=0')
    if gap > 0.05:
        return _fail('kg_ss_gap', scenario, 'WARN', gap,
                     f"|K_g[T] - K_g*| / K_g* = {gap:.1%}  — T_transition too short for K_g convergence")
    return _pass('kg_ss_gap', scenario, 'WARN')


def chk_no_nan_inf(macro, budget, scenario):
    bad = []
    for src in (macro, budget):
        for k, v in src.items():
            try:
                arr = np.asarray(v, dtype=float)
                if not np.all(np.isfinite(arr)):
                    bad.append(k)
            except (TypeError, ValueError):
                pass
    if bad:
        return _fail('no_nan_inf', scenario, 'FAIL', len(bad),
                     f"NaN/Inf in: {bad}")
    return _pass('no_nan_inf', scenario, 'FAIL')


def chk_positive_quantities(macro, scenario):
    bad = {}
    for key in ['Y', 'C', 'K', 'L', 'w']:
        arr = _arr(macro, key)
        if arr is not None:
            mn = float(arr.min())
            if mn <= 0:
                bad[key] = mn
    if bad:
        return _fail('positive_quantities', scenario, 'FAIL',
                     max(abs(v) for v in bad.values()),
                     f"Non-positive values: { {k: f'{v:.4f}' for k, v in bad.items()} }")
    return _pass('positive_quantities', scenario, 'FAIL')


def chk_convergence(converged, scenario):
    if not converged:
        return _fail('convergence', scenario, 'FAIL', 1.0, 'converged=False')
    return _pass('convergence', scenario, 'FAIL')


def chk_budget_identity(budget, scenario):
    PD  = _arr(budget, 'primary_deficit')
    S   = _arr(budget, 'total_spending')
    R   = _arr(budget, 'total_revenue')
    if PD is None or S is None or R is None:
        return _skip('budget_identity', scenario, 'FAIL', 'missing keys')
    resid = float(np.max(np.abs(PD - (S - R))))
    if resid > IDENTITY_TOL:
        return _fail('budget_identity', scenario, 'FAIL', resid,
                     f"max |PD - (spending - revenue)| = {resid:.2e}")
    return _pass('budget_identity', scenario, 'FAIL')


def chk_debt_accumulation(budget, B_gdp_path, Y, r_path, scenario):
    PD   = _arr(budget, 'primary_deficit')
    Ygdp = np.asarray(Y, dtype=float)
    Bgdp = np.asarray(B_gdp_path, dtype=float)
    r    = np.asarray(r_path, dtype=float)
    T    = len(PD)
    # Reconstruct B levels: B_gdp_path[t] = B[t] / Y[min(t, T-1)]
    Y_ext = np.append(Ygdp, Ygdp[-1])
    B = Bgdp * Y_ext           # length T+1
    resid = np.abs(B[1:] - ((1 + r[:T]) * B[:T] + PD))
    mx = float(resid.max())
    if mx > IDENTITY_TOL * float(np.abs(B).mean() + 1):
        return _fail('debt_accumulation', scenario, 'FAIL', mx,
                     f"max |B[t+1] - (1+r)*B[t] - PD[t]| = {mx:.2e}")
    return _pass('debt_accumulation', scenario, 'FAIL')


def chk_nfa_accounting(macro, B_gdp_path, scenario):
    A  = _arr(macro, 'A')
    Kd = _arr(macro, 'K_domestic')
    NFA = _arr(macro, 'NFA')
    if A is None or Kd is None or NFA is None:
        return _skip('nfa_accounting', scenario, 'FAIL', 'no SOE variables')
    Y  = _arr(macro, 'Y')
    T  = len(A)
    B  = np.asarray(B_gdp_path[:T], dtype=float) * Y
    resid = np.abs(A - Kd - NFA - B)
    mx = float(resid.max())
    scale = float(np.abs(A).mean()) + 1e-8
    if mx / scale > IDENTITY_TOL * 10:
        return _fail('nfa_accounting', scenario, 'FAIL', mx,
                     f"max |A - K_dom - NFA - B| = {mx:.2e}  (rel {mx/scale:.2e})")
    return _pass('nfa_accounting', scenario, 'FAIL')


def chk_tax_revenue(budget, macro, params, scenario):
    """
    Approximate check: in a heterogeneous-agent OLG model, tax revenues do not
    equal τ × macro_aggregate exactly (unemployed pay no payroll tax, retirees
    have different consumption bases, etc.).  These are WARN-tier plausibility
    checks, not hard identities.  A relative error > 50% signals a likely bug;
    10-50% is expected approximation error.

    tax_k base: r × A (household wealth), not r × K_domestic, because households
    earn r on all their savings (domestic capital + NFA + bonds) in a SOE.
    """
    results = []
    w  = _arr(macro, 'w')
    L  = _arr(macro, 'L')
    C  = _arr(macro, 'C')
    r  = _arr(macro, 'r')
    A  = _arr(macro, 'A')   # total household wealth — correct base for capital tax
    wL = w * L if (w is not None and L is not None) else None
    rA = r * A if (r is not None and A is not None) else None
    checks = [
        ('tax_l', 'tau_l_path', wL),
        ('tax_c', 'tau_c_path', C),
        ('tax_p', 'tau_p_path', wL),
        ('tax_k', 'tau_k_path', rA),
    ]
    for rev_key, rate_key, base in checks:
        rev  = _arr(budget, rev_key)
        rate = np.asarray(params.get(rate_key, []), dtype=float)
        if rev is None or base is None or len(rate) == 0:
            results.append(_skip(f'tax_revenue_{rev_key}', scenario, 'WARN',
                                 'missing data'))
            continue
        T_base = len(base)
        if len(rate) < T_base:
            rate = np.concatenate([rate, np.full(T_base - len(rate), rate[-1])])
        expected = rate[:T_base] * base
        resid = np.abs(rev - expected)
        scale = float(np.abs(expected).mean()) + 1e-8
        mx_rel = float(resid.max()) / scale
        if mx_rel > 0.50:   # > 50%: likely a model bug
            results.append(_fail(f'tax_revenue_{rev_key}', scenario, 'WARN',
                                 float(resid.max()),
                                 f"max |{rev_key} - τ·base| rel = {mx_rel:.0%}  (>50% threshold)"))
        elif mx_rel > FOC_TOL:
            results.append(_fail(f'tax_revenue_{rev_key}', scenario, 'WARN',
                                 float(resid.max()),
                                 f"max |{rev_key} - τ·base| rel = {mx_rel:.0%}  (approx error, expected in HAM)"))
        else:
            results.append(_pass(f'tax_revenue_{rev_key}', scenario, 'WARN'))
    return results


def chk_firm_foc(macro, params, scenario):
    results = []
    alpha = params.get('alpha')
    delta = params.get('delta')
    if alpha is None or delta is None:
        return [_skip('firm_foc_mpk', scenario, 'WARN', 'no params'),
                _skip('firm_foc_mpl', scenario, 'WARN', 'no params')]

    Y  = _arr(macro, 'Y')
    L  = _arr(macro, 'L')
    Kd = _arr(macro, 'K_domestic')
    w  = _arr(macro, 'w')
    r  = _arr(macro, 'r')

    # MPK: r + δ = α · Y / K_domestic
    if Kd is not None and r is not None and Y is not None:
        lhs = r + delta
        rhs = alpha * Y / Kd
        resid = np.abs(lhs - rhs)
        mx_rel = float(resid.max()) / (float(np.abs(rhs).mean()) + 1e-8)
        if mx_rel > FOC_TOL:
            results.append(_fail('firm_foc_mpk', scenario, 'WARN', float(resid.max()),
                                 f"max |r+δ - α·Y/K| = {resid.max():.2e}  (rel {mx_rel:.2e})"))
        else:
            results.append(_pass('firm_foc_mpk', scenario, 'WARN'))
    else:
        results.append(_skip('firm_foc_mpk', scenario, 'WARN', 'no K_domestic or r'))

    # MPL: w = (1-α) · Y / L
    if w is not None and Y is not None and L is not None:
        rhs = (1 - alpha) * Y / L
        resid = np.abs(w - rhs)
        mx_rel = float(resid.max()) / (float(np.abs(rhs).mean()) + 1e-8)
        if mx_rel > FOC_TOL:
            results.append(_fail('firm_foc_mpl', scenario, 'WARN', float(resid.max()),
                                 f"max |w - (1-α)·Y/L| = {resid.max():.2e}  (rel {mx_rel:.2e})"))
        else:
            results.append(_pass('firm_foc_mpl', scenario, 'WARN'))
    else:
        results.append(_skip('firm_foc_mpl', scenario, 'WARN', 'missing macro'))

    return results


def chk_wealth_output_ratio(macro, scenario):
    A = _arr(macro, 'A')
    Y = _arr(macro, 'Y')
    if A is None or Y is None:
        return _skip('wealth_output_ratio', scenario, 'WARN', 'missing A or Y')
    ratio = float(np.mean(A / Y))
    if ratio < WEALTH_LO or ratio > WEALTH_HI:
        return _fail('wealth_output_ratio', scenario, 'WARN', ratio,
                     f"mean(A/Y) = {ratio:.2f}, expected [{WEALTH_LO}, {WEALTH_HI}]")
    return _pass('wealth_output_ratio', scenario, 'WARN')


def chk_debt_financed_neutrality(base_macro, cf_macro, params, scenario):
    """In SOE with exogenous r and no tax change, K/L/Y/C must be identical."""
    results = []
    n_sim  = params.get('n_sim', 500)
    tol_rel = NEUTRAL_REL + 3.0 / np.sqrt(max(n_sim, 1))  # scale with sim noise
    for key in ['K', 'L', 'Y', 'C']:
        base = _arr(base_macro, key)
        cf   = _arr(cf_macro,   key)
        if base is None or cf is None:
            results.append(_skip(f'debt_neutral_{key}', scenario, 'WARN', f'no {key}'))
            continue
        mean_base = float(np.abs(base).mean()) + 1e-8
        mx_rel = float(np.max(np.abs(cf - base))) / mean_base
        if mx_rel > tol_rel:
            results.append(_fail(f'debt_neutral_{key}', scenario, 'WARN',
                                 mx_rel,
                                 f"max |Δ{key}| / mean = {mx_rel:.4f} > {tol_rel:.4f}"))
        else:
            results.append(_pass(f'debt_neutral_{key}', scenario, 'WARN'))
    return results


def chk_bisection_target(B_gdp_path, target, scenario):
    terminal = float(B_gdp_path[-1])
    resid = abs(terminal - target)
    if resid > BISECT_TOL:
        return _fail('bisection_target', scenario, 'FAIL', resid,
                     f"terminal B/Y = {terminal:.4f}, target = {target:.4f}, gap = {resid:.4f}")
    return _pass('bisection_target', scenario, 'FAIL')


def chk_shock_g_path(base_budget, cf_budget, delta_G, scenario):
    base_G = _arr(base_budget, 'govt_spending')
    cf_G   = _arr(cf_budget,   'govt_spending')
    dG     = np.asarray(delta_G, dtype=float)
    if base_G is None or cf_G is None:
        return _skip('shock_g_path', scenario, 'FAIL', 'no govt_spending in budget')
    T = min(len(base_G), len(cf_G), len(dG))
    resid = np.abs(cf_G[:T] - base_G[:T] - dG[:T])
    mx = float(resid.max())
    if mx > IDENTITY_TOL:
        return _fail('shock_g_path', scenario, 'FAIL', mx,
                     f"max |G_cf - G_base - ΔG| = {mx:.2e}")
    return _pass('shock_g_path', scenario, 'FAIL')


def chk_labor_response_sign(base_macro, cf_macro, adjustment_scalar, scenario):
    """If τ_l cut (Δ < 0) → L_cf > L_base; if τ_l hike (Δ > 0) → L_cf < L_base."""
    if adjustment_scalar is None or abs(adjustment_scalar) < 1e-6:
        return _skip('labor_response_sign', scenario, 'WARN', 'no tax adjustment')
    L_base = _arr(base_macro, 'L')
    L_cf   = _arr(cf_macro,   'L')
    if L_base is None or L_cf is None:
        return _skip('labor_response_sign', scenario, 'WARN', 'no L')
    expected_sign = -1 if adjustment_scalar > 0 else 1   # hike → less L; cut → more L
    actual_sign   = np.sign(np.mean(L_cf - L_base))
    if actual_sign != expected_sign and actual_sign != 0:
        return _fail('labor_response_sign', scenario, 'WARN', float(abs(actual_sign - expected_sign)),
                     f"Δτ_l = {adjustment_scalar:+.4f}, expected mean(ΔL) {'>' if expected_sign>0 else '<'} 0, "
                     f"got mean(ΔL) = {float(np.mean(L_cf - L_base)):.4e}")
    return _pass('labor_response_sign', scenario, 'WARN')


def chk_consumption_response_sign(base_macro, cf_macro, adjustment_scalar, scenario):
    """Same direction: tax cut → higher income → C up."""
    if adjustment_scalar is None or abs(adjustment_scalar) < 1e-6:
        return _skip('consumption_response_sign', scenario, 'WARN', 'no tax adjustment')
    C_base = _arr(base_macro, 'C')
    C_cf   = _arr(cf_macro,   'C')
    if C_base is None or C_cf is None:
        return _skip('consumption_response_sign', scenario, 'WARN', 'no C')
    expected_sign = -1 if adjustment_scalar > 0 else 1
    actual_sign   = np.sign(np.mean(C_cf - C_base))
    if actual_sign != expected_sign and actual_sign != 0:
        return _fail('consumption_response_sign', scenario, 'WARN',
                     float(abs(actual_sign - expected_sign)),
                     f"Δτ_l = {adjustment_scalar:+.4f}, expected mean(ΔC) {'>' if expected_sign>0 else '<'} 0, "
                     f"got mean(ΔC) = {float(np.mean(C_cf - C_base)):.4e}")
    return _pass('consumption_response_sign', scenario, 'WARN')


def chk_calibration_ratios(cf_budget, cf_macro, params, scenario):
    """Spending/revenue flows should be in the right ballpark relative to Y."""
    results = []
    Y = _arr(cf_macro, 'Y')
    if Y is None:
        return [_skip('calib_ratio', scenario, 'WARN', 'no Y')]
    mean_Y = float(Y.mean())
    targets = [
        ('pension',      'pensions_over_Y'),
        ('ui',           'ui_over_Y'),
        ('gov_health',   'health_over_Y'),
        ('govt_spending','G_over_Y'),
    ]
    for bud_key, param_key in targets:
        target = params.get(param_key)
        if target is None:
            continue
        arr = _arr(cf_budget, bud_key)
        if arr is None:
            continue
        actual = float(arr.mean()) / mean_Y
        rel_err = abs(actual - target) / (abs(target) + 1e-8)
        if rel_err > RATIO_TOL:
            results.append(_fail(f'calib_{bud_key}_over_Y', scenario, 'WARN',
                                 rel_err,
                                 f"mean({bud_key}/Y) = {actual:.3f}, target = {target:.3f} "
                                 f"(rel err {rel_err:.0%})"))
        else:
            results.append(_pass(f'calib_{bud_key}_over_Y', scenario, 'WARN'))
    return results


# ---------------------------------------------------------------------------
# Run all checks for one experiment (one shock type, one scenario key)
# ---------------------------------------------------------------------------

def run_scenario_checks(exp_data, scenario_key, params, shock_type):
    """
    exp_data : dict with keys baseline, counterfactual, base_budget, cf_budget,
               B_gdp_path, converged, adjustment_scalar
    scenario_key : 'baseline' | 'debt_financed' | 'tax_financed'
    """
    label    = f"{shock_type}/{scenario_key}"
    base_mac = exp_data.get('baseline', {})
    cf_mac   = exp_data.get('counterfactual', {})
    base_bud = exp_data.get('base_budget', {})
    cf_bud   = exp_data.get('cf_budget', {})
    B_gdp    = exp_data.get('B_gdp_path', [])
    conv     = exp_data.get('converged', True)
    adj_scl  = exp_data.get('adjustment_scalar')
    r_path   = cf_mac.get('r', base_mac.get('r', []))
    Y        = cf_mac.get('Y', base_mac.get('Y', []))

    results = []

    # --- FAIL tier ---
    results.append(chk_no_nan_inf(cf_mac, cf_bud, label))
    results.append(chk_positive_quantities(cf_mac, label))
    results.append(chk_convergence(conv, label))
    results.append(chk_budget_identity(cf_bud, label))
    if len(B_gdp) > 1 and len(Y) > 0:
        results.append(chk_debt_accumulation(cf_bud, B_gdp, Y, r_path, label))
    results.append(chk_nfa_accounting(cf_mac, B_gdp, label))
    results += chk_tax_revenue(cf_bud, cf_mac, params, label)

    # Shock path check
    delta_G  = params.get('delta_G_path')
    delta_Ig = params.get('delta_Ig_path')
    if shock_type == 'G' and delta_G is not None and scenario_key != 'baseline':
        results.append(chk_shock_g_path(base_bud, cf_bud, delta_G, label))

    # Bisection target (tax-financed only)
    # terminal_debt_gdp: check B[T]/Y[T] == target (stock condition)
    # terminal_flow_balance: check PD[T-1]/Y[T-1] ≈ (g-r)*target (flow condition)
    balance_cond = exp_data.get('balance_condition', 'terminal_debt_gdp')
    if scenario_key == 'tax_financed':
        if balance_cond == 'terminal_flow_balance':
            # Verify the flow condition held: PD[T-1]/Y[T-1] should be near zero
            # (or near (g-r)*target if target != 0).  We check |PD/Y| < BISECT_TOL.
            PD = _arr(cf_bud, 'primary_deficit')
            Y_arr = np.asarray(Y, dtype=float) if len(Y) > 0 else None
            if PD is not None and Y_arr is not None and len(Y_arr) > 0:
                pd_over_y = float(abs(PD[-1])) / (float(abs(Y_arr[-1])) + 1e-8)
                if pd_over_y > BISECT_TOL:
                    results.append(_fail('bisection_flow_target', label, 'FAIL', pd_over_y,
                                         f"|PD[T]/Y[T]| = {pd_over_y:.4f} > {BISECT_TOL} "
                                         f"(terminal_flow_balance not satisfied)"))
                else:
                    results.append(_pass('bisection_flow_target', label, 'FAIL'))
        elif len(B_gdp) > 0:
            target = params.get('target_debt_gdp', 0.0)
            results.append(chk_bisection_target(B_gdp, target, label))

    # Terminal convergence
    if scenario_key != 'baseline':
        results.append(chk_terminal_converged(exp_data, label))
        results.append(chk_kg_ss_gap(exp_data, params, label))

    # --- WARN tier ---
    results += chk_firm_foc(cf_mac, params, label)
    results.append(chk_wealth_output_ratio(cf_mac, label))

    if scenario_key == 'debt_financed':
        results += chk_debt_financed_neutrality(base_mac, cf_mac, params, label)

    if scenario_key == 'tax_financed':
        results.append(chk_labor_response_sign(base_mac, cf_mac, adj_scl, label))
        results.append(chk_consumption_response_sign(base_mac, cf_mac, adj_scl, label))

    results += chk_calibration_ratios(cf_bud, cf_mac, params, label)

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

STATUS_ICONS = {'PASS': '✓', 'FAIL': '✗', 'WARN': '⚠', 'SKIP': '–'}
STATUS_ORDER = {'FAIL': 0, 'WARN': 1, 'SKIP': 2, 'PASS': 3}


def print_report(all_results):
    # Sort: FAILs first, then WARNs, then SKIPs, then PASSes
    all_results = sorted(all_results, key=lambda r: (STATUS_ORDER[r.status], r.scenario, r.name))

    n_fail = sum(1 for r in all_results if r.status == 'FAIL')
    n_warn = sum(1 for r in all_results if r.status == 'WARN')
    n_pass = sum(1 for r in all_results if r.status == 'PASS')
    n_skip = sum(1 for r in all_results if r.status == 'SKIP')

    col_w = max(len(r.name) for r in all_results) + 2
    scn_w = max(len(r.scenario) for r in all_results) + 2

    print()
    print("=" * 80)
    print("  FISCAL RESULTS EVALUATION")
    print("=" * 80)
    print(f"  {'CHECK':<{col_w}} {'SCENARIO':<{scn_w}} {'STATUS':<6}  {'VIOLATION / NOTE'}")
    print("-" * 80)

    last_status = None
    for r in all_results:
        if r.status != last_status and last_status is not None:
            print()
        last_status = r.status
        icon = STATUS_ICONS[r.status]
        note = r.message if r.message else ''
        viol = f" [{r.violation:.3e}]" if r.violation > 0 and r.status not in ('SKIP', 'PASS') else ''
        print(f"  {r.name:<{col_w}} {r.scenario:<{scn_w}} {icon} {r.status:<5}  {note}{viol}")

    print("-" * 80)
    print(f"  SUMMARY:  {n_fail} FAIL  |  {n_warn} WARN  |  {n_skip} SKIP  |  {n_pass} PASS")
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Validate fiscal_results.json')
    parser.add_argument('--input',  required=True, help='Path to fiscal_results.json')
    parser.add_argument('--config', default=None,  help='Optional: path to calibration JSON for extra checks')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    params = data.get('params', {})

    # Merge calibration targets from config if provided (overrides what's in params)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        fiscal = cfg.get('fiscal', {})
        for key in ['pensions_over_Y', 'ui_over_Y', 'health_over_Y', 'G_over_Y', 'tax_revenue_over_Y']:
            if key in fiscal:
                params[key] = fiscal[key]
        prod = cfg.get('production', {})
        for key in ['alpha', 'delta', 'eta_g']:
            if key in prod and key not in params:
                params[key] = prod[key]

    if not params:
        print("WARNING: no 'params' section in JSON and no --config provided. "
              "Many checks will be skipped. Re-run fiscal figures to embed params.", file=sys.stderr)

    all_results = []
    scenario_keys = ['baseline', 'debt_financed', 'tax_financed']

    for shock_type in ('G', 'Ig'):
        shock_data = data.get(shock_type)
        if shock_data is None:
            continue
        for scn_key in scenario_keys:
            exp = shock_data.get(scn_key)
            if exp is None:
                continue
            all_results += run_scenario_checks(exp, scn_key, params, shock_type)

    print_report(all_results)

    n_fail = sum(1 for r in all_results if r.status == 'FAIL')
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == '__main__':
    main()
