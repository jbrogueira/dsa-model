"""
Fiscal experiment figures — demo script.

Runs three scenarios on the fast-test OLG configuration and produces figures
for each shock type.

Run:
    cd code
    python run_fiscal_figures.py               # G shock, NumPy backend (default)
    python run_fiscal_figures.py --shock Ig    # I_g (public investment) shock
    python run_fiscal_figures.py --shock both  # both shock types
    python run_fiscal_figures.py --backend jax # JAX backend
"""

import argparse
import functools
import json
import os
import sys
import platform
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # avoid Metal backend on macOS
import time

# Force unbuffered print so progress is visible over SSH / pipes
print = functools.partial(print, flush=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display required
import matplotlib.pyplot as plt

from lifecycle_perfect_foresight import LifecycleConfig
from olg_transition import OLGTransition
from fiscal_experiments import (
    FiscalScenario,
    run_fiscal_scenario,
    compare_scenarios,
    debt_fan_chart,
    fiscal_multiplier,
)

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['numpy', 'jax'], default='numpy')
parser.add_argument('--shock', choices=['G', 'Ig', 'both'], default='G',
                    help='Fiscal shock type: G (govt spending), Ig (public investment), or both')
parser.add_argument('--config', type=str, default=None,
                    help='JSON config file (same format as calibration input)')
parser.add_argument('--n-sim', type=int, default=None,
                    help='Override simulation size')
args = parser.parse_args()

if args.backend == 'jax':
    import jax
    devices = jax.devices()
    dev_type = devices[0].platform.upper() if devices else 'UNKNOWN'
    print(f"JAX backend: {dev_type} ({len(devices)} device(s): {devices})")

OUTPUT_DIR = 'output/fiscal_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Build OLG model
# ---------------------------------------------------------------------------

if args.config:
    # Build from JSON config file
    import json
    from calibrate import build_olg_transition

    with open(args.config) as f:
        config_data = json.load(f)

    economy, paths, T_TR = build_olg_transition(config_data, backend=args.backend)
    economy.output_dir = OUTPUT_DIR
    N_SIM = args.n_sim or config_data.get('transition', {}).get('n_sim', 2000)

    r_path = paths['r_path']
    tax_paths = {k: paths[k] for k in
                 ['tau_c_path', 'tau_l_path', 'tau_p_path', 'tau_k_path',
                  'pension_replacement_path']}

    # Compute I_g from data ratio
    prod = config_data.get('production', {})
    I_g_path = np.full(T_TR, prod.get('delta_g', 0.05) * prod.get('K_g', 0.0))

    # Calibrate G from data ratio
    print("Calibrating baseline G …")
    _calib = economy.simulate_transition(
        r_path=r_path, I_g_path=I_g_path, n_sim=50, verbose=False, **tax_paths
    )
    Y_path = np.asarray(_calib['Y'])
    G_over_Y = paths.get('G_over_Y', 0.13)
    G_path = np.full(T_TR, G_over_Y * Y_path.mean())
    B_over_Y = config_data.get('fiscal', {}).get('B_over_Y', 0.0)
    B_initial = B_over_Y * Y_path.mean()
    target_B_Y = B_over_Y  # tax-financed: return to initial debt ratio
    print(f"  mean(Y) = {Y_path.mean():.4f},  G/Y = {G_over_Y},  mean(G) = {G_path.mean():.4f}")
    print(f"  B/Y = {B_over_Y},  B_initial = {B_initial:.4f}")

else:
    # Hardcoded fast-test parameters (backward compatible)
    T_LC  = 20
    N_H   = 1
    N_SIM = args.n_sim or 2000

    config = LifecycleConfig(
        T              = T_LC,
        beta           = 0.96,
        gamma          = 2.0,
        n_a            = 50,
        n_y            = 4,
        n_h            = N_H,
        retirement_age = 15,
        education_type = 'medium',
        labor_supply   = True,
        nu             = 1.0,
        phi            = 2.0,
        survival_probs = np.linspace(0.995, 0.90, T_LC).reshape(T_LC, N_H),
    )

    economy = OLGTransition(
        lifecycle_config  = config,
        alpha             = 0.33,
        delta             = 0.05,
        A                 = 1.0,
        pop_growth        = 0.02,
        birth_year        = 2005,
        current_year      = 2020,
        education_shares  = {'medium': 1.0},
        eta_g             = 0.10,
        K_g_initial       = 1.0,
        backend           = args.backend,
        output_dir        = OUTPUT_DIR,
    )

    T_TR = 40
    r_path   = np.full(T_TR, 0.04)
    I_g_path = np.full(T_TR, 0.05 * 1.0)

    tax_paths = dict(
        tau_l_path               = np.full(T_TR, 0.15),
        tau_c_path               = np.full(T_TR, 0.18),
        tau_p_path               = np.full(T_TR, 0.20),
        tau_k_path               = np.full(T_TR, 0.20),
        pension_replacement_path = np.full(T_TR, 0.60),
    )

    print("Calibrating baseline G (30 %% of Y) …")
    _calib = economy.simulate_transition(
        r_path=r_path, I_g_path=I_g_path, n_sim=50, verbose=False, **tax_paths
    )
    Y_path = np.asarray(_calib['Y'])
    G_path = np.full(T_TR, 0.30 * Y_path.mean())
    B_initial = 0.0
    target_B_Y = 0.0  # hardcoded test: no initial debt, target stays at zero
    print(f"  mean(Y) = {Y_path.mean():.4f},  mean(G) = {G_path.mean():.4f}")

base_paths = dict(r_path=r_path, G_path=G_path, I_g_path=I_g_path, **tax_paths)

# ---------------------------------------------------------------------------
# 3. Define scenarios
# ---------------------------------------------------------------------------

# Extra periods beyond T_transition to show post-target dynamics
N_POST = 20

# --- Scenario A: pure baseline (no shock, debt residual) ---
scn_base = FiscalScenario(
    name      = 'baseline',
    financing = 'debt',
    B_initial = B_initial,
    n_post    = N_POST,
)

# --- G shock: 2% of mean(Y) ---
delta_G = np.full(T_TR, 0.02 * Y_path.mean())

scn_g_debt = FiscalScenario(
    name         = 'G_shock_debt',
    delta_G_path = delta_G,
    financing    = 'debt',
    B_initial    = B_initial,
    n_post       = N_POST,
)

scn_g_taul = FiscalScenario(
    name              = 'G_shock_tau_l',
    delta_G_path      = delta_G,
    financing         = 'tau_l',
    balance_condition = 'terminal_flow_balance',
    target_debt_gdp   = target_B_Y,
    B_initial         = B_initial,
    pop_growth        = float(economy.pop_growth),
    n_post            = N_POST,
)

# --- I_g shock: same absolute size as G shock ---
delta_Ig = np.full(T_TR, 0.02 * Y_path.mean())

scn_ig_debt = FiscalScenario(
    name           = 'Ig_shock_debt',
    delta_I_g_path = delta_Ig,
    financing      = 'debt',
    B_initial      = B_initial,
    n_post         = N_POST,
)

scn_ig_taul = FiscalScenario(
    name              = 'Ig_shock_tau_l',
    delta_I_g_path    = delta_Ig,
    financing         = 'tau_l',
    balance_condition = 'terminal_flow_balance',
    target_debt_gdp   = target_B_Y,
    B_initial         = B_initial,
    pop_growth        = float(economy.pop_growth),
    n_post            = N_POST,
)

# ---------------------------------------------------------------------------
# 4. Run experiments
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Running fiscal experiments …")
print("=" * 60)

t0 = time.time()


def run_experiment_set(shock_type):
    if shock_type == 'G':
        scn_debt, scn_taul = scn_g_debt, scn_g_taul
        labels = ('G shock (debt)', 'G shock (τ_l)')
    else:
        scn_debt, scn_taul = scn_ig_debt, scn_ig_taul
        labels = ('I_g shock (debt)', 'I_g shock (τ_l)')

    print(f"\n[1/3] Baseline …")
    res_base = run_fiscal_scenario(economy, scn_base, base_paths, n_sim=N_SIM, verbose=False)
    print(f"[2/3] {shock_type} shock — debt-financed …")
    res_debt = run_fiscal_scenario(economy, scn_debt, base_paths, n_sim=N_SIM, verbose=False)
    print(f"[3/3] {shock_type} shock — labour-tax-financed …")
    res_taul = run_fiscal_scenario(economy, scn_taul, base_paths, n_sim=N_SIM,
                                   verbose=False, bisect_tol=1e-3)
    return res_base, res_debt, res_taul, labels


shock_types = ['G'] if args.shock == 'G' else ['Ig'] if args.shock == 'Ig' else ['G', 'Ig']
experiment_results = {st: run_experiment_set(st) for st in shock_types}

print(f"\nAll experiments done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 5. Post-processing: attach derived quantities to each result
# ---------------------------------------------------------------------------

for res_base, res_debt, res_taul, _ in experiment_results.values():
    for res in (res_base, res_debt, res_taul):
        _T = len(res.cf_macro['Y'])
        _B = res.B_path[:_T]
        _r = np.asarray(res.cf_macro['r'])
        res.cf_budget['interest_payments'] = _r * _B
        _Kg = res.cf_macro.get('K_g')
        if _Kg is not None:
            res.cf_macro['K_g_Y'] = np.asarray(_Kg)[:_T] / np.asarray(res.cf_macro['Y'])

# ---------------------------------------------------------------------------
# 6. Figure variables
# ---------------------------------------------------------------------------

MACRO_VARS = ['Y', 'K_domestic', 'L', 'C', 'B_gdp_path', 'A', 'K_g_Y']
MACRO_LABELS = {
    'Y':          'Output (Y)',
    'K_domestic': 'Domestic capital (K)',
    'L':          'Labour (L)',
    'C':          'Consumption (C)',
    'B_gdp_path': 'Debt / GDP (B/Y)',
    'A':          'Household wealth (A)',
    'K_g_Y':      'Public capital / Y (K_g/Y)',
}

PRICE_VARS   = ['w', 'r']
PRICE_LABELS = {'w': 'Wage rate (w)', 'r': 'Interest rate (r)'}

FISCAL_VARS = [
    'primary_deficit',
    'tax_l', 'tax_c', 'tax_p', 'tax_k',
    'ui', 'pension', 'govt_spending', 'public_investment',
    'interest_payments',
]
FISCAL_LABELS = {
    'primary_deficit':   'Primary deficit',
    'tax_l':             'Labour tax',
    'tax_c':             'Consumption tax',
    'tax_p':             'Payroll tax',
    'tax_k':             'Capital tax',
    'ui':                'UI benefits',
    'pension':           'Pensions',
    'govt_spending':     'Govt spending (G)',
    'public_investment': 'Public investment (I_g)',
    'interest_payments': 'Interest payments (r·B)',
}

# ---------------------------------------------------------------------------
# 7. Print key scalars + produce figures per experiment set
# ---------------------------------------------------------------------------

print("\n--- Key results ---")
print("\n--- Saving figures ---")

for shock_type, (res_base, res_debt, res_taul, (label_debt, label_taul)) in experiment_results.items():
    p = shock_type.lower() + '_'
    SCENARIOS_  = [res_base, res_debt, res_taul]
    SCN_LABELS_ = ['baseline', label_debt, label_taul]
    title_suffix = f'({shock_type} shock)'

    print(f"\n  [{shock_type}] Baseline         : final B/Y = {res_base.B_gdp_path[-1]*100:.1f}%")
    print(f"  [{shock_type}] {label_debt:<20}: final B/Y = {res_debt.B_gdp_path[-1]*100:.1f}%")
    print(f"  [{shock_type}] {label_taul:<20}: converged={res_taul.converged}, "
          f"Δτ_l = {res_taul.adjustment_scalar*100:+.2f} pp, "
          f"final B/Y = {res_taul.B_gdp_path[-1]*100:.1f}%")

    shock_var = 'govt_spending' if shock_type == 'G' else 'public_investment'
    mult = fiscal_multiplier(res_base, res_debt, shock_variable=shock_var)
    print(f"  [{shock_type}] Fiscal multiplier ({shock_type} shock, debt): {np.nanmean(mult):.3f}")

    compare_scenarios(res_base, res_debt, res_taul,
        variables  = MACRO_VARS,
        var_labels = MACRO_LABELS,
        title      = f'Macro Overview {title_suffix}',
        output_dir = OUTPUT_DIR,
        filename   = f'{p}macro_overview.png',
    )
    plt.close('all')

    compare_scenarios(res_base, res_debt, res_taul,
        variables  = PRICE_VARS,
        var_labels = PRICE_LABELS,
        title      = f'Prices — SOE sanity check {title_suffix}',
        output_dir = OUTPUT_DIR,
        filename   = f'{p}prices_sanity.png',
    )
    plt.close('all')

    compare_scenarios(res_base, res_debt, res_taul,
        variables  = FISCAL_VARS,
        var_labels = FISCAL_LABELS,
        title      = f'Fiscal Decomposition {title_suffix}',
        output_dir = OUTPUT_DIR,
        filename   = f'{p}fiscal_decomp.png',
    )
    plt.close('all')

    debt_fan_chart(SCENARIOS_, SCN_LABELS_,
        output_dir = OUTPUT_DIR,
        filename   = f'{p}debt_fan_chart.png',
    )
    plt.close('all')

# ---------------------------------------------------------------------------
# 8. Save numerical results to JSON for remote inspection
# ---------------------------------------------------------------------------

def _result_to_dict(res):
    """Convert a FiscalScenarioResult to a JSON-serializable dict."""
    d = {
        'converged': bool(res.converged) if res.converged is not None else None,
        'terminal_converged': bool(res.terminal_converged),
        'terminal_drift': {k: float(v) for k, v in res.terminal_drift.items()},
        'balance_condition': res.scenario.balance_condition,
        'T_balance': res.T_balance,
        'n_post': res.scenario.n_post,
        'adjustment_scalar': float(res.adjustment_scalar) if res.adjustment_scalar else None,
        'B_gdp_path': [float(x) for x in res.B_gdp_path],
    }
    for label, macro in [('baseline', res.base_macro), ('counterfactual', res.cf_macro)]:
        d[label] = {}
        for k, v in macro.items():
            try:
                d[label][k] = [float(x) for x in np.asarray(v)]
            except (TypeError, ValueError):
                pass
    for label, budget in [('base_budget', res.base_budget), ('cf_budget', res.cf_budget)]:
        d[label] = {}
        for k, v in budget.items():
            try:
                d[label][k] = [float(x) for x in np.asarray(v)]
            except (TypeError, ValueError):
                pass
    return d

results_out = {}
for shock_type, (res_base, res_debt, res_taul, (label_debt, label_taul)) in experiment_results.items():
    shock_var = 'govt_spending' if shock_type == 'G' else 'public_investment'
    mult = fiscal_multiplier(res_base, res_debt, shock_variable=shock_var)
    results_out[shock_type] = {
        'baseline': _result_to_dict(res_base),
        'debt_financed': _result_to_dict(res_debt),
        'tax_financed': _result_to_dict(res_taul),
        'fiscal_multiplier_mean': float(np.nanmean(mult)),
        'fiscal_multiplier_path': [float(x) for x in mult],
    }

params_out = {
    'alpha':         float(economy.alpha),
    'delta':         float(economy.delta),
    'eta_g':         float(economy.eta_g),
    'labor_supply':  bool(getattr(economy.lifecycle_config, 'labor_supply', False)),
    'n_sim':         int(N_SIM),
    'T_transition':  int(T_TR),
    'B_initial':     float(B_initial),
    'target_debt_gdp': float(target_B_Y),
    'tau_l_path':    [float(x) for x in base_paths['tau_l_path']],
    'tau_c_path':    [float(x) for x in base_paths['tau_c_path']],
    'tau_p_path':    [float(x) for x in base_paths['tau_p_path']],
    'tau_k_path':    [float(x) for x in base_paths['tau_k_path']],
    'delta_G_path':  [float(x) for x in delta_G] if 'G'  in shock_types else None,
    'delta_Ig_path': [float(x) for x in delta_Ig] if 'Ig' in shock_types else None,
}
if args.config:
    fiscal = config_data.get('fiscal', {})
    for key in ['pensions_over_Y', 'ui_over_Y', 'health_over_Y', 'G_over_Y', 'tax_revenue_over_Y']:
        if key in fiscal:
            params_out[key] = fiscal[key]

results_out['params'] = params_out

results_path = os.path.join(OUTPUT_DIR, 'fiscal_results.json')
with open(results_path, 'w') as f:
    json.dump(results_out, f, indent=2)

print(f"\nFigures saved to {OUTPUT_DIR}/")
print(f"Numerical results saved to {results_path}")
