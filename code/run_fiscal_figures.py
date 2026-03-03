"""
Fiscal experiment figures — quick demo script.

Runs three scenarios on the fast-test OLG configuration and produces:
  output/fiscal_test/comparison_g_shock.png  — G shock: debt vs tax financing
  output/fiscal_test/debt_fan_chart.png      — B/Y paths for all scenarios

Run:
    cd code
    python run_fiscal_figures.py               # NumPy backend (default)
    python run_fiscal_figures.py --backend jax # JAX backend
"""

import argparse
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import time
import numpy as np
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
args = parser.parse_args()

OUTPUT_DIR = 'output/fiscal_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Build OLG model  (fast-test parameters: T=20, n_a=30, n_sim=300)
# ---------------------------------------------------------------------------

T_LC  = 20   # lifecycle periods
N_H   = 1
N_SIM = 4000

config = LifecycleConfig(
    T              = T_LC,
    beta           = 0.96,
    gamma          = 2.0,
    n_a            = 60,
    n_y            = 2,
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

# ---------------------------------------------------------------------------
# 2. Calibrate baseline G = 30% of Y
#    K_g is held constant: I_g = delta_g * K_g_initial = 0.05 * 1.0 = 0.05
#    Run one cheap preliminary simulation to read off Y, then set G_path.
# ---------------------------------------------------------------------------

T_TR = 40   # transition periods

r_path  = np.full(T_TR, 0.04)          # constant 4 %
I_g_path = np.full(T_TR, 0.05 * 1.0)  # delta_g * K_g_initial → K_g constant

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
G_path = 0.30 * Y_path
print(f"  mean(Y) = {Y_path.mean():.4f},  mean(G) = {G_path.mean():.4f}")

base_paths = dict(r_path=r_path, G_path=G_path, I_g_path=I_g_path, **tax_paths)

# The G shock is 2 % of mean(Y) — a concrete absolute amount
delta_G = np.full(T_TR, 0.02 * Y_path.mean())

# ---------------------------------------------------------------------------
# 3. Define scenarios
# ---------------------------------------------------------------------------

# --- Scenario A: pure baseline (no shock, debt residual) ---
scn_base = FiscalScenario(
    name      = 'baseline',
    financing = 'debt',
)

# --- Scenario B: +2% of Y G shock, debt-financed ---
scn_g_debt = FiscalScenario(
    name         = 'G_shock_debt',
    delta_G_path = delta_G,
    financing    = 'debt',
)

# --- Scenario C: same G shock, balanced via labour tax ---
scn_g_taul = FiscalScenario(
    name              = 'G_shock_tau_l',
    delta_G_path      = delta_G,
    financing         = 'tau_l',
    balance_condition = 'terminal_debt_gdp',
    target_debt_gdp   = 0.0,
)

# ---------------------------------------------------------------------------
# 4. Run experiments
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Running fiscal experiments …")
print("=" * 60)

t0 = time.time()

print("\n[1/3] Baseline …")
res_base   = run_fiscal_scenario(economy, scn_base,   base_paths, n_sim=N_SIM, verbose=False)

print("[2/3] G shock — debt-financed …")
res_g_debt = run_fiscal_scenario(economy, scn_g_debt, base_paths, n_sim=N_SIM, verbose=False)

print("[3/3] G shock — labour-tax-financed …")
res_g_taul = run_fiscal_scenario(economy, scn_g_taul, base_paths, n_sim=N_SIM, verbose=False,
                                 bisect_tol=1e-3)

print(f"\nAll experiments done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 5. Print key scalars
# ---------------------------------------------------------------------------

print("\n--- Key results ---")
print(f"  Baseline         : final B/Y = {res_base.B_gdp_path[-1]*100:.1f}%")
print(f"  G shock (debt)   : final B/Y = {res_g_debt.B_gdp_path[-1]*100:.1f}%")
print(f"  G shock (tau_l)  : converged={res_g_taul.converged}, "
      f"Δτ_l = {res_g_taul.adjustment_scalar*100:+.2f} pp, "
      f"final B/Y = {res_g_taul.B_gdp_path[-1]*100:.1f}%")

mult = fiscal_multiplier(res_base, res_g_debt, shock_variable='govt_spending')
print(f"  Fiscal multiplier (G shock, debt): {np.nanmean(mult):.3f}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------

print("\n--- Saving figures ---")

# ── post-process: attach derived quantities to each result ──────────────────
# A is already in cf_macro (= K_path = total household wealth from simulation).
# interest_payments = r_t · B_t  (debt-service cost from fiscal accounting)
for res in (res_base, res_g_debt, res_g_taul):
    _T = len(res.cf_macro['Y'])
    _B = res.B_path[:_T]                              # start-of-period debt
    _r = np.asarray(res.cf_macro['r'])
    res.cf_budget['interest_payments'] = _r * _B      # r_t · B_t

SCENARIOS   = [res_base, res_g_debt, res_g_taul]
SCN_LABELS  = ['baseline', 'G shock (debt)', 'G shock (τ_l)']

# Figure 1 — Macro overview: Y, K_domestic, L, C, B/Y, A
MACRO_VARS = ['Y', 'K_domestic', 'L', 'C', 'B_gdp_path', 'A']
MACRO_LABELS = {
    'Y':          'Output (Y)',
    'K_domestic': 'Domestic capital (K)',
    'L':          'Labour (L)',
    'C':          'Consumption (C)',
    'B_gdp_path': 'Debt / GDP (B/Y)',
    'A':          'Household wealth (A)',
}
fig1 = compare_scenarios(
    res_base, res_g_debt, res_g_taul,
    variables  = MACRO_VARS,
    var_labels = MACRO_LABELS,
    title      = 'Macro Overview',
    output_dir = OUTPUT_DIR,
    filename   = 'macro_overview.png',
)
plt.show()

# Figure 2 — Prices sanity check: w and r should be constant in SOE
PRICE_VARS   = ['w', 'r']
PRICE_LABELS = {'w': 'Wage rate (w)', 'r': 'Interest rate (r)'}
fig2 = compare_scenarios(
    res_base, res_g_debt, res_g_taul,
    variables  = PRICE_VARS,
    var_labels = PRICE_LABELS,
    title      = 'Prices (SOE sanity check)',
    output_dir = OUTPUT_DIR,
    filename   = 'prices_sanity.png',
)
plt.show()

# Figure 3 — Fiscal decomposition
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
fig3 = compare_scenarios(
    res_base, res_g_debt, res_g_taul,
    variables  = FISCAL_VARS,
    var_labels = FISCAL_LABELS,
    title      = 'Fiscal Decomposition',
    output_dir = OUTPUT_DIR,
    filename   = 'fiscal_decomp.png',
)
plt.show()

# Figure 4 — Debt fan chart
fig4 = debt_fan_chart(
    scenarios  = SCENARIOS,
    labels     = SCN_LABELS,
    output_dir = OUTPUT_DIR,
    filename   = 'debt_fan_chart.png',
)
plt.show()

print(f"\nFigures saved to {OUTPUT_DIR}/")
