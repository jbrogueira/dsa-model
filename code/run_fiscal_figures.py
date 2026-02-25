"""
Fiscal experiment figures — quick demo script.

Runs four scenarios on the fast-test OLG configuration and produces:
  output/fiscal_test/comparison_g_shock.png      — G shock: debt vs tax financing
  output/fiscal_test/comparison_pension_cut.png  — pension cut: debt vs tax financing
  output/fiscal_test/debt_fan_chart.png          — B/Y paths for all four scenarios

Run:
    cd code
    python run_fiscal_figures.py
"""

import os
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

OUTPUT_DIR = 'output/fiscal_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Build OLG model  (fast-test parameters: T=20, n_a=30, n_sim=300)
# ---------------------------------------------------------------------------

T_LC  = 20   # lifecycle periods
N_H   = 1
N_SIM = 300

config = LifecycleConfig(
    T              = T_LC,
    beta           = 0.96,
    gamma          = 2.0,
    n_a            = 30,
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
    output_dir        = OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# 2. Calibrate baseline G = 30% of Y
#    eta_g = 0, so Y does not depend on G — run one cheap preliminary
#    simulation to read off Y, then set G_path = 0.30 * Y_path.
# ---------------------------------------------------------------------------

T_TR = 25   # transition periods

periods = np.arange(T_TR)
r_path  = 0.04 + (0.03 - 0.04) * (periods / (T_TR - 1))   # 4 % → 3 %

tax_paths = dict(
    tau_l_path               = np.full(T_TR, 0.15),
    tau_c_path               = np.full(T_TR, 0.18),
    tau_p_path               = np.full(T_TR, 0.20),
    tau_k_path               = np.full(T_TR, 0.20),
    pension_replacement_path = np.full(T_TR, 0.80),
)

print("Calibrating baseline G (30 %% of Y) …")
_calib = economy.simulate_transition(
    r_path=r_path, n_sim=50, verbose=False, **tax_paths
)
Y_path = np.asarray(_calib['Y'])
G_path = 0.30 * Y_path
print(f"  mean(Y) = {Y_path.mean():.4f},  mean(G) = {G_path.mean():.4f}")

base_paths = dict(r_path=r_path, G_path=G_path, **tax_paths)

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

# --- Scenario D: -10 pp pension replacement, savings via labour tax cut ---
scn_pens = FiscalScenario(
    name               = 'pension_cut_tau_l',
    delta_pension_path = np.full(T_TR, -0.10),   # 80 % → 70 %
    financing          = 'tau_l',
    balance_condition  = 'terminal_debt_gdp',
    target_debt_gdp    = 0.0,
)

# ---------------------------------------------------------------------------
# 4. Run experiments
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Running fiscal experiments …")
print("=" * 60)

t0 = time.time()

print("\n[1/4] Baseline …")
res_base   = run_fiscal_scenario(economy, scn_base,   base_paths, n_sim=N_SIM, verbose=False)

print("[2/4] G shock — debt-financed …")
res_g_debt = run_fiscal_scenario(economy, scn_g_debt, base_paths, n_sim=N_SIM, verbose=False)

print("[3/4] G shock — labour-tax-financed …")
res_g_taul = run_fiscal_scenario(economy, scn_g_taul, base_paths, n_sim=N_SIM, verbose=False,
                                 bisect_tol=1e-3)

print("[4/4] Pension cut — labour-tax-financed …")
res_pens   = run_fiscal_scenario(economy, scn_pens,   base_paths, n_sim=N_SIM, verbose=False,
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
print(f"  Pension cut      : converged={res_pens.converged}, "
      f"Δτ_l = {res_pens.adjustment_scalar*100:+.2f} pp, "
      f"final B/Y = {res_pens.B_gdp_path[-1]*100:.1f}%")

mult = fiscal_multiplier(res_base, res_g_debt, shock_variable='govt_spending')
print(f"  Fiscal multiplier (G shock, debt): {np.nanmean(mult):.3f}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------

print("\n--- Saving figures ---")

# Figure 1: G shock comparison (debt vs tax financing)
fig1 = compare_scenarios(
    res_base, res_g_debt, res_g_taul,
    variables  = ['Y', 'K', 'primary_deficit', 'B_gdp_path', 'total_revenue', 'total_spending'],
    output_dir = OUTPUT_DIR,
    filename   = 'comparison_g_shock.png',
)
plt.show()

# Figure 2: Pension cut comparison (baseline vs pension cut)
fig2 = compare_scenarios(
    res_base, res_pens,
    variables  = ['Y', 'K', 'primary_deficit', 'B_gdp_path', 'total_revenue', 'total_spending'],
    output_dir = OUTPUT_DIR,
    filename   = 'comparison_pension_cut.png',
)
plt.show()

# Figure 3: Debt fan chart across all four scenarios
fig3 = debt_fan_chart(
    scenarios  = [res_base, res_g_debt, res_g_taul, res_pens],
    labels     = ['baseline', 'G shock (debt)', 'G shock (τ_l)', 'pension cut (τ_l)'],
    output_dir = OUTPUT_DIR,
    filename   = 'debt_fan_chart.png',
)
plt.show()

print(f"\nFigures saved to {OUTPUT_DIR}/")
