"""
Re-measure the transition baseline primary balance under the current
calibration (_derived.theta), with other_net_spending = 0, to reset the
baseline fiscal closure.

Procedure (FISCAL_EXPERIMENTS_STATUS.md, 2026-05-28 Fix 2):
    other_net_spending_over_Y = (measured baseline primary surplus / Y)
                                - (target primary surplus / Y)
Target: Greek 2023 primary balance = +1.95% of Y.

Usage: python measure_baseline_closure.py [backend] [n_sim]
"""
import os, sys, platform, json
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

CONFIG = 'calibration_input_GR.json'
BACKEND = sys.argv[1] if len(sys.argv) > 1 else 'jax'
N_SIM = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
TARGET_SURPLUS_OVER_Y = 0.0195

from calibrate import build_olg_transition

with open(CONFIG) as f:
    config_data = json.load(f)

theta = config_data['_derived']['theta']
print("theta in config:", theta)
print(f"backend={BACKEND}, n_sim={N_SIM}")

economy, paths, T_TR = build_olg_transition(config_data, backend=BACKEND)

r_path = paths['r_path']
tax_paths = {k: paths[k] for k in
             ['tau_c_path', 'tau_l_path', 'tau_p_path', 'tau_k_path',
              'pension_replacement_path']}
prod = config_data.get('production', {})
I_g_warmup = np.full(T_TR, prod.get('delta_g', 0.05) * prod.get('K_g', 0.0))

print("warmup sim (n_sim=50) for mean(Y) ...", flush=True)
_calib = economy.simulate_transition(r_path=r_path, I_g_path=I_g_warmup,
                                     n_sim=50, verbose=False, **tax_paths)
meanY = float(np.asarray(_calib['Y']).mean())
G_over_Y = paths.get('G_over_Y', 0.13)
I_g_over_Y = paths.get('I_g_over_Y', 0.03)
defense_over_Y = paths.get('defense_over_Y', 0.0)
G_path = np.full(T_TR, G_over_Y * meanY)
I_g_path = np.full(T_TR, I_g_over_Y * meanY)
defense_path = np.full(T_TR, defense_over_Y * meanY)
other_path = np.zeros(T_TR)  # closure knob forced to 0 for the measurement
print(f"warmup mean(Y) = {meanY:.5f}; G/Y={G_over_Y} I_g/Y={I_g_over_Y} "
      f"def/Y={defense_over_Y} other/Y=0 (forced)", flush=True)

print(f"baseline no-shock sim (n_sim={N_SIM}, backend={BACKEND}) ...", flush=True)
res = economy.simulate_transition(
    r_path=r_path, govt_spending_path=G_path, I_g_path=I_g_path,
    defense_spending_path=defense_path, other_net_spending_path=other_path,
    n_sim=N_SIM, verbose=False, **tax_paths)
Yp = np.asarray(res['Y'])
budget = economy.compute_government_budget_path(n_sim=N_SIM, verbose=False)

deficit_over_Y = np.asarray(budget['primary_deficit']) / Yp
surplus_over_Y = -deficit_over_Y

print("\n--- primary surplus / Y path (every 5 periods) ---")
idx = list(range(0, T_TR, 5)) + [T_TR - 1]
print("  t:    " + " ".join(f"{i:8d}" for i in idx))
print("  s/Y:  " + " ".join(f"{surplus_over_Y[i]:+8.4f}" for i in idx))

s_t0 = float(surplus_over_Y[0])
s_mean = float(surplus_over_Y.mean())
print(f"\nprimary surplus / Y at t=0   : {s_t0:+.4f}")
print(f"primary surplus / Y mean     : {s_mean:+.4f}")
print(f"target primary surplus / Y   : {TARGET_SURPLUS_OVER_Y:+.4f}")
print(f"\nother_net_spending_over_Y (from t=0)  : {s_t0 - TARGET_SURPLUS_OVER_Y:+.4f}")
print(f"other_net_spending_over_Y (from mean) : {s_mean - TARGET_SURPLUS_OVER_Y:+.4f}")
print(f"(current config value: "
      f"{config_data.get('fiscal', {}).get('other_net_spending_over_Y')})")
print("\nDONE")
