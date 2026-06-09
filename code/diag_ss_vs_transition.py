"""
Diagnostic: does the transition baseline-at-rest coincide with the calibration
steady state?

Computes, like-for-like, item ratios (item / Y) on two paths:
  (A) SS side: calibrate.py stationary cross-section, calibrated theta, age-weighted.
  (B) Transition side: one no-shock baseline simulate_transition (G/I_g/defense/other
      wired exactly as run_fiscal_figures config branch), per-period government budget.

Prints the t=0 transition ratios vs the SS ratios, plus the full Y path and item
paths across periods so we can tell a TRANSIENT (t=0 matches SS, later drifts)
from a LEVEL mismatch (t=0 already off => convention bug).
"""
import os, sys, platform, json
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

CONFIG = 'calibration_input_GR.json'
BACKEND = sys.argv[1] if len(sys.argv) > 1 else 'numpy'
N_SIM = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

from calibrate import (load_config, build_olg_transition, run_model_moments,
                       compute_fiscal_ratios)

with open(CONFIG) as f:
    config_data = json.load(f)

# ----------------------------------------------------------------------------
# (A) SS side
# ----------------------------------------------------------------------------
print("=" * 70)
print("(A) SS side — calibrate.py stationary cross-section")
print("=" * 70)
loaded = load_config(CONFIG)
spec = loaded['spec']
# load_config injects _derived.K_over_L / w into the dict it returns; the
# separately-loaded config_data above still has K_over_L=None, which would make
# compute_fiscal_ratios return Y<=0. Use the load_config dict for the SS side.
cfg_ss = loaded['config_data']
print("SS K_over_L =", cfg_ss.get('_derived', {}).get('K_over_L'))
spec.n_sim = N_SIM          # dataclass: assign directly
spec.backend = BACKEND
theta_dict = config_data['_derived']['theta']
theta = np.array([theta_dict[p.name] for p in spec.params])
print("theta:", {p.name: float(v) for p, v in zip(spec.params, theta)})
print(f"n_sim={spec.n_sim}, seed={spec.seed}, w={spec.w:.5f}, r={spec.r:.4f}")

m_model, panels = run_model_moments(theta, spec, return_panels=True)
ss = compute_fiscal_ratios(panels, spec, cfg_ss)
if 'error' in ss:
    print("compute_fiscal_ratios error:", ss['error']); raise SystemExit(1)
print(f"\nSS Y level = {ss['Y']:.5f},  L = {ss['L']:.5f}")
ss_keys = ['tax_revenue_over_Y', 'tax_c_over_Y', 'tax_l_over_Y', 'tax_p_over_Y',
           'tax_k_over_Y', 'pensions_over_Y', 'ui_over_Y', 'health_gov_over_Y',
           'C_over_Y', 'K_over_Y', 'primary_balance_over_Y']
for k in ss_keys:
    print(f"  SS {k:28s} = {ss[k]:+.4f}")

# ----------------------------------------------------------------------------
# (B) Transition side — one no-shock baseline
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("(B) Transition side — no-shock baseline")
print("=" * 70)
economy, paths, T_TR = build_olg_transition(config_data, backend=BACKEND)

r_path = paths['r_path']
tax_paths = {k: paths[k] for k in
             ['tau_c_path', 'tau_l_path', 'tau_p_path', 'tau_k_path',
              'pension_replacement_path']}
prod = config_data.get('production', {})
I_g_warmup = np.full(T_TR, prod.get('delta_g', 0.05) * prod.get('K_g', 0.0))

print("warmup sim (n_sim=50) for mean(Y) ...")
_calib = economy.simulate_transition(r_path=r_path, I_g_path=I_g_warmup,
                                     n_sim=50, verbose=False, **tax_paths)
meanY = float(np.asarray(_calib['Y']).mean())
G_over_Y = paths.get('G_over_Y', 0.13)
I_g_over_Y = paths.get('I_g_over_Y', 0.03)
defense_over_Y = paths.get('defense_over_Y', 0.0)
other_over_Y = paths.get('other_net_spending_over_Y', 0.0)
G_path = np.full(T_TR, G_over_Y * meanY)
I_g_path = np.full(T_TR, I_g_over_Y * meanY)
defense_path = np.full(T_TR, defense_over_Y * meanY)
other_path = np.full(T_TR, other_over_Y * meanY)
print(f"warmup mean(Y) = {meanY:.5f}; G/Y={G_over_Y} I_g/Y={I_g_over_Y} "
      f"def/Y={defense_over_Y} other/Y={other_over_Y}")

print(f"baseline no-shock sim (n_sim={N_SIM}, backend={BACKEND}) ...")
res = economy.simulate_transition(
    r_path=r_path, govt_spending_path=G_path, I_g_path=I_g_path,
    defense_spending_path=defense_path, other_net_spending_path=other_path,
    n_sim=N_SIM, verbose=False, **tax_paths)
Yp = np.asarray(res['Y'])
budget = economy.compute_government_budget_path(n_sim=N_SIM, verbose=False)

def ratio(key):
    return np.asarray(budget[key]) / Yp

tr = {
    'Y': Yp,
    'tax_revenue_over_Y': ratio('total_revenue'),
    'tax_c_over_Y': ratio('tax_c'),
    'tax_l_over_Y': ratio('tax_l'),
    'tax_p_over_Y': ratio('tax_p'),
    'tax_k_over_Y': ratio('tax_k'),
    'pensions_over_Y': ratio('pension'),
    'ui_over_Y': ratio('ui'),
    'health_gov_over_Y': ratio('gov_health'),
    'primary_deficit_over_Y': ratio('primary_deficit'),
}

# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPARISON: SS vs transition t=0  (and transition t=last)")
print("=" * 70)
print(f"{'item':28s} {'SS':>10s} {'tr[t=0]':>10s} {'tr[t=-1]':>10s} "
      f"{'d(t0-SS)%':>10s}")
cmp_keys = ['tax_revenue_over_Y', 'tax_c_over_Y', 'tax_l_over_Y',
            'tax_p_over_Y', 'tax_k_over_Y', 'pensions_over_Y', 'ui_over_Y',
            'health_gov_over_Y']
for k in cmp_keys:
    sv = ss.get(k, float('nan'))
    t0 = tr[k][0]
    tl = tr[k][-1]
    pct = 100 * (t0 - sv) / sv if abs(sv) > 1e-12 else float('nan')
    print(f"{k:28s} {sv:>10.4f} {t0:>10.4f} {tl:>10.4f} {pct:>+10.1f}")

print(f"\n{'Y level':28s} {ss['Y']:>10.4f} {tr['Y'][0]:>10.4f} "
      f"{tr['Y'][-1]:>10.4f} {100*(tr['Y'][0]-ss['Y'])/ss['Y']:>+10.1f}")
print(f"{'SS primary_balance/Y':28s} {ss['primary_balance_over_Y']:>10.4f}")
print(f"{'tr primary_DEFICIT/Y [t0]':28s} {tr['primary_deficit_over_Y'][0]:>10.4f}"
      f"  (surplus = {-tr['primary_deficit_over_Y'][0]:+.4f})")

print("\n--- transition Y path (every 5 periods) ---")
idx = list(range(0, T_TR, 5)) + [T_TR - 1]
print("  t:    " + " ".join(f"{i:7d}" for i in idx))
print("  Y:    " + " ".join(f"{Yp[i]:7.4f}" for i in idx))
print("  taxp: " + " ".join(f"{tr['tax_p_over_Y'][i]:7.4f}" for i in idx))
print("  pens: " + " ".join(f"{tr['pensions_over_Y'][i]:7.4f}" for i in idx))
print("  rev:  " + " ".join(f"{tr['tax_revenue_over_Y'][i]:7.4f}" for i in idx))
print("\nDONE")
