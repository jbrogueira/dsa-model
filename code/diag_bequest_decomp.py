"""
Decomposition of the SS-vs-transition item-ratio gap (route 2, handoff 2026-06-09).

Three like-for-like computations of government-budget item ratios (item / Y):
  (A) SS side: calibrate.py stationary cross-section (no bequest receipts).
  (B) Transition no-shock baseline, recompute_bequests=False (open circuit,
      like-for-like with the SS side).
  (C) Transition no-shock baseline, recompute_bequests=True (closed circuit,
      bequest fixed-point loop).

Gap decomposition per item:
  gap_total  = (B) - (A)        solve-path difference (stationary single-cohort
                                solve vs MIT-stitched cohort solves), bequests off
  d_bequest  = (C) - (B)        bequest-redistribution contribution
Both sides draw survival from the same data 2020 life table (config
survival_probs + transition survival_table), so demographics are controlled.

Usage: python diag_bequest_decomp.py [backend] [n_sim]
"""
import os, sys, platform, json
if platform.system() == 'Darwin':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

CONFIG = 'calibration_input_GR.json'
BACKEND = sys.argv[1] if len(sys.argv) > 1 else 'jax'
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
cfg_ss = loaded['config_data']
spec.n_sim = N_SIM
spec.backend = BACKEND
theta_dict = config_data['_derived']['theta']
theta = np.array([theta_dict[p.name] for p in spec.params])
print("theta:", {p.name: float(v) for p, v in zip(spec.params, theta)})

m_model, panels = run_model_moments(theta, spec, return_panels=True)
ss = compute_fiscal_ratios(panels, spec, cfg_ss)
if 'error' in ss:
    print("compute_fiscal_ratios error:", ss['error']); raise SystemExit(1)
print(f"SS Y level = {ss['Y']:.5f},  L = {ss['L']:.5f}", flush=True)

# ----------------------------------------------------------------------------
# Transition runner (shared by B and C)
# ----------------------------------------------------------------------------
KEYS = [('tax_revenue_over_Y', 'total_revenue'),
        ('tax_c_over_Y', 'tax_c'), ('tax_l_over_Y', 'tax_l'),
        ('tax_p_over_Y', 'tax_p'), ('tax_k_over_Y', 'tax_k'),
        ('pensions_over_Y', 'pension'), ('ui_over_Y', 'ui'),
        ('health_gov_over_Y', 'gov_health')]

def run_transition(recompute_bequests):
    economy, paths, T_TR = build_olg_transition(config_data, backend=BACKEND)
    r_path = paths['r_path']
    tax_paths = {k: paths[k] for k in
                 ['tau_c_path', 'tau_l_path', 'tau_p_path', 'tau_k_path',
                  'pension_replacement_path']}
    prod = config_data.get('production', {})
    I_g_warmup = np.full(T_TR, prod.get('delta_g', 0.05) * prod.get('K_g', 0.0))
    _calib = economy.simulate_transition(r_path=r_path, I_g_path=I_g_warmup,
                                         n_sim=50, verbose=False, **tax_paths)
    meanY = float(np.asarray(_calib['Y']).mean())
    G_path = np.full(T_TR, paths.get('G_over_Y', 0.13) * meanY)
    I_g_path = np.full(T_TR, paths.get('I_g_over_Y', 0.03) * meanY)
    defense_path = np.full(T_TR, paths.get('defense_over_Y', 0.0) * meanY)
    other_path = np.full(T_TR, paths.get('other_net_spending_over_Y', 0.0) * meanY)
    print(f"  baseline sim (n_sim={N_SIM}, recompute_bequests={recompute_bequests}) ...",
          flush=True)
    res = economy.simulate_transition(
        r_path=r_path, govt_spending_path=G_path, I_g_path=I_g_path,
        defense_spending_path=defense_path, other_net_spending_path=other_path,
        n_sim=N_SIM, verbose=recompute_bequests,
        recompute_bequests=recompute_bequests, **tax_paths)
    Yp = np.asarray(res['Y'])
    budget = economy.compute_government_budget_path(n_sim=N_SIM, verbose=False)
    out = {'Y': Yp,
           'bequest_converged': getattr(economy, '_bequest_converged', None),
           'bequest_iters': getattr(economy, '_bequest_iter_count', None),
           'bequest_transfers_over_Y':
               np.asarray(budget['bequest_transfers']) / Yp}
    for ss_key, b_key in KEYS:
        out[ss_key] = np.asarray(budget[b_key]) / Yp
    return out

print("\n" + "=" * 70)
print("(B) Transition baseline — recompute_bequests=False (open circuit)")
print("=" * 70)
tr_off = run_transition(False)

print("\n" + "=" * 70)
print("(C) Transition baseline — recompute_bequests=True (closed circuit)")
print("=" * 70)
tr_on = run_transition(True)
print(f"bequest loop: converged={tr_on['bequest_converged']}, "
      f"iters={tr_on['bequest_iters']}")

# ----------------------------------------------------------------------------
# Decomposition table
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DECOMPOSITION: SS (A) vs transition off (B) vs transition on (C), t=0")
print("=" * 70)
print(f"{'item':24s} {'SS(A)':>9s} {'off(B)':>9s} {'on(C)':>9s} "
      f"{'B-A %':>8s} {'C-B %':>8s} {'C-A %':>8s}")
for ss_key, _ in KEYS:
    a = ss.get(ss_key, float('nan'))
    b = tr_off[ss_key][0]
    c = tr_on[ss_key][0]
    pba = 100 * (b - a) / a if abs(a) > 1e-12 else float('nan')
    pcb = 100 * (c - b) / b if abs(b) > 1e-12 else float('nan')
    pca = 100 * (c - a) / a if abs(a) > 1e-12 else float('nan')
    print(f"{ss_key:24s} {a:>9.4f} {b:>9.4f} {c:>9.4f} "
          f"{pba:>+8.1f} {pcb:>+8.1f} {pca:>+8.1f}")

print(f"\n{'Y level':24s} {ss['Y']:>9.4f} {tr_off['Y'][0]:>9.4f} "
      f"{tr_on['Y'][0]:>9.4f} "
      f"{100*(tr_off['Y'][0]-ss['Y'])/ss['Y']:>+8.1f} "
      f"{100*(tr_on['Y'][0]-tr_off['Y'][0])/tr_off['Y'][0]:>+8.1f} "
      f"{100*(tr_on['Y'][0]-ss['Y'])/ss['Y']:>+8.1f}")
print(f"bequest_transfers/Y t=0: off={tr_off['bequest_transfers_over_Y'][0]:.4f} "
      f"on={tr_on['bequest_transfers_over_Y'][0]:.4f}")

print("\n--- path means (periods 0..-1) ---")
print(f"{'item':24s} {'off mean':>9s} {'on mean':>9s}")
for ss_key, _ in KEYS:
    print(f"{ss_key:24s} {tr_off[ss_key].mean():>9.4f} {tr_on[ss_key].mean():>9.4f}")
print(f"{'Y mean':24s} {tr_off['Y'].mean():>9.4f} {tr_on['Y'].mean():>9.4f}")

print("\nDONE")
