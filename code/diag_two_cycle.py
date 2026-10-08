"""Rerun the baseline transition under one switch at a time, to locate the
two-year alternation in labour input. The lump-sum and output-tax paths are
the baseline's (baseline_paths.npz), so no fixed point is solved.

Variants: base | const_u (unemployment at its 2023 rate) | const_ret (retirement
age constant, no cohort table) | no_split (each cohort retires at its larger
part's age) | no_pindex (pension index constant) | no_lump (no lump-sum
transfer) | smooth_lump (five-year moving average of the baseline's lump-sum
path) | const_lump (the 2023 level throughout).

Usage (from code/): python3 diag_two_cycle.py --variant base --out output/diag/base.npz
"""
import argparse
import os

import numpy as np

from calibrate import load_config, build_olg_transition

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='calibration_input_GR.json')
ap.add_argument('--baseline', default='output/calibration_growth/baseline_paths.npz')
ap.add_argument('--variant', default='base')
ap.add_argument('--backend', default='jax')
ap.add_argument('--n-sim', type=int, default=2000)
ap.add_argument('--out', required=True)
args = ap.parse_args()

L = load_config(args.config)
cfg = L['config_data']
tr = cfg.setdefault('transition', {})
if args.variant == 'const_u':
    tr.pop('unemployment_index_file', None)
elif args.variant == 'const_ret':
    tr.pop('retirement_age_file', None)
elif args.variant == 'no_pindex':
    tr.pop('pension_index_file', None)
economy, tp, T_TR = build_olg_transition(cfg, backend=args.backend)
if args.variant == 'no_split':
    for k, parts in list(economy.cohort_retirement.items()):
        J, lam, sh = max(parts, key=lambda p: p[2])
        economy.cohort_retirement[k] = [(J, lam, 1.0)]

base = np.load(args.baseline, allow_pickle=True)
lump = np.asarray(base['lump_sum_path'], float)[:T_TR]
tau = np.asarray(base['tau_y_path'], float)[:T_TR]
if args.variant == 'no_lump':
    lump = np.zeros_like(lump)
elif args.variant == 'smooth_lump':      # centred five-year moving average of the baseline's path
    pad = np.pad(lump, (2, 2), mode='edge')
    lump = np.convolve(pad, np.ones(5) / 5, mode='valid')
elif args.variant == 'const_lump':       # the baseline's 2023 level throughout
    lump = np.full_like(lump, lump[0])
prod = cfg['production']
I_g = (prod.get('delta_g', 0.05) + economy.growth_factors(T_TR) - 1.0) * prod.get('K_g', 0.0)
tax = {k: tp[k] for k in ('tau_c_path', 'tau_l_path', 'tau_p_path', 'tau_k_path', 'pension_replacement_path')}
fisc = cfg.get('fiscal', {})
r_B_full = (np.asarray(tp['r_B_path'], float) if tp.get('r_B_path') is not None
            else np.full(T_TR, float(cfg['prices']['r_B'])))
res = economy.simulate_transition(r_path=tp['r_path'], I_g_path=I_g, n_sim=args.n_sim, verbose=False,
                                  G_over_Y=fisc.get('G_over_Y', 0.0), defense_over_Y=fisc.get('defense_over_Y', 0.0),
                                  tau_y_path=tau, lump_sum_path=lump,
                                  education_over_Y0=tp.get('education_over_Y0', 0.0),
                                  education_index_path=tp.get('education_index_path'),
                                  foreign_transfer_over_Y=tp.get('foreign_transfer_over_Y'),
                                  unemployment_index_path=tp.get('unemployment_index_path'),
                                  r_B_path=r_B_full, **tax)
bud = economy.compute_government_budget_path(n_sim=args.n_sim, verbose=False)
out = {k: np.asarray(res[k], float) for k in ('Y', 'L', 'C', 'K', 'A') if k in res}
out['pension'] = np.asarray(bud['pension'], float)
# Labour by age and the cohort weights, per period, from the period cache.
pv = getattr(economy, '_policy_version', 0)
lab = {}; wts = {}
for (t, ns, sb, v), rec in getattr(economy, '_period_cache', {}).items():
    if ns == args.n_sim and v == pv:
        sh = np.asarray(rec['education_shares_array'], float)
        lab[t] = (sh[:, None] * np.asarray(rec['labor_by_age_edu'], float)).sum(0)
        wts[t] = np.asarray(rec['cohort_sizes_t'], float)
if lab:
    ts = sorted(lab)
    out['labor_by_age'] = np.array([lab[t] for t in ts]); out['weights_by_age'] = np.array([wts[t] for t in ts]); out['periods'] = np.array(ts)
out['base_year'] = int(economy.current_year)
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
np.savez(args.out, **out)
Lp = out['L']; by = out['base_year']; dl = np.diff(np.log(Lp))
print(f'variant {args.variant}: T_TR={T_TR}, L[0]={Lp[0]:.4f}')
for a, b in ((2024, 2033), (2034, 2043), (2044, 2053), (2054, 2063), (2074, 2083), (2084, 2093), (2094, 2103)):
    x = dl[a - by - 1:b - by]
    print(f'  dlog L {a}-{b}: std {100 * x.std():.2f}%  ac1 {np.corrcoef(x[:-1], x[1:])[0, 1]:+.2f}')
