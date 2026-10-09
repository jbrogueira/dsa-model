"""
Checks before the recalibration with the endogenous grid method
(docs/EGM_PLAN.md section 5), at the configuration's current theta.

  python egm_grid_check.py config --n-a 200 [--savings-solver grid] --out CFG.json
      writes calibration_input_GR.json with another grid size or solver
  python egm_grid_check.py moments --config CFG.json --out MOM.json
      targeted moments of the base-year cross-section, the mass of wealth
      above a = 20 and a = 40, and the run time of the cross-section
  python egm_grid_check.py compare --runs DIR_100 DIR_200 [DIR_grid] --out TABLE.md
      the statistics of section 5.2 from each run's fiscal_results.json and
      moments.json, and the rule for the grid size

Each run directory holds the fiscal_results.json of
run_fiscal_figures.py --shock Ig --scenarios debt and the moments.json above.
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROUGH_YEARS = (2030, 2100)
# section 5.2: largest difference between n_a = 100 and 200 for which the
# recalibration uses 100
TOL = {'multiplier': 0.01, 'dev_2060': 0.05, 'tau_y': 0.05, 'moment_rel': 0.01,
       'roughness': 0.005}


def make_config(args):
    raw = json.load(open(args.base))
    raw['model']['n_a'] = int(args.n_a)
    raw.setdefault('household', {})['savings_solver'] = args.savings_solver
    with open(args.out, 'w') as fh:
        json.dump(raw, fh, indent=2)
    print(f'wrote {args.out}: n_a = {args.n_a}, savings_solver = {args.savings_solver}')


def moments(args):
    from calibrate import load_config, theta_from_config, run_model_moments, _agent_weights
    L = load_config(args.config)
    spec = L['spec']
    theta = theta_from_config(L['config_data'], spec, verbose=False)
    t0 = time.time()
    m, panels = run_model_moments(theta, spec, return_panels=True)
    dt = time.time() - t0
    # Share of wealth held above a = 20 and 40 in the base-year cross-section,
    # with the weights the moments use (age weights, education shares, masses).
    above = {}
    for cut in (20.0, 40.0):
        num = den = 0.0
        for edu, panel in panels.items():
            sel, w = _agent_weights(panel, spec, edu)
            a = np.asarray(panel.a_sim, dtype=float)[sel]
            num += float((w * a * (a > cut)).sum())
            den += float((w * a).sum())
        above[f'wealth_share_above_{int(cut)}'] = num / den
    out = {'config': args.config, 'n_a': int(L['config_data']['model']['n_a']),
           'savings_solver': L['config_data'].get('household', {}).get('savings_solver', 'grid'),
           'seconds_cross_section': dt,
           'moments': {mom.name: {'model': float(v), 'data': float(mom.value)}
                       for mom, v in zip(spec.moments, m)}}
    out.update(above)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


def _run_stats(d):
    res = json.load(open(os.path.join(d, 'fiscal_results.json')))
    mom = json.load(open(os.path.join(d, 'moments.json')))
    p = res['params']
    base_year = int(p['base_year'])
    S = res['Ig']['debt_financed']
    Yb = np.asarray(S['baseline']['Y'], dtype=float)
    Yc = np.asarray(S['counterfactual']['Y'], dtype=float)
    Cb = np.asarray(S['baseline']['C'], dtype=float)
    Cc = np.asarray(S['counterfactual']['C'], dtype=float)
    years = base_year + np.arange(len(Yb))
    dev_Y = 100.0 * (Yc / Yb - 1.0)
    dev_C = 100.0 * (Cc - Cb) / Yb
    win = (years >= ROUGH_YEARS[0]) & (years <= ROUGH_YEARS[1])
    rough = {k: float(np.std(np.diff(v[win], n=2))) for k, v in (('Y', dev_Y), ('C', dev_C))}
    i60 = int(np.where(years == 2060)[0][0])
    Bb = np.asarray(res['Ig']['baseline']['B_gdp_path'], dtype=float)
    Bc = np.asarray(S['B_gdp_path'], dtype=float)
    tau = np.asarray(p['tau_y_path'], dtype=float)
    ty = base_year + np.arange(len(tau))
    tau_26_60 = float(np.mean(tau[(ty >= 2026) & (ty <= 2060)]))
    return {
        'n_a': mom['n_a'], 'savings_solver': mom['savings_solver'],
        'multiplier': S['multiplier'],
        'dev_Y_2060_pct': float(dev_Y[i60]),
        'dev_B_2060_pp': float(100.0 * (Bc[i60] - Bb[i60])),
        'tau_y_2026_60_pct': 100.0 * tau_26_60,
        'roughness_pp': rough,
        'moments': mom['moments'],
        'wealth_share_above_20': mom['wealth_share_above_20'],
        'wealth_share_above_40': mom['wealth_share_above_40'],
        'seconds_cross_section': mom['seconds_cross_section'],
    }


def compare(args):
    stats = [_run_stats(d) for d in args.runs]
    lines = ['| statistic | ' + ' | '.join(f"{s['savings_solver']} n_a={s['n_a']}" for s in stats) + ' |',
             '|---' * (len(stats) + 1) + '|']

    def row(label, vals, fmt='{:.4f}'):
        lines.append(f'| {label} | ' + ' | '.join(fmt.format(v) for v in vals) + ' |')
    for k in ('impact', 'cumulative_10y', 'cumulative_horizon'):
        row(f'I_g multiplier, {k}', [s['multiplier'][k] for s in stats])
    row('output deviation 2060, % of baseline', [s['dev_Y_2060_pct'] for s in stats])
    row('debt deviation 2060, pp of output', [s['dev_B_2060_pp'] for s in stats])
    row('baseline output tax 2026-60, %', [s['tau_y_2026_60_pct'] for s in stats])
    row('roughness of the output deviation, pp', [s['roughness_pp']['Y'] for s in stats], '{:.5f}')
    row('roughness of the consumption deviation, pp', [s['roughness_pp']['C'] for s in stats], '{:.5f}')
    for name in stats[0]['moments']:
        row(f"{name} (data {stats[0]['moments'][name]['data']})",
            [s['moments'][name]['model'] for s in stats])
    row('wealth share above a = 20', [s['wealth_share_above_20'] for s in stats])
    row('wealth share above a = 40', [s['wealth_share_above_40'] for s in stats])
    row('cross-section run time, s', [s['seconds_cross_section'] for s in stats], '{:.1f}')

    egm = {s['n_a']: s for s in stats if s['savings_solver'] == 'egm'}
    verdict = []
    if 100 in egm and 200 in egm:
        a, b = egm[100], egm[200]
        checks = {
            'multiplier': max(abs(a['multiplier'][k] - b['multiplier'][k])
                              for k in ('impact', 'cumulative_10y', 'cumulative_horizon'))
            < TOL['multiplier'],
            'dev_2060': max(abs(a['dev_Y_2060_pct'] - b['dev_Y_2060_pct']),
                            abs(a['dev_B_2060_pp'] - b['dev_B_2060_pp'])) < TOL['dev_2060'],
            'tau_y': abs(a['tau_y_2026_60_pct'] - b['tau_y_2026_60_pct']) < TOL['tau_y'],
            'moments': all(abs(a['moments'][k]['model'] - b['moments'][k]['model'])
                           < TOL['moment_rel'] * abs(b['moments'][k]['model'])
                           for k in a['moments']),
            'roughness': max(max(s['roughness_pp'].values()) for s in (a, b)) < TOL['roughness'],
        }
        verdict = ['', 'Section 5.2 rule (EGM, n_a = 100 against 200):']
        verdict += [f'- {k}: {"holds" if v else "fails"}' for k, v in checks.items()]
        verdict.append(f"- recalibrate at n_a = {100 if all(checks.values()) else 200}")
    text = '\n'.join(lines + verdict) + '\n'
    with open(args.out, 'w') as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('config')
    c.add_argument('--base', default=os.path.join(HERE, 'calibration_input_GR.json'))
    c.add_argument('--n-a', type=int, required=True)
    c.add_argument('--savings-solver', choices=('grid', 'egm'), default='egm')
    c.add_argument('--out', required=True)
    m = sub.add_parser('moments')
    m.add_argument('--config', required=True)
    m.add_argument('--out', required=True)
    k = sub.add_parser('compare')
    k.add_argument('--runs', nargs='+', required=True)
    k.add_argument('--out', required=True)
    args = ap.parse_args()
    {'config': make_config, 'moments': moments, 'compare': compare}[args.cmd](args)


if __name__ == '__main__':
    main()
