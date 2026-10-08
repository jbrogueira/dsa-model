"""Numbers the calibration report's text cites, from the saved runs (no solve).

Reads the baseline_paths.npz of the report baseline, of scenario 1 (the 2023
output-tax rate throughout) and of the constant-unemployment baseline, and
prints the statistics the baseline and scenario sections quote: output and
growth, the budget lines, the debt path, and the debt ratio of 2060 with the
projection's primary balance or growth in place of the model's.

Usage (from code/): python3 reports/report_numbers.py [--config calibration_input_GR.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import baseline_closure as bc  # noqa: E402

OUT = os.path.join(HERE, '..', 'output')


def load(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def debt_with(P, raw, dsa, pb_proj=False, growth_proj=False, first=2026, last=2060):
    """Debt ratio path with the projection's primary balance and/or growth in
    place of the model's over first..last; everything else the model's."""
    by = int(P['base_year'])
    Y = np.asarray(P['Y'], float).copy()
    G = np.asarray(P['growth_factor'], float)
    rev, spend = np.asarray(P['budget_total_revenue'], float), np.asarray(P['budget_total_spending'], float)
    pb_ratio = (rev - spend) / Y
    years = by + np.arange(len(Y))
    proj = {int(y): i for i, y in enumerate(dsa['years'])}
    if pb_proj:
        for t, y in enumerate(years):
            if first <= y <= last:
                pb_ratio[t] = float(dsa['primary_balance'][proj[y]]) / 100.0 \
                    if abs(float(dsa['primary_balance'][proj[y]])) > 0.5 else float(dsa['primary_balance'][proj[y]])
    if growth_proj:
        g_scale = 100.0 if np.nanmax(np.abs(dsa['real_growth'])) > 0.5 else 1.0
        for t, y in enumerate(years):
            if t > 0 and first <= y <= last:
                Y[t] = Y[t - 1] * (1.0 + float(dsa['real_growth'][proj[y]]) / g_scale) / G[t - 1]
    out = bc.debt_paths(years, Y, G, pb_ratio * Y, P['r_B_path'],
                        float(raw['fiscal']['B_over_Y']), dsa)
    return years, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, '..', 'calibration_input_GR.json'))
    ap.add_argument('--root', default=OUT)
    ap.add_argument('--fiscal', default=None, help='fiscal_results.json of the policy experiments')
    args = ap.parse_args()
    raw = json.load(open(args.config))
    dsa = bc.load_dsa_projection(raw)
    runs = {'baseline': 'calibration_growth', 'pinned': 'calibration_growth_tau_pinned',
            'constant_u': 'calibration_growth_constant_u'}
    data = {k: load(os.path.join(args.root, v, 'baseline_paths.npz')) for k, v in runs.items()}
    pct = lambda x: 100.0 * x

    for name, P in data.items():
        if P is None:
            print(f'== {name}: missing')
            continue
        by = int(P['base_year'])
        T = len(P['Y'])
        years = by + np.arange(T)
        ix = lambda y: int(y - by)
        Y = np.asarray(P['Y'], float)
        G = np.asarray(P['growth_factor'], float)
        cl = bc.debt_from_run(P, raw, dsa=dsa)
        d, pb, g = cl['debt'], cl['primary_balance'], cl['growth']
        tau = np.asarray(P['tau_y_path'], float)
        bud = lambda k: np.asarray(P['budget_' + k], float) / Y
        print(f'\n== {name} (base year {by})')
        print('  tau_y: 2023 %.4f  2025 %.4f  2026 %.4f  2060 %.4f  2100 %.4f'
              % tuple(tau[ix(y)] for y in (2023, 2025, 2026, 2060, 2100)))
        print('  debt %:', ' '.join('%d:%.1f' % (y, pct(d[ix(y)])) for y in (2025, 2030, 2040, 2050, 2060, 2070, 2080, 2100, 2150, 2200) if ix(y) < T))
        neg = [y for y in years[ix(2026):] if d[ix(y)] < 0]
        print('  debt first below zero:', neg[0] if neg else None)
        print('  primary balance %:', ' '.join('%d:%.2f' % (y, pct(pb[ix(y)])) for y in (2023, 2025, 2030, 2040, 2050, 2060, 2070, 2080, 2100)))
        w = slice(ix(2026), ix(2060) + 1)
        print('  balance 2026-60: min %.2f max %.2f mean %.2f; projection mean %.2f'
              % (pct(pb[w].min()), pct(pb[w].max()), pct(pb[w].mean()), pct(np.nanmean(cl['primary_balance_projection'][w]))))
        print('  sfa 2024 %.2f 2025 %.2f 2026 %.2f' % tuple(pct(cl['sfa'][ix(y)]) for y in (2024, 2025, 2026)))
        print('  output per person, 2023 = 1:', ' '.join('%d:%.3f' % (y, Y[ix(y)] / Y[0]) for y in (2025, 2026, 2030, 2040, 2050, 2055, 2060, 2070, 2100)))
        lo = ix(2026) + int(np.argmin(Y[ix(2026):ix(2100)]))
        print('  output trough after 2026: %d at %.3f' % (years[lo], Y[lo] / Y[0]))
        print('  output 2025 vs 2023 %+.2f%%, 2030 vs 2025 %+.2f%%' % (pct(Y[ix(2025)] / Y[0] - 1), pct(Y[ix(2030)] / Y[ix(2025)] - 1)))
        for a, b in ((2026, 2060), (2026, 2030), (2031, 2040), (2041, 2050), (2051, 2060)):
            print('  growth %d-%d mean %.2f%%' % (a, b, pct(np.mean(g[ix(a):ix(b) + 1]))))
        if 'w' in P:
            w_ = np.asarray(P['w'], float)
            kdy = np.asarray(P['K_domestic'], float) / Y
            print('  wage 2026/2025 %+.2f%%, K_dom/Y 2026/2025 %+.2f%%' % (pct(w_[ix(2026)] / w_[ix(2025)] - 1), pct(kdy[ix(2026)] / kdy[ix(2025)] - 1)))
        lines = ['tax_c', 'tax_l', 'tax_p', 'tax_k', 'bequest_tax', 'tax_y', 'foreign_transfer', 'total_revenue',
                 'pension', 'gov_health', 'ui', 'transfers', 'govt_spending', 'defense_spending',
                 'public_investment', 'education', 'lump_sum', 'total_spending']
        for y in (2023, 2025, 2030, 2040, 2050, 2051, 2060, 2070, 2100):
            print('  %d: ' % y + ' '.join('%s %.2f' % (k, pct(bud(k)[ix(y)])) for k in lines))
        pen = bud('pension')
        pk = ix(2026) + int(np.argmax(pen[ix(2026):ix(2080)]))
        print('  pensions peak %d at %.2f' % (years[pk], pct(pen[pk])))
        edu = bud('education')[ix(2023):ix(2100) + 1]
        print('  education range 2023-2100 %.2f-%.2f' % (pct(edu.min()), pct(edu.max())))
        print('  investment/Y 2023 %.2f, Gamma_T/Gamma_0 rise of (delta_g+Gamma-1) %.1f%%'
              % (pct(bud('public_investment')[0]),
                 pct((float(raw['production']['delta_g']) + G[-1] - 1) / (float(raw['production']['delta_g']) + G[0] - 1) - 1)))
        if dsa is not None:
            for lab, kw in (('projection balance', dict(pb_proj=True)), ('projection growth', dict(growth_proj=True)),
                            ('both', dict(pb_proj=True, growth_proj=True))):
                yrs, o = debt_with(P, raw, dsa, **kw)
                print('  2060 debt with %s: %.1f' % (lab, pct(o['debt'][ix(2060)])))
    P, Q = data['baseline'], data['constant_u']
    if P is not None and Q is not None:
        by = int(P['base_year']); ix = lambda y: int(y - by)
        Yb, Yc = np.asarray(P['Y'], float), np.asarray(Q['Y'], float)
        print('\n== constant unemployment against the baseline')
        print('  output ratio:', ' '.join('%d:%+.2f%%' % (y, pct(Yc[ix(y)] / Yb[ix(y)] - 1)) for y in (2026, 2030, 2040, 2050, 2060)))
        ub = np.asarray(P['budget_ui'], float) / Yb; uc = np.asarray(Q['budget_ui'], float) / Yc
        print('  UI/Y difference:', ' '.join('%d:%+.2f' % (y, pct(uc[ix(y)] - ub[ix(y)])) for y in (2026, 2030, 2040, 2050, 2060)))
    P = data['pinned']
    if P is not None:
        by = int(P['base_year']); ix = lambda y: int(y - by)
        Y = np.asarray(P['Y'], float)
        print('\n== scenario 1 revenue by line, 2023 and 2050 (% of output)')
        for k in ('total_revenue', 'tax_c', 'tax_l', 'tax_p', 'tax_k', 'bequest_tax', 'tax_y', 'foreign_transfer', 'total_spending'):
            a = np.asarray(P['budget_' + k], float) / Y
            print('  %-16s %.2f -> %.2f (%+.2f)' % (k, pct(a[0]), pct(a[ix(2050)]), pct(a[ix(2050)] - a[0])))

    if args.fiscal:
        experiment_numbers(args.fiscal, os.path.join(args.root, 'calibration_growth', 'baseline_paths.npz'))


def experiment_numbers(results, baseline):
    """Statistics of the policy experiments quoted in the text."""
    import fiscal_figures as ff
    R, G, by = ff.load(results, baseline)
    t = lambda y: int(y - by)
    pct = lambda x: 100.0 * x
    base = R['G']['baseline']
    db = ff.end_of_year_debt(base, G)
    nfa_b = np.asarray(base['counterfactual']['NFA'], float) / np.asarray(base['counterfactual']['Y'], float)
    tau = np.asarray(R['params']['tau_y_path'], float)
    print('\n== experiments: their baseline')
    print('  tau_y 2025 %.4f 2026 %.4f' % (tau[t(2025)], tau[t(2026)]))
    print('  debt %:', ' '.join('%d:%.0f' % (y, pct(db[t(y)])) for y in (2060, 2070, 2080, 2100, 2200)))
    print('  NFA/Y 2200 %.2f; debt change 2070-80 %.0f points' % (nfa_b[t(2200)], pct(db[t(2080)] - db[t(2070)])))
    for shock in ('G', 'Ig'):
        S = R[shock]
        key = 'govt_spending' if shock == 'G' else 'public_investment'
        print(f'\n== {shock}')
        print('  cumulative multiplier, horizon %.2f' % ff.cumulative_multiplier(S, G, key))
        s = S['debt_financed']
        dY = np.asarray(s['counterfactual']['Y'], float) - np.asarray(s['baseline']['Y'], float)
        dX = np.asarray(s['cf_budget'][key], float) - np.asarray(s['base_budget'][key], float)
        g = np.r_[G, np.full(max(0, len(dY) - len(G)), G[-1])][:len(dY)]
        sc = np.concatenate([[1.0], np.cumprod(g[:-1])])
        w = slice(t(2023), t(2033))
        print('  cumulative multiplier, 2023-32 %.2f' % ((sc[w] * dY[w]).sum() / (sc[w] * dX[w]).sum()))
        if shock == 'Ig':
            kgy = np.asarray(s['counterfactual']['K_g'], float)
            print('  K_g per person over base-year output: 2023 %.2f 2050 %.2f 2100 %.2f'
                  % (kgy[t(2023)] / np.asarray(s['baseline']['Y'], float)[0] * 1.0,
                     kgy[t(2050)] / np.asarray(s['baseline']['Y'], float)[0], kgy[t(2100)] / np.asarray(s['baseline']['Y'], float)[0]))
            print('  K_g level: 2023 %.3f 2050 %.3f 2100 %.3f' % (kgy[t(2023)], kgy[t(2050)], kgy[t(2100)]))
        for k in ('debt_financed', 'tax_financed', 'nfa_constrained'):
            s = S[k]
            dYp, dC = ff.dev(s, 'Y'), ff.dev(s, 'C')
            dL = ff.dev(s, 'L')
            d = ff.end_of_year_debt(s, G)
            adj = s.get('adjustment_scalar')
            first_small = next((y for y in range(2024, 2190) if abs(dYp[t(y)]) < 0.1 and all(abs(dYp[t(z)]) < 0.1 for z in range(y, 2150))), None)
            print('  %-16s dtau %s | Y 2023 %+.2f 2023-32 %+.2f 2050 %+.2f 2100 %+.2f | L 2023 %+.2f 2023-32 %+.2f | C 2023 %+.2f 2100 %+.2f | debt 2060 %.0f 2200 %.0f (base %.0f, diff %+.0f) | |dY|<0.1 from %s'
                  % (k, '--' if adj is None else '%.2f' % pct(adj), dYp[t(2023)], dYp[w].mean(), dYp[t(2050)], dYp[t(2100)],
                     dL[t(2023)], dL[w].mean(), dC[t(2023)], dC[t(2100)], pct(d[t(2060)]), pct(d[t(2200)]),
                     pct(db[t(2200)]), pct(d[t(2200)] - db[t(2200)]), first_small))


if __name__ == '__main__':
    main()
