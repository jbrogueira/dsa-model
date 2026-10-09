"""Figure and table comparing the three output-tax rules for the report.

Reads, for each rule, the baseline_paths.npz and baseline_closure.npz that
fill_report.py --run-baseline wrote (run_tau_rules_2026-10-09.sh), and writes
tau_rules.pdf and tau_rules_body.tex in the same folder. No solve.

Usage (from code/):
    python reports/tau_rules_figure.py [--root output/tau_rules_2026-10-09]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline_figures import SERIES, INK, INK2, _style  # noqa: E402

RULES = (('ramp', 'A: linear to the 2060 rate'),
         ('pinned', 'B: the 2023 rate throughout'),
         ('const', 'C: one rate from 2023'))


def load(root, rule):
    d = os.path.join(root, rule)
    P = np.load(os.path.join(d, 'baseline_paths.npz'))
    Z = np.load(os.path.join(d, 'baseline_closure.npz'))
    years = np.asarray(Z['years'], int)
    Y, G = np.asarray(P['Y'], float), np.asarray(P['growth_factor'], float)
    n = len(years)
    growth = np.full(n, np.nan)
    growth[1:] = G[1:n] * Y[1:n] / Y[:n - 1] - 1.0
    return {'years': years, 'tau': np.asarray(Z['tau_y_path'], float)[:n],
            'debt': np.asarray(Z['debt'], float), 'pb': np.asarray(Z['primary_balance'], float),
            'debt_proj': np.asarray(Z['debt_projection'], float),
            'pb_proj': np.asarray(Z['primary_balance_projection'], float),
            'growth': growth, 'Y': Y[:n], 'L': np.asarray(P['L'], float)[:n],
            'C': np.asarray(P['C'], float)[:n]}


def figure(R, out_pdf, last_year=2070):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.8))
    ref = R[RULES[0][0]]
    x = ref['years']
    n = int(np.searchsorted(x, last_year)) + 1
    panels = (
        (ax[0, 0], lambda r: 100 * r['tau'], 'Output tax rate, %', False),
        (ax[0, 1], lambda r: 100 * r['debt'], 'Debt / output, end of year, %', True),
        (ax[0, 2], lambda r: 100 * r['pb'], 'Primary balance / output, %', True),
        (ax[1, 0], lambda r: 100 * r['growth'], 'Output growth, %', False),
        (ax[1, 1], lambda r: 100 * (r['L'] / r['L'][0] - 1), 'Labour input, % from 2023', False),
        (ax[1, 2], lambda r: 100 * (r['C'] / r['C'][0] - 1), 'Consumption, % from 2023', False),
    )
    for a, f, title, proj in panels:
        for i, (k, lab) in enumerate(RULES):
            a.plot(x[:n], f(R[k])[:n], color=SERIES[i], lw=1.5, label=lab)
        if proj:
            key = 'debt_proj' if 'Debt' in title else 'pb_proj'
            has = ~np.isnan(ref[key][:n])
            a.plot(x[:n][has], 100 * ref[key][:n][has], color=INK2, lw=1.3, ls='--',
                   label='DSM forecast (ESM)')
        if title.startswith(('Primary', 'Labour', 'Consumption')):
            a.axhline(0.0, color=INK2, lw=0.6, zorder=1)
        a.legend(frameon=False, fontsize=7, labelcolor=INK, handlelength=1.8)
        _style(a, title)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out_pdf)
    plt.close(fig)
    print('  wrote', os.path.basename(out_pdf))


def table(R):
    yrs = (2030, 2040, 2050, 2060, 2070, 2100)
    ref = R[RULES[0][0]]
    at = lambda r, key, y: r[key][int(np.searchsorted(r['years'], y))]
    span = lambda r, key: float(np.nanmean(r[key][int(np.searchsorted(r['years'], 2026)):
                                                  int(np.searchsorted(r['years'], 2060)) + 1]))
    rows = []
    for k, lab in RULES:
        r = R[k]
        rows.append(f'{lab} & {100 * at(r, "tau", 2023):.2f} & {100 * at(r, "tau", 2060):.2f} & '
                    + ' & '.join(f'{100 * at(r, "debt", y):.0f}' for y in yrs)
                    + f' & {100 * span(r, "pb"):.2f} & {100 * span(r, "growth"):.2f}'
                    + f' & {at(r, "Y", 2023):.3f} \\\\')
    proj = ' & '.join('{--}' if np.isnan(at(ref, 'debt_proj', y)) else f'{100 * at(ref, "debt_proj", y):.0f}'
                      for y in yrs)
    pbp = span(ref, 'pb_proj')
    rows.append(f'DSM forecast (ESM) & {{--}} & {{--}} & {proj} & {100 * pbp:.2f} & {{--}} & {{--}} \\\\')
    head = (' & \\multicolumn{2}{c}{$\\tau_y$, \\%} & \\multicolumn{6}{c}{Debt / output, \\%} '
            '& {Balance} & {Growth} & {Output} \\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-9}\n'
            ' & {2023} & {2060} & ' + ' & '.join(f'{{{y}}}' for y in yrs)
            + ' & {2026--60} & {2026--60} & {2023} \\\\')
    return ('\\begin{tabular}{@{}l' + 'c' * 11 + '@{}}\n\\toprule\n' + head + '\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(HERE, '..', 'output', 'tau_rules_2026-10-09'))
    args = ap.parse_args()
    R = {k: load(args.root, k) for k, _ in RULES}
    figure(R, os.path.join(args.root, 'tau_rules.pdf'))
    with open(os.path.join(args.root, 'tau_rules_body.tex'), 'w') as fh:
        fh.write(table(R))
    print('  wrote tau_rules_body.tex')
    for k, lab in RULES:
        r = R[k]
        print(f'  {lab}: tau 2023 {100 * r["tau"][0]:.2f}, 2030 {100 * r["tau"][7]:.2f}, '
              f'2060 {100 * r["tau"][37]:.2f}; L 2026/2025 {100 * (r["L"][3] / r["L"][2] - 1):+.2f}%, '
              f'C 2026/2025 {100 * (r["C"][3] / r["C"][2] - 1):+.2f}%')


if __name__ == '__main__':
    main()
