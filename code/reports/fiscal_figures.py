"""Figures and table of the policy experiments for the calibration report.

Reads ``fiscal_results.json`` (written by ``run_fiscal_figures.py``) and
writes, in the same folder, ``fiscal_G.pdf`` and ``fiscal_Ig.pdf`` (output,
consumption, debt and net foreign assets under each financing rule) and
``fiscal_body.tex`` (the table of responses). No transition is solved here.

Debt is the stock at the end of the year over the same year's output,
Gamma_t B_{t+1} / Y_t, the convention of the baseline figures; the results
file stores B_t / Y_t.

Usage (from code/):
    python reports/fiscal_figures.py [--results output/fiscal_2026-10-06_ret_annual/fiscal_results.json]
                                     [--baseline output/calibration_growth/baseline_paths.npz]
                                     [--last-year 2100]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_figures import SERIES, INK, INK2, _style  # noqa: E402

SCENARIOS = [('debt_financed', 'debt-financed'),
             ('tax_financed', r'$\tau_l$, debt-ratio target'),
             ('nfa_constrained', r'$\tau_l$, net-foreign-asset target')]
SHOCKS = {'G': 'Government consumption', 'Ig': 'Public investment'}


def load(results, baseline):
    R = json.load(open(results))
    m = np.load(baseline, allow_pickle=True)
    G = np.asarray(m['growth_factor'], float)
    base_year = int(m['base_year'])
    return R, G, base_year


def end_of_year_debt(s, G):
    """Gamma_t B_{t+1} / Y_t from the stored B_t / Y_t."""
    B = np.asarray(s['B_gdp_path'], float)
    Y = np.asarray(s['counterfactual']['Y'], float)
    n = len(Y) - 1
    g = np.r_[G, np.full(max(0, n - len(G)), G[-1])][:n]
    return g * B[1:n + 1] * Y[1:n + 1] / Y[:n]


def dev(s, key):
    b = np.asarray(s['baseline'][key], float)
    c = np.asarray(s['counterfactual'][key], float)
    return 100.0 * (c / b - 1.0)


def shock_figure(R, G, base_year, shock, out_pdf, plt, last_year):
    S = R[shock]
    n = last_year - base_year + 1
    x = base_year + np.arange(n)
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.6))

    def lines(a, series, title, zero=True, loc='best'):
        for i, (lab, y) in enumerate(series):
            a.plot(x, y[:n], color=SERIES[i], lw=1.5, label=lab)
        if zero:
            a.axhline(0.0, color=INK2, lw=0.6, zorder=1)
        a.legend(frameon=False, fontsize=7.5, labelcolor=INK, handlelength=1.8, loc=loc)
        _style(a, title)

    lines(ax[0, 0], [(lab, dev(S[k], 'Y')) for k, lab in SCENARIOS],
          'Output, % deviation from the baseline')
    lines(ax[0, 1], [(lab, dev(S[k], 'C')) for k, lab in SCENARIOS],
          'Consumption, % deviation from the baseline')
    base = S['baseline']
    lines(ax[1, 0], [('baseline', end_of_year_debt(base, G))]
          + [(lab, end_of_year_debt(S[k], G)) for k, lab in SCENARIOS],
          'Debt / output, end of year', zero=False, loc='upper left')
    nfa = lambda s: (np.asarray(s['counterfactual']['NFA'], float)
                     / np.asarray(s['counterfactual']['Y'], float))
    lines(ax[1, 1], [('baseline', nfa(base))] + [(lab, nfa(S[k])) for k, lab in SCENARIOS],
          'Net foreign assets / output', loc='upper right')
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


def cumulative_multiplier(S, G, key):
    """Sum of the output response over the sum of the spending increase, both
    as aggregate flows (the detrended per-capita paths re-trended by Gamma)."""
    s = S['debt_financed']
    dY = np.asarray(s['counterfactual']['Y'], float) - np.asarray(s['baseline']['Y'], float)
    dG = np.asarray(s['cf_budget'][key], float) - np.asarray(s['base_budget'][key], float)
    g = np.r_[G, np.full(max(0, len(dY) - len(G)), G[-1])][:len(dY)]
    scale = np.concatenate([[1.0], np.cumprod(g[:-1])])
    return float((scale * dY).sum() / (scale * dG).sum())


def table(R, G, base_year):
    def fmt(v, places=1, sign=False):
        return f'{v:+.{places}f}' if sign else f'{v:.{places}f}'
    rows = []
    for shock, name in SHOCKS.items():
        S = R[shock]
        rows.append(f'\\multicolumn{{9}}{{l}}{{\\itshape {name}}}\\\\')
        for k, lab in SCENARIOS:
            s = S[k]
            t = lambda y: y - base_year
            dY, dC = dev(s, 'Y'), dev(s, 'C')
            d = end_of_year_debt(s, G)
            nfa = (np.asarray(s['counterfactual']['NFA'], float)
                   / np.asarray(s['counterfactual']['Y'], float))
            adj = s.get('adjustment_scalar')
            cells = ['{--}' if adj is None else fmt(100 * adj, 2, True),
                     fmt(dY[t(2023)], 1, True), fmt(dY[t(2023):t(2033)].mean(), 1, True),
                     fmt(dY[t(2050)], 1, True), fmt(dY[t(2100)], 1, True),
                     fmt(dC[t(2100)], 1, True),
                     fmt(100 * d[t(2060)], 0), fmt(100 * d[t(2200)], 0),
                     fmt(nfa[t(2200)], 2, True)]
            rows.append(f'\\quad {lab} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--results',
                    default=os.path.join(here, '..', 'output', 'fiscal_2026-10-06_ret_annual', 'fiscal_results.json'))
    ap.add_argument('--baseline',
                    default=os.path.join(here, '..', 'output', 'calibration_growth', 'baseline_paths.npz'))
    ap.add_argument('--last-year', type=int, default=2100)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R, G, base_year = load(args.results, args.baseline)
    outdir = os.path.dirname(os.path.abspath(args.results))
    for shock in SHOCKS:
        shock_figure(R, G, base_year, shock, os.path.join(outdir, f'fiscal_{shock}.pdf'),
                     plt, args.last_year)
    body = table(R, G, base_year)
    colspec = ('@{}l' + 'S[table-format=+1.2]' + 'S[table-format=+1.1]' * 5
               + 'S[table-format=+3.0]' * 2 + 'S[table-format=+1.2]@{}')
    head = (' & {$\\Delta\\tau_l$, pp} & \\multicolumn{4}{c}{Output, \\% from baseline}'
            ' & {$C$, \\%} & \\multicolumn{2}{c}{Debt / output, \\%} & {$NFA/Y$} \\\\\n'
            '\\cmidrule(lr){3-6}\\cmidrule(lr){8-9}\n'
            ' & & {2023} & {2023--32} & {2050} & {2100} & {2100} & {2060} & {2200} & {2200} \\\\')
    with open(os.path.join(outdir, 'fiscal_body.tex'), 'w') as fh:
        fh.write('\\begin{tabular}{' + colspec + '}\n\\toprule\n' + head
                 + '\n\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n')
    print('  wrote fiscal_body.tex')
    for shock, key in (('G', 'govt_spending'), ('Ig', 'public_investment')):
        print(f'  cumulative multiplier, {shock}: {cumulative_multiplier(R[shock], G, key):.3f}')


if __name__ == '__main__':
    main()
