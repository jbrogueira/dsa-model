"""Figures and tables of the policy experiments for the calibration report.

Reads ``fiscal_results.json`` (written by ``run_fiscal_figures.py``) and
writes, in the same folder (or --out-dir), for each shock in the file:

- ``fiscal_<shock>.pdf``: output, consumption, debt, net foreign assets, the
  labour-tax rate and the shock's spending line under each financing scheme;
- with distributional outputs, ``fiscal_<shock>_age.pdf`` (means by age
  group, % deviation from the baseline), ``fiscal_<shock>_ineq.pdf`` (change
  in the inequality measures), ``fiscal_<shock>_welfare.pdf`` (the
  consumption-equivalent variation by year of birth and by income quintile);
- ``fiscal_<shock>_body.tex``: the table of the exercise (multipliers, the
  tax change, debt, inequality, welfare), one column per scheme;

and ``fiscal_body.tex``, the table of responses across shocks. No transition
is solved here. Shocks and schemes absent from the file are skipped.

Debt is the stock at the end of the year over the same year's output,
Gamma_t B_{t+1} / Y_t, the convention of the baseline figures; the results
file stores B_t / Y_t.

Usage (from code/):
    python reports/fiscal_figures.py [--results output/policy_2026-10-XX/fiscal_results.json]
                                     [--baseline output/calibration_growth/baseline_paths.npz]
                                     [--last-year 2070] [--out-dir DIR]
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
             ('tax_financed_window', r'$\tau_l$ over the cut, debt-ratio target'),
             ('nfa_constrained', r'$\tau_l$, net-foreign-asset target')]
DECOMPOSITION = [('debt_financed_kappa_only', 'coverage only'),
                 ('debt_financed_m_only', 'medical spending only')]
SHOCKS = {'G': 'Government consumption', 'Ig': 'Public investment',
          'health': 'Temporary cut in public health spending'}
AGE_VARS = [('consumption', 'Consumption'), ('hours', 'Hours of the employed'),
            ('labour_income', 'Labour income of the employed'),
            ('disp_income', 'Disposable income'), ('assets', 'Assets')]
INEQ_PANELS = [('disp_income', 'gini', 'Gini, disposable income'),
               ('disp_income_net_health', 'gini', 'Gini, disposable income less health'),
               ('consumption', 'gini', 'Gini, consumption'),
               ('labour_income', 'gini', 'Gini, labour income of the employed'),
               ('wealth', 'gini', 'Gini, wealth'),
               ('wealth', 'top10_share', 'Top-10% wealth share'),
               ('disp_income', 'p90_p10', 'P90/P10, disposable income'),
               ('consumption', 'p90_p10', 'P90/P10, consumption')]


def load(results, baseline):
    R = json.load(open(results))
    p = R.get('params', {})
    if baseline and os.path.exists(baseline):
        m = np.load(baseline, allow_pickle=True)
        G = np.asarray(m['growth_factor'], float)
        base_year = int(m['base_year'])
    else:
        G = np.asarray(p.get('growth_factor_path') or [1.0], float)
        base_year = int(p.get('base_year', 2023))
    return R, G, base_year


def present(S, pairs):
    return [(k, lab) for k, lab in pairs if k in S]


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


def ratio(s, side, key):
    src = s['cf_budget' if side == 'cf' else 'base_budget']
    mac = s['counterfactual' if side == 'cf' else 'baseline']
    return np.asarray(src[key], float) / np.asarray(mac['Y'], float)


def _lines(a, x, n, series, title, zero=True, loc='best'):
    """One line per series; the baseline in grey, the schemes in the palette
    in the order of SCENARIOS, so a scheme has the same colour in every panel."""
    i = 0
    for lab, y in series:
        if lab == 'baseline':
            a.plot(x, np.asarray(y)[:n], color=INK2, lw=1.2, ls='--', label=lab)
            continue
        a.plot(x, np.asarray(y)[:n], color=SERIES[i % len(SERIES)], lw=1.5, label=lab)
        i += 1
    if zero:
        a.axhline(0.0, color=INK2, lw=0.6, zorder=1)
    a.legend(frameon=False, fontsize=7, labelcolor=INK, handlelength=1.8, loc=loc)
    _style(a, title)


def _shock_year(R, base_year):
    p = R.get('params', {})
    return int(p.get('shock_year', base_year + int(p.get('shock_period', 0) or 0)))


def shock_figure(R, G, base_year, shock, out_pdf, plt, last_year):
    S = R[shock]
    scn = present(S, SCENARIOS)
    n = min(last_year - base_year + 1, len(S['baseline']['counterfactual']['Y']) - 1)
    x = base_year + np.arange(n)
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.8))
    _lines(ax[0, 0], x, n, [(lab, dev(S[k], 'Y')) for k, lab in scn],
           'Output, % deviation from the baseline')
    _lines(ax[0, 1], x, n, [(lab, dev(S[k], 'C')) for k, lab in scn],
           'Consumption, % deviation from the baseline')
    base = S['baseline']
    _lines(ax[0, 2], x, n, [('baseline', end_of_year_debt(base, G))]
           + [(lab, end_of_year_debt(S[k], G)) for k, lab in scn],
           'Debt / output, end of year', zero=False, loc='upper left')
    nfa = lambda s: (np.asarray(s['counterfactual']['NFA'], float)
                     / np.asarray(s['counterfactual']['Y'], float))
    _lines(ax[1, 0], x, n, [('baseline', nfa(base))] + [(lab, nfa(S[k])) for k, lab in scn],
           'Net foreign assets / output', loc='upper right')
    tl = lambda s: 100.0 * np.asarray(s['counterfactual'].get('tau_l', s['counterfactual']['Y']), float)
    if 'tau_l' in base['counterfactual']:
        _lines(ax[1, 1], x, n, [('baseline', tl(base))] + [(lab, tl(S[k])) for k, lab in scn],
               r'Labour income tax rate, %', zero=False)
    else:
        ax[1, 1].set_visible(False)
    if shock == 'health':
        s0 = S[scn[0][0]]
        _lines(ax[1, 2], x, n,
               [('government, baseline', 100 * ratio(s0, 'base', 'gov_health')),
                ('government, cut', 100 * ratio(s0, 'cf', 'gov_health')),
                ('households, baseline', 100 * ratio(s0, 'base', 'oop_health')),
                ('households, cut', 100 * ratio(s0, 'cf', 'oop_health'))],
               'Health spending / output, %', zero=False)
    else:
        key = 'govt_spending' if shock == 'G' else 'public_investment'
        _lines(ax[1, 2], x, n, [(lab, 100 * (ratio(S[k], 'cf', key) - ratio(S[k], 'base', key)))
                                for k, lab in scn],
               ('Change in government consumption' if shock == 'G'
                else 'Change in public investment') + ' / output, pp')
    for a in ax.flat:
        a.axvline(_shock_year(R, base_year), color=INK2, lw=0.5, ls=':')
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


def _dist_years(R, base_year):
    return base_year + np.asarray(R['params']['reporting_periods'], int)


def age_figure(R, base_year, shock, out_pdf, plt):
    S = R[shock]
    scn = present(S, SCENARIOS)
    years = _dist_years(R, base_year)
    groups = list(S['baseline']['distribution']['age_groups'][0].keys())
    fig, ax = plt.subplots(len(scn), len(AGE_VARS), figsize=(3.0 * len(AGE_VARS), 2.4 * len(scn)),
                           squeeze=False)
    for i, (k, lab) in enumerate(scn):
        b = S['baseline']['distribution']['age_groups']
        c = S[k]['distribution']['age_groups']
        for j, (var, vlab) in enumerate(AGE_VARS):
            series = []
            for g in groups:
                yb = np.array([row[g][var] for row in b], float)
                yc = np.array([row[g][var] for row in c], float)
                series.append((g, 100.0 * (yc / yb - 1.0)))
            _lines(ax[i, j], years, len(years), series, f'{vlab}, % ({lab})')
    fig.tight_layout(h_pad=1.0)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


def _ineq(s, measure, stat):
    return np.array([row[measure].get(stat, np.nan) for row in s['distribution']['inequality']], float)


def ineq_figure(R, base_year, shock, out_pdf, plt):
    S = R[shock]
    scn = present(S, SCENARIOS)
    years = _dist_years(R, base_year)
    ncol = 4
    nrow = (len(INEQ_PANELS) + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(12, 2.8 * nrow), squeeze=False)
    for a, (measure, stat, title) in zip(ax.flat, INEQ_PANELS):
        b = _ineq(S['baseline'], measure, stat)
        scale = 100.0 if stat in ('gini', 'top10_share') else 1.0
        _lines(a, years, len(years),
               [(lab, scale * (_ineq(S[k], measure, stat) - b)) for k, lab in scn],
               title + (', change in points' if scale == 100.0 else ', change'))
    fig.tight_layout(h_pad=1.0)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


def welfare_figure(R, shock, out_pdf, plt):
    S = R[shock]
    scn = [(k, lab) for k, lab in present(S, SCENARIOS) if 'welfare' in S[k]]
    entry_age = int(R['params'].get('entry_age', 25))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for i, (k, lab) in enumerate(scn):
        w = S[k]['welfare']
        by = np.asarray(w['entry_year'], int) - entry_age
        ax[0].plot(by, 100 * np.asarray(w['cohort_cev'], float), color=SERIES[i], lw=1.5, label=lab)
    if scn:
        w = S[scn[0][0]]['welfare']
        alive = np.asarray(w['alive_in_shock_year'], bool)
        by = np.asarray(w['entry_year'], int) - entry_age
        if alive.any():
            ax[0].axvspan(by[alive].min() - 0.5, by[alive].max() + 0.5, color='#ecebe7', zorder=0)
    ax[0].axhline(0.0, color=INK2, lw=0.6)
    ax[0].legend(frameon=False, fontsize=7, labelcolor=INK)
    _style(ax[0], 'Consumption equivalent by year of birth, % (shaded: alive in the shock year)')
    width = 0.8 / max(len(scn), 1)
    for i, (k, lab) in enumerate(scn):
        q = 100 * np.asarray(S[k]['welfare']['quintile_cev'], float)
        ax[1].bar(np.arange(1, len(q) + 1) + (i - (len(scn) - 1) / 2) * width, q, width,
                  color=SERIES[i], label=lab)
    ax[1].axhline(0.0, color=INK2, lw=0.6)
    ax[1].set_xticks(np.arange(1, 6))
    ax[1].legend(frameon=False, fontsize=7, labelcolor=INK)
    _style(ax[1], 'By quintile of disposable income in the shock year, %')
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


def exercise_table(R, G, base_year, shock):
    """Rows: statistics; columns: the schemes of the exercise."""
    S = R[shock]
    scn = present(S, SCENARIOS)
    ys = _shock_year(R, base_year)
    t = lambda y: y - base_year
    fmt = lambda v, p=2, sign=True: ('{--}' if v is None or not np.isfinite(v)
                                     else f'{v:+.{p}f}' if sign else f'{v:.{p}f}')
    rows = []

    def row(label, vals, p=2, sign=True):
        rows.append(f'{label} & ' + ' & '.join(fmt(v, p, sign) for v in vals) + ' \\\\')

    rows.append(f'\\multicolumn{{{len(scn) + 1}}}{{l}}{{\\itshape Multiplier}}\\\\')
    row(f'\\quad impact ({ys})', [S[k]['multiplier']['impact'] for k, _ in scn])
    row(f'\\quad cumulative {ys}--{ys + 9}', [S[k]['multiplier']['cumulative_10y'] for k, _ in scn])
    row('\\quad cumulative over the horizon', [S[k]['multiplier']['cumulative_horizon'] for k, _ in scn])
    row('$\\Delta\\tau_l$, pp', [None if S[k].get('adjustment_scalar') is None
                                 else 100 * S[k]['adjustment_scalar'] for k, _ in scn])
    rows.append(f'\\multicolumn{{{len(scn) + 1}}}{{l}}{{\\itshape Debt / output, end of year, \\%}}\\\\')
    def at(a, i):
        return float(a[i]) if 0 <= i < len(a) else None
    for y in (2030, 2040, 2060, 2070):
        vals = [at(end_of_year_debt(S[k], G), t(y)) for k, _ in scn]
        row(f'\\quad {y}', [None if v is None else 100 * v for v in vals], 1, sign=False)
    if 'distribution' in S['baseline']:
        years = list(_dist_years(R, base_year))
        rows.append(f'\\multicolumn{{{len(scn) + 1}}}{{l}}{{\\itshape Change in the Gini, points}}\\\\')
        for measure, mlab in (('disp_income', 'disposable income'), ('consumption', 'consumption'),
                              ('labour_income', 'labour income'), ('wealth', 'wealth')):
            b = _ineq(S['baseline'], measure, 'gini')
            for y in (ys, 2030, 2036, 2070):
                if y in years:
                    r = years.index(y)
                    row(f'\\quad {mlab}, {y}',
                        [100 * (_ineq(S[k], measure, 'gini')[r] - b[r]) for k, _ in scn])
    if all('welfare' in S[k] for k, _ in scn):
        rows.append(f'\\multicolumn{{{len(scn) + 1}}}{{l}}{{\\itshape Consumption-equivalent '
                    f'variation, \\%}}\\\\')
        row(f'\\quad cohorts alive in {ys}, mean', [100 * S[k]['welfare']['alive_mean'] for k, _ in scn])
        entry_age = int(R['params'].get('entry_age', 25))
        for ey in (ys + 1, ys + 10, ys + 30):
            vals = []
            for k, _ in scn:
                w = S[k]['welfare']
                vals.append(100 * w['cohort_cev'][w['entry_year'].index(ey)]
                            if ey in w['entry_year'] else None)
            row(f'\\quad entering in {ey} (born {ey - entry_age})', vals)
    head = (' & ' + ' & '.join('{' + lab + '}' for _, lab in scn) + ' \\\\')
    colspec = '@{}l' + 'S[table-format=+2.2]' * len(scn) + '@{}'
    return ('\\begin{tabular}{' + colspec + '}\n\\toprule\n' + head + '\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def table(R, G, base_year):
    """Responses across shocks (the table of earlier rounds), dated from the
    shock year."""
    ys = _shock_year(R, base_year)
    t = lambda y: y - base_year
    # The last year reported: 2200, or the end of the transition if earlier
    yl = min(2200, base_year + int(R['params'].get('T_transition', 10 ** 4)) - 1)

    def fmt(v, places=1, sign=False):
        return f'{v:+.{places}f}' if sign else f'{v:.{places}f}'
    rows = []
    for shock, name in SHOCKS.items():
        if shock not in R:
            continue
        S = R[shock]
        rows.append(f'\\multicolumn{{9}}{{l}}{{\\itshape {name}}}\\\\')
        for k, lab in present(S, SCENARIOS):
            s = S[k]
            dY, dC = dev(s, 'Y'), dev(s, 'C')
            d = end_of_year_debt(s, G)
            nfa = (np.asarray(s['counterfactual']['NFA'], float)
                   / np.asarray(s['counterfactual']['Y'], float))
            adj = s.get('adjustment_scalar')
            last = min(t(yl), len(d) - 1, len(dY) - 1)
            y2100 = min(2100, yl)
            cells = ['{--}' if adj is None else fmt(100 * adj, 2, True),
                     fmt(dY[t(ys)], 1, True), fmt(dY[t(ys):t(ys) + 10].mean(), 1, True),
                     fmt(dY[t(min(2050, yl))], 1, True), fmt(dY[t(y2100)], 1, True),
                     fmt(dC[t(y2100)], 1, True),
                     fmt(100 * d[t(min(2060, yl))], 0), fmt(100 * d[last], 0),
                     fmt(nfa[last], 2, True)]
            rows.append(f'\\quad {lab} & ' + ' & '.join(cells) + ' \\\\')
    colspec = ('@{}l' + 'S[table-format=+1.2]' + 'S[table-format=+1.1]' * 5
               + 'S[table-format=+3.0]' * 2 + 'S[table-format=+1.2]@{}')
    head = (' & {$\\Delta\\tau_l$, pp} & \\multicolumn{4}{c}{Output, \\% from baseline}'
            ' & {$C$, \\%} & \\multicolumn{2}{c}{Debt / output, \\%} & {$NFA/Y$} \\\\\n'
            '\\cmidrule(lr){3-6}\\cmidrule(lr){8-9}\n'
            f' & & {{{ys}}} & {{{ys}--{str(ys + 9)[2:]}}} & {{2050}} & {{{min(2100, yl)}}} & '
            f'{{{min(2100, yl)}}} & {{2060}} & {{{yl}}} & {{{yl}}} \\\\')
    return ('\\begin{tabular}{' + colspec + '}\n\\toprule\n' + head
            + '\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--results',
                    default=os.path.join(here, '..', 'output', 'fiscal_2026-10-06_ret_annual', 'fiscal_results.json'))
    ap.add_argument('--baseline',
                    default=os.path.join(here, '..', 'output', 'calibration_growth', 'baseline_paths.npz'))
    ap.add_argument('--last-year', type=int, default=2070)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R, G, base_year = load(args.results, args.baseline)
    outdir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    for shock in [s for s in SHOCKS if s in R]:
        S = R[shock]
        shock_figure(R, G, base_year, shock, os.path.join(outdir, f'fiscal_{shock}.pdf'),
                     plt, args.last_year)
        if 'distribution' in S['baseline']:
            age_figure(R, base_year, shock, os.path.join(outdir, f'fiscal_{shock}_age.pdf'), plt)
            ineq_figure(R, base_year, shock, os.path.join(outdir, f'fiscal_{shock}_ineq.pdf'), plt)
            welfare_figure(R, shock, os.path.join(outdir, f'fiscal_{shock}_welfare.pdf'), plt)
        if all('multiplier' in S[k] for k, _ in present(S, SCENARIOS)):
            with open(os.path.join(outdir, f'fiscal_{shock}_body.tex'), 'w') as fh:
                fh.write(exercise_table(R, G, base_year, shock))
            print(f'  wrote fiscal_{shock}_body.tex')
        for k, lab in present(S, SCENARIOS):
            m = S[k].get('multiplier')
            if m:
                print(f'  {shock}, {k}: multiplier impact {m["impact"]:.3f}, '
                      f'10 years {m["cumulative_10y"]:.3f}, horizon {m["cumulative_horizon"]:.3f}')
        if shock == 'health':
            dec = present(S, DECOMPOSITION)
            if dec and 'welfare' in S.get('debt_financed', {}):
                tot = np.asarray(S['debt_financed']['welfare']['cohort_cev'], float)
                parts = [np.asarray(S[k]['welfare']['cohort_cev'], float) for k, _ in dec]
                resid = tot - sum(parts)
                print(f'  health decomposition of the mean CEV of cohorts alive in the shock year: '
                      f'total {100 * S["debt_financed"]["welfare"]["alive_mean"]:+.3f}%, '
                      + ', '.join(f'{lab} {100 * S[k]["welfare"]["alive_mean"]:+.3f}%'
                                  for k, lab in dec)
                      + f'; residual by cohort max |.| {100 * np.max(np.abs(resid)):.3f}%')
    with open(os.path.join(outdir, 'fiscal_body.tex'), 'w') as fh:
        fh.write(table(R, G, base_year))
    print('  wrote fiscal_body.tex')


if __name__ == '__main__':
    main()
