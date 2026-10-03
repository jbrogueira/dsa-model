"""Figures of the baseline transition, read from the saved paths.

Draws the main aggregates and the government accounts of the no-policy-change
baseline from ``baseline_paths.npz`` (written by ``fill_report.py
--run-baseline``), so no transition is solved here. The debt ratio is not in
the saved paths: taxes and spending are fixed in the baseline and the debt is
external, so it follows from the saved primary deficit by the recursion the
fiscal experiments use (``fiscal_experiments.compute_debt_path``), from the
config's B/Y in the base year at the config's r_B.

Usage (from code/):
    python reports/baseline_figures.py [--config calibration_input_GR.json]
                                       [--outdir output/calibration_growth]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fiscal_experiments import compute_debt_path  # noqa: E402

# Categorical slots in fixed order (dataviz reference palette, light surface).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'


def load(npz, config):
    d = np.load(npz, allow_pickle=True)
    cfg = json.load(open(config))
    Y = np.asarray(d['Y'], float)
    T = len(Y)
    years = int(d['base_year']) + np.arange(T)

    def b(k):
        return np.asarray(d['budget_' + k], float)[:T]

    G = np.asarray(d['growth_factor'], float)[:T]
    delta = float(d['delta'])
    Kd = np.asarray(d['K_domestic'], float)
    inv = G[:T - 1] * Kd[1:T] - (1.0 - delta) * Kd[:T - 1]
    B_over_Y0 = float(cfg.get('fiscal', {}).get('B_over_Y', 0.0))
    r_B = float(cfg['prices']['r_B'])
    B = compute_debt_path(b('primary_deficit'), np.full(T, r_B),
                          B_initial=B_over_Y0 * Y[0], growth_factor=G)
    return dict(d=d, Y=Y, T=T, years=years, b=b, inv=inv, B=B[:T],
                B_over_Y0=B_over_Y0, r_B=r_B, G=G)


def _style(ax, title):
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7.5, width=0.6)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _lines(ax, x, series, legend='below', ncol=2):
    for i, (lab, y) in enumerate(series):
        ax.plot(x[:len(y)], y, color=SERIES[i], lw=1.5, label=lab)
    if len(series) > 1:
        kw = dict(frameon=False, fontsize=7.5, ncol=ncol, labelcolor=INK,
                  handlelength=1.6, columnspacing=1.0)
        if legend == 'below':
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), **kw)
        else:
            ax.legend(**kw)


# Centred window for I/Y. Investment is a first difference of K^dom, which is
# about 6.5% of the stock, so it carries ~20 times the stock's sampling noise
# (measured on the 2026-10-03 baseline: 2.5% of its level per period, against
# 0.1% for K^dom); the other series are levels and are plotted raw.
SMOOTH_I = 5


def centred_mean(x, w):
    """Centred moving average; the window shrinks symmetrically at the ends."""
    x = np.asarray(x, float)
    h = w // 2
    return np.array([x[max(0, i - min(h, i, len(x) - 1 - i)):
                       i + min(h, i, len(x) - 1 - i) + 1].mean() for i in range(len(x))])


def aggregates_figure(p, out_pdf, plt):
    """Indexed aggregates and ratios to output.

    y, k^dom and l share one index: K_g per capita is constant and the
    exogenous return fixes K/L, so Y, K^dom and L are proportional (checked
    here to 1e-10 before plotting y alone).
    """
    d, Y, x, T = p['d'], p['Y'], p['years'], p['T']
    ix = {k: np.asarray(d[k], float)[:T] / float(d[k][0]) for k in ('Y', 'K_domestic', 'L')}
    assert max(np.abs(ix['Y'] - ix['L']).max(), np.abs(ix['Y'] - ix['K_domestic']).max()) < 1e-10
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    _lines(ax[0], x, [(r'$\hat y=\hat k^{dom}=\hat\ell$', ix['Y']),
                      (r'$\hat c$', np.asarray(d['C'], float)[:T] / float(d['C'][0])),
                      (r'$\hat a$', np.asarray(d['A'], float)[:T] / float(d['A'][0]))], ncol=3)
    ax[0].axhline(1.0, color=INK2, lw=0.6, zorder=1)
    _style(ax[0], 'Detrended per capita, 2023 = 1')
    _lines(ax[1], x, [('$C/Y$', np.asarray(d['C'], float)[:T] / Y),
                      (f'$I/Y$ ({SMOOTH_I}-year centred mean)',
                       centred_mean(p['inv'] / Y[:-1], SMOOTH_I))])
    _style(ax[1], 'Consumption and investment / output')
    _lines(ax[2], x, [('$A/Y$', np.asarray(d['A'], float)[:T] / Y),
                      ('$K^{dom}/Y$', np.asarray(d['K_domestic'], float)[:T] / Y),
                      ('$NFA/Y$', np.asarray(d['NFA'], float)[:T] / Y)], ncol=3)
    ax[2].axhline(0.0, color=INK2, lw=0.6, zorder=1)
    _style(ax[2], 'Household wealth / output')
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


def fiscal_figure(p, out_pdf, plt):
    b, Y, x = p['b'], p['Y'], p['years']
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.2))
    rev = [('consumption tax', b('tax_c')), ('labour income tax', b('tax_l')),
           ('payroll tax', b('tax_p')), ('capital income tax', b('tax_k')),
           ('bequest tax', b('bequest_tax'))]
    _lines(ax[0, 0], x, [(lab, v / Y) for lab, v in rev], ncol=3)
    _style(ax[0, 0], 'Revenue / output')
    spend = [('pensions', b('pension')), ('public health', b('gov_health')),
             ('UI and minimum income', b('ui') + b('transfers')),
             ('$G$ and defence', b('govt_spending') + b('defense_spending')),
             ('public investment', b('public_investment')),
             ('other net spending $O$', b('other_net_spending'))]
    _lines(ax[0, 1], x, [(lab, v / Y) for lab, v in spend], ncol=3)
    ax[0, 1].axhline(0.0, color=INK2, lw=0.6, zorder=1)
    _style(ax[0, 1], 'Spending / output')
    _lines(ax[1, 0], x, [('', b('primary_deficit') / Y)])
    ax[1, 0].axhline(0.0, color=INK2, lw=0.6, zorder=1)
    _style(ax[1, 0], 'Primary deficit / output')
    _lines(ax[1, 1], x, [('', p['B'] / Y)])
    _style(ax[1, 1], f"Debt / output ($B/Y$ = {p['B_over_Y0']:.2f} in 2023, "
                     f"$r_B$ = {p['r_B']:.3f})")
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(here, '..', 'calibration_input_GR.json'))
    ap.add_argument('--outdir', default=os.path.join(here, '..', 'output', 'calibration_growth'))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = load(os.path.join(args.outdir, 'baseline_paths.npz'), args.config)
    aggregates_figure(p, os.path.join(args.outdir, 'baseline_aggregates.pdf'), plt)
    fiscal_figure(p, os.path.join(args.outdir, 'baseline_fiscal.pdf'), plt)


if __name__ == '__main__':
    main()
