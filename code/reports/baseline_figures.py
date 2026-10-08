"""Figures of the baseline transition, read from the saved paths.

Draws the main aggregates and the government accounts of the no-policy-change
baseline from ``baseline_paths.npz`` (written by ``fill_report.py
--run-baseline``), so no transition is solved here. The debt ratio is not in
the saved paths. Debt and the primary balance come from
``baseline_closure.debt_from_run`` (the model's own balance, the 2024-25
ratios imposed, the projection's stock-flow rows to 2060); with a debt
projection configured a third figure sets the baseline against it. Without one, taxes and spending shares
are fixed and debt follows from the saved primary deficit by the recursion the
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
from baseline_closure import debt_from_run, load_dsa_projection  # noqa: E402

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
    rb_path = (np.asarray(d['r_B_path'], float)[:T] if 'r_B_path' in d.files
               else np.full(T, r_B))
    B = compute_debt_path(b('primary_deficit'), rb_path,
                          B_initial=B_over_Y0 * Y[0], growth_factor=G)
    dsa = load_dsa_projection(cfg)
    saved = {k: d[k] for k in d.files}
    debt = debt_from_run(saved, cfg, dsa=dsa, r_B_path=rb_path)
    return dict(d=d, Y=Y, T=T, years=years, b=b, inv=inv, B=B[:T],
                B_over_Y0=B_over_Y0, r_B=r_B, G=G, dsa=dsa, debt=debt, cfg=cfg,
                tau_y=(np.asarray(d['tau_y_path'], float)[:T] if 'tau_y_path' in d.files else None))


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
    # With a constant output tax y, k^dom and l share one index; the tax's
    # ramp after 2060 moves K/L, so the three are drawn separately when they
    # differ.
    same = max(np.abs(ix['Y'] - ix['L']).max(), np.abs(ix['Y'] - ix['K_domestic']).max()) < 1e-8
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    first = ([(r'$\hat y=\hat k^{dom}=\hat\ell$', ix['Y'])] if same
             else [(r'$\hat y$', ix['Y']), (r'$\hat k^{dom}$', ix['K_domestic']),
                   (r'$\hat\ell$', ix['L'])])
    _lines(ax[0], x, first + [(r'$\hat c$', np.asarray(d['C'], float)[:T] / float(d['C'][0])),
                              (r'$\hat a$', np.asarray(d['A'], float)[:T] / float(d['A'][0]))],
           ncol=3)
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
    def bz(k):
        return b(k) if ('budget_' + k) in p['d'].files else np.zeros_like(Y)
    rev = [('consumption tax', b('tax_c')), ('labour income tax', b('tax_l')),
           ('payroll tax', b('tax_p')), ('capital income tax', b('tax_k')),
           ('bequest tax', b('bequest_tax')), ('output tax', bz('tax_y')),
           ('transfer from the EU', bz('foreign_transfer'))]
    _lines(ax[0, 0], x, [(lab, v / Y) for lab, v in rev], ncol=3)
    _style(ax[0, 0], 'Revenue / output')
    c = p['debt']
    spend = [('pensions', b('pension')), ('public health', b('gov_health')),
             ('UI and minimum income', b('ui') + b('transfers')),
             ('$G$ and defence', b('govt_spending') + b('defense_spending')),
             ('public investment', b('public_investment')),
             ('education', bz('education')), ('lump-sum transfer', bz('lump_sum'))]
    other = bz('other_net_spending')
    if np.any(other != 0.0):
        spend.append(('other net spending $O$', other))
    _lines(ax[0, 1], x, [(lab, v / Y) for lab, v in spend], ncol=3)
    ax[0, 1].axhline(0.0, color=INK2, lw=0.6, zorder=1)
    _style(ax[0, 1], 'Spending / output')
    series = [('primary balance', c['primary_balance'])]
    if p['tau_y'] is not None:
        series.append((r'output tax rate $\tau_y$', p['tau_y']))
    _lines(ax[1, 0], x, series, ncol=2)
    ax[1, 0].axhline(0.0, color=INK2, lw=0.6, zorder=1)
    _style(ax[1, 0], 'Primary balance and output tax rate / output')
    _lines(ax[1, 1], x, [('', c['debt'])])
    _style(ax[1, 1], f"Debt / output, end of year ({c['debt'][0]:.2f} in 2023)")
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


# Public pension spending over GDP and the average effective retirement age,
# 2024 Ageing Report, Country Fiche EL (December 2023), Tables 6 and 4.
FICHE_YEARS = (2022, 2030, 2040, 2050, 2060, 2070)
FICHE_PENSIONS = (0.145, 0.127, 0.137, 0.140, 0.127, 0.120)


def projection_figure(p, out_pdf, plt, last_year=2070):
    """The baseline against the Commission's debt projection."""
    c, dsa, Y, x = p['debt'], p['dsa'], p['Y'], p['years']
    n = int(np.searchsorted(x, last_year)) + 1
    b = p['b']
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.6))

    def two(a, model, proj_x, proj_y, title, labels=('baseline', 'projection')):
        a.plot(x[:n], model[:n], color=SERIES[0], lw=1.5, label=labels[0])
        a.plot(proj_x, proj_y, color=SERIES[1], lw=1.5, ls='--', label=labels[1])
        a.legend(frameon=False, fontsize=7.5, labelcolor=INK, handlelength=1.8)
        _style(a, title)

    two(ax[0, 0], c['debt'], dsa['years'], dsa['debt'], 'Debt / output, end of year')
    ax[0, 1].plot(x[:n], c['primary_balance'][:n], color=SERIES[0], lw=1.5,
                  label='primary balance')
    if p['tau_y'] is not None:
        ax[0, 1].plot(x[:n], p['tau_y'][:n], color=SERIES[2], lw=1.5,
                      label=r'output tax rate $\tau_y$')
    ax[0, 1].plot(dsa['years'], dsa['primary_balance'], color=SERIES[1], lw=1.5, ls='--',
                  label='primary balance, projection')
    ax[0, 1].axhline(0.0, color=INK2, lw=0.6, zorder=1)
    ax[0, 1].legend(frameon=False, fontsize=7.5, labelcolor=INK, handlelength=1.8)
    _style(ax[0, 1], 'Primary balance and output tax rate / output')
    two(ax[1, 0], b('pension') / Y, FICHE_YEARS, FICHE_PENSIONS,
        'Public pensions / output', labels=('baseline', '2024 Ageing Report'))
    growth = centred_mean(np.nan_to_num(c['growth'], nan=c['growth'][1]), 5)
    two(ax[1, 1], growth, dsa['years'], dsa['real_growth'],
        'Real output growth (baseline: 5-year centred mean)')
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')


def demography_figure(cfg, out_pdf, plt, last_year=2120):
    """The demographic inputs: population aged 25-99 and entering cohorts,
    the age structure, dependency and retirees per non-retired person, and the
    retirement age with life expectancy at 65. Read from the data files the
    configuration names; no run is needed."""
    here = os.path.dirname(os.path.abspath(__file__))
    trans = cfg.get('transition', {})
    path = lambda key: os.path.join(here, '..', trans[key])
    dem = np.load(path('demography_file'))
    ret = np.load(path('retirement_age_file'))
    years = dem['pop_years'].astype(int)
    pop = np.asarray(dem['pop'], float)                   # (years, T) ages 25..24+T
    n = int(np.searchsorted(years, last_year)) + 1
    x = years[:n]
    total = pop.sum(1)
    ent_y = dem['entrant_years'].astype(int)
    ent = np.asarray(dem['entrants'], float)
    e0 = ent[np.searchsorted(ent_y, years[0])]
    entrants = np.array([ent[np.searchsorted(ent_y, y)] for y in x]) / e0
    ages = 25 + np.arange(pop.shape[1])
    share = lambda lo, hi: pop[:n, (ages >= lo) & (ages <= hi)].sum(1) / total[:n]
    dep = pop[:n, ages >= 65].sum(1) / pop[:n, ages <= 64].sum(1)
    # Retirees per non-retired person from the cohort retirement table.
    tab = dict(zip(ret['entry_years'].tolist(), zip(ret['J_R'].tolist(), ret['share_later'].tolist())))
    lo_e, hi_e = min(tab), max(tab)
    R = np.zeros(n); W = np.zeros(n)
    for i, y in enumerate(x):
        for j in range(pop.shape[1]):
            JR, sl = tab[int(min(max(y - j, lo_e), hi_e))]
            r = 1.0 if j >= JR + 1 else ((1.0 - sl) if j >= JR else 0.0)
            R[i] += pop[i, j] * r
            W[i] += pop[i, j] * (1 - r)
    ry = ret['years'].astype(int)
    rpath = np.array([ret['retirement_age_path'][np.searchsorted(ry, y)] for y in x])
    ey = ret['e65_years'].astype(int)
    e65 = np.array([ret['e65'][np.searchsorted(ey, min(y, ey[-1]))] for y in x])

    fig, ax = plt.subplots(2, 2, figsize=(10, 5.6))
    _lines(ax[0, 0], x, [('population aged 25\u201399', total[:n] / total[0]),
                         ('cohort entering at 25', entrants)], ncol=2)
    ax[0, 0].axhline(1.0, color=INK2, lw=0.6, zorder=1)
    _style(ax[0, 0], 'Population, 2023 = 1')
    _lines(ax[0, 1], x, [('25\u201339', share(25, 39)), ('40\u201364', share(40, 64)),
                         ('65\u201399', share(65, 99))], ncol=3)
    _style(ax[0, 1], 'Shares of the population aged 25\u201399')
    _lines(ax[1, 0], x, [('aged 65\u201399 per person aged 25\u201364', dep),
                         ('retired per non-retired person', R / W)], ncol=1)
    _style(ax[1, 0], 'Dependency')
    _lines(ax[1, 1], x, [('average effective retirement age', rpath),
                         ('65 + life expectancy at 65', 65.0 + e65)], ncol=1)
    _style(ax[1, 1], 'Retirement age and life expectancy, years')
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  wrote {os.path.basename(out_pdf)}')
    return dict(years=x, total=total[:n] / total[0], entrants=entrants, dep=dep, RW=R / W,
                s2539=share(25, 39), s4064=share(40, 64), s6599=share(65, 99),
                ret=rpath, e65=e65)


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
    demography_figure(p['cfg'], os.path.join(args.outdir, 'demography.pdf'), plt)
    if p['dsa'] is not None:
        projection_figure(p, os.path.join(args.outdir, 'baseline_projection.pdf'), plt)


if __name__ == '__main__':
    main()
