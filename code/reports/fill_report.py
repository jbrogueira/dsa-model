#!/usr/bin/env python3
"""Fill the calibration report from a calibration run.

Reads the config, the markdown report that calibrate.py writes, and (optionally)
a baseline transition, and emits the LaTeX table bodies and figures that
calibration_report.tex \input{}s.  Nothing here re-runs the SMM.

    python fill_report.py --config ../calibration_input_GR.json \
        --calib-report ../output/calibration_growth/reports/calibration_GR_*.md \
        --outdir ../output/calibration_growth --run-baseline --n-sim 1000

Without --run-baseline the figure and the growth table are written with
placeholders, so the report still compiles.
"""
import argparse, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

NP = r'{\textemdash}'          # placeholder cell


# ---------------------------------------------------------------- parsing ---
def parse_md_table(text, heading):
    """Return the rows of the markdown table under *heading* as lists."""
    m = re.search(rf'^##\s+{re.escape(heading)}\s*$', text, re.M)
    if not m:
        return []
    rest = text[m.end():]
    stop = re.search(r'^##\s+', rest, re.M)
    block = rest[:stop.start()] if stop else rest
    rows = []
    for line in block.strip().split('\n'):
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if all(set(c) <= set('-: ') for c in cells):      # separator
            continue
        rows.append(cells)
    return rows[1:] if rows else []                        # drop the header


def num(x, default=None):
    try:
        return float(str(x).replace('%', '').replace(',', ''))
    except (TypeError, ValueError):
        return default


def fmt(x, places=4):
    return NP if x is None else f'{x:.{places}f}'


# ----------------------------------------------------------------- tables ---
SMM_LABELS = {
    'nu': ('$\\nu$', 'labour disutility weight', 'average hours'),
    'beta': ('$\\beta$', 'discount factor', '$A/Y$'),
    'tau_p': ('$\\tau_p$', 'payroll tax rate', 'payroll revenue$/Y$'),
    'pension_replacement_default': ('$\\rho^{pens}$', 'pension replacement rate', 'pensions$/Y$'),
    'm_good': ('$m^{good}$', 'medical cost scale', 'public health$/Y$'),
    'ui_replacement_rate': ('$\\rho^{ui}$', 'UI replacement rate', 'UI$/Y$'),
}


def params_table(cfg, n0=None, n_inf=None):
    """n0 is the realised population growth of the base year from the
    demographic path and n_inf the terminal rate; the configuration's
    pop_growth scalar is the terminal rate only, and printing it as 'n'
    while the header printed the realised value put two numbers for one
    symbol in one report."""
    ext, prod, pri, mod = (cfg['external_params'], cfg['production'],
                           cfg['prices'], cfg['model'])
    th = cfg.get('_derived', {}).get('theta', {})
    g = ext.get('trend_growth', 0.0)
    if n_inf is None:
        n_inf = ext.get('pop_growth', 0.0)
    rows_ext = [
        ('$g$', 'labour productivity growth, per capita', g, '2024 Ageing Report'),
        ('$n_0$', 'population growth, base year', n0, 'EUROPOP2023, demography file'),
        ('$n_\\infty$', 'population growth, terminal', n_inf, 'assumption (tail after 2120)'),
        ('$\\gamma$', 'curvature of consumption utility', mod.get('gamma'), 'log utility'),
        ('$\\varphi$', 'inverse Frisch elasticity', mod.get('phi'), ''),
        ('$\\alpha$', 'private capital share', prod.get('alpha'), ''),
        ('$\\delta$', 'private depreciation', prod.get('delta'), 'standard value'),
        ('$\\eta_g$', 'public capital elasticity', prod.get('eta_g'), ''),
        ('$K_g/Y$', 'public capital ratio', prod.get('K_g'), 'IMF ICSD'),
        ('$\\delta_g$', 'public capital depreciation', prod.get('delta_g'),
         '$I_g/K_g-(\\Gamma-1)$'),
        ('$r$', 'world return on capital', pri.get('r'), ''),
        ('$r_B$', 'sovereign rate', pri.get('r_B'), 'implicit rate 2012--24'),
        ('$\\tau_c,\\tau_l,\\tau_k$', 'consumption, labour, capital tax rates',
         ext.get('tau_c'), 'effective rates'),
        ('$b_{min}$', 'minimum pension floor', ext.get('pension_min_floor'),
         'national pension, L.4387/2016'),
        ('$\\underline{c}$', 'means-tested consumption floor', ext.get('transfer_floor'),
         'guaranteed minimum income'),
        ('$\\tau^{beq}$', 'tax on accidental bequests', ext.get('tau_beq'), ''),
        ('$f$', 'job-finding probability', ext.get('job_finding_rate'),
         'Eurostat \\texttt{une\\_ltu\\_a}'),
        ('$\\kappa$', 'public share of medical spending', ext.get('kappa'),
         'Eurostat \\texttt{hlth\\_sha11\\_hf}'),
        ('$T$, $J_R$', 'lifespan, retirement age', mod.get('T'), ''),
    ]
    # One row per parameter the configuration lists for the SMM; a parameter
    # the last fit did not cover shows no value rather than its initial.
    rows_cal = []
    for p in cfg.get('calibration', {}).get('params', []):
        sym, desc, ident = SMM_LABELS.get(p['name'], (p['name'], p.get('path', ''), ''))
        rows_cal.append((sym, desc, th.get(p['name']), ident))
    rows_pin = [
        ('$A$', 'total factor productivity', prod.get('A_tfp'),
         'normalisation, $\\hat y=1$'),
        ('$O/Y$', 'other net spending', cfg['fiscal'].get('other_net_spending_over_Y'),
         'data primary balance'),
    ]
    out = ['\\multicolumn{4}{l}{\\itshape Externally set}\\\\']
    for sym, desc, val, src in rows_ext:
        out.append(f'{sym} & {desc} & {fmt(val, 4)} & {src} \\\\')
    out.append('\\midrule\n\\multicolumn{4}{l}{\\itshape Calibrated jointly by SMM}\\\\')
    for sym, desc, val, src in rows_cal:
        out.append(f'{sym} & {desc} & {fmt(val, 4)} & {src} \\\\')
    out.append('\\midrule\n\\multicolumn{4}{l}{\\itshape Pinned outside the SMM}\\\\')
    for sym, desc, val, src in rows_pin:
        out.append(f'{sym} & {desc} & {fmt(val, 4)} & {src} \\\\')
    return '\n'.join(out)


LABEL = {'average_hours': 'Average hours', 'A_over_Y': '$A/Y$',
         'tax_p_over_Y': 'Payroll revenue$/Y$', 'pensions_over_Y': 'Pensions$/Y$',
         'health_gov_over_Y': 'Public health$/Y$', 'I_g_over_Y': '$I_g/Y$',
         'ui_over_Y': 'UI$/Y$', 'interest_over_Y': 'Interest$/Y$ $(r_BB/Y)$',
         'primary_balance_over_Y': 'Household primary balance$/Y$', 'G_over_Y': '$G/Y$',
         'disposable_income_gini': 'Disposable income Gini',
         'disposable_p90_p10': 'Disposable income p90/p10',
         'B_over_Y': '$B/Y$', 'health_oop_over_Y': 'Out-of-pocket health$/Y$'}
# Fiscal ratios the model does not target but the data measure. I_g/Y and the
# household primary balance are out: the first is a policy input, the second
# has no data counterpart (it excludes G, I_g, defence and the residual O/Y).
# Interest/Y is out too: the "data" value 0.0312 is r_B x B/Y with both taken
# from the config (0.019 x 1.64), so the model reproduces it by construction.
UNTARGETED = ['ui_over_Y']
# Distributional checks, as (model moment, data key in the config's
# `untargeted` block). The model side is the disposable measure, since the
# Eurostat series are disposable; see _moment_disposable_income_gini in
# calibrate.py for what can and cannot be matched.
UNTARGETED_DIST = [('disposable_income_gini', 'income_gini'),
                   ('disposable_p90_p10', 'p90_p10_income')]


def live_moments(panels, spec, cfg):
    """Model moments at the config's own theta and A_tfp.

    The calibration report is written inside an SMM round, so its moments
    predate the last A_tfp normalisation and the closure re-pin. Recomputing
    them here keeps the table consistent with the config it documents.
    """
    from calibrate import MOMENT_DISPATCH, compute_fiscal_ratios
    targeted = []
    for mom in spec.moments:
        model = float(MOMENT_DISPATCH[mom.compute_key](panels, spec))
        dev = 100.0 * (model / mom.value - 1.0) if mom.value else None
        targeted.append((mom.name, mom.value, model, dev, mom.weight))
    fr = compute_fiscal_ratios(panels, spec, cfg)
    fisc = cfg.get('fiscal', {})
    untarg_cfg = cfg.get('untargeted', {})
    untargeted = []
    for mom_key, data_key in UNTARGETED_DIST:
        data = untarg_cfg.get(data_key)
        if data is None:
            continue
        model = float(MOMENT_DISPATCH[mom_key](panels, spec))
        untargeted.append((mom_key, data, model, 100.0 * (model / data - 1.0)))
    targeted_keys = {mom.compute_key for mom in spec.moments} | {mom.name for mom in spec.moments}
    for key in UNTARGETED:
        if key in targeted_keys:
            continue                       # it is a target in this configuration
        model = None if 'error' in fr else fr.get(key)
        data = fisc.get(key)
        if data is None or model is None:
            continue                       # nothing to compare against
        untargeted.append((key, data, model, 100.0 * (model / data - 1.0)))
    return {'targeted': targeted, 'untargeted': untargeted}


def moments_table(md, live=None, targeted_keys=()):
    out = ['\\multicolumn{5}{l}{\\itshape Targeted}\\\\']
    if live:
        for name, data, model, dev, wt in live['targeted']:
            out.append(f'{LABEL.get(name, name)} & {fmt(data)} & {fmt(model)} '
                       f'& {fmt(dev, 2)} & {fmt(wt, 2)} \\\\')
    else:
        for r in parse_md_table(md, 'Targeted Moments'):
            name, data, model, dev, wt = (r + [''] * 5)[:5]
            out.append(f'{LABEL.get(name, name)} & {fmt(num(data))} & {fmt(num(model))} '
                       f'& {fmt(num(dev), 2)} & {fmt(num(wt), 2)} \\\\')
    out.append('\\midrule\n\\multicolumn{5}{l}{\\itshape Not targeted}\\\\')
    if live:
        rows = list(live['untargeted'])
    else:
        ratios = {r[0]: r for r in parse_md_table(md, 'Fiscal Ratios (model vs data, share of Y)')}
        rows = []
        for key in UNTARGETED:
            if key in targeted_keys:
                continue
            r = ratios.get(key)
            if r is None:
                rows.append((key, None, None, None))
                continue
            data, model = num(r[2]), num(r[1])
            # The markdown row's last column is the absolute gap; recompute
            # the relative deviation so the cell means what the header says.
            dev = 100.0 * (model / data - 1.0) if (data and model is not None) else None
            rows.append((key, data, model, dev))
    for key, data, model, dev in rows:
        if data is None:
            continue                       # no data counterpart, no comparison
        out.append(f'{LABEL.get(key, key)} & {fmt(data)} & {fmt(model)} & {fmt(dev, 2)} & \\\\')
    return '\n'.join(out)


def age_table(data_shares, model_t0=None, model_T=None, n_t0=None, n_T=None):
    def col(d, key):
        return NP if d is None else fmt(d.get(key), 3)
    rows = [('Share 25--39', 'young'), ('Share 40--64', 'mid'),
            ('Share 65--84', 'old'), ('Dependency (65--84 / 25--64)', 'oadr')]
    out = []
    for label, key in rows:
        out.append(f'{label} & {col(data_shares, key)} & {col(model_t0, key)} '
                   f'& {col(model_T, key)} \\\\')
    out.append(f'Population growth $n$ & {NP} & {fmt(n_t0, 4)} & {fmt(n_T, 4)} \\\\')
    return '\n'.join(out)


def shares_from_weights(w):
    """w over model ages 0..59 = real 25..84."""
    w = np.asarray(w, float); w = w / w.sum()
    young, mid, old = w[:15].sum(), w[15:40].sum(), w[40:].sum()
    return {'young': young, 'mid': mid, 'old': old, 'oadr': old / (young + mid)}


def growth_table(paths, gamma_minus_1, g, noise=None, t_stable=None):
    """paths: {label: detrended series}.

    The trend is the mean log growth per period over the periods in which the
    population has settled, since only there is the detrended series meant to
    be flat. t_stable is the first such period; without it the second half of
    the horizon is used.
    """
    order = [('$Y$', 'Y'), ('$C$', 'C'), ('$K^{dom}$', 'K_domestic'),
             ('$A$ (wealth)', 'A'), ('$L$', 'L'), ('$K_g$', 'K_g'), ('$B$', 'B')]
    out = []
    for label, key in order:
        x = paths.get(key) if paths else None
        if x is None or len(np.asarray(x)) < 8:
            trend = None
        else:
            x = np.asarray(x, float)
            h = len(x) // 2 if t_stable is None else min(int(t_stable), len(x) - 3)
            trend = 100 * (np.log(x[-1]) - np.log(x[h])) / (len(x) - 1 - h)
        out.append(f'{label} & {fmt(100*gamma_minus_1, 3)} & {fmt(100*g, 3)} '
                   f'& {fmt(trend, 3)} \\\\')
    out.append('\\midrule')
    out.append(f'Theory & {fmt(100*gamma_minus_1, 3)} & {fmt(100*g, 3)} & {{0}} \\\\')
    out.append(f'Noise floor ($g=n=0$) & {{}} & {{}} & {fmt(noise, 3)} \\\\')
    return '\n'.join(out)


def implied_stats(panels, spec, cfg):
    """Model statistics recomputed in the definitions the data use.

    Each calibrated parameter is identified by an aggregate ratio, but the data
    also measure the same object directly, and the two need not agree because
    the bases differ: tau_p applies to the whole wage bill with no contribution
    ceiling, rho_pens multiplies a career-average base rather than the final
    wage. Reporting revenue, base and rate separately shows where the distance
    sits.
    """
    from calibrate import compute_fiscal_ratios
    T = spec.base_config.T
    J_R = spec.base_config.retirement_age
    aw = spec.age_weights if spec.age_weights is not None else np.ones(T) / T
    tot = {k: 0.0 for k in ('wage', 'tax_p', 'pension_new', 'wage_pre',
                            'w_new', 'w_pre')}
    for edu, panel in panels.items():
        sh = spec.education_shares[edu]
        alive = panel.alive_sim.astype(bool)
        for t in range(T):
            a_t = alive[t]
            if not np.any(a_t):
                continue
            wt = sh * aw[t]
            # gross wage income excludes UI, which effective_y_sim includes
            wage = panel.effective_y_sim[t, a_t] - panel.ui_sim[t, a_t]
            tot['wage']  += wt * float(np.mean(wage))
            tot['tax_p'] += wt * float(np.mean(panel.tax_p_sim[t, a_t]))
            if t == J_R:                    # first period of retirement
                tot['pension_new'] += sh * float(np.mean(panel.pension_sim[t, a_t]))
                tot['w_new'] += sh
            if t == J_R - 1:                # last working period
                tot['wage_pre'] += sh * float(np.mean(wage))
                tot['w_pre'] += sh
    fr = compute_fiscal_ratios(panels, spec, cfg)
    Y = None if 'error' in fr else float(fr['Y'])
    ssc = tot['tax_p'] / tot['wage'] if tot['wage'] else None
    base = tot['wage'] / Y if Y else None
    repl = ((tot['pension_new'] / tot['w_new']) / (tot['wage_pre'] / tot['w_pre'])
            if tot['w_new'] and tot['w_pre'] and tot['wage_pre'] else None)
    return {'ssc_rev': None if Y is None else tot['tax_p'] / Y,
            'ssc_base': base, 'ssc_rate': ssc, 'replacement': repl}


# Data counterparts, Greece 2023, as shares. Contribution revenue and the
# factor-income shares are Eurostat (`nasa_10_nf_tr`, `nama_10_gdp`; see
# code/data_inventory.md 1.11); the base is compensation of employees plus
# mixed income, so the self-employed are inside it as they are in the model.
# The replacement rate is the sheet `Parameters` of DATA_GR.xlsx.
# The first three model entries follow from the production function and the
# tax base, not from the simulation: with L in efficiency units the wage bill
# is (1-alpha) Y, so the base is 1-alpha and the rate is tau_p itself; only the
# replacement row is a simulated statistic.
DATA_COUNTERPART = [
    ('ssc_rev',     'Social contributions / output',              0.130),
    ('ssc_base',    'Contribution base / output (model: $1-\\alpha$)', 0.570),
    ('ssc_rate',    'Contribution rate on that base (model: $\\tau^p$)', 0.228),
    ('replacement', 'Pension at retirement / last wage',          0.762750),
]


def implied_table(stats, cfg):
    out = []
    for key, label, data in DATA_COUNTERPART:
        model = (stats or {}).get(key)
        dev = (100 * (model / data - 1)) if (model and data) else None
        out.append(f'{label} & {fmt(data, 3)} & {fmt(model, 3)} & {fmt(dev, 1)} \\\\')
    return '\n'.join(out)


def goods_market_residual(paths, cfg, economy, T_tr, budget=None):
    """Resource-constraint residual of the no-debt baseline, as a share of output.

    Per capita and detrended, with NFA_p = A - K_dom the household sector's
    foreign assets (no sovereign debt is tracked in this run) and M total
    medical spending, the flows the code books satisfy

      Gamma_t NFA_p[t+1] - NFA_p[t]
        = Y + r NFA_p - C - I_priv - G - I_g - D - O - M + PD,

    where I_priv = Gamma_t K_dom[t+1] - (1-delta) K_dom[t] and PD is the
    primary deficit, which nobody finances in a baseline without a debt
    path: the taxes, benefits, transfers and the bequest tax net out between
    households and the government, leaving the government's purchases and the
    unfinanced deficit. The residual returned is

      C - (Y + r NFA_p - I_priv - G - I_g - D - O - M - dNFA_p + PD),

    every government line read from the budget the run produced, so that a
    spending share not passed to the simulation does not enter through the
    back door. Until 2026-10-02 the function omitted M and PD, took the
    spending from configuration ratios whether or not the budget carried them,
    and attributed the 0.1-0.2 residual to accidental bequests.

    None when the budget or a required path is absent rather than guessing.
    """
    need = ('Y', 'C', 'K_domestic', 'NFA')
    if budget is None or any(paths.get(k) is None for k in need):
        return None
    Y = np.asarray(paths['Y'], float)
    C = np.asarray(paths['C'], float)
    Kd = np.asarray(paths['K_domestic'], float)
    NFA = np.asarray(paths['NFA'], float)
    G_arr = np.asarray(economy.growth_factors(T_tr), float)
    delta = float(cfg['production'].get('delta', 0.07))
    kappa = float(cfg['external_params'].get('kappa', 1.0))
    n = min(len(Y), len(C), len(Kd), len(NFA)) - 1
    if n < 2:
        return None

    def line(key):
        v = budget.get(key)
        return np.zeros(n) if v is None else np.asarray(v, float)[:n]

    I_priv = G_arr[:n] * Kd[1:n + 1] - (1.0 - delta) * Kd[:n]
    dNFA = G_arr[:n] * NFA[1:n + 1] - NFA[:n]
    r = np.asarray(paths.get('r', np.full(len(Y), float(getattr(economy, 'r_star', 0.04) or 0.04))), float)
    nfi = r[:n] * NFA[:n]
    gov_purchases = (line('govt_spending') + line('public_investment')
                     + line('defense_spending') + line('other_net_spending'))
    M = line('gov_health') / kappa            # public + out-of-pocket medical spending
    PD = line('primary_deficit')
    resid = C[:n] - (Y[:n] + nfi - I_priv - gov_purchases - M - dNFA + PD)
    return {'resource': resid / Y[:n]}


# ---------------------------------------------------------------- figures ---
def make_figures(paths, w_data, w_model_t0, outdir):
    """Write the age-distribution figure always, the panels only with a run.

    An empty panel figure would overwrite one drawn from a real baseline
    transition, so when there is no transition the existing file is left alone
    and the template falls back to its placeholder.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels_pdf = os.path.join(outdir, 'baseline_panels.pdf')
    if paths is None and os.path.exists(panels_pdf):
        print('  kept baseline_panels.pdf (no baseline transition in this run)')
    else:
        _panels_figure(paths, panels_pdf, plt)
    _age_figure(w_data, w_model_t0, outdir, plt)


def _panels_figure(paths, out_pdf, plt):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    if paths:
        for key, lab in [('Y', r'$\hat y$'), ('C', r'$\hat c$'),
                         ('K_domestic', r'$\hat k^{dom}$'), ('L', r'$\hat\ell$'),
                         ('A', r'$\hat a$')]:
            x = paths.get(key)
            if x is None:
                continue
            x = np.asarray(x, float)
            ax[0].plot(x / x[0], label=lab, lw=1.4)
        ax[0].axhline(1.0, color='0.7', lw=0.8, zorder=0)
        ax[0].set_title('Detrended aggregates, each indexed to 1 in 2023')
        ax[0].set_xlabel('period ($t=0$ is 2023)')
        ax[0].legend(frameon=False, ncol=3, fontsize=8)
        Y = np.asarray(paths.get('Y'), float)
        for key, lab in [('B', '$B/Y$'), ('NFA', '$NFA/Y$'),
                         ('K_g', '$K_g/Y$'), ('A', '$A/Y$')]:
            x = paths.get(key)
            if x is None:
                continue
            x = np.asarray(x, float)[:len(Y)]
            ax[1].plot(x / Y[:len(x)], label=lab, lw=1.4)
        ax[1].axhline(0.0, color='0.7', lw=0.8, zorder=0)
        ax[1].set_title('Ratios to output')
        ax[1].set_xlabel('period ($t=0$ is 2023)')
        ax[1].legend(frameon=False, ncol=2, fontsize=8)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print('  wrote baseline_panels.pdf')


def _age_figure(w_data, w_model_t0, outdir, plt):
    fig, a = plt.subplots(figsize=(5.2, 3.0))
    ages = np.arange(25, 85)
    if w_data is not None:
        a.plot(ages, np.asarray(w_data) / np.sum(w_data), label='data 2023', lw=1.5)
    if w_model_t0 is not None:
        a.plot(ages, np.asarray(w_model_t0) / np.sum(w_model_t0),
               label='model $t=0$', lw=1.5, ls='--')
    a.set_xlabel('age'); a.set_ylabel('share'); a.legend(frameon=False, fontsize=8)
    a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'age_distribution.pdf'))
    plt.close(fig)
    print('  wrote age_distribution.pdf')


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='../calibration_input_GR.json')
    ap.add_argument('--calib-report', default=None,
                    help='markdown report from calibrate.py; newest if omitted')
    ap.add_argument('--outdir', default='../output/calibration_growth')
    ap.add_argument('--run-baseline', action='store_true')
    ap.add_argument('--n-sim', type=int, default=1000)
    ap.add_argument('--backend', default='jax')
    ap.add_argument('--implied', action='store_true',
                    help='compute the data-comparable statistics only')
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    report = args.calib_report
    if report is None:
        # The config records which run wrote its theta, so use that rather than
        # whichever markdown happens to be on the machine -- on a fresh
        # checkout that is some other run entirely, and the label lies.
        meta = cfg.get('_derived', {}).get('theta_metadata', {})
        src = meta.get('source_report')
        if src:
            for cand in (src, os.path.join(os.path.dirname(args.config), src),
                         os.path.join(outdir, 'reports', os.path.basename(src))):
                if os.path.exists(cand):
                    report = cand
                    break
            if report is None:
                print(f'  calibration report named by the config is not here: '
                      f'{os.path.basename(src)}; tables stay live from the config')
    if report is None and args.calib_report is None:
        cands = []
        for d in (os.path.join(outdir, 'reports'),
                  os.path.join(os.path.dirname(args.config), 'output', 'calibration')):
            if os.path.isdir(d):
                cands += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.md')]
        report = max(cands, key=os.path.basename) if cands else None
    md = open(report).read() if report else ''
    if report:
        print(f'calibration report: {os.path.basename(report)}')

    g = cfg['external_params'].get('trend_growth', 0.0)
    n = cfg['external_params'].get('pop_growth', 0.0)          # terminal rate; realised n_0 below
    n_inf_cfg = n
    gamma_minus_1 = (1 + g) * (1 + n) - 1                       # overwritten by the realised path

    # First period on the balanced growth path, from the demographic sidecar.
    t_stable = None
    demog = os.path.join(os.path.dirname(args.config), '..', 'data',
                         'demography_GR.npz')
    if os.path.exists(demog):
        _d = np.load(demog)
        t_stable = int(_d['stable_year']) - int(_d['base_year'])

    # Base-year age distribution (25-84). The demographic sidecar carries it as
    # cross_section_base and is tracked, whereas DATA_GR.xlsx is gitignored as
    # *.xlsx and the .npy cache is untracked -- neither reaches a fresh
    # checkout, which is how this column came out empty on the instance.
    w_data = None
    _demog = os.path.join(os.path.dirname(args.config), '..', 'data',
                          'demography_GR.npz')
    if os.path.exists(_demog):
        _cs = np.asarray(np.load(_demog)['cross_section_base'], dtype=float)
        w_data = _cs / _cs.sum()
    else:
        print(f'  age distribution unavailable: {_demog} not found')

    # The age weights come from the demographic path, so the age-structure
    # table and its figure need no solve; only the panels and the growth table
    # need a baseline transition.
    from calibrate import load_config, build_olg_transition
    L = load_config(args.config)
    economy, tp, T_TR = build_olg_transition(L['config_data'], backend=args.backend)

    def living_shares(t):
        """Shares of the living population by age at period t.

        The transition's weights are cohort sizes at entry, because its means
        run over all agents with the dead holding zero. The data column counts
        the living, so the weights are carried through cumulative survival
        before the two are compared.
        """
        w = economy._cohort_weights(t)
        cum = np.empty(economy.T)
        for j in range(economy.T):
            sched = economy._cohort_survival_schedule(t - j)
            cum[j] = float(np.prod(np.mean(sched, axis=1)[:j])) if j else 1.0
        out = w * cum
        return out / out.sum()

    w_model_t0 = living_shares(0)
    w_model_T = living_shares(T_TR - 1)
    G_path = economy.growth_factors(T_TR)
    n = float(G_path[0] / (1.0 + g) - 1.0)          # realised, not the config scalar
    n_T = float(G_path[-1] / (1.0 + g) - 1.0)
    gamma_minus_1 = float(G_path[0] - 1.0)           # Gamma_0 - 1, the base year
    gammaT_minus_1 = float(G_path[-1] - 1.0)         # Gamma_T - 1, the balanced growth path

    paths = None
    if args.run_baseline:
        prod = L['config_data']['production']
        I_g = ((prod.get('delta_g', 0.05) + economy.growth_factors(T_TR) - 1.0)
               * prod.get('K_g', 0.0))
        tax = {k: tp[k] for k in ('tau_c_path', 'tau_l_path', 'tau_p_path',
                                  'tau_k_path', 'pension_replacement_path')}
        print(f'baseline transition: T={T_TR}, n_sim={args.n_sim}, backend={args.backend}')
        fisc = L['config_data'].get('fiscal', {})
        # The spending shares go into the simulation so that the budget it
        # produces carries G, defence and the balancing item; the resource
        # constraint below is read off that budget, not off the configuration.
        res = economy.simulate_transition(r_path=tp['r_path'], I_g_path=I_g,
                                          n_sim=args.n_sim, verbose=False,
                                          G_over_Y=fisc.get('G_over_Y', 0.0),
                                          defense_over_Y=fisc.get('defense_over_Y', 0.0),
                                          other_net_over_Y=fisc.get('other_net_spending_over_Y', 0.0),
                                          **tax)
        paths = {k: np.asarray(v) for k, v in res.items()
                 if isinstance(v, (list, np.ndarray)) and np.ndim(v) == 1}
        budget = economy.compute_government_budget_path(n_sim=args.n_sim, verbose=False)
        budget = {k: np.asarray(v) for k, v in budget.items()
                  if isinstance(v, (list, np.ndarray)) and np.ndim(v) == 1}
        # Keep the paths, not just the picture of them. Questions about the
        # baseline -- why a ratio moves, where output turns -- otherwise cost a
        # full transition to answer again.
        np.savez(os.path.join(outdir, 'baseline_paths.npz'),
                 **{k: v for k, v in paths.items()},
                 **{('budget_' + k): v for k, v in budget.items()},
                 growth_factor=economy.growth_factors(T_TR),
                 delta=float(L['config_data']['production'].get('delta', 0.07)),
                 base_year=int(economy.current_year))
        print('  wrote baseline_paths.npz')

        # The resource constraint is the one identity that does not cancel a
        # normalisation error: every other check in the repo compares two
        # quantities that carry the same denominator, so a mismatch divides out.
        res = goods_market_residual(paths, L['config_data'], economy, T_TR, budget=budget)
        if res is not None:
            wr = float(np.max(np.abs(res['resource'])))
            print(f'  resource constraint, max |residual| / Y = {wr:.4f} '
                  f'(t=0: {res["resource"][0]:+.4f})')
            # Every booked flow is in the identity, so only the sampling noise
            # of a finite n_sim and the discrete timing of deaths remain.
            if wr > 0.01:
                print('  WARNING: the resource constraint is off by more than '
                      'sampling noise explains; a flow is missing or mis-dated')

    stats = None
    live = None
    if args.run_baseline or args.implied:
        from calibrate import run_model_moments
        import dataclasses
        L2 = L
        sp = dataclasses.replace(L2['spec'], backend=args.backend, n_sim=args.n_sim)
        from calibrate import theta_from_config
        th = theta_from_config(cfg, sp)
        print(f'implied statistics: solving at n_sim={args.n_sim} ...')
        _, panels = run_model_moments(th, sp, return_panels=True)
        # load_config injects _derived.K_over_L; the raw JSON has it as None,
        # which makes compute_fiscal_ratios return an error instead of Y.
        stats = implied_stats(panels, sp, L2['config_data'])
        live = live_moments(panels, sp, L2['config_data'])
        for key, lbl, dat in DATA_COUNTERPART:
            v = stats.get(key)
            print(f'  {lbl:40s} model {v if v is not None else float("nan"):.4f}'
                  f'   data {dat:.4f}')

    def wrap(colspec, header, body):
        return ('\\begin{tabular}{@{}' + colspec + '@{}}\n\\toprule\n'
                + header + ' \\\\\n\\midrule\n' + body
                + '\n\\bottomrule\n\\end{tabular}')

    frag = {
        'implied_body.tex': wrap(
            'lS[table-format=1.3]S[table-format=1.3]S[table-format=+3.1]',
            ' & {Data} & {Model} & {\\% dev}', implied_table(stats, cfg)),
        'params_body.tex': wrap(
            'llS[table-format=2.4]l',
            'Symbol & Description & {Value} & Source / identified by',
            params_table(cfg, n0=n, n_inf=n_T)),
        'moments_body.tex': wrap(
            'lS[table-format=1.4]S[table-format=1.4]S[table-format=+2.2]S[table-format=3.2]',
            ' & {Data} & {Model} & {\\% dev} & {Weight}',
            moments_table(md, live, targeted_keys={m.get('compute_key', m.get('name'))
                                                   for m in cfg.get('calibration', {}).get('targets', [])}
                          | {m.get('name') for m in cfg.get('calibration', {}).get('targets', [])})),
        'age_body.tex': wrap(
            'lS[table-format=1.3]S[table-format=1.3]S[table-format=1.3]',
            ' & {Data, 2023} & {Model, $t=0$} & {Model, terminal}',
            age_table(shares_from_weights(w_data) if w_data is not None else None,
                      shares_from_weights(w_model_t0) if w_model_t0 is not None else None,
                      shares_from_weights(w_model_T) if w_model_T is not None else None,
                      n, n_T)),
        'growth_body.tex': wrap(
            'lS[table-format=1.3]S[table-format=1.3]S[table-format=+1.3]',
            ' & {Level (\\%)} & {Per capita (\\%)} & {Detrended trend (\\%)}',
            growth_table(paths, gammaT_minus_1, g, t_stable=t_stable)),
    }
    # Without a baseline transition there is nothing to put in the growth
    # table's trend column or in the transition panels. Writing them anyway
    # destroys the output of a run that did have one, so skip them instead.
    if paths is None:
        for name in ('growth_body.tex',):
            if os.path.exists(os.path.join(outdir, name)):
                frag.pop(name, None)
                print(f'  kept {name} (no baseline transition in this run)')

    for name, body in frag.items():
        with open(os.path.join(outdir, name), 'w') as fh:
            fh.write(body + '\n')
        print(f'  wrote {name}')

    make_figures(paths, w_data, w_model_t0, outdir)

    meta = {'run': os.path.basename(report) if report else None,
            'g': g, 'n': n, 'n_T': n_T, 'Gamma_minus_1': gamma_minus_1,
            'GammaT_minus_1': gammaT_minus_1,
            'r_B': cfg['prices'].get('r_B'),
            'objective': num((parse_md_table(md, 'Summary') or [['', '']])[0][1])}
    with open(os.path.join(outdir, 'header.tex'), 'w') as fh:
        run = (meta['run'] or '').replace('_', '\\_') or NP
        for name, val in (('runlabel', run), ('gval', fmt(g, 4)),
                          ('nval', fmt(n, 4)), ('nTval', fmt(n_T, 4)),
                          ('rBval', fmt(meta['r_B'], 4)),
                          ('Gammaval', fmt(gamma_minus_1, 5)),
                          ('GammaTval', fmt(gammaT_minus_1, 5))):
            fh.write(f"\\providecommand{{\\{name}}}{{}}"
                     f"\\renewcommand{{\\{name}}}{{{val}}}\n")
    print('  wrote header.tex')


if __name__ == '__main__':
    main()
