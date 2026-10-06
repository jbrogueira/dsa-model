"""
Fiscal closure of the baseline: other net spending as the residual that
delivers a target path for the primary balance, and the debt path it implies.

Other net spending O is primary expenditure less revenue with no explicit line
in the model. No household pays or receives it, so its path can be set after
the household block is solved, from the run's own budget lines:

  O_t / Y_t = pb_t^{ex O} - pb_t^{target},

with pb^{ex O} the primary balance before O, as a share of output. The target
is

  base year      the model's own primary balance at fiscal.other_net_spending_over_Y,
                 which pin_baseline_closure.py sets to the outturn;
  to the end of  the primary balance of the Commission's projection
  the projection (data/dsa_projection_GR.npz: the Debt Sustainability Monitor's
                 2024 value, then the DSA file to 2060);
  afterwards     the primary balance that holds the debt ratio at its last
                 projected value, pb_t = d_{t-1} [(1 + r_B)/(1 + g_t) - 1].

Debt is the stock at the end of the year over the same year's output, the
convention of the data and of the projection:

  d_t = d_{t-1} (1 + r_B) / (1 + g_t) - pb_t + sfa_t,     d_base = fiscal.B_over_Y,

with g_t the growth of output, Gamma_{t-1} y_t / y_{t-1} - 1 in detrended
per-capita units, and r_B a real rate. The stock-flow adjustment sfa is the
projection's in its years. In the years between the base year and the first
year of the projection it is the residual that reproduces the debt ratios of
the data (2024) and of the projection's first year (2025); it is zero after
the projection.

In levels the recursion is D_t = (1 + r_B) D_{t-1} / Gamma_{t-1} + PD_t + SFA_t,
the law of motion of fiscal_experiments.compute_debt_path with D_t = Gamma_t B_{t+1}.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dsa_projection(raw):
    """The projection named by fiscal.dsa_projection_file, or None."""
    rel = raw.get('fiscal', {}).get('dsa_projection_file')
    if not rel:
        return None
    path = rel if os.path.isabs(rel) else os.path.join(HERE, rel)
    if not os.path.exists(path):
        print(f'  [baseline_closure] dsa_projection_file not found: {path}')
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def output_growth(Y, growth_factor):
    """g_t = Gamma_{t-1} y_t / y_{t-1} - 1 for t >= 1; nan at t = 0."""
    Y = np.asarray(Y, dtype=float)
    G = np.asarray(growth_factor, dtype=float)
    return np.r_[np.nan, G[:len(Y) - 1] * Y[1:] / Y[:-1] - 1.0]


def closure_paths(years, Y, growth_factor, primary_balance_ex_other, other_over_Y_base,
                  r_B, debt_over_Y_base, dsa):
    """Target primary balance, O/Y, stock-flow adjustment and debt, by year.

    years                      calendar years of the run, (T,), years[0] the base year
    Y, growth_factor           detrended per-capita output and Gamma_t, (T,)
    primary_balance_ex_other   revenue less primary spending before O, level, (T,)
    other_over_Y_base          O/Y in the base year (the config's scalar)
    r_B                        real sovereign rate, scalar or (T,)
    debt_over_Y_base           debt at the end of the base year over its output
    dsa                        load_dsa_projection()

    Returns a dict of (T,) arrays, all shares of output: 'primary_balance',
    'other_net_over_Y', 'sfa', 'debt', 'interest', and 'growth' (a rate).
    """
    years = np.asarray(years, dtype=int)
    Y = np.asarray(Y, dtype=float)
    T = len(Y)
    g = output_growth(Y, growth_factor)
    rB = np.broadcast_to(np.asarray(r_B, dtype=float), (T,))
    pb_ex = np.asarray(primary_balance_ex_other, dtype=float)[:T] / Y

    proj = {int(y): i for i, y in enumerate(dsa['years'])}
    first, last = int(dsa['years'][0]), int(dsa['years'][-1])
    mon_year = int(dsa['monitor_year'])

    pb = np.empty(T)
    sfa = np.zeros(T)
    d = np.empty(T)
    interest = np.full(T, np.nan)
    pb[0] = pb_ex[0] - float(other_over_Y_base)
    d[0] = float(debt_over_Y_base)
    for t in range(1, T):
        y = int(years[t])
        carried = d[t - 1] * (1.0 + rB[t]) / (1.0 + g[t])
        interest[t] = d[t - 1] * rB[t] / (1.0 + g[t])
        if y < first:
            # Between the base year and the projection: the Monitor's year, and
            # the debt ratio of the data.
            if y != mon_year:
                raise ValueError(f'no primary balance for {y}: the projection starts in '
                                 f'{first} and the Monitor value is for {mon_year}')
            pb[t] = float(dsa['monitor_primary_balance'])
            sfa[t] = float(dsa['monitor_debt']) - (carried - pb[t])
        elif y <= last:
            i = proj[y]
            pb[t] = float(dsa['primary_balance'][i])
            # The first projected year reproduces the projection's own stock.
            sfa[t] = (float(dsa['debt'][i]) - (carried - pb[t]) if y == first
                      else float(dsa['sfa'][i]))
        else:
            pb[t] = carried - d[t - 1]
        d[t] = carried - pb[t] + sfa[t]
    return {'primary_balance': pb, 'other_net_over_Y': pb_ex - pb, 'sfa': sfa,
            'debt': d, 'interest': interest, 'growth': g,
            'primary_balance_ex_other': pb_ex}


def closure_from_run(paths, raw, dsa=None):
    """closure_paths() for a saved baseline (the arrays of baseline_paths.npz).

    `paths` maps names to arrays: Y, growth_factor, base_year and the budget
    lines budget_total_revenue, budget_total_spending, budget_other_net_spending.
    None when no projection is configured.
    """
    dsa = load_dsa_projection(raw) if dsa is None else dsa
    if dsa is None:
        return None
    Y = np.asarray(paths['Y'], dtype=float)
    T = len(Y)
    years = int(paths['base_year']) + np.arange(T)
    rev = np.asarray(paths['budget_total_revenue'], dtype=float)[:T]
    spend = np.asarray(paths['budget_total_spending'], dtype=float)[:T]
    other = np.asarray(paths['budget_other_net_spending'], dtype=float)[:T]
    pb_ex = rev - (spend - other)
    fiscal = raw.get('fiscal', {})
    out = closure_paths(years, Y, np.asarray(paths['growth_factor'], dtype=float)[:T],
                        pb_ex, other[0] / Y[0], float(raw['prices']['r_B']),
                        float(fiscal.get('B_over_Y', 0.0)), dsa)
    out['years'] = years
    return out
