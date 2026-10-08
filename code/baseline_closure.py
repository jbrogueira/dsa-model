"""
Debt dynamics and the fixed point of the baseline.

The government budget has no residual line: the primary balance is the model's
own, with the tax on gross output tau_y pinned in the base year to the
outturn (normalize_A_tfp.py). The Commission's projection is a comparison. This
module does the government-side accounting on a run:

Debt. The ratio d_t is the stock at the end of year t over the year's output,
the convention of the data and of the projection,

    d_t = d_{t-1} (1 + r_B,t) / (1 + g_t) - pb_t + sfa_t,     d_2023 = fiscal.B_over_Y,

with g_t = Gamma_{t-1} y_t / y_{t-1} - 1 the growth of output in detrended
per-capita units, r_B,t the real sovereign rate of the year (prices.r_B_file)
and pb_t the model's primary balance over output. In levels this is the law
of motion of fiscal_experiments.compute_debt_path, D_t = Gamma_t B_{t+1}.

The two years between the base year and the projection, 2024 and 2025, are
history: d_2024 is the data's ratio (the Debt Sustainability Monitor's 154.2%)
and d_2025 the projection's starting ratio (146.1%). The flow that reconciles
each with the recursion is recorded as that year's stock-flow adjustment.
Over 2026-2060 the adjustment is the projection's (the deferred interest on
the official loans to 2032); after 2060 it is zero.

Output tax. tau_y is constant at its base-year value through 2060 and moves
linearly over the following fiscal.tau_y_ramp_years (ten: 2061-2070) to the
rate that holds the debt ratio at its 2070 value over the following ten
years, d_2080 = d_2070, and stays there. (The one-year condition
pb_2070 = d_2070 [(1 + r_B,2070)/(1 + g_2070) - 1] is reported too; at a
debt ratio above one it moves with the growth rate of a single year, so the
ten-year window is the condition solved.) That rate is a fixed point: tau_y
moves the wage and so the household block, so solve_baseline() iterates full
transitions (a secant on the window residual, steps bounded).

Lump-sum transfer. The transfer per adult is lump_sum_over_Y times the run's
own output, a level path the household solve needs before output is known;
solve_baseline() iterates it with the tax rate. Output enters as a centred
moving average over lump_smooth_years: next-period assets are chosen on the
asset grid, so a household's consumption and hours jump when a small change
in resources moves its choice to the next node, and a transfer tied to the
unsmoothed path sustains a two-year alternation in labour input through the
fixed point. The experiments hold both paths fixed at the baseline's.
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


def tau_y_path(T, base_year, tau_base, tau_terminal, ramp_years=10, last_fixed_year=2060):
    """The output-tax path by period: tau_base through last_fixed_year (a
    scalar, or a (T,) path whose values up to that year are kept), linear from
    the rate of last_fixed_year to tau_terminal over the next ramp_years years,
    tau_terminal after."""
    t_fix = int(last_fixed_year - base_year)
    base = np.asarray(tau_base, dtype=float)
    out = (np.full(T, float(base)) if base.ndim == 0
           else np.concatenate([base[:T], np.full(max(T - len(base), 0), base[-1])]))
    start = float(out[min(t_fix, T - 1)])
    for t in range(T):
        if t <= t_fix:
            continue
        s = min((t - t_fix) / float(ramp_years), 1.0)
        out[t] = start + s * (float(tau_terminal) - start)
    return out


def debt_paths(years, Y, growth_factor, primary_balance, r_B, debt_over_Y_base, dsa,
               terminal_year=2070, window=10):
    """Debt, the stock-flow adjustment and the projection's series by year.

    years, Y, growth_factor   calendar years (T,), detrended per-capita output and Gamma_t
    primary_balance           the model's primary balance, level (T,)
    r_B                       real sovereign rate, scalar or (T,) by period
    debt_over_Y_base          debt at the end of the base year over its output
    dsa                       load_dsa_projection() (None: no pins, no adjustment)

    Returns a dict of (T,) arrays, shares of output unless noted: 'debt',
    'primary_balance', 'sfa', 'interest', 'growth' (a rate), 'stabilising_pb'
    (the balance that keeps d_t = d_{t-1}), 'debt_projection' and
    'primary_balance_projection' (nan outside the projection's years), the
    scalar 'terminal_residual' = pb - stabilising_pb in terminal_year, and
    'window_residual' = d in terminal_year + window less d in terminal_year
    (over the years available, 'window_years'; nan when none).
    """
    years = np.asarray(years, dtype=int)
    Y = np.asarray(Y, dtype=float)
    T = len(Y)
    g = output_growth(Y, growth_factor)
    rB = np.broadcast_to(np.asarray(r_B, dtype=float), (T,))
    pb = np.asarray(primary_balance, dtype=float)[:T] / Y

    proj, first, last, mon_year = {}, None, None, None
    if dsa is not None:
        proj = {int(y): i for i, y in enumerate(dsa['years'])}
        first, last = int(dsa['years'][0]), int(dsa['years'][-1])
        mon_year = int(dsa['monitor_year'])

    d = np.empty(T)
    sfa = np.zeros(T)
    interest = np.full(T, np.nan)
    stab = np.full(T, np.nan)
    d_proj = np.full(T, np.nan)
    pb_proj = np.full(T, np.nan)
    d[0] = float(debt_over_Y_base)
    for t in range(1, T):
        y = int(years[t])
        carried = d[t - 1] * (1.0 + rB[t]) / (1.0 + g[t])
        interest[t] = d[t - 1] * rB[t] / (1.0 + g[t])
        stab[t] = carried - d[t - 1]
        if dsa is not None and y == mon_year:
            d[t] = float(dsa['monitor_debt'])
            sfa[t] = d[t] - (carried - pb[t])
            pb_proj[t] = float(dsa['monitor_primary_balance'])
            d_proj[t] = d[t]
        elif dsa is not None and y == first:
            d[t] = float(dsa['debt'][proj[y]])
            sfa[t] = d[t] - (carried - pb[t])
            pb_proj[t] = float(dsa['primary_balance'][proj[y]])
            d_proj[t] = d[t]
        elif dsa is not None and first < y <= last:
            sfa[t] = float(dsa['sfa'][proj[y]])
            d[t] = carried - pb[t] + sfa[t]
            pb_proj[t] = float(dsa['primary_balance'][proj[y]])
            d_proj[t] = float(dsa['debt'][proj[y]])
        else:
            d[t] = carried - pb[t]
    t_term = int(terminal_year - years[0])
    resid = float(pb[t_term] - stab[t_term]) if 0 < t_term < T else np.nan
    W = min(int(window), T - 1 - t_term) if 0 < t_term < T else 0
    w_resid = float(d[t_term + W] - d[t_term]) if W >= 1 else np.nan
    return {'debt': d, 'primary_balance': pb, 'sfa': sfa, 'interest': interest,
            'growth': g, 'stabilising_pb': stab, 'debt_projection': d_proj,
            'primary_balance_projection': pb_proj, 'terminal_residual': resid,
            'window_residual': w_resid, 'window_years': int(W),
            'terminal_year': int(terminal_year)}


def debt_from_run(paths, raw, dsa=None, r_B_path=None):
    """debt_paths() for a saved baseline (the arrays of baseline_paths.npz).

    `paths` maps names to arrays: Y, growth_factor, base_year and the budget
    lines budget_total_revenue and budget_total_spending. The rate path is
    paths['r_B_path'] when saved, else the argument, else prices.r_B.
    """
    Y = np.asarray(paths['Y'], dtype=float)
    T = len(Y)
    years = int(paths['base_year']) + np.arange(T)
    rev = np.asarray(paths['budget_total_revenue'], dtype=float)[:T]
    spend = np.asarray(paths['budget_total_spending'], dtype=float)[:T]
    rb = paths.get('r_B_path')
    if rb is None:
        rb = r_B_path
    if rb is None:
        rb = float(raw['prices']['r_B'])
    fiscal = raw.get('fiscal', {})
    out = debt_paths(years, Y, np.asarray(paths['growth_factor'], dtype=float)[:T],
                     rev - spend, rb, float(fiscal.get('B_over_Y', 0.0)),
                     load_dsa_projection(raw) if dsa is None else dsa)
    out['years'] = years
    out['tau_y_path'] = (np.asarray(paths['tau_y_path'], dtype=float)[:T]
                         if paths.get('tau_y_path') is not None else None)
    return out


def centred_mean(x, years):
    """Centred moving average over `years` points, edge-padded; x itself for years <= 1."""
    x = np.asarray(x, dtype=float)
    if years <= 1:
        return x
    half = int(years) // 2
    pad = np.pad(x, (half, half), mode='edge')
    return np.convolve(pad, np.ones(int(years)) / float(years), mode='valid')


def solve_baseline(run, config_data, T_tr, base_year, lump_sum_over_Y, tau_base,
                   r_B_path, growth_factor, Y_init=None, tau_terminal_init=None,
                   ramp_years=10, max_iter=8, tol_Y=1e-4, tol_pb=1e-4, verbose=True,
                   step_max=0.03, tau_bounds=(-0.10, 0.60), tol_tau=5e-4,
                   match_projection=False, lump_smooth_years=5):
    """Fixed point of the baseline over the lump-sum level path and the
    terminal output-tax rate.

    run(lump_sum_path, tau_y_path) -> (Y_path, budget) runs one transition
    with those household inputs and returns detrended per-capita output (at
    least T_tr long) and the budget dict (levels). Returns a dict with the
    converged 'lump_sum_path', 'tau_y_path', 'tau_terminal', the last run's
    'Y', 'budget' and 'debt' (debt_paths()), and 'iterations'.

    With match_projection the rate is a path over 2024-2060 as well: in each
    iteration the rate of every projected year moves by (pb_projection -
    pb)/0.7, so that the model's primary balance reproduces the Commission's
    (the closure of 2026-10-05 carried by the output tax instead of a residual
    line; a fixed point over full transitions, since the path moves the
    wage). The ramp then starts from the 2060 rate. The default, a constant
    rate to 2060, is the baseline of the report.

    The lump sum of iteration k is lump_sum_over_Y times the output of
    iteration k-1 (Y_init, or one, before the first run), output taken as a
    centred moving average over lump_smooth_years (see the module
    docstring). The terminal rate
    is updated by a secant on the window residual of debt_paths(), the change
    in the debt ratio over the ten years after the terminal year; the first
    step takes a slope of -0.7 per year of the window per unit of the rate
    (one unit of the rate raises the balance by about 0.7 of output a year:
    the direct revenue less the fall of the labour-income bases). Steps are
    bounded by step_max and the rate by tau_bounds.
    """
    dsa = load_dsa_projection(config_data)
    fiscal = config_data.get('fiscal', {})
    B_over_Y = float(fiscal.get('B_over_Y', 0.0))
    years = base_year + np.arange(T_tr)
    Y_prev = np.ones(T_tr) if Y_init is None else np.asarray(Y_init, dtype=float)[:T_tr]
    tau_T = float(tau_base if tau_terminal_init is None else tau_terminal_init)
    tau_fixed = np.full(T_tr, float(tau_base))     # the rates up to 2060
    hist = []          # (tau_T, residual)
    out = None
    for k in range(1, max_iter + 1):
        lump = float(lump_sum_over_Y) * centred_mean(Y_prev, lump_smooth_years)
        tau = tau_y_path(T_tr, base_year, tau_fixed, tau_T, ramp_years)
        Y, budget = run(lump, tau)
        Y = np.asarray(Y, dtype=float)[:T_tr]
        pb_level = (np.asarray(budget['total_revenue'], dtype=float)[:T_tr]
                    - np.asarray(budget['total_spending'], dtype=float)[:T_tr])
        debt = debt_paths(years, Y, np.asarray(growth_factor, dtype=float)[:T_tr], pb_level,
                          np.asarray(r_B_path, dtype=float)[:T_tr], B_over_Y, dsa)
        resid = debt['window_residual']
        W = debt['window_years']
        # A horizon that ends at or before the terminal year (a test run) has
        # no ramp to solve: only the lump-sum path is iterated.
        update_tau = not np.isnan(resid)
        if not update_tau:
            resid = 0.0
        dY = float(np.max(np.abs(Y - Y_prev) / Y))
        proj_gap = 0.0
        if match_projection:
            pb_proj = debt['primary_balance_projection']
            has = ~np.isnan(pb_proj)
            has[0] = False
            gap = np.where(has, pb_proj - debt['primary_balance'], 0.0)
            proj_gap = float(np.max(np.abs(gap)))
            tau_fixed = np.clip(tau_fixed + np.clip(gap / 0.7, -step_max, step_max),
                                tau_bounds[0], tau_bounds[1])
        out = {'lump_sum_path': lump, 'tau_y_path': tau, 'tau_terminal': tau_T, 'Y': Y,
               'budget': budget, 'debt': debt, 'iterations': k,
               'projection_gap': proj_gap}
        if verbose:
            t70 = debt['terminal_year'] - base_year
            print(f'  baseline fixed point {k}: tau_y^T = {tau_T:.5f}, debt ratio change over '
                  f'{W} years from {debt["terminal_year"]} = {resid:+.5f}, one-year residual '
                  f'{debt["terminal_residual"]:+.5f}, max |dY|/Y = {dY:.2e}, '
                  f'debt {100 * debt["debt"][t70]:.1f}% in {debt["terminal_year"]}'
                  + (f', max |pb - projection| = {proj_gap:.5f}' if match_projection else ''),
                  flush=True)
        hist.append((tau_T, resid))
        if abs(resid) < tol_pb * max(W, 1) and dY < tol_Y and proj_gap < tol_pb:
            break
        # Two successive rates within tol_tau with a settled output path: the
        # residual is at the resolution of the model's response.
        if (len(hist) >= 2 and abs(hist[-1][0] - hist[-2][0]) < tol_tau and dY < tol_Y):
            break
        if not update_tau:
            Y_prev = Y
            continue
        # Secant on the window residual (first step at the a-priori slope,
        # -0.7 per year of the window); a slope of the wrong sign or a step
        # beyond step_max falls back to the a-priori slope or the bound.
        prior = -0.7 * max(W, 1)
        slope = prior
        if len(hist) >= 2 and hist[-1][0] != hist[-2][0] and hist[-1][1] != hist[-2][1]:
            s_est = (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0])
            if 0.2 * abs(prior) <= -s_est <= 5.0 * abs(prior):
                slope = s_est
        step = -resid / slope
        step = float(np.clip(step, -step_max, step_max))
        tau_T = float(np.clip(tau_T + step, tau_bounds[0], tau_bounds[1]))
        Y_prev = Y
    return out
