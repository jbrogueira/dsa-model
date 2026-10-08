"""
Build the model's demographic path for Greece → data/demography_GR.npz.

Combines the historical life tables in data/survival_GR.npz (Eurostat
demo_mlifetable, 1961-2023) with the EUROPOP2023 baseline projection in
data/europop2023_GR.npz (2022-2100, built by build_europop_GR.py) and an
assumed tail, and turns them into the two objects the model consumes:

  px[year, j]    probability of surviving from exact age 25+j to 26+j in
                 that year.  Historical through 2023, projected 2024-2100,
                 then held at the 2100 schedule.
  entrants[year] size of the cohort entering at real age 25 in that year.

Net migration is routed through the entry age (TREND_GROWTH_PLAN Step 0): the
model creates cohorts only at age 25, so the entering cohort is the residual
that makes the model's 25-84 population equal the projection's,

    B_y = P_y - sum_{j>=1} B_{y-j} * S_j(y-j),

with the cohorts already alive in 2023 pinned by the measured cross-section
divided by cumulative survival.

The entering-cohort series is then smoothed with a centred moving average of
SMOOTH_WINDOW years in logs over the cohorts that are measured or projected
(entering 1949-2100), and the population is rebuilt from the smoothed series.
Adjacent single-year cohorts in the Greek age distribution differ in size by
5% on average and by up to 24% at the oldest ages, and every budget line
priced by cohort would otherwise jump with whichever cohort enters, retires or
leaves the model in a given year.  With the five-year window the 25-99 total
stays within 0.3% of EUROPOP2023 in every year to 2100 and the old-age ratio
moves by less than 0.005; the measured 2023 cross-section is kept in the file
as cross_section_measured.

The historical life table starts in 1961.  The cohorts over 86 in 2023
entered before that, and their survival in those years is taken as the 1961
schedule.  Those entries describe survival at ages a cohort has already
passed, which never enters a solution: the 2023 cross-section is reproduced
exactly whatever the past schedule, and from 2024 on only projected survival
is applied.

Past the projection the population is taken to a balanced growth path:
mortality is held at the 2100 schedule and entering-cohort growth is ramped
linearly to n_inf over RAMP_YEARS and held there.  The age distribution is
then exactly stable one full lifespan after the ramp ends -- from 2180, when
every cohort alive both entered under the constant regime and has faced the
constant life table throughout -- so the terminal state is a genuine balanced
growth path rather than an approach to one.

Usage (from code/):  python3 build_demography_GR.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(DATA, 'demography_GR.npz')

ENTRY_AGE = 25
T = 75                          # model horizon → real ages 25..99
SMOOTH_WINDOW = 5               # centred moving average of entering cohorts, in logs
BASE_YEAR = 2023                # t = 0
PROJ_END = 2100                 # last year of EUROPOP2023
RAMP_YEARS = 20                 # 2100 → 2120
N_INF = 0.0                     # terminal growth of entering cohorts
ENTRANT_END = 2210              # covers T_transition = 180 from 2023
PX_END = ENTRANT_END + T        # last year the cohort entering in ENTRANT_END needs


def main():
    hist = np.load(os.path.join(DATA, 'survival_GR.npz'))
    proj = np.load(os.path.join(DATA, 'europop2023_GR.npz'))
    hy, py = list(hist['years']), list(proj['years'])

    # ---------------------------------------------------------------- px ---
    first_entry = BASE_YEAR - T + 1                        # oldest cohort alive in 2023
    years = np.arange(min(hy[0], first_entry), PX_END + 1)
    px = np.zeros((len(years), T))
    for i, y in enumerate(years):
        if y < hy[0]:
            px[i] = hist['px'][0]                          # 1961 schedule before 1961
        elif y <= 2023:
            px[i] = hist['px'][hy.index(y)]
        elif y <= PROJ_END:
            px[i] = proj['px'][py.index(y)]
        else:
            px[i] = proj['px'][py.index(PROJ_END)]
    yi = {int(y): i for i, y in enumerate(years)}

    def S(k, j):
        """Cumulative survival of the cohort entering in year k, to model age j."""
        if j <= 0:
            return 1.0
        return float(np.prod([px[yi[k + a], a] for a in range(j)]))

    # ---------------------------------------------------------- entrants ---
    cross_2023 = proj['pop'][py.index(BASE_YEAR)]          # living, by model age
    B = {}
    for j in range(T):                                     # cohorts alive in 2023
        B[BASE_YEAR - j] = cross_2023[j] / S(BASE_YEAR - j, j)

    pop_proj = proj['pop'].sum(axis=1)                     # projected 25-84 total
    for y in range(BASE_YEAR + 1, PROJ_END + 1):           # entry-age residual rule
        carried = sum(B[y - j] * S(y - j, j) for j in range(1, T) if (y - j) in B)
        B[y] = pop_proj[py.index(y)] - carried
    assert all(B[y] > 0 for y in B), 'entry-age rule produced a non-positive cohort'

    # Smooth the measured and projected cohorts (entering 1949-2100): centred
    # moving average in logs, edge-padded at both ends.
    sm_years = np.arange(first_entry, PROJ_END + 1)
    raw_B = np.array([B[int(y)] for y in sm_years])
    half = SMOOTH_WINDOW // 2
    padded = np.pad(np.log(raw_B), (half, half), mode='edge')
    smooth_B = np.exp(np.convolve(padded, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                                  mode='valid'))
    B_measured = dict(B)
    for y, b in zip(sm_years, smooth_B):
        B[int(y)] = float(b)

    # Tail: ramp the growth rate of entering cohorts to N_INF, then hold.
    g_last = float(np.mean([B[y] / B[y - 1] - 1
                            for y in range(PROJ_END - 9, PROJ_END + 1)]))
    for i, y in enumerate(range(PROJ_END + 1, PROJ_END + RAMP_YEARS + 1), start=1):
        g = g_last + (N_INF - g_last) * i / RAMP_YEARS
        B[y] = B[y - 1] * (1.0 + g)
    for y in range(PROJ_END + RAMP_YEARS + 1, ENTRANT_END + 1):
        B[y] = B[y - 1] * (1.0 + N_INF)

    e_years = np.arange(BASE_YEAR - T + 1, ENTRANT_END + 1)
    entrants = np.array([B[int(y)] for y in e_years])

    # ------------------------------------------------- population by year ---
    # Living count by model age, model definition: entrants x cumulative survival.
    p_years = np.arange(BASE_YEAR, ENTRANT_END + 1)
    pop = np.zeros((len(p_years), T))
    for i, y in enumerate(p_years):
        for j in range(T):
            pop[i, j] = B[int(y) - j] * S(int(y) - j, j)
    level = pop.sum(axis=1)
    n_path = np.full(len(p_years), N_INF)
    n_path[:-1] = level[1:] / level[:-1] - 1.0
    assert abs(n_path[-2] - N_INF) < 1e-12, 'population has not settled at the horizon'

    sh = pop / pop.sum(axis=1, keepdims=True)
    moved = np.abs(np.diff(sh, axis=0)).max(axis=1)
    stable_year = int(p_years[1:][moved < 1e-14][0])

    np.savez(OUT,
             years=years.astype(int), px=px.astype(float),
             entrant_years=e_years.astype(int), entrants=entrants.astype(float),
             pop_years=p_years.astype(int), pop=pop.astype(float),
             pop_level=level.astype(float), n_path=n_path.astype(float),
             cross_section_base=pop[list(p_years).index(BASE_YEAR)].astype(float),
             cross_section_measured=cross_2023.astype(float),
             base_year=BASE_YEAR, n_inf=N_INF, stable_year=stable_year,
             model_ages=np.arange(T, dtype=int),
             real_ages=np.arange(ENTRY_AGE, ENTRY_AGE + T, dtype=int))
    print(f'wrote {os.path.relpath(OUT)}')
    print(f'  px        years {years[0]}..{years[-1]}  shape {px.shape}')
    print(f'  entrants  years {e_years[0]}..{e_years[-1]}')
    print(f'  ramp from {100*g_last:+.3f}%/yr (mean over {PROJ_END-9}-{PROJ_END}) '
          f'to {100*N_INF:+.2f}% by {PROJ_END+RAMP_YEARS}')

    # ------------------------------------------------------------ checks ---
    def pi(y):
        return list(p_years).index(y)

    rec = pop[pi(BASE_YEAR)]
    print(f'\n  smoothing: std of the log change between adjacent cohorts '
          f'{np.diff(np.log(raw_B)).std():.3f} measured, '
          f'{np.diff(np.log(smooth_B)).std():.3f} smoothed; '
          f'base-year cross-section moved by {100*np.abs(rec/cross_2023 - 1).max():.1f}% at most')
    err = max(abs(level[pi(y)] / pop_proj[py.index(y)] - 1)
              for y in range(BASE_YEAR, PROJ_END + 1))
    print(f'  {ENTRY_AGE}-{ENTRY_AGE+T-1} total within {100*err:.2f}% of EUROPOP2023 '
          f'over {BASE_YEAR}-{PROJ_END}')

    print(f'\n  population growth and old-age dependency, 65-{ENTRY_AGE+T-1} over 25-64')
    for y in (2030, 2050, 2070, 2100, 2120, 2150, 2179):
        i = pi(y)
        o = pop[i, 40:].sum() / pop[i, :40].sum()
        print(f'    {y}  n = {100*n_path[i]:+.3f}%   OADR {o:.3f}')

    print(f'\n  age distribution exactly constant from {stable_year} '
          f'(entrants constant from {PROJ_END+RAMP_YEARS+1}, mortality from '
          f'{PROJ_END+1}, one lifespan later)')
    print(f'  a transition starting {BASE_YEAR} reaches it at t = {stable_year-BASE_YEAR}')


if __name__ == '__main__':
    main()
