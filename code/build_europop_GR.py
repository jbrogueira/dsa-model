"""
Build the EUROPOP2023 demographic path for Greece → data/europop2023_GR.npz.

Source: Eurostat dissemination API, baseline variant (BSL), Greece, 2022-2100.
  proj_23np      population on 1 January by single year of age and sex
  proj_23naasmr  assumed age-specific mortality rates by sex
  proj_23nanmig  assumed net migration by age and sex

The model enters at real age 25 (model age 0), T = 60 → real ages 25-84, and
its two demographic inputs are the size of each entering cohort and a survival
schedule. Both are taken here on the projection's own definitions:

  entrants[y]  count of 25-year-olds on 1 January of year y
  px[y, j]     probability of surviving from exact age 25+j to 26+j in year y

Mortality is published as a central rate m by sex; the two sexes are combined
at the projected population weights of that age and year, and the rate is
turned into a probability by q = m / (1 + m/2), px = 1 - q. Reproduces the
observed 2023 life table (demo_mlifetable, the source of survival_GR.npz) to
2.3e-3 at worst, so the projected table continues the historical one.

The model has no migration, so a cohort's simulated size follows entrants x
cumulative survival while the projection also adds migrants after age 25. The
script reports that gap; it is the price of reading demography off a
projection built with migration into a model without it.

Usage (from code/):  python3 build_europop_GR.py [--refresh]
"""
import argparse
import json
import os
import urllib.request

import numpy as np

API = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data'
RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'europop2023_raw')
OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'europop2023_GR.npz')
ENTRY_AGE = 25
T = 60                                   # model horizon → real ages 25..84
DATASETS = {'proj_23np': ('T', 'M', 'F'), 'proj_23naasmr': ('M', 'F'),
            'proj_23nanmig': ('T',)}


def fetch(dataset, sex, refresh=False):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f'{dataset}_EL_{sex}.json')
    if refresh or not os.path.exists(path):
        url = (f'{API}/{dataset}?format=JSON&lang=en&geo=EL'
               f'&sex={sex}&projection=BSL')
        print(f'  GET {dataset} sex={sex}')
        with urllib.request.urlopen(url, timeout=180) as r:
            data = r.read()
        with open(path, 'wb') as fh:
            fh.write(data)
    return json.load(open(path))


def as_age_year(doc, real_ages):
    """JSON-stat document → array (len(real_ages), n_years) plus the years."""
    ids, size = doc['id'], doc['size']
    cats = {k: (list(doc['dimension'][k]['category']['index'])
                if isinstance(doc['dimension'][k]['category']['index'], dict)
                else doc['dimension'][k]['category']['index']) for k in ids}
    flat = np.full(int(np.prod(size)), np.nan)
    for k, v in doc['value'].items():
        flat[int(k)] = np.nan if v is None else v
    cube = flat.reshape(size)
    ia, it = ids.index('age'), ids.index('time')
    panel = np.moveaxis(cube, [ia, it], [0, 1]).reshape(len(cats['age']),
                                                        len(cats['time']))
    col = {}
    for i, lab in enumerate(cats['age']):
        if lab == 'Y_LT1':
            col[0] = i
        elif lab.startswith('Y') and lab[1:].isdigit():
            col[int(lab[1:])] = i
    years = np.array([int(y) for y in cats['time']])
    return np.array([panel[col[a]] for a in real_ages]), years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true',
                    help='re-download even if the raw JSON is cached')
    args = ap.parse_args()

    real_ages = np.arange(ENTRY_AGE, ENTRY_AGE + T)
    print('EUROPOP2023, Greece, baseline variant')
    raw = {}
    for ds, sexes in DATASETS.items():
        for s in sexes:
            raw[(ds, s)] = fetch(ds, s, args.refresh)

    pop_T, years = as_age_year(raw[('proj_23np', 'T')], real_ages)
    pop_M, _ = as_age_year(raw[('proj_23np', 'M')], real_ages)
    pop_F, _ = as_age_year(raw[('proj_23np', 'F')], real_ages)
    m_M, _ = as_age_year(raw[('proj_23naasmr', 'M')], real_ages)
    m_F, _ = as_age_year(raw[('proj_23naasmr', 'F')], real_ages)
    mig, _ = as_age_year(raw[('proj_23nanmig', 'T')], real_ages)

    m_T = (m_M * pop_M + m_F * pop_F) / (pop_M + pop_F)
    px = 1.0 - m_T / (1.0 + 0.5 * m_T)
    assert np.all((px > 0) & (px <= 1)), 'px out of (0, 1]'

    entrants = pop_T[0]                                   # 25-year-olds, by year

    # Model-implied age distribution: each cohort is its entering size carried
    # forward by survival alone. The projection adds migrants after age 25.
    implied = np.zeros_like(pop_T)
    for i, y in enumerate(years):
        for j in range(T):
            k = i - j                                     # year that cohort entered
            if k < 0:
                implied[j, i] = np.nan
                continue
            s = np.prod([px[a, k + a] for a in range(j)]) if j else 1.0
            implied[j, i] = entrants[k] * s
    ok = ~np.isnan(implied)
    gap = np.full_like(implied, np.nan)
    gap[ok] = implied[ok] / pop_T[ok] - 1.0

    np.savez(OUT,
             years=years.astype(int),
             px=px.T.astype(float),                       # (Ny, 60), like survival_GR
             pop=pop_T.T.astype(float),
             net_migration=mig.T.astype(float),
             entrants=entrants.astype(float),
             model_ages=np.arange(T, dtype=int),
             real_ages=real_ages.astype(int))
    print(f'\nwrote {os.path.relpath(OUT)}: years {years[0]}..{years[-1]}, '
          f'px {px.T.shape}, real ages {real_ages[0]}..{real_ages[-1]}')

    def at(y):
        return list(years).index(y)

    print('\nEntering cohort (25-year-olds) and its growth rate')
    for y in (2023, 2030, 2040, 2050, 2060, 2070, 2083, 2100):
        i = at(y)
        gr = entrants[i] / entrants[i - 1] - 1 if i else np.nan
        print(f'  {y}  {entrants[i]:>10,.0f}   {100*gr:+.2f}%')

    print('\nTotal 25-84 population and its growth rate')
    tot = pop_T.sum(axis=0)
    for y in (2023, 2030, 2040, 2050, 2060, 2070, 2083, 2100):
        i = at(y)
        gr = tot[i] / tot[i - 1] - 1 if i else np.nan
        print(f'  {y}  {tot[i]:>10,.0f}   {100*gr:+.2f}%')

    oadr = pop_T[40:].sum(axis=0) / pop_T[:40].sum(axis=0)   # 65-84 over 25-64
    print('\nOld-age dependency ratio, 65-84 over 25-64')
    print('  ' + '  '.join(f'{y}: {oadr[at(y)]:.3f}'
                           for y in (2023, 2040, 2060, 2083, 2100)))

    print('\nModel-implied minus projected population (no migration in the model)')
    for y in (2033, 2043, 2063, 2083):
        i = at(y)
        col = gap[:, i]
        f = np.isfinite(col)
        print(f'  {y}: ages with a full model history {f.sum():2d}/60, '
              f'mean {100*np.nanmean(col[f]):+.1f}%, '
              f'worst {100*col[f][np.nanargmax(np.abs(col[f]))]:+.1f}%')


if __name__ == '__main__':
    main()
