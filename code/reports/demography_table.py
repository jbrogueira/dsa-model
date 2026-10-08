"""Demographic statistics: observed (Eurostat), EUROPOP2023 baseline, the 2024
Ageing Report country fiche for Greece, and the model -> demography_body.tex.

Sources (all read from cached raw files; nothing is fetched here):
  data/eurostat_raw/demo_pjan_EL_T_2022_2025.json   Population on 1 January by age and sex
      API: .../data/demo_pjan?geo=EL&sex=T&unit=NR&time=2022&time=2023&time=2024&time=2025
  data/eurostat_raw/demo_gind_EL_2020_2024.json     Population change, demographic balance
      API: .../data/demo_gind?geo=EL&time=2020..2024
  data/eurostat_raw/demo_mlexpec_EL_2020_2024.json  Life expectancy by age and sex, age 65
      API: .../data/demo_mlexpec?geo=EL&age=Y65&time=2020..2024
  data/eurostat_raw/demo_find_EL_2020_2024.json     Fertility indicators, total fertility rate
      API: .../data/demo_find?geo=EL&indic_de=TOTFERRT&time=2020..2024
  data/europop2023_raw/proj_23np_EL_T.json, proj_23nanmig_EL_T.json
      EUROPOP2023 baseline (BSL): population on 1 January by age; net migration by age
  data/retirement_age_GR.npz                        e65 by year from the EUROPOP2023 mortality
      assumptions (build_retirement_age_GR.py); the model's retirement age path
  data/demography_GR.npz                            the model's population (build_demography_GR.py)
  2024 Ageing Report, country fiche for Greece (December 2023), Table 2 "Main
      demographic variables evolution", section 2.1: typed below as FICHE.
  The API root is https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0;
  files were read on 2026-10-08 (dataset 'updated' stamps printed below).

Usage (from code/): python3 reports/demography_table.py [--outdir ../output/calibration_growth]
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data')
YEARS = [2022, 2023, 2024, 2025, 2030, 2050, 2070]

# 2024 Ageing Report, country fiche for Greece, Table 2 (section 2.1).
FICHE = {
    'pop': {2022: 10438, 2030: 10004, 2040: 9475, 2050: 8935, 2060: 8318, 2070: 7777},
    'oadr': {2022: 39.0, 2030: 46.0, 2040: 60.6, 2050: 74.4, 2060: 72.1, 2070: 66.0},
    'e65_m': {2022: 18.7, 2030: 19.8, 2040: 20.9, 2050: 22.0, 2060: 23.0, 2070: 23.9},
    'e65_f': {2022: 21.7, 2030: 22.7, 2040: 23.8, 2050: 24.8, 2060: 25.8, 2070: 26.7},
    'mig': {2022: 21.5, 2030: -4.3, 2040: 5.2, 2050: 8.2, 2060: 12.6, 2070: 19.5},
}


def jsonstat(path):
    doc = json.load(open(path))
    ids, size = doc['id'], doc['size']
    cats = {k: list(doc['dimension'][k]['category']['index']) for k in ids}
    flat = np.full(int(np.prod(size)), np.nan)
    for k, v in doc['value'].items():
        flat[int(k)] = np.nan if v is None else v
    cube = flat.reshape(size)

    def get(**sel):
        idx = []
        for k in ids:
            if k in sel:
                idx.append(cats[k].index(str(sel[k])))
            elif len(cats[k]) == 1:
                idx.append(0)
            else:
                raise KeyError(f'dimension {k} needs a selection: {cats[k][:5]}...')
        return float(cube[tuple(idx)])
    return get, cats, doc.get('updated')


def by_age(get, cats, year):
    """Population by single age 0..99 plus the open group at 100, for one year."""
    out = {}
    for lab in cats['age']:
        if lab == 'Y_LT1':
            out[0] = get(age=lab, time=year)
        elif re.fullmatch(r'Y\d+', lab):
            out[int(lab[1:])] = get(age=lab, time=year)
        elif lab in ('Y_OPEN', 'Y_GE100'):
            out[100] = get(age=lab, time=year)
    ages = np.array(sorted(out))
    return ages, np.array([out[a] for a in ages])


def stats(ages, P):
    s = lambda lo, hi: P[(ages >= lo) & (ages <= hi)].sum()
    return {'pop': P.sum() / 1e3, 'pop2599': s(25, 99) / 1e3,
            'oadr': 100 * s(65, 200) / s(20, 64), 'oadr_m': 100 * s(65, 99) / s(25, 64),
            'sh85': 100 * s(85, 200) / s(65, 200), 'sh85_m': 100 * s(85, 99) / s(65, 99),
            'a25': s(25, 25) / 1e3}


def add_growth(d, key='pop2599', out='g2599'):
    """Growth of the 25-99 population from the previous year, %, where both years are present."""
    for y in list(d):
        if y - 1 in d and key in d[y - 1] and key in d[y]:
            d[y][out] = 100 * (d[y][key] / d[y - 1][key] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(HERE, '..', 'output', 'calibration_growth'))
    args = ap.parse_args()
    raw = os.path.join(DATA, 'eurostat_raw')

    # ---------------------------------------------------------- observed ---
    obs = {}
    get, cats, upd = jsonstat(os.path.join(raw, 'demo_pjan_EL_T_2022_2025.json'))
    print('demo_pjan updated', upd)
    for y in (2022, 2023, 2024, 2025):
        obs[y] = stats(*by_age(get, cats, y))
    add_growth(obs)
    get, cats, upd = jsonstat(os.path.join(raw, 'demo_gind_EL_2020_2024.json'))
    print('demo_gind updated', upd)
    for y in (2022, 2023, 2024):
        obs[y].update(births=get(indic_de='LBIRTH', time=y) / 1e3, deaths=get(indic_de='DEATH', time=y) / 1e3,
                      mig=get(indic_de='CNMIGRAT', time=y) / 1e3)
    get, cats, upd = jsonstat(os.path.join(raw, 'demo_mlexpec_EL_2020_2024.json'))
    print('demo_mlexpec updated', upd)
    for y in (2022, 2023, 2024):
        obs[y].update(e65_m=get(sex='M', time=y), e65_f=get(sex='F', time=y))
    get, cats, upd = jsonstat(os.path.join(raw, 'demo_find_EL_2020_2024.json'))
    print('demo_find updated', upd)
    for y in (2022, 2023, 2024):
        obs[y].update(tfr=get(time=y))

    # --------------------------------------------------------- EUROPOP2023 ---
    eu = {}
    get, cats, _ = jsonstat(os.path.join(DATA, 'europop2023_raw', 'proj_23np_EL_T.json'))
    for y in sorted(y for y in set(YEARS) | {y - 1 for y in YEARS} if y >= 2022):
        eu[y] = stats(*by_age(get, cats, y))
    add_growth(eu)
    get, cats, _ = jsonstat(os.path.join(DATA, 'europop2023_raw', 'proj_23nanmig_EL_T.json'))
    for y in YEARS:
        ages, M = by_age(get, cats, y)
        eu[y]['mig'] = M.sum() / 1e3
    ret = np.load(os.path.join(DATA, 'retirement_age_GR.npz'))
    ey = ret['e65_years'].astype(int).tolist()
    for y in YEARS:
        eu[y]['e65_m'] = float(ret['e65_men'][ey.index(y)])
        eu[y]['e65_f'] = float(ret['e65_women'][ey.index(y)])

    # --------------------------------------------------------------- model ---
    dem = np.load(os.path.join(DATA, 'demography_GR.npz'))
    pop = np.asarray(dem['pop'], float); py = dem['pop_years'].astype(int).tolist()
    T = pop.shape[1]; ages = 25 + np.arange(T)
    ent = np.asarray(dem['entrants'], float); ey2 = dem['entrant_years'].astype(int).tolist()
    tab = dict(zip(ret['entry_years'].tolist(), zip(ret['J_R'].tolist(), ret['share_later'].tolist())))
    lo_e, hi_e = min(tab), max(tab)
    ry = ret['years'].astype(int).tolist()
    mod = {}
    for y in YEARS:
        if y not in py:
            continue
        P = pop[py.index(y)]
        s = lambda lo, hi: P[(ages >= lo) & (ages <= hi)].sum()
        R = W = 0.0
        for j in range(T):
            JR, sl = tab[int(min(max(y - j, lo_e), hi_e))]
            r = 1.0 if j >= JR + 1 else ((1.0 - sl) if j >= JR else 0.0)
            R += P[j] * r; W += P[j] * (1 - r)
        mod[y] = {'pop2599': P.sum() / 1e3, 'oadr_m': 100 * s(65, 99) / s(25, 64),
                  'sh85_m': 100 * s(85, 99) / s(65, 99), 'a25': ent[ey2.index(y)] / 1e3,
                  'rw': R / W, 'ret': float(ret['retirement_age_path'][ry.index(y)]),
                  'e65_m': eu[y]['e65_m'], 'e65_f': eu[y]['e65_f']}
        if y - 1 in py:
            mod[y]['g2599'] = 100 * (pop[py.index(y)].sum() / pop[py.index(y - 1)].sum() - 1)

    # ---------------------------------------------------------------- table ---
    def cell(d, y, k, f):
        v = d.get(y, {}).get(k)
        return '{\\textemdash}' if v is None or (isinstance(v, float) and np.isnan(v)) else f % v
    rows = []
    def block(title, d, items):
        rows.append('\\addlinespace\\multicolumn{%d}{@{}l}{\\emph{%s}}\\\\' % (len(YEARS) + 1, title))
        for label, key, f in items:
            rows.append(label + ' & ' + ' & '.join(cell(d, y, key, f) for y in YEARS) + ' \\\\')
    block('Observed (Eurostat); population on 1 January, flows in the year', obs, [
        ('Population, thousand', 'pop', '%.0f'), ('Population aged 25--99, thousand', 'pop2599', '%.0f'),
        ('Persons 65+ per 100 aged 20--64', 'oadr', '%.1f'), ('Persons 65--99 per 100 aged 25--64', 'oadr_m', '%.1f'),
        ('Share of 85+ in 65+, \\%', 'sh85', '%.1f'), ('25-year-olds, thousand', 'a25', '%.1f'),
        ('Live births, thousand', 'births', '%.1f'), ('Deaths, thousand', 'deaths', '%.1f'),
        ('Net migration, thousand', 'mig', '%.1f'), ('Life expectancy at 65, men', 'e65_m', '%.1f'),
        ('Life expectancy at 65, women', 'e65_f', '%.1f'), ('Total fertility rate', 'tfr', '%.2f')])
    block('EUROPOP2023 baseline (the Ageing Report\'s population)', eu, [
        ('Population, thousand', 'pop', '%.0f'), ('Population aged 25--99, thousand', 'pop2599', '%.0f'),
        ('Persons 65+ per 100 aged 20--64', 'oadr', '%.1f'), ('Persons 65--99 per 100 aged 25--64', 'oadr_m', '%.1f'),
        ('Share of 85+ in 65+, \\%', 'sh85', '%.1f'), ('25-year-olds, thousand', 'a25', '%.1f'),
        ('Net migration, thousand', 'mig', '%.1f'), ('Life expectancy at 65, men', 'e65_m', '%.1f'),
        ('Life expectancy at 65, women', 'e65_f', '%.1f')])
    fiche = {y: {k: FICHE[k].get(y) for k in FICHE} for y in YEARS}
    block('2024 Ageing Report, country fiche for Greece, Table 2', fiche, [
        ('Population, thousand', 'pop', '%.0f'), ('Persons 65+ per 100 aged 20--64', 'oadr', '%.1f'),
        ('Net migration, thousand', 'mig', '%.1f'), ('Life expectancy at 65, men', 'e65_m', '%.1f'),
        ('Life expectancy at 65, women', 'e65_f', '%.1f')])
    block('Model (ages 25--99, entering cohorts smoothed)', mod, [
        ('Population aged 25--99, thousand', 'pop2599', '%.0f'), ('Persons 65--99 per 100 aged 25--64', 'oadr_m', '%.1f'),
        ('Share of 85--99 in 65--99, \\%', 'sh85_m', '%.1f'), ('Cohort entering at 25, thousand', 'a25', '%.1f'),
        ('Retired per non-retired person', 'rw', '%.2f'), ('Retirement age, years', 'ret', '%.1f')])
    head = ('\\begin{tabular}{@{}l' + 'S[table-format=5.1]' * len(YEARS) + '@{}}\n\\toprule\n & '
            + ' & '.join('{%d}' % y for y in YEARS) + ' \\\\\n\\midrule\n')
    body = head + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, 'demography_body.tex')
    open(out, 'w').write(body)
    print('wrote', os.path.relpath(out))

    # Short table: statistics defined the same way in the data, the projection
    # and the model, then a few that are not directly comparable.
    SY = [2023, 2025, 2030, 2050, 2070]
    srows = []
    def triple(label, key, f, srcs=(('data', obs), ('projection', eu), ('model', mod))):
        srows.append('%s \\\\' % label + '')
        for name, d in srcs:
            srows.append('\\quad %s & ' % name + ' & '.join(cell(d, y, key, f) for y in SY) + ' \\\\')
    srows.append('\\multicolumn{%d}{@{}l}{\\emph{Defined identically}}\\\\' % (len(SY) + 1))
    triple('Population aged 25--99, thousand', 'pop2599', '%.0f')
    triple('Growth of the population aged 25--99 from the previous year, \\%', 'g2599', '%+.2f')
    triple('Persons 65--99 per 100 aged 25--64', 'oadr_m', '%.1f')
    triple('Share of 85--99 in 65--99, \\%', 'sh85_m', '%.1f')
    srows.append('\\addlinespace\\multicolumn{%d}{@{}l}{\\emph{Not directly comparable}}\\\\' % (len(SY) + 1))
    triple('Persons 65+ per 100 aged 20--64 (the Ageing Report\'s ratio)', 'oadr', '%.1f',
           srcs=(('data', obs), ('projection', eu), ('Ageing Report fiche', fiche)))
    triple('25-year-olds (data, projection); cohort entering at 25 (model), thousand', 'a25', '%.1f')
    triple('Life expectancy at 65, men', 'e65_m', '%.1f')
    triple('Life expectancy at 65, women', 'e65_f', '%.1f')
    srows.append('Retired per non-retired person, model & ' + ' & '.join(cell(mod, y, 'rw', '%.2f') for y in SY) + ' \\\\')
    srows.append('Average effective retirement age, model, years & ' + ' & '.join(cell(mod, y, 'ret', '%.1f') for y in SY) + ' \\\\')
    shead = ('\\begin{tabular}{@{}l' + 'S[table-format=4.2]' * len(SY) + '@{}}\n\\toprule\n & '
             + ' & '.join('{%d}' % y for y in SY) + ' \\\\\n\\midrule\n')
    sbody = shead + '\n'.join(srows) + '\n\\bottomrule\n\\end{tabular}\n'
    out2 = os.path.join(args.outdir, 'demography_short_body.tex')
    open(out2, 'w').write(sbody)
    print('wrote', os.path.relpath(out2))
    for src, d in (('observed', obs), ('EUROPOP', eu), ('model', mod)):
        print(src, {y: {k: round(v, 2) for k, v in d[y].items()} for y in d})


if __name__ == '__main__':
    main()
