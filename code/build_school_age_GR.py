"""
Build the school-age population index for Greece -> data/school_age_GR.npz.

Public education spending scales with the school-age population relative to
the model's population. Both come from EUROPOP2023 (Eurostat proj_23np,
baseline variant, population on 1 January by single year of age, 2022-2100),
cached in data/europop2023_raw/proj_23np_EL_T.json by build_europop_GR.py:

  S_t      population aged 5-24
  N_t      population aged 25-84 (the model's ages)
  index_t  = (S_t / N_t) / (S_2023 / N_2023).

The index is one in the model's base year, 2023. It holds the 2022 value
before 2022 and the 2100 value after 2100, over 1900-2400.

Usage (from code/):  python3 build_school_age_GR.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', 'data', 'europop2023_raw', 'proj_23np_EL_T.json')
OUT = os.path.join(HERE, '..', 'data', 'school_age_GR.npz')

BASE_YEAR = 2023
SCHOOL_AGES = np.arange(5, 25)             # 5-24
MODEL_AGES = np.arange(25, 85)             # 25-84
FIRST_YEAR, LAST_YEAR = 1900, 2400
PRINT_YEARS = (2023, 2030, 2040, 2050, 2060, 2070, 2100)


def as_age_year(doc, real_ages):
    """JSON-stat document -> array (len(real_ages), n_years) plus the years.

    Same logic as build_europop_GR.as_age_year: age codes 'Y_LT1' -> 0 and
    'Y<n>' -> n; the open and aggregate codes ('Y_GE100', 'TOTAL', 'Y_LT15',
    'Y_GE65', ...) are not single ages and are skipped.
    """
    ids, size = doc['id'], doc['size']
    cats = {}
    for k in ids:
        index = doc['dimension'][k]['category']['index']
        cats[k] = sorted(index, key=index.get) if isinstance(index, dict) else list(index)
    flat = np.full(int(np.prod(size)), np.nan)
    for k, v in doc['value'].items():
        flat[int(k)] = np.nan if v is None else v
    cube = flat.reshape(size)
    ia, it = ids.index('age'), ids.index('time')
    panel = np.moveaxis(cube, [ia, it], [0, 1]).reshape(len(cats['age']), len(cats['time']))
    col = {}
    for i, lab in enumerate(cats['age']):
        if lab == 'Y_LT1':
            col[0] = i
        elif lab.startswith('Y') and lab[1:].isdigit():
            col[int(lab[1:])] = i
    years = np.array([int(y) for y in cats['time']])
    return np.array([panel[col[a]] for a in real_ages]), years


def main():
    doc = json.load(open(RAW))
    school, data_years = as_age_year(doc, SCHOOL_AGES)
    model, _ = as_age_year(doc, MODEL_AGES)
    if np.isnan(school).any() or np.isnan(model).any():
        raise ValueError('missing population cells')
    S, N = school.sum(axis=0), model.sum(axis=0)
    ratio = S / N
    base = float(ratio[data_years == BASE_YEAR][0])
    data_index = ratio / base

    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    index = np.interp(years, data_years, data_index)     # flat outside 2022-2100
    np.savez(OUT, years=years, index=index, base_year=BASE_YEAR,
             data_years=data_years, S=S, N=N,
             school_ages=SCHOOL_AGES, model_ages=MODEL_AGES)
    for y in PRINT_YEARS:
        k = int(np.where(data_years == y)[0][0])
        print(f'  {y}: S {S[k] / 1e3:8.1f} thousand  N {N[k] / 1e3:8.1f} thousand  '
              f'S/N {ratio[k]:.4f}  index {float(index[years == y][0]):.4f}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
