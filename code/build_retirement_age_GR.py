"""
Build the statutory retirement age path for Greece → data/retirement_age_GR.npz.

Law 4336/2015 links the minimum and statutory retirement ages to the change in
life expectancy at 65 of the whole population, re-examined every three years
from 2021 (European Commission, 2024 Ageing Report, Country Fiche EL, December
2023, p. 6 item (x) and p. 9 section 1.1.5). The model has a single retirement
age, taken to be the statutory one:

  S_y = 67 + e65(r(y)) - e65(2023),    r(y) = last review year <= y,

with reviews in 2024, 2027, ... (the three-year cycle that started in 2021),
S_y = 67 for y <= 2023, and e65 the unisex period life expectancy at 65 (the
mean of men and women) from the EUROPOP2023 baseline mortality assumptions,
ages 65 to 100+. The projection ends in 2100; the model's mortality is held at
the 2100 schedule after that, so e65 and S are held there too.

The model is annual and ages are whole years. A cohort entering at real age
25 in year k retires at the whole age a that solves a = round(S_{k + a - 25}):
the statutory age in force in the year it reaches it, rounded to the nearest
year. Its model retirement index is J_R = a - 25, the first period of
retirement.

e65 reproduces the fiche's projections (men 18.7 in 2022 and 23.9 in 2070,
women 21.7 and 26.7); `test_retirement_age.py` checks that.

Usage (from code/):  python3 build_retirement_age_GR.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RAW = os.path.join(DATA, 'europop2023_raw')
OUT = os.path.join(DATA, 'retirement_age_GR.npz')

ENTRY_AGE = 25
BASE_YEAR = 2023
BASE_AGE = 67                   # statutory age in 2023
FIRST_REVIEW = 2024             # three-year cycle from 2021: 2021, 2024, 2027, ...
REVIEW_EVERY = 3
PROJ_END = 2100                 # last year of EUROPOP2023
FIRST_ENTRY = 1939              # oldest cohort alive in 2023 (real age 84)
LAST_ENTRY = 2210               # last entering cohort of the demographic path


def mortality_rates(sex):
    """EUROPOP2023 baseline mortality rates m[year, age], ages 0..100+."""
    d = json.load(open(os.path.join(RAW, f'proj_23naasmr_EL_{sex}.json')))
    ai = d['dimension']['age']['category']['index']
    ti = d['dimension']['time']['category']['index']
    na, nt = len(ai), len(ti)
    ages = sorted(ai, key=ai.get)
    years = [int(y) for y in sorted(ti, key=ti.get)]
    if ages[65] != 'Y65' or ages[-1] != 'Y_GE100':
        raise ValueError(f'unexpected age coding: {ages[65]}, {ages[-1]}')
    m = np.full((nt, na), np.nan)
    for k, v in d['value'].items():
        k = int(k)
        m[k % nt, k // nt] = v        # age is the outer index, time the inner one
    if np.isnan(m).any():
        raise ValueError(f'missing mortality rates for {sex}')
    return np.array(years), m


def life_expectancy_65(m_row):
    """Period life expectancy at 65 from rates m_65..m_100+.

    q = 1 - exp(-m), deaths at mid-year, and the open interval 100+ closed
    with L = l/m.
    """
    mx = np.asarray(m_row[65:], float)
    q = 1.0 - np.exp(-mx[:-1])
    l = np.cumprod(np.r_[1.0, 1.0 - q])               # survivors at 65..100
    L = l[:-1] * (1.0 - q) + 0.5 * l[:-1] * q
    return float(L.sum() + l[-1] / mx[-1])


def e65_path():
    yM, mM = mortality_rates('M')
    yF, mF = mortality_rates('F')
    if not np.array_equal(yM, yF):
        raise ValueError('men and women cover different years')
    eM = np.array([life_expectancy_65(r) for r in mM])
    eF = np.array([life_expectancy_65(r) for r in mF])
    return yM, eM, eF, 0.5 * (eM + eF)


def statutory_age(years, e_years, e65):
    """S_y for each year in `years` under the review rule."""
    e = dict(zip(e_years.tolist(), e65.tolist()))
    e_last = e[PROJ_END]
    out = np.empty(len(years))
    for i, y in enumerate(years):
        if y < FIRST_REVIEW:
            out[i] = BASE_AGE
            continue
        r = FIRST_REVIEW + REVIEW_EVERY * ((y - FIRST_REVIEW) // REVIEW_EVERY)
        out[i] = BASE_AGE + e.get(r, e_last) - e[BASE_YEAR]
    return out


def cohort_retirement_age(entry_year, S):
    """Whole retirement age a = round(S_{entry + a - 25}) for one cohort.

    S is non-decreasing, so iterating from the base age converges upward; the
    loop guards against a two-cycle by keeping the larger age.
    """
    a = BASE_AGE
    seen = set()
    while a not in seen:
        seen.add(a)
        y = entry_year + a - ENTRY_AGE
        a_new = int(np.floor(S(y) + 0.5))
        if a_new == a:
            return a
        a = a_new
    return max(seen)


def main():
    e_years, eM, eF, e65 = e65_path()
    years = np.arange(FIRST_ENTRY, LAST_ENTRY + 100)
    S = statutory_age(years, e_years, e65)
    S_of = dict(zip(years.tolist(), S.tolist()))
    entry = np.arange(FIRST_ENTRY, LAST_ENTRY + 1)
    age = np.array([cohort_retirement_age(int(k), S_of.__getitem__) for k in entry])
    np.savez(OUT, e65_years=e_years, e65_men=eM, e65_women=eF, e65=e65,
             years=years, statutory_age=S, entry_years=entry,
             retirement_real_age=age, J_R=age - ENTRY_AGE,
             base_year=BASE_YEAR, base_age=BASE_AGE, entry_age=ENTRY_AGE)
    for y in (2023, 2024, 2030, 2040, 2050, 2070, 2100):
        print(f'  {y}: e65 {e65[list(e_years).index(min(y, PROJ_END))]:.2f}  '
              f'statutory age {S_of[y]:.2f}')
    for k in (1939, 1981, 1990, 2000, 2023, 2050, 2100, 2210):
        print(f'  cohort entering {k} (aged 25): retires at {age[k - FIRST_ENTRY]}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
