"""
Build the retirement age path for Greece → data/retirement_age_GR.npz.

The model's retirement age is the average effective retirement age, the age at
which people start receiving an old-age, early or disability pension: 63.8 in
2022 (European Commission, 2024 Ageing Report, Country Fiche EL, December 2023,
p. 23), against a statutory age of 67. Law 4336/2015 links the minimum and
statutory ages to the change in life expectancy at 65, re-examined every three
years from 2021 (p. 6 item (x), p. 9 section 1.1.5). The effective age is moved
by the same rule:

  R_y = 63.8 + e65(r(y)) - e65(2023),    r(y) = last review year <= y,

with reviews in 2024, 2027, ... (the three-year cycle that started in 2021),
R_y = 63.8 for y <= 2023, and e65 the unisex period life expectancy at 65 (the
mean of men and women) from the EUROPOP2023 baseline mortality assumptions,
ages 65 to 100+. The projection ends in 2100; the model's mortality is held at
the 2100 schedule after that, so e65 and R are held there too.

A cohort entering at real age 25 in year k retires on average at the age A_k
that solves A_k = R_{k + round(A_k) - 25}: the age in force in the year it
reaches it. The model is annual, so the cohort is split: a share
1 - frac(A_k) retires at floor(A_k), the rest one year later. Aggregates then
move continuously with R instead of in one-year steps. The model retirement
index of the first part is J_R = floor(A_k) - 25, its first period of
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
BASE_AGE = 63.8                 # average effective retirement age, 2022 (fiche p. 23)
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
    """R_y for each year in `years` under the review rule."""
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


def cohort_retirement_age(entry_year, R):
    """Average retirement age A = R_{entry + round(A) - 25} of one cohort.

    R is non-decreasing, so iterating from the base age converges upward; the
    loop guards against a two-cycle by keeping the larger age.
    """
    A = float(BASE_AGE)
    seen = set()
    while round(A) not in seen:
        seen.add(round(A))
        A_new = float(R(entry_year + int(round(A)) - ENTRY_AGE))
        if round(A_new) == round(A):
            return A_new
        A = A_new
    return float(R(entry_year + max(seen) - ENTRY_AGE))


def main():
    e_years, eM, eF, e65 = e65_path()
    years = np.arange(FIRST_ENTRY, LAST_ENTRY + 100)
    R = statutory_age(years, e_years, e65)
    R_of = dict(zip(years.tolist(), R.tolist()))
    entry = np.arange(FIRST_ENTRY, LAST_ENTRY + 1)
    A = np.array([cohort_retirement_age(int(k), R_of.__getitem__) for k in entry])
    low = np.floor(A).astype(int)
    share_high = A - low
    np.savez(OUT, e65_years=e_years, e65_men=eM, e65_women=eF, e65=e65,
             years=years, retirement_age_path=R, entry_years=entry,
             retirement_real_age=A, J_R=low - ENTRY_AGE, share_later=share_high,
             base_year=BASE_YEAR, base_age=BASE_AGE, entry_age=ENTRY_AGE)
    for y in (2023, 2024, 2030, 2040, 2050, 2070, 2100):
        print(f'  {y}: e65 {e65[list(e_years).index(min(y, PROJ_END))]:.2f}  '
              f'retirement age {R_of[y]:.2f}')
    for k in (1939, 1960, 1990, 2000, 2023, 2050, 2100, 2210):
        i = k - FIRST_ENTRY
        print(f'  cohort entering {k} (aged 25): average age {A[i]:.2f} = '
              f'{1 - share_high[i]:.2f} at {low[i]}, {share_high[i]:.2f} at {low[i] + 1}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
