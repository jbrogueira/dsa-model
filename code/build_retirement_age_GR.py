"""
Build the retirement age path for Greece → data/retirement_age_GR.npz.

The model's retirement age is the average effective retirement age, the age at
which people start receiving an old-age, early or disability pension. Its path
to 2070 is the projection of the 2024 Ageing Report (European Commission,
Country Fiche EL, December 2023, Table 4, p. 23): 63.8 in 2022, 65.5 in 2030,
66.4 in 2040, 66.6 in 2050, 67.4 in 2060 and 67.9 in 2070, interpolated
linearly between those years and equal to 63.8 before 2022. The statutory age
is 67 in 2022.

After 2070 the path rises with life expectancy at 65 at a pass-through of
0.75 (`PASS_THROUGH`), applied year by year:

  R_y = 67.9 + 0.75 (e65(y) - e65(2070)),

with e65 the unisex period life expectancy at 65 (the mean of men and women)
from the EUROPOP2023 baseline mortality assumptions, ages 65 to 100+. Law
4336/2015 links the minimum and statutory ages one for one to longevity
(fiche p. 6 item (x), p. 9 section 1.1.5). The pass-through is the ratio in
the fiche's own projection over 2022-70. The statutory age rises 5.5 years,
from 67 to 72.5 (Table 1a, p. 12-13). The average effective retirement age
rises 4.1 years, from 63.8 to 67.9 (Table 4). The ratio is 0.745, rounded to
0.75. The law re-examines the ages every three years. Applied literally the
path would move in steps every three years, and each step would move the
hours of the cohorts at the margin in one year, so the annual path is used.
The projection ends in 2100. The model's mortality is held at the 2100
schedule after that, so e65 and R are held there too. The one-for-one path
used before gave 70.3 in 2100; this one gives 69.7. `rule_path` in the
output is the literal review rule, R_y = 63.8 + e65(r(y)) - e65(2023) with
r(y) the last review year (2024, 2027, ...), the path used before the Ageing
Report's projection was adopted.

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
# Average effective retirement age, fiche Table 4 (p. 23).
FICHE_YEARS = (2022, 2030, 2040, 2050, 2060, 2070)
FICHE_AGES = (63.8, 65.5, 66.4, 66.6, 67.4, 67.9)
# Rise in the average effective retirement age per year of e65 after 2070:
# the fiche's 2022-70 rise in the effective age (4.1 years, Table 4) over the
# rise in the statutory age (5.5 years, Table 1a), which follows e65 one for one.
PASS_THROUGH = 0.75
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


def effective_age(years, e_years, e65):
    """R_y for each year in `years`: the fiche's path to 2070, then
    R_y = 67.9 + PASS_THROUGH * (e65(y) - e65(2070)), year by year."""
    years = np.asarray(years)
    e = dict(zip(e_years.tolist(), e65.tolist()))
    e_at = lambda y: e.get(int(y), e[PROJ_END])
    out = np.interp(years, FICHE_YEARS, FICHE_AGES)      # flat outside the range
    late = years > FICHE_YEARS[-1]
    de65 = np.array([e_at(y) for y in years[late]]) - e_at(FICHE_YEARS[-1])
    out[late] = FICHE_AGES[-1] + PASS_THROUGH * de65
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
    R = effective_age(years, e_years, e65)
    rule = statutory_age(years, e_years, e65)
    R_of = dict(zip(years.tolist(), R.tolist()))
    entry = np.arange(FIRST_ENTRY, LAST_ENTRY + 1)
    A = np.array([cohort_retirement_age(int(k), R_of.__getitem__) for k in entry])
    low = np.floor(A).astype(int)
    share_high = A - low
    np.savez(OUT, e65_years=e_years, e65_men=eM, e65_women=eF, e65=e65,
             years=years, retirement_age_path=R, rule_path=rule, entry_years=entry,
             retirement_real_age=A, J_R=low - ENTRY_AGE, share_later=share_high,
             base_year=BASE_YEAR, base_age=BASE_AGE, entry_age=ENTRY_AGE,
             pass_through=PASS_THROUGH)
    print(f'  after 2070: R = {FICHE_AGES[-1]} + {PASS_THROUGH} * (e65(y) - e65(2070))')
    for y in (2023, 2024, 2030, 2040, 2050, 2070, 2080, 2100):
        print(f'  {y}: e65 {e65[list(e_years).index(min(y, PROJ_END))]:.2f}  '
              f'retirement age {R_of[y]:.2f}')
    for k in (1939, 1960, 1990, 2000, 2023, 2050, 2100, 2210):
        i = k - FIRST_ENTRY
        print(f'  cohort entering {k} (aged 25): average age {A[i]:.2f} = '
              f'{1 - share_high[i]:.2f} at {low[i]}, {share_high[i]:.2f} at {low[i] + 1}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
