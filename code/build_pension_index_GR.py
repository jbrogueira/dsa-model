"""
Build the path of the pension per pensioner for Greece -> data/pension_index_GR.npz.

The model's replacement rate follows the projection of the 2024 Ageing Report
(European Commission, Country Fiche EL, December 2023) for the average pension
relative to output per employed person:

  index_y = (public pension spending / GDP)_y / (pensioners / employment)_y,

with gross public pension spending from Table 6 (p. 26) and the number of
pensioners and employment from Table 10 (p. 34), both given for 2022 and every
tenth year from 2030 to 2070. The index is interpolated linearly between those
years, held at its 2070 value afterwards, and normalised to one in the model's
base year, 2023, where the calibration fits the level of pension spending.
Before the base year it is one.

Usage (from code/):  python3 build_pension_index_GR.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'pension_index_GR.npz')

BASE_YEAR = 2023
FICHE_YEARS = (2022, 2030, 2040, 2050, 2060, 2070)
PENSIONS_OVER_GDP = (14.5, 12.7, 13.7, 14.0, 12.7, 12.0)            # Table 6, % of GDP
PENSIONERS = (2460.4, 2503.4, 2764.9, 2958.6, 2742.0, 2510.8)       # Table 10, thousand
EMPLOYMENT = (4155.2, 4034.5, 3705.7, 3384.8, 3191.4, 3131.6)       # Table 10, thousand


def fiche_index():
    """Pension per pensioner over GDP per employed person, 2022 = 1."""
    x = (np.array(PENSIONS_OVER_GDP) / 100.0) / (np.array(PENSIONERS) / np.array(EMPLOYMENT))
    return x / x[0]


def index_path(years):
    """The index for each calendar year in `years`, one in the base year."""
    years = np.asarray(years, dtype=float)
    raw = np.interp(years, FICHE_YEARS, fiche_index())     # flat outside the range
    base = float(np.interp(BASE_YEAR, FICHE_YEARS, fiche_index()))
    return np.where(years <= BASE_YEAR, 1.0, raw / base)


def main():
    years = np.arange(BASE_YEAR - 100, FICHE_YEARS[-1] + 1)
    idx = index_path(years)
    np.savez(OUT, years=years, index=idx, base_year=BASE_YEAR,
             fiche_years=np.array(FICHE_YEARS), fiche_index=fiche_index(),
             pensions_over_gdp=np.array(PENSIONS_OVER_GDP),
             pensioners=np.array(PENSIONERS), employment=np.array(EMPLOYMENT))
    for y, v in zip(FICHE_YEARS, fiche_index()):
        print(f'  {y}: fiche index {v:.4f}   model index {float(index_path([y])[0]):.4f}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
