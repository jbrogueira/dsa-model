"""
Build the path of the unemployment rate of 25-64 year olds for Greece
-> data/unemployment_index_GR.npz.

The model's unemployment rates by education group are the 2023 Eurostat rates
for ages 25-64 (lfsa_urgaed, ISCED 0-2, 3-4 and 5-8: 12.3, 11.6 and 7.7
percent). Each of them is scaled over the transition by the index

  index_y = u_y / u_2023,

where u_y is the unemployment rate of 25-64 year olds, both sexes, all
education levels, in calendar year y. The index is one up to the base year,
2023. After it the rate follows

  2024-2025  the Eurostat annual rates (lfsa_urgaed, geo EL, sex T, age
             Y25-64, isced11 TOTAL): 10.2 in 2023, 9.5 in 2024, 8.3 in 2025;
  2026-2027  the Commission's Spring 2026 forecast of the unemployment rate
             for ages 15-74, 8.3 and 7.9, times the 2023 ratio of the 25-64
             rate to the 15-74 rate, 10.2 / 11.1;
  2028-2050  a straight line to the 2050 rate of the 2024 Ageing Report for
             ages 20-64, 6.6 (Table II.1.50), times the 2023 ratio of the
             25-64 rate to the Report's 20-64 rate, 10.2 / 12.09, where 12.09
             interpolates the Report's 2022 and 2030 values, 12.4 and 9.9;
  2051-2055  a straight line to the Report's 2055 rate, 6.5, converted the
             same way;
  after 2055 constant.

The Report's rates for 2025 to 2045 (10.9 falling to 7.5) lie above the 2025
outturn and are not used; only its 2050 and 2055 levels enter.

Sources. Eurostat dissemination API, dataset lfsa_urgaed, raw JSON cached in
data/eurostat_raw/. European Commission, European Economic Forecast, Spring
2026, Greece page (economy-finance.ec.europa.eu), table "Indicators 2025 2026
2027", row "Unemployment (%)": 8.9, 8.3, 7.9. European Commission, 2024 Ageing
Report, Economic and Budgetary Projections for the EU Member States
(2022-2070), Institutional Paper 279, Table II.1.50, row EL (p. 197).

Usage (from code/):  python3 build_unemployment_path_GR.py [--refresh]
"""
import argparse
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data'
RAW = os.path.join(HERE, '..', 'data', 'eurostat_raw')
OUT = os.path.join(HERE, '..', 'data', 'unemployment_index_GR.npz')

BASE_YEAR = 2023
FIRST_YEAR, LAST_YEAR = 1900, 2400
DATA_YEARS = (2023, 2024, 2025)                   # Eurostat outturns, ages 25-64

# Commission Spring 2026 forecast, unemployment rate, ages 15-74, percent.
# The 2025 entry is the outturn printed in the same table; it stands in for
# 2025 only if Eurostat has not yet published that year.
EC_RATES_15_74 = {2025: 8.9, 2026: 8.3, 2027: 7.9}

# 2024 Ageing Report, Table II.1.50, unemployment rate 20-64, row EL, percent.
AR_YEARS = (2022, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070)
AR_RATES = (12.4, 10.9, 9.9, 9.4, 8.5, 7.5, 6.6, 6.5, 6.5, 6.5, 6.5)
AR_ANCHOR_YEARS = (2050, 2055)                    # the only Report values used
AR_BASE_BRACKET = (2022, 2030)                    # Report years that bracket 2023

SOURCES = (
    'Eurostat, lfsa_urgaed (unemployment rates by educational attainment '
    'level), geo EL, sex T, isced11 TOTAL, unit PC, ages Y25-64 and Y15-74; '
    'dissemination API, raw JSON in data/eurostat_raw/',
    'European Commission, European Economic Forecast, Spring 2026, Greece: '
    'table "Indicators 2025 2026 2027", row "Unemployment (%)" 8.9 8.3 7.9 '
    '(ages 15-74); https://economy-finance.ec.europa.eu/economic-surveillance'
    '-eu-economies/greece/economic-forecast-greece_en, read 2026-10-07',
    'European Commission, 2024 Ageing Report, Economic and Budgetary '
    'Projections for the EU Member States (2022-2070), Institutional Paper '
    '279, Table II.1.50 Unemployment rate (20-64y), row EL, p. 197',
)


def fetch(age, refresh=False):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f'lfsa_urgaed_EL_T_{age}_TOTAL.json')
    if refresh or not os.path.exists(path):
        url = (f'{API}/lfsa_urgaed?format=JSON&lang=en&geo=EL&sex=T'
               f'&age={age}&isced11=TOTAL&unit=PC')
        print(f'  GET lfsa_urgaed age={age}')
        with urllib.request.urlopen(url, timeout=180) as r:
            data = r.read()
        with open(path, 'wb') as fh:
            fh.write(data)
    return json.load(open(path))


def as_year_series(doc):
    """JSON-stat document holding one time series -> {year: value}."""
    ids, size = doc['id'], doc['size']
    assert all(n == 1 for k, n in zip(ids, size) if k != 'time'), \
        'the document holds more than one series'
    cats = doc['dimension']['time']['category']['index']
    years = sorted(cats, key=cats.get) if isinstance(cats, dict) else list(cats)
    flat = np.full(int(np.prod(size)), np.nan)
    for k, v in doc['value'].items():
        flat[int(k)] = np.nan if v is None else v
    return {int(y): float(v) for y, v in zip(years, flat) if np.isfinite(v)}


def ar_rate(year):
    """Table II.1.50, row EL, linear between the Report's years."""
    return float(np.interp(year, AR_YEARS, AR_RATES))


def ar_rate_base_year():
    """The Report's 2023 rate: linear between its 2022 and 2030 values."""
    y0, y1 = AR_BASE_BRACKET
    return float(np.interp(BASE_YEAR, (y0, y1), (ar_rate(y0), ar_rate(y1))))


def rate_path(years, data, forecast_15_74, ratio_15_74, ratio_ar):
    """The 25-64 unemployment rate in percent for each calendar year.

    data            {year: 25-64 rate}, the base year and the outturns after it
    forecast_15_74  {year: 15-74 rate} for the years after the last outturn
    ratio_15_74     25-64 rate over 15-74 rate in the base year
    ratio_ar        25-64 rate over the Report's 20-64 rate in the base year
    Linear between the knots; the base-year value before the base year; the
    last knot's value after it.
    """
    assert max(data) < min(forecast_15_74) < min(AR_ANCHOR_YEARS)
    knots = dict(data)
    knots.update({y: v * ratio_15_74 for y, v in forecast_15_74.items()})
    knots.update({y: ar_rate(y) * ratio_ar for y in AR_ANCHOR_YEARS})
    ky = np.array(sorted(knots), dtype=float)
    kv = np.array([knots[int(y)] for y in ky])
    years = np.asarray(years, dtype=float)
    return np.where(years <= BASE_YEAR, data[BASE_YEAR], np.interp(years, ky, kv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true',
                    help='re-download even if the raw JSON is cached')
    args = ap.parse_args()

    print('Unemployment rate of 25-64 year olds, Greece')
    raw25, raw15 = fetch('Y25-64', args.refresh), fetch('Y15-74', args.refresh)
    u25, u15 = as_year_series(raw25), as_year_series(raw15)
    print(f'  lfsa_urgaed updated {raw25["updated"]}, '
          f'last year {max(u25)} (25-64), {max(u15)} (15-74)')

    last_data = max(y for y in DATA_YEARS if y in u25)
    data = {y: u25[y] for y in range(BASE_YEAR, last_data + 1)}
    forecast = {y: v for y, v in EC_RATES_15_74.items() if y > last_data}
    for y in DATA_YEARS:
        if y > last_data:
            print(f'  Eurostat has no {y} rate yet; the Commission\'s {y} '
                  f'rate for ages 15-74, {EC_RATES_15_74[y]}, is converted '
                  'and used instead')

    base = data[BASE_YEAR]
    ratio_15_74 = base / u15[BASE_YEAR]
    ratio_ar = base / ar_rate_base_year()
    print(f'  2023: 25-64 {base:.1f}, 15-74 {u15[BASE_YEAR]:.1f}, '
          f'Ageing Report 20-64 {ar_rate_base_year():.4f}')
    print(f'  ratios: 25-64 / 15-74 {ratio_15_74:.6f}, '
          f'25-64 / 20-64 (Report) {ratio_ar:.6f}')

    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    rate = rate_path(years, data, forecast, ratio_15_74, ratio_ar)
    index = rate / base

    np.savez(OUT,
             years=years.astype(int),
             index=index.astype(float),
             rate_25_64=rate.astype(float),
             base_year=BASE_YEAR,
             data_years=np.array(sorted(data), dtype=int),
             data_rates=np.array([data[y] for y in sorted(data)]),
             rate_15_74_2023=u15[BASE_YEAR],
             forecast_years=np.array(sorted(forecast), dtype=int),
             forecast_rates=np.array([forecast[y] for y in sorted(forecast)]),
             ratio_15_74=ratio_15_74,
             ar_years=np.array(AR_YEARS, dtype=int),
             ar_rates=np.array(AR_RATES),
             ar_anchor_years=np.array(AR_ANCHOR_YEARS, dtype=int),
             ar_rate_2023=ar_rate_base_year(),
             ratio_ar=ratio_ar,
             sources=np.array(SOURCES))

    at = {int(y): i for i, y in enumerate(years)}
    print('\n  year   index   rate 25-64 (%)')
    for y in (*range(2023, 2031), 2040, 2050, 2055, 2060):
        print(f'  {y}   {index[at[y]]:.4f}   {rate[at[y]]:.3f}')
    print(f'\nwrote {os.path.relpath(OUT)}: years {years[0]}..{years[-1]}')


if __name__ == '__main__':
    main()
