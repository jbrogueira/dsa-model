"""
Build the path of the sovereign rate for Greece -> data/r_B_path_GR.npz.

The model's r_B is the real effective interest rate on general government
debt, by calendar year:

  i_t = (interest payments / GDP)_t / (debt / GDP)_{t-1}
  r_t = (1 + i_t) / (1 + pi_t) - 1,

with pi the growth rate of the GDP deflator, nominal GDP over chain-linked
real GDP (Eurostat nama_10_gdp, data/nomgdp_GR.json and data/realgdp_GR.json).

  2023-2024  the data: interest payments and debt from data/DATA_GR.xlsx,
             sheet DATA, codes 40 and 49.
  2025       the data sheet has no interest payments for 2025, and the
             Spring 2026 documents in lit-review/data-debt (country report,
             post-programme surveillance report) give the 2025 headline
             balance but no interest expenditure. The nominal rate is the
             2026 nominal effective rate of the debt projection (1.77%),
             deflated by the 2025 deflator growth (2.78%).
  2026-2060  the real effective rate of the debt projection,
             data/dsa_projection_GR.npz (build_dsa_projection_GR.py).
  2061-2070  linear from the 2060 value to 2%.
  2071-2400  2%.
  1900-2022  the 2023 value.

Usage (from code/):  python3 build_r_B_path_GR.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
XLSX = os.path.join(DATA, 'DATA_GR.xlsx')
NOMGDP = os.path.join(DATA, 'nomgdp_GR.json')
REALGDP = os.path.join(DATA, 'realgdp_GR.json')
DSA = os.path.join(DATA, 'dsa_projection_GR.npz')
OUT = os.path.join(DATA, 'r_B_path_GR.npz')

CODE_INTEREST, CODE_DEBT = 40, 49          # DATA_GR.xlsx, sheet DATA, row 1
DATA_YEARS = (2023, 2024)
PROXY_YEAR = 2025
PROXY_SOURCE_YEAR = 2026                   # nominal effective rate taken from this year of the projection
TERMINAL_RATE = 0.02
TERMINAL_YEAR = 2070
FIRST_YEAR, LAST_YEAR = 1900, 2400
PRINT_YEARS = (2023, 2024, 2025, 2026, 2030, 2040, 2050, 2060, 2065, 2070, 2100)


def read_data_sheet():
    """{year: (interest payments / GDP, debt / GDP)} from sheet DATA."""
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, data_only=True)['DATA']
    cols = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)
            if ws.cell(1, c).value is not None}
    out = {}
    for r in range(5, ws.max_row + 1):
        y = ws.cell(r, 1).value
        if y is None:
            continue
        out[int(y)] = (ws.cell(r, cols[CODE_INTEREST]).value, ws.cell(r, cols[CODE_DEBT]).value)
    return out


def read_jsonstat_series(path):
    """{year: value} from a JSON-stat file whose only varying dimension is time."""
    doc = json.load(open(path))
    sizes = [n for n in doc['size'] if n > 1]
    if len(sizes) != 1:
        raise ValueError(f'{path}: expected time as the only varying dimension, got sizes {doc["size"]}')
    index = doc['dimension']['time']['category']['index']
    return {int(y): doc['value'][str(i)] for y, i in index.items() if str(i) in doc['value']}


def deflator_growth(years):
    nom, real = read_jsonstat_series(NOMGDP), read_jsonstat_series(REALGDP)
    level = {y: nom[y] / real[y] for y in nom if y in real}
    return np.array([level[y] / level[y - 1] - 1.0 for y in years])


def main():
    sheet = read_data_sheet()
    dsa = np.load(DSA)
    dsa_years, rr = dsa['years'], dsa['real_effective_rate']
    nominal_proxy = float(dsa['nominal_effective_rate'][dsa_years == PROXY_SOURCE_YEAR][0])

    years_in = np.array(DATA_YEARS + (PROXY_YEAR,))
    pi = deflator_growth(years_in)
    pi_dsa = float(dsa['deflator_growth'][dsa_years == PROXY_YEAR][0])
    if abs(pi[-1] - pi_dsa) > 1e-3:
        raise ValueError(f'{PROXY_YEAR} deflator growth: data {pi[-1]:.4f} vs projection {pi_dsa:.4f}')
    i_nom = np.array([sheet[y][0] / sheet[y - 1][1] for y in DATA_YEARS] + [nominal_proxy])
    r_data = (1.0 + i_nom) / (1.0 + pi) - 1.0

    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    r = np.full(years.shape, np.nan)
    r[years <= DATA_YEARS[0]] = r_data[0]
    for y, v in zip(years_in[1:], r_data[1:]):
        r[years == y] = v
    for y, v in zip(dsa_years, rr):
        if y > PROXY_YEAR:
            r[years == y] = v
    last_proj = int(dsa_years[-1])
    v_last = float(rr[dsa_years == last_proj][0])
    ramp = (years > last_proj) & (years < TERMINAL_YEAR)
    r[ramp] = v_last + (TERMINAL_RATE - v_last) * (years[ramp] - last_proj) / (TERMINAL_YEAR - last_proj)
    r[years >= TERMINAL_YEAR] = TERMINAL_RATE
    if np.isnan(r).any():
        raise ValueError('r_B path has gaps')

    notes = (f'{PROXY_YEAR}: no interest payments in the data sheet and no interest expenditure line '
             f'for {PROXY_YEAR} in lit-review/data-debt/country_report_EL_2026.txt or '
             f'pps_assessments_spring2026.txt (Greece chapter); nominal rate = the projection\'s '
             f'{PROXY_SOURCE_YEAR} nominal effective rate {100 * nominal_proxy:.3f}%, deflator growth '
             f'{100 * pi[-1]:.3f}% from the GDP data. {DATA_YEARS[0]}-{DATA_YEARS[-1]}: DATA_GR.xlsx '
             f'codes {CODE_INTEREST} and {CODE_DEBT}, deflator from nomgdp_GR.json / realgdp_GR.json. '
             f'{PROXY_YEAR + 1}-{last_proj}: dsa_projection_GR.npz real_effective_rate. '
             f'{last_proj + 1}-{TERMINAL_YEAR}: linear to {TERMINAL_RATE}. After: {TERMINAL_RATE}. '
             f'Before {DATA_YEARS[0]}: the {DATA_YEARS[0]} value.')
    np.savez(OUT, years=years, r_B=r,
             data_years=years_in, implicit_nominal=i_nom, deflator_growth=pi, data_real=r_data,
             proxy_year=PROXY_YEAR, nominal_proxy=nominal_proxy, notes=notes)

    for y, i, p, v in zip(years_in, i_nom, pi, r_data):
        tag = ' (nominal rate = projection %d)' % PROXY_SOURCE_YEAR if y == PROXY_YEAR else ''
        print(f'  {y}: nominal {100 * i:.3f}%  deflator growth {100 * p:.3f}%  real {100 * v:.3f}%{tag}')
    for y in PRINT_YEARS:
        print(f'  r_B {y}: {100 * float(r[years == y][0]):.3f}%')
    m = (years >= PROXY_YEAR + 1) & (years <= last_proj)
    print(f'  mean {PROXY_YEAR + 1}-{last_proj}: {100 * r[m].mean():.3f}%')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
