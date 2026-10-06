"""
Read the debt projection for Greece into data/dsa_projection_GR.npz.

Source: data/2026-09-28 GR DSA Spring Forecast 2026.xlsx, one projection of
government debt and gross financing needs for 2025-2060 in the layout of the
Commission's Debt Sustainability Monitor, started from the Spring 2026
Economic Forecast. The file gives no units. Its accounting identities hold
with flows and stocks in % of GDP and nominal GDP in EUR bn, and main() checks
them:

  GFN_t  = amortisation_t + interest_t - pb_t + sfa_fin_t
  d_t    = d_{t-1} (1 + i_t) / (1 + gn_t) - pb_t + sfa_fin_t + sfa_debt_t
  i_t    = interest_t Y_t / (d_{t-1} Y_{t-1})
  rr_t   = (1 + i_t) / (1 + pi_t) - 1

with gn nominal growth, pi deflator growth and rr the real effective rate.

The file starts in 2025. The two years between the model's base year and the
file come from the Debt Sustainability Monitor 2025 (Institutional Paper 332,
February 2026, table "Greece - baseline scenario"): debt 154.2% of GDP and a
primary balance of 4.7% in 2024. Debt in 2023 is 164.28% (fiscal.B_over_Y,
DATA_GR.xlsx code 49).

All series are stored as fractions of GDP (rates as fractions), by year.

Usage (from code/):  python3 build_dsa_projection_GR.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
XLSX = os.path.join(DATA, '2026-09-28 GR DSA Spring Forecast 2026.xlsx')
OUT = os.path.join(DATA, 'dsa_projection_GR.npz')

# Row of each series on the sheet (years are in row 2, columns B..AK).
ROWS = {
    'debt': 3, 'amortisation': 4, 'interest': 9, 'gfn': 14, 'nominal_gdp': 18,
    'real_growth': 19, 'deflator_growth': 20, 'primary_balance': 21,
    'spb_before_ageing': 22, 'cyclical': 23, 'one_offs': 24, 'cost_of_ageing': 25,
    'property_income': 26, 'sfa_financing': 28, 'sfa_debt': 29,
    'nominal_effective_rate': 30, 'real_effective_rate': 31,
}
LEVELS = ('nominal_gdp',)                     # EUR bn, not rescaled
# Debt Sustainability Monitor 2025, Greece, baseline: year, debt, primary balance (% of GDP).
MONITOR_2024 = (2024, 154.2, 4.7)


def read_sheet():
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, data_only=True)['Sheet1']

    def row(r):
        return np.array([np.nan if ws.cell(r, c).value is None else float(ws.cell(r, c).value)
                         for c in range(2, 38)])
    years = row(2).astype(int)
    if years[0] != 2025 or years[-1] != 2060:
        raise ValueError(f'unexpected year range {years[0]}-{years[-1]}')
    return years, {k: row(r) for k, r in ROWS.items()}


def check_identities(s):
    """Largest absolute deviation of each identity, in % of GDP or pp."""
    d, Y = s['debt'], s['nominal_gdp']
    gn = Y[1:] / Y[:-1] - 1.0
    sfa = s['sfa_financing'] + s['sfa_debt']
    out = {
        'gfn': np.nanmax(np.abs(s['gfn'] - (s['amortisation'] + s['interest']
                                           - s['primary_balance'] + s['sfa_financing']))),
        'debt': np.max(np.abs(d[1:] - (d[:-1] / (1 + gn) + s['interest'][1:]
                                       - s['primary_balance'][1:] + sfa[1:]))),
        'rate': np.max(np.abs(s['nominal_effective_rate'][1:]
                              - 100 * s['interest'][1:] * Y[1:] / (d[:-1] * Y[:-1]))),
        'real_rate': np.max(np.abs(
            s['real_effective_rate'][1:]
            - 100 * ((1 + s['nominal_effective_rate'][1:] / 100)
                     / (1 + s['deflator_growth'][1:] / 100) - 1))),
        'growth': np.max(np.abs(100 * gn - 100 * ((1 + s['real_growth'][1:] / 100)
                                                  * (1 + s['deflator_growth'][1:] / 100) - 1))),
    }
    return out


def main():
    years, s = read_sheet()
    dev = check_identities(s)
    for k, v in dev.items():
        print(f'  identity {k}: largest deviation {v:.2e}')
        if v > 1e-6:
            raise ValueError(f'identity {k} fails by {v:.3g}')
    out = {k: (v if k in LEVELS else v / 100.0) for k, v in s.items()}
    out['sfa'] = out['sfa_financing'] + out['sfa_debt']
    np.savez(OUT, years=years, **out,
             monitor_year=MONITOR_2024[0], monitor_debt=MONITOR_2024[1] / 100.0,
             monitor_primary_balance=MONITOR_2024[2] / 100.0)
    rr = out['real_effective_rate'][1:]
    print(f'  debt {100 * out["debt"][0]:.1f}% in {years[0]}, {100 * out["debt"][-1]:.1f}% in {years[-1]}')
    print(f'  mean real effective rate 2026-2060: {100 * rr.mean():.3f}%')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
