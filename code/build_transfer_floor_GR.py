"""
Derive the guaranteed income level of the Greek minimum income scheme ->
data/transfer_floor_GR.json.

The level is the model's minimum_income (y_min): an unemployed household of
working age or a retiree whose non-capital income net of out-of-pocket medical
spending falls short of it receives the difference (docs/EGM_PLAN.md section
2). Until 2026-10-09 the same level was the means-tested consumption floor
transfer_floor, which tops up resources including assets. In the model's
units -- detrended output per living person aged 25-99, normalised to 1 in the
base year -- it is

    minimum_income = annual guaranteed minimum income / (nominal GDP / population 25-99),

the same denominator as the pension floor (build_pension_floor_GR.py), whose
cached nominal GDP this script reuses.

Amount: the Guaranteed Minimum Income (Elachisto Eggyimeno Eisodima, formerly
KEA, L.4389/2016 art. 235 and later decisions): EUR 200 per month for a
single-adult household, plus EUR 100 per additional adult and EUR 50 per
child. The single-adult amount is the one that maps onto a model household.
The amount is from the Ministry of Labour / OPEKA programme pages and should
be checked against the vintage in force in the base year.

Spending in the data: Eurostat ESSPROS books the scheme under the social
exclusion function, means-tested periodic cash benefits for income support
(spr_exp_fex, spdep CASH_P_INC, spdepm MT, % of GDP); the unemployment
function's means-tested cash benefits (spr_exp_fun, spdep CASH, spdepm MT)
are recorded beside it. Responses are cached in data/eurostat_raw.

Usage (from code/):  python3 build_transfer_floor_GR.py
"""
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
GDP_CACHE = os.path.join(DATA, 'nomgdp_GR.json')
OUT = os.path.join(DATA, 'transfer_floor_GR.json')
RAW = os.path.join(DATA, 'eurostat_raw')
API = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/'
ESSPROS = {
    'social_exclusion_income_support_mt': ('spr_exp_fex', 'spdep=CASH_P_INC&spdepm=MT'),
    'unemployment_cash_mt': ('spr_exp_fun', 'spdep=CASH&spdepm=MT'),
}

AMOUNTS = {
    'gmi_single_adult': (200.0, 'Guaranteed Minimum Income, single-adult household, per month'),
    'gmi_two_adults': (300.0, 'two adults, no children'),
}
CHOSEN = 'gmi_single_adult'


def nominal_gdp(year):
    if not os.path.exists(GDP_CACHE):
        raise SystemExit('run build_pension_floor_GR.py first (it caches nominal GDP)')
    d = json.load(open(GDP_CACHE))
    lab = {v: k for k, v in d['dimension']['time']['category']['index'].items()}
    vals = {int(lab[int(k)]): float(v) for k, v in d['value'].items() if v is not None}
    return vals[year] * 1e6, d.get('updated')


def esspros_series(code, query, refresh=False):
    """{year: % of GDP} for Greece, all schemes, from 2015."""
    os.makedirs(RAW, exist_ok=True)
    tag = query.replace('&', '_').replace('=', '-')
    path = os.path.join(RAW, f'{code}_EL_{tag}.json')
    if refresh or not os.path.exists(path):
        url = (f'{API}{code}?format=JSON&lang=en&geo=EL&unit=PC_GDP&spscheme=TOTAL'
               f'&{query}&sinceTimePeriod=2015')
        with urllib.request.urlopen(url, timeout=120) as r:
            open(path, 'wb').write(r.read())
    d = json.load(open(path))
    years = {v: k for k, v in d['dimension']['time']['category']['index'].items()}
    return ({int(years[int(k)]): float(v) for k, v in d['value'].items() if v is not None},
            d.get('updated'))


def main():
    demog = np.load(os.path.join(DATA, 'demography_GR.npz'))
    base_year = int(demog['base_year'])
    pop = float(np.asarray(demog['cross_section_base'], dtype=float).sum())
    gdp, vintage = nominal_gdp(base_year)
    per_person = gdp / pop
    rows = {k: {'monthly_eur': m, 'annual_eur': 12 * m, 'transfer_floor': 12 * m / per_person, 'note': n}
            for k, (m, n) in AMOUNTS.items()}
    esspros = {}
    for name, (code, query) in ESSPROS.items():
        series, updated = esspros_series(code, query)
        esspros[name] = {'dataset': code, 'filter': query, 'unit': '% of GDP',
                         'updated': updated, 'values': {str(y): v for y, v in sorted(series.items())}}
    out = {
        'quantity': 'minimum_income (guaranteed income level of the minimum income benefit); '
                    'the same level was transfer_floor (means-tested consumption floor) until 2026-10-09',
        'units': 'detrended output per living person aged 25-99, base year normalised to 1',
        'base_year': base_year,
        'chosen': CHOSEN,
        'value': rows[CHOSEN]['transfer_floor'],
        'minimum_income': rows[CHOSEN]['transfer_floor'],
        'denominator': {'nominal_gdp_eur': gdp, 'nominal_gdp_vintage': vintage,
                        'population_25_84': pop, 'output_per_living_25_84_eur': per_person},
        'amounts': rows,
        'amount_source': ('Greek Guaranteed Minimum Income (L.4389/2016 art. 235 and implementing '
                          'decisions), Ministry of Labour / OPEKA programme pages; amount to be '
                          'checked against the base-year vintage.'),
        'esspros_spending': esspros,
        'caveats': [
            'The benefit tops up the lump sum plus after-tax UI or pension, net of '
            'out-of-pocket medical spending, of the unemployed of working age and of '
            'retirees; the GMI wealth test and capital-income test are not modelled.',
            'A constant level in detrended units is indexed at g in levels.',
        ],
    }
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'wrote {os.path.relpath(OUT)}')
    print(f'  output per living person EUR {per_person:,.0f}')
    for k, r in rows.items():
        mark = ' <- chosen' if k == CHOSEN else ''
        print(f'  {k:18s} EUR {r["monthly_eur"]:6.0f}/mo  minimum_income = {r["transfer_floor"]:.4f}{mark}')
    for name, e in esspros.items():
        print(f'  ESSPROS {name}: 2023 {e["values"].get("2023")}% of GDP')


if __name__ == '__main__':
    main()
