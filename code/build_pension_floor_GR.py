"""
Derive the minimum pension floor b_min for Greece -> data/pension_floor_GR.json.

b_min is the flat national-pension component of the model's pension,
PENS = max(rho * ybar, b_min), expressed in the model's units: detrended output
per living person aged 25-99, normalised to 1 in the base year.

    b_min = annual statutory amount / (nominal GDP / population aged 25-99)

The denominator is the part that is easy to get wrong. The model contains no one
outside 25-99, so its output is per living person in that band -- not per capita
of the whole population, which would understate the denominator by about 30% and
overstate b_min by the same.

Sources. Eurostat publishes aggregate pension expenditure, not statutory benefit
amounts, so the amount comes from Greek law and the Commission's country fiche;
Eurostat supplies the denominator.

  amount      national pension, L.4387/2016: EUR 384/month base, indexed to
              EUR 413.76 from January 2023 at >= 20 years of contributions;
              EUR 387.90 at 15 years. 2024 Ageing Report country fiche for
              Greece (DG ECFIN) and the Ministry of Labour's pension pages.
  GDP         Eurostat nama_10_gdp, B1GQ, CP_MEUR, Greece -- cached by this
              script at data/nomgdp_GR.json.
  population  data/demography_GR.npz, cross_section_base, summed over model
              ages 0-74 (real ages 25-99), base year 2023.

Usage (from code/):  python3 build_pension_floor_GR.py [--refresh]
"""
import argparse
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
GDP_CACHE = os.path.join(DATA, 'nomgdp_GR.json')
OUT = os.path.join(DATA, 'pension_floor_GR.json')
GDP_URL = ('https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/'
           'data/nama_10_gdp?format=JSON&lang=en&geo=EL&na_item=B1GQ'
           '&unit=CP_MEUR')

# Statutory monthly amounts, EUR. The qualification years matter: the national
# pension is reduced proportionally between 15 and 20 years of contributions.
AMOUNTS = {
    'national_pension_20y_2023': (413.76, 'indexed from January 2023, >=20 years'),
    'national_pension_15y_2023': (387.90, 'indexed from January 2023, 15 years'),
    'national_pension_base': (384.00, 'L.4387/2016 base rate, before indexation'),
}
CHOSEN = 'national_pension_20y_2023'


def nominal_gdp(year, refresh=False):
    if refresh or not os.path.exists(GDP_CACHE):
        print('  GET nama_10_gdp B1GQ CP_MEUR geo=EL')
        with urllib.request.urlopen(GDP_URL, timeout=120) as r:
            open(GDP_CACHE, 'wb').write(r.read())
    d = json.load(open(GDP_CACHE))
    lab = {v: k for k, v in d['dimension']['time']['category']['index'].items()}
    vals = {int(lab[int(k)]): float(v) for k, v in d['value'].items() if v is not None}
    return vals[year] * 1e6, d.get('updated')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    args = ap.parse_args()

    demog = np.load(os.path.join(DATA, 'demography_GR.npz'))
    base_year = int(demog['base_year'])
    pop = float(np.asarray(demog['cross_section_base'], dtype=float).sum())
    gdp, gdp_updated = nominal_gdp(base_year, args.refresh)
    per_person = gdp / pop

    rows = {}
    for key, (monthly, note) in AMOUNTS.items():
        annual = monthly * 12.0
        rows[key] = {'monthly_eur': monthly, 'annual_eur': annual,
                     'b_min': annual / per_person, 'note': note}

    out = {
        'quantity': 'pension_min_floor (b_min)',
        'units': ('detrended output per living person aged 25-99, '
                  'base year normalised to 1'),
        'base_year': base_year,
        'chosen': CHOSEN,
        'value': rows[CHOSEN]['b_min'],
        'denominator': {
            'nominal_gdp_eur': gdp,
            'nominal_gdp_source': 'Eurostat nama_10_gdp, B1GQ, CP_MEUR, geo=EL',
            'nominal_gdp_vintage': gdp_updated,
            'population_25_84': pop,
            'population_source': 'data/demography_GR.npz, cross_section_base',
            'output_per_living_25_84_eur': per_person,
        },
        'amounts': rows,
        'amount_source': ('Greek L.4387/2016 national pension; 2024 Ageing Report '
                          'country fiche for Greece (DG ECFIN); Ministry of Labour '
                          'pension pages. Eurostat does not publish statutory '
                          'benefit amounts.'),
        'caveats': [
            'The model has no one outside 25-99, so output is per living person in '
            'that band, not per capita.',
            'A constant b_min in detrended units is a floor indexed at g, which is '
            'the right treatment for a statutory minimum uprated with earnings.',
            'The level doubles as a share of output only while A_tfp normalises '
            'base-year output to 1.',
            'b_min is not an SMM parameter, so the SMM absorbs changes to it '
            'through rho_pens.',
        ],
    }
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)

    print(f'wrote {os.path.relpath(OUT)}')
    print(f'  nominal GDP {base_year}      EUR {gdp/1e9:,.1f} bn')
    print(f'  population 25-99        {pop:,.0f}')
    print(f'  output per living person EUR {per_person:,.0f}')
    print()
    for key, r in rows.items():
        mark = ' <- chosen' if key == CHOSEN else ''
        print(f'  {key:30s} EUR {r["monthly_eur"]:7.2f}/mo  '
              f'b_min = {r["b_min"]:.4f}{mark}')


if __name__ == '__main__':
    main()
