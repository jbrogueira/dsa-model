"""
Derive the means-tested consumption floor for Greece -> data/transfer_floor_GR.json.

The floor is the model's transfer_floor: a household whose resources fall
short of it receives the difference as a government transfer. In the model's
units -- detrended output per living person aged 25-84, normalised to 1 in the
base year -- it is

    transfer_floor = annual guaranteed minimum income / (nominal GDP / population 25-84),

the same denominator as the pension floor (build_pension_floor_GR.py), whose
cached nominal GDP this script reuses.

Amount: the Guaranteed Minimum Income (Elachisto Eggyimeno Eisodima, formerly
KEA, L.4389/2016 art. 235 and later decisions): EUR 200 per month for a
single-adult household, plus EUR 100 per additional adult and EUR 50 per
child. The single-adult amount is the one that maps onto a model household.
The amount is from the Ministry of Labour / OPEKA programme pages and should
be checked against the vintage in force in the base year.

Usage (from code/):  python3 build_transfer_floor_GR.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
GDP_CACHE = os.path.join(DATA, 'nomgdp_GR.json')
OUT = os.path.join(DATA, 'transfer_floor_GR.json')

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


def main():
    demog = np.load(os.path.join(DATA, 'demography_GR.npz'))
    base_year = int(demog['base_year'])
    pop = float(np.asarray(demog['cross_section_base'], dtype=float).sum())
    gdp, vintage = nominal_gdp(base_year)
    per_person = gdp / pop
    rows = {k: {'monthly_eur': m, 'annual_eur': 12 * m, 'transfer_floor': 12 * m / per_person, 'note': n}
            for k, (m, n) in AMOUNTS.items()}
    out = {
        'quantity': 'transfer_floor (means-tested consumption floor)',
        'units': 'detrended output per living person aged 25-84, base year normalised to 1',
        'base_year': base_year,
        'chosen': CHOSEN,
        'value': rows[CHOSEN]['transfer_floor'],
        'denominator': {'nominal_gdp_eur': gdp, 'nominal_gdp_vintage': vintage,
                        'population_25_84': pop, 'output_per_living_25_84_eur': per_person},
        'amounts': rows,
        'amount_source': ('Greek Guaranteed Minimum Income (L.4389/2016 art. 235 and implementing '
                          'decisions), Ministry of Labour / OPEKA programme pages; amount to be '
                          'checked against the base-year vintage.'),
        'caveats': [
            'The floor applies to resources after taxes and out-of-pocket medical '
            'spending, before the hours adjustment, as compute_budget evaluates it.',
            'A constant floor in detrended units is indexed at g in levels.',
        ],
    }
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'wrote {os.path.relpath(OUT)}')
    print(f'  output per living person EUR {per_person:,.0f}')
    for k, r in rows.items():
        mark = ' <- chosen' if k == CHOSEN else ''
        print(f'  {k:18s} EUR {r["monthly_eur"]:6.0f}/mo  transfer_floor = {r["transfer_floor"]:.4f}{mark}')


if __name__ == '__main__':
    main()
