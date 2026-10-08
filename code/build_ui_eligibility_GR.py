"""
UI eligibility probability for Greece -> data/ui_eligibility_GR.json.

In the model a household that moves from employment into unemployment is
eligible for UI with probability p (external_params.ui_eligibility_prob) and
then receives UI in the first year of the spell; no household receives UI
from the second year on. p is set to the share of the unemployed in the first
year of a spell (duration under 12 months) who report receiving benefits or
assistance in the Labour Force Survey, ages 25-64 (the model's working ages).

Series (Eurostat dissemination API, geo EL, sex T):
  lfsa_ugadra  unemployed by duration and registration/benefit status, unit PC;
               regis_es UNE_BEN (receiving benefits or assistance), durations
               M_LT12 and M_GE12: the receipt share within each duration class.
  lfsa_ugad    unemployed by duration, thousands, by duration band (summed to
               under and over 12 months; NRP and OTH excluded): the weights
               that turn the two receipt shares into the share of all unemployed.

The value for all unemployed, p x (1 - long-term share), is the model's
counterpart of the last row and is reported as an untargeted moment.

Usage (from code/):  python3 build_ui_eligibility_GR.py [--refresh] [--year 2023]
"""
import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RAW = os.path.join(DATA, 'eurostat_raw')
OUT = os.path.join(DATA, 'ui_eligibility_GR.json')
API = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/'
QUERY = '?format=JSON&lang=en&geo=EL&sex=T&sinceTimePeriod=2019'
LT12 = ('M_LT1', 'M1-2', 'M3-5', 'M6-11')
GE12 = ('M12-17', 'M18-23', 'M24-47', 'M_GE48')


def fetch(code, refresh=False):
    path = os.path.join(RAW, f'{code}_EL_T_2019.json')
    if refresh or not os.path.exists(path):
        print(f'  GET {code}')
        with urllib.request.urlopen(API + code + QUERY, timeout=120) as r:
            open(path, 'wb').write(r.read())
    return json.load(open(path))


def cells(d):
    """{(label per dimension, ...): value} from a JSON-stat response."""
    ids, sizes = d['id'], d['size']
    labels = [{v: k for k, v in d['dimension'][dim]['category']['index'].items()} for dim in ids]
    out = {}
    for k, v in d['value'].items():
        if v is None:
            continue
        sub, rem = [], int(k)
        for n in reversed(sizes):
            sub.append(rem % n)
            rem //= n
        sub = sub[::-1]
        out[tuple(labels[i][s] for i, s in enumerate(sub))] = float(v)
    return out, ids


def series(d, **fixed):
    """{year: value} at the fixed labels of the other dimensions."""
    vals, ids = cells(d)
    out = {}
    for key, v in vals.items():
        lab = dict(zip(ids, key))
        if all(lab[k] == fv for k, fv in fixed.items()):
            out[int(lab['time'])] = v
    return dict(sorted(out.items()))


def table(refresh, age):
    ra = fetch('lfsa_ugadra', refresh)
    du = fetch('lfsa_ugad', refresh)
    ben_lt = series(ra, age=age, regis_es='UNE_BEN', duration='M_LT12')
    ben_ge = series(ra, age=age, regis_es='UNE_BEN', duration='M_GE12')
    n_lt = {}
    n_ge = {}
    for band in LT12 + GE12:
        s = series(du, age=age, duration=band)
        target = n_lt if band in LT12 else n_ge
        for y, v in s.items():
            target[y] = target.get(y, 0.0) + v
    rows = {}
    for y in sorted(set(ben_lt) & set(ben_ge) & set(n_lt) & set(n_ge)):
        ltu = n_ge[y] / (n_lt[y] + n_ge[y])
        rows[y] = {
            'receiving_lt12': ben_lt[y] / 100.0,
            'receiving_ge12': ben_ge[y] / 100.0,
            'long_term_share': ltu,
            'receiving_all': ((1 - ltu) * ben_lt[y] + ltu * ben_ge[y]) / 100.0,
        }
    return rows, ra.get('updated')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--year', type=int, default=2023,
                    help='base year of the chosen value (decided 2026-10-08)')
    args = ap.parse_args()

    rows, vintage = table(args.refresh, 'Y25-64')
    rows_1574, _ = table(args.refresh, 'Y15-74')
    if args.year not in rows:
        raise SystemExit(f'no data for {args.year}; have {sorted(rows)}')
    chosen = rows[args.year]
    out = {
        'quantity': 'ui_eligibility_prob (p), probability that a new unemployment spell is eligible for UI',
        'definition': ('share of the unemployed aged 25-64 with a spell under 12 months '
                       'who receive benefits or assistance (LFS, self-reported)'),
        'source': ('Eurostat lfsa_ugadra (unit PC, regis_es UNE_BEN, durations M_LT12 and M_GE12) '
                   'and lfsa_ugad (thousands, duration bands), geo EL, sex T'),
        'vintage': vintage,
        'year': args.year,
        'value': chosen['receiving_lt12'],
        'validation': {
            'ui_recipient_share': chosen['receiving_all'],
            'definition': 'share of all unemployed aged 25-64 receiving benefits or assistance',
        },
        'by_year_25_64': rows,
        'receiving_all_15_74': {y: r['receiving_all'] for y, r in rows_1574.items()},
        'caveats': [
            'Receipt is self-reported in the LFS and covers benefits or assistance of any kind.',
            '2020 is raised by the pandemic measures.',
            'The regular benefit lasts 5 to 12 months by days worked (MITOS); the model pays '
            'a full year to an eligible household.',
            'Spending counterpart (the UI/Y target): ESSPROS spr_exp_fun, unemployment function, '
            'periodic cash benefits for full unemployment, 0.56% of GDP in 2023.',
        ],
    }
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'wrote {os.path.relpath(OUT)}  (vintage {vintage})')
    print('  year  <12m  >=12m  LTU share  all (25-64)  all (15-74)')
    for y, r in rows.items():
        mark = ' <- chosen' if y == args.year else ''
        print(f'  {y}  {r["receiving_lt12"]:.3f}  {r["receiving_ge12"]:.3f}  {r["long_term_share"]:.3f}'
              f'      {r["receiving_all"]:.3f}        {rows_1574.get(y, {}).get("receiving_all", float("nan")):.3f}{mark}')


if __name__ == '__main__':
    main()
