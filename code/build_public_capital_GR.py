"""
Bring the public capital stock to the model's base year → prints K_g/Y for 2023.

The config's K_g/Y = 0.745 is the IMF Investment and Capital Stock Dataset
value for **2019**, the last year ICSD covers, used as if it were the base-year
ratio. This rolls the stock forward to 2023 by perpetual inventory,

    K_g[t+1] = (1 - delta_g) * K_g[t] + I_g[t],

on real levels, using the national accounts' public investment share for
2020-2023 (DATA_GR.xlsx sheet DATA, code 45) and Greek real GDP in chain-linked
2015 prices (Eurostat nama_10_gdp, B1GQ, CLV15_MEUR).

delta_g is not assumed: it is estimated from the ICSD stock itself over
--delta-window, so the roll-forward uses the depreciation implicit in the
series it extends. Reported with the spread across years so the reader can see
how well-determined it is.

Usage (from code/):  python3 build_public_capital_GR.py [--base-year 2023]
"""
import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
GDP_CACHE = os.path.join(DATA, 'realgdp_GR.json')
GDP_URL = ('https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/'
           'data/nama_10_gdp?format=JSON&lang=en&geo=EL&na_item=B1GQ'
           '&unit=CLV15_MEUR')


def real_gdp(refresh=False):
    """Greek real GDP by year, chain-linked 2015 prices (EUR million)."""
    if refresh or not os.path.exists(GDP_CACHE):
        print('  GET nama_10_gdp B1GQ CLV15_MEUR geo=EL')
        with urllib.request.urlopen(GDP_URL, timeout=120) as r:
            open(GDP_CACHE, 'wb').write(r.read())
    d = json.load(open(GDP_CACHE))
    lab = {v: k for k, v in d['dimension']['time']['category']['index'].items()}
    return {int(lab[int(k)]): float(v) for k, v in d['value'].items()
            if v is not None}


def icsd_ratios():
    """K_g/Y and I_g/Y from the IMF ICSD extract, as fractions."""
    d = pd.read_csv(os.path.join(DATA, 'IMF_ICSD_GR.csv'))
    def series(ind):
        s = d[d.indicator == ind][['year', 'value']].dropna()
        return {int(y): v / 100.0 for y, v in zip(s.year, s.value)}
    return series('CAPSTCK_S13_Q_POGDP_PT'), series('P51G_S13_Q_POGDP_PT')


def na_public_investment():
    """Public investment / Y from DATA_GR.xlsx, sheet DATA, code 45."""
    df = pd.read_excel(os.path.join(DATA, 'DATA_GR.xlsx'), sheet_name='DATA',
                       header=None)
    col = next(j for j in range(df.shape[1])
               if str(df.iloc[0, j]).replace('.0', '') == '45')
    out = {}
    for i in range(3, len(df)):
        y = str(df.iloc[i, 0]).replace('.0', '')
        if y.isdigit() and pd.notna(df.iloc[i, col]):
            out[int(y)] = float(df.iloc[i, col])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-year', type=int, default=2023)
    ap.add_argument('--delta-window', type=int, nargs=2, default=(2000, 2019))
    ap.add_argument('--refresh', action='store_true')
    args = ap.parse_args()

    Y = real_gdp(args.refresh)
    kg_r, ig_r = icsd_ratios()
    ig_na = na_public_investment()
    last = max(kg_r)
    print(f'ICSD covers K_g/Y through {last}: {kg_r[last]:.4f}')

    # Real levels, one unit throughout (EUR mn, 2015 prices).
    K = {y: kg_r[y] * Y[y] for y in kg_r if y in Y}
    I_icsd = {y: ig_r[y] * Y[y] for y in ig_r if y in Y}

    lo, hi = args.delta_window
    deltas = [(K[y] + I_icsd[y] - K[y + 1]) / K[y]
              for y in range(lo, hi) if y in K and y + 1 in K and y in I_icsd]
    delta_g = float(np.mean(deltas))
    print(f'delta_g implied by the ICSD stock over {lo}-{hi}: '
          f'{delta_g:.5f}  (sd {np.std(deltas):.5f}, '
          f'range {min(deltas):.5f}-{max(deltas):.5f}, n={len(deltas)})')

    # Roll forward on national-accounts public investment.
    k = K[last]
    print(f'\n{"year":6s} {"I_g/Y":>8s} {"K_g real":>12s} {"K_g/Y":>8s}')
    print(f'{last:<6d} {ig_r[last]:8.5f} {k:12.0f} {k / Y[last]:8.4f}   (ICSD)')
    for y in range(last, args.base_year):
        i_g = ig_na[y] * Y[y]
        k = (1.0 - delta_g) * k + i_g
        print(f'{y + 1:<6d} {ig_na[y]:8.5f} {k:12.0f} {k / Y[y + 1]:8.4f}')

    kg_base = k / Y[args.base_year]
    print(f'\nK_g/Y in {args.base_year}: {kg_base:.4f}   '
          f'(config holds {0.745:.4f}, the {last} value)')
    print(f'I_g/Y in {args.base_year}: {ig_na[args.base_year]:.5f}   '
          f'(config holds 0.03530)')
    print('\nK_g enters the firm FOC through K_over_L, so changing it moves w '
          'and\nrequires re-running the SMM, normalize_A_tfp and '
          'pin_baseline_closure.')


if __name__ == '__main__':
    main()
