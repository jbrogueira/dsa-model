"""Public capital and public investment in Greece, 1960-2025 -> public_capital.pdf.

Left panel: general-government capital stock over GDP. 1960-2019 is the IMF
Investment and Capital Stock Dataset (constant prices); 2020-2025 extends that
stock by perpetual inventory exactly as ``build_public_capital_GR.py`` does
(national-accounts public investment, Eurostat real GDP, delta_g estimated
from the ICSD stock over 2000-2019). The dots are the config's 2023 targets, K_g and
I_g_over_Y.

Right panel: general-government gross fixed capital formation over GDP from
the three sources the calibration touches: ICSD (1960-2019), DATA_GR.xlsx
sheet DATA code 45 (the series that drives the roll-forward and sets I_g/Y),
and Eurostat gov_10a_main P51G (cached in data/gov_accounts_GR.json, through
2025).

Nothing is fetched here; all inputs are cached files.

Usage (from code/): python3 reports/public_capital_figure.py [--outdir ../output/calibration_growth]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from build_public_capital_GR import (DATA, icsd_ratios,  # noqa: E402
                                     na_public_investment, real_gdp)
from baseline_figures import INK, INK2, SERIES, _style  # noqa: E402

LAST_YEAR = 2025
BASE_YEAR = 2023
DELTA_WINDOW = (2000, 2019)


def public_capital_path():
    """K_g/Y by year: ICSD through its last year, perpetual inventory after."""
    Y = real_gdp()
    kg_r, ig_r = icsd_ratios()
    ig_na = na_public_investment()
    last = max(kg_r)

    K = {y: kg_r[y] * Y[y] for y in kg_r if y in Y}
    I_icsd = {y: ig_r[y] * Y[y] for y in ig_r if y in Y}
    lo, hi = DELTA_WINDOW
    delta_g = float(np.mean([(K[y] + I_icsd[y] - K[y + 1]) / K[y]
                             for y in range(lo, hi)
                             if y in K and y + 1 in K and y in I_icsd]))

    ext = {last: kg_r[last]}
    k = K[last]
    for y in range(last, LAST_YEAR):
        k = (1.0 - delta_g) * k + ig_na[y] * Y[y]
        ext[y + 1] = k / Y[y + 1]
    return kg_r, ext, delta_g


def eurostat_p51g():
    d = json.load(open(os.path.join(DATA, 'gov_accounts_GR.json')))
    v = d['expenditure']['P51G']['values']
    return {int(y): x / 100.0 for y, x in v.items() if x is not None}


def _xy(d, lo=None, hi=None):
    ys = sorted(y for y in d if (lo is None or y >= lo) and (hi is None or y <= hi))
    return np.array(ys), np.array([d[y] for y in ys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, '..', 'calibration_input_GR.json'))
    ap.add_argument('--outdir', default=os.path.join(HERE, '..', 'output', 'calibration_growth'))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = json.load(open(args.config))
    kg_target = float(cfg['production']['K_g'])
    ig_target = float(cfg['fiscal']['I_g_over_Y'])
    kg_icsd, kg_ext, delta_g = public_capital_path()
    _, ig_icsd = icsd_ratios()
    # Code 45 is blank before 1999 and read as zero; keep positive entries.
    ig_na = {y: v for y, v in na_public_investment().items() if v > 0}
    ig_es = eurostat_p51g()
    print(f'delta_g (ICSD, {DELTA_WINDOW[0]}-{DELTA_WINDOW[1]}): {delta_g:.5f}')
    for y in sorted(kg_ext):
        print(f'  {y}  K_g/Y {kg_ext[y]:.4f}')

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))

    x, v = _xy(kg_icsd)
    ax[0].plot(x, v, color=SERIES[0], lw=1.5, label='IMF ICSD')
    x, v = _xy(kg_ext)
    ax[0].plot(x, v, color=SERIES[0], lw=1.5, ls='--',
               label='Perpetual inventory from the 2019 ICSD stock')
    ax[0].plot([BASE_YEAR], [kg_target], 'o', color=INK, ms=4,
               label=f'Calibration target, {BASE_YEAR} ({kg_target:.3f})')
    _style(ax[0], 'Public capital / GDP')

    for i, (lab, d) in enumerate([('IMF ICSD', ig_icsd),
                                  ('National accounts (DATA_GR, code 45)', ig_na),
                                  ('Eurostat gov_10a_main, P51G', ig_es)]):
        x, v = _xy(d, hi=LAST_YEAR)
        ax[1].plot(x, v, color=SERIES[i], lw=1.5, label=lab)
    ax[1].plot([BASE_YEAR], [ig_target], 'o', color=INK, ms=4,
               label=f'Calibration target, {BASE_YEAR} ({ig_target:.3f})')
    _style(ax[1], 'Public investment / GDP')

    for a in ax:
        a.set_xlim(1960, LAST_YEAR)
        a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False,
                 fontsize=7.5, ncol=1, labelcolor=INK, handlelength=1.8)
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, 'public_capital.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(out)}')


if __name__ == '__main__':
    main()
