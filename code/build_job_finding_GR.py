"""
Derive the annual job-finding probability f for Greece -> data/job_finding_GR.json.

The income process moves a household out of unemployment with probability f
per year. With a constant monthly hazard the share of the unemployed stock
whose spell has lasted twelve months or more equals the probability of not
leaving within a year, so

    f = 1 - (long-term unemployed / unemployed),

the long-term share being Eurostat's une_ltu_a in percent of unemployment
(duration of 12 months or more, ages 15-74, both sexes). The separation rates
follow from f and the education-specific unemployment rates inside the model
(lifecycle_perfect_foresight._income_process), so f is the one free constant.

Until 2026-10-02 the configuration carried f = 0.5 with no recorded source.

Usage (from code/):  python3 build_job_finding_GR.py [--refresh] [--window 2015 2024]
"""
import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CACHE = os.path.join(DATA, 'une_ltu_GR.json')
OUT = os.path.join(DATA, 'job_finding_GR.json')
URL = ('https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/'
       'data/une_ltu_a?format=JSON&lang=en&geo=EL&unit=PC_UNE&sex=T&age=Y15-74')


def ltu_share(refresh=False):
    if refresh or not os.path.exists(CACHE):
        print('  GET une_ltu_a PC_UNE sex=T age=Y15-74 geo=EL')
        with urllib.request.urlopen(URL, timeout=120) as r:
            open(CACHE, 'wb').write(r.read())
    d = json.load(open(CACHE))
    # JSON-stat: values are keyed by a flat row-major index over every
    # dimension in d['id'] with sizes d['size']. The response carries two
    # duration indicators (LTU: 12 months or more; VLTU: 24 or more), so the
    # flat index is unravelled and only LTU is kept.
    ids, sizes = d['id'], d['size']
    pos = {dim: i for i, dim in enumerate(ids)}
    time_lab = {v: k for k, v in d['dimension']['time']['category']['index'].items()}
    ltu_idx = d['dimension']['indic_em']['category']['index']['LTU']
    vals = {}
    for k, v in d['value'].items():
        if v is None:
            continue
        sub = []
        rem = int(k)
        for n in reversed(sizes):
            sub.append(rem % n)
            rem //= n
        sub = sub[::-1]
        if sub[pos['indic_em']] != ltu_idx:
            continue
        vals[int(time_lab[sub[pos['time']]])] = float(v) / 100.0
    return dict(sorted(vals.items())), d.get('updated')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--window', nargs=2, type=int, default=(2023, 2023),
                    metavar=('FIRST', 'LAST'),
                    help='years averaged for the chosen value (default: the base year 2023, '
                         'consistent with the other base-year targets; decided 2026-10-02)')
    args = ap.parse_args()

    share, vintage = ltu_share(args.refresh)
    first, last = args.window
    win = {y: s for y, s in share.items() if first <= y <= last}
    if not win:
        raise SystemExit(f'no observations in {first}-{last}; have {min(share)}-{max(share)}')
    mean_share = sum(win.values()) / len(win)
    by_year = {y: {'ltu_share': s, 'f': 1.0 - s} for y, s in share.items()}
    out = {
        'quantity': 'job_finding_rate (f), annual probability of leaving unemployment',
        'definition': 'f = 1 - share of the unemployed with a spell of 12 months or more',
        'source': 'Eurostat une_ltu_a, unit PC_UNE, sex T, age Y15-74, geo EL',
        'vintage': vintage,
        'window': [first, last],
        'value': 1.0 - mean_share,
        'ltu_share_window_mean': mean_share,
        'by_year': by_year,
        'caveats': [
            'Assumes a constant exit hazard within the year, so the long-term share '
            'of the stock equals one minus the annual exit probability.',
            'The share is for ages 15-74; the model covers 25-84 and has no exit '
            'to inactivity, so every exit is a job finding.',
            'The window is a choice: the share fell from about 0.73 (2014) to '
            'about 0.5 (2023-24), so a base-year value and a crisis-era average differ.',
        ],
    }
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'wrote {os.path.relpath(OUT)}')
    print(f'  years available {min(share)}-{max(share)}')
    for y, s in share.items():
        mark = ' <- window' if first <= y <= last else ''
        print(f'  {y}: long-term share {s:.3f}  ->  f = {1 - s:.3f}{mark}')
    print(f'  window {first}-{last}: mean share {mean_share:.3f}  ->  f = {1 - mean_share:.3f}  (chosen)')


if __name__ == '__main__':
    main()
