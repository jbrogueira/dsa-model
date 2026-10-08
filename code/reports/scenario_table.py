"""Debt and the primary balance by scenario against the Commission's
projection -> scenarios_body.tex.

Scenarios are baseline_paths.npz files written by fill_report.py --run-baseline:
  --baseline   the report's baseline (the output-tax rule of the configuration)
  --pinned     the 2023 rate held over the whole horizon, no terminal adjustment
  --constant-u the baseline's rule with the unemployment rate at its 2023 level
Any file that is missing is shown as dashes.

Usage (from code/): python3 reports/scenario_table.py [--outdir ../output/calibration_growth]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import baseline_closure as bc  # noqa: E402

YEARS = [2025, 2030, 2040, 2050, 2060, 2070]


def load_run(path, raw, dsa):
    if path is None or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    P = {k: d[k] for k in d.files}
    out = bc.debt_from_run(P, raw, dsa=dsa)
    by = int(P['base_year'])
    return {'debt': out['debt'], 'pb': out['primary_balance'], 'tau': np.asarray(P['tau_y_path'], float),
            'proj_debt': out['debt_projection'], 'proj_pb': out['primary_balance_projection'], 'base_year': by}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, '..', 'calibration_input_GR.json'))
    ap.add_argument('--baseline', default=os.path.join(HERE, '..', 'output', 'calibration_growth', 'baseline_paths.npz'))
    ap.add_argument('--pinned', default=os.path.join(HERE, '..', 'output', 'calibration_growth_tau_pinned', 'baseline_paths.npz'))
    ap.add_argument('--constant-u', default=os.path.join(HERE, '..', 'output', 'calibration_growth_constant_u', 'baseline_paths.npz'))
    ap.add_argument('--outdir', default=os.path.join(HERE, '..', 'output', 'calibration_growth'))
    args = ap.parse_args()
    raw = json.load(open(args.config))
    dsa = bc.load_dsa_projection(raw)
    runs = {'baseline': load_run(args.baseline, raw, dsa), 'pinned': load_run(args.pinned, raw, dsa),
            'constant_u': load_run(args.constant_u, raw, dsa)}
    ref = next(r for r in runs.values() if r is not None)
    by = ref['base_year']

    def cell(arr, y, f='%.1f', scale=100.0):
        if arr is None:
            return '{--}'
        t = y - by
        if t < 0 or t >= len(arr) or not np.isfinite(arr[t]):
            return '{--}'
        return f % (scale * arr[t])
    rows = []
    rows.append('\\multicolumn{%d}{@{}l}{\\emph{Debt, end of year over output, \\%%}}\\\\' % (len(YEARS) + 1))
    rows.append('DSM forecast (ESM) & ' + ' & '.join(cell(ref['proj_debt'], y) for y in YEARS) + ' \\\\')
    rows.append('Baseline: rate set to the 2060 projection & ' + ' & '.join(cell(runs['baseline'] and runs['baseline']['debt'], y) for y in YEARS) + ' \\\\')
    rows.append('Rate at its 2023 value throughout & ' + ' & '.join(cell(runs['pinned'] and runs['pinned']['debt'], y) for y in YEARS) + ' \\\\')
    rows.append('Baseline rule, unemployment at its 2023 rate & ' + ' & '.join(cell(runs['constant_u'] and runs['constant_u']['debt'], y) for y in YEARS) + ' \\\\')
    rows.append('\\addlinespace\\multicolumn{%d}{@{}l}{\\emph{Primary balance, \\%% of output}}\\\\' % (len(YEARS) + 1))
    rows.append('DSM forecast (ESM) & ' + ' & '.join(cell(ref['proj_pb'], y) for y in YEARS) + ' \\\\')
    rows.append('Baseline & ' + ' & '.join(cell(runs['baseline'] and runs['baseline']['pb'], y) for y in YEARS) + ' \\\\')
    rows.append('Rate at its 2023 value throughout & ' + ' & '.join(cell(runs['pinned'] and runs['pinned']['pb'], y) for y in YEARS) + ' \\\\')
    rows.append('Baseline rule, unemployment at its 2023 rate & ' + ' & '.join(cell(runs['constant_u'] and runs['constant_u']['pb'], y) for y in YEARS) + ' \\\\')
    rows.append('\\addlinespace\\multicolumn{%d}{@{}l}{\\emph{Output tax, \\%% of output}}\\\\' % (len(YEARS) + 1))
    rows.append('Baseline & ' + ' & '.join(cell(runs['baseline'] and runs['baseline']['tau'], y, '%.2f') for y in YEARS) + ' \\\\')
    rows.append('Rate at its 2023 value throughout & ' + ' & '.join(cell(runs['pinned'] and runs['pinned']['tau'], y, '%.2f') for y in YEARS) + ' \\\\')
    rows.append('Baseline rule, unemployment at its 2023 rate & ' + ' & '.join(cell(runs['constant_u'] and runs['constant_u']['tau'], y, '%.2f') for y in YEARS) + ' \\\\')
    head = ('\\begin{tabular}{@{}l' + 'S[table-format=3.2]' * len(YEARS) + '@{}}\n\\toprule\n & '
            + ' & '.join('{%d}' % y for y in YEARS) + ' \\\\\n\\midrule\n')
    body = head + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, 'scenarios_body.tex')
    open(out, 'w').write(body)
    print('wrote', os.path.relpath(out))
    for k, r in runs.items():
        if r is not None:
            print(k, 'debt', ' '.join('%d:%.1f' % (y, 100 * r['debt'][y - by]) for y in YEARS), '| tau 2030 %.4f 2070 %.4f' % (r['tau'][2030 - by], r['tau'][2070 - by]))


if __name__ == '__main__':
    main()
