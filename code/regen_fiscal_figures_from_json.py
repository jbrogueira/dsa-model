#!/usr/bin/env python
"""Regenerate fiscal-shock figures from a saved fiscal_results.json.

Reads the numerical paths already stored by run_fiscal_figures.py and redraws
the macro-overview, prices-sanity, fiscal-decomposition, and debt-fan figures
WITHOUT re-running any simulation.  Adds the NFA/Y panel to the macro overview.

The stored 'counterfactual' macro NFA is the full NFA (A - K_domestic - B); the
baseline curve uses the standalone baseline run's counterfactual block, which is
also the full NFA.  (The 'baseline' block nested under each shock scenario is the
uncorrected partial NFA and is NOT used here.)

Usage:
    python regen_fiscal_figures_from_json.py \
        --json output/fiscal_test/fiscal_results.json \
        --output-dir output/fiscal_test
"""
import argparse
import json
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fiscal_experiments import compare_scenarios, debt_fan_chart, _nfa_ca_paths

# Same panel definitions as run_fiscal_figures.py (kept in sync manually).
MACRO_VARS = ['Y', 'K_domestic', 'L', 'C', 'B_gdp_path', 'A', 'A_gdp', 'NFA_gdp', 'K_g_Y']
MACRO_LABELS = {
    'Y':          'Output (Y)',
    'K_domestic': 'Domestic capital (K)',
    'L':          'Labour (L)',
    'C':          'Consumption (C)',
    'B_gdp_path': 'Debt / GDP (B/Y)',
    'A':          'Household wealth (A)',
    'A_gdp':      'Household wealth / Y (A/Y)',
    'NFA_gdp':    'Net foreign assets / Y (NFA/Y)',
    'K_g_Y':      'Public capital / Y (K_g/Y)',
}
PRICE_VARS   = ['w', 'r']
PRICE_LABELS = {'w': 'Wage rate (w)', 'r': 'Interest rate (r)'}
FISCAL_VARS = [
    'primary_deficit_gdp',
    'tax_l_gdp', 'tax_c_gdp', 'tax_p_gdp', 'tax_k_gdp',
    'ui_gdp', 'pension_gdp', 'govt_spending', 'public_investment',
    'defense_spending', 'other_net_spending',
    'interest_payments',
]
FISCAL_LABELS = {
    'primary_deficit_gdp': 'Primary deficit / Y',
    'tax_l_gdp':         'Labour tax / Y',
    'tax_c_gdp':         'Consumption tax / Y',
    'tax_p_gdp':         'Payroll tax / Y',
    'tax_k_gdp':         'Capital tax / Y',
    'ui_gdp':            'UI benefits / Y',
    'pension_gdp':       'Pensions / Y',
    'govt_spending':     'Govt spending (G)',
    'public_investment': 'Public investment (I_g)',
    'defense_spending':  'Defense',
    'other_net_spending':'Other net spending',
    'interest_payments': 'Interest payments (r_B·B)',
}


def _to_arrays(d):
    """Convert a dict of JSON lists into a dict of numpy arrays (skip scalars)."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, list):
            out[k] = np.asarray(v, dtype=float)
        else:
            out[k] = v
    return out


def _rebuild_result(entry, name, r_b=None):
    """Build a minimal stand-in for FiscalScenarioResult from a JSON entry.

    Uses only the attributes that compare_scenarios / debt_fan_chart read:
    cf_macro, cf_budget, B_gdp_path, NFA_path, CA_path, scenario.name, T_balance.
    NFA is the stored (full) counterfactual NFA.

    With *r_b* set, the interest_payments line is recomputed as r_B·B from the
    stored B/Y and Y paths (JSONs written before the r_B fix in
    run_fiscal_figures.py carry an interest line at the capital return r).
    """
    cf_macro  = _to_arrays(entry.get('counterfactual'))
    cf_budget = _to_arrays(entry.get('cf_budget'))
    B_gdp     = np.asarray(entry['B_gdp_path'], dtype=float) if entry.get('B_gdp_path') else None
    if r_b is not None and B_gdp is not None and 'Y' in cf_macro:
        Y = np.asarray(cf_macro['Y'], dtype=float)
        T = len(Y)
        cf_budget['interest_payments'] = float(r_b) * B_gdp[:T] * Y
    NFA, CA   = _nfa_ca_paths(cf_macro)  # cf_macro['NFA'] is the full NFA
    return SimpleNamespace(
        scenario=SimpleNamespace(name=name),
        cf_macro=cf_macro,
        cf_budget=cf_budget,
        B_gdp_path=B_gdp,
        NFA_path=NFA,
        CA_path=CA,
        T_balance=entry.get('T_balance'),
    )


def regen(json_path, output_dir, shocks=None, r_b=None):
    with open(json_path) as fh:
        data = json.load(fh)

    shock_keys = [k for k in data if k != 'params']
    if shocks:
        shock_keys = [s for s in shock_keys if s in shocks]

    for shock in shock_keys:
        g = data[shock]
        p = shock.lower() + '_'
        title_suffix = f'({shock} shock)'

        spec = [('baseline',        'baseline'),
                ('debt_financed',   f'{shock} shock (debt)'),
                ('tax_financed',    f'{shock} shock (τ_l, debt target)'),
                ('nfa_constrained', f'{shock} shock (τ_l, NFA@T)')]
        present = [(k, lbl) for k, lbl in spec if k in g]
        missing = [k for k, _ in spec if k not in g]
        if missing:
            print(f"[{shock}] WARNING: scenarios absent from JSON, skipped: {missing}")
        results = [_rebuild_result(g[k], lbl, r_b=r_b) for k, lbl in present]
        labels  = [lbl for _, lbl in present]
        res_base, *cfs = results

        print(f"[{shock}] regenerating figures from {json_path} -> {output_dir}")

        compare_scenarios(res_base, *cfs,
            variables=MACRO_VARS, var_labels=MACRO_LABELS,
            title=f'Macro Overview {title_suffix}',
            output_dir=output_dir, filename=f'{p}macro_overview.png')
        plt.close('all')

        compare_scenarios(res_base, *cfs,
            variables=PRICE_VARS, var_labels=PRICE_LABELS,
            title=f'Prices — SOE sanity check {title_suffix}',
            output_dir=output_dir, filename=f'{p}prices_sanity.png')
        plt.close('all')

        compare_scenarios(res_base, *cfs,
            variables=FISCAL_VARS, var_labels=FISCAL_LABELS,
            title=f'Fiscal Decomposition {title_suffix}',
            output_dir=output_dir, filename=f'{p}fiscal_decomp.png')
        plt.close('all')

        debt_fan_chart(results, labels,
            output_dir=output_dir, filename=f'{p}debt_fan_chart.png')
        plt.close('all')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', default='output/fiscal_test/fiscal_results.json')
    ap.add_argument('--output-dir', default='output/fiscal_test')
    ap.add_argument('--shock', nargs='*', default=None,
                    help='Restrict to these shock keys (e.g. G Ig). Default: all in file.')
    ap.add_argument('--r-b', type=float, default=None,
                    help='Sovereign rate r_B; recompute the interest line as '
                         'r_B·B from the stored paths (default: keep stored line).')
    args = ap.parse_args()
    regen(args.json, args.output_dir, shocks=args.shock, r_b=args.r_b)
