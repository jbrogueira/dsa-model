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
import os
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

# Budget-components chart: one line per budget item, as a share of Y.
# Taxes solid, spending dashed; colors from the validated dataviz palette.
BUDGET_COMPONENTS = [
    # key                  label                        color      linestyle
    ('tax_l',              'Labour tax',                '#104281', '-'),
    ('tax_p',              'Payroll tax',               '#6da7ec', '-'),
    ('tax_c',              'Consumption tax',           '#256abf', '-'),
    ('tax_k',              'Capital tax',               '#1baf7a', '-'),
    ('pension',            'Pensions',                  '#008300', '--'),
    ('gov_health',         'Govt health',               '#e87ba4', '--'),
    ('ui',                 'UI benefits',               '#eda100', '--'),
    ('govt_spending',      'Govt spending (G)',         '#e34948', '--'),
    ('defense_spending',   'Defense',                   '#4a3aa7', '--'),
    ('public_investment',  'Public investment (I_g)',   '#eb6834', '--'),
    ('other_net_spending', 'Other net spending',        '#898781', '--'),
]


def budget_components_chart(result, title, output_dir, filename,
                            base_budget=None, base_Y=None):
    """One line per budget item (/Y, in %) plus the primary balance (/Y).

    Items are plotted with their budget sign: taxes and spending positive,
    other_net_spending negative (it is a net-revenue residual).  The primary
    balance is total revenue minus primary spending.

    With *base_budget* and *base_Y* given, every line is the deviation of the
    scenario's share of Y from the baseline's share of Y, in pp of Y.
    """
    cb = result.cf_budget
    Y = np.asarray(result.cf_macro['Y'], dtype=float)
    years = np.arange(len(Y))
    deviations = base_budget is not None

    def share(budget, y, key):
        return np.asarray(budget[key], dtype=float) / y * 100.0

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    for key, label, color, linestyle in BUDGET_COMPONENTS:
        s = share(cb, Y, key)
        if deviations:
            s = s - share(base_budget, base_Y, key)
        ax.plot(years, s, color=color, linestyle=linestyle,
                linewidth=1.8, label=label)

    pb = -share(cb, Y, 'primary_deficit')
    if deviations:
        pb = pb + share(base_budget, base_Y, 'primary_deficit')
    ax.plot(years, pb, color='#0b0b0b', linewidth=2.4, zorder=5,
            label='Primary balance')
    ax.axhline(0.0, color='#c3c2b7', linewidth=0.8, zorder=1)

    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Deviation from baseline (pp of Y)' if deviations
                  else 'Share of Output Y (%)')
    ax.set_xlim(years[0], years[-1])
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved {path}")


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


SCENARIO_LABELS = {
    'baseline':        'baseline',
    'debt_financed':   '{shock} shock (debt)',
    'tax_financed':    '{shock} shock (τ_l, debt target)',
    'nfa_constrained': '{shock} shock (τ_l, NFA@T)',
}


def regen_budget_components(json_path, output_dir, shock, scenario):
    with open(json_path) as fh:
        data = json.load(fh)
    entry = data[shock][scenario]
    label = SCENARIO_LABELS.get(scenario, scenario).format(shock=shock)
    result = _rebuild_result(entry, label)
    print(f"[{shock}/{scenario}] budget-components chart from {json_path} -> {output_dir}")
    budget_components_chart(result,
        title=f'Budget components and primary balance — {label}',
        output_dir=output_dir,
        filename=f'{shock.lower()}_{scenario}_budget_components.png')
    plt.close('all')
    budget_components_chart(result,
        title=f'Budget components and primary balance — {label} — deviations from baseline',
        output_dir=output_dir,
        filename=f'{shock.lower()}_{scenario}_budget_components_dev.png',
        base_budget=_to_arrays(entry['base_budget']),
        base_Y=np.asarray(entry['baseline']['Y'], dtype=float))
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
    ap.add_argument('--budget-components', nargs=2, metavar=('SHOCK', 'SCENARIO'),
                    default=None,
                    help='Draw only the stacked budget-components chart for one '
                         'shock/scenario (e.g. Ig tax_financed) and exit.')
    args = ap.parse_args()
    if args.budget_components:
        regen_budget_components(args.json, args.output_dir, *args.budget_components)
    else:
        regen(args.json, args.output_dir, shocks=args.shock, r_b=args.r_b)
