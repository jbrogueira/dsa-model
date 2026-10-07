"""
Build the general government accounts of Greece, % of GDP, 2019-2025
-> data/gov_accounts_GR.json.

Source: Eurostat dissemination API, general government (S13), unit PC_GDP, geo EL.
  gov_10a_main   main aggregates by ESA 2010 item (revenue items carry the suffix
                 REC, expenditure items PAY); 2019-2025
  gov_10a_taxag  taxes and social contributions by detailed ESA item; 2019-2024
  gov_10a_exp    expenditure by COFOG function and ESA item; 2019-2024

The items are those of docs/BUDGET_ALIGNMENT_PLAN.md, section 2: the ESA revenue
and expenditure lines of the government budget identity and, by COFOG function,
the purchases of goods and services (D1, P2, D632, P51G) and cash social
benefits (D62). Each item carries the plan's model line (the budget line of the
OLG model it maps to; "none" where the model has no such line; null where the
plan assigns no line, as for aggregates).

Detail items that gov_10a_main does not carry (excises D214A, import taxes
D212, other taxes on products D214, property taxes D29A, other income taxes
D51C-D51E, other current taxes D59, imputed contributions D612, pension
contributions D6111 and D6131) are taken from gov_10a_taxag. The plan's "other
income taxes" line is the sum D51C + D51D + D51E, computed here. Changes in
inventories are published together with acquisitions of valuables (P52_P53).

If the API cannot be reached and no cached response exists, the script reads
the frozen Commission workbook data/2026-09-28_GR_DSA_Spring_Forecast_2026_v1.xlsx
(sheets "Balance " and "Sheet3", an extract of gov_10a_main of 2026-10-06) for
the items it contains and marks the file "source": "frozen xlsx copy".

The raw JSON-stat responses are cached under data/eurostat_raw/.

Usage (from code/):  python3 build_gov_accounts_GR.py [--refresh] [--xlsx-fallback] [--out PATH]
"""
import argparse
import datetime as dt
import json
import os
import re
import socket
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RAW = os.path.join(DATA, 'eurostat_raw')
OUT = os.path.join(DATA, 'gov_accounts_GR.json')
XLSX = os.path.join(DATA, '2026-09-28_GR_DSA_Spring_Forecast_2026_v1.xlsx')
API = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data'
FIRST_YEAR, LAST_YEAR = 2019, 2025
YEARS = list(range(FIRST_YEAR, LAST_YEAR + 1))

COFOG_FUNCTIONS = {
    'TOTAL': ('Total', 'G, I_g, defence, public health, pensions, UI, floor'),
    'GF02': ('Defence', 'defence'),
    'GF07': ('Health', 'public health'),
    'GF09': ('Education', 'in G'),
    'GF10': ('Social protection', None),
    'GF1001': ('Sickness and disability', None),
    'GF1002': ('Old age', 'pensions'),
    'GF1003': ('Survivors', 'pensions'),
    'GF1004': ('Family and children', None),
    'GF1005': ('Unemployment', 'UI'),
    'GF1006': ('Housing', None),
    'GF1007': ('Social exclusion n.e.c.', 'floor'),
    'GF1008': ('R&D social protection', None),
    'GF1009': ('Social protection n.e.c.', None),
}
COFOG_ITEMS = ('TE', 'D1', 'P2', 'D632', 'P51G', 'D62')

# key: (dataset, Eurostat na_item code, label, model line)
REVENUE = {
    'TR': ('gov_10a_main', 'TR', 'Total revenue', None),
    'D2': ('gov_10a_main', 'D2REC', 'Taxes on production and imports', None),
    'D211': ('gov_10a_main', 'D211REC', 'Value added type taxes (VAT)', 'tau_c C'),
    'D214A': ('gov_10a_taxag', 'D214A', 'Excise duties and consumption taxes', 'tau_c C'),
    'D212': ('gov_10a_taxag', 'D212', 'Taxes and duties on imports excluding VAT', 'none'),
    'D214': ('gov_10a_taxag', 'D214',
             'Taxes on products, except VAT and import taxes (includes D214A)', None),
    'D29': ('gov_10a_main', 'D29REC', 'Other taxes on production', 'none'),
    'D29A': ('gov_10a_taxag', 'D29A', 'Taxes on land, buildings and other structures', None),
    'D5': ('gov_10a_main', 'D5REC', 'Current taxes on income, wealth, etc.', None),
    'D51A': ('gov_10a_main', 'D51A_C1REC',
             'Taxes on individual or household income including holding gains', 'tau_l base'),
    'D51B': ('gov_10a_main', 'D51B_C2REC',
             'Taxes on the income or profits of corporations including holding gains',
             'tau_k r A'),
    'D51C_D51D_D51E': ('gov_10a_taxag', ('D51C', 'D51D', 'D51E'),
                       'Other taxes on income (holding gains, lottery winnings, n.e.c.)',
                       'none'),
    'D59': ('gov_10a_taxag', 'D59', 'Other current taxes', 'none'),
    'D61': ('gov_10a_main', 'D61REC', 'Net social contributions', 'tau_p w L'),
    'D611': ('gov_10a_main', 'D611REC', "Employers' actual social contributions", None),
    'D613': ('gov_10a_main', 'D613REC', "Households' actual social contributions", None),
    'D612': ('gov_10a_taxag', 'D612', "Employers' imputed social contributions", None),
    'D6111': ('gov_10a_taxag', 'D6111', "Employers' actual pension contributions", None),
    'D6131': ('gov_10a_taxag', 'D6131', "Households' actual pension contributions", None),
    'D91': ('gov_10a_main', 'D91REC', 'Capital taxes', 'bequest tax'),
    'P11_P12_P131': ('gov_10a_main', 'P11_P12_P131',
                     'Market output, output for own final use and payments for '
                     'non-market output', 'none'),
    'D4': ('gov_10a_main', 'D4REC', 'Property income', 'none'),
    'D7': ('gov_10a_main', 'D7REC', 'Other current transfers', 'none'),
    'D9': ('gov_10a_main', 'D9REC', 'Capital transfers', 'none'),
}
EXPENDITURE = {
    'TE': ('gov_10a_main', 'TE', 'Total expenditure', None),
    'D1': ('gov_10a_main', 'D1PAY', 'Compensation of employees', 'in G, defence, health'),
    'P2': ('gov_10a_main', 'P2', 'Intermediate consumption', 'in G, defence, health'),
    'D632': ('gov_10a_main', 'D632PAY',
             'Social transfers in kind, purchased market production', 'health (kappa m)'),
    'P51G': ('gov_10a_main', 'P51G', 'Gross fixed capital formation', 'I_g'),
    'D62': ('gov_10a_main', 'D62PAY', 'Social benefits other than social transfers in kind',
            'pensions, UI, floor'),
    'D3': ('gov_10a_main', 'D3PAY', 'Subsidies', 'none'),
    'D7': ('gov_10a_main', 'D7PAY', 'Other current transfers', 'none'),
    'D9': ('gov_10a_main', 'D9PAY', 'Capital transfers', 'none'),
    'D41': ('gov_10a_main', 'D41PAY', 'Interest', 'r_B B'),
    'P3': ('gov_10a_main', 'P3', 'Final consumption expenditure', None),
    'P52_P53': ('gov_10a_main', 'P52_P53',
                'Changes in inventories and acquisitions less disposals of valuables', None),
    'B9': ('gov_10a_main', 'B9', 'Net lending (+) / net borrowing (-)', None),
}

# Section 2 of docs/BUDGET_ALIGNMENT_PLAN.md (2026-10-07), 2023 values.
PLAN_2023 = {
    'revenue': {'TR': 48.1, 'D2': 17.1, 'D211': 8.8, 'D214A': 3.4, 'D29': 2.9, 'D29A': 1.3,
                'D5': 10.1, 'D51A': 5.9, 'D51B': 2.7, 'D51C_D51D_D51E': 0.5, 'D59': 1.0,
                'D61': 13.0, 'D612': 1.8, 'D91': 0.1, 'P11_P12_P131': 3.1, 'D4': 1.1,
                'D7': 1.2, 'D9': 2.5},
    'revenue_sums': {'D212 + D214 - D214A (other taxes on products)': (2.0, ('D212', 'D214'), ('D214A',)),
                     'D611 + D613 (actual contributions)': (11.2, ('D611', 'D613'), ()),
                     'D6111 + D6131 (pension contributions)': (8.1, ('D6111', 'D6131'), ())},
    'expenditure': {'TE': 49.5, 'D1': 10.5, 'P2': 5.5, 'D632': 3.2, 'P51G': 4.0, 'D62': 17.3,
                    'D3': 2.0, 'D7': 1.3, 'D9': 2.3, 'D41': 3.4},
    'cofog': {('GF02', 'D1'): 1.4, ('GF02', 'P2'): 0.5, ('GF02', 'D632'): 0.0, ('GF02', 'P51G'): 0.4,
              ('GF07', 'D1'): 1.6, ('GF07', 'P2'): 1.7, ('GF07', 'D632'): 2.3, ('GF07', 'P51G'): 0.2,
              ('GF09', 'D1'): 2.7, ('GF09', 'P2'): 0.3, ('GF09', 'D632'): 0.1, ('GF09', 'P51G'): 0.5,
              ('TOTAL', 'D1'): 10.5, ('TOTAL', 'P2'): 5.5, ('TOTAL', 'D632'): 3.2, ('TOTAL', 'P51G'): 4.0,
              ('GF1002', 'D62'): 12.0, ('GF1003', 'D62'): 2.0, ('GF1001', 'D62'): 1.4,
              ('GF1004', 'D62'): 0.8, ('GF1005', 'D62'): 0.6, ('GF1007', 'D62'): 0.3,
              ('GF09', 'D62'): 0.1},
}


def fetch(dataset, extra, refresh=False):
    """Download one dataset for EL / S13 / PC_GDP since FIRST_YEAR; cache the raw JSON."""
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f'{dataset}_EL_S13_PC_GDP.json')
    if refresh or not os.path.exists(path):
        url = (f'{API}/{dataset}?format=JSON&lang=en&geo=EL&unit=PC_GDP&sector=S13'
               f'&sinceTimePeriod={FIRST_YEAR}{extra}')
        print(f'  GET {dataset}')
        with urllib.request.urlopen(url, timeout=180) as r:
            data = r.read()
        with open(path, 'wb') as fh:
            fh.write(data)
    return json.load(open(path))


def unravel(doc):
    """JSON-stat document -> {(category code per dimension): value}, {same key: flag}."""
    ids, size = doc['id'], doc['size']
    labels = {k: {i: c for c, i in doc['dimension'][k]['category']['index'].items()}
              for k in ids}
    values, flags = {}, {}
    status = doc.get('status', {})
    for k, v in doc['value'].items():
        rem, sub = int(k), []
        for n in reversed(size):
            sub.append(rem % n)
            rem //= n
        sub = sub[::-1]
        key = tuple(labels[ids[i]][sub[i]] for i in range(len(ids)))
        values[key] = v
        if k in status:
            flags[key] = status[k]
    return values, flags


class Table:
    """Values of one dataset indexed by (na_item[, cofog99]) and year."""

    def __init__(self, doc, with_cofog=False):
        values, flags = unravel(doc)
        ids = doc['id']
        i_item, i_time = ids.index('na_item'), ids.index('time')
        i_fun = ids.index('cofog99') if with_cofog else None
        self.values, self.flags = {}, {}
        for key, v in values.items():
            if v is None:
                continue
            k = (key[i_fun], key[i_item]) if with_cofog else key[i_item]
            self.values.setdefault(k, {})[int(key[i_time])] = float(v)
            if key in flags:
                self.flags.setdefault(k, {})[int(key[i_time])] = flags[key]
        self.items = set(self.values)
        self.updated = doc.get('updated')
        self.years = sorted({y for d in self.values.values() for y in d})

    def series(self, k):
        if k not in self.values:
            raise KeyError(f'item {k} not in the response')
        return dict(sorted(self.values[k].items())), dict(sorted(self.flags.get(k, {}).items()))


def summed(table, codes):
    """Sum of several items, year by year, over the years every item carries."""
    parts = [table.series(c)[0] for c in codes]
    years = set.intersection(*(set(p) for p in parts))
    return {y: round(sum(p[y] for p in parts), 1) for y in sorted(years)}, {}


def from_api(refresh):
    main = Table(fetch('gov_10a_main', '', refresh))
    taxag = Table(fetch('gov_10a_taxag', '', refresh))
    cofog_query = ''.join(f'&cofog99={f}' for f in COFOG_FUNCTIONS)
    exp = Table(fetch('gov_10a_exp', cofog_query, refresh), with_cofog=True)
    tables = {'gov_10a_main': main, 'gov_10a_taxag': taxag, 'gov_10a_exp': exp}

    def block(spec):
        out = {}
        for key, (ds, code, label, line) in spec.items():
            t = tables[ds]
            if isinstance(code, tuple):
                vals, flags = summed(t, code)
                code_str = ' + '.join(code)
            else:
                vals, flags = t.series(code)
                code_str = code
            entry = {'label': label, 'model_line': line, 'dataset': ds, 'code': code_str,
                     'values': {str(y): v for y, v in vals.items()}}
            if flags:
                entry['flags'] = {str(y): f for y, f in flags.items()}
            out[key] = entry
        return out

    cofog = {}
    for fun in COFOG_FUNCTIONS:
        cofog[fun] = {}
        for item in COFOG_ITEMS:
            vals, _ = exp.series((fun, item))
            cofog[fun][item] = {str(y): v for y, v in vals.items()}

    datasets = {name: {'updated': t.updated, 'years': [t.years[0], t.years[-1]]}
                for name, t in tables.items()}
    return 'Eurostat API (gov_10a_main, gov_10a_taxag, gov_10a_exp)', datasets, \
        block(REVENUE), block(EXPENDITURE), cofog


# Frozen workbook, Sheet3: five Eurostat extracts in sequence, each with its own
# header rows ("Dataset: ", the COFOG function for gov_10a_exp, "TIME"). Row labels
# of each block -> (block, key) of this file.
XLSX_MAIN = {
    'Net lending (+)/net borrowing (-)': ('expenditure', 'B9'),
    'Final consumption expenditure': ('expenditure', 'P3'),
    'Gross fixed capital formation': ('expenditure', 'P51G'),
    'Taxes on production and imports, revenue': ('revenue', 'D2'),
    'Interest, expenditure': ('expenditure', 'D41'),
    'Current taxes on income, wealth, etc., revenue': ('revenue', 'D5'),
    'Net social contributions, revenue': ('revenue', 'D61'),
    'Total general government expenditure': ('expenditure', 'TE'),
    'Total general government revenue': ('revenue', 'TR'),
    'Social benefits other than social transfers in kind, expenditure, of which COFOG 10.2 old age':
        ('cofog', ('GF1002', 'D62')),
    'Social benefits other than social transfers in kind, expenditure, of which COFOG 10.5 unemployment':
        ('cofog', ('GF1005', 'D62')),
}
XLSX_TAXAG = {
    'Taxes on individual or household income including holding gains': ('revenue', 'D51A'),
    'Taxes on the income or profits of corporations including holding gains': ('revenue', 'D51B'),
}
XLSX_EXP = {                                   # by COFOG function of the block
    'Final consumption expenditure': 'P3',
    'Gross fixed capital formation': 'P51G',
    'Social benefits other than social transfers in kind': 'D62',
    'Social transfers in kind - purchased market production': 'D632',
    'Total general government expenditure': 'TE',
}
XLSX_FUNCTIONS = {'Defence': 'GF02', 'Health': 'GF07'}
# "Balance " sheet rows that repeat Sheet3 items; read only to confirm the two agree.
XLSX_BALANCE = {'Total general government revenue': ('revenue', 'TR'),
                'Total general government expenditure': ('expenditure', 'TE'),
                'CIT': ('revenue', 'D51B'),
                'Interest expenditure': ('expenditure', 'D41'),
                'Defence expenditure': ('cofog', ('GF02', 'TE')),
                'Health expenditure': ('cofog', ('GF07', 'TE')),
                'Unemployment benefits': ('cofog', ('GF1005', 'D62')),
                'Pensions (excl. defence and health)': ('cofog', ('GF1002', 'D62'))}


def _year_header(row):
    return [int(c) if c is not None and str(c).strip().isdigit() else None for c in row]


def _row_values(header, row):
    vals = {}
    for y, v in zip(header, row):
        if y is not None and FIRST_YEAR <= y <= LAST_YEAR and isinstance(v, (int, float)):
            vals[str(y)] = round(float(v), 1)
    return vals


def from_xlsx():
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, data_only=True, read_only=True)

    got, extracted = {}, None
    dataset = function = header = None
    for row in wb['Sheet3'].iter_rows(values_only=True):
        a, b = row[0], row[1]
        if isinstance(a, str) and a.startswith('Data extracted on'):
            extracted = extracted or a
            dataset = function = header = None
        elif a == 'Dataset: ' and isinstance(b, str):
            m = re.search(r'\[(gov_10a_\w+?)__', b)
            dataset = m.group(1) if m else None
        elif isinstance(a, str) and a.startswith('Classification of the functions'):
            function = XLSX_FUNCTIONS.get(row[2])
        elif a == 'TIME':
            header = _year_header(row)
        elif a == 'Greece' and header is not None:
            if dataset == 'gov_10a_main' and b in XLSX_MAIN:
                got[XLSX_MAIN[b]] = _row_values(header, row)
            elif dataset == 'gov_10a_taxag' and b in XLSX_TAXAG:
                got[XLSX_TAXAG[b]] = _row_values(header, row)
            elif dataset == 'gov_10a_exp' and function and b in XLSX_EXP:
                got[('cofog', (function, XLSX_EXP[b]))] = _row_values(header, row)

    header = None
    for row in wb['Balance '].iter_rows(values_only=True):
        if row[0] == 'Country':
            header = _year_header(row)
        elif row[0] == 'Greece' and header is not None and row[1] in XLSX_BALANCE:
            key = XLSX_BALANCE[row[1]]
            other = _row_values(header, row)
            mine = got.get(key, {})
            bad = {y: (mine[y], other[y]) for y in mine if y in other and abs(mine[y] - other[y]) > 0.05}
            if bad:
                print(f'  Sheet3 and "Balance " disagree on {key}: {bad}')

    revenue, expenditure, cofog = {}, {}, {}
    for (blk, key), vals in got.items():
        if blk == 'cofog':
            cofog.setdefault(key[0], {})[key[1]] = vals
        else:
            spec = (REVENUE if blk == 'revenue' else EXPENDITURE)[key]
            (revenue if blk == 'revenue' else expenditure)[key] = {
                'label': spec[2], 'model_line': spec[3], 'dataset': 'xlsx Sheet3', 'code': spec[1],
                'values': vals}
    years = sorted({int(y) for e in list(revenue.values()) + list(expenditure.values())
                    for y in e['values']})
    datasets = {'xlsx': {'updated': extracted, 'years': [years[0], years[-1]] if years else None}}
    return 'frozen xlsx copy', datasets, revenue, expenditure, cofog


def compare_with_plan(revenue, expenditure, cofog):
    """Print every 2023 value of the plan's section 2 against the file."""
    def get(block, key):
        return block.get(key, {}).get('values', {}).get('2023')

    print('\n2023 values against docs/BUDGET_ALIGNMENT_PLAN.md section 2 (plan / file)')
    diffs = 0
    for name, blk, spec in (('revenue', revenue, PLAN_2023['revenue']),
                            ('expenditure', expenditure, PLAN_2023['expenditure'])):
        for key, plan in spec.items():
            v = get(blk, key)
            mark = '' if v is not None and abs(v - plan) < 0.05 else '   <-- differs'
            diffs += bool(mark)
            print(f'  {name:12s} {key:16s} {plan:6.1f} / {v if v is None else f"{v:.1f}"}{mark}')
    for name, (plan, plus, minus) in PLAN_2023['revenue_sums'].items():
        parts = [get(revenue, k) for k in plus + minus]
        if any(p is None for p in parts):
            v = None
        else:
            v = sum(get(revenue, k) for k in plus) - sum(get(revenue, k) for k in minus)
        mark = '' if v is not None and abs(v - plan) < 0.05 else '   <-- differs'
        diffs += bool(mark)
        print(f'  {"revenue":12s} {name:52s} {plan:6.1f} / {v if v is None else f"{v:.1f}"}{mark}')
    for (fun, item), plan in PLAN_2023['cofog'].items():
        v = cofog.get(fun, {}).get(item, {}).get('2023')
        mark = '' if v is not None and abs(v - plan) < 0.05 else '   <-- differs'
        diffs += bool(mark)
        print(f'  {"cofog":12s} {fun + " " + item:16s} {plan:6.1f} / {v if v is None else f"{v:.1f}"}{mark}')
    print(f'  {diffs} value(s) differ from the plan')


def consistency(revenue, expenditure, cofog):
    """Accounting checks at one-decimal rounding; printed, not enforced."""
    def rv(k, y):
        return revenue.get(k, {}).get('values', {}).get(str(y))

    def ev(k, y):
        return expenditure.get(k, {}).get('values', {}).get(str(y))

    print('\nAccounting checks (rounding of the published one-decimal values is 0.1 per item)')
    for y in YEARS:
        if rv('TR', y) is None or ev('TE', y) is None:
            continue
        print(f'  {y}: TR - TE = {rv("TR", y) - ev("TE", y):5.1f}, B9 = {ev("B9", y)}', end='')
        if all(rv(k, y) is not None for k in ('D2', 'D211', 'D212', 'D214', 'D29')):
            print(f';  D2 - (D211 + D212 + D214 + D29) = '
                  f'{rv("D2", y) - rv("D211", y) - rv("D212", y) - rv("D214", y) - rv("D29", y):5.1f}',
                  end='')
        if all(rv(k, y) is not None for k in ('D61', 'D611', 'D612', 'D613')):
            print(f';  D61 - (D611 + D612 + D613) = '
                  f'{rv("D61", y) - rv("D611", y) - rv("D612", y) - rv("D613", y):5.1f}', end='')
        print()
    for item in ('D1', 'P2', 'D632', 'P51G', 'D62', 'TE'):
        t = cofog.get('TOTAL', {}).get(item, {}).get('2023')
        m = expenditure.get(item, {}).get('values', {}).get('2023')
        if t is not None and m is not None:
            print(f'  2023 COFOG TOTAL {item} {t:5.1f} vs gov_10a_main {m:5.1f}')
    g10 = cofog.get('GF10', {}).get('D62', {}).get('2023')
    parts = [cofog.get(f, {}).get('D62', {}).get('2023')
             for f in ('GF1001', 'GF1002', 'GF1003', 'GF1004', 'GF1005', 'GF1006', 'GF1007',
                       'GF1008', 'GF1009')]
    if g10 is not None and all(p is not None for p in parts):
        print(f'  2023 D62 social protection GF10 {g10:5.1f} vs sum of groups {sum(parts):5.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true',
                    help='re-download even if the raw JSON is cached')
    ap.add_argument('--xlsx-fallback', action='store_true',
                    help='read the frozen workbook instead of the API')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    print('General government accounts, Greece, % of GDP')
    if args.xlsx_fallback:
        source, datasets, revenue, expenditure, cofog = from_xlsx()
    else:
        try:
            source, datasets, revenue, expenditure, cofog = from_api(args.refresh)
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            print(f'  API unreachable ({e}); reading the frozen workbook')
            source, datasets, revenue, expenditure, cofog = from_xlsx()

    out = {
        'source': source,
        'extracted': dt.date.today().isoformat(),
        'unit': 'percent of GDP',
        'geo': 'EL', 'sector': 'S13',
        'years': YEARS,
        'datasets': datasets,
        'model_line_note': ('Model line of docs/BUDGET_ALIGNMENT_PLAN.md section 2: the budget '
                            'line of the OLG model the item maps to; "none" where the model has '
                            'no such line; null where the plan assigns none (aggregates and '
                            'sub-items)'),
        'revenue': revenue,
        'expenditure': expenditure,
        'cofog_functions': {f: {'label': lab, 'model_line': line}
                            for f, (lab, line) in COFOG_FUNCTIONS.items()},
        'cofog': cofog,
    }
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f'wrote {os.path.relpath(args.out)}  (source: {source})')
    for name, d in datasets.items():
        print(f'  {name}: updated {d["updated"]}, years {d["years"]}')

    def show(block, title):
        print(f'\n{title}')
        for key, e in block.items():
            v = e['values']
            print(f'  {key:16s} ' + ' '.join(f'{y}:{v.get(str(y), "   . "):>5}' for y in YEARS)
                  + f'   {e["model_line"]}')
    show(revenue, 'Revenue')
    show(expenditure, 'Expenditure')
    print('\nCOFOG, 2023 (D1, P2, D632, P51G, TE, D62)')
    for fun, items in cofog.items():
        print(f'  {fun:7s} ' + ' '.join(f'{it}:{items.get(it, {}).get("2023", "."):>5}'
                                       for it in COFOG_ITEMS))

    consistency(revenue, expenditure, cofog)
    compare_with_plan(revenue, expenditure, cofog)


if __name__ == '__main__':
    main()
