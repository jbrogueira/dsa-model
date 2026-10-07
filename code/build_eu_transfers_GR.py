"""
Build the net transfer from the EU budget to Greece -> data/eu_transfers_GR.npz.

The transfer is EU spending in Greece less Greece's contribution to the EU
budget, in % of GDP, by calendar year:

  transfer_t = (operating expenditure_t + NGEU_t - national contribution_t) / GDP_t.

Operating expenditure is EU budget spending in Greece without the
administration heading; the national contribution is the VAT- and GNI-based
own resources, the plastics own resource and the corrections, without the
customs duties collected for the EU. This is the Commission's operating
budgetary balance before its rescaling of contributions to the level of
allocated expenditure (the rescaled balance, 2014-2020, is kept as
obb_adjusted). NGEU spending in Greece is counted from 2021: the Recovery and
Resilience Facility (RRF) grants at the dates of the Commission's payments,
plus the other NGEU lines (ERDF and ESF top-ups, Just Transition Fund,
EAFRD). The RRF loans are not transfers and are not counted.

Sources
  European Commission, "EU spending and revenue - Data 2000-2025" (xlsx,
  29 September 2026), one sheet per year, column EL, EUR million: TOTAL
  EXPENDITURE, the administration heading (5 to 2020, 7 from 2021), TOTAL
  national contribution (to 2020; from 2021 TOTAL own resources less customs
  duties and sugar levies plus TOTAL balances and adjustments), TOTAL NGEU and
  its RRF line (from 2021), GNI.
  RRF grant payments: Commission and Greek government releases listed in
  RRF_GRANTS, with dates.
  GDP: Eurostat nama_10_gdp, current prices (data/nomgdp_GR.json); 2026 from
  the debt projection (data/dsa_projection_GR.npz).

The budget data's RRF line agrees with the dated payments in 2021-2023 and
2025 (within 2%) and is EUR 728 million below them in 2024; the dated
payments are used and both series are saved.

Path
  2014-2025  data.
  2026       the 2014-2020 mean of the EU budget balance plus the RRF grants
             of 2026: the April payment, the request under assessment and
             the rest of the grant allocation, assumed paid by the Facility's
             31 December 2026 deadline (flag 'projected').
  2027-2400  the 2014-2020 mean of the EU budget balance.
  1900-2013  the 2014 value.

Usage (from code/):  python3 build_eu_transfers_GR.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
NOMGDP = os.path.join(DATA, 'nomgdp_GR.json')
DSA = os.path.join(DATA, 'dsa_projection_GR.npz')
OUT = os.path.join(DATA, 'eu_transfers_GR.npz')

FIRST_YEAR, LAST_YEAR = 1900, 2400
MEAN_YEARS = (2014, 2020)                 # the constant after the RRF: mean of these years
LAST_RRF_YEAR = 2026
PRINT_YEARS = range(2019, 2031)

BUDGET_URL = ('https://commission.europa.eu/strategy-and-policy/eu-budget/long-term-eu-budget/'
              '2021-2027/spending-and-revenue_en')
BUDGET_FILE = 'eu_budget_spending_and_revenue_2000-2025.xlsx (29 September 2026)'

# EUR million, sheet = year, column EL (16). expenditure: TOTAL EXPENDITURE; administration:
# heading 5 ADMINISTRATION (2014-2020) or 7 European Public Administration (2021-2025);
# contribution: TOTAL national contribution (2014-2020) or TOTAL own resources - customs
# duties - sugar levies + TOTAL balances and adjustments (2021-2025); ngeu: TOTAL NGEU;
# rrf: NGEU row 2.2.21 European Recovery and Resilience Facility; gni: Gross National
# Income; eu_opex, eu_contribution: expenditure less administration and the contribution
# summed over the member-state columns.
BUDGET = {
    2014: dict(expenditure=7094.97, administration=37.54, contribution=1826.60, ngeu=0.0, rrf=0.0,
               gni=178380.62, eu_opex=120884.01, eu_contribution=116531.84),
    2015: dict(expenditure=6209.66, administration=28.92, contribution=1205.60, ngeu=0.0, rrf=0.0,
               gni=176522.67, eu_opex=122656.86, eu_contribution=118604.32),
    2016: dict(expenditure=5849.90, administration=24.95, contribution=1570.16, ngeu=0.0, rrf=0.0,
               gni=176187.86, eu_opex=109834.02, eu_contribution=112080.16),
    2017: dict(expenditure=5130.07, administration=27.82, contribution=1247.74, ngeu=0.0, rrf=0.0,
               gni=178035.30, eu_opex=103632.31, eu_contribution=94968.62),
    2018: dict(expenditure=4870.08, administration=32.80, contribution=1487.70, ngeu=0.0, rrf=0.0,
               gni=183070.28, eu_opex=122141.54, eu_contribution=122123.73),
    2019: dict(expenditure=5257.71, administration=34.38, contribution=1516.60, ngeu=0.0, rrf=0.0,
               gni=186263.95, eu_opex=125712.71, eu_contribution=123402.92),
    2020: dict(expenditure=7414.45, administration=31.21, contribution=1654.54, ngeu=0.0, rrf=0.0,
               gni=164887.92, eu_opex=138825.26, eu_contribution=140223.38),
    2021: dict(expenditure=6282.26, administration=38.88, contribution=1570.66, ngeu=3466.44, rrf=2310.09,
               gni=181904.13, eu_opex=138807.19, eu_contribution=139597.88),
    2022: dict(expenditure=5851.22, administration=39.35, contribution=1726.36, ngeu=2593.15, rrf=1717.77,
               gni=203136.08, eu_opex=140580.79, eu_contribution=129381.93),
    2023: dict(expenditure=5852.22, administration=42.34, contribution=1634.38, ngeu=3669.54, rrf=3405.21,
               gni=216871.18, eu_opex=133171.26, eu_contribution=127484.78),
    2024: dict(expenditure=5100.58, administration=46.38, contribution=1714.73, ngeu=998.09, rrf=429.27,
               gni=230167.71, eu_opex=112260.59, eu_contribution=121016.66),
    2025: dict(expenditure=5710.24, administration=119.44, contribution=1907.71, ngeu=3612.40, rrf=3389.20,
               gni=242221.45, eu_opex=121976.09, eu_contribution=132397.52),
}

# RRF grant payments to Greece: (date, EUR million, description, source).
RRF_GRANTS = (
    ('2021-08-09', 2310.09,
     'pre-financing, 13% of the grant allocation; EUR 3.96 bn with the loan pre-financing',
     'amount: EU spending and revenue 2021, NGEU row 2.2.21; date: '
     'https://www.ot.gr/2021/08/09/english-edition/recovery-fund-the-commission-disbursed-the-first-4-billion-euros-for-greece'),
    ('2022-04-08', 1720.0,
     'first payment; EUR 3.56 bn with EUR 1.84 bn of loans',
     'https://www.ot.gr/2022/04/08/english-edition/recovery-fund-greece-received-the-first-3-6-billion-euros-the-9-projects-that-are-starting/'),
    ('2023-01-19', 1720.0,
     'second payment; EUR 3.56 bn with EUR 1.84 bn of loans; Commission decision C(2023) 344 of 12 January 2023',
     'https://amna.gr/en/article/700374/European-Commission-approves-second-RRF-payment-to-Greece-of-36-bln-euros; '
     'https://commission.europa.eu/system/files/2023-01/C_2023_344_1_EN.pdf'),
    ('2023-12-28', 1690.0,
     'third payment; EUR 3.64 bn with EUR 1.95 bn of loans',
     'https://www.ot.gr/2023/12/28/english-edition/eu-commission-disburses-e3-64-billion-to-greece-under-the-rrf/'),
    ('2024-01-25', 158.7,
     'REPowerEU chapter pre-financing',
     'https://amna.gr/print/854243'),
    ('2024-10-16', 998.6,
     'fourth payment, grants',
     'https://greece20.gov.gr/en/?p=34099'),
    ('2025-05-02', 1350.0,
     'fifth payment; EUR 3.13 bn with EUR 1.78 bn of loans',
     'https://greece20.gov.gr/en/5th-payment-en/'),
    ('2025-11-24', 2100.0,
     'sixth payment, grants; approved 26 November 2025, paid the same week',
     'https://greece20.gov.gr/en/?p=38399; https://www.tovima.com/finance/ec-disburses-planned-e2-1bn-to-greece/'),
    ('2026-04-23', 884.0,
     'seventh payment; EUR 1.18 bn with EUR 294 m of loans (split from the preliminary assessment)',
     'https://greece20.gov.gr/en/?p=40183; https://en.protothema.gr/?p=785133'),
)
RRF_GRANT_ALLOCATION = 18220.0            # EUR million, https://greece20.gov.gr/en/?p=40183
RRF_2026_REQUESTED = 866.0                # eighth grant request under assessment, 1 September 2026, https://ered.gr/?p=84810


def gdp_by_year():
    """Nominal GDP, EUR million, from Eurostat (to 2025) and the debt projection (2026)."""
    doc = json.load(open(NOMGDP))
    index = doc['dimension']['time']['category']['index']
    gdp = {int(y): doc['value'][str(i)] for y, i in index.items() if str(i) in doc['value']}
    dsa = np.load(DSA)
    for y, v in zip(dsa['years'], dsa['nominal_gdp']):
        gdp.setdefault(int(y), 1000.0 * float(v))
    return gdp


def rrf_grants_by_year():
    out = {}
    for date, amount, _, _ in RRF_GRANTS:
        out[int(date[:4])] = out.get(int(date[:4]), 0.0) + amount
    return out


def main():
    gdp = gdp_by_year()
    rrf_paid = rrf_grants_by_year()
    data_years = np.array(sorted(BUDGET))
    budget = np.array([BUDGET[y]['expenditure'] - BUDGET[y]['administration'] - BUDGET[y]['contribution']
                       for y in data_years])                                  # EU budget without NGEU
    ngeu_other = np.array([BUDGET[y]['ngeu'] - BUDGET[y]['rrf'] for y in data_years])
    rrf_budget = np.array([BUDGET[y]['rrf'] for y in data_years])
    rrf_dated = np.array([rrf_paid.get(int(y), 0.0) for y in data_years])
    gdp_data = np.array([gdp[int(y)] for y in data_years])
    net = budget + ngeu_other + rrf_dated
    net_budget_rrf = budget + ngeu_other + rrf_budget
    data_ratio = net / gdp_data

    mean_mask = (data_years >= MEAN_YEARS[0]) & (data_years <= MEAN_YEARS[1])
    constant = float((budget[mean_mask] / gdp_data[mean_mask]).mean())

    paid_total = sum(a for _, a, _, _ in RRF_GRANTS)
    rrf_2026_paid = rrf_paid.get(LAST_RRF_YEAR, 0.0)
    rrf_2026_remaining = RRF_GRANT_ALLOCATION - paid_total - RRF_2026_REQUESTED
    rrf_2026 = rrf_2026_paid + RRF_2026_REQUESTED + rrf_2026_remaining
    ratio_2026 = constant + rrf_2026 / gdp[LAST_RRF_YEAR]

    adj_mask = data_years <= 2020
    obb_adjusted = np.array([(BUDGET[y]['expenditure'] - BUDGET[y]['administration'])
                             - BUDGET[y]['contribution'] * BUDGET[y]['eu_opex'] / BUDGET[y]['eu_contribution']
                             for y in data_years[adj_mask]])

    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    transfer = np.full(years.shape, constant)
    flags = np.full(years.shape, f'mean_{MEAN_YEARS[0]}_{MEAN_YEARS[1]}', dtype=object)
    transfer[years < data_years[0]] = data_ratio[0]
    flags[years < data_years[0]] = f'held_{data_years[0]}'
    for y, v in zip(data_years, data_ratio):
        transfer[years == y] = v
        flags[years == y] = 'data'
    flags[years == 2024] = 'data_rrf_source_conflict'
    transfer[years == LAST_RRF_YEAR] = ratio_2026
    flags[years == LAST_RRF_YEAR] = 'projected'
    if np.isnan(transfer).any():
        raise ValueError('transfer path has gaps')

    sources = [f'{BUDGET_URL} ; file {BUDGET_FILE}, column EL, EUR million; rows per sheet: TOTAL EXPENDITURE, '
               'administration heading, national contribution (see BUDGET in build_eu_transfers_GR.py)']
    sources += [f'{y}: expenditure {b["expenditure"]:.2f}, administration {b["administration"]:.2f}, '
                f'contribution {b["contribution"]:.2f}, NGEU {b["ngeu"]:.2f} of which RRF {b["rrf"]:.2f}, '
                f'GNI {b["gni"]:.2f}' for y, b in BUDGET.items()]
    sources += [f'RRF grants {d}: EUR {a:.2f} m, {desc}; {src}' for d, a, desc, src in RRF_GRANTS]
    sources += [f'RRF grant allocation EUR {RRF_GRANT_ALLOCATION:.0f} m; eighth grant request EUR '
                f'{RRF_2026_REQUESTED:.0f} m under assessment (https://ered.gr/?p=84810, 1 September 2026); '
                f'rest of the allocation EUR {rrf_2026_remaining:.2f} m assumed paid in {LAST_RRF_YEAR}',
                'GDP: Eurostat nama_10_gdp CP_MEUR (data/nomgdp_GR.json); 2026: data/dsa_projection_GR.npz nominal_gdp',
                f'2024: the budget data\'s RRF line (EUR {BUDGET[2024]["rrf"]:.2f} m) is below the dated payments '
                f'(EUR {rrf_paid[2024]:.2f} m); the dated payments are used']
    np.savez(OUT, years=years, transfer_over_Y=transfer, flags=np.array(flags, dtype=str),
             data_years=data_years, eu_budget_net=budget, ngeu_other=ngeu_other,
             rrf_grants_dated=rrf_dated, rrf_grants_budget=rrf_budget, net_transfer=net,
             net_transfer_budget_rrf=net_budget_rrf, gdp=gdp_data,
             gni=np.array([BUDGET[y]['gni'] for y in data_years]),
             obb_adjusted_years=data_years[adj_mask], obb_adjusted=obb_adjusted,
             constant_after_rrf=constant, rrf_2026_paid=rrf_2026_paid, rrf_2026_requested=RRF_2026_REQUESTED,
             rrf_2026_remaining=rrf_2026_remaining, gdp_2026=gdp[LAST_RRF_YEAR],
             sources=np.array(sources, dtype=str))

    print(f'  EU budget balance without NGEU, mean {MEAN_YEARS[0]}-{MEAN_YEARS[1]}: {100 * constant:.3f}% of GDP')
    print(f'  RRF grants {LAST_RRF_YEAR}: paid {rrf_2026_paid:.0f} + requested {RRF_2026_REQUESTED:.0f} '
          f'+ rest of allocation {rrf_2026_remaining:.0f} = EUR {rrf_2026:.0f} m')
    print('  year  transfer/GDP   budget  NGEU-other  RRF(dated)  RRF(budget)  flag')
    for y in PRINT_YEARS:
        k = years == y
        parts = ''
        if y in BUDGET:
            j = int(np.where(data_years == y)[0][0])
            parts = f'{budget[j]:8.1f}  {ngeu_other[j]:9.1f}  {rrf_dated[j]:10.1f}  {rrf_budget[j]:11.1f}'
        print(f'  {y}  {100 * float(transfer[k][0]):7.3f}%   {parts:44s}  {flags[k][0]}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
