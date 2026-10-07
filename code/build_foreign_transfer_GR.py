"""
Net transfers from the EU budget to the Greek general government, % of GDP,
by calendar year -> data/foreign_transfer_GR.npz.

The line is the government's net receipts: current and capital transfers
received from EU institutions (ESA items D.7 and D.9) less Greece's payment
of own resources (D.7 paid to the EU). It is the flow the government budget
books; payments the EU makes to other residents (direct payments to farmers
under the common agricultural policy, grants to firms) are not government
revenue and are not in it. data/eu_transfers_GR.npz, built by
build_eu_transfers_GR.py, has the broader series of all EU payments to Greece
less the national contribution (3.5% of GDP in 2023) for reference.

Values, % of GDP (general government accounts, Eurostat gov_10a_main, the
EU components as in BUDGET_ALIGNMENT_PLAN.md section 2.1):

  2023  D.7 from the EU 0.3 + D.9 from the EU 2.3 - own resources 0.7 = 1.9
  2024  D.9 from the EU 2.2, the other two unchanged                 = 1.8
  2025  D.9 from the EU 3.1 (RRF grants), the other two unchanged    = 2.7
  2026  held at the 2025 value: the last year of RRF grant payments
  2027- 1.0, the level of the years before the Recovery and Resilience
        Facility (2014-2020), when the receipts were structural and cohesion
        funds and the rural development part of the agricultural policy

The 2027 level is a judgement about the next multiannual financial
framework, which is not known; it is flagged in `flags`. Years before 2023
are set to the 2023 value (the model starts in 2023; the line is not used
before). Years run 1900-2400 for the transition's horizon.

Usage (from code/):  python3 build_foreign_transfer_GR.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'foreign_transfer_GR.npz')

ANCHORS = {2023: 1.9, 2024: 1.8, 2025: 2.7, 2026: 2.7}
AFTER_RRF = 1.0          # from 2027, % of GDP
FIRST_AFTER = 2027


def transfer_path(years):
    years = np.asarray(years, dtype=int)
    out = np.empty(len(years))
    flags = np.empty(len(years), dtype=object)
    for i, y in enumerate(years):
        if y in ANCHORS:
            out[i] = ANCHORS[y] / 100.0
            flags[i] = 'accounts' if y <= 2025 else 'held_2025'
        elif y < 2023:
            out[i] = ANCHORS[2023] / 100.0
            flags[i] = 'held_2023'
        else:
            out[i] = AFTER_RRF / 100.0
            flags[i] = 'pre_rrf_level_judgement'
    return out, flags


def main():
    years = np.arange(1900, 2401)
    path, flags = transfer_path(years)
    np.savez(OUT, years=years, transfer_over_Y=path, flags=flags.astype(str),
             anchors_years=np.array(sorted(ANCHORS)),
             anchors_values=np.array([ANCHORS[y] for y in sorted(ANCHORS)]) / 100.0,
             after_rrf=AFTER_RRF / 100.0, first_after=FIRST_AFTER)
    for y in (2023, 2024, 2025, 2026, 2027, 2030, 2060):
        i = int(np.where(years == y)[0][0])
        print(f'  {y}: {100 * path[i]:.1f}% of GDP  ({flags[i]})')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
