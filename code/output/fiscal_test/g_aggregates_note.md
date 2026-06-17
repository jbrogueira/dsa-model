# G shock — baseline transition: aggregates and GBC

Source: `fiscal_results.json` (G shock, baseline block). Figures checked: `g_macro_overview.png`, `g_fiscal_decomp.png`, `g_prices_sanity.png`, `g_debt_fan_chart.png`. `T_balance = 60`; 80 periods.

## Main aggregates

- **Y**: flat, 0.449 → 0.452 (+0.6%); in the SOE at fixed `r`, K/L and Y/L are pinned, so output moves only with L.
- **K_domestic**: flat, 1.646 → 1.657; pinned to L by the firm FOC at the fixed world `r`.
- **L**: flat, 0.237 → 0.238.
- **C**: rises, 0.269 → 0.273 (+1.4%).
- **w**: constant at 1.271; exogenous in the SOE.
- **r**: constant at 0.040; the exogenous world rate.
- **A** (household wealth): rises, 1.794 → 1.823 (+1.6%).
- **A/Y**: rises slightly, 3.996 → 4.034.
- **NFA/Y**: falls steadily, −1.32 → −4.26; with A/Y and K/Y roughly flat, NFA/Y declines nearly one-for-one with the rising B/Y (NFA = A − K_domestic − B).
- **B/Y**: rises ×2.8, 1.646 → 4.626; `r` (4%) exceeds growth (≈0) and the ≈2%-of-Y primary surplus is below the debt-stabilizing level.
- **K_g/Y**: not present in this run (no public capital; panel empty).

## Government budget constraint (shares of Y)

- **total_revenue**: flat, 0.344 → 0.346.
- **tax_l**: flat, 0.070.
- **tax_c**: flat, 0.109 → 0.110.
- **tax_p**: constant, 0.130.
- **tax_k**: flat, 0.036.
- **pension**: rises slightly, 0.159 → 0.162 (population aging).
- **ui**: flat, 0.013.
- **gov_health**: flat, 0.054.
- **govt_spending (G)**: constant, 0.130 (pinned as a share of Y).
- **public_investment (I_g)**: constant, 0.030.
- **defense_spending**: constant, 0.030.
- **other_net_spending**: constant, −0.091; calibration residual pinned at the initial steady state.
- **bequest_transfers**: flat, 0.039.
- **interest_payments (r·B)**: rises ×2.8, 0.066 → 0.185; tracks the rising B/Y at `r` = 4%.
- **primary surplus** (= −primary_deficit/Y): flat, 0.0205 → 0.0186; held near the calibration target `primary_balance_target_over_Y` = 0.0195.

\newpage

# G shock — debt-financed vs baseline

Policy: G raised by 2% of Y from t=0, financed entirely by debt (no tax change); `balance_condition = terminal_debt_gdp`; no instrument adjustment; converged. Real allocation is unchanged because at the fixed world `r` there is no crowding out and G does not enter the household problem.

## Main aggregates vs baseline

- **Y**: unchanged (Δ = 0 at every period).
- **K_domestic**: unchanged (Δ = 0); K/L is pinned by `r`, which does not move.
- **L**: unchanged (Δ = 0).
- **C**: unchanged (Δ = 0); households face the same prices, taxes, and lifetime budget.
- **A**: unchanged (Δ = 0).
- **A/Y**: unchanged (Δ = 0).
- **NFA/Y**: far below baseline, −2.29 at `T_balance` and −3.96 at the end (in units of Y); the extra debt is absorbed abroad one-for-one (A and K_domestic unchanged).
- **B/Y**: far above baseline, +2.29 at `T_balance` and +3.96 at the end (8.58 vs 4.63), as the unfunded G accumulates.
- **w**, **r**: unchanged (exogenous).

## Government budget constraint vs baseline

- **govt_spending (G)**: +2.0 pp of Y at all periods (the shock).
- **interest_payments (r·B)**: above baseline and widening, +9.2 pp of Y at `T_balance` and +15.8 pp at the end, compounding on the higher debt.
- **primary surplus**: 2.0 pp of Y below baseline at every period, a constant gap (G up 2 pp, revenue unchanged); because the baseline surplus is only ≈1.9% of Y, the debt-financed level itself runs from +0.05% at t=0 to −0.14% (a small primary deficit) at the end.
- **total_revenue**, **tax_l**, **tax_c**, **tax_p**, **tax_k**, **pension**, **ui**, **gov_health**, **public_investment**, **defense_spending**, **other_net_spending**, **bequest_transfers**: unchanged from baseline (Δ = 0).

\newpage

# G shock — labour-tax financed, debt target vs baseline

Policy: G raised by 2% of Y, financed by labour tax; `balance_condition = terminal_debt_gdp` pinned to the baseline terminal B/Y; Δτ_l = +3.17 pp; converged. Higher τ_l lowers lifetime income: households work more, consume less, and save less.

## Main aggregates vs baseline

- **Y**: above baseline, +0.0045 (+1.0%) at `T_balance` and the end; the income effect on labour raises L.
- **K_domestic**: above baseline, +0.017 (+1.0%); rises with L at fixed K/L.
- **L**: above baseline, +0.0024 (+1.0%); income effect of the tax dominates.
- **C**: below baseline, −0.0076 (−2.8%); lower after-tax lifetime income.
- **A**: below baseline, −0.041 (−2.2%); reduced saving.
- **A/Y**: below baseline, −0.13 of Y throughout.
- **NFA/Y**: below baseline, −0.14 at `T_balance` and −0.17 at the end (in units of Y); tracks the lower A/Y (K/Y unchanged, B/Y near baseline).
- **B/Y**: at baseline by construction at `T_balance` (+0.014), drifting +0.04 above by the end.
- **w**, **r**: unchanged (exogenous).

## Government budget constraint vs baseline

- **tax_l**: +2.2 pp of Y (the +3.17 pp rate rise on the labour base).
- **total_revenue**: +1.7 pp of Y, from the higher labour tax.
- **govt_spending (G)**: +2.0 pp of Y (the shock).
- **primary surplus**: −0.1 pp of Y; the tax rise nearly offsets the higher G, leaving B/Y on its target path.
- **interest_payments (r·B)**: ≈ baseline (+0.06 pp at `T_balance`, +0.17 pp at the end), since B/Y is held to target.
- **pension**: −0.16 pp of Y; higher Y lowers the pension share.
- **public_investment**, **defense_spending**, **other_net_spending**, **bequest_transfers**, **tax_c**, **tax_p**, **tax_k**, **ui**, **gov_health**: ≈ unchanged from baseline.

\newpage

# G shock — labour-tax financed, NFA@T target vs baseline

Policy: G raised by 2% of Y, financed by labour tax; `balance_condition = terminal_nfa_gdp` pinned to the baseline terminal NFA/Y; Δτ_l = +3.47 pp; converged (to the closure's bracket, gap +0.018 of Y). The larger tax rise consolidates the budget enough to return external assets to baseline at T.

## Main aggregates vs baseline

- **Y**: above baseline, +0.0049 (+1.1%); same income-effect channel as the debt-target case, slightly stronger.
- **K_domestic**: above baseline, +0.018 (+1.1%).
- **L**: above baseline, +0.0026 (+1.1%).
- **C**: below baseline, −0.0086 (−3.1%); the largest consumption drop of the three closures.
- **A**: below baseline, −0.052 (−2.9%).
- **A/Y**: below baseline, −0.16 of Y throughout.
- **NFA/Y**: returns to baseline at `T_balance` (+0.018 of Y, the target) and runs +0.12 of Y above by the end.
- **B/Y**: below baseline, −0.18 at `T_balance` and −0.28 at the end; the extra consolidation lowers debt to offset the lower A/Y and pin NFA/Y.
- **w**, **r**: unchanged (exogenous).

## Government budget constraint vs baseline

- **tax_l**: +2.4 pp of Y (the +3.47 pp rate rise).
- **total_revenue**: +1.8 pp of Y.
- **govt_spending (G)**: +2.0 pp of Y (the shock).
- **primary surplus**: +0.05 pp of Y; runs slightly tighter than baseline to push B/Y down.
- **interest_payments (r·B)**: below baseline, −0.70 pp at `T_balance` and −1.12 pp at the end, on the lower debt stock.
- **pension**: −0.17 pp of Y; higher Y lowers the pension share.
- **public_investment**, **defense_spending**, **other_net_spending**, **bequest_transfers**, **tax_c**, **tax_p**, **tax_k**, **ui**, **gov_health**: ≈ unchanged from baseline.

## Figure cross-check

Consistent. `g_fiscal_decomp.png` plots the primary deficit, the four taxes, UI, and pensions as shares of Y and the remaining lines (G, I_g, defense, other-net, interest) as levels; Y is ≈flat so level and share directions coincide. The debt-financed B/Y explosion and its NFA/Y mirror, the τ_l scenarios' lower A/Y and C, and the NFA@T scenario's NFA/Y returning to baseline at T all match their macro-overview panels (B/Y also in the fan chart); w and r are flat and identical across scenarios in the prices panel; the K_g/Y panel is empty, matching the absent `K_g`.
