# Aligning the government budget with the general government accounts

Written 2026-10-07; decisions taken the same day (§5); implementation done on 2026-10-07/08 (§0). Companion to `EC_ALIGNMENT_PLAN.md` (debt dynamics, retirement age, pension index); this plan supersedes its item 8 (the closure) and implements its item 5 (the unemployment path).

## 0. Handoff (read first)

Status on 2026-10-07: every decision is taken (§5; the shared doc "Budget alignment plan" has the same table, rows 1–18). Where a "Proposal" in §3 and §5 disagree, §5 is authoritative; §6 corrects §3 where it says so. Implementation done on 2026-10-07/08 for every item of §4 except the age range to 100 (§3.17, scoping first): the data builders; the configuration; the code (the firm conditions with τ_y in `firm_conditions.py`; `tax_y` and the transfer from abroad as revenue; education and the lump-sum transfer as spending; the lump sum in both household solvers; the unemployment path as per-cohort income transition matrices; the real sovereign-rate path; the stock-flow adjustment and the terminal ramp in `baseline_closure.py`; the joint (A_tfp, τ_y) pin in `normalize_A_tfp.py --pin-tau-y`); the tests (`test_fiscal_restructure.py`); the recalibration on an A100; and the report. The O line stays in the code at zero. The fiscal-driver fix of §6.5 is confirmed by the I_g rerun in `output/fiscal_2026-10-07_a0/`.

Targets for 2023, % of output, all on the ESA purchases basis net of imputed contributions (§3.10, §5): pensions 14.0; public health 5.3 (total 8.4, κ 0.631); education 2.6 (new line, school-age driver per adult, run's own wage); G 7.0 (own wage bill and intermediate consumption of functions outside defence, health, education); defence 1.66 (flat; code takes a path); I_g 4.0 constant with δ_g 0.046; lump-sum transfer 3.5 (new line, uniform per adult, untaxed); social contributions 11.2; unemployment benefits 0.6 and the floor as now; τ_c, τ_l, τ_k, bequest tax as now. O/Y removed: an output tax on firms τ_y at a constant rate pinned to the 2023 primary balance (about 0.06 after the lines below), plus a transfer from abroad at the EU net flow (§3.16). Unemployment rates by education follow the Ageing Report's aggregate path (§3.15). Age range extended to 100 before the recalibration (§3.17, scoping first).

Order (§4): data builders → config → code (firm conditions with τ_y in one function, `tax_y`, education, lump sum, foreign transfer, defence and τ_y paths, unemployment path, age extension, sfa values, terminal ramp) → tests → recalibration on a GPU instance with the τ_y root-find in the loop (§6.3) → report (§4 step 5, §6.2). Confirm with the user before any long run.


## 1. Purpose

The baseline reproduces the Commission's primary balance in every year through the closure line O/Y ("other net spending"). In 2023 O/Y is −8.9% of output. The general government accounts (Eurostat, ESA 2010) show which revenue and spending the model does not have, which lines it has at the wrong level, and which lines will move with demographics over the transition. This plan restructures the fiscal targets and the tax assumptions so that O/Y contains only revenue and spending without a counterpart in the model, and so that the lines with a demographic path are named.

Three kinds of change, with different payoffs:

- **A. Changes that alter household behaviour or the fiscal wedges**: the pension target, the health target, the tax rates and their bases, transfers to the working-age population. These change the calibrated parameters (ρ_pens, m, ν, β) and the fiscal experiments.
- **B. Changes that alter the path of O/Y over 2023–2070 without touching households**: lines whose share of output moves with demographics (education) or is transitory (EU grants, energy subsidies). Since O/Y is set to the Commission's primary balance, these change what O/Y absorbs, not the primary balance.
- **C. Relabelling**: items that are constant shares of output with no household counterpart (sales, property income, taxes on production and property, subsidies, capital transfers). Naming them changes nothing in the model; it makes the benchmark table legible. These belong in the calibration report as a memo decomposition of O/Y, not in the code.

## 2. The data

Eurostat `gov_10a_main`, `gov_10a_taxag`, `gov_10a_exp` (COFOG by ESA item), Greece, % of GDP, extracted 2026-10-07. The xlsx (`data/2026-09-28_GR_DSA_Spring_Forecast_2026_v1.xlsx`, sheets `Balance ` and `Sheet3`) carries a subset of the same extract (2026-10-06); every overlapping value agrees. COFOG stops in 2024. The Commission's cost-of-ageing items are from the 2024 Ageing Report (projections volume Table I.1.x, `lit-review/data-debt/ageing2024_projections_ip279.txt` line 1177; fiche Table 6).

### 2.1 Revenue, 2023 (total 48.1)

| ESA item | 2019 | 2023 | 2024 | 2025 | Model line | Model 2023 |
|---|---|---|---|---|---|---|
| Taxes on production and imports (D2) | 17.4 | 17.1 | 17.0 | 17.2 | | |
| of which VAT (D211) | 8.3 | 8.8 | 9.0 | 9.5 | τ_c C | 9.1 (all of τ_c C) |
| of which excise and consumption taxes (D214A) | 4.0 | 3.4 | 3.3 | | τ_c C | |
| of which other taxes on products (D212, D214B–I) | 1.9 | 2.0 | 1.9 | | none | |
| of which other taxes on production (D29) | 3.2 | 2.9 | 2.8 | 2.6 | none | |
| of which property (D29A) | 1.7 | 1.3 | 1.2 | | | |
| Current taxes on income and wealth (D5) | 9.1 | 10.1 | 11.2 | 11.1 | | |
| of which household income (D51A) | 5.9 | 5.9 | 6.6 | | τ_l base | 7.1 |
| of which corporate income (D51B) | 1.9 | 2.7 | 3.0 | | τ_k r A | 3.6 |
| of which other income taxes (D51D, D51E) | 0.4 | 0.5 | 0.6 | | none | |
| of which other current taxes (D59; capital 0.7, licences 0.4) | 1.2 | 1.0 | 1.0 | | none | |
| Social contributions (D61) | 14.4 | 13.0 | 13.3 | 13.1 | τ_p w L | 13.0 |
| of which actual (D611 + D613) | 12.0 | 11.2 | 11.7 | | | |
| of which imputed, civil servants (D612) | 2.4 | 1.8 | 1.7 | | | |
| of which pension contributions (D6111 + D6131) | 8.2 | 8.1 | 8.5 | | | |
| Capital taxes (D91) | 0.1 | 0.1 | 0.1 | 0.1 | bequest tax | 2.3 |
| Sales of goods and services (P11 + P12 + P131) | 2.9 | 3.1 | 3.1 | 3.3 | none | |
| Property income (D4) | 0.9 | 1.1 | 1.0 | 0.8 | none | |
| Other current transfers (D7; from the EU 0.3) | 2.2 | 1.2 | 1.3 | 1.2 | none | |
| Capital transfers (D9; from the EU 2.3 in 2023, 3.1 in 2025) | 1.6 | 2.5 | 2.5 | 3.4 | none | |

Consumption taxes in the Commission's implicit-tax-rate sense (VAT, excises, part of the other product taxes) are about 12.7% of GDP in 2023, which is 18.4% of household consumption (69.2% of GDP). The config's τ_c = 18.18% is this rate.

### 2.2 Expenditure, 2023 (total 49.5; primary 46.1)

By ESA item:

| ESA item | 2019 | 2023 | 2024 | 2025 | Model line | Model 2023 |
|---|---|---|---|---|---|---|
| Compensation of employees (D1) | 11.8 | 10.5 | 10.4 | 10.2 | in G, defence, health | |
| Intermediate consumption (P2) | 4.7 | 5.5 | 5.2 | 5.7 | in G, defence, health | |
| Social transfers in kind, purchased (D632; health 2.3) | 2.8 | 3.2 | 2.8 | 2.8 | health (κ m) | |
| Gross fixed capital formation (P51G) | 2.5 | 4.0 | 4.3 | 4.8 | I_g | 3.86 |
| Cash social benefits (D62) | 18.6 | 17.3 | 16.7 | 16.4 | pensions, UI, floor | 16.0 + 0.6 + 0.1 |
| Subsidies (D3; energy 1.5 in 2023, 0.4 in 2019) | 1.3 | 2.0 | 1.4 | 1.3 | none | |
| Other current transfers (D7; EU own resources 0.7) | 1.5 | 1.3 | 1.4 | 1.4 | none | |
| Capital transfers (D9; investment grants 1.7) | 1.4 | 2.3 | 2.3 | 2.5 | none | |
| Interest (D41) | 3.0 | 3.4 | 3.5 | 3.2 | r_B B | 1.7 |

Purchases of goods and services (D1 + P2 + D632 + P51G) by function, 2023:

| Function (COFOG) | D1 | P2 | D632 | P51G | Total | Model line | Model 2023 |
|---|---|---|---|---|---|---|---|
| Defence (GF02) | 1.4 | 0.5 | 0 | 0.4 | 2.3 | defence | 3.0 |
| Health (GF07) | 1.6 | 1.7 | 2.3 | 0.2 | 5.8 | public health | 5.4 |
| Education (GF09) | 2.7 | 0.3 | 0.1 | 0.5 | 3.6 | in G | |
| All other functions | 4.8 | 3.0 | 0.8 | 2.9 | 11.5 | G, I_g | 13.0 + 3.86 |
| Total | 10.5 | 5.5 | 3.2 | 4.0 | 23.2 | | 25.3 |

The file's "public consumption" lines are final consumption expenditure (P3 = D1 + P2 + D632 + consumption of fixed capital 3.3 − sales 3.1), so they carry the depreciation of public capital, which the model has in δ_g K_g, and net out sales. The table above uses the ESA items, which match the model's budget identity (gross investment feeds K_g; depreciation is not a spending line).

Cash social benefits (D62) by function, 2023: old age 12.0, survivors 2.0, sickness and disability 1.4 (of which disability pensions about 0.4), family and children 0.8, unemployment 0.6, social exclusion 0.3, education 0.1. ESSPROS pensions 2023 (old age 11.3, survivors 2.3, disability 0.4): 14.0.

Cost of ageing in the 2024 Ageing Report, Greece, % of GDP:

| Item | 2022 | 2030 | 2040 | 2050 | 2060 | 2070 | Change 2022–70 |
|---|---|---|---|---|---|---|---|
| Pensions (fiche Table 6) | 14.5 | 12.7 | 13.7 | 14.0 | 12.7 | 12.0 | −2.5 |
| Health care | 5.4 | | | | | | +0.6 |
| Long-term care | 0.1 | | | | | | 0.0 |
| Education | 3.4 | | | | | | −0.5 |
| Total | 23.4 | | | | | | −2.4 |

The decade values for health and education are in the projections volume (lines 6524 ff. and the education chapter); not transcribed here. The Commission's primary-balance path after 2027, which the closure reproduces, embeds these four items. The country report (SWD(2026) 208, Annex 2) has the same items from 2025.

### 2.3 Model, 2023 (grid-50 calibration, `code/output/calibration_growth/baseline_paths.npz`)

Revenue 35.0: τ_c C 9.1, τ_l 7.1, τ_p 13.0, τ_k 3.6, bequest tax 2.3. Spending without O 42.0: pensions 16.0, health 5.4, UI 0.6, floor top-ups 0.1, G 13.0, defence 3.0, I_g 3.86. O −8.9. Primary balance 1.95.

## 3. Items

Each item: the data, the model, the proposal, what it changes, what the user must decide.

### 3.1 Pension target and concept (A)

Data: old-age cash benefits 12.0; old age plus survivors 14.0; ESSPROS pensions 14.0; Ageing Report pensions 14.5 in 2022 (old age, survivors, disability pensions). The pension index path (`build_pension_index_GR.py`) is built from fiche Table 6 (14.5 → 12.0) and Table 10, so it is in the Ageing Report concept.

Model: `pensions_over_Y` 0.16, undocumented in `data_inventory.md`; `CALIBRATION_FIX_CHECKLIST.md` (2026-05-18) records 0.120 as the data value and the change was never made.

Proposal: `pensions_over_Y` = 0.140 (2023, Ageing Report concept: old age 11.3, survivors 2.3, disability 0.4). Survivors' pensions are paid to the retired and derived from the deceased's pension; the model has no couples, so they are part of the benefit of the retired. ρ_pens falls from 0.235 to about 0.21 (less than proportionally because the indexed floor binds for part of the distribution). The target becomes consistent with the index path in level and concept.

Check after recalibration: the model's pension path by decade against Table 6. The 2026-10-06 run gives 16.0 (2023), 14.3 (2030), 15.8 (2050), 10.2 (2070) against 14.5, 12.7, 14.0, 12.0: the 2070 value undershoots by 1.8 pp even after scaling by the level, which the index alone should not produce.

Decision: 0.140 (recommended) or 0.120 with survivors in item 3.2.

### 3.2 Cash benefits outside pensions and unemployment (A or C)

Data 2023: sickness and disability 1.0 (after moving disability pensions to 3.1), family and children 0.8, social exclusion 0.3, education 0.1; in-kind social protection outside health 0.8 (housing 0.2, social exclusion 0.4, family 0.1). Total 3.0, of which 2.2 cash to the working-age population.

Model: the means-tested floor pays 0.1.

Proposal: book the 3.0 as a named constant share of output in the memo decomposition (C). A household-side version (a lump-sum transfer to ages 25–64, taxed under τ_l, with the working-age population as its driver) would add a term to the household budget constraint and the budget identity; the income effect on labour supply and saving at 2% of output is small, and the Commission's projection holds these items constant in GDP, so the path adds nothing against the benchmark. Not recommended now.

Decision (2026-10-07, §5): a lump-sum transfer line to households. A uniform amount per adult, the same at every age and education level, untaxed, received every period. Target: cash social benefits (D62, 17.3) less the pension line (14.0) and unemployment benefits (0.6), plus social transfers in kind purchased in social protection (D632, 0.8): 3.5% of GDP in 2023, so that pensions, unemployment benefits and the lump sum add up to the accounts' benefits to households (the 0.4 by which the ESSPROS pension figure falls short of the COFOG one sits in the lump sum). Constant share of output along the transition, uniform per adult of the model population (`lump_sum_over_Y`, with a `(T,)` path option like defence). The calibration report states this definition next to the number. The output tax carries 3.5 pp of output more. The in-kind 0.8 could instead stay in G as resource use; it is a transfer by the rule of §3.5.

### 3.3 Public health: level and concept (A)

Data: purchases of care for households (D1 + P2 + D632 in GF07) 5.6 in 2023 (5.9 in 2022); COFOG total 5.9 (adds investment 0.2, R&D 0.1, public health services 0.1); Ageing Report 5.4 in 2022; country report 5.2 in 2025. Out-of-pocket health consumption 3.1 (`DATA_GR.xlsx`).

Model: `health_gov_over_Y` 0.054 (equals the Ageing Report 2022 value), κ 0.662, total 8.2, out-of-pocket 2.8.

Proposal: `health_gov_over_Y` = 0.056 (2023 purchases, same ESA basis as the other spending lines) and `health_total_over_Y` = 0.087 (5.6 + 3.1), which gives κ = 0.644 if κ is to reproduce the split; or keep κ 0.662 and accept out-of-pocket 2.9. The Ageing Report's 5.4 is the alternative if the benchmark over the transition is the Ageing Report path (+0.6 pp by 2070) rather than the accounts. Either way the model's health path by decade should be reported against the Ageing Report's.

Decision: 0.056 (ESA purchases) or 0.054 (Ageing Report); whether κ is re-derived. Taken 2026-10-07 (§5): purchases net of imputed contributions, 5.3; total 8.4 with out-of-pocket 3.1; κ = 5.3/8.4 = 0.631.

### 3.4 Education as a separate line with a demographic driver (B)

Data: purchases 3.6 in 2023 (D1 2.7, P2 0.3, D632 0.1, GFCF 0.5), COFOG total 4.0; Ageing Report 3.4 in 2022, falling 0.5 pp by 2070 with the school-age population.

Model: inside G (13.0). The config's G is constant in output, so the fall is absorbed by O/Y.

Proposal: a spending line E_t = (E/Y)_2023 · Y_2023 · (N^{5–24}_t / N^{5–24}_2023) · (w_t / w_2023) in levels, i.e. per-student cost growing with wages and the number of students from EUROPOP2023 (`proj_23np`, ages 5–24; `build_europop_GR.py` fetches the dataset but keeps ages 25–84, so the builder is extended to save the school-age population path). (E/Y)_2023 = 0.031 (D1 + P2 + D632), with education investment 0.5 staying in I_g. No household side: students are outside the model population. In the code the line enters `compute_government_budget` like `defense_spending_path` (a new term in `total_spending`, which is a change to the budget identity in `olg_transition.py` and needs approval). The existing `schooling_years`, `child_cost_profile` and `education_subsidy_rate` features model the agent's own schooling cost and are off; they are not what is needed here.

Effect: O/Y no longer absorbs the −0.5 pp; the model's cost of ageing has three of the Commission's four items (long-term care, 0.1, is left out).

Decision: approve the new line.

### 3.5 G and defence levels (A for the level; the path is a decision)

Data 2023: purchases outside defence, health and education 7.8 (D1 4.8 + P2 3.0) plus in-kind transfers 0.8 = 8.6; defence purchases 1.9 (D1 1.4, P2 0.5) plus investment 0.4; COFOG defence total 2.2 (2023 and 2024). NATO's 3.0% definition includes military pensions and a different equipment basis.

Model: `G_over_Y` 0.13, `defense_over_Y` 0.03 (NATO/SIPRI), I_g includes defence and health investment (0.46 and 0.21 of the 3.86).

Proposal: `G_over_Y` = 0.086, `defense_over_Y` = 0.019 for 2023, with investment of all functions staying in I_g. For the path: the Commission's primary-balance projection embeds its own defence assumption (the country report notes defence rising from 2.2% in 2024); a defence path above 1.9 would be a scenario, not the baseline, unless the user wants the baseline to carry the announced increase.

Decision: the two levels; whether defence has a path. Taken 2026-10-07 (§5): G is the government's own wage bill and intermediate consumption, net of imputed contributions, 7.0; in-kind social protection 0.8 (D632, COFOG 10) is a transfer in kind to households and goes to the household-transfer memo; defence 1.66.

### 3.6 Public investment and EU financing (B)

Data: GFCF 4.0 (2023), 4.3 (2024), 4.8 (2025); capital transfers received from the EU 2.3, 2.2, 3.1 (RRF and structural funds), investment grants paid 1.7, 1.6, 1.8. The RRF ends in 2026; cohesion funds continue at a lower level.

Model: `I_g_over_Y` 0.03862 constant; K_g accumulates from it with η_g 0.05, so the path matters for output.

Proposal: I_g/Y 0.040 (2023), the accounts' 0.043 and 0.048 for 2024–25, then a constant share of output. The DSM 2025 baseline is a no-fiscal-policy-change scenario that fixes the structural primary balance before the cost of ageing at its last forecast-year level (`dsm2025_ip332.txt` lines 254–276, Table 2.5); it does not project investment separately, so the post-2027 share is a model choice, and a constant share is the reading of unchanged policy. The level of that constant (the 2025 value with RRF, or the 2023 value) is the decision. EU grants and the grant-financed investment are close to neutral for the primary balance, so they stay in the memo decomposition (C) with their transitory profile noted.

Decision: the 2024–25 values and the post-2027 rule.

### 3.7 Consumption tax: rate versus base (A for the rate; the base is structural)

Data: consumption taxes 12.7 on household consumption 69.2 (18.4%); the Commission's implicit rate is the config's 18.18%. The rest of D2, 4.4 (other taxes on production 2.9, of which property 1.3 and pollution 0.9; other taxes on products 1.5), is not a tax on household consumption.

Model: τ_c C = 9.1 because C/Y is 50.1: private investment 21.8 against 12.0 (K/Y = α/(r + δ) = 3.67), trade balance 0 against −4.8, public purchases 25.3 against 23.2.

Proposal: keep τ_c = 0.1818. The rate sets the consumption-leisure wedge, which is what the experiments use; the base gap (3.6 pp of revenue) is a national-accounts mismatch that targets cannot fix without changing α, δ or r, or giving the small open economy a trade deficit. Report the gap as a memo item. Book the 4.4 of non-consumption D2 as a memo item (C). Sensitivity, if wanted: τ_c calibrated to 12.7% of output on the model's base (τ_c ≈ 0.25), which raises the wedge.

Decision: none, unless the sensitivity is wanted.

### 3.8 Labour income tax (A)

Data: household income taxes 5.9 (D51A), 6.4 with the other income taxes (D51D, D51E); the file's "PIT" 7.4 adds other current taxes (D59, 1.0: taxes on capital 0.7, licences 0.4), which are not income taxes. The Greek personal income tax is progressive (rates 9–44% with a tax credit).

Model: flat τ_l = 0.10 (`data_inventory.md`: the Commission's labour ITR 40.58% less the contribution part) on net wages, pensions and UI, base 70.6% of output, revenue 7.1. After item 3.1 the base falls by 2.0 and revenue to about 6.9.

Proposal, minimal: keep τ_l = 0.10; report the +0.5 against 6.4 in the benchmark table. Alternative: the progressive schedule already in the solver (`tax_progressive`, `_hsv_tax(taxable_income, tax_kappa_hsv, tax_eta)`, off in the config), with the level parameter set to reproduce 6.4% of output and the progressivity parameter from an external estimate for Greece (EUROMOD or OECD tax-benefit tables; no source in the repo). This changes the marginal wedge, the response to the τ_l-financed experiments, and what a "Δτ_l" means (a shift of the level parameter). It is a separate project: solver tests, recalibration, and a rewrite of the experiment definitions.

Decision: minimal now (recommended); progressive schedule as a later item.

### 3.9 Capital taxes and property taxes (A if a wealth tax is added; C otherwise)

Data: corporate income tax 2.7; household taxes on capital income inside D51A (dividends 5%, interest 15%, rents 15–45%; no split published); current taxes on capital 0.7 (D59A) and recurrent property tax 1.3 (D29A, ENFIA). Broad capital-related taxes at least 4.7.

Model: τ_k = 0.2236 on r A, revenue 3.6 (A/Y about 4).

Proposal: keep τ_k. Book property taxes (2.0) as a memo item (C); the model has no housing. A wealth tax on A at 0.5% would raise 2.0 but cuts the after-tax return from 3.1% to 2.6% and forces a β recalibration; not recommended without a housing block.

Decision: none.

### 3.10 Social contributions: concept (documentation)

Data: 13.0 includes imputed contributions of civil servants 1.8 (unfunded scheme; the mirror entry is in compensation of employees, D12, inside G); actual contributions 11.2, of which pension contributions 8.1. Compensation of employees is 35.0% of GDP and the self-employed earn mixed income outside it, while the model's labour share is 0.67; the effective τ_p of 0.194 on the model's base is accordingly below the statutory rate of about 0.36.

Proposal: keep `tax_p_over_Y` = 0.13 (the imputed part is a closed circuit: it is matched by D12 inside the purchases that G reproduces) and state the concept in `data_inventory.md`. The alternative, 0.112 with G lowered by 1.8, is equivalent for the primary balance and changes the labour wedge.

Decision: 0.112 with the purchase lines net of imputed contributions, taken 2026-10-07 (§5).

### 3.11 Bequest tax (documentation)

Revenue 2.3 against capital taxes of 0.1 in the data. It is the model's treatment of accidental bequests (confiscated since 2026-10-02), not a data line. Reported in the benchmark table as "no data counterpart". A lump-sum rebate would move 2.3 into O/Y.

### 3.12 Memo decomposition of O/Y after items 3.1 and 3.3–3.6 (C)

With pensions 14.0, health 5.3, G 7.0, education 2.6, defence 1.66, I_g 4.0, a lump-sum transfer 3.5, social contributions 11.2 (decisions of 2026-10-07: purchase lines net of imputed contributions, in-kind social protection outside G, the lump-sum line of §3.2) and the other lines unchanged, the model's revenue is about 33.1 (τ_l 6.9) and spending without O about 38.8, so O/Y ≈ −7.7 in 2023 (from −8.9); the transfer from abroad of §6.4 item 3 takes 1.9 off it, so the output tax carries about 5.8% of output. Its data decomposition, % of GDP:

| Component | Contribution |
|---|---|
| Revenue without a model line: sales 3.1, property income 1.1, other current transfers 1.2, capital transfers 2.5, capital taxes 0.1, other taxes on production 2.9, other taxes on products 1.5, other current taxes 1.0, imputed social contributions 1.8 | −15.2 |
| Primary expenditure without a model line: subsidies 2.0, other current transfers 1.3, capital transfers 2.3, inventories −0.1, imputed contributions inside public wages 1.8 | +7.3 |
| Consumption tax base (model 9.1 against 12.7; C/Y 50.1 against 69.2) | −3.6 |
| Bequest tax (no data counterpart) | +2.3 |
| Labour income tax above household income taxes (6.9 against 6.4) | +0.5 |
| Capital income tax above corporate income tax (3.6 against 2.7) | +0.9 |
| Floor top-ups (0.1, inside other cash benefits in the data) | −0.1 |
| Total | −7.7 (lines sum to −7.9; rounding of one-decimal items) |

The table is the identity O = (Rev_model − Rev_data) − (Exp_model − Exp_data) with the primary balance common to both sides; it gives −7.7. (The four model-specific lines carried the opposite signs until 2026-10-07; they nearly cancel, which hid the error.) The pension line mixes sources: 14.0 is ESSPROS, "other cash benefits 2.1" is COFOG less the 0.4 of disability pensions; on COFOG alone old age 12.0 + survivors 2.0 + disability pensions 0.4 = 14.4, so 0.4 of D62 is in neither line. The report table should use one source. The benchmark section of the calibration report should carry this table for 2023 and the same lines for 2024–25 from the accounts.

### 3.13 Transition benchmark (B)

Over 2024–25 the data primary balance rises 2.9 pp through income taxes (+1.0), other revenue (+0.7), lower purchases, pensions and other expenditure; the model's own lines move 0.6 and O/Y supplies the rest (−8.9 → −11.2). After the restructure the same comparison should be made line by line (the accounts run to 2025, COFOG to 2024). Beyond 2027 the benchmark is the Ageing Report by decade: pensions (Table 6), health, education, each against the model's line, and the implied O/Y path. The target for the restructure is a flatter O/Y path: the 2030–2060 values are now −7.2, −6.2, −6.0, −5.4, and the part of that drift due to education (−0.5 pp by 2070) and the pension level should disappear.

### 3.14 Modelling O/Y with incidence (A)

O/Y is the net revenue the government needs from lines the model does not have. In the code it is a number in the budget identity that no household or firm pays. In a small open economy that makes it a transfer from abroad: it finances spending without taking resources from residents, so household consumption and net foreign assets are higher than with domestic incidence (about 5% of output a year after items 3.1–3.6, 9% now). In the data only the EU transfers (net 1.9% of GDP in 2023) come from abroad; the rest is paid by residents (taxes on production and products not on consumption 4.4, sales 3.1, other 2.0) and spent on residents (cash and in-kind benefits 2.9, subsidies and capital transfers to firms 4.3, other 1.3). After the restructure O/Y is about −5.0 in 2023 and moves over time: the current path (−8.9, −11.3, −11.2, −7.2, −6.2, −6.0, −5.4 for 2023, 2024, 2025, 2030, 2040, 2050, 2060) shifted by about +3.9.

Production tax. A tax τ_y on gross output paid by firms has a data counterpart (4.4% of GDP) and a base that follows output, which is the convention O already has (`_spend` multiplies the ratio path by Y_t, so O scales with output in the experiments). What the tax adds is incidence. With r exogenous the firm's conditions become (1 − τ_y) α Y/K = r + δ and (1 − τ_y)(1 − α) Y/L = w, so K/Y falls by the factor (1 − τ_y) and the wage by (1 − τ_y)^(1/(1−α)). At τ_y = 0.044: K/Y 3.67 → 3.51, private investment 21.8 → 20.8% of output, w −6.5%. The baseline absorbs the level in the A_tfp normalisation and the recalibration; the wedge is constant and small; in the experiments the revenue rises with output. The tax cannot carry the time variation: a rate that moves with the closure (about 0.05 in 2023, 0.074 in 2024–25, 0.033 in 2030, 0.015 in 2060) moves wages by about ±3.5% and the capital stock with it, households foresee it, and the closure's dynamics would enter every experiment.

Lump-sum residual. The time-varying part is a uniform transfer per adult T_t (negative = tax) in the household budget constraint, set so that the primary balance equals the Commission's. Non-distorting at the margin; its income effect feeds back on consumption, the consumption tax and the floor, so the closure becomes a fixed point over the transition (2–3 transition solves; the experiments hold the path fixed, as they hold O now). After the production tax at 0.044 the residual is roughly −0.6 in 2023, −3.0 in 2024–25, +1.1 in 2030, +2.9 in 2060 (% of output). The 2024–25 step is the Commission's revenue measures without a model counterpart; the drift to 2060 is the gap between the model's own structural primary balance and the Commission's (pension path against Table 6, drift of the tax bases). The transfer line makes the gap visible; it does not remove it. A per-adult lump sum is regressive and part of a tax on the poorest returns as floor top-ups; a version proportional to income is a flat tax.

Household side: residents pay the 4.4 through lower wages and receive the residual, so consumption falls by about 3 pp of output relative to the current treatment (the consumption-tax base gap widens by that much) and net foreign assets no longer receive the inflow.

Validity in the small open economy. A source-based tax on domestic production is well defined with r exogenous, but its incidence is the textbook one: capital is perfectly elastic and labour immobile, so the whole burden is shifted to labour and the tax is dominated by a direct tax on labour (Diamond–Mirrlees production efficiency; Gordon 1986). Holding hours fixed, at τ_y = 0.044 the domestic capital stock falls 6.5% (0.24 of output moves into NFA), GDP 2.2%, the take-home wage 6.5%, labour income 4.35% of output; revenue is 4.30% of output, so the national-income loss on the capital margin is 0.05% of output. What it distorts: (1) the location of capital, the only new margin: K/Y 3.67 → 3.51, private investment −1 pp of output, GDP −2.2%, NFA +0.24 of output, the level absorbed by the A_tfp normalisation; (2) labour supply through the wage, like a payroll tax: the wedge 1 − (1−τ_y)(1−τ_p)(1−τ_l) rises from 27.5% to 30.7%, plus a 2.1% fall in the marginal product of labour from the lower capital stock; ν re-absorbs the level in the baseline, but the marginal excess burden of τ_l in the financed experiments is higher; (3) the public-capital channel: the private capital response to K_g is scaled by (1−τ_y)^(1/(1−α)) = 0.935, so the I_g multiplier falls by a few percent and the G multiplier is unchanged. It does not touch the saving margin: households earn r on all assets whatever τ_y is. The taxes the 4.4 stands for (land and buildings 1.3, pollution 0.9, licences 0.5, stamp, transaction, insurance and car taxes 1.5) fall on immobile or transaction bases, not on mobile capital, so the output tax adds a capital-location distortion that the actual taxes do not have.

Recommendation after discussion. A levy conditional on age and education is non-distorting only because education is exogenous in the model, and it has no policy counterpart; dropped. The output tax is the coherent instrument for the permanent part. It is what the national accounts call taxes on production, a wedge between GDP at market prices and factor incomes, which the model lacks; it is paid by firms, so its incidence falls on labour income in proportion without any rule across households; and its two effects go toward the data: K/Y falls from 3.67 to 3.51 (the model's private investment is 21.8% of output against 12.0 in the data) and the labour wedge rises from 27.5% to 30.7% (the Commission's implicit rate on labour is 40.6%, on a narrower base). The deadweight loss on the capital margin is 0.05% of output. For the time-varying residual the choice is between (i) a time-varying rate τ_y,t and (ii) a uniform lump sum per adult. With (i), K/L adjusts within the year to each change in the rate because capital is frictionless: the 2024 step (0.05 → 0.074) cuts the capital stock 3.8%, so investment falls from about 21% to about 8% of output in that year and runs about 4 pp a year above normal over 2026–2030 as the rate comes down; GDP moves ±1–2% and wages ±4% with the closure. The path is the same in the baseline and the experiments, so it differences out to first order. With (ii) nothing in K or w moves; the levy is about ±3% of output per adult (±870 EUR a year at 2023 GDP per adult), regressive, with the bottom caught by the floor. Both need the closure as a fixed point. Recommended: τ_y fixed at 0.044 plus the uniform lump-sum residual (ii); τ_y,t for everything (i) is acceptable if the investment swings are accepted.

| Option | Incidence | Carries the time variation | Distortion | Cost | Verdict |
|---|---|---|---|---|---|
| A. Keep O as a number (current) | none: an inflow from abroad | yes | none | none | incoherent: consumption and NFA overstated |
| B. Production tax at a fixed 0.044 plus a uniform lump-sum residual | firms (w, K) and households | lump-sum | constant wedge on K and w | firm conditions, one household term, closure fixed point, recalibration | recommended: data counterpart, incidence on labour income without a rule, K/Y and the labour wedge move toward the data |
| C. Production tax with a time-varying rate as the closure | firms | yes | K and investment swing with the closure (2024: investment 21% → 8% of output) | firm conditions, fixed point | acceptable; same path in baseline and experiments |
| D. Lump-sum residual only, uniform per adult | households | yes | none | one household term, fixed point | regressive; the by-age-and-type variant is dropped (not lump-sum in substance) |
| E. Fold into the existing rates (τ_c 0.25, τ_k broadened) | households | partly | raises the wedges | recalibration | not recommended: the rates are the Commission's effective rates |
| F. Add an EU transfer line from abroad (net 1.9, about 1.0 after 2027) | abroad | own path | none | a data path | optional refinement of D: NFA right by 1–2% of output a year |

Decision (2026-10-07): the output tax τ_y on firms carries all of O (option C with a constant default). The code takes a rate path τ_y,t; the default is a constant rate. Under a constant rate the primary balance is the model's own from 2024 on and the Commission's path is a comparison, not a constraint: τ_y is set in the base year to the 2023 primary balance (about 0.05 after items 3.1–3.6) and the debt path follows from the model's primary balance, r_B and the stock-flow adjustment. Under the time-varying path the closure of 2026-10-05 applies (the Commission's primary balance 2024–2060, then the debt-stabilising balance), computed as a fixed point on τ_y,t. The baseline for the report and the experiments is the constant rate (decided 2026-10-07, §5); the Commission's primary balance and debt become comparison series in the report, and the stock-flow adjustment from the DSA file stays in the debt recursion.

Code: τ_y,t enters `K_over_L` and `w_path` in `olg_transition.py` (firm conditions) and the budget as revenue τ_y,t Y_t; O is removed from the budget identity; `baseline_closure.py` either calibrates the constant rate in the base year or iterates on the path. No household term. Equation changes, need approval.

### 3.15 Unemployment path: labour input from the Ageing Report (A)

Data: the Ageing Report's employment rate of 20–64-year-olds rises from 66.1% in 2022 to 74.7% in 2070 (country fiche, Table 3). Three parts: the share of the labour force at work rises from 87.6% to 93.5% (unemployment 12.4% in 2022, 9.9% in 2030, 8.5% in 2040, 6.6% in 2050, 6.5% in 2060 and 2070), about +4.5 pp of the employment rate; participation of 55–64 (57.4% → 78.2%) and 65–74 (9.3% → 24.3%), the effective retirement age 63.8 → 67.9, about +4 pp; prime-age participation nearly flat. Productivity per worker is the Report's in both, g = 1.7%.

Model: the retirement-age path is in (2026-10-06); participation is flat by construction; the unemployment rates by education are fixed at their 2023 values (0.123, 0.116, 0.077 for ages 25–64). Output growth over 2026–60 is 0.65% against the projection's 1.08% (calibration report, "Why the debt paths differ"); the missing part is the unemployment decline.

Proposal: each education group's unemployment rate follows the Report's aggregate path in proportion, u_{e,t} = u_{e,2023} · u^{AR}_t / u^{AR}_{2023}, with the fiche's decade values interpolated linearly by year (2023 by interpolation between 2022 and 2030) and held at the 2070 value after. In the model the unemployment rate is the stationary rate of a two-state chain with a fixed job-finding rate f and a separation rate s = u/(1−u) · f, so the path is a separation rate by calendar year and the income transition matrix varies by year; each cohort faces the sequence by age (year = entry year + age), the base-year cross-section uses the 2023 matrix, and before 2023 the rate is held at its 2023 value. Config: `edu_params[*].unemployment_rate` stays the 2023 level; a new `labor.unemployment_index_path` by calendar year carries the Report's ratio u^{AR}_t / u^{AR}_{2023}, one index for all groups.

Effect: employment of the model population rises about 4.5% of the labour force by 2050, adding about 0.17 pp a year to output growth over 2023–50 (log(93.5/87.6) over 38 years), roughly 40% of the growth gap; the bases of the labour-income tax and of contributions rise with it and unemployment benefits fall (0.6% of output toward about 0.3), so the model's own primary balance improves by roughly 0.7–1.0 pp of output by 2050 relative to a flat path, adding to the balance gap of §6.2. In 2023 nothing changes: the pre-2023 rates are held at the 2023 value, so the simulated 2023 cross-section has the same unemployment, tax bases and benefits, and the output-tax pin is unaffected; the bases drift along the transition. The path enters the calibration only through anticipation: the base-year cross-section solves each 2023 cohort over its lifecycle and already carries the cohort's survival, retirement age and pension-index path; carried the same way, the known decline in unemployment risk lowers precautionary wealth a little in 2023, so the wealth moment and hence β and ν move slightly. Carrying it keeps the transition's t=0 equal to the calibration cross-section (the preflight check); leaving it out of the cross-section frees the ordering at the cost of that discrepancy. How much of the remaining growth gap the retirement path already delivers, and how much is hours per worker and capital deepening, is read off the run.

Decision (2026-10-07): add it; carried into the base-year cross-section like the pension index (recommended), hence before the recalibration (§5).

### 3.16 Transfer from abroad: the EU net flow (A)

Data: Greece's net receipts from the EU budget (payments received less the own-resources contribution of 0.7% of GDP) plus RRF grants: about 1.9% of GDP in 2023 (the audit's figure, to be rebuilt from the source below). Source: the Commission's EU budget financial reports (net operating budgetary balance by member state, % of GNI → % of GDP with Eurostat nominal GDP) for 2014–2024, and the RRF grant disbursements to Greece by year (Commission RRF scoreboard) for 2021–2026. Path: data to 2025; 2026 the last RRF grant year; from 2027 constant at the 2014–2020 average of the net operating balance (the audit's "about 1.0", to be verified from the reports); held after.

Model: a revenue line `foreign_transfer_t = foreign_transfer_over_Y_t · Y_t` that no resident pays, entering the budget identity and NFA accumulation exactly as O does today (a flow from abroad). Implementation: the existing O line is kept under this name with the data path, with the sign convention of a receipt (`other_net_spending` and `other_net_over_Y` renamed `foreign_transfer`/`foreign_transfer_over_Y`, sign flipped), so that the readers listed in §6.3 are renamed rather than removed and the closure logic of `baseline_closure.py` is deleted. The output tax then carries the residual net of this flow (about 5.8% of output in 2023, §3.12).

Decision (2026-10-07, §6.4 item 3, doc row 10): add it.

### 3.17 Population aged 85 and over: age range to 100 (A)

Decision (2026-10-07, doc row 14): extend the model's age range from 25–84 to 25–100 before the recalibration, since the pension and health paths and the wealth target change with it (the pension path scaled to 14.0 gives 8.9% of output in 2070 against the fiche's 12.0; §6.2).

Done 2026-10-08: ages 25–99, T = 75. `build_europop_GR.py`, `build_survival_GR.py` and `build_demography_GR.py` carry 75 ages; the historical life table ends in an open group at 85, so ages 85–99 in every historical year take the EUROPOP2023 mortality assumption of 2022, and survival before 1961 (the cohorts over 86 in 2023) takes the 1961 schedule; neither enters a solution, since a cohort's survival at ages already passed is never evaluated and the 2023 cross-section is reproduced exactly. The health cost profile `m_age_profile` is rebuilt from the Ageing Report's EU14 groups placed at their midpoints (27, 32, …, 97), linear between them, flat outside, normalised to a population-weighted mean of one over 25–99; the earlier 60-entry profile had placed each group one bin too old. `wage_age_profile` is one over the added ages. `education_shares` take the 2021 census groups 85–89, 90–94 and 95–99 by cohort (0.3141, 0.3990, 0.2869). The two floors are per living person aged 25–99 (`pension_min_floor` 0.1751, `transfer_floor` 0.0846). The school-age index's denominator and the retirement table's first entry cohort (1924) follow. The calibration targets are unchanged. The same rebuild smooths the entering-cohort series with a centred five-year moving average in logs over the cohorts entering 1949–2100, after which the 25–99 total is within 0.27% of EUROPOP2023 in every year to 2100; the measured 2023 cross-section is kept in the file as `cross_section_measured`.

Scoping, to be done first and written here: the demography builder (`build_europop_GR.py` keeps ages 25–84; EUROPOP has single ages to 100+; `data/demography_GR.npz` fields `px`, `entrants`, `pop`, `pop_level` by age); survival probabilities to 100 (`demo_mlexpec`/life tables; the health-state survival `survival_probs` by health); the health process and medical cost profile a(j) beyond 84 (source and extrapolation); the base-year cross-section and education shares for 85–100 (2021 census cohorts, as for 75–84); the lifecycle grid T from 60 to 76 periods in both solvers, the JAX batch shapes and the runtime (+27% per solve); the cohort retirement table and pension index by cohort for the older cohorts; the calibration targets (A/Y with the 85+ wealth, pensions 14.0 unchanged as a total, health 5.3 unchanged as a total); tests on the demography identities. Not to be started before the scoping is approved.

## 4. Implementation order

1. Data builders. `build_gov_accounts_GR.py` → `data/gov_accounts_GR.json`: the ESA items and COFOG purchases of section 2 for 2019–2025, fetched from the Eurostat API like `build_europop_GR.py`, with the model mapping as a field per line. The xlsx is the frozen copy. Documented in `data_inventory.md`.
2. Config: `pensions_over_Y` 0.140, `health_gov_over_Y` 0.053 (`health_total_over_Y` 0.084, κ 0.631), `G_over_Y` 0.070, `defense_over_Y` 0.0166, `I_g_over_Y` 0.040 (constant, δ_g 0.046), new `education_over_Y` 0.026 with the school-age population path, new `lump_sum_over_Y` 0.035, `tax_p_over_Y` 0.112 (values net of imputed contributions, §5).
3. Code: the foreign transfer line (§3.16: O renamed with the data path); the sfa values and the terminal ramp (§5); the unemployment path (§3.15: a separation rate by calendar year in the income process, both backends, the base-year cross-section on the 2023 matrix); the education line and the lump-sum transfer in `compute_government_budget` and `compute_government_budget_path` (new terms in the budget identity; the lump sum also enters the household period budget constraint in both backends; approved 2026-10-07); the output tax τ_y,t in the firm conditions (`K_over_L`, `w_path`) and in the budget as revenue, O removed (item 3.14), with a scalar or `(T,)` rate in the config and the closure either calibrating the scalar in the base year or iterating on the path; a `(T,)` option for `defense_over_Y` (already supported by `_spend`); the extended demography builder; tests for the new lines and for the budget identity.
4. Recalibration (θ: ν, β, ρ_pens, m_good, ρ_ui; then A_tfp; then the closure) on a rented A100: about one hour per round at the 2026-10-06 timing.
5. Report: the benchmark table of section 3.12 for 2023 and 2024–25, the cost-of-ageing table by decade, the output-tax rate and what it carries; a sentence per line stating the data definition it is calibrated to, the lump-sum transfer in particular (§3.2).
6. Later items, each a separate decision: progressive income tax (3.8), household-side transfers (3.2), defence path (3.5), investment share and trade balance (3.7).

## 5. Decisions

Taken by the user on 2026-10-07 (in the shared doc "Budget alignment plan"):

- Pension target 0.140 (Ageing Report concept). Accepted.
- Health 0.056 (ESA purchases, 2023). Accepted; after the imputed-contribution decision below the line is 0.053, `health_total_over_Y` 0.084 (out-of-pocket 3.1 from the data) and κ 0.631.
- Education line at 0.031 with the school-age population driver. Accepted.
- G 0.086 and defence 0.019, defence flat in the baseline. Accepted, with the code able to take a time-varying defence path (a `(T,)` ratio array, as `_spend` already allows) so that the announced increase can be run as a scenario.
- I_g 0.040 (2023), 0.043 and 0.048 (2024–25), then constant at 0.040. Accepted.
- Later items (progressive income tax, household-side transfers, investment share and trade balance) deferred. Accepted.

- O/Y replaced by an output tax on firms carrying all of it; the code takes a rate path, the default is a constant rate. Decided 2026-10-07.

- Baseline: the constant rate. τ_y is set in 2023 to the outturn primary balance; from 2024 the primary balance and the debt path are the model's own and the Commission's projection is a comparison. This replaces the closure rule of 2026-10-05 (`EC_ALIGNMENT_PLAN.md` item 8). The time-varying path τ_y,t that reproduces the Commission's primary balance stays available as an option. Decided 2026-10-07.

Taken by the user on 2026-10-07 after the audit (§6.4; doc rows 8–16):

- Stock-flow adjustment 2024–25: fixed values from the data: sfa_t = d_t − d_{t−1}(1+i_t)/(1+γ_t) + pb_t with the data's debt ratios, implicit nominal rate i_t (interest/debt_{t−1}), nominal growth γ_t and primary balance pb_t (Eurostat `gov_10dd_edpt1`, `gov_10a_main`, `nama_10_gdp`); for 2025 this is the DSA file's own sfa row (6.24% of GDP), for 2024 it is computed. The two numbers go into the config (`fiscal.sfa_2024`, `fiscal.sfa_2025`) and replace the residuals at `baseline_closure.py` 107 and 112; the model's 2024–25 debt then differs from the data by its primary-balance gap (about 2.3 pp a year, §6.2).
- Terminal rule: constant τ_y through 2060, then a smooth path rather than a step: a linear ramp over 2061–2070 to the rate that holds the debt ratio at its 2070 value; the ramp length is a parameter (`fiscal.tau_y_ramp_years`, 10). The terminal rate τ_y^T makes the debt ratio in 2080 equal to its 2070 value, d_2080 = d_2070 (sfa is zero after 2060); the one-year condition pb_2070 = d_2070 · ((1+r_B)/Γ_2070 − 1) is reported but not solved, since at a ratio above one it moves with the growth rate of a single year. The ratio depends on the ramp, so the rate is a fixed point in one scalar, iterated together with the lump-sum path, each iteration a full transition solve since τ_y moves the wage; τ_y is held at τ_y^T after 2070 and `eval_fiscal_results.py` checks the change of the debt ratio over 2070–2080 (`terminal_debt_window`, WARN tier). The time-varying path τ_y,t that reproduces the Commission's primary balance over 2024–2060 is `fiscal.tau_y_mode: "projection"` (`baseline_closure.solve_baseline`, `match_projection`); it is not run for the report.
- EU transfers: a transfer line from abroad with a path (option F) beside τ_y; definition and path in §3.16.
- I_g: constant 0.040 as a stationary level, δ_g re-derived to 0.046; the 2024–25 bump of the earlier decision is dropped.
- Population aged 85 and over: extend the model's age range to 100 before the next recalibration. Scoping (survival data, health process, base-year cross-section beyond 84, solver grid, runtime) is a separate step and plan section.
- Equation changes for the output tax (firm conditions in one function with (1−τ_y), `tax_y` and education lines in the budget identity, O removed, the closure pin replaced by a root-find on (A_tfp, τ_y) inside the calibration loop): approved.
- Fiscal driver wage-path defect (§6.5): fix now, add the A[0] check to the evaluator, rerun the I_g experiment before the restructure.

- τ_p concept: 0.112 (actual contributions) with the purchase lines net of imputed contributions. G is further restricted to the government's own wage bill and intermediate consumption of the functions outside defence, health and education; the 0.8 of in-kind social protection (D632, COFOG 10) is a transfer in kind to households and joins the household-transfer memo. Targets: G 0.070, defence 0.0166, health 0.053, education 0.026; O/Y memo ≈ −4.2 (§3.12).

- Lump-sum transfer to households (doc row 17): added, uniform per adult, untaxed, 3.5% of GDP in 2023 by the definition of §3.2, constant share of output with a path option; the calibration report states the definition. Supersedes the deferral of household-side transfers in row 6 for this item.

- Education line driver: the run's own wage (the line responds inside the public-investment experiment, like the other purchase lines priced at the run's own values).

- Unemployment path (§3.15, doc row 18): added, each education group's rate scaled by the Ageing Report's aggregate path (12.4% in 2022 to 6.5% from 2060), implemented as a separation rate by calendar year, before the recalibration.

Taken on 2026-10-07 (evening):

- EC plan item 1: r_B is a path by calendar year (the data's real effective rate to 2025, the projection's real effective rate over 2026–60, linear to 2% by 2070, 2% after); `prices.r_B` is the terminal value (`EC_ALIGNMENT_PLAN.md` item 1).
- Stock-flow adjustment 2024–25: the data's debt ratios are imposed (154.2% in 2024, 146.1% in 2025) and the flows that reconcile them with the recursion at the model's own primary balance, real rate and growth are recorded as those years' adjustments. This replaces the fixed-value decision above. The file's 2025 cell (6.24% of GDP) is not used; it does not close the identity from the 2024 stock.
- Unemployment path values: the 2024–25 outturns, the Spring 2026 forecast for 2026–27, then linear to the Report's 2050 and 2055 levels scaled to ages 25–64 (index 0.546 in 2050, 0.538 from 2055). The Report's 2025–45 values are not used.
- Transfer from abroad: the general government's net receipts from the EU budget (D.7 and D.9 received from the EU less own resources): 1.9% of GDP in 2023, 1.8 in 2024, 2.7 in 2025 and 2026, 1.0 from 2027 (`data/foreign_transfer_GR.npz`). The broader series of all EU payments to Greece less the national contribution (3.5% of GDP in 2023, `data/eu_transfers_GR.npz`) includes payments to farmers and firms and is not government revenue.
- Lump-sum transfer: λ times the run's own output, iterated with the terminal tax rate in the baseline's fixed point; the experiments hold the level path.
- Retirement age after 2070: 0.75 of the change in life expectancy at 65 (`EC_ALIGNMENT_PLAN.md` item 4).
- δ_g: 0.04477 = 0.040/0.703 less Γ_0 − 1 (0.01212), not 0.046.
- Not adopted: the pension path by cohort and age; health unit costs growing with output per person (`EC_ALIGNMENT_PLAN.md` items 2 and 6).

No decision is open; the equation changes are approved (doc row 15).

## 6. Audit of 2026-10-07

Two context-free reviews of this plan against the code (one on the economics, one on the code paths), before any code is written. Findings that change the plan, ranked; items marked "verified" were checked against the code by the author of this plan afterwards, the rest are by the reviewers' reading.

### 6.1 Errors in this plan, corrected above

- §3.12: the four model-specific lines carried the wrong sign; the table now sums to −5.0 exactly. The pension and other-benefit lines mix ESSPROS and COFOG (0.4 of cash benefits in neither).
- §3.14: the firm conditions and the incidence numbers are confirmed (K/Y × (1−τ_y), w and K/L × (1−τ_y)^(1/(1−α)) = 0.935, Y × 0.978 at fixed hours, loss 0.05% of output). Corrections: τ_p is an SMM parameter targeting tax_p/Y = 0.13 and its base wL/Y falls to (1−τ_y)(1−α) = 0.640, so τ_p refits to about 0.203 and the recalibrated wedge is about 31.4%, not 30.7%; the claim that the I_g multiplier falls because the capital response is "scaled by 0.935" conflates the level with the elasticity, which is η_g/(1−α) with or without τ_y, so the multiplier is unchanged to first order; under the code's timing I_t = Γ_t K_{t+1} − (1−δ)K_t a rate step in 2024 is booked in 2023's investment (moot under the constant rate); the "data counterpart 4.4" is loose: the national-accounts wedge net of subsidies and consumption taxes is D2 − D3 − 12.7 = 2.4, and the ~0.05 that τ_y carries is the budget residual (13.4 of unmodelled revenue less 8.4 of unmodelled spending), to be described as such.
- §3.4: the education formula is in levels; in the model's per-capita detrended units the driver is the school-age population relative to the model's 25–99 population, s_t = (S_t/N_t)/(S_2023/N_2023), and e_t = e_2023 · y_2023 · (w_t/w_2023) · s_t with w_t the detrended wage; S_t/S_2023 alone overstates E by the fall of N_t (25–30% by 2070). The raw EUROPOP cache (`data/europop2023_raw/proj_23np_EL_*.json`) holds all ages; the projection ends 2100 and the transition runs 180 periods plus `n_post`, so s_t is held at its last value.
- §3.10: τ_p = 0.13 including imputed contributions is a closed circuit for the primary balance, not for incidence: in the model the 1.8 is a real payroll tax (2.8 pp of τ_p) and a real purchase inside G, defence, health and education (D122 is 17% of D1). The incidence-consistent choice is 0.112 with the four purchase lines net of D122 (G 7.8, defence 1.66, health 5.3, education 2.6). Decision reopened (§6.4).

### 6.2 Design gaps under the constant-rate baseline

- Stock-flow adjustment 2024–25 (verified): `baseline_closure.py` lines 107 and 112 define sfa_2024 = monitor_debt − (carried − pb) and sfa_2025 = dsa_debt_2025 − (carried − pb) with pb the Commission's. With the model's own pb the same formula would make debt hit 154.2 and 146.1 by construction and the two years would be a hidden closure. The sfa for 2024–25 must be fixed numbers from the data (data pb, nominal growth and implicit rate, re-expressed on the model's real basis), after which the model's 2024–25 debt differs from the data by its pb gap. Decision (§6.4).
- Terminal rule: the old closure held the debt ratio at its 2060 value from 2061; the constant rate has no anchor. With n → 0 the long-run Γ − 1 = 1.7% exceeds r_B = 1.05%, so a steady surplus of s sends the debt ratio to −s/(Γ − 1 − r_B): a 3% surplus gives −460% of output. `_check_terminal_convergence`, the rest-point check in `eval_fiscal_results.py` and the experiments' target B/Y at T_bal (the "closure's constant level") lose their meaning. Decision (§6.4).
- EU transfers: option C puts the EU net inflow (1.9% of GDP in 2023, about 1.0 after 2027) on residents; the labour wedge is overstated by about 3 pp of τ_y-equivalent and NFA accumulation understated by that flow. Option F (a transfer line from abroad with a path, the mechanism O has today, sized at the EU net flow) is the fix. Decision (§6.4).
- Comparability of the model's own pb with the Commission's from 2024, what differs by construction: no cyclical component or one-offs; the 2024–25 revenue measures; EU transfers and RRF investment (pb-neutral, current-account non-neutral); property income; the Commission holds the structural balance before ageing at 2.3 with a constant revenue ratio while the model's bases drift; health unit costs with trend productivity; no one over 99; unemployment flat. Under the old closure O absorbed these; under the constant rate they are in the model's pb and debt. The report must list them next to the comparison.
  Size, from the 2026-10-06 baseline with a constant residual in place of the closure path (the first-order stand-in for the constant output tax, before the retargets): the model's own primary balance is 2.3 pp of output below the outturn in 2025 (2.6 against 4.9) and then above the Commission's path by 1.7 pp in 2030, 2.9 in 2050 and 3.5 in 2060 (2.4 on average over 2026–60), because the Commission holds the structural balance before ageing at 2.3 while the model's revenue ratio rises with wealth (consumption tax +1.8 pp, capital-income tax +1.2, bequest tax +0.6 by 2050) and pensions fall from 16.0 to 13.2 by 2060, against health +2.1 pp by 2050. Debt recursion from the 2025 ratio with the projection's sfa: 136.0 in 2030, 105.7 in 2040, 79.8 in 2050, 39.8 in 2060 against the projection's 103.4. Swapping one input at a time in 2060: the Commission's balance instead of the model's gives 125.8 (the old closure's path), the projection's rate path 31.4, the projection's growth 25.4. The two sources of the report (constant r_B against the projection's rate path, output growth 0.65% against 1.08%) keep their sign and size (about +8 and +14 points in 2060, +3 and +20 under the old closure, where the rate's early and late differences offset); the balance becomes the largest source and reverses the sign of the gap. The retargets and new lines move the balance gap: pensions at 14.0 scale the fall by 7/8 (−0.35 pp by 2060), the EU transfer line falls from 1.9 to about 1.0 after 2027 (−0.9 pp), the age range to 100 adds the 85+ pensions and health (the fiche's 12.0 against the model's 8.9 in 2070 bounds it at about −2 to −3 pp late in the horizon), the education line adds +0.5 pp by 2070, the lump sum is a constant share. The report's section "Why the debt paths differ" is rewritten after the recalibration with the three sources quantified this way.

### 6.3 Implementation requirements the plan omitted (code audit)

- Firm conditions in one function. They are computed at five sites: `olg_transition.py` 2375–2377 (wage path) and 2593–2597 (a second copy that sets K_domestic, hence NFA and Y; verified), `_marginal_products_njit` 1026–1031 (used by `factor_prices()` and tests), `calibrate.compute_equilibrium_prices` 1352–1356 (the steady-state site; `_compute_ss_aggregates` and `compute_fiscal_ratios` reuse its K_over_L; `load_config` derives the household wage from it, so the SMM sees τ_y through w only), `eval_fiscal_results.chk_firm_foc` 375–398 (re-derives both conditions without τ_y and would fail on every scenario). `_compute_wage_path_njit` 1070–1080 is dead code. Change: one `firm_conditions(r, A, K_g_factor, α, δ, τ_y) → (K_over_L, w)` called from all sites, τ_y stamped into `params` by `run_fiscal_figures.py` and read by the evaluator. The household side needs no change: both backends take `w_path` as given.
- Calibration order. τ_y moves every SMM moment through w, so the pin cannot stay a one-shot subtraction after the loop (`pin_baseline_closure.py` 103–111). Per round: (1) SMM for θ at fixed (A_tfp, τ_y); (2) a two-dimensional root-find at fixed θ on (A_tfp, τ_y) with targets Y_ss = 1 and the full primary balance = `primary_balance_target_over_Y`, extending `normalize_A_tfp.eval_ss`; (3) convergence on the SMM residual, |Y_ss − 1| and |pb − target|. τ_y as a seventh SMM parameter is not possible without recomputing w per evaluation (w is fixed at `load_config`). The pin is at the base-year calibration cross-section, where the SMM holds, not at the stationary state. Config key `fiscal.tau_y` (scalar) or `fiscal.tau_y_path`.
- Budget plumbing: a `tax_y` line in `total_revenue` (2118), the return dict, `compute_government_budget_path` keys (2692–2699), `calibrate.compute_fiscal_ratios` (1925–1957, the bequest-tax precedent), `eval chk_tax_revenue`; τ_y as a path on the object clamps at its last value for `n_post`; if passed through the scenario it is copied by `_apply_shock` and carried in `base_paths`.
- Removing O: readers at `olg_transition.py` 138, 253–256, 283, 290, 2143, 2180, 2265–2267, 2309–2321, 2695; `calibrate.py` 1832, 1970–1978; `baseline_closure.py` 117, 137–141 (`closure_from_run` would KeyError); `pin_baseline_closure.py` (whole script); `fiscal_experiments.py` 571, 610, 612, 619, 673, 677; `run_fiscal_figures.py` 112, 186, 192, 230–232, 439, 453, 574–578; `eval_fiscal_results.py` 272 (the goods-market check must drop O and add education, else it fails by ~3% of Y against a 2% tolerance); `reports/fill_report.py` 146, 297–311, 447, 644, 662–677; `reports/baseline_figures.py` 137, 154–156, 188–192, 281 (and the assertion at 105–106 that Y, L and K_domestic move in proportion, which fails under a time-varying τ_y or a drifting K_g); `regen_fiscal_figures_from_json.py` 49, 63, 81, 90; `test_olg_transition.py` 3360–3363; `CLAUDE.md` 28, 37, 167–168, 211–212; `data_inventory.md` 60–77; `EC_ALIGNMENT_PLAN.md` §8. Either keep the budget key at 0 for one release or update every reader.
- Paths in the config: `defense_over_Y` and `I_g_over_Y` are read as scalars in `calibrate.compute_fiscal_ratios` 1967–1978, `pin_baseline_closure.py` 102–111 and 126, `fill_report.py` 642–644, `run_fiscal_figures.py` 109–112, `calibrate.build_olg_transition` 1829–1832; only the transition (`_spend`, `_apply_shock`, `_extend_base_paths`) takes a `(T,)` ratio. A path goes under a separate key by calendar year (`fiscal.defense_over_Y_path`), the base-year scalar stays, and `build_olg_transition` resolves it.
- I_g path: `simulate_transition` rejects `I_g_over_Y` when η_g ≠ 0 (2325–2329, verified); the production flow passes I_g as the stationary level (δ_g + Γ_t − 1)·K_g, computed independently in `run_fiscal_figures.py` 100–116, `reports/fill_report.py` 630–632 and `pin_baseline_closure.py` 97–99. A 2024–25 ratio path needs a level path from a preliminary baseline's Y (feedback ignored) and one shared helper. With I_g/Y 0.040 and K_g 0.703, stationarity needs δ_g ≈ 0.046 (config 0.04281, set for 0.03862); otherwise K_g drifts and `chk_kg_ss_gap` warns. Decision (§6.4).
- Education line: an `education_spending_path`/`education_over_Y` pair like defence (constructor 135–138 and 244–256, `_active_*` 280–290, `simulate_transition` args 2264–2267 and 2307–2321, the `_spend` call 2140–2143, the return dict and `total_spending` 2161–2180, `compute_government_budget_path` 2692–2699, `_apply_shock` 570/609/618, `_run_one_simulation` 672–677, `fill_report.py` 640–645, `MACRO`/`FISCAL_VARS` in `run_fiscal_figures.py` 435–453 and `regen_fiscal_figures_from_json.py` 45–81, `eval chk_goods_market` 271–272 and `chk_calibration_ratios` 526–530). `_spend` cannot be reused as is (ratio × Y_t has the wrong driver). Whether E follows the run's own w_t (then it responds inside the I_g experiment) or the baseline's: decision (§6.4).
- Unemployment path: `lifecycle_perfect_foresight._income_process` (505–546) builds one transition matrix per education group from `edu_params[*].unemployment_rate`, `job_finding_rate` and the cap `max_job_separation_rate`; the backward induction and the NumPy simulation read a single `P_y`, the JAX batched solve and simulation take one matrix per group, `_stationary_dist_cache` is keyed on the group ("P_y_2d fixed for object lifetime"), and `olg_transition.py` 679 reads the scalar rate. A path means a matrix per age for each cohort (year = entry year + age), the 2023 matrix for the base-year cross-section and the stationary distribution, the path in the household cache key and in `pre_transition_paths` (identical in baseline and counterfactual, so the stitching is unaffected), and `calibrate.compute_unemployment_rate` on the 2023 matrix. The exact solver and simulation sites beyond these are to be listed at implementation; this is a change to an exogenous process, not to an equation.
- Lump-sum transfer: the household receives it every period at every age, so it is a new term in the period budget constraint of `lifecycle_perfect_foresight.py` (the NumPy solver and simulation) and of the JAX batched solver and simulation; `bequest_lumpsum` (received once at entry, `lifecycle_perfect_foresight.py` 792–793) is the nearest precedent but not reusable. On the transition it is a per-period amount per adult, T_t = `lump_sum_over_Y` · Y_t / N_t, fed through the cohort paths like the pension and wage paths (`_extract_cohort_path`), stitched for the MIT baseline, and carried in `pre_transition_paths`; the `_household_cache` key must include it. Budget: a `lump_sum` line in `total_spending`, the return dict, `compute_government_budget_path`, `calibrate.compute_fiscal_ratios`, `_apply_shock`, `_extend_base_paths`, the evaluator's goods-market and calibration-ratio checks, the report's fiscal variables. It is a transfer, so the resource constraint is unchanged.
- Time-varying τ_y option: `closure_from_run` must read `budget_tax_y` and iterate; each iteration is a full household solve (τ_y,t feeds w, so the household cache cannot serve it). `_household_cache` keys on `w_path_full`, so a τ_y change misses automatically; a path identical in baseline and counterfactual keeps A[0] exact for the same reason the I_g case does; extend `check_a0_predetermination.py` with a non-constant τ_y path on both backends.
- Tests. Fail or change: `test_olg_transition.py` 3360–3363; eval checks `chk_firm_foc`, `chk_goods_market`, `chk_tax_revenue`, `chk_calibration_ratios`; `reports/baseline_figures.py` 106. New: τ_y = 0 bit-identical to today on both backends; with τ_y > 0, K_d/Y = (1−τ_y)α/(r+δ), wL = (1−τ_y)(1−α)Y and τ_y Y + wL + (r+δ)K_d = Y on the transition and in `compute_equilibrium_prices`; SS w equals the transition's w at t = 0; w_path[t] ∝ (1−τ_y,t)^(1/(1−α)) for a path; budget identity with `tax_y` and `education`; A[0] with a τ_y path; the pin reproduces the base-year primary balance; the education formula and its clamping past the data; a `(T,)` defence path through `_apply_shock`, `_extend_base_paths`, `compute_fiscal_ratios` and the pin with the scalar key.
- Also omitted from §4: `pin_baseline_closure.py` rewrite, `eval_fiscal_results.py`, `regen_fiscal_figures_from_json.py`, `reports/*`, `data_inventory.md`, `CLAUDE.md`, `chain_fiscal_after_loop.sh` (gates on the A[0] check), the two backends.

### 6.4 Decisions reopened or added by the audit

Calls recorded in §5; none open.

1. Stock-flow adjustment 2024–25: fixed data-based values (recommended) or the residual off the model's pb (a hidden two-year closure). Replaced on 2026-10-07 (evening) by imposing the data's 2024 and 2025 ratios; see §5.
2. Terminal rule after 2060: constant τ_y throughout and accept the debt ratio's drift (toward large net assets at r_B < g), or constant τ_y through 2060 and from 2061 the rate that holds the debt ratio at its 2060 value (recommended: the old rule's tail, keeps the experiments' target and the terminal check meaningful).
3. EU transfers: add option F (a transfer from abroad, net 1.9 in 2023 to about 1.0 after 2027, with a path) beside τ_y (recommended by both reviews), or keep everything on τ_y.
4. τ_p: 0.13 with the purchase lines as they are, or 0.112 with the purchase lines net of imputed contributions (G 7.8, defence 1.66, health 5.3, education 2.6; incidence-consistent).
5. I_g: constant 0.040 as a stationary level with δ_g re-derived to 0.046 (recommended: the 2024–25 bump adds about 1.6% to K_g and 0.1% to Y and needs a level path from a preliminary run), or the 2024–25 path with a preliminary baseline.
6. Education line driver: the run's own w_t (recommended) or the baseline's.
7. The 85+ population: done 2026-10-08, the age range is 25–99 (§3.17).

### 6.5 Pre-existing defect found on the way (not caused by the plan; confirmed by a run on 2026-10-07, fixed the same day)

`fiscal_experiments.run_baseline` builds the pre-transition dict at line 1335 from `base_paths.get('w_path')` before the baseline simulation, when `base_paths` has no `w_path` (verified: `_build_pre_transition_paths` copies the keys present), stores it as `base_paths['_pre_transition_paths']`, and sets `base_paths['w_path']` only afterwards (1359); the dict is never updated and the dispatcher reuses it (1433). In `solve_cohort_problems` this makes `_base_w_ext = None` (1411) and the stitching model falls back to the counterfactual wage path (1497), the bug fixed on 2026-03-06. It is masked while `_mit_baseline_cache` holds the baseline's solved models (1365–1376). `run_fiscal_figures.py` line 241 reruns the baseline with the closure path, served from the household cache: that call passes `pre_transition_paths=None` (1351), the id test at `olg_transition.py` 2425–2428 clears `_mit_baseline_cache` (verified), and the cache-served run sets `birth_cohort_solutions = None` (2507, verified), so the cache is not repopulated. Every scenario then solves its stitching models with counterfactual wages. For τ_l shocks w is unchanged and A[0] holds; for the I_g shock (w moves with K_g) A[0] is not predetermined. `check_a0_predetermination.py` starts from a fresh scenario and does not exercise this path; `eval_fiscal_results.py` has no A[0] check (0 FAIL on the 2026-10-06 run says nothing about it). Fix: set `pre_tp['w_path'] = base_paths['w_path']` (and `r_path`) at 1359, and do not clear `_mit_baseline_cache` on a cache-served rerun (or repopulate it from kept models); add an A[0] check to the evaluator; confirm with a harness (baseline run, cache-served rerun, I_g scenario) comparing A[0]. The I_g multiplier of the 2026-10-06 run (1.43) is unconfirmed until then.

Run of 2026-10-07 (small model of `check_a0_predetermination.py`, both backends): along the driver's sequence the dictionary's wage path is `None` and the stitching cache is empty after the cache-served rerun, as read. With the driver's shock size A[0] still matched the baseline exactly, because the contaminated pre-transition policies did not move any agent's asset choice on the grid; with a shock that raises the wage by 3.8% in the first year A[0] fell 0.43% below the baseline on NumPy. The saved 2026-10-06 results have A[0] equal to the baseline to all digits in every scenario, so the published I_g numbers are those of correct stitching. Fix applied: `run_baseline` writes the wage path (and the interest-rate path if missing) into the pre-transition dict once the baseline has run and clears the stitching cache before refilling it; `simulate_transition` leaves the stitching cache in place on a call without pre-transition paths; `run_fiscal_scenario` takes the object's last wage path for a baseline supplied by the caller without one; `solve_cohort_problems` warns when the dict lacks the wage or interest-rate path; `eval_fiscal_results.py` has an A[0] check (`a0_predetermined`, FAIL tier, tolerance 1e-10) on every non-baseline scenario; `check_a0_predetermination.py` has the driver sequence with the large shock (`Ig+rerun`, both backends), which fails on the old code and passes on the fixed one. 50 fiscal tests pass. Production rerun of the I_g set on the Mac (2026-10-07, 194 min): `output/fiscal_2026-10-07_a0/`. It reproduces `fiscal_2026-10-06_ret_annual`, the run the report uses, to floating-point precision in every scenario (output, consumption, wealth, hours, wage and the debt ratio within 1e-14; Δτ_l 0.034239 and 0.038276; multiplier 1.4011); the evaluator passes `a0_predetermined` on every scenario, 0 FAIL. The report's fiscal section therefore stands.
