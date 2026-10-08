# UI eligibility at the start of a spell

Plan, 2026-10-08. Not implemented.

## 1. Why

The model pays UI to every household in the first year of an unemployment spell. In the 2023 Labour Force Survey for Greece, one third of the unemployed in the first year of a spell receive benefits. The model matches aggregate UI spending (0.6% of GDP) because ρ^ui is fitted to it, so it spreads that spending over about three times as many recipients as the data.

| Unemployed receiving benefits, % | Model now | Greece 2023, ages 25–64 |
|---|---|---|
| Spell under 12 months | 100 | 33.2 |
| Spell of 12 months or more | 0 | 2.7 |
| All unemployed | ≈ 44 | 14.8 |

The model's 44% is the first-year share implied by f = 0.44 (by construction 1 − f is the long-term share). It is approximate: households that enter unemployed at age 25 have z_last = 0 and receive nothing.

## 2. Data

| Series | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| Receiving benefits, spell < 12 months, % (25–64) | 36.7 | 44.8 | 37.6 | 37.0 | 33.2 | 34.2 |
| Receiving benefits, spell ≥ 12 months, % (25–64) | 4.5 | 6.6 | 2.9 | 3.0 | 2.7 | 2.1 |
| Share of unemployed with spell ≥ 12 months (25–64) | 0.717 | 0.678 | 0.651 | 0.660 | 0.604 | 0.570 |
| All unemployed receiving benefits, % (25–64) | 13.6 | 18.9 | 15.0 | 14.6 | 14.8 | 15.9 |
| Same, ages 15–74 | 13.0 | 18.2 | 13.9 | 13.4 | 14.1 | 14.9 |

The last two rows weight the two receipt shares by the unemployed counts by duration. 2020 is raised by the pandemic measures.

Sources:

- Eurostat `lfsa_ugadra`, unemployed persons by duration of unemployment and distinction registration/benefits, unit PC, `regis_es` = UNE_BEN ("receiving benefits/assistance", self-reported in the LFS), durations M_LT12 and M_GE12, geo EL, sex T. https://ec.europa.eu/eurostat/databrowser/view/lfsa_ugadra/default/table
- Eurostat `lfsa_ugad`, unemployed persons by duration of unemployment, thousands, by single duration band (summed to under and over 12 months; NRP excluded). https://ec.europa.eu/eurostat/databrowser/view/lfsa_ugad/default/table
- Eurostat `spr_exp_fun` (ESSPROS), expenditure on the unemployment function, 2023: periodic cash benefits for full unemployment 0.56% of GDP, means-tested benefits 0.01%. This is the UI/Y target's counterpart. https://ec.europa.eu/eurostat/databrowser/view/spr_exp_fun/default/table
- Eurostat `une_ltu_a`, long-term unemployment share (ages 15–74), the source of f (`data/job_finding_GR.json`).
- MITOS (Greek government procedures portal), regular unemployment benefit: 5 to 12 months depending on days worked. https://en.mitos.gov.gr/index.php/ΔΔ:Regular_unemployment_benefit
- MITOS, long-term unemployment benefit, means-tested, after the 12-month regular benefit is exhausted. https://en.mitos.gov.gr/index.php/%CE%94%CE%94:Long-term_unemployment_benefit
- A new benefit scheme was announced for 1 April 2026 (To Vima); its effect on duration and coverage is not checked. https://www.tovima.com/society/greece-updates-unemployment-benefits/

All series are fetched by the Eurostat dissemination API (`https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/<code>?format=JSON&geo=EL`).

## 3. Model change

A new parameter, the eligibility probability p^ui. When an employed household becomes unemployed (z > 0 this year, z′ = 0 next year, next year still a working age), it is eligible with probability p^ui. The draw is carried in z_last, so no state is added:

- z_last′ = z with probability p^ui, z_last′ = 0 with probability 1 − p^ui, if z > 0, z′ = 0 and j + 1 ∈ 𝒲^b;
- z_last′ = z otherwise (unchanged rule).

An ineligible household has z_last = 0 and receives no UI for the whole spell, as an eligible one does from the second year. The draw is not applied on the transition into retirement, because there z_last fixes the pension base. The UI formula is unchanged. p^ui = 1 reproduces the current model exactly.

Calibration: p^ui is set externally to the share of the unemployed in the first year of a spell who receive benefits, 0.332 (2023, ages 25–64). ρ^ui stays in the SMM against UI/Y = 0.006 and rises by about a factor of three. The model's share of all unemployed receiving UI, p^ui × (1 − long-term share) ≈ 0.332 × 0.44 = 0.146, becomes a validation moment against 0.148.

## 4. Code

Single source: `external_params.ui_eligibility_prob` (default 1.0) → `LifecycleConfig.ui_eligibility_prob`, read by both backends; JAX threads it like `trend_growth` (model_params, the `solve_period_jax` unpack, `in_axes`; `model_params` is sliced by index, so append it).

Sites (line numbers at the current `trend-growth` HEAD, 1f96906):

1. JAX solve, expected continuation of a worker (`lifecycle_jax.py` around 404–416, `EV_working`): for z′ = 0 and z > 0 the continuation becomes p^ui·V(0, z_last′ = z) + (1 − p^ui)·V(0, z_last′ = 0); not at the last working age.
2. NumPy solve (`lifecycle_perfect_foresight.py:1087`, `next_val = self.V[t+1, …, i_y]`): same mixture for `i_y_next == 0`, `i_y > 0`.
3. JAX Monte Carlo simulation (`lifecycle_jax.py:1001`, `new_i_y_last`): a uniform draw when the household moves from employment into unemployment at a working age.
4. NumPy Monte Carlo simulation (the `i_y_last` update in the simulate loop near `lifecycle_perfect_foresight.py:1369`): same draw.
5. Exact aggregation, JAX (`lifecycle_jax.py:1369–1377`, `working = einsum(...)`) and NumPy (`lifecycle_perfect_foresight.py:1549–1561`): for z′ = 0 move mass (1 − p^ui) of each y_last = z > 0 column to y_last = 0.
6. Split retirement cohorts: the "next age is still working" condition uses each kernel's own retirement age (the split cohorts carry two).
7. Moments: add `ui_recipient_share` (unemployed with T^UI > 0 over unemployed, ages 25–64) to the calibration's untargeted moments and to `untargeted` in the config (0.148).
8. Builder `build_ui_eligibility_GR.py` → `data/ui_eligibility_GR.json` with the table of §2 and the chosen value; the config cites it.
9. `code/docs/dsa_economic_problem.md:29` states z_last stays frozen during a spell; correct it to the rule above.

## 5. Tests

- p^ui = 1 reproduces current policies, moments and a transition exactly (no-op guard), both backends.
- NumPy against JAX at p^ui = 0.33: policies and exact-aggregation moments.
- Exact aggregation at p^ui = 0.33: the UI recipient share equals p^ui times the share of the unemployed in the first year of a spell; UI/Y scales with p^ui at fixed ρ^ui.
- Monte Carlo against exact aggregation for the recipient share (sampling tolerance).
- The pension base is unchanged by p^ui for households unemployed in their last working year.
- `check_a0_predetermination.py` and the budget identity test.

## 6. Runs

1. Local: tests, then one calibration evaluation at the current θ with p^ui = 0.332 to see the moment shifts (UI/Y falls to about a third; A/Y and hours move slightly through precautionary saving).
2. GPU: scale loop (SMM with ρ^ui free, joint A and τ^y), baseline with the 2060 debt rule, G and I^g experiments, evaluator. About one hour on an A100 at the last timings. Confirm before launching.
3. Docs: model text (eligibility in the UI item and in the z_last rule of the Bellman equation), Table 1 row for p^ui with the sources above, moments table row for the recipient share, the reply to Ramon updated.

## 7. Decisions (taken 2026-10-08)

1. p^ui = 0.332, the 2023 value.
2. Ages 25–64.
3. One p^ui for all education groups.

Separate item, not part of this change: f is computed from the 15–74 long-term share in `une_ltu_a` (56.0% in 2023), while the 25–64 share computed from `lfsa_ugad` is 60.4%, which would give f = 0.40. Decided 2026-10-08: f stays at 0.44.
