# Continuous savings choice: minimum income benefit and the endogenous grid method

Plan, 2026-10-08; audited and revised 2026-10-09. Not implemented.

## 1. Why

The household chooses a' among the nodes of the asset grid (`argmax` in `lifecycle_jax.solve_period_jax`; the same grid search in `lifecycle_perfect_foresight._solve_state_choice`), and the exact aggregation moves mass from node to node. A small change in prices moves some households to the next node, so the deviation of a counterfactual from the baseline contains discrete jumps. In the I_g experiment of `output/fiscal_2026-10-08f` the year-to-year jaggedness of the output and consumption deviations is about 0.05–0.1 pp. Aggregation is exact, so none of this is sampling error. The grid runs from 0 to 50 with curvature 1.5: near mean wealth (about 3) adjacent nodes are 0.30 apart at n_a = 100, roughly 10% of mean wealth, and 0.15 apart at n_a = 200. More nodes shrink the jumps without removing them.

The plan makes a' a continuous choice, solved by the endogenous grid method (Carroll, 2006) with endogenous labour (Barillas and Fernández-Villaverde, 2007), and spreads each household's mass over the two grid nodes around a' (Young, 2010). The aggregates are then continuous in prices.

The endogenous grid method requires a value function concave in a. The means-tested consumption floor breaks concavity: it tops resources before saving up to 0.0846, so below a threshold a̲(z, z_last) resources equal the floor whatever a is, V is flat in a, and its slope jumps at the threshold. The Euler equation then has multiple solutions and the method needs an upper-envelope step (Iskhakov, Jørgensen, Rust and Schjerning, 2017), which vectorises poorly. Step 1 replaces the floor with an income-tested minimum income benefit that depends on discrete states only, which keeps V concave.

## 2. Step 1: minimum income benefit in place of the consumption floor

### Model

The Guaranteed Minimum Income guarantees a household an income of EUR 200 a month for a single adult and pays the difference when the household's own income falls short (`data/transfer_floor_GR.json`). In the model the guaranteed income level is y_min = 0.0846 of output per person aged 25–84 in 2023. Unemployed households of working age and retirees, at every age, receive the minimum income benefit

  b_t(j, z, z_last, ξ) = max(0, y_min − y_t(j, z, z_last, ξ)),   y_t = T^ls_t + I_t(j, z, z_last, ξ) − (1 − κ_t) m(j),

and the employed receive none. I_t is after-tax UI, (1 − τ^l_t) T^UI(z_last, ξ), for the unemployed and the after-tax pension, (1 − τ^l_t) P_t(z_last, ξ), for retirees, so y_t is the household's non-capital income net of out-of-pocket medical spending. The benefit raises y_t to y_min and is zero when y_t exceeds it. y_t depends on (t, j, z, z_last, ξ) and not on a, so b_t is a single number at each of these states, the budget stays linear in a and V stays concave. The benefit is untaxed. The wealth test and the capital-income test of the GMI are dropped. Through y_t the benefit moves with τ^l_t and T^ls_t: a higher labour tax raises it for UI recipients and retirees, and a change in the lump sum is offset one for one where the benefit is positive.

The employed are left out because an income test on their earnings would make resources max(y_min, X + MW ℓ), a convex function of hours, and the budget set in (c, ℓ) non-convex.

Effects to report, not targeted:

- spending on the benefit, % of output, against the floor's 0.08% in the current baseline. A rough count, not from the model: the unemployed without UI receive about y_min − T^ls = 0.05, which puts spending on the unemployed at 0.25–0.3% of output. Retirees at the pension floor do not qualify in the base year (after-tax floor 0.158 plus the lump sum exceeds y_min plus medical spending). With the pension index the after-tax floor falls to about 0.098 by 2070, and retirees at the oldest ages, where medical spending reaches 0.086, qualify;
- the zero-wealth share (model 0.142, data 0.011), by age group;
- the benefit's counterpart in the data. Eurostat ESSPROS books the GMI under the social exclusion function (means-tested cash benefits). The unemployment function's means-tested cash benefits are 0.01% of GDP in 2023 (`spr_exp_fun`). The social exclusion series is to be fetched and recorded in `data/transfer_floor_GR.json`.

Resources stay positive at every state. An unemployed or retired household has resources before saving of at least R_t a + y_min ≥ y_min, including along the health-coverage experiment's κ and medical-spending paths. An employed household has MW > 0, so the hours condition gives positive consumption at a = 0 even where T^ls falls short of medical spending.

### Code

- The floor term `max(0, transfer_floor − budget)` is replaced by `b_t` for the unemployed of working age and for retirees at the five sites that compute it: `compute_budget_jax` (`lifecycle_jax.py:271`), `_state_outcomes_jax` (`:936`), `LifecycleModelPerfectForesight._compute_budget` (`lifecycle_perfect_foresight.py:888`), `_simulate_sequential` (`:1390`) and `_exact_columns` (`:1532`). Each needs τ^l_t, the lump sum, κ_t, m(j), UI and the pension, which all five already hold.
- Plumbing of the parameter: the positional `transfer_floor` slot of the batched kernels (`olg_transition.py:608, 982, 1098`; `distribution_stats.py:154`), the MIT-baseline override (`olg_transition.py:1723–1731`, `pre_tp['transfer_floor']` in `fiscal_experiments.py:69`) and `simulate_transition(transfer_floor=)` (`olg_transition.py:2534–2656`) carry `minimum_income` alongside it. Config key `external_params.minimum_income` (y_min); `transfer_floor` set to zero in the GR configs and refused together with `minimum_income` (the two are alternatives).
- The panel element that records the floor's top-up (`transfer_sim`, the 23rd) records `b_t`; `compute_government_budget` books it in the existing `transfers` line, `compute_fiscal_ratios` likewise. The name of the budget line stays; its definition changes in the docs.
- `fiscal_experiments.py` `financing='transfer_floor'` keeps working on the old floor only; refused with `minimum_income`.
- `build_transfer_floor_GR.py`: write `minimum_income` (same value) and the ESSPROS series.
- Docs: `CLAUDE.md`, `docs/dsa_economic_problem.md`, the calibration report's parameter row (`reports/fill_report.py`) and text. The paper's equation for T^f (`docs/DSA-LSA model.tex`) changes after the recalibration (§8).

### Tests (`test_minimum_income.py`)

- Both backends agree on policies and exact means with the benefit on.
- The benefit depends on states only: b_t is equal across the asset nodes at every (t, j, z, z_last, ξ). Concavity of V is tested under the endogenous grid method (§4); grid-search V is the maximum of finitely many concave functions of a and need not be concave.
- y_t + b_t ≥ y_min at every unemployed and retired state, with b_t = 0 for the employed.
- Budget identity with the benefit booked.
- `minimum_income = 0` and `transfer_floor = 0` reproduce the current code at zero floor bit for bit. The benefit is gated on y_min > 0, as the floor is gated on `transfer_floor > 0` (`lifecycle_jax.py:266`): at y_min = 0 the formula would pay max(0, −y_t) to the unemployed without UI whose medical spending exceeds the lump sum.

Step 1 runs with the existing grid search and is usable on its own.

## 3. Step 2: savings policy as a level, mass split over two nodes

Every consumer of the savings policy moves from a grid index to a level. Done before the new solve, with the grid search kept, so the change can be checked bit for bit: a grid-search policy is a level that sits on a node, and its lottery weight is one.

### Representation

- New policy arrays `a_next_policy` (float, level of a' in detrended units), same shape as now, `(T, n_a, n_y, n_h, n_y_last)` and the per-α stacks `a_next_policy_alpha`. The integer `a_policy` and `a_policy_alpha` are removed, so every reader that is missed fails loudly.
- The lottery: for a' ∈ [a_k, a_{k+1}], index k = `searchsorted(a_grid, a') − 1` (clipped to [0, n_a − 2]) and weight ω = (a_{k+1} − a')/(a_{k+1} − a_k) on a_k, 1 − ω on a_{k+1}. a' is clipped to [a_min, a_max]. One helper per backend (`lottery_jax`, `lottery_np`), used everywhere. `searchsorted` uses `side='left'`: a policy on node n then gives k = n − 1 and ω = 0 (k = 0 and ω = 1 at n = 0), so the whole mass moves through the (1 − ω) term. The push-forward applies the ω scatter and then the (1 − ω) scatter as two calls; with that order every nonzero contribution reaches its node in the order of the current single scatter, and the ω scatter adds exact zeros.

### Readers to change

| Site | Change |
|---|---|
| `exact_age_means_jax`, `LifecycleModelPerfectForesight._exact_columns` / `exact_age_means` | push-forward of the distribution: scatter `ω·μ` to k and `(1−ω)·μ` to k+1 (`.at[].add` / `np.add.at`) |
| `exact_panel` (both), `_cross_section_exact` | same push-forward |
| `_agent_step_jax`, `_simulate_sequential` | agents stay on the grid: an agent moves to node k+1 if a uniform draw ≥ ω, to k otherwise. JAX draws from `fold_in(key, 2)`; NumPy draws only when ω is strictly inside (0, 1), so a grid-search policy reproduces the current random streams |
| `_state_outcomes_jax`, the NumPy panel | a' is read as a level; savings, wealth carried out and the bequest measure `(1+g)·a'` use it directly |
| `olg_transition.py` (≈18 sites) | MIT-shock stitching of the policy stacks, `_policy_stack`, the household cache key, the on-device route (`jax_policies_on_device`), the per-age diagnostic print |
| `distribution_stats.py` | reads the level stack |
| `_cross_section` / `cross_section_batched` (simulated calibration route) | as the simulation |

### Tests

- Grid search with the new representation reproduces HEAD bit for bit on both backends: policies (as levels), exact means, MC panels, a transition, a fiscal scenario (reference arrays from the current commit, as in `tests_data/policy_reference_e80b119.npz`). On CPU only: scatter-add with repeated indices has no fixed summation order on GPU.
- The lottery preserves mean assets exactly: Σ (ω a_k + (1−ω) a_{k+1}) μ = Σ a' μ to 1e-14.
- MC panel means converge to the exact means as n_sim grows (existing test, rerun with a policy off the nodes).
- The ≈41 sites in `test_olg_transition.py` that index `a_policy` move to the level arrays.

## 4. Step 3: the endogenous grid method

### Household problem in a period

Detrended budget for a working-age employed household (z > 0):

  (1+τ^c_t) c + (1+g) a' = R_t a + MW_t ℓ + X_t,   R_t = 1 + (1−τ^k_t) r_t,   MW_t = w_t ε_j z e^ξ (1−τ^p_t)(1−τ^l_t),

where X_t collects the state-only terms (lump sum, bequest receipt at entry, minus out-of-pocket medical spending). For the unemployed X_t adds after-tax UI and b_t, and ℓ = 0; for retirees X_t adds the after-tax pension and b_t, and ℓ = 0. Utility u(c) − ν ℓ^{1+φ}/(1+φ).

For each state (z, z_last, h, ξ) and each a' on the grid:

1. **Expected marginal value.** By the envelope condition V_{a,t+1}(a) = R_{t+1} u'(c_{t+1}(a))/(1+τ^c_{t+1}), from next period's consumption policy on the grid. Its expectation E_t V_{a,t+1}(a', ·) uses the same operators as the value now: the P_h einsum, the UI eligibility mixture in the z' = 0 continuation (linear, so it carries over), the income transition (age-dependent where `P_y_age_health`), the retired continuation at the household's own z_last, and survival π_t.
2. **Euler equation.** u'(c) (1+g)/(1+τ^c_t) = β π_t E_t V_{a,t+1}(a'), so c = (u')^{-1}(β π_t (1+τ^c_t) E_t V_{a,t+1}(a') / (1+g)). With log utility c = (1+g) / (β π_t (1+τ^c_t) E_t V_{a,t+1}(a')).
3. **Hours.** ν ℓ^φ = MW_t u'(c)/(1+τ^c_t), so ℓ = (MW_t u'(c) / (ν (1+τ^c_t)))^{1/φ}, capped at 64 as now. Closed form: no root-finder in the unconstrained region.
4. **Endogenous wealth.** a = ((1+τ^c_t) c + (1+g) a' − MW_t ℓ − X_t) / R_t.
5. **Interpolation onto the grid.** The pairs (a, a') are increasing in a under concavity. a'(a) on the grid nodes by linear interpolation in a; above the largest endogenous point by linear extrapolation, clipped to a_max. At each node (c, ℓ) are then recomputed from the budget at that a' jointly with the hours condition, as in step 6 with a' in place of 0 (one hours solve per state). Interpolating c and ℓ separately would keep the budget, which is linear in (a, c, a', ℓ), but violate the hours condition between endogenous points, and in the extrapolation region it would break the budget once a' is clipped.
6. **Borrowing limit.** For grid nodes below the endogenous a at a' = 0, a' = 0 and (c, ℓ) solve the budget with a' = 0 jointly with the hours condition. With log utility this is the problem `solve_labor_log_jax` already solves (the hours equation ν ℓ^φ (ζ + ℓ) = 1); other γ use `solve_labor_robust_jax`. For the unemployed and retirees c = (R_t a + X_t)/(1+τ^c_t).
7. **Value function.** V_t(a) = u(c) − ν ℓ^{1+φ}/(1+φ) + β π_t E_t V_{t+1}(a'), with E_t V_{t+1} linear in a' between nodes. Needed for the welfare measures (`distribution_stats.py`, consumption-equivalent variation) and for tests, not for the solve.

The terminal period is unchanged (a' = 0). The pension floor and the minimum income benefit enter X_t only.

### Refusals

The solver refuses, with a clear error: `tax_progressive` (the HSV schedule makes MW depend on income), `schooling_years > 0` with child costs, `retirement_window` (already refused), `transfer_floor > 0` (the non-concavity of §1), γ ≠ 1 with g ≠ 0 (with additive disutility of hours a balanced growth path requires log utility, and the detrended Bellman equation of both solvers scales only the price of a' by 1 + g). Config key `household.savings_solver: "egm" | "grid"`; `"grid"` stays as the validation reference.

### Code

- JAX: `solve_period_egm_jax` beside `solve_period_jax`, same signature and outputs `(V_t, a_next_t, c_t, l_t)`; `solve_lifecycle_jax` picks the kernel from a static flag. The scan carries next period's consumption policy as well as V. The batched kernels (`_solve_lifecycle_jax_batched*`, `_tr`, `_pyc` variants) take the flag as a static argument.
- NumPy: `_solve_period_egm`, vectorised over states per period (no per-state loop). It is the reference the tests compare JAX against.
- Interpolation of a non-uniform endogenous grid onto the fixed grid: `jnp.interp` per state under `vmap`; `np.interp` in NumPy.

### Tests (`test_egm.py`)

- Euler residuals |1 − β π E V_a / (u'(c)(1+g)/(1+τ^c))|: below 1e-10 at the endogenous points of unconstrained states. Off the grid (a fine grid of a between nodes) the residuals carry the interpolation error of c_{t+1}; their mean and maximum log10 are reported, not bounded.
- Hours condition holds to 1e-10 at every employed state with ℓ below the cap.
- V is concave in a at every state (second differences ≤ 1e-12 in absolute terms).
- Convergence to the grid search: at fixed prices, EGM at n_a = 100 against grid search at n_a = 2,000 on a test economy with n_y = 3 (each period array of the grid search is then about 0.3 GB); policies within the fine grid's spacing. The tolerance on the exact age means of assets, consumption and hours is set from EGM at n_a = 100 against EGM at n_a = 400 on the same economy.
- Continuity: exact aggregate assets as a function of r over a fine sweep (r ± 1e-4 in 20 steps). No first difference exceeds three times the median first difference; grid search fails the same test. Second differences are not tested: the lottery weight is piecewise linear in a', so the aggregate has kinks where an a' crosses a node.
- Backends agree to 1e-10 (policies, V, exact means).
- Monotonicity of a'(a) and c(a) in a at every state.
- UI eligibility: recipient share and UI spending as in `test_ui_eligibility.py`, rerun under EGM.
- MIT stitching: assets in the shock year predetermined (`check_a0_predetermination.py`, evaluator `a0_predetermined`) on a `--tiny` fiscal run with `shock_period` 0 and 3.
- The policy-exercise tests (`test_policy_exercises.py`) and the fiscal tests rerun under EGM; reference arrays regenerated where they hold grid-search numbers.

## 5. Step 4: checks before recalibration

1. Run time: one production cross-section and one baseline transition, EGM against grid search, on the A100. Expected faster (no n_a × n_a hours solve); to be measured.
2. Grid size under EGM, at the current θ (no recalibration): baseline and the debt-financed I_g experiment at n_a = 100, 200. Report the roughness of the output and consumption deviations (standard deviation of second differences, 2030–2100), the I_g multiplier, the 2060 output and debt deviations, the baseline output tax rate over 2026–60 (the rate that brings the 2060 debt ratio to the projection's, `fiscal.tau_y_mode: "debt"`, so the 2060 debt ratio itself is equal at both sizes) and the targeted moments. Recalibrate at 100 if all of the following hold, at 200 otherwise:

| Statistic | 100 and 200 differ by less than |
|---|---|
| I_g multiplier | 0.01 |
| 2060 output and debt deviations | 0.05 pp of output |
| baseline output tax rate over 2026–60 | 0.05 pp |
| targeted moments | 1% of the moment |
| roughness of the output and consumption deviations | below 0.005 pp at both sizes |
3. Grid placement: mass above a = 20 in the base-year cross-section; if small, lower a_max or raise the curvature.
4. Positive consumption at every state along the baseline and the health-coverage experiment's κ and medical-spending paths (assertion at model build), and y_t + b_t ≥ y_min at every unemployed and retired state.

## 6. Step 5: runs

1. Recalibration (scale loop) with the minimum income benefit, EGM and the grid of §5.2, on the A100.
2. Baseline transition, scenario 1 (the 2023 output tax throughout) and the constant-unemployment baseline.
3. The policy exercises (`run_policy_exercises.sh`: I_g and health coverage, unanticipated in 2026, three financing rules), and the G/I_g set if the report keeps it.
4. Calibration report: parameter table (the consumption-floor row becomes the guaranteed income level y_min), the solution-method paragraph, baseline numbers, the two new experiments (POLICY_EXERCISES_PLAN §7). Paper calibration section after the report.

## 7. Order and size

Steps 1 and 2 are independent of each other and each is checkable against the current code (Step 1 at a zero benefit, Step 2 bit for bit with grid search). Step 3 needs both. Step 2 is the largest in lines touched; Step 3 is the largest in new code. No recalibration until Step 4 is done: every reported number changes once, in Step 5.

## 8. Decisions

- Taken (2026-10-08): the consumption floor is replaced by the income-tested minimum income benefit of §2; the savings choice is solved by the endogenous grid method with the two-node lottery.
- Taken (2026-10-09): the benefit covers the unemployed of working age and retirees at every age; its income test nets out out-of-pocket medical spending; the grid rule of §5.2 uses the tolerances in its table.
- Taken (2026-10-09): the grid-search solver stays in both backends as the reference (`household.savings_solver: "grid"`), used by the convergence and continuity tests; a later model change is made in both solvers or refused in one.
- Open: the paper's wording for the benefit, deferred until after the recalibration.
