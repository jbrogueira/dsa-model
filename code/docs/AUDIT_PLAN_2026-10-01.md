# Audit plan — model description, calibration, consistency audit

Written 2026-10-01, re-pinned to `3784ae5` the same evening. Status: **not
started**; the five decisions at the end are open.

## Object

Branch `trend-growth` at `3784ae5` (2026-10-01 22:32), config
`code/calibration_input_GR.json`, data sidecars `data/demography_GR.npz`,
`data/survival_GR.npz`, `data/europop2023_GR.npz`, and the parameter record
`data/pension_floor_GR.json`. The baseline is the
demographic transition from the measured 2023 cross-section with no policy
change. The fiscal layer (debt law, financing rules, terminal conditions) is
in scope as equations; the G and I_g experiment results are not, since none
exist under trend growth.

State of the object at the time of writing: `_derived.theta` holds the five
parameters fitted at 12:31 on 2026-10-01 on the single-solve calibration
route; `A_tfp` and the closure `O/Y` were pinned in the same run. The code
has moved since, and every move below supersedes those numbers:

- `calibration.base_year_cohorts` switched on (`08833c3`, 15:24); the
  transition's aggregates per living person (`a49e0be`, 15:36); the survival
  double-count in `_agent_weights` fixed (`6362091`, 15:43).
- The JAX solve values the pension at the last working age, as NumPy does
  (`0c645d4`, 21:11). θ was fitted on the JAX backend, so the fit used the
  wrong pension base.
- The SMM has a sixth parameter, `ui_replacement_rate`, and a sixth target,
  UI/Y (`01d382a`, 22:18); `theta_from_config` fills the sixth from its
  initial value when `_derived.theta` lacks it. The closure is pinned against
  the I_g level the transition spends, not the config ratio, and the pension
  fund accrues at r_B (`01d382a`).
- `pension_min_floor` 0.15 → 0.1671, derived in `build_pension_floor_GR.py`
  (`ae9900b`, 22:21).
- `terminal_debt_gdp` dates the stock and the flow together; `_birth_sim_cache`
  and `_period_cache` carry `_policy_version`, so the bequest fixed point
  iterates; the cumulative multiplier re-trends (`6ab03e7`, 21:35).
- Deleted: `_build_population_weights` and `_slice_means_njit`; raising now:
  `fertility_path`, `retirement_window`, a positive `transfer_floor`
  (`7aecdb8`, `01d382a`, `49fb3cc`). `_alive_fraction` refuses `n_h > 1`.
- `diag_ss_vs_transition.py`, `validate_backends.py` and
  `regen_fiscal_figures_from_json.py` now run the production fiscal branch
  (`999e59c`); `eval_fiscal_results.py` gates on the goods market (`3784ae5`).

The numbers in the config and in `output/calibration_growth/*.tex` predate
all of this. The commit messages of 2026-10-01 record each fix's measured
effect; Part III lists them as fixed before HEAD, not as findings.

## Deliverable

One document, `code/reports/model_audit_2026-10-01.{tex,pdf}` with a
markdown twin, about 10 pages. Style: referee or auditor report — facts,
locations, consequences; no framing of what the paper should claim.

| Part | Content | Length |
|---|---|---|
| I. The model economy | Section order of McGrattan, Miyachi and Peralta-Alva (2019, `lit-review/McGrattan-et-al-Japan.pdf.pdf`, §3): demographics; portfolios and returns (who holds domestic capital, foreign assets and sovereign debt, at which rate); prices and policy; the household problem, working and retired; technology; government budget constraints; equilibrium and the detrended balanced-growth path. Equations are what the code does. Notation follows the draft's `docs/DSA-LSA model.tex` where the objects coincide. No code references in the body. | 4 pp |
| II. Calibration | Same layout as `code/reports/calibration_report.tex`: externally set; fitted jointly by SMM (targets, weights, which moment identifies which parameter); pinned outside the SMM (A_tfp, closure); demographics (2023 cross-section, EUROPOP2023 path, tail). Reuses the report's table bodies in `output/calibration_growth/`; these are untracked files written by `fill_report.py` on 2026-10-01 at 14:36–14:46 from the 12:31 θ on the code before `b3a0d47`, which git does not record, so Part II states it. Every number carries the run it came from and a stale flag where the code has moved since. | 2 pp |
| III. Audit | Summary verdict; findings ranked Critical / Major / Minor, each with the fact, the file:line, the consequence and what was verified; the six items of `OPEN_ISSUES_2026-07-30.md` with their status at HEAD; the dead-code inventory (B6's classes a–f) with the evidence per item; what could not be verified without a production-scale run, which includes A[0] predetermination under the Step 0 demography unless C1b is run. Whether a dead item is deleted is a later decision, not part of this report. | 3–4 pp |
| Appendix | Code-to-equation correspondence (object, equation, file:line, both backends); the full dead-code inventory, one row per definition with its class, the search pattern and the result; the list of checks run and their outcomes. | 2 pp |

## Phases

### A — reconnaissance (done 2026-10-01)

Re-run on `b27e7a3..3784ae5` (11 commits, 21 files) the same evening; the
state paragraph above is the result. Established: repo layout and the "what
solves what" table in `README.md`;
the McGrattan et al. section order; today's commits and their messages; Step
0 of `code/docs/TREND_GROWTH_PLAN.md`; the July open-issues list; the calibration
report template and the run outputs it reads; the function maps of
`olg_transition.py`, `calibrate.py`, `fiscal_experiments.py`,
`lifecycle_perfect_foresight.py`; the test classes.

### B — six context-free agents in parallel, read-only

Each agent reads source files, the config and the data sidecars only.
Forbidden: `docs/*.tex`, `code/docs/*.md` (B5 excepted), `code/code_report_2026-06-15.md`,
the model sections of `README.md`, the assistant's memory, and the git
history (`git log`, `git show`, `git blame`): the commit messages of
2026-10-01 record findings. `code/CLAUDE.md` cannot be forbidden by
instruction: the harness injects it into any agent that reads a file under
`code/` (verified 2026-10-01 with a probe agent; the global CLAUDE.md and
the memory files are not injected), and it states the UI share of L, the
open bequest circuit, the UI-after-one-period behaviour, the B_initial
sizing and the Y_ss normalisation — items 1–5 of the open-issues list. So
before launch, move `code/CLAUDE.md` to the scratchpad; restore it after the
last agent returns and confirm with `git status`. The report states that the
agents ran without it. No agent is told what the others cover or what is
suspected; the questions below are classes of question, not findings. Each
returns equations with file:line pointers and a findings list that separates
"the code does X" from "X is inconsistent with Y".

| Agent | Files | Questions |
|---|---|---|
| B1 household | `lifecycle_perfect_foresight.py`, `lifecycle_jax.py` | Preferences; income, health and survival processes; budget constraints working and retired; every tax base; pension base and its indexation; UI; transfer floor; bequest receipt; the (1+g) on next-period assets; discounting with survival; terminal value; asset-grid bounds; the labour FOC. Where do the two backends differ? |
| B2 initial condition and cohorts | `calibrate.py` (`base_year_cross_section`, `base_year_cohort_survival`, `base_year_age_weights`, `compute_age_weights`), `olg_transition.py` (`solve_cohort_problems`, `_solve_cohorts_jax_batched`, `_simulate_cohorts_jax_batched` and its `_as_alpha_indexed`, `_extract_cohort_path`, stitching and `_mit_baseline_cache`, `_cohort_survival_schedule`, `_survival_schedule_at_year`, initial assets, bequest path), `fiscal_experiments.py` (`_build_pre_transition_paths`), `normalize_A_tfp.py`, `pin_baseline_closure.py`, `run_fiscal_figures.py` (B_initial, K_g initial) | What is assumed at t = 0: which cohorts exist; the prices, taxes and survival each was solved against; where its t = 0 assets come from; what pins K_g(0), B(0), NFA(0), w(0). Is the object the SMM targets the same object as the transition's t = 0 — weights, survival, seeds, n_sim? Which of θ, A_tfp and the closure are computed on which route? Where do the JAX batched solve and simulate depart from the per-cohort NumPy path: the α grid, the policy indexing, per-cohort survival, what the stitching overwrites? |
| B3 aggregation, transition, terminal state | `olg_transition.py` (`_entrant_weights`, `_alive_fraction`, `_aggregation_weights`, `growth_factors`, `_growth_at`, `_compute_all_cross_sections`, `compute_aggregates`, `compute_government_budget`, `simulate_transition`), `fiscal_experiments.py` (`compute_debt_path`, `_balance_residual`, `_check_terminal_convergence`, `_extend_base_paths`, `_nfa_ca_paths`, `_correct_base_macro_nfa`), `reports/fill_report.py` (goods-market residual, flatness statistic), `build_demography_GR.py` | Per-living-person aggregation and the Γ_t in each stock recursion (B, K_g, NFA/CA, pension fund); the goods-market identity, including what happens to the assets of agents who die; the labour aggregate; timing of stocks; the government budget line by line and the closure; what cohorts born after T − 60 face beyond T; the tail (mortality held at 2100, entrant ramp, n_∞) and whether every terminal condition uses Γ_T; whether the baseline has a rest point for debt at all. |
| B4 calibration procedure | `calibrate.py` (moment functions, `_agent_weights`, `_compute_ss_aggregates`, both routes of `run_model_moments`, `smm_objective`, `calibrate`, `load_config`, `theta_from_config`), `run_scale_loop.sh`, `run_step0_baseline.sh`, `normalize_A_tfp.py`, `pin_baseline_closure.py`, `build_pension_floor_GR.py`, `reports/fill_report.py`, `calibration_input_GR.json`, `data/pension_floor_GR.json` | What each target measures and on which population; which of the six parameters identifies which moment, and which of them `_derived.theta` holds; the SMM ↔ A_tfp loop and its convergence test; what `_derived.theta_metadata` says against the live code path; which recorded numbers are stale and why. |
| B5 known-issue status | May also read `code/docs/OPEN_ISSUES_2026-07-30.md` | For each of items 1–6: is the behaviour described still in the code at HEAD? Evidence by file:line. |
| B6 dead code | Every `code/*.py`, `code/reports/*.py`, `code/*.sh` and `calibration_input_GR.json`; may read the file table at the top of `README.md` for the entry points | Inventory every definition — module functions, methods, classes, njit kernels, config keys — and for each the callers found by a repo-wide search of `code/`, counting dictionary dispatch (`MOMENT_DISPATCH`), `getattr`, the multiprocessing wrapper and the shell drivers as callers. Classify: (a) unreachable — no caller outside its own definition; (b) test-only — reached only from `test_*.py`; (c) switched off in production — reachable only through a flag, kwarg, backend branch or config key that the production config and drivers never set (feature defaults, `recompute_bequests`, the NumPy branches); (d) dead inside live code — branches whose condition cannot hold under the production config, parameters read and never used, caches written and never read, two implementations of one quantity, fallbacks that mask a missing input; (e) config keys no code reads; (f) scripts no driver calls whose outputs nothing reads. Per item: the definition's file:line, the pattern searched, the result. No view on whether an item should be deleted. |

B6's production entry points are the drivers `run_scale_loop.sh`,
`run_step0_baseline.sh`, `chain_fiscal_after_loop.sh`, `run_cost_and_figure.sh`
and the scripts they call (`calibrate.py`, `normalize_A_tfp.py`,
`pin_baseline_closure.py`, `run_fiscal_figures.py`,
`regen_fiscal_figures_from_json.py`, `eval_fiscal_results.py`,
`reports/fill_report.py`, `diag_ss_vs_transition.py`,
`check_a0_predetermination.py`, `validate_backends.py`,
`diag_bequest_decomp.py`), the `build_*.py` data builders and
`health_flag_decomposition.py`; the test files are the second class of
caller. B6 starts from a mechanical inventory — an `ast` walk over `code/`
listing every definition, then a search for each name; `vulture` is not in
the venv and is not to be installed — and confirms every candidate by
reading the call sites. String-keyed lookups and dynamic dispatch are the
usual false positives.

### C — numerical checks

Local, CPU (`source ~/venvs/jax-arm/bin/activate`, `JAX_PLATFORM_NAME=cpu`),
run without asking, durations checked first, anything over ~2 min in the
background with a monitor:

- `pytest test_olg_transition.py -k "TestDemographicPath or TestBaseYearCrossSection or TestTrendGrowthHousehold or TestTrendGrowthStocks or TestCohortBatchedSurvival or TestCrossRoutineLevels or TestPensionBaseAcrossBackends"`
  (the cost-ratio test skips without a GPU).
- `check_a0_predetermination.py`: a toy harness — T = 20, n_a = 30, n_sim =
  50, T_tr = 10, one education group, fixed r and tax paths, no demographic
  path — that checks the stitching mechanism on both backends in minutes. It
  reads neither the config nor θ, so it says nothing about the production
  economy.
- Spot checks at test-fixture scale from Python: Σ_j w_j S_j = 1 at several
  t; Γ_t from the living population against `growth_factors`; the survival
  diagonal of three cohorts against the sidecar; the single-solve and cohort
  routes agreeing under one shared schedule.
- **C0, cost of the cohort route:** time one `base_year_cross_section` call
  for one education group at the production grid and `n_sim=200`, from a
  scratchpad script. The SMM has never run on this route; the only
  production fit (12:31) is single-solve. Three education groups times the
  objective evaluations of the 12:31 run (from its log; bounded by `maxiter`
  otherwise) bounds the cost of C2 and of each cross-section build in C1.
  Decisions 1 and 3 wait for this number.

Production scale, GPU (Verda or the A100 recipe in memory):

- **C1:** the two scripts each solve their own baseline transition and each
  build the base-year cross-section on the cohort route —
  `diag_ss_vs_transition.py jax 2000` (cross-section, then transition) and
  `reports/fill_report.py --run-baseline --implied --n-sim 2000` (transition,
  then cross-section). Two transitions and two cross-sections, not one.
  Gives the t = 0 gap, the goods-market residual in both forms and the
  flatness column. Tests the machinery, not the fit. Duration: twice the
  transition time in the 12:31 run's logs
  (`output/calibration_growth/step0_a100/`) plus twice C0's figure.
- **C1b, optional, one more transition:** A[0] predetermination at
  production scale — a scratchpad script that runs `run_fiscal_scenario`
  with a τ_l shock on the production config and compares A[0] with the
  baseline. No repo script does this; without it the check is listed as not
  verified.
- **C2:** `run_step0_baseline.sh` — recalibration on the cohort route, then
  the same checks. Gives current numbers for Part II. Duration unknown until
  C0; each objective evaluation is 60 solves per education group where the
  12:31 run had one.

What solves what: Parts I and III and the whole of Phase B solve nothing.
Part II at the stale θ solves nothing. Recalibration (`run_scale_loop.sh`)
solves only the base-year cross-section through `run_model_moments`, which
`normalize_A_tfp.py` and `pin_baseline_closure.py` also call; no
transition. On the cohort route `base_year_cross_section` solves each of
the 60 cohorts per education group in its own `model.solve()` call, in
sequence, and simulates each to its base-year age; the batched cohort solve
exists only in the transition (`_solve_cohorts_jax_batched`). The
transition is solved only by C1, C1b and the second half of C2.

### D — verification and consolidation

Re-read the code at every Critical and Major finding; discard what does not
reproduce; merge duplicates; rank. Repeat B6's search for every dead-code
item and read the call site; an item stays only if the search reproduces. Only then compare Part I against the
draft's `docs/DSA-LSA model.tex` at submodule commit `da44637` (2026-09-21).
The draft predates the branch, so a disagreement that
`code/docs/TREND_GROWTH_PLAN.md` settles is booked as a settled change, not
a finding; every other disagreement is a finding ("the draft says / the
code does"). No edits to code, config or the draft; the scratchpad scripts
of C0 and C1b are not repo changes.

### E — write and compile

LaTeX in `code/reports/`, `latexmk -pdf`; markdown twin alongside. Style
per `~/.claude/CLAUDE.md`.

## Decisions open

1. Audit the code at HEAD with the stale θ, numbers flagged (recommended),
   or recalibrate first (C2, GPU time unknown until C0; does not solve the
   transition until its second half).
2. Agents code-only; the draft's notation adopted only at the writing stage.
3. GPU, decided after C0: C1 now (recommended), C1 + C1b, C2 now, or
   neither.
4. Output: LaTeX + PDF in `code/reports/` with a markdown twin (recommended),
   or markdown only.
5. Scope: baseline machinery plus the fiscal layer's equations and terminal
   conditions (recommended), plus the dead-code inventory; experiment
   results excluded.

## Cost

Phase B 45–60 min wall clock (B6 reads every file); Phase C local 10–20 min
plus C0, minutes to tens of minutes; Phases D–E 1.5–2 h. C1 is two
transitions and two cross-sections of GPU time, sized from the 12:31 logs
and C0; C1b one more transition; C2 unsized until C0.

## How to start

Confirm HEAD is still `3784ae5` (`git log -1`); if not, re-run Phase A on
the diff. Run C0 and the local Phase C checks first; answer the five
decisions with C0's number in hand. Move `code/CLAUDE.md` to the scratchpad.
Launch the six Phase B agents in one message with `run_in_background: true`
and the questions above verbatim. When the last agent returns, restore
`code/CLAUDE.md` and confirm with `git status`.
