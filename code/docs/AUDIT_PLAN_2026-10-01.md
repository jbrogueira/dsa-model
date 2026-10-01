# Audit plan — model description, calibration, consistency audit

Written 2026-10-01. Status: **not started**; the five decisions at the end
are open.

## Object

Branch `trend-growth` at `342f3e1` (2026-10-01 16:12), config
`code/calibration_input_GR.json`, data sidecars `data/demography_GR.npz`,
`data/survival_GR.npz`, `data/europop2023_GR.npz`. The baseline is the
demographic transition from the measured 2023 cross-section with no policy
change. The fiscal layer (debt law, financing rules, terminal conditions) is
in scope as equations; the G and I_g experiment results are not, since none
exist under trend growth.

State of the object at the time of writing: the five SMM parameters, `A_tfp`
and the closure `O/Y` in the config were fitted at 12:31 on 2026-10-01 on
the single-solve calibration route. Since then `calibration.base_year_cohorts`
was switched on (`08833c3`, 15:24), the transition's aggregates were put per
living person (`a49e0be`, 15:36) and the survival double-count in
`_agent_weights` was fixed (`6362091`, 15:43). The code is current; the
numbers in the config and in `output/calibration_growth/*.tex` are not.

## Deliverable

One document, `code/reports/model_audit_2026-10-01.{tex,pdf}` with a
markdown twin, about 10 pages. Style: referee or auditor report — facts,
locations, consequences; no framing of what the paper should claim.

| Part | Content | Length |
|---|---|---|
| I. The model economy | Section order of McGrattan, Miyachi and Peralta-Alva (2019, `lit-review/McGrattan-et-al-Japan.pdf.pdf`, §3): demographics; portfolios and returns (who holds domestic capital, foreign assets and sovereign debt, at which rate); prices and policy; the household problem, working and retired; technology; government budget constraints; equilibrium and the detrended balanced-growth path. Equations are what the code does. Notation follows the draft's `docs/DSA-LSA model.tex` where the objects coincide. No code references in the body. | 4 pp |
| II. Calibration | Same layout as `code/reports/calibration_report.tex`: externally set; fitted jointly by SMM (targets, weights, which moment identifies which parameter); pinned outside the SMM (A_tfp, closure); demographics (2023 cross-section, EUROPOP2023 path, tail). Reuses the report's table bodies in `output/calibration_growth/`. Every number carries the run it came from and a stale flag where the code has moved since. | 2 pp |
| III. Audit | Summary verdict; findings ranked Critical / Major / Minor, each with the fact, the file:line, the consequence and what was verified; the six items of `OPEN_ISSUES_2026-07-30.md` with their status at HEAD; what could not be verified without a production-scale run. | 3 pp |
| Appendix | Code-to-equation correspondence (object, equation, file:line, both backends); the list of checks run and their outcomes. | 1–2 pp |

## Phases

### A — reconnaissance (done 2026-10-01)

Established: repo layout and the "what solves what" table in `README.md`;
the McGrattan et al. section order; today's commits and their messages; Step
0 of `TREND_GROWTH_PLAN.md`; the July open-issues list; the calibration
report template and the run outputs it reads; the function maps of
`olg_transition.py`, `calibrate.py`, `fiscal_experiments.py`,
`lifecycle_perfect_foresight.py`; the test classes.

### B — five context-free agents in parallel, read-only

Each agent reads source files, the config and the data sidecars only.
Forbidden: `docs/*.tex`, `code/docs/*.md` (B5 excepted), `code/code_report_2026-06-15.md`,
the model sections of `README.md` and `code/CLAUDE.md`, `TREND_GROWTH_PLAN.md`,
the assistant's memory. No agent is told what the others cover or what is
suspected; the questions below are classes of question, not findings. Each
returns equations with file:line pointers and a findings list that separates
"the code does X" from "X is inconsistent with Y".

| Agent | Files | Questions |
|---|---|---|
| B1 household | `lifecycle_perfect_foresight.py`, `lifecycle_jax.py` | Preferences; income, health and survival processes; budget constraints working and retired; every tax base; pension base and its indexation; UI; transfer floor; bequest receipt; the (1+g) on next-period assets; discounting with survival; terminal value; asset-grid bounds; the labour FOC. Where do the two backends differ? |
| B2 initial condition and cohorts | `calibrate.py` (`base_year_cross_section`, `base_year_cohort_survival`, `base_year_age_weights`, `compute_age_weights`), `olg_transition.py` (`solve_cohort_problems`, `_extract_cohort_path`, stitching and `_mit_baseline_cache`, `_cohort_survival_schedule`, `_survival_schedule_at_year`, initial assets, bequest path), `fiscal_experiments.py` (`_build_pre_transition_paths`), `normalize_A_tfp.py`, `pin_baseline_closure.py`, `run_fiscal_figures.py` (B_initial, K_g initial) | What is assumed at t = 0: which cohorts exist; the prices, taxes and survival each was solved against; where its t = 0 assets come from; what pins K_g(0), B(0), NFA(0), w(0). Is the object the SMM targets the same object as the transition's t = 0 — weights, survival, seeds, n_sim? Which of θ, A_tfp and the closure are computed on which route? |
| B3 aggregation, transition, terminal state | `olg_transition.py` (`_entrant_weights`, `_alive_fraction`, `_aggregation_weights`, `_build_population_weights`, `growth_factors`, `_growth_at`, `_compute_all_cross_sections`, `compute_aggregates`, `compute_government_budget`, `simulate_transition`), `fiscal_experiments.py` (`compute_debt_path`, `_balance_residual`, `_check_terminal_convergence`, `_extend_base_paths`, `_nfa_ca_paths`, `_correct_base_macro_nfa`), `reports/fill_report.py` (goods-market residual, flatness statistic), `build_demography_GR.py` | Per-living-person aggregation and the Γ_t in each stock recursion (B, K_g, NFA/CA, pension fund); the goods-market identity, including what happens to the assets of agents who die; the labour aggregate; timing of stocks; the government budget line by line and the closure; what cohorts born after T − 60 face beyond T; the tail (mortality held at 2100, entrant ramp, n_∞) and whether every terminal condition uses Γ_T; whether the baseline has a rest point for debt at all. |
| B4 calibration procedure | `calibrate.py` (moment functions, `_agent_weights`, `_compute_ss_aggregates`, both routes of `run_model_moments`, `smm_objective`, `calibrate`), `run_scale_loop.sh`, `run_step0_baseline.sh`, `normalize_A_tfp.py`, `pin_baseline_closure.py`, `reports/fill_report.py`, `calibration_input_GR.json` | What each target measures and on which population; which parameter identifies which moment; the SMM ↔ A_tfp loop and its convergence test; what `_derived.theta_metadata` says against the live code path; which recorded numbers are stale and why. |
| B5 known-issue status | May also read `code/docs/OPEN_ISSUES_2026-07-30.md` | For each of items 1–6: is the behaviour described still in the code at HEAD? Evidence by file:line. |

### C — numerical checks

Local, CPU (`source ~/venvs/jax-arm/bin/activate`, `JAX_PLATFORM_NAME=cpu`),
run without asking, durations checked first, anything over ~2 min in the
background with a monitor:

- `pytest test_olg_transition.py -k "TestDemographicPath or TestBaseYearCrossSection or TestTrendGrowthHousehold or TestTrendGrowthStocks or TestCohortBatchedSurvival"`
  (the cost-ratio test skips without a GPU).
- Spot checks at test-fixture scale from Python: Σ_j w_j S_j = 1 at several
  t; Γ_t from the living population against `growth_factors`; the survival
  diagonal of three cohorts against the sidecar; the single-solve and cohort
  routes agreeing under one shared schedule.

Production scale, GPU (Verda or the A100 recipe in memory):

- **C1, about 1 h:** one baseline transition at the current θ —
  `diag_ss_vs_transition.py jax 2000`, `check_a0_predetermination.py`,
  `reports/fill_report.py --run-baseline --n-sim 2000`. Gives the t = 0 gap,
  the A[0] check, the goods-market residual and the flatness column. Tests
  the machinery, not the fit.
- **C2, several hours:** `run_step0_baseline.sh` — recalibration on the
  cohort route, then the same checks. Gives current numbers for Part II.

What solves what: Parts I and III and the whole of Phase B solve nothing.
Part II at the stale θ solves nothing. Recalibration (`run_scale_loop.sh`)
solves only the base-year cross-section — on the cohort route, 60 cohorts
per education group in one batched solve, each simulated to its base-year
age — through `run_model_moments`, which `normalize_A_tfp.py` and
`pin_baseline_closure.py` also call; no transition. The transition is
solved only by C1 and by the second half of C2.

### D — verification and consolidation

Re-read the code at every Critical and Major finding; discard what does not
reproduce; merge duplicates; rank. Only then compare Part I against the
draft's `model.tex` and the settled decisions in `TREND_GROWTH_PLAN.md`, and
book each disagreement as a finding ("the draft says / the code does"). No
edits to code, config or the draft.

### E — write and compile

LaTeX in `code/reports/`, `latexmk -pdf`; markdown twin alongside. Style
per `~/.claude/CLAUDE.md`.

## Decisions open

1. Audit the code at HEAD with the stale θ, numbers flagged (recommended),
   or recalibrate first (C2, several GPU hours; does not solve the
   transition until its second half).
2. Agents code-only; the draft's notation adopted only at the writing stage.
3. GPU: C1 now (recommended), C2 now, or neither.
4. Output: LaTeX + PDF in `code/reports/` with a markdown twin (recommended),
   or markdown only.
5. Scope: baseline machinery plus the fiscal layer's equations and terminal
   conditions (recommended); experiment results excluded.

## Cost

Phase B 30–45 min wall clock; Phase C local 10–20 min; Phases D–E 1.5–2 h.
C1 adds about an hour of GPU time.

## How to start

Confirm HEAD is still `342f3e1` (`git log -1`); if not, re-run Phase A on
the diff. Answer the five decisions. Launch the five Phase B agents in one
message with `run_in_background: true` and the questions above verbatim;
start the Phase C local tests in the same message.
