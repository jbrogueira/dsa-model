# Fiscal Experiments — Status Handoff

Last updated: 2026-07-14 (G+Ig rerun at r_B=0 under the K_g calibration; §4 + calibration appendix synced and pushed — see `## Session 2026-07-14`). **STATUS:** current calibration is the K_g-activated one (θ: ν=20.656, β=0.9419, τ_p=0.1978, ρ_pens=0.1697, m=0.0845; A_tfp=1.6061; closure −0.096295). The draft's §4 numbers come from `output/fiscal_test_kg_rB0/` (r_B=0); the r_B=0.021 counterpart is `output/fiscal_test_kg/`. Open: eval loop skips `nfa_constrained`; `g_aggregates_note` stale (regenerate via `/fiscal-note`).

---

## Session 2026-07-14: §4 finished at r_B=0 — kg-calibration rerun on V100, draft + calibration appendix synced, eval checks fixed

The r_B=0 G+Ig set under the 2026-07-10 calibration — the numbers the draft's §4 was written for. Results in `output/fiscal_test_kg_rB0/`; config `calibration_input_GR_rB0.json` (= working config with the single edit `prices.r_B → 0.0`; `pin_baseline_closure.py` reproduces −0.096295 at r_B=0, re-verifying the closure's r_B-invariance under the new calibration — no re-pin, no recalibration).

- **Results (n_sim=2000, T=60+20, all τ_l scenarios converged):** baseline terminal B/Y at T_bal **0.7423** (vs 3.8863 at r_B=0.021 — with zero debt service the ≈1.5%-of-Y primary surplus amortises debt); Δτ_l debt-target G **+3.07pp** / Ig **+2.83pp**, NFA-target G **+3.36pp** / Ig **+3.19pp**; Ig cumulative multiplier **0.829**, G **0.0**. Acceptance gates: baseline & debt-financed real paths (Y,C,L,A) bit-identical to `fiscal_test_kg` (r_B is government-side only; 0.0 max rel diff); closed-form B[t+1]=B[t]+PD[t] holds at 5e-16.
- **Run mechanics (for the next GPU run):** peak resident ≈29–30 GB. The Verda 6vCPU/23GB V100 variant OOM-killed the first launch at 22.9 GB (SIGKILL → no traceback, and the block-buffered `nohup` log hid it — relaunch with `python -u`); with a 16 GB swapfile it completed but swap-thrashed (D-state, ~3% CPU efficiency): **949.7 min vs ~50 min** on the July-10 instance. Use a ≥32 GB-RAM variant. Monitoring gotcha: `pgrep -f <script>` over ssh matches the probing shell itself — bracket the pattern (`pgrep -f "[r]un_fiscal_figures"`).
- **Draft synced and pushed (Overleaf `2b5e87e`):** §4.1 parameters (I_g stated as the level δ_g·K_g0 with I_g/Y=0.0353 at the calibration SS; O=−0.096; baseline surplus ≈1.5% of Y; terminal B/Y 0.74); G-shock numbers refreshed (Δτ_l +3.1/+3.4pp; C −2.5/−2.7%; A −1.4%); Ig subsection filled — four figures live plus results paragraphs; all 8 PNGs in `docs/output/fiscal_test/` replaced with the kg_rB0 vintage. Calibration appendix + `data.tex` at the 2026-07-10 θ: A=1.606 (Y_ss=1 normalization stated), K_g block active (K_g/Y=0.745 IMF ICSD, δ_g=0.0474, η_g=0.05), new †† footnote on r_B (calibration at the measured 0.021, experiments at 0), workflow gains the SMM↔A-normalization fixed point + closure re-pin, moments table refreshed (Q=7.4e-9; Y/L 3.77; wealth Gini 0.279; zero-wealth 2.5%). Full draft compiles clean. `m`'s table description changed to "medical-cost scale (model units)" — the old "share of mean income" units predate the Y_ss=1 normalization; verify.
- **`eval_fiscal_results.py` drift fixed** (was 10 spurious FAILs on both accepted runs): `chk_debt_accumulation` accrues at r_B (from JSON `params` or `--config` `prices.r_B`; pre-r_B JSONs without `--config` fall back to r and fail spuriously — pass `--config`); `chk_bisection_target` compares B/Y at `T_balance` against the same shock's baseline `B_gdp_path[T_balance]` (the closure design since 2026-06-15) instead of the static config 1.64 at end-of-extension; the shock-path check is mode-aware (`chk_shock_line_path`: ratio for config-run G, level for Ig with `eta_g≠0`) and a `shock_ig_path` check was added. `run_fiscal_figures.py` now embeds `r_B`, `shock_mode_G`, `shock_mode_Ig` in the results JSON. Both `fiscal_test_kg` and `fiscal_test_kg_rB0`: **0 FAIL / 24 WARN / 104 PASS** (WARNs are the known classes: ui/Y above target, HAM tax-approximation lines, labor-response sign under income effects, terminal S_pens/NFA drift).
- **Open:** the eval main loop iterates baseline/debt_financed/tax_financed only — `nfa_constrained` blocks are never checked; `output/fiscal_test/g_aggregates_note.{md,pdf}` predates all current runs — regenerate via `/fiscal-note` from `fiscal_test_kg_rB0/fiscal_results.json`.

---

## Session 2026-07-10: public capital (K_g) activated — IMF K_g/Y, Y_ss=1 recalibration on V100, G+Ig run

Full activation of the public-capital channel per `PUBLIC_CAPITAL_KG_PLAN.md`; committed in `17ce0a0` (code + data + decisions) and `dd5be23` (calibrated config + results). Detail in the plan doc §§4–7 and memory.

- **Decisions:** K_g/Y pinned to IMF 0.745 (ICSD, Greece general government, 2019 — latest; dataset ends 2019); I_g/Y = 0.0353 (DATA-sheet mean 2015–19, = ICSD 3.52%); δ_g = 0.0353/0.745 = 0.04738255; reference year stays 2023. Data: `data/IMF_ICSD_GR.csv` (force-added past `*.csv` ignore), inventory §1.12.
- **Code:** with `eta_g ≠ 0`, `run_fiscal_figures.py --config` passes baseline I_g as a constant **level** δ_g·K_g (K_g flat in baseline) and a **level** Ig-shock delta 0.02·Y(0); `_apply_shock` gained a per-line mixed mode (I_g level alongside G/defense/other ratios when `base_paths` has no `I_g_over_Y` key). `check_a0_predetermination.py` gained an Ig/`eta_g≠0` case — all four cases (τ_l, Ig × numpy, jax) pass with |diff| exactly 0.0.
- **Recalibration at Y_ss=1** (new `normalize_A_tfp.py` + `run_scale_loop.sh`, run on a Verda V100): A_tfp = 1.60606497 (Y_ss = 1.0005, so K_g = 0.745 = K_g/Y); θ = ν 20.6558 / β 0.941873 / τ_p 0.197779 / ρ_pens 0.169694 / m_good 0.084453 (all interior); closure re-pinned −0.096295 (SS primary balance = +0.0195 target). All 5 targets at 0.0% (Q = 7.4e-9), report `calibration_GR_20260710_153659.md`. Gotchas hit and fixed: (i) `calibrate.py` writes `_derived.theta` **only on formal convergence** — the loop now aborts if the write-back line is absent; (ii) the old `m_good` upper bound 0.1 was in pre-normalization units and binds at Y_ss=1 — raised to 0.2; (iii) the SMM↔A_tfp alternation contracts slowly (ratio ≈0.45/round) — Aitken-extrapolating the A_tfp sequence (seed 1.6069) converged it in one further round.
- **G+Ig results at r_B=0.021** (`output/fiscal_test_kg/`, n_sim=2000, T=60+20, ~50 min on 1×V100): Ig cumulative multiplier **0.829** (production channel active: K_g → 1.167 long run, +2.3% on Y) vs G **0.000**; Δτ_l debt-target G +3.03pp / Ig +2.74pp, NFA-target G +3.17pp / Ig +2.91pp; baseline terminal B/Y at T_bal = 3.8863.
- **Draft (Overleaf `9542542`):** §4 gained the Public Investment Shock subsection — level-path design and transmission mechanism only, **no numbers** — because §4 is written at r_B=0 while these results are r_B=0.021. Figure blocks stay commented; the `ig_*.png` currently in `docs/output/fiscal_test/` are stale pre-K_g vintage.
- **Open:** rerun G+Ig at `prices.r_B=0` under this calibration (fresh V100: ~15 min setup + ~50 min; only the τ_l scenarios need live solves — baseline/debt-financed debt paths are closed-form from the primary-deficit path), then update §4.1's parameter values (I_g level 0.0353, O = −0.0963), refresh the G-shock numbers, fill the Ig subsection, replace the draft figures.

---

## Session 2026-07-09: experiment-design section expanded — baseline transition, shock timing, financing mechanics

Paper-only session (docs submodule, pushed to Overleaf `602882b`, `cb4b180`). No code changes.

- **§4.1 Baseline** now describes how the baseline transition is solved. Initial condition: cohorts alive at t=0 simulated from birth under constant prices/policies, each with its birth-year survival diagonal; B_0 = 1.64·Y_0; O_t pinned at the initial steady state. Drivers: demographic composition (historical-survival cohorts replaced by last-table cohorts; output moves < 1% over the horizon) vs. the fiscal drift (primary surplus ≈ 1.9% of Y, below the debt-stabilising level r_B·B_t/Y_t ≈ 3.4%; B/Y 1.64 → 3.40 at T). Terminal point: real allocation ≈ stationary at T (= lifecycle length, pre-transition cohorts have exited); fiscal position is not — the financing closures target the baseline's own value at T. Numbers verified against `output/fiscal_test/fiscal_results_G_20260616.json`.
- **§4.1 Shock** now states the MIT timing (unanticipated at t=0; date-0 wealth cross-section identical across scenarios, matching the verified A[0] predetermination), the own-output share convention (ratio mode: G^cf = 0.15·Y^cf), and the channel (G enters neither preferences nor production).
- **§4.1 Financing regimes** now separates the debt case (single transition solve, behaviour unchanged) from the τ_l case: modified regula falsi on the scalar Δτ^l; each residual evaluation is a full transition solve; targets pinned to the drifting baseline's terminal values (tolerance 1e-3 on the terminal ratio); condition imposed at T only — drift resumes post-horizon (tax-financed B/Y 3.41 at T → 4.75 at T+20).
- **Phrasing fix**: the debt-drift statement is now conditioned on the primary balance falling short of r_B·B_t/Y_t, not on "r_B > g" alone; g dropped (undefined in the paper; baseline output is trendless).
- **Overleaf-side edits pulled** (user): abstract/introduction files renamed to the "DSA-LSA 2607" prefix; fiscal-multiplier paragraph and prices-sanity figure removed from the experiments section (no dangling references).

---

## Session 2026-07-07: draft synced to code; moments report at fixed θ; interest line at r_B

Paper-draft update (docs submodule, pushed to Overleaf `48f0d5a`..`3a8649d`) plus two code-side items.

- **Moments report at the current θ** — `output/calibration/calibration_GR_20260707_095127.md`, produced by a single objective evaluation at `_derived.theta` (no optimizer; JAX, n_sim=10,000, seed 42, 33.8 s). All five targets match to 4 decimals (Q = 1.5e-8). Untargeted under the new θ: C/Y 0.601, wealth Gini 0.287, income Gini 0.376, p90/p10 5.43, zero-wealth 2.7%, u 14.3%, total tax revenue/Y 0.345. This is the source for the draft's moments table (`DSA-LSA calibration.tex`, Table `tab:baseline-moments`). The 2026-06-14 recalibration itself only left a log (`recalib_20260614.log`); this report fills the gap.
- **Figure interest line fixed to r_B.** `run_fiscal_figures.py` plotted `interest_payments = r · B` at the capital return (4%) while `compute_debt_path` accumulates B at `r_B` (2.1%) — display-only inconsistency, the law of motion was already correct. Now `r_B · B` (falls back to `r` when `olg.r_B` is unset, preserving the hardcoded test branch). `regen_fiscal_figures_from_json.py` gains `--r-b <rate>` to recompute the line from stored `B/Y · Y` paths when re-plotting JSONs written before the fix. G figures regenerated from `fiscal_results.json` with `--r-b 0.021`; only `g_fiscal_decomp.png` changed (the other three are interest-free and reproduce bit-identically). `g_aggregates_note.md` still describes the old 4% line — regenerate via `/fiscal-note` after the next run.
- **Draft sync** (see the docs submodule log for detail): model section equations aligned with the code (multiplicative tax wedge, permanent effect e^α + education strata, m(j) age profile, career-average pension, Rev^l base incl. UI and pensions, defence + other-net GBC lines, debt at r_B, bequests to newborns, cohort life-table survival); calibration appendix at the recalibrated θ with the new moments table; experiments section rewritten to the current G-shock runs (three financing regimes, terminal-stock closures, ratio-mode shock, B_0/Y = 1.64, T = 60); the I_g section is commented out pending a rerun.

---

## Session 2026-06-17 (b): fiscal figures — NFA/Y + A/Y panels, /Y GBC decomposition, base_macro NFA fix, fiscal-note skill

Plotting/diagnostics layer + one accounting fix. Figures re-plotted from the existing `fiscal_results.json` via the new `regen_fiscal_figures_from_json.py` (no re-simulation); the underlying data is unchanged, so "redo fiscal figures under new θ" stands.

- **`base_macro['NFA']` correction.** `FiscalScenarioResult.base_macro['NFA']` was the partial `A − K_domestic` (uncorrected) while `cf_macro['NFA']` was the full `A − K_domestic − B`; a baseline-vs-cf NFA comparison was off by the debt level (≈3.35×Y). New `_correct_base_macro_nfa()` applies the `− B_base` correction in `run_debt_financed`, `run_tax_financed`, and `run_nfa_constrained`. The existing figures were unaffected (they read `cf_macro`/`NFA_path`); the stored result object is now consistent. 39/39 fiscal tests pass.
- **Generic `<line>_gdp` ratio key in `compare_scenarios`** (any `cf_macro`/`cf_budget` line ÷ Y). Macro overview adds **NFA/Y** and **A/Y** panels (9-panel grid); fiscal decomposition plots primary deficit + the four taxes + UI + pensions as **/Y**, leaving G/I_g/defense/other-net/interest as levels.
- **`regen_fiscal_figures_from_json.py` (new):** rebuilds lightweight result stand-ins from a saved JSON and calls the existing `compare_scenarios`/`debt_fan_chart`. No simulation.
- **`/fiscal-note` project skill (`.claude/skills/fiscal-note/`):** reads the JSON, cross-checks the four figures, writes a one-page-per-scenario note (baseline absolute; counterfactuals as deviations from baseline) and renders PDF. Produced `output/fiscal_test/g_aggregates_note.{md,pdf}` (G shock; baseline + 3 closures). Debt-financed G has zero real effect (SOE, fixed r) — only B/Y and NFA/Y diverge; τ_l closures: Δτ_l = +3.17 pp (debt target), +3.47 pp (NFA@T).

---

## Session 2026-06-17: fiscal run-time — Illinois root finder, JAX backend, equivalence validation

Runtime-only work; model results unchanged (verified equivalent across backends).

- **`run_tax_financed` root finder: pure bisection → Illinois (modified regula falsi).** Keeps the verified opposite-sign bracket (same robustness, same converged Δτ to within `tol`) but steps via secant interpolation, with the retained-endpoint-residual halving that guards against the regula-falsi stall, and a midpoint fallback if the secant step would leave the bracket. Iterations drop from ~10–15 to 1 interior step on the test scenario (residual ~1e-16; the τ_l balance residual is near-linear in Δτ under SOE — r exogenous, w fixed when K_g doesn't move). Both τ_l figure variants (`terminal_debt_gdp`, `terminal_nfa_gdp`) use this path. 39/39 fiscal tests pass.
- **`validate_backends.py` (new): NumPy-vs-JAX equivalence.** Runs the baseline transition kernel (cohort solve + MC simulation + aggregation + government budget — the only backend-dependent computation; the fiscal layer on top is pure NumPy) under both backends and compares aggregate paths. The backends use different PRNGs (MT19937 vs ThreeFry) at the fixed seed 42 (`simulate_transition` exposes no seed), so aggregates differ by Monte-Carlo noise ∝1/√n_sim, not machine precision; a per-path n_sim-scaling test separates sampling noise from a systematic solver discrepancy. Result: PASS (gap ratio 0.558 ≈ √(300/1200)=0.5; deterministic r/w/K_g exact to 0).
- **Measured: JAX ~25× faster than NumPy on CPU** (66 s vs 1601 s, one baseline, quick grids) — serial cohort loop vs `vmap`. Run the fiscal figures with `--backend jax`; GPU is additional upside, not a prerequisite. On the GPU box confirm the JAX device printed by `validate_backends.py` says `cuda` and that `JAX_PLATFORM_NAME` is not pinned to cpu.
- **`run_fiscal_figures.py`: total wall-clock printout** at the end, tagged backend/shock/n_sim.

---

## Session 2026-06-16 (b): terminal NFA target as a Type B closure

Added `balance_condition='terminal_nfa_gdp'` (field `target_nfa_gdp`) — the external-balance analogue of `terminal_debt_gdp`. A single τ_l (or other Type B instrument) is bisected so full NFA/Y at the balance period T_bal equals a target; the interior NFA path is free. Residual: `target_nfa_gdp − NFA[T_bal−1]/Y[T_bal−1]` with full NFA = `macro['NFA'] (A − K_dom) − B`. `_balance_residual` gained an `NFA_partial` arg; `run_tax_financed` now keeps the simulation macro to pass it.

This is the convergent alternative to the Type C pointwise band: one instrument, one terminal target, root-find like the debt closure (verified: Δτ_l converges, terminal NFA/Y hits target to ~1e-4). Use it when you want NFA returned to a level at T_bal rather than tracked every period.

`run_fiscal_figures.py`: the 4th G/Ig scenario (`scn_*_nfa`) is now this terminal-NFA Type B closure, pinned at runtime to the **baseline terminal NFA/Y** (analogue of pinning the debt target to baseline terminal B/Y). It replaced the earlier Type C pointwise-band scenario. Curves per figure: baseline / debt / τ_l(debt target) / τ_l(NFA@T). Output filenames unchanged; `fiscal_results.json` gains an `nfa_constrained` block. Full fiscal suite 39/39 passes.

Type C (band around baseline, `nfa_limit`/`ca_limit`) is unchanged and still available; it just isn't the scenario wired into the figures.

---

## Session 2026-06-16: Type C NFA constraint → band around baseline path

The NFA/CA constraint in `run_nfa_constrained` was an absolute one-sided floor (`NFA_t ≥ −nfa_limit`). It is now a **band around the baseline path**: `NFA_t ≥ NFA_base_t − nfa_limit` (and `CA_t ≥ CA_base_t − ca_limit`), with `nfa_limit`/`ca_limit` reinterpreted as the band half-width; `0.0` enforces exact baseline tracking. This avoids the absolute floor rendering the no-shock baseline itself infeasible (baseline NFA drifts to ≈ −426% of Y over the horizon from −132% at t=0).

Changes (`fiscal_experiments.py`):
- `_check_nfa_violation` takes optional `NFA_base`/`CA_base`; floor is the baseline path minus the limit (absolute `−limit` retained as fallback when no baseline passed).
- `run_nfa_constrained` builds the baseline **full** NFA/CA reference (partial `A − K_domestic` minus the baseline no-shock debt path) and a `_full_nfa_ca` helper.
- Fixed a latent inconsistency (dormant — no limit was ever set in production): Step 1 checked the full NFA while the Mode I/II bisection checked the partial NFA. All checks now use full NFA (net of that iteration's debt).

Behavior of the two modes (unchanged in mechanism, clarified in `solver_architecture.md`):
- **Mode I (`financing='debt'`)**: the specified ΔG is treated as a **ceiling**, not delivered in full — the bisection finds the largest fraction `s·ΔG` (s∈[0,1]) that stays in the band. `nfa_limit=0 ⇒ s=0` (any deficit lowers NFA below baseline). This answers a feasibility question, not a financing one.
- **Mode II (`financing≠'debt'`, e.g. `tau_l`)**: the full ΔG is delivered; the financing instrument rises to the smallest value that restores the band. `nfa_limit=0 ⇒` instrument rises until NFA tracks baseline exactly.

Verified (small test economy, both modes): the counterfactual NFA lands exactly on the band edge; full fiscal suite 39/39 passes. Not yet wired into `run_fiscal_figures.py` (no Type C scenario constructed for G/Ig); no `nfa_limit`/`ca_limit` calibrated in `calibration_input_GR.json`.

---

## Session 2026-06-14: labor-supply FOC audit + fix

The one-period aggregate-labor spike at transition t=1 in the tax-financed counterfactual (flagged from the fiscal figures) was diagnosed, then audited independently (a macro-theorist deriving the correct conditions from the budget/utility primitives + a code-extractor reporting the implemented formulas, both blind to the hypothesis), and fixed.

### Root cause(s)

The household problem has exactly one FOC (labor-leisure; assets are grid-enumerated, retirement is a discrete max, UI/pension/schooling are accounting). It had five defects:

1. **Solver non-convergence (the spike).** The 2-iteration Newton (`newton_labor_jax`, `_solve_labor_newton`) seeded its guess where the implied `c(l)` is negative → clamped to 1e-10 → derivative explodes → iteration frozen at a spurious non-root. Hours under-stated by up to ~0.4; more iterations did not help. At young, borrowing-constrained ages this gave erratic labor; MIT alignment turned each cohort's onset wobble into the aggregate t=0 dip / t=1 overshoot. Pre-existing (persisted at n_alpha=1; lives in the per-agent sim, not the aggregation fixed on 2026-06-12).
2. **Tax wedge (both backends).** FOC `net_wage` used additive `(1−τ_l−τ_p)`; the budget taxes labor income net of payroll, so the correct marginal after-tax wage is multiplicative `(1−τ_p)(1−τ_l)`.
3. **κ(t) omitted (NumPy only).** NumPy FOC `net_wage` dropped the wage-age profile κ(t) its own budget includes; JAX included it. Made NumPy ≠ JAX.
4. **Missing 1/(1+τ_c) (both backends).** Correct FOC `ν·l^φ = c^{−γ}·MW/(1+τ_c)`; solver targeted `ν·l^φ = c^{−γ}·net_wage` (the `dc_dl=net_wage/(1+τ_c)` term entered only Newton's step, not the root). Biased hours high by ~(1+τ_c)^{1/φ}≈1.12.
5. **Spurious retired/unemployed disutility (both backends).** Retired/unemployed carry `l=1.0` (to zero `delta_budget`), so the objective subtracted `ν/(1+φ)` of disutility they don't incur — biasing value levels feeding the retirement-window choice and EV.

### Fix (applied; both backends)

Corrected FOC `ν·l^φ = c^{−γ}·MW/(1+τ_c)`, `MW = w·κ(t)·y·h·e^α·(1−τ_p)(1−τ_l)` (flat; progressive keeps `(1−τ_p)`), `c(l)=c_guess+MW·(l−1)/(1+τ_c)`, `delta_budget=MW·(l−1)`. Solved by **projected-Newton** on the monotone residual `G(l)=ν·l^φ·(1+τ_c)−c(l)^{−γ}·MW`, bracketed to `c(l)>0`, Newton steps clipped into the bracket (corners snap to the bound; interior gets Newton speed). Vectorized/branchless JAX (`solve_labor_robust_jax`, 8 iters → ~1e-11); scalar with early-break NumPy (`_solve_labor_newton`). Disutility applied only to working-age employed states.

### Verification

- Both solvers match an independent bracketing root-finder for the corrected FOC to ~1e-14.
- A[0] predetermination still exact (0.0) on both backends.
- `olg_transition.py --test` runs on both backends; they now agree to MC-noise (~0.4%) where the κ omission previously separated them.
- The t=1 spike is gone: the per-age t=1 labor deviation, previously a uniform +0.001…+0.0016 across all ages, is now mixed-sign MC noise (±0.0004).

### Recalibration (done, same session)

Defects 2–4 change labor supply, so the SMM was re-run under the corrected solver (`calibrate.py --config … --backend jax`, n_sim=10000, Nelder-Mead). Converged to obj ≈ 2e-8 (MC floor), all five moments matching targets to ≤0.01%:

| param | new (corrected FOC) | old (pre-fix) |
|---|---|---|
| ν | 36.907 | 28.670 |
| β | 0.94317 | 0.98541 |
| τ_p | 0.19776 | 0.19786 |
| ρ_pens | 0.16629 | 0.16122 |
| m | 0.04277 | 0.04162 |

Moment fits at new θ: average_hours 0.4100/0.41, A/Y 4.0004/4.0, tax_p/Y 0.1300/0.130, pensions/Y 0.1600/0.160, health_gov/Y 0.0540/0.054 (max gap 0.01%). The big moves (ν +29%, β −4.2pp) are the FOC corrections propagating: the corrected wedge + 1/(1+τ_c) changed labor supply, so ν rose to still hit hours and β fell to still hit A/Y. New θ written to `calibration_input_GR.json._derived.theta`.

Mechanics note: I stopped the optimizer once θ was frozen to 5 sig figs (it was grinding the simplex toward an absolute xatol=1e-6 on ν≈37, ~7 sig figs, which MC discretization makes slow); θ was parsed from the converged log line, written to the config, and re-validated by an independent moment eval (table above). The script's auto-write fires only on scipy `success`, which the early stop bypassed — hence the manual (but validated) write.

### Still stale — next steps

The fiscal figures in `output/fiscal_test/` predate the FOC fix and the recalibration; re-run with `run_fiscal_figures.py --config … --shock both --backend jax`. (The baseline closure has been redone under the new θ — see the next section.)

---

## Session 2026-06-14 (cont.): baseline closure pinned at the initial steady state

### Design change

`other_net_spending_over_Y` is now pinned at the **initial steady state**, not measured off a transition. It is a structural constant that makes the initial-point government budget consistent with the data primary balance; the baseline transition takes it as given and produces a time-varying primary-deficit path as a model output. The closure is no longer re-tuned to force the transition's t=0 primary balance to the target.

Procedure (interest excluded throughout, matching the transition's primary balance; `pb_house` is the household-side SS balance from `compute_fiscal_ratios`):

```
pb_house = (tax_revenue − pension − ui − gov_health) / Y           [stationary SS]
s_SS     = pb_house − (G_over_Y + I_g_over_Y + defense_over_Y)      [full, other=0]
other_net_spending_over_Y = s_SS − primary_balance_target_over_Y
```

Implementation:
- `compute_fiscal_ratios` (calibrate.py) now also returns `primary_balance_full_over_Y` (nets out G, I_g, defense, other) and `closure_other_over_Y` (the pin above). Adds keys only — no existing equation/moment changed.
- `primary_balance_target_over_Y = 0.0195` (Greek 2023) is a config key in the `fiscal` block, replacing the previously hardcoded target.
- `pin_baseline_closure.py` (replaces `measure_baseline_closure.py`) runs ONE stationary lifecycle solve (no transition), reuses `run_model_moments` + `compute_fiscal_ratios`, prints the pin, and writes it under `--write`.

Result under the new θ: SS household primary balance/Y = +0.1184; (G+I_g+defense)/Y = 0.19; full SS primary surplus (other=0) = −0.0716; pinned **`other_net_spending_over_Y = −0.091122`** (was −0.0889, the transition-t=0 value). The small change is the SS-vs-t=0 cross-section gap below.

### Why SS and transition-t=0 differ (for future reference)

**Cross-section (SS vs transition t=0).** Two channels, not only demographics:
1. *Demographics* — age-structure weights (fertility / population growth) and the survival schedule. The SS uses stationary weights + the calibration's fixed survival; t=0 uses births-only weights + per-cohort data survival. Differs whenever these don't coincide.
2. *History of predetermined wealth* — agents alive at t=0 accumulated assets over their own life histories under whatever pre-t=0 price/tax/survival paths the transition assumes. If those past paths (or the t=0 prices/taxes) differ from the calibration's stationary values, the t=0 asset/consumption distribution differs from the stationary one even with identical demographics.

If demographics, survival, and all prices/taxes are stationary and equal across the two (and pre-t=0 history is the same stationary equilibrium), they coincide up to Monte-Carlo noise.

**meanY vs Y[0].** Not numerical error. The baseline transition is non-stationary (fertility path + longevity improvement + per-cohort data survival → evolving population and capital/labor), so Y(t) is a genuine path and its time-average ≠ its initial value. The n_sim=50 warmup adds minor MC noise on top, but the substantive gap is the non-stationarity.

---

## Session 2026-06-14 (cont. 2): exogenous spending lines as GDP shares of Y(t)

### Change

The exogenous fiscal lines (G, I_g, defense, other-net closure) were previously fixed **levels** `ratio × meanY` (constant across the transition, sized off an n_sim=50 warmup mean). They are now **fixed shares of Y(t)**: the SS-calibrated ratio is passed through and the budget forms `level = ratio · Y_path[t]` using each run's own realized output. Decisions taken: each scenario indexes to its **own** Y(t) (true fixed-share-of-GDP; in tax-financed runs the spending levels respond to the tax-induced output change, a behavioral change to the experiment, not a normalization); the G/I_g shock magnitudes are **2% of Y(t)** each period; `B_initial = B_over_Y · Y(0)`; **gov_health is unchanged** (it stays a real per-agent medical-goods cost, so its GDP share drifts off 0.054 along the path — a real resource cost, not an imposed share).

This is the consistent counterpart to the SS-pinned closure: `other_net_spending_over_Y` is a ratio, so applying it as `ratio · Y(t)` (not `ratio · meanY`) keeps the share at its pinned value.

### Items audited

- **Now GDP-share (changed):** G, I_g, defense, other-net, and the G/I_g shock deltas; `B_initial` uses Y(0).
- **Already move with the economy (unchanged):** taxes (rates fixed; revenue is the sum of per-agent payments), pensions (replacement × past income), UI (replacement × wage), bequest tax, debt_service = `r_B · B(t)`.
- **Real cost, not a share (deliberately unchanged):** gov_health.

### Implementation (additive; level-path API preserved as fallback)

- `OLGTransition`: `simulate_transition` takes `G_over_Y/I_g_over_Y/defense_over_Y/other_net_over_Y` (scalar or `(T,)`), stored as `_active_*_over_Y`; `compute_government_budget` uses `ratio · Y_path[t]` when a ratio is set, else the level path. `I_g_over_Y` raises when `eta_g != 0` (I_g→K_g→Y simultaneity needs a fixed point). The lines have no household/production/NFA coupling at `eta_g=0`, so this is exact and single-pass; each scenario's `self.Y_path` makes it scenario-specific automatically.
- `fiscal_experiments._apply_shock`: ratio branch (gated on `*_over_Y` in `base_paths`) composes ratio cf entries (baseline ratio + ratio shock) and leaves the level paths None; `_run_one_simulation` forwards the ratio kwargs.
- `run_fiscal_figures.py --config`: passes ratios, `delta_G=delta_Ig=0.02` (ratio), `B_initial = B_over_Y · Y(0)`; the hardcoded test branch stays in level mode.

Verified: ratio mode gives G/Y exactly the target share every period; level mode unchanged; the `eta_g≠0` guard fires; `test_fiscal_experiments.py` 39/39 pass.

### Still stale — next steps

Re-run the fiscal figures under the new θ + SS-pinned closure + GDP-share spending: `run_fiscal_figures.py --config calibration_input_GR.json --shock both --backend jax`. The outputs in `output/fiscal_test/` predate all three.

---

Prior (2026-06-12): SS-vs-transition gap RESOLVED — a multi-agent code audit found four bugs/mismatches (K/L/C unpack swap since 2026-03-04; L-units 1/w convention; JAX batched α inconsistency; MIT-stitching staleness), all fixed and verified. Item ratios match the calibration SS to ±0.6%; closure was reset to `other_net_spending_over_Y = −0.0889` (now stale again after the 2026-06-14 FOC fix). Detail below.

---

## Session 2026-06-12: code audit of the SS-vs-transition gap (4 independent auditors + 3 adversarial verifiers; no runs)

Audit design: two blind formula-extraction agents (one per pipeline), one solver-input parity agent, one free-range hunter — none shown prior hypotheses, run results, docs, or git history. Surviving candidates were each re-verified by a fresh agent instructed to refute. Findings, in order of importance:

### Bug 1 — K/L/C unpack swap in `simulate_transition` (since `9496ee1`, 2026-03-04)

`_aggregate_capital_labor_njit` returns `(K, C, L)` (`olg_transition.py:838`), but `simulate_transition` unpacks `K_path[t], L_path[t], C_path[t] = ...` (`olg_transition.py:2072`). So **`L_path` carries aggregate consumption and `C_path` carries the wage-valued labor aggregate**; `Y_path`, `K_domestic`, and `NFA_path` are all built from the consumption aggregate (`olg_transition.py:2090–2100`). `compute_aggregates` (`olg_transition.py:1684–1691`) handles the same return correctly (`K, C, L = ...; return K, L, C`); the bug entered when `9496ee1` inlined the njit call for speed but kept the `compute_aggregates`-style unpack order. Affects every `simulate_transition` since 2026-03-04 (both backends): fiscal figures, the 2026-05-28 and 2026-06-11 closure measurements, the SS-vs-transition diagnostics. Found by an adversarial verifier while refuting the 1/w claim below.

### Bug 2 — JAX batched α-fixed-effect inconsistency (transition JAX backend only)

`_solve_cohorts_jax_batched` solves only the `alpha_mult=1.0` policy and stores it with a singleton α axis (`olg_transition.py:472–476, 545–548`), while `_simulate_cohorts_jax_batched` draws per-agent α indices over the full `n_alpha=5` grid and scales wages/UI/pensions by `exp(α)` (`olg_transition.py:688–701, 748, 768`). The kernel lookup `a_policy[alpha_idx, ...]` (`lifecycle_jax.py:661–663`) clamps the out-of-bounds index to 0 (JAX gather semantics, verified with a minimal snippet), so all agents follow the α-neutral policy while incomes carry heterogeneous multipliers. No gate, fallback, or repair path exists. Affected: every `OLGTransition` run with `backend='jax'` and `n_alpha>1` (the GR config). Not affected: NumPy transitions; the standalone `LifecycleModelJAX` used by the SMM calibration (loops all α correctly) — **the recalibrated θ is clean**.

### Convention mismatch — L units (exact 1/w wedge, survives even after Bug 1 is fixed)

`compute_fiscal_ratios` converts the aggregated wage bill to efficiency labor (`L = agg['labor_income'] / w`, `calibrate.py:1260`, also `:570`) before `Y = A·(K_over_L·L)^α·L^(1−α)`. The transition never divides by w: the per-agent `effective_y_sim` is wage-valued (`w·κ(j)·y·l·exp(α) + UI`, `lifecycle_perfect_foresight.py:1174–1175`) and flows into `L_path` → `K_domestic` → `Y_path` unscaled. Y is linear in L on both sides with the same coefficient, so after fixing Bug 1 the transition item/Y ratios would still be exactly 1/w (= 0.787 at this config, w = 1.2706) times the SS ratios — a uniform −21.3%. Reconciling requires choosing one convention (equation-level change; not applied).

### Verified small / excluded

- **Survival source**: SS uses the static 2020 table everywhere; the transition gives each cohort its calendar diagonal clamped to [1961, 2023] (cohort born at t=0 gets 2020 at age 0, then 2021/2022/2023-clamped; cohorts born t≥3 get pure 2023). Real but quantitatively small: diagonal cumulative survival to age 59 is 0.504 (oldest cohort) vs 0.523 (2020 table); cross-section age-share differences ~0.3% total variation. Cannot explain a ~12% gap.
- **Population-weight normalization** (births-only vs survival-inclusive, alive-conditional vs all-n_sim means): cancels in item/Y ratios because Y is linear in L — confirmed independently by two agents. The 2026-06-09 part-1 correction stands.

### Secondary findings (recorded, not pursued)

- SMM internal inconsistency: pooled moments (e.g. `average_hours`, ginis) weight alive observations by `s_e·ω(t)/n_sim` where ω(t) already contains cumulative survival — survival enters twice (ω(t)·S(t) effectively), unlike the ratio moments which count it once (`calibrate.py:333–344` vs `:543–561`).
- MIT stitching overwrites only the 5-D `a_policy/c_policy/l_policy` (`olg_transition.py:1247–1254, 1308–1315`) while both simulate paths read `*_policy_alpha` — whether the stitched arrays are ever consumed (NumPy views may save this; JAX arrays are immutable) needs a dedicated check.
- `_at` zero-fills exogenous spending paths beyond their length while `B_path` clamps to its last value (`olg_transition.py:1745–1759`); `bequest_transfers` enters neither `total_spending` nor (with the loop off) household income; `transfer_floor` outlays have no budget line (inactive at floor 0); `fiscal.B_over_Y`=1.64 vs `transition.B_over_Y`=1.7 in the JSON, only the former read; `transition.recompute_bequests` in the JSON is never read by `run_from_config` (CLI flag only).

### Implications

- The ~12% common-factor gap is consistent in sign and rough size with Bug 1 (Y built from C instead of w·L: the wedge becomes the data-dependent factor ≈ L_SS/C_transition ≈ 0.26/0.30), with Bug 2 adding item-specific distortions on JAX runs; exact attribution requires re-running after fixes.
- Contaminated and to be re-measured after fixes: `other_net_spending_over_Y` (−0.1042 and the earlier −0.1056), the 2026-06-11 decomposition table, all fiscal figures since 2026-03-04, all JAX-backend transition output.
- Unaffected: the SMM θ (standalone pipeline, both bugs absent); the multiplier-0.000 structural diagnosis (a debt-financed G shock has no household channel regardless of which aggregate enters Y).

**No code changes applied** — fixes to `olg_transition.py:2072` (unpack order), the L/w convention, and the JAX batched α path await user decision.

### Fixes applied (same day, user-approved "fix all")

1. **Unpack swap**: `K_path[t], C_path[t], L_path[t] = self._aggregate_capital_labor_njit(...)` — matches the njit's `(K, C, L)` return; comment added.
2. **L units**: after aggregation, `L_path = L_path / w_path` — converts the wage-valued `effective_y_sim` aggregate to efficiency units before production, same convention as `calibrate.py` (`L = labor_income / w`). `results['L']` is now in efficiency units. `compute_aggregates()` still returns the raw wage-valued labor mean (no external callers; not changed).
3. **JAX batched α solve**: `_solve_cohorts_jax_batched` now outer-loops the permanent-FE grid (one batched sweep per α node, `alpha_mult` is a traced vmap arg so no retracing), stacks `(n_alpha, T, ...)` per-α policies on each model, scalar attributes alias α=0 — mirroring `LifecycleModelPerfectForesight.solve` and `LifecycleModelJAX.solve`. Verified: batched per-α policies bitwise-identical (0.0 max diff) to the standalone JAX solve, distinct across α nodes.
4. **MIT stitching staleness (4th bug, found while fixing 3)**: both simulate paths read `*_policy_alpha` (`lifecycle_perfect_foresight.py:1126,1132,1211`; `olg_transition.py:599-601` via `_as_alpha_indexed`), but stitching rebound only the scalar 5-D `a_policy/c_policy/l_policy` — a no-op for the simulation on BOTH backends (the scalar arrays alias α=0 only until rebound by `.copy()`/setattr). Fixed: both stitching blocks (NumPy `olg_transition.py:~1247`, JAX `~1308`) now also stitch `a_policy_alpha/c_policy_alpha/l_policy_alpha` over `[:, :pre]`. The 2026-03-08 "A[0] verified" note predates Phase 8's switch of the simulate read path to the α arrays.

Verification: fiscal suite **39/39**, OLG suite **73 passed / 17 deselected** (documented exclusions only, 45 min) — regression-clean. `olg_transition.py --test` passes on both backends, agreeing to MC noise. New check script `check_a0_predetermination.py`: τ_l-shock fiscal scenario with n_alpha=3, σ_α>0 — A[0] diff exactly 0.0 on both backends (the old code fails this by construction).

### Post-fix closure re-measurement and gap validation

**Closure** (`measure_baseline_closure.py`, JAX, n_sim=2000, other_net=0): baseline primary surplus **−6.94% of Y** (mean; t=0 −6.83%; was −8.47% under the bugs). New value, config updated:

```
other_net_spending_over_Y = −0.0694 − 0.0195 = −0.0889   (was −0.1042; pre-audit −0.1056)
```

The plug is now below the data's other-revenue line (9.4%), which the old −10.6% exceeded. Log: `output/closure_remeasure_postfix.log`.

**Gap validation** (`diag_ss_vs_transition.py`, JAX, n_sim=3000): every item ratio now agrees between the calibration SS and the transition baseline to **±0.6%** (was −9.5 to −14.6%): tax_p/Y 0.1300 vs 0.1300, pensions/Y 0.1592 vs 0.1602, health_gov/Y 0.0539 vs 0.0539, total revenue/Y 0.3477 vs 0.3480. Transition t=0 primary surplus +2.03% with the new closure (target +1.95%; n_sim difference). The transition baseline now reproduces the calibrated steady state — "the SMM matches the base year" carries to the transition. The remaining **Y-level** difference (transition 0.4368 vs SS 0.4936, −11.5%) is the per-birth vs per-living normalization, which cancels in all ratios — expected, not a bug. Log: `output/ss_vs_transition_postfix.log`.

The 2026-06-09 part-1 open question and the 2026-06-11 route-2 decomposition are thereby superseded: the gap was Bug 1 + the L-units convention, not bequests or behavior.

### Pre-counterfactual reconciliation check (same day)

Checked the fiscal pipeline wiring before re-running counterfactuals:

- `transition.B_over_Y = 1.7` was **dead config** — nothing read it. The pipeline uses `fiscal.B_over_Y = 1.64` for `B_initial` and the tax-financed return target, so baseline interest = r_B·1.64 = 3.44% of Y, matching the 2023 data (3.4%) by construction. (Earlier statement that the transition used 1.7 was wrong.)
- `transition.G_over_Y` / `transition.I_g_over_Y` were also dead (live values come from the `fiscal` block); `transition.recompute_bequests` was never read (CLI flag only). All four dead keys removed from the JSON.
- Post-horizon paths: `_extend_base_paths` clamp-extends all spending paths over `n_post`, and `run_fiscal_figures` puts G/I_g/defense/other into `base_paths` — the `_at` zero-fill footgun never triggers on this pipeline.
- Closure consistency: `run_fiscal_figures` uses the same n_sim (2000), warmup-mean(Y) scaling, and spending shares as the closure measurement, so −0.0889 carries over. Standing condition: closure measured with the bequest circuit open (what the fiscal pipeline runs); re-measure if bequests are switched on there.
- Multiplier: nothing to decide — ΔY ≡ 0 for debt-financed G is the model's theoretical implication and the 0.000 line in the output is simply that fact. Dropped as an open item.

### Code cleanup (same day)

Dead code/config eliminated; output verified bit-identical pre/post (test-mode Deficit/GDP −25.66986% NumPy / −25.74057% JAX; A[0] check values unchanged to the last digit):

- **JSON**: removed dead `transition.{B_over_Y, G_over_Y, I_g_over_Y, recompute_bequests}` keys.
- **`olg_transition.py`**: removed the SS-profiles block in `solve_cohort_problems` (3 full lifecycle solves + sims per call; outputs `ss_asset_profiles`/`ss_earnings_profiles`/`ss_asset_distributions` had no consumers) — a per-call speedup; removed the dead `use_initial_distribution` param and `_jax_policy_batch` (assigned `{}` in 3 places, never read since the batched simulate reads model objects); removed dead locals.
- **`compute_aggregates()`**: now converts L to efficiency units via `self.w_path[t]` (consistent with `simulate_transition`); raises if called before a transition has set `w_path`.
- **`calibrate.py`**: removed unused `import sys`, dead locals (`ret_age`, `n_sim`, `w`), and the never-filled `'transfer'` key in `compute_fiscal_ratios`.
- **`fiscal_experiments.py` / `run_fiscal_figures.py` / `lifecycle_jax.py` / `lifecycle_perfect_foresight.py` / `eval_fiscal_results.py`**: pyflakes-driven removal of unused imports and dead locals. Pyflakes now clean (f-string style warnings left).

Verification of the cleanup: smoke tests bit-identical on both backends (test-mode Deficit/GDP and A[0] check values unchanged to the last digit); config loads with `B_over_Y = 1.64`, `other = −0.0889`; fiscal suite re-run **39/39**; pyflakes clean. **The OLG and calibrate suites were NOT re-run after the cleanup** (the run was killed to free CPU for an interactive fiscal-figures run) — re-run `pytest test_olg_transition.py -k "not (TestJAXBackend or TestNewFeaturesJAX or TestLaborSupplyJAX or TestEndogenousRetirementJAX or test_l_sim_in_output or test_public_capital_increases_output)"` and `pytest test_calibrate.py` before relying on the cleanup commit. Expected clean: the cleanup only removed code verified dead (no consumers) and outputs are bit-identical. **STATUS:** baseline fiscal closure re-measured under the new θ (`other_net_spending_over_Y = −0.1042`, was −0.1056); cumulative-multiplier-0.000 diagnosed as structural (debt-financed G shock has no household channel in SOE mode, ΔY ≡ 0). See `## Session 2026-06-11` at top. Prior: 2026-06-09 recalibration against the data 2020 life table (part 3).

---

## Session 2026-06-11: closure re-measured under new θ + multiplier-0.000 diagnosis

### Route 1 — baseline fiscal closure reset (done)

Re-measured the transition baseline primary balance under `_derived.theta` = {ν=28.67, β=0.985, τ_p=0.198, ρ_pens=0.161, m=0.0416} with `other_net_spending = 0` (script: `measure_baseline_closure.py`, JAX, n_sim=2000, G/I_g/defense wired as in the config branch). Result: **primary surplus −8.47% of Y** (mean over the flat 60-period path; t=0 −8.43%; was −8.6% under the old θ). New closure, same formula as 2026-05-28 Fix 2:

```
other_net_spending_over_Y = −0.0847 − 0.0195 = −0.1042   (config updated; was −0.1056)
```

The shift is exact (no household feedback), so the baseline primary balance now equals the Greek 2023 value (+1.95%) by construction; no re-run needed. Warmup mean(Y) = 0.5011.

### Route 3 — cumulative multiplier 0.000 (diagnosed, structural)

Inspected `output/fiscal_test/fiscal_results.json` (G shock, file of 2026-05-20). `fiscal_multiplier()` compares the no-shock baseline scenario against `debt_financed`. In the debt-financed run, ΔG = 0.0107 per period but **every household-relevant path is bit-identical** between baseline and counterfactual: max |Δr| = |Δw| = |Δτ_l| = |Δτ_c| = |Δτ_p| = |Δτ_k| = 0, hence ΔL = ΔA = ΔC = ΔK_domestic = ΔY = 0 exactly. Only B/GDP (ends 228% vs −183%) and NFA differ.

This is the model structure, not a numerical bug: in SOE mode r is exogenous, w is pinned by the firm FOC given (r, K_g), a G goods shock does not enter the household budget or production, and debt financing leaves all tax paths unchanged over the horizon — so the debt-financed G multiplier is exactly 0 by construction. Cross-check: `tax_financed` shows ΔY ≠ 0 (max 0.005), since τ_l moves. An I_g shock would also give ΔY ≠ 0 (K_g enters production and w). Whether to report the multiplier off a different scenario/shock is an open modeling choice.

### Route 2 — SS-vs-transition gap decomposition (run done; bequests are NOT the driver)

Script `diag_bequest_decomp.py` (JAX, n_sim=3000, new θ, same 2020 survival both sides): (A) SS stationary cross-section vs transition baseline with (B) `recompute_bequests=False` and (C) `=True`. Per-item gaps at t=0: B−A = solve-path difference (stationary single-cohort solve vs MIT-stitched cohort solves), C−B = bequest-redistribution contribution. Log: `output/bequest_decomp.log`.

| item | SS(A) | off(B) | on(C) | B−A % | C−B % | C−A % |
|---|---|---|---|---|---|---|
| tax_revenue/Y | 0.3480 | 0.3076 | 0.3029 | −11.6 | −1.5 | −13.0 |
| tax_c/Y | 0.1123 | 0.0959 | 0.0959 | −14.6 | −0.0 | −14.6 |
| tax_l/Y | 0.0700 | 0.0629 | 0.0615 | −10.1 | −2.2 | −12.1 |
| tax_p/Y | 0.1300 | 0.1177 | 0.1147 | −9.5 | −2.5 | −11.8 |
| pensions/Y | 0.1602 | 0.1407 | 0.1392 | −12.2 | −1.0 | −13.1 |
| ui/Y | 0.0128 | 0.0114 | 0.0113 | −11.2 | −1.0 | −12.0 |
| health_gov/Y | 0.0539 | 0.0476 | 0.0471 | −11.8 | −1.0 | −12.7 |
| **Y level** | 0.4936 | 0.4944 | 0.4994 | **+0.2** | +1.0 | +1.2 |

Findings (observed, this run):

1. **The Y-level gap is gone under the new θ**: SS Y 0.4936 vs transition 0.4944 (+0.2%; was +7.3% under the old θ per the 2026-06-09 part-1 diagnostic). The remaining gap is entirely in the item numerators relative to Y.
2. **Bequest redistribution explains only −1.0 to −2.5 pp** of the per-item gap (C−B), and it moves the transition *away* from the SS, not toward it.
3. **The solve-path difference (B−A) carries the gap**: −9.5 to −14.6% across items with bequests off on both sides and identical survival. All items are low by a similar factor (~−12%) while Y and (by the firm FOC) the K/L ratio agree, which is consistent with a composition/weighting difference in how the two sides aggregate item numerators — not yet isolated.
4. Bequest loop: converged in 2 iterations; iteration-2 max change exactly 0.00e+00 (assets live on a discrete grid, so unchanged grid choices give bit-identical bequests; recipients of the period-t lumpsum are cohorts born at t, most of whose deaths fall outside the 60-period window). Not verified further.

**Open (next step for route 2):** isolate the remaining B−A item gap. Y levels agree but every item/Y is ~12% lower in the transition; candidates are the cross-section weighting of item numerators (calibrate `age_weights` vs transition per-cohort means with births-only weights — see the part-1/part-2 correction: ratios were argued to agree algebraically, which this run contradicts at the item level) and per-item composition (e.g., wage-weighted vs head-count L). A per-item, per-age comparison of the two cross-sections would localize it.

Prior: GG-accounts data audit done; plumbing fix + `other_net_spending` residual to pin the baseline primary balance. See `## Session 2026-05-26`.

Prior: double-count fix landed (commit `8c45250`); baseline calibration not yet aligned to the corrected accounting. See `## Session 2026-05-20 / 2026-05-21`.

Earlier resolution: 2026-05-18. See `## Resolution (2026-05-18)` mid-file.

Original handoff (2026-05-11) preserved below for context.

---

## Session 2026-06-09 (part 3): recalibrate against data 2020 life table

The calibration's stationary `survival_probs` was a hand-entered vector differing from the Eurostat data by up to 0.037 (highest at old ages: px₈₄ old 0.960 vs data 0.923). Re-sourced it from `data/survival_GR.npz` year **2020** (= transition `current_year`, the "initial equilibrium" year; 2020 vs 2023 differ only in the 3rd–4th decimal). Both calibration uses of survival pick it up from the JSON array (`build_lifecycle_config` → lifecycle solve; `compute_age_weights` → cross-section weights). Warm-started the optimizer from the prior 5-param optimum.

5-param SMM (ν, β, τ_p, ρ_pens, m), JAX/CPU, n_sim=10000, **Converged: True** (objective 0.0, 8828 s), theta auto-written:

| Param | prior (hand-entered surv) | **new (data 2020 surv)** | | Moment | Data | Model |
|---|---|---|---|---|---|---|
| ν | 27.30 | **28.67** | | average_hours | 0.41 | 0.410 |
| β | 0.977 | **0.985** | | A/Y | 4.0 | 4.000 |
| τ_p | 0.197 | **0.198** | | SSC/Y | 0.130 | 0.130 |
| ρ_pens | 0.147 | **0.161** | | pensions/Y | 0.160 | 0.160 |
| m | 0.0393 | **0.0416** | | health_gov/Y | 0.054 | 0.054 |

Direction as expected from lower old-age survival: more end-of-life mortality risk lowers effective discounting (β·π), so β rises to hold A/Y=4.0; fewer survivors reach pension/high-medical ages, so ρ_pens and m rise to hold pensions/Y and health_gov/Y. τ_p and ν roughly unchanged. Report: `output/calibration/calibration_GR_20260609_183208.md`.

**Not yet done:** re-run the transition baseline + fiscal experiments under the new θ and data survival; the `other_net_spending_over_Y = −0.1056` closure was measured under the old calibration and should be re-measured. Then decompose the ~13% SS-vs-transition gap (part 1 open question) — now both sides draw survival from the same 2020 source, so the demographic contribution to the gap is removed and any residual isolates the behavioral (bequest / cohort-MIT-solve) sources.

---

## Session 2026-06-09 (part 2): data-driven cohort survival + correction of the weighting-norm diagnosis

### What was implemented

Greek cohort survival is now read from data instead of a hand-entered vector + synthetic improvement rate.

- `code/build_survival_GR.py` → `data/survival_GR.npz`: Eurostat `demo_mlifetable` px (DATA_GR.xlsx 'Survival rates'), years 1961–2023, model age j ↔ real age 25+j, px (63, 60). 1960 dropped (missing).
- `OLGTransition` gains `survival_table=(years, px)`. When set, `_survival_schedule_at_year` returns the data **period table** for the true calendar year (= internal birth_year-anchored clock + (current_year−birth_year)), clamped to [1961, 2023]. Each cohort is solved and simulated along its calendar diagonal (`_cohort_survival_schedule`) — **cohort-historical** for the past, **held at 2023** for future transition years (2024–2079). User's choice (2026-06-09): cohort-historical at t=0, hold-at-2023 forward.
- `build_olg_transition` loads the npz via the opt-in config key `transition.survival_data_file` (only if the px age dim matches model T) and passes `survival_table`. Added to `calibration_input_GR.json`.
- **JAX bug fixed**: `_solve_cohorts_jax_batched` / `_simulate_cohorts_jax_batched` passed `ref.survival_probs` (cohort 0's schedule) as a *shared* vmap arg (`in_axes=None`). With per-cohort survival every cohort would have used cohort 0's table. Changed `survival_probs` `in_axes` to `0` in both batched kernels (`lifecycle_jax.py`) and stack per-cohort schedules in `olg_transition.py`. Validated: a 2-cohort batched solve with survival 1.0 vs 0.7 now yields different policies (mean a_policy 17.98 vs 15.46). For the common case (all cohorts share one schedule) the stacked array is identical across cohorts → behaviour unchanged. NumPy already solved per cohort (correct but slow — 120 distinct cohort solves).

### Correction to the 2026-06-09 (part 1) weighting-norm diagnosis

Part 1 claimed the transition's `cohort_sizes` "missing cumulative survival" was a confirmed contributor to the ~13% SS-vs-transition item-ratio gap, and proposed forcing `cohort_sizes = age_weights`. **That is wrong** and was NOT implemented (a survival-weighted `cohort_sizes` was written, then reverted).

Reason: the transition's per-cohort age means divide by **n_sim** (all agents) and **dead agents hold 0** (verified: `mean_over_n_sim(age 59) = 0.166 = 0.68·0.244 = survival · E[a|alive]`). So survival is **already baked into every transition mean**. `calibrate.py` instead weights by `births·S(j)` and averages over **alive** agents. The two are algebraically identical per age (`births·S·E[X|alive]`), so **aggregate ratios agree by construction** — the survival factor cancels. Adding survival to `cohort_sizes` would **double-count** it. The births-only weights are correct; with data cohort survival, mortality is cohort-specific in the simulation, and the time-invariant births weights remain correct.

**Therefore the ~13% ratio gap is not from the weighting norm.** Remaining candidate sources (undecomposed): bequest treatment under `recompute_bequests=false`, and behavioral differences between calibrate's single stationary lifecycle solve and the transition's cohort/MIT solves.

### Calibration ↔ transition norm

Both already use the same effective norm (`births × survival`, taken with their respective mean conventions), so ratios agree. `calibrate.py:compute_age_weights` is unchanged; its stationary `survival_probs` vector is still the config vector (not re-sourced from the npz). Re-sourcing the base-year `survival_probs` from `survival_GR.npz` would change the SMM moments and require a recalibration — not done.

Files: `build_survival_GR.py`, `lifecycle_jax.py` (2 `in_axes`), `olg_transition.py` (`survival_table`, `_survival_schedule_at_year`, per-cohort batching), `calibrate.py:build_olg_transition`, `calibration_input_GR.json`. Data: `data/survival_GR.npz`, `data_inventory.md` § 1.3.

---

## Session 2026-06-09 (part 1): SS-vs-transition gap — level mismatch, not a transient

### Question

The 2026-05-28 caveat measured the transition baseline ITEM ratios ~8–15% off the SS calibration and attributed it partly to "demographic cohort-weighting over 2020–80." But the Greek config has **no** `fertility_path` / `survival_improvement_rate`, so `cohort_sizes_path` is never built and demographics are stationary; prices are flat (`r_initial=r_final=0.04`). With stationary demographics and flat prices the no-shock baseline should reproduce the calibration SS *by construction*. The caveat compared the transition MEAN over 2020–80, which cannot distinguish a true t=0 level mismatch from a transient. This session ran the t=0 check.

### Method

`diag_ss_vs_transition.py` (left in `code/`): (A) SS side = `calibrate.py` stationary cross-section at the calibrated θ, age-weighted, via `run_model_moments` + `compute_fiscal_ratios`; (B) transition side = one no-shock baseline `simulate_transition` (G/I_g/defense/`other_net` wired exactly as the `run_fiscal_figures` config branch), then `compute_government_budget_path`. JAX/CPU, n_sim=3000. (Gotcha: the on-disk `_derived.K_over_L` is `None` — only `theta` is persisted — so `compute_fiscal_ratios` must be fed the dict `load_config` returns, not a fresh `json.load`, else it hits its `Y<=0` early-return.)

### Finding — flat baseline at a different level (case 1, convention mismatch)

The transition baseline is **flat across all 60 periods** (Y=0.5189 at t=0 and t=59; every item/Y constant to 3 decimals), so it IS at rest — not a transient. But it sits at a different steady state than the calibration:

| item / Y | SS | transition t=0 | t=59 | Δ(t0−SS) |
|---|---|---|---|---|
| total revenue | 0.3500 | 0.3019 | 0.3025 | −13.7% |
| consumption tax | 0.1148 | 0.0959 | 0.0959 | −16.5% |
| labour tax | 0.0702 | 0.0616 | 0.0618 | −12.2% |
| payroll (SSC) | 0.1294 | 0.1142 | 0.1146 | −11.7% |
| capital tax | 0.0356 | 0.0302 | 0.0302 | −15.2% |
| pensions | 0.1616 | 0.1392 | 0.1392 | −13.8% |
| gov health | 0.0539 | 0.0467 | 0.0467 | −13.4% |
| **Y level** | **0.4835** | **0.5189** | **0.5189** | **+7.3%** |

Both objects are the same economy (same θ, r/w, demographics, policy), so they should coincide. The flat ~13% item offset + 7.3% higher Y localizes the discrepancy to the two aggregation code paths (`calibrate.py` stationary cross-section vs `olg_transition.py` cohort aggregation), not to demographics or a transient.

### Confirmed contributor — `age_weights` ≠ `cohort_sizes`

The demographic weights the two paths use are different objects:
- `calibrate.py` `compute_age_weights`: ω(j) = (1+g)^(−j) · S(j), with cumulative survival S(j) thinning older ages.
- `olg_transition.py` `_cohort_sizes_njit`: size(j) = exp(g·(birth_year_of_cohort − base)) — pure population-growth scaling by birth cohort, **no survival term**.

They diverge up to ~30% at old ages and have **opposite shape at the top**: `age_weights` falls at old ages (0.0160→0.0145, survival-thinned), `cohort_sizes` rises (0.0192→0.0196, no thinning; under g<0 older birth cohorts get larger weight). Normalized max abs diff 0.0051. This is *a* contributor; the residual (bequest treatment under `recompute_bequests=false`, per-capita normalization in `compute_aggregates`) is not yet decomposed.

### Caveat on the table

The SS `primary_balance_over_Y` reported by `compute_fiscal_ratios` (+0.122) is NOT comparable to the transition primary surplus (+0.020): the SS ratio nets revenue only against pensions+UI+health and omits G/I_g/defense/`other_net`. Adding those (13+3+3−10.6 = +8.4% net spending) reconciles +12.2% → ~+3.8% ≈ transition +2.0% once the ~13% item gap is applied. So the budget lines are mutually consistent; the open issue is purely the ~13% level offset.

### Implication for the baseline closure

Until the two computations of the same steady state are reconciled (starting by making the demographic weights identical), "the SMM matches the base year" does NOT carry to "the transition baseline matches the base year," and the `other_net_spending_over_Y = −0.1056` plug is absorbing this code discrepancy rather than a genuine accounting residual. Next cheap step: re-run the transition aggregation forcing `cohort_sizes = age_weights` and measure how much of the 13% closes; the remainder is bequests + normalization.

---

## Session 2026-05-26: GG-accounts data audit + baseline fiscal closure

### Data audit — which government-account lines the model omits

Source: `data/DATA_GR.xlsx`, sheet `DATA`. Reference year **2023** (2024 reports zero on the itemized social-benefit lines — incomplete). All values are % of GDP.

| Line (DATA sheet code) | 2023 | In model? |
|---|---|---|
| **Revenue (total 48.2%)** | | |
| Taxes on consumption (23) | 17.10 | yes — `tau_c` |
| Taxes on labour (25) | 5.93 | yes — `tau_l` |
| Taxes on profits (26) | 2.71 | yes — `tau_k` |
| Social security contributions (28) | 13.00 | yes — `tau_p` |
| **Other revenues (22)** | **9.43** | **no** |
| **Primary expenditure (46.2%)** | | |
| Pensions (77) | 12.02 | yes |
| Unemployment (79) | 0.61 | yes — UI |
| Means-tested (80) | 1.11 | yes — `transfer_floor` |
| Health, in-kind (81) | 2.33 | yes — `health_gov` |
| **Education benefits (82)** | **1.40** | **no** |
| Public investment (45) | 3.86 | line exists, **=0 in the run** |
| **GG Other expenditure (30)** | **26.15** | only `G` (13%) proxies it |
| Interest (40) | 3.39 | yes — `debt_service` |
| **Primary balance (48)** | **+1.95** | — |

(2022 primary balance −0.07%. Greek post-program target band ≈ 2.0–3.5%.)

### Model baseline primary surplus

Direct read of `primary_deficit/Y` from `output/fiscal_test/fiscal_results.json` (post-fix G run): **+5.2% of GDP, stable across all 60 periods** (t0 +5.16%, mean +5.19%). The 2026-05-21 "~10%" figure was a back-of-envelope from the B/Y drift; the direct measurement supersedes it.

### Two separable causes of the +5.2% surplus

1. **Plumbing — config lines not wired into the baseline budget (~6 pp).**
   - `run_fiscal_figures.py` config branch (line ~84) computes `I_g_path = delta_g · K_g`. Greek config has `K_g = 0` → `I_g = 0`, so `fiscal.I_g_over_Y = 0.03` is never applied.
   - `fiscal.defense_over_Y = 0.03` is in the JSON but `base_paths` never passes `defense_spending_path` → defense = 0.
   - Both show as exactly `0.0000` in `base_budget`. Wiring them adds ~6 pp of primary spending → surplus +5.2% → ≈ **−0.8%**.

2. **Structurally absent accounts.** Other revenues (+9.4%), most of GG Other expenditure (model `G`=13% vs data public consumption 19.4% / other-expenditure bucket 26.2%), education benefits (1.4%). Net of genuinely-absent lines ≈ −5 pp of GDP of spending; with the un-wired I_g+defense, the model omits ≈ −11 pp net spending relative to the full accounts. The residual's natural sign is *net spending*.

### Debt-stabilizing surplus (model recursion)

`compute_debt_path` uses `B[t+1] = (1+r_B)·B[t] + PD` (no growth term). Stationary `B/Y = b` requires primary surplus `= r_B·b = 0.021·1.64 = 3.44%`.

### Residual to pin (off the model's realized surplus, not the data arithmetic)

| Target primary surplus | from current baseline (+5.2%) | after wiring I_g+defense (−0.8%) |
|---|---|---|
| Data 2023 (+1.95%) | net spending +3.2% | net **revenue** +2.8% |
| Debt-stabilizing (+3.44%) | net spending +1.8% | net revenue +4.2% |

### Tension (the open closure choice)

Matching the data surplus (+1.95%) is **below** the model's debt-stabilizing surplus (3.44%), so the baseline B/Y rises. Greek debt/GDP fell in 2022–23 mainly through high nominal GDP growth, a channel this stationary model lacks. Closure (a) = target 3.44% → stationary baseline; closure (b) = target 1.95% → accept B/Y drift.

### Implementation (this session)

1. **Plumbing fix.** `run_fiscal_figures.py` config branch: `I_g_path` from `fiscal.I_g_over_Y × mean(Y)`; `defense_spending_path` from `fiscal.defense_over_Y × mean(Y)`, both routed through `base_paths`.
2. **`other_net_spending` parameter.** New exogenous net-primary-spending line = (other expenditure − other revenue), added to `total_spending` in `compute_government_budget`. Set from `fiscal.other_net_spending_over_Y × mean(Y)`. Routed through `simulate_transition` (with `_active_` override) and `fiscal_experiments` base_paths/cf, identical to `govt_spending_path`. Does **not** enter the household budget, so not added to `pre_transition_paths`. Defaults None/0 → exact prior behaviour, no test changes.
   - `other_net_spending_over_Y` is the single knob to pin the baseline primary balance to whichever target the closure choice selects.

---

## Session 2026-05-27: objective reframe + labour-share / SSC investigation

**Objective reframe (user).** Stationary B/Y was never a target. What matters: the government budget — its several items *and* the primary balance — match the data. So a single `other_net_spending` plug on the bottom line is insufficient; items must be reconciled. The `−0.0276` value set on 2026-05-26 is therefore provisional and will be revisited.

**Regression check.** Full OLG suite (`test_olg_transition.py`, ex documented JAX/hang exclusions): **74 passed, 16 deselected** (44 min). The plumbing + `other_net_spending` changes are regression-clean. Fiscal suite 39/39.

**Item-level audit** (model baseline vs Greek 2023, per Y): SSC overshoots (+9.3 pp), consumption tax short (−7.5 pp), pensions over (+4.3 pp); these are endogenous. Largest is SSC.

**Labour-share / SSC investigation** (full data + Gollin adjustments recorded in `data_inventory.md` § 1.11):
- Model `(1−α)=0.67` is the *total* return to labour; raw compensation of employees (0.35) is the wrong target because it excludes self-employed labour (mixed income B.3G = 22% of GDP). Gollin (2002) adjustments put α in [0.34, 0.45]; model α=0.33 ≈ Gollin Adj. 1. **Decision: keep α=0.33** (document as Adj. 1).
- SSC base ≠ α. Employees pay ~32–38% (capped); self-employed pay **flat-rate categories, not income-linked** (2020 reform), effective **~8–9% on mixed income** (two convergent estimates). Employee share of labour income = 0.614.

**Approach change for SSC (user).** Do **not** impose a fraction/rate. Instead **calibrate `tau_p` to match SSC/GDP** (data 0.130), then read off the implied `tau_p` and check it falls between the two group rates (~9% self-employed, ~32–38% employee), share-weighted (expected ~0.20–0.23, since SSC/Y ≈ `tau_p` × labour-base/Y and the model base/Y ≈ 0.66). This validates the labour-income/SSC structure rather than hard-coding it.
- Note: `tau_p` enters the household budget (net wage), so it is coupled to the SMM hours target — calibration must be joint (add `tau_p` as param, `tax_p/Y` as target) or an outer fixed point, not a one-shot fiscal adjustment.

**Next step:** implement the `tau_p` calibration and run it; compare implied `tau_p` to 0.09 / 0.32–0.38.

### Outcome (2026-05-28)

`tax_p_over_Y` added as SMM moment, `tau_p` (path `tau_p_default`) as a third SMM parameter; weights switched to percent-deviation (`1/value^2`: hours 5.949, A/Y 0.0625, SSC 59.172). First full run at `phi=2` did not converge cleanly: the (unfittable-at-phi=2) hours moment dominated and the optimizer sacrificed the A/Y and SSC matches to chase it. Diagnosis: hours has little independent leverage from `nu` once A/Y is held (separable preferences → consumption response offsets), so it needs `phi`.

Set `phi = 2.0 → 1.5` (Frisch 0.5 → 0.67; justified because the model's only behavioural labour margin is intensive — the unemployment state is exogenous zero-productivity — so a single margin must carry the aggregate Frisch ~0.8, [[Chetty_AER2011]]). Full run (n_sim=10000, 29 min) **converged**:

| Param | Value | | Moment | Data | Model |
|---|---|---|---|---|---|
| nu | 26.69 | | average_hours | 0.41 | 0.417 |
| beta | 0.972 | | A/Y | 4.0 | 3.99 |
| tau_p | **0.198** | | SSC/Y | 0.130 | 0.130 |
| phi | 1.5 (fixed) | | | | |

`tau_p = 0.198` lands at the labour-income-share-weighted blend of the employee (~0.38) and self-employed (~0.09) effective rates — validates the labour-income/SSC structure. Written to `_derived.theta`; `build_olg_transition` now reads calibrated taxes from `_derived.theta`, so the transition picks up `tau_p=0.198`.

**Untargeted moments that shifted (worse) under the new calibration:** pensions/Y 0.192 (+3.2 pp, was ~matched — pension flow scales with the calibrated wage/earnings profile), health total 0.099 (+1.7 pp), wealth Gini 0.30 (vs 0.58). Pensions/health need `rho_pens`/`m` revisited; wealth Gini is the Phase 9 (bequest + initial wealth) item.

Draft `DSA-LSA calibration.tex` updated: Table 1 params (nu, beta, tau_p, phi=1.5), baseline-moments table Model column, and the targeted/fiscal-residual/C-over-Y/distributional paragraphs. `phi=1.5` footnote + `Chetty_AER2011` and `Gollin_JPE2002` bib entries added.

**Still open:** baseline fiscal closure (`other_net_spending`) — see 2026-05-28 Fix 1/2 below.

### Fix 1 / Fix 2 (2026-05-28, later)

**Fix 1 — pension/health overshoot.** Folded `pension_replacement_default` and `m_good` into the SMM (5 params: `nu, beta, tau_p, rho_pens, m`; 5 targets: hours, A/Y, SSC/Y, pensions/Y, health_gov/Y; percent weights). They belong in the SMM because both shift saving → A/Y, so external re-adjustment would need iteration. Converged-in-substance (params stable 180+ iters, Q=2.5e-4) but hit maxiter=400 → "Converged: False", so the auto-writer (gated on `convergence`) did not write `_derived.theta`; **theta written manually**. New SS calibration:

| Param | Value | | SS moment | Data | Model |
|---|---|---|---|---|---|
| nu | 27.30 | | hours | 0.41 | 0.415 |
| beta | 0.977 | | A/Y | 4.0 | 3.99 |
| tau_p | 0.197 | | SSC/Y | 0.130 | 0.129 |
| rho_pens | 0.147 | | pensions/Y | 0.160 | 0.161 |
| m | 0.0393 | | health_gov/Y | 0.054 | 0.054 |

All five SS moments match ≤1.2%. rho_pens, m interior (no bounds; pension floor not binding). Untargeted distributional moments worsened (wealth Gini 0.33, zero-wealth 5.7%, income Gini 0.376) — Phase 9.

**Fix 2 — baseline fiscal closure.** Measured the transition baseline primary balance (build_olg_transition + one baseline sim, G/I_g/defense wired, other_net=0, n_sim=2000): **primary surplus −8.6% of Y** (a deficit). Set `other_net_spending_over_Y = -0.0861 - 0.0195 = -0.1056` so the baseline primary balance equals the Greek 2023 data value (+1.95%). `other_net` has zero household feedback (added to `total_spending` post-simulation), so the shift is exact — no re-run. Formula to re-derive: `other_net = (measured baseline primary surplus) - (target primary surplus)`.

**Caveat surfaced — SS vs transition gap.** The transition baseline ITEM ratios differ from the SS calibration: transition tax_p/Y 0.114 (SS 0.129), pension/Y 0.139 (SS 0.161), gov_health/Y 0.047 (SS 0.054), total revenue 0.302 (SS 0.350). Cause: transition mean Y (~0.52) ≠ SS Y (0.483) plus demographic cohort-weighting over 2020–80. So the SMM matches items to data in the **stationary cross-section**, but the **transition** baseline (where the experiments run) is ~8–15% off, and the −10.6% closure is larger than the data's other-revenue line (9.4%) because it also absorbs this gap. Open question: whether to calibrate against transition moments (heavy — each SMM eval = full transition) or accept the SS calibration with the closure forcing only the transition primary balance to data.

---

## Where we are (2026-05-11)

The fiscal-experiment code path through `run_fiscal_figures.py` is **functionally intact** after the Phase 8 σ_α merge (`4a9e4aa`, 2026-05-08) and the follow-up plot/income-moment fixes (`599454b`, 2026-05-11). Smoke tests confirmed:

- Minimal hardcoded run (`python run_fiscal_figures.py --shock G`): three scenarios complete, four figures saved, ~28 min on NumPy.
- FE-on Greek run (`python run_fiscal_figures.py --config calibration_input_GR.json --shock G --backend jax` with `JAX_PLATFORM_NAME=cpu`): three scenarios complete, bisection converged, four figures saved, **54.5 min** on JAX/CPU (n_alpha=5 multiplies the cost of every cohort solve in the transition).

**However, the economic results are not sensible.** This is a *separate* problem from anything Phase 8 touched.

## Headline symptoms (FE-on Greek run)

| Quantity | Value | Interpretation |
|---|---|---|
| Baseline `final B/Y` | 23,378 % (234× GDP) | Debt explodes — the baseline is not a fiscal steady state |
| Debt-financed G shock `final B/Y` | 24,492 % | Larger explosion than baseline |
| Tax-financed G shock `Δτ_l` | +109 pp | Bisection converges only by pushing τ_l from 0.10 to ~1.19 |
| Tax-financed G shock `final B/Y` | −92 % | Government becomes a net creditor |
| Cumulative fiscal multiplier | 0.000 | Almost certainly a units issue or baseline-noise division |

## Diagnosis (where the issue is *not* and where it *probably* is)

**Not Phase 8.** The same residual was visible in the no-FE Greek calibration (`c9b3f14`). It's pre-Phase-8.

**Probably the fiscal accounting in `olg_transition.py` plus the input ratios in `calibration_input_GR.json`.** The Phase 8.7 calibration report (`code/output/calibration/calibration_GR_20260508_145856.md`) shows the smoking gun in the "Fiscal Ratios" table:

| Ratio | Model | Greek data | Gap |
|---|---|---|---|
| `tax_revenue / Y` | 0.97 | 0.40 | model collects 2.4× the data |
| `pensions / Y` | 2.37 | 0.16 | model spends 15× the data |
| `UI / Y` | 0.04 | 0.01 | model 4× the data |
| `health / Y` | 0.019 | 0.054 | model is 0.35× the data |
| `interest / Y` | 0.07 | 0.03 | model 2× the data |
| `primary_balance / Y` | −1.42 (deficit) | — | implies wild divergence |

So in the **steady state**, the model already runs a primary deficit of ~140% of GDP. Once we start the transition, debt compounds at `(1+r)` and grows without bound.

The most likely arithmetic culprits, in order of suspicion:

1. **Pension level.** Pension formula is `PENS = ρ · w · κ_{J_R} · [λ z_last + (1-λ) z̄]`. With `ρ = 0.50`, `w ≈ 1.15`, `κ_{J_R} ≈ 1.06`, `z̄ ≈ 1`, this gives `PENS ≈ 0.61` per retiree-period. Aggregated across the retired share (≈35% of lifecycle), pensions are ~0.21 per-capita per period. Model `Y` is ~11.9 per period for the same denominator (population-weighted income). So `pensions / Y ≈ 0.21 / 11.9 ≈ 0.018` *per agent*. The reported 2.37 model number is ~130× higher — implies the model is **summing pensions across some aggregation that doesn't share the same denominator as `Y`**. Almost certainly an unweighted-sum-over-cohorts vs population-share normalization mismatch in `compute_fiscal_ratios()` or `_compute_ss_aggregates()` in `calibrate.py`, or in `compute_government_budget_path()` in `olg_transition.py`.

2. **Tax revenue normalization.** Same direction of mismatch — model collects 0.97 of Y when data shows 0.40. Either aggregating tax revenue without the right population weights, or dividing by an income aggregate that excludes some component (e.g., excludes pensions when tax is on total income).

3. **Units of `B_initial`.** `calibration_input_GR.json` sets `B/Y = 1.70`. The script computes `B_initial = B_over_Y * Y_path.mean()`. If `Y_path.mean()` is "model Y" (which is ~12) but the rest of the fiscal accounting operates in different units, `B_initial` is initialized in the wrong scale and accumulates incorrectly.

## What to do next session

Order of attack:

1. **Read** the steady-state aggregator: `_compute_ss_aggregates()` in `calibrate.py` (around line 524), and the per-period budget accumulator: `compute_government_budget_path()` in `olg_transition.py`.
2. **Add a units assertion**: build the per-period accounting via hand calculation for one period in one education stratum with `n_alpha=1` and tiny `n_sim`, and compare to what the function reports. Discrepancy should pinpoint the mis-weighting.
3. **Check the simulation tuple → aggregate mapping** in `_panel_to_age_means` and downstream: `effective_y_sim` includes wage and UI together; `pension_sim` is separate. If the aggregator sums them with different age-weights or different education-share multipliers, the levels diverge.
4. **Validate against a closed-economy sanity check**: build a tiny calibration where the agent's lifetime budget constraint is solvable by hand (constant wage, no health, no UI, no pension floor), simulate, and verify the reported `tax_revenue/Y` and `pensions/Y` match the analytical answer.
5. Once the unit/aggregation is right, the baseline B/Y path should be near-stationary at `1.70` and the bisection on τ_l should land at a reasonable Δτ_l (~1–5 pp for a 2%-of-Y G shock).

## Reproducer

The FE-on run that produced the symptoms:
```
source ~/venvs/jax-arm/bin/activate
JAX_PLATFORM_NAME=cpu python run_fiscal_figures.py \
    --config calibration_input_GR.json --shock G --backend jax
```

Outputs land in `code/output/fiscal_test/`. Full stdout log: `/tmp/fiscal_GR_FEon.log` (latest).

## Pointers

- Phase 8 work (where it landed, what changed): `code/docs/IMPLEMENTATION_PLAN.md` § Phase 8.
- Phase 9 plan (warm-glow bequest + initial wealth — separate from this fiscal issue): same file § Phase 9.
- Calibration report with the fiscal-ratio table: `code/output/calibration/calibration_GR_20260508_145856.md`.
- Hand calculation of expected pension/Y in the diagnosis section above.

## What is NOT broken (don't waste time re-checking)

- Phase 8 σ_α plumbing: smoke-tested, regression-tested. With `n_alpha=1` the new code path collapses to pre-Phase-8 bit-exact; with `n_alpha=5` simulated `Var(log y)` matches the LIS target.
- Plot generation: fixed in `599454b`.
- Income-moment double-counting: fixed in `599454b` (UI was being summed twice in `income_gini` and `p90_p10_income`).
- JAX backend on macOS: works with `JAX_PLATFORM_NAME=cpu`. The Metal float64 issue is documented; do not retry Metal.

---

## Resolution (2026-05-18)

**Root cause was a single bug:** `mu_y` in `calibration_input_*.json` is the unconditional mean of log y per education stratum (per LIS estimation `data/lis/02_estimate_ar1.py:236`), but `lifecycle_perfect_foresight.py:_income_process()` was passing it as the AR(1) intercept to `tauchen()`. With ρ=0.95 this scaled the unconditional log-y mean by 20×. For Greek "high" edu (`mu_y=0.259`), the discretized stationary mean of log y was 5.18 instead of 0.259, producing y-grid levels around 110-290 instead of 0.8-2.1. Cross-section weighted mean y was 55 (should be 1.1).

This single error explained the entire fiscal-ratio blow-up:
- inflated `y_last` ⇒ inflated pensions
- inflated `effective_y` ⇒ inflated tax revenue
- inflated `Y` ⇒ understated `health/Y` (numerator was independent of y, denominator wasn't)
- inflated household income ⇒ `C > Y` (resource constraint violated)

The aggregator (`_compute_ss_aggregates`, `compute_government_budget`) was **correct** — numerator and denominator were always in the same per-living-person units. The bug was upstream, in the income-process discretization.

### Commits

1. **`8efa408`** — `mu_y` fix at `lifecycle_perfect_foresight.py:467`. Pass `(1-rho_y)*mu_y` as the intercept so `mu_y` is the unconditional mean of log y. Includes:
   - `test_income_process.py` (new) — data-driven regression test that auto-discovers `calibration_input_*.json` and asserts `mean(log y_grid) == mu_y` to 1e-12.
   - Updated default `edu_params` in `LifecycleConfig` so existing tests' y-grids stay bit-identical.
   - `lifecycle_jax.py` — auto-set `JAX_PLATFORMS=cpu` on macOS ARM (no more env-var ritual).
   - `IMPLEMENTATION_PLAN.md` — Phase 10 (test-suite audit) added.

2. **`99313fb`** — Class 1: data-target triage. Updates JSON fiscal targets to Greek 2023 data values; replaces single `health_over_Y` with `health_{gov,oop,total}_over_Y`; introduces `r_B` (sovereign rate) distinct from `r` (private K return). Code changes in `calibrate.py`, `olg_transition.py`, `eval_fiscal_results.py`, `run_fiscal_figures.py`.

3. **`93cec78`** — Class 1 fixup: Greek total health spending is ~8% of GDP (OECD), not 5.4% (data-sheet partial). Updated targets to match and rescaled `m_good` accordingly.

4. **`9c0bf90`** — Class 2: pension generosity. `pension_replacement_default 0.50 → 0.25`, `pension_min_floor 0.40 → 0.15`. Reason: model pension formula `ρ·w·y_last·α_mult` omits hours, so ρ represents replacement of "full-time potential earnings" not realised earnings. With Greek headline replacement 76% and realised aggregate ratio ~50%, ρ_model ≈ 0.25 calibrates the model concept to the data flow.

### Post-recalibration state

Re-calibrated Greek baseline (n_sim=10000, JAX/CPU, 33 min):

| Calibrated param | Value |
|---|---|
| ν | 36.57 |
| β | 1.019 (above 1; effective β·survival well below 1) |

Targeted moments:

| Moment | Target | Model | Dev |
|---|---|---|---|
| A_over_Y | 4.0 | 4.003 | exact |
| average_hours | 0.41 | 0.488 | +19% |

Fiscal ratios (untargeted):

| Ratio | Model | Data | Dev |
|---|---|---|---|
| `interest/Y` | 0.034 | 0.034 | exact |
| `K/Y` | 3.00 | (firm FOC) | exact |
| `health_total/Y` | 0.092 | 0.082 | +12% |
| `health_gov/Y` | 0.061 | 0.054 | +13% |
| `tax_revenue/Y` | 0.437 | 0.400 | +9% |
| `pensions/Y` | 0.221 | 0.160 | +38% |
| `ui/Y` | 0.011 | 0.006 | (small abs) |

Full-balance picture (including exogenous G/Ig/defense/transfers): primary balance ~ −6% of Y vs Greek 2023 +2%. The remaining gap is dominated by the pension overshoot.

Calibration report: `code/output/calibration/calibration_GR_20260518_184335.md`.

### Open items (as of 2026-05-18)

- **Tighten pensions further.** Model 0.221 vs target 0.16. Likely needs another small reduction in `pension_replacement_default` (0.25 → ~0.18) followed by re-calibration.
- **Wealth-distribution residuals** (`wealth_gini = 0.38` vs 0.58, `zero_wealth_fraction = 2.9%` vs 1.1%) — Phase 9 (warm-glow bequest + initial wealth distribution).
- **Hours overshoots +19%.** Class 3 of `docs/CALIBRATION_FIX_CHECKLIST.md` proposes adding φ (Frisch curvature) as a third free parameter to close the trade-off with A/Y.
- **Fiscal experiments not yet re-validated.** The `run_fiscal_figures.py` smoke test should be re-run on the post-bug-fix baseline to confirm the debt path no longer explodes. Expected: debt path stable around target B/Y=1.64; bisection on τ_l for tax-financed shock should converge to plausible Δτ (~1-3 pp).

---

## Session 2026-05-19: cohort-config bug + workflow fix

Pension tightening + delta lowering committed (`7f905cb`, `6f43d97`). Re-ran fiscal experiment, **debt still blew up to B/Y = 10,624%** in the baseline. The calibration's primary balance was sustainable (~−1% of Y), but the transition's apparent primary deficit was much larger. Two compounding causes identified.

### Cause 1: workflow gap (auto-write theta to JSON)

`calibrate.py` wrote SMM results only to a Markdown report; `build_olg_transition` read `calibration_input_GR.json` and picked up the *initial guesses* for `nu`, `beta`. Fiscal experiments ran at uncalibrated parameters silently.

Fix (`1578e30`): on SMM convergence, `calibrate.py` now writes calibrated theta to `_derived.theta` in the JSON. `build_lifecycle_config` reads `_derived.theta` and overrides the corresponding fields. Initial guesses in `model.*` are preserved on disk for traceability and as the next SMM's starting point.

### Cause 2: cohort lifecycle config built from scratch

`solve_cohort_problems` constructed each cohort's `LifecycleConfig(...)` with only an enumerated subset of fields (T, beta, gamma, n_a, n_y, n_h, retirement_age, paths, `_feature_kwargs`). All unlisted fields fell back to **dataclass defaults** — most critically `edu_params`, `n_alpha`, `wage_age_profile`, `kappa`, `m_good`, `pension_avg_weight`, and the initial-condition fields.

For the Greek baseline this meant cohort lifecycle models silently ran with `mu_y = {5/3, 10/3, 4.0}` (the defaults' rescaled values) and `n_alpha = 1`, producing a `y_grid` for medium edu of `[0, 21.9, 25.8, 30.4, 35.9]` instead of the Greek-calibrated `[0, 0.58, 0.83, 1.20, 1.73]`. Cohort effective_y was 5–9× the standalone lifecycle value, which is exactly what blew up the transition's primary deficit and debt path. The bug had been silent: tests built OLGTransition with a fully-populated lifecycle config and never noticed `solve_cohort_problems` discarded most of it.

Fix (`f2b0e9f`): replace `LifecycleConfig(...)` with `self.lifecycle_config._replace(...)` at three sites — `ss_config` (line 1011), per-cohort `config` (line 1101), and the MIT-baseline `base_config` (line 1150). `_replace` preserves every field on the base lifecycle_config and overrides only the cohort-specific paths and feature kwargs.

**Verification** (medium edu, birth_period=0, n_sim=500):

| | Before fix | After fix | Standalone |
|---|---|---|---|
| `n_alpha` | 1 | 5 | 5 |
| `y_grid[1]` | 21.90 | 0.577 | 0.577 |
| cohort `l_m[0]` | 4.65 | 0.90 | 0.83 |

### Post-fix fiscal experiment

Re-run of G-shock fiscal experiment (`run_fiscal_figures.py --config calibration_input_GR.json --shock G --backend jax`, ~49 min on JAX/CPU, n_sim=2000):

| Quantity | Pre-cohort-fix | **Post-cohort-fix** |
|---|---|---|
| `mean(Y)` warmup | 8.45 | **0.537** (matches calibration's per-capita 0.561) |
| `B_initial` | 13.86 | 0.88 |
| Baseline final B/Y | 10,624 % | **961 %** |
| G-shock debt B/Y | 11,735 % | 2,074 % |
| G-shock τ_l Δτ_l | +114 pp | **+6.64 pp** (plausible Greek policy) |
| Cumulative multiplier | 0.000 | 0.000 |

`mean(Y)` now matches the calibration's per-capita Y — the 16× unit mismatch is gone. The bisection produces a credible Greek fiscal-policy move (raise `τ_l` by 6.6 pp to stabilise debt).

Baseline B/Y still grows to 961 % over 80 periods, but this is now a **real economic result**, not a unit bug. Back-of-envelope with `r_B = 0.021`, residual primary deficit ≈ 1.1 % of Y, and pop growth `g = −0.57 %`:

```
d_80 = (1+r_B)^80 · d_0 + deficit · [(1+r_B)^80 − 1] / r_B
     = 5.27 · 1.64 + 0.011 · 4.27 / 0.021 ≈ 10.9   (~1090 %)
```

Observed 961 % is consistent. The residual primary deficit comes from the structural L/Y and C/Y gaps documented in `docs/CALIBRATION_FIX_CHECKLIST.md` (Class 3+) and the +38 % pension overshoot from the morning's run.

### Open items going forward

- **Cumulative multiplier = 0.000** — worth a 30-min check. Could be Ricardian-equivalence pattern in the lifecycle model (forward-looking agents internalise future taxes) or a computation issue in `compare_scenarios`.
- **Close the residual primary-deficit gap** (~3 pp of Y). Sources: tax base mismatches (L/Y mechanically 0.67 vs Greek 0.36), pension overshoot, missing "Other revenues" / "Other primary spending" lines.
- **Phase 9** (warm-glow bequest + initial wealth) remains the path for the wealth-distribution residuals.
- **Class 3** (free φ as third SMM parameter) remains the path for closing the hours fit.

---

## Session 2026-05-20 / 2026-05-21: solver-architecture audit + sovereign-debt accounting fix

### What was done

1. **Tree 2 audit** of `code/docs/solver_architecture.md` against the live `fiscal_experiments.py` and `olg_transition.py`. Twelve corrections applied: dispatcher field is `scenario.financing` (not `scenario.balance`); NFA constraint is a one-sided floor (not a corridor); `fiscal_multiplier` is undiscounted; budget identity had spurious `r·NFA_gov` revenue term and a missing `defense_spending` spending line; the doc's `_simulate_sequential` was misplaced (the top-level call is `_ensure_cohort_panel_cache`); signatures of `_apply_shock`, `_extract_cohort_path`, `_compute_output_path_njit` corrected.
2. **Interest double-count exposed.** `compute_government_budget` defined `primary_deficit = total_spending − total_revenue` where `total_spending` already included `debt_service = r_B · B_t`. `compute_debt_path` then accumulated `B[t+1] = (1+r_path[t]) · B[t] + primary_deficit[t]`, adding interest again via the `(1+r)·B` term. Sovereign rate `r_B` was scalar (default `None`, fallback to capital `r`).
3. **Fix applied (commit `8c45250`)**:
   - `debt_service` dropped from `total_spending`. The field labelled `primary_deficit` is now the textbook primary deficit (spending excluding interest, minus revenue). `debt_service` is still reported as a separate budget line.
   - `r_B_path` introduced: built once per `simulate_transition()` call as the scalar `r_B` broadcast to length `T_transition`, falling back to the capital `r_path` when `r_B is None`. Lives on `OLGTransition.r_B_path`; also plumbed through `base_paths['r_B_path']` so all three financing branches (`run_debt_financed`, `run_tax_financed`, `run_nfa_constrained`) pick it up.
   - `compute_debt_path` renamed its second positional arg `r_path → r_B_path`. Recursion is now `B[t+1] = (1 + r_B_path[t]) · B[t] + primary_deficit[t]`.
   - `_balance_residual`'s `r_terminal` now sourced from `r_B_path[-1]`, so `terminal_flow_balance`'s `(g − r) · target` references the sovereign rate as it should.
   - No test changes required: with `r_B = None` in test configs, `r_B_path` mirrors `r_path`, preserving exact prior behaviour. (`test_sovereign_debt_in_budget` keeps passing because `r_B_path[0] = r_path[0] = 0.04`.)

### Post-fix G-shock run (Greek config, JAX/CPU, 43 min)

`python run_fiscal_figures.py --config calibration_input_GR.json --shock G --backend jax`

| Quantity | 2026-05-19 (pre-fix) | **2026-05-20 (post-fix)** |
|---|---|---|
| Baseline final B/Y | +961 % | **−183 %** |
| G-shock debt-financed final B/Y | +2,074 % | +228 % |
| G-shock τ_l-financed Δτ_l | +6.64 pp | **+1.56 pp** |
| G-shock τ_l-financed final B/Y | (target 164%) | +27.6 % |
| Cumulative fiscal multiplier | 0.000 | 0.000 |
| mean(Y) baseline | 0.537 | 0.537 |

**Sign flip of baseline drift** (+961 % → −183 %) is the diagnostic signature of the fix. Pre-fix recursion `B[t+1] ≈ (1+r+r_B)·B + PD_primary` had B compounding at the sum of two rates; post-fix `B[t+1] = (1+r_B)·B + PD_primary` correctly compounds at one rate, but the same calibration's `Spd_excl_interest − Rev` is now strongly negative (primary surplus of roughly 10 pp of GDP sustained), so B/Y falls fast instead of growing.

### Diagnosis

The fix is mathematically correct. What it exposed: the baseline G/Y, tax rates, and `B_initial` in `calibration_input_GR.json` were implicitly aligned against the *mis-labelled* `primary_deficit` field (which contained `r_B · B`). With debt service removed, the same parameters imply a much larger primary surplus than the stationarity identity requires.

**SS stationarity** for `B/Y = b` requires `PD_primary / Y = (g − r_B) · b`. For Greece (`r_B = 0.021`, `g ≈ 0`, `b = 1.64`) this is a primary surplus of 3.4 % of GDP. The post-fix baseline is producing a primary surplus closer to 10 % of GDP — much too tight.

**SMM is unaffected.** Calibrated `(ν, β)` come from matching `(avg_hours, A/Y)`; neither moment depends on how the budget is labelled or how B accumulates. No retune of `(ν, β)` required.

### Open question: choose a fiscal closure

- **(a) SS-residual instrument.** Add one fiscal lever (residual transfer, lump-sum, or one tax) to the baseline and pin it so the model's `PD_primary / Y` equals `(g − r_B) · b₀`. Then `b₀ = B_initial / Y[0]` is a fixed point of the law of motion and the baseline B/Y is stationary. Smallest diff. Probably ≤ 50 LOC in `olg_transition.py` + JSON.
- **(b) Accept drift; recalibrate `B_initial` and target `b` to model long-run.** Treat data Greece as a transition state. Requires running the baseline to its long-run B/Y and using that as the data target. Conceptually heavier; affects every downstream comparison.

### Cumulative multiplier 0.000

Still open from 2026-05-19. The SOE pins `K_domestic = (K/L)·L` via firm FOC at exogenous `r`, so Y reacts to G only through `L` — and only if labor supply is elastic enough. Worth confirming this is "Ricardian-equivalence + SOE" by inspecting the per-period `multiplier_path` in `fiscal_results.json`, not a print-precision artifact. 30-min diagnostic.

### Pointers

- Solver writeup: `code/docs/solver_architecture.md` (Tree 1 = SS calibration, Tree 2 = fiscal transition; Tree 2 was rewritten this session).
- Fix commit: `8c45250` ("Use r_B for sovereign debt law of motion; fix interest double-count").
- Run log: `code/output/fiscal_test/run_G.log`.
- Numerical results: `code/output/fiscal_test/fiscal_results.json`.
