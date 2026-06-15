# Public capital (K_g) activation + τ_l-matches-baseline-debt — handoff plan

Status as of 2026-06-15. Picks up an in-progress change set; nothing below is committed.

## 1. Goal

Two independent threads from this session:

**(A) τ_l closure that matches the baseline transition's terminal debt.** Done (code applied,
byte-compiles, not yet run).

**(B) Turn on public capital** with `eta_g=0.05`, `K_g=1` normalized at the initial steady
state, so the I_g shock has a production channel. Parameter edits applied; the supporting
normalization + δ_g + I_g handling are NOT yet done.

## 2. What changed this session (uncommitted)

- `code/run_fiscal_figures.py`
  - Dropped static `balance_condition`/`target_debt_gdp` from `scn_g_taul` and `scn_ig_taul`.
  - In `run_experiment_set`, after the baseline runs: set the τ_l closure to
    `balance_condition='terminal_debt_gdp'`, `target_debt_gdp = B_base[T_bal]/Y_base[T_bal-1]`
    (baseline transition's terminal debt/GDP), and print the resolved target.
  - Shared by both the `--config` and hardcoded-test branches, so (A) is live in config mode.
- `code/calibration_input_GR.json` → `production` block:
  - `K_g`: 0.0 → **1.0**
  - `eta_g`: 0.0 → **0.05**
  - `delta_g`: still **0.0**  ← MUST be set before any config run (see §4).

## 3. Design decision (agreed)

`K_g=1` is a neutral normalization at t=0 only because `K_g_factor = K_g^eta_g = 1` (calibrate.py
~915; olg_transition.py:817). For `K_g=1` to be an actual initial **steady state**, the public-
capital law of motion `K_g[t]=(1−δ_g)K_g[t−1]+I_g[t−1]` (olg_transition.py:1914–1915) needs
`I_g_ss = δ_g·K_g`. Two consequences:

1. **δ_g is pinned by ratios, not levels:** `δ_g = (I_g/Y) / (K_g/Y)`. No Y level required.
2. **Normalize initial-SS Y to 1** (via `A_tfp`) so `K_g=1 ⟺ K_g/Y=1`, then `δ_g = (I_g/Y)/1 =
   I_g/Y`. Caveat: Y_ss is endogenous (`Y_ss=(Y/L)_ss·L_ss`, L_ss from the household block) and
   with CRRA (γ=2) + separable labor disutility hours respond to the wage level, so Y_ss is NOT
   proportional to A_tfp. Normalizing means **adding a 1-D calibration target: solve for the
   scalar A_tfp s.t. initial-SS Y = 1.** It is not a closed-form rescaling.

**Avoid the fixed point.** Passing baseline I_g as a constant **level** `I_g = δ_g·K_g` keeps K_g
exactly at 1 in the baseline, so there is no I_g↔K_g↔Y simultaneity and the `I_g_over_Y` +
`eta_g≠0` rejection (olg_transition.py:1892–1896) never fires. The I_g shock adds on top. (The
earlier "I_g = constant share of Y(t)" route would require an outer fixed-point loop wrapping the
whole perfect-foresight solve — explicitly NOT the chosen path.)

## 4. Pending code changes

Ordered; each is small.

1. **Set δ_g as a primitive in the config** (`production.delta_g`). Value = target `I_g/Y`
   divided by target `K_g/Y` (= `I_g/Y` once Y_ss=1, K_g/Y=1). With I_g/Y=0.03 and K_g/Y=1 →
   δ_g=0.03. If the IMF K_g/Y target (§5) differs, recompute: δ_g=(I_g/Y)/(K_g/Y).
   - Note: data I_g/Y (DATA sheet, code 45) = 3.86% in 2023; config currently has
     `fiscal.I_g_over_Y = 0.03`. Decide whether to update to 0.0386 (§6).

2. **Config-mode I_g: switch from share to level** in `code/run_fiscal_figures.py` `--config`
   branch (lines ~89, ~102, ~178). With `eta_g≠0`:
   - Replace `base_paths['I_g_over_Y'] = I_g_over_Y` with a constant level path
     `base_paths['I_g_path'] = np.full(T_TR, delta_g * K_g_initial)` (= δ_g since K_g=1).
   - The warmup `I_g_warmup` (line 89) already computes `δ_g·K_g`; reuse that value as the level.
   - Keep `I_g_over_Y` ratio mode only when `eta_g==0` (backward compatible).

3. **A_tfp normalization (Y_ss=1)** — add to the calibration step (`calibrate.py`). One scalar
   root-find: choose `A_tfp` s.t. the initial stationary solve returns Y=1. Confirm where the
   initial-SS Y is computed and whether `pin_baseline_closure.py`'s stationary solve can be
   reused as the inner evaluation. This recalibration interacts with the SMM block — sequence it
   so A_tfp is solved with the other externally-set params fixed, then re-run SMM if needed.

4. **Re-pin fiscal closure.** `fiscal.other_net_spending_over_Y` was pinned at the initial SS
   (`pin_baseline_closure.py`). Changing A_tfp (and adding baseline I_g spending if I_g/Y moves)
   shifts the initial-point budget, so re-run `pin_baseline_closure.py --write` after §4.3.

5. **Tests / sanity.** With `eta_g≠0`: confirm no `I_g_over_Y` path reaches `simulate_transition`
   (else line-1892 raise). Check `check_a0_predetermination.py` still passes (Ig shock path is the
   one sensitive to K_g→w; see MEMORY MIT-stitching bug fixed 2026-03-06).

## 5. Data search — IMF (K_g/Y target)

The repo has **no public-capital stock series** — only flows (I_g/Y, I/Y) in `data/DATA_GR.xlsx`
DATA sheet. To discipline `K_g/Y` rather than choose it:

- **Source:** IMF Investment and Capital Stock Dataset (FAD, "IMF Investment and Capital Stock
  Dataset" / IMFInvest). Reports general-government capital stock for Greece, level and as % of
  GDP, plus private and PPP capital. Public capital is built by IMF via PIM from public investment.
- **Fetch:** download the dataset (Excel), extract Greece general-government capital stock /
  GDP for the reference year (match the model's reference year — 2023 used elsewhere, e.g. debt
  B/Y=1.64, primary balance). Record the value and vintage.
- **Save:** add to `data/` (new file, e.g. `IMF_ICSD_GR.xlsx` or an extracted CSV) and document in
  `code/data_inventory.md` (new row under §1.2 or a new external-source subsection), mirroring how
  other external series are logged.
- **Use:** set `K_g/Y` target = IMF value; recompute `δ_g=(I_g/Y)/(K_g/Y)`; if normalizing Y_ss=1,
  set `K_g` level so K_g/Y matches (with Y_ss=1, `K_g = K_g/Y`). NOTE: this may move K_g off 1.0
  — decide whether to keep the K_g=1 normalization (and let K_g/Y be whatever Y_ss delivers) or
  pin K_g/Y to the IMF number (and let K_g≠1). See open decision §6.
- **Cross-check:** PIM the DATA-sheet I_g/Y series (1995–2024, ~30 yrs) under the chosen δ_g and
  compare the implied K_g/Y to the IMF figure as a consistency check (seed-sensitivity over 30 yrs
  is a known weakness).
- Do NOT quote an IMF K_g/Y number from memory; pull it from the dataset.

## 6. Open decisions for next session

1. **K_g=1 vs K_g/Y=IMF.** Keep the clean `K_g=1` normalization (K_g/Y falls out of Y_ss), or pin
   `K_g/Y` to the IMF value (K_g≠1, δ_g from the two ratios). The latter disciplines the public-
   capital channel to data; the former is cleaner bookkeeping.
2. **I_g/Y = 0.03 vs 0.0386 (2023 data).** δ_g and the baseline I_g level both depend on this.
3. **Reference year** for the K_g/Y (and I_g/Y) target — 2023 is the default used elsewhere.

## 7. Verification once implemented

- `python -m py_compile` on edited files.
- Config-mode dry run: `python run_fiscal_figures.py --config calibration_input_GR.json --shock Ig`
  (full sim, >2 min — run with `Bash run_in_background:true` + a divergence Monitor per the
  user's run-policy). Confirm: no line-1892 raise; baseline K_g[t]≈1 flat; printed τ_l
  `target_debt_gdp` = baseline terminal B/Y; bisection converges.
- Confirm initial-SS Y≈1 after A_tfp normalization.
- Re-run `pin_baseline_closure.py` and check the t=0 primary balance vs `primary_balance_target`.

## 8. Key file:line references

- τ_l closure injection: `run_fiscal_figures.py` `run_experiment_set` (~line 266).
- I_g_over_Y rejection: `olg_transition.py:1892–1896`.
- K_g law of motion: `olg_transition.py:1911–1915` (uses lagged I_g[t−1]).
- K_g→w: `olg_transition.py:1934–1941`. K_g_factor in production: `:817`.
- Equilibrium prices / K_g_factor: `calibrate.py:901–927` (`compute_equilibrium_prices`).
- Config I_g handling: `run_fiscal_figures.py:88–113, 174–180`.
- I_g/Y data: `data/DATA_GR.xlsx` DATA sheet, code 45 (col index 12): 2023=0.0386.
