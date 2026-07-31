# Open issues — model/text audit, 2026-07-30

Findings from an audit of the paper draft (`docs/DSA-LSA *.tex`) against the code
behind the reported results. The reported run is
`output/fiscal_test_kg_rB0/` (r_B=0, K_g active, JAX, n_sim=2000, 14 July);
every figure in `docs/output/` is byte-identical to it and to
`output/health_flag/`. Calibration reference:
`output/calibration/calibration_GR_20260710_153659.md`.

No code was changed. Each item below is deferred; the text is being corrected to
describe current behaviour in the meantime.

---

## 1. Aggregate labour input includes UI benefits — Open

**Behaviour.** `L` is aggregated from `effective_y_sim = wage_income + ui_sim`
(`lifecycle_perfect_foresight.py:1205`) and divided by `w`
(`olg_transition.py:1518, 2102`). Verified in the output: at every `t`,
`w·L − tax_p/τ^p = UI` to all digits, and UI is **1.887 % of w·L**.

**Conflict.** Every written description has `L_t = ∫ κ_j z e^α ℓ dμ` — paper
`model.tex:128`, `code_report_2026-06-15.md:55`. Nothing documents the UI term.
It also breaks the implied identity `𝓑^lab = w_t·L_t`: the labour share is 0.670
while the payroll-tax base is 0.657 of Y.

**Fix (4 one-liners; the UI aggregate is already computed at each site).**

| Site | Change |
|---|---|
| `olg_transition.py:2088-2102` | `_ui_all` is unpacked and unused; aggregate it with the same weights into `UI_path[t]`, then `L_path = (L_path - UI_path) / w_path` |
| `olg_transition.py:1671-1679` (`compute_aggregates`) | subtract the weighted `px["ui_by_age_edu"]` before `L /= w_path[t]` |
| `calibrate.py:569` | `L = (agg['labor_income'] - agg['ui']) / spec.w` |
| `calibrate.py:1255` | same |

**Predicted effect.** `Y = A(K^g)^{η_g}(K/L)^α·L` is linear in L, so L, K and Y
all fall 1.887 %; `w`, `Y/L`, `K/Y` unchanged. `𝓑^lab` becomes exactly `w·L`, so
SSC/Y = τ^p(1−α) and the 0.130 target is hit at **τ^p = 0.194** instead of
0.1978. `A/Y` rises to 4.077 before re-fitting, so β falls. `K^g/Y` and `I^g/Y`
along the transition rise a further 1.9 % (see item 5).

---

## 2. Bequest circuit is open in the reported experiments — Open, decision needed

**Behaviour.** `FiscalScenario.recompute_bequests` defaults to `False`
(`fiscal_experiments.py:130`) and `run_fiscal_figures.py` never sets it, while
`olg_transition.py`'s own CLI defaults to `True`. With
`bequest_lumpsum_path=None` the newborn transfer is zero
(`olg_transition.py:2041-2056`). `output/fiscal_test_kg_rB0/run_rB0.log` has no
bequest-loop output, confirming the loop did not run. `tau_beq = 0.0` (absent
from both GR configs). `calibrate.py` never sets `bequest_lumpsum` either, so the
stationary calibration and the transition are consistently open and β = 0.942
was fitted with the leak.

**Consequence.** Accidental bequests of **3.77 % of Y per period** leave the
economy: dying agents' assets are recorded (`total_bequests` in the budget dict)
but neither taxed nor handed to newborns.

**Fix.** Set `recompute_bequests=True` on the six `FiscalScenario` constructions
in `run_fiscal_figures.py` (or flip the dataclass default), and add the same
closure to `calibrate.py`'s stationary cross-section so both sides agree. `A/Y`
rises, β falls.

**Decision.** Close the circuit, or keep it open as a stated modelling choice.

---

## 3. `z_last` is the previous income state, not the last *employed* one — Open, decision needed

**Behaviour.** While working, `z'_last = z` unconditionally, including `z = 0`:
`lifecycle_perfect_foresight.py:1244` (simulation) and `:973` (continuation-value
index); `lifecycle_jax.py:765` (simulation) and `:306-314`, where `EV_working` is
broadcast over the `y_last` axis so `P_y`'s row index doubles as `y_last_next`.
Frozen at retirement in both backends.

**Consequence.** `y_grid[0] = 0`, so a second consecutive unemployment period
pays **zero UI** — P(spell continues) = 1 − λ^find = 0.5. An agent unemployed in
the last working year retires on the (1−λ) career-average term only. Not
documented anywhere as intentional.

**Fix (4 edits).**
- `lifecycle_perfect_foresight.py:1244` → `if i_y[i] > 0: i_y_last[i] = i_y[i]`
- `lifecycle_perfect_foresight.py:973` → index the continuation value by `i_y if i_y > 0 else i_y_last`
- `lifecycle_jax.py:765` → `jnp.where(is_retired | (i_y == 0), i_y_last, i_y)`
- `lifecycle_jax.py:306-314` → replace the broadcast with
  `EV_full = einsum('ij,ajhl->aihl', P_y, EV_h)`; take the i-diagonal slice for
  i > 0 and the l-indexed slice for i = 0, selected with `jnp.where(i_y == 0, …)`.
  `EV_full` is (n_a, n_y, n_h, n_y) = 100×5×1×5 — no memory concern.

**Moment consequence to weigh first.** UI/Y is already 0.013 against 0.006 in the
data; freezing `z_last` raises it further.

---

## 4. `B_initial` is sized off the warmup simulation — Open

`run_fiscal_figures.py:119` computes `B_initial = B_over_Y · Y0` from a 50-draw
warmup (`Y(0) = 0.8816`, `run_rB0.log`), while the reported run has
`Y(0) = 0.8853`. Result: **B/Y = 1.633 at t=0, not 1.64**. Fix: raise the warmup
`n_sim`, or rescale `B_initial` after the first full baseline run.

---

## 5. Stationary `Y_ss = 1` does not carry to the transition — Open, decision needed

Calibration: `Y = 1.0005`. Transition `t=0`: **`Y = 0.8853` (−11.5 %)**;
`output/ss_vs_transition_postfix.log` records the same gap. Structural, not a
bug: `calibrate.py` uses the single-year `survival_probs` from the config, the
transition uses the cohort diagonals of `data/survival_GR.npz`. Every share of Y
is preserved to within 0.6 %, but the two lines specified as **levels** are not:

| Set at the stationary normalization | Along the transition |
|---|---|
| `K^g_0/Y = 0.745` (IMF ICSD 2019) | 0.8416 (+13 %) |
| `I^g/Y = 0.0353` (ICSD 2015–19) | 0.0399 (+13 %) |

**Option.** Re-target `normalize_A_tfp.py` on the transition's `Y(0)` instead of
the stationary `Y`; the residual is then a full baseline transition, usable at
`n_sim ≈ 200`. Changes the object being normalized. Otherwise the ICSD ratios
hold only where the calibration is reported, not where the figures are drawn.

---

## 6. Silent config fallbacks — Open (zero behavioural change)

`pension_avg_weight` is absent from both GR configs, so `calibrate.py:1093`
derives **λ = 0.4434** from `(1−ρ_z^{J_R})/(J_R(1−ρ_z))`. This is the intended
career-average approximation (`IMPLEMENTATION_PLAN.md:51`,
`calibration_plan.md:63,85`), but nothing in the config records it. Same for
`tau_beq`. Write both explicitly into `calibration_input_GR.json` and
`calibration_input_GR_rB0.json`.

---

## Re-run cost if items 1–4 are taken together

`run_scale_loop.sh` (SMM ↔ A-normalization ↔ closure re-pin, ~40–60 min at
n_sim=10,000) → `chain_fiscal_after_loop.sh` (**15.8 h GPU**, measured) →
`build_health_model_baseline.py` + `health_flag_decomposition.py`. Every number
in the paper's §4, §5 and appendix would then have to be re-derived.

---

## Verified consistent (no action)

Working and retired budget constraints and all four tax bases (τ^l on
(1−τ^p)·wage + UI and on pensions; τ^p on wages only; τ^k on r·a; τ^c on c);
OOP = (1−κ)m(j); HSub = κΣ_j m(j)N_{t,j}; production function and both FOCs;
`K_domestic`; NFA = A − K − B; B_{t+1} = (1+r_B)B_t + PD_t and the PD definition
including O_t; λ^sep = min(u/(1−u)λ^find, λ̄^sep) (never binding: 0.098/0.094/0.056);
Tauchen with 4 employed nodes plus z=0; Gauss–Hermite α with n_α=5;
`m_age_profile` mean 0.999995; survival table 1961–2023 × real ages 25–84;
SMM weights 1/m_data²; logit-transformed Nelder–Mead with a differential-evolution
option (`calibrate.py:258, 722-801`); MIT predetermination (A[0] identical to 8 dp
across all eight scenarios); bisect tol 1e−3; N_POST = 20. All parameter values
in the paper's tables match the configs to the reported precision except items 4
and 5 above; all entries in the baseline-moments table match the calibration
report; all reported experiment numbers match `fiscal_results.json`.
