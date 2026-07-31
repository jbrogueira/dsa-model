# The economic problem (model terms only)

This document states the economic problem the code solves, in model terms only; it makes no reference to the code. The companion implementation document maps each object to the code. Notation authority is the compiled source `docs/DSA-LSA model.tex` (with `government_sector` quantities and `DSA-LSA experiments.tex`); the subject is what the code actually solves. Where the two diverge, the divergence is a footnote and is collected in §10. The non-compiled richer draft in `DSA-LSA main.tex` (human capital, health state, CES capital, child/schooling costs, long-term bonds) is not implemented and is not described here.

## 1. What is being solved

A small open economy populated by overlapping generations of finitely-lived, heterogeneous households (Greece). The world interest rate path `{r_t}` is exogenous; given `r_t` the firm's first-order conditions pin the private capital–labor ratio and the wage `w_t`. Households face idiosyncratic productivity and employment risk, retire at a fixed age, receive pensions, and die stochastically leaving accidental bequests. The government levies consumption, labor, payroll, and capital taxes, pays pensions, unemployment insurance, and a health subsidy, consumes goods, invests in public capital, and issues sovereign debt to foreigners.

The experiment is a debt-sustainability analysis: a permanent fiscal shock — either to government consumption `G` or to public investment `I^g` — financed either by issuing debt or by raising the labor tax, computed as a perfect-foresight transition between the pre-shock steady state and the post-shock path. The case is: small open economy, exogenous `r`; perfect-foresight transition; partial equilibrium in prices (factor prices follow from `r_t` and `K^g_t`, not from a capital-market clearing condition) with general-equilibrium fiscal feedback through the debt and tax instruments.

## 2. Environment

Time is discrete, `t = 0, 1, …`. Households live for `J = 60` periods indexed by age `j ∈ {1, …, J}` (model age `j` = calendar age `24 + j`, so the cohort spans ages 25–84). Two stages: working `j ∈ 𝒲 = {1, …, J_R}` and retired `j ∈ ℛ = {J_R+1, …, J}`, with fixed retirement at `J_R = 39` (statutory age 64).¹ Survival from age `j` to `j+1` is stochastic with probability `π(j) ∈ (0,1]`, age-dependent and cohort-specific.² Cohort sizes grow at the exogenous population rate `g_N` (negative — population aging); a cohort of age `j` carries weight `Λ^{-j}` with `Λ = 1 + g_N`.

Households are heterogeneous in: education `e ∈ {low, medium, high}` (fixed at birth, education-specific income process and unemployment rate); a permanent productivity fixed effect `α_i`, drawn at birth from a mean-zero distribution with education-specific variance `σ_α²` and discretized on a 5-node Gauss–Hermite grid; the idiosyncratic productivity state `z`; and the last-employed productivity state `z_last` (the pension and unemployment-benefit base).

Preferences are time-separable with discount factor `β` and survival-adjusted continuation. Period utility over consumption `c` and labor `ℓ ∈ [0,1]` is CRRA in consumption plus iso-elastic labor disutility,
$$u(c, \ell) = \frac{c^{1-\sigma}}{1-\sigma} - \nu\,\frac{\ell^{1+\varphi}}{1+\varphi}, \tag{1}$$
with risk aversion `σ = 2`, labor-disutility scale `ν`, and inverse Frisch elasticity `φ = 1.5` (intensive-margin Frisch `1/φ ≈ 0.67`). Labor disutility applies only to the employed working-age.

## 3. Agents' problem

### 3.1 Idiosyncratic income

Productivity is a Markov chain `z ∈ {0, z_1, …, z_{n_y-1}}` (`n_y = 5`), where `z = 0` is unemployment and the employed states `{z_1, …, z_4}` are a Tauchen discretization of a log-normal AR(1),³
$$\log z' = \rho_z \log z + \eta', \qquad \eta' \sim N(0, \sigma_\eta^2), \tag{2}$$
with education-specific `ρ_z = 0.95` and `σ_η ∈ {0.054, 0.086, 0.075}`. The level of productivity that enters wage income is scaled by a deterministic age profile `κ_j` and by the permanent effect through `e^{α_i}`. Employed wage income is
$$y^L = w_t\,\kappa_j\,z\,e^{\alpha_i}. \tag{3}$$
The last-employed state evolves as `z_last' = z·𝟙[z>0] + z_last·𝟙[z=0]`. Transitions in and out of unemployment use a job-finding rate `λ^{find}` and a separation rate `λ^{sep} = min((u/(1-u))·λ^{find}, λ̄^{sep})` with education-specific unemployment rate `u`.

### 3.2 Budget constraints

Working-age `j ∈ 𝒲`, with assets `a` (beginning of period) and bound `a' ≥ a̲`:
$$(1+\tau^c)\,c + a' = \big(1 + (1-\tau^k) r_t\big) a + \underbrace{y^L + T^{UI} - \tau^p y^L - \tau^l\big(y^L - \tau^p y^L + T^{UI}\big)}_{\text{after-tax labor income and UI}} - (1-\kappa)\,m_j + \beta_t^{lump}, \tag{4}$$
where the payroll tax `τ^p` falls on wage income `y^L` only, the labor income tax `τ^l` falls on the base `y^L - τ^p y^L + T^{UI}` (payroll-deductible, UI taxable, §10 item 8); `T^{UI} = ρ^{ui} w_t κ_j z_last e^{α_i}` if `z = 0`, else `0`; `m_j` is age-dependent⁴ medical cost of which the government covers a share `κ` and the household pays `(1-κ)m_j` out of pocket; `β_t^{lump}` is the lump-sum bequest transfer (§3.4); and a minimum-consumption floor `c̲` is enforced by an additional transfer when (4) would otherwise imply `c < c̲`. Capital income `r_t a` is taxed at `τ^k`, consumption at `τ^c`, labor income at `τ^l + τ^p`.

Retired `j ∈ ℛ` (labor `ℓ = 0`):
$$(1+\tau^c)\,c + a' = \big(1 + (1-\tau^k) r_t\big) a + (1-\tau^l)\,\text{PENS} - (1-\kappa)\,m_j + \beta_t^{lump}, \tag{5}$$
the pension taxed at `τ^l` but not `τ^p`. The pension is set at retirement from the wage and a blend of last-employed and career-average productivity, with a floor:
$$\text{PENS} = \max\!\Big(\rho^{pens}\, w_t\, \kappa_{J_R}\,\big[\lambda\, z_{last} + (1-\lambda)\,\bar z\big]\, e^{\alpha_i},\; b_{\min}\Big), \tag{6}$$
where `ρ^{pens}` is the replacement rate, `λ` weights the last-employed state against mean employed productivity `z̄`, and `b_min` is the floor. The live configuration leaves `λ` unset, so it is derived from the career-average approximation `λ = (1 - ρ_z^{J_R})/(J_R(1-ρ_z)) ≈ 0.443` (at `ρ_z = 0.95`, `J_R = 39`): a near-even blend, not last-state-only.⁵

### 3.3 Recursive problem

The individual state is `(j, z, z_last, a)` (and the fixed type `(e, α_i)`); the aggregate state is `S_t = (r_t, w_t, K^g_t, μ_t)`. Working-age,
$$V_j(z, z_{last}, a; S_t) = \max_{c,\,\ell,\,a' \ge a̲}\; u(c,\ell) + \beta\,\pi(j)\,\mathbb{E}\big[V_{j+1}(z', z_{last}', a'; S_{t+1}) \,\big|\, z\big] \quad \text{s.t. (4).} \tag{7}$$
Retired,
$$V_j^R(z_{last}, a; S_t) = \max_{c,\,a' \ge a̲}\; u(c,0) + \beta\,\pi(j)\,\mathbb{E}\big[V_{j+1}^R(z_{last}, a'; S_{t+1})\big] \quad \text{s.t. (5).} \tag{8}$$
The terminal period `j = J` has no continuation. The intratemporal labor choice of the employed satisfies the first-order condition⁶
$$\nu\,\ell^{\varphi} = c^{-\sigma}\,\frac{MW}{1+\tau^c}, \qquad MW = w_t\,\kappa_j\,z\,e^{\alpha_i}\,(1-\tau^p)(1-\tau^l), \tag{9}$$
bracketed to the feasible region `c(\ell) > 0`.

### 3.4 Bequests

Assets of the deceased are taxed at `τ^β` and the after-tax remainder is redistributed lump-sum to surviving members of the same cohort, giving the per-survivor transfer `β_t^{lump}` in (4)–(5). The transfer is a fixed point: it depends on aggregate accidental bequests, which depend on the saving policy, which in turn depends on the transfer.

## 4. Technology and equilibrium

Output is Cobb–Douglas in private capital `K_t`, effective labor `L_t`, and public capital `K^g_t`:
$$Y_t = A\,(K^g_t)^{\eta_g}\,K_t^{\alpha}\,L_t^{1-\alpha}, \tag{10}$$
with private capital share `α = 0.33`, public-capital elasticity `η_g`, and TFP `A = 1`. Given the exogenous `r_t`, the firm's first-order conditions pin the capital–labor ratio and the wage:
$$\frac{K_t}{L_t} = \left(\frac{\alpha A (K^g_t)^{\eta_g}}{r_t + \delta}\right)^{\frac{1}{1-\alpha}}, \qquad w_t = (1-\alpha)\,A\,(K^g_t)^{\eta_g}\left(\frac{K_t}{L_t}\right)^{\alpha}, \tag{11}$$
with private depreciation `δ = 0.05`. Public capital accumulates from public investment:
$$K^g_{t+1} = (1-\delta_g)\,K^g_t + I^g_t. \tag{12}$$

Aggregate effective labor and the private capital stock follow from the household distribution `μ_t` and the firm ratio (11):
$$L_t = \int_{j\in\mathcal{W}} \kappa_j\, z\, \ell\, e^{\alpha_i}\, d\mu_t, \qquad K_t = \frac{K_t}{L_t}\cdot L_t. \tag{13}$$

Household wealth `A_t = ∫ a\, dμ_t` need not equal `K_t + B_t`; the residual is net foreign assets,
$$\text{NFA}_t = A_t - K_t - B_t, \tag{14}$$
with `B_t` sovereign debt. The current account is `ΔNFA_t`, and the goods-market identity is `C_t + I_t + G_t + I^g_t + ΔNFA_t = Y_t`, with private investment `I_t = K_{t+1} - (1-δ)K_t`.⁷

## 5. Government and budget constraint

Revenues and outlays, all integrated over the period distribution `μ_t`:

| Line | Definition |
| :-- | :-- |
| Consumption tax | `Rev^c_t = τ^c_t C_t` |
| Labor income tax | `Rev^l_t = τ^l_t 𝓑^{lab}_t`, base `𝓑^{lab}_t = ∫_{𝒲} w_t κ_j z e^{α_i} ℓ\, dμ_t` |
| Payroll tax | `Rev^p_t = τ^p_t 𝓑^{lab}_t` |
| Capital income tax | `Rev^k_t = τ^k_t r_t A_t` |
| Bequest tax | `BeqTax_t = τ^β ∫ (1-π(j)) a\, dμ_t` |
| Unemployment insurance | `UI_t = ∫_{𝒲,\,z=0} T^{UI}\, dμ_t` |
| Pensions | `PENS^{out}_t = ∫_{ℛ} \text{PENS}\, dμ_t` |
| Health subsidy | `HSub_t = κ ∫ m_j\, dμ_t` |
| Government consumption | `G_t` |
| Public investment | `I^g_t` |
| Defense | `D_t` |
| Other net primary | `O_t` (residual closure, §9)⁸ |

The primary deficit is
$$\text{PD}_t = G_t + I^g_t + D_t + O_t + UI_t + PENS^{out}_t + HSub_t - \big(Rev^c_t + Rev^l_t + Rev^p_t + Rev^k_t + BeqTax_t\big), \tag{15}$$
and sovereign debt evolves at the sovereign service rate `r^B`,⁹
$$B_{t+1} = (1 + r^B)\,B_t + \text{PD}_t. \tag{16}$$
An optional pension trust fund accumulates `S^{pens}_{t+1} = (1+r_t) S^{pens}_t + Rev^p_t - PENS^{out}_t`. The progressive labor schedule `τ^l(y) = 1 - κ^{tax} y^{-η^{tax}}` is available but off in the baseline (flat `τ^l`).

## 6. The policy / experiment

Two permanent fiscal shocks, each of size `Δ = 0.02·Ȳ` (2% of mean baseline output), announced at `t = 0` with full perfect foresight over all future paths:

- **G shock:** `G^{cf}_t = G^{base}_t + Δ` for all `t ≥ 0`.
- **`I^g` shock:** `I^{g,cf}_t = I^{g,base}_t + Δ`, accumulating in `K^g_t` via (12), raising `Y` via (10) and `w` via (11).

**Eligibility.** The shock is economy-wide; no subset of cohorts, education types, or income states is targeted or exempt. It enters every household's problem only through the aggregate prices and tax rates and, for the `I^g` shock, through public capital `K^g_t` in (10)–(11). The shocked spending line is itself a pure government outlay (it does not appear in any household budget).

**Baseline:** constant policy paths; `r = 0.04`; `G^{base} = 0.13·Y_t`; `I^{g,base}_t = δ_g K^g_0` (sustains steady-state public capital); initial debt `B_0 = 1.64·Y_0`. Baseline and counterfactual coincide at `t = 0`: initial wealth `A_0` and the distribution `μ_0` are predetermined (§7).

**Financing regimes and the fate of every budget term.** Both regimes leave the unshocked spending lines at their baseline paths: the other of `{G, I^g}`, defense `D_t`, pensions `PENS^{out}_t`, unemployment insurance `UI_t`, health subsidy `HSub_t`, and the other-net residual `O_t` each follow the baseline (the `Y(t)`-share lines re-scale to the realized counterfactual output; the residual `O_t` is held at its pinned share, §9). The shocked line follows its post-shock path above. (i) **Debt-financed:** all tax rates `(τ^c, τ^l, τ^p, τ^k)` stay at baseline; sovereign debt `B_t` is the residual of (16). (ii) **Labor-tax-financed:** a constant increment `Δτ^l` is added to the baseline labor-tax path for all `t ≥ 0` (the other three tax rates unchanged), `Δτ^l` set by a one-dimensional search so the counterfactual's terminal debt ratio matches a target. The source condition is the flow target (17); the live experiment script instead matches the counterfactual terminal `B/Y` to the baseline transition's terminal `B/Y` (a stock-ratio target, §10 item 12).

**Terminal condition.** With transition horizon `T_{trans} = 60`, the labor-tax case targets the primary balance that stabilizes debt at its initial ratio `b^* = B_0/Y_0`:
$$\frac{\text{PD}_{T_{trans}-1}}{Y_{T_{trans}-1}} = (g - r)\,b^*, \tag{17}$$
where `g` is the population growth rate. This is a finite-horizon target imposed at `T_{trans}-1`, not a fully converged long-run steady state; aggregates need not have converged to a stationary path by `T_{trans}`.

**Counterfactual.** Each shock-financing pair is compared against the no-shock baseline; the fiscal multiplier in the debt-financed case is
$$\mathcal{M}_t = \frac{Y^{cf}_t - Y^{base}_t}{\Delta}. \tag{18}$$

Mechanism (exogenous `r_t`): under debt financing the G shock leaves `K_t/L_t`, `Y_t`, `w_t` unchanged through (11), so `𝓜 ≈ 0` while `B_t/Y_t` rises; the `I^g` shock raises `Y` and `w` through the `K^g` channel, giving a positive, growing multiplier and a smaller financing tax than the G shock.

## 7. Equilibrium concept and the solution object

**Steady state:** households solve (7)–(8) under constant `(r, w, τ, ρ^{pens})`; the cohort age structure is stationary; aggregates follow from the optimal plans. Prices are not iterated — `w` follows from (11) given `r` and `K^g`.

**Transition:** a perfect-foresight path of length `T_{trans}`. Every cohort born at or after `t = 0` knows the entire future of `(r_t, w_t, τ_t, ρ^{pens}_t, K^g_t)`. The initial wealth `A_0` and the distribution `μ_0` are predetermined and identical across counterfactuals (an MIT-shock boundary condition): cohorts alive before `t = 0` hold their pre-shock policies for their pre-transition ages, so the counterfactual cannot move wealth that was chosen before the announcement. The solution object is the family of age-profiles `{V_j, c_j, ℓ_j, a'_j}` per cohort and education, the implied aggregate paths `{K_t, L_t, Y_t, C_t, A_t, K^g_t, B_t, \text{NFA}_t}`, and the government budget path. No capital-market fixed point is solved; the only outer fixed points are the within-cohort bequest transfer (§3.4) and, in the labor-tax case, the labor-tax increment chosen to satisfy (17).

## 8. Welfare / objective measure

No consumption-equivalent or social-welfare objective is defined in the source.¹⁰ The reported quantities are the fiscal multiplier (18), the terminal debt-stabilizing balance target (17), and the debt-to-GDP path `B_t/Y_t` (the debt-sustainability outcome).

## 9. Calibration of the host economy

Greece. The income process is estimated externally from LIS/EU-SILC; five parameters are calibrated internally by the method of simulated moments (SMM), the rest set externally.

**Externally set:**

| Symbol | Meaning | Value | Source |
| :-- | :-- | :-- | :-- |
| `J` | lifecycle length (from age 25) | 60 | demographic |
| `J_R` | retirement period (age 64) | 39 | Eurostat |
| `σ` | CRRA risk aversion | 2.0 | external |
| `φ` | inverse Frisch (`1/φ = 0.67`) | 1.5 | Chetty (2011) |
| `a̲` | borrowing constraint | 0 | institutional |
| `α` | private capital share | 0.33 | Gollin (2002) |
| `δ` | private depreciation | 0.05 | OECD |
| `A` | TFP | 1.0 | normalized (units) |
| `η_g` | public-capital elasticity | 0.05 | Baxter–King (1993) |
| `δ_g` | public depreciation | 0.0 | — |
| `K^g_0` | initial public capital | 1.0 | normalized (units) |
| `τ^l` | labor income tax | 0.10 | EC implicit |
| `τ^c` | consumption tax | 0.1818 | Eurostat implicit |
| `τ^k` | capital income tax | 0.2236 | Eurostat effective |
| `τ^β` | bequest tax | 0.0 | — |
| `ρ^{ui}` | UI replacement | 0.0945 | OAED rules |
| `b_min` | min pension floor (wage units) | 0.15 | Greek min/avg pension |
| `κ` | government health coverage | 0.662 | OECD Health |
| `λ^{find}` | job-finding rate | 0.50 | Eurostat |
| `λ̄^{sep}` | separation-rate cap | 0.10 | — |
| `B_0/Y` | initial debt / GDP | 1.64 | Eurostat 2023 |
| `G/Y` | government consumption / GDP | 0.13 | Eurostat |
| `I^g/Y` | public investment / GDP | 0.03 | Eurostat |
| `D/Y` | defense / GDP | 0.03 | SIPRI/NATO |
| `g_N` | population growth | −0.00573 | Eurostat |
| `r` | private capital return | 0.04 | 10y Greek bond |
| `r^B` | sovereign debt service | 0.021 | effective Greek yield |
| primary-balance target / `Y` | terminal closure | 0.0195 | — |
| `O/Y` | other net primary (residual) | −0.091122 | pinned at initial steady state |
| `T_{trans}` | transition horizon | 60 | — |

**Internally calibrated (SMM, 2026-06-09):** five parameters `θ = (β, ν, τ^p, ρ^{pens}, m)` minimize the weighted distance between model and data moments, `Q(θ) = [m^{data} - m^{model}(θ)]' W [m^{data} - m^{model}(θ)]`, `W` diagonal.

| Symbol | Meaning | Value | Dominant target moment |
| :-- | :-- | :-- | :-- |
| `β` | discount factor | 0.943 | `A/Y = 4.00` |
| `ν` | labor-disutility scale | 36.91 | mean hours `h̄ = 0.41` |
| `τ^p` | payroll tax | 0.198 | social contributions / `Y = 0.13` |
| `ρ^{pens}` | pension replacement | 0.166 | pensions / `Y = 0.16` |
| `m` | per-capita medical cost (share of mean income) | 0.0428 | government health / `Y = 0.054` |

**Education-specific (external, LIS):**

| | Low | Medium | High |
| :-- | :-- | :-- | :-- |
| population share | 0.234 | 0.471 | 0.295 |
| unemployment `u` | 0.165 | 0.158 | 0.101 |
| mean log productivity (dev.) | −0.153 | 0.000 | +0.259 |
| persistence `ρ_z` | 0.95 | 0.95 | 0.95 |
| innovation `σ_η` | 0.054 | 0.086 | 0.075 |
| fixed-effect `σ_α` | 0.367 | 0.259 | 0.318 |

## 10. Relationship to the source

The live problem matches `DSA-LSA model.tex` in structure — preferences (1), income (2)–(3), budgets (4)–(6), Bellman (7)–(8), production (10)–(12), NFA (14), primary deficit (15), debt (16) — and the experiments match `DSA-LSA experiments.tex` (shocks, financing, terminal condition (17), multiplier (18)). It diverges in the following respects, stated as facts:

1. **Public capital is on in the live baseline.** The source baseline disables public capital (`η_g = δ_g = K^g_0 = 0`), reinstating `η_g > 0` only for the `I^g` experiment. The live configuration runs `η_g = 0.05`, `K^g_0 = 1.0` in the baseline.
2. **Calibrated parameters differ from the source table.** `DSA-LSA calibration.tex` reports `β = 0.977`, `ν = 27.30`, `ρ^{pens} = 0.147`, `m = 0.039` (these match the pre-SMM starting values); the live calibration (2026-06-09) is `β = 0.943`, `ν = 36.91`, `ρ^{pens} = 0.166`, `m = 0.0428`. `τ^p = 0.198` agrees.
3. **The debt service rate is the sovereign rate, not the capital return.** Equation (16) uses `r^B = 0.021`; the source `eq:debt` is written with `r_t`. With `r = 0.04 ≠ r^B`, the two differ.
4. **Defense and a residual closure line enter the budget.** Source `eq:primary-deficit` omits defense; the live primary deficit (15) includes both defense `D_t` and an exogenous other-net residual `O_t`, the latter pinned at the initial steady state so the initial-point budget matches the primary-balance target.
5. **Medical cost is age-dependent.** The source treats `m` as a fixed per-capita cost; the live model scales it by a deterministic age profile `m_j`.
6. **The labor-supply first-order condition (9) is not transcribed in the source;** it is implied by Bellman (7) and stated here from the optimality condition the code solves.
7. **The explicit goods resource constraint is not in the compiled source;** (14) is, and the resource identity in §4 footnote 7 follows from it.
8. **Unemployment benefits are taxed and the payroll tax is deductible from the labor-tax base.** Source `eq:bc-work` writes after-tax labor income as `(1-τ^l-τ^p)y^L + T^{UI}` (UI untaxed, both taxes on the same base `y^L`); the code taxes `T^{UI}` at `τ^l` and excludes `τ^p y^L` from the `τ^l` base, as in (4).
9. **A permanent productivity fixed effect `e^{α_i}` scales wage, UI, and pension** in (3), (4), and (6); the source income and pension equations omit `α_i`. The code carries it through the wage base and the marginal wage in (9).
10. **A lump-sum bequest transfer `β_t^{lump}` enters the household budgets** (4)–(5); source `eq:bc-work`/`eq:bc-ret` have no such term, though the source describes the redistribution (§3.4).
11. **The live baseline is the Greek calibration, not the illustrative baseline in `DSA-LSA experiments.tex`.** That file states `G^{base} = 0.30·Ȳ`, `B_0 = 0`, `T_{trans} = 40`; the live calibration runs `G^{base} = 0.13·Y_t`, `B_0 = 1.64·Y_0`, `T_{trans} = 60` (matching `DSA-LSA calibration.tex`).
12. **The labor-tax closure targets a stock debt ratio, not the flow condition (17).** The source `eq:balance-condition` is the flow target `PD/Y = (g-r)b^*`; the live experiment script sets `Δτ^l` so the counterfactual's terminal `B/Y` equals the baseline transition's terminal `B/Y`. The flow target is implemented in the code but not the branch the script selects.
13. **The live pension blend weight is `λ ≈ 0.443`, not `λ = 1`.** The source pension base (6) carries a general blend `λ`; the live model derives `λ` from a career-average approximation rather than setting last-state-only.

---

¹ Source treats retirement as exogenous; an endogenous retirement window exists in the non-compiled draft and is off in the live model.
² Cohort-specific survival is built from period life tables along each cohort's calendar diagonal; population weights remain births-only because survival is already embedded in per-cohort means.
³ [CITATION: Tauchen (1986), AR(1) discretization]. The underlying cross-sectional variance profile `Var(u_j) = σ_α² + σ_η²(1-ρ^{2j})/(1-ρ²)` is fit by weighted NLLS on LIS/EU-SILC Greece (GR03–GR21), with `ρ` fixed at 0.95 (weakly identified cross-sectionally).
⁴ See §10 item 5.
⁵ `λ = 1` would recover the last-employed-state base; the live derived `λ ≈ 0.443` blends it with career-average productivity `z̄` (mean employed productivity).
⁶ See §10 item 6.
⁷ See §10 item 7; the explicit goods identity appears only in the non-compiled draft.
⁸ See §10 item 4.
⁹ See §10 item 3.
¹⁰ The SMM criterion `Q(θ)` is an estimation objective, not a welfare measure.
