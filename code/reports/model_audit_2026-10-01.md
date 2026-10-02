# Object and method

**Object.** The code under `code/` at commit `3784ae5` (2026-10-01 22:32); the working tree is `dce6352`, which differs from it only by the audit plan file. Configuration `calibration_input_GR.json` (modified 22:25), the data files `data/demography_GR.npz`, `data/survival_GR.npz`, `data/europop2023_GR.npz`, and `data/pension_floor_GR.json`. The baseline is the demographic transition from the measured 2023 cross-section with no policy change. The fiscal layer is in scope as equations; no G or $`I_g`$ experiment result exists under trend growth and none is audited.

**Method.** Six context-free agents read the source, the configuration and the data files in parallel (22:53–23:15), each on one block of questions: household problem; initial condition and cohorts; aggregation, transition and terminal state; calibration procedure; status of the six items of `code/docs/OPEN_ISSUES_2026-07-30.md`; dead code. They were forbidden the draft, the project notes, the earlier code report, the model sections of the README, the assistant’s memory and the git history; `code/CLAUDE.md` was moved out of the tree before launch and restored after the last agent returned (`git status` clean apart from the pre-existing `docs` submodule pointer). Every Critical and Major claim below was re-read in the source by the author of this report; the dead-code searches for the unreachable items were repeated. Local numerical checks ran on an 8-core Apple CPU (JAX 0.8.0, float64); the GPU runs of the plan (C1, C1b, C2) were not run, since no instance was available in the session; a reduced-scale substitute for C1 ran overnight on the CPU. Appendix C lists every check with its outcome.

**Reading guide.** Part I states the model as the code implements it, in the notation of the draft’s model section where the objects coincide, without parameter values and without solution details. Part II states the solution method, the calibration and the provenance of every recorded number. Part III ranks the findings, reports the status of the six July items, summarises the dead-code inventory, lists what could not be verified, records the comments received on the first draft, and lists the changes made after the audit. The appendices give the code-to-equation correspondence, the dead-code inventory, and the checks.

# The model economy as implemented

Time $`t = 0, 1, \dots`$ indexes periods, period $`t`$ being calendar year $`2023 + t`$, over a horizon $`T_{tr}`$. Ages are $`j = 1, \dots, J`$ with real age $`24 + j`$; working ages $`\mathcal{W} = \{1, \dots, J_R\}`$ and retired ages $`\mathcal{R} = \{J_R + 1, \dots, J\}`$. Education $`e \in \{\text{low}, \text{medium}, \text{high}\}`$ with fixed population shares $`s_e`$. Labour productivity grows at the exogenous rate $`g`$; every quantity below is detrended by $`Z_t = (1+g)^t`$, and aggregates are per capita, meaning per person aged $`25`$ to $`24 + J`$ alive in the period. The hats of the draft are dropped.

## Demographics

A cohort enters at age 1. The cohort entering in year $`y`$ has size $`N^0_y`$ and faces at age $`j`$ the survival probability $`\pi_j(y)`$ read from the period life table of the calendar year in which it reaches that age: the observed tables through 2023, the EUROPOP2023 baseline mortality for 2024–2100, and the 2100 table for every later year. Cumulative survival to age $`j`$ is $`S_j(y) = \prod_{i<j} \pi_i(y)`$. The sizes of the entering cohorts are set as follows. For years up to 2023 they are recovered from the measured 2023 population: the number of people of each age in 2023 divided by their cumulative survival, so that the model’s 2023 population equals the measured one age by age. For 2024–2100 the entering cohort is the number that makes the model’s population aged $`25`$ to $`24 + J`$ equal to the EUROPOP2023 projection, which books net migration at the entry age. After 2100 the growth rate of the entering cohort is brought linearly to zero between 2100 and 2120 and held at zero afterwards. The age distribution of the population is then constant from 2179, one lifetime later.

The population alive in period $`t`$ is $`N_t = \sum_j N^0_{t-j+1} S_j(t-j+1)`$, its growth rate $`n_t = N_{t+1}/N_t - 1`$, and the growth factor of every detrended per-capita stock is
``` math
\Gamma_t = (1+g)(1+n_t), \qquad \Gamma_t = \Gamma_T \equiv (1+g)(1+n_\infty) \ \text{for } t \ge 156 \ (\text{year } 2179).
```

A per-capita aggregate is the population-share-weighted sum of cohort averages: with $`\tilde{x}_{e,j}(t)`$ the average of a variable over the surviving members of the cohort aged $`j`$ in period $`t`$ and education $`e`$,
``` math
X_t = \sum_e s_e \sum_{j=1}^{J} \frac{N^0_{t-j+1}\, S_j(t-j+1)}{N_t}\; \tilde{x}_{e,j}(t).
```
Because a cohort’s share of the population one period later is its current share divided by $`(1+n_t)`$, the household’s detrending by $`(1+g)`$ and the stocks’ by $`\Gamma_t`$ are consistent.

## Assets and returns

Households hold one asset $`a \ge 0`$ that earns the world return $`r`$ whatever it finances, taxed at $`\tau^k`$ on the interest. Domestic private capital $`K_t`$ is chosen by firms (§<a href="#sec:tech" data-reference-type="ref" data-reference="sec:tech">1.5</a>); sovereign debt $`B_t`$ is held abroad and serviced at the sovereign rate $`r_B`$. The household sector’s foreign assets are $`A_t - K_t`$, and the economy’s net foreign asset position nets the government’s external debt out of them,
``` math
\mathrm{NFA}_t = A_t - K_t - B_t, \qquad A_t = \text{per-capita household wealth.}
```
There is no portfolio choice and no risk premium; $`r`$ and $`r_B`$ are two exogenous prices. The assets of households who die are accidental bequests. The model has a bequest tax $`\tau^{beq}`$ and a lump-sum transfer of the remainder to the entering cohort; at the audited commit both are switched off, so accidental bequests leave the economy (finding M5). Since 2026-10-02 the configuration taxes them away in full, $`\tau^{beq} = 1`$ (§<a href="#sec:postaudit" data-reference-type="ref" data-reference="sec:postaudit">3.8</a>).

## Prices and policy

The interest rate $`r_t`$ is exogenous and the wage $`w_t`$ is the firm’s marginal product of labour given the public capital stock (§<a href="#sec:tech" data-reference-type="ref" data-reference="sec:tech">1.5</a>). The policy instruments are:

- four proportional tax rates: $`\tau^c`$ on consumption, $`\tau^l`$ on labour income net of social contributions, on unemployment benefits and on pensions, $`\tau^p`$ (social contributions) on wage income only, and $`\tau^k`$ on interest income;

- a pay-as-you-go pension paid from the general budget: a replacement rate $`\rho`$ applied to a pension base built from the retiree’s last income state and the career average (equation <a href="#eq:pens" data-reference-type="eqref" data-reference="eq:pens">[eq:pens]</a>), subject to a minimum pension $`b_{\min}`$; the benefit is constant in detrended units, so in levels it grows with productivity;

- unemployment insurance replacing a fraction $`\rho^{ui}`$ of the earnings the household’s previous income state would have delivered, for the first period of a spell;

- public coverage of a fraction $`\kappa`$ of each person’s age-dependent medical expenditure $`m(j)`$, the rest paid out of pocket;

- government consumption $`G_t`$, defence spending $`D_t`$ and other net spending $`O_t`$, each a fixed share of output; $`O_t`$ is the balancing item, the net of expenditure and revenue items the model does not represent, set once so that the base-year primary balance equals the data and held as a share of output thereafter;

- public investment $`I^g_t`$, set in the baseline at the level that holds detrended per-capita public capital constant;

- sovereign debt, which absorbs the primary balance at rate $`r_B`$. In the tax-financed experiments the labour income tax rate is raised by a constant amount from $`t = 0`$, chosen so that the economy reaches a target debt ratio, or a target net-foreign-asset ratio, at the end of the horizon. No other instrument moves.

## The household problem

**State and shocks.** $`(e, \alpha, j, z, z_{last}, a)`$. The permanent productivity effect $`\alpha \sim N(0, \sigma^{2}_{\alpha,e})`$ is drawn at entry. The income state $`z \in \{0, z_1, \dots, z_{n-1}\}`$: $`z = 0`$ is unemployment; the employed states follow an autoregressive process in logs with persistence $`\rho_z`$ and an education-specific mean $`\mu_e`$ and innovation variance $`\sigma^2_{\varepsilon,e}`$. From unemployment a job is found with probability $`f`$, the new state being drawn with equal probability from the employed states; from employment a separation occurs with probability $`s_e`$, set so that the stationary unemployment rate equals the education-specific rate $`u_e`$. $`z_{last}`$ is the previous period’s state, whatever it was, and is frozen at retirement. Health has one state; medical expenditure $`m(j) = m\,\tilde{m}_j`$ is an age profile scaled by a constant. At entry $`a = 0`$, $`z`$ is drawn from the stationary distribution of the income process, and $`z_{last} = z`$.

**Preferences.** $`u(c, \ell) = \log c - \nu\,\ell^{1+\varphi}/(1+\varphi)`$. Hours $`\ell \in [0, 1]`$ are chosen by the employed; the unemployed and the retired supply $`\ell = 0`$. Discounting is $`\beta\,\pi_j`$; death leaves no bequest motive. With log consumption the detrended problem needs no growth adjustment of $`\beta`$.

**Budget constraints**, in detrended units, with $`\kappa_j`$ the age-productivity profile and $`y^L = w\,\kappa_j\,z\,e^{\alpha}`$ the wage per unit of hours:
``` math
\begin{aligned}
\text{working:}\quad & (1+\tau^c)\,c + (1+g)\,a' = \bigl(1 + (1-\tau^k) r\bigr) a + (1-\tau^l)\bigl[(1-\tau^p)\, y^L \ell + T^{UI}\bigr] - (1-\kappa)\, m(j), \label{eq:bcw}\\
\text{retired:}\quad & (1+\tau^c)\,c + (1+g)\,a' = \bigl(1 + (1-\tau^k) r\bigr) a + (1-\tau^l)\,\mathrm{PENS} - (1-\kappa)\, m(j), \label{eq:bcr}\\
& T^{UI} = \mathbf{1}\{z = 0\}\,\rho^{ui}\, w\, \kappa_j\, z_{last}\, e^{\alpha}, \qquad
\mathrm{PENS} = \max\Bigl\{\rho\, w_{J_R}\bigl[\lambda\,\kappa_{J_R} z_{last} + (1-\lambda)\,\bar{\kappa}\,\bar{z}_e\bigr] e^{\alpha},\; b_{\min}\Bigr\}, \label{eq:pens}
\end{aligned}
```
with $`a' \ge 0`$, $`w_{J_R}`$ the cohort’s wage in its last working year, $`\bar{\kappa}`$ the mean of the age profile over working ages and $`\bar{z}_e`$ the mean of the employed income states. The weight $`\lambda = (1 - \hat\rho^{J_R})/(J_R(1-\hat\rho))`$ is the coefficient of a career average on its terminal value for an autoregressive process with persistence $`\hat\rho`$ over $`J_R`$ periods; $`\hat\rho`$ is a separate constant, not the $`\rho_z`$ of the income process, and the two are equal in the current calibration. Because $`z_{last}`$ is the previous state including $`z = 0`$, UI is paid for one period per spell, an entrant drawn unemployed receives none, and a household unemployed at $`J_R`$ retires on the $`(1-\lambda)`$ term alone. There is no means-tested consumption floor and no bequest is received.

**Labour supply.** For an employed household and each candidate $`a'`$, hours satisfy
``` math
\nu\,\ell^{\varphi}\,(1+\tau^c) = \frac{1}{c}\,\mathrm{MW}, \qquad \mathrm{MW} = w\,\kappa_j\, z\, e^{\alpha}\,(1-\tau^p)(1-\tau^l),
```
with $`c`$ the consumption implied by <a href="#eq:bcw" data-reference-type="eqref" data-reference="eq:bcw">[eq:bcw]</a> at those hours and that $`a'`$: the marginal disutility of an hour equals the marginal utility of the after-tax wage it buys. The solution is restricted to $`[0, 1]`$ (§<a href="#sec:solution" data-reference-type="ref" data-reference="sec:solution">2.1</a>).

**Bellman equations.** Working age:
``` math
V_j(z, z_{last}, a) = \max_{a', \ell}\; u(c, \ell) + \beta\,\pi_j\, \sum_{z'} P_z(z' \mid z)\, V_{j+1}(z', z, a'), \qquad \text{s.t. } \eqref{eq:bcw}.
```
Retired age, as implemented in both backends:
``` math
V^R_j(z_{last}, a) = \max_{a'}\; u(c, 0) + \beta\,\pi_j\, V^R_{j+1}(0, a'), \qquad \text{s.t. } \eqref{eq:bcr},
```
that is, the continuation value is evaluated at $`z_{last} = 0`$ for every retiree, while the pension in <a href="#eq:bcr" data-reference-type="eqref" data-reference="eq:bcr">[eq:bcr]</a> and in the simulation depends on the retiree’s own $`z_{last}`$ (finding C1). At the terminal age $`a' = 0`$. A household whose resources are non-positive, which happens at $`a = 0`$ when unemployed with $`z_{last} = 0`$, since it has no benefit and owes its out-of-pocket medical cost, is handled by a default rule described in §<a href="#sec:solution" data-reference-type="ref" data-reference="sec:solution">2.1</a> (finding M1).

**Death.** After consumption and taxes, the household dies with probability $`1 - \pi_j`$; its beginning-of-period assets are recorded as an accidental bequest.

## Technology

``` math
Y_t = A\,(K^g_t)^{\eta_g}\, K_t^{\alpha}\, L_t^{1-\alpha}, \qquad
L_t = \sum_e s_e \sum_{j \in \mathcal{W}} \frac{N^0_{t-j+1} S_j(t-j+1)}{N_t}\; \widetilde{\bigl(\kappa_j\, z\, e^{\alpha}\, \ell\bigr)}_{e,j}(t),
```
labour input being the per-capita sum of hours in efficiency units. In the implementation $`L_t`$ is obtained by dividing wage income plus unemployment benefits by the wage, which equals the expression above plus $`\mathrm{UI}_t/w_t`$ (finding M2). Given $`r`$, the firm’s conditions pin the capital–labour ratio, the wage and domestic capital,
``` math
\frac{K_t}{L_t} = \left(\frac{r + \delta}{\alpha A (K^g_t)^{\eta_g}}\right)^{1/(\alpha - 1)}, \qquad
w_t = (1-\alpha)\, A\,(K^g_t)^{\eta_g} \left(\frac{K_t}{L_t}\right)^{\alpha}, \qquad K_t = \frac{K_t}{L_t}\, L_t,
```
so $`K_t/Y_t = \alpha/(r+\delta)`$ in every period. Public capital follows
``` math
K^g_{t+1} = \frac{(1-\delta_g)\, K^g_t + I^g_t}{\Gamma_t},
```
and the baseline investment level $`I^g_t = (\delta_g + \Gamma_t - 1)K^g_0`$ keeps it constant.

## Government budget constraints

With $`\Sigma`$ the per-capita aggregation of §1.1,
``` math
\begin{aligned}
\mathrm{Rev}^c_t &= \tau^c\, C_t, &
\mathrm{Rev}^l_t &= \tau^l\bigl[(1-\tau^p)\mathcal{B}^{lab}_t + \mathrm{UI}_t + \mathrm{PENS}^{out}_t\bigr], &
\mathrm{Rev}^p_t &= \tau^p\,\mathcal{B}^{lab}_t, &
\mathrm{Rev}^k_t &= \tau^k\, r\, A_t,\\
\mathrm{UI}_t &= \Sigma\, T^{UI}, &
\mathrm{PENS}^{out}_t &= \Sigma\, \mathrm{PENS}, &
\mathrm{HSub}_t &= \kappa\,\Sigma\, m(j), &
\mathcal{B}^{lab}_t &= \Sigma\, y^L \ell,
\end{aligned}
```
``` math
\begin{aligned}
\mathrm{PD}_t &= \mathrm{UI}_t + \mathrm{PENS}^{out}_t + \mathrm{HSub}_t + G_t + I^g_t + D_t + O_t - \bigl(\mathrm{Rev}^c_t + \mathrm{Rev}^l_t + \mathrm{Rev}^p_t + \mathrm{Rev}^k_t\bigr), \label{eq:pd}\\
B_{t+1} &= \frac{(1+r_B)\, B_t + \mathrm{PD}_t}{\Gamma_t}, \qquad B_0 = \bar{b}\; Y_0, \label{eq:debt}
\end{aligned}
```
with $`\bar{b}`$ the base-year debt ratio (in the implementation $`Y_0`$ is taken from a separate low-precision run, finding M9). Interest $`r_B B_t`$ is reported alongside the primary balance. A pension fund $`S_{t+1} = [(1+r_B) S_t + \mathrm{Rev}^p_t - \mathrm{PENS}^{out}_t]/\Gamma_t`$, $`S_0 = 0`$, is a memorandum item that feeds nothing. The net foreign asset position evolves as $`\mathrm{CA}_t = \Gamma_t \mathrm{NFA}_{t+1} - \mathrm{NFA}_t`$ with $`\mathrm{NFA}_t = A_t - K_t - B_t`$.

**Financing rules and terminal conditions.** Debt financing leaves $`\tau^l`$ unchanged and lets $`B_t`$ follow <a href="#eq:debt" data-reference-type="eqref" data-reference="eq:debt">[eq:debt]</a>. Tax financing adds a constant $`\Delta\tau^l`$ from $`t = 0`$, found by a root finder, so that either $`B_{T-1}/Y_{T-1}`$ equals a target debt ratio or $`\mathrm{NFA}_{T-1}/Y_{T-1}`$ equals a target net-foreign-asset ratio at the end of the horizon; the targets are the baseline’s own terminal ratios. After the horizon the simulation continues for twenty periods with every path held at its terminal value (finding C2). A rest-point condition $`\mathrm{PD}_t/Y_t = (\Gamma_T - 1 - r_B)\, b`$ exists in the code but is not used. A terminal-convergence check compares the last two periods of $`K`$, $`K^g`$, $`L`$, $`Y`$, $`C`$, $`A`$, NFA and $`S`$; $`B`$ is not among them (finding M4).

## Equilibrium and the detrended balanced-growth path

Each cohort solves its lifecycle problem given the sequences of prices, taxes and transfers it faces over its own lifetime, under the mortality of its own birth year. Cohorts already alive in 2023 are assumed to have lived their earlier years at the 2023 prices and policies, starting from zero assets at entry; their wealth in 2023 is therefore the wealth that a lifetime at those prices generates, not a measured distribution. Cohorts entering after the horizon face prices and policies held at their terminal values. Aggregates are the population-weighted sums of §1.1; the wage is set by the firm, domestic capital by the capital–labour ratio, and the net foreign asset position absorbs the difference between household wealth and domestic capital plus sovereign debt. In the baseline, aggregate changes come from the changing age composition of the population and the sizes of the entering cohorts; cohort-level changes come from each cohort’s own survival schedule. With tax rates fixed, the age distribution enters no household’s problem.

The terminal state is a balanced growth path: from 2179 the age distribution of the population, the growth factor $`\Gamma_T`$ and every cohort’s survival schedule are constant, so the detrended per-capita aggregates are constant; levels grow at $`\Gamma_T - 1`$ and per-capita quantities at $`g`$. On that path the debt recursion <a href="#eq:debt" data-reference-type="eqref" data-reference="eq:debt">[eq:debt]</a> has root $`(1+r_B)/\Gamma_T`$: when $`r_B > \Gamma_T - 1`$ a constant debt ratio requires a primary surplus of $`(r_B - (\Gamma_T - 1))\,b`$ per period, and nothing in the baseline imposes it (finding M4).

The economy’s resource constraint, per capita and detrended, with $`M_t = \Sigma\, m(j)`$ total medical spending and $`\mathrm{Beq}_t`$ the accidental bequests of those who die in $`t`$ (which leave the economy in the baseline), is
``` math
\Gamma_t\mathrm{NFA}_{t+1} - \mathrm{NFA}_t = Y_t + r\,\mathrm{NFA}_t + (r - r_B) B_t - C_t - \bigl[\Gamma_t K_{t+1} - (1-\delta)K_t\bigr] - G_t - I^g_t - D_t - O_t - M_t - \mathrm{Beq}_t ,
```
where $`r\,\mathrm{NFA}_t + (r - r_B)B_t`$ is net factor income from abroad: households earn $`r`$ on their foreign assets $`A_t - K_t`$ while the government pays $`r_B`$ on $`B_t`$. The residual that the report generator and the results checker compute omits $`M_t`$ and $`(r - r_B)B_t`$, and the implementation’s output carries the unemployment-benefit term of finding M2 (finding M3).

# Solution method and calibration

## Solution and simulation

**Household problem.** Backward induction over age on a fixed asset grid of 100 points, $`a_i = 200\,(i/99)^{1.5}`$, eight of them below $`a = 4`$; the savings choice is a search over the grid with consumption as the residual, without interpolation. The employed income states are a four-point Tauchen discretisation of the autoregressive process two standard deviations wide; the permanent effect $`\alpha`$ is a five-point Gauss–Hermite discretisation of its normal distribution. For each candidate $`a'`$ the labour first-order condition is solved by a safeguarded Newton iteration on $`[0, 1]`$ and the solution restricted to that interval, so that where the bound binds the condition holds as an inequality (§3.7). The unemployed and the retired carry a placeholder $`\ell = 1`$ in the decision arrays with zero disutility and zero wage income, which is $`\ell = 0`$ economically; the simulated sample records $`\ell = 1`$ for the unemployed and $`0`$ for retirees, and neither value enters any aggregate or moment. A state with non-positive resources is assigned $`c = 10^{-10}`$, $`a' = 0`$ and a value with no continuation term (finding M1). A means-tested consumption floor exists as a parameter; the solver refuses a positive value, and the configuration sets zero. Two implementations exist, NumPy and JAX; the production chain uses JAX, which solves the cohorts of an education group together.

**Simulation and aggregation.** Each cohort and education group is simulated with $`n_{sim}`$ households from entry. The per-cohort average over the households the cohort had at entry, counting those who have died as zero, equals cumulative survival times the average among survivors, so weighting those averages by entering-cohort sizes and dividing by the living share gives the per-capita aggregate of §1.1. Different seeds are used per cohort and education group in the transition; in the calibration all three education groups of a cohort share one seed.

**Two constructions of the 2023 cross-section.** The *single-lifecycle* construction solves one lifecycle problem per education group under one survival schedule and reads the age-$`j`$ outcome off that same lifetime for every $`j`$. The *cohort-by-cohort* construction solves one lifecycle problem per 2023 cohort, each on its own survival schedule, and reads each cohort at its own age; it is how the transition builds its $`t = 0`$ and costs 60 solves per education group where the other costs one. The configuration now selects the cohort-by-cohort construction for the calibration, the normalisation and the closure.

**Counterfactuals.** Cohorts alive in 2023 keep their baseline decision rules for the years before 2023, so that household wealth at $`t = 0`$ is the same in every scenario; the construction passes the regression check on both implementations (Appendix C). The tax-financed experiments find $`\Delta\tau^l`$ by the Illinois method to a tolerance of $`10^{-3}`$.

## Externally set

Layout as in `reports/calibration_report.tex`. The table bodies are snapshots, taken for this audit, of the untracked files in `output/calibration_growth/` that `reports/fill_report.py` wrote: the parameter, moment, implied and age bodies and the header at 22:23 on 2026-10-01 from the current configuration; the growth body at 14:36 and the figure at 14:46 from a local baseline run whose configuration state is not recorded. Git records none of them. Every model number in them, and every fitted or set value in the configuration, comes from the 12:31 run on an A100 or from the two scripts that followed it, on the single-lifecycle construction, with five fitted parameters and a pension floor of 0.15; the code and the configuration have moved since (Part III, C3). Numbers that the live pipeline would not reproduce are flagged <span style="color: red!70!black"><span class="smallcaps">stale</span></span>.

<table>
<caption>Parameters as printed by the report generator on 2026-10-01 22:23. Row flags: <span class="math inline"><em>n</em></span> is the configuration’s terminal rate <span class="math inline"><em>n</em><sub>∞</sub> = 0</span>, whereas the realised path starts at <span class="math inline"><em>n</em><sub>0</sub> = −0.0048</span> (the header of the same run prints <span class="math inline">−0.0048</span>); <span class="math inline"><em>K</em><sub><em>g</em></sub>/<em>Y</em></span> is 0.703, the 2019 IMF stock rolled to 2023; <span class="math inline"><em>b</em><sub>min</sub> = 0.1671</span> is the value in force now, while every fitted value in the next block was obtained at 0.15 (<span style="color: red!70!black"><span class="smallcaps">stale</span></span> for the block as a whole); the five SMM rows omit the sixth parameter <span class="math inline"><em>ρ</em><sup><em>u</em><em>i</em></sup></span> that the configuration lists; <span class="math inline"><em>A</em></span> and <span class="math inline"><em>O</em>/<em>Y</em></span> were set at the 12:31 <span class="math inline"><em>θ</em></span> on the single-lifecycle construction (<span style="color: red!70!black"><span class="smallcaps">stale</span></span>).</caption>
<thead>
<tr>
<th style="text-align: left;">Symbol</th>
<th style="text-align: left;">Description</th>
<th style="text-align: right;"><span>Value</span></th>
<th style="text-align: left;">Source / identified by</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4" style="text-align: left;"><em>Externally set</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>g</em></span></td>
<td style="text-align: left;">labour productivity growth, per capita</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: left;">2024 Ageing Report</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>n</em></span></td>
<td style="text-align: left;">population growth</td>
<td style="text-align: right;">0.00</td>
<td style="text-align: left;">EUROPOP2023</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>γ</em></span></td>
<td style="text-align: left;">curvature of consumption utility</td>
<td style="text-align: right;">1.00</td>
<td style="text-align: left;">log utility</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>φ</em></span></td>
<td style="text-align: left;">inverse Frisch elasticity</td>
<td style="text-align: right;">1.50</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>α</em></span></td>
<td style="text-align: left;">private capital share</td>
<td style="text-align: right;">0.33</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>δ</em></span></td>
<td style="text-align: left;">private depreciation</td>
<td style="text-align: right;">0.05</td>
<td style="text-align: left;">standard value</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>η</em><sub><em>g</em></sub></span></td>
<td style="text-align: left;">public capital elasticity</td>
<td style="text-align: right;">0.05</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>K</em><sub><em>g</em></sub>/<em>Y</em></span></td>
<td style="text-align: left;">public capital ratio</td>
<td style="text-align: right;">0.70</td>
<td style="text-align: left;">IMF ICSD</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>δ</em><sub><em>g</em></sub></span></td>
<td style="text-align: left;">public capital depreciation</td>
<td style="text-align: right;">0.04</td>
<td style="text-align: left;"><span class="math inline"><em>I</em><sub><em>g</em></sub>/<em>K</em><sub><em>g</em></sub> − (<em>Γ</em> − 1)</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>r</em></span></td>
<td style="text-align: left;">world return on capital</td>
<td style="text-align: right;">0.04</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>r</em><sub><em>B</em></sub></span></td>
<td style="text-align: left;">sovereign rate</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: left;">implicit rate 2012–24</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>τ</em><sub><em>c</em></sub>, <em>τ</em><sub><em>l</em></sub>, <em>τ</em><sub><em>k</em></sub></span></td>
<td style="text-align: left;">consumption, labour, capital tax rates</td>
<td style="text-align: right;">0.18</td>
<td style="text-align: left;">effective rates</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>b</em><sub><em>m</em><em>i</em><em>n</em></sub></span></td>
<td style="text-align: left;">minimum pension floor</td>
<td style="text-align: right;">0.17</td>
<td style="text-align: left;">national pension, L.4387/2016</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>κ</em></span></td>
<td style="text-align: left;">public share of medical spending</td>
<td style="text-align: right;">0.66</td>
<td style="text-align: left;">Eurostat <code>hlth_sha11_hf</code></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>T</em></span>, <span class="math inline"><em>J</em><sub><em>R</em></sub></span></td>
<td style="text-align: left;">lifespan, retirement age</td>
<td style="text-align: right;">60.00</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Calibrated jointly by SMM</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>ν</em></span></td>
<td style="text-align: left;">labour disutility weight</td>
<td style="text-align: right;">13.28</td>
<td style="text-align: left;">average hours</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>β</em></span></td>
<td style="text-align: left;">discount factor</td>
<td style="text-align: right;">1.02</td>
<td style="text-align: left;"><span class="math inline"><em>A</em>/<em>Y</em></span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>τ</em><sub><em>p</em></sub></span></td>
<td style="text-align: left;">payroll tax rate</td>
<td style="text-align: right;">0.20</td>
<td style="text-align: left;">payroll revenue<span class="math inline">/<em>Y</em></span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>ρ</em><sup><em>p</em><em>e</em><em>n</em><em>s</em></sup></span></td>
<td style="text-align: left;">pension replacement rate</td>
<td style="text-align: right;">0.24</td>
<td style="text-align: left;">pensions<span class="math inline">/<em>Y</em></span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>m</em><sup><em>g</em><em>o</em><em>o</em><em>d</em></sup></span></td>
<td style="text-align: left;">medical cost scale</td>
<td style="text-align: right;">0.09</td>
<td style="text-align: left;">public health<span class="math inline">/<em>Y</em></span></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Pinned outside the SMM</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>A</em></span></td>
<td style="text-align: left;">total factor productivity</td>
<td style="text-align: right;">1.43</td>
<td style="text-align: left;">normalisation, <span class="math inline"><em>ŷ</em> = 1</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>O</em>/<em>Y</em></span></td>
<td style="text-align: left;">other net spending</td>
<td style="text-align: right;">-0.11</td>
<td style="text-align: left;">data primary balance</td>
</tr>
</tbody>
</table>

Values not in the table, from the configuration: $`J = 60`$, $`J_R = 39`$, $`T_{tr} = 180`$; $`r = 0.04`$, $`r_B = 0.019`$; $`\tau^c = 0.1818`$, $`\tau^l = 0.10`$, $`\tau^k = 0.2236`$; $`\delta_g = 0.04281`$; $`G/Y = 0.13`$, $`D/Y = 0.03`$, $`\bar b = 1.6428`$; $`s_e = (0.2343, 0.4705, 0.2952)`$; $`u_e = (0.1645, 0.158, 0.1005)`$; $`f = 0.5`$, with no recorded source in the code or configuration (§3.7); $`\rho_z = 0.95`$, $`\mu_e = (-0.1529, 0, 0.259)`$, $`\sigma_{\varepsilon,e} = (0.0538, 0.0858, 0.0750)`$, $`\sigma_{\alpha,e} = (0.367, 0.259, 0.318)`$; $`\hat\rho = 0.95`$, hence $`\lambda = 0.4434`$; $`\tau^{beq} = 0`$ by default; $`n_\infty = 0`$; $`n_{sim} = 2\,000`$ per cohort and education group in the transition and in the cohort-by-cohort calibration, $`10\,000`$ per education group in the single-lifecycle construction. The age profiles $`\kappa_j`$ and $`\tilde m_j`$ are 60-entry vectors in the configuration. Derived: $`\Gamma_0 = 1.01212`$, $`\Gamma_T = 1.017`$, $`K/Y = 3.667`$, $`w = 2.1054`$ at the current $`A`$; the old-age dependency ratio (65–84 over 25–64) is 0.359 in 2023 and 0.468 on the terminal path; the living share of the ever-entered population is 0.899 in 2023 and 0.970 on the terminal path. The configuration’s 60-entry survival vector is the 2020 period table and enters no production computation, since both constructions take cohort-specific schedules from the demography file.

## Fitted jointly by SMM

Targets, weights $`1/m^2_{data}`$, and the parameter each is labelled as identifying (a labelling in the report generator, not a constraint of the optimiser):

<div class="center">

| Target | Population | Data | Weight | Labelled parameter |
|:---|:---|---:|---:|:---|
| Average hours | employed working-age households | 0.4100 | 5.95 | $`\nu`$ |
| $`A/Y`$ | whole population | 4.0000 | 0.06 | $`\beta`$ |
| Payroll revenue$`/Y`$ | $`\tau^p \times`$ wage bill | 0.1300 | 59.17 | $`\tau^p`$ |
| Pensions$`/Y`$ | retirees | 0.1600 | 39.06 | $`\rho`$ |
| Public health$`/Y`$ | $`\kappa\, m \sum_j \omega_j \tilde m_j / Y`$ | 0.0540 | 342.94 | $`m`$ |
| UI$`/Y`$ | working-age unemployed | 0.0060 | 27777.78 | $`\rho^{ui}`$ (never fitted) |

</div>

Means are taken among the living at each age and weighted by the measured 2023 age shares and the education shares. Two targets are identities given $`Y`$ and UI$`/Y`$: payroll revenue$`/Y = \tau^p(1 - \alpha - \mathrm{UI}/Y)`$ because $`w L/Y \equiv 1 - \alpha`$ and the base excludes UI, and public health$`/Y = \kappa\, m \cdot 0.904951/Y`$ because medical spending does not depend on the solution; the fitted $`\tau^p = 0.197414`$ and $`m = 0.089724`$ reproduce these identities at the 12:31 run’s $`Y = 0.9954`$ and UI$`/Y = 0.0115`$. The objective is $`\sum_i W_i (m_i^{data} - m_i^{model})^2`$ minimised by Nelder–Mead in logit-transformed parameters (tolerance $`10^{-5}`$ in the shell scripts, 500 iterations at most, same seeds at every evaluation); $`\theta`$ is written to the configuration only on convergence.

<table>
<caption>Moments as printed on 2026-10-01 22:23, read from the 12:31 markdown report (<span style="color: red!70!black"><span class="smallcaps">stale</span></span>): fitted at <span class="math inline"><em>A</em> = 1.42370</span> before the normalisation moved <span class="math inline"><em>A</em></span> to <span class="math inline">1.42761</span>, at which the normalisation log shows <span class="math inline"><em>A</em>/<em>Y</em> = 4.0439</span> (+1.10%) and public health<span class="math inline">/<em>Y</em> = 0.0537</span>; UI<span class="math inline">/<em>Y</em></span> is filed as not targeted although the configuration targets it, and its deviation cell prints the absolute gap <span class="math inline">+0.005</span> as 0.01 (relative: +83%). Five parameters, single-lifecycle construction, <span class="math inline"><em>n</em><sub><em>s</em><em>i</em><em>m</em></sub> = 10 000</span> per education type, pension floor 0.15, 169 Nelder–Mead iterations, 286 s.</caption>
<thead>
<tr>
<th style="text-align: left;"></th>
<th style="text-align: right;"><span>Data</span></th>
<th style="text-align: right;"><span>Model</span></th>
<th style="text-align: right;"><span>% dev</span></th>
<th style="text-align: right;"><span>Weight</span></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Targeted</em></td>
</tr>
<tr>
<td style="text-align: left;">Average hours</td>
<td style="text-align: right;">0.4100</td>
<td style="text-align: right;">0.4100</td>
<td style="text-align: right;">0.00</td>
<td style="text-align: right;">5.95</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>A</em>/<em>Y</em></span></td>
<td style="text-align: right;">4.0000</td>
<td style="text-align: right;">4.0002</td>
<td style="text-align: right;">0.00</td>
<td style="text-align: right;">0.06</td>
</tr>
<tr>
<td style="text-align: left;">Payroll revenue<span class="math inline">/<em>Y</em></span></td>
<td style="text-align: right;">0.1300</td>
<td style="text-align: right;">0.1300</td>
<td style="text-align: right;">0.00</td>
<td style="text-align: right;">59.17</td>
</tr>
<tr>
<td style="text-align: left;">Pensions<span class="math inline">/<em>Y</em></span></td>
<td style="text-align: right;">0.1600</td>
<td style="text-align: right;">0.1600</td>
<td style="text-align: right;">0.00</td>
<td style="text-align: right;">39.06</td>
</tr>
<tr>
<td style="text-align: left;">Public health<span class="math inline">/<em>Y</em></span></td>
<td style="text-align: right;">0.0540</td>
<td style="text-align: right;">0.0540</td>
<td style="text-align: right;">-0.00</td>
<td style="text-align: right;">342.94</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Not targeted</em></td>
</tr>
<tr>
<td style="text-align: left;">UI<span class="math inline">/<em>Y</em></span></td>
<td style="text-align: right;">0.0060</td>
<td style="text-align: right;">0.0110</td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;"></td>
</tr>
</tbody>
</table>

**The 12:31 run and what followed.** `run_scale_loop.sh` starts the SMM from the stored $`\theta`$, runs it, then re-normalises $`A`$ so that base-year output equals one at the fitted $`\theta`$ (root finder, tolerance $`10^{-4}`$), and declares the outer loop converged when the residual *before* the update, $`Y(\theta_r, A_{r-1}) - 1`$, is below $`5\times 10^{-3}`$; on 2026-10-01 it was $`-4.60\times 10^{-3}`$ and $`A`$ moved from 1.42370 to 1.42761 after the test had passed. The pair written to the configuration is therefore $`(\theta_r, A_r)`$, which no step evaluates jointly. The balancing item was then set at that pair: household-side primary balance $`+0.1045`$ of output, discretionary spending $`0.1986`$, target $`+0.0195`$, hence $`O/Y = -0.113636`$.

**Construction used.** The 12:31 fit used the single-lifecycle construction: one lifecycle per education type on the configuration’s survival vector, 10 000 households, every age read off the same households. The configuration now selects the cohort-by-cohort construction for the SMM, the normalisation and the balancing item. The construction is not recorded in any log or in the metadata; the attribution rests on the output levels in the records depending on $`n_{sim}`$, which the cohort-by-cohort construction ignores. The cohort-by-cohort construction costs 60 solves per education group where the other costs one (Appendix C, C0).

## Set outside the SMM

$`A = 1.42761013`$ (normalisation $`Y = 1`$ at the base-year equilibrium, single-lifecycle construction, 2026-10-01 after 12:31; <span style="color: red!70!black"><span class="smallcaps">stale</span></span>) and $`O/Y = -0.113636`$ (balancing item, same chain; <span style="color: red!70!black"><span class="smallcaps">stale</span></span>). The script that sets the balancing item now recomputes $`I_g/Y`$ from the level the transition spends, $`(\delta_g + \Gamma_0 - 1)K^g/Y = 0.03862`$, but writes the value formed from the configuration’s ratio 0.03862; the two coincide to five decimals today.

|                                   |  Data | Model | % dev |
|:----------------------------------|------:|------:|------:|
| Social contributions / output     | 0.130 |     — |     — |
| Contribution base / output        | 0.570 |     — |     — |
| Contribution rate on that base    | 0.228 |     — |     — |
| Pension at retirement / last wage | 0.763 |     — |     — |

Calibrated parameters against their direct data counterparts. The 22:23 run wrote placeholders (no `--implied` flag); the A100 values of 13:41 survive only in `step0_report.log`: contribution base$`/Y`$ 0.6587, contribution rate 0.1974, pension at retirement over last wage 0.5723 against data 0.570, 0.228, 0.763. The first two model values are identities, $`1 - \alpha - \mathrm{UI}/Y`$ and $`\tau^p`$; only the replacement row is a simulated statistic.

## Demographics

|                            | Data, 2023 | Model, $`t=0`$ | Model, terminal |
|:---------------------------|-----------:|---------------:|----------------:|
| Share 25–39                |      0.229 |          0.229 |           0.257 |
| Share 40–64                |      0.507 |          0.507 |           0.424 |
| Share 65–84                |      0.264 |          0.264 |           0.319 |
| Dependency (65–84 / 25–64) |      0.359 |          0.359 |           0.468 |
| Population growth $`n`$    |          — |        -0.0048 |          0.0000 |

Age structure, 2026-10-01 22:23, from the demography file and the transition’s population weights; current, no solve involved. Model $`t = 0`$ reproduces the data by construction of the demography file (Appendix C, spot checks 1–3).

|                         | Level (%) | Per capita (%) | Detrended trend (%) |
|:------------------------|----------:|---------------:|--------------------:|
| $`Y`$                   |     1.700 |          1.700 |              -0.015 |
| $`C`$                   |     1.700 |          1.700 |               0.014 |
| $`K^{dom}`$             |     1.700 |          1.700 |              -0.015 |
| $`A`$ (wealth)          |     1.700 |          1.700 |              -0.012 |
| $`L`$                   |     1.700 |          1.700 |              -0.015 |
| $`K_g`$                 |     1.700 |          1.700 |               0.000 |
| $`B`$                   |     1.700 |          1.700 |                   — |
| Theory                  |     1.700 |          1.700 |                   0 |
| Noise floor ($`g=n=0`$) |           |                |                   — |

Implied growth rates, written 14:36 on 2026-10-01 by a local baseline run at $`n_{sim}`$ and configuration not recorded (<span style="color: red!70!black"><span class="smallcaps">stale</span></span>, provenance unknown). The Level and Per-capita columns are $`\Gamma_T - 1`$ and $`g`$ by construction; the detrended-trend column over $`t = 156, \dots, 179`$ is the flatness test. $`B`$ is absent because that run tracked no debt.

# Audit

## Summary

The code at the audited commit implements the economy of Part I. Two defects in the equations as coded are Critical: the retired household’s continuation value is evaluated at the wrong pension in both implementations (C1), and the configuration-driven fiscal script cannot complete a baseline because its 200-period horizon overruns the entering-cohort table, which ends in 2210 (C2). The recorded parameter vector, $`A`$ and balancing item were produced under a pension floor, a parameter count and a construction of the 2023 cross-section that the live pipeline no longer has, so no number in Part II is reproducible as the code stands (C3). Among the Major items, the labour aggregate includes unemployment benefits, the accidental-bequest circuit is open, the goods-market residual that the results checker fails runs on is not the model’s resource constraint, the baseline debt ratio has no rest point under the balancing item as set and the terminal check does not look at debt, non-positive-resource states are handled by a default rule that drops the continuation value, two scripts fail on their own on the sixth parameter, and the report tables carry several internal inconsistencies. Of the six July items, five are unchanged in mechanism and one (the base-year-versus-transition level gap) has had its mechanism replaced for the production configuration; a reduced-scale run on the CPU puts the gap at $`+1.2\%`$ on the cohort-by-cohort construction. The aggregation, the demographic path, the growth isomorphism of the household problem, the stock recursions and the predetermination of initial wealth across scenarios check out at test scale; the production-scale checks that need a GPU were not run.

## Findings

Each item: the fact, its location, the consequence, and how it was verified. “Both implementations” means `lifecycle_perfect_foresight.py` (NumPy) and `lifecycle_jax.py` (JAX).

### Critical

**C1. Retired continuation value indexed at $`z_{last} = 0`$ (both implementations).** In the retired branch of the Bellman step the continuation is $`V_{j+1}(a', z = 0, h', z_{last} = 0)`$ — `lifecycle_perfect_foresight.py:993` (`self.V[t + 1, i_a_next, 0, i_h_next, 0]`) and `lifecycle_jax.py:332` (`V_next[:, 0, :, 0]`) — while the pension in the same period’s budget depends on $`z_{last}`$ with $`\lambda = 0.4434`$ (`:727--733`; JAX `:150--154`) and the simulation keeps $`z_{last}`$ fixed through retirement and pays the pension at the household’s own $`z_{last}`$ (`:1270--1271`, `:1194--1200`; JAX `:790`, `:716--720`). Every retiree with $`z_{last} > 0`$ therefore chooses $`a'`$ as if its pension base fell next period to $`(1-\lambda)\bar{\kappa}\bar{z}_e`$: at the mean employed state and $`\alpha = 0`$ the pension actually paid is 0.457/0.560/0.711 (low/medium/high) against 0.247/0.303/0.385 assumed for next period, a ratio of 0.54 at every education level, with the floor not binding. The working-age continuation at $`J_R`$ reads the same retired value, so pre-retirement saving is affected too. Consequence: the retiree consumption and asset profiles, aggregate household wealth $`A`$ (the $`A/Y`$ target and hence $`\beta`$), capital-income-tax revenue and the size of accidental bequests are computed from decision rules that solve a different problem from the one simulated. Agent B1’s small-model check ($`J = 14`$, correcting the index to $`z_{last}`$) changed $`a'`$ at 86% of retired states and reversed the slope of retiree consumption. Verified by reading both call sites and the budget and simulation lines; production magnitude not measured. The draft’s retired Bellman equation carries $`V^R_{j+1}(z_{last}, a')`$, so the code disagrees with the draft here.

**C2. The configuration-driven fiscal script overruns the entering-cohort table.** `run_fiscal_figures.py` asks for twenty periods beyond the horizon on every scenario (`n_post = 20`, `:218` onwards), so `run_fiscal_scenario` extends every path and runs the transition for $`T_{tr} + 20 = 200`$ periods (`fiscal_experiments.py:730--736`). The population weights need the entering cohort of year $`2023 + t`$, and `_entrant_weights` raises `ValueError` when that year exceeds the table’s last year 2210 (`olg_transition.py:947--953`; the demography file is built “to cover T_transition = 180”, `build_demography_GR.py:48`). Verified by calling the method on the production economy: $`t = 187`$ returns, $`t = 188`$ raises (“entering cohorts cover 1964..2210 but period t=188 needs 2152..2211”); `growth_factors` clips the same overrun instead. Consequence: `python run_fiscal_figures.py --config calibration_input_GR.json` cannot complete its baseline, so no fiscal number can be produced on the current configuration. No reported number is corrupted: both `fiscal_results.json` files in `output/` are 60-period runs from July, without the growth and aggregation stamps.

**C3. No recorded number is reproducible by the live pipeline.** The configuration’s $`\theta`$ (five values), $`A = 1.42761013`$ and $`O/Y = -0.113636`$ come from the 12:31 chain with `pension_min_floor = 0.15` (every one of the seven calibration reports since 29 September prints 0.15), five parameters and targets, and the single-lifecycle construction; the configuration now has the floor at 0.1671 (`build_pension_floor_GR.py`, 22:21), six parameters and targets with UI$`/Y`$ at weight 27 777.78, and `base_year_cohorts = true`; `calibrate.py` and `pin_baseline_closure.py` were edited after the run (file times 22:29 and 22:15). In addition $`\theta`$ was fitted at $`A = 1.42370`$ and sits beside $`A = 1.42761`$, at which the model’s $`A/Y`$ is 4.0439 (normalisation log) while the moments table prints 4.0002. Consequence: every model column in Part II describes a configuration that no longer exists; the direction of the change in $`\theta`$ is not readable from the code (the test file records that the cohort-by-cohort construction moved hours by +2.6% and pensions$`/Y`$ by $`-3.1`$% on the earlier production run); at the stored $`\theta`$ the CPU run at $`n_{sim} = 200`$ puts hours at $`+1.6\%`$ and $`A/Y`$ at $`+6.3\%`$ of their targets on the live configuration (Appendix C, one noisy draw). Verified against the configuration, the reports and the logs; the agents’ attribution of the construction is an inference from the $`n_{sim}`$-dependence of the recorded levels.

### Major

**M1. Non-positive resources handled by a default rule with no continuation (both implementations).** When no $`a'`$ leaves $`c > 0`$, the solver returns $`c = 10^{-10}`$, $`a' = 0`$ and $`V = u(c)`$ with no $`\beta\pi_j\mathbb{E}V_{j+1}`$ term (`lifecycle_perfect_foresight.py:1020--1024`; `lifecycle_jax.py:354--361`). The state $`a = 0`$, $`z = 0`$, $`z_{last} = 0`$ has resources $`-(1-\kappa)m(j) \in [-0.010, -0.035]`$ because UI is zero there and out-of-pocket medical spending is due. Every entrant starts at $`a = 0`$ with $`z_{last} = z`$ and $`z`$ drawn from the stationary distribution, so the share of entrants in that state at entry equals the education unemployment rate (16.5%, 15.8%, 10.1%); it recurs for any zero-asset worker in the second year of a spell (probability 0.5). Consequences: the medical outlays are recorded although the budget cannot pay them, so the household budget identity fails by the shortfall in those household-periods; $`c = 10^{-10}`$ enters the consumption aggregate and every consumption-distribution statistic; the value of hitting $`a = 0`$ while unemployed, against which neighbouring states’ precautionary saving is computed, omits the future. Verified by reading the default rule, the entry state and the budget. Share of household-periods not measured.

**M2. Labour input includes unemployment benefits (July item 1, unchanged).** `effective_y_sim = wage_income + ui_sim` (`lifecycle_perfect_foresight.py:1232`; `lifecycle_jax.py:734`) is the per-age labour mean and is divided by $`w`$ to form $`L`$ (`olg_transition.py:2173`; `calibrate.py:628, :1524`); the social-contribution base is wage income alone (`:1236`; JAX `:746`). Dividing wage income by the wage gives exactly hours in efficiency units, so the discrepancy from the model’s $`L_t`$ is the benefit term alone: $`L`$, $`K`$, $`Y`$ and $`wL`$ are overstated by UI$`/(wL) \approx 0.011/0.67 \approx 1.7\%`$ at the logged UI$`/Y`$; the same convention holds in calibration and transition, so $`\theta`$ absorbs it and the levels and the multiplier denominator carry it. Verified.

**M3. The goods-market residual is not the model’s resource constraint.** `reports/fill_report.py:302--364` and `eval_fiscal_results.py:225--262` compute $`C - (Y - I - (G/Y + D/Y + O/Y)Y - I^g - \Delta\mathrm{NFA})`$ and an “open” form that subtracts $`r\,\mathrm{NFA}`$. Against the resource constraint in §1.7, two booked flows are missing: total medical spending $`M_t`$ (household side `:768`, government side `olg_transition.py:1824`) and $`(r - r_B)B_t`$. A third term appears because of M2: the implementation’s $`Y_t`$ and $`K_t`$ contain the benefit inside $`L_t`$, while the government also pays it, so the identity on the flows as booked carries an extra $`-\mathrm{UI}_t`$. On the configuration’s own ratios the open form cannot be smaller in magnitude than $`M/Y + \mathrm{UI}/Y - (r - r_B)B/Y \approx 0.082 + 0.006 - 0.034 = 0.054`$ of output in a debt-financed run and about 0.108 in the report’s baseline, before the bequests that the docstrings credit with the whole residual (0.02–0.04, with “NFA$`/Y`$ near $`-5`$”, which the code cannot produce since $`K/Y = 3.667`$ bounds $`(A - K)/Y`$ below at $`-3.667`$). In the results checker the $`r\,\mathrm{NFA}`$ term is always zero because `params[’r’]` is never written by `run_fiscal_figures.py` nor by the checker’s configuration merge, so the checker always tests the closed form. Consequence: the one FAIL-tier identity check added at `3784ae5` fails a run on an identity the model does not satisfy; its threshold of 0.08 would trip for reasons unrelated to a normalisation error. Verified by reading both implementations; the algebra is agent B3’s derivation, re-read. The CPU run at $`n_{sim} = 200`$ (Appendix C) measures the open form at $`-0.197`$ of output at $`t = 0`$ in the report’s baseline, of which the identity’s known terms account for $`-0.162`$, leaving $`-0.035`$ for the unmeasured accidental bequests.

**M4. No rest point for debt; the terminal check ignores $`B`$; the $`\tau^l`$ target is the baseline’s drifted ratio.** With $`r_B = 0.019 > \Gamma_T - 1 = 0.017`$ the recursion <a href="#eq:debt" data-reference-type="eqref" data-reference="eq:debt">[eq:debt]</a> has root 1.00197 on the terminal path and a stationary $`B/Y`$ needs a primary surplus of 0.197% of output per unit of $`B/Y`$; during the transition $`\Gamma_t - 1 - r_B`$ runs from $`-0.0069`$ to $`-0.002`$. `_check_terminal_convergence` tracks $`K`$, $`K^g`$, $`L`$, $`Y`$, $`C`$, $`A`$, NFA and $`S`$ and not $`B`$ (`fiscal_experiments.py:411--442`), so “terminal converged” is compatible with a drifting debt ratio; `run_fiscal_figures.py:326--328` sets the tax-financed target to the baseline’s own terminal $`B/Y`$. Consequence: the reported $`\Delta\tau^l`$ is defined relative to wherever the baseline’s debt ratio has drifted to at $`t = 180`$ and depends on the horizon. The last logged baseline (13:41, on the pre-edit code) shows pensions$`/Y`$ rising from 0.157 to 0.289 at $`t = 25`$ and settling at 0.199, revenue from 0.328 to 0.357; adding the fixed ratios gives a terminal primary deficit of about 0.3% of output (back-of-envelope). The CPU run at $`n_{sim} = 200`$ gives a terminal full primary deficit of about 0.06% of output (Appendix C). Verified by reading; the baseline $`B/Y`$ path on the current code does not exist.

**M5. The bequest circuit is open (July item 2, unchanged).** `recompute_bequests` defaults to `False` (`fiscal_experiments.py:130`) and none of the seven scenarios in `run_fiscal_figures.py` sets it; the redistribution loop runs only when it is set (`olg_transition.py:2064--2067`); the transfer to the entering cohort is zero without it (`:1181--1185`); `bequest_transfers` is in the budget dictionary but in neither revenue nor spending (`:1784`, `:1824--1825`); $`\tau^{beq} = 0`$; the calibration never sets a transfer. The wealth of the dead (about 3.8% of output per period in the July runs; not re-measured) leaves the economy, and $`\beta`$ was fitted with that outflow. The recorded bequest is the dying household’s beginning-of-period $`a`$ rather than the $`(1+g)a'`$ it carried out of the period (`lifecycle_perfect_foresight.py:1262`; JAX `:802`), and when the loop is on the transfer goes to the entering cohort of the same period with no $`\Gamma_t`$ scaling and never to cohorts born before $`t = 0`$. The draft states that bequests are taxed and redistributed. Verified. Closed on 2026-10-02 by taxing bequests away in full and correcting the measure (§<a href="#sec:postaudit" data-reference-type="ref" data-reference="sec:postaudit">3.8</a>).

**M6. Two scripts fail on their own on the sixth parameter.** `normalize_A_tfp.py:93` and `pin_baseline_closure.py:60` build $`\theta`$ as `[theta_dict[p.name] for p in spec.params]`; the stored $`\theta`$ lacks `ui_replacement_rate`, so both raise `KeyError` before any solve. `theta_from_config` exists for this case and is used by `fill_report.py`, `diag_ss_vs_transition.py` and `diag_bequest_decomp.py`. Inside `run_scale_loop.sh` the failure does not occur, because `calibrate.py` rewrites $`\theta`$ with all six keys on convergence before the two scripts run; re-setting the balancing item on its own, which is what the floor change calls for, fails. Verified by reading both lines and the configuration.

**M7. The scale loop never evaluates the written $`(\theta, A)`$ pair.** Described in §2.3: the convergence test uses the residual at the pre-update $`A`$, after which $`A`$ is still rewritten; on 2026-10-01 the pass margin was $`0.4\times 10^{-3}`$ and the moments at the written pair deviate by up to 1.1% ($`A/Y`$). Verified in `run_scale_loop.sh:55--61` and the log.

**M8. The report tables are internally inconsistent.** (a) The implied-statistics rows “contribution base$`/Y`$” and “contribution rate” are the identities $`1 - \alpha - \mathrm{UI}/Y`$ and $`\tau^p`$ (`fill_report.py:261--273`), presented against data 0.570 and 0.228. (b) UI$`/Y`$ is filed under “Not targeted” (`UNTARGETED = [’ui_over_Y’]`, `:121`) while the configuration targets it, and on the markdown path its “% dev” cell prints the absolute gap. (c) The parameter table prints $`n = 0.00`$ from `external_params.pop_growth` (`:60`) and the header of the same run prints $`n = -0.0048`$ (`:533`); $`\Gamma - 1 = 0.01700`$ in the header and the growth table’s Level column use $`n = 0`$, which equals $`\Gamma_T - 1`$ only because the configuration’s terminal rate coincides with the demography file’s $`n_\infty`$. (d) $`b_{\min} = 0.17`$ is printed beside $`\theta`$ fitted at 0.15, and five SMM rows beside six configured parameters. Verified in the generator and the bodies.

**M9. $`B_0`$ and the $`I_g`$ shock are sized off a 50-household preliminary run (July item 4, unchanged).** `run_fiscal_figures.py:102--106, :123` and, for the shock, `:266--267`; the full runs use 2 000 households per cohort. $`B_0/Y_0`$ differs from 1.6428 by the preliminary run’s sampling error, which propagates through the whole $`B/Y`$ path and the tax-financed target. Verified.

### Minor

1.  **Dating of the tax-financed target.** `run_fiscal_figures.py:326` sets the target as $`B_{T_{bal}}/Y_{T_{bal}-1}`$ with a comment saying it matches the residual’s convention; `_balance_residual` now evaluates $`B_{T_{bal}-1}/Y_{T_{bal}-1}`$ (`fiscal_experiments.py:335--337`, changed 2026-10-01). The counterfactual is required to reach at $`t = 179`$ the ratio the baseline reaches at $`t = 180`$; bias in $`\Delta\tau^l`$ of order 0.01 pp. Verified.

2.  **G multiplier when G is a share of output.** $`G_t = 0.13\,Y_t`$ in the baseline and $`0.15\,Y^{cf}_t`$ in the shock, so $`\Delta G_t = 0.02 Y^{cf}_t + 0.13\,\Delta Y_t`$ and the reported multiplier is $`m/(1 + 0.13 m)`$ for a true impact $`m`$; the $`I_g`$ shock is a level. Read, not re-derived.

3.  **The balancing-item script writes the configuration-ratio value.** `pin_baseline_closure.py:76--95` recomputes $`I_g/Y`$ from the level and prints it, then writes `ratios[’closure_other_over_Y’]`, formed from `fiscal.I_g_over_Y` (`:104`); identical to five decimals today. Verified.

4.  **$`z_{last}`$ convention (July item 3, unchanged).** UI lasts one period per spell, entrants drawn unemployed get none, and a worker unemployed at $`J_R`$ retires on the $`(1-\lambda)`$ term; four sites, both implementations. The draft now describes the one-period UI; the retirement consequence is not in the draft.

5.  **Silent defaults (July item 6, unchanged).** $`\lambda`$ from a constant $`\hat\rho = 0.95`$ (`calibrate.py:1342`), not from `edu_params.rho_y`; $`\tau^{beq}`$ from the dataclass default; neither key in the configuration.

6.  **Pre-2023 wage path is `None` in every fiscal run.** The baseline paths that pre-2023 cohorts keep in a counterfactual are assembled before `base_paths[’w_path’]` is set and on a local copy (`fiscal_experiments.py:1226, :1244--1248, :1278`); harmless because the baseline decision rules are stored before any counterfactual runs and $`w_0`$ is the same in every scenario, and each scenario call re-runs the baseline (cost only).

7.  **Pension fund at $`r_B`$.** The memorandum recursion accrues at $`r_B`$ (`olg_transition.py:2305--2317`); the draft and the plan write $`r_t`$. Feeds nothing.

8.  **Implementation and seed details.** The JAX single-model `simulate` ignores `T_sim`, so the cohort-by-cohort construction simulates every cohort over the full horizon on JAX and reads row $`j`$ (no numerical effect); the SMM seeds cohort $`j`$ with $`42 + j`$ for all three education types (common random numbers across education); the JAX simulation of cohorts solved together and the single-model JAX simulation split keys differently for $`n_\alpha > 1`$; the single-model JAX solve drops `bequest_lumpsum` while the kernel for cohorts solved together carries it (inert at zero); dead households’ $`\ell`$ is 1 in NumPy and 0 in JAX (not aggregated).

9.  **Records.** The report prints $`n_{sim} = 10\,000`$ where the cohort-by-cohort construction uses 2 000 per cohort; `theta_metadata.source_report` names a directory that holds only July reports; the normalisation and balancing-item docstrings say “one lifecycle problem”; `fill_report.py:119` says the interest “data” is 0.0312 while the configuration holds 0.0339; the configuration’s survival vector is the 2020 table; the docstrings’ “0.47 spread in the probability of reaching 84” is 0.25 on the demography file (0.505–0.759).

10. **Test fixtures with a positive consumption floor.** The constructor now refuses `transfer_floor > 0`, so the two household-isomorphism tests fail at fixture construction and three further tests that build such fixtures (`test_olg_transition.py:1160, :1410, :1436`) would too; the isomorphism itself holds exactly with the floor at zero (Appendix C).

11. **Entry points that cannot complete.** `python olg_transition.py` without flags builds $`n_h = 2`$ and `_alive_fraction` refuses it; the `lifecycle_perfect_foresight.py --test` block unpacks 19 names from a 22-tuple. Neither is a production path (Appendix B).

12. **Two live configurations diverge.** `calibration_input_GR_rB0.json` (July: $`K^g = 0.745`$, 60 periods, no demography file) is tracked, read by nothing in `code/`, and still describes the July mechanism of item 5.

13. **Hours of the unemployed in the simulated sample.** The sample records $`\ell = 1`$ for unemployed households (a placeholder from the decision arrays) and $`0`$ for retirees; the hours moment excludes the unemployed and $`L_t`$ is built from income, so the value reaches no aggregate or moment.

## The six items of the July open-issues list at HEAD

<div class="center">

| Item | Status | Evidence at HEAD |
|:--:|:---|:---|
| 1 $`L`$ includes UI | still present | M2; four sites, both implementations; only line numbers moved. |
| 2 bequest circuit open | still present | M5; seven (not six) scenarios, none sets the flag; the configuration path of the command line also defaults to off. |
| 3 $`z_{last}`$ = previous state incl. 0 | still present | m4; four sites, both implementations. |
| 4 $`B_0`$ from the preliminary run | still present | M9; the $`I_g`$ shock level shares it. |
| 5 $`Y_{ss} = 1`$ not carried to $`t = 0`$ | changed | The mechanism the note names (configuration survival vector against cohort schedules) is gone for the production configuration: both constructions now take cohort schedules from the demography file and the calibration weights the measured cross-section. The last production-scale measurement (13:41, single-lifecycle, $`n_{sim} = 2\,000`$) was $`-9.5\%`$; the CPU run on the cohort-by-cohort construction at $`n_{sim} = 200`$ gives $`+1.2\%`$ (Appendix C), one draw. The aggregation conventions, seeds and random-number generators still differ. For the rB0 configuration the item stands as written. $`K^g`$ is now 0.703, not 0.745. |
| 6 silent defaults | still present | m5; the note’s “derives $`\lambda`$ from $`\rho_z`$” overstates it: a constant 0.95 that coincides with $`\rho_z`$. |

</div>

## Code against the draft’s model section

The draft (`docs/DSA-LSA model.tex` at submodule commit `da44637`, 2026-09-21) predates the branch. Disagreements that `code/docs/TREND_GROWTH_PLAN.md` settles are booked as settled changes: CRRA utility with curvature $`\sigma`$ (code: $`\log c`$); population growing at a constant $`g_N`$ with weights $`\Lambda^{-j}`$ (code: measured 2023 cross-section and the EUROPOP path); $`a'`$ without the growth factor (code: $`(1+g)a'`$); the public-capital and debt laws without $`\Gamma`$ (code: divided by $`\Gamma_t`$); no statement on who holds $`B`$ (plan: external official-sector debt at $`r_B`$). Every other disagreement is a finding:

- The draft’s retired Bellman equation continues with $`V^R_{j+1}(z_{last}, a')`$; the code continues with $`z_{last} = 0`$ (C1).

- The draft’s $`L_t = \int \kappa_j z e^{\alpha}\ell\, d\mu`$; the code’s $`L_t`$ includes UI$`/w`$ (M2).

- The draft taxes bequests at $`\tau^{beq}`$ and redistributes them to newborns; the code records them and lets them leave (M5).

- The draft enforces “a minimum consumption floor via a lump-sum transfer when needed”; the code has no floor and refuses a positive one (m10). The default rule of M1 is the only thing that happens at the bottom.

- The draft’s pension fund accrues at $`r_t`$; the code’s at $`r_B`$ (m7).

- The draft describes UI as “zero once a spell runs beyond one period”, which matches the code; it does not state that the $`\lambda`$ term of the pension is zero for a household unemployed at $`J_R`$ (m4).

The pension formula, the four tax bases, the health line, the production function and both firm conditions, $`\mathrm{NFA} = A - K - B`$, the primary-deficit definition with $`D_t`$ and $`O_t`$, and the cohort-specific survival along calendar years agree between draft and code.

## Dead-code inventory

Agent B6 parsed the 28 Python files with `ast` (611 definitions: 324 in non-test modules, 287 collected by pytest), indexed every name and attribute reference with its file and line, kept string-only hits apart, counted the four shell scripts as callers, and confirmed each candidate by reading its call sites against the production configuration and scripts. Counts of tabulated rows (a row can cover several definitions): (a) unreachable 6; (b) test-only 11 rows, 15 definitions; (c) switched off in production 31; (d) dead inside live code 39; (e) configuration keys without production effect 13; (f) scripts no shell script calls whose outputs nothing reads 5; plus 3 observations that are not dead code. The repeated search reproduced the five unreachable definitions (`solve_labor_hours_jax`, `_solve_labor_hours`, `_compute_wage_path_njit`, `_get_cached_cohort_panel`, `compute_aggregates`: definition only, one docstring mention); the sixth item is a set of branches of `compare_scenarios` that no caller reaches. Items in (d) with a bearing on correctness rather than tidiness: a stored simulation result (`_birth_sim_cache`) written and never read under JAX; the budget keys `debt_service`, `new_borrowing`, `fiscal_deficit` identically zero or duplicate in production yet written to the results JSON; `simulation.n_sim` printed but not used on the cohort-by-cohort construction; the pension-weight constant of m5; the duplicated aggregate implementations (`_compute_ss_aggregates` against `compute_fiscal_ratios`; the regeneration script against the figure script). Whether any item is deleted is not part of this report. The full inventory, with the pattern searched and the result per item, is Appendix B; the raw 611-row table is `audit_2026-10-01/dead_code_inventory.csv`.

## Not verified

- The production-scale magnitude of C1 and M1 (requires the 60-cohort solve at the production grid; the agents’ checks were small models).

- The $`t = 0`$ output gap, the goods-market residual and the flatness column at the production $`n_{sim} = 2\,000`$: C1 of the plan was not run (no GPU in the session). A CPU run at $`n_{sim} = 200`$ measured them once (Appendix C): gap $`+1.2\%`$, open residual $`-0.197`$, trends within $`0.07\%`$ per year; not repeated across seeds.

- Predetermination of initial wealth across scenarios under the Step 0 demography at production scale (C1b): not run. The mechanism passes on the small test economy on both implementations.

- Numerical agreement of the two implementations at production scale; the construction and implementation behind the recorded $`\theta`$ (inferred, not logged); the baseline $`B/Y`$ path and terminal primary balance on the current code; the size of the accidental-bequest outflow and of the sampling error in $`Y_0`$.

- Whether the user-level skills outside `code/` call `eval_fiscal_results.py` and `regen_fiscal_figures_from_json.py` (B6 classified them as manual command-line tools).

## Comments received on the first draft (2026-10-02)

Comments that overlap a finding above were folded into it: bequests must be taxed away or transferred (M5); labour input should be the sum of hours in efficiency units, not of income (M2, where dividing wage income by the wage gives exactly that, so the benefit term is the whole discrepancy); unemployment benefits do not belong in the resource constraint (M2 and M3). Three do not overlap and are recorded here with the facts the code gives.

1.  **Should $`B`$ be counted in NFA?** $`\mathrm{NFA}_t = A_t - K_t - B_t`$ is the economy’s net foreign asset position: household foreign assets $`A_t - K_t`$ less the government’s external debt. Under the assumption that $`B`$ is held abroad it must be netted out, with a negative sign, to obtain the country’s position; the household sector’s own position is $`A_t - K_t`$. The code uses the economy-wide definition for the NFA target of the tax-financed experiments and for the current account, and the household definition nowhere; the two are consistent and the report labels which is which.

2.  **How was the job-finding probability calibrated?** $`f = 0.5`$ is a constant in the configuration (`external_params.job_finding_rate`) with no recorded source in the code, the configuration or the calibration reports; the separation rates are derived from it and the education unemployment rates. Whether $`f`$ was ever calibrated, and to what, is not recorded.

3.  **Hours should not be restricted to $`[0, 1]`$.** The solver restricts the labour first-order condition’s solution to $`[0, 1]`$ (`lifecycle_perfect_foresight.py:841--891`; `lifecycle_jax.py:56--93`), so where either bound binds the condition holds as an inequality. Whether the upper bound binds for any household-period at the production parameters was not measured; average hours among the employed are 0.41.

## Changes made after the audit (2026-10-02)

Made at the user’s request after the review of the first draft; they postdate the audited commit. The bequest change is committed as `ff76dd4`; the rest is in the working tree at the time of writing. Decisions taken by the user: a booked consumption floor for the non-positive-resource states (M1); debt tracked and reported rather than pinned (M4); no upper bound on hours (c3); the job-finding probability calibrated to unemployment duration (c2).

1.  **Accidental bequests taxed away in full** (M5). `external_params.tau_beq = 1.0`; the base-year budget in `calibrate.py` now carries the bequest line, so the balancing item and the transition see the same revenue. The recorded bequest is the wealth the dying household carried out of the period, $`(1+g)a'`$, in both implementations. Checked as reported in the first version of this section; committed as `ff76dd4`.

2.  **C1, retired continuation value.** Both solvers read $`V_{j+1}`$ at the retiree’s own $`z_{last}`$ (`lifecycle_perfect_foresight.py`, retired branch of `_solve_state_choice`; `lifecycle_jax.py`, `solve_period_jax`, where the $`y_{last}`$ axis is kept through the health expectation). Check: on a small model the retired value varies with $`z_{last}`$, is nondecreasing in it, and the two implementations agree to $`10^{-14}`$ with identical savings rules.

3.  **M2, labour input.** $`L = (\text{wage income} + \mathrm{UI} - \mathrm{UI})/w`$ in the transition, in the unused `compute_aggregates`, and in both aggregators of `calibrate.py`. Check: $`wL`$ equals the wage bill, payroll tax over its rate, to $`10^{-10}`$ in a transition with a positive payroll rate.

4.  **C2, horizon beyond the entering-cohort table.** `_entrant_weights` extrapolates the entering cohort past the table at the terminal rate $`n_\infty`$, which is exact because the table ends after the ramp; `growth_factors` already clipped. Check: period 199 returns the same weights as 187 on the production configuration.

5.  **M6 and m3, the two standalone scripts.** `normalize_A_tfp.py` and `pin_baseline_closure.py` read $`\theta`$ with `theta_from_config`; the closure script writes the value built on the level-based $`I_g/Y`$.

6.  **M1, consumption floor, booked.** `external_params.transfer_floor = 0.0807`: the Greek guaranteed minimum income for a single adult, EUR 200 a month, over output per person aged 25–84 (`build_transfer_floor_GR.py`, `data/transfer_floor_GR.json`; the amount is to be checked against the base-year vintage). The budget function returns the top-up it granted; both simulations record it per household (`transfer_sim`, the 23rd panel array), the per-age means carry it as a 12th element, the transition’s budget books it as an outlay (`transfers`) and the base-year budget nets it in the primary balance. The two refusals of a positive floor are gone; the JAX wrapper refuses a floor together with schooling, whose child costs the simulated transfer does not replicate. Check: with a floor of 0.08 on a small model the household budget identity holds to $`10^{-14}`$ on both implementations with the transfer included, and the transition’s total spending equals the sum of its lines including transfers. With the floor above the largest out-of-pocket medical cost, the non-positive-resource states of M1 no longer occur.

7.  **M4, debt in the terminal check.** `_check_terminal_convergence` takes the debt path and reports its drift at the slow tolerance; the three fiscal runners pass it. No rest-point rule is imposed, by decision.

8.  **M7, the scale loop.** `run_scale_loop.sh` converges only if, in addition to the pre-update residual, every targeted moment at the written $`(\theta, A)`$ pair is within `MOM_TOL` (default 0.5%), read from the normalisation’s own printout.

9.  **M9 and m1, the fiscal driver.** The preliminary run is at the full sample size, so $`B_0/Y_0`$ equals the configured ratio exactly and the $`I_g`$ shock is sized off the same output; the tax-financed target is dated $`B_{T_{bal}-1}/Y_{T_{bal}-1}`$ as the residual is; the results now stamp $`r`$, $`\kappa`$, $`\tau^{beq}`$ and the floor.

10. **M3, the resource-constraint checks.** The report generator’s baseline receives the spending shares and computes the budget; its residual is $`C - (Y + r\,\mathrm{NFA}_p - I - G - I^g - D - O - M - \Delta\mathrm{NFA}_p + \mathrm{PD})`$ on the budget’s own lines, with $`M`$ total medical spending, warning above 1%. The results checker’s residual is $`C - (Y + r\,\mathrm{NFA} + (r - r_B)B - I - \text{purchases} - M - \Delta\mathrm{NFA})`$, with $`r`$ now read from the results and the debt path passed in, failing above 2%. Neither has been run on a production transition yet.

11. **M8, the report tables.** The parameter table prints $`n_0`$ and $`n_\infty`$ as separate rows, every parameter the configuration lists for the SMM (a never-fitted one shows no value), the floor, $`\tau^{beq}`$ and $`f`$; UI$`/Y`$ is not filed as untargeted when it is a target, and the markdown route recomputes the relative deviation; the two identity rows of the implied table are labelled as such; the header carries $`\Gamma_0 - 1`$ and $`\Gamma_T - 1`$ and the growth table’s level column uses $`\Gamma_T - 1`$.

12. **c3, hours.** Both labour solvers bracket upward by doubling until the condition’s residual turns positive, so hours above one are admitted and the condition holds with equality at every interior solution. Check: with a small disutility weight both implementations choose hours above one, and a direct solve returns a residual of zero at $`\ell = 3.84`$.

13. **c2, job-finding probability.** `build_job_finding_GR.py` derives $`f = 1 - \text{long-term share of unemployment}`$ from Eurostat `une_ltu_a` (ages 15–74, both sexes; `data/job_finding_GR.json`); the user chose the base year: share 0.560 in 2023, $`f = 0.440`$, written to the configuration (it was 0.5, unsourced). Alternatives recorded in the file: 0.385 over 2015–2024, 0.406 over 2019–2024, 0.464 for 2024.

14. **m13, hours of the unemployed.** The simulated panel records $`\ell = 0`$ for the unemployed on both implementations; the two labour-supply tests that asserted the old placeholder were updated.

15. **m10, test fixtures.** With the floor booked the constructor accepts a positive floor again, so the household-isomorphism tests run; six regression tests for the items above were added (`TestAuditFixes20261002`).

16. **Found by the full test suite after the changes, and fixed.** The weighted Gini coefficient (`calibrate.compute_gini` with weights) summed the cumulative income share after each observation instead of the trapezoid mean of the shares before and after it, so every weighted Gini was understated (0.067 instead of 0.267 for the values 1 to 5 with equal weights; negative on skewed weights). Pre-existing: identical on the audited commit, and the suite’s own test asserted agreement only to 0.05. The wealth, earnings, income and disposable-income Gini rows of every calibration report to date are affected; none is a target. Fixed to the trapezoid rule, with an exact test against an expanded sample. A second failing test, which asserted that a higher pension floor raises retiree consumption, encoded a claim the model does not imply: with the continuation value now read at the household’s own last income state, the fixture smooths by consuming more before retirement and 0.05% less after it. The test now asserts what the theory does imply, that the value function weakly rises at every state and the floor binds.

17. **Left open.** Everything above changes what the SMM fits to, so $`\theta`$, $`A`$ and the balancing item must be re-fitted (C3); the resource-constraint checks and the fiscal pipeline have not yet been exercised on a production transition; the `retirement_window` path, the HSV schedule and the other switched-off features of Appendix B were not touched.

# Code-to-equation correspondence

Line numbers are from the files at `3784ae5`. PF = `lifecycle_perfect_foresight.py`, JX = `lifecycle_jax.py`, OT = `olg_transition.py`, FE = `fiscal_experiments.py`, CA = `calibrate.py`, RF = `run_fiscal_figures.py`, FR = `reports/fill_report.py`.

| Object | Equation (Part I) | NumPy / scripts | JAX |
|:---|:---|:---|:---|
| Utility, disutility | $`\log c`$; $`\nu \ell^{1+\varphi}/(1+\varphi)`$ | PF:629–634, 893–897 | JX:35–41, 96–98 |
| Discounting with survival | $`\beta\,\pi_j\,\mathbb{E}V_{j+1}`$ | PF:946, 1012 | JX:338–339, 345 |
| Working budget | <a href="#eq:bcw" data-reference-type="eqref" data-reference="eq:bcw">[eq:bcw]</a>; $`(1+g)a'`$ | PF:742–768, 974 | JX:163–190, 288 |
| Retired budget, pension | <a href="#eq:bcr" data-reference-type="eqref" data-reference="eq:bcr">[eq:bcr]</a>, <a href="#eq:pens" data-reference-type="eqref" data-reference="eq:pens">[eq:pens]</a> | PF:724–740, 1194–1200 | JX:140–161, 713–720 |
| UI | $`T^{UI}`$ in <a href="#eq:pens" data-reference-type="eqref" data-reference="eq:pens">[eq:pens]</a> | PF:743–747, 1215–1218 | JX:164–168, 723–727 |
| Tax bases | $`\tau^c, \tau^l, \tau^p, \tau^k`$ | PF:751, 758, 739, 761–763, 1235–1254 | JX:170, 176, 159, 183–185, 743–763 |
| Labour FOC and bounds | $`\nu\ell^{\varphi}(1+\tau^c) = \mathrm{MW}/c`$ on $`[0,1]`$ | PF:841–891 | JX:56–93, 298–310 |
| Working continuation | $`\sum_{z'} P_z V_{j+1}(z', z, a')`$ | PF:997–1002 | JX:320–330 |
| Retired continuation (C1) | $`V_{j+1}(0, a')`$ | PF:993 | JX:332 |
| Terminal age; default rule (M1) | $`a' = 0`$; $`c = 10^{-10}`$, no continuation | PF:953–965; 1020–1024 | JX:366–439; 354–361 |
| Asset grid | $`a_i = 200(i/99)^{1.5}`$ | PF:435–447 | JX:1032 (same array) |
| Income process | Tauchen + $`z = 0`$; $`f`$, $`s_e`$; $`\alpha`$ discretisation | PF:496–534, 538–559 | copied, JX:1015, 1032–1037 |
| Entry state | $`a = 0`$, $`z \sim`$ stationary, $`z_{last} = z`$ | PF:1137–1155; OT:630–642 | JX:1201–1225 |
| $`z_{last}`$ update; freeze | previous state incl. 0 | PF:1270–1271 | JX:790 |
| Death, bequest record | $`1 - \pi_j`$; $`a`$ recorded | PF:1256–1264 | JX:797–803 |
| Cohort survival schedule | $`\pi_j(y)`$, clamped years | OT:1023–1040, 1047–1066 | same (schedules stacked, OT:440–444, 689–692) |
| Cohort paths before 2023 | period-0 values | OT:58–80, 1154–1166 | same |
| Baseline rules kept in counterfactuals | pre-2023 ages | OT:1202–1273 | OT:1308–1345 |
| Population weights, living share | §1.1 | OT:938–956, 973–1021 | same |
| $`\Gamma_t`$ | $`(1+g)(1+n_t)`$, clipped at the table end | OT:905–936 | same |
| Aggregates $`A, C, L`$ | per capita; $`L = \Sigma(\cdot)/w`$ | OT:806–823, 2163–2173 | same |
| Firm conditions, $`K`$, $`Y`$ | §1.5 | OT:2002–2004, 2185–2198; CA:1069–1096 | same |
| Public capital | $`K^g_{t+1} = [(1-\delta_g)K^g_t + I^g_t]/\Gamma_t`$ | OT:1972–1978 | same |
| Baseline $`I^g`$ level | $`(\delta_g + \Gamma_t - 1)K^g`$ | RF:98–99; FR:540–541 | — |
| Government budget | <a href="#eq:pd" data-reference-type="eqref" data-reference="eq:pd">[eq:pd]</a> | OT:1735–1851 | same |
| Debt law | <a href="#eq:debt" data-reference-type="eqref" data-reference="eq:debt">[eq:debt]</a> | FE:264–297 | — |
| $`B_0`$ (M9) | $`\bar b\,Y_0`$ | RF:102–106, 123 | — |
| NFA, CA | $`A - K - B`$; $`\Gamma_t \mathrm{NFA}_{t+1} - \mathrm{NFA}_t`$ | FE:650–667, 755–758 | — |
| Pension fund (memorandum) | $`[(1+r_B)S + \mathrm{Rev}^p - \mathrm{PENS}]/\Gamma_t`$ | OT:2305–2317 | — |
| Balance residual, targets | $`B_{T_{bal}-1}/Y_{T_{bal}-1}`$; NFA; rest point | FE:326–357; RF:320–334 | — |
| Terminal-convergence check | $`K, K^g, L, Y, C, A`$, NFA, $`S`$ | FE:406–461 | — |
| Horizon extension (C2) | 20 periods | RF:218; FE:217–243, 730–736 | — |
| Goods-market residual (M3) | §1.7 | FR:302–364; `eval_fiscal_results.py:225--262` | — |
| SMM cross-section (cohort by cohort) | 60 solves per education type | CA:748–817, 826–836 | same classes |
| Moments, weights | among the living, 2023 shares | CA:328–353, 587–695, 1174–1189 | — |
| Scale loop, balancing item | §2.3 | `run_scale_loop.sh:23--68`; `pin_baseline_closure.py:60--120` | — |

# Dead-code inventory

Agent B6’s inventory, reproduced from its report; classes (a)–(f) as defined in the audit plan. Line numbers are from the files at `3784ae5`. No view on deletion.

## Method

**Mechanical inventory.** `b6_tools/inventory.py` (standard library only) parses the 28 Python files (`code/*.py`, `code/reports/*.py`) with `ast` and records every `FunctionDef`/`AsyncFunctionDef`/`ClassDef` with its decorators (`njit`, `staticmethod`, `pytest.mark.*`), its enclosing class, and its line range: **611 definitions** (324 in non-test modules, 287 in `test_*.py`). It then records every `ast.Name` and `ast.Attribute` reference in all files with file:line, every string constant that contains a definition name (string-only hits are reported separately and never counted as callers), and every word-boundary hit in the five `*.sh` drivers (comments stripped). A reference inside the definition's own line range is a self-reference, not a caller. The raw result, one row per definition with caller counts by class (`n_refs_nontest`, `n_refs_test`, `n_refs_shell`, `n_self_refs`, `n_string_hits`) and the first twelve call sites, is `B6_inventory.csv` next to this file (a `class`/`note` column has been appended after the manual pass).

**Caller classes.** (i) production drivers: `run_scale_loop.sh`, `run_step0_baseline.sh`, `chain_fiscal_after_loop.sh`, `run_cost_and_figure.sh` and the scripts they invoke (`calibrate.py --config … --backend jax`, `normalize_A_tfp.py --backend jax --write`, `pin_baseline_closure.py --backend jax --write`, `diag_ss_vs_transition.py jax`, `check_a0_predetermination.py` (runs **both** backends), `reports/fill_report.py --backend jax --run-baseline --implied`, `run_fiscal_figures.py --config … --shock both --backend jax`, `pytest test_olg_transition.py::TestCohortBatchedSurvival`); the manual CLIs the task lists (`eval_fiscal_results.py`, `regen_fiscal_figures_from_json.py`, `validate_backends.py`, `diag_bequest_decomp.py`, `health_flag_decomposition.py`, the `build_*.py` builders) — none of the `.sh` drivers invokes these, they count as callers only for what they themselves reach; (ii) `test_*.py`. Dictionary dispatch (`MOMENT_DISPATCH`, calibrate.py:698-723), `getattr` lookups (`_as_alpha_indexed`, olg_transition.py:575-579), the `multiprocessing` wrappers and the shell drivers were counted as callers. Every candidate was then confirmed by reading the call sites and, for flags, the production config `calibration_input_GR.json` and the drivers.

**Config keys.** `calibration_input_GR.json` is read by `load_config`/`build_lifecycle_config`/`build_olg_transition` (calibrate.py:1197-1454): every `model.*` and `external_params.*` key is forwarded as a `LifecycleConfig` kwarg (1296-1302, `pop_growth` excepted), `_derived.theta` overwrites the matching fields (1314-1329), `fiscal.*` keys are read by name in `compute_fiscal_ratios`/`build_olg_transition`/`pin_baseline_closure.py`/`run_fiscal_figures.py`/`eval_fiscal_results.py`/`reports/fill_report.py`, and `untargeted.*` by `generate_report` and `fill_report`. A key counts as read only if the read has an effect on a production output; keys read into an object that nothing then consults are listed under (e).

**Production configuration facts used for class (c)/(d):** `model.n_h=1`, `model.n_alpha=5`, `model.labor_supply=true`, `model.tax_progressive=false`, `external_params.transfer_floor=0.0`, `production.eta_g=0.05`, `prices.r_B=0.019`, `transition.demography_file=../data/demography_GR.npz` (present; `px` shape (310, 60), `cross_section_base` length 60 = `model.T`), `transition.r_initial=r_final=0.04`, `calibration.base_year_cohorts=true`, `simulation.n_sim=10000`, `simulation.n_sim_cohorts=2000`, `transition.n_sim=2000`; `run_fiscal_figures.py` scenarios use `financing in {'debt','tau_l'}`, `balance_condition in {'terminal_debt_gdp','terminal_nfa_gdp'}`, `recompute_bequests=False`, `adjustment_profile=None`, `n_post=20`, no `nfa_limit`/`ca_limit`.

## Items

### (a) Unreachable

| \# | definition (file:line) | kind | pattern searched | result (callers) | evidence read |
|:---|:---|:---|:---|:---|:---|
|  | `lifecycle_jax.py:50` | function solve_labor_hours_jax | `\bsolve_labor_hours_jax\b over code/**.py, *.sh` | definition only (0 callers, 0 string hits) | solve_period_jax:303 and \_solve_terminal_period_jax:426 call solve_labor_robust_jax; second implementation of the labour FOC without the (1+tau_c) wedge |
| 2 | `lifecycle_perfect_foresight.py:821` | method LifecycleModelPerfectForesight.\_solve_labor_hours | `\b_solve_labor_hours\b` | definition only | \_solve_state_choice:956,979 call \_solve_labor_newton; this version omits kappa(t) and the (1+tau_c) wedge |
| 3 | `olg_transition.py:839` | staticmethod @njit OLGTransition.\_compute_wage_path_njit | `\b_compute_wage_path_njit\b` | definition only | simulate_transition computes w inline from the firm FOC at 2002-2004 |
| 4 | `olg_transition.py:1458` | method OLGTransition.\_get_cached_cohort_panel | `\b_get_cached_cohort_panel\b` | definition only | all consumers read \_cohort_panel_cache directly (1521, 1615) |
| 5 | `olg_transition.py:1708` | method OLGTransition.compute_aggregates | `\bcompute_aggregates\b` | definition only; 1 string hit (compute_government_budget docstring 1739) | simulate_transition aggregates inline at 2160-2167; own docstring says no caller remains |
| 6 | `fiscal_experiments.py:1341-1342, 1394-1397` | branches in compare_scenarios (variables=None default; direct NFA/CA keys) | `compare_scenarios\( call sites; variables=` | run_fiscal_figures 442-466, regen 224-239, test_fiscal_experiments:612 all pass variables; none lists NFA or CA | the default list \[Y, primary_deficit, B_gdp_path, NFA\] and the NFA/CA key branches have no caller |

### (b) Test-only

| \# | definition (file:line) | kind | pattern searched | result (callers) | evidence read |
|:---|:---|:---|:---|:---|:---|
|  | `calibrate.py:134` | function compute_wealth_gini | `\bcompute_wealth_gini\b` | test_calibrate.py:102,104,110,114 only | unweighted single-panel versions; the SMM/report use the age- and education-weighted *moment*\* functions (372-584), which do not call these |
| 2 | `calibrate.py:151` | function compute_zero_wealth_fraction | `\bcompute_zero_wealth_fraction\b` | test_calibrate.py:120,126,134 only | unweighted single-panel versions; the SMM/report use the age- and education-weighted *moment*\* functions (372-584), which do not call these |
| 3 | `calibrate.py:160` | function compute_wealth_to_income_by_age | `\bcompute_wealth_to_income_by_age\b` | test_calibrate.py:217 only | unweighted single-panel versions; the SMM/report use the age- and education-weighted *moment*\* functions (372-584), which do not call these |
| 4 | `calibrate.py:177` | function compute_unemployment_rate | `\bcompute_unemployment_rate\b` | test_calibrate.py:142,150,159 only | unweighted single-panel versions; the SMM/report use the age- and education-weighted *moment*\* functions (372-584), which do not call these |
| 5 | `calibrate.py:185` | function compute_health_distribution_by_age | `\bcompute_health_distribution_by_age\b` | test_calibrate.py:167,178 only | unweighted single-panel versions; the SMM/report use the age- and education-weighted *moment*\* functions (372-584), which do not call these |
| 6 | `calibrate.py:198` | function compute_average_hours | `\bcompute_average_hours\b` | test_calibrate.py:188,198 only | unweighted single-panel versions; the SMM/report use the age- and education-weighted *moment*\* functions (372-584), which do not call these |
| 7 | `calibrate.py:206` | function compute_consumption_gini | `\bcompute_consumption_gini\b` | test_calibrate.py:206 only | unweighted single-panel versions; the SMM/report use the age- and education-weighted *moment*\* functions (372-584), which do not call these |
| 8 | `fiscal_experiments.py:186,191,201,207` | functions uniform_profile, linear_phase_in, back_loaded, exponential_convergence | `\b(uniform_profilelinear_phase_inback_loadedexponential_convergence)\b` | test_fiscal_experiments.py:185-209, 458-463 only | FiscalScenario.adjustment_profile is never set outside tests (grep adjustment_profile: run_fiscal_figures/check_a0 none), so \_get_psi:467 always returns ones |
| 9 | `olg_transition.py:851, 856, 788` | methods production_function, factor_prices; staticmethod @njit \_production_function_njit | `\b(production_functionfactor_prices_production_function_njit)\b` | test_olg_transition.py:80,89,117,1593-1594,1608-1609; \_production_function_njit called only from production_function:854 | the transition uses \_compute_output_path_njit (2198) and inline FOCs |
| 10 | `olg_transition.py:2687` | function get_test_config | `\bget_test_config\b` | 29 test refs; one non-test caller run_fast_test:2721 (CLI only, see f) | test fixture |
| 11 | `fiscal_experiments.py:1250-1252` | branch run_fiscal_scenario: 'base_macro' in base_paths | `'base_macro'` | test_fiscal_experiments.py:354-358 builds bp2 with base_macro/base_budget; run_fiscal_figures never does | production always re-runs the baseline inside run_fiscal_scenario |

### (c) Switched off in production

| \# | definition (file:line) | kind | pattern searched | result (callers) | evidence read |
|:---|:---|:---|:---|:---|:---|
|  | `calibrate.py:881, 908-964 (nested de_callback:911, polish_obj:942, polish_cb:947)` | function smm_objective_bounded; differential-evolution branch of calibrate() | `--methoddifferential_evolutionsmm_objective_bounded` | callers: calibrate():913,921 only inside the DE branch; run_scale_loop.sh:42 passes no --method (default Nelder-Mead); no test passes method= | CLI flag never set by a driver or test |
| 2 | `calibrate.py:1772, 1828-1845, 1860-1863, 1899-1919` | function default_spec; main() branches --test, no --config, --output | `default_specargs\.testargs\.output` | default_spec: main:1861 (no --config) and test_calibrate.py:401; drivers always pass --config and never --test/--output | run_scale_loop.sh:42 is the only production invocation |
| 3 | `calibrate.py:839-864` | single-solve branch of run_model_moments | `cohort_survival is not Nonebase_year_cohorts` | production config calibration.base_year_cohorts=true (calibration_input_GR.json:373) -\> load_config:1223 sets cohort_survival -\> run_model_moments takes the 826-837 branch; tests at test_olg_transition.py:2786,2993 clear cohort_survival | the non-cohort stationary panel is reached only by tests |
| 4 | `calibrate.py:1263 (w argument)` | parameter w of build_lifecycle_config | `build_lifecycle_config\(` | all callers (load_config:1206, build_olg_transition:1360, test_income_process.py:18,34) pass raw only | the w is not None branch 1271-1272 never runs |
| 5 | `lifecycle_perfect_foresight.py:1041, 1352, 1314, 1402, 1340` | methods \_solve_backward_parallel, \_simulate_parallel, \_combine_simulation_results; functions \_solve_period_wrapper, \_simulate_agent_batch | `parallel=n_jobs=` | only lifecycle_perfect_foresight.py:1467,1471 (the module **main** --test block) pass parallel=True; no driver, script or test does | multiprocessing path; its only caller is itself broken (see d: 1472-1475) |
| 6 | `lifecycle_perfect_foresight.py:394` | method \_print_income_diagnostics | `LifecycleModelPerfectForesight\(LifecycleModelJAX\(cls\(cfg` | every constructor call in calibrate.py:804,850, olg_transition.py:1200,1251,1254, lifecycle_jax.py:1015(forwards), tests passes verbose=False; only lifecycle_perfect_foresight.py:1465 (**main**) passes True | reached only from the module **main** |
| 7 | `lifecycle_perfect_foresight.py:734-737,753-756,831-832,863,1239-1241,1247-1249; lifecycle_jax.py:44 (_hsv_tax),156-160,173-177,299,422,751-759` | HSV progressive-tax branches | `tax_progressive` | config model.tax_progressive=false (json:18); set True only in test_olg_transition.py:1127,1330,1435 | under JAX the jnp.where evaluates \_hsv_tax eagerly but discards it; tax_progressive is a static argname so XLA removes it |
| 8 | `lifecycle_perfect_foresight.py:166-168, 775-778; lifecycle_jax.py:195-197, 497-501, 539-541, 572` | schooling_years / child_cost_profile / education_subsidy_rate feature | `schooling_yearschild_cost_profileeducation_subsidy_rate` | no config key, no driver; test_olg_transition.py:1226,1392,1438 only | feature defaults 0 / zeros / 0.0 |
| 9 | `lifecycle_perfect_foresight.py:175, 364-370, 789-790, 796-797; lifecycle_jax.py:489, 504-505, 781-785, 1069-1072` | P_y_by_age_health (4-D income transitions) feature | `P_y_by_age_health=` | never set anywhere: only forwarded as \_lc.P_y_by_age_health (olg_transition:1101) and self.P_y_4d; no test sets it | the P_y_age_health=True branches have no caller at all (a-adjacent) |
| 10 | `lifecycle_perfect_foresight.py:178-179, 1147-1153; lifecycle_jax.py:1212-1223; olg_transition.py:634-636` | initial_assets / initial_asset_distribution | `initial_assets=initial_asset_distribution=` | test_olg_transition.py:2162-2204 only | production starts every agent at a_grid\[0\] |
| 11 | `lifecycle_perfect_foresight.py:146; olg_transition.py:1108, 1780-1782, 1866-1877` | tau_beq (bequest tax) | `tau_beq` | no config key; test_olg_transition.py:2041,2059,2079 only | in production bequest_tax=0 and bequest_transfers=total_bequests identically |
| 12 | `lifecycle_perfect_foresight.py:147, 771-772; lifecycle_jax.py:193, 557; olg_transition.py:1181-1185, 1853-1884, 2064-2111` | bequest_lumpsum / recompute_bequests loop / \_compute_bequest_lumpsum_path | `recompute_bequests=Truebequest_lumpsum` | run_fiscal_figures leaves FiscalScenario.recompute_bequests=False; set True only by diag_bequest_decomp.py:90 (case C) and test_fiscal_experiments.py:119,136 | off in the calibration/fiscal/report chain; reached by one diagnostic script and tests |
| 13 | `lifecycle_perfect_foresight.py:84-120, 224-260, 378-382, 568-625` | n_h\>1 health machinery: h_moderate/h_poor/m_moderate/m_poor/P_h_young/middle/old/P_h, **post_init** P_h build, \_health_process n_h in {2,3} branches | `n_h=2n_h=3P_h=` | config model.n_h=1; test_olg_transition.py:39 builds n_h=2 (constructor only); run_full_simulation:2819 uses n_h=2 (CLI, cannot complete, see d); nobody passes P_h= | with n_h=1: m_grid_base=linspace(m_good,m_poor,1)=\[m_good\], h_grid=\[h_good\], P_h=ones |
| 14 | `lifecycle_perfect_foresight.py:803-804; lifecycle_jax.py:492-493, 897-898; olg_transition.py:438-441, 695-698` | survival_probs is None fallbacks (no-mortality) | `survival_probs is None_ones_surv` | production always has schedules (config survival_probs + demography); test_olg_transition.py:2000 and fixtures without a schedule | fallback to ones |
| 15 | `olg_transition.py:132, 276, 1044-1045, 1114` | survival_improvement_rate (legacy longevity scaling) | `survival_improvement_rate` | default 0.0 everywhere in production; test_olg_transition.py:2121 only | the data path (\_surv_px) supersedes it: \_survival_schedule_at_year returns at 1040 before the legacy branch |
| 16 | `olg_transition.py:104,122,126,129 (ctor paths), 1806-1809 obj_level fallbacks; fiscal_experiments.py:1227-1230` | constructor-level I_g_path / govt_spending_path / defense_spending_path / other_net_spending_path | `govt_spending_path=defense_spending_path=other_net_spending_path=I_g_path=` | production passes per-call paths/ratios (run_fiscal_figures 195-205 via \_apply_shock); ctor forms set only in test_olg_transition.py:1266 and diag_bequest_decomp.py:88-89 (per call) | the \_spend(..., obj_level) fallback is never the value used in production |
| 17 | `olg_transition.py:108, 183-186, 1814-1822` | constructor B_path and the debt-service block of compute_government_budget | `B_path=` | test_olg_transition.py:1552 only; fiscal_experiments computes B outside the OLG object (compute_debt_path) | in production debt_service=0, new_borrowing=0 and fiscal_deficit==primary_deficit for every t |
| 18 | `olg_transition.py:124, 213, 2306` | constructor S_pens_initial | `S_pens_initial` | test_olg_transition.py:1624,1656 only | the S_pens recursion itself (2305-2318) is live: it starts at 0 and accumulates tax_p - pension |
| 19 | `olg_transition.py:863-903, 1897, 1965-1966` | set_cohort_sizes_path_from_pop_growth / pop_growth_path kwarg | `pop_growth_path` | olg_transition.py:2773, 2891 (run_fast_test / run_full_simulation, CLI only); no driver or test | production builds cohort weights from the demographic entrants (1928-1929) |
| 20 | `olg_transition.py:1988-1990; fiscal_experiments.py:1236-1239; run_fiscal_figures.py:365-367; eval_fiscal_results.py:549-551` | r_B is None fallbacks to the capital rate | `r_B` | config prices.r_B=0.019 (json:35) is always set by build_olg_transition:1410 | legacy fallback |
| 21 | `olg_transition.py:1886 (w_path kwarg), 2009-2012` | simulate_transition 'Using provided wage path' branch | `simulate_transition\(.*w_path=` | test_olg_transition.py:431,557,617,707,792,869 pass a constant w_path; every script passes None | production derives w from the firm FOC (1993-2008) |
| 22 | `olg_transition.py:1414-1438` | multi-agent-batch branch of \_ensure_cohort_panel_cache (n_agent_batches\>1) | `sim_agent_batch_size` | production n_sim=2000 (transition.n_sim) vs batch 10000 (calibrate.py:1419 default) -\> n_agent_batches==1; no test sets sim_agent_batch_size | reached only if n_sim exceeds 10000 |
| 23 | `olg_transition.py:1201-1202, 1253-1277, 1439-1450, 366-404 (_simulate_birth_cohort_cached)` | NumPy-backend branches of the transition | `backend != 'jax'_simulate_birth_cohort_cached` | drivers: run_scale_loop/run_step0/chain/run_cost all pass --backend jax to calibrate, normalize, pin, fill_report, run_fiscal_figures; the NumPy branches are reached by check_a0_predetermination.py:74 (both backends; called by run_step0_baseline.sh:46 and chain_fiscal_after_loop.sh:27), validate_backends.py:236, and diag\_\*.py when given numpy | switched off in the calibration/fiscal/report chain; on in two diagnostic drivers and the tests |
| 24 | `fiscal_experiments.py:350-366 (terminal_flow_balance, pv_balance, period_balance); 862-884 (+_objective:865); 120 (discount_rate); 303 (r_terminal)` | \_balance_residual conditions other than terminal_debt_gdp / terminal_nfa_gdp; run_tax_financed period_balance block | `balance_condition\s*=` | run_fiscal_figures.py:327,335 set only 'terminal_debt_gdp' and 'terminal_nfa_gdp'; check_a0:57,65 'terminal_debt_gdp'; pv_balance/period_balance only in test_fiscal_experiments.py:296,305,410,430 | three balance conditions and the discount_rate / r_terminal inputs are unused in production |
| 25 | `fiscal_experiments.py:587-602 (tau_c, tau_k, tau_p, pension_replacement, transfer_floor financing); 90-93 (delta_tau_c/k/p_path, delta_pension_path)` | \_apply_shock financing branches and shock fields other than G/I_g/tau_l | `financing\s*=delta_tau_c_pathdelta_tau_k_pathdelta_tau_p_pathdelta_pension_path` | production: financing in {'debt','tau_l'} (run_fiscal_figures 223-297), shocks delta_G_path/delta_I_g_path; check_a0 uses delta_tau_l_path; tests: tau_c (409), transfer_floor (258); the four delta_tau_c/k/p/pension fields are set nowhere, not even in tests | transfer_floor financing is also dead by construction (see d) |
| 26 | `fiscal_experiments.py:126-127, 691-713, 1000-1187 (+_full_nfa_ca:1037, _feasible:1064, _nfa_ok_at:1104)` | nfa_limit / ca_limit, \_check_nfa_violation, run_nfa_constrained | `nfa_limitca_limit` | run_fiscal_figures never sets nfa_limit/ca_limit (its 'nfa' scenarios use balance_condition terminal_nfa_gdp through run_tax_financed); test_fiscal_experiments.py:541 sets nfa_limit=200 | the dispatcher at 1295-1301 never routes a production scenario here |
| 27 | `fiscal_experiments.py:564-566; run_fiscal_figures.py:202-203, 266; validate_backends.py:109-112` | GDP-share I_g (I_g_over_Y) mode | `I_g_over_Y` | enabled only when eta_g==0; production eta_g=0.05 (json:42); simulate_transition:1952 refuses it otherwise | level-mode I_g is the live path |
| 28 | `fiscal_experiments.py:1480-1481` | fiscal_multiplier growth_factors=None branch | `fiscal_multiplier\(` | run_fiscal_figures.py:438,510 pass growth_factors; test_fiscal_experiments.py:630 does not | untrended fallback |
| 29 | `run_fiscal_figures.py:136-193, 206-211, 229, 269` | hardcoded fast-test branch (no --config) | `run_fiscal_figures\.py` | chain_fiscal_after_loop.sh:35 always passes --config; no other driver or test invokes the script | demo configuration |
| 30 | `reports/fill_report.py:169-172, 177-181` | markdown-parsed fallback of moments_table (live is None) | `--implied--run-baseline` | run_step0_baseline.sh:52-54 and run_cost_and_figure.sh:31-33 pass both flags, so live is always set | parse_md_table/num remain live for header.tex:642 |
| 31 | `eval_fiscal_results.py:592-608; 86-88` | bisection_flow_target branch; scalar Gamma fallback in \_growth_factor | `terminal_flow_balancegrowth_factor_path` | production JSON carries balance_condition 'terminal_debt_gdp' (run_fiscal_figures 327) and params.growth_factor_path (548) | legacy-JSON paths |

### (d) Dead inside live code

| \# | definition (file:line) | kind | pattern searched | result (callers) | evidence read |
|:---|:---|:---|:---|:---|:---|
|  | `calibrate.py:75-77` | wrap_sim_output 21-tuple padding branch | `return \(a_sim.*alpha_idx_panelresult_21 \+` | both simulate() implementations return 22 arrays (lifecycle_perfect_foresight.py:1282-1285; lifecycle_jax.py:963); test_calibrate.py:384 feeds a 21-tuple | condition cannot hold for a real panel |
| 2 | `calibrate.py:587-637 vs 1474-1527` | \_compute_ss_aggregates and compute_fiscal_ratios: two implementations of the age-weighted aggregates and Y | `_compute_ss_aggregatescompute_fiscal_ratios` | both live: the first behind the SMM ratio moments and normalize_A_tfp:95, the second behind pin_baseline_closure:68, fill_report:143,270, diag\_\*; they take K_over_L from different places (spec.production vs config_data\[\_derived\]) | same loop body duplicated; identical only when load_config built both |
| 3 | `calibrate.py:1099, 1192-1194` | compute_age_weights via the base_year_age_weights fallback | `compute_age_weightscross_section_base` | data/demography_GR.npz exists with cross_section_base of length 60 == model.T, so base_year_age_weights returns at 1189; fallback reached only by tests (test_olg_transition.py:2436,2440,3006) | fallback masks a missing sidecar by printing a message and using the stationary approximation |
| 4 | `calibrate.py:1337-1347` | pension_avg_weight derivation | `pension_avg_weightp\``[``'name'\``]`` == 'rho_y'` | config has no pension_avg_weight key and no 'rho_y' entry in calibration.params, so rho_init stays the literal 0.95 (1342); edu_params.\*.rho_y (json:65,72,79) is not consulted | coincides numerically today (rho_y=0.95) but the config value does not reach this formula |
| 5 | `calibrate.py:1384-1397` | survival_data_file block of build_olg_transition | `survival_data_filesurvival_table is None` | demography_GR.npz loads first (1373-1379, px (310,60) matches T=60) so survival_table is already set at 1385 | dead in production; see also (e) transition.survival_data_file |
| 6 | `calibrate.py:1427` | exponential r-path branch (r_i != r_f) | `r_initialr_finalr_decay` | config r_initial=r_final=0.04 (json:396-397) -\> np.full branch | transition.r_decay has no effect |
| 7 | `calibrate.py:1314-1329 (effect on model.beta, model.nu, external_params.tau_p, pension_replacement_default, m_good)` | \_derived.theta overwrite of config values | `derived_theta_derived` | json model.beta=0.96 (13), model.nu=68.35 (16), external_params.tau_p=0.34 (48), pension_replacement_default=0.18 (50), m_good=0.048 (55) are replaced by \_derived.theta (22-26) at 1329; build_olg_transition:\_tax reads theta first (1434-1436); generate_report:1655 still prints the external_params values | five parameter values in the file are read but never used while \_derived.theta exists |
| 8 | `calibrate.py:1249, 783, 826-830 (spec.n_sim vs n_sim_cohorts)` | simulation.n_sim read into spec.n_sim but unused for the production panels | `spec\.n_simn_sim_cohorts` | with base_year_cohorts=true, run_model_moments -\> base_year_cross_section(theta, spec) with n_sim=None -\> spec.n_sim_cohorts (783); spec.n_sim is then only printed (calibrate.py:903,1628; pin_baseline_closure.py:64); same for calibrate.py --n-sim, diag_ss_vs_transition.py:44 and fill_report.py:581 (its --n-sim does govern the baseline transition at 546) | simulation.n_sim=10000 (json:388) does not determine the SMM Monte Carlo size; n_sim_cohorts=2000 (json:390) does |
| 9 | `lifecycle_perfect_foresight.py:781-783, 301-309; lifecycle_jax.py:205-207; olg_transition.py:1958-1962, 2251-2252, 1234-1236; fiscal_experiments.py:595-602, 621-623, 640` | transfer_floor machinery | `transfer_floor` | LifecycleModelPerfectForesight.**init**:301 raises NotImplementedError for transfer_floor\>0 (LifecycleModelJAX builds that object at 1015), so every branch that handles a positive floor is unreachable; config external_params.transfer_floor=0.0 (json:53) | dead by construction; tests at test_olg_transition.py:1160,1410,1436,2368 construct positive floors (not checked here whether they are expected to raise) |
| 10 | `lifecycle_perfect_foresight.py:809-812, 815-819, 912-929; 1359, 1370-1388` | retirement_window branches (\_is_retired_at window case, \_in_retirement_window, dual solve in \_solve_period and \_solve_period_wrapper) | `retirement_window` | **init**:316-324 raises for any non-None window (test_olg_transition.py:1926 asserts the raise), so \_in_retirement_window always returns False | dead by construction |
| 11 | `lifecycle_perfect_foresight.py:1160-1163; lifecycle_jax.py:1230-1232; olg_transition.py:644-646` | initial_avg_earnings is None else-branches | `initial_avg_earnings` | LifecycleConfig.initial_avg_earnings defaults to 0.0 (180) and nothing sets it to None (grep initial_avg_earnings=None: none) | branch cannot run |
| 12 | `lifecycle_perfect_foresight.py:45, 294` | LifecycleConfig.N_earnings_history / self.N_earnings_history | `N_earnings_history` | assigned at 294, read nowhere | parameter read and never used |
| 13 | `lifecycle_perfect_foresight.py:1414-1690` | **main** --test block | `= results$ (1472-1475)` | unpacks 19 names from the 22-tuple simulate() returns (1282-1285) -\> ValueError before any output | the only caller of the parallel code path (see c) cannot complete |
| 14 | `lifecycle_jax.py:101-103 (P_y, P_h_t), 117 (labor_hours)` | compute_budget_jax parameters P_y, P_h_t unused in the body; labor_hours never varied | `def compute_budget_jaxcompute_budget_jax\(` | body 134-209 references neither P_y nor P_h_t; both call sites (261-284, 392-414) omit labor_hours (default 1.0) | parameters read and never used |
| 15 | `lifecycle_jax.py:1052, 1122-1152` | LifecycleModelJAX.bequest_lumpsum not forwarded by LifecycleModelJAX.solve | `bequest_lumpsum` | solve() passes no bequest_lumpsum= to \_solve_lifecycle_jax_jit (default 0.0); the attribute is read only by the batched solve in olg_transition.py:433 | on the single-model JAX path (calibrate.py SMM) a non-zero config.bequest_lumpsum would be ignored; moot while recompute_bequests is off |
| 16 | `lifecycle_jax.py:1102, 1173 (**kwargs)` | LifecycleModelJAX.solve/simulate accept and ignore \*\*kwargs | `def solve\(self, verbose=False, \*\*kwargs\)def simulate\(.*\*\*kwargs` | parallel/n_jobs accepted for interface parity, never read | parameters never used |
| 17 | `olg_transition.py:42-45` | \_panel_to_age_means legacy \>=19 branch | `len\(panel_data\)` | both backends return 22 (see wrap_sim_output item) | condition cannot hold |
| 18 | `olg_transition.py:329-364; 1535-1558; 1648-1668` | \_slice_mean_single_age_njit and the raw-panel branches of \_period_cross_section / \_compute_all_cross_sections | `len\(panel_data\) == 11_slice_mean_single_age_njit` | \_cohort_panel_cache is filled with 11-tuples of per-age means on both backends (1412-1413, 1436-1438, 1445-1450 via \_simulate_birth_cohort_cached:402), so len(panel_data)==11 always | njit kernel and two else-branches unreachable; a third implementation of the per-age mean |
| 19 | `olg_transition.py:1493-1591` | \_period_cross_section body (cache miss path) | `_period_cross_section\(_period_cache\``[` | \_compute_all_cross_sections:1686-1704 pre-populates \_period_cache for every t with key (t, n_sim, 42, pv); compute_government_budget_path is called with the same n_sim right after simulate_transition (fiscal_experiments.py:646, diag/fill/validate), so 1490-1491 hits; a miss needs a different n_sim; \_compute_bequest_lumpsum_path (recompute_bequests only) also hits | second implementation of the per-period cross-section, reached only on cache miss |
| 20 | `olg_transition.py:762` | \_birth_sim_cache write in the JAX batched simulate | `_birth_sim_cache` | the only reader is \_simulate_birth_cohort_cached:382 (NumPy branch of \_ensure_cohort_panel_cache:1445); under backend=jax nothing reads the entry; it also stores a raw 22-tuple where the reader expects an 11-tuple | cache written and never read |
| 21 | `olg_transition.py:766-784, 971` | \_create_cohort_sizes / \_cohort_sizes_njit -\> self.cohort_sizes | `self\.cohort_sizes\b` | read only at \_cohort_weights:971 when no cohort_sizes_path and no demography; production has demography (json:401) so simulate_transition:1928-1929 builds cohort_sizes_path; tests at test_olg_transition.py:2438 read \_cohort_sizes_njit directly | computed in **init** and unused in production |
| 22 | `olg_transition.py:189, 195 (pop_growth, growth_factor)` | OLGTransition.pop_growth and scalar growth_factor | `self\.growth_factor\bself\.pop_growth` | growth_factor read by \_growth_at:913 only when growth_factor_path is None (simulate_transition sets it at 1927 first) and by growth_factors:924 only when \_demog is None; pop_growth feeds \_cohort_sizes_njit (unused, above) and run_fiscal_figures params_out:544 | with the demographic sidecar both are inert; see (e) external_params.pop_growth |
| 23 | `olg_transition.py:196-197, 1037, 1044, 1060` | OLGTransition.birth_year | `birth_year` | in \_cohort_survival_schedule cal_year=birth_year+birth_period+j (1060) and \_survival_schedule_at_year true_cal=cal_year+(current_year-birth_year) (1037): birth_year cancels; remaining readers \_cohort_sizes_njit (unused) and set_cohort_sizes_path_from_pop_growth (CLI only) | parameter with no effect on production output; see (e) transition.birth_year |
| 24 | `olg_transition.py:1041-1045` | legacy branch of \_survival_schedule_at_year | `_surv_px is not None` | production passes survival_table (calibrate.py:1414) so 1036-1040 returns | reached only without a survival table (tests/fixtures) |
| 25 | `olg_transition.py:106, 179, 2181, 2195` | economy_type and the closed-economy branch (K_domestic is None) | `economy_type=` | only calibrate.py:1408 passes economy_type='soe'; default is 'soe'; nothing passes anything else | closed-economy branch unreachable |
| 26 | `olg_transition.py:120, 231-233, 2472, 2554, 2678; run_fiscal_figures.py:81` | OLGTransition.output_dir | `output_dir` | read only by the three plot\_\* methods, which no driver or test calls; **init** still mkdirs it; run_fiscal_figures:81 assigns it and never reads it | parameter whose only readers are unreachable; side effect: creates ./output at construction |
| 27 | `olg_transition.py:131, 269-275` | fertility_path tombstone | `fertility_path` | raises NotImplementedError; test_olg_transition.py:2142 asserts the raise | parameter kept only to refuse |
| 28 | `olg_transition.py:1814-1822 -> keys debt_service, new_borrowing, fiscal_deficit` | budget keys constant in production | `B_path is not None` | B_path is never set on the object in production (see c), so debt_service=new_borrowing=0 and fiscal_deficit==primary_deficit; run_fiscal_figures writes them to fiscal_results.json; eval_fiscal_results reads none of them (grep debt_servicenew_borrowingfiscal_deficit: none) | three JSON series that carry no information in production |
| 29 | `olg_transition.py:2809-2914` | run_full_simulation | `n_h=_n_h_alive_fraction` | builds n_h=2 (2819,2839); simulate_transition -\> \_compute_all_cross_sections -\> \_aggregation_weights -\> \_alive_fraction raises NotImplementedError for n_h\>1 (993-1001) | the default branch of `python olg_transition.py` cannot complete |
| 30 | `olg_transition.py:2605-2606` | plot_lifecycle_comparison use_crn=False legacy seed | `use_crn` | no caller passes use_crn | inside an already unreachable method |
| 31 | `fiscal_experiments.py:1432` | `if not False:` in compare_scenarios | `if not False` | constant condition | always prints |
| 32 | `fiscal_experiments.py:530-533 (G, I_g, defense, other) in ratio mode` | \_apply_shock computes level paths that ratio mode discards | `ratio_mode` | production always ratio_mode=True (run_fiscal_figures 201-205); the G/defense/other level reads at 530-533 are then unused (I_g is used) | minor |
| 33 | `run_fiscal_figures.py:70-71, 118, 208-211` | defense_path / other_path level variables | `defense_pathother_path` | None on both branches (71, 118); the `is not None` tests at 208 and 210 can never be true | dead conditionals |
| 34 | `regen_fiscal_figures_from_json.py:31-65 vs run_fiscal_figures.py:377-413; 184-198 vs 260-274` | duplicated panel definitions and duplicated Gamma-rebuild block | `MACRO_VARSgrowth_factor_path` | regen:30 says 'kept in sync manually'; the Gamma block is copy-pasted inside regen() and regen_budget_components() | two implementations of one table / one computation |
| 35 | `reports/fill_report.py:108-115 (I_g_over_Y, interest_over_Y, primary_balance_over_Y, G_over_Y, B_over_Y, health_oop_over_Y entries)` | LABEL dict entries never looked up | `LABEL\.getUNTARGETED` | LABEL is consulted for targeted names (108-110 subset, 166,171) and for UNTARGETED=\['ui_over_Y'\] (121) and UNTARGETED_DIST keys (126-127); the six listed keys are never passed | dead dictionary entries |
| 36 | `reports/fill_report.py:358-360` | r_star fallback in goods_market_residual | `paths\.get\('r'` | paths always contains 'r' (simulate_transition result 2257 -\> fill_report 547-548), so the default argument is built and discarded; this is the only reader of OLGTransition.r_star (olg_transition.py:180) besides tests | fallback that can never apply; r_star is otherwise an attribute set and never used |
| 37 | `reports/fill_report.py:580` | L2 = L alias | `L2` | alias of the same object | trivial |
| 38 | `diag_ss_vs_transition.py:44, 46, 83, 85` | spec.n_sim override, theta_dict, meanY, I_g_over_Y | `theta_dictmeanYI_g_over_Yspec\.n_sim` | theta_dict (46), meanY (83) and I_g_over_Y (85) are assigned and never read; spec.n_sim=N_SIM (44) does not change the SS panels (cohort path uses n_sim_cohorts) | the script argument n_sim governs only the transition side |
| 39 | `diag_bequest_decomp.py:48` | theta_dict | `theta_dict` | assigned, never read | trivial |

### (e) Config keys without production effect

| \# | definition (file:line) | kind | pattern searched | result (callers) | evidence read |
|:---|:---|:---|:---|:---|:---|
|  | `calibration_input_GR.json:400 transition.survival_data_file` | config key | `survival_data_file` | read at calibrate.py:1384; consulted only if survival_table is None (1385), which the demography block (1373-1379) already set | no effect in production |
| 2 | `calibration_input_GR.json:398 transition.r_decay (and r_initial/r_final 396-397 equal to prices.r)` | config key | `r_decayr_initialr_final` | calibrate.py:1423-1427: r_decay enters only the r_i != r_f branch; r_i==r_f==0.04 | r_decay has no effect; r_initial/r_final matter only as a constant level |
| 3 | `calibration_input_GR.json:394 transition.birth_year` | config key | `birth_year` | calibrate.py:1412 -\> OLGTransition.birth_year; cancels in the survival lookups (olg_transition.py:1037,1060), otherwise feeds only cohort_sizes (unused with demography) and the CLI-only pop_growth_path device | no effect on production output |
| 4 | `calibration_input_GR.json:58 external_params.pop_growth` | config key | `pop_growth` | skipped for LifecycleConfig (calibrate.py:1300-1301); OLGTransition.pop_growth inert with the demographic sidecar (see d); base_year_age_weights fallback dead; remaining readers: reports/fill_report.py:60,484-485 (params table, Level column, header Gammaval), run_fiscal_figures.py:544 (params_out), eval_fiscal_results.py:87,714 (fallback) | affects report text only, not the model solution |
| 5 | `calibration_input_GR.json:89-150 survival_probs` | config key (60-vector) | `survival_probs` | read into LifecycleConfig (calibrate.py:1330-1331); overridden per cohort in the calibration (calibrate.py:797-800, base_year_cohorts=true) and in the transition (olg_transition.py:1112-1116,1178-1180, survival_table set); residual readers are the recompute_bequests gate (2066), the legacy survival branch (1041) and the age-weights fallback (1193), all inactive in production | the 60 numbers do not enter any production computation |
| 6 | `calibration_input_GR.json:13 model.beta, :16 model.nu, :48 external_params.tau_p, :50 pension_replacement_default, :55 m_good` | config values overwritten by \_derived.theta | `_derived.*theta` | see (d) calibrate.py:1314-1329 | read, then replaced; printed unchanged by generate_report |
| 7 | `calibration_input_GR.json:388 simulation.n_sim` | config key | `n_sim` | see (d) calibrate.py:1249/783: with base_year_cohorts=true only simulation.n_sim_cohorts (390) sizes the SMM panels; simulation.n_sim is printed only | no effect on production numbers |
| 8 | `calibration_input_GR.json:381-385 untargeted.{earnings_var_mean, earnings_var_slope, consumption_gini, mean_assets, median_wealth_to_income} = null` | config keys | `config_data\.get\('untargeted'` | read by generate_report:1701-1714 and rendered as an em-dash when None; fill_report reads only income_gini and p90_p10_income (126-127) | inert null entries |
| 9 | `calibration_input_GR.json:286 fiscal.interest_over_Y` | config key | `interest_over_Y` | read only through the comparisons loop of compute_fiscal_ratios (1585-1591) and generate_report (1736-1739); fill_report excludes it on purpose (116-120) and its LABEL entry is dead | report-only; the model value is r_B\*B_over_Y by construction |
| 10 | `calibration_input_GR.json:283-285 fiscal.health_oop_over_Y, health_total_over_Y, tax_revenue_over_Y` | config keys | `health_oop_over_Yhealth_total_over_Ytax_revenue_over_Y` | generate_report comparisons; copied into fiscal_results.json params (run_fiscal_figures.py:564-566) and eval params (eval_fiscal_results.py:703-705) but chk_calibration_ratios checks only pensions/ui/health_gov/G (500-505) | read, carried into outputs, never used by a check |
| 11 | `calibration_input_GR.json:278 fiscal.I_g_over_Y` | config key | `I_g_over_Y` | calibrate.py:1449 -\> paths (consumed by run_fiscal_figures/validate_backends only when eta_g==0, by diag_bequest_decomp.py:82 as a level), compute_fiscal_ratios:1571 (report ratio primary_balance_full_over_Y), pin_baseline_closure.py:97 (printed); the production transition sizes I_g from K_g instead (pin_baseline_closure.py:74-95, run_fiscal_figures.py:98-99) | report/diagnostic-only in production |
| 12 | `calibration_input_GR.json:29 _derived.theta_metadata.calibration_date` | config key | `calibration_datetheta_metadata` | written by calibrate.py:1891-1894; fill_report.py:461-462 reads source_report only | never read |
| 13 | `code/calibration_input_GR_rB0.json` | config file | `rB0calibration_input_GR_rB0` | no reference in code/*.py, code/*.sh, code/reports/\*.py | config file nothing reads |

### (f) Scripts / outputs nothing reads

| \# | definition (file:line) | kind | pattern searched | result (callers) | evidence read |
|:---|:---|:---|:---|:---|:---|
|  | `olg_transition.py:2712, 2809, 2917, 2925, 2959, 2993` | CLI: run_fast_test, run_full_simulation, \_parse_backend, run_from_config, main | `olg_transition\.py` | no .sh driver and no test runs `python olg_transition.py`; outputs output/test/*.png and output/*.png are read by nothing in code/; run_full_simulation cannot complete (see d) | together with plot_government_budget:2358, plot_transition:2481, plot_lifecycle_comparison:2563, \_default_plot_filename:2346, \_sparse_int_ticks:320, whose only callers are these |
| 2 | `lifecycle_jax.py:1292-1371` | CLI --test block of lifecycle_jax.py | `lifecycle_jax\.py --test` | no driver or test; setup_jax.sh:51 only prints a suggestion to run `olg_transition.py --test` | standalone smoke test |
| 3 | `build_public_capital_GR.py` | builder script | `build_public_capital_GRrealgdp_GR` | prints K_g/Y and delta_g; writes only its download cache data/realgdp_GR.json; no code reads either | hand-run calibration aid; result transcribed into production.K_g/delta_g by hand |
| 4 | `build_pension_floor_GR.py` | builder script | `pension_floor_GRnomgdp_GR` | writes data/pension_floor_GR.json and cache data/nomgdp_GR.json; grep over code/ finds no reader of either | hand-run aid for external_params.pension_min_floor |
| 5 | `build_survival_GR.py` | builder script | `survival_GR` | writes data/survival_GR.npz; read by build_demography_GR.py:53 (live chain) and by the dead survival_data_file branch | live only as an input to build_demography_GR.py |

### Observations (not dead code)

| \# | definition (file:line) | kind | pattern searched | result (callers) | evidence read |
|:---|:---|:---|:---|:---|:---|
|  | `olg_transition.py:1793-1799` | \_ratio_at array branch in compute_government_budget | `G_over_Y=defense_over_Y=other_net_over_Y=` | all production callers pass scalars (run_fiscal_figures 201-205 via \_apply_shock \_rbase -\> arrays of length T; NOTE: \_rbase returns arrays, so the array branch IS live for fiscal runs; the scalar branch is live for diag_ss_vs_transition/validate_backends) | both branches reached; listed to record the check (not dead) |
| 2 | `eval_fiscal_results.py:723, 729-734` | scenario_keys omits 'nfa_constrained' | `scenario_keys` | run_fiscal_figures writes four scenario blocks per shock (512-516); eval checks three; regen reads all four (209-212) | not dead code: an output block no check reads |
| 3 | `calibrate.py:1655-1656` | generate_report prints external_params verbatim | `config_data\``[``.external_params.\``]` | prints tau_p=0.34, pension_replacement_default=0.18, m_good=0.048 although \_derived.theta replaced them at build time (1314-1329) | report shows values the run did not use |

### Definitions flagged test-only by count but not tabulated

The mechanical pass flagged four definitions test-only by count that the location map did not attach to a §3 row directly: `fiscal_experiments.py:191 linear_phase_in`, `:201 back_loaded`, `:207 exponential_convergence` (all inside the (b) row for the adjustment profiles) and `olg_transition.py:856 OLGTransition.factor_prices` (inside the (b) row with `production_function`). They are confirmed (b). The `ref_sites` column of `B6_inventory.csv` lists each call site.

## Not settled by reading

1.  **Skill-level callers outside `code/`.** `eval_fiscal_results.py` and `regen_fiscal_figures_from_json.py` are invoked by no `.sh` driver in `code/`; whether the user's `/eval-fiscal` and `/fiscal-note` skills call them could not be checked because `~/.claude` was out of scope for this audit. They are classified as manual CLIs, not (f).

2.  **`run_cost_and_figure.sh` → `TestCohortBatchedSurvival`.** The driver runs a test class; its reach into the JAX batched solve was taken from the test file (test_olg_transition.py:2666-2748) without executing it.

3.  **Tests that construct `transfer_floor>0`** (test_olg_transition.py:1160, 1410, 1436, 2368) would now hit the `NotImplementedError` at lifecycle_perfect_foresight.py:301. Whether they currently fail or are marked expected-fail was not checked (no test run was made).

4.  **`jnp.where` eager evaluation.** Under JAX the `tax_progressive` and `transfer_floor` alternatives are traced even when false (lifecycle_jax.py:156-160, 173-177, 205-207); they are classified by their effect on the result, not by whether XLA executes them.

5.  **Reachability through `getattr` strings.** `_as_alpha_indexed` (olg_transition.py:575) and the `getattr(self.lifecycle_config, 'tau_beq'/'transfer_floor', …)` reads were traced by hand; no other dynamic attribute access on user-defined names was found, but the `ast` pass cannot prove absence inside f-strings.

6.  **`_ratio_at` branches** (olg_transition.py:1793-1799) are listed under (c) only to record that both branches were checked; they are live.

7.  **Whether `cohort_sizes` (olg_transition.py:236) is used by any test through `OLGTransition` without demography** was confirmed for test_olg_transition.py:2438 only; other fixtures without a demography also take the 971 fallback at run time, which is why the item is (d) for production rather than (a).

Files: report `B6_dead_code.md`; raw+classified inventory `B6_inventory.csv`; tools `b6_tools/inventory.py`, `b6_tools/classify.py`, reference dump `b6_tools/refs.json` — all under `scratchpad/phaseB/`.

# Checks run and their outcomes

All local, on an 8-core Apple CPU with 24 GB, JAX 0.8.0 on CPU in float64, the `jax-arm` virtual environment, on 2026-10-01 between 22:50 and 23:30 local time, plus the overnight run described below. Scripts and logs are in the session scratchpad, not in the repository.

| Check | Outcome |
|:---|:---|
| C0, cost of the cohort-by-cohort construction | One education group, production grid ($`J = 60`$, $`n_a = 100`$, $`n_y = 5`$, $`n_\alpha = 5`$), JAX, $`n_{sim} = 200`$: `base_year_cross_section` 956 s, 15.9 s per cohort; a single solve 11.5 s (no compile reuse across instances), ratio about 83. Scaling: one SMM evaluation on this construction is three groups, about 48 min on this CPU; the A100 did a three-group single-lifecycle evaluation in about 1.3 s against 35 s here (about 30 times faster), so one cohort-by-cohort cross-section is about 3 min of A100 time, one SMM evaluation about 3 min, and the 169-iteration round of 12:31 would be of the order of 10 hours; C2 is tens of GPU hours. C1 is about 10 min of cross-sections plus two transitions, which the 13:41 chain completed within about an hour. |
| Spot check 1, living share | `_alive_fraction(t)` equals the demography file’s living over ever-entered to $`2\times 10^{-16}`$ at $`t \in \{0, 1, 10, 30, 60, 100, 150, 179\}`$; the weights times cumulative survival sum to 1 at each (true by construction, as the test file notes). <span class="smallcaps">ok</span> |
| Spot check 2, $`\Gamma_t`$ | `growth_factors(180)[t]` $`= (1+g)\,N_{t+1}/N_t`$ exactly (forward convention); $`\Gamma_t = 1.017`$ for all $`t \ge 156`$; the model’s population from entering cohorts and the code’s cohort survival equals the demography file’s population to $`2\times 10^{-16}`$ at six dates. <span class="smallcaps">ok</span> |
| Spot check 3, survival schedules | `_cohort_survival_schedule` equals the demography file’s $`p(\text{year}, j)`$ exactly for cohorts born 1964, 1993, 2023, 2073, 2143, and the historical table where it covers; the table is constant after 2100 and equals the EUROPOP 2100 row. Probability of reaching 84: 0.514, 0.633, 0.759, 0.840, 0.842. <span class="smallcaps">ok</span> |
| Spot check 4, the two constructions agree under one schedule | Covered by `TestBaseYearCrossSection::test_shared_schedule_reproduces_the_single_solve_exactly`, passed. <span class="smallcaps">ok</span> |
| pytest selection (26 tests, serial) | 23 passed, 1 skipped (cost ratio of cohorts solved together, GPU only), 2 failed: the two household-isomorphism cases fail at fixture construction because the fixture sets `transfer_floor = 0.05` and the constructor now raises; stale fixture, not a model failure. 320 s, of which 296 s in `test_kg_flat_at_stationary_investment` (NumPy transition). |
| Isomorphism with the floor at zero | Same fixture otherwise, both implementations: `a_policy` identical; $`\max|\Delta c| \le 8.9\times 10^{-15}`$, $`\max|\Delta \ell| = 2.2\times 10^{-16}`$, $`\max|\Delta V| \le 5.3\times 10^{-15}`$. The growth problem on grid $`G`$ at $`r`$ equals the no-growth problem on $`(1+g)G`$ at $`\tilde{r} = 0.017801`$. <span class="smallcaps">ok</span> |
| `check_a0_predetermination.py` | Small test economy ($`J = 20`$, $`n_a = 30`$, $`n_{sim} = 50`$, $`T_{tr} = 10`$, $`n_\alpha = 3`$), $`\tau^l`$ and $`I_g`$ shocks, both implementations: $`|A_0^{cf} - A_0^{base}| = 0`$ in all four cases; values identical to the 13:41 record. Says nothing about the production economy. <span class="smallcaps">ok</span> |
| Entering-cohort table bound (C2) | `_entrant_weights(187)` returns; `(188)` and `(199)` raise; `growth_factors(200)` clips. <span class="smallcaps">verified</span> |
| Dead-code search repeat | Five unreachable definitions reproduced (definition only); `compare_scenarios` is called with explicit variables at every site. <span class="smallcaps">verified</span> |
| C1 (GPU) | Not run: no instance in the session. |
| C1b (GPU) | Not run. |
| C2 (GPU) | Not run; sized by C0 at tens of GPU hours. |
| Reduced-scale substitute for C1 | Cohort-by-cohort cross-section and one baseline transition at $`n_{sim} = 200`$, JAX on CPU, production configuration and $`\theta`$ (`c1_local.py`): completed, 2 866 s + 10 969 s; results in the note below the table. |
| `check_a0_predetermination.py` after the 2026-10-02 change | Re-run after the bequest edits of §<a href="#sec:postaudit" data-reference-type="ref" data-reference="sec:postaudit">3.8</a>: all four cases $`|A_0^{cf} - A_0^{base}| = 0`$, values identical to the pre-change run to ten decimals (the harness has $`\tau^{beq} = 0`$, and the measure does not enter wealth); the vmapped simulation with the new positional argument runs on both implementations. <span class="smallcaps">ok</span> |

**Reduced-scale substitute for C1, completed 2026-10-02 (started 23:14 on 2026-10-01).** Production configuration and the stored $`\theta`$, JAX on CPU, $`n_{sim} = 200`$ per cohort on both sides; the cohort-by-cohort cross-section took 2 866 s and the baseline transition (180 periods, 239 cohort solves per education group) 10 969 s. Results, one draw each, with different seeds on the two sides:

<div class="center">

|                       | Base-year equilibrium | Transition $`t = 0`$ | % gap |
|:----------------------|----------------------:|---------------------:|------:|
| Output $`Y`$          |                1.0075 |               1.0198 | +1.22 |
| $`A/Y`$               |                4.2534 |               4.2654 | +0.28 |
| $`C/Y`$               |                0.5037 |               0.5039 | +0.04 |
| Revenue$`/Y`$         |                0.3295 |               0.3297 | +0.06 |
| Payroll revenue$`/Y`$ |                0.1300 |               0.1302 | +0.15 |
| Pensions$`/Y`$        |                0.1588 |               0.1574 | -0.88 |
| UI$`/Y`$              |                0.0112 |               0.0106 |  -5.4 |
| Public health$`/Y`$   |                0.0533 |               0.0529 | -0.75 |
| $`K/Y`$               |                3.6667 |               3.6667 |     0 |

</div>

The base-year moments at the stored $`\theta`$ on the live configuration are hours 0.4167 (target 0.41, $`+1.6\%`$), $`A/Y`$ 4.253 ($`+6.3\%`$), payroll revenue$`/Y`$ 0.1300, pensions$`/Y`$ 0.1588 ($`-0.8\%`$), public health$`/Y`$ 0.0534 ($`-1.2\%`$), UI$`/Y`$ 0.0112 (target 0.006). At 200 households per cohort $`A/Y`$ alone can move by several percent between seeds, so these bound the drift of C3 loosely; the direction of hours and $`A/Y`$ is up. The $`t = 0`$ output gap of $`+1.2\%`$ compares with $`-9.5\%`$ in the 13:41 record on the single-lifecycle construction; at this $`n_{sim}`$ it is within sampling error of zero and was not tested against a second seed.

Baseline path (per capita, detrended): $`Y`$ falls from 1.020 at $`t = 0`$ to 0.823 at $`t = 26`$ ($`-19\%`$), recovers to 0.97 by $`t = 75`$, oscillates between 0.93 and 0.99 and settles at 0.96; pensions$`/Y`$ rise from 0.157 to 0.286 at $`t = 25`$ and settle at 0.198; public health$`/Y`$ from 0.053 to 0.077 and back to 0.060; $`A/Y`$ from 4.27 to 6.05 at $`t = 26`$, settling at 5.35–5.44; revenue$`/Y`$ from 0.330 to 0.390 and back to 0.359; $`K^g`$ exactly constant. The budget of this run carries no $`G`$, $`D`$ or $`O`$ (the report’s baseline passes no ratios); adding the configuration shares ($`0.13 + 0.03 - 0.1136 = 0.0464`$ of output) gives a full primary balance of $`+2.45\%`$ of output at $`t = 0`$ (target $`+1.95\%`$), a deficit of $`7.3\%`$ at $`t = 25`$, and a deficit of about $`0.06\%`$ on the terminal path, where a constant $`B/Y`$ needs a surplus of $`0.197\%`$ per unit of $`B/Y`$ (M4).

Goods-market residual as the report computes it: closed form $`-0.173`$ of output at $`t = 0`$, open form $`-0.197`$, closed form $`-0.093`$ at $`t = 179`$; both far beyond the 0.08 threshold. The identity on the flows as booked accounts for it: in this budget $`\mathrm{PD}/Y = -0.0705`$, $`M/Y = 0.0806`$, $`\mathrm{UI}/Y = 0.0106`$, so the known terms sum to $`-0.162`$, leaving $`-0.035`$ of output for the unmeasured accidental bequests, the order of the 3.8% recorded in July (M3).

Flatness over $`t = 156, \dots, 179`$, in percent per year: $`Y`$ $`+0.010`$, $`C`$ $`-0.004`$, $`K`$ $`+0.010`$, $`L`$ $`+0.010`$, $`K^g`$ $`0.000`$, household wealth $`A`$ $`-0.067`$. The sampling-noise floor at this $`n_{sim}`$ was not measured. Log and paths: session scratchpad `c1/c1_local.log`, `c1/c1_local_paths.npz`.
