# Fiscal Scenarios — Assumptions

Notation follows the paper draft: the model in `DSA-LSA model.tex` and the
experiments in `DSA-LSA experiments.tex`. Two permanent shocks — government
consumption $G_t$ and public investment $I_t^g$ — are each studied under two
financing regimes (debt and labour tax) against a no-shock baseline.

## Environment

**Production and prices.** Output and the public-capital law of motion are
$$Y_t = A\,(K_t^g)^{\eta_g}\,K_t^{\alpha}\,L_t^{1-\alpha},
\qquad K_{t+1}^g = (1-\delta_g)\,K_t^g + I_t^g.$$
The interest rate $r_t$ is exogenous, so the firm's first-order conditions pin
the capital–labour ratio and the wage:
$$\frac{K_t}{L_t}=\Big(\tfrac{\alpha A (K_t^g)^{\eta_g}}{r_t+\delta}\Big)^{1/(1-\alpha)},
\qquad w_t=(1-\alpha)A(K_t^g)^{\eta_g}\Big(\tfrac{K_t}{L_t}\Big)^{\alpha}.$$
Any policy that leaves $r_t$, $K_t^g$, and the tax rates $(\tau^c,\tau^l,\tau^p,\tau^k)$
unchanged leaves $w_t$ and every household price unchanged, hence leaves
$Y_t,\,C_t,\,L_t$ unchanged.

**Government primary balance.** Extending equation (primary-deficit) of the
paper with a defence line $D_t$ and a net residual line $O_t$,
$$\text{PD}_t = \text{UI}_t+\text{PENS}_t^{\text{out}}+\text{HSub}_t
+G_t+I_t^g+D_t+O_t
-\big(\text{Rev}_t^c+\text{Rev}_t^l+\text{Rev}_t^p+\text{Rev}_t^k+\text{BeqTax}_t\big).$$
$\text{UI}_t,\text{PENS}_t^{\text{out}},\text{HSub}_t$ and the revenues
$\text{Rev}_t^{c,l,p,k}$ are determined by household behaviour and demographics.
The discretionary lines are fixed shares of output,
$$G_t=\gamma Y_t,\quad I_t^g=\iota Y_t,\quad D_t=\delta_D Y_t,\quad O_t=\omega Y_t,$$
with $\gamma=0.13,\ \iota=0.03,\ \delta_D=0.03$ constant. The residual share
$\omega$ is fixed at the initial steady state so that the initial primary
balance matches the data target, $\text{PD}_0/Y_0=-\bar s$ with $\bar s=0.0195$
(a surplus of $1.95\%$ of output); this gives $\omega=-0.091$.

**Debt and external account.** Sovereign debt and net foreign assets evolve as
$$B_{t+1}=(1+r_B)\,B_t+\text{PD}_t,\qquad
B_0=b^\ast Y_0,\quad b^\ast=1.64,\qquad
\text{NFA}_t=A_t-K_t-B_t,$$
where $r_B$ is the (exogenous, constant) sovereign rate and $A_t=\int a\,d\mu_t$
is household wealth. Net foreign assets close the resource constraint
$C_t=Y_t-I_t-G_t-I_t^g-\Delta\text{NFA}_t$.

**Across scenarios.** Demographics (survival $\pi(j)$, growth $g_N$) are
exogenous and time-varying, so the baseline is itself a transition, not a
stationary point. Agents have perfect foresight over all paths, and the initial
distribution $\mu_0$ is predetermined: households enter the transition with the
same wealth in every scenario, so scenarios differ only from $t=0$ onward.

## Shocks

Both shocks are permanent and equal to $2\%$ of contemporaneous output,
$\Delta_t=0.02\,Y_t$:
$$G_t^{cf}=G_t^{base}+\Delta_t,\qquad I_t^{g,cf}=I_t^{g,base}+\Delta_t.$$
Government consumption is wasteful (it enters neither $u$ nor $Y_t$). Public
investment acts only through $K_t^g$, hence only when $\eta_g>0$.

## Financing regimes

**Debt-financed.** Tax rates are held at baseline; the extra spending is
absorbed by debt through the law of motion above. Only $B_t/Y_t$ and
$\text{NFA}_t$ move.

**Labour-tax-financed.** A constant $\Delta\tau^l$ is added to the labour-tax
path, $\tau^l_t=\tau^l+\Delta\tau^l$, chosen so that the counterfactual's
terminal debt-to-GDP ratio equals the **baseline transition's** terminal ratio:
$$\frac{B^{cf}_{T_{trans}}}{Y^{cf}_{T_{trans}-1}}
 =\frac{B^{base}_{T_{trans}}}{Y^{base}_{T_{trans}-1}},
\qquad T_{trans}=60.$$
The baseline is debt-financed, so its terminal ratio is itself endogenous (debt
drifts along the no-shock path); the labour tax is raised just enough that the
shock leaves terminal debt/GDP where the baseline leaves it. Raising $\tau^l$
lowers the after-tax wage $(1-\tau^l-\tau^p)\,y^L$, changing labour supply and
output; $\text{PENS}_t^{\text{out}}$ and $\text{UI}_t$ respond through $\mu_t$.

## Fiscal multiplier

For the debt-financed scenarios,
$$\mathcal{M}_t=\frac{Y_t^{cf}-Y_t^{base}}{\Delta_t}.$$

## The five scenarios

| Scenario | Shock | Financing | What it isolates |
|---|---|---|---|
| Baseline | none | debt residual | reference path; $B_t$ evolves from the calibrated $\text{PD}_t$ |
| $G$, debt | $\Delta_t=0.02Y_t$ | debt residual | $r_t,w_t,\tau$ unchanged and $G_t$ wasteful $\Rightarrow Y_t,C_t,L_t$ unchanged; only $B_t/Y_t$, $\text{NFA}_t$ move; $\mathcal{M}_t\approx 0$ |
| $G$, labour tax | $\Delta_t=0.02Y_t$ | $\Delta\tau^l$, terminal $B/Y$ = baseline | permanent $\Delta\tau^l$ funding a $2\%$-of-$Y$ rise in $G_t$ so terminal debt/GDP matches the no-shock path; output responds via $w_t(1-\tau^l)$ |
| $I^g$, debt | $\Delta_t=0.02Y_t$ | debt residual | raises $K_t^g$; with $\eta_g=0$ no output effect, mirroring $G$-debt |
| $I^g$, labour tax | $\Delta_t=0.02Y_t$ | $\Delta\tau^l$, terminal $B/Y$ = baseline | same balance condition as $G$; $\Delta\tau^l$ smaller than the $G$ case when $\eta_g>0$ (expanding tax base) |

The terminal debt/GDP comparison is evaluated at $T_{trans}$; results extend
$n_{post}=20$ periods beyond it to show post-target dynamics.

**Current Greek calibration.** $\eta_g=0$ and $K_0^g=0$, so the two
public-investment scenarios carry no output channel: the debt-financed one
coincides with $G$-debt, and the labour-tax one differs from the $G$ case only
through the $2\%$-of-$Y$ accounting line. They become distinct exercises once
public capital is productive ($\eta_g>0$).

## Relation to the current paper draft (`DSA-LSA experiments.tex`)

The implemented exercise differs from the experiments section as currently
written, in ways the draft should be reconciled with:

- **Shock size.** Here $\Delta_t=0.02\,Y_t$ (a share of contemporaneous output);
  the draft writes $\Delta=0.02\,\bar Y$ with $\bar Y$ mean baseline output (a
  constant level). The spending lines $G_t,I_t^g,D_t,O_t$ are likewise shares of
  $Y_t$, not levels.
- **Debt interest.** The debt law uses a separate sovereign rate $r_B$; the
  draft's equation (debt) uses the capital return $r_t$.
- **Balance condition.** The labour-tax scenarios target the **baseline
  transition's terminal debt/GDP** (a level match, $B^{cf}_{T}/Y^{cf}_{T-1}=
  B^{base}_{T}/Y^{base}_{T-1}$); the draft's equation (balance-condition)
  instead targets the flow rest point $\text{PD}_{T-1}/Y_{T-1}=(g-r)b^\ast$ with
  a fixed $b^\ast$.
- **Primary-deficit lines.** Defence $D_t$ and the residual closure $O_t$ are
  included here but absent from equation (primary-deficit) in the draft. $O_t$
  is pinned at the initial steady state to hit $\text{PD}_0/Y_0=-\bar s$.
- **Calibration constants.** $b^\ast=1.64$, $T_{trans}=60$, $\gamma=0.13$ for the
  Greek calibration; the draft's worked example uses the test configuration
  ($B_0=0\Rightarrow b^\ast=0$, $G^{base}=0.30\,\bar Y$, $T_{trans}=40$).
