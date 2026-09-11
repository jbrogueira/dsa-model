# Literature Review: Sovereign debt sustainability with non-debt government liabilities in heterogeneous-agent OLG small open economies

*Generated 2026-09-10 | Depth: deep | For: the DSA–LSA paper (Brogueira de Sousa, Marimon, Slawinska, Zavalloni; draft of 30 July 2026)*
*Roughly 280 papers and documents screened by five search agents (119 + 100 + 41 + 23 entries plus 25 official documents), plus a Semantic Scholar repeat pass on 11 September that added 55 verified entries; 21 read in full text, 6 further PDFs read in targeted sections; see "Method and limits" and the Addendum.*

The review tests five candidate contributions (H1–H5, defined in `PLAN_dsa-lsa-contributions.md`) against the literature. For each it returns a verdict, the nearest existing work, the differentiator that survives, and the model extension it requires. The choice among candidates is the authors'; this document supplies the evidence.

---

## Executive summary

Debt sustainability analysis in practice (European Commission S1/S2, IMF SRDSF, ECB, ESM) is debt accounting around an exogenous primary balance, with ageing costs imported from partial-equilibrium projections; its academic basis (Bohn 1998; Ghosh et al. 2013; D'Erasmo–Mendoza–Zhang 2016; Debrun et al. 2019) is debt-only by construction. The quantitative OLG literature that jointly carries pensions, health and debt and computes a required fiscal adjustment is a Japan cluster (Braun–Joines 2015; Kitao 2015; İmrohoroğlu–Kitao–Yamada 2016, 2019; McGrattan–Miyachi–Peralta-Alva 2018), all closed-economy or exogenous-price accounting models. The one OLG paper calibrated to Greece as a small open economy (Glomm–Jung–Tran 2018) studies debt reduction with public capital, pensions and public employment but has stable demographics, no health or UI line and no liability accounting. No paper found builds a heterogeneous-agent OLG small open economy for a European country that carries debt, pensions, health, UI and public capital jointly and solves for the adjustment that satisfies a constraint on all liabilities (H1 survives). No paper reads realised Greek health or pension cuts against a structural counterfactual as a haircut on promised liabilities (H2 survives; the current health decomposition is arithmetic and belongs to the same class as the Ageing Report's, so the surviving content is the pension-side re-solve). The defence-financing question (H3) has become crowded in 2025–2026: a CEPR paper studies the NATO-target permanent increase in a heterogeneous-agent OLG with a PAYG system, and the ESM chapter co-authored by Zavalloni compares the same three financing options; what remains is generational and liability-by-liability incidence in a high-debt SOE, which requires a welfare module the code does not have. The r_B < r wedge (H4) is standard in the Japan literature and has two market theories (convenience yield; bubble premium); no quantitative OLG sustainability analysis sources it in official lending, but the draft's front matter does not yet frame it and the modelling choice on who earns the wedge is open. The pension leg of the liability ledger (H5) already exists officially (Eurostat accrued-to-date entitlements: Greece 403 % of GDP in 2021); no Greek public sector balance sheet with a health liability exists.

## Contribution verdict table

| Candidate | Verdict | Nearest work | Differentiator that survives | Model extension required | Main referee objection |
|---|---|---|---|---|---|
| **H1** Structural LSA: primary balance and each liability line as GE outcomes; adjustment satisfying a constraint on all liabilities | **Not pre-empted.** | Kitao 2015; Braun–Joines 2015; McGrattan et al. 2018 (Japan, closed GE OLG, pensions + health + debt, required tax); İKY 2019 (accounting, all lines); Glomm–Jung–Tran 2018 (Greece, SOE OLG, public capital + pensions, debt reduction); Castro et al. 2016 (Portugal, Blanchard–Yaari NK DSGE, ageing shocks under a fiscal rule); Bettendorf et al. 2011 (Netherlands, representative-cohort GA model); EC S2 (four ageing items, partial equilibrium) | Joint HA-OLG SOE with incomplete markets carrying debt + pensions + health + UI + public capital + NFA, calibrated to Greece; each budget line reported as an equilibrium outcome; the GE correction to S2-type indicators is computable | Activate the ageing transition (fertility and survival paths are implemented, not used); state the sustainability criterion explicitly (terminal B/Y or NFA/Y, or a fiscal limit) | "With exogenous r and no uncertainty fiscal space is infinite" (Blanchard 2023 via Corneo); "European labour taxes sit near the Laffer peak" (D'Erasmo–Mendoza–Zhang); "Glomm–Jung–Tran already did Greece" |
| **H2** Model-based measurement of partial default on non-debt liabilities (look-back) | **Not pre-empted; current evidence thin.** | Perotti 2021 (Greek health cuts by component, accounting); Leventi–Matsaganis 2020 (NPV of pension entitlements removed 2010–13, by cohort, micro); OECD 2013 / Medeiros–Schwierz 2013 / Ageing Report growth decompositions (demography, income, residual; no coverage term, no decline); D'Erasmo–Mendoza 2016, 2021 (default on domestic bonds); Novy-Marx–Rauh 2011 (pension promises as defaultable debt); Evans–Kotlikoff–Phillips 2012 ("forced to default on its promised payment") | Reading an observed expenditure path against a structural counterfactual under stable rules and observed demographics; coverage as an explicit factor; the same identity applied to endogenous lines via re-solves | The pension-side counterfactual re-solves (draft §5.4, not run); a present-value haircut measure anchored to Eurostat Table 29 (403 % of GDP) | "The health split is arithmetic and the Ageing Report does the same three-way split"; "Perotti already documented the cuts" |
| **H3** Financing a permanent G or I_g increase in a high-debt SOE: which liabilities and generations pay | **Partly pre-empted (2025–2026).** | Boullot–Cahn–Challe–Matheron 2026 (HA-OLG, PAYG + tax-transfer, NATO-target permanent increase, financing alternatives; country and cohort reporting ⚠️ unverified); Lauwers–Michou–Ricci–Zavalloni 2026 (ESM, two-sector DSGE, reallocation / labour tax / debt with 20-year stabilisation, self-financing 25–53 cents); Bokan et al. 2025 (ECB, RA and HANK, labour-tax financing lowers multipliers); Kudrna–Tran 2018 and Glomm–Jung–Tran 2018 (SOE OLG consolidation, welfare by generation); Braun–Joines 2015 (GA + compensating variation by cohort across financing paths) | Defence-type G versus public investment in a SOE with NFA absorption; incidence on each liability line (pension and UI shares); the one-generation absorption rule; Greek calibration; r_B wedge | Welfare module: the code stores no value functions or welfare measures (grep of the solvers and fiscal code, 2026-09-10), so cohort compensating variations need a solver-side extension | "Zero multiplier of debt-financed G at a fixed world rate is textbook"; "Boehm 2020 finds the opposite ordering of G and I_g multipliers"; "the ESM already did the arithmetic" |
| **H4** Sustainability with r_B ≤ g < r, wedge sourced in official lending | **Not pre-empted; thin as stated.** | Reis 2021 (r < g < m, bubble premium); Jiang et al. 2024, 2025 and Brunnermeier–Merkel–Sannikov 2024 (convenience yields); Bellon–Gnewuch–Orlandi–Zavalloni 2026 (safe-asset externality, euro area); Corsetti–Erce–Uy 2018, 2020 and Mimir–Önder 2025 (official lending in default models); the Japan cluster (r_B = 1 % vs 3–4 %, sourced in JGB yields); Hansen–İmrohoroğlu 2023 (central bank as below-market holder) | An official-lender wedge inside an OLG sustainability analysis; the experiment "LSA at the ESM rate versus the market rate" for Greece | Settle who earns the wedge on household holdings of B (open in `TREND_GROWTH_PLAN.md`); add the framing to abstract and introduction, which do not mention it | Blanchard's infinite-fiscal-space point; Mauro–Zhou (defaults follow negative r − g); Brumm et al. 2024 and Cao 2024 (low r does not license debt in OLG) |
| **H5** Model-consistent PSBS liability ledger for Greece | **Partly pre-empted; nothing delivered in the draft.** | Eurostat Table 29 (Greece 331 / 374 / 403 % of GDP, 2015 / 2018 / 2021); Deboeck–Eckerfelt 2020; Arévalo et al. 2019 (EC generational accounts incl. Greece); Koshima et al. 2021 (intertemporal PSBS, G7); IMF PSBS (Greece general-government tier only; social-security pensions excluded by construction); Kaier–Müller 2015; Heer–Polito–Wickens 2023 ("pension space") | Present values of every line under the model's equilibrium paths, including health and UI, reconciled to Table 29 | Substantial data work; none exists | Debrun et al. 2019: "concerns about net worth do not cause sovereign crises or defaults, unmanageable gross debt dynamics does" |

**Ranked reading of the evidence** (criteria: distance to the nearest neighbour; what has to be built; fit with the sovereign-debt-analysis literature the authors target).

1. **H1** has the largest distance to any neighbour and needs the least new code. Its exposure is conceptual: the sustainability criterion must be stated against Blanchard's infinite-fiscal-space point and against Glomm–Jung–Tran's Greek SOE precedent.
2. **H2** is conceptually distinct from everything found; the "partial default on a promised liability" reading has only abstract precedents. Its current empirical content (health) is arithmetic; the pension-side re-solve is what would separate it from Perotti 2021 and the Ageing Report.
3. **H4** is the candidate most specific to sovereign debt analysis. It is a modelling assumption today, not a result. The experiment that turns it into a result is the comparison of liability sustainability at the ESM rate and at the market rate, which the framework can run once the wedge's incidence is settled.
4. **H3** has the most recent and most direct neighbours, one of them by a coauthor. Its surviving content is generational and liability-line incidence, which requires the welfare module.
5. **H5** is infrastructure. Its pension leg exists officially; its value lies in the health and UI legs and in model consistency, and the draft has none of it yet.

---

## 1. The core question

Debt sustainability analysis asks whether a projected primary-balance path stabilises debt. The primary balance is an input. Ageing costs enter, when they enter, as exogenous projections: the Commission's S2 indicator is "the permanent adjustment of the structural primary balance (SPB) in 2027 that would stabilise public debt in the long term", equal to the initial budgetary position plus "ageing costs, comprising the projected change in public spending on pensions, healthcare, long-term care and education as provided by the 2024 Ageing Report" (Debt Sustainability Monitor 2025, IP 332, ch. 3). The Ageing Report itself "is based on a simple accounting framework which ignores general equilibrium behavioural reactions" (Baksa–Munkacsi–Nerlich 2020). For Greece the S2 indicator is −0.4 % of GDP (DSM 2025; −0.8 in DSM 2024), with pensions contributing −0.6 pp and health +0.7 pp; long-term risk is classified low while medium-term risk is high because of the debt level.

The paper under review asks a different question: whether the set of promised liabilities, debt included, is jointly sustainable when every line is an equilibrium outcome of household saving and labour supply, demographics and policy rules in a small open economy. Two literatures already contain parts of that question. Generational accounting (Auerbach–Gokhale–Kotlikoff 1991, 1994; Kotlikoff 2002; Green–Kotlikoff 2006) treats debt and transfer promises symmetrically and states that "failure to satisfy this constraint means that the government will default on its liabilities", but "does not incorporate general equilibrium feedback effects" (Auerbach–Gokhale–Kotlikoff 1994). The quantitative OLG literature on ageing supplies the feedback but, outside Japan, has carried one liability at a time.

## 2. Foundational framework

**Sustainability definitions.** Bohn (1998) shows that a positive response of the primary surplus to debt suffices for the intertemporal budget constraint; Bohn (2007) shows that stationarity or cointegration tests are not needed. Ghosh et al. (2013) add "fiscal fatigue": a debt limit where the primary-balance response falls below r − g. Debrun, Ostry, Willems and Wyplosz (2019) separate solvency ("a pure prediction" with "no operational meaning") from sustainability, adopt the IMF (2002) definition, and give the fiscal-fatigue debt limit d** = pb̄/Γ*, "literally the edge of a cliff". D'Erasmo, Mendoza and Zhang (2016) define the structural approach: "the initial debt that is sustainable is the one determined by the present value of primary balances evaluated using equilibrium allocations and prices"; in their two-country neoclassical model European labour taxes are "near the peak of the dynamic Laffer curve" and "cannot restore fiscal solvency" after a 20 pp debt increase. Collard, Habib and Rochet (2015) compute maximum sustainable debt for 23 OECD countries from the primary-surplus capacity and the growth distribution. All of these are debt-only.

**Operational criteria in OLG models.** The Japan papers use four different criteria: a transversality condition with terminal debt/GDP fixed at 1.0 and one tax adjusting before the terminal date (Braun–Joines 2015, Definition 2); a trigger-and-retire rule that starts retiring debt at 250 % of output and targets 60 % (Hansen–İmrohoroğlu 2016); a period-by-period balanced budget with debt held at 100 % of GDP (Kitao 2015); and no criterion at all, reading sustainability off the debt path (İKY 2016, 2019). The draft's criterion, a terminal debt-to-GDP or NFA-to-GDP ratio equal to the baseline's at T = 60, belongs to the first family and should be stated as such.

**Liabilities as a single class.** Buiter (1983, 1985) built the comprehensive public-sector balance sheet with the present value of social-insurance commitments and defined the permanent primary-deficit gap; Blejer–Cheasty (1991) surveyed deficit concepts including implicit social-security liabilities; Bohn (1992) constructed US government balance sheets and income statements for 1947–1989. Kotlikoff (2002) states that "official government debt is, as a matter of neoclassical economic theory, an artifice of fiscal taxonomy". This is the theoretical basis for treating pension and health promises and bonds as one liability class, and it is old. Debrun et al. (2019) reply that "concerns about net worth do not cause sovereign crises or defaults, unmanageable gross debt dynamics does" and that the Commission's S2 "is simply the wedge in the intertemporal budget constraint".

**OLG lineage.** Auerbach–Kotlikoff (1987), Storesletten (2000, "Sustaining fiscal policy through immigration"), De Nardi–İmrohoroğlu–Sargent (1999), Conesa–Krueger (1999), Krueger–Ludwig (2007), and the multi-country ageing models of Börsch-Supan–Ludwig–Winter (2006) and Attanasio–Kitao–Violante (2007) are the standing references; Auclert–Malmberg–Martenet–Rognlie (2021) is the current multi-country OLG on demographics, wealth and r*. The authors' own OLG engine is Díaz-Saavedra–Marimon–Brogueira de Sousa (2023 JEEA; 2022 SERIEs), which reports the open-versus-closed difference at stake in the SOE choice (welfare gain of the Backpack 0.96 % of consumption open, 16.14 % closed).

## 3. Evidence by strand

### 3.1 Debt sustainability practice and its academic basis (serves H1, H4)

Official practice is uniform in structure. The Commission's baseline is a "no-fiscal-policy-change" projection; behavioural content appears only as a 0.6 fiscal multiplier in alternative scenarios (DSM 2024, 2025). The IMF's SRDSF (2022) attaches an optional demographics module that projects pension and health financing needs to 2050 by mechanical age profiles on top of WEO baselines. The ECB's framework (Bouabdallah et al. 2017) is a ten-year accounting DSA with ageing costs as a narrative scenario and a contingent-liability indicator, and it excludes debt held by official creditors from the maturity-convergence assumption. The ESM's DSA is gross-financing-needs based (Gabriele–Erce–Athanasopoulou–Rojas 2017: at high debt "an increase of one percentage point in GFN translates into an increase in the spread of 10 basis points"); its stochastic version tilts BVAR fan charts to a deterministic baseline (Hitaj–Zavalloni–Zigraiova 2025); its tool inventory lists DSGE models and no OLG model. The ESM's own statements on Greece record the wedge H4 uses: EFSF loans "will continue to be financed for the next 30 years at very low interest rates", weighted average maturity 32.5 years after the 2018 measures.

The academic frontier of DSA moved from accounting to probabilistic statements (Celasun–Debrun–Ostry 2006; Zenios et al. 2021; Willems–Zettelmeyer 2022) and to the r − g debate (Section 3.6). Eichengreen–Panizza (2016) document how rarely primary surpluses of 5 % of GDP are sustained for a decade, a constraint on any labour-tax adjustment the model computes. For Greece specifically, Zettelmeyer–Kreplin–Panizza (2017) run an accounting DSA under alternative official-rate assumptions, and Debrun et al. (2019, Box 2) record that the 2010 programme projected a debt peak of 149 % of GDP in 2013 while the ratio "quickly shot up to about 180 percent", with growth 3.4 pp/yr and primary balances 3.2 pp/yr below programme over 2010–2017.

### 3.2 Quantitative OLG fiscal sustainability with ageing (serves H1, H3)

**The Japan cluster** is the literature the framework belongs to.

- İmrohoroğlu–Kitao–Yamada (2016 IER) is a micro-data-based large-scale OLG *accounting* model: "We do not model individual decisions on consumption/saving and labor/leisure choices"; interest rates and wage growth are exogenous constants (3 % on savings, "the average of the return on government bonds, 1%, and the return on private capital, 4%"); health is not a separate line but sits inside age-invariant per-capita purchases, and "public health insurance reform … [is] left for future research". Net debt reaches 477 % of GDP by 2060 on the baseline; "a 35% consumption tax is needed" to hold it near 80 %. The 2019 paper (Journal of the Economics of Ageing) adds health and long-term care as government lines with age-dependent coverage (70–90 %) and finds net debt of 625 % of GDP by 2070 absent reform; a package of retirement age 67, pensions −10 %, copays 20 %, higher female employment and a 15 % consumption tax brings debt to 87 % in 2050.
- Braun–Joines (2015 JEDC) is a closed-economy GE OLG with representative cohorts, pensions under the 2004 rules, exogenous age-dependent medical and LTC costs with government coverage, and a government borrowing rate held 1.24 pp below the return on capital. Sustainability is a transversality condition with terminal debt/GDP of 1.0. The steady state needs a 26.4 % consumption tax; delaying consolidation to 2039 needs 57 %; a constant tax from 2018 needs about 36 %. Welfare is reported by generational accounts and compensating variation by birth cohort, and cohorts 2000–2045 prefer a constant tax to higher copays while later cohorts prefer copays. This is the template for H3's "which generations pay".
- Kitao (2015 JEDC) is a closed GE life-cycle model with idiosyncratic wage risk, pensions, exogenous health and LTC profiles with copays, debt fixed at 100 % of GDP and an exogenous 1 % rate on debt below the endogenous capital return. The consumption tax must rise from 5 % to 19.3 % in the long run and peaks at 47.9 % along the transition; a labour-tax alternative needs +13.5 pp and leaves output 13 % lower.
- Hansen–İmrohoroğlu (2016 RED) is a representative-agent closed economy with bonds in utility (an endogenous below-market bond return); the consumption tax needed is 47–61 %, and the labour tax alone cannot raise the revenue at a Frisch elasticity of 0.5. Hansen–İmrohoroğlu (2023 RED) adds Bank of Japan holdings, the only quantitative sustainability paper found with an institutional below-market holder of part of the debt.
- McGrattan–Miyachi–Peralta-Alva (2018 IMF WP) is a GE OLG "specifically parameterized to match both the macroeconomic and microeconomic level data of Japan" that compares consumption-tax, contribution, debt and copayment financing of retirement, health and LTC costs; it is a closed economy ("Because this is a closed economy, this …", p. 1290 of the extracted text).
- Kitao–Yamada (2026, Japanese Economic Review) survey this literature; İmrohoroğlu–Ino (2025) give an Aiyagari version.

Every paper in the cluster is closed-economy or exogenous-price, none has UI or public capital, none has a liability-accounting object, and the r_B < r wedge is calibrated to observed JGB yields.

**Small-open-economy OLG fiscal adjustment.** Kudrna–Tran (2018 JMacro) compare immediate and gradual deficit elimination in Australia through income-tax hikes, consumption-tax hikes or transfer cuts in a computable OLG with exogenous world rate, with welfare by generation and by rich and poor; Kudrna–Tran–Woodland (2019 Macro Dyn) do the same for ageing-driven gaps. **Glomm–Jung–Tran (2018 Macroeconomic Dynamics; Towson WP 2013-01)** is the Greek precedent: "The benchmark model is calibrated to Greece at the beginning of the 21st century"; "We assume a small open economy. Capital is free to move across borders" at "the fixed world interest rate … and the country specific risk premium"; households differ by skill and age under mortality risk and a borrowing constraint, with stable demographics; the government has public capital in a public-good production function, public-sector employment, separate public- and private-sector pension schemes, and lumps "health care and welfare programs" into residual expenditure. They reduce debt from 105 to 85 % of GDP with consumption or income taxes, public-sector pension cuts or public-investment cuts; output falls up to 6 % in the first ten years and rises 4.3–5 % in the long run; "the current old and middle age generations experience welfare losses while current young workers and future generations are beneficiaries". Differences from the draft: no idiosyncratic income risk found in the text ⚠️, no ageing transition, no health coverage or UI line, a market risk premium rather than an official-lending rate, debt reduction rather than a spending increase, and no liability accounting.

**European OLG sustainability.** Bettendorf et al. (2011 De Economist) use the CPB's GAMMA, a "Generational Accounting Model with Maximizing Agents", an applied GE model with several generations that computes a Dutch sustainability gap from pensions and health with generational accounts; its abstract is silent on idiosyncratic risk and the open-economy assumption ⚠️. Castro–Maria–Félix–Braz (2016 Macro Dyn; BdP WP 4/2013) use PESSOA, an open-economy New Keynesian model with Blanchard–Yaari non-Ricardian households for Portugal in the euro area, feed it Ageing Report shocks to pensions, health and LTC, and close the budget with a fiscal rule on the debt target; they position the paper as adding "to the existing literature on debt sustainability analysis" because the standard accumulation equation "does not capture interdependencies", and they note that "most of the analysis on ageing issues is conducted on fully-fledged overlapping generation models, which lack open economy features". Baksa–Munkacsi–Nerlich (2020 ECB WP 2396) integrate Ageing Report inputs into a Gertler-type OLG for Germany and Slovakia to price pension-reform reversals; Greece is named as facing court-ordered reversals but not modelled. Heer–Polito–Wickens (2020 JEDC) compute the fiscal limit for pensions under ageing in a life-cycle model with distortionary taxes for the US and 14 European countries (Greece's inclusion ⚠️), and their 2023 "pension space" indicator is the closest model-based sustainability metric for a single liability. Díaz-Giménez–Díaz-Saavedra (2025 EER) is the nearest published HA-OLG open-economy sustainability paper for a euro-area country, pensions only. Hougaard Jensen et al. (2021) do the Danish DREAM exercise. Dolls et al. (2017 ITAX) project demographic effects on 27 EU budgets with EUROMOD, Greece included, without debt dynamics. Arévalo–Berti–Caretta–Eckefeldt (2019 EC DP 112) produce generational accounts for all Member States on the Commission's debt-projection model, Greece included, without behavioural response.

No paper in this strand is at once heterogeneous-agent with incomplete markets, a small open economy, European, and carrying debt, pensions, health, UI and public capital jointly with a required-adjustment computation. Four targeted Semantic Scholar searches for Greek OLG fiscal work returned only Glomm–Jung–Tran (via other agents) and Yoshino–Miyamoto–Terada-Hagiwara (2025), a modified Domar-condition paper.

### 3.3 Health in life-cycle macro and health-spending decompositions (serves H2)

The life-cycle literature treats out-of-pocket medical risk as a driver of old-age saving (De Nardi–French–Jones 2010; Kopecky–Koreshkova 2014; De Nardi–French–Jones 2016 survey) and public coverage as insurance with a consumption floor (Braun–Kopecky–Koreshkova 2017, 2019); the fiscal side under ageing appears in Attanasio–Kitao–Violante (2010), Jung–Tran–Chambers (2017 EER) and Conesa et al. (2018). Coverage and copayment changes appear as *prospective* instruments (Hsu–Yamada 2019 SJE; Ihori et al. 2011; Hagiwara 2024; Lim 2020), never as a realised cut measured against a counterfactual. The draft's deterministic age-cost profile omits the precautionary-saving channel of a coverage cut that this literature centres on.

The comparator for the look-back exercise is the health-projection methodology. All four sources read decompose the *growth* of real per-capita public health spending into a demographic effect (population times age profile), an income effect (elasticity 0.7–1.1; 1.5 in the Ageing Report risk scenario) and a residual labelled excess cost growth or non-demographic drivers, projected forward at unchanged policy. Greek numbers: 5.9 % per year over 1995–2009 = 0.6 age + 2.7 income + 2.5 residual at elasticity 0.8 (de la Maisonneuve–Oliveira Martins 2013, Table 1); 2.8 % over 1988–2010 = 0.2 age + 1.3 income − 0.3 price + 1.7 residual (Medeiros–Schwierz 2013, Table 10); the 2024 Ageing Report projects Greek public health spending from 5.4 % of GDP in 2022 to 5.9 % in 2070 (+0.6 pp, demographic contribution +0.3). None of these decomposes an observed decline, none has a coverage term (coverage is folded into the base-year profile), and none uses a Shapley split. The draft's coverage / age-composition / residual split therefore has no direct analogue, but it is arithmetic in exogenous primitives, exactly as these are.

### 3.4 Greek austerity and welfare-state cuts (serves H2, H3)

Structural counterfactuals of Greek austerity treat spending cuts as aggregate instruments. Gourinchas–Philippon–Vayanos (2017) lump government spending and transfers into one item (G = T = 19.1 % of GDP), never mention pensions, and attribute about 50 % of the peak-to-trough output drop to fiscal consolidation; their footnote 25 records that "Greek debt was refinanced by official creditors at low rates … while the secondary market rate was high". House–Proebsting–Tesar (2020 JME) measure austerity as purchases forecast errors with no categories and find a multiplier near 2. Economides–Papageorgiou–Philippopoulos (2021), Dellas–Malliaropulos–Papageorgiou–Vourvachaki (2024 JEDC) and Papageorgiou–Vourvachaki (2017) use Bank of Greece DSGE models with a rich public sector but no cohorts and no liability framing.

The cuts themselves are documented, not modelled. Perotti (2021 Economic Policy) gives the component-level account of Greek public health spending and outcomes and argues the outcome deterioration was smaller than the Lancet literature (Kentikelenis et al. 2014) claimed; Kanavos–Souliotis (2017) and Economou et al. (2014) record the 6 %-of-GDP public health ceiling of the first programme, pharmaceutical clawbacks and coverage gaps for the uninsured; the Country Health Profiles record per-capita spending down 28 % between 2009 and 2015 and a public financing share of 59 % in 2015 and 62.1 % in 2021, the lowest in the EU against 81.1 %, with out-of-pocket payments at 33–35 %. On pensions, Panageas–Tinios (2017) and Symeonidis (2016) give the chronology of more than a dozen cuts, and Leventi–Matsaganis (2020) compute the net present value of accrued entitlements removed from 2008 private-sector retirees by the 2010–13 cuts, by cohort, without general equilibrium. Callegari–Michou–Sławińska–Tomasone–Zigraiova (2025 ESM WP 73) show empirically that "rigid" spending (pensions, public wages) shifts consolidation onto investment.

The nearest conceptual precedents for "partial default on a promised liability" are Evans–Kotlikoff–Phillips (2012), where "the government is forced to default on its promised payment to the contemporaneous elderly", Novy-Marx–Rauh (2011), who price US state pension promises as defaultable debt, and the domestic-default literature (D'Erasmo–Mendoza 2016 JEEA, 2021 JME; Ferrière 2015; Deng 2024 ⚠️), which defaults on bonds held domestically. Arrears are treated as de facto default by Checherita-Westphal–Klemm–Viefers (2016) and Flynn–Pessoa (2014).

### 3.5 Financing permanent spending increases; defence (serves H3)

Tax smoothing (Barro 1979) prescribes debt finance for temporary spending and tax finance for permanent spending; Ohanian (1997 AER) quantifies the welfare cost of tax-financed Korea versus debt-financed World War II in a representative-agent model. Empirically, Ramey–Zubairy (2018) find military-news multipliers of 0.6–1.0, and Antolín-Díaz–Surico (2025 AER 115(7)) find that military spending has decades-long output effects through R&D while "the effects of public investment are shorter-lived". Boehm (2020 JME) finds a government-consumption multiplier near 0.8 and an investment multiplier near zero for short-lived shocks, the reverse of the draft's ordering (0 and 0.83 for permanent shocks at a fixed world rate), which the permanence of the shock and the accumulation of public capital must explain. Bom–Ligthart (2014) give an output elasticity of public capital of 0.083 in the short run and 0.122 in the long run against the draft's 0.05.

Heterogeneous-agent fiscal work with European calibrations covers generic spending and consolidation: Brinca–Holter–Krusell–Malafry (2016 JME), Brinca et al. (2021 IER), Ferriere–Navarro (2025 REStud), Hagedorn–Manovskii–Mitman (2019), Druedahl et al. (2025, SOE-HANK). The 2025–2026 defence literature is dense: Boullot–Cahn–Challe–Matheron (2026 CEPR DP 21270) study "a permanent increase in government spending of the magnitude implied by the 2025 change in NATO 'core defence' spending target" in "a calibrated Overlapping-Generations model with Heterogeneous Agents and a rich fiscal side that includes fully specified tax-and-transfer and pay-as-you-go social-security systems" and find "a tradeoff, when choosing among fiscal adjustments, between mitigating aggregate crowding-out of private consumption versus reducing consumption inequality" (country, open-economy status and cohort reporting ⚠️ not accessible); Lauwers–Michou–Ricci–Zavalloni (2026, ESM Euro Area Stability Watch ch. 2) compare reallocation, labour-tax and debt financing with a 20-year stabilisation of a +1.5 pp of GDP defence increase in a two-sector DSGE, finding self-financing of 25 cents per euro without and 53 cents with productivity spillovers, cumulative multipliers of 0.8 to 1.5, and GDP 2.5 % above baseline after two decades; Bokan et al. (2025 ECB Economic Bulletin) report two-year multipliers of 0.42–1.13 across ECB models and sharply lower multipliers under full labour-tax financing; the Commission (Spring 2025 Forecast box), the IMF (WEO April 2026 ch. 2; Furceri et al. 2026) and the OECD (Conigrave–Shin 2026) add multipliers, debt effects and the crowding-out of social spending. Ilzetzki (2025 Kiel Report) surveys the field. None of these is a high-debt SOE with a liability ledger and an r_B wedge, and none reports incidence by cohort and liability line; Braun–Joines (2015) is the template for that reporting.

### 3.6 r − g, wedges, fiscal limits and official lending (serves H4)

Blanchard (2019 AER) argues that r < g makes rollover feasible at no fiscal cost and that the welfare cost depends on r relative to the marginal product of capital; his 2023 book (via Corneo's review) states the condition the draft must confront: "If the interest rate is exogenous and there is no uncertainty, the government's fiscal space is infinite and debt sustainability is not an issue." Reis (2021 BIS WP 939; unpublished ⚠️) is the exact configuration of the draft, r < g < m, with the debt bound set by the bubble premium m − r rather than by r − g. Mehrotra–Sergeyev (2021 JME), Kocherlakota (2022, 2023 IER), Mian–Straub–Sufi (2025 AER), Aguiar–Amador–Arellano (2024 AER) and Angeletos–Lian–Wolf (2024 Econometrica) give conditions under which low rates license debt in heterogeneous-agent economies; Brumm–Feng–Kotlikoff–Kubler (2024 AEJ Macro), including an open-economy OLG, and Cao (2024 IMF WP) find Pareto gains only under implausible calibrations. Mauro–Zhou (2021) show that negative r − g is common and that defaults follow it. Convenience-yield theories of the wedge are Jiang–Lustig–Van Nieuwerburgh–Xiaolan (2024 Econometrica; 2025 NBER WP 34307 on the eurozone), Brunnermeier–Merkel–Sannikov (2024 JPE "Safe Assets"), Choi–Kirpalani–Perez (2025 JPE), and Bellon–Gnewuch–Orlandi–Zavalloni (2026 ESM WP 78: euro-area governments over-issue by 5 % of GDP). Fiscal limits through Laffer curves are Trabandt–Uhlig (2011 JME, Greece included), Bi (2012), Heer–Polito–Wickens (2020) and Holter–Krueger–Stepanchuk (2019).

Official lending as the source of a below-market rate is documented and modelled outside OLG: Corsetti–Erce–Uy (2018 CEPR DP 13292; 2020 RIO) in a default model with ESM-type maturities and rates; Mimir–Önder (2025 ESM WP 70) with concessional credit lines for Portugal; Schumacher–Weder di Mauro (2015 BPEA ⚠️) and Zettelmeyer–Kreplin–Panizza (2017) in accounting; Ardagna–Caselli (2014 AEJ Macro) on the negotiated terms. In the Japan cluster the wedge is universal (İKY: 1 % versus 3–4 %; Braun–Joines: fixed 1.24 pp; Kitao: 1 % versus the endogenous return; Hansen–İmrohoroğlu: bonds in utility) and calibrated to JGB yields. Heimberger (2023 JIMF) finds euro-area members face higher r − g risk than stand-alone sovereigns. No paper found places an official-lender rate wedge inside an OLG sustainability analysis.

### 3.7 Default with heterogeneous agents (positioning only)

The authors' bibliography covers classic default, restructuring and official lending. The sub-strand that matters for a no-default Greek application is domestic and distributional default: D'Erasmo–Mendoza (2016 JEEA, 2021 JME), Ferrière (2015), Jeon–Kabukcuoglu (2018 JEDC), Deng (2024 AEJ Macro ⚠️), Andreasen–Sandleris–Van der Ghote (2019 JME ⚠️), Bianchi–Ottonello–Presno (2023 JPE ⚠️), and Tran-Xuan (2026 IMF WP), who builds "a heterogeneous-agent small open economy in which redistribution relies on distortionary labor taxation and the government lacks commitment", calibrated to Italy; none defaults on welfare-state promises. Bocola–Bornstein–Dovis (2019) quantify euro-crisis default risk; the Financial Stability Fund papers (Ábrahám–Carceles-Poveda–Liu–Marimon 2025 REStud; Liu–Marimon–Wicht 2023 JIE; Callegari–Marimon–Wicht–Zavalloni 2023 RED) are the theory under which sovereign debt is safe and the limited-enforcement constraint is the DSA.

### 3.8 The authors' own work

The Fund papers establish constrained-efficient long-term contingent contracts under limited enforcement (and moral hazard in ACLM 2025), identify the Fund's country risk assessment with the DSA (Liu–Marimon–Wicht: the Fund "announces the level of liabilities the country can sustain"), deliver safe sovereign debt, and compute required Fund capacity and maturity (CMWZ 2023, Italy: 2.9-year maturity, 90 % of GDP capacity). They have no OLG, no non-debt liabilities, no welfare-state calibration, no r_B ≤ g and no wedge between the safe rate and the return on capital. The OLG engine is the Backpack line (Díaz-Saavedra–Marimon–Brogueira de Sousa 2023 JEEA, 2022 SERIEs; Díaz-Giménez–Díaz-Saavedra 2009 RED, 2017 JPEF), one liability at a time. Ábrahám–Brogueira de Sousa–Marimon–Mayr (2023 EER) evaluate a welfare-state liability (UI) in a multi-country GE model with SURE-type low-rate borrowing, the closest precedent for financing a liability at a below-market rate. On the ESM side, Sławińska co-authored WP 73 (consolidation composition); Zavalloni co-authored the ESM stochastic DSA brief (2025), WP 78 (convenience yields, 2026), the defence chapter (2026), Battistini–Callegari–Zavalloni (2019 ECB WP 2268, dynamic fiscal limits), and an unpublished 2024 paper with Marimon and Callegari, "Fiscal Rules with a Financial Stability Fund", listed only in Marimon's CV ⚠️, whose content bears on H1's "sustainability as a constraint on fiscal-policy design". The draft cites three of these works; the EER and SERIEs papers are absent from the bibliography; the draft writes "Fiscal Stability Fund" where the cited titles say "Financial Stability Fund".

## 4. Recent frontier (2023–2026)

- **Defence financing** (Section 3.5): Boullot–Cahn–Challe–Matheron (CEPR DP 21270, March 2026); Lauwers–Michou–Ricci–Zavalloni (ESM, August 2026); Bokan et al. (ECB, 2025); Commission Spring 2025 Forecast box; IMF WEO April 2026 ch. 2 and Furceri et al. (IMF WP 2026/053); Conigrave–Shin (OECD, 2026); Ilzetzki (Kiel, 2025); Marzian–Trebesch (Kiel, 2025); Antolín-Díaz–Surico (AER 2025); Antonova–Luetticke–Müller (2026, "The Military Multiplier"). The IMF finds that "about two-thirds of the additional spending is financed through higher budget deficits" in past buildups and that "sustained defense buildups … risk crowding out other public priorities—most notably social spending", which is the liability-composition margin of H3.
- **r − g and safe assets**: Mian–Straub–Sufi (AER 2025); Aguiar–Amador–Arellano (AER 2024); Angeletos–Lian–Wolf (Econometrica 2024); Brunnermeier–Merkel–Sannikov ("Safe Assets", JPE 2024; the bubble paper is R&R at REStud as of April 2026); Brumm–Feng–Kotlikoff–Kubler (AEJ Macro 2024); Kocherlakota (IER 2023); Cao (IMF WP 2024); Jiang et al. on eurozone convenience yields (NBER WP 34307, 2025); Bellon–Gnewuch–Orlandi–Zavalloni (ESM WP 78, 2026); Choi–Kirpalani–Perez (JPE 2025).
- **OLG sustainability**: Díaz-Giménez–Díaz-Saavedra (EER 2025, Spain HA-OLG open economy, pensions); Heer–Polito–Wickens (2023/2026, "pension space"); İmrohoroğlu–Ino (2025); Hansen–İmrohoroğlu (RED 2023); Kitao–Yamada (2026 survey); Katagiri et al. (2025); Okamoto (2024, 2025); Tran-Xuan (IMF WP 2026); Platzer (IMFER 2026, r* projections).
- **Official practice**: DSM 2025 (IP 332, February 2026) and its Greek fiche (S2 −0.4, S1 1.5, long-term risk low, medium-term high); the ESM's stochastic DSA brief (December 2025); IMF WP/24/181 on European debt with spending pressures of about 5½ % of GDP by 2050 for advanced European economies; Darvas–Welslau–Zettelmeyer (Bruegel 2023, 2024a, 2024b) on the DSA-based framework, with Greece's required medium-term SPB at 1.9–2.1 % of GDP.

## 5. Consensus

- Sustainability in practice is a statement about debt dynamics under an exogenous primary-balance path; ageing costs enter as exogenous projections; general-equilibrium feedback is absent from every official framework examined.
- The Japan OLG literature agrees that, absent reform, ageing-related spending requires consumption-tax rates between 20 and 60 %, that labour taxes alone run into the Laffer limit, and that delay raises the required adjustment.
- The r_B < r wedge is a standard calibration device; its size in Japan (1–3 pp) is of the order the draft uses (2–4 pp).
- Health-spending growth decomposes into demography, income and a residual; the residual dominates historically (Greece 1.7–2.5 %/yr before 2010).
- Greek austerity was purchases-heavy in the structural literature's measurement and accounted for around half of the output collapse; the category-level content of the cuts is documented outside the structural literature.

## 6. Active debates

- Whether r < g licenses higher debt (Blanchard; Mian–Straub–Sufi; Aguiar–Amador–Arellano) or not once risk, heterogeneity and generational incidence are counted (Brumm et al.; Cao; Kocherlakota; Mauro–Zhou).
- Whether the wedge between the sovereign rate and the return on capital is a convenience yield (Jiang et al.; Brunnermeier et al.; Bellon et al.), a bubble premium (Reis), or an institutional transfer (official lending; central-bank holdings).
- Whether public sector balance sheets and intertemporal net worth are useful sustainability objects (IMF Fiscal Monitor 2018; Koshima et al. 2021) or not (Debrun et al. 2019).
- Whether defence buildups should be debt-financed (Ilzetzki; Marzian–Trebesch; Barro's tax smoothing for temporary spending) or tax-financed when permanent, and how much self-financing the supply side yields (25–53 cents per euro in the ESM chapter; multipliers 0.4–1.9 across institutions).
- Whether Greek austerity's welfare-state cuts damaged outcomes as much as claimed (Kentikelenis et al. vs Perotti).

## 7. The gaps

**H1.** No model computes the adjustment that keeps *all* promised liabilities of a European small open economy sustainable in general equilibrium. The closest joint-liability models are Japanese and closed (or accounting); the closest European SOE models are representative-cohort or Blanchard–Yaari, or single-liability; the Greek SOE OLG (Glomm–Jung–Tran) has stable demographics and lumps health into residual spending. The Commission's S2 has no general-equilibrium correction, and the size of that correction for a country with Greece's debt and demographic path has not been computed.

**H2.** No paper measures a realised cut in a welfare-state line as a haircut on a promised liability against a structural counterfactual. The measurement objects exist separately: accrued-to-date pension entitlements (Eurostat), the chronology and micro NPV of Greek pension cuts (Symeonidis; Leventi–Matsaganis), the component-level account of health cuts (Perotti), and the growth decompositions of the projection literature. The gap is the structural counterfactual for the endogenous lines; the health line, being exogenous, does not fill it.

**H3.** The 2026 heterogeneous-agent OLG defence paper covers the aggregate-versus-inequality trade-off across financing options; the institutional papers cover multipliers and debt paths. Nobody reports which liability lines absorb a permanent defence increase and which cohorts pay, in a high-debt SOE where debt is absorbed abroad, or compares that with a public-investment increase of the same size.

**H4.** Official lending at below-market rates is documented for Greece and modelled in default and accounting frameworks; the r_B < r wedge is standard in OLG sustainability models but always sourced in market yields. Nobody has asked how much of a high-debt country's liability sustainability rests on the official-lending wedge, or who bears it, in an OLG with a welfare state.

**H5.** Greece has an official accrued-to-date pension entitlement figure (403 % of GDP, 2021) and Commission generational accounts, but no public sector balance sheet at all in the IMF database, no health liability anywhere, and no valuation consistent with an equilibrium model.

## 8. How the framework fills them, and what must be built

| Gap | What the current framework delivers | What is missing |
|---|---|---|
| H1 | HA-OLG SOE with incomplete markets; debt, pensions, UI, health coverage, public capital, NFA in one budget; Greek calibration; permanent-shock transitions with labour-tax closure to a terminal B/Y or NFA/Y | Ageing transition switched on (fertility and survival-improvement paths exist in code); an explicit sustainability criterion stated against Braun–Joines Definition 2 and Blanchard's exogenous-r point; a reported S2-type indicator with and without GE feedback |
| H2 | Exact Shapley split of the health gap 2009–2023 into coverage, age composition and residual; the identity for pensions written down (§5.4) | Pension-side counterfactual re-solves; present-value haircut measure; reconciliation with Eurostat Table 29 and Leventi–Matsaganis |
| H3 | G and I_g shocks under three financing regimes; budget line by line (Δτ_l +3.1/+3.4 pp for G, +2.8/+3.2 pp for I_g; I_g multiplier 0.76–0.83; G multiplier zero under debt financing) | Cohort and education-type welfare (no value functions or welfare measures are stored in the code); a defence-specific import share or production channel if the comparison with the ECB and ESM results is to be like-for-like |
| H4 | r_B separate from r; experiments at r_B = 0 with r = 4 % | Decide whether households earn r or r_B on B; add the framing to the front matter; run the ESM-rate versus market-rate comparison |
| H5 | Nothing | The ledger |

## 9. Pre-empted referee objections

1. *"The Commission's S2 already integrates pensions, health, long-term care and education."* It does, as exogenous Ageing Report paths added to a debt-accounting gap; the only behavioural link is a 0.6 multiplier in alternative scenarios (DSM 2024, 2025). The framework computes each line as an equilibrium outcome; the size of the difference is a result to report.
2. *"Kotlikoff said it all: debt is a labelling convention."* Green–Kotlikoff (2006) and Kotlikoff (2002) establish label-invariance; generational accounting "does not incorporate general equilibrium feedback effects" (AGK 1994). The claim to make is quantitative, not conceptual.
3. *"The Japan papers compute the adjustment with pensions, health and debt jointly."* They are closed-economy or exogenous-price accounting models without UI or public capital; the SOE with NFA absorption changes the incidence of debt financing (zero crowding out at a fixed world rate), and the European institutional setting sets the sovereign rate.
4. *"The health decomposition is arithmetic, like the Ageing Report's."* Correct for the health line, which is exogenous in the model (draft §5 says so). The model-based content is in the endogenous lines and requires re-solves.
5. *"Zero multiplier for debt-financed G at a fixed world rate is textbook."* Yes; the experiment's object is the budget composition and the required labour-tax increment, not the multiplier. Boehm (2020) and the ECB's 0.42–1.13 range are the benchmarks to state.
6. *"Why no default when Bocola–Bornstein–Dovis quantified euro-crisis default risk for these countries?"* The Fund papers (ACLM 2025; LMW 2023) give the theory under which debt is safe; GPV's footnote 25 and the ESM's lending terms give the facts; Glomm–Jung–Tran show what a risk-premium shock does in a Greek SOE OLG. The no-default assumption is a stated institutional case, not a general claim.
7. *"Perfect foresight and no aggregate risk exclude stochastic DSA."* True; Celasun–Debrun–Ostry (2006), Zenios et al. (2021) and the ESM brief (2025) are the stochastic comparators. The framework's output is the deterministic required adjustment under alternative rules, the object the Japan literature reports.
8. *"The title says ageing; the experiments hold the age distribution fixed."* The fertility and survival-improvement paths are implemented but not activated in the reported runs; the Ageing Report's Greek path (pensions 14.5 to 12.0 % of GDP by 2070) is the natural input.
9. *"Wealth Gini 0.28 against 0.58."* Any distributional claim in H3 needs either the bequest motive and initial-wealth heterogeneity the draft lists as pending, or a statement that incidence is by cohort and education type rather than by wealth.
10. *"Households earn 4 % on holdings of B while the government pays 0–2 %."* The open modelling choice in `TREND_GROWTH_PLAN.md`; Reis (2021) and Brunnermeier et al. (2024) are the references for who receives a convenience or bubble premium, Corsetti–Erce–Uy (2020) for who receives an official-lending subsidy.

## 10. Method and limits

- Google Scholar returned empty results for all 15 queries; its agent resolved metadata through web search instead, so no citation counts are available for strands S1, S2, S5, S6.
- Semantic Scholar was rate-limited throughout the first pass (shared key with a second session); its snippet, paper-details and recommendation endpoints failed on every attempt under concurrent load, so citation-graph expansion on 10 September came from four `paper_citations` calls. The step was repeated on 11 September with one call at a time and completed (see Addendum).
- The session's web-search budget (200 calls) was exhausted during the official-documents and survey sweeps; later verification used direct fetches only.
- Full texts read: İKY 2016 and 2019, Braun–Joines 2015, Hansen–İmrohoroğlu 2016, Kitao 2015, Evans–Kotlikoff–Phillips 2012, D'Erasmo–Mendoza–Zhang 2016 (NBER version), Debrun et al. 2019 (IMF conference draft), Yared 2019, Auclert–Rognlie–Straub 2025, Auerbach–Gokhale–Kotlikoff 1994, de la Maisonneuve–Oliveira Martins 2013, Przywara 2010, Medeiros–Schwierz 2013 and 2015, Ageing Report 2024 (both volumes), DSM 2024 and 2025, IMF SRDSF 2022, ECB OP 185, De Nardi–French–Jones 2016; targeted sections of House–Proebsting–Tesar 2020, Gourinchas–Philippon–Vayanos 2017, Kotlikoff 2002, Feldstein–Liebman 2002, Glomm–Jung–Tran (WP 2013), Castro et al. (WP 2013), McGrattan et al. 2018.
- Unverified items are flagged ⚠️ in the text and in the paper summaries: Boullot et al. 2026 (country, cohort reporting); Bettendorf et al. 2011 (idiosyncratic risk, open economy); Heer–Polito–Wickens 2020 (Greece among the 14); Kaier–Müller 2015 (Greece among the 18); Reis (no journal version found); Brunnermeier–Merkel–Sannikov bubble paper (R&R status from an author site); Deng, Andreasen et al., Bianchi–Ottonello–Presno (identifiers from memory); Schumacher–Weder di Mauro 2015 (from search only); the unpublished Marimon–Zavalloni–Callegari 2024 paper (CV only).
- Phase 7.5 (writing-pattern mining) was skipped: no top-5 journal article was read in full in this run.

---

## Paper-by-paper summaries

### İmrohoroğlu, Kitao, Yamada (2016) — Japan fiscal balance, accounting OLG
- **Title**: Achieving Fiscal Balance in Japan
- **Published in**: International Economic Review 57(1), 117–154 | Citations: 76 (WP record)
- **Question**: What are the paths of Japanese debt and the pension fund under current law, and which policies restore balance?
- **Data**: Japanese micro data by age, gender, employment type, earnings, assets and pension category; IPSS demographics.
- **Identification**: Large-scale OLG accounting model; no optimisation; exogenous rates (3 % savings, 1 % JGB, 2 % pension fund) and 1.5 % wage growth.
- **Findings**: Net debt/GDP 162 % (2020) to 477 % (2060); a 35 % consumption tax stabilises debt near 80 %; sensitivity of 2060 debt to r_B from 253 % (−1 %) to 998 % (3 %).
- **Limitation**: No behavioural response; health not a separate line; open/closed status not stated.
- **Relevance**: H1 nearest accounting counterpart; H4 fixed r_B wedge sourced in yields.

### İmrohoroğlu, Kitao, Yamada (2019) — adds health and LTC lines
- **Title**: Fiscal Sustainability in Japan: What to Tackle?
- **Published in**: Journal of the Economics of Ageing 14, 100205 | Citations: 21
- **Question**: Which policy combinations achieve sustainability once health and LTC insurance are explicit?
- **Data**: 2015 Japanese data; 2017 IPSS projections.
- **Identification**: Same accounting OLG; health coverage 70–90 % by age, LTC 90 %.
- **Findings**: Net debt 625 % of GDP by 2070 absent reform; 2050 components: pensions 5.6, health 5.25, LTC 3.1 % of GDP; package (FRA 67, −10 % pensions, 20 % copays, female employment, 15 % consumption tax) gives 87 % in 2050.
- **Limitation**: Accounting model; zero copay elasticity.
- **Relevance**: H1 (all lines jointly); H2 (coverage rates as explicit parameters).

### Braun, Joines (2015) — graying Japan, terminal-debt criterion
- **Title**: The implications of a graying Japan for government policy
- **Published in**: JEDC 57, 1–23 | Citations: 117 / 22 influential
- **Question**: Is Japanese fiscal policy sustainable, when must adjustment occur, and who pays?
- **Data**: IPSS demographics; MHLW medical and LTC profiles; 2004 pension rules.
- **Identification**: Closed GE OLG, representative cohorts; r^g 1.24 pp below the return on capital; transversality with terminal debt/GDP = 1.0.
- **Findings**: Steady-state consumption tax 26.4 %; 57 % if consolidation waits until 2039; ≈36 % constant from 2018; copays for 70+ cut the tax path; cohorts 2000–2045 prefer constant taxes, later cohorts copays.
- **Limitation**: Closed; no idiosyncratic risk; exogenous medical spending.
- **Relevance**: H1 criterion; H3 generational-incidence template; H4 wedge.

### Kitao (2015) — HA-OLG fiscal cost of ageing
- **Title**: Fiscal cost of demographic transition in Japan
- **Published in**: JEDC 54, 37–58 | Citations: 90
- **Question**: Which tax path balances the budget through the demographic transition?
- **Data**: Japanese demographics; Lise et al. wage process; MHLW health and LTC profiles.
- **Identification**: Closed GE life-cycle model with idiosyncratic wage risk, participation margin; debt fixed at 100 % of GDP; r^d = 1 % exogenous.
- **Findings**: Consumption tax 5 % to 19.3 % long run, peak 47.9 % (2083); labour-tax alternative +13.5 pp with output 13 % lower; pension cuts of 20–40 % and NRA 70 lower the peak to 19–37 %.
- **Limitation**: Closed; debt path exogenous.
- **Relevance**: H1 closest HA structure; H3 consumption versus labour tax with cohort welfare.

### McGrattan, Miyachi, Peralta-Alva (2018) — financing options incl. debt
- **Title**: On Financing Retirement, Health, and Long-term Care in Japan
- **Published in**: IMF WP 18/249 | Citations: 8
- **Question**: Which financing option for ageing costs performs best?
- **Data**: Japanese macro and micro data.
- **Identification**: Closed GE OLG; consumption tax versus contributions versus debt versus copays.
- **Findings**: Gradual consumption-tax increases deliver better macro performance and higher welfare for most individuals than the alternatives.
- **Limitation**: Closed economy (stated in text).
- **Relevance**: H1 closest joint-liability financing comparison.

### Glomm, Jung, Tran (2018) — the Greek SOE OLG precedent
- **Title**: Fiscal Austerity Measures: Spending Cuts vs. Tax Increases
- **Published in**: Macroeconomic Dynamics 22(2), 501–540 | Citations: 20
- **Question**: What are the macro and generational welfare effects of risk-premium shocks and debt reduction in a high-debt SOE?
- **Data**: "Calibrated to Greece at the beginning of the 21st century."
- **Identification**: OLG with skill heterogeneity, mortality risk, borrowing constraint; SOE at a fixed world rate plus country risk premium; public capital in public-good production; public employment; two pension schemes; stable demographics.
- **Findings**: Debt cut from 105 to 85 % of GDP; output falls up to 6 % in the first ten years, rises 4.3–5 % in the long run; spending-based reform dominates in the long run; old and middle-aged cohorts lose, young and future cohorts gain.
- **Limitation**: No ageing transition; health lumped into residual spending; no UI; risk premium rather than official-lending wedge; idiosyncratic income risk not found in the text ⚠️.
- **Relevance**: H1 and H3 nearest Greek precedent; the first paper a referee will cite.

### Castro, Maria, Félix, Braz (2016) — Portugal, PESSOA
- **Title**: Aging and fiscal sustainability in a small euro area economy
- **Published in**: Macroeconomic Dynamics 21(7) | Citations: 17
- **Question**: Macroeconomic impact of ageing in a small euro-area economy under a fiscal rule.
- **Data**: 2012 Ageing Report projections for Portugal.
- **Identification**: Open-economy NK DSGE with Blanchard–Yaari non-Ricardian households; ageing shocks to pensions, health and LTC; fiscal rule on a debt target.
- **Findings**: Impact depends on the pace of ageing, accrued rights and the policy response; two policy options compared (tax financing versus replacement-rate cut).
- **Limitation**: Representative-cohort DSGE; "rude simplifications" of demographics and social security (authors' words).
- **Relevance**: H1 nearest euro-area SOE; positions itself as adding to DSA; states that OLG models "lack open economy features".

### Kudrna, Tran (2018) — SOE budget repair with generational incidence
- **Title**: Comparing budget repair measures for a small open economy with growing debt
- **Published in**: Journal of Macroeconomics 55, 162–183 | Citations: 2
- **Question**: Immediate versus gradual deficit elimination via taxes or transfer cuts.
- **Data**: Australia.
- **Identification**: Computable OLG SOE with exogenous world rate.
- **Findings**: Welfare by generation and by rich and poor differs across instruments and pace.
- **Limitation**: Pensions only; no health, UI, public capital or wedge.
- **Relevance**: H1 and H3 SOE design match.

### D'Erasmo, Mendoza, Zhang (2016) — structural sustainability
- **Title**: What is a Sustainable Public Debt?
- **Published in**: Handbook of Macroeconomics vol. 2, ch. 32 | Citations: 32 (S2)
- **Question**: How should sustainability be assessed after the post-2008 debt surge?
- **Data**: US 1791–2014; 1951–2013 panel; US–EU15 calibration.
- **Identification**: Bohn reaction functions; two-country neoclassical model with dynamic Laffer curves; domestic default with two wealth types.
- **Findings**: European labour taxes near the Laffer peak "cannot restore fiscal solvency"; sustainable debt falls sharply with default risk.
- **Limitation**: Representative agent; no pensions, health or ageing.
- **Relevance**: H1 definition; Laffer check on the draft's labour-tax closures; H4.

### Debrun, Ostry, Willems, Wyplosz (2019) — the practitioners' survey
- **Title**: Debt Sustainability (CEPR DP title: Public Debt Sustainability)
- **Published in**: Abbas, Pienkowski, Rogoff (eds), Sovereign Debt: A Guide for Economists and Practitioners, OUP | Citations: n/a
- **Question**: What do solvency and sustainability mean operationally?
- **Identification**: Survey; fiscal-fatigue debt limit; stress tests versus fan charts.
- **Findings**: "Solvency … has no operational meaning"; INW and PSBS are discussed and dismissed ("concerns about net worth do not cause sovereign crises or defaults"); Greece box on programme projections versus outcomes.
- **Limitation**: Draft version read ⚠️.
- **Relevance**: H1 definitions; H5 direct interlocutor.

### Auerbach, Gokhale, Kotlikoff (1994) — generational accounting
- **Title**: Generational Accounting: A Meaningful Way to Evaluate Fiscal Policy
- **Published in**: JEP 8(1), 73–94
- **Findings**: Intertemporal constraint on all liabilities; "failure to satisfy this constraint means that the government will default on its liabilities"; US 1991 imbalance 111 % (65 % at r = 3 %); closing it needs +11.7 % on taxes or −24.9 % on transfers.
- **Limitation**: "Does not incorporate general equilibrium feedback effects."
- **Relevance**: H1, H2, H5 conceptual basis.

### Gourinchas, Philippon, Vayanos (2017) — Greek crisis analytics
- **Title**: The Analytics of the Greek Crisis
- **Published in**: NBER Macroeconomics Annual 31 | Citations: high (not retrieved)
- **Identification**: SOE DSGE in a monetary union; sovereign rate = world rate + expected losses; G and T follow one rule, G = T = 19.1 % of GDP.
- **Findings**: Fiscal consolidation ≈ 50 % of the peak-to-trough output drop; sudden stop and sovereign risk ≈ 40 %; footnote 25 records refinancing by official creditors at low rates.
- **Limitation**: No pension or health lines; no cohorts.
- **Relevance**: H2 (no category-level cuts); H4 (wedge documented, not used).

### House, Proebsting, Tesar (2020) — austerity multipliers
- **Title**: Austerity in the aftermath of the Great Recession
- **Published in**: JME 115, 37–63
- **Identification**: 29-country NK DSGE; austerity as purchases forecast errors.
- **Findings**: Multiplier ≈ 2; GIIPS output 18 % below trend by 2014; debt ratios rose because of austerity.
- **Limitation**: Purchases only; no categories.
- **Relevance**: H2 not triggered; H3 demand-side benchmark.

### Perotti (2021) — Greek health cuts by component
- **Title**: The human side of austerity: health spending and outcomes during the Greek crisis
- **Published in**: Economic Policy 36(105)
- **Findings**: Component-level account of public health cuts; outcome deterioration smaller than the Lancet literature claims.
- **Limitation**: Accounting; no counterfactual; no liability framing.
- **Relevance**: H2 nearest on the health side.

### Leventi, Matsaganis (2020) — pension entitlements removed
- **Title**: Disentangling annuities and transfers: the case of Greek retirement benefits
- **Published in**: European Journal of Social Security 22(3) ⚠️
- **Findings**: NPV of contributions and benefits for 2008 private-sector retirees under pre-2010 rules and after the 2010–13 cuts, by cohort.
- **Limitation**: Micro; no general equilibrium.
- **Relevance**: H2 nearest on the pension side; the haircut measure to reconcile with.

### Evans, Kotlikoff, Phillips (2012) — game over
- **Title**: Game Over: Simulating Unsustainable Fiscal Policy
- **Published in**: NBER WP 17917; in Fiscal Policy after the Financial Crisis (2013)
- **Identification**: Two-period stochastic OLG; fixed real transfer to the old.
- **Findings**: Median time to infeasibility two 30-year periods; "the government is forced to default on its promised payment to the contemporaneous elderly"; fiscal gaps double before shutdown.
- **Relevance**: H1/H2 language of default on a non-debt promise.

### Blanchard (2019, 2023) — r < g
- **Title**: Public Debt and Low Interest Rates (AER 109(4)); Fiscal Policy under Low Interest Rates (MIT Press)
- **Findings**: r < g historically normal; rollover feasible at no fiscal cost; welfare cost depends on r versus the marginal product of capital; via Corneo (2023): with exogenous r and no uncertainty fiscal space is infinite.
- **Relevance**: H4 the objection to answer; not in the draft's bibliography.

### Reis (2021) — r < g < m
- **Title**: The constraint on public debt when r < g but g < m
- **Published in**: BIS WP 939; CEPR DP 15950; CFM DP 2021-11 (no journal version found ⚠️) | Citations: 65
- **Findings**: Debt capacity bounded by the bubble premium m − r, not by r − g.
- **Relevance**: H4 exact configuration; comparator for the source of the wedge.

### Boullot, Cahn, Challe, Matheron (2026) — HA-OLG defence buildup
- **Title**: Aggregate and Distributional Implications of a Military Buildup
- **Published in**: CEPR DP 21270, March 2026
- **Findings**: Permanent NATO-target increase in an HA-OLG with full tax-transfer and PAYG systems; trade-off between aggregate crowding-out and consumption inequality across fiscal adjustments.
- **Limitation**: Country, open-economy status and cohort reporting not accessible ⚠️.
- **Relevance**: H3 strongest challenge.

### Lauwers, Michou, Ricci, Zavalloni (2026) — ESM defence chapter
- **Title**: Security at what cost? Defence spending, growth, and the fiscal arithmetic
- **Published in**: ESM Euro Area Stability Watch 2026, ch. 2
- **Findings**: +1.5 pp of GDP over ten years; self-financing 25 cents per euro without and 53 cents with spillovers; multipliers 0.8–1.5; GDP 2.5 % above baseline after two decades; labour-tax financing erodes self-financing; permanently higher debt raises the cost ≈5 %.
- **Limitation**: Two-sector DSGE; no cohorts; no country detail.
- **Relevance**: H3 coauthor overlap.

### Eurostat Table 29 / Deboeck–Eckerfelt (2020) — the pension ledger that exists
- **Findings**: Greek unfunded (government-managed) pension entitlements 331 / 374 / 403 % of GDP in 2015 / 2018 / 2021 (Eurostat API); "not … part of government debt"; Deboeck–Eckerfelt argue the figure is not a sustainability measure.
- **Relevance**: H5 partial pre-emption; H2 pricing anchor.

---


## Addendum (2026-09-11): Semantic Scholar repeat pass

The Semantic Scholar step was rerun as a single agent issuing one call at a time with forced pauses: 66 calls, 8 transient failures (5 server disconnects, 3 rate limits), every one recovered on retry; the recommendation pass, nine citation-graph expansions, seven snippet searches and 22 of 23 metadata resolutions that had failed on 10 September all completed. Yesterday's failures were load-related (two callers on a one-request-per-second key; bursts of two or more calls), not a broken server. Fifty-five entries with verified metadata were added to the bibliography.

**No verdict changes.** The one item that could have moved H1 is settled: Baksa–Munkácsi (2016, Bank of Lithuania WP 32), the OGRE-type OLG for Southern Europe, is calibrated to Portugal and Spain; Greece is not covered (RePEc abstract, read 11 September). Boullot–Cahn–Challe–Matheron (2026) is not indexed in Semantic Scholar, so its calibration country and cohort reporting remain unverified ⚠️. The phrase "liability sustainability analysis" has no prior use in the economics full texts indexed by Semantic Scholar.

**Additions by candidate.**
- H1: Beetsma–Busse–Larch–Romp (2025, JPEF) combine the Debt Sustainability Monitor and the Ageing Report to compute the extra growth each Member State, Greece included, needs to keep debt on the Commission's baseline under alternative demographic scenarios; pure accounting, the official-indicator comparator. Bernardino–Franco–Teles Morais (2024, SSRN) is a recent European OLG on the fiscal burden of ageing and immigration (liability set unverified ⚠️). Greek items surfaced by the Greece-targeted search are actuarial or descriptive only (Symeonidis–Tinios–Xenos 2021; Nektarios–Tinios 2019). The Glomm–Jung–Tran citation graph contains no OLG follow-up for Greece, only the Bank-of-Greece representative-agent DSGE cluster (Dimakopoulou–Economides–Philippopoulos 2022; Economides–Papageorgiou–Philippopoulos 2020).
- H2: Erce–Mallucci ("Selective Sovereign Defaults", Dallas Fed WP 127) and Shakhnov–Paczos (2026, Macroeconomic Dynamics) formalise default on one creditor class, the nearest formal analogue to defaulting on one class of claimants; Hindriks–Çetin (2025, JPEF) and Börsch-Supan–Ludwig (2010) treat pension-reform reversals as the sustainability risk. Nothing measures a realised cut as default.
- H3: Sundram (2026, REStat) shows that in a small-open-economy HANK the cumulative fiscal multiplier is exactly one and deficits are not self-financing because foreign debt is repaid, the closest analytical statement of the draft's NFA-absorption mechanism; Piguillem–Riboni (2024, AER) on debt dynamics with rigid spending lines; Asonuma–Joo (2019, 2021) on spending composition and public capital under a debt constraint; Bolouri–Lohse–Qari (2025, Kyklos): in a German survey the over-50s prefer debt financing of defence and the young prefer taxes.
- H4: the low-income-country Debt Sustainability Framework is the one structural treatment of concessional rates inside a sustainability analysis: the IMF–IDA (2004) statement, quoted in Gill–Pinto (2023), that sustainability under official financing "is largely de-linked from the sentiments of the market", and the Buffie–Berg–Pattillo–Portillo–Zanna (2012) representative-agent model with a concessional and a market borrowing rate plus public investment (metadata from a citing paper ⚠️). Reinhart–Trebesch (2015, BPEA) document two centuries of Greek dependence on official lenders. Theory: Abel–Panageas (2022) on perpetual primary deficits in a dynamically efficient economy; Angeletos–Collard–Dellas (2023, JPE) on the liquidity premium; Brumm–Feng–Kotlikoff–Kubler (2022, JPubE); Miao–Su (2024, AEJ Macro); Elenev–Landvoigt–Shultz–Van Nieuwerburgh (2021) on the central bank as fiscal capacity; Mitchener–Trebesch (2023, JEL) as survey. Still no OLG sustainability analysis with an official-lender wedge.
- H5: Castañer–Garvey–Pérez-Salamero–Vidal-Meliá (2025) already convert Eurostat Table 29 into an actuarial balance sheet with a net-worth solvency measure (Spain: assets cover 72 % of liabilities); Brede–Henn (2019) build Finland's public sector balance sheet with the accrued pension leg (301 % of GDP). Neither is Greek, neither has a health or UI leg, neither is consistent with an equilibrium model. Weddige (Peter Lang monograph) and Girodet et al. (OECD) are the data-method sources; an OECD figure of Greek implicit pension liabilities above 200 % of GDP around 1999 (Mylonas–de la Maisonneuve) is a pointer to verify ⚠️.

**Metadata resolved** for the entries that carried a warning in the strand sections: De Nardi–İmrohoroğlu–Sargent 1999; De Nardi–French–Jones 2010; Kopecky–Koreshkova 2014; Braun–Kopecky–Koreshkova 2017, 2019; Jung–Tran 2016; Zhao 2014; Conesa et al. 2018; Hall–Jones 2007; Fonseca et al. 2021; Hosseini–Kopecky–Zhao 2022; Pashchenko–Porapakkarm 2013; Bassetto–Cui 2018; Bi 2012; Holter–Krueger–Stepanchuk 2019; Deng; Andreasen–Sandleris–Van der Ghote 2019; Bianchi–Ottonello–Presno 2023; Arellano–Bai–Mihalache 2024. Barro (2023) "r minus g" could not be resolved by title search. Full details in the BibTeX below.

## Bibliography (APA, core entries; full list in the BibTeX below and in `dsa-lsa-contributions.bib`)

Ábrahám, Á., Carceles-Poveda, E., Liu, Y., & Marimon, R. (2025). On the optimal design of a Financial Stability Fund. *Review of Economic Studies*. https://doi.org/10.1093/restud/rdaf076
Aguiar, M., Amador, M., & Arellano, C. (2024). Micro risks and (robust) Pareto-improving policies. *American Economic Review, 114*(11), 3669–3713.
Antolín-Díaz, J., & Surico, P. (2025). The long-run effects of government spending. *American Economic Review, 115*(7), 2376–2413.
Auerbach, A. J., Gokhale, J., & Kotlikoff, L. J. (1994). Generational accounting: A meaningful way to evaluate fiscal policy. *Journal of Economic Perspectives, 8*(1), 73–94.
Baksa, D., Munkácsi, Z., & Nerlich, C. (2020). A framework for assessing the costs of pension reform reversals. ECB Working Paper 2396.
Bettendorf, L., van der Horst, A., Draper, N., van Ewijk, C., de Mooij, R., & ter Rele, H. (2011). Ageing and the conflict of interest between generations. *De Economist, 159*(3), 257–278.
Blanchard, O. (2019). Public debt and low interest rates. *American Economic Review, 109*(4), 1197–1229.
Blanchard, O. (2023). *Fiscal policy under low interest rates*. MIT Press.
Bohn, H. (1998). The behavior of U.S. public debt and deficits. *Quarterly Journal of Economics, 113*(3), 949–963.
Bokan, N., Jacquinot, P., Lalik, M., Müller, G., Priftis, R., & Rigato, R. (2025). Macroeconomic impacts of higher defence spending: A model-based assessment. *ECB Economic Bulletin*, 6/2025.
Boullot, M., Cahn, C., Challe, E., & Matheron, J. (2026). Aggregate and distributional implications of a military buildup. CEPR Discussion Paper 21270.
Braun, R. A., & Joines, D. H. (2015). The implications of a graying Japan for government policy. *Journal of Economic Dynamics and Control, 57*, 1–23.
Brumm, J., Feng, X., Kotlikoff, L. J., & Kubler, F. (2024). When interest rates go low, should public debt go high? *American Economic Journal: Macroeconomics, 16*(4), 432–469.
Buiter, W. H. (1985). A guide to public sector debt and deficits. *Economic Policy, 1*(1), 13–61.
Castro, G., Maria, J. R., Félix, R. M., & Braz, C. R. (2016). Aging and fiscal sustainability in a small euro area economy. *Macroeconomic Dynamics, 21*(7).
Collard, F., Habib, M., & Rochet, J.-C. (2015). Sovereign debt sustainability in advanced economies. *Journal of the European Economic Association, 13*(3), 381–420.
Corsetti, G., Erce, A., & Uy, T. (2018). Debt sustainability and the terms of official support. CEPR Discussion Paper 13292.
Debrun, X., Ostry, J. D., Willems, T., & Wyplosz, C. (2019). Debt sustainability. In S. A. Abbas, A. Pienkowski, & K. Rogoff (Eds.), *Sovereign debt: A guide for economists and practitioners*. Oxford University Press.
D'Erasmo, P., & Mendoza, E. G. (2016). Distributional incentives in an equilibrium model of domestic sovereign default. *Journal of the European Economic Association, 14*(1), 7–44.
D'Erasmo, P., Mendoza, E. G., & Zhang, J. (2016). What is a sustainable public debt? In J. B. Taylor & H. Uhlig (Eds.), *Handbook of Macroeconomics* (Vol. 2, pp. 2493–2597). Elsevier.
Díaz-Saavedra, J., Marimon, R., & Brogueira de Sousa, J. (2023). A worker's backpack as an alternative to PAYG pension systems. *Journal of the European Economic Association, 21*(5), 1944–1993.
European Commission. (2024). *2024 Ageing Report*. Institutional Paper 279.
European Commission. (2026). *Debt Sustainability Monitor 2025*. Institutional Paper 332.
Evans, R. W., Kotlikoff, L. J., & Phillips, K. L. (2012). Game over: Simulating unsustainable fiscal policy. NBER Working Paper 17917.
Ghosh, A. R., Kim, J. I., Mendoza, E. G., Ostry, J. D., & Qureshi, M. S. (2013). Fiscal fatigue, fiscal space and debt sustainability in advanced economies. *Economic Journal, 123*(566), F4–F30.
Glomm, G., Jung, J., & Tran, C. (2018). Fiscal austerity measures: Spending cuts vs. tax increases. *Macroeconomic Dynamics, 22*(2), 501–540.
Gourinchas, P.-O., Philippon, T., & Vayanos, D. (2017). The analytics of the Greek crisis. *NBER Macroeconomics Annual, 31*, 1–81.
Hansen, G. D., & İmrohoroğlu, S. (2016). Fiscal reform and government debt in Japan: A neoclassical perspective. *Review of Economic Dynamics, 21*, 201–224.
Heer, B., Polito, V., & Wickens, M. R. (2020). Population aging, social security and fiscal limits. *Journal of Economic Dynamics and Control, 116*, 103913.
House, C. L., Proebsting, C., & Tesar, L. L. (2020). Austerity in the aftermath of the Great Recession. *Journal of Monetary Economics, 115*, 37–63.
İmrohoroğlu, S., Kitao, S., & Yamada, T. (2016). Achieving fiscal balance in Japan. *International Economic Review, 57*(1), 117–154.
İmrohoroğlu, S., Kitao, S., & Yamada, T. (2019). Fiscal sustainability in Japan: What to tackle? *Journal of the Economics of Ageing, 14*, 100205.
Kitao, S. (2015). Fiscal cost of demographic transition in Japan. *Journal of Economic Dynamics and Control, 54*, 37–58.
Kotlikoff, L. J. (2002). Generational policy. In *Handbook of Public Economics* (Vol. 4, pp. 1873–1932). Elsevier.
Kudrna, G., & Tran, C. (2018). Comparing budget repair measures for a small open economy with growing debt. *Journal of Macroeconomics, 55*, 162–183.
Lauwers, A., Michou, M., Ricci, L., & Zavalloni, L. (2026). Security at what cost? Defence spending, growth, and the fiscal arithmetic. *ESM Euro Area Stability Watch 2026*, ch. 2.
Leventi, C., & Matsaganis, M. (2020). Disentangling annuities and transfers: The case of Greek retirement benefits. *European Journal of Social Security, 22*(3).
Liu, Y., Marimon, R., & Wicht, A. (2023). Making sovereign debt safe with a Financial Stability Fund. *Journal of International Economics, 145*, 103834.
Mauro, P., & Zhou, J. (2021). r − g < 0: Can we sleep more soundly? *IMF Economic Review, 69*, 197–229.
McGrattan, E. R., Miyachi, K., & Peralta-Alva, A. (2018). On financing retirement, health, and long-term care in Japan. IMF Working Paper 18/249.
Mian, A., Straub, L., & Sufi, A. (2025). A Goldilocks theory of fiscal deficits. *American Economic Review, 115*(12), 4253–4291.
Perotti, R. (2021). The human side of austerity: Health spending and outcomes during the Greek crisis. *Economic Policy, 36*(105).
Reis, R. (2021). The constraint on public debt when r < g but g < m. BIS Working Paper 939.
Storesletten, K. (2000). Sustaining fiscal policy through immigration. *Journal of Political Economy, 108*(2), 300–323.
Yared, P. (2019). Rising government debt: Causes and solutions for a decades-old trend. *Journal of Economic Perspectives, 33*(2), 115–140.

## BibTeX

```bibtex
% Literature review: DSA-LSA paper, 2026-09-10. Entries marked "note = {UNVERIFIED ...}" carry metadata not confirmed on a primary page.

@article{imrohorogluAchievingFiscalBalance2016,
  author = {{\.I}mrohoro{\u{g}}lu, Selahattin and Kitao, Sagiri and Yamada, Tomoaki},
  title = {Achieving Fiscal Balance in Japan},
  journal = {International Economic Review},
  year = {2016}, volume = {57}, number = {1}, pages = {117--154},
  doi = {10.1111/iere.12150}
}
@article{imrohorogluFiscalSustainabilityJapan2019,
  author = {{\.I}mrohoro{\u{g}}lu, Selahattin and Kitao, Sagiri and Yamada, Tomoaki},
  title = {Fiscal Sustainability in Japan: What to Tackle?},
  journal = {The Journal of the Economics of Ageing},
  year = {2019}, volume = {14}, pages = {100205},
  doi = {10.1016/j.jeoa.2019.100205}
}
@article{braunImplicationsGrayingJapan2015,
  author = {Braun, R. Anton and Joines, Douglas H.},
  title = {The implications of a graying Japan for government policy},
  journal = {Journal of Economic Dynamics and Control},
  year = {2015}, volume = {57}, pages = {1--23},
  doi = {10.1016/j.jedc.2015.05.002}
}
@article{hansenFiscalReformGovernment2016,
  author = {Hansen, Gary D. and {\.I}mrohoro{\u{g}}lu, Selahattin},
  title = {Fiscal reform and government debt in Japan: A neoclassical perspective},
  journal = {Review of Economic Dynamics},
  year = {2016}, volume = {21}, pages = {201--224},
  doi = {10.1016/j.red.2015.04.001}
}
@article{hansenDemographicChangeGovernment2023,
  author = {Hansen, Gary D. and {\.I}mrohoro{\u{g}}lu, Selahattin},
  title = {Demographic change, government debt and fiscal sustainability in Japan: The impact of bond purchases by the Bank of Japan},
  journal = {Review of Economic Dynamics},
  year = {2023}, volume = {50}, pages = {88--105},
  doi = {10.1016/j.red.2023.07.007}
}
@article{kitaoFiscalCostDemographic2015,
  author = {Kitao, Sagiri},
  title = {Fiscal cost of demographic transition in Japan},
  journal = {Journal of Economic Dynamics and Control},
  year = {2015}, volume = {54}, pages = {37--58},
  doi = {10.1016/j.jedc.2015.02.015}
}
@article{kitaoSustainableSocialSecurity2014,
  author = {Kitao, Sagiri},
  title = {Sustainable social security: Four options},
  journal = {Review of Economic Dynamics},
  year = {2014}, volume = {17}, number = {4}, pages = {756--779},
  doi = {10.1016/j.red.2013.11.004}
}
@techreport{mcgrattanFinancingRetirementHealth2018,
  author = {McGrattan, Ellen R. and Miyachi, Kazuaki and Peralta-Alva, Adrian},
  title = {On Financing Retirement, Health, and Long-term Care in Japan},
  institution = {International Monetary Fund}, type = {IMF Working Paper}, number = {18/249},
  year = {2018}, doi = {10.5089/9781484384718.001}
}
@article{kitaoSurveyDemographicsFamily2026,
  author = {Kitao, Sagiri and Yamada, Tomoaki},
  title = {Survey of demographics, family, and fiscal sustainability in Japan: macroeconomic approaches},
  journal = {The Japanese Economic Review},
  year = {2026}, volume = {77}, number = {3}, pages = {471--507},
  doi = {10.1007/s42973-026-00250-y}
}
@article{glommFiscalAusterityMeasures2018,
  author = {Glomm, Gerhard and Jung, Juergen and Tran, Chung},
  title = {Fiscal Austerity Measures: Spending Cuts vs. Tax Increases},
  journal = {Macroeconomic Dynamics},
  year = {2018}, volume = {22}, number = {2}, pages = {501--540},
  doi = {10.1017/S1365100516000298},
  note = {Calibrated to Greece; small open economy with world rate plus risk premium (Towson WP 2013-01)}
}
@article{kudrnaComparingBudgetRepair2018,
  author = {Kudrna, George and Tran, Chung},
  title = {Comparing budget repair measures for a small open economy with growing debt},
  journal = {Journal of Macroeconomics},
  year = {2018}, volume = {55}, pages = {162--183},
  doi = {10.1016/j.jmacro.2017.10.005}
}
@article{kudrnaFacingDemographicChallenges2019,
  author = {Kudrna, George and Tran, Chung and Woodland, Alan},
  title = {Facing Demographic Challenges: Pension Cuts or Tax Hikes?},
  journal = {Macroeconomic Dynamics},
  year = {2019}, volume = {23}, number = {2}, pages = {625--673},
  doi = {10.1017/S1365100516001292}
}
@article{castroAgeingFiscalSustainability2016,
  author = {Castro, Gabriela and Maria, Jos{\'e} R. and F{\'e}lix, Ricardo Mourinho and Braz, Cl{\'a}udia Rodrigues},
  title = {Aging and Fiscal Sustainability in a Small Euro Area Economy},
  journal = {Macroeconomic Dynamics},
  year = {2016}, volume = {21}, number = {7},
  doi = {10.1017/S1365100515001029},
  note = {Banco de Portugal WP 4/2013; PESSOA, Blanchard-Yaari NK DSGE}
}
@article{bettendorfAgeingConflictInterest2011,
  author = {Bettendorf, Leon and van der Horst, Albert and Draper, Nick and van Ewijk, Casper and de Mooij, Ruud and ter Rele, Harry},
  title = {Ageing and the Conflict of Interest Between Generations},
  journal = {De Economist},
  year = {2011}, volume = {159}, number = {3}, pages = {257--278},
  doi = {10.1007/s10645-011-9166-5}
}
@techreport{baksaFrameworkAssessingCosts2020,
  author = {Baksa, D{\'a}niel and Munk{\'a}csi, Zsuzsa and Nerlich, Carolin},
  title = {A Framework for Assessing the Costs of Pension Reform Reversals},
  institution = {European Central Bank}, type = {ECB Working Paper}, number = {2396},
  year = {2020}
}
@article{heerPopulationAgingSocial2020,
  author = {Heer, Burkhard and Polito, Vito and Wickens, Michael R.},
  title = {Population aging, social security and fiscal limits},
  journal = {Journal of Economic Dynamics and Control},
  year = {2020}, volume = {116}, pages = {103913},
  doi = {10.1016/j.jedc.2020.103913}
}
@article{diazgimenezPublicPensionsReforms2025,
  author = {D{\'i}az-Gim{\'e}nez, Javier and D{\'i}az-Saavedra, Juli{\'a}n},
  title = {Public pensions reforms: Financial and political sustainability},
  journal = {European Economic Review},
  year = {2025}, volume = {175}, pages = {104988}
}
@article{dollsFiscalSustainabilityDemographic2017,
  author = {Dolls, Mathias and Doorley, Karina and Paulus, Alari and Schneider, Hilmar and Siegloch, Sebastian and Sommer, Eric},
  title = {Fiscal sustainability and demographic change: a micro-approach for 27 EU countries},
  journal = {International Tax and Public Finance},
  year = {2017}, volume = {24}, number = {4}, pages = {575--615},
  doi = {10.1007/s10797-017-9462-3}
}
@article{storeslettenSustainingFiscalPolicy2000,
  author = {Storesletten, Kjetil},
  title = {Sustaining Fiscal Policy through Immigration},
  journal = {Journal of Political Economy},
  year = {2000}, volume = {108}, number = {2}, pages = {300--323},
  doi = {10.1086/262120}
}
@techreport{evansGameOverSimulating2012,
  author = {Evans, Richard W. and Kotlikoff, Laurence J. and Phillips, Kerk L.},
  title = {Game Over: Simulating Unsustainable Fiscal Policy},
  institution = {NBER}, type = {Working Paper}, number = {17917},
  year = {2012}, doi = {10.3386/w17917}
}
@incollection{derasmoWhatSustainablePublic2016,
  author = {D'Erasmo, Pablo and Mendoza, Enrique G. and Zhang, Jing},
  title = {What is a Sustainable Public Debt?},
  booktitle = {Handbook of Macroeconomics}, volume = {2}, pages = {2493--2597},
  editor = {Taylor, John B. and Uhlig, Harald}, publisher = {Elsevier},
  year = {2016}, doi = {10.1016/bs.hesmac.2016.03.013}
}
@incollection{debrunPublicDebtSustainability2019,
  author = {Debrun, Xavier and Ostry, Jonathan D. and Willems, Tim and Wyplosz, Charles},
  title = {Debt Sustainability},
  booktitle = {Sovereign Debt: A Guide for Economists and Practitioners},
  editor = {Abbas, S. Ali and Pienkowski, Alex and Rogoff, Kenneth}, publisher = {Oxford University Press},
  year = {2019}, note = {CEPR DP 14010 (title: Public Debt Sustainability)}
}
@article{bohnBehaviorUSPublic1998,
  author = {Bohn, Henning},
  title = {The Behavior of U.S. Public Debt and Deficits},
  journal = {Quarterly Journal of Economics},
  year = {1998}, volume = {113}, number = {3}, pages = {949--963},
  doi = {10.1162/003355398555793}
}
@article{bohnAreStationarityCointegration2007,
  author = {Bohn, Henning},
  title = {Are stationarity and cointegration restrictions really necessary for the intertemporal budget constraint?},
  journal = {Journal of Monetary Economics},
  year = {2007}, volume = {54}, number = {7}, pages = {1837--1847},
  doi = {10.1016/j.jmoneco.2006.12.012}
}
@article{mendozaInternationalEvidenceFiscal2008,
  author = {Mendoza, Enrique G. and Ostry, Jonathan D.},
  title = {International evidence on fiscal solvency: Is fiscal policy ``responsible''?},
  journal = {Journal of Monetary Economics},
  year = {2008}, volume = {55}, number = {6}, pages = {1081--1093},
  doi = {10.1016/j.jmoneco.2008.06.003}
}
@article{ghoshFiscalFatigueFiscal2013,
  author = {Ghosh, Atish R. and Kim, Jun I. and Mendoza, Enrique G. and Ostry, Jonathan D. and Qureshi, Mahvash S.},
  title = {Fiscal Fatigue, Fiscal Space and Debt Sustainability in Advanced Economies},
  journal = {Economic Journal},
  year = {2013}, volume = {123}, number = {566}, pages = {F4--F30},
  doi = {10.1111/ecoj.12010}
}
@article{collardSovereignDebtSustainability2015,
  author = {Collard, Fabrice and Habib, Michel and Rochet, Jean-Charles},
  title = {Sovereign Debt Sustainability in Advanced Economies},
  journal = {Journal of the European Economic Association},
  year = {2015}, volume = {13}, number = {3}, pages = {381--420},
  doi = {10.1111/jeea.12135}
}
@article{celasunPrimarySurplusBehavior2006,
  author = {Celasun, Oya and Debrun, Xavier and Ostry, Jonathan D.},
  title = {Primary Surplus Behavior and Risks to Fiscal Sustainability in Emerging Market Countries: A ``Fan-Chart'' Approach},
  journal = {IMF Staff Papers},
  year = {2006}, volume = {53}, number = {3}, pages = {401--425}
}
@article{zeniosRiskManagementSustainable2021,
  author = {Zenios, Stavros A. and Consiglio, Andrea and Athanasopoulou, Marialena and Moshammer, Edmund and Gavilan, Angel and Erce, Aitor},
  title = {Risk Management for Sustainable Sovereign Debt Financing},
  journal = {Operations Research},
  year = {2021}, volume = {69}, number = {3}, pages = {755--773},
  doi = {10.1287/opre.2020.2055}
}
@article{willemsSovereignDebtSustainability2022,
  author = {Willems, Tim and Zettelmeyer, Jeromin},
  title = {Sovereign Debt Sustainability and Central Bank Credibility},
  journal = {Annual Review of Financial Economics},
  year = {2022}, volume = {14}, pages = {75--93},
  doi = {10.1146/annurev-financial-112921-110812}
}
@article{mauroRMinusG2021,
  author = {Mauro, Paolo and Zhou, Jing},
  title = {r -- g < 0: Can We Sleep More Soundly?},
  journal = {IMF Economic Review},
  year = {2021}, volume = {69}, pages = {197--229},
  doi = {10.1057/s41308-020-00128-y}
}
@article{eichengreenSurplusAmbitionCan2016,
  author = {Eichengreen, Barry and Panizza, Ugo},
  title = {A surplus of ambition: can Europe rely on large primary surpluses to solve its debt problem?},
  journal = {Economic Policy},
  year = {2016}, volume = {31}, number = {85}, pages = {5--49},
  doi = {10.1093/epolic/eiv016}
}
@article{wyploszDebtSustainabilityAssessment2011,
  author = {Wyplosz, Charles},
  title = {Debt Sustainability Assessment: Mission Impossible},
  journal = {Review of Economics and Institutions},
  year = {2011}, volume = {2}, number = {3},
  doi = {10.5202/rei.v2i3.42}
}
@article{blanchardPublicDebtLow2019,
  author = {Blanchard, Olivier},
  title = {Public Debt and Low Interest Rates},
  journal = {American Economic Review},
  year = {2019}, volume = {109}, number = {4}, pages = {1197--1229},
  doi = {10.1257/aer.109.4.1197}
}
@book{blanchardFiscalPolicyLow2023,
  author = {Blanchard, Olivier},
  title = {Fiscal Policy under Low Interest Rates},
  publisher = {MIT Press}, address = {Cambridge, MA},
  year = {2023}
}
@techreport{reisConstraintPublicDebt2021,
  author = {Reis, Ricardo},
  title = {The constraint on public debt when r < g but g < m},
  institution = {Bank for International Settlements}, type = {BIS Working Paper}, number = {939},
  year = {2021}, note = {UNVERIFIED: no journal version found as of 2026-09}
}
@article{mehrotraDebtSustainabilityLow2021,
  author = {Mehrotra, Neil R. and Sergeyev, Dmitriy},
  title = {Debt sustainability in a low interest rate world},
  journal = {Journal of Monetary Economics},
  year = {2021}, volume = {124}, pages = {S1--S18},
  doi = {10.1016/j.jmoneco.2021.09.001},
  note = {UNVERIFIED pages}
}
@article{jiangUSPublicDebt2024,
  author = {Jiang, Zhengyang and Lustig, Hanno and Van Nieuwerburgh, Stijn and Xiaolan, Mindy Z.},
  title = {The U.S. Public Debt Valuation Puzzle},
  journal = {Econometrica},
  year = {2024}, volume = {92}, number = {4}, pages = {1309--1347},
  doi = {10.3982/ECTA20497},
  note = {UNVERIFIED volume/pages}
}
@techreport{jiangBondConvenienceYields2025,
  author = {Jiang, Zhengyang and Lustig, Hanno and Van Nieuwerburgh, Stijn and Xiaolan, Mindy Z.},
  title = {Bond Convenience Yields in the Eurozone Currency Union},
  institution = {NBER}, type = {Working Paper}, number = {34307}, year = {2025}
}
@article{brunnermeierSafeAssets2024,
  author = {Brunnermeier, Markus K. and Merkel, Sebastian and Sannikov, Yuliy},
  title = {Safe Assets},
  journal = {Journal of Political Economy},
  year = {2024}, volume = {132}, number = {11}, pages = {3603--3657}
}
@article{mianGoldilocksTheoryFiscal2025,
  author = {Mian, Atif and Straub, Ludwig and Sufi, Amir},
  title = {A Goldilocks Theory of Fiscal Deficits},
  journal = {American Economic Review},
  year = {2025}, volume = {115}, number = {12}, pages = {4253--4291},
  doi = {10.1257/aer.20220308}
}
@article{aguiarMicroRisksRobust2024,
  author = {Aguiar, Mark and Amador, Manuel and Arellano, Cristina},
  title = {Micro Risks and (Robust) Pareto-Improving Policies},
  journal = {American Economic Review},
  year = {2024}, volume = {114}, number = {11}, pages = {3669--3713},
  doi = {10.1257/aer.20230128}
}
@article{brummWhenInterestRates2024,
  author = {Brumm, Johannes and Feng, Xiangyu and Kotlikoff, Laurence J. and Kubler, Felix},
  title = {When Interest Rates Go Low, Should Public Debt Go High?},
  journal = {American Economic Journal: Macroeconomics},
  year = {2024}, volume = {16}, number = {4}, pages = {432--469},
  doi = {10.1257/mac.20230154}
}
@article{kocherlakotaPublicDebtBubbles2023,
  author = {Kocherlakota, Narayana R.},
  title = {Public Debt Bubbles in Heterogeneous Agent Models with Tail Risk},
  journal = {International Economic Review},
  year = {2023}, volume = {64}, number = {2}, pages = {491--509},
  doi = {10.1111/iere.12613}
}
@article{angeletosCanDeficitsFinance2024,
  author = {Angeletos, George-Marios and Lian, Chen and Wolf, Christian K.},
  title = {Can Deficits Finance Themselves?},
  journal = {Econometrica},
  year = {2024}, volume = {92}, number = {5}, pages = {1351--1390}
}
@techreport{caoCostlyIncreasesPublic2024,
  author = {Cao, Yongquan},
  title = {Costly Increases in Public Debt when r < g},
  institution = {International Monetary Fund}, type = {IMF Working Paper}, number = {24/10},
  year = {2024}, doi = {10.5089/9798400263620.001}
}
@article{trabandtLafferCurveRevisited2011,
  author = {Trabandt, Mathias and Uhlig, Harald},
  title = {The Laffer curve revisited},
  journal = {Journal of Monetary Economics},
  year = {2011}, volume = {58}, number = {4}, pages = {305--327},
  doi = {10.1016/j.jmoneco.2011.07.003}
}
@article{heimbergerPublicDebtRG2023,
  author = {Heimberger, Philipp},
  title = {Public debt and r-g risks in advanced economies: Eurozone versus stand-alone},
  journal = {Journal of International Money and Finance},
  year = {2023}, volume = {136}, pages = {102877},
  doi = {10.1016/j.jimonfin.2023.102877}
}
@techreport{corsettiDebtSustainabilityTerms2018,
  author = {Corsetti, Giancarlo and Erce, Aitor and Uy, Timothy},
  title = {Debt Sustainability and the Terms of Official Support},
  institution = {CEPR}, type = {Discussion Paper}, number = {13292}, year = {2018}
}
@article{corsettiOfficialSectorLending2020,
  author = {Corsetti, Giancarlo and Erce, Aitor and Uy, Timothy},
  title = {Official sector lending during the euro area crisis},
  journal = {Review of International Organizations},
  year = {2020}, volume = {15}, pages = {667--705},
  note = {UNVERIFIED pages}
}
@techreport{mimirConcessionalCreditLines2025,
  author = {Mimir, Yasin and {\"O}nder, Yasin K{\"u}r{\c{s}}at},
  title = {Concessional credit lines for sovereigns in financial distress},
  institution = {European Stability Mechanism}, type = {ESM Working Paper}, number = {70}, year = {2025}
}
@techreport{bellonTooMuchGood2026,
  author = {Bellon, Matthieu and Gnewuch, Matthias and Orlandi, Fabrice and Zavalloni, Luca},
  title = {Too Much of a Good Thing? Safe Assets, Spillovers, and the Lack of Fiscal Policy Coordination},
  institution = {European Stability Mechanism}, type = {ESM Working Paper}, number = {78}, year = {2026}
}
@techreport{hitajDebtSustainabilityAnalysis2025,
  author = {Hitaj, Ermal and Zavalloni, Luca and Zigraiova, Diana},
  title = {Debt sustainability analysis under uncertainty: mapping the past into the future},
  institution = {European Stability Mechanism}, type = {ESM Brief}, number = {5}, year = {2025}
}
@techreport{callegariSpendingCompositionFiscal2025,
  author = {Callegari, Giovanni and Michou, Maria and S{\l}awi{\'n}ska, Kamila and Tomasone, Sara and Zigraiova, Diana},
  title = {Spending Composition and Fiscal Consolidation: Enhancing Resilience in the Face of Economic Shocks},
  institution = {European Stability Mechanism}, type = {ESM Working Paper}, number = {73}, year = {2025}
}
@techreport{gabrieleDebtStocksMeet2017,
  author = {Gabriele, Carmine and Erce, Aitor and Athanasopoulou, Marialena and Rojas, Juan},
  title = {Debt Stocks Meet Gross Financing Needs: A Flow Perspective into Sustainability},
  institution = {European Stability Mechanism}, type = {ESM Working Paper}, number = {24}, year = {2017}
}
@incollection{lauwersSecurityWhatCost2026,
  author = {Lauwers, Alexandre and Michou, Maria and Ricci, Luca and Zavalloni, Luca},
  title = {Security at what cost? Defence spending, growth, and the fiscal arithmetic},
  booktitle = {Euro Area Stability Watch 2026}, chapter = {2},
  publisher = {European Stability Mechanism}, year = {2026},
  note = {UNVERIFIED first names of first and third authors}
}
@techreport{boullotAggregateDistributionalImplications2026,
  author = {Boullot, Mathieu and Cahn, Christophe and Challe, Edouard and Matheron, Julien},
  title = {Aggregate and Distributional Implications of a Military Buildup},
  institution = {CEPR}, type = {Discussion Paper}, number = {21270}, year = {2026}
}
@article{bokanMacroeconomicImpactsHigher2025,
  author = {Bokan, Nikola and Jacquinot, Pascal and Lalik, Magdalena and M{\"u}ller, Georg and Priftis, Romanos and Rigato, Rodolfo},
  title = {Macroeconomic impacts of higher defence spending: a model-based assessment},
  journal = {ECB Economic Bulletin},
  year = {2025}, number = {6}
}
@techreport{ilzetzkiGunsGrowthEconomic2025,
  author = {Ilzetzki, Ethan},
  title = {Guns and Growth: The Economic Consequences of Defense Buildups},
  institution = {Kiel Institute for the World Economy}, type = {Kiel Report}, number = {2}, year = {2025}
}
@article{ohanianMacroeconomicEffectsWar1997,
  author = {Ohanian, Lee E.},
  title = {The Macroeconomic Effects of War Finance in the United States: World War II and the Korean War},
  journal = {American Economic Review},
  year = {1997}, volume = {87}, number = {1}, pages = {23--40}
}
@article{barroDeterminationPublicDebt1979,
  author = {Barro, Robert J.},
  title = {On the Determination of the Public Debt},
  journal = {Journal of Political Economy},
  year = {1979}, volume = {87}, number = {5}, pages = {940--971},
  doi = {10.1086/260807}
}
@article{rameyGovernmentSpendingMultipliers2018,
  author = {Ramey, Valerie A. and Zubairy, Sarah},
  title = {Government Spending Multipliers in Good Times and in Bad: Evidence from US Historical Data},
  journal = {Journal of Political Economy},
  year = {2018}, volume = {126}, number = {2}, pages = {850--901},
  doi = {10.1086/696277}
}
@article{antolindiazLongRunEffectsGovernment2025,
  author = {Antol{\'i}n-D{\'i}az, Juan and Surico, Paolo},
  title = {The Long-Run Effects of Government Spending},
  journal = {American Economic Review},
  year = {2025}, volume = {115}, number = {7}, pages = {2376--2413},
  doi = {10.1257/aer.20231278}
}
@article{bomWhatHaveWe2014,
  author = {Bom, Pedro R. D. and Ligthart, Jenny E.},
  title = {What Have We Learned from Three Decades of Research on the Productivity of Public Capital?},
  journal = {Journal of Economic Surveys},
  year = {2014}, volume = {28}, number = {5}, pages = {889--916},
  doi = {10.1111/joes.12037}
}
@article{boehmGovernmentConsumptionInvestment2020,
  author = {Boehm, Christoph E.},
  title = {Government consumption and investment: Does the composition of purchases affect the multiplier?},
  journal = {Journal of Monetary Economics},
  year = {2020}, volume = {115}, pages = {80--93},
  doi = {10.1016/j.jmoneco.2019.05.003}
}
@article{bouakezPublicInvestmentTime2017,
  author = {Bouakez, Hafedh and Guillard, Michel and Roulleau-Pasdeloup, Jordan},
  title = {Public investment, time to build, and the zero lower bound},
  journal = {Review of Economic Dynamics},
  year = {2017}, volume = {23}, pages = {60--79},
  doi = {10.1016/j.red.2016.09.001}
}
@article{abiadMacroeconomicEffectsPublic2016,
  author = {Abiad, Abdul and Furceri, Davide and Topalova, Petia},
  title = {The macroeconomic effects of public investment: Evidence from advanced economies},
  journal = {Journal of Macroeconomics},
  year = {2016}, volume = {50}, pages = {224--240},
  doi = {10.1016/j.jmacro.2016.07.005}
}
@article{brincaFiscalMultipliers21st2016,
  author = {Brinca, Pedro and Holter, Hans A. and Krusell, Per and Malafry, Laurence},
  title = {Fiscal multipliers in the 21st century},
  journal = {Journal of Monetary Economics},
  year = {2016}, volume = {77}, pages = {53--69},
  doi = {10.1016/j.jmoneco.2015.09.005}
}
@article{brincaFiscalConsolidationPrograms2021,
  author = {Brinca, Pedro and Ferreira, Miguel H. and Franco, Francesco and Holter, Hans A. and Malafry, Laurence},
  title = {Fiscal Consolidation Programs and Income Inequality},
  journal = {International Economic Review},
  year = {2021}, volume = {62}, number = {1}, pages = {405--460},
  doi = {10.1111/iere.12482}
}
@article{ferriereHeterogeneousEffectsGovernment2025,
  author = {Ferriere, Axelle and Navarro, Gaston},
  title = {The Heterogeneous Effects of Government Spending: It's All About Taxes},
  journal = {Review of Economic Studies},
  year = {2025}, volume = {92}, number = {2}, pages = {1061--1125}
}
@article{auclertFiscalMonetaryPolicy2025,
  author = {Auclert, Adrien and Rognlie, Matthew and Straub, Ludwig},
  title = {Fiscal and Monetary Policy with Heterogeneous Agents},
  journal = {Annual Review of Economics},
  year = {2025}, volume = {17},
  note = {NBER WP 32991}
}
@article{yaredRisingGovernmentDebt2019,
  author = {Yared, Pierre},
  title = {Rising Government Debt: Causes and Solutions for a Decades-Old Trend},
  journal = {Journal of Economic Perspectives},
  year = {2019}, volume = {33}, number = {2}, pages = {115--140},
  doi = {10.1257/jep.33.2.115}
}
@article{auerbachGenerationalAccountingMeaningful1994,
  author = {Auerbach, Alan J. and Gokhale, Jagadeesh and Kotlikoff, Laurence J.},
  title = {Generational Accounting: A Meaningful Way to Evaluate Fiscal Policy},
  journal = {Journal of Economic Perspectives},
  year = {1994}, volume = {8}, number = {1}, pages = {73--94},
  doi = {10.1257/jep.8.1.73}
}
@incollection{auerbachGenerationalAccountsMeaningful1991,
  author = {Auerbach, Alan J. and Gokhale, Jagadeesh and Kotlikoff, Laurence J.},
  title = {Generational Accounts: A Meaningful Alternative to Deficit Accounting},
  booktitle = {Tax Policy and the Economy}, volume = {5}, pages = {55--110},
  editor = {Bradford, David}, publisher = {MIT Press}, year = {1991}
}
@incollection{kotlikoffGenerationalPolicy2002,
  author = {Kotlikoff, Laurence J.},
  title = {Generational Policy},
  booktitle = {Handbook of Public Economics}, volume = {4}, chapter = {27}, pages = {1873--1932},
  editor = {Auerbach, Alan J. and Feldstein, Martin}, publisher = {Elsevier}, year = {2002}
}
@techreport{greenGeneralRelativityFiscal2006,
  author = {Green, Jerry and Kotlikoff, Laurence J.},
  title = {On the General Relativity of Fiscal Language},
  institution = {NBER}, type = {Working Paper}, number = {12344}, year = {2006}
}
@article{buiterMeasurementPublicSector1983,
  author = {Buiter, Willem H.},
  title = {Measurement of the Public Sector Deficit and Its Implications for Policy Evaluation and Design},
  journal = {IMF Staff Papers},
  year = {1983}, volume = {30}, number = {2}, pages = {306--349}
}
@article{buiterGuidePublicSector1985,
  author = {Buiter, Willem H.},
  title = {A Guide to Public Sector Debt and Deficits},
  journal = {Economic Policy},
  year = {1985}, volume = {1}, number = {1}, pages = {13--61}
}
@article{blejerMeasurementFiscalDeficits1991,
  author = {Blejer, Mario I. and Cheasty, Adrienne},
  title = {The Measurement of Fiscal Deficits: Analytical and Methodological Issues},
  journal = {Journal of Economic Literature},
  year = {1991}, volume = {29}, number = {4}, pages = {1644--1678}
}
@article{bohnBudgetDeficitsGovernment1992,
  author = {Bohn, Henning},
  title = {Budget deficits and government accounting},
  journal = {Carnegie-Rochester Conference Series on Public Policy},
  year = {1992}, volume = {37}, pages = {1--83}
}
@article{kaierNewFiguresUnfunded2015,
  author = {Kaier, Klaus and M{\"u}ller, Christoph},
  title = {New figures on unfunded public pension entitlements across Europe: concept, results and applications},
  journal = {Empirica},
  year = {2015}, volume = {42}, number = {4}, pages = {865--895},
  doi = {10.1007/s10663-015-9285-3}
}
@techreport{koshimaCostFuturePolicy2021,
  author = {Koshima, Yugo and Harris, Jason and Tieman, Alexander F. and De Sanctis, Alessandro},
  title = {The Cost of Future Policy: Intertemporal Public Sector Balance Sheets in the G7},
  institution = {International Monetary Fund}, type = {IMF Working Paper}, number = {21/128}, year = {2021}
}
@techreport{alvesPublicSectorBalance2020,
  author = {Alves, Miguel and De Clerck, Sage and Gamboa-Arbelaez, Juliana},
  title = {Public Sector Balance Sheet Database: Overview and Guide for Compilers and Users},
  institution = {International Monetary Fund}, type = {IMF Working Paper}, number = {20/130}, year = {2020}
}
@incollection{deboeckTakingStockImplicit2020,
  author = {Deboeck, Ben and Eckefeldt, Per},
  title = {Taking stock of implicit pension liabilities},
  booktitle = {Quarterly Report on the Euro Area}, volume = {19}, number = {2}, chapter = {III},
  publisher = {European Commission, DG ECFIN}, year = {2020}
}
@techreport{arevaloIntergenerationalDimensionFiscal2019,
  author = {Ar{\'e}valo, Pedro and Berti, Katia and Caretta, Alessandra and Eckefeldt, Per},
  title = {The Intergenerational Dimension of Fiscal Sustainability},
  institution = {European Commission, DG ECFIN}, type = {Discussion Paper}, number = {112}, year = {2019}
}
@article{novymarxPublicPensionPromises2011,
  author = {Novy-Marx, Robert and Rauh, Joshua},
  title = {Public Pension Promises: How Big Are They and What Are They Worth?},
  journal = {Journal of Finance},
  year = {2011}, volume = {66}, number = {4}, pages = {1211--1249},
  doi = {10.1111/j.1540-6261.2011.01664.x}
}
@article{gourinchasAnalyticsGreekCrisis2017,
  author = {Gourinchas, Pierre-Olivier and Philippon, Thomas and Vayanos, Dimitri},
  title = {The Analytics of the Greek Crisis},
  journal = {NBER Macroeconomics Annual},
  year = {2017}, volume = {31}, pages = {1--81},
  doi = {10.1086/690239}
}
@article{houseAusterityAftermathGreat2020,
  author = {House, Christopher L. and Proebsting, Christian and Tesar, Linda L.},
  title = {Austerity in the aftermath of the Great Recession},
  journal = {Journal of Monetary Economics},
  year = {2020}, volume = {115}, pages = {37--63},
  doi = {10.1016/j.jmoneco.2019.05.004}
}
@article{perottiHumanSideAusterity2021,
  author = {Perotti, Roberto},
  title = {The human side of austerity: health spending and outcomes during the Greek crisis},
  journal = {Economic Policy},
  year = {2021}, volume = {36}, number = {105}, pages = {121--164},
  doi = {10.1093/epolic/eiaa027},
  note = {UNVERIFIED pages and DOI}
}
@article{leventiDisentanglingAnnuitiesTransfers2020,
  author = {Leventi, Chrysa and Matsaganis, Manos},
  title = {Disentangling annuities and transfers: The case of Greek retirement benefits},
  journal = {European Journal of Social Security},
  year = {2020}, volume = {22}, number = {3},
  doi = {10.1177/1388262720941879}
}
@article{economidesAusterityAssistanceInstitutions2021,
  author = {Economides, George and Papageorgiou, Dimitris and Philippopoulos, Apostolis},
  title = {Austerity, Assistance and Institutions: Lessons from the Greek Sovereign Debt Crisis},
  journal = {Open Economies Review},
  year = {2021}, volume = {32}, number = {3}, pages = {435--478},
  doi = {10.1007/s11079-020-09613-3}
}
@article{dellasFiscalPolicyInformal2024,
  author = {Dellas, Harris and Malliaropulos, Dimitris and Papageorgiou, Dimitris and Vourvachaki, Evangelia},
  title = {Fiscal policy with an informal sector},
  journal = {Journal of Economic Dynamics and Control},
  year = {2024}, volume = {160}
}
@incollection{panageasPensionsArrestingRace2017,
  author = {Panageas, Stavros and Tinios, Platon},
  title = {Pensions: Arresting a Race to the Bottom},
  booktitle = {Beyond Austerity: Reforming the Greek Economy},
  editor = {Meghir, Costas and Pissarides, Christopher A. and Vayanos, Dimitri and Vettas, Nikolaos},
  publisher = {MIT Press}, year = {2017}
}
@incollection{kanavosReformingHealthCare2017,
  author = {Kanavos, Panos and Souliotis, Kyriakos},
  title = {Reforming Health Care in Greece: Balancing Fiscal Adjustment with Health Care Needs},
  booktitle = {Beyond Austerity: Reforming the Greek Economy},
  editor = {Meghir, Costas and Pissarides, Christopher A. and Vayanos, Dimitri and Vettas, Nikolaos},
  publisher = {MIT Press}, year = {2017}
}
@article{kentikelenisGreecesHealthCrisis2014,
  author = {Kentikelenis, Alexander and Karanikolos, Marina and Reeves, Aaron and McKee, Martin and Stuckler, David},
  title = {Greece's health crisis: from austerity to denialism},
  journal = {The Lancet},
  year = {2014}, volume = {383}, number = {9918}, pages = {748--753},
  doi = {10.1016/S0140-6736(13)62291-6}
}
@techreport{symeonidisGreekPensionReform2016,
  author = {Symeonidis, Georgios},
  title = {The Greek Pension Reform Strategy 2010--2016},
  institution = {World Bank}, type = {Social Protection and Labor Discussion Paper}, number = {1601}, year = {2016}
}
@techreport{delamaisonneuvePublicSpendingHealth2013,
  author = {de la Maisonneuve, Christine and Oliveira Martins, Joaquim},
  title = {Public Spending on Health and Long-term Care: A New Set of Projections},
  institution = {OECD}, type = {OECD Economic Policy Paper}, number = {6}, year = {2013}
}
@techreport{przywaraProjectingFutureHealth2010,
  author = {Przywara, Bartosz},
  title = {Projecting future health care expenditure at European level: drivers, methodology and main results},
  institution = {European Commission, DG ECFIN}, type = {European Economy Economic Papers}, number = {417}, year = {2010}
}
@techreport{medeirosEstimatingDriversProjecting2013,
  author = {Medeiros, Jo{\~a}o and Schwierz, Christoph},
  title = {Estimating the drivers and projecting long-term public health expenditure in the European Union: Baumol's ``cost disease'' revisited},
  institution = {European Commission, DG ECFIN}, type = {European Economy Economic Papers}, number = {507}, year = {2013}
}
@techreport{europeancommission2024AgeingReport2024,
  author = {{European Commission}},
  title = {2024 Ageing Report: Economic and Budgetary Projections for the EU Member States (2022--2070)},
  institution = {European Commission, DG ECFIN}, type = {Institutional Paper}, number = {279}, year = {2024}
}
@techreport{europeancommissionDebtSustainabilityMonitor2026,
  author = {{European Commission}},
  title = {Debt Sustainability Monitor 2025},
  institution = {European Commission, DG ECFIN}, type = {Institutional Paper}, number = {332}, year = {2026}
}
@techreport{imfStaffGuidanceNote2022,
  author = {{International Monetary Fund}},
  title = {Staff Guidance Note on the Sovereign Risk and Debt Sustainability Framework for Market Access Countries},
  institution = {International Monetary Fund}, type = {Policy Paper}, number = {2022/039}, year = {2022}
}
@techreport{bouabdallahDebtSustainabilityAnalysis2017,
  author = {Bouabdallah, Othman and Checherita-Westphal, Cristina and Warmedinger, Thomas and de Stefani, Roberta and Drudi, Francesco and Setzer, Ralph and Westphal, Andreas},
  title = {Debt sustainability analysis for euro area sovereigns: a methodological framework},
  institution = {European Central Bank}, type = {Occasional Paper}, number = {185}, year = {2017}
}
@article{denardiWhyDoElderly2010,
  author = {De Nardi, Mariacristina and French, Eric and Jones, John Bailey},
  title = {Why Do the Elderly Save? The Role of Medical Expenses},
  journal = {Journal of Political Economy},
  year = {2010}, volume = {118}, number = {1}, pages = {39--75},
  doi = {10.1086/651674}
}
@article{denardiSavingsAfterRetirement2016,
  author = {De Nardi, Mariacristina and French, Eric and Jones, John Bailey},
  title = {Savings After Retirement: A Survey},
  journal = {Annual Review of Economics},
  year = {2016}, volume = {8}, pages = {177--204},
  doi = {10.1146/annurev-economics-080315-015127}
}
@article{jungAgingHealthFinancing2017,
  author = {Jung, Juergen and Tran, Chung and Chambers, Matthew},
  title = {Aging and health financing in the U.S.: A general equilibrium analysis},
  journal = {European Economic Review},
  year = {2017}, volume = {100}, pages = {428--462},
  doi = {10.1016/j.euroecorev.2017.09.005}
}
@article{hsuPopulationAgingHealth2019,
  author = {Hsu, Minchung and Yamada, Tomoaki},
  title = {Population Aging, Health Care, and Fiscal Policy Reform: The Challenges for Japan},
  journal = {Scandinavian Journal of Economics},
  year = {2019}, volume = {121}, number = {2}, pages = {547--577},
  doi = {10.1111/sjoe.12280}
}
@article{derasmoDistributionalIncentivesEquilibrium2016,
  author = {D'Erasmo, Pablo and Mendoza, Enrique G.},
  title = {Distributional Incentives in an Equilibrium Model of Domestic Sovereign Default},
  journal = {Journal of the European Economic Association},
  year = {2016}, volume = {14}, number = {1}, pages = {7--44},
  doi = {10.1111/jeea.12168}
}
@article{derasmoHistoryRememberedOptimal2021,
  author = {D'Erasmo, Pablo and Mendoza, Enrique G.},
  title = {History remembered: Optimal sovereign default on domestic and external debt},
  journal = {Journal of Monetary Economics},
  year = {2021}, volume = {117}, pages = {969--989},
  doi = {10.1016/j.jmoneco.2020.07.006}
}
@techreport{tranxuanSovereignDebtSustainability2026,
  author = {Tran-Xuan, Monica},
  title = {Sovereign Debt Sustainability and Redistribution},
  institution = {International Monetary Fund}, type = {IMF Working Paper}, year = {2026},
  doi = {10.5089/9798229039055.001}
}
@article{checheritawestphalGovernmentsPaymentDiscipline2016,
  author = {Checherita-Westphal, Cristina and Klemm, Alexander and Viefers, Paul},
  title = {Governments' payment discipline: The macroeconomic impact of public payment delays and arrears},
  journal = {Journal of Macroeconomics},
  year = {2016}, volume = {47}, pages = {147--165},
  doi = {10.1016/j.jmacro.2015.12.003}
}
@article{abrahamOptimalDesignFinancial2025,
  author = {{\'A}brah{\'a}m, {\'A}rp{\'a}d and Carceles-Poveda, Eva and Liu, Yan and Marimon, Ramon},
  title = {On the Optimal Design of a Financial Stability Fund},
  journal = {Review of Economic Studies},
  year = {2025}, doi = {10.1093/restud/rdaf076}
}
@article{liuMakingSovereignDebt2023,
  author = {Liu, Yan and Marimon, Ramon and Wicht, Adrien},
  title = {Making sovereign debt safe with a financial stability fund},
  journal = {Journal of International Economics},
  year = {2023}, volume = {145}, pages = {103834},
  doi = {10.1016/j.jinteco.2023.103834}
}
@article{callegariLenderLastResort2023,
  author = {Callegari, Giovanni and Marimon, Ramon and Wicht, Adrien and Zavalloni, Luca},
  title = {On a Lender of Last Resort with a Central Bank and a Stability Fund},
  journal = {Review of Economic Dynamics},
  year = {2023}, volume = {50}, pages = {106--130},
  doi = {10.1016/j.red.2023.07.004},
  note = {UNVERIFIED DOI}
}
@article{abrahamDesignEuropeanUnemployment2023,
  author = {{\'A}brah{\'a}m, {\'A}rp{\'a}d and Brogueira de Sousa, Jo{\~a}o and Marimon, Ramon and Mayr, Lukas},
  title = {On the design of a European Unemployment Insurance System},
  journal = {European Economic Review},
  year = {2023}, volume = {156}, pages = {104469},
  doi = {10.1016/j.euroecorev.2023.104469}
}
@article{diazsaavedraWorkersBackpackAlternative2023,
  author = {D{\'i}az-Saavedra, Juli{\'a}n and Marimon, Ramon and Brogueira de Sousa, Jo{\~a}o},
  title = {A Worker's Backpack as an Alternative to PAYG Pension Systems},
  journal = {Journal of the European Economic Association},
  year = {2023}, volume = {21}, number = {5}, pages = {1944--1993},
  doi = {10.1093/jeea/jvad021}
}
@article{diazsaavedraIntroducingAustrianBackpack2022,
  author = {D{\'i}az-Saavedra, Juli{\'a}n and Marimon, Ramon and Brogueira de Sousa, Jo{\~a}o},
  title = {Introducing an Austrian backpack in Spain},
  journal = {SERIEs},
  year = {2022}, volume = {13}, pages = {513--556},
  doi = {10.1007/s13209-022-00263-x}
}
@article{diazgimenezDelayingRetirementSpain2009,
  author = {D{\'i}az-Gim{\'e}nez, Javier and D{\'i}az-Saavedra, Juli{\'a}n},
  title = {Delaying Retirement in Spain},
  journal = {Review of Economic Dynamics},
  year = {2009}, volume = {12}, number = {1}, pages = {147--167},
  doi = {10.1016/j.red.2008.06.001}
}

% --- Added 2026-09-11 by the Semantic Scholar repeat pass (55 entries) ---
@article{sundramFiscalPolicySmall2026,
  author  = {Sundram, Jacob},
  title   = {Fiscal Policy in Small Open Economies: The International Intertemporal {Keynesian} Cross},
  journal = {Review of Economics and Statistics},
  year    = {2026},
  doi     = {10.1162/rest.a.1832},
  note    = {S2 paperId 37998bb6f2bcdc5f3729bfa21c2f8ebe469979ef}
}

@article{fernandezbastidasMindNarrowingGender2025,
  author  = {Fern{\'a}ndez-Bastidas, Roc{\'\i}o and Pycroft, Jonathan},
  title   = {Mind the (Narrowing) Gender Gap: Implications of Labour Market Changes for {EU} Pension Sustainability},
  journal = {Economic Modelling},
  year    = {2025},
  doi     = {10.1016/j.econmod.2025.107185},
  note    = {S2 paperId e650d5f3d31a66454fb13efa1d5f9e0e14f57618}
}

@article{linShiftingPayasyougoIndividual2021,
  author  = {Lin, Hsuan-Chih and Tanaka, Atsuko and Wu, Po-Shyan},
  title   = {Shifting from Pay-as-You-Go to Individual Retirement Accounts: A Path to a Sustainable Pension System},
  journal = {Journal of Macroeconomics},
  year    = {2021},
  doi     = {10.1016/j.jmacro.2021.103329},
  note    = {S2 paperId 5cc004400279769a7f88f96cfe80cb970cb740e6}
}

@article{dimakopoulouECBsPolicyRecovery2022,
  author  = {Dimakopoulou, Vasiliki and Economides, George and Philippopoulos, Apostolis},
  title   = {The {ECB}'s Policy, the {Recovery Fund} and the Importance of Trust and Fiscal Corrections: The Case of {Greece}},
  journal = {Economic Modelling},
  year    = {2022},
  doi     = {10.1016/j.econmod.2022.105846},
  note    = {S2 paperId 7c1d25a913f8e71cb306b0a342300c4f740ced24}
}

@techreport{economidesMacroeconomicPolicyLessons2020,
  author      = {Economides, George and Papageorgiou, Dimitris and Philippopoulos, Apostolis},
  title       = {Macroeconomic Policy Lessons for {Greece} from the Debt Crisis},
  institution = {SSRN},
  type        = {Working Paper},
  year        = {2020},
  doi         = {10.2139/ssrn.3570290},
  note        = {S2 paperId 5dd5c8e32a18e5245432860db6e2ef1569c10ed5}
}

@article{wangFiscalStimulusHighdebt2021,
  author  = {Wang, Shu-Ling},
  title   = {Fiscal Stimulus in a High-Debt Economy? A {DSGE} Analysis},
  journal = {Economic Modelling},
  year    = {2021},
  doi     = {10.1016/j.econmod.2021.02.009},
  note    = {S2 paperId b6e20fb54a23a9505316166f275c838931869b16; country calibration not verified}
}

@article{hindriksSustainabilityPensionReforms2025,
  author  = {Hindriks, Jean and {\c{C}}etin, Sefane},
  title   = {Sustainability of Pension Reforms: An {EU}-Wide Political Stress Test},
  journal = {Journal of Pension Economics and Finance},
  year    = {2025},
  doi     = {10.1017/s1474747225100103},
  note    = {S2 paperId d51c49293de5cef4f164508e8a17626b7cdd82b9}
}

@article{ramosherreraFiscalSustainabilityAging2020,
  author  = {Ramos-Herrera, Mar{\'\i}a del Carmen and Sosvilla-Rivero, Sim{\'o}n},
  title   = {Fiscal Sustainability in Aging Societies: Evidence from Euro Area Countries},
  journal = {Sustainability},
  volume  = {12},
  number  = {24},
  pages   = {10276},
  year    = {2020},
  doi     = {10.3390/su122410276},
  note    = {S2 paperId 3996236b3aedc54c46c8f883f78078ae40090913}
}

@article{miyazawaCapitalMarketIntegration2019,
  author  = {Miyazawa, Kazuo and Ogawa, Hikaru and Tamai, Toshiki},
  title   = {Capital Market Integration and Fiscal Sustainability},
  journal = {European Economic Review},
  year    = {2019},
  doi     = {10.1016/j.euroecorev.2019.103305},
  note    = {S2 paperId c87e3bcce58c71842de69674b664e3cffb66dfeb; two-country endogenous-growth model, not OLG}
}

@techreport{baksaAgingPensionReforms2016,
  author      = {Baksa, D{\'a}niel and Munk{\'a}csi, Zsuzsa},
  title       = {Aging, (Pension) Reforms and the Shadow Economy in {Southern Europe}},
  institution = {Bank of Lithuania},
  type        = {Working Paper},
  number      = {32},
  year        = {2016},
  note        = {RePEc:lie:wpaper:32; S2 paperId f8df1fbe678efcf7c9dd6c9e966aee3e91f7b825; abstract elided in S2, country set unverified}
}

@techreport{bernardinoCostsBuildingWalls2024,
  author      = {Bernardino, Tiago and Franco, Francesco and Teles Morais, Lu{\'\i}s},
  title       = {The Costs of Building Walls: Immigration and the Fiscal Burden of Aging in {Europe}},
  institution = {SSRN},
  type        = {Working Paper},
  year        = {2024},
  doi         = {10.2139/ssrn.4932922},
  note        = {S2 paperId d6b7d8b72d2d7be6d3cad0de6b5cbe056b18caa2}
}

@article{magnaniGeneralEquilibriumEvaluation2011,
  author  = {Magnani, Riccardo},
  title   = {A General Equilibrium Evaluation of the Sustainability of the New Pension Reforms in {Italy}},
  journal = {Research in Economics},
  volume  = {65},
  number  = {1},
  year    = {2011},
  doi     = {10.1016/j.rie.2010.02.001},
  note    = {S2 paperId 9bfe8fdb41dc4d0d4d72c90e002f2885099f3877}
}

@techreport{borschsupanOldEuropeAges2010,
  author      = {B{\"o}rsch-Supan, Axel and Ludwig, Alexander},
  title       = {Old {Europe} Ages: Reforms and Reform Backlashes},
  institution = {National Bureau of Economic Research},
  type        = {Working Paper},
  number      = {15744},
  year        = {2010},
  doi         = {10.3386/w15744},
  note        = {S2 paperId fb0eec0c4d961cabc1075fd6b25297bfa482c1f6}
}

@techreport{abelRunningPrimaryDeficits2022,
  author      = {Abel, Andrew B. and Panageas, Stavros},
  title       = {Running Primary Deficits Forever in a Dynamically Efficient Economy: Feasibility and Optimality},
  institution = {National Bureau of Economic Research},
  type        = {Working Paper},
  number      = {30554},
  year        = {2022},
  doi         = {10.3386/w30554},
  note        = {S2 paperId 62df949a86e038f67c82d88d28143a8c7402cc69}
}

@article{angeletosPublicDebtPrivate2023,
  author  = {Angeletos, George-Marios and Collard, Fabrice and Dellas, Harris},
  title   = {Public Debt as Private Liquidity: Optimal Policy},
  journal = {Journal of Political Economy},
  year    = {2023},
  doi     = {10.1086/725170},
  note    = {S2 paperId e909953b5fe25e2a8b0082e69a64573d0c96e8db (S2 year 2016 = WP; year from DOI)}
}

@article{brummAreDeficitsFree2022,
  author  = {Brumm, Johannes and Feng, Xiangyu and Kotlikoff, Laurence and Kubler, Felix},
  title   = {Are Deficits Free?},
  journal = {Journal of Public Economics},
  year    = {2022},
  doi     = {10.1016/j.jpubeco.2022.104627},
  note    = {S2 paperId 4bee94fbb6cfe0cb9771e627b71c5d43d98e81e4}
}

@article{piguillemStickySpendingSequestration2024,
  author  = {Piguillem, Facundo and Riboni, Alessandro},
  title   = {Sticky Spending, Sequestration, and Government Debt},
  journal = {American Economic Review},
  year    = {2024},
  doi     = {10.1257/aer.20210935},
  note    = {S2 paperId 75513f64d9eccc8d94fba88eb4c2003441dbac21}
}

@article{miaoFiscalMonetaryPolicy2024,
  author  = {Miao, Jianjun and Su, Dongling},
  title   = {Fiscal and Monetary Policy Interactions in a Model with Low Interest Rates},
  journal = {American Economic Journal: Macroeconomics},
  year    = {2024},
  doi     = {10.1257/mac.20220232},
  note    = {S2 paperId b9a783d860c5deecced145fd780206866ae67ae0}
}

@article{mitchenerSovereignDebtTwentyfirst2023,
  author  = {Mitchener, Kris James and Trebesch, Christoph},
  title   = {Sovereign Debt in the Twenty-First Century},
  journal = {Journal of Economic Literature},
  year    = {2023},
  doi     = {10.1257/jel.20211362},
  note    = {S2 paperId 66e215006e0d3ee8e6c9ef910b267406ee34fed4}
}

@techreport{jiangMeasuringUSFiscal2022,
  author      = {Jiang, Zhengyang and Lustig, Hanno and Van Nieuwerburgh, Stijn and Xiaolan, Mindy Z.},
  title       = {Measuring {US} Fiscal Capacity Using Discounted Cash Flow Analysis},
  institution = {SSRN},
  type        = {Working Paper},
  year        = {2022},
  doi         = {10.2139/ssrn.4058541},
  note        = {S2 paperId 83d5106f4209c70fe34593034eae3a5eb7428e14}
}

@techreport{elenevCanMonetaryPolicy2021,
  author      = {Elenev, Vadim and Landvoigt, Tim and Shultz, Patrick J. and Van Nieuwerburgh, Stijn},
  title       = {Can Monetary Policy Create Fiscal Capacity?},
  institution = {SSRN},
  type        = {Working Paper},
  year        = {2021},
  doi         = {10.2139/ssrn.3896402},
  note        = {S2 paperId 71d554df30c4fb0ad395011600779d2740afbc4b}
}

@article{reinhartPitfallsExternalDependence2015,
  author  = {Reinhart, Carmen M. and Trebesch, Christoph},
  title   = {The Pitfalls of External Dependence: {Greece}, 1829--2015},
  journal = {Brookings Papers on Economic Activity},
  year    = {2015},
  doi     = {10.1353/eca.2015.0000},
  note    = {S2 paperId 4825df06b9e83f530cfa30794e1707e76dd42a73}
}

@techreport{erceSelectiveSovereignDefaults2012,
  author      = {Erce, Aitor and Mallucci, Enrico},
  title       = {Selective Sovereign Defaults},
  institution = {Federal Reserve Bank of Dallas, Globalization and Monetary Policy Institute},
  type        = {Working Paper},
  number      = {127},
  year        = {2012},
  doi         = {10.24149/gwp127},
  note        = {S2 paperId 38d32b14a8ee60f6945a9b96c1c8d1c938fda5b5; authors and year as in the S2 record}
}

@article{shakhnovSovereignDebtIssuance2026,
  author  = {Shakhnov, Kirill and Paczos, Wojtek},
  title   = {Sovereign Debt Issuance and Selective Default},
  journal = {Macroeconomic Dynamics},
  year    = {2026},
  doi     = {10.1017/S1365100525100825},
  note    = {S2 paperId b727d2b355aaee83236084d4fcbd8a5a37855230}
}

@techreport{asonumaSovereignDebtOverhang2019,
  author      = {Asonuma, Tamon and Joo, Hyungseok},
  title       = {Sovereign Debt Overhang, Expenditure Composition and Debt Restructurings},
  institution = {SSRN},
  type        = {Working Paper},
  year        = {2019},
  doi         = {10.2139/ssrn.3456125},
  note        = {S2 paperId dc356f9da57a8437d9bf447c8c6c1f959fb76cb7}
}

@techreport{asonumaPublicCapitalFiscal2021,
  author      = {Asonuma, Tamon and Joo, Hyungseok},
  title       = {Public Capital and Fiscal Constraint in Sovereign Debt Crises},
  institution = {SSRN},
  type        = {Working Paper},
  year        = {2021},
  doi         = {10.2139/ssrn.3862268},
  note        = {S2 paperId 4d1d9151ea6d55fdc53417c78ceb164e99f9d49d}
}

@article{senQuantifyingEffectsAgeing2026,
  author  = {{\c{S}}en, H{\"u}seyin and Kaya, Ay{\c{s}}e},
  title   = {Quantifying the Effects of Ageing on the Composition of Government Spending},
  journal = {Policy Studies},
  year    = {2026},
  doi     = {10.1080/01442872.2026.2654505},
  note    = {S2 paperId 10ab4f6f4ded1b2009baf3fa9210112267ac5951}
}

@article{symeonidisEnhancingPensionAdequacy2021,
  author  = {Symeonidis, Georgios and Tinios, Platon and Xenos, Panos},
  title   = {Enhancing Pension Adequacy While Reducing the Fiscal Budget and Creating Essential Capital for Domestic Investments and Growth: Analysing the Risks and Outcomes in the Case of {Greece}},
  journal = {Risks},
  volume  = {9},
  number  = {1},
  pages   = {8},
  year    = {2021},
  doi     = {10.3390/risks9010008},
  note    = {S2 corpusId 234450881}
}

@techreport{nektariosGreekPensionReforms2019,
  author      = {Nektarios, Milton and Tinios, Platon},
  title       = {The {Greek} Pension Reforms},
  institution = {World Bank},
  year        = {2019},
  doi         = {10.1596/31630},
  note        = {S2 corpusId 159414981}
}

@article{beetsmaSomeIntergenerationalArithmetic2025,
  author  = {Beetsma, Roel and Busse, Matthias and Larch, Martin and Romp, Ward},
  title   = {Some Intergenerational Arithmetic to Control Public Debt in the {EU}},
  journal = {Journal of Pension Economics and Finance},
  year    = {2025},
  doi     = {10.1017/s1474747225100085},
  note    = {S2 corpusId 280953080}
}

@article{castanerTransformingEurostatsTable2025,
  author  = {Casta{\~n}er, Anna and Garvey, Anne M. and P{\'e}rez-Salamero Gonz{\'a}lez, Juan Manuel and Vidal-Meli{\'a}, Carlos},
  title   = {Transforming {Eurostat}'s {Table 29} into an Actuarial Balance Sheet: A Net Worth Approach to Assessing Public Pension Solvency},
  journal = {Journal of Risk and Financial Management},
  volume  = {18},
  number  = {9},
  pages   = {528},
  year    = {2025},
  doi     = {10.3390/jrfm18090528},
  note    = {S2 corpusId 281512101}
}

@article{bredeFinlandsPublicSector2019,
  author  = {Brede, Maren and Henn, Christian},
  title   = {{Finland}'s Public Sector Balance Sheet},
  journal = {Baltic Journal of Economics},
  year    = {2019},
  doi     = {10.1080/1406099X.2019.1585062},
  note    = {S2 corpusId 159339496}
}

@techreport{gillMakingLowincomeCountry2023,
  author      = {Gill, Indermit and Pinto, Brian},
  title       = {Making the Low-Income Country Debt Sustainability Framework Fit for Purpose},
  institution = {World Bank},
  type        = {Policy Research Working Paper},
  number      = {10602},
  year        = {2023},
  doi         = {10.1596/1813-9450-10602},
  note        = {S2 corpusId 265267318}
}

@article{carterGunsButterGrowth2020,
  author  = {Carter, Jeff and Ondercin, Heather L. and Palmer, Glenn},
  title   = {Guns, Butter, and Growth: The Consequences of Military Spending Reconsidered},
  journal = {Political Research Quarterly},
  year    = {2020},
  doi     = {10.1177/1065912919890417},
  note    = {S2 corpusId 212983801}
}

@article{bolouriInterdependentPreferencesFinancing2025,
  author  = {Bolouri, Armin A. and Lohse, Timm H. and Qari, Salmai},
  title   = {Interdependent Preferences for Financing and Providing Public Goods---The Case of National Defense},
  journal = {Kyklos},
  year    = {2025},
  doi     = {10.1111/kykl.12442},
  note    = {S2 corpusId 276888717}
}

@incollection{weizsackerIntroductionPrivateWealth2021,
  author    = {von Weizs{\"a}cker, Carl Christian and Kr{\"a}mer, Hagen M.},
  title     = {Introduction: Private Wealth and Public Debt},
  booktitle = {Saving and Investment in the Twenty-First Century},
  publisher = {Springer},
  year      = {2021},
  doi       = {10.1007/978-3-030-75031-2_1},
  note      = {S2 corpusId 237973584}
}

@article{utrerogonzalezDefenceSpendingInstitutional2019,
  author  = {Utrero-Gonz{\'a}lez, Natalia and Hromcov{\'a}, Jana and Callado-Mu{\~n}oz, Francisco J.},
  title   = {Defence Spending, Institutional Environment and Economic Growth: Case of {NATO}},
  journal = {Defence and Peace Economics},
  year    = {2019},
  doi     = {10.1080/10242694.2017.1400292},
  note    = {S2 corpusId 158154106}
}
@article{denardiProjectedUSDemographics1999,
  author  = {De Nardi, Mariacristina and {\.I}mrohoro{\u{g}}lu, Selahattin and Sargent, Thomas J.},
  title   = {Projected {U.S.} Demographics and Social Security},
  journal = {Review of Economic Dynamics},
  volume  = {2},
  number  = {3},
  year    = {1999},
  doi     = {10.1006/redy.1999.0067}
}

@article{kopeckyImpactMedicalNursing2014,
  author  = {Kopecky, Karen A. and Koreshkova, Tatyana},
  title   = {The Impact of Medical and Nursing Home Expenses on Savings},
  journal = {American Economic Journal: Macroeconomics},
  volume  = {6},
  number  = {3},
  year    = {2014},
  doi     = {10.1257/mac.6.3.29}
}

@article{braunOldSickAlone2017,
  author  = {Braun, R. Anton and Kopecky, Karen A. and Koreshkova, Tatyana},
  title   = {Old, Sick, Alone, and Poor: A Welfare Analysis of Old-Age Social Insurance Programmes},
  journal = {Review of Economic Studies},
  volume  = {84},
  number  = {2},
  year    = {2017},
  doi     = {10.1093/restud/rdw016}
}

@article{braunOldFrailUninsured2019,
  author  = {Braun, R. Anton and Kopecky, Karen A. and Koreshkova, Tatyana},
  title   = {Old, Frail, and Uninsured: Accounting for Features of the {U.S.} Long-Term Care Insurance Market},
  journal = {Econometrica},
  volume  = {87},
  number  = {3},
  year    = {2019},
  doi     = {10.3982/ECTA15295}
}

@article{jungMarketInefficiencyInsurance2016,
  author  = {Jung, Juergen and Tran, Chung},
  title   = {Market Inefficiency, Insurance Mandate and Welfare: {U.S.} Health Care Reform 2010},
  journal = {Review of Economic Dynamics},
  volume  = {20},
  year    = {2016},
  doi     = {10.1016/j.red.2016.02.002}
}

@article{zhaoSocialSecurityRise2014,
  author  = {Zhao, Kai},
  title   = {Social Security and the Rise in Health Spending},
  journal = {Journal of Monetary Economics},
  volume  = {64},
  year    = {2014},
  doi     = {10.1016/j.jmoneco.2014.02.005}
}

@article{conesaMacroeconomicEffectsMedicare2018,
  author  = {Conesa, Juan Carlos and Costa, Daniela and Kamali, Parisa and Kehoe, Timothy J. and Nyg{\r{a}}rd, Vegard M. and Raveendranathan, Gajendran and Saxena, Akshar},
  title   = {Macroeconomic Effects of {Medicare}},
  journal = {The Journal of the Economics of Ageing},
  volume  = {11},
  year    = {2018},
  doi     = {10.1016/j.jeoa.2017.06.002}
}

@article{hallValueLifeRise2007,
  author  = {Hall, Robert E. and Jones, Charles I.},
  title   = {The Value of Life and the Rise in Health Spending},
  journal = {Quarterly Journal of Economics},
  volume  = {122},
  number  = {1},
  year    = {2007},
  doi     = {10.1162/qjec.122.1.39}
}

@article{fonsecaAccountingRiseHealth2021,
  author  = {Fonseca, Raquel and Michaud, Pierre-Carl and Galama, Titus and Kapteyn, Arie},
  title   = {Accounting for the Rise of Health Spending and Longevity},
  journal = {Journal of the European Economic Association},
  volume  = {19},
  number  = {1},
  year    = {2021},
  doi     = {10.1093/jeea/jvaa003}
}

@article{hosseiniEvolutionHealthLife2022,
  author  = {Hosseini, Roozbeh and Kopecky, Karen A. and Zhao, Kai},
  title   = {The Evolution of Health over the Life Cycle},
  journal = {Review of Economic Dynamics},
  volume  = {45},
  year    = {2022},
  note    = {S2 record is Atlanta Fed WP 2019-12, doi 10.29338/WP2019-12; journal DOI not returned}
}

@article{pashchenkoQuantitativeAnalysisHealth2013,
  author  = {Pashchenko, Svetlana and Porapakkarm, Ponpoje},
  title   = {Quantitative Analysis of Health Insurance Reform: Separating Regulation from Redistribution},
  journal = {Review of Economic Dynamics},
  volume  = {16},
  number  = {3},
  year    = {2013},
  doi     = {10.1016/j.red.2012.09.002}
}

@article{bassettoFiscalTheoryPrice2018,
  author  = {Bassetto, Marco and Cui, Wei},
  title   = {The Fiscal Theory of the Price Level in a World of Low Interest Rates},
  journal = {Journal of Economic Dynamics and Control},
  volume  = {89},
  year    = {2018},
  doi     = {10.1016/j.jedc.2018.01.006}
}

@article{biSovereignDefaultRisk2012,
  author  = {Bi, Huixin},
  title   = {Sovereign Default Risk Premia, Fiscal Limits, and Fiscal Policy},
  journal = {European Economic Review},
  volume  = {56},
  number  = {3},
  year    = {2012},
  note    = {S2 holds only a 2011 Bank of Canada WP record (paperId d89310b9376e8b7efceab69e898dc71bcb882da3, 248 cites) with misattributed authors; journal DOI not verified today}
}

@article{holterHowDoTax2019,
  author  = {Holter, Hans A. and Krueger, Dirk and Stepanchuk, Serhiy},
  title   = {How Do Tax Progressivity and Household Heterogeneity Affect {Laffer} Curves?},
  journal = {Quantitative Economics},
  volume  = {10},
  number  = {4},
  year    = {2019},
  doi     = {10.3982/QE653}
}

@techreport{dengInequalityTaxationSovereign2019,
  author      = {Deng, Minjie},
  title       = {Inequality, Taxation, and Sovereign Default Risk},
  institution = {SSRN},
  type        = {Working Paper},
  year        = {2019},
  doi         = {10.2139/ssrn.3545501},
  note        = {S2 paperId 511e718c30f87f72b23500e117e3e6444ab5d180; journal version (AEJ: Macroeconomics 2024, from memory) not indexed}
}

@article{andreasenPoliticalEconomySovereign2019,
  author  = {Andreasen, Eugenia and Sandleris, Guido and Van der Ghote, Alejandro},
  title   = {The Political Economy of Sovereign Defaults},
  journal = {Journal of Monetary Economics},
  volume  = {104},
  year    = {2019},
  doi     = {10.1016/j.jmoneco.2018.09.003}
}

@article{bianchiFiscalStimulusSovereign2023,
  author  = {Bianchi, Javier and Ottonello, Pablo and Presno, Ignacio},
  title   = {Fiscal Stimulus under Sovereign Risk},
  journal = {Journal of Political Economy},
  volume  = {131},
  number  = {9},
  year    = {2023},
  doi     = {10.1086/724317}
}

@article{arellanoDeadlyDebtCrises2024,
  author  = {Arellano, Cristina and Bai, Yan and Mihalache, Gabriel},
  title   = {Deadly Debt Crises: {COVID-19} in Emerging Markets},
  journal = {The Review of Economic Studies},
  year    = {2024},
  note    = {S2 record is Minneapolis Fed Staff Report 603 (2020), doi 10.21034/sr.603; journal DOI not returned}
}
```
