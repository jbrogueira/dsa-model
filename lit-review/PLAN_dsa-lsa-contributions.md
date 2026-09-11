# Plan: deep literature review for the DSA→LSA paper

*Prepared 2026-09-10 from the draft at `docs/` (Overleaf `9eafb13`, dated 30 July 2026), `docs/ref.bib` (462 entries), `code/docs/OPEN_ISSUES_2026-07-30.md`, `code/docs/model_vs_implementation.md`, `code/docs/TREND_GROWTH_PLAN.md`.*

## 1. Objective

Run the `/lit-review` skill at `deep` depth on the literature on sovereign debt analysis, organised so that the output answers one question: which two or three contributions can a paper built on the current model framework make that no existing paper makes. The review tests five candidate contributions (Section 3) against the literature and returns, for each, a verdict, the nearest existing paper, the differentiator that survives, and the model extension it requires.

The final choice of contributions is the authors'. The review supplies the evidence.

## 2. What the framework can deliver today

Contributions must be deliverable with the model as it stands or with extensions already specified. The review therefore needs the following facts about the framework.

**Implemented and reported (draft §2–§5).**
- Heterogeneous-agent OLG, 60 annual periods from age 25, exogenous retirement at 64, stochastic survival on the cohort's own life-table diagonal.
- Small open economy: `r = 0.04` exogenous, wage from the firm FOC, `NFA = A − K − B`. Sovereign rate `r_B` separate from `r` (0.021 in calibration, 0 in experiments, imposing `r_B − g = 0`).
- Cobb–Douglas with public capital (`η_g = 0.05`, `K_g/Y = 0.745`).
- Government budget with four proportional taxes, bequest tax, PAYG pensions with floor, UI, health coverage `κ` of an exogenous age-cost profile, transfer floor, `G`, `I_g`, defence `D`, residual `O_t`.
- Idiosyncratic income and unemployment risk; three education strata; permanent productivity effect; borrowing constraint at zero.
- Deterministic transitions under perfect foresight after an unanticipated permanent shock (MIT shock, `A[0]` predetermined).
- Experiments: `G` +2 % of Y and `I_g` +2 % of `Y_0`, each under debt financing, labour-tax financing to a terminal `B/Y`, and labour-tax financing to a terminal `NFA/Y`. Reported results: `Δτ_l` +3.1/+3.4 pp (G), +2.8/+3.2 pp (`I_g`); `I_g` cumulative multiplier 0.76 through `T`, 0.83 over the full path; debt-financed `G` multiplier zero.
- Look-back exercise (§5): exact Shapley split of the Greek government-health-spending gap 2009–2023 into coverage, age composition, residual. This split is analytic in exogenous primitives; it does not use the solved equilibrium.

**Implemented, not activated in the reported runs.** Fertility path and survival-improvement rate (the ageing transition — the reported experiments hold cohort weights fixed), progressive labour tax (HSV form), endogenous retirement window, pension trust fund, child costs and schooling phase, bequest tax and the bequest redistribution loop.

**Specified, not implemented.** Trend growth `g = r_B = 1 %` (`TREND_GROWTH_PLAN.md`, seven steps open). Welfare and redistribution reporting across cohorts and education types (the introduction promises it; nothing is reported).

**Outside the framework.** Aggregate uncertainty (no stochastic DSA fan charts), sovereign default, endogenous `r`, nominal side, income-dependent medical costs, bequest motive, human capital accumulation.

**Calibration facts a referee will use.** Wealth Gini 0.28 against 0.58 in the HFCS; `C/Y` 8.8 pp below data; UI/Y 0.013 against 0.006; six implementation gaps recorded in `OPEN_ISSUES_2026-07-30.md` (aggregate labour includes UI; bequest circuit open; `z_last` definition; `B/Y` 1.633 not 1.64; `Y_ss = 1` does not carry to `t = 0`; silent config fallbacks).

**Terminology to carry into the review.** Liability Sustainability Analysis (LSA); Debt Sustainability Analysis (DSA); Public Sector Balance Sheet (PSBS); promised-liabilities; partial default (of non-debt liabilities); Fiscal Stability Fund in the draft's text, Financial Stability Fund in the cited titles (ACLM 2025; Liu–Marimon–Wicht) — the review should use the cited titles' term and flag the draft's inconsistency.

## 3. Candidate contributions (hypotheses the review tests)

Each candidate has a kill test: the paper that, if it exists, removes the claim of novelty. Five candidates are listed so that the review can eliminate two or three.

### H1. Structural LSA: the primary balance and the cost of non-debt liabilities as equilibrium outcomes

*Claim.* In DSA practice the primary balance is an input. In this framework it is the outcome of household saving, labour supply, demographics, and policy rules, and every liability line (pensions, UI, health, interest) is priced jointly in general equilibrium. The framework therefore delivers the fiscal adjustment that satisfies an intertemporal constraint on all liabilities, with the behavioural feedback that partial-equilibrium projections omit.

*Nearest literature.* Official: European Commission Fiscal Sustainability Report (S1/S2 indicators include the cost of ageing: pensions, health, long-term care, education, unemployment benefits) and the Ageing Report 2024 (AWG projections); IMF SRDSF (2022). Academic: generational accounting and the fiscal gap (Auerbach–Gokhale–Kotlikoff 1991, 1994; Gokhale–Smetters 2003; Green–Kotlikoff 2006 on fiscal language); Bohn (1998, 2007) fiscal reaction functions; D'Erasmo–Mendoza–Zhang (2016, Handbook of Macroeconomics) on structural sustainability; the Japan OLG fiscal-sustainability papers (Braun–Joines 2015; Hansen–İmrohoroğlu 2016; İmrohoroğlu–Kitao–Yamada 2016, 2019; Kitao 2015); Evans–Kotlikoff–Phillips (2012) for the US; Storesletten (2000) "Sustaining fiscal policy through immigration".

*Kill test.* A heterogeneous-agent OLG model of a European small open economy that carries debt, pensions, health, UI and public capital jointly and computes the adjustment required for an intertemporal constraint. Verify in particular: (i) whether İmrohoroğlu–Kitao–Yamada treat Japan as a small open economy with exogenous `r`; (ii) whether the Baksa–Munkacsi OGRE model (IMF, Southern Europe) covers Greece with this scope; (iii) any Bank of Greece, ECB, or EC OLG application to Greece.

*What must survive for the claim.* The precise list of what the Japan papers include and exclude; the size of the general-equilibrium correction to S2-type indicators that the framework can compute and the Commission's method cannot.

### H2. Model-based measurement of partial default on non-debt liabilities (the look-back use)

*Claim.* The calibrated model under stable rules and observed demographics is the counterfactual against which observed spending paths are read; deviations are a quantified partial default on promised liabilities. The Greek public-health cuts of 2010–2014 are the first application.

*Nearest literature.* Accounting decompositions of health-spending growth into demographic, income, and residual (excess-cost-growth) components: the Ageing Report methodology (Przywara 2010), de la Maisonneuve–Oliveira Martins (2013, OECD), Medeiros–Schwierz (2015). Greek health under austerity: Kentikelenis et al. (2014, Lancet), Economou et al. (2014, WHO), the OECD/European Observatory country profiles. Greek pension cuts: Panageas–Tinios in Meghir–Pissarides–Vayanos–Vettas (eds, 2017, *Beyond Austerity*). Default on domestic claims: D'Erasmo–Mendoza (2016 JEEA; 2021 JME), Arellano–Mateos-Planas–Ríos-Rull (partial default, cited), Amador–Phelan (reputation and partial default, cited), Reinhart–Sbrancia (2015) on financial repression, expenditure arrears as hidden default (Checherita-Westphal–Klemm–Viefers 2016; Flynn–Pessoa 2014). In-kind cuts in consolidations: Sánchez-Gil (2025, cited).

*Kill test.* A structural counterfactual of Greek austerity that quantifies the welfare-state cuts as a liability default (candidates to read in full: House–Proebsting–Tesar 2020 JME; Chodorow-Reich–Karabarbounis–Kekre 2023, cited; Gourinchas–Philippon–Vayanos 2017; Sánchez-Gil 2025; Economides–Papageorgiou–Philippopoulos 2021; Dellas–Malliaropulos–Papageorgiou–Vourvachaki 2024).

*Known weakness the review must size.* The reported health decomposition is arithmetic because the health line is exogenous; it is in the same class as the Ageing Report's decomposition. The model-based content lies in the endogenous components (pensions, tax bases), which need counterfactual re-solves (draft §5.4) that have not been run. The review should establish whether the pension-side counterfactual, once run, has a precedent.

### H3. Financing a permanent spending increase in a high-debt SOE: which liabilities and which generations pay

*Claim.* For a defence-type increase in `G` and for public investment, the framework identifies which non-debt liabilities adjust endogenously under each financing instrument (pension and UI shares of output, health coverage if used as an instrument) and the incidence across cohorts and education types, under a one-generation absorption rule.

*Nearest literature.* War finance and tax-versus-debt financing (Barro 1979; Ohanian 1997 AER; Ramey 2011; Ramey–Zubairy 2018; Antolín-Díaz–Surico 2024/25 on long-run effects of military spending; Ilzetzki 2025 Kiel "Guns and Growth"; IMF REO Europe Nov 2025, cited). Public investment (Baxter–King 1993, Leeper–Walker–Yang 2010, Ramey 2020, all cited; Bom–Ligthart 2014; Bouakez–Guillard–Roulleau-Pascale 2017/2020; Boehm 2020 JME; Abiad–Furceri–Topalova 2016; Coenen et al. 2012). Heterogeneous-agent incidence of fiscal policy (Brinca–Holter–Krusell–Malafry 2016; Brinca et al. 2021 IER; Ferriere–Navarro 2024 REStud; Hagedorn–Manovskii–Mitman 2019; Auclert–Rognlie–Straub 2018; Bhandari–Evans–Golosov–Sargent 2017). Open-economy multipliers (Ilzetzki–Mendoza–Végh 2013, Broner et al. 2019, Priftis–Zimic 2018, all cited; Farhi–Werning 2016; Nakamura–Steinsson 2014). Intergenerational incidence of debt (Diamond 1965 and Auerbach–Kotlikoff 1987, cited; Blanchard 2019; Brumm–Feng–Kotlikoff–Kubler 2022/2024; Aguiar–Amador–Arellano 2024; Kocherlakota 2023; Angeletos–Lian–Wolf 2024; Mian–Straub–Sufi 2024). EU fiscal-framework adjustment paths of 4–7 years (Darvas–Welslau–Zettelmeyer 2023/2024, partly cited).

*Kill test.* A heterogeneous-agent OLG study of defence-spending financing with generational incidence for a euro-area country.

*Extension required.* Welfare and redistribution reporting (code check: whether cohort value functions are already stored). The zero multiplier of debt-financed `G` at fixed `r` is a standard SOE result and is not itself a contribution.

### H4. LSA under a sovereign rate below the return on capital

*Claim.* The framework separates `r_B` (0–2 %, the ESM-era effective Greek rate) from `r` (4 %). Liability sustainability with `r_B ≤ g < r` is a distinct object from the `r − g` debate, and the wedge has an institutional source (official lending; the Financial Stability Fund theory of ACLM 2025 and Liu–Marimon–Wicht) rather than a convenience-yield source.

*Nearest literature.* Blanchard (2019 AER); Mehrotra–Sergeyev (2021 JME); Reis (2022) "The constraint on public debt when r < g but g < m"; Jiang–Lustig–Van Nieuwerburgh–Xiaolan (2024 Econometrica; JLNSX 2020 cited); Brunnermeier–Merkel–Sannikov (2024); Bassetto–Cui (2018); Kocherlakota (2023 IER); Barro (2023 RED); Mauro–Zhou (2021); Bohn (1995); Abel–Mankiw–Summers–Zeckhauser (1989); Cochrane (2020, 2021, 2022, cited); Willems–Zettelmeyer (2022 ARFE); official-lending terms (Corsetti–Erce–Uy 2020, cited; ESM papers). Fiscal limits and Laffer curves (Trabandt–Uhlig 2011 JME, includes Greece; Bi 2012 EER; Leeper–Walker 2011; Holter–Krueger–Stepanchuk 2019 QE; Guner–Lopez-Daneri–Ventura 2016 JME).

*Kill test.* A quantitative OLG with an official-lender rate wedge used for sustainability analysis.

*Extension required.* Settle who earns the wedge on household holdings of `B` (open thread in `TREND_GROWTH_PLAN.md`: households currently earn `r` on all wealth while the government pays `r_B`). Reis and Brunnermeier et al. are the references for that choice.

### H5. Data infrastructure: a model-consistent liability ledger for Greece (PSBS)

*Claim.* The introduction lists constructing the PSBS as a contribution; the draft contains none of it. The review must map what already exists so the contribution, if kept, is the model-consistent mapping rather than the data.

*Nearest literature.* IMF Fiscal Monitor Oct 2018 (cited) and the IMF PSBS database (Yousefi 2019; Bova–Dippelsman–Rideout–Schaechter 2013); Eurostat/ECB accrued-to-date pension entitlements (ESA 2010 supplementary Table 29, triennial since reference year 2015; Kaier–Müller 2015); implicit pension debt (Holzmann–Palacios–Zviniene 2004; Franco 1995; Feldstein 1974); the comprehensive-balance-sheet tradition (Buiter 1983, 1985; Blejer–Cheasty 1991 JEL; Bohn 1992); Ball–Detter–Fölster (2015); generational accounting for Europe (Raffelhüschen 1999, European Economy).

*Kill test.* An existing PSBS for Greece that includes implicit pension and health liabilities. Verify whether Greece is in the IMF PSBS country set.

### Strand for positioning only (no candidate): sovereign default and limited enforcement

`ref.bib` already covers classic default, restructuring, seniority, and the Financial Stability Fund line in depth. The review adds only the heterogeneous-agent and domestic-default sub-strand (D'Erasmo–Mendoza 2016, 2021; Ferriere 2015; Deng 2024 AEJ Macro; Andreasen–Sandleris–Van der Ghote 2019 JME; Bianchi–Ottonello–Presno 2023 JPE; Arellano–Bai–Mihalache 2024) and the euro-crisis quantitative default papers already cited (Bocola–Bornstein–Dovis 2019; Aguiar–Amador–Farhi–Gopinath 2015), to justify the no-default assumption for Greece under ESM lending.

## 4. Search strands (Phase 1)

Nine strands. Each names the questions it answers, the anchors the agents start from, and the queries. Anchors marked (cited) are in `ref.bib`; the review verifies the others.

| # | Strand | Serves | Anchors | Queries |
|---|---|---|---|---|
| S1 | DSA methodology and official frameworks | H1, H4 | EC Debt Sustainability Monitor 2024 (cited); EC Fiscal Sustainability Report 2024; EC Ageing Report 2024; IMF SRDSF staff guidance 2022; Bouabdallah et al. 2017 ECB OP; Blanchard–Leandro–Zettelmeyer 2021; Darvas–Welslau–Zettelmeyer 2023, 2024 (cited); Debrun–Ostry–Willems–Wyplosz 2019; Wyplosz 2011; Bohn 1998, 2007; Mendoza–Ostry 2008; Ghosh et al. 2013 EJ; D'Erasmo–Mendoza–Zhang 2016; Collard–Habib–Rochet 2015 JEEA; Zenios et al. 2021; Celasun–Debrun–Ostry 2006; Gabriele–Erce–Athanasopoulou–Rojas 2017 ESM | "debt sustainability analysis framework", "fiscal reaction function debt sustainability", "S2 sustainability indicator cost of ageing", "stochastic debt sustainability analysis", "ESM debt sustainability", plus author searches Slawinska, Zavalloni |
| S2 | Fiscal gap, generational accounting, PSBS, implicit liabilities | H1, H5 | Auerbach–Gokhale–Kotlikoff 1991, 1994; Kotlikoff 1992; Green–Kotlikoff 2006; Gokhale–Smetters 2003; Buiter 1983, 1985; Blejer–Cheasty 1991; Bohn 1992; Holzmann–Palacios–Zviniene 2004; Kaier–Müller 2015; Eurostat Table 29 methodology; IMF FM 2018 (cited); Bova et al. 2013; Yousefi 2019; Ball–Detter–Fölster 2015; Feldstein 1974; Raffelhüschen 1999 | "generational accounting", "fiscal gap intertemporal budget", "public sector balance sheet implicit liabilities", "accrued-to-date pension entitlements", "implicit pension debt" |
| S3 | Quantitative OLG fiscal sustainability with ageing | H1, H3 | Auerbach–Kotlikoff 1987 (cited); De Nardi–İmrohoroğlu–Sargent 1999; Kotlikoff–Smetters–Walliser 2007; Nishiyama–Smetters 2007; Evans–Kotlikoff–Phillips 2012; Kitao 2014, 2015, 2018; Braun–Joines 2015; Hansen–İmrohoroğlu 2016; İmrohoroğlu–Kitao–Yamada 2016, 2019; Kitao–Mikoshiba 2020; McGrattan–Prescott 2017; Attanasio–Kitao–Violante 2007 (cited), 2011; Krueger–Ludwig 2007 (cited); Börsch-Supan–Ludwig–Winter 2006 (cited); Ludwig–Schelkle–Vogel 2012; Vogel–Ludwig–Börsch-Supan 2017; Bielecki–Brzoza-Brzezina–Kolasa 2020, 2022; Papetti 2021; Cooley–Henriksen–Nusbaum 2019 (cited); Auclert–Malmberg–Martenet–Rognlie 2024; Eggertsson–Mehrotra–Robbins 2019; Storesletten 2000; Baksa–Munkacsi (OGRE); Díaz-Giménez–Díaz-Saavedra 2009, 2017; Díaz-Saavedra–Marimon–Brogueira de Sousa 2023 (cited); Conesa–Krueger 1999 (cited); Huggett–Ventura 1999; Hosseini–Shourideh 2019; Harenberg–Ludwig 2019 | "overlapping generations fiscal sustainability aging", "achieving fiscal balance OLG", "demographic transition government debt heterogeneous agents", "pension reform OLG small open economy Europe", "OLG model Greece" |
| S4 | Health and long-term care in life-cycle macro; health-spending projection methods | H2 | De Nardi–French–Jones 2010; Kopecky–Koreshkova 2014; Braun–Kopecky–Koreshkova 2017, 2019; Jung–Tran 2016; Jung–Tran–Chambers 2017; Zhao 2014; Conesa et al. 2018 (Medicare); Hall–Jones 2007; Fonseca et al. 2021; Frankovic–Kuhn 2019; Hosseini–Kopecky–Zhao 2022; Capatina 2015; Pashchenko–Porapakkarm 2013; Jeske–Kitao 2009; Przywara 2010; de la Maisonneuve–Oliveira Martins 2013; Medeiros–Schwierz 2015; Sánchez-Gil 2025 (cited) | "public health insurance coverage OLG general equilibrium", "health expenditure projections demographic residual excess cost growth", "in-kind benefits fiscal consolidation welfare" |
| S5 | Greek crisis, austerity, welfare-state cuts | H2, H3 | Gourinchas–Philippon–Vayanos 2017; Chodorow-Reich–Karabarbounis–Kekre 2023 (cited); House–Proebsting–Tesar 2020; Zettelmeyer–Trebesch–Gulati 2013 (cited); Meghir–Pissarides–Vayanos–Vettas (eds) 2017; Economides–Papageorgiou–Philippopoulos 2021; Dellas–Malliaropulos–Papageorgiou–Vourvachaki 2024; Papageorgiou–Vourvachaki 2017; Ardagna–Caselli 2014; Alesina–Favero–Giavazzi 2019; Blanchard–Leigh 2013; Kentikelenis et al. 2014; Economou et al. 2014; OECD/EOHSP Greece profile 2023; Tinios on pension cuts | "Greek depression macroeconomics", "Greece austerity structural model counterfactual", "Greece pension cuts 2010-2016", "Greece health expenditure austerity" |
| S6 | Public investment, defence, financing permanent spending | H3 | listed under H3 | "military spending financing taxes debt", "defence spending Europe fiscal 2025", "public investment multiplier general equilibrium", "fiscal consolidation heterogeneous agents inequality", "government spending small open economy multiplier" |
| S7 | r−g, safe rates, convenience yields, fiscal limits | H4 | listed under H4 | "r minus g public debt sustainability", "convenience yield government debt valuation", "fiscal limit Laffer curve heterogeneous agents", "official lending concessional rate sustainability" |
| S8 | Default with heterogeneous agents; domestic and partial default | positioning | listed above | "sovereign default heterogeneous agents inequality", "domestic sovereign default distributional", "expenditure arrears hidden default" |
| S9 | Coauthors' and user's own work | all | Marimon FSF line (ACLM 2025; Liu–Marimon–Wicht; Ferrari–Liu–Marimon–Simpson-Bell 2024; Marimon–Wicht 2021, all cited); Ábrahám–Brogueira de Sousa–Marimon–Mayr 2023 EER; Díaz-Saavedra–Marimon–Brogueira de Sousa 2022, 2023; ESM working and discussion papers by Slawinska and Zavalloni | author searches |

Agent mapping (five parallel agents, per the skill): Agent 1 (Semantic Scholar) runs S3, S4, S7 with citation-graph expansion from the Japan papers, De Nardi–French–Jones, and Blanchard 2019; Agent 2 (Google Scholar) runs S1 and S2 title searches for policy documents; Agent 3 (surveys and handbooks) targets D'Erasmo–Mendoza–Zhang 2016, Debrun et al. 2019, Aguiar–Amador 2014 (cited), Auclert–Rognlie–Straub handbook chapter, Tomz–Wright 2013 (cited); Agent 4 (frontier working papers) runs S5, S6, S8 on NBER/CEPR/SSRN/IMF/ECB/ESM/Bruegel/Kiel 2023–2026; Agent 5 (own work) runs S9 and reads `~/Work/research/_anchors/summary.md`.

## 5. Ranking rule (Phase 2), adjusted

The skill ranks by citations × recency × relevance. Two adjustments for this review:
- Official and institutional documents (EC, IMF, ESM, ECB, OECD, Eurostat) are must-include when they define the practice the paper positions against, regardless of citation count. This applies to S1 and S2.
- Papers already in `ref.bib` are recorded as "in library" and not re-summarised unless they are in the deep-read list below.

Coverage of `ref.bib` as of today: dense on sovereign default, limited enforcement, restructuring, seniority, official lending, the FSF line, EU fiscal governance, demographics and current accounts, and pension-system design. Thin or absent: quantitative OLG fiscal sustainability (Japan and US applications), health in life-cycle macro, DSA methodology beyond the EC Monitor, r−g and convenience yields, fiscal limits, Greek fiscal-crisis structural work, PSBS and implicit-liability measurement. The search budget goes to the thin strands.

## 6. Deep-read list (Phase 3)

Read in full, in this order of priority. Extract for each: what liabilities the model carries, open or closed economy, how sustainability is defined, whether heterogeneity matters for the result, and the fiscal adjustment reported.

1. İmrohoroğlu–Kitao–Yamada 2016 IER and 2019; Braun–Joines 2015 JEDC; Hansen–İmrohoroğlu 2016 RED; Kitao 2015 JEDC.
2. D'Erasmo–Mendoza–Zhang 2016 (Handbook); Debrun–Ostry–Willems–Wyplosz 2019; Bohn 1998.
3. EC Fiscal Sustainability Report 2024 (S2 methodology chapter); EC Ageing Report 2024 (health and pension methodology); IMF SRDSF 2022 staff guidance; EC Debt Sustainability Monitor 2024 (cited).
4. Auerbach–Gokhale–Kotlikoff 1994 JEP; Green–Kotlikoff 2006; Kaier–Müller 2015; IMF FM 2018 (cited).
5. House–Proebsting–Tesar 2020; Chodorow-Reich–Karabarbounis–Kekre 2023 (cited); Gourinchas–Philippon–Vayanos 2017; Sánchez-Gil 2025 (cited); Panageas–Tinios 2017.
6. Blanchard 2019; Reis 2022; Mehrotra–Sergeyev 2021; Jiang–Lustig–Van Nieuwerburgh–Xiaolan 2024; Brunnermeier–Merkel–Sannikov 2024; Trabandt–Uhlig 2011.
7. Ohanian 1997; Antolín-Díaz–Surico 2024/25; Ilzetzki 2025; IMF REO Europe Nov 2025 (cited); Brinca et al. 2021; Ferriere–Navarro 2024.
8. De Nardi–French–Jones 2010; Braun–Kopecky–Koreshkova 2019; Jung–Tran–Chambers 2017; de la Maisonneuve–Oliveira Martins 2013.
9. D'Erasmo–Mendoza 2016 JEEA; Deng 2024; Bocola–Bornstein–Dovis 2019 (cited).
10. Auclert–Malmberg–Martenet–Rognlie 2024; Bielecki–Brzoza-Brzezina–Kolasa 2022.

Around 40 papers; abstract-only fallback flagged per the skill.

## 7. Synthesis structure (Phase 5)

Replace the skill's generic "The Gap" section with a contribution verdict table, one row per candidate H1–H5:

| Candidate | Verdict (novel / partly pre-empted / pre-empted) | Nearest paper(s) | Differentiator that survives | Model extension required | Referee objection |

Then a ranked recommendation of two or three candidates with the reasoning, followed by the strand-by-strand narrative in the skill's top-journal style (contributions first, then N strands, each ending in positioning against the closest work).

## 8. Hostile-referee questions (Phase 6)

Run one targeted search for each before finalising.
1. The Commission's S2 indicator already includes pensions, health, long-term care, education and unemployment benefits. What does general equilibrium add, and how large is the correction?
2. Kotlikoff has argued since the 1980s that explicit debt is a labelling convention and that generational accounts are the right object. Is LSA a rename?
3. The Japan OLG papers compute the fiscal adjustment for debt, pensions and health jointly. What is new beyond the country?
4. The health decomposition is arithmetic and the Ageing Report performs the same three-way split. Where is the model?
5. A zero multiplier for debt-financed `G` at a fixed world rate is textbook. What is the experiment's contribution?
6. Why no default in a Greek application when Bocola–Bornstein–Dovis (2019) quantified euro-crisis default risk? Does the ESM/FSF argument carry the weight?
7. Perfect foresight and no aggregate risk exclude stochastic DSA (fan charts in Celasun–Debrun–Ostry 2006, Zenios et al. 2021, and the Commission's stochastic projections). Is the framework a DSA tool at all?
8. The title says ageing; the experiments hold the age distribution fixed. (Fertility and survival paths are implemented but not activated.)
9. Wealth Gini 0.28 against 0.58: can the model speak to distributional incidence?
10. Households earn `r = 4 %` on holdings of `B` while the government pays `r_B = 0`. Who receives the wedge?

## 9. Facts to verify during the review

- Whether İmrohoroğlu–Kitao–Yamada (2016, 2019) assume a small open economy with exogenous `r`, and whether health spending is a government line in their model.
- Whether Greece is in the IMF PSBS country set (Fiscal Monitor Oct 2018 and the database).
- The exact component list of the Commission's S2 cost-of-ageing term and its reference year in the 2024 Fiscal Sustainability Report.
- Whether the Eurostat/ECB accrued-to-date pension-entitlements table (ESA 2010 Table 29) is published for Greece and for which reference years.
- Whether Baksa–Munkacsi's OGRE model has a Greek calibration.
- Publication status and venues of: Antolín-Díaz–Surico; Brunnermeier–Merkel–Sannikov; Mian–Straub–Sufi; Aguiar–Amador–Arellano; Reis; Deng.
- Whether any ESM paper by Slawinska or Zavalloni already sets out an LSA-type framework.
- Whether the code stores cohort value functions, so that welfare reporting (H3) is a reporting change rather than a solver change.

## 10. Execution notes

**Backends (checked 2026-09-10).** The `semantic-scholar` server failed to connect at session start because a second Claude Code session already held TCP port 8000 through the server's optional HTTP bridge; the user-scope registration now sets `SEMANTIC_SCHOLAR_ENABLE_HTTP_BRIDGE=0` and the server completes the MCP handshake with all 16 tools (runbook entry in `~/.claude/skills/lit-review/ARCHITECTURE.md`). A session started before the fix needs `/mcp` reconnect. Keyed API requests returned HTTP 200 directly but 429 through the server during the probe; if 429s persist during the run, the rate limit is shared with the other live session's server, so pause that session or space the queries. The `google-scholar` server, WebSearch and WebFetch are available.

**Invocation.**
```
/lit-review topic: sovereign debt sustainability analysis with non-debt government liabilities (pensions, health, public capital) in heterogeneous-agent OLG small open economies, depth: deep, for: DSA-LSA paper, seminal: Auerbach Kotlikoff 1987, Bohn 1998, Blanchard 2019, Imrohoroglu Kitao Yamada 2016
```
Run from the repo root so that output lands in `lit-review/`, not in the Overleaf submodule.

**Outputs.** `lit-review/dsa-lsa-contributions.md` (review with the verdict table), `lit-review/dsa-lsa-contributions.bib`, Zotero staging with tags `to-read, macro, fiscal, social-security`, lit notes for the deep-read papers. Nothing is written to `docs/`.

**Scope limits.** The review does not draft introduction text and does not restate the draft's results as findings. Verdicts are evidence for the authors' decision.
