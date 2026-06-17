---
name: fiscal-note
description: From a fiscal_results.json, cross-check the saved figures and write a very short note (one page per scenario, one sentence per variable) on the evolution of the main aggregates and GBC items; emit markdown and PDF.
argument-hint: [path/to/fiscal_results.json]
---

# Fiscal Aggregates Note

Read the numerical paths in a `fiscal_results.json` produced by `run_fiscal_figures.py` /
`regen_fiscal_figures_from_json.py`, verify they are consistent with the figures saved in the
same folder, and write a short markdown note — **one page per scenario present** — describing
how each main aggregate and government-budget-constraint (GBC) item evolves. The baseline page
reports absolute evolution; each policy-counterfactual page reports the variable's deviation
**from the baseline**. Then render the note to PDF.

Do not run any simulation. Work only from the saved JSON and PNGs.

## Arguments

`$ARGUMENTS` is an optional path to the results JSON.
- empty → default to `code/output/fiscal_test/fiscal_results.json`
- one path → use it as `RESULTS_JSON`

Let `FOLDER` = the directory containing `RESULTS_JSON`. The note and the figures live there.

## Process

1. **Load the data.** Read `RESULTS_JSON`. Top-level keys are shock names (e.g. `G`, `Ig`)
   plus `params`. For each shock, the relevant blocks are `baseline`, `debt_financed`,
   `tax_financed`, `nfa_constrained` (some may be absent — skip those). Within each block,
   `counterfactual` holds the macro dict (`Y, K_domestic, L, C, A, NFA, r, w`, and `K_g` if
   present) and `cf_budget` holds the GBC items; `B_gdp_path` holds B/Y; `T_balance` is the
   balance horizon. The baseline curve is `baseline.counterfactual` (full NFA, B already
   subtracted) — never use the `baseline` block nested under a shock scenario (its NFA is the
   uncorrected partial A − K_domestic).

2. **Extract evolution.** With a short inline `python3` script, for every variable below compute
   for the BASELINE: the start value (t=0), the value at `T_balance`, the end value (last
   period), and the monotone direction (rising / falling / flat / hump). For each
   policy-counterfactual scenario (`debt_financed`, `tax_financed`, `nfa_constrained`) also
   compute the **deviation from baseline** (cf − base) at `T_balance` and at the end, plus the
   scenario's policy facts: financing instrument, `balance_condition`, target value,
   `adjustment_scalar` (the scalar Δ on the instrument, e.g. Δτ_l in pp), and `converged`. The
   baseline curve for every comparison is `baseline.counterfactual`. Express stocks that the
   figures plot as ratios in the same ratio units:
   - Aggregates: `Y`, `K_domestic`, `L`, `C`, `w`, `r`, `A`, `A/Y`, `NFA/Y`
     (`NFA/Y = counterfactual['NFA'] / Y`), `B/Y` (`B_gdp_path`), `K_g/Y` if `K_g` present.
   - GBC items (as shares of Y, i.e. `cf_budget[k] / Y`): `total_revenue`, `tax_l`, `tax_c`,
     `tax_p`, `tax_k`, `pension`, `ui`, `gov_health`, `govt_spending`, `public_investment`,
     `defense_spending`, `other_net_spending`, `bequest_transfers`, `primary_deficit`
     (report as the primary surplus = −primary_deficit/Y), `interest_payments`.

3. **Cross-check against the figures.** The figures share the JSON's filename stem prefix
   `<shock lower>_` in `FOLDER`: `<p>macro_overview.png`, `<p>fiscal_decomp.png`,
   `<p>prices_sanity.png`, `<p>debt_fan_chart.png`. Read each PNG and confirm that the
   direction you extracted for every variable matches the plotted curve (e.g. baseline B/Y
   rising in both fan chart and the B/Y panel; A/Y panel falling for the τ_l scenarios; each
   GBC line in fiscal_decomp matching its computed share). For any variable whose number and
   plotted curve disagree, do NOT smooth it over — list the mismatch explicitly in the note
   under a "Figure cross-check" line and state which source (JSON vs figure) you trust and why.
   If a variable has no panel, say it is not plotted rather than claiming a match.

4. **Write the note.** Create `FOLDER/<shock lower>_aggregates_note.md` (one file per shock
   present), with **one page per scenario** in this order: `baseline`, then each counterfactual
   present (`debt_financed`, `tax_financed`, `nfa_constrained`). Separate pages with a raw
   LaTeX `\newpage` on its own line. Keep it very short.

   - **Baseline page** (absolute evolution):
     - One line stating the source JSON, the figures checked, and `T_balance`.
     - `## Main aggregates` — one sentence per aggregate variable.
     - `## Government budget constraint` — one sentence per GBC item.
     Each sentence: lead with the direction and the t=0 → end magnitude, then the one-clause
     mechanism if not obvious.

   - **Each counterfactual page** (deviation from baseline):
     - A title naming the scenario, and one header line with the policy: the shock, the
       financing instrument, the `balance_condition` and its target, `adjustment_scalar`
       (e.g. "Δτ_l = +3.17 pp"), and whether it converged.
     - `## Main aggregates vs baseline` — one sentence per aggregate variable.
     - `## Government budget constraint vs baseline` — one sentence per GBC item.
     Each sentence: lead with the sign and size of the deviation from baseline (level Δ or % for
     levels; pp-of-Y for ratios and GBC shares), report it at `T_balance` and at the end if they
     differ, then the one-clause mechanism. State explicitly any variable that is unchanged from
     baseline (Δ = 0).

   For all pages: no preamble, no adjectives, no trailing summary. Report facts only (what the
   series does relative to baseline) — do not state what it implies for any argument or claim.
   Quote numbers from the JSON, not the figures. Close the file with a final
   `## Figure cross-check` section (one line: "consistent" or the list of mismatches), since the
   figures overlay all scenarios.

5. **Render PDF.** From `FOLDER`, convert the note with pandoc + xelatex, using a font with full
   symbol coverage so `→ − ≈ × ·` render:
   ```
   pandoc <note>.md -o <note>.pdf --pdf-engine=xelatex \
     -V geometry:margin=2.5cm -V fontsize=11pt -V mainfont="Arial Unicode MS"
   ```
   Confirm there are no "Missing character" warnings; if the font is unavailable, fall back to
   another broad-coverage system font.

6. **Report.** Print the `.md` and `.pdf` path(s) written, the page count (= scenarios present),
   and the "Figure cross-check" verdict for each shock.

## Style

Follow the repository writing conventions: direct and precise, one idea per sentence, say it
once. Use the established variable names (`B/Y`, `NFA/Y`, `K_domestic`, `primary surplus`,
`other_net_spending`) verbatim; do not coin new terms. State each caveat once, where it matters.
