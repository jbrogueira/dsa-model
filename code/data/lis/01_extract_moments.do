*! 01_extract_moments.do
*! Compute LIS moments for income process estimation.
*!
*! Modes:
*!   LOCAL_RUN = 1  → reads ../../../data/it20ip.dta (Italy 2020 LIS sample)
*!   LOCAL_RUN = 0  → LISSY submission against pooled Greek waves
*!
*! Subsets reported (all stratified by education educ in {1,2,3}):
*!   emp_fyft  : employees, full-year-full-time, log(pi11)        [primary]
*!   emp_all   : employees, no fyft filter,       log(pi11)        [robustness]
*!   selfemp   : self-employed,                   log(pi12)        [diagnostic]
*!
*! Output: a plain-text moments file, LISSY-compatible (no individual data,
*! no graphics; only aggregated cell statistics with cell-size threshold).

clear all
set more off

*-----------------------------------------------------------
* 0. Configuration
*-----------------------------------------------------------
local LOCAL_RUN = 0

if `LOCAL_RUN' == 1 {
    local datasets "it20"
    local local_path "../../../data"
}
else {
    local datasets "gr03 gr04 gr05 gr06 gr07 gr08 gr09 gr10 gr11 gr12 gr13 gr14 gr15 gr16 gr17 gr18 gr19 gr20 gr21"
}

local kvars "pid hid did dname year wave grossnet currency age sex pi11 pi12 pilabour educ educlev edyrs status1 lfs emp emp_ilo weeks weeksft hours1 fyft wexptl pwgt"

*-----------------------------------------------------------
* 1. Pool waves
*-----------------------------------------------------------
tempfile pooled
local first = 1
foreach d of local datasets {
    if `LOCAL_RUN' == 1 {
        use "`local_path'/`d'ip.dta", clear
    }
    else {
        use ${`d'p}, clear
    }
    keep `kvars'
    if `first' == 1 {
        save `pooled', replace
        local first = 0
    }
    else {
        append using `pooled'
        save `pooled', replace
    }
}
use `pooled', clear

*-----------------------------------------------------------
* 2. Common filters: working-age, valid education
*-----------------------------------------------------------
keep if !missing(age) & inrange(age, 25, 60)
keep if inlist(educ, 1, 2, 3)
gen age5 = 25 + 5*floor((age-25)/5)

tempfile master
save `master'

*-----------------------------------------------------------
* 3. Header — emitted via display so it lands in the LISSY log
*-----------------------------------------------------------
display "==== LIS income process moments ===="
display "datasets: `datasets'"
display "common filters: age 25-60, educ in {1,2,3}"
display "subsets:"
display "  emp_fyft = pi11>0 & fyft==1;                                  logy=log(pi11)"
display "  emp_all  = pi11>0;                                            logy=log(pi11)"
display "  selfemp  = self-employed (status1 in {200,210,220,240}) & pi12>0; logy=log(pi12)"
display "trim: top/bottom 0.5% of logy within (educ,year)"
display "residualization: logy on i.year FE, within education, weighted by pwgt"
display ""

*-----------------------------------------------------------
* 3b. Diagnostic: tabulate labour-force / employment indicators
*     so we can see which codes are populated in the pool
*-----------------------------------------------------------
display "==== Diagnostics: labour-force indicators in master ===="
display "[D1] status1 (frequencies, including missing)"
tabulate status1, missing
display ""
display "[D2] emp (LIS employed flag)"
tabulate emp, missing
display ""
display "[D3] emp_ilo (ILO employed flag)"
tabulate emp_ilo, missing
display ""
display "[D4] lfs (labour-force status)"
tabulate lfs, missing
display ""
display "[D5] fyft (full-year-full-time flag)"
tabulate fyft, missing
display ""
display "[D6] pi11>0 by status1 (cross-tab, missing included)"
gen byte pi11_pos = (pi11 > 0 & !missing(pi11))
tabulate status1 pi11_pos, missing
drop pi11_pos
display ""

*-----------------------------------------------------------
* 4. Loop over subsets
*-----------------------------------------------------------
foreach S in emp_fyft emp_all selfemp {

    use `master', clear

    if "`S'" == "emp_fyft" {
        keep if !missing(pi11) & pi11 > 0
        keep if !missing(fyft) & fyft == 1
        gen logy = log(pi11)
        local incvar "pi11"
    }
    else if "`S'" == "emp_all" {
        keep if !missing(pi11) & pi11 > 0
        gen logy = log(pi11)
        local incvar "pi11"
    }
    else if "`S'" == "selfemp" {
        keep if inlist(status1, 200, 210, 220, 240)
        keep if !missing(pi12) & pi12 > 0
        gen logy = log(pi12)
        local incvar "pi12"
    }

    quietly count
    if r(N) == 0 {
        display "==== subset = `S' (incvar = `incvar') — EMPTY, skipped ===="
        display ""
        continue
    }

    * Trim top/bottom 0.5% of logy within (educ,year)
    sort educ year
    by educ year: egen p005 = pctile(logy), p(0.5)
    by educ year: egen p995 = pctile(logy), p(99.5)
    drop if logy < p005 | logy > p995
    drop p005 p995

    * Residualize logy on year FE within education
    gen double u = .
    quietly levelsof educ, local(elist)
    foreach e of local elist {
        quietly count if educ == `e'
        if r(N) > 0 {
            capture quietly reg logy i.year if educ == `e' [pw=pwgt]
            if _rc == 0 {
                tempvar resid_e
                quietly predict double `resid_e' if e(sample), residuals
                quietly replace u = `resid_e' if educ == `e'
                drop `resid_e'
            }
            else {
                quietly summarize logy if educ == `e' [aw=pwgt]
                quietly replace u = logy - r(mean) if educ == `e'
            }
        }
    }

    display "==== subset = `S' (incvar = `incvar') ===="

    * [1] Wave-level metadata + sample sizes
    display "[1] Wave-level metadata and sample sizes"
    display "subset  dname  year  grossnet  n  mean_logy  var_logy  var_u"
    quietly levelsof dname, local(dlist)
    foreach dn of local dlist {
        quietly summarize year if dname == "`dn'"
        local yr = r(mean)
        quietly summarize grossnet if dname == "`dn'"
        local gn = r(mean)
        quietly count if dname == "`dn'"
        local n = r(N)
        quietly summarize logy if dname == "`dn'" [aw=pwgt]
        local mly = r(mean)
        local vly = r(Var)
        quietly summarize u if dname == "`dn'" [aw=pwgt]
        local vu = r(Var)
        display "`S'  `dn'  " %4.0f (`yr') "  " %4.0f (`gn') "  " ///
            %7.0f (`n') "  " %9.4f (`mly') "  " %9.4f (`vly') "  " %9.4f (`vu')
    }
    display ""

    * [2] Mean log earnings by education
    display "[2] Mean log earnings by education (pooled across waves)"
    display "subset  educ  n  mean_logy  sd_logy"
    foreach e of local elist {
        quietly summarize logy if educ == `e' [aw=pwgt]
        display "`S'  `e'  " %7.0f (r(N)) "  " %9.4f (r(mean)) "  " %9.4f (r(sd))
    }
    display ""

    * [3] Var(u) by single-year age x education (n>=10)
    display "[3] Var(u) by single-year age x education (n>=10)"
    display "subset  educ  age  n  var_u"
    foreach e of local elist {
        forvalues a = 25/60 {
            quietly count if educ == `e' & age == `a' & !missing(u)
            local n = r(N)
            if `n' >= 10 {
                quietly summarize u if educ == `e' & age == `a' [aw=pwgt]
                display "`S'  `e'  " %3.0f (`a') "  " %5.0f (`n') "  " %9.6f (r(Var))
            }
        }
    }
    display ""

    * [4] Var(u) by 5-year age band x education (n>=30)
    display "[4] Var(u) by 5-year age band x education (n>=30)"
    display "subset  educ  age_lo  age_hi  n  mean_u  var_u"
    foreach e of local elist {
        forvalues b = 25(5)55 {
            local bhi = `b' + 4
            quietly count if educ == `e' & age5 == `b' & !missing(u)
            local n = r(N)
            if `n' >= 30 {
                quietly summarize u if educ == `e' & age5 == `b' [aw=pwgt]
                display "`S'  `e'  " %3.0f (`b') "  " %3.0f (`bhi') "  " ///
                    %5.0f (`n') "  " %9.6f (r(mean)) "  " %9.6f (r(Var))
            }
        }
    }
    display ""

    * [5] Log earnings percentiles by 5-year age band x education (n>=30)
    display "[5] logy percentiles by age band x education (n>=30)"
    display "subset  educ  age_lo  age_hi  n  p10  p50  p90"
    foreach e of local elist {
        forvalues b = 25(5)55 {
            local bhi = `b' + 4
            quietly count if educ == `e' & age5 == `b' & !missing(logy)
            local n = r(N)
            if `n' >= 30 {
                _pctile logy if educ == `e' & age5 == `b' [aw=pwgt], p(10 50 90)
                display "`S'  `e'  " %3.0f (`b') "  " %3.0f (`bhi') "  " ///
                    %5.0f (`n') "  " %9.4f (r(r1)) "  " %9.4f (r(r2)) "  " %9.4f (r(r3))
            }
        }
    }
    display ""

    * [6] Var(u) by 5-year experience band x education (n>=30) — robustness
    display "[6] Var(u) by 5-year experience band x education (n>=30)"
    display "subset  educ  exp_lo  exp_hi  n  var_u"
    quietly count if !missing(wexptl)
    if r(N) > 0 {
        capture drop exp5
        gen exp5 = 5*floor(wexptl/5)
        forvalues b = 0(5)40 {
            local bhi = `b' + 4
            foreach e of local elist {
                quietly count if educ == `e' & exp5 == `b' & !missing(u)
                local n = r(N)
                if `n' >= 30 {
                    quietly summarize u if educ == `e' & exp5 == `b' [aw=pwgt]
                    display "`S'  `e'  " %3.0f (`b') "  " %3.0f (`bhi') "  " ///
                        %5.0f (`n') "  " %9.6f (r(Var))
                }
            }
        }
    }
    else {
        display "`S'  wexptl missing across all waves — skipped"
    }
    display ""
}

display "==== END ===="
