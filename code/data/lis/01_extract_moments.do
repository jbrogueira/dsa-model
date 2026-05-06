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
    local outname  "output/moments_it20.txt"
}
else {
    local datasets "gr03 gr04 gr05 gr06 gr07 gr08 gr09 gr10 gr11 gr12 gr13 gr14 gr15 gr16 gr17 gr18 gr19 gr20 gr21"
    local outname  "moments_GR_pooled.txt"
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
* 3. Open output file + write header
*-----------------------------------------------------------
capture file close fh
file open fh using "`outname'", write replace text

file write fh "==== LIS income process moments ====" _n
file write fh "datasets: `datasets'" _n
file write fh "common filters: age 25-60, educ in {1,2,3}" _n
file write fh "subsets:" _n
file write fh "  emp_fyft = employees (status1 in {110,120}), fyft==1, pi11>0; logy=log(pi11)" _n
file write fh "  emp_all  = employees (status1 in {110,120}), pi11>0;          logy=log(pi11)" _n
file write fh "  selfemp  = self-employed (status1 in {200,210,220,240}), pi12>0; logy=log(pi12)" _n
file write fh "trim: top/bottom 0.5% of logy within (educ,year)" _n
file write fh "residualization: logy on i.year FE, within education, weighted by pwgt" _n
file write fh _n

*-----------------------------------------------------------
* 4. Loop over subsets
*-----------------------------------------------------------
foreach S in emp_fyft emp_all selfemp {

    use `master', clear

    if "`S'" == "emp_fyft" {
        keep if inlist(status1, 110, 120)
        keep if !missing(fyft) & fyft == 1
        keep if !missing(pi11) & pi11 > 0
        gen logy = log(pi11)
        local incvar "pi11"
    }
    else if "`S'" == "emp_all" {
        keep if inlist(status1, 110, 120)
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
        file write fh "==== subset = `S' (incvar = `incvar') — EMPTY, skipped ====" _n _n
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
    levelsof educ, local(elist)
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

    file write fh "==== subset = `S' (incvar = `incvar') ====" _n

    * [1] Wave-level metadata + sample sizes
    file write fh "[1] Wave-level metadata and sample sizes" _n
    file write fh "subset  dname  year  grossnet  n  mean_logy  var_logy  var_u" _n
    levelsof dname, local(dlist)
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
        file write fh "`S'  `dn'  " %4.0f (`yr') "  " %4.0f (`gn') "  " ///
            %7.0f (`n') "  " %9.4f (`mly') "  " %9.4f (`vly') "  " %9.4f (`vu') _n
    }
    file write fh _n

    * [2] Mean log earnings by education
    file write fh "[2] Mean log earnings by education (pooled across waves)" _n
    file write fh "subset  educ  n  mean_logy  sd_logy" _n
    foreach e of local elist {
        quietly summarize logy if educ == `e' [aw=pwgt]
        file write fh "`S'  `e'  " %7.0f (r(N)) "  " %9.4f (r(mean)) "  " %9.4f (r(sd)) _n
    }
    file write fh _n

    * [3] Var(u) by single-year age x education (n>=10)
    file write fh "[3] Var(u) by single-year age x education (n>=10)" _n
    file write fh "subset  educ  age  n  var_u" _n
    foreach e of local elist {
        forvalues a = 25/60 {
            quietly count if educ == `e' & age == `a' & !missing(u)
            local n = r(N)
            if `n' >= 10 {
                quietly summarize u if educ == `e' & age == `a' [aw=pwgt]
                file write fh "`S'  `e'  " %3.0f (`a') "  " %5.0f (`n') "  " %9.6f (r(Var)) _n
            }
        }
    }
    file write fh _n

    * [4] Var(u) by 5-year age band x education (n>=30)
    file write fh "[4] Var(u) by 5-year age band x education (n>=30)" _n
    file write fh "subset  educ  age_lo  age_hi  n  mean_u  var_u" _n
    foreach e of local elist {
        forvalues b = 25(5)55 {
            local bhi = `b' + 4
            quietly count if educ == `e' & age5 == `b' & !missing(u)
            local n = r(N)
            if `n' >= 30 {
                quietly summarize u if educ == `e' & age5 == `b' [aw=pwgt]
                file write fh "`S'  `e'  " %3.0f (`b') "  " %3.0f (`bhi') "  " ///
                    %5.0f (`n') "  " %9.6f (r(mean)) "  " %9.6f (r(Var)) _n
            }
        }
    }
    file write fh _n

    * [5] Log earnings percentiles by 5-year age band x education (n>=30)
    file write fh "[5] logy percentiles by age band x education (n>=30)" _n
    file write fh "subset  educ  age_lo  age_hi  n  p10  p50  p90" _n
    foreach e of local elist {
        forvalues b = 25(5)55 {
            local bhi = `b' + 4
            quietly count if educ == `e' & age5 == `b' & !missing(logy)
            local n = r(N)
            if `n' >= 30 {
                _pctile logy if educ == `e' & age5 == `b' [aw=pwgt], p(10 50 90)
                file write fh "`S'  `e'  " %3.0f (`b') "  " %3.0f (`bhi') "  " ///
                    %5.0f (`n') "  " %9.4f (r(r1)) "  " %9.4f (r(r2)) "  " %9.4f (r(r3)) _n
            }
        }
    }
    file write fh _n

    * [6] Var(u) by 5-year experience band x education (n>=30) — robustness
    file write fh "[6] Var(u) by 5-year experience band x education (n>=30)" _n
    file write fh "subset  educ  exp_lo  exp_hi  n  var_u" _n
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
                    file write fh "`S'  `e'  " %3.0f (`b') "  " %3.0f (`bhi') "  " ///
                        %5.0f (`n') "  " %9.6f (r(Var)) _n
                }
            }
        }
    }
    else {
        file write fh "`S'  wexptl missing across all waves — skipped" _n
    }
    file write fh _n
}

file write fh "==== END ====" _n
file close fh

display "Output written to: `outname'"
