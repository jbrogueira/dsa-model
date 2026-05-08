"""
02_estimate_ar1.py — Stage 2 minimum-distance estimation of the AR(1) labor
productivity process per education stratum, from the LISSY-returned moments file.

Reads moments_GR_pooled.txt and fits

    Var(u_j) = sigma_alpha^2 + sigma_eta^2 * (1 - rho^{2j}) / (1 - rho^2)

by weighted nonlinear least squares to the cross-sectional variance profile.
j is years since model age 25 (the lifecycle model anchor). sigma_alpha
captures pre-25 dispersion that loads onto education-type mean dispersion in
the calibrated model (which has no fixed effect); sigma_eta and rho feed
into the model's AR(1) income process.

Outputs ar1_estimates_GR.json with three specs per education stratum:
  - joint:       (rho, sigma_eta, sigma_alpha) jointly estimated
  - fixed_rho:   rho fixed at 0.95, (sigma_eta, sigma_alpha) estimated
  - rho_grid:    sensitivity at rho in {0.90, 0.95, 0.97, 0.99}

Usage:
    python 02_estimate_ar1.py
    python 02_estimate_ar1.py --moments output/moments_GR_pooled.txt
    python 02_estimate_ar1.py --subset emp_all --age-resolution band
"""

import argparse
import json
import os
import re
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize


EDU_MAP = {1: "low", 2: "medium", 3: "high"}
RHO_GRID = [0.90, 0.95, 0.97, 0.99]
RHO_BASELINE = 0.95
J_OFFSET = 25  # j = age - 25 (model period 0 = age 25)


# ---------------------------------------------------------------------------
# Variance profile and fitter
# ---------------------------------------------------------------------------

def variance_profile(j, sigma_alpha, sigma_eta, rho):
    """Var(u_j) under permanent + AR(1) decomposition with z_0 = 0.

    Var(u_j) = sigma_alpha^2 + sigma_eta^2 * (1 - rho^{2j}) / (1 - rho^2).
    """
    j = np.asarray(j, dtype=float)
    sa2 = sigma_alpha ** 2
    se2 = sigma_eta ** 2
    if rho >= 1.0:
        return sa2 + se2 * j
    return sa2 + se2 * (1.0 - rho ** (2.0 * j)) / (1.0 - rho ** 2)


def fit_profile(j, var_u, weights, rho_fixed=None,
                x0_joint=(0.30, 0.10, 0.90),
                x0_fixed=(0.30, 0.10)) -> Dict:
    """Weighted NLS fit. Returns dict with sigma_alpha, sigma_eta, rho, ssr."""
    j = np.asarray(j, dtype=float)
    var_u = np.asarray(var_u, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()  # normalize for numerical stability

    if rho_fixed is None:
        def loss(p):
            sa, se, rh = p
            pred = variance_profile(j, sa, se, rh)
            return float(np.sum(w * (var_u - pred) ** 2))

        bounds = [(0.0, 2.0), (0.0, 1.0), (0.0, 0.9999)]
        res = minimize(loss, x0_joint, bounds=bounds, method="L-BFGS-B",
                       options=dict(maxiter=500, ftol=1e-12))
        sa, se, rh = res.x
    else:
        def loss(p):
            sa, se = p
            pred = variance_profile(j, sa, se, rho_fixed)
            return float(np.sum(w * (var_u - pred) ** 2))

        bounds = [(0.0, 2.0), (0.0, 1.0)]
        res = minimize(loss, x0_fixed, bounds=bounds, method="L-BFGS-B",
                       options=dict(maxiter=500, ftol=1e-12))
        sa, se = res.x
        rh = rho_fixed

    return {
        "sigma_alpha": float(sa),
        "sigma_eta": float(se),
        "rho": float(rh),
        "ssr": float(res.fun),
        "success": bool(res.success),
    }


# ---------------------------------------------------------------------------
# Moments-file parser
# ---------------------------------------------------------------------------

_SUBSET_RE = re.compile(r"^==== subset = (\S+) \(incvar = (\S+)\) ====")
_SECTION_RE = re.compile(r"^\[(\d)\]")


def parse_moments(path: str) -> Dict[str, Dict[str, list]]:
    """Parse a LISSY moments file. Returns a nested dict keyed by subset, with
    sub-dicts holding parsed rows from sections [2], [3], [4]."""
    with open(path) as f:
        lines = f.readlines()

    out: Dict[str, Dict[str, list]] = {}
    cur_subset = None
    cur_section = None

    for raw in lines:
        line = raw.rstrip()
        m = _SUBSET_RE.match(line)
        if m:
            cur_subset = m.group(1)
            out[cur_subset] = {"mean_logy": [], "var_u_age": [], "var_u_band": []}
            cur_section = None
            continue
        m = _SECTION_RE.match(line)
        if m:
            cur_section = int(m.group(1))
            continue
        if cur_subset is None or cur_section is None:
            continue
        s = line.strip()
        if not s or s.startswith("subset "):
            continue
        parts = line.split()
        if cur_section == 2 and len(parts) == 5 and parts[0] == cur_subset:
            try:
                out[cur_subset]["mean_logy"].append({
                    "educ": int(parts[1]),
                    "n": int(parts[2]),
                    "mean_logy": float(parts[3]),
                    "sd_logy": float(parts[4]),
                })
            except ValueError:
                pass
        elif cur_section == 3 and len(parts) == 5 and parts[0] == cur_subset:
            try:
                out[cur_subset]["var_u_age"].append({
                    "educ": int(parts[1]),
                    "age": int(parts[2]),
                    "n": int(parts[3]),
                    "var_u": float(parts[4]),
                })
            except ValueError:
                pass
        elif cur_section == 4 and len(parts) == 7 and parts[0] == cur_subset:
            try:
                out[cur_subset]["var_u_band"].append({
                    "educ": int(parts[1]),
                    "age_lo": int(parts[2]),
                    "age_hi": int(parts[3]),
                    "n": int(parts[4]),
                    "mean_u": float(parts[5]),
                    "var_u": float(parts[6]),
                })
            except ValueError:
                pass

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_profile(rows, age_resolution: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (j, var_u, weights) arrays from parsed rows."""
    if age_resolution == "single":
        ages = np.array([r["age"] for r in rows], dtype=float)
        ns = np.array([r["n"] for r in rows], dtype=float)
        vs = np.array([r["var_u"] for r in rows], dtype=float)
    else:
        ages = np.array([(r["age_lo"] + r["age_hi"]) / 2.0 for r in rows], dtype=float)
        ns = np.array([r["n"] for r in rows], dtype=float)
        vs = np.array([r["var_u"] for r in rows], dtype=float)
    return ages - J_OFFSET, vs, ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments", default="output/moments_GR_pooled.txt")
    parser.add_argument("--subset", default="emp_fyft", choices=["emp_fyft", "emp_all", "selfemp"])
    parser.add_argument("--age-resolution", default="single", choices=["single", "band"])
    parser.add_argument("--output", default="output/ar1_estimates_GR.json")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    moments_path = args.moments if os.path.isabs(args.moments) else os.path.join(here, args.moments)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(here, args.output)

    print(f"Reading: {moments_path}")
    parsed = parse_moments(moments_path)
    if args.subset not in parsed:
        raise ValueError(f"Subset {args.subset!r} not in moments file. Found: {list(parsed)}")
    data = parsed[args.subset]
    if not data["mean_logy"]:
        raise ValueError(f"No mean_logy block found for subset {args.subset!r}")

    edu_data = {}
    print(f"\nSubset: {args.subset}   Age resolution: {args.age_resolution}\n")

    for edu_int, edu_str in EDU_MAP.items():
        mly = next((r for r in data["mean_logy"] if r["educ"] == edu_int), None)
        if mly is None:
            print(f"  educ={edu_int} ({edu_str}): no mean_logy row, skipping")
            continue

        rows = [r for r in (data["var_u_age"] if args.age_resolution == "single" else data["var_u_band"])
                if r["educ"] == edu_int]
        if not rows:
            print(f"  educ={edu_int} ({edu_str}): no variance-profile rows, skipping")
            continue

        j, var_u, weights = _build_profile(rows, args.age_resolution)

        joint = fit_profile(j, var_u, weights, rho_fixed=None)
        fixed = fit_profile(j, var_u, weights, rho_fixed=RHO_BASELINE)
        grid = []
        for rh in RHO_GRID:
            r = fit_profile(j, var_u, weights, rho_fixed=rh)
            grid.append({"rho": rh, "sigma_eta": r["sigma_eta"],
                         "sigma_alpha": r["sigma_alpha"], "ssr": r["ssr"]})

        edu_data[edu_str] = {
            "n": mly["n"],
            "mu_logy": mly["mean_logy"],
            "sd_logy": mly["sd_logy"],
            "n_cells": int(len(rows)),
            "j_min": float(j.min()),
            "j_max": float(j.max()),
            "joint": joint,
            "fixed_rho": fixed,
            "rho_grid": grid,
        }

        print(f"  educ={edu_int} ({edu_str}): n={mly['n']:>6}   mu_logy={mly['mean_logy']:.4f}   "
              f"cells={len(rows):>2}")
        print(f"    joint:                 rho={joint['rho']:.4f}   sigma_eta={joint['sigma_eta']:.4f}   "
              f"sigma_alpha={joint['sigma_alpha']:.4f}   ssr={joint['ssr']:.2e}")
        print(f"    rho fixed at 0.95:                     sigma_eta={fixed['sigma_eta']:.4f}   "
              f"sigma_alpha={fixed['sigma_alpha']:.4f}   ssr={fixed['ssr']:.2e}")
        print(f"    sensitivity (sigma_eta by rho):  ", end="")
        print("   ".join(f"rho={g['rho']:.2f}->{g['sigma_eta']:.4f}" for g in grid))
        print()

    n_total = sum(d["n"] for d in edu_data.values())
    out = {
        "source": args.moments,
        "subset": args.subset,
        "age_resolution": args.age_resolution,
        "j_definition": f"j = age - {J_OFFSET} (model period 0 = age {J_OFFSET})",
        "rho_baseline": RHO_BASELINE,
        "rho_grid": RHO_GRID,
        "estimation_date": date.today().isoformat(),
        "n_total": n_total,
        "edu_params": edu_data,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
