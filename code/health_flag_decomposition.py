#!/usr/bin/env python
"""Health-expenditure flag decomposition (see docs/HEALTH_FLAG_DECOMPOSITION.md).

Decomposes, year by year, the gap between observed government health spending and
the model's calibrated prediction into interpretable factors with an exact
residual, using a Shapley decomposition of the multiplicative identity

    g = kappa * Abar * psi          (g = gov health / GDP)

kappa = coverage (gov share of health expenditure), Abar = age-cost index
(population age shares x model age profile), psi = residual unit cost (relative
price of health, arrears, real-use intensity, and anything the model omits) —
the flag.  A two-way grouping g = kappa * chi (chi = CHE/GDP) is also reported.

Inputs:
    data/health_model_baseline_GR.npz   (build_health_model_baseline.py)
    data/health_flag_GR.csv             (build_health_flag_data.py)

Usage:
    python health_flag_decomposition.py
    python health_flag_decomposition.py --plot output/health_flag/
"""
import argparse
import csv
from itertools import permutations

import numpy as np


# ---------------------------------------------------------------------------
# Shapley decomposition of a multiplicative identity g = prod_i F_i
# ---------------------------------------------------------------------------

def shapley_multiplicative(model_vals, data_vals):
    """Exact Shapley contributions to g = prod_i F_i as each F_i moves model->data.

    model_vals, data_vals: equal-length sequences of the factor values.
    Returns an array c with c[i] = contribution of factor i, satisfying
    sum_i c[i] = prod(data_vals) - prod(model_vals) exactly, independent of the
    factor ordering.
    """
    m = np.asarray(model_vals, dtype=float)
    d = np.asarray(data_vals, dtype=float)
    n = len(m)
    if len(d) != n:
        raise ValueError("model_vals and data_vals must have equal length")

    def value(mask):
        """Product with factors in `mask` (a tuple of bools) taken from data."""
        v = 1.0
        for i in range(n):
            v *= d[i] if mask[i] else m[i]
        return v

    contrib = np.zeros(n)
    perms = list(permutations(range(n)))
    for order in perms:
        mask = [False] * n
        base = value(tuple(mask))
        for i in order:
            mask[i] = True
            nxt = value(tuple(mask))
            contrib[i] += (nxt - base)
            base = nxt
    contrib /= len(perms)
    return contrib


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

def load_model_benchmark(npz_path, period=0):
    """Model calibrated benchmark (default: stationary baseline t=0)."""
    z = np.load(npz_path)
    kappa = float(z["kappa"])
    Abar = float(z["Abar"][period])
    g = float(z["g_model"][period])
    chi = g / kappa
    psi = chi / Abar
    return dict(kappa=kappa, Abar=Abar, chi=chi, psi=psi, g=g,
                current_year=int(z["current_year"]), period=period)


def load_data(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (int(v) if k == "year" else float(v)) for k, v in r.items()})
    return rows


# ---------------------------------------------------------------------------
# Per-year decomposition
# ---------------------------------------------------------------------------

def decompose(model, data_rows):
    """Return per-year records with the three-way and two-way contributions."""
    out = []
    km, Am, pm, chim, gm = (model[k] for k in ("kappa", "Abar", "psi", "chi", "g"))
    for r in data_rows:
        kd = r["kappa_data"]
        Ad = r["Abar_data"]
        gd = r["gov_health_gdp"]
        chid = r["che_gdp"]
        pd_ = chid / Ad                       # residual psi^d = chi^d / Abar^d

        # three-way: g = kappa * Abar * psi
        c3 = shapley_multiplicative([km, Am, pm], [kd, Ad, pd_])
        # two-way: g = kappa * chi
        c2 = shapley_multiplicative([km, chim], [kd, chid])

        gap = gd - gm
        out.append(dict(
            year=r["year"], g_data=gd, g_model=gm, gap=gap,
            kappa_data=kd, Abar_data=Ad, psi_data=pd_, chi_data=chid,
            c_coverage=c3[0], c_demography=c3[1], c_residual=c3[2],
            c2_coverage=c2[0], c2_che=c2[1],
            resid_check3=gap - c3.sum(), resid_check2=gap - c2.sum(),
            identity_err=gd - kd * Ad * pd_,
        ))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(model, recs):
    pp = 100.0
    print(f"\nModel benchmark (calibrated, t={model['period']}, year {model['current_year']}): "
          f"g={model['g']*pp:.3f}%  kappa={model['kappa']:.3f}  "
          f"Abar={model['Abar']:.4f}  chi={model['chi']*pp:.3f}%  psi={model['psi']:.5f}")
    print("\nThree-way decomposition — contributions to (g_data - g_model), pp of GDP")
    hdr = ("year", "g_data", "g_mod", "gap", "coverage", "demog", "residual", "sum-err")
    print("  " + " ".join(f"{h:>8}" for h in hdr))
    for r in recs:
        print("  " + " ".join(f"{v:>8.3f}" for v in (
            r["year"], r["g_data"]*pp, r["g_model"]*pp, r["gap"]*pp,
            r["c_coverage"]*pp, r["c_demography"]*pp, r["c_residual"]*pp,
            r["resid_check3"]*pp)))
    maxerr = max(abs(r["resid_check3"]) for r in recs)
    maxid = max(abs(r["identity_err"]) for r in recs)
    print(f"\n  max |additivity error| = {maxerr:.2e}   "
          f"max |identity error g - k*A*psi| = {maxid:.2e}")


def write_csv(recs, path):
    cols = ["year", "g_data", "g_model", "gap", "kappa_data", "Abar_data",
            "psi_data", "chi_data", "c_coverage", "c_demography", "c_residual",
            "c2_coverage", "c2_che"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in recs:
            f.write(",".join(f"{r[c]:.8g}" for c in cols) + "\n")
    print(f"  wrote {path}")


def _new_axes(figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, plt.subplots(figsize=figsize)


def plot_decomposition(model, recs, output_dir):
    """Stacked factor contributions to the data-model gap, with the total gap line."""
    import os
    plt, (fig, ax) = _new_axes((11, 6))
    pp = 100.0
    years = [r["year"] for r in recs]
    cov = np.array([r["c_coverage"] for r in recs]) * pp
    dem = np.array([r["c_demography"] for r in recs]) * pp
    res = np.array([r["c_residual"] for r in recs]) * pp
    gap = np.array([r["gap"] for r in recs]) * pp

    comps = [("Coverage", cov, "#256abf"),
             ("Demographics", dem, "#1baf7a"),
             ("Residual (flag)", res, "#e34948")]
    pos = np.zeros(len(years))
    neg = np.zeros(len(years))
    for label, c, color in comps:
        base = np.where(c >= 0, pos, neg)
        ax.bar(years, c, bottom=base, width=0.8, color=color,
               edgecolor="white", linewidth=0.6, label=label)
        pos = np.where(c >= 0, base + c, pos)
        neg = np.where(c >= 0, neg, base + c)
    ax.plot(years, gap, "o-", color="#0b0b0b", linewidth=2.0, markersize=5,
            label="Total gap (data - model)")
    ax.axhline(0.0, color="#c3c2b7", linewidth=0.8)
    ax.set_title("Government health spending: data - model gap, decomposed")
    ax.set_xlabel("Year")
    ax.set_ylabel("Contribution to gap (pp of GDP)")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "health_flag_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_inputs(model, recs, output_dir):
    """Data vs model for each factor behind the identity g = kappa * Abar * psi."""
    import os
    plt, (fig, axes) = _new_axes((11, 7))
    fig.clf()
    axes = fig.subplots(2, 2)
    years = np.array([r["year"] for r in recs])
    DATA = "#256abf"
    MODEL = "#898781"

    panels = [
        (axes[0, 0], "Government health / GDP (%)",
         np.array([r["g_data"] for r in recs]) * 100.0, model["g"] * 100.0),
        (axes[0, 1], "Coverage  kappa = gov / CHE",
         np.array([r["kappa_data"] for r in recs]), model["kappa"]),
        (axes[1, 0], "Age-cost index  Abar",
         np.array([r["Abar_data"] for r in recs]), model["Abar"]),
        (axes[1, 1], "Residual unit cost  psi",
         np.array([r["psi_data"] for r in recs]), model["psi"]),
    ]
    for ax, title, data_series, model_val in panels:
        ax.plot(years, data_series, "o-", color=DATA, linewidth=1.8,
                markersize=4, label="data")
        ax.axhline(model_val, color=MODEL, linewidth=1.8, linestyle="--",
                   label="model (calibrated)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Health-flag inputs: data vs calibrated model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "health_flag_inputs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot(model, recs, output_dir):
    plot_decomposition(model, recs, output_dir)
    plot_inputs(model, recs, output_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-npz", default="data/health_model_baseline_GR.npz")
    ap.add_argument("--data-csv", default="data/health_flag_GR.csv")
    ap.add_argument("--period", type=int, default=0,
                    help="model baseline period used as the benchmark (default 0)")
    ap.add_argument("--out-csv", default="data/health_flag_decomposition_GR.csv")
    ap.add_argument("--plot", default="output/health_flag", metavar="DIR",
                    help="directory for the decomposition + inputs figures")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    args = ap.parse_args()

    model = load_model_benchmark(args.model_npz, period=args.period)
    data_rows = load_data(args.data_csv)
    recs = decompose(model, data_rows)
    print_table(model, recs)
    write_csv(recs, args.out_csv)
    if not args.no_plot:
        plot(model, recs, args.plot)


if __name__ == "__main__":
    main()
