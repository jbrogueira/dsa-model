#!/usr/bin/env python
"""Build the model's baseline health structure for the flag decomposition.

The model's government health spending for an individual of age j is
    g^m(j) = kappa * m_good * a(j)
(education- and income-independent with n_h=1), so the period-t aggregate level is
    G^h_t = kappa * m_good * sum_j w_{j,t} phi(j,t) a(j)
with w_{j,t} the birth-cohort weights and phi(j,t) the fraction of the age-j
cohort alive at t.  Both are deterministic functions of the model's demographics
(cohort weights + cohort-diagonal survival tables) and do NOT depend on the
economic solution, so they are reconstructed analytically here — no transition
solve is run.

Validation: the analytic level G^h_0 reproduces the live baseline gov_health
level in the fiscal_results.json to Monte-Carlo tolerance (~1e-3 relative).  The
aggregate share g^m_t = G^h_t / Y_t uses the equilibrium output Y_t, which is
read from that baseline (Y_0 != 1: the transition's t=0 output differs from the
Y_ss=1 normalisation).

Output (data/health_model_baseline_GR.npz), same schema consumed by
health_flag_decomposition.py:
    model_age (T,), real_age (T,), a_profile (T,), kappa, m_good,
    alive_frac (T_tr,T), pop_share (T_tr,T), Abar (T_tr,),
    g_model (T_tr,), gov_health_level (T_tr,), Y (T_tr,), current_year
"""
import argparse
import json
import numpy as np


def cohort_alive_frac(economy, t, T):
    """phi(j,t): fraction of the age-j cross-section cohort alive at period t.

    The cohort at cross-section age j at period t is born at birth_period = t - j;
    alive fraction = cumulative product of its cohort-diagonal survival schedule
    over ages 0..j-1 (the same construction the aggregation's simulation realises
    in expectation)."""
    af = np.ones(T)
    for j in range(T):
        sched = economy._cohort_survival_schedule(birth_period=t - j)  # (T,n_h) or None
        if sched is None:
            af[j] = 1.0
            continue
        s = sched[:, :].mean(axis=1)          # average over health states (n_h=1: identity)
        af[j] = float(np.prod(s[:j])) if j > 0 else 1.0
    return af


def build(config_path, baseline_json, backend="numpy"):
    from calibrate import build_olg_transition

    cfg = json.load(open(config_path))
    economy, paths, T_tr = build_olg_transition(cfg, backend=backend)
    economy.T_transition = int(T_tr)
    T = int(economy.T)
    kappa = float(economy.lifecycle_config.kappa)
    m_good = float(economy.lifecycle_config.m_good)
    a = np.asarray(economy.lifecycle_config.m_age_profile, dtype=float)

    # Baseline equilibrium output path (paper's n_sim=2000 run).
    jd = json.load(open(baseline_json))
    shock = "G" if "G" in jd else next(k for k in jd if k != "params")
    base = jd[shock]["debt_financed"]
    Y_json = np.asarray(base["baseline"]["Y"], dtype=float)
    gh_json = np.asarray(base["base_budget"]["gov_health"], dtype=float)
    T_tr = min(int(T_tr), len(Y_json))

    alive_frac = np.zeros((T_tr, T))
    pop_share = np.zeros((T_tr, T))
    Abar = np.zeros(T_tr)
    gov_level = np.zeros(T_tr)
    for t in range(T_tr):
        w = np.asarray(economy._cohort_weights(t), dtype=float)   # births weights (sum 1)
        af = cohort_alive_frac(economy, t, T)
        stock = w * af
        s = stock.sum()
        alive_frac[t] = af
        pop_share[t] = stock / s if s > 0 else 0.0
        Abar[t] = float(np.sum(pop_share[t] * a))
        gov_level[t] = kappa * m_good * float(np.sum(w * af * a))

    Y = Y_json[:T_tr]
    g_model = gov_level / Y

    # Validation against the live aggregate.
    rel = np.max(np.abs(gov_level - gh_json[:T_tr]) / np.maximum(gh_json[:T_tr], 1e-12))
    print(f"analytic vs live gov_health level: max rel err = {rel:.2e}")
    if rel > 5e-3:
        print("  WARNING: analytic health level departs from the live baseline by >0.5%")

    return dict(
        model_age=np.arange(T), real_age=25 + np.arange(T), a_profile=a,
        kappa=kappa, m_good=m_good,
        alive_frac=alive_frac, pop_share=pop_share, Abar=Abar,
        g_model=g_model, gov_health_level=gov_level, Y=Y,
        current_year=int(economy.current_year),
    ), rel


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="calibration_input_GR.json")
    ap.add_argument("--baseline-json",
                    default="output/fiscal_test_kg_rB0/fiscal_results.json")
    ap.add_argument("--out", default="data/health_model_baseline_GR.npz")
    args = ap.parse_args()

    art, rel = build(args.config, args.baseline_json)
    np.savez(args.out, **art)
    print(f"\nSaved {args.out}")
    print(f"  g_model  t=0 = {art['g_model'][0]:.5f}  (live baseline gov_health/Y)")
    print(f"  Abar     t=0 = {art['Abar'][0]:.5f}")
    print(f"  psi^m    t=0 = {art['g_model'][0]/art['kappa']/art['Abar'][0]:.5f}")
    print(f"  alive_frac(oldest) t=0 = {art['alive_frac'][0][-1]:.4f}")
    print(f"  identity kappa*Abar*psi = {art['kappa']*art['Abar'][0]*(art['g_model'][0]/art['kappa']/art['Abar'][0]):.5f}")
