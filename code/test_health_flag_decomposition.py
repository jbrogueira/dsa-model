"""Tests for the health-expenditure flag decomposition.

Pure-math tests (Shapley additivity, order-invariance, closed forms, anchor
property) always run.  Artefact-dependent tests skip if the model npz / data csv
have not been built yet.
"""
import math
import os
from itertools import combinations

import numpy as np
import pytest

from health_flag_decomposition import (
    shapley_multiplicative, load_model_benchmark, load_data, decompose,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NPZ = os.path.join(HERE, "data", "health_model_baseline_GR.npz")
DATA_CSV = os.path.join(HERE, "data", "health_flag_GR.csv")


# --- reference Shapley (subset form) to cross-check the permutation form ------

def shapley_reference(m, d):
    m = np.asarray(m, float); d = np.asarray(d, float)
    n = len(m)

    def value(S):
        v = 1.0
        for i in range(n):
            v *= d[i] if i in S else m[i]
        return v

    c = np.zeros(n)
    for i in range(n):
        others = [k for k in range(n) if k != i]
        for r in range(len(others) + 1):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in combinations(others, r):
                c[i] += w * (value(set(S) | {i}) - value(set(S)))
    return c


# --- pure-math tests ----------------------------------------------------------

def test_additivity_exact():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = rng.integers(2, 6)
        m = rng.uniform(0.2, 2.0, n)
        d = rng.uniform(0.2, 2.0, n)
        c = shapley_multiplicative(m, d)
        assert c.sum() == pytest.approx(np.prod(d) - np.prod(m), rel=0, abs=1e-12)


def test_matches_reference_subset_form():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = rng.integers(2, 5)
        m = rng.uniform(0.2, 2.0, n)
        d = rng.uniform(0.2, 2.0, n)
        assert np.allclose(shapley_multiplicative(m, d), shapley_reference(m, d),
                           atol=1e-12)


def test_order_invariance():
    m = np.array([0.662, 0.90, 0.081])
    d = np.array([0.60, 0.86, 0.095])
    c = shapley_multiplicative(m, d)
    perm = [2, 0, 1]
    cp = shapley_multiplicative(m[perm], d[perm])
    assert np.allclose(cp, c[perm], atol=1e-12)


def test_two_factor_closed_form():
    rng = np.random.default_rng(2)
    for _ in range(20):
        m = rng.uniform(0.2, 2.0, 2)
        d = rng.uniform(0.2, 2.0, 2)
        c = shapley_multiplicative(m, d)
        c0 = 0.5 * (d[0] - m[0]) * (m[1] + d[1])
        c1 = 0.5 * (d[1] - m[1]) * (m[0] + d[0])
        assert np.allclose(c, [c0, c1], atol=1e-12)


def test_no_change_gives_zero_contribution():
    m = np.array([0.662, 0.90, 0.081])
    c = shapley_multiplicative(m, m.copy())
    assert np.allclose(c, 0.0, atol=1e-14)


def test_single_factor_change_isolated():
    # only factor 1 moves -> all contribution on factor 1, exactly its marginal effect
    m = np.array([0.662, 0.90, 0.081])
    d = m.copy(); d[1] = 0.86
    c = shapley_multiplicative(m, d)
    assert c[0] == pytest.approx(0.0, abs=1e-14)
    assert c[2] == pytest.approx(0.0, abs=1e-14)
    assert c[1] == pytest.approx(np.prod(d) - np.prod(m), abs=1e-14)


# --- artefact-dependent tests -------------------------------------------------

@pytest.mark.skipif(not (os.path.exists(MODEL_NPZ) and os.path.exists(DATA_CSV)),
                    reason="run build_health_model_baseline.py and build_health_flag_data.py first")
def test_identity_holds_on_data():
    model = load_model_benchmark(MODEL_NPZ)
    recs = decompose(model, load_data(DATA_CSV))
    for r in recs:
        # g_data = kappa_data * Abar_data * psi_data  (psi defined residually)
        assert abs(r["identity_err"]) < 1e-9


@pytest.mark.skipif(not (os.path.exists(MODEL_NPZ) and os.path.exists(DATA_CSV)),
                    reason="artefacts not built")
def test_contributions_sum_to_gap_on_data():
    model = load_model_benchmark(MODEL_NPZ)
    recs = decompose(model, load_data(DATA_CSV))
    for r in recs:
        three = r["c_coverage"] + r["c_demography"] + r["c_residual"]
        assert three == pytest.approx(r["gap"], abs=1e-9)   # float64 roundoff over permutations
        two = r["c2_coverage"] + r["c2_che"]
        assert two == pytest.approx(r["gap"], abs=1e-9)


@pytest.mark.skipif(not (os.path.exists(MODEL_NPZ) and os.path.exists(DATA_CSV)),
                    reason="artefacts not built")
def test_anchor_year_zero_when_data_equals_model():
    # Feed a synthetic data row identical to the model: gap and all contributions 0.
    model = load_model_benchmark(MODEL_NPZ)
    synth = [dict(year=9999,
                  kappa_data=model["kappa"],
                  Abar_data=model["Abar"],
                  gov_health_gdp=model["g"],
                  che_gdp=model["chi"])]
    r = decompose(model, synth)[0]
    assert r["gap"] == pytest.approx(0.0, abs=1e-12)
    assert r["c_coverage"] == pytest.approx(0.0, abs=1e-12)
    assert r["c_demography"] == pytest.approx(0.0, abs=1e-12)
    assert r["c_residual"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(not (os.path.exists(MODEL_NPZ) and os.path.exists(DATA_CSV)),
                    reason="artefacts not built")
def test_model_benchmark_reproduces_calibration_target():
    # kappa^m * Abar^m * psi^m must equal the live baseline g^m exactly.
    model = load_model_benchmark(MODEL_NPZ)
    assert model["kappa"] * model["Abar"] * model["psi"] == pytest.approx(model["g"], rel=1e-12)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
