"""Tests for SMM calibration infrastructure."""

import copy
import numpy as np
import pytest

from calibrate import (
    SimPanel,
    wrap_sim_output,
    compute_gini,
    compute_earnings_variance_by_age,
    compute_wealth_gini,
    compute_zero_wealth_fraction,
    compute_wealth_to_income_by_age,
    compute_unemployment_rate,
    compute_health_distribution_by_age,
    compute_average_hours,
    compute_consumption_gini,
    CalibrationParam,
    TargetMoment,
    CalibrationSpec,
    apply_params,
    theta_to_unbounded,
    unbounded_to_theta,
    run_model_moments,
    smm_objective,
    calibrate,
    default_spec,
    MOMENT_DISPATCH,
)
from lifecycle_perfect_foresight import LifecycleConfig


# ===================================================================
# 1. Moment unit tests
# ===================================================================

class TestGini:
    def test_equal_values(self):
        """Gini of identical values is 0."""
        assert compute_gini(np.ones(100)) == pytest.approx(0.0, abs=1e-10)

    def test_maximum_inequality(self):
        """One person has everything → Gini close to 1."""
        x = np.zeros(1000)
        x[0] = 1.0
        g = compute_gini(x)
        assert g > 0.99

    def test_known_value(self):
        """Uniform [0, 1] has Gini = 1/3."""
        np.random.seed(0)
        x = np.random.uniform(0, 1, 100_000)
        assert compute_gini(x) == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_weighted(self):
        """Weighted Gini with equal weights matches unweighted."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g1 = compute_gini(x)
        g2 = compute_gini(x, weights=np.ones(5))
        assert g1 == pytest.approx(g2, abs=0.05)

    def test_empty(self):
        assert compute_gini(np.array([])) == 0.0


class TestEarningsVariance:
    def test_basic(self):
        T, n = 10, 1000
        np.random.seed(42)
        ey = np.exp(np.random.randn(T, n) * 0.5 + 1.0)
        employed = np.ones((T, n), dtype=bool)
        alive = np.ones((T, n), dtype=bool)
        v = compute_earnings_variance_by_age(ey, employed, alive, 10)
        assert v.shape == (10,)
        assert np.all(np.isfinite(v))
        assert np.all(v > 0)

    def test_dead_excluded(self):
        T, n = 5, 100
        ey = np.ones((T, n))
        employed = np.ones((T, n), dtype=bool)
        alive = np.ones((T, n), dtype=bool)
        alive[3, :] = False  # all dead at age 3
        v = compute_earnings_variance_by_age(ey, employed, alive, 5)
        assert np.isnan(v[3])
        assert v[0] == pytest.approx(0.0, abs=1e-10)

    def test_unemployed_excluded(self):
        T, n = 3, 50
        ey = np.ones((T, n)) * 2.0
        employed = np.ones((T, n), dtype=bool)
        employed[1, :25] = False
        alive = np.ones((T, n), dtype=bool)
        v = compute_earnings_variance_by_age(ey, employed, alive, 3)
        # Age 1: only 25 employed, all with same value → var = 0
        assert v[1] == pytest.approx(0.0, abs=1e-10)


class TestWealthGini:
    def test_equal_wealth(self):
        a = np.ones((10, 100))
        alive = np.ones((10, 100), dtype=bool)
        assert compute_wealth_gini(a, alive) == pytest.approx(0.0, abs=1e-10)

    def test_with_ages(self):
        a = np.ones((10, 100))
        a[5, :] = np.arange(100) + 1.0  # unequal at age 5
        alive = np.ones((10, 100), dtype=bool)
        g_5 = compute_wealth_gini(a, alive, ages=[5])
        g_0 = compute_wealth_gini(a, alive, ages=[0])
        # Age 5 has inequality, age 0 has none
        assert g_5 > g_0
        assert g_0 == pytest.approx(0.0, abs=1e-10)


class TestZeroWealth:
    def test_all_positive(self):
        a = np.ones((5, 100))
        alive = np.ones((5, 100), dtype=bool)
        assert compute_zero_wealth_fraction(a, alive) == 0.0

    def test_half_zero(self):
        a = np.zeros((1, 100))
        a[0, 50:] = 1.0
        alive = np.ones((1, 100), dtype=bool)
        assert compute_zero_wealth_fraction(a, alive) == pytest.approx(0.5)

    def test_dead_excluded(self):
        a = np.zeros((1, 100))
        alive = np.zeros((1, 100), dtype=bool)
        alive[0, :50] = True
        a[0, :50] = 1.0  # all alive have positive wealth
        assert compute_zero_wealth_fraction(a, alive) == 0.0


class TestUnemploymentRate:
    def test_full_employment(self):
        T, n = 5, 100
        employed = np.ones((T, n), dtype=bool)
        retired = np.zeros((T, n), dtype=bool)
        alive = np.ones((T, n), dtype=bool)
        assert compute_unemployment_rate(employed, retired, alive) == 0.0

    def test_known_rate(self):
        T, n = 1, 1000
        employed = np.ones((T, n), dtype=bool)
        employed[0, :100] = False  # 10% unemployed
        retired = np.zeros((T, n), dtype=bool)
        alive = np.ones((T, n), dtype=bool)
        assert compute_unemployment_rate(employed, retired, alive) == pytest.approx(0.1)

    def test_retired_excluded(self):
        T, n = 1, 100
        employed = np.zeros((T, n), dtype=bool)
        retired = np.ones((T, n), dtype=bool)
        alive = np.ones((T, n), dtype=bool)
        # All retired → no labor force → 0
        assert compute_unemployment_rate(employed, retired, alive) == 0.0


class TestHealthDistribution:
    def test_single_state(self):
        h = np.zeros((5, 100), dtype=int)
        alive = np.ones((5, 100), dtype=bool)
        dist = compute_health_distribution_by_age(h, alive, 3)
        assert dist.shape == (5, 3)
        np.testing.assert_allclose(dist[:, 0], 1.0)
        np.testing.assert_allclose(dist[:, 1:], 0.0)

    def test_even_split(self):
        n = 300
        h = np.zeros((1, n), dtype=int)
        h[0, :100] = 0
        h[0, 100:200] = 1
        h[0, 200:] = 2
        alive = np.ones((1, n), dtype=bool)
        dist = compute_health_distribution_by_age(h, alive, 3)
        np.testing.assert_allclose(dist[0], [1/3, 1/3, 1/3], atol=1e-10)


class TestAverageHours:
    def test_all_one(self):
        T, n = 3, 100
        l = np.ones((T, n))
        emp = np.ones((T, n), dtype=bool)
        alive = np.ones((T, n), dtype=bool)
        ret = np.zeros((T, n), dtype=bool)
        assert compute_average_hours(l, emp, alive, ret) == pytest.approx(1.0)

    def test_excludes_retired(self):
        T, n = 2, 100
        l = np.ones((T, n)) * 0.5
        l[1, :] = 0.0  # retired hours
        emp = np.ones((T, n), dtype=bool)
        alive = np.ones((T, n), dtype=bool)
        ret = np.zeros((T, n), dtype=bool)
        ret[1, :] = True
        assert compute_average_hours(l, emp, alive, ret) == pytest.approx(0.5)


class TestConsumptionGini:
    def test_equal(self):
        c = np.ones((5, 100)) * 3.0
        alive = np.ones((5, 100), dtype=bool)
        assert compute_consumption_gini(c, alive) == pytest.approx(0.0, abs=1e-10)


class TestWealthToIncome:
    def test_basic(self):
        T, n = 5, 200
        a = np.ones((T, n)) * 10.0
        ey = np.ones((T, n)) * 2.0
        alive = np.ones((T, n), dtype=bool)
        emp = np.ones((T, n), dtype=bool)
        ratio = compute_wealth_to_income_by_age(a, ey, alive, emp, 5)
        np.testing.assert_allclose(ratio, 5.0)


# ===================================================================
# 2. Parameter mapping tests
# ===================================================================

class TestApplyParams:
    def test_joint_edu_param(self):
        config = LifecycleConfig()
        params = [CalibrationParam('rho_y', 'edu_params.*.rho_y', 0.8, 0.99, 0.95)]
        new_cfg = apply_params(config, params, np.array([0.90]))
        for edu in new_cfg.edu_params:
            assert new_cfg.edu_params[edu]['rho_y'] == 0.90
        # Original unchanged
        for edu in config.edu_params:
            assert config.edu_params[edu]['rho_y'] == 0.97

    def test_type_specific_edu_param(self):
        config = LifecycleConfig()
        params = [CalibrationParam('rho_y_low', 'edu_params.low.rho_y', 0.8, 0.99, 0.90)]
        new_cfg = apply_params(config, params, np.array([0.85]))
        assert new_cfg.edu_params['low']['rho_y'] == 0.85
        assert new_cfg.edu_params['medium']['rho_y'] == 0.97  # unchanged

    def test_top_level_param(self):
        config = LifecycleConfig()
        params = [CalibrationParam('jfr', 'job_finding_rate', 0.1, 0.9, 0.5)]
        new_cfg = apply_params(config, params, np.array([0.7]))
        assert new_cfg.job_finding_rate == 0.7
        assert config.job_finding_rate == 0.5  # original unchanged

    def test_no_mutation(self):
        config = LifecycleConfig()
        original_edu = copy.deepcopy(config.edu_params)
        params = [CalibrationParam('sigma_y', 'edu_params.*.sigma_y', 0.01, 0.1, 0.03)]
        apply_params(config, params, np.array([0.05]))
        assert config.edu_params == original_edu


# ===================================================================
# 3. Logit roundtrip tests
# ===================================================================

class TestLogitTransform:
    def test_roundtrip(self):
        params = [
            CalibrationParam('a', 'x', 0.0, 1.0, 0.5),
            CalibrationParam('b', 'y', 0.8, 0.999, 0.95),
            CalibrationParam('c', 'z', 0.01, 0.5, 0.1),
        ]
        theta = np.array([0.5, 0.95, 0.1])
        x = theta_to_unbounded(theta, params)
        theta2 = unbounded_to_theta(x, params)
        np.testing.assert_allclose(theta, theta2, atol=1e-10)

    def test_bounds_respected(self):
        params = [CalibrationParam('a', 'x', 2.0, 5.0, 3.5)]
        for x_val in [-100, -1, 0, 1, 100]:
            theta = unbounded_to_theta(np.array([x_val]), params)
            assert 2.0 <= theta[0] <= 5.0
            # Extreme values approach but don't reach bounds
        theta_low = unbounded_to_theta(np.array([-100.0]), params)
        theta_high = unbounded_to_theta(np.array([100.0]), params)
        assert theta_low[0] == pytest.approx(2.0, abs=1e-6)
        assert theta_high[0] == pytest.approx(5.0, abs=1e-6)

    def test_midpoint_maps_to_zero(self):
        params = [CalibrationParam('a', 'x', 0.0, 1.0, 0.5)]
        x = theta_to_unbounded(np.array([0.5]), params)
        assert x[0] == pytest.approx(0.0, abs=1e-10)


# ===================================================================
# 4. Integration tests — tiny model
# ===================================================================

class TestIntegration:
    @pytest.fixture
    def tiny_spec(self):
        base_config = LifecycleConfig(
            T=10, retirement_age=7, n_a=15, n_y=3, n_h=1,
            beta=0.96, gamma=2.0,
        )
        params = [
            CalibrationParam('rho_y', 'edu_params.*.rho_y', 0.80, 0.99, 0.95),
            CalibrationParam('sigma_y', 'edu_params.*.sigma_y', 0.01, 0.10, 0.03),
        ]
        moments = [
            TargetMoment('earnings_var_mean', 0.10, 1.0, 'earnings_var_mean'),
            TargetMoment('wealth_gini', 0.50, 1.0, 'wealth_gini'),
        ]
        return CalibrationSpec(
            params=params,
            moments=moments,
            base_config=base_config,
            education_shares={'medium': 1.0},
            n_sim=100,
            seed=42,
        )

    def test_run_model_moments_finite(self, tiny_spec):
        theta = np.array([0.95, 0.03])
        m = run_model_moments(theta, tiny_spec)
        assert m.shape == (2,)
        assert np.all(np.isfinite(m))

    def test_smm_objective_finite(self, tiny_spec):
        theta = np.array([0.95, 0.03])
        x = theta_to_unbounded(theta, tiny_spec.params)
        obj = smm_objective(x, tiny_spec)
        assert np.isfinite(obj)
        assert obj >= 0.0

    def test_moments_positive(self, tiny_spec):
        theta = np.array([0.95, 0.03])
        m = run_model_moments(theta, tiny_spec)
        # Earnings variance should be non-negative, Gini should be positive
        assert m[0] >= 0.0  # earnings_var_mean
        assert m[1] >= 0.0  # wealth_gini


# ===================================================================
# 5. Optimizer smoke test
# ===================================================================

class TestOptimizer:
    def test_smoke(self):
        """Run calibrate() for a few iterations on a tiny model."""
        base_config = LifecycleConfig(
            T=10, retirement_age=7, n_a=15, n_y=3, n_h=1,
            beta=0.96, gamma=2.0,
        )
        params = [
            CalibrationParam('sigma_y', 'edu_params.*.sigma_y', 0.01, 0.10, 0.03),
        ]
        moments = [
            TargetMoment('wealth_gini', 0.50, 1.0, 'wealth_gini'),
        ]
        spec = CalibrationSpec(
            params=params,
            moments=moments,
            base_config=base_config,
            education_shares={'medium': 1.0},
            n_sim=50,
            seed=42,
        )
        result = calibrate(spec, maxiter=5, verbose=False)
        assert 'theta' in result
        assert 'objective' in result
        assert 'model_moments' in result
        assert 'data_moments' in result
        assert 'convergence' in result
        assert 'elapsed_seconds' in result
        assert np.isfinite(result['objective'])
        assert len(result['theta']) == 1
        assert len(result['model_moments']) == 1


# ===================================================================
# 6. SimPanel wrapper test
# ===================================================================

class TestSimPanel:
    def test_wrap(self):
        """wrap_sim_output converts 21-tuple to named fields."""
        arrays = tuple(np.zeros((5, 10)) for _ in range(21))
        panel = wrap_sim_output(arrays)
        assert isinstance(panel, SimPanel)
        assert panel.a_sim.shape == (5, 10)
        assert panel.alive_sim.shape == (5, 10)
        assert panel.bequest_sim.shape == (5, 10)


# ===================================================================
# 7. MOMENT_DISPATCH coverage
# ===================================================================

class TestDispatch:
    def test_all_keys_callable(self):
        for key, fn in MOMENT_DISPATCH.items():
            assert callable(fn), f"{key} is not callable"

    def test_unknown_key_raises(self):
        spec = default_spec(n_sim=50)
        spec.moments.append(TargetMoment('bad', 0.0, 1.0, 'nonexistent_key'))
        theta = np.array([p.initial for p in spec.params])
        with pytest.raises(ValueError, match='nonexistent_key'):
            run_model_moments(theta, spec)
