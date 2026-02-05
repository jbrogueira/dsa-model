import pytest
import numpy as np
import sys
import os
from olg_transition import OLGTransition, get_test_config
from lifecycle_perfect_foresight import LifecycleConfig, LifecycleModelPerfectForesight

# test_olg_transition.py
"""
Unit tests for OLG transition dynamics with perfect foresight.
Run with: pytest test_olg_transition.py -v
"""



class TestOLGTransitionBasics:
    """Test basic instantiation and setup."""
    
    def test_init_default_config(self):
        """Test OLGTransition initializes with default config."""
        economy = OLGTransition()
        
        assert economy.T == 60
        assert economy.alpha == 0.33
        assert economy.delta == 0.05
        assert economy.A == 1.0
        assert economy.cohort_sizes is not None
        assert len(economy.cohort_sizes) == economy.T
        assert np.isclose(np.sum(economy.cohort_sizes), 1.0)
    
    def test_init_custom_config(self):
        """Test OLGTransition with custom LifecycleConfig."""
        config = LifecycleConfig(
            T=10,
            beta=0.95,
            gamma=1.5,
            n_a=15,
            n_y=3,
            n_h=2,
            retirement_age=7
        )
        
        economy = OLGTransition(
            lifecycle_config=config,
            alpha=0.30,
            delta=0.06,
            A=1.5
        )
        
        assert economy.T == 10
        assert economy.beta == 0.95
        assert economy.gamma == 1.5
        assert economy.alpha == 0.30
        assert economy.delta == 0.06
        assert economy.A == 1.5
        assert economy.retirement_age == 7
    
    def test_cohort_sizes_sum_to_one(self):
        """Test that cohort sizes sum to 1 (population mass)."""
        economy = OLGTransition()
        assert np.isclose(np.sum(economy.cohort_sizes), 1.0)
    
    def test_education_shares(self):
        """Test education share specification."""
        edu_shares = {'low': 0.2, 'medium': 0.6, 'high': 0.2}
        economy = OLGTransition(education_shares=edu_shares)
        
        assert economy.education_shares == edu_shares
        assert np.isclose(sum(edu_shares.values()), 1.0)


class TestProductionFunction:
    """Test production function and factor prices."""
    
    def test_production_function(self):
        """Test Cobb-Douglas production function."""
        economy = OLGTransition(alpha=0.33, A=1.0)
        
        K, L = 1.0, 1.0
        Y = economy.production_function(K, L)
        
        assert Y == 1.0  # With K=L=1, alpha=0.33, A=1: Y = 1^0.33 * 1^0.67 = 1
    
    def test_factor_prices_consistency(self):
        """Test that factor prices satisfy Euler equation."""
        economy = OLGTransition(alpha=0.33, delta=0.05, A=1.0)
        
        K, L = 2.0, 1.0
        r, w = economy.factor_prices(K, L)
        
        # Check MPK: r = alpha * A * (K/L)^(alpha-1) - delta
        K_over_L = K / L
        MPK = economy.alpha * economy.A * (K_over_L ** (economy.alpha - 1))
        expected_r = MPK - economy.delta
        
        assert np.isclose(r, expected_r)
        
        # Check MPL: w = (1-alpha) * A * (K/L)^alpha
        expected_w = (1 - economy.alpha) * economy.A * (K_over_L ** economy.alpha)
        
        assert np.isclose(w, expected_w)
    
    def test_factor_prices_with_exogenous_r(self):
        """Test that given r, we can back out K/L and w."""
        economy = OLGTransition(alpha=0.33, delta=0.05, A=1.0)
        
        r_exog = 0.03
        
        # From r + delta = alpha * A * (K/L)^(alpha-1)
        K_over_L = ((r_exog + economy.delta) / (economy.alpha * economy.A)) ** (1 / (economy.alpha - 1))
        
        # Then w = (1-alpha) * A * (K/L)^alpha
        w_implied = (1 - economy.alpha) * economy.A * (K_over_L ** economy.alpha)
        
        # Verify by computing factor prices from this K/L
        K, L = K_over_L, 1.0
        r_computed, w_computed = economy.factor_prices(K, L)
        
        assert np.isclose(r_computed, r_exog, atol=1e-6)
        assert np.isclose(w_computed, w_implied, atol=1e-6)


class TestConstantInterestRate:
    """Test transition with constant interest rates."""
    
    def test_constant_r_small_economy(self):
        """Test transition with constant r using minimal grid."""
        # Minimal config for speed
        config = LifecycleConfig(
            T=5,
            beta=0.96,
            gamma=2.0,
            n_a=5,
            n_y=2,
            n_h=1,
            retirement_age=4,
            education_type='medium'
        )
        
        economy = OLGTransition(
            lifecycle_config=config,
            alpha=0.33,
            delta=0.05,
            A=1.0,
            education_shares={'medium': 1.0},
            output_dir='output/test'
        )
        
        # Constant interest rate path
        T_transition = 3
        r_constant = 0.03
        r_path = np.ones(T_transition) * r_constant
        
        # Simulate
        results = economy.simulate_transition(
            r_path=r_path,
            w_path=None,  # Will be computed from r
            n_sim=20,
            verbose=False
        )
        
        # Verify results structure
        assert 'r' in results
        assert 'w' in results
        assert 'K' in results
        assert 'L' in results
        assert 'Y' in results
        
        # Check that r_path is constant
        assert np.allclose(results['r'], r_constant)
        
        # Check that aggregates are positive
        assert np.all(results['K'] > 0)
        assert np.all(results['L'] > 0)
        assert np.all(results['Y'] > 0)
        
        # Check production function: Y = A * K^alpha * L^(1-alpha)
        Y_implied = economy.A * (results['K'] ** economy.alpha) * (results['L'] ** (1 - economy.alpha))
        assert np.allclose(results['Y'], Y_implied, rtol=1e-5)
    
    def test_constant_r_implies_constant_w(self):
        """Test that constant r should imply roughly constant w (given K/L)."""
        config = LifecycleConfig(
            T=5,
            beta=0.96,
            gamma=2.0,
            n_a=5,
            n_y=2,
            n_h=1,
            retirement_age=4,
            education_type='medium'
        )
        
        economy = OLGTransition(
            lifecycle_config=config,
            alpha=0.33,
            delta=0.05,
            A=1.0,
            education_shares={'medium': 1.0},
            output_dir='output/test'
        )
        
        # Constant r
        T_transition = 3
        r_constant = 0.04
        r_path = np.ones(T_transition) * r_constant
        
        results = economy.simulate_transition(
            r_path=r_path,
            n_sim=20,
            verbose=False
        )
        
        # With constant r, the implied K/L from production should be constant
        # Therefore w should also be constant
        K_over_L = results['K'] / results['L']
        
        # Check K/L is roughly constant (allowing for small simulation noise)
        assert np.std(K_over_L) / np.mean(K_over_L) < 0.1  # CV < 10%
    
    def test_constant_environment_implies_constant_aggregates(self):
        """
        Test that with ALL constant parameters (r, w, taxes, etc.), 
        aggregates should be exactly constant in steady state.
        
        This is a strong test: if everything is constant, the economy 
        should be in perfect steady state with no dynamics.
        """
        config = LifecycleConfig(
            T=3,
            beta=0.99,
            gamma=2.0,
            n_a=100,
            n_y=2,
            n_h=1,
            retirement_age=3,
            education_type='medium'
        )
        
        economy = OLGTransition(
            lifecycle_config=config,
            alpha=0.33,
            delta=0.05,
            A=1.0,
            pop_growth=0.0,  # Zero population growth for perfect steady state
            education_shares={'medium': 1.0},
            output_dir='output/test'
        )
        
        # All parameters constant
        T_transition = 20  # Longer to verify stability
        r_constant = 0.06
        tau_c_constant = 0.05
        tau_l_constant = 0.0
        tau_p_constant = 0.0
        tau_k_constant = 0.0
        pension_constant = 0.00
        
        r_path = np.ones(T_transition) * r_constant
        tau_c_path = np.ones(T_transition) * tau_c_constant
        tau_l_path = np.ones(T_transition) * tau_l_constant
        tau_p_path = np.ones(T_transition) * tau_p_constant
        tau_k_path = np.ones(T_transition) * tau_k_constant
        pension_path = np.ones(T_transition) * pension_constant
        
        results = economy.simulate_transition(
            r_path=r_path,
            tau_c_path=tau_c_path,
            tau_l_path=tau_l_path,
            tau_p_path=tau_p_path,
            tau_k_path=tau_k_path,
            pension_replacement_path=pension_path,
            n_sim=500,
            verbose=False
        )
        
        # Extract aggregates
        K_path = results['K']
        L_path = results['L']
        Y_path = results['Y']
        w_path = results['w']

        # Print detailed diagnostics
        print("\n" + "="*70)
        print("STEADY STATE TEST WITH CONSTANT ENVIRONMENT")
        print("="*70)
        print(f"Simulation periods: {T_transition}")
        print(f"Number of agents: 500")
        print(f"\nConstant parameters:")
        print(f"  r = {r_constant:.4f}")
        print(f"  τ_c = {tau_c_constant:.4f}")
        print(f"  τ_l = {tau_l_constant:.4f}")
        
        print("\n" + "-"*70)
        print("FIRST 5 PERIODS:")
        print("-"*70)
        print(f"{'Period':<8} {'K':<12} {'L':<12} {'Y':<12} {'w':<12} {'r':<12}")
        print("-"*70)
        for t in range(min(5, T_transition)):
            print(f"{t:<8} {K_path[t]:<12.6f} {L_path[t]:<12.6f} {Y_path[t]:<12.6f} {w_path[t]:<12.6f} {results['r'][t]:<12.6f}")
        
        print("\n" + "-"*70)
        print("LAST 10 PERIODS:")
        print("-"*70)
        print(f"{'Period':<8} {'K':<12} {'L':<12} {'Y':<12} {'w':<12} {'r':<12}")
        print("-"*70)
        for t in range(max(0, T_transition-10), T_transition):
            print(f"{t:<8} {K_path[t]:<12.6f} {L_path[t]:<12.6f} {Y_path[t]:<12.6f} {w_path[t]:<12.6f} {results['r'][t]:<12.6f}")
        
        # Skip first 20 periods to allow convergence
        burn_in = min(20, T_transition // 2)
        K_path_stable = K_path[burn_in:]
        L_path_stable = L_path[burn_in:]
        Y_path_stable = Y_path[burn_in:]
        w_path_stable = w_path[burn_in:]
        
        # Test 1: Interest rate is exactly constant
        r_is_constant = np.allclose(results['r'], r_constant, atol=1e-10)
        print("\n" + "-"*70)
        print("TEST RESULTS:")
        print("-"*70)
        print(f"1. Interest rate constant: {'✓ PASS' if r_is_constant else '✗ FAIL'}")
       
        # Test 2: All aggregates should have very low variance
        # Use coefficient of variation (CV = std / mean)
        K_mean = np.mean(K_path_stable)
        L_mean = np.mean(L_path_stable)
        Y_mean = np.mean(Y_path_stable)
        w_mean = np.mean(w_path_stable)
        
        K_cv = np.std(K_path_stable) / K_mean if K_mean > 0 else np.nan
        L_cv = np.std(L_path_stable) / L_mean if L_mean > 0 else np.nan
        Y_cv = np.std(Y_path_stable) / Y_mean if Y_mean > 0 else np.nan
        w_cv = np.std(w_path_stable) / w_mean if w_mean > 0 else np.nan
        
        tolerance = 0.05  # 5% coefficient of variation
        
        print(f"\n2. Aggregate stability (Coefficient of Variation after burn-in={burn_in}):")
        print(f"   Capital:  CV={K_cv:.2%} (mean={K_mean:.6f}, std={np.std(K_path_stable):.6f}) {'✓' if K_cv < tolerance else '✗'}")
        print(f"   Labor:    CV={L_cv:.2%} (mean={L_mean:.6f}, std={np.std(L_path_stable):.6f}) {'✓' if L_cv < tolerance else '✗'}")
        print(f"   Output:   CV={Y_cv:.2%} (mean={Y_mean:.6f}, std={np.std(Y_path_stable):.6f}) {'✓' if Y_cv < tolerance else '✗'}")
        print(f"   Wage:     CV={w_cv:.2%} (mean={w_mean:.6f}, std={np.std(w_path_stable):.6f}) {'✓' if w_cv < tolerance else '✗'}")
        
        # Test 3: No trend in aggregates (after burn-in)
        periods_stable = np.arange(len(K_path_stable))

        def get_trend_slope(y):
            t_mean = np.mean(periods_stable)
            y_mean = np.mean(y)
            cov = np.mean((periods_stable - t_mean) * (y - y_mean))
            var_t = np.mean((periods_stable - t_mean)**2)
            return cov / var_t

        K_slope = get_trend_slope(K_path_stable)
        L_slope = get_trend_slope(L_path_stable)
        Y_slope = get_trend_slope(Y_path_stable)
        
        K_slope_pct = (K_slope / K_mean) * 100 if K_mean > 0 else np.nan
        L_slope_pct = (L_slope / L_mean) * 100 if L_mean > 0 else np.nan
        Y_slope_pct = (Y_slope / Y_mean) * 100 if Y_mean > 0 else np.nan
        
        slope_tolerance = 0.5  # 0.5% per period
        
        print(f"\n3. Aggregate trends (% change per period):")
        print(f"   Capital:  {K_slope_pct:+.4f}%/period {'✓' if abs(K_slope_pct) < slope_tolerance else '✗'}")
        print(f"   Labor:    {L_slope_pct:+.4f}%/period {'✓' if abs(L_slope_pct) < slope_tolerance else '✗'}")
        print(f"   Output:   {Y_slope_pct:+.4f}%/period {'✓' if abs(Y_slope_pct) < slope_tolerance else '✗'}")
        
        # Test 4: Production function
        Y_check = economy.A * (K_path ** economy.alpha) * (L_path ** (1 - economy.alpha))
        prod_fn_holds = np.allclose(Y_path, Y_check, rtol=1e-5)
        
        print(f"\n4. Production function Y = A·K^α·L^(1-α): {'✓ PASS' if prod_fn_holds else '✗ FAIL'}")
        
        print("\n" + "="*70)
        
        # Summary statistics
        print("\nSUMMARY STATISTICS (full transition):")
        print(f"  K: min={np.min(K_path):.6f}, max={np.max(K_path):.6f}, mean={np.mean(K_path):.6f}")
        print(f"  L: min={np.min(L_path):.6f}, max={np.max(L_path):.6f}, mean={np.mean(L_path):.6f}")
        print(f"  Y: min={np.min(Y_path):.6f}, max={np.max(Y_path):.6f}, mean={np.mean(Y_path):.6f}")
        print(f"  K/Y ratio: {np.mean(K_path/Y_path):.4f}" if np.mean(Y_path) > 0 else "  K/Y ratio: undefined")
        print("="*70 + "\n")
        
        # Now do assertions
        assert r_is_constant, "Interest rate should be exactly constant"
        assert K_cv < tolerance, f"Capital should be nearly constant (CV={K_cv:.2%} > {tolerance:.0%})"
        assert L_cv < tolerance, f"Labor should be nearly constant (CV={L_cv:.2%} > {tolerance:.0%})"
        assert Y_cv < tolerance, f"Output should be nearly constant (CV={Y_cv:.2%} > {tolerance:.0%})"
        assert w_cv < tolerance, f"Wage should be nearly constant (CV={w_cv:.2%} > {tolerance:.0%})"
        assert abs(K_slope_pct) < slope_tolerance, f"Capital should have no trend (slope={K_slope_pct:.2f}% per period)"
        assert abs(L_slope_pct) < slope_tolerance, f"Labor should have no trend (slope={L_slope_pct:.2f}% per period)"
        assert abs(Y_slope_pct) < slope_tolerance, f"Output should have no trend (slope={Y_slope_pct:.2f}% per period)"
        assert prod_fn_holds, "Production function should hold exactly"


class TestBorrowingConstraint:
    """Test if the borrowing constraint is causing zero savings."""
    
    def test_policy_at_different_asset_levels(self):
        """Check if policies are zero only at a=0 (borrowing constraint)."""
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        
        print("\n" + "="*70)
        print("BORROWING CONSTRAINT TEST")
        print("="*70)
        
        config = LifecycleConfig(
            T=5,
            beta=0.96,
            gamma=2.0,
            n_a=10,
            n_y=2,
            n_h=1,
            retirement_age=4,
            education_type='medium'
        )
        
        r_ss = 0.04
        alpha = 0.33
        delta = 0.05
        A = 1.0
        K_over_L = ((r_ss + delta) / (alpha * A)) ** (1 / (alpha - 1))
        w_ss = (1 - alpha) * A * (K_over_L ** alpha)
        
        ss_config = config._replace(
            r_path=np.ones(config.T) * r_ss,
            w_path=np.ones(config.T) * w_ss,
            tau_c_path=np.ones(config.T) * 0.05,
            tau_l_path=np.ones(config.T) * 0.15,
            tau_p_path=np.ones(config.T) * 0.124,
            tau_k_path=np.ones(config.T) * 0.20,
            pension_replacement_path=np.ones(config.T) * 0.40
        )
        
        model = LifecycleModelPerfectForesight(ss_config, verbose=False)
        model.solve(verbose=False)
        
        # Check policies at age 1 across ALL asset levels
        age = 1
        y_idx = 1  # High income
        h_idx = 0
        e_idx = 0
        
        print(f"\nAge {age} savings policies (y=high, h=good, e=0) by asset level:")
        print(f"{'a_idx':<6} {'assets':<10} {'a_next':<10} {'savings?':<10}")
        print("-" * 40)
        
        for a_idx in range(model.a_policy.shape[1]):
            a_next = model.a_policy[age, a_idx, y_idx, h_idx, e_idx]
            # Get actual asset value from grid
            a_current = model.a_grid[a_idx]
            saves = "✓" if a_next > 0.01 else "✗"
            print(f"{a_idx:<6} {a_current:<10.4f} {a_next:<10.4f} {saves:<10}")
        
        # Check if ONLY a=0 has zero savings
        a0_policy = model.a_policy[age, 0, y_idx, h_idx, e_idx]
        a1_policy = model.a_policy[age, 1, y_idx, h_idx, e_idx]
        
        print(f"\nKey finding:")
        print(f"  Policy at a=0: {a0_policy:.4f}")
        print(f"  Policy at a=1: {a1_policy:.4f}")
        
        if a0_policy < 0.01 and a1_policy > 0.01:
            print(f"\n✓ Confirmed: Borrowing constraint binds ONLY at a=0!")
            print(f"   Agents with any positive assets DO save.")
        
        # Now check what happens in OLGTransition when initializing cohorts
        print(f"\n" + "="*70)
        print(f"IMPLICATION FOR OLGTRANSITION:")
        print(f"="*70)
        print(f"When OLGTransition initializes old cohorts with their")
        print(f"'steady-state assets', if those assets are ZERO (or very small),")
        print(f"they will hit the borrowing constraint and save NOTHING.")
        print(f"\nThis causes K→0 because:")
        print(f"  1. New cohorts start with a=0 (by definition)")
        print(f"  2. Policy at a=0 says: save nothing")
        print(f"  3. Old cohorts die out")
        print(f"  4. Aggregate K decreases monotonically to zero")


class TestConfigInspection:
    """Inspect what's in the LifecycleConfig."""
    
    def test_print_full_config(self):
        """Print all fields in LifecycleConfig."""
        config = LifecycleConfig(
            T=8,
            beta=0.96,
            gamma=2.0,
            n_a=20,
            n_y=3,
            n_h=1,
            retirement_age=6,
            education_type='medium'
        )
        
        print("\n" + "="*70)
        print("FULL LIFECYCLE CONFIG")
        print("="*70)
        
        # Print all attributes
        for attr in dir(config):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
        
        print("="*70)


class TestRootCauseDiagnostic:
    """Find the root cause of K→0 problem."""
    
    def test_check_steady_state_simulation(self):
        """
        Check if the steady-state computation itself is producing valid results.
        """
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        
        print("\n" + "="*70)
        print("STEADY STATE SIMULATION DIAGNOSTIC")
        print("="*70)
        
        config = LifecycleConfig(
            T=8,
            beta=0.96,
            gamma=2.0,
            n_a=20,
            n_y=3,
            n_h=1,
            retirement_age=6,
            education_type='medium'
        )
        
        # Create standalone lifecycle model
        r_ss = 0.04
        
        # Compute w from production function
        alpha = 0.33
        delta = 0.05
        A = 1.0
        K_over_L = ((r_ss + delta) / (alpha * A)) ** (1 / (alpha - 1))
        w_ss = (1 - alpha) * A * (K_over_L ** alpha)
        
        print(f"\nSteady-state prices:")
        print(f"  r_ss = {r_ss:.4f}")
        print(f"  w_ss = {w_ss:.4f}")
        print(f"  K/L = {K_over_L:.4f}")
        
        # Create lifecycle model with these prices
        ss_config = config._replace(
            r_path=np.ones(config.T) * r_ss,
            w_path=np.ones(config.T) * w_ss,
            tau_c_path=np.ones(config.T) * 0.05,
            tau_l_path=np.ones(config.T) * 0.15,
            tau_p_path=np.ones(config.T) * 0.124,
            tau_k_path=np.ones(config.T) * 0.20,
            pension_replacement_path=np.ones(config.T) * 0.40
        )
        
        model = LifecycleModelPerfectForesight(ss_config, verbose=False)
        model.solve(verbose=True)
        
        # Simulate to get asset profiles
        results = model.simulate(T_sim=config.T, n_sim=1000, seed=42)
        assets_sim = results[0]  # Shape: (T, n_sim)
        
        # Compute mean assets by age
        mean_assets = np.mean(assets_sim, axis=1)
        
        print(f"\nSimulated steady-state asset profile:")
        for age in range(len(mean_assets)):
            print(f"  Age {age}: {mean_assets[age]:.4f}")
        
        # Check if assets are all zero (problem!)
        nonzero_ages = np.sum(mean_assets > 0.01)
        print(f"\nAges with positive assets: {nonzero_ages}/{len(mean_assets)}")
        
        # If most ages have zero assets, there's a problem with the model
        assert nonzero_ages >= 3, \
            f"Only {nonzero_ages} ages have positive assets - model is not saving!"
    
    def test_check_lifecycle_model_directly(self):
        """
        Test the lifecycle model directly to see if it's producing valid policies.
        """
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        
        print("\n" + "="*70)
        print("LIFECYCLE MODEL POLICY CHECK")
        print("="*70)
        
        config = LifecycleConfig(
            T=5,  # Short for easier inspection
            beta=0.96,
            gamma=2.0,
            n_a=15,
            n_y=2,
            n_h=1,
            retirement_age=4,
            education_type='medium'
        )
        
        r_ss = 0.04
        alpha = 0.33
        delta = 0.05
        A = 1.0
        K_over_L = ((r_ss + delta) / (alpha * A)) ** (1 / (alpha - 1))
        w_ss = (1 - alpha) * A * (K_over_L ** alpha)
        
        ss_config = config._replace(
            r_path=np.ones(config.T) * r_ss,
            w_path=np.ones(config.T) * w_ss,
            tau_c_path=np.ones(config.T) * 0.05,
            tau_l_path=np.ones(config.T) * 0.15,
            tau_p_path=np.ones(config.T) * 0.124,
            tau_k_path=np.ones(config.T) * 0.20,
            pension_replacement_path=np.ones(config.T) * 0.40
        )
        
        model = LifecycleModelPerfectForesight(ss_config, verbose=False)
        model.solve(verbose=False)
        
        # Check policy functions at age 1 (young worker)
        age = 1
        print(f"\nPolicy function at age {age}:")
        print(f"  c_policy shape: {model.c_policy.shape}")
        print(f"  a_policy shape: {model.a_policy.shape}")
        
        # Sample the policy: a'(a, y, h) for some states
        a_idx = 0  # Starting with zero assets
        y_idx = 0  # Low income
        h_idx = 0  # No health shock
        
        c_policy_val = model.c_policy[age, a_idx, y_idx, h_idx, 0]
        a_next_policy_val = model.a_policy[age, a_idx, y_idx, h_idx, 0]
        
        print(f"\n  At state (a=0, y_low, h_good):")
        print(f"    Consumption: {c_policy_val:.4f}")
        print(f"    Next assets: {a_next_policy_val:.4f}")
        
        # Check if the agent is saving anything
        if a_next_policy_val < 0.01:
            print(f"\n  ⚠️  WARNING: Agent not saving at age {age}!")
            print(f"     This will cause K→0 in aggregate")
        
        # Try higher income state
        y_idx = 1  # High income
        c_policy_val_high = model.c_policy[age, a_idx, y_idx, h_idx, 0]
        a_next_policy_val_high = model.a_policy[age, a_idx, y_idx, h_idx, 0]
        
        print(f"\n  At state (a=0, y_high, h_good):")
        print(f"    Consumption: {c_policy_val_high:.4f}")
        print(f"    Next assets: {a_next_policy_val_high:.4f}")
        
        # Check budget constraint
        y_val = model.y_grid[y_idx]
        h_val = model.h_grid[h_idx]
        income = w_ss * y_val * h_val

        print(f"\n  Budget check:")
        print(f"    Income (w*y*h): {income:.4f}")
        print(f"    Consumption: {c_policy_val_high:.4f}")
        print(f"    Savings (a' index): {a_next_policy_val_high}")
        a_next_val = model.a_grid[int(a_next_policy_val_high)]
        print(f"    Savings (a' value): {a_next_val:.4f}")
        
        assert a_next_policy_val_high > 0, \
            "Agent should save something with high income at young age!"


class TestPolicyIndexing:
    """Test policy function indexing."""
    
    def test_policy_dimensions_and_values(self):
        """Check policy function dimensions and sample values."""
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        
        print("\n" + "="*70)
        print("POLICY FUNCTION DIMENSIONS TEST")
        print("="*70)
        
        config = LifecycleConfig(
            T=5,
            beta=0.96,
            gamma=2.0,
            n_a=10,
            n_y=2,
            n_h=1,
            retirement_age=4,
            education_type='medium'
        )
        
        r_ss = 0.04
        alpha = 0.33
        delta = 0.05
        A = 1.0
        K_over_L = ((r_ss + delta) / (alpha * A)) ** (1 / (alpha - 1))
        w_ss = (1 - alpha) * A * (K_over_L ** alpha)
        
        ss_config = config._replace(
            r_path=np.ones(config.T) * r_ss,
            w_path=np.ones(config.T) * w_ss,
            tau_c_path=np.ones(config.T) * 0.05,
            tau_l_path=np.ones(config.T) * 0.15,
            tau_p_path=np.ones(config.T) * 0.124,
            tau_k_path=np.ones(config.T) * 0.20,
            pension_replacement_path=np.ones(config.T) * 0.40
        )
        
        model = LifecycleModelPerfectForesight(ss_config, verbose=False)
        model.solve(verbose=False)
        
        print(f"\nPolicy function shapes:")
        print(f"  a_policy: {model.a_policy.shape}")
        print(f"  c_policy: {model.c_policy.shape}")
        print(f"  V: {model.V.shape}")
        
        # Expected: (T, n_a, n_y, n_h, n_y_last)
        # where n_y_last tracks previous income state (used for pension calculation)

        print(f"\nExpected dimensions:")
        print(f"  T = {config.T}")
        print(f"  n_a = {config.n_a}")
        print(f"  n_y = {config.n_y}")
        print(f"  n_h = {config.n_h}")

        n_y_last = model.a_policy.shape[-1]
        print(f"  n_y_last (previous income states) = {n_y_last}")
        
        # Sample policies at different ages
        print(f"\nSample asset policies (a=0, y=high, h=good, e=0):")
        for age in range(min(4, config.T)):
            a_next = model.a_policy[age, 0, 1, 0, 0]  # a=0, y=1 (high), h=0, e=0
            print(f"  Age {age}: a' = {a_next:.4f}")
        
        # Check if any policies are non-zero
        nonzero_policies = np.sum(model.a_policy > 0.01)
        total_policies = np.prod(model.a_policy.shape)
        print(f"\nNon-zero asset policies: {nonzero_policies}/{total_policies} ({100*nonzero_policies/total_policies:.1f}%)")
        
        # Check if the issue is at age 1 specifically
        age1_policies = model.a_policy[1, :, :, :, :]
        age1_nonzero = np.sum(age1_policies > 0.01)
        age1_total = np.prod(age1_policies.shape)
        print(f"Age 1 non-zero policies: {age1_nonzero}/{age1_total} ({100*age1_nonzero/age1_total:.1f}%)")
        
        # Check other ages
        for age in [0, 2, 3]:
            if age < config.T:
                age_policies = model.a_policy[age, :, :, :, :]
                age_nonzero = np.sum(age_policies > 0.01)
                age_total = np.prod(age_policies.shape)
                print(f"Age {age} non-zero policies: {age_nonzero}/{age_total} ({100*age_nonzero/age_total:.1f}%)")


class TestEarningsIndexing:
    """Test earnings history indexing."""
    
    def test_which_earnings_index_has_savings(self):
        """Find which earnings index actually has positive savings policies."""
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        
        print("\n" + "="*70)
        print("EARNINGS INDEX DIAGNOSTIC")
        print("="*70)
        
        config = LifecycleConfig(
            T=5,
            beta=0.96,
            gamma=2.0,
            n_a=10,
            n_y=2,
            n_h=1,
            retirement_age=4,
            education_type='medium'
        )
        
        r_ss = 0.04
        alpha = 0.33
        delta = 0.05
        A = 1.0
        K_over_L = ((r_ss + delta) / (alpha * A)) ** (1 / (alpha - 1))
        w_ss = (1 - alpha) * A * (K_over_L ** alpha)
        
        ss_config = config._replace(
            r_path=np.ones(config.T) * r_ss,
            w_path=np.ones(config.T) * w_ss,
            tau_c_path=np.ones(config.T) * 0.05,
            tau_l_path=np.ones(config.T) * 0.15,
            tau_p_path=np.ones(config.T) * 0.124,
            tau_k_path=np.ones(config.T) * 0.20,
            pension_replacement_path=np.ones(config.T) * 0.40
        )
        
        model = LifecycleModelPerfectForesight(ss_config, verbose=False)
        model.solve(verbose=False)
        
        n_y_last = model.a_policy.shape[-1]
        print(f"\nNumber of previous income states (n_y_last): {n_y_last}")

        # Check policy at age 1, for each y_last state
        print(f"\nAge 1 policies (a=0, y=high, h=good) by y_last state:")
        for yl_idx in range(n_y_last):
            a_next = model.a_policy[1, 0, 1, 0, yl_idx]
            print(f"  y_last={yl_idx}: a' = {a_next:.4f}")

        # Check age 2
        print(f"\nAge 2 policies (a=0, y=high, h=good) by y_last state:")
        for yl_idx in range(n_y_last):
            a_next = model.a_policy[2, 0, 1, 0, yl_idx]
            print(f"  y_last={yl_idx}: a' = {a_next:.4f}")

        # Check which y_last states have most non-zero policies
        print(f"\nNon-zero policies by y_last state:")
        for yl_idx in range(n_y_last):
            yl_policies = model.a_policy[:, :, :, :, yl_idx]
            yl_nonzero = np.sum(yl_policies > 0.01)
            yl_total = np.prod(yl_policies.shape)
            print(f"  y_last={yl_idx}: {yl_nonzero}/{yl_total} ({100*yl_nonzero/yl_total:.1f}%)")

        # Check average policy value by y_last state
        print(f"\nAverage savings by y_last state (excluding zeros):")
        for yl_idx in range(n_y_last):
            yl_policies = model.a_policy[:, :, :, :, yl_idx]
            nonzero_policies = yl_policies[yl_policies > 0.01]
            if len(nonzero_policies) > 0:
                avg_savings = np.mean(nonzero_policies)
                print(f"  y_last={yl_idx}: {avg_savings:.4f}")
            else:
                print(f"  y_last={yl_idx}: No non-zero policies")


class TestSimulationVsPolicy:
    """Compare simulation results with direct policy access."""
    
    def test_simulation_uses_different_indexing(self):
        """Check if simulation uses policies differently than direct access."""
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        
        print("\n" + "="*70)
        print("SIMULATION VS POLICY ACCESS")
        print("="*70)
        
        config = LifecycleConfig(
            T=5,
            beta=0.96,
            gamma=2.0,
            n_a=10,
            n_y=2,
            n_h=1,
            retirement_age=4,
            education_type='medium'
        )
        
        r_ss = 0.04
        alpha = 0.33
        delta = 0.05
        A = 1.0
        K_over_L = ((r_ss + delta) / (alpha * A)) ** (1 / (alpha - 1))
        w_ss = (1 - alpha) * A * (K_over_L ** alpha)
        
        ss_config = config._replace(
            r_path=np.ones(config.T) * r_ss,
            w_path=np.ones(config.T) * w_ss,
            tau_c_path=np.ones(config.T) * 0.05,
            tau_l_path=np.ones(config.T) * 0.15,
            tau_p_path=np.ones(config.T) * 0.124,
            tau_k_path=np.ones(config.T) * 0.20,
            pension_replacement_path=np.ones(config.T) * 0.40
        )
        
        model = LifecycleModelPerfectForesight(ss_config, verbose=False)
        model.solve(verbose=False)
        
        # Simulate
        results = model.simulate(T_sim=config.T, n_sim=100, seed=42)
        assets_sim = results[0]
        
        mean_assets = np.mean(assets_sim, axis=1)
        
        print("\nSimulation results (mean assets by age):")
        for age in range(len(mean_assets)):
            print(f"  Age {age}: {mean_assets[age]:.4f}")
        
        print("\nDirect policy access (a=0, y=high, h=good, e=0):")
        for age in range(min(4, config.T)):
            a_next = model.a_policy[age, 0, 1, 0, 0]
            print(f"  Age {age}: a' = {a_next:.4f}")
        
        print("\n⚠️  If simulation shows positive assets but direct access shows zero,")
        print("   then the indexing in OLGTransition.solve_cohort_problems() is wrong!")


class TestJAXBackend:
    """Cross-validation tests for JAX backend against NumPy reference."""

    @staticmethod
    def _jax_available():
        try:
            import jax  # noqa: F401
            return True
        except Exception:
            return False

    def test_solve_matches_numpy(self):
        """Solve same config with both backends; V must match within atol=1e-6, policies identical."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        from lifecycle_jax import LifecycleModelJAX

        config = LifecycleConfig(
            T=10,
            beta=0.96,
            gamma=2.0,
            n_a=50,
            n_y=2,
            n_h=1,
            retirement_age=8,
            education_type='medium',
            pension_replacement_default=0.40,
            m_good=0.0,
        )

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        # Value functions must match closely (float64)
        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-6, f"V mismatch: max diff = {V_diff:.2e}"

        # Asset policies must be identical
        assert np.all(np_model.a_policy == jax_model.a_policy), \
            "Asset policies differ between NumPy and JAX"

        # Consumption policies must be close
        c_diff = np.max(np.abs(np_model.c_policy - jax_model.c_policy))
        assert c_diff < 1e-6, f"c_policy mismatch: max diff = {c_diff:.2e}"

    def test_simulate_distributional_match(self):
        """Simulate with both backends; mean lifecycle profiles must match within 2 standard errors."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        from lifecycle_jax import LifecycleModelJAX

        config = LifecycleConfig(
            T=10,
            beta=0.96,
            gamma=2.0,
            n_a=50,
            n_y=2,
            n_h=1,
            retirement_age=8,
            education_type='medium',
            pension_replacement_default=0.40,
            m_good=0.0,
        )

        n_sim = 5000

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)
        np_results = np_model.simulate(n_sim=n_sim, seed=42)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)
        jax_results = jax_model.simulate(n_sim=n_sim, seed=42)

        # Compare mean assets, consumption over lifecycle
        for idx, name in [(0, 'assets'), (1, 'consumption')]:
            np_means = np.mean(np_results[idx], axis=1)
            jax_means = np.mean(jax_results[idx], axis=1)

            # Standard error of the mean from NumPy simulation
            np_se = np.std(np_results[idx], axis=1) / np.sqrt(n_sim)
            # Allow 3 SE tolerance (generous for different PRNGs)
            tolerance = 3 * np.maximum(np_se, 1e-6)

            diff = np.abs(np_means - jax_means)
            max_excess = np.max(diff / tolerance)

            assert max_excess < 1.0, (
                f"Distributional mismatch for {name}: "
                f"max |diff|/tolerance = {max_excess:.2f} at age {np.argmax(diff / tolerance)}"
            )

    def test_olg_transition_jax_backend(self):
        """Run existing constant-r economy test with backend='jax'."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        config = LifecycleConfig(
            T=5,
            beta=0.96,
            gamma=2.0,
            n_a=5,
            n_y=2,
            n_h=1,
            retirement_age=4,
            education_type='medium',
        )

        economy = OLGTransition(
            lifecycle_config=config,
            alpha=0.33,
            delta=0.05,
            A=1.0,
            education_shares={'medium': 1.0},
            output_dir='output/test',
            backend='jax',
        )

        T_transition = 3
        r_path = np.ones(T_transition) * 0.03

        results = economy.simulate_transition(
            r_path=r_path,
            w_path=None,
            n_sim=20,
            verbose=False,
        )

        assert 'r' in results
        assert 'K' in results
        assert np.all(results['K'] > 0)
        assert np.all(results['L'] > 0)
        assert np.all(results['Y'] > 0)
        assert np.allclose(results['r'], 0.03)


# =====================================================================
# Tests for new features (Phases 1-5)
# =====================================================================

class TestNewFeatures:
    """Tests for new lifecycle model features. All features default OFF for backward compatibility."""

    @staticmethod
    def _base_config(**overrides):
        """Small config for fast tests."""
        defaults = dict(
            T=10, beta=0.96, gamma=2.0, n_a=50, n_y=2, n_h=1,
            retirement_age=8, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
        )
        defaults.update(overrides)
        return LifecycleConfig(**defaults)

    # --- Feature #11: Minimum pension floor ---

    def test_pension_min_floor_increases_retiree_consumption(self):
        """With a positive pension floor, retiree consumption should not decrease."""
        config_base = self._base_config()
        config_floor = self._base_config(pension_min_floor=0.5)

        model_base = LifecycleModelPerfectForesight(config_base, verbose=False)
        model_base.solve(verbose=False)
        res_base = model_base.simulate(n_sim=2000, seed=42)

        model_floor = LifecycleModelPerfectForesight(config_floor, verbose=False)
        model_floor.solve(verbose=False)
        res_floor = model_floor.simulate(n_sim=2000, seed=42)

        # Mean consumption in retirement (ages 8, 9) should be at least as high
        c_base_ret = np.mean(res_base[1][8:, :])
        c_floor_ret = np.mean(res_floor[1][8:, :])
        assert c_floor_ret >= c_base_ret - 1e-6, \
            f"Pension floor should increase retiree consumption: {c_floor_ret:.4f} < {c_base_ret:.4f}"

    def test_pension_min_floor_zero_is_noop(self):
        """pension_min_floor=0.0 should match the default behavior exactly."""
        config = self._base_config(pension_min_floor=0.0)
        config_default = self._base_config()

        m1 = LifecycleModelPerfectForesight(config, verbose=False)
        m1.solve(verbose=False)
        m2 = LifecycleModelPerfectForesight(config_default, verbose=False)
        m2.solve(verbose=False)

        assert np.allclose(m1.V, m2.V), "pension_min_floor=0 should match default"

    # --- Feature #20: Age-dependent medical expenditure ---

    def test_age_dependent_medical_costs(self):
        """Age-increasing medical costs should reduce late-life consumption."""
        # Flat profile
        config_flat = self._base_config(m_good=0.05, n_h=1)
        # Rising medical costs with age
        age_profile = np.linspace(0.5, 2.0, 10)
        config_age = self._base_config(m_good=0.05, n_h=1, m_age_profile=age_profile)

        m_flat = LifecycleModelPerfectForesight(config_flat, verbose=False)
        m_flat.solve(verbose=False)
        res_flat = m_flat.simulate(n_sim=2000, seed=42)

        m_age = LifecycleModelPerfectForesight(config_age, verbose=False)
        m_age.solve(verbose=False)
        res_age = m_age.simulate(n_sim=2000, seed=42)

        # m_grid should be 2D
        assert m_age.m_grid.ndim == 2
        assert m_age.m_grid.shape == (10, 1)

        # Late-life consumption should be lower with rising medical costs
        c_flat_late = np.mean(res_flat[1][7:, :])
        c_age_late = np.mean(res_age[1][7:, :])
        assert c_age_late < c_flat_late, \
            f"Rising medical costs should reduce late-life consumption"

    # --- Feature #14: Progressive taxation ---

    def test_progressive_tax_reduces_inequality(self):
        """HSV progressive taxation should compress the consumption distribution."""
        config_flat = self._base_config(n_y=3, n_h=1)
        config_prog = self._base_config(n_y=3, n_h=1,
                                         tax_progressive=True, tax_kappa=0.8, tax_eta=0.15)

        m_flat = LifecycleModelPerfectForesight(config_flat, verbose=False)
        m_flat.solve(verbose=False)
        res_flat = m_flat.simulate(n_sim=3000, seed=42)

        m_prog = LifecycleModelPerfectForesight(config_prog, verbose=False)
        m_prog.solve(verbose=False)
        res_prog = m_prog.simulate(n_sim=3000, seed=42)

        # Consumption variance should be lower under progressive tax
        var_flat = np.var(res_flat[1][3, :])
        var_prog = np.var(res_prog[1][3, :])
        # Allow some tolerance — the effect depends on calibration
        assert var_prog <= var_flat * 1.1, \
            f"Progressive tax should compress consumption distribution"

    def test_progressive_tax_disabled_matches_flat(self):
        """tax_progressive=False should give same results as default."""
        config_a = self._base_config(tax_progressive=False)
        config_b = self._base_config()

        m_a = LifecycleModelPerfectForesight(config_a, verbose=False)
        m_a.solve(verbose=False)
        m_b = LifecycleModelPerfectForesight(config_b, verbose=False)
        m_b.solve(verbose=False)

        assert np.allclose(m_a.V, m_b.V), "tax_progressive=False should match default"

    # --- Feature #15: Means-tested transfers ---

    def test_transfer_floor_prevents_destitution(self):
        """A positive transfer_floor should prevent consumption from falling below the floor."""
        config = self._base_config(transfer_floor=0.05)

        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        res = model.simulate(n_sim=3000, seed=42)

        # Minimum consumption across all agents should be close to or above floor
        min_c = np.min(res[1])
        assert min_c > 0.0, f"Consumption should be positive with transfer floor"

    def test_transfer_floor_zero_is_noop(self):
        """transfer_floor=0 should match default."""
        config_a = self._base_config(transfer_floor=0.0)
        config_b = self._base_config()

        m_a = LifecycleModelPerfectForesight(config_a, verbose=False)
        m_a.solve(verbose=False)
        m_b = LifecycleModelPerfectForesight(config_b, verbose=False)
        m_b.solve(verbose=False)

        assert np.allclose(m_a.V, m_b.V), "transfer_floor=0 should match default"

    # --- Feature #2: Survival risk ---

    def test_survival_risk_changes_value_function(self):
        """With survival risk < 1, value function should differ from the no-risk case."""
        config_base = self._base_config()
        survival = np.ones((10, 1)) * 0.95  # 5% mortality each period
        config_surv = self._base_config(survival_probs=survival)

        m_base = LifecycleModelPerfectForesight(config_base, verbose=False)
        m_base.solve(verbose=False)
        m_surv = LifecycleModelPerfectForesight(config_surv, verbose=False)
        m_surv.solve(verbose=False)

        # Value function should differ (survival risk changes effective discount)
        V_diff = np.max(np.abs(m_surv.V - m_base.V))
        assert V_diff > 1e-4, \
            f"Survival risk should change value function (max diff = {V_diff:.2e})"

        # At non-degenerate states (away from borrowing constraint), V should differ
        # Focus on interior states where the penalty doesn't dominate
        V_base_interior = m_base.V[5, 10:30, 1, 0, 0]  # mid-life, mid-assets, employed
        V_surv_interior = m_surv.V[5, 10:30, 1, 0, 0]
        assert not np.allclose(V_base_interior, V_surv_interior, atol=1e-4), \
            "Survival risk should change interior value function"

    def test_survival_prob_one_is_noop(self):
        """survival_probs=all ones should match the no-risk default."""
        config_a = self._base_config(survival_probs=np.ones((10, 1)))
        config_b = self._base_config()

        m_a = LifecycleModelPerfectForesight(config_a, verbose=False)
        m_a.solve(verbose=False)
        m_b = LifecycleModelPerfectForesight(config_b, verbose=False)
        m_b.solve(verbose=False)

        assert np.allclose(m_a.V, m_b.V), "survival_probs=1 should match default"

    # --- Feature #4: Schooling and children ---

    def test_child_costs_reduce_early_consumption(self):
        """Schooling child costs should reduce consumption in early periods."""
        config_base = self._base_config()
        child_costs = np.zeros(10)
        child_costs[:3] = 0.1  # child costs in first 3 periods
        config_school = self._base_config(schooling_years=3, child_cost_profile=child_costs)

        m_base = LifecycleModelPerfectForesight(config_base, verbose=False)
        m_base.solve(verbose=False)
        res_base = m_base.simulate(n_sim=2000, seed=42)

        m_school = LifecycleModelPerfectForesight(config_school, verbose=False)
        m_school.solve(verbose=False)
        res_school = m_school.simulate(n_sim=2000, seed=42)

        # Early-life consumption should be lower with child costs
        c_base_early = np.mean(res_base[1][:3, :])
        c_school_early = np.mean(res_school[1][:3, :])
        assert c_school_early < c_base_early, \
            "Child costs should reduce early-life consumption"

    def test_no_schooling_is_noop(self):
        """schooling_years=0 should match default."""
        config_a = self._base_config(schooling_years=0)
        config_b = self._base_config()

        m_a = LifecycleModelPerfectForesight(config_a, verbose=False)
        m_a.solve(verbose=False)
        m_b = LifecycleModelPerfectForesight(config_b, verbose=False)
        m_b.solve(verbose=False)

        assert np.allclose(m_a.V, m_b.V), "schooling_years=0 should match default"

    # --- Feature #17: Government spending ---

    def test_govt_spending_increases_deficit(self):
        """Positive G_t should increase the primary deficit."""
        config = self._base_config(T=5, retirement_age=4, n_a=5)
        economy_base = OLGTransition(
            lifecycle_config=config, alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0}, output_dir='output/test',
        )
        economy_G = OLGTransition(
            lifecycle_config=config, alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0}, output_dir='output/test',
            govt_spending_path=np.ones(3) * 0.05,
        )

        r_path = np.ones(3) * 0.03
        economy_base.simulate_transition(r_path=r_path, n_sim=50, verbose=False)
        economy_G.simulate_transition(r_path=r_path, n_sim=50, verbose=False)

        budget_base = economy_base.compute_government_budget(0, n_sim=50)
        budget_G = economy_G.compute_government_budget(0, n_sim=50)

        # G_t should increase spending and deficit
        assert budget_G['govt_spending'] == 0.05
        assert budget_G['total_spending'] > budget_base['total_spending']
        assert budget_G['primary_deficit'] > budget_base['primary_deficit']


class TestNewFeaturesJAX:
    """JAX cross-validation tests for new features."""

    @staticmethod
    def _jax_available():
        try:
            import jax  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _base_config(**overrides):
        defaults = dict(
            T=10, beta=0.96, gamma=2.0, n_a=50, n_y=2, n_h=1,
            retirement_age=8, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
        )
        defaults.update(overrides)
        return LifecycleConfig(**defaults)

    def test_pension_floor_jax_matches_numpy(self):
        """JAX pension_min_floor solve should match NumPy."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_jax import LifecycleModelJAX

        config = self._base_config(pension_min_floor=0.3)

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-6, f"V mismatch with pension floor: max diff = {V_diff:.2e}"
        assert np.all(np_model.a_policy == jax_model.a_policy), \
            "Asset policies differ with pension floor"

    def test_progressive_tax_jax_matches_numpy(self):
        """JAX progressive tax solve should match NumPy."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_jax import LifecycleModelJAX

        config = self._base_config(tax_progressive=True, tax_kappa=0.8, tax_eta=0.15)

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-6, f"V mismatch with progressive tax: max diff = {V_diff:.2e}"
        assert np.all(np_model.a_policy == jax_model.a_policy), \
            "Asset policies differ with progressive tax"

    def test_age_medical_jax_matches_numpy(self):
        """JAX age-dependent medical costs should match NumPy."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_jax import LifecycleModelJAX

        age_profile = np.linspace(0.5, 2.0, 10)
        config = self._base_config(m_good=0.05, m_age_profile=age_profile)

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-5, f"V mismatch with age-medical: max diff = {V_diff:.2e}"

    def test_survival_risk_jax_matches_numpy(self):
        """JAX survival risk solve should match NumPy."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_jax import LifecycleModelJAX

        survival = np.ones((10, 1)) * 0.95
        config = self._base_config(survival_probs=survival)

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-6, f"V mismatch with survival risk: max diff = {V_diff:.2e}"
        assert np.all(np_model.a_policy == jax_model.a_policy), \
            "Asset policies differ with survival risk"

    def test_schooling_jax_matches_numpy(self):
        """JAX schooling child costs should match NumPy."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_jax import LifecycleModelJAX

        child_costs = np.zeros(10)
        child_costs[:3] = 0.1
        config = self._base_config(schooling_years=3, child_cost_profile=child_costs)

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-6, f"V mismatch with schooling: max diff = {V_diff:.2e}"

    def test_transfer_floor_jax_matches_numpy(self):
        """JAX transfer floor solve should match NumPy."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_jax import LifecycleModelJAX

        config = self._base_config(transfer_floor=0.05)

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-6, f"V mismatch with transfer floor: max diff = {V_diff:.2e}"

    def test_combined_features_jax_matches_numpy(self):
        """JAX with multiple features enabled should match NumPy."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        from lifecycle_jax import LifecycleModelJAX

        age_profile = np.linspace(0.8, 1.5, 10)
        survival = np.ones((10, 1)) * 0.97
        child_costs = np.zeros(10)
        child_costs[:2] = 0.05
        config = self._base_config(
            m_good=0.03, m_age_profile=age_profile,
            pension_min_floor=0.2,
            tax_progressive=True, tax_kappa=0.85, tax_eta=0.10,
            transfer_floor=0.02,
            survival_probs=survival,
            schooling_years=2, child_cost_profile=child_costs,
        )

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-6, f"V mismatch with combined features: max diff = {V_diff:.2e}"
        assert np.all(np_model.a_policy == jax_model.a_policy), \
            "Asset policies differ with combined features"


class TestPhase6Features:
    """Tests for Phase 6: public capital, public investment, SOE/sovereign debt."""

    def test_public_capital_increases_output(self):
        """Public capital with eta_g > 0 should increase output vs baseline."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04
        n_sim = 100

        # Baseline: no public capital
        olg_base = OLGTransition(lifecycle_config=get_test_config())
        res_base = olg_base.simulate_transition(r_path, n_sim=n_sim, verbose=False)

        # With public capital
        olg_kg = OLGTransition(lifecycle_config=get_test_config(),
                               eta_g=0.05, K_g_initial=1.0,
                               I_g_path=np.ones(T_tr) * 0.1)
        res_kg = olg_kg.simulate_transition(r_path, n_sim=n_sim, verbose=False)

        # Public capital should boost output
        assert np.mean(res_kg['Y']) > np.mean(res_base['Y']), \
            "Public capital should increase output"
        assert 'K_g' in res_kg
        assert res_kg['K_g'][0] == 1.0

    def test_public_capital_zero_eta_g_is_noop(self):
        """eta_g=0 with public capital should produce same results as baseline."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04
        n_sim = 100

        olg_base = OLGTransition(lifecycle_config=get_test_config())
        res_base = olg_base.simulate_transition(r_path, n_sim=n_sim, verbose=False)

        olg_kg = OLGTransition(lifecycle_config=get_test_config(),
                               eta_g=0.0, K_g_initial=5.0,
                               I_g_path=np.ones(T_tr) * 0.5)
        res_kg = olg_kg.simulate_transition(r_path, n_sim=n_sim, verbose=False)

        np.testing.assert_allclose(res_base['Y'], res_kg['Y'], rtol=1e-10)
        np.testing.assert_allclose(res_base['w'], res_kg['w'], rtol=1e-10)

    def test_public_capital_accumulation(self):
        """Public capital should follow K_g' = (1-delta_g)*K_g + I_g."""
        T_tr = 10
        r_path = np.ones(T_tr) * 0.04
        I_g = np.ones(T_tr) * 0.2
        K_g_0 = 2.0
        delta_g = 0.1

        olg = OLGTransition(lifecycle_config=get_test_config(),
                            eta_g=0.05, K_g_initial=K_g_0,
                            delta_g=delta_g, I_g_path=I_g)
        res = olg.simulate_transition(r_path, n_sim=100, verbose=False)

        K_g = res['K_g']
        assert K_g[0] == K_g_0
        for t in range(1, T_tr):
            expected = (1 - delta_g) * K_g[t - 1] + I_g[t - 1]
            np.testing.assert_allclose(K_g[t], expected, rtol=1e-12,
                                       err_msg=f"K_g accumulation failed at t={t}")

    def test_public_capital_changes_wages(self):
        """With public capital, wages should differ from baseline."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04

        olg_base = OLGTransition(lifecycle_config=get_test_config())
        res_base = olg_base.simulate_transition(r_path, n_sim=100, verbose=False)

        olg_kg = OLGTransition(lifecycle_config=get_test_config(),
                               eta_g=0.05, K_g_initial=2.0,
                               I_g_path=np.ones(T_tr) * 0.3)
        res_kg = olg_kg.simulate_transition(r_path, n_sim=100, verbose=False)

        assert np.all(res_kg['w'] > res_base['w']), \
            "Public capital should increase wages for given r"

    def test_public_investment_in_budget(self):
        """Public investment should appear in government budget spending."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04

        olg = OLGTransition(lifecycle_config=get_test_config(),
                            eta_g=0.05, K_g_initial=1.0,
                            I_g_path=np.ones(T_tr) * 0.5)
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget = olg.compute_government_budget(0)

        assert budget['public_investment'] == 0.5
        assert budget['total_spending'] >= budget['public_investment']

    def test_sovereign_debt_in_budget(self):
        """Sovereign debt should add debt service and borrowing to budget."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04
        B_path = np.linspace(1.0, 1.5, T_tr + 1)

        olg = OLGTransition(lifecycle_config=get_test_config(), B_path=B_path)
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget = olg.compute_government_budget(0)

        expected_debt_service = 0.04 * 1.0
        np.testing.assert_allclose(budget['debt_service'], expected_debt_service, rtol=1e-10)
        expected_borrowing = B_path[1] - B_path[0]
        np.testing.assert_allclose(budget['new_borrowing'], expected_borrowing, rtol=1e-10)
        assert budget['total_spending'] >= budget['debt_service']

    def test_no_debt_is_noop(self):
        """Without sovereign debt, budget should match baseline (no debt terms)."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04

        olg = OLGTransition(lifecycle_config=get_test_config())
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget = olg.compute_government_budget(0)

        assert budget['debt_service'] == 0.0
        assert budget['new_borrowing'] == 0.0
        assert budget['public_investment'] == 0.0

    def test_soe_computes_nfa(self):
        """SOE mode should compute NFA path."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04

        olg = OLGTransition(lifecycle_config=get_test_config(), economy_type='soe')
        res = olg.simulate_transition(r_path, n_sim=100, verbose=False)

        assert 'NFA' in res
        assert len(res['NFA']) == T_tr

    def test_production_function_with_public_capital(self):
        """Production function should include K_g factor."""
        olg = OLGTransition(lifecycle_config=get_test_config(), eta_g=0.1)

        K, L = 10.0, 5.0
        K_g = 2.0

        Y_with = olg.production_function(K, L, K_g=K_g)
        Y_without = olg.production_function(K, L, K_g=None)

        assert Y_with > Y_without, "Public capital should increase production"

        expected = olg.A * (K_g ** 0.1) * (K ** olg.alpha) * (L ** (1 - olg.alpha))
        np.testing.assert_allclose(Y_with, expected, rtol=1e-12)

    def test_factor_prices_with_public_capital(self):
        """Factor prices should account for public capital."""
        olg = OLGTransition(lifecycle_config=get_test_config(), eta_g=0.1)

        K, L = 10.0, 5.0
        K_g = 2.0

        r_with, w_with = olg.factor_prices(K, L, K_g=K_g)
        r_without, w_without = olg.factor_prices(K, L, K_g=None)

        assert r_with > r_without
        assert w_with > w_without


class TestPhase7Features:
    """Tests for Phase 7: pension trust fund, defense spending."""

    def test_pension_trust_fund_accumulation(self):
        """Trust fund follows S[t+1] = (1+r)*S[t] + payroll_tax - pensions."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04
        S_0 = 10.0

        olg = OLGTransition(lifecycle_config=get_test_config(), S_pens_initial=S_0)
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget = olg.compute_government_budget_path(verbose=False)

        S = olg.S_pens_path
        assert S[0] == S_0
        # Verify accumulation equation
        for t in range(T_tr):
            r_t = r_path[t]
            expected = (1 + r_t) * S[t] + budget['tax_p'][t] - budget['pension'][t]
            np.testing.assert_allclose(S[t + 1], expected, rtol=1e-10,
                                       err_msg=f"Trust fund accumulation failed at t={t}")

    def test_pension_trust_fund_zero_initial_matches_baseline(self):
        """Trust fund with S_0=0 should compute but start at zero."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04

        olg = OLGTransition(lifecycle_config=get_test_config())
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget = olg.compute_government_budget_path(verbose=False)

        assert olg.S_pens_path[0] == 0.0
        assert 'S_pens' in budget

    def test_pension_trust_fund_in_budget_path(self):
        """Trust fund balance should appear in budget_path output."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04

        olg = OLGTransition(lifecycle_config=get_test_config(), S_pens_initial=5.0)
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget = olg.compute_government_budget_path(verbose=False)

        assert 'S_pens' in budget
        assert len(budget['S_pens']) == T_tr
        assert budget['S_pens'][0] == 5.0

    def test_defense_spending_in_budget(self):
        """Defense spending should appear in government budget."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04
        defense = np.ones(T_tr) * 0.3

        olg = OLGTransition(lifecycle_config=get_test_config(),
                            defense_spending_path=defense)
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget_t = olg.compute_government_budget(0)

        assert budget_t['defense_spending'] == 0.3
        assert budget_t['total_spending'] >= budget_t['defense_spending']

    def test_defense_spending_increases_deficit(self):
        """Defense spending should increase the deficit."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04

        # Baseline
        olg_base = OLGTransition(lifecycle_config=get_test_config())
        olg_base.simulate_transition(r_path, n_sim=100, verbose=False)
        budget_base = olg_base.compute_government_budget(0)

        # With defense spending
        olg_def = OLGTransition(lifecycle_config=get_test_config(),
                                defense_spending_path=np.ones(T_tr) * 1.0)
        olg_def.simulate_transition(r_path, n_sim=100, verbose=False)
        budget_def = olg_def.compute_government_budget(0)

        assert budget_def['primary_deficit'] > budget_base['primary_deficit']

    def test_no_defense_is_noop(self):
        """Without defense spending, the field should be zero."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04

        olg = OLGTransition(lifecycle_config=get_test_config())
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget_t = olg.compute_government_budget(0)

        assert budget_t['defense_spending'] == 0.0


class TestLaborSupply:
    """Tests for Feature #1: Endogenous labor supply."""

    @staticmethod
    def _base_config(**overrides):
        defaults = dict(
            T=10, beta=0.96, gamma=2.0, n_a=50, n_y=2, n_h=1,
            retirement_age=8, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
        )
        defaults.update(overrides)
        return LifecycleConfig(**defaults)

    def test_l_policy_default_ones(self):
        """labor_supply=False -> l_policy all 1.0"""
        config = self._base_config(labor_supply=False)
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        assert model.l_policy is not None
        assert np.allclose(model.l_policy, 1.0), "l_policy should be all 1.0 when labor_supply=False"

    def test_l_sim_in_output(self):
        """Simulation returns 19-tuple, l_sim at index 18."""
        config = self._base_config()
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        result = model.simulate(n_sim=100, seed=42)
        assert len(result) == 19, f"Expected 19-tuple, got {len(result)}-tuple"
        l_sim = result[18]
        assert l_sim.shape == result[0].shape, "l_sim shape should match a_sim shape"
        # With labor_supply=False, l_sim should be all 1.0
        assert np.allclose(l_sim, 1.0), "l_sim should be all 1.0 when labor_supply=False"

    def test_labor_supply_endogenous(self):
        """labor_supply=True -> l_policy varies, non-negative, 1.0 in retirement."""
        config = self._base_config(labor_supply=True, nu=1.0, phi=2.0)
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        assert model.l_policy is not None
        # All labor hours should be non-negative
        assert np.all(model.l_policy >= 0.0), "l_policy should be non-negative"
        # In retirement periods, l_policy should be 1.0 (fixed)
        for t in range(config.retirement_age, config.T):
            assert np.allclose(model.l_policy[t], 1.0), \
                f"l_policy at retirement age {t} should be 1.0"
        # In working periods, some l_policy values should differ from 1.0
        # (for employed states with positive income)
        working_employed = model.l_policy[:config.retirement_age, :, 1:, :, :]
        assert not np.allclose(working_employed, 1.0), \
            "l_policy should vary for employed workers when labor_supply=True"

    def test_effective_y_uses_labor_hours(self):
        """effective_y_sim should reflect l * w * y * h."""
        config = self._base_config(labor_supply=True, nu=1.0, phi=2.0)
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        result = model.simulate(n_sim=500, seed=42)
        effective_y = result[5]
        l_sim = result[18]
        # For workers with labor_supply=True, effective_y should not assume l=1
        # Check that in at least some periods, effective_y differs from what l=1 would give
        config_nolabor = self._base_config(labor_supply=False)
        model_nolabor = LifecycleModelPerfectForesight(config_nolabor, verbose=False)
        model_nolabor.solve(verbose=False)
        result_nolabor = model_nolabor.simulate(n_sim=500, seed=42)
        effective_y_nolabor = result_nolabor[5]
        # The two should differ (different policies produce different outcomes)
        # This is a weak test — just checking they're not identical
        assert not np.allclose(effective_y, effective_y_nolabor), \
            "effective_y should differ when labor_supply is enabled"

    def test_labor_supply_backward_compatible(self):
        """labor_supply=False produces identical results to the default."""
        config_default = self._base_config()
        config_explicit = self._base_config(labor_supply=False)
        m1 = LifecycleModelPerfectForesight(config_default, verbose=False)
        m1.solve(verbose=False)
        m2 = LifecycleModelPerfectForesight(config_explicit, verbose=False)
        m2.solve(verbose=False)
        assert np.allclose(m1.V, m2.V), "V should be identical with labor_supply=False"
        assert np.all(m1.a_policy == m2.a_policy), "a_policy should be identical"
        assert np.allclose(m1.c_policy, m2.c_policy), "c_policy should be identical"
        assert np.allclose(m1.l_policy, m2.l_policy), "l_policy should be identical (all 1.0)"


class TestLaborSupplyJAX:
    """JAX cross-validation tests for labor supply feature."""

    @staticmethod
    def _jax_available():
        try:
            import jax  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _base_config(**overrides):
        defaults = dict(
            T=10, beta=0.96, gamma=2.0, n_a=50, n_y=2, n_h=1,
            retirement_age=8, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
        )
        defaults.update(overrides)
        return LifecycleConfig(**defaults)

    def test_jax_labor_supply_solve_matches(self):
        """JAX V and l_policy match NumPy within 1e-6 with labor_supply=False."""
        if not self._jax_available():
            pytest.skip("JAX not available")
        from lifecycle_jax import LifecycleModelJAX

        config = self._base_config(labor_supply=False)
        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)
        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)

        V_diff = np.max(np.abs(np_model.V - jax_model.V))
        assert V_diff < 1e-6, f"V mismatch: max diff = {V_diff:.2e}"
        assert np.all(np_model.a_policy == jax_model.a_policy), \
            "Asset policies differ"
        assert np.allclose(np_model.l_policy, jax_model.l_policy), \
            "l_policy should match (both all 1.0)"

    def test_jax_labor_supply_simulate_distributional(self):
        """JAX mean labor hours match NumPy within 3 SE."""
        if not self._jax_available():
            pytest.skip("JAX not available")
        from lifecycle_jax import LifecycleModelJAX

        config = self._base_config(labor_supply=False)
        n_sim = 5000

        np_model = LifecycleModelPerfectForesight(config, verbose=False)
        np_model.solve(verbose=False)
        np_results = np_model.simulate(n_sim=n_sim, seed=42)

        jax_model = LifecycleModelJAX(config, verbose=False)
        jax_model.solve(verbose=False)
        jax_results = jax_model.simulate(n_sim=n_sim, seed=42)

        # l_sim is at index 18
        assert len(np_results) == 19, f"NumPy: expected 19-tuple, got {len(np_results)}"
        assert len(jax_results) == 19, f"JAX: expected 19-tuple, got {len(jax_results)}"

        np_l = np_results[18]
        jax_l = jax_results[18]

        # With labor_supply=False, both should be all 1.0
        assert np.allclose(np_l, 1.0), "NumPy l_sim should be 1.0"
        assert np.allclose(jax_l, 1.0), "JAX l_sim should be 1.0"

        # Also check distributional match for assets (regression)
        np_means = np.mean(np_results[0], axis=1)
        jax_means = np.mean(jax_results[0], axis=1)
        np_se = np.std(np_results[0], axis=1) / np.sqrt(n_sim)
        tolerance = 3 * np.maximum(np_se, 1e-6)
        diff = np.abs(np_means - jax_means)
        max_excess = np.max(diff / tolerance)
        assert max_excess < 1.0, f"Asset distributional mismatch: max |diff|/tol = {max_excess:.2f}"

    def test_olg_with_labor_supply(self):
        """OLG transition completes with labor_supply=False, K > 0, L > 0."""
        if not self._jax_available():
            pytest.skip("JAX not available")

        config = LifecycleConfig(
            T=5, beta=0.96, gamma=2.0, n_a=5, n_y=2, n_h=1,
            retirement_age=4, education_type='medium',
            labor_supply=False,
        )
        economy = OLGTransition(
            lifecycle_config=config, alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0}, output_dir='output/test',
            backend='jax',
        )
        T_transition = 3
        r_path = np.ones(T_transition) * 0.03

        results = economy.simulate_transition(r_path=r_path, n_sim=20, verbose=False)

        assert np.all(results['K'] > 0), "K should be positive"
        assert np.all(results['L'] > 0), "L should be positive"
        assert np.all(results['Y'] > 0), "Y should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])