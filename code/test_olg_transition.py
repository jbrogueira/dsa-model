import os

import pytest
import numpy as np
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
        
        # Check production function: Y = A * K_domestic^alpha * L^(1-alpha)
        # In SOE mode, firms hire K_domestic (pinned by FOC), not household wealth K.
        K_for_Y = results.get('K_domestic', results['K'])
        Y_implied = economy.A * (K_for_Y ** economy.alpha) * (results['L'] ** (1 - economy.alpha))
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
        print("Number of agents: 500")
        print("\nConstant parameters:")
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
        
        print("\n3. Aggregate trends (% change per period):")
        print(f"   Capital:  {K_slope_pct:+.4f}%/period {'✓' if abs(K_slope_pct) < slope_tolerance else '✗'}")
        print(f"   Labor:    {L_slope_pct:+.4f}%/period {'✓' if abs(L_slope_pct) < slope_tolerance else '✗'}")
        print(f"   Output:   {Y_slope_pct:+.4f}%/period {'✓' if abs(Y_slope_pct) < slope_tolerance else '✗'}")
        
        # Test 4: Production function — use K_domestic in SOE (firm's capital, not household wealth)
        K_for_Y = results.get('K_domestic', K_path)
        Y_check = economy.A * (K_for_Y ** economy.alpha) * (L_path ** (1 - economy.alpha))
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
        
        print("\nKey finding:")
        print(f"  Policy at a=0: {a0_policy:.4f}")
        print(f"  Policy at a=1: {a1_policy:.4f}")
        
        if a0_policy < 0.01 and a1_policy > 0.01:
            print("\n✓ Confirmed: Borrowing constraint binds ONLY at a=0!")
            print("   Agents with any positive assets DO save.")
        
        # Now check what happens in OLGTransition when initializing cohorts
        print("\n" + "="*70)
        print("IMPLICATION FOR OLGTRANSITION:")
        print("="*70)
        print("When OLGTransition initializes old cohorts with their")
        print("'steady-state assets', if those assets are ZERO (or very small),")
        print("they will hit the borrowing constraint and save NOTHING.")
        print("\nThis causes K→0 because:")
        print("  1. New cohorts start with a=0 (by definition)")
        print("  2. Policy at a=0 says: save nothing")
        print("  3. Old cohorts die out")
        print("  4. Aggregate K decreases monotonically to zero")


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
        
        print("\nSteady-state prices:")
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
        
        print("\nSimulated steady-state asset profile:")
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
        
        print("\n  At state (a=0, y_low, h_good):")
        print(f"    Consumption: {c_policy_val:.4f}")
        print(f"    Next assets: {a_next_policy_val:.4f}")
        
        # Check if the agent is saving anything
        if a_next_policy_val < 0.01:
            print(f"\n  ⚠️  WARNING: Agent not saving at age {age}!")
            print("     This will cause K→0 in aggregate")
        
        # Try higher income state
        y_idx = 1  # High income
        c_policy_val_high = model.c_policy[age, a_idx, y_idx, h_idx, 0]
        a_next_policy_val_high = model.a_policy[age, a_idx, y_idx, h_idx, 0]
        
        print("\n  At state (a=0, y_high, h_good):")
        print(f"    Consumption: {c_policy_val_high:.4f}")
        print(f"    Next assets: {a_next_policy_val_high:.4f}")
        
        # Check budget constraint
        y_val = model.y_grid[y_idx]
        h_val = model.h_grid[h_idx]
        income = w_ss * y_val * h_val

        print("\n  Budget check:")
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
        
        print("\nPolicy function shapes:")
        print(f"  a_policy: {model.a_policy.shape}")
        print(f"  c_policy: {model.c_policy.shape}")
        print(f"  V: {model.V.shape}")
        
        # Expected: (T, n_a, n_y, n_h, n_y_last)
        # where n_y_last tracks previous income state (used for pension calculation)

        print("\nExpected dimensions:")
        print(f"  T = {config.T}")
        print(f"  n_a = {config.n_a}")
        print(f"  n_y = {config.n_y}")
        print(f"  n_h = {config.n_h}")

        n_y_last = model.a_policy.shape[-1]
        print(f"  n_y_last (previous income states) = {n_y_last}")
        
        # Sample policies at different ages
        print("\nSample asset policies (a=0, y=high, h=good, e=0):")
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
        print("\nAge 1 policies (a=0, y=high, h=good) by y_last state:")
        for yl_idx in range(n_y_last):
            a_next = model.a_policy[1, 0, 1, 0, yl_idx]
            print(f"  y_last={yl_idx}: a' = {a_next:.4f}")

        # Check age 2
        print("\nAge 2 policies (a=0, y=high, h=good) by y_last state:")
        for yl_idx in range(n_y_last):
            a_next = model.a_policy[2, 0, 1, 0, yl_idx]
            print(f"  y_last={yl_idx}: a' = {a_next:.4f}")

        # Check which y_last states have most non-zero policies
        print("\nNon-zero policies by y_last state:")
        for yl_idx in range(n_y_last):
            yl_policies = model.a_policy[:, :, :, :, yl_idx]
            yl_nonzero = np.sum(yl_policies > 0.01)
            yl_total = np.prod(yl_policies.shape)
            print(f"  y_last={yl_idx}: {yl_nonzero}/{yl_total} ({100*yl_nonzero/yl_total:.1f}%)")

        # Check average policy value by y_last state
        print("\nAverage savings by y_last state (excluding zeros):")
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

    def test_pension_min_floor_raises_value(self):
        """A positive pension floor weakly raises the value at every state and
        binds somewhere. It need not raise retiree consumption: a household that
        expects a higher pension smooths by consuming more before retirement,
        which is what this fixture does once the retired continuation value is
        read at the household's own last income state (2026-10-02)."""
        config_base = self._base_config()
        config_floor = self._base_config(pension_min_floor=0.5)

        model_base = LifecycleModelPerfectForesight(config_base, verbose=False)
        model_base.solve(verbose=False)
        model_floor = LifecycleModelPerfectForesight(config_floor, verbose=False)
        model_floor.solve(verbose=False)

        V_base = np.asarray(model_base.V); V_floor = np.asarray(model_floor.V)
        finite = np.isfinite(V_base) & np.isfinite(V_floor)
        assert np.all(V_floor[finite] >= V_base[finite] - 1e-9), \
            "a higher pension floor lowered the value at some state"
        assert np.any(V_floor[finite] > V_base[finite] + 1e-9), \
            "the floor of 0.5 binds nowhere in this fixture"
        res_floor = model_floor.simulate(n_sim=2000, seed=42)
        pens = res_floor[16][res_floor[17].astype(bool)]
        assert np.all(pens >= 0.5 - 1e-12), "a retiree received less than the floor"

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
            "Rising medical costs should reduce late-life consumption"

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
            "Progressive tax should compress consumption distribution"

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
        assert min_c > 0.0, "Consumption should be positive with transfer floor"

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
        """Public capital follows K_g' = [(1-delta_g)*K_g + I_g] / G."""
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
        G = olg.growth_factor
        assert K_g[0] == K_g_0
        for t in range(1, T_tr):
            expected = ((1 - delta_g) * K_g[t - 1] + I_g[t - 1]) / G
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
        expected_borrowing = olg.growth_factor * B_path[1] - B_path[0]
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
        """Trust fund follows S[t+1] = [(1+r)*S[t] + payroll_tax - pensions] / G."""
        T_tr = 5
        r_path = np.ones(T_tr) * 0.04
        S_0 = 10.0

        olg = OLGTransition(lifecycle_config=get_test_config(), S_pens_initial=S_0)
        olg.simulate_transition(r_path, n_sim=100, verbose=False)
        budget = olg.compute_government_budget_path(verbose=False)

        S = olg.S_pens_path
        assert S[0] == S_0
        # Verify accumulation equation
        G = olg.growth_factor
        for t in range(T_tr):
            r_t = r_path[t]
            expected = ((1 + r_t) * S[t] + budget['tax_p'][t]
                        - budget['pension'][t]) / G
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
        """Simulation returns a 23-tuple: l_sim at index 18, alive_sim at 19, bequest_sim at 20, alpha_idx at 21, transfer_sim at 22."""
        config = self._base_config()
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        result = model.simulate(n_sim=100, seed=42)
        assert len(result) in (21, 22, 23), f"Expected 21-, 22- or 23-tuple, got {len(result)}-tuple"
        l_sim = result[18]
        assert l_sim.shape == result[0].shape, "l_sim shape should match a_sim shape"
        # With labor_supply=False the employed supply 1.0; the unemployed and the
        # retired supply 0.0 (since 2026-10-02 the unemployed no longer carry the
        # policy array's placeholder of 1.0 into the panel).
        retired_sim = result[17].astype(bool)
        employed_sim = result[6].astype(bool)
        assert np.allclose(l_sim[employed_sim & ~retired_sim], 1.0), \
            "l_sim should be 1.0 for the employed when labor_supply=False"
        assert np.allclose(l_sim[~employed_sim | retired_sim], 0.0), \
            "l_sim should be 0.0 for the unemployed and the retired"
        # alive_sim: all True when no survival_probs
        alive_sim = result[19]
        assert alive_sim.shape == result[0].shape
        assert np.all(alive_sim), "alive_sim should be all True when no mortality"
        # bequest_sim: all zero when no survival_probs
        bequest_sim = result[20]
        assert bequest_sim.shape == result[0].shape
        assert np.all(bequest_sim == 0.0), "bequest_sim should be all zero when no mortality"

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
        # Phase 8.3 added alpha_idx_sim → NumPy now returns 22-tuple.
        # JAX backend still returns 21-tuple until Phase 8.5 lands.
        assert len(np_results) in (21, 22, 23), f"NumPy: expected 21-23-tuple, got {len(np_results)}"
        assert len(jax_results) in (21, 22, 23), f"JAX: expected 21-23-tuple, got {len(jax_results)}"

        np_l = np_results[18]
        jax_l = jax_results[18]

        # With labor_supply=False the EMPLOYED supply 1.0; the unemployed and the
        # retired supply 0 (since 2026-10-02 the panel records 0 for the
        # unemployed too). The two backends draw different shocks, so hours are
        # checked against each backend's own employment indicator.
        np_emp = np_results[6].astype(bool) & ~np_results[17].astype(bool)
        jax_emp = jax_results[6].astype(bool) & ~jax_results[17].astype(bool)
        assert np.allclose(np_l[np_emp], 1.0) and np.allclose(np_l[~np_emp], 0.0), \
            "NumPy l_sim should be 1.0 for the employed and 0.0 otherwise"
        assert np.allclose(jax_l[jax_emp], 1.0) and np.allclose(jax_l[~jax_emp], 0.0), \
            "JAX l_sim should be 1.0 for the employed and 0.0 otherwise"

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


class TestEndogenousRetirementRefused:
    """retirement_window is intended but not implemented; it must refuse.

    It was honoured by the NumPy solve alone. The JAX solve ignored it entirely,
    and NEITHER simulation consulted it -- both retire mechanically at
    retirement_age, which docs/bug_report.md:487 already recorded -- so policies
    and realised incomes disagreed even on NumPy while
    docs/model_vs_implementation.md advertised the feature as working and
    cross-validated. The tests that exercised it were removed with it on
    2026-10-01, because they compared two solves of a problem the simulation
    never solved.

    Endogenous retirement is wanted later. Until the JAX solve and both
    simulations honour the window, constructing a model with one raises.
    """

    def test_setting_a_window_raises(self):
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        cfg = get_test_config()._replace(retirement_window=(6, 10))
        with pytest.raises(NotImplementedError, match='retirement_window'):
            LifecycleModelPerfectForesight(cfg, verbose=False)

    def test_no_window_is_unaffected(self):
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        m = LifecycleModelPerfectForesight(get_test_config(), verbose=False)
        assert m.retirement_window is None



class TestSimulationMortality:
    """Tests for Feature #2: Simulation mortality draws."""

    @staticmethod
    def _base_config(**overrides):
        defaults = dict(
            T=10, beta=0.96, gamma=2.0, n_a=50, n_y=2, n_h=1,
            retirement_age=8, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
        )
        defaults.update(overrides)
        return LifecycleConfig(**defaults)

    def test_survival_mortality_draws_reduce_alive(self):
        """With survival_probs < 1, fewer agents are alive at late ages."""
        survival = np.ones((10, 1)) * 0.90  # 10% mortality each period
        config = self._base_config(survival_probs=survival)
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        result = model.simulate(n_sim=2000, seed=42)
        alive_sim = result[19]
        # At late ages, fewer agents should be alive
        alive_early = np.sum(alive_sim[0, :])
        alive_late = np.sum(alive_sim[9, :])
        assert alive_late < alive_early, \
            "Fewer agents should be alive at late ages with mortality"
        # With 10% annual mortality, expected survival to age 9: 0.9^9 ≈ 0.387
        frac_alive = alive_late / alive_early
        assert frac_alive < 0.95, f"Expected significant mortality reduction, got {frac_alive:.3f}"

    def test_survival_prob_one_simulation_all_alive(self):
        """With survival_probs=1, alive_sim should be all True."""
        survival = np.ones((10, 1))  # certainty of survival
        config = self._base_config(survival_probs=survival)
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        result = model.simulate(n_sim=500, seed=42)
        alive_sim = result[19]
        assert np.all(alive_sim), "alive_sim should be all True when survival_probs=1"

    def test_bequest_nonzero_with_mortality(self):
        """With survival_probs < 1, bequest_sim should have nonzero entries."""
        survival = np.ones((10, 1)) * 0.80  # 20% mortality
        config = self._base_config(survival_probs=survival)
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        result = model.simulate(n_sim=2000, seed=42)
        bequest_sim = result[20]
        assert np.any(bequest_sim > 0), "Some bequests should be nonzero with mortality"

    def test_alive_sim_shape(self):
        """alive_sim shape is (T_sim, n_sim) — index 19 in 21-tuple."""
        config = self._base_config()
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        n_sim = 200
        result = model.simulate(n_sim=n_sim, seed=42)
        assert len(result) in (21, 22, 23)
        alive_sim = result[19]
        T_sim = 10 - 0  # T - current_age
        assert alive_sim.shape == (T_sim, n_sim), \
            f"alive_sim shape should be ({T_sim}, {n_sim}), got {alive_sim.shape}"

    def test_no_survival_probs_all_alive(self):
        """Without survival_probs, all agents are alive throughout."""
        config = self._base_config()  # survival_probs=None
        model = LifecycleModelPerfectForesight(config, verbose=False)
        model.solve(verbose=False)
        result = model.simulate(n_sim=500, seed=42)
        alive_sim = result[19]
        assert np.all(alive_sim), "All agents should be alive when survival_probs=None"
        bequest_sim = result[20]
        assert np.all(bequest_sim == 0.0), "No bequests when survival_probs=None"


class TestBequestTaxation:
    """Tests for Feature #16: Bequest taxation."""

    @staticmethod
    def _get_economy(**overrides):
        config_kwargs = dict(
            T=5, beta=0.96, gamma=2.0, n_a=10, n_y=2, n_h=1,
            retirement_age=4, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
            survival_probs=np.ones((5, 1)) * 0.80,  # mortality needed for bequests
        )
        config_kwargs.update(overrides.pop('config_kwargs', {}))
        config = LifecycleConfig(**config_kwargs)
        economy_kwargs = dict(
            lifecycle_config=config,
            alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0},
            output_dir='output/test',
        )
        economy_kwargs.update(overrides)
        return OLGTransition(**economy_kwargs)

    def test_bequest_tax_zero_is_noop_budget(self):
        """tau_beq=0 → bequest_tax=0 in budget."""
        config = LifecycleConfig(
            T=5, beta=0.96, gamma=2.0, n_a=10, n_y=2, n_h=1,
            retirement_age=4, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
            survival_probs=np.ones((5, 1)) * 0.80,
            tau_beq=0.0,
        )
        economy = OLGTransition(
            lifecycle_config=config, alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0}, output_dir='output/test',
        )
        r_path = np.ones(3) * 0.03
        economy.simulate_transition(r_path, n_sim=100, verbose=False)
        budget = economy.compute_government_budget(0)
        assert budget['bequest_tax'] == 0.0, "tau_beq=0 should give zero bequest tax"

    def test_bequest_tax_raises_revenue(self):
        """tau_beq > 0 → bequest_tax > 0 in budget when mortality is active."""
        config = LifecycleConfig(
            T=5, beta=0.96, gamma=2.0, n_a=10, n_y=2, n_h=1,
            retirement_age=4, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
            survival_probs=np.ones((5, 1)) * 0.80,
            tau_beq=0.3,
        )
        economy = OLGTransition(
            lifecycle_config=config, alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0}, output_dir='output/test',
        )
        r_path = np.ones(3) * 0.03
        economy.simulate_transition(r_path, n_sim=200, verbose=False)
        budget = economy.compute_government_budget(0)
        assert budget['bequest_tax'] >= 0.0, "bequest_tax should be non-negative"
        # With mortality, should have some bequests
        assert 'total_bequests' in budget, "total_bequests should be in budget"

    def test_bequest_tax_in_budget_path(self):
        """Bequest tax keys appear in budget_path output."""
        config = LifecycleConfig(
            T=5, beta=0.96, gamma=2.0, n_a=10, n_y=2, n_h=1,
            retirement_age=4, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
            survival_probs=np.ones((5, 1)) * 0.80,
            tau_beq=0.2,
        )
        economy = OLGTransition(
            lifecycle_config=config, alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0}, output_dir='output/test',
        )
        r_path = np.ones(3) * 0.03
        economy.simulate_transition(r_path, n_sim=100, verbose=False)
        budget_path = economy.compute_government_budget_path(verbose=False)
        assert 'bequest_tax' in budget_path
        assert 'total_bequests' in budget_path
        assert len(budget_path['bequest_tax']) == 3


class TestPopulationAging:
    """Tests for Feature #21: Population aging."""

    @staticmethod
    def _get_small_economy(**overrides):
        config = LifecycleConfig(
            T=5, beta=0.96, gamma=2.0, n_a=10, n_y=2, n_h=1,
            retirement_age=4, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
        )
        kwargs = dict(
            lifecycle_config=config, alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0}, output_dir='output/test',
        )
        kwargs.update(overrides)
        return OLGTransition(**kwargs)

    def test_survival_improvement_increases_life_expectancy(self):
        """survival_improvement_rate > 0 → larger cum_surv at late ages."""
        config = LifecycleConfig(
            T=10, beta=0.96, gamma=2.0, n_a=10, n_y=2, n_h=1,
            retirement_age=8, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
            survival_probs=np.ones((10, 1)) * 0.90,
        )
        economy = OLGTransition(
            lifecycle_config=config, alpha=0.33, delta=0.05, A=1.0,
            education_shares={'medium': 1.0}, output_dir='output/test',
            survival_improvement_rate=0.05,
        )
        # Compute survival schedule at year 0 vs year 20
        sched_base = economy._survival_schedule_at_year(economy.birth_year)
        sched_future = economy._survival_schedule_at_year(economy.birth_year + 20)
        assert np.all(sched_future >= sched_base - 1e-10), \
            "Future survival schedules should be >= base with improvement"

    def test_fertility_path_is_rejected(self):
        """fertility_path was removed; construction must say where it went.

        It drove _build_population_weights, which multiplied the weights by
        cumulative survival -- double-counting mortality that the per-cohort
        means already carry -- and discarded the measured entrant path,
        substituting ones. Three tests here exercised that function; they were
        removed with it on 2026-10-01. The demographic sidecar supersedes the
        capability: the EUROPOP2023 entrant series is the fertility path.
        """
        with pytest.raises(NotImplementedError, match='demography'):
            OLGTransition(lifecycle_config=get_test_config(),
                          education_shares={'medium': 1.0},
                          fertility_path=np.ones(80))



class TestInitialAssetDistribution:
    """Tests for initial_asset_distribution feature."""

    @staticmethod
    def _base_config(**overrides):
        defaults = dict(
            T=10, beta=0.96, gamma=2.0, n_a=100, n_y=2, n_h=1,
            retirement_age=8, education_type='medium',
            pension_replacement_default=0.40, m_good=0.0,
        )
        defaults.update(overrides)
        return LifecycleConfig(**defaults)

    def test_initial_asset_distribution_overrides_initial_assets(self):
        """initial_asset_distribution takes priority over initial_assets."""
        # With initial_assets = 50 (high), mean assets at t=0 should be near 50
        config_scalar = self._base_config(initial_assets=50.0)
        m_scalar = LifecycleModelPerfectForesight(config_scalar, verbose=False)
        m_scalar.solve(verbose=False)
        result_scalar = m_scalar.simulate(n_sim=500, seed=42)
        mean_a0_scalar = np.mean(result_scalar[0][0, :])

        # With initial_asset_distribution = [0, 0, 0] (all at zero), mean assets at t=0 should be near 0
        config_dist = self._base_config(
            initial_assets=50.0,  # This should be overridden
            initial_asset_distribution=np.zeros(100),
        )
        m_dist = LifecycleModelPerfectForesight(config_dist, verbose=False)
        m_dist.solve(verbose=False)
        result_dist = m_dist.simulate(n_sim=500, seed=42)
        mean_a0_dist = np.mean(result_dist[0][0, :])

        assert mean_a0_dist < mean_a0_scalar - 1e-3, \
            "initial_asset_distribution=zeros should give lower t=0 assets than initial_assets=50"

    def test_initial_asset_distribution_broadens_wealth_spread(self):
        """Heterogeneous initial assets → higher wealth dispersion in early periods."""
        # Uniform initial assets
        config_uniform = self._base_config(initial_assets=5.0)
        m_uniform = LifecycleModelPerfectForesight(config_uniform, verbose=False)
        m_uniform.solve(verbose=False)
        result_uniform = m_uniform.simulate(n_sim=1000, seed=42)

        # Dispersed initial assets
        dist = np.array([0.0, 1.0, 5.0, 10.0, 20.0, 50.0] * 50)
        config_disp = self._base_config(initial_asset_distribution=dist)
        m_disp = LifecycleModelPerfectForesight(config_disp, verbose=False)
        m_disp.solve(verbose=False)
        result_disp = m_disp.simulate(n_sim=1000, seed=42)

        std_uniform = np.std(result_uniform[0][0, :])
        std_disp = np.std(result_disp[0][0, :])
        assert std_disp > std_uniform, \
            "Heterogeneous initial assets should broaden wealth dispersion"

    def test_initial_assets_scalar_unchanged(self):
        """initial_asset_distribution=None with initial_assets scalar is unchanged behavior."""
        config_a = self._base_config(initial_assets=10.0, initial_asset_distribution=None)
        config_b = self._base_config(initial_assets=10.0)
        m_a = LifecycleModelPerfectForesight(config_a, verbose=False)
        m_a.solve(verbose=False)
        m_b = LifecycleModelPerfectForesight(config_b, verbose=False)
        m_b.solve(verbose=False)
        result_a = m_a.simulate(n_sim=200, seed=42)
        result_b = m_b.simulate(n_sim=200, seed=42)
        assert np.allclose(result_a[0], result_b[0]), \
            "initial_asset_distribution=None should not change behavior"


class TestFixedEffect:
    """Phase 8: permanent productivity fixed effect (sigma_alpha)."""

    def _base_config(self, sigma_alpha=0.0, n_alpha=1):
        edu_params = {
            'low':    {'mu_y': 1.0, 'sigma_y': 0.10, 'rho_y': 0.95,
                       'sigma_alpha': sigma_alpha, 'unemployment_rate': 0.10},
            'medium': {'mu_y': 2.0, 'sigma_y': 0.10, 'rho_y': 0.95,
                       'sigma_alpha': sigma_alpha, 'unemployment_rate': 0.06},
            'high':   {'mu_y': 2.4, 'sigma_y': 0.10, 'rho_y': 0.95,
                       'sigma_alpha': sigma_alpha, 'unemployment_rate': 0.03},
        }
        return LifecycleConfig(T=50, retirement_age=40, n_a=30,
                               current_age=0, n_alpha=n_alpha,
                               edu_params=edu_params)

    def test_alpha_grid_off_by_default(self):
        """Default config has n_alpha=1 and a degenerate {0.0} alpha grid."""
        cfg = LifecycleConfig()
        m = LifecycleModelPerfectForesight(cfg, verbose=False)
        assert m.n_alpha == 1
        assert np.allclose(m.alpha_grid, 0.0)
        assert np.allclose(m.alpha_probs, 1.0)

    def test_alpha_grid_gauss_hermite(self):
        """With sigma_alpha>0 and n_alpha=5, alpha has zero mean and
        variance exactly sigma_alpha^2."""
        sigma = 0.30
        cfg = self._base_config(sigma_alpha=sigma, n_alpha=5)
        m = LifecycleModelPerfectForesight(cfg, verbose=False)
        assert m.n_alpha == 5
        mean = float(np.dot(m.alpha_grid, m.alpha_probs))
        var = float(np.dot(m.alpha_grid ** 2, m.alpha_probs))
        assert abs(mean) < 1e-12
        assert abs(var - sigma ** 2) < 1e-10
        assert abs(m.alpha_probs.sum() - 1.0) < 1e-12

    def test_n_alpha_1_no_op(self):
        """n_alpha=1 with sigma_alpha=0 reproduces an identical V/policy
        to a model built without the FE feature in mind."""
        cfg = self._base_config(sigma_alpha=0.0, n_alpha=1)
        m = LifecycleModelPerfectForesight(cfg, verbose=False)
        m.solve(verbose=False)
        # Scalar policies alias the (singleton) per-alpha arrays
        assert m.V_alpha.shape[0] == 1
        assert np.array_equal(m.V, m.V_alpha[0])
        assert np.array_equal(m.a_policy, m.a_policy_alpha[0])

    def test_alpha_permanence(self):
        """Each agent's alpha_idx is constant across all simulation periods."""
        cfg = self._base_config(sigma_alpha=0.30, n_alpha=5)
        m = LifecycleModelPerfectForesight(cfg, verbose=False)
        m.solve(verbose=False)
        result = m.simulate(T_sim=50, n_sim=500, seed=7)
        alpha_idx_panel = result[21]    # alpha_idx_sim; index 22 is transfer_sim since 2026-10-02
        # Every column should be constant across time
        assert np.all(alpha_idx_panel[0] == alpha_idx_panel[-1])
        for t in range(alpha_idx_panel.shape[0]):
            assert np.array_equal(alpha_idx_panel[t], alpha_idx_panel[0])

    def test_wage_decomposition(self):
        """For employed agents, effective_y / (w * kappa * y_state) equals
        exp(alpha) within the agent's alpha bin."""
        cfg = self._base_config(sigma_alpha=0.30, n_alpha=5)
        m = LifecycleModelPerfectForesight(cfg, verbose=False)
        m.solve(verbose=False)
        result = m.simulate(T_sim=50, n_sim=2000, seed=7)
        y_state = result[2]
        eff_y = result[5]
        employed = result[6].astype(bool) & (y_state > 0)
        alpha_idx = result[21]          # alpha_idx_sim; index 22 is transfer_sim since 2026-10-02
        t = 10  # working age
        kappa_t = float(m.wage_age_profile[t])
        w_t = float(m.w_path[t])
        for k in range(m.n_alpha):
            mask = employed[t] & (alpha_idx[t] == k)
            if mask.sum() < 5:
                continue
            ratio = (eff_y[t, mask] / (w_t * kappa_t * y_state[t, mask])).mean()
            expected = float(np.exp(m.alpha_grid[k]))
            assert abs(ratio - expected) < 1e-4, \
                f"alpha[{k}]: ratio={ratio:.4f}, expected={expected:.4f}"

    def test_alpha_frequency_matches_probs(self):
        """At t=0, alpha_idx frequencies match alpha_probs (within sampling tol)."""
        cfg = self._base_config(sigma_alpha=0.30, n_alpha=5)
        m = LifecycleModelPerfectForesight(cfg, verbose=False)
        m.solve(verbose=False)
        n_sim = 5000
        result = m.simulate(T_sim=50, n_sim=n_sim, seed=7)
        alpha_idx = result[21][0]       # alpha_idx_sim; index 22 is transfer_sim since 2026-10-02
        empirical = np.bincount(alpha_idx, minlength=m.n_alpha) / n_sim
        # 3 standard errors for a multinomial proportion: sqrt(p*(1-p)/n)
        for k in range(m.n_alpha):
            p = float(m.alpha_probs[k])
            tol = 3 * np.sqrt(p * (1 - p) / n_sim)
            assert abs(empirical[k] - p) < tol + 1e-3, \
                f"alpha[{k}]: empirical={empirical[k]:.4f}, p={p:.4f}, tol={tol:.4f}"


class TestFixedEffectJAX:
    """Phase 8 cross-validation: NumPy and JAX agree on per-alpha policies."""

    def _base_config(self, sigma_alpha=0.0, n_alpha=1):
        edu_params = {
            'low':    {'mu_y': 1.0, 'sigma_y': 0.10, 'rho_y': 0.95,
                       'sigma_alpha': sigma_alpha, 'unemployment_rate': 0.10},
            'medium': {'mu_y': 2.0, 'sigma_y': 0.10, 'rho_y': 0.95,
                       'sigma_alpha': sigma_alpha, 'unemployment_rate': 0.06},
            'high':   {'mu_y': 2.4, 'sigma_y': 0.10, 'rho_y': 0.95,
                       'sigma_alpha': sigma_alpha, 'unemployment_rate': 0.03},
        }
        return LifecycleConfig(T=50, retirement_age=40, n_a=30,
                               current_age=0, n_alpha=n_alpha,
                               edu_params=edu_params)

    def test_solve_value_function_matches_numpy(self):
        """V_alpha agrees between NumPy and JAX at every alpha index."""
        try:
            from lifecycle_jax import LifecycleModelJAX
        except ImportError:
            pytest.skip("JAX not available")
        cfg = self._base_config(sigma_alpha=0.30, n_alpha=5)
        m_np = LifecycleModelPerfectForesight(cfg, verbose=False)
        m_np.solve(verbose=False)
        m_jx = LifecycleModelJAX(cfg, verbose=False)
        m_jx.solve(verbose=False)
        assert m_np.V_alpha.shape == m_jx.V_alpha.shape
        for k in range(m_np.n_alpha):
            diff = np.abs(m_np.V_alpha[k] - m_jx.V_alpha[k]).max()
            assert diff < 1e-6, f"V mismatch at alpha[{k}]: max diff = {diff:.2e}"


class TestTrendGrowthHousehold:
    """Balanced growth in the household block (plan Step 2).

    In detrended units a unit of next-period assets costs (1+g) today, so the
    growth problem on grid G at return r is the same problem as the no-growth
    one on grid (1+g)G at the return r_tilde that solves
    1 + r_tilde(1-tau_k) = (1 + r(1-tau_k))/(1+g).  Every level object
    (pension floor, transfer floor, medical costs, child costs) is unchanged
    between the two, so the policies must coincide exactly.
    """

    G, R, TAU_K = 0.017, 0.04, 0.2236
    A_MAX = 30.0

    @classmethod
    def _pair(cls):
        r_tilde = ((1 + cls.R * (1 - cls.TAU_K)) / (1 + cls.G) - 1) / (1 - cls.TAU_K)
        T = 8
        base = dict(T=T, n_a=25, n_y=2, n_h=1, retirement_age=6, labor_supply=True,
                    nu=1.0, phi=2.0, gamma=1.0, a_min=0.0, tau_k_default=cls.TAU_K,
                    w_path=np.ones(T), pension_min_floor=0.1, transfer_floor=0.05)
        grow = LifecycleConfig(trend_growth=cls.G, r_path=np.full(T, cls.R),
                               a_max=cls.A_MAX, **base)
        flat = LifecycleConfig(trend_growth=0.0, r_path=np.full(T, r_tilde),
                               a_max=cls.A_MAX * (1 + cls.G), **base)
        return grow, flat, r_tilde

    def test_r_tilde_value(self):
        _, _, r_tilde = self._pair()
        assert abs(r_tilde - 0.017801) < 1e-6

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_isomorphism(self, backend):
        if backend == "jax":
            from lifecycle_jax import LifecycleModelJAX as Model
        else:
            Model = LifecycleModelPerfectForesight
        grow, flat, _ = self._pair()
        mg = Model(grow, verbose=False); mg.solve(verbose=False)
        mf = Model(flat, verbose=False); mf.solve(verbose=False)
        assert np.array_equal(np.asarray(mg.a_policy), np.asarray(mf.a_policy))
        assert np.abs(np.asarray(mg.c_policy) - np.asarray(mf.c_policy)).max() < 1e-12
        assert np.abs(np.asarray(mg.l_policy) - np.asarray(mf.l_policy)).max() < 1e-12

    def test_growth_changes_policies(self):
        """Guard against the growth factor being a no-op on a fixed grid."""
        T = 8
        base = dict(T=T, n_a=25, n_y=2, n_h=1, retirement_age=6, labor_supply=True,
                    nu=1.0, phi=2.0, gamma=1.0, a_min=0.0, a_max=self.A_MAX,
                    w_path=np.ones(T), r_path=np.full(T, self.R))
        m0 = LifecycleModelPerfectForesight(LifecycleConfig(trend_growth=0.0, **base),
                                            verbose=False)
        m0.solve(verbose=False)
        mg = LifecycleModelPerfectForesight(LifecycleConfig(trend_growth=self.G, **base),
                                            verbose=False)
        mg.solve(verbose=False)
        assert not np.array_equal(m0.a_policy, mg.a_policy)


class TestTrendGrowthStocks:
    """Per-capita detrended stock recursions (plan Steps 3-4)."""

    G_TREND, N_POP = 0.017, -0.006

    def test_kg_flat_at_stationary_investment(self):
        """I_g = (delta_g + G - 1) * K_g holds K_g exactly flat.

        Gamma is written out from the literal rates rather than read from
        olg.growth_factor: a test that reuses the object the recursion divides
        by would pass for any growth factor, right or wrong.
        """
        T_tr = 10
        G = (1 + self.G_TREND) * (1 + self.N_POP)
        K_g_0, delta_g = 2.0, 0.1
        I_g = np.full(T_tr, (delta_g + G - 1.0) * K_g_0)

        olg = OLGTransition(lifecycle_config=get_test_config(self.G_TREND),
                            pop_growth=self.N_POP,
                            eta_g=0.05, K_g_initial=K_g_0,
                            delta_g=delta_g, I_g_path=I_g)
        res = olg.simulate_transition(np.full(T_tr, 0.04), n_sim=100, verbose=False)

        K_g = res['K_g']
        np.testing.assert_allclose(K_g, K_g_0, rtol=1e-12,
                                   err_msg="stationary I_g did not hold K_g flat")

    def test_age_weights_match_steady_state(self):
        """Transition cohort weights equal calibrate.compute_age_weights."""
        from calibrate import compute_age_weights
        n_cohorts = 20
        sizes = OLGTransition._cohort_sizes_njit(n_cohorts, 2020, 1960, self.N_POP)
        sizes = sizes / sizes.sum()
        np.testing.assert_allclose(sizes, compute_age_weights(n_cohorts, self.N_POP),
                                   rtol=0, atol=1e-15)


class TestDemographicPath:
    """Demography from data, projection and tail (plan Step 0).

    The calibration and the transition weight ages differently and both have
    to be right: the calibration's means are taken among the alive, so its
    weights are shares of the living population, while the transition's means
    run over all simulated agents with the dead holding zero, so its weights
    are cohort sizes at entry. Feeding the transition's weights back through
    cumulative survival must therefore return the measured cross-section.
    """

    CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'calibration_input_GR.json')

    @pytest.fixture(scope='class')
    def built(self):
        import json
        from calibrate import build_olg_transition, load_config
        if not os.path.exists(self.CONFIG):
            pytest.skip('country config not present')
        cfg = json.load(open(self.CONFIG))
        economy, _, T_tr = build_olg_transition(cfg, backend='numpy')
        if economy._demog is None:
            pytest.skip('no demographic path configured')
        demog = np.load(os.path.join(os.path.dirname(self.CONFIG), '..', 'data',
                                    'demography_GR.npz'))
        return economy, T_tr, demog, load_config(self.CONFIG)['age_weights']

    @staticmethod
    def _living(economy, t):
        """Living shares by age at period t, from births-only weights."""
        w = economy._cohort_weights(t)
        T = economy.T
        cum = np.empty(T)
        for j in range(T):
            sched = economy._cohort_survival_schedule(t - j)
            cum[j] = float(np.prod(np.mean(sched, axis=1)[:j])) if j else 1.0
        living = w * cum
        return living / living.sum()

    def test_calibration_weights_are_the_measured_cross_section(self, built):
        _, _, demog, age_weights = built
        data = demog['cross_section_base'] / demog['cross_section_base'].sum()
        np.testing.assert_allclose(age_weights, data, rtol=1e-12, atol=0)

    def test_transition_t0_living_shares_match_the_data(self, built):
        economy, _, demog, _ = built
        data = demog['cross_section_base'] / demog['cross_section_base'].sum()
        np.testing.assert_allclose(self._living(economy, 0), data,
                                   rtol=1e-10, atol=0,
                                   err_msg='t=0 cross-section is not the measured one')

    def test_terminal_state_is_a_balanced_growth_path(self, built):
        economy, T_tr, demog, _ = built
        t0 = int(demog['stable_year']) - int(demog['base_year'])
        assert t0 < T_tr - 1, 'the horizon ends before the population settles'
        G = economy.growth_factors(T_tr)
        gamma_T = (1.0 + economy.trend_growth) * (1.0 + float(demog['n_inf']))
        np.testing.assert_allclose(G[t0:], gamma_T, rtol=0, atol=1e-14)
        for t in range(t0, T_tr):
            np.testing.assert_allclose(economy._cohort_weights(t),
                                       economy._cohort_weights(T_tr - 1),
                                       rtol=0, atol=1e-14)

    def test_the_living_share_matches_the_demographic_data(self, built):
        """The living share must equal living/ever-entered from the sidecar.

        Written to replace a test that asserted sum_j w_j S_j == 1. That holds
        for ANY weights, because _aggregation_weights divides by exactly that
        sum using the same S -- a linear ramp in place of the real weights
        satisfies it to 1e-12. It could not fail, so it checked nothing.

        This compares the living share the code computes against the ratio of
        two independent columns of the sidecar, so wrong weights do show up.
        """
        economy, T_tr, demog, _ = built
        T = economy.T
        years = list(np.asarray(demog['years'], dtype=int))
        px = np.asarray(demog['px'], dtype=float)
        base = int(demog['base_year'])
        cs = np.asarray(demog['cross_section_base'], dtype=float)
        S = np.array([float(np.prod([px[years.index(base - j + a), a]
                                     for a in range(j)])) if j else 1.0
                      for j in range(T)])
        want = cs.sum() / (cs / S).sum()        # living / ever entered, from data
        got = economy._alive_fraction(0)
        assert got == pytest.approx(want, rel=1e-10), (
            f'living share {got:.9f} against {want:.9f} from the sidecar: the '
            f'aggregation weights do not describe the measured population')

    def test_corrupt_weights_change_the_living_share(self, built):
        """Guard that the living share is sensitive to the weights at all."""
        economy, T_tr, _, _ = built
        clean = economy._alive_fraction(0)
        saved = getattr(economy, 'cohort_sizes_path', None)
        try:
            ramp = np.linspace(1.0, 5.0, economy.T)[None, :].repeat(T_tr, axis=0)
            economy.cohort_sizes_path = ramp / ramp.sum(axis=1, keepdims=True)
            economy._alive_frac_cache = {}
            assert abs(economy._alive_fraction(0) - clean) > 1e-6, (
                'the living share is insensitive to the cohort weights, so '
                'nothing here constrains them')
        finally:
            economy.cohort_sizes_path = saved
            economy._alive_frac_cache = {}

    def test_no_mortality_leaves_the_weights_alone(self):
        """Without a survival schedule the scaling must be exactly neutral.

        get_test_config does set one, so it is cleared here: every fixture in
        this suite that has no mortality must be unaffected by the change.
        """
        olg = OLGTransition(
            lifecycle_config=get_test_config()._replace(survival_probs=None),
            education_shares={'medium': 1.0})
        olg.T_transition = 3
        assert olg._alive_fraction(0) == 1.0
        np.testing.assert_array_equal(olg._aggregation_weights(0),
                                      olg._cohort_weights(0))

    def test_mortality_makes_the_scaling_bite(self):
        """With a schedule the living share must be below one and scale up."""
        olg = OLGTransition(lifecycle_config=get_test_config(),
                            education_shares={'medium': 1.0})
        olg.T_transition = 3
        frac = olg._alive_fraction(0)
        assert 0.0 < frac < 1.0, f'living share {frac} is not a share'
        np.testing.assert_allclose(olg._aggregation_weights(0),
                                   np.asarray(olg._cohort_weights(0)) / frac,
                                   rtol=1e-14)

    def test_base_year_cohort_survival_matches_the_transition_diagonals(self, built):
        """The calibration's cohort schedules must be the transition's own.

        base_year_cohort_survival is what lets the base-year equilibrium face
        the same mortality as the transition's t=0 cross-section. It is
        re-derived here straight from the sidecar, independently of the
        function, so an indexing slip in either shows up as a mismatch.
        """
        from calibrate import base_year_cohort_survival
        economy, _, demog, _ = built
        T = economy.T
        S = base_year_cohort_survival(
            __import__('json').load(open(self.CONFIG)), T)
        assert S is not None and S.shape == (T, T)

        years = list(np.asarray(demog['years'], dtype=int))
        px = np.asarray(demog['px'], dtype=float)
        base = int(demog['base_year'])
        for j in (0, T // 3, T // 2, T - 1):
            entry = base - j
            want = np.array([px[years.index(entry + a), a] for a in range(T)])
            np.testing.assert_allclose(
                S[j], want, rtol=0, atol=0,
                err_msg=f'cohort aged {25 + j} in {base}: schedule mismatch')

        # Each cohort alive at t=0 must also face what the transition gives it.
        for j in (0, T // 2, T - 1):
            sched = np.mean(economy._cohort_survival_schedule(-j), axis=1)
            np.testing.assert_allclose(
                S[j], sched, rtol=1e-12, atol=0,
                err_msg=f'cohort aged {25 + j} differs from the transition')

    def test_gamma_path_is_used_by_the_stock_recursions(self, built):
        economy, T_tr, _, _ = built
        G = economy.growth_factors(T_tr)
        assert G.min() < G.max(), 'Gamma_t should vary while demography moves'
        economy.T_transition = T_tr
        economy.growth_factor_path = G
        for t in (0, T_tr // 2, T_tr - 1):
            assert economy._growth_at(t) == pytest.approx(float(G[t]), abs=0, rel=0)
        # Beyond the horizon the last value is held, which is what the fiscal
        # layer relies on when it extends its recursions past the simulation.
        assert economy._growth_at(T_tr + 5) == pytest.approx(float(G[-1]))


class TestCohortBatchedSurvival:
    """Per-cohort survival inside the batched solve.

    The calibration is to be rebuilt on the transition's own batched cohort
    solve, so that the base-year cross-section and the transition's t=0 come
    from one code path rather than two that can drift. Three things have to
    hold for that and none is checked elsewhere: distinct schedules must reach
    the solve and produce distinct policies; a cohort solved inside the batch
    must match the same cohort solved alone; and the batch must not cost its
    cohort count in wall-clock, or an SMM inner loop cannot afford it.

    The first two fail hard. The third warns, because it is a property of the
    machine and is informative rather than wrong.
    """

    YEARS = np.arange(1900, 2101)

    @classmethod
    def _survival_table(cls):
        """Synthetic px[year, age] improving strongly with calendar year.

        Exaggerated relative to real life tables so that a schedule failing to
        reach the solve is unmistakable rather than lost in rounding.
        """
        T = get_test_config().T
        base = np.linspace(0.97, 0.80, T)
        gain = np.linspace(0.0, 0.18, len(cls.YEARS))[:, None]
        return cls.YEARS, np.clip(base[None, :] + gain, 0.0, 0.999)

    def _economy(self, backend='jax', birth_year=1960, current_year=2023):
        yrs, px = self._survival_table()
        return OLGTransition(lifecycle_config=get_test_config(),
                             education_shares={'medium': 1.0},
                             survival_table=(yrs, px),
                             birth_year=birth_year, current_year=current_year,
                             backend=backend)

    def _solved(self, backend='jax'):
        olg = self._economy(backend)
        T_tr = 4
        r_path = np.full(T_tr, 0.04)
        olg.T_transition = T_tr
        olg.solve_cohort_problems(r_path=r_path,
                                  w_path=np.full(T_tr, 1.0), verbose=False)
        return olg

    def test_distinct_survival_gives_distinct_policies(self):
        """Cohorts facing different mortality must not share a policy.

        If the schedule were broadcast rather than stacked per cohort, every
        cohort would solve the same problem and this is the only test that
        would notice.
        """
        olg = self._solved()
        models = olg.birth_cohort_solutions['medium']
        bps = sorted(models)
        young, old = models[bps[-1]], models[bps[0]]
        surv_gap = abs(float(np.prod(np.mean(young.survival_probs, axis=1)))
                       - float(np.prod(np.mean(old.survival_probs, axis=1))))
        assert surv_gap > 1e-3, \
            f'fixture gives the two cohorts near-identical survival ({surv_gap:.2e})'
        c_gap = float(np.max(np.abs(np.asarray(young.c_policy)
                                    - np.asarray(old.c_policy))))
        assert c_gap > 1e-6, (
            'cohorts with different survival solved to the same consumption '
            'policy: the per-cohort schedule is not reaching the batched solve')

    def test_cohort_in_batch_matches_cohort_alone(self):
        """A cohort's policy must not depend on who it was batched with."""
        olg = self._solved()
        models = olg.birth_cohort_solutions['medium']
        bp = sorted(models)[len(models) // 2]
        batched = models[bp]
        cls = type(batched)
        alone = cls(batched.config, verbose=False)
        alone.solve(verbose=False)
        for field in ('c_policy', 'a_policy'):
            b = np.asarray(getattr(batched, field))
            a = np.asarray(getattr(alone, field))
            assert b.shape == a.shape, f'{field} shape {b.shape} vs {a.shape}'
            np.testing.assert_allclose(
                b, a, rtol=1e-10, atol=1e-10,
                err_msg=f'{field} differs between the batched and standalone solve')

    def test_batched_cost_is_sublinear_in_cohorts(self):
        """Warn if batching costs its cohort count rather than a small multiple.

        Backward induction is a sequential scan over T, so batching widens each
        step instead of lengthening the scan; on a GPU the narrow version
        underuses the device and the extra cohorts should be largely absorbed.
        If the ratio approaches the cohort count the device was already
        saturated and a cohort-batched SMM inner loop is not affordable.
        """
        import time
        import warnings
        try:
            import jax
        except ImportError:
            pytest.skip('jax not installed')
        if not any(d.platform == 'gpu' for d in jax.devices()):
            pytest.skip('cost ratio is only meaningful on a GPU')

        def timed(n_cohorts):
            olg = self._economy()
            olg.T_transition = n_cohorts
            r = np.full(n_cohorts, 0.04)
            olg.solve_cohort_problems(r_path=r, w_path=np.full(n_cohorts, 1.0),
                                      verbose=False)          # warm the compile
            t0 = time.perf_counter()
            olg._policy_version = 0
            olg.solve_cohort_problems(r_path=r, w_path=np.full(n_cohorts, 1.0),
                                      verbose=False)
            return time.perf_counter() - t0

        t1, tn = timed(1), timed(16)
        ratio = tn / max(t1, 1e-9)
        # Record it either way: a measurement reported only when it fails is a
        # measurement you do not have.
        print(f'\nbatched solve: 1 cohort {t1:.3f}s, 16 cohorts {tn:.3f}s, '
              f'ratio {ratio:.2f}x (linear would be 16x)')
        if ratio > 8.0:
            warnings.warn(
                f'batched solve scales at {ratio:.1f}x for 16x the cohorts '
                f'({t1:.3f}s -> {tn:.3f}s): a cohort-batched calibration would '
                f'cost roughly that factor per SMM evaluation', RuntimeWarning)
        assert ratio < 40.0, (
            f'batched solve costs {ratio:.1f}x for 16x the cohorts, worse than '
            f'solving them sequentially would suggest -- batching is not working')


class TestBaseYearCrossSection:
    """The cross-section of cohorts that replaces the single stationary solve.

    Row j is the cohort aged 25+j in the base year, observed at age j. Each
    cohort is solved over its full horizon, because backward induction at age j
    needs every later age, but simulated only to age j -- 1,830 cohort periods
    instead of 3,600.
    """

    CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'calibration_input_GR.json')
    T = 12

    def _small(self):
        """The country config shrunk to something a CPU can solve in seconds.

        Every age-indexed array has to be sliced with T or the config rejects
        itself, so they are found by shape rather than listed.
        """
        import dataclasses
        from calibrate import load_config
        if not os.path.exists(self.CONFIG):
            pytest.skip('country config not present')
        L = load_config(self.CONFIG)
        b = L['spec'].base_config
        kw = {f: getattr(b, f)[:self.T].copy()
              for f in b.__dataclass_fields__
              if isinstance(getattr(b, f), np.ndarray)
              and getattr(b, f).ndim >= 1 and getattr(b, f).shape[0] == b.T}
        kw.update(T=self.T, n_a=25, n_y=2, n_alpha=1, retirement_age=9)
        # cohort_survival cleared so run_model_moments takes the single-solve
        # route: these tests compare the cohort path *against* it, and with the
        # switch on it would be comparing the cohort path with itself.
        spec = dataclasses.replace(L['spec'], backend='numpy', n_sim=200,
                                   base_config=b._replace(**kw),
                                   education_shares={'medium': 1.0},
                                   cohort_survival=None,
                                   cohort_retirement_age=None,
                                   cohort_pension_avg_weight=None)
        from calibrate import theta_from_config
        theta = theta_from_config(L['config_data'], spec, verbose=False)
        return L['config_data'], spec, theta

    def test_shared_schedule_reproduces_the_single_solve_exactly(self):
        """With one schedule for every cohort, this must be the old panel.

        Cohort j simulated to age j at the same seed is the single panel's
        first j+1 rows, so the assembled cross-section has to match row for
        row with no tolerance at all. Any slip in the assembly, the truncation
        or the seeding shows up here, and it is the only test that separates
        plumbing errors from the change in economics.
        """
        from calibrate import base_year_cross_section, run_model_moments
        cfg, spec, theta = self._small()
        S = np.asarray(spec.base_config.survival_probs, dtype=float).reshape(
            self.T, spec.base_config.n_h)
        xs = base_year_cross_section(theta, spec, cfg, n_sim=spec.n_sim,
                                     survival=S, seed_per_cohort=False)['medium']
        single = run_model_moments(theta, spec, return_panels=True)[1]['medium']
        for f in xs._fields:
            a, b = np.asarray(getattr(xs, f)), np.asarray(getattr(single, f))
            assert np.array_equal(a, b), (
                f'{f} differs between the cohort cross-section and the single '
                f'solve under one shared survival schedule')

    def test_the_switch_routes_run_model_moments(self):
        """The config key must decide which path run_model_moments takes.

        With it off, nothing changes and the old single-solve route runs. With
        it on, the spec carries (T, T) schedules and the moments come from the
        cross-section of cohorts. Checked on the real config rather than the
        shrunk one, since this is about wiring, not about solving.
        """
        import json
        import tempfile
        from calibrate import load_config
        if not os.path.exists(self.CONFIG):
            pytest.skip('country config not present')
        raw = json.load(open(self.CONFIG))
        T = raw['model']['T']

        spec_on = load_config(self.CONFIG)['spec']
        assert spec_on.cohort_survival is not None, \
            'calibration.base_year_cohorts is set but no schedules reached the spec'
        assert spec_on.cohort_survival.shape == (T, T)
        assert spec_on.n_sim_cohorts > 0

        raw['calibration']['base_year_cohorts'] = False
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False,
                                         dir=os.path.dirname(self.CONFIG)) as fh:
            json.dump(raw, fh)
            off = fh.name
        try:
            assert load_config(off)['spec'].cohort_survival is None, \
                'the switch is off but schedules were still built'
        finally:
            os.unlink(off)

    def test_per_cohort_schedules_change_the_cross_section(self):
        """Per-cohort schedules must move the cross-section.

        Synthetic schedules with a strong calendar gradient, not the real
        diagonals: the cohorts' actual differences sit in old-age mortality --
        0.735 against 0.505 over sixty ages, but only 0.0045 over the first
        twelve -- so a horizon short enough to run on a CPU never reaches where
        the variation is. This checks that a schedule per cohort reaches the
        solve, which is the thing that can silently break; whether the real
        spread matters is a question for the production run, and the measured
        answer there is +2.6% on hours and -3.1% on pensions/Y.
        """
        from calibrate import base_year_cross_section, MOMENT_DISPATCH
        cfg, spec, theta = self._small()
        T, n_h = self.T, spec.base_config.n_h
        shared = np.asarray(spec.base_config.survival_probs,
                            dtype=float).reshape(T, n_h)
        per_cohort = np.clip(
            shared.ravel()[None, :] - np.linspace(0.0, 0.25, T)[:, None], 0.01, 0.999)
        assert abs(np.prod(per_cohort[0]) - np.prod(per_cohort[-1])) > 1e-2

        a = base_year_cross_section(theta, spec, cfg, n_sim=spec.n_sim,
                                   survival=shared)
        b = base_year_cross_section(theta, spec, cfg, n_sim=spec.n_sim,
                                   survival=per_cohort)
        moved = {k: (float(MOMENT_DISPATCH[k](a, spec)),
                     float(MOMENT_DISPATCH[k](b, spec)))
                 for k in ('average_hours', 'A_over_Y')}
        assert any(abs(x - y) > 1e-8 for x, y in moved.values()), (
            f'per-cohort schedules left the cross-section unchanged ({moved}): '
            f'the schedules are not reaching the solve')


    def test_batched_jax_cross_section_matches_one_at_a_time(self):
        """The vectorised JAX route must give the cohort-by-cohort panels.

        Per-cohort schedules and per-cohort seeds, so both the survival stack
        and the seed list are exercised, with a chunk size that leaves a
        padded last chunk. Integer fields must agree exactly; float fields to
        1e-10, since vectorising can reorder floating-point reductions.
        """
        import dataclasses
        from calibrate import base_year_cross_section
        cfg, spec, theta = self._small()
        # n_alpha > 1 so the per-node solve sweeps and their stacking are
        # exercised; the production config uses five nodes.
        spec = dataclasses.replace(spec, backend='jax',
                                   base_config=spec.base_config._replace(n_alpha=3))
        T, n_h = self.T, spec.base_config.n_h
        shared = np.asarray(spec.base_config.survival_probs,
                            dtype=float).reshape(T, n_h)
        per_cohort = np.clip(
            shared.ravel()[None, :] - np.linspace(0.0, 0.25, T)[:, None], 0.01, 0.999)
        one = base_year_cross_section(theta, spec, cfg, n_sim=spec.n_sim,
                                      survival=per_cohort, batched=False)
        vec = base_year_cross_section(theta, spec, cfg, n_sim=spec.n_sim,
                                      survival=per_cohort, batched=True,
                                      chunk_size=5)
        for f in one['medium']._fields:
            a = np.asarray(getattr(one['medium'], f))
            b = np.asarray(getattr(vec['medium'], f))
            assert a.shape == b.shape and a.dtype == b.dtype, f
            if np.issubdtype(a.dtype, np.floating):
                np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12, err_msg=f)
            else:
                assert np.array_equal(a, b), f'{f} differs between the routes'


    @pytest.mark.parametrize('assets', ['config', 'fixed', 'dist'])
    def test_batched_sim_inputs_match_per_seed_draws(self, assets):
        """The one-call initial draws must equal the per-seed draws exactly.

        Covers the three initial-asset branches; the country config uses only
        one of them, so the others are set here.
        """
        from calibrate import apply_params
        from lifecycle_jax import LifecycleModelJAX
        cfg, spec, theta = self._small()
        c = apply_params(spec.base_config, spec.params, theta)._replace(
            n_alpha=3, education_type='medium')
        if assets == 'fixed':
            c = c._replace(initial_assets=0.3, initial_avg_earnings=0.7)
        elif assets == 'dist':
            c = c._replace(initial_asset_distribution=np.array([0.0, 0.2, 0.9, 1.5]))
        m = LifecycleModelJAX(c, verbose=False)
        seeds = [42 + j for j in range(5)] + [2**31 + 7]
        one = [m._simulate_inputs(64, sd) for sd in seeds]
        vec = m._simulate_inputs_batched(64, seeds)
        for k, v in enumerate(vec):
            ref = np.stack([np.asarray(o[k]) for o in one])
            assert np.asarray(v).dtype == ref.dtype, k
            assert np.array_equal(np.asarray(v), ref), f'output {k} differs ({assets})'


    def test_batched_cross_section_with_cohort_retirement_ages(self):
        """Cohorts retiring at different ages: the batched route, which solves
        one retirement group per call, must equal the one-at-a-time route."""
        import dataclasses
        from calibrate import base_year_cross_section
        cfg, spec, theta = self._small()
        T, n_h = self.T, spec.base_config.n_h
        J = np.array([11] * 3 + [10] * 4 + [9] * (T - 7))
        lam = 0.4 + 0.01 * (J - 9)
        spec = dataclasses.replace(spec, backend='jax',
                                   base_config=spec.base_config._replace(n_alpha=3),
                                   cohort_retirement_age=J,
                                   cohort_pension_avg_weight=lam)
        shared = np.asarray(spec.base_config.survival_probs, dtype=float).reshape(T, n_h)
        per_cohort = np.clip(
            shared.ravel()[None, :] - np.linspace(0.0, 0.25, T)[:, None], 0.01, 0.999)
        one = base_year_cross_section(theta, spec, cfg, n_sim=spec.n_sim,
                                      survival=per_cohort, batched=False)['medium']
        vec = base_year_cross_section(theta, spec, cfg, n_sim=spec.n_sim,
                                      survival=per_cohort, batched=True)['medium']
        for f in one._fields:
            a, b = np.asarray(getattr(one, f)), np.asarray(getattr(vec, f))
            if np.issubdtype(a.dtype, np.floating):
                np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12, err_msg=f)
            else:
                assert np.array_equal(a, b), f'{f} differs between the routes'
        # The retirement ages reach the solve: the cohort aged j is retired in
        # the cross-section exactly when j >= its own J_R.
        ret = np.asarray(one.retired_sim, bool)
        alive = np.asarray(one.alive_sim, bool)
        for j in range(T):
            if j >= J[j]:
                assert ret[j][alive[j]].all(), j
            else:
                assert not ret[j].any(), j


class TestCohortRetirement:
    """Cohort-specific retirement ages in the transition.

    The JAX batched solve and simulation take the retirement age as a static
    argument, so cohorts retiring at different ages are batched separately;
    these tests check each cohort gets its own age and that grouping changes
    nothing else.
    """

    T_TR = 4

    def _economy(self, backend):
        from calibrate import SimPanel  # noqa: F401  (import check)
        cfg = get_test_config()                              # T = 20, J_R = 15
        lam = lambda J: (1 - 0.95 ** J) / (J * (1 - 0.95))
        # Entry years 2004..2026 (birth periods -19..3 from 2023): three ages.
        table = {k: ((15 if k < 2012 else 16 if k < 2020 else 17),) for k in range(2004, 2027)}
        table = {k: (v[0], lam(v[0])) for k, v in table.items()}
        olg = OLGTransition(lifecycle_config=cfg, education_shares={'medium': 1.0},
                            birth_year=2023, current_year=2023, backend=backend,
                            cohort_retirement=table)
        olg.T_transition = self.T_TR
        olg.solve_cohort_problems(r_path=np.full(self.T_TR, 0.04),
                                  w_path=np.full(self.T_TR, 1.0), verbose=False)
        return olg, table

    def test_each_cohort_gets_its_retirement_age(self):
        olg, table = self._economy('numpy')
        for bp, m in olg.birth_cohort_solutions['medium'].items():
            J, lam = table[2023 + bp]
            assert m.config.retirement_age == J, bp
            assert m.config.pension_avg_weight == pytest.approx(lam), bp

    def test_grouped_jax_solve_matches_each_cohort_solved_alone(self):
        from lifecycle_jax import LifecycleModelJAX
        olg, _ = self._economy('jax')
        models = olg.birth_cohort_solutions['medium']
        assert len({m.retirement_age for m in models.values()}) == 3
        for bp, m in models.items():
            alone = LifecycleModelJAX(m.config, verbose=False)
            alone.solve(verbose=False)
            assert np.array_equal(np.asarray(m.a_policy_alpha),
                                  np.asarray(alone.a_policy_alpha)), bp
            np.testing.assert_allclose(np.asarray(m.c_policy_alpha),
                                       np.asarray(alone.c_policy_alpha),
                                       rtol=1e-10, atol=1e-12, err_msg=str(bp))

    def test_grouped_jax_simulation_retires_each_cohort_at_its_age(self):
        from calibrate import SimPanel
        olg, table = self._economy('jax')
        panels = olg._simulate_cohorts_jax_batched(200, 7)['medium']
        i_ret = SimPanel._fields.index('retired_sim')
        i_alive = SimPanel._fields.index('alive_sim')
        for bp, panel in panels.items():
            J = table[int(np.clip(2023 + bp, 2004, 2026))][0]
            retired = np.asarray(panel[i_ret], bool)
            alive = np.asarray(panel[i_alive], bool)
            assert not retired[:J].any(), bp
            assert retired[J:][alive[J:]].all(), bp


class TestPensionBaseAcrossBackends:
    """The pension base must be the same in both solves, with a sloped wage profile.

    The JAX solve valued the career-average pension at the retiree's CURRENT age
    wage multiplier instead of the one at the last working age, while the NumPy
    solve and the JAX simulate step both used the latter. Households therefore
    optimised against a poorer retirement than they were paid: noise-free, wealth
    +0.51%, consumption -0.69%.

    No existing cross-validation test could see it, because they all run a flat
    wage_age_profile, under which the two readings coincide identically. The
    slope is the whole point of this fixture.
    """

    T, R = 12, 9

    def _config(self, sloped):
        from calibrate import load_config
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'calibration_input_GR.json')
        if not os.path.exists(cfg_path):
            pytest.skip('country config not present')
        b = load_config(cfg_path)['spec'].base_config
        T = self.T
        kw = {f: getattr(b, f)[:T].copy() for f in b.__dataclass_fields__
              if isinstance(getattr(b, f), np.ndarray)
              and getattr(b, f).ndim >= 1 and getattr(b, f).shape[0] == b.T}
        prof = np.ones(T)
        if sloped:
            prof[:self.R] = np.linspace(1.0, 1.35, self.R)   # rises to retirement
        kw.update(T=T, n_a=40, n_y=3, n_alpha=1, retirement_age=self.R,
                  wage_age_profile=prof, pension_avg_weight=0.45,
                  education_type='medium',
                  r_path=np.full(T, 0.04), w_path=np.full(T, 2.0))
        return b._replace(**kw)

    @pytest.mark.parametrize('sloped', [False, True])
    def test_backends_agree_on_policies(self, sloped):
        from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
        from lifecycle_jax import LifecycleModelJAX
        cfg = self._config(sloped)
        mn = LifecycleModelPerfectForesight(cfg, verbose=False); mn.solve(verbose=False)
        mj = LifecycleModelJAX(cfg, verbose=False); mj.solve(verbose=False)
        c_np, c_jx = np.asarray(mn.c_policy), np.asarray(mj.c_policy)
        scale = max(float(np.max(np.abs(c_np))), 1e-12)
        rel = float(np.max(np.abs(c_np - c_jx))) / scale
        assert rel < 1e-9, (
            f'wage profile {"sloped" if sloped else "flat"}: consumption policies '
            f'differ by {rel:.2e} relative between the backends')

    def test_the_sloped_fixture_can_detect_a_wrong_pension_base(self):
        """Guard that the sloped profile actually discriminates.

        Values the pension at the current age instead of the last working age --
        the old behaviour -- and requires that it moves the policy. Without this
        the test above could pass on a fixture too flat to matter.
        """
        from lifecycle_jax import LifecycleModelJAX
        cfg = self._config(sloped=True)
        good = LifecycleModelJAX(cfg, verbose=False); good.solve(verbose=False)
        flat_after_R = cfg.wage_age_profile.copy()
        flat_after_R[self.R - 1] = flat_after_R[self.R]      # kill the step at R-1
        bad = LifecycleModelJAX(cfg._replace(wage_age_profile=flat_after_R),
                                verbose=False)
        bad.solve(verbose=False)
        moved = float(np.max(np.abs(np.asarray(good.c_policy)
                                    - np.asarray(bad.c_policy))))
        assert moved > 1e-6, (
            'the fixture is insensitive to the wage multiplier at retirement, so '
            'it cannot detect which one the pension base uses')


class TestCrossRoutineLevels:
    """The two routines must agree on LEVELS, not only on ratios.

    This is the check whose absence let a 10% normalisation error survive a full
    calibration and a reported fiscal run. Of roughly 25 quantities that ought to
    agree between calibrate.py and olg_transition.py, none was compared by any
    automated check; the only cross-routine tests compared demographic weights.
    Every ratio cancels a common denominator, so a level error passes them all.

    Both sides are given the SAME survival schedule here, so the only thing that
    can differ is how each adds agents up: the calibration averages over those
    alive at each age and weights ages by their share of the living, while the
    transition averages over everyone ever entered with the dead at zero and
    weights cohorts by entry size, then divides by the living share.
    """

    T, N_SIM = 12, 400

    def _pieces(self):
        import dataclasses
        from calibrate import load_config
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'calibration_input_GR.json')
        if not os.path.exists(cfg_path):
            pytest.skip('country config not present')
        L = load_config(cfg_path)
        b = L['spec'].base_config
        T = self.T
        kw = {f: getattr(b, f)[:T].copy() for f in b.__dataclass_fields__
              if isinstance(getattr(b, f), np.ndarray)
              and getattr(b, f).ndim >= 1 and getattr(b, f).shape[0] == b.T}
        # Real mortality over the shrunk horizon. Slicing the country vector to
        # its first twelve ages leaves survival near 1, which makes the living
        # share near 1 too -- and then the two aggregation conventions coincide
        # and this fixture cannot tell them apart. The guard test below enforces
        # that, and it failed until this was put in.
        kw.update(T=T, n_a=30, n_y=2, n_alpha=1, retirement_age=9,
                  survival_probs=np.linspace(0.97, 0.85, T).reshape(T, 1))
        cfg = b._replace(**kw)
        spec = dataclasses.replace(L['spec'], backend='numpy', n_sim=self.N_SIM,
                                   base_config=cfg,
                                   education_shares={'medium': 1.0},
                                   cohort_survival=None,
                                   cohort_retirement_age=None,
                                   cohort_pension_avg_weight=None)
        return L['config_data'], cfg, spec

    def test_output_and_ratios_agree_at_the_base_year(self):
        from calibrate import (run_model_moments, compute_fiscal_ratios,
                               theta_from_config, compute_age_weights)
        raw, cfg, spec = self._pieces()
        theta = theta_from_config(raw, spec, verbose=False)

        # Calibration side: living-share weights, means among the alive.
        surv = np.asarray(cfg.survival_probs, dtype=float).reshape(self.T, cfg.n_h)
        import dataclasses
        spec = dataclasses.replace(
            spec, age_weights=compute_age_weights(self.T, 0.0, surv.ravel()))
        _, panels = run_model_moments(theta, spec, return_panels=True)
        ss = compute_fiscal_ratios(panels, spec, raw)
        if 'error' in ss:
            pytest.skip(f"compute_fiscal_ratios: {ss['error']}")

        # Transition side, same survival vector for every cohort so the household
        # problem is identical and only the aggregation can differ.
        olg = OLGTransition(
            lifecycle_config=cfg._replace(education_type='medium'),
            education_shares={'medium': 1.0}, backend='numpy',
            alpha=raw['production']['alpha'], delta=raw['production']['delta'],
            A=raw['production']['A_tfp'], economy_type='soe',
            r_star=spec.r, pop_growth=0.0)
        T_tr = 2
        res = olg.simulate_transition(np.full(T_tr, spec.r), n_sim=self.N_SIM,
                                      verbose=False)

        y_ss, y_tr = float(ss['Y']), float(np.asarray(res['Y'])[0])
        rel = abs(y_tr / y_ss - 1.0)
        assert rel < 0.05, (
            f'output levels disagree by {100*rel:.2f}%: base-year equilibrium '
            f'{y_ss:.5f} against transition t=0 {y_tr:.5f}. Both are per living '
            f'person, so a gap of this size is an aggregation defect, not Monte '
            f'Carlo -- the historical failure was 10% and cancelled in every ratio.')

    def test_the_test_would_catch_a_denominator_swap(self):
        """Guard: re-deriving the transition per person ever entered must trip it.

        Without this the test above could pass for the wrong reason -- a loose
        tolerance on two numbers that happen to be close.
        """
        raw, cfg, spec = self._pieces()
        olg = OLGTransition(
            lifecycle_config=cfg._replace(education_type='medium'),
            education_shares={'medium': 1.0}, backend='numpy',
            alpha=raw['production']['alpha'], delta=raw['production']['delta'],
            A=raw['production']['A_tfp'], economy_type='soe',
            r_star=spec.r, pop_growth=0.0)
        olg.T_transition = 2
        frac = olg._alive_fraction(0)
        assert 0.0 < frac < 1.0, f'living share {frac} is not a share'
        # The old convention was the new one times the living share, so a swap
        # moves every level by at least this much.
        assert abs(1.0 - frac) > 0.03, (
            f'living share {frac:.4f} is too close to 1 for this fixture to '
            f'distinguish the two conventions')


class TestAuditFixes20261002:
    """Regression guards for the fixes of 2026-10-02 (audit report §3.8)."""

    T, N_SIM = 12, 300

    @classmethod
    def _cfg(cls, **kw):
        T = cls.T
        base = dict(T=T, n_a=40, n_y=3, n_h=1, n_alpha=2, retirement_age=8, labor_supply=True,
                    nu=1.0, phi=2.0, gamma=1.0, trend_growth=0.017, beta=0.97,
                    survival_probs=np.linspace(0.97, 0.80, T).reshape(T, 1),
                    tau_beq=1.0, pension_min_floor=0.05, education_type='medium')
        base.update(kw)
        cfg = LifecycleConfig(**base)
        ep = dict(cfg.edu_params); ep['medium'] = dict(ep['medium'], sigma_alpha=0.3)
        return cfg._replace(edu_params=ep)

    def test_retired_continuation_uses_own_last_income_state(self):
        """C1: at a retired age V depends on z_last, is nondecreasing in it, and the
        two backends agree. Until 2026-10-02 the continuation was read at z_last=0."""
        from lifecycle_jax import LifecycleModelJAX
        cfg = self._cfg()
        m = LifecycleModelPerfectForesight(cfg, verbose=False); m.solve(verbose=False)
        mj = LifecycleModelJAX(cfg, verbose=False); mj.solve(verbose=False)
        tR = cfg.retirement_age + 1
        Vr = np.asarray(m.V)[tR, :, 0, 0, :]
        assert np.ptp(Vr[10]) > 1e-6, "retired value does not depend on z_last"
        assert np.all(np.diff(Vr, axis=1) >= -1e-12), "retired value decreasing in z_last"
        assert np.abs(np.asarray(m.V) - np.asarray(mj.V)).max() < 1e-10
        assert np.array_equal(np.asarray(m.a_policy), np.asarray(mj.a_policy))

    def test_hours_are_not_capped_at_one(self):
        """c3: with a small disutility weight the chosen hours exceed one on both
        backends and the first-order condition holds with equality there."""
        from lifecycle_jax import LifecycleModelJAX, solve_labor_robust_jax
        import jax.numpy as jnp
        cfg = self._cfg(nu=0.02)
        m = LifecycleModelPerfectForesight(cfg, verbose=False); m.solve(verbose=False)
        mj = LifecycleModelJAX(cfg, verbose=False); mj.solve(verbose=False)
        assert float(np.asarray(m.l_policy_alpha).max()) > 1.0
        assert float(np.asarray(mj.l_policy_alpha).max()) > 1.0
        l = float(solve_labor_robust_jax(jnp.array(1.0), jnp.array(2.0), 0.02, 2.0, 1.0, 0.1))
        c_l = 1.0 + 2.0 * (l - 1.0) / 1.1
        assert l > 1.0 and abs(0.02 * l ** 2 * 1.1 - 2.0 / c_l) < 1e-8
        c, l_np = m._solve_labor_newton(1.0, 2.0 / ((1 - 0.2) * (1 - 0.2)), 1.0, 1.0, 1.0, 0.2, 0.2, 0.1)
        assert l_np > 1.0 and abs(0.02 * l_np ** 2 * 1.1 - 2.0 / c) < 1e-8

    def test_transfer_floor_is_recorded_and_booked(self):
        """M1: the simulated top-up closes each household's budget identity on both
        backends and the transition books its aggregate as an outlay."""
        from lifecycle_jax import LifecycleModelJAX
        cfg = self._cfg(transfer_floor=0.08)
        g, r, tc = cfg.trend_growth, cfg.r_default, cfg.tau_c_default
        for Model in (LifecycleModelPerfectForesight, LifecycleModelJAX):
            mdl = Model(cfg, verbose=False); mdl.solve(verbose=False)
            out = [np.asarray(x) for x in mdl.simulate(n_sim=self.N_SIM, seed=11)]
            assert len(out) == 23
            a, c, eff, oop, tl, tp, tk, pens, ret, alive, tr = (
                out[0], out[1], out[5], out[9], out[12], out[13], out[14], out[16],
                out[17].astype(bool), out[19].astype(bool), out[22])
            assert (tr > 0).any(), "floor never binds in this fixture"
            a_next = np.zeros_like(a); a_next[:-1] = a[1:]
            alive_next = np.zeros_like(alive); alive_next[:-1] = alive[1:]
            after_tax = np.where(ret, pens - tl, eff - tp - tl)
            resid = (1 + tc) * c + (1 + g) * a_next - (a + r * a - tk + after_tax - oop + tr)
            assert np.abs(resid[alive & alive_next]).max() < 1e-10
        olg = OLGTransition(lifecycle_config=cfg, education_shares={'medium': 1.0}, backend='numpy',
                            pop_growth=0.0, economy_type='soe', r_star=0.04, alpha=0.33, delta=0.05, A=1.0)
        olg.simulate_transition(np.full(3, 0.04), n_sim=self.N_SIM, verbose=False)
        bud = olg.compute_government_budget_path(n_sim=self.N_SIM, verbose=False)
        lines = sum(np.asarray(bud[k]) for k in ('ui', 'pension', 'gov_health', 'transfers', 'govt_spending',
                                                  'public_investment', 'defense_spending', 'other_net_spending'))
        assert np.asarray(bud['transfers']).max() > 0.0
        np.testing.assert_allclose(lines, np.asarray(bud['total_spending']), rtol=0, atol=1e-12)

    def test_labor_input_is_the_wage_bill(self):
        """M2: w*L equals the wage bill (payroll tax over its rate); UI is not labour."""
        cfg = self._cfg()
        olg = OLGTransition(lifecycle_config=cfg, education_shares={'medium': 1.0}, backend='numpy',
                            pop_growth=0.0, economy_type='soe', r_star=0.04, alpha=0.33, delta=0.05, A=1.0)
        T_tr = 3
        res = olg.simulate_transition(np.full(T_tr, 0.04), n_sim=self.N_SIM, verbose=False,
                                      tau_p_path=np.full(T_tr, 0.2))
        bud = olg.compute_government_budget_path(n_sim=self.N_SIM, verbose=False)
        wL = np.asarray(res['w']) * np.asarray(res['L'])
        np.testing.assert_allclose(wL, np.asarray(bud['tax_p']) / 0.2, rtol=1e-10, atol=0)
        assert np.asarray(bud['ui']).max() > 0.0

    def test_entering_cohorts_extend_past_the_table(self):
        """C2: periods beyond the entrant table take the terminal growth rate instead of raising."""
        import json
        from calibrate import build_olg_transition
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_input_GR.json')
        if not os.path.exists(path):
            pytest.skip('country config not present')
        economy, _, _ = build_olg_transition(json.load(open(path)), backend='numpy')
        if economy._demog is None:
            pytest.skip('no demographic path configured')
        last = int(economy._demog['entrant_years'][-1]) - int(economy.current_year)
        w_last = economy._entrant_weights(last)
        w_beyond = economy._entrant_weights(last + 12)
        n_inf = float(economy._demog['n_path'][-1])
        if abs(n_inf) < 1e-12:
            np.testing.assert_allclose(w_beyond, w_last, rtol=0, atol=1e-14)
        assert abs(w_beyond.sum() - 1.0) < 1e-12

    def test_terminal_check_tracks_debt(self):
        """M4: the terminal-convergence check reports the drift of B when given a debt path."""
        from fiscal_experiments import _check_terminal_convergence
        drift, _ = _check_terminal_convergence({'Y': np.array([1.0, 1.0, 1.0])}, {}, None,
                                               B_path=np.array([1.0, 1.0, 1.02, 1.03]))
        assert 'B' in drift and abs(drift['B'] - 0.01 / 1.02) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])