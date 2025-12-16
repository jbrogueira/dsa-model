import pytest
import numpy as np
import sys
import os
from olg_transition import OLGTransition, OLGConfig, get_test_config
from lifecycle_perfect_foresight import LifecycleConfig

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
            T=100,
            beta=0.96,
            gamma=2.0,
            n_a=50,
            n_y=2,
            n_h=1,
            retirement_age=6,
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
        T_transition = 10  # Longer to verify stability
        r_constant = 0.04
        tau_c_constant = 0.05
        tau_l_constant = 0.15
        tau_p_constant = 0.124
        tau_k_constant = 0.20
        pension_constant = 0.40
        
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
            n_sim=100,  # More simulations for accuracy
            verbose=False
        )
        
        # Extract aggregates
        K_path = results['K']
        L_path = results['L']
        Y_path = results['Y']
        w_path = results['w']

        # Skip first 20 periods to allow convergence
        burn_in = min(20, T_transition // 2)
        K_path_stable = K_path[burn_in:]
        L_path_stable = L_path[burn_in:]
        Y_path_stable = Y_path[burn_in:]
        w_path_stable = w_path[burn_in:]
        
        # Test 1: Interest rate is exactly constant
        assert np.allclose(results['r'], r_constant, atol=1e-10), \
            "Interest rate should be exactly constant"
       
        # Test 2: All aggregates should have very low variance
        # Use coefficient of variation (CV = std / mean)
        K_cv = np.std(K_path_stable) / np.mean(K_path_stable)
        L_cv = np.std(L_path_stable) / np.mean(L_path_stable)
        Y_cv = np.std(Y_path_stable) / np.mean(Y_path_stable)
        w_cv = np.std(w_path_stable) / np.mean(w_path_stable)
        
        tolerance = 0.05  # 5% coefficient of variation (generous for simulation noise)
        
        assert K_cv < tolerance, \
            f"Capital should be nearly constant (CV={K_cv:.2%} > {tolerance:.0%})"
        assert L_cv < tolerance, \
            f"Labor should be nearly constant (CV={L_cv:.2%} > {tolerance:.0%})"
        assert Y_cv < tolerance, \
            f"Output should be nearly constant (CV={Y_cv:.2%} > {tolerance:.0%})"
        assert w_cv < tolerance, \
            f"Wage should be nearly constant (CV={w_cv:.2%} > {tolerance:.0%})"
        
        # Test 3: No trend in aggregates (use linear regression slope)
        periods = np.arange(T_transition)
        
        # Fit linear trend: y = a + b*t, we want b ≈ 0
        def get_trend_slope(y):
            # Simple OLS: b = cov(t,y) / var(t)
            t_mean = np.mean(periods)
            y_mean = np.mean(y)
            cov = np.mean((periods - t_mean) * (y - y_mean))
            var_t = np.mean((periods - t_mean)**2)
            return cov / var_t
        
        K_slope = get_trend_slope(K_path)
        L_slope = get_trend_slope(L_path)
        Y_slope = get_trend_slope(Y_path)
        
        # Normalize slopes by mean to get percentage change per period
        K_slope_pct = (K_slope / np.mean(K_path)) * 100
        L_slope_pct = (L_slope / np.mean(L_path)) * 100
        Y_slope_pct = (Y_slope / np.mean(Y_path)) * 100
        
        slope_tolerance = 0.5  # 0.5% per period
        
        assert abs(K_slope_pct) < slope_tolerance, \
            f"Capital should have no trend (slope={K_slope_pct:.2f}% per period)"
        assert abs(L_slope_pct) < slope_tolerance, \
            f"Labor should have no trend (slope={L_slope_pct:.2f}% per period)"
        assert abs(Y_slope_pct) < slope_tolerance, \
            f"Output should have no trend (slope={Y_slope_pct:.2f}% per period)"
        
        # Test 4: Production function holds throughout
        Y_check = economy.A * (K_path ** economy.alpha) * (L_path ** (1 - economy.alpha))
        assert np.allclose(Y_path, Y_check, rtol=1e-5), \
            "Production function should hold exactly"
        
        # Print diagnostics
        print("\n" + "="*60)
        print("STEADY STATE TEST WITH CONSTANT ENVIRONMENT")
        print("="*60)
        print(f"Simulation periods: {T_transition}")
        print(f"Number of agents: 100")
        print(f"\nConstant parameters:")
        print(f"  r = {r_constant:.4f}")
        print(f"  τ_c = {tau_c_constant:.4f}")
        print(f"  τ_l = {tau_l_constant:.4f}")
        print(f"\nAggregate stability (Coefficient of Variation):")
        print(f"  Capital:  {K_cv:.2%} (< {tolerance:.0%} ✓)" if K_cv < tolerance else f"  Capital:  {K_cv:.2%} (> {tolerance:.0%} ✗)")
        print(f"  Labor:    {L_cv:.2%} (< {tolerance:.0%} ✓)" if L_cv < tolerance else f"  Labor:    {L_cv:.2%} (> {tolerance:.0%} ✗)")
        print(f"  Output:   {Y_cv:.2%} (< {tolerance:.0%} ✓)" if Y_cv < tolerance else f"  Output:   {Y_cv:.2%} (> {tolerance:.0%} ✗)")
        print(f"  Wage:     {w_cv:.2%} (< {tolerance:.0%} ✓)" if w_cv < tolerance else f"  Wage:     {w_cv:.2%} (> {tolerance:.0%} ✗)")
        print(f"\nAggregate trends (% change per period):")
        print(f"  Capital:  {K_slope_pct:+.2f}% (|·| < {slope_tolerance}% ✓)" if abs(K_slope_pct) < slope_tolerance else f"  Capital:  {K_slope_pct:+.2f}% (|·| > {slope_tolerance}% ✗)")
        print(f"  Labor:    {L_slope_pct:+.2f}% (|·| < {slope_tolerance}% ✓)" if abs(L_slope_pct) < slope_tolerance else f"  Labor:    {L_slope_pct:+.2f}% (|·| > {slope_tolerance}% ✗)")
        print(f"  Output:   {Y_slope_pct:+.2f}% (|·| < {slope_tolerance}% ✓)" if abs(Y_slope_pct) < slope_tolerance else f"  Output:   {Y_slope_pct:+.2f}% (|·| > {slope_tolerance}% ✗)")
        print(f"\nMean values:")
        print(f"  K = {np.mean(K_path):.4f} ± {np.std(K_path):.4f}")
        print(f"  L = {np.mean(L_path):.4f} ± {np.std(L_path):.4f}")
        print(f"  Y = {np.mean(Y_path):.4f} ± {np.std(Y_path):.4f}")
        print(f"  w = {np.mean(w_path):.4f} ± {np.std(w_path):.4f}")
        print(f"  K/Y = {np.mean(K_path/Y_path):.4f}")
        print("="*60)


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
            a_current = model.config.a_grid[a_idx] if hasattr(model.config, 'a_grid') else a_idx
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
            pension_replacement_path=np.ones(config.T) * 0.40  # Changed from pension_replacement
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
            pension_replacement_path=np.ones(config.T) * 0.40  # Changed from pension_replacement
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
        eff_labor = model.config.age_efficiency_profile[age]
        y_grid = model.config.income_process['y_grid']
        income = w_ss * eff_labor * y_grid[y_idx]
        
        print(f"\n  Budget check:")
        print(f"    Income (w*eff*y): {income:.4f}")
        print(f"    Consumption: {c_policy_val_high:.4f}")
        print(f"    Savings: {a_next_policy_val_high:.4f}")
        print(f"    Total: {c_policy_val_high + a_next_policy_val_high:.4f}")
        
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
        
        # Expected: (T, n_a, n_y, n_h, n_e)
        # where n_e is number of avg earnings states (for pensions)
        
        print(f"\nExpected dimensions:")
        print(f"  T = {config.T}")
        print(f"  n_a = {config.n_a}")
        print(f"  n_y = {config.n_y}")
        print(f"  n_h = {config.n_h}")
        
        # Check if last dimension is for earnings history
        n_e = model.a_policy.shape[-1]
        print(f"  n_e (earnings history states) = {n_e}")
        
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
        
        n_e = model.a_policy.shape[-1]
        print(f"\nNumber of earnings states: n_e = {n_e}")
        
        # Check policy at age 1, for each earnings state
        print(f"\nAge 1 policies (a=0, y=high, h=good) by earnings state:")
        for e_idx in range(n_e):
            a_next = model.a_policy[1, 0, 1, 0, e_idx]
            print(f"  e={e_idx}: a' = {a_next:.4f}")
        
        # Check age 2
        print(f"\nAge 2 policies (a=0, y=high, h=good) by earnings state:")
        for e_idx in range(n_e):
            a_next = model.a_policy[2, 0, 1, 0, e_idx]
            print(f"  e={e_idx}: a' = {a_next:.4f}")
        
        # Check which earnings states have most non-zero policies
        print(f"\nNon-zero policies by earnings state:")
        for e_idx in range(n_e):
            e_policies = model.a_policy[:, :, :, :, e_idx]
            e_nonzero = np.sum(e_policies > 0.01)
            e_total = np.prod(e_policies.shape)
            print(f"  e={e_idx}: {e_nonzero}/{e_total} ({100*e_nonzero/e_total:.1f}%)")
        
        # Check average policy value by earnings state
        print(f"\nAverage savings by earnings state (excluding zeros):")
        for e_idx in range(n_e):
            e_policies = model.a_policy[:, :, :, :, e_idx]
            nonzero_policies = e_policies[e_policies > 0.01]
            if len(nonzero_policies) > 0:
                avg_savings = np.mean(nonzero_policies)
                print(f"  e={e_idx}: {avg_savings:.4f}")
            else:
                print(f"  e={e_idx}: No non-zero policies")


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])