import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from lifecycle_perfect_foresight import LifecycleModelPerfectForesight, LifecycleConfig
import os
from datetime import datetime
from numba import njit
from functools import partial
from dataclasses import dataclass, field, replace
from typing import Optional  # <-- Add this import

# Suppress RuntimeWarning from numpy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class OLGConfig:
    pension_replacement_default: float = 0.40

    # === Initial conditions ===
    initial_assets: Optional[float] = None
    initial_avg_earnings: Optional[float] = None

    def _replace(self, **changes):  # <-- Add this method
        """Return a new instance with specified fields replaced."""
        return replace(self, **changes)

    def __post_init__(self):
        """Post-initialization setup and validation."""
        # Add derived parameters to edu_params
        pass


class OLGTransition:
    """
    Overlapping Generations Economy for Transition Dynamics with Perfect Foresight.
    
    Takes exogenous interest rate path and simulates the economy's response.
    All agents know the entire future path of interest rates and wages.
    Includes retirement, pensions, education heterogeneity, taxes, UI, and health.
    """
    
    def __init__(self,
                 # Lifecycle configuration (defaults from LifecycleConfig)
                 lifecycle_config=None,
                 # Production parameters
                 alpha=0.33,
                 delta=0.05,
                 A=1.0,
                 # Demographic parameters
                 pop_growth=0.01,
                 birth_year=1960,
                 current_year=2020,
                 # Education distribution
                 education_shares=None,
                 # Output settings
                 output_dir='output'):
        
        # Use provided config or create default
        if lifecycle_config is None:
            self.lifecycle_config = LifecycleConfig()
        else:
            self.lifecycle_config = lifecycle_config
        
        # Store parameters from config
        self.T = self.lifecycle_config.T
        self.beta = self.lifecycle_config.beta
        self.gamma = self.lifecycle_config.gamma
        self.n_a = self.lifecycle_config.n_a
        self.n_y = self.lifecycle_config.n_y
        self.n_h = self.lifecycle_config.n_h
        self.retirement_age = self.lifecycle_config.retirement_age
        
        # Production parameters
        self.alpha = alpha
        self.delta = delta
        self.A = A
        
        # Demographics
        self.pop_growth = pop_growth
        self.birth_year = birth_year
        self.current_year = current_year
        self.n_cohorts = self.T
        
        # Education distribution
        if education_shares is None:
            self.education_shares = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
        else:
            self.education_shares = education_shares
        
        # Output directory
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create cohort sizes (demographic structure)
        self.cohort_sizes = self._create_cohort_sizes()
        
        # Transition path storage
        self.T_transition = None
        self.r_path = None
        self.w_path = None
        self.K_path = None
        self.L_path = None
        self.Y_path = None
        self.cohort_models = None
    
    def _create_cohort_sizes(self):
        """Create demographic structure with different cohort sizes."""
        cohort_sizes = self._cohort_sizes_njit(
            self.n_cohorts, self.current_year, self.birth_year, self.pop_growth
        )
        cohort_sizes = cohort_sizes / cohort_sizes.sum()
        return cohort_sizes
    
    @staticmethod
    @njit
    def _cohort_sizes_njit(n_cohorts, current_year, birth_year, pop_growth):
        """JIT-compiled cohort size calculation."""
        cohort_sizes = np.zeros(n_cohorts)
        for i in range(n_cohorts):
            age = i
            birth_yr = current_year - age
            years_since_base = birth_yr - birth_year
            cohort_sizes[i] = np.exp(pop_growth * years_since_base)
        return cohort_sizes
    
    @staticmethod
    @njit
    def _production_function_njit(K, L, alpha, A):
        """JIT-compiled production function."""
        return A * (K ** alpha) * (L ** (1 - alpha))
    
    @staticmethod
    @njit
    def _marginal_products_njit(K, L, alpha, delta, A):
        """JIT-compiled marginal products and factor prices."""
        MPK = alpha * A * (K ** (alpha - 1)) * (L ** (1 - alpha))
        MPL = (1 - alpha) * A * (K ** alpha) * (L ** (-alpha))
        r = MPK - delta
        w = MPL
        return r, w
    
    @staticmethod
    @njit
    def _aggregate_capital_labor_njit(assets_by_age_edu, labor_by_age_edu, 
                                      cohort_sizes, education_shares_array):
        """JIT-compiled aggregation of capital and labor across age and education."""
        n_edu, T = assets_by_age_edu.shape
        K = 0.0
        L = 0.0
        
        for edu in range(n_edu):
            for age in range(T):
                weight = cohort_sizes[age] * education_shares_array[edu]
                K += weight * assets_by_age_edu[edu, age]
                L += weight * labor_by_age_edu[edu, age]
        
        return K, L
    
    @staticmethod
    @njit
    def _compute_output_path_njit(K_path, L_path, alpha, A):
        """JIT-compiled computation of output path."""
        T = len(K_path)
        Y_path = np.zeros(T)
        for t in range(T):
            Y_path[t] = A * (K_path[t] ** alpha) * (L_path[t] ** (1 - alpha))
        return Y_path
    
    @staticmethod
    @njit
    def _compute_wage_path_njit(K_path, L_path, alpha, delta, A):
        """JIT-compiled computation of wage path from aggregates."""
        T = len(K_path)
        w_path = np.zeros(T)
        for t in range(T):
            MPL = (1 - alpha) * A * (K_path[t] ** alpha) * (L_path[t] ** (-alpha))
            w_path[t] = MPL
        return w_path
    
    def production_function(self, K, L):
        """Cobb-Douglas production function."""
        return self._production_function_njit(K, L, self.alpha, self.A)
    
    def factor_prices(self, K, L):
        """Compute factor prices from production function."""
        return self._marginal_products_njit(K, L, self.alpha, self.delta, self.A)
    
    def solve_cohort_problems(self, r_path, w_path, 
                          tau_c_path=None, tau_l_path=None, 
                          tau_p_path=None, tau_k_path=None,
                          pension_replacement_path=None,
                          verbose=False):
        """
        Solve lifecycle problems for all cohorts given full price paths.
        
        Key indexing:
        - r_path, w_path, etc. are indexed by CALENDAR TIME (0 to T_transition + T - 1)
        - Each cohort born at calendar time t faces prices from t to t+T-1
        - Policy functions are indexed by LIFECYCLE AGE (0 to T-1)
        """
        if verbose:
            print("\nSolving cohort lifecycle problems with perfect foresight...")
            print(f"  Education types: {list(self.education_shares.keys())}")
        
        # --- STEADY-STATE PROFILES (for initial conditions) ---
        if verbose: print("  Computing initial steady-state profiles...")
        r_ss = r_path[0]
        w_ss = w_path[0]
        self.ss_asset_profiles = {}
        self.ss_earnings_profiles = {}

        for edu_type in self.education_shares.keys():
            ss_config = LifecycleConfig(
                T=self.T, beta=self.beta, gamma=self.gamma, current_age=0,
                education_type=edu_type, n_a=self.n_a, n_y=self.n_y, n_h=self.n_h,
                retirement_age=self.retirement_age,
                r_path=np.ones(self.T) * r_ss, w_path=np.ones(self.T) * w_ss,
                tau_c_path=np.ones(self.T) * (tau_c_path[0] if tau_c_path is not None else 0),
                tau_l_path=np.ones(self.T) * (tau_l_path[0] if tau_l_path is not None else 0),
                tau_p_path=np.ones(self.T) * (tau_p_path[0] if tau_p_path is not None else 0),
                tau_k_path=np.ones(self.T) * (tau_k_path[0] if tau_k_path is not None else 0),
                pension_replacement_path=np.ones(self.T) * (pension_replacement_path[0] if pension_replacement_path is not None else 0.4)
            )
            ss_model = LifecycleModelPerfectForesight(ss_config, verbose=False)
            ss_model.solve(verbose=False)
            results = ss_model.simulate(T_sim=self.T, n_sim=1000, seed=42)
            self.ss_asset_profiles[edu_type] = np.mean(results[0], axis=1)
            self.ss_earnings_profiles[edu_type] = np.mean(results[15], axis=1)
            if verbose:
                print(f"    {edu_type}: age {self.retirement_age} assets = {self.ss_asset_profiles[edu_type][self.retirement_age]:.4f}, avg_earnings = {self.ss_earnings_profiles[edu_type][self.retirement_age]:.4f}")

        # --- SOLVE FOR UNIQUE BIRTH COHORTS ---
        birth_cohort_solutions = {}
        
        if verbose: print("\n  Solving for unique birth cohorts...")
        
        # Define the range of birth cohorts we need to solve for
        min_birth_period = 1 - self.T  # Oldest cohort alive at t=0
        max_birth_period = self.T_transition - 1  # Last cohort born during transition
        
        for edu_type in self.education_shares.keys():
            birth_cohort_solutions[edu_type] = {}
            
            for birth_period in range(min_birth_period, max_birth_period + 1):
                if verbose and birth_period % 10 == 0:
                    print(f"    Solving for cohort born at t={birth_period}...")

                # --- KEY FIX: Extract the correct calendar time slice ---
                # This cohort is born at calendar time 'birth_period'
                # They will live from birth_period to birth_period + T - 1
                
                if birth_period < 0:
                    # Cohort born before transition starts
                    # They experience steady-state until t=0, then transition prices
                    pre_periods = -birth_period
                    cohort_r = np.concatenate([
                        np.ones(pre_periods) * r_path[0],  # Steady state before t=0
                        r_path[0:self.T - pre_periods]      # Transition prices
                    ])
                    cohort_w = np.concatenate([
                        np.ones(pre_periods) * w_path[0],
                        w_path[0:self.T - pre_periods]
                    ])
                    cohort_tau_c = np.concatenate([
                        np.ones(pre_periods) * (tau_c_path[0] if tau_c_path is not None else 0),
                        (tau_c_path[0:self.T - pre_periods] if tau_c_path is not None else np.zeros(self.T - pre_periods))
                    ])
                    cohort_tau_l = np.concatenate([
                        np.ones(pre_periods) * (tau_l_path[0] if tau_l_path is not None else 0),
                        (tau_l_path[0:self.T - pre_periods] if tau_l_path is not None else np.zeros(self.T - pre_periods))
                    ])
                    cohort_tau_p = np.concatenate([
                        np.ones(pre_periods) * (tau_p_path[0] if tau_p_path is not None else 0),
                        (tau_p_path[0:self.T - pre_periods] if tau_p_path is not None else np.zeros(self.T - pre_periods))
                    ])
                    cohort_tau_k = np.concatenate([
                        np.ones(pre_periods) * (tau_k_path[0] if tau_k_path is not None else 0),
                        (tau_k_path[0:self.T - pre_periods] if tau_k_path is not None else np.zeros(self.T - pre_periods))
                    ])
                    cohort_pension = np.concatenate([
                        np.ones(pre_periods) * (pension_replacement_path[0] if pension_replacement_path is not None else 0.4),
                        (pension_replacement_path[0:self.T - pre_periods] if pension_replacement_path is not None else np.ones(self.T - pre_periods) * 0.4)
                    ])
                else:
                    # Cohort born during or after transition starts
                    # Extract prices from birth_period to birth_period + T - 1
                    end_period = birth_period + self.T
                    
                    cohort_r = r_path[birth_period:end_period]
                    cohort_w = w_path[birth_period:end_period]
                    cohort_tau_c = tau_c_path[birth_period:end_period] if tau_c_path is not None else np.zeros(self.T)
                    cohort_tau_l = tau_l_path[birth_period:end_period] if tau_l_path is not None else np.zeros(self.T)
                    cohort_tau_p = tau_p_path[birth_period:end_period] if tau_p_path is not None else np.zeros(self.T)
                    cohort_tau_k = tau_k_path[birth_period:end_period] if tau_k_path is not None else np.zeros(self.T)
                    cohort_pension = pension_replacement_path[birth_period:end_period] if pension_replacement_path is not None else np.ones(self.T) * 0.4

                # Create and solve the model for this birth cohort
                config = LifecycleConfig(
                    T=self.T, beta=self.beta, gamma=self.gamma, current_age=0,
                    education_type=edu_type, n_a=self.n_a, n_y=self.n_y, n_h=self.n_h,
                    retirement_age=self.retirement_age,
                    r_path=cohort_r, w_path=cohort_w,
                    tau_c_path=cohort_tau_c, tau_l_path=cohort_tau_l,
                    tau_p_path=cohort_tau_p, tau_k_path=cohort_tau_k,
                    pension_replacement_path=cohort_pension
                )
                
                model = LifecycleModelPerfectForesight(config, verbose=False)
                model.solve(verbose=False)
                birth_cohort_solutions[edu_type][birth_period] = model

        # --- ASSIGN SOLVED MODELS TO THE COHORT_MODELS GRID ---
        if verbose: print("\n  Assigning solved models to transition grid...")
        self.cohort_models = {}
        for t in range(self.T_transition):
            self.cohort_models[t] = {}
            for edu_type in self.education_shares.keys():
                self.cohort_models[t][edu_type] = {}
                for age in range(self.T):
                    birth_period = t - age
                    
                    # Get the pre-solved model for this cohort's birth period
                    solved_model = birth_cohort_solutions[edu_type][birth_period]
                    
                    # Set initial conditions
                    if birth_period < 0:
                        # Cohort born before transition: use steady-state initial conditions
                        cohort_age_at_transition = -birth_period  # How old they are when transition starts
                        
                        # Get steady-state asset and earnings profiles
                        initial_assets = self.ss_asset_profiles[edu_type][cohort_age_at_transition]
                        initial_avg_earnings = self.ss_earnings_profiles[edu_type][cohort_age_at_transition]
                        
                        # Update config with initial conditions
                        solved_model.config = solved_model.config._replace(
                            initial_assets=initial_assets,
                            initial_avg_earnings=initial_avg_earnings
                        )

                    # Create instance config with current age
                    instance_config = solved_model.config._replace(
                        current_age=age
                    )
                    
                    # Create a new model instance
                    instance_model = LifecycleModelPerfectForesight(instance_config, verbose=False)
                    
                    # Copy the relevant slice of policy functions
                    # The solved model has policies indexed by lifecycle age (0 to T-1)
                    # This agent is currently at lifecycle age 'age'
                    # So they need policies from age to T-1
                    instance_model.V = solved_model.V[age:, :, :, :, :]
                    instance_model.c_policy = solved_model.c_policy[age:, :, :, :, :]
                    instance_model.a_policy = solved_model.a_policy[age:, :, :, :, :]
                    
                    # Store under the LOOP variable age (not reassigned age!)
                    self.cohort_models[t][edu_type][age] = instance_model

        if verbose:
            print("All cohort problems assigned!")
    
    def compute_aggregates(self, t, n_sim=10000):
        """Compute aggregate capital and labor for period t."""
        n_edu = len(self.education_shares)
        education_types = list(self.education_shares.keys())
        
        assets_by_age_edu = np.zeros((n_edu, self.T))
        labor_by_age_edu = np.zeros((n_edu, self.T))
        education_shares_array = np.array([self.education_shares[edu] for edu in education_types])
        
        for edu_idx, edu_type in enumerate(education_types):
            for age in range(self.T):
                birth_period = t - age
                
                model = self.cohort_models[t][edu_type][age]
                
                # Simulate from current age to end of life
                remaining_periods = self.T - age
                
                # Handle edge case: agents in their last period
                if remaining_periods <= 0:
                    assets_by_age_edu[edu_idx, age] = 0.0
                    labor_by_age_edu[edu_idx, age] = 0.0
                    continue
                
                results = model.simulate(T_sim=remaining_periods, n_sim=n_sim, 
                                        seed=42 + t * 100 + age)
                
                (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
                 ui_sim, m_sim, oop_m_sim, gov_m_sim,
                 tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                 pension_sim, retired_sim) = results
                
                # --- ENHANCED DEBUG: Check what simulate() returned ---
                if t == 0 and edu_idx == 0 and age < 5:
                    print(f"    Age {age}: remaining_periods={remaining_periods}")
                    print(f"             effective_y_sim type={type(effective_y_sim)}, shape={effective_y_sim.shape if hasattr(effective_y_sim, 'shape') else 'N/A'}")
                    print(f"             y_sim shape={y_sim.shape if hasattr(y_sim, 'shape') else 'N/A'}")
                    if effective_y_sim.ndim >= 1:
                        print(f"             effective_y_sim[0] sample: {effective_y_sim[0] if effective_y_sim.ndim == 1 else effective_y_sim[0, :3]}")
                
                # CRITICAL FIX: Manually set initial assets for cohorts born before transition
                if birth_period < 0 and hasattr(self, 'ss_asset_profiles'):
                    initial_assets = self.ss_asset_profiles[edu_type][age]
                    if a_sim.ndim >= 1:
                        a_sim[0, :] = initial_assets  # Override simulated initial assets
                
                # --- FIX: Robust handling of all array dimensions ---
                # Extract mean asset value for this age/education group
                if np.isscalar(a_sim) or a_sim.ndim == 0:
                    assets_by_age_edu[edu_idx, age] = float(a_sim)
                elif a_sim.ndim == 1:
                    assets_by_age_edu[edu_idx, age] = np.mean(a_sim)
                else:  # ndim == 2
                    assets_by_age_edu[edu_idx, age] = np.mean(a_sim[0, :])
                
                # Extract mean labor supply for this age/education group
                if np.isscalar(effective_y_sim) or effective_y_sim.ndim == 0:
                    labor_by_age_edu[edu_idx, age] = float(effective_y_sim)
                elif effective_y_sim.ndim == 1:
                    labor_by_age_edu[edu_idx, age] = np.mean(effective_y_sim)
                else:  # ndim == 2
                    labor_by_age_edu[edu_idx, age] = np.mean(effective_y_sim[0, :])
                
                # --- DEBUG: Print to diagnose zero labor issue ---
                if t == 0 and edu_idx == 0 and age < 5:
                    print(f"    Age {age}: assets={assets_by_age_edu[edu_idx, age]:.4f}, "
                          f"labor={labor_by_age_edu[edu_idx, age]:.4f}, "
                          f"effective_y_sim shape={effective_y_sim.shape}")
        
        K, L = self._aggregate_capital_labor_njit(
            assets_by_age_edu, labor_by_age_edu,
            self.cohort_sizes, education_shares_array
        )
        
        return K, L
    
    def simulate_transition(self, r_path, w_path=None,
                           tau_c_path=None, tau_l_path=None,
                           tau_p_path=None, tau_k_path=None,
                           pension_replacement_path=None,
                           n_sim=10000, verbose=True):
        """
        Simulate transition dynamics with exogenous interest rate path.
        
        With exogenous r, the capital-labor ratio K/L is pinned down by:
            r = α * A * (K/L)^(α-1) - δ
        
        This determines the wage:
            w = (1-α) * A * (K/L)^α
        
        Parameters
        ----------
        r_path : array_like
            Exogenous interest rate path
        w_path : array_like, optional
            If provided, uses this wage path (for testing)
            If None, computes wage from production function given r
        """
        r_path = np.array(r_path)
        self.T_transition = len(r_path)
        
        # Extend r_path for cohorts born before transition
        r_path_full = np.concatenate([r_path, np.ones(self.T) * r_path[-1]])
        
        # Compute wage path from production function
        if w_path is None:
            if verbose:
                print("\nComputing wage path from production function...")
                print(f"  Using r + δ = MPK = α * A * (K/L)^(α-1)")
            
            # From MPK: r + δ = α * A * (K/L)^(α-1)
            # Solve for K/L: K/L = [(r + δ) / (α * A)]^(1/(α-1))
            # Then: w = MPL = (1-α) * A * (K/L)^α
            
            K_over_L = np.power((r_path + self.delta) / (self.alpha * self.A), 
                                1.0 / (self.alpha - 1.0))
            w_path = (1 - self.alpha) * self.A * np.power(K_over_L, self.alpha)
            
            if verbose:
                print(f"  Initial: r={r_path[0]:.4f} → K/L={K_over_L[0]:.4f} → w={w_path[0]:.4f}")
                print(f"  Final:   r={r_path[-1]:.4f} → K/L={K_over_L[-1]:.4f} → w={w_path[-1]:.4f}")
        else:
            w_path = np.array(w_path)
            if verbose:
                print("\nUsing provided wage path")
        
        # Extend w_path for cohorts born before transition
        w_path_full = np.concatenate([w_path, np.ones(self.T) * w_path[-1]])
        
        # Extend tax paths
        if tau_c_path is not None:
            tau_c_path = np.array(tau_c_path)
            tau_c_path_full = np.concatenate([tau_c_path, np.ones(self.T) * tau_c_path[-1]])
        else:
            tau_c_path_full = None
        
        if tau_l_path is not None:
            tau_l_path = np.array(tau_l_path)
            tau_l_path_full = np.concatenate([tau_l_path, np.ones(self.T) * tau_l_path[-1]])
        else:
            tau_l_path_full = None
        
        if tau_p_path is not None:
            tau_p_path = np.array(tau_p_path)
            tau_p_path_full = np.concatenate([tau_p_path, np.ones(self.T) * tau_p_path[-1]])
        else:
            tau_p_path_full = None
        
        if tau_k_path is not None:
            tau_k_path = np.array(tau_k_path)
            tau_k_path_full = np.concatenate([tau_k_path, np.ones(self.T) * tau_k_path[-1]])
        else:
            tau_k_path_full = None
        
        if pension_replacement_path is not None:
            pension_replacement_path = np.array(pension_replacement_path)
            pension_path_full = np.concatenate([pension_replacement_path, 
                                               np.ones(self.T) * pension_replacement_path[-1]])
        else:
            pension_path_full = None
        
        if verbose:
            print("\n" + "=" * 60)
            print("Simulating OLG Transition with Exogenous Interest Rates")
            print("=" * 60)
            print(f"Transition periods: {self.T_transition}")
            print(f"Initial r: {r_path[0]:.4f}, w: {w_path[0]:.4f}")
            print(f"Final r: {r_path[-1]:.4f}, w: {w_path[-1]:.4f}")
            print(f"Retirement age: {self.retirement_age}")
            print(f"Education groups: {list(self.education_shares.keys())}")
        
        # Solve all cohort problems with perfect foresight of r and w
        self.solve_cohort_problems(
            r_path_full, w_path_full,
            tau_c_path=tau_c_path_full,
            tau_l_path=tau_l_path_full,
            tau_p_path=tau_p_path_full,
            tau_k_path=tau_k_path_full,
            pension_replacement_path=pension_path_full,
            verbose=verbose
        )
        
        if verbose:
            print("\nComputing aggregate quantities...")
        
        # Compute aggregates from household decisions
        K_path = np.zeros(self.T_transition)
        L_path = np.zeros(self.T_transition)
        
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            K_path[t], L_path[t] = self.compute_aggregates(t, n_sim=n_sim)
        
        # Compute output
        Y_path = self._compute_output_path_njit(K_path, L_path, self.alpha, self.A)
        
        # Verify consistency: check if implied r from aggregates matches exogenous r
        if verbose:
            print("\nVerifying consistency with production function...")
            r_implied, w_implied = self._marginal_products_njit(
                K_path[0], L_path[0], self.alpha, self.delta, self.A
            )
            print(f"  Period 0:")
            print(f"    Exogenous r: {r_path[0]:.4f}, Implied r: {r_implied:.4f}")
            print(f"    Computed w:  {w_path[0]:.4f}, Implied w: {w_implied:.4f}")
            
            if self.T_transition > 1:
                r_implied_end, w_implied_end = self._marginal_products_njit(
                    K_path[-1], L_path[-1], self.alpha, self.delta, self.A
                )
                print(f"  Period {self.T_transition-1}:")
                print(f"    Exogenous r: {r_path[-1]:.4f}, Implied r: {r_implied_end:.4f}")
                print(f"    Computed w:  {w_path[-1]:.4f}, Implied w: {w_implied_end:.4f}")
        
        # Store results
        self.r_path = r_path
        self.w_path = w_path
        self.K_path = K_path
        self.L_path = L_path
        self.Y_path = Y_path
        
        if verbose:
            print("\n" + "=" * 60)
            print("Transition Simulation Complete")
            print("=" * 60)
            print(f"\nSummary Statistics:")
            print(f"  Average K: {np.mean(K_path):.4f}")
            print(f"  Average L: {np.mean(L_path):.4f}")
            print(f"  Average Y: {np.mean(Y_path):.4f}")
            print(f"  Average w: {np.mean(w_path):.4f}")
            print(f"  Average K/Y: {np.mean(K_path/Y_path):.4f}")
            print(f"  K range: [{np.min(K_path):.4f}, {np.max(K_path):.4f}]")
            print(f"  L range: [{np.min(L_path):.4f}, {np.max(L_path):.4f}]")
        
        return {'r': self.r_path, 'w': self.w_path, 'K': self.K_path, 
                'L': self.L_path, 'Y': self.Y_path}
    
    def plot_transition(self, save=True, show=True, filename=None):
        """Plot transition dynamics."""
        if self.r_path is None:
            raise ValueError("Must simulate transition first")
        
        fig = plt.figure(figsize=(15, 10))
        periods = np.arange(self.T_transition)
        
        plt.subplot(3, 3, 1)
        plt.plot(periods, self.r_path * 100, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Interest Rate (%)')
        plt.title('Interest Rate Path (Exogenous)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        plt.plot(periods, self.w_path, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Wage')
        plt.title('Wage Path')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        plt.plot(periods, self.K_path, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Capital')
        plt.title('Aggregate Capital')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 4)
        plt.plot(periods, self.L_path, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Labor')
        plt.title('Aggregate Labor')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5)
        plt.plot(periods, self.Y_path, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Output')
        plt.title('Aggregate Output')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 6)
        K_Y_ratio = self.K_path / self.Y_path
        plt.plot(periods, K_Y_ratio, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('K/Y Ratio')
        plt.title('Capital-Output Ratio')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 7)
        K_growth = np.diff(self.K_path) / self.K_path[:-1] * 100
        plt.plot(periods[1:], K_growth, linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Period')
        plt.ylabel('Growth Rate (%)')
        plt.title('Capital Growth Rate')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        Y_growth = np.diff(self.Y_path) / self.Y_path[:-1] * 100
        plt.plot(periods[1:], Y_growth, linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Period')
        plt.ylabel('Growth Rate (%)')
        plt.title('Output Growth Rate')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 9)
        factor_ratio = (self.r_path + self.delta) / self.w_path
        plt.plot(periods, factor_ratio, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('(r+δ)/w')
        plt.title('Relative Factor Prices')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('OLG Transition Dynamics (Exogenous r)', fontsize=14, y=0.995)
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"transition_dynamics_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_lifecycle_comparison(self, periods_to_plot, edu_type='medium',
                                  ages_to_plot=None,
                                  n_sim=5000, save=True, show=True, filename=None):
        """Compare lifecycle profiles at different points in transition."""
        if self.cohort_models is None:
            raise ValueError("Must simulate transition first")
        
        # Default to just newborns if not specified
        if ages_to_plot is None:
            ages_to_plot = [0]
        
        fig = plt.figure(figsize=(15, 12))
        
        # Loop over both periods and starting ages
        for t in periods_to_plot:
            if t >= self.T_transition:
                continue
            
            for age_at_t in ages_to_plot:
                if age_at_t >= self.T:
                    continue
                
                # Get the cohort model for this period and age
                model = self.cohort_models[t][edu_type][age_at_t]
                remaining_life = self.T - age_at_t
                
                # Skip if no remaining life
                if remaining_life <= 0:
                    continue
                
                results = model.simulate(T_sim=remaining_life, n_sim=n_sim, seed=42)
                
                (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
                 ui_sim, m_sim, oop_m_sim, gov_m_sim,
                 tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                 pension_sim, retired_sim) = results
                
                # --- FIX: Handle different array shapes ---
                # Ensure all arrays are 2D (T_sim, n_sim)
                def ensure_2d(arr):
                    """Convert scalar or 1D array to 2D array."""
                    if np.isscalar(arr) or arr.ndim == 0:
                        return np.array([[arr]])
                    elif arr.ndim == 1:
                        return arr.reshape(-1, 1)
                    else:
                        return arr
                
                a_sim = ensure_2d(a_sim)
                c_sim = ensure_2d(c_sim)
                effective_y_sim = ensure_2d(effective_y_sim)
                pension_sim = ensure_2d(pension_sim)
                employed_sim = ensure_2d(employed_sim)
                oop_m_sim = ensure_2d(oop_m_sim)
                avg_earnings_sim = ensure_2d(avg_earnings_sim)
                tax_c_sim = ensure_2d(tax_c_sim)
                tax_l_sim = ensure_2d(tax_l_sim)
                tax_p_sim = ensure_2d(tax_p_sim)
                tax_k_sim = ensure_2d(tax_k_sim)
                
                # Adjust ages to show actual age (20 + lifecycle age)
                ages = np.arange(a_sim.shape[0]) + 20 + age_at_t
                
                # Create label showing period, starting age, and interest rate
                label = f't={t}, start_age={20+age_at_t}, r={self.r_path[t]:.3f}'
                
                # Plot 1: Assets
                plt.subplot(3, 3, 1)
                plt.plot(ages, np.mean(a_sim, axis=1), label=label, linewidth=2)
                
                # Plot 2: Consumption
                plt.subplot(3, 3, 2)
                plt.plot(ages, np.mean(c_sim, axis=1), label=label, linewidth=2)
                
                # Plot 3: Labor Income
                plt.subplot(3, 3, 3)
                plt.plot(ages, np.mean(effective_y_sim, axis=1), label=label, linewidth=2)
                
                # Plot 4: Pension Benefits
                plt.subplot(3, 3, 4)
                plt.plot(ages, np.mean(pension_sim, axis=1), label=label, linewidth=2)
                
                # Plot 5: Employment Rate
                plt.subplot(3, 3, 5)
                plt.plot(ages, np.mean(employed_sim, axis=1), label=label, linewidth=2)
                
                # Plot 6: Total Taxes
                plt.subplot(3, 3, 6)
                total_tax = tax_c_sim + tax_l_sim + tax_p_sim + tax_k_sim
                plt.plot(ages, np.mean(total_tax, axis=1), label=label, linewidth=2)
                
                # Plot 7: Savings Rate
                plt.subplot(3, 3, 7)
                capital_income = self.r_path[t] * a_sim
                # FIX: effective_y_sim already includes wage (w*y*h), don't multiply again
                labor_income_flow = effective_y_sim + pension_sim
                total_income = capital_income + labor_income_flow
                savings = total_income - c_sim
                savings_rate = savings / (total_income + 1e-10)
                plt.plot(ages, np.mean(savings_rate, axis=1), label=label, linewidth=2)
                
                # Plot 8: Out-of-Pocket Health Expenditures
                plt.subplot(3, 3, 8)
                plt.plot(ages, np.mean(oop_m_sim, axis=1), label=label, linewidth=2)
                
                # Plot 9: Average Earnings History
                plt.subplot(3, 3, 9)
                plt.plot(ages, np.mean(avg_earnings_sim, axis=1), label=label, linewidth=2)
        
        # Add vertical line at retirement age for all subplots
        retirement_age_actual = 20 + self.retirement_age
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            plt.axvline(x=retirement_age_actual, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.5, label='Retirement' if i == 1 else '')
        
        # Configure each subplot
        plt.subplot(3, 3, 1)
        plt.xlabel('Age')
        plt.ylabel('Assets')
        plt.title('Asset Profiles Over Transition')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        plt.xlabel('Age')
        plt.ylabel('Consumption')
        plt.title('Consumption Profiles')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        plt.xlabel('Age')
        plt.ylabel('Labor Income')
        plt.title('Labor Income Profiles')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 4)
        plt.xlabel('Age')
        plt.ylabel('Pension')
        plt.title('Pension Benefits')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5)
        plt.xlabel('Age')
        plt.ylabel('Employment Rate')
        plt.title('Employment Rate')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 6)
        plt.xlabel('Age')
        plt.ylabel('Total Taxes')
        plt.title('Total Tax Payments')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 7)
        plt.xlabel('Age')
        plt.ylabel('Savings Rate')
        plt.title('Savings Rate Profiles')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        plt.xlabel('Age')
        plt.ylabel('OOP Health Exp.')
        plt.title('Out-of-Pocket Health Expenditures')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 9)
        plt.xlabel('Age')
        plt.ylabel('Avg Earnings')
        plt.title('Average Earnings History')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Lifecycle Profiles Over Transition ({edu_type.capitalize()} Education)', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lifecycle_comparison_{edu_type}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def compute_government_budget(self, t, n_sim=10000):
        """Compute government revenues, expenditures, and deficit for period t."""
        n_edu = len(self.education_shares)
        education_types = list(self.education_shares.keys())
        education_shares_array = np.array([self.education_shares[edu] for edu in education_types])
        
        # Initialize aggregates
        total_tax_c = 0.0
        total_tax_l = 0.0
        total_tax_p = 0.0
        total_tax_k = 0.0
        total_ui = 0.0
        total_pension = 0.0
        total_gov_health = 0.0
        
        for edu_idx, edu_type in enumerate(education_types):
            age_dict = self.cohort_models[t].get(edu_type, {})
            
            for age in range(self.T):
                if age not in age_dict:
                    continue
                
                model = age_dict[age]
                remaining_periods = self.T - age
                
                if remaining_periods <= 0:
                    continue
                
                results = model.simulate(
                    T_sim=remaining_periods,
                    n_sim=n_sim,
                    seed=42 + 1000 * t + 100 * edu_idx + age,
                )
                
                (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim,
                 employed_sim, ui_sim, m_sim, oop_m_sim, gov_m_sim,
                 tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                 pension_sim, retired_sim) = results
                
                # Weight by cohort size and education share
                weight = self.cohort_sizes[age] * education_shares_array[edu_idx]
                
                # Extract period-0 values (current period)
                if tax_c_sim.ndim >= 2:
                    tax_c = np.mean(tax_c_sim[0, :])
                    tax_l = np.mean(tax_l_sim[0, :])
                    tax_p = np.mean(tax_p_sim[0, :])
                    tax_k = np.mean(tax_k_sim[0, :])
                    ui = np.mean(ui_sim[0, :])
                    pension = np.mean(pension_sim[0, :])
                    gov_health = np.mean(gov_m_sim[0, :])
                elif tax_c_sim.ndim == 1:
                    tax_c = np.mean(tax_c_sim)
                    tax_l = np.mean(tax_l_sim)
                    tax_p = np.mean(tax_p_sim)
                    tax_k = np.mean(tax_k_sim)
                    ui = np.mean(ui_sim)
                    pension = np.mean(pension_sim)
                    gov_health = np.mean(gov_m_sim)
                else:
                    tax_c = float(tax_c_sim)
                    tax_l = float(tax_l_sim)
                    tax_p = float(tax_p_sim)
                    tax_k = float(tax_k_sim)
                    ui = float(ui_sim)
                    pension = float(pension_sim)
                    gov_health = float(gov_m_sim)
                
                # Aggregate
                total_tax_c += weight * tax_c
                total_tax_l += weight * tax_l
                total_tax_p += weight * tax_p
                total_tax_k += weight * tax_k
                total_ui += weight * ui
                total_pension += weight * pension
                total_gov_health += weight * gov_health
        
        # Compute totals
        total_revenue = total_tax_c + total_tax_l + total_tax_p + total_tax_k
        total_spending = total_ui + total_pension + total_gov_health
        primary_deficit = total_spending - total_revenue
        
        return {
            'tax_c': total_tax_c,
            'tax_l': total_tax_l,
            'tax_p': total_tax_p,
            'tax_k': total_tax_k,
            'total_revenue': total_revenue,
            'ui': total_ui,
            'pension': total_pension,
            'gov_health': total_gov_health,
            'total_spending': total_spending,
            'primary_deficit': primary_deficit
        }
    
    def compute_government_budget_path(self, n_sim=10000, verbose=True):
        """Compute government budget for all transition periods."""
        if self.cohort_models is None:
            raise ValueError("Must simulate transition first")
        
        if verbose:
            print("\nComputing government budget path...")
        
        # Initialize storage
        budget_path = {
            'tax_c': np.zeros(self.T_transition),
            'tax_l': np.zeros(self.T_transition),
            'tax_p': np.zeros(self.T_transition),
            'tax_k': np.zeros(self.T_transition),
            'total_revenue': np.zeros(self.T_transition),
            'ui': np.zeros(self.T_transition),
            'pension': np.zeros(self.T_transition),
            'gov_health': np.zeros(self.T_transition),
            'total_spending': np.zeros(self.T_transition),
            'primary_deficit': np.zeros(self.T_transition)
        }
        
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            
            budget_t = self.compute_government_budget(t, n_sim=n_sim)
            
            for key in budget_path.keys():
                budget_path[key][t] = budget_t[key]
        
        self.budget_path = budget_path
        
        if verbose:
            print("\nGovernment Budget Summary:")
            print(f"  Average revenue:     {np.mean(budget_path['total_revenue']):.4f}")
            print(f"    Consumption tax:   {np.mean(budget_path['tax_c']):.4f}")
            print(f"    Labor income tax:  {np.mean(budget_path['tax_l']):.4f}")
            print(f"    Payroll tax:       {np.mean(budget_path['tax_p']):.4f}")
            print(f"    Capital income tax:{np.mean(budget_path['tax_k']):.4f}")
            print(f"  Average spending:    {np.mean(budget_path['total_spending']):.4f}")
            print(f"    UI benefits:       {np.mean(budget_path['ui']):.4f}")
            print(f"    Pensions:          {np.mean(budget_path['pension']):.4f}")
            print(f"    Health spending:   {np.mean(budget_path['gov_health']):.4f}")
            print(f"  Average deficit:     {np.mean(budget_path['primary_deficit']):.4f}")
            print(f"  Deficit/GDP:         {np.mean(budget_path['primary_deficit'] / self.Y_path):.2%}")
        
        return budget_path
    
    def plot_government_budget(self, save=True, show=True, filename=None):
        """Plot government budget constraint components."""
        if not hasattr(self, 'budget_path') or self.budget_path is None:
            print("Computing government budget path...")
            self.compute_government_budget_path(n_sim=10000, verbose=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        periods = np.arange(self.T_transition)
        
        # Plot 1: Total Revenue and Spending
        ax = axes[0, 0]
        ax.plot(periods, self.budget_path['total_revenue'], 
                label='Total Revenue', linewidth=2, color='green')
        ax.plot(periods, self.budget_path['total_spending'], 
                label='Total Spending', linewidth=2, color='red')
        ax.fill_between(periods, self.budget_path['total_revenue'], 
                        self.budget_path['total_spending'],
                        where=(self.budget_path['total_spending'] > self.budget_path['total_revenue']),
                        alpha=0.3, color='red', label='Deficit')
        ax.fill_between(periods, self.budget_path['total_revenue'], 
                        self.budget_path['total_spending'],
                        where=(self.budget_path['total_revenue'] > self.budget_path['total_spending']),
                        alpha=0.3, color='green', label='Surplus')
        ax.set_xlabel('Period')
        ax.set_ylabel('Amount')
        ax.set_title('Revenue vs Spending', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Tax Revenue Breakdown
        ax = axes[0, 1]
        ax.plot(periods, self.budget_path['tax_c'], label='Consumption Tax', linewidth=2)
        ax.plot(periods, self.budget_path['tax_l'], label='Labor Income Tax', linewidth=2)
        ax.plot(periods, self.budget_path['tax_p'], label='Payroll Tax', linewidth=2)
        ax.plot(periods, self.budget_path['tax_k'], label='Capital Income Tax', linewidth=2)
        ax.set_xlabel('Period')
        ax.set_ylabel('Tax Revenue')
        ax.set_title('Tax Revenue by Type', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Spending Breakdown
        ax = axes[0, 2]
        ax.plot(periods, self.budget_path['ui'], label='UI Benefits', linewidth=2)
        ax.plot(periods, self.budget_path['pension'], label='Pensions', linewidth=2)
        ax.plot(periods, self.budget_path['gov_health'], label='Health Spending', linewidth=2)
        ax.set_xlabel('Period')
        ax.set_ylabel('Spending')
        ax.set_title('Government Spending by Category', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Primary Deficit
        ax = axes[1, 0]
        ax.plot(periods, self.budget_path['primary_deficit'], linewidth=2, color='purple')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(periods, 0, self.budget_path['primary_deficit'],
                        where=(self.budget_path['primary_deficit'] > 0),
                        alpha=0.3, color='red', label='Deficit')
        ax.fill_between(periods, 0, self.budget_path['primary_deficit'],
                        where=(self.budget_path['primary_deficit'] < 0),
                        alpha=0.3, color='green', label='Surplus')
        ax.set_xlabel('Period')
        ax.set_ylabel('Primary Deficit')
        ax.set_title('Primary Deficit (Spending - Revenue)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Deficit as % of GDP
        ax = axes[1, 1]
        deficit_gdp_ratio = (self.budget_path['primary_deficit'] / self.Y_path) * 100
        ax.plot(periods, deficit_gdp_ratio, linewidth=2, color='darkred')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(periods, 0, deficit_gdp_ratio,
                        where=(deficit_gdp_ratio > 0),
                        alpha=0.3, color='red')
        ax.fill_between(periods, 0, deficit_gdp_ratio,
                        where=(deficit_gdp_ratio < 0),
                        alpha=0.3, color='green')
        ax.set_xlabel('Period')
        ax.set_ylabel('Deficit/GDP (%)')
        ax.set_title('Primary Deficit as % of GDP', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Revenue and Spending as % of GDP
        ax = axes[1, 2]
        revenue_gdp = (self.budget_path['total_revenue'] / self.Y_path) * 100
        spending_gdp = (self.budget_path['total_spending'] / self.Y_path) * 100
        ax.plot(periods, revenue_gdp, label='Revenue/GDP', linewidth=2, color='green')
        ax.plot(periods, spending_gdp, label='Spending/GDP', linewidth=2, color='red')
        ax.set_xlabel('Period')
        ax.set_ylabel('% of GDP')
        ax.set_title('Fiscal Ratios to GDP', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Government Budget Constraint Over Transition', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"government_budget_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Government budget plot saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()


# Test and example code
def get_test_config():
    """Return a 'lightning fast' configuration for quick structural tests."""
    config = LifecycleConfig(
        T=10,
        beta=0.96,
        gamma=2.0,
        n_a=10,
        n_y=2,
        n_h=1,
        retirement_age=8,
        education_type='medium'
    )
    return config


def run_fast_test():
    """Run OLG transition with minimal parameters for fast testing."""
    import sys
    import time
    
    print("=" * 60)
    print("RUNNING FAST TEST MODE")
    print("=" * 60)
    
    config = get_test_config()
    
    print("\nTest configuration:")
    print(f"  T = {config.T} periods (e.g., ages 20-{20 + config.T - 1})")
    print(f"  retirement_age = {config.retirement_age} (e.g., age {20 + config.retirement_age})")
    print(f"  n_a = {config.n_a} asset grid points")
    print(f"  n_sim = 50 simulations")
    print()

    economy = OLGTransition(
        lifecycle_config=config,
        alpha=0.33,
        delta=0.05,
        A=1.0,
        pop_growth=0.01,
        birth_year=1960,
        current_year=2020,
        education_shares={'medium': 1.0},
        output_dir='output/test'
    )
    
    T_transition = 5  # Very short transition period
    r_initial = 0.04
    r_final = 0.02
    
    periods = np.arange(T_transition)
    r_path = r_initial + (r_final - r_initial) * (periods / (T_transition - 1))
    
    # Tax rates and pension
    tau_c_path = np.ones(T_transition) * 0.05
    tau_l_path = np.ones(T_transition) * 0.15
    tau_p_path = np.ones(T_transition) * 0.124
    tau_k_path = np.ones(T_transition) * 0.20
    pension_replacement_path = np.ones(T_transition) * 0.40
    
    print(f"Simulating transition from r={r_initial:.3f} to r={r_final:.3f}")
    print(f"Transition periods: {T_transition}")
    
    start = time.time()
    results = economy.simulate_transition(
        r_path=r_path,
        w_path=None,
        tau_c_path=tau_c_path,
        tau_l_path=tau_l_path,
        tau_p_path=tau_p_path,
        tau_k_path=tau_k_path,
        pension_replacement_path=pension_replacement_path,
        n_sim=50, # Reduced simulations
        verbose=True
    )
    end = time.time()
    
    print(f"\n{'=' * 60}")
    print(f"Test completed in {end - start:.2f} seconds")
    print(f"{'=' * 60}")
    
    print("\nGenerating plots for visual inspection...")
    economy.plot_transition(save=True, show=False, 
                           filename='test_transition_dynamics.png')
    
    economy.plot_lifecycle_comparison(
        periods_to_plot=[0],
        ages_to_plot=[0, 5, 9],  # Adjusted for T=10
        edu_type='medium',
        n_sim=50, # Reduced simulations
        save=True,
        show=False,
        filename='test_lifecycle_comparison.png'
    )
    
    # Add government budget plot
    economy.compute_government_budget_path(n_sim=50, verbose=True)
    economy.plot_government_budget(save=True, show=False,
                                   filename='test_government_budget.png')
    
    print("\nTest plots saved to 'output/test' directory:")
    print("  - test_transition_dynamics.png")
    print("  - test_lifecycle_comparison.png")
    print("  - test_government_budget.png")
    
    return economy, results


def run_full_simulation():
    """Run full OLG transition simulation."""
    import sys
    import time
    
    print("=" * 60)
    print("RUNNING FULL SIMULATION")
    print("=" * 60)
    
    config = LifecycleConfig(
        T=40,
        beta=0.98,
        gamma=2.0,
        n_a=30,
        n_y=3,
        n_h=3,
        retirement_age=30,
        education_type='medium'
    )
    
    economy = OLGTransition(
        lifecycle_config=config,
        alpha=0.33,
        delta=0.05,
        A=1.0,
        pop_growth=0.01,
        birth_year=1960,
        current_year=2020,
        education_shares={'low': 0.3, 'medium': 0.5, 'high': 0.2},
        output_dir='output'
    )
    
    T_transition = 15
    r_initial = 0.04
    r_final = 0.02
    
    periods = np.arange(T_transition)
    r_path = r_initial + (r_final - r_initial) * (1 - np.exp(-periods / 5))
    
    print(f"\nSimulating transition from r={r_initial:.3f} to r={r_final:.3f}")
    
    start = time.time()
    results = economy.simulate_transition(
        r_path=r_path,
        w_path=None,
        n_sim=500,
        verbose=True
    )
    end = time.time()
    
    print(f"\nTotal simulation time: {end - start:.2f} seconds")
    
    economy.plot_transition(save=True, show=False)
    
    for edu_type in ['low', 'medium', 'high']:
        economy.plot_lifecycle_comparison(
            periods_to_plot=[0, 5, 10, 14],
            edu_type=edu_type,
            n_sim=500,
            save=True,
            show=False
        )
    
    # Add government budget plot
    economy.compute_government_budget_path(n_sim=500, verbose=True)
    economy.plot_government_budget(save=True, show=False)
    
    print("\nAll plots saved to 'output' directory")
    
    return economy, results


def main():
    """
    Main entry point. Check for --test flag and run accordingly.
    """
    if '--test' in sys.argv:
        economy, results = run_fast_test()
    else:
        economy, results = run_full_simulation()
    
    return economy, results


if __name__ == "__main__":
    economy, results = main()