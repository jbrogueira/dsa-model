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
        OPTIMIZED: Solves only for unique birth cohorts to avoid redundant calculations.
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

        # --- OPTIMIZED SOLVER LOOP ---
        # This dictionary will store solved models for each birth cohort
        birth_cohort_solutions = {}
        
        if verbose: print("\n  Solving for unique birth cohorts...")
        
        # Define the range of birth cohorts we need to solve for
        # Cohorts born before t=0 up to cohorts born at the end of the transition
        min_birth_period = 1 - self.T
        max_birth_period = self.T_transition - 1
        
        for edu_type in self.education_shares.keys():
            birth_cohort_solutions[edu_type] = {}
            
            for birth_period in range(min_birth_period, max_birth_period + 1):
                if verbose and birth_period % 10 == 0:
                    print(f"    Solving for cohort born at t={birth_period}...")

                # Get the price paths for this specific birth cohort
                cohort_r = r_path[max(0, birth_period) : birth_period + self.T]
                cohort_w = w_path[max(0, birth_period) : birth_period + self.T]
                
                # Pad with initial steady-state prices for cohorts born before t=0
                if birth_period < 0:
                    pre_periods = -birth_period
                    cohort_r = np.concatenate([np.ones(pre_periods) * r_path[0], cohort_r])
                    cohort_w = np.concatenate([np.ones(pre_periods) * w_path[0], cohort_w])

                # For simplicity, assume taxes are constant for this optimization
                # A full implementation would handle time-varying taxes similarly
                cohort_tau_c = np.ones(self.T) * (tau_c_path[0] if tau_c_path is not None else 0)
                cohort_tau_l = np.ones(self.T) * (tau_l_path[0] if tau_l_path is not None else 0)
                cohort_tau_p = np.ones(self.T) * (tau_p_path[0] if tau_p_path is not None else 0)
                cohort_tau_k = np.ones(self.T) * (tau_k_path[0] if tau_k_path is not None else 0)
                cohort_pension = np.ones(self.T) * (pension_replacement_path[0] if pension_replacement_path is not None else 0.4)

                # Create and solve the model for this birth cohort
                config = LifecycleConfig(
                    T=self.T, beta=self.beta, gamma=self.gamma, current_age=0, # Always solve from age 0
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
                    
                    # Create a new config with the correct current_age and initial conditions
                    if birth_period < 0:
                        initial_assets = self.ss_asset_profiles[edu_type][age]
                        initial_avg_earnings = self.ss_earnings_profiles[edu_type][age]
                    else:
                        initial_assets = 0.0
                        initial_avg_earnings = 0.0

                    # Create a new config for this specific (t, age) instance
                    # This is cheap as it just copies data
                    instance_config = solved_model.config._replace(
                        current_age=age,
                        initial_assets=initial_assets,
                        initial_avg_earnings=initial_avg_earnings
                    )
                    
                    # Create a new model instance, but copy the solved policy functions
                    # This avoids re-solving the model
                    instance_model = LifecycleModelPerfectForesight(instance_config, verbose=False)
                    instance_model.V = solved_model.V
                    instance_model.c_policy = solved_model.c_policy
                    instance_model.a_policy = solved_model.a_policy
                    
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
                
                results = model.simulate(T_sim=remaining_periods, n_sim=n_sim, 
                                        seed=42 + t * 100 + age)
                
                (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
                 ui_sim, m_sim, oop_m_sim, gov_m_sim,
                 tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                 pension_sim, retired_sim) = results
                
                # CRITICAL FIX: Manually set initial assets for cohorts born before transition
                if birth_period < 0 and hasattr(self, 'ss_asset_profiles'):
                    initial_assets = self.ss_asset_profiles[edu_type][age]
                    
                    # DEBUG OUTPUT for first period, first few ages
                    if t == 0 and age < 5:
                        print(f"\n  DEBUG compute_aggregates: t={t}, age={age}, birth_period={birth_period}")
                        print(f"    initial_assets from ss_profile: {initial_assets:.4f}")
                        print(f"    a_sim[0, :5] BEFORE override: {a_sim[0, :5]}")
                    
                    a_sim[0, :] = initial_assets  # Override simulated initial assets
                    
                    if t == 0 and age < 5:
                        print(f"    a_sim[0, :5] AFTER override: {a_sim[0, :5]}")
                
                # Take assets and labor at time 0 of simulation (which is their current age)
                assets_by_age_edu[edu_idx, age] = np.mean(a_sim[0, :])
                labor_by_age_edu[edu_idx, age] = np.mean(effective_y_sim[0, :])
        
        # DEBUG: Print aggregation for first period
        if t == 0:
            print(f"\n  DEBUG Aggregation at t=0:")
            print(f"    assets_by_age_edu[0, :]: {assets_by_age_edu[0, :]}")
            print(f"    labor_by_age_edu[0, :]: {labor_by_age_edu[0, :]}")
            print(f"    cohort_sizes: {self.cohort_sizes}")
            print(f"    education_shares_array: {education_shares_array}")
        
        K, L = self._aggregate_capital_labor_njit(
            assets_by_age_edu, labor_by_age_edu,
            self.cohort_sizes, education_shares_array
        )
        
        if t == 0:
            print(f"    Aggregated K: {K:.4f}, L: {L:.4f}")
        
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
                                  ages_to_plot=None,  # ← NEW parameter
                                  n_sim=5000, save=True, show=True, filename=None):
        """Compare lifecycle profiles at different points in transition.
        
        Parameters
        ----------
        periods_to_plot : list
            List of transition periods to plot
        edu_type : str
            Education type to plot
        ages_to_plot : list or None
            List of ages at period 0 to plot. If None, plots only newborns (age 0).
            E.g., [0, 5, 8] plots cohorts aged 0, 5, and 8 at the start of transition.
        n_sim : int
            Number of simulations for each cohort
        """
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
                
                results = model.simulate(T_sim=remaining_life, n_sim=n_sim, seed=42)
                
                (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
                 ui_sim, m_sim, oop_m_sim, gov_m_sim,
                 tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                 pension_sim, retired_sim) = results
                
                # Adjust ages to show actual age (20 + lifecycle age)
                ages = np.arange(remaining_life) + 20 + age_at_t
                
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
                labor_income_flow = effective_y_sim * self.w_path[t] + pension_sim
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


# Test and example code
def get_test_config():
    """Return a minimal configuration for fast testing."""
    config = LifecycleConfig(
        T=20,                    # Short lifecycle (20 years)
        beta=0.96,
        gamma=2.0,
        n_a=20,                  # Small asset grid
        n_y=3,                   # Fewer income states
        n_h=2,
        retirement_age=15,       # Retire at period 15 (e.g., age 35)
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
    print(f"  n_sim = 100 simulations")
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
    
    T_transition = 10  # Short transition period
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
        n_sim=100,
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
        ages_to_plot=[0, 10, 16],  # Newborn (20), Mid-career (30), and Retiree (36)
        edu_type='medium',
        n_sim=100,
        save=True,
        show=False,
        filename='test_lifecycle_comparison.png'
    )
    
    print("\nTest plots saved to 'output/test' directory:")
    print("  - test_transition_dynamics.png")
    print("  - test_lifecycle_comparison.png")
    
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