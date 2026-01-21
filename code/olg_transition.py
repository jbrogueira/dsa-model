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
import matplotlib.ticker as mticker

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

        # Cache for stochastic simulations (avoid re-simulating same cell in multiple routines)
        self._sim_cache = {}

        # NEW: remember last Monte Carlo size used in simulate_transition()
        self._last_n_sim: Optional[int] = None

    @staticmethod
    def _seed_u32(x: int) -> int:
        """Map any integer (incl. negative/large) into NumPy's allowed seed range."""
        return int(np.uint32(x))

    @staticmethod
    def _sparse_int_ticks(x_min: int, x_max: int, step: int = 5):
        """Integer ticks from x_min..x_max inclusive, spaced by `step`."""
        x_min = int(x_min)
        x_max = int(x_max)
        step = max(1, int(step))
        return np.arange(x_min, x_max + 1, step, dtype=int)

    @staticmethod
    @njit
    def _slice_means_njit(a_sim, effective_y_sim, tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim,
                         ui_sim, pension_sim, gov_m_sim, T: int):
        """
        Compute per-age means for a single cohort simulation (all arrays are shape (T, n_sim)).
        Returns 9 arrays of length T.
        """
        n_sim = a_sim.shape[1]

        a_mean = np.zeros(T)
        labor_mean = np.zeros(T)

        tax_c_mean = np.zeros(T)
        tax_l_mean = np.zeros(T)
        tax_p_mean = np.zeros(T)
        tax_k_mean = np.zeros(T)

        ui_mean = np.zeros(T)
        pension_mean = np.zeros(T)
        gov_health_mean = np.zeros(T)

        for age in range(T):
            sa = 0.0
            sl = 0.0
            stc = 0.0
            stl = 0.0
            stp = 0.0
            stk = 0.0
            sui = 0.0
            spen = 0.0
            sg = 0.0

            for j in range(n_sim):
                sa += a_sim[age, j]
                sl += effective_y_sim[age, j]
                stc += tax_c_sim[age, j]
                stl += tax_l_sim[age, j]
                stp += tax_p_sim[age, j]
                stk += tax_k_sim[age, j]
                sui += ui_sim[age, j]
                spen += pension_sim[age, j]
                sg += gov_m_sim[age, j]

            inv = 1.0 / n_sim
            a_mean[age] = sa * inv
            labor_mean[age] = sl * inv
            tax_c_mean[age] = stc * inv
            tax_l_mean[age] = stl * inv
            tax_p_mean[age] = stp * inv
            tax_k_mean[age] = stk * inv
            ui_mean[age] = sui * inv
            pension_mean[age] = spen * inv
            gov_health_mean[age] = sg * inv

        return (a_mean, labor_mean,
                tax_c_mean, tax_l_mean, tax_p_mean, tax_k_mean,
                ui_mean, pension_mean, gov_health_mean)

    def _simulate_cached(self, t, edu_type, age, remaining_periods, n_sim, seed):
        """Run model.simulate(...) once per key and reuse results."""
        seed = self._seed_u32(seed)
        key = (t, edu_type, age, remaining_periods, n_sim, seed)
        if key in self._sim_cache:
            return self._sim_cache[key]
        model = self.cohort_models[t][edu_type][age]
        res = model.simulate(T_sim=remaining_periods, n_sim=n_sim, seed=seed)
        self._sim_cache[key] = res
        return res
    
    def _simulate_birth_cohort_cached(self, edu_type, birth_period, n_sim, seed):
        """Simulate a full birth cohort once from age 0 and cache it."""
        if not hasattr(self, "_birth_sim_cache"):
            self._birth_sim_cache = {}

        seed = self._seed_u32(seed)
        key = (edu_type, int(birth_period), n_sim, seed)
        if key in self._birth_sim_cache:
            return self._birth_sim_cache[key]

        model = self.birth_cohort_solutions[edu_type][int(birth_period)]
        # Full lifecycle simulation from age 0
        res = model.simulate(T_sim=self.T, n_sim=n_sim, seed=seed)
        self._birth_sim_cache[key] = res
        return res

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

    # --- Demographics: time-varying cohort weights (ageing experiments) -----------------

    def set_cohort_sizes_path_from_pop_growth(self, pop_growth_path):
        """
        Create time-varying cross-sectional cohort weights cohort_sizes_path[t, age].

        This enables "ageing over time" (e.g., declining/negative population growth implies
        relatively smaller newborn cohorts and larger retiree shares in later periods).

        pop_growth_path: array-like, length T_transition
            Growth rate used in the exponential cohort-size rule in each calendar period t.

        Notes
        -----
        This is a *reduced-form* demographic device: each period's cross-sectional age shares
        are reweighted using exp(g_t * years_since_base) and normalized to sum to 1.
        It does not model births/deaths jointly with changing total population.
        """
        pop_growth_path = np.asarray(pop_growth_path, dtype=float)

        if getattr(self, "T_transition", None) is None:
            raise ValueError("T_transition must be set before building cohort_sizes_path.")
        if pop_growth_path.shape[0] != int(self.T_transition):
            raise ValueError("pop_growth_path must have length T_transition.")

        cohort_sizes_path = np.zeros((int(self.T_transition), int(self.T)), dtype=float)

        # At calendar time t (with year current_year + t), age 'age' implies birth year:
        # birth_yr = (current_year + t) - age.
        for t in range(int(self.T_transition)):
            g = float(pop_growth_path[t])
            for age in range(int(self.T)):
                birth_yr = (int(self.current_year) + int(t)) - int(age)
                years_since_base = birth_yr - int(self.birth_year)
                cohort_sizes_path[t, age] = np.exp(g * years_since_base)

            s = float(np.sum(cohort_sizes_path[t, :]))
            if s > 0:
                cohort_sizes_path[t, :] /= s
            else:
                cohort_sizes_path[t, :] = 0.0

        self.cohort_sizes_path = cohort_sizes_path

    def _cohort_weights(self, t):
        """Return age weights for calendar period t (time-varying if cohort_sizes_path exists)."""
        if hasattr(self, "cohort_sizes_path") and self.cohort_sizes_path is not None:
            return self.cohort_sizes_path[int(t), :]
        return self.cohort_sizes

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
                education_type=edu_type, n_a=self.n_a, n_y=self.n_y,
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
            results = ss_model.simulate(T_sim=self.T, n_sim=100, seed=42)
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

                 # DEBUG: Print asset policy for cohorts born during transition
                if verbose and birth_period >= 0 and birth_period < 5:
                    print(f"\n    → Cohort born at t={birth_period} ({edu_type}):")
                    print(f"       Price paths: r={cohort_r[:3]} ... {cohort_r[-2:]}")
                    print(f"                    w={cohort_w[:3]} ... {cohort_w[-2:]}")
                    
                    # Show asset policy at age 0 (newborn)
                    age = 0
                    # a_policy shape: (T, n_a, n_y, n_h, n_e)
                    # Show policy for median asset state, first income/health state
                    mid_a = model.config.n_a // 2
                    a_next_idx = model.a_policy[age, mid_a, 0, 0, 0]
                    a_next_level = model.a_grid[a_next_idx]
                    print(f"       Asset policy at age {age}: a'={a_next_level:.6f} (idx={a_next_idx}, from a={model.a_grid[mid_a]:.6f})")
                    
                    # Show mean asset policy across all states
                    mean_a_policy = np.mean(model.a_policy[age, :, :, :, :])
                    max_a_policy = np.max(model.a_policy[age, :, :, :, :])
                    print(f"       Mean a' at age {age}: {mean_a_policy:.6f}, Max a': {max_a_policy:.6f}")
                    
                    # Check if saving is happening
                    if mean_a_policy < 0.01:
                        print(f"       ⚠️  WARNING: Near-zero savings for this cohort!")

                birth_cohort_solutions[edu_type][birth_period] = model

        # Store birth cohort solutions for later cohort-level simulation/slicing
        self.birth_cohort_solutions = birth_cohort_solutions

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

                    # IMPORTANT: do NOT slice policies.
                    # The model's simulation uses t_sim indexed from 0 but maps to lifecycle_age
                    # using current_age. Policies should remain indexed on absolute lifecycle age (0..T-1).
                    instance_model.V = solved_model.V
                    instance_model.c_policy = solved_model.c_policy
                    instance_model.a_policy = solved_model.a_policy
                    
                    # Store under the LOOP variable age (not reassigned age!)
                    self.cohort_models[t][edu_type][age] = instance_model

        if verbose:
            print("All cohort problems assigned!")
    
    def _crn_seed(self, *, edu_idx: int, birth_period: int, base: int = 42) -> int:
        """
        Common-random-numbers seed rule used across aggregation and fiscal calculations.

        Seed depends ONLY on (education group, birth cohort). Use different `base` values
        only if you intentionally want different random streams.
        """
        raw = int(base) + 10_000 * int(birth_period) + 1_000_000 * int(edu_idx)
        return self._seed_u32(raw)

    def _ensure_cohort_panel_cache(self, n_sim: Optional[int] = None, seed_base: int = 42, verbose: bool = False):
        """
        Precompute (once) all cohort Monte Carlo panels needed to build any (t,age) slice
        during the transition. This eliminates repeated cohort simulations inside
        per-period aggregation routines.

        Cache key: (n_sim, seed_base).
        Cached object: dict[edu_type][birth_period] -> tuple of simulation arrays.
        """
        if n_sim is None:
            if getattr(self, "_last_n_sim", None) is None:
                raise ValueError("n_sim is None and no previous simulate_transition() n_sim is stored.")
            n_sim = int(self._last_n_sim)
        else:
            n_sim = int(n_sim)

        if not hasattr(self, "_cohort_panel_cache"):
            self._cohort_panel_cache = {}

        cache_key = (int(n_sim), int(seed_base))
        if cache_key in self._cohort_panel_cache:
            return

        education_types = list(self.education_shares.keys())
        min_birth_period = -(self.T - 1)                 # cohorts already alive at t=0
        max_birth_period = self.T_transition - 1          # cohorts born during transition

        if verbose:
            print(f"Precomputing cohort panels for birth_period in [{min_birth_period}, {max_birth_period}] "
                  f"(n_sim={n_sim}, seed_base={seed_base}) ...")

        panels = {edu_type: {} for edu_type in education_types}

        for edu_idx, edu_type in enumerate(education_types):
            for b in range(min_birth_period, max_birth_period + 1):
                seed = self._crn_seed(edu_idx=edu_idx, birth_period=int(b), base=int(seed_base))
                panels[edu_type][int(b)] = self._simulate_birth_cohort_cached(
                    edu_type=edu_type,
                    birth_period=int(b),
                    n_sim=int(n_sim),
                    seed=int(seed),
                )

        self._cohort_panel_cache[cache_key] = panels

    def _get_cached_cohort_panel(self, *, edu_type: str, birth_period: int,
                                 n_sim: Optional[int] = None, seed_base: int = 42, verbose: bool = False):
        """Helper to fetch a cached cohort simulated panel, precomputing if needed."""
        if n_sim is None:
            if getattr(self, "_last_n_sim", None) is None:
                raise ValueError("n_sim is None and no previous simulate_transition() n_sim is stored.")
            n_sim = int(self._last_n_sim)
        else:
            n_sim = int(n_sim)

        self._ensure_cohort_panel_cache(n_sim=n_sim, seed_base=seed_base, verbose=verbose)
        return self._cohort_panel_cache[(int(n_sim), int(seed_base))][edu_type][int(birth_period)]

    def _period_cross_section(self, t: int, n_sim: int):
        """
        Build (and cache) all per-(edu,age) objects needed for period-t aggregates + budget.

        IMPORTANT: This version does NOT run new MC simulations. It slices from the
        precomputed cohort panel cache.
        """
        if not hasattr(self, "_period_cache"):
            self._period_cache = {}

        t = int(t)
        seed_base = 42
        n_sim = int(n_sim)
        key = (t, n_sim, int(seed_base))

        if key in self._period_cache:
            return self._period_cache[key]

        # Ensure we have all cohort panels in memory for this (n_sim, seed_base)
        self._ensure_cohort_panel_cache(n_sim=n_sim, seed_base=seed_base, verbose=False)

        education_types = list(self.education_shares.keys())
        n_edu = len(education_types)
        education_shares_array = np.array([self.education_shares[edu] for edu in education_types], dtype=float)
        cohort_sizes_t = self._cohort_weights(t)

        assets_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        labor_by_age_edu = np.zeros((n_edu, self.T), dtype=float)

        tax_c_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_l_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_p_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_k_by_age_edu = np.zeros((n_edu, self.T), dtype=float)

        ui_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        pension_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        gov_health_by_age_edu = np.zeros((n_edu, self.T), dtype=float)

        panels = self._cohort_panel_cache[(n_sim, int(seed_base))]

        for edu_idx, edu_type in enumerate(education_types):
            edu_panels = panels[edu_type]
            for age in range(self.T):
                birth_period = t - age

                (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
                 ui_sim, m_sim, oop_m_sim, gov_m_sim,
                 tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                 pension_sim, retired_sim) = edu_panels[int(birth_period)]

                # Fast per-age means (Numba kernel if you added it; else fallback to np.mean)
                if hasattr(self, "_slice_means_njit"):
                    (a_mean, labor_mean,
                     tax_c_mean, tax_l_mean, tax_p_mean, tax_k_mean,
                     ui_mean, pension_mean, gov_health_mean) = self._slice_means_njit(
                        a_sim, effective_y_sim,
                        tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim,
                        ui_sim, pension_sim, gov_m_sim,
                        int(self.T)
                    )

                    assets_by_age_edu[edu_idx, age] = float(a_mean[age])
                    labor_by_age_edu[edu_idx, age] = float(labor_mean[age])

                    tax_c_by_age_edu[edu_idx, age] = float(tax_c_mean[age])
                    tax_l_by_age_edu[edu_idx, age] = float(tax_l_mean[age])
                    tax_p_by_age_edu[edu_idx, age] = float(tax_p_mean[age])
                    tax_k_by_age_edu[edu_idx, age] = float(tax_k_mean[age])

                    ui_by_age_edu[edu_idx, age] = float(ui_mean[age])
                    pension_by_age_edu[edu_idx, age] = float(pension_mean[age])
                    gov_health_by_age_edu[edu_idx, age] = float(gov_health_mean[age])
                else:
                    assets_by_age_edu[edu_idx, age] = float(np.mean(a_sim[age, :]))
                    labor_by_age_edu[edu_idx, age] = float(np.mean(effective_y_sim[age, :]))

                    tax_c_by_age_edu[edu_idx, age] = float(np.mean(tax_c_sim[age, :]))
                    tax_l_by_age_edu[edu_idx, age] = float(np.mean(tax_l_sim[age, :]))
                    tax_p_by_age_edu[edu_idx, age] = float(np.mean(tax_p_sim[age, :]))
                    tax_k_by_age_edu[edu_idx, age] = float(np.mean(tax_k_sim[age, :]))

                    ui_by_age_edu[edu_idx, age] = float(np.mean(ui_sim[age, :]))
                    pension_by_age_edu[edu_idx, age] = float(np.mean(pension_sim[age, :]))
                    gov_health_by_age_edu[edu_idx, age] = float(np.mean(gov_m_sim[age, :]))

        out = {
            "education_types": education_types,
            "education_shares_array": education_shares_array,
            "cohort_sizes_t": cohort_sizes_t,
            "assets_by_age_edu": assets_by_age_edu,
            "labor_by_age_edu": labor_by_age_edu,
            "tax_c_by_age_edu": tax_c_by_age_edu,
            "tax_l_by_age_edu": tax_l_by_age_edu,
            "tax_p_by_age_edu": tax_p_by_age_edu,
            "tax_k_by_age_edu": tax_k_by_age_edu,
            "ui_by_age_edu": ui_by_age_edu,
            "pension_by_age_edu": pension_by_age_edu,
            "gov_health_by_age_edu": gov_health_by_age_edu,
        }
        self._period_cache[key] = out
        return out

    def compute_aggregates(self, t, n_sim: Optional[int] = None):
        """Compute aggregate capital and labor for period t (reuses period cross-section cache)."""
        if n_sim is None:
            if self._last_n_sim is None:
                raise ValueError("n_sim is None and no previous simulate_transition() n_sim is stored.")
            n_sim = int(self._last_n_sim)

        px = self._period_cross_section(t=int(t), n_sim=int(n_sim))

        K, L = self._aggregate_capital_labor_njit(
            px["assets_by_age_edu"],
            px["labor_by_age_edu"],
            px["cohort_sizes_t"],
            px["education_shares_array"],
        )
        return K, L

    def compute_government_budget(self, t, n_sim: Optional[int] = None):
        """
        Compute government revenues, expenditures, and deficit for period t.

        NOTE: This routine reuses the same Monte Carlo cohort simulations as compute_aggregates()
        via _period_cross_section(), so we do NOT re-simulate cohorts here.
        """
        if n_sim is None:
            if self._last_n_sim is None:
                raise ValueError("n_sim is None and no previous simulate_transition() n_sim is stored.")
            n_sim = int(self._last_n_sim)

        px = self._period_cross_section(t=int(t), n_sim=int(n_sim))

        # Aggregate budget components from the already-computed per-(edu,age) means
        total_tax_c = 0.0
        total_tax_l = 0.0
        total_tax_p = 0.0
        total_tax_k = 0.0
        total_ui = 0.0
        total_pension = 0.0
        total_gov_health = 0.0

        n_edu = px["assets_by_age_edu"].shape[0]
        for edu_idx in range(n_edu):
            for age in range(self.T):
                weight = float(px["cohort_sizes_t"][age]) * float(px["education_shares_array"][edu_idx])

                total_tax_c += weight * float(px["tax_c_by_age_edu"][edu_idx, age])
                total_tax_l += weight * float(px["tax_l_by_age_edu"][edu_idx, age])
                total_tax_p += weight * float(px["tax_p_by_age_edu"][edu_idx, age])
                total_tax_k += weight * float(px["tax_k_by_age_edu"][edu_idx, age])

                total_ui += weight * float(px["ui_by_age_edu"][edu_idx, age])
                total_pension += weight * float(px["pension_by_age_edu"][edu_idx, age])
                total_gov_health += weight * float(px["gov_health_by_age_edu"][edu_idx, age])

        total_revenue = total_tax_c + total_tax_l + total_tax_p + total_tax_k
        total_spending = total_ui + total_pension + total_gov_health
        primary_deficit = total_spending - total_revenue

        return {
            "tax_c": total_tax_c,
            "tax_l": total_tax_l,
            "tax_p": total_tax_p,
            "tax_k": total_tax_k,
            "total_revenue": total_revenue,
            "ui": total_ui,
            "pension": total_pension,
            "gov_health": total_gov_health,
            "total_spending": total_spending,
            "primary_deficit": primary_deficit,
        }

    def simulate_transition(self, r_path, w_path=None,
                           tau_c_path=None, tau_l_path=None,
                           tau_p_path=None, tau_k_path=None,
                           pension_replacement_path=None,
                           n_sim=10000, verbose=True,
                           pop_growth_path=None):
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
        pop_growth_path : array_like, optional
            If provided, creates time-varying cohort weights (ageing over time).
        """
        r_path = np.array(r_path)
        self.T_transition = len(r_path)

        # NEW: build time-varying cohort sizes if requested
        if pop_growth_path is not None:
            self.set_cohort_sizes_path_from_pop_growth(pop_growth_path)
        
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
        
        # Store n_sim so other routines can reuse it by default
        self._last_n_sim = int(n_sim)

        # clear caches for this run
        self._sim_cache = {}
        self._birth_sim_cache = {}
        self._period_cache = {}
        self._cohort_panel_cache = {}

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

        # Precompute cohort panels ONCE (requires birth_cohort_solutions from solve_cohort_problems)
        self._ensure_cohort_panel_cache(n_sim=int(n_sim), seed_base=42, verbose=verbose)
        
        if verbose:
            print("\nComputing aggregate quantities...")
        
        # Compute aggregates from household decisions
        K_path = np.zeros(self.T_transition)
        L_path = np.zeros(self.T_transition)
        
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            K_path[t], L_path[t] = self.compute_aggregates(t, n_sim=None)  # reuse stored n_sim
        
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
    
    def compute_government_budget_path(self, n_sim: Optional[int] = None, verbose=True):
        """Compute government budget for all transition periods."""
        if self.cohort_models is None:
            raise ValueError("Must simulate transition first")

        if n_sim is None:
            if self._last_n_sim is None:
                raise ValueError("n_sim is None and no previous simulate_transition() n_sim is stored.")
            n_sim = int(self._last_n_sim)

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
            
            budget_t = self.compute_government_budget(t, n_sim=int(n_sim))
            
            for key in budget_path.keys():
                budget_path[key][t] = budget_t[key]
        
        self.budget_path = budget_path
        
        if verbose:
            print("\nGovernment Budget Summary:")
            print(f"  Average revenue:     {np.mean(budget_path['total_revenue']):.2f}")
            print(f"    Consumption tax:   {np.mean(budget_path['tax_c']):.2f}")
            print(f"    Labor income tax:  {np.mean(budget_path['tax_l']):.2f}")
            print(f"    Payroll tax:       {np.mean(budget_path['tax_p']):.2f}")
            print(f"    Capital income tax:{np.mean(budget_path['tax_k']):.2f}")
            print(f"  Average spending:    {np.mean(budget_path['total_spending']):.2f}")
            print(f"    UI benefits:       {np.mean(budget_path['ui']):.2f}")
            print(f"    Pensions:          {np.mean(budget_path['pension']):.2f}")
            print(f"    Health spending:   {np.mean(budget_path['gov_health']):.2f}")
            print(f"  Average deficit:     {np.mean(budget_path['primary_deficit']):.2f}")
            print(f"  Deficit/GDP:         {np.mean(budget_path['primary_deficit'] / self.Y_path):.5%}")
        
        return budget_path
    
    def _default_plot_filename(self, prefix: str, ext: str = "png") -> str:
        """
        Default unique filename for plots to avoid accidental overwrites.

        Includes timestamp + key scenario parameters.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_sim = getattr(self, "_last_n_sim", None)
        ttr = getattr(self, "T_transition", None)
        edu_tag = "x".join(sorted(list(getattr(self, "education_shares", {}).keys()))) or "edu"
        return f"{prefix}_{ts}_Ttr{ttr}_T{self.T}_nsim{n_sim}_{edu_tag}.{ext}"

    def plot_government_budget(self, save=True, show=True, filename=None):
        """Plot government budget constraint components (drops an initial burn-in window)."""
        if not hasattr(self, "budget_path") or self.budget_path is None:
            self.compute_government_budget_path(n_sim=None, verbose=True)

        Ttr = int(self.T_transition)
        burn = int(min(20, Ttr // 2))

        periods_full = np.arange(Ttr, dtype=int)
        periods = periods_full[burn:]

        # NEW: sparse x-ticks (every 5 periods)
        x_ticks = self._sparse_int_ticks(int(periods[0]), int(periods[-1]), step=5) if periods.size else np.array([], dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        def _slice(x):
            return np.asarray(x)[burn:]

        # Plot 1
        ax = axes[0, 0]
        ax.plot(periods, _slice(self.budget_path["total_revenue"]), label="Total Revenue", linewidth=2, color="green")
        ax.plot(periods, _slice(self.budget_path["total_spending"]), label="Total Spending", linewidth=2, color="red")
        ax.fill_between(
            periods, _slice(self.budget_path["total_revenue"]), _slice(self.budget_path["total_spending"]),
            where=(_slice(self.budget_path["total_spending"]) > _slice(self.budget_path["total_revenue"])),
            alpha=0.3, color="red", label="Deficit"
        )
        ax.fill_between(
            periods, _slice(self.budget_path["total_revenue"]), _slice(self.budget_path["total_spending"]),
            where=(_slice(self.budget_path["total_revenue"]) > _slice(self.budget_path["total_spending"])),
            alpha=0.3, color="green", label="Surplus"
        )
        ax.set_xlabel("Period")
        ax.set_ylabel("Amount")
        ax.set_title("Revenue vs Spending", fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2
        ax = axes[0, 1]
        ax.plot(periods, _slice(self.budget_path["tax_c"]), label="Consumption Tax", linewidth=2)
        ax.plot(periods, _slice(self.budget_path["tax_l"]), label="Labor Income Tax", linewidth=2)
        ax.plot(periods, _slice(self.budget_path["tax_p"]), label="Payroll Tax", linewidth=2)
        ax.plot(periods, _slice(self.budget_path["tax_k"]), label="Capital Income Tax", linewidth=2)
        ax.set_xlabel("Period")
        ax.set_ylabel("Tax Revenue")
        ax.set_title("Tax Revenue by Type", fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3
        ax = axes[0, 2]
        ax.plot(periods, _slice(self.budget_path["ui"]), label="UI Benefits", linewidth=2)
        ax.plot(periods, _slice(self.budget_path["pension"]), label="Pensions", linewidth=2)
        ax.plot(periods, _slice(self.budget_path["gov_health"]), label="Health Spending", linewidth=2)
        ax.set_xlabel("Period")
        ax.set_ylabel("Spending")
        ax.set_title("Government Spending by Category", fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4
        ax = axes[1, 0]
        ax.plot(periods, _slice(self.budget_path["primary_deficit"]), linewidth=2, color="purple")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.fill_between(periods, 0, _slice(self.budget_path["primary_deficit"]),
                        where=(_slice(self.budget_path["primary_deficit"]) > 0),
                        alpha=0.3, color="red", label="Deficit")
        ax.fill_between(periods, 0, _slice(self.budget_path["primary_deficit"]),
                        where=(_slice(self.budget_path["primary_deficit"]) < 0),
                        alpha=0.3, color="green", label="Surplus")
        ax.set_xlabel("Period")
        ax.set_ylabel("Primary Deficit")
        ax.set_title("Primary Deficit (Spending - Revenue)", fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5
        ax = axes[1, 1]
        deficit_gdp_ratio = (_slice(self.budget_path["primary_deficit"]) / _slice(self.Y_path)) * 100
        ax.plot(periods, deficit_gdp_ratio, linewidth=2, color="darkred")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.fill_between(periods, 0, deficit_gdp_ratio, where=(deficit_gdp_ratio > 0), alpha=0.3, color="red")
        ax.fill_between(periods, 0, deficit_gdp_ratio, where=(deficit_gdp_ratio < 0), alpha=0.3, color="green")
        ax.set_xlabel("Period")
        ax.set_ylabel("Deficit/GDP (%)")
        ax.set_title("Primary Deficit as % of GDP", fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.grid(True, alpha=0.3)

        # Plot 6
        ax = axes[1, 2]
        revenue_gdp = (_slice(self.budget_path["total_revenue"]) / _slice(self.Y_path)) * 100
        spending_gdp = (_slice(self.budget_path["total_spending"]) / _slice(self.Y_path)) * 100
        ax.plot(periods, revenue_gdp, label="Revenue/GDP", linewidth=2, color="green")
        ax.plot(periods, spending_gdp, label="Spending/GDP", linewidth=2, color="red")
        ax.set_xlabel("Period")
        ax.set_ylabel("% of GDP")
        ax.set_title("Fiscal Ratios to GDP", fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle("Government Budget Constraint Over Transition", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save:
            if filename is None:
                filename = self._default_plot_filename("government_budget")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Government budget plot saved to: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_transition(self, save=True, show=True, filename=None):
        """Plot r, w, K, L, Y over the transition (drops an initial burn-in window)."""
        if self.K_path is None or self.L_path is None or self.Y_path is None:
            raise ValueError("Run simulate_transition() before plotting.")

        Ttr = int(self.T_transition)
        burn = int(min(20, Ttr // 2))

        periods_full = np.arange(Ttr, dtype=int)
        periods = periods_full[burn:]

        # NEW: sparse x-ticks (every 5 periods)
        x_ticks = self._sparse_int_ticks(int(periods[0]), int(periods[-1]), step=5) if periods.size else np.array([], dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        # r
        axes[0].plot(periods, np.asarray(self.r_path)[burn:], linewidth=2)
        axes[0].set_title("Interest rate r")
        axes[0].set_xlabel("Period")
        axes[0].set_xticks(x_ticks)
        axes[0].grid(True, alpha=0.3)

        # w
        axes[1].plot(periods, np.asarray(self.w_path)[burn:], linewidth=2)
        axes[1].set_title("Wage w")
        axes[1].set_xlabel("Period")
        axes[1].set_xticks(x_ticks)
        axes[1].grid(True, alpha=0.3)

        # K
        axes[2].plot(periods, np.asarray(self.K_path)[burn:], linewidth=2)
        axes[2].set_title("Aggregate capital K")
        axes[2].set_xlabel("Period")
        axes[2].set_xticks(x_ticks)
        axes[2].grid(True, alpha=0.3)

        # L
        L_series = np.asarray(self.L_path)[burn:]
        axes[3].plot(periods, L_series, linewidth=2)
        axes[3].set_title("Aggregate labor L")
        axes[3].set_xlabel("Period")
        axes[3].set_xticks(x_ticks)
        axes[3].grid(True, alpha=0.3)

        # NEW: widen y-axis to reduce visual jitter (around mean level)
        if L_series.size:
            mu = float(np.mean(L_series))
            band = 0.25  # around L≈1.22 this shows [~0.97, ~1.47]; increase to flatten more
            axes[3].set_ylim(mu - band, mu + band)

        # Y
        axes[4].plot(periods, np.asarray(self.Y_path)[burn:], linewidth=2)
        axes[4].set_title("Output Y")
        axes[4].set_xlabel("Period")
        axes[4].set_xticks(x_ticks)
        axes[4].grid(True, alpha=0.3)

        # K/Y
        ky = np.asarray(self.K_path) / np.asarray(self.Y_path)
        axes[5].plot(periods, ky[burn:], linewidth=2)
        axes[5].set_title("K/Y")
        axes[5].set_xlabel("Period")
        axes[5].set_xticks(x_ticks)
        axes[5].grid(True, alpha=0.3)

        plt.suptitle("Transition Dynamics", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save:
            if filename is None:
                filename = self._default_plot_filename("transition")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Transition plot saved to: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_lifecycle_comparison(
        self,
        birth_periods=(0, 5),
        edu_type="medium",
        n_sim: Optional[int] = None,
        save=True,
        show=True,
        filename=None,
        use_crn: bool = True,
        seed_base: int = 42,
    ):
        """
        Plot lifecycle profiles for two cohorts.

        If use_crn=True, reuse the same cohort Monte Carlo draws as aggregates/budget
        (seed depends only on cohort + education), avoiding extra MC runs.
        """
        if self.cohort_models is None:
            raise ValueError("Run simulate_transition() before plotting.")

        if n_sim is None:
            if self._last_n_sim is None:
                raise ValueError("n_sim is None and no previous simulate_transition() n_sim is stored.")
            n_sim = int(self._last_n_sim)

        birth_periods = list(birth_periods)
        if len(birth_periods) != 2:
            raise ValueError("birth_periods must have length 2.")

        cohort_labels = [f"cohort born in period t={int(b)}" for b in birth_periods]

        # Map edu_type -> edu_idx for CRN seed rule
        education_types = list(self.education_shares.keys())
        if edu_type not in education_types:
            raise ValueError(f"edu_type='{edu_type}' not in education_shares keys: {education_types}")
        edu_idx = education_types.index(edu_type)

        series = []
        for b in birth_periods:
            b = int(b)
            if use_crn:
                seed = self._crn_seed(edu_idx=edu_idx, birth_period=b, base=int(seed_base))
            else:
                seed = 999 + 10_000 * b  # legacy behavior (extra MC stream)

            results = self._simulate_birth_cohort_cached(
                edu_type=edu_type,
                birth_period=b,
                n_sim=int(n_sim),
                seed=seed,
            )

            (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
             ui_sim, m_sim, oop_m_sim, gov_m_sim,
             tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
             pension_sim, retired_sim) = results

            max_age = min(self.T - 1, self.T_transition - 1 - b)
            ages = np.arange(0, max_age + 1, dtype=int)

            mean_a = np.mean(a_sim[ages, :], axis=1)
            mean_c = np.mean(c_sim[ages, :], axis=1)
            mean_y_eff = np.mean(effective_y_sim[ages, :], axis=1)
            emp_rate = np.mean(employed_sim[ages, :], axis=1)
            ui_rate = np.mean(ui_sim[ages, :] > 0, axis=1).astype(float)
            mean_pension = np.mean(pension_sim[ages, :], axis=1)

            series.append(
                {"b": b, "ages": ages, "a": mean_a, "c": mean_c, "y_eff": mean_y_eff,
                 "emp": emp_rate, "ui": ui_rate, "pension": mean_pension}
            )

        # --- Plot: 6 panels (2x3) ---
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.ravel()

        def _plot_panel(ax, key, title, ylabel):
            for s, lbl in zip(series, cohort_labels):
                if s["ages"].size == 0:
                    continue
                order = np.argsort(s["ages"])
                ax.plot(
                    s["ages"][order],
                    s[key][order],
                    marker="o",
                    linewidth=2,
                    label=lbl,
                )
            ax.set_title(title)
            ax.set_xlabel("Lifecycle age")
            ax.set_ylabel(ylabel)

            # Ensure integer x-axis for ages (0..T-1)
            age_ticks = np.arange(0, int(self.T), 5, dtype=int)
            ax.set_xticks(age_ticks)

            ax.grid(True, alpha=0.3)
            ax.legend()

        _plot_panel(axes[0], "a", f"Mean assets by age (edu={edu_type})", "Mean assets")
        _plot_panel(axes[1], "c", f"Mean consumption by age (edu={edu_type})", "Mean consumption")
        _plot_panel(axes[2], "y_eff", f"Effective labor income by age (edu={edu_type})", "Effective income")
        _plot_panel(axes[3], "emp", f"Employment rate by age (edu={edu_type})", "Employment rate")
        _plot_panel(axes[4], "ui", f"UI recipiency by age (edu={edu_type})", "UI recipiency")
        _plot_panel(axes[5], "pension", f"Mean pension by age (edu={edu_type})", "Mean pension")

        plt.suptitle("Lifecycle Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            if filename is None:
                filename = self._default_plot_filename("lifecycle_comparison")
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Lifecycle comparison plot saved to: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

def get_test_config():
    """Return a minimal LifecycleConfig for fast testing."""
    config = LifecycleConfig(
        T=20,
        beta=0.99,
        gamma=1.0,
        n_a=100,
        n_y=2,
        n_h=1,
        retirement_age=15,
        education_type='medium'
    )
    return config


def run_fast_test():
    """Run OLG transition with minimal parameters for fast testing."""
    import sys
    import time

    # Single MC knob for this run
    N_SIM_TEST = 100

    print("=" * 60)
    print("RUNNING FAST TEST MODE")
    print("=" * 60)

    config = get_test_config()

    print("\nTest configuration:")
    print(f"  T = {config.T} periods (e.g., ages 20-{20 + config.T - 1})")
    print(f"  retirement_age = {config.retirement_age} (e.g., age {20 + config.retirement_age})")
    print(f"  n_a = {config.n_a} asset grid points")
    print(f"  n_sim = {N_SIM_TEST} simulations")
    print()

    economy = OLGTransition(
        lifecycle_config=config,
        alpha=0.33,
        delta=0.05,
        A=1.0,
        pop_growth=0.02,
        birth_year=2005,
        current_year=2020,
        education_shares={'medium': 1.0},
        output_dir='output/test'
    )
    
    T_transition = 25  
    r_initial = 0.04
    r_final = 0.03
    
    periods = np.arange(T_transition)
    r_path = r_initial + (r_final - r_initial) * (periods / (T_transition - 1))
    
    # Tax rates and pension
    tau_c_path = np.ones(T_transition) * 0.18
    tau_l_path = np.ones(T_transition) * 0.15
    tau_p_path = np.ones(T_transition) * 0.2
    tau_k_path = np.ones(T_transition) * 0.2
    pension_replacement_path = np.ones(T_transition) * 0.8
    
    print(f"Simulating transition from r={r_initial:.3f} to r={r_final:.3f}")
    print(f"Transition periods: {T_transition}")
       
    # NEW: ageing population experiment (declining pop growth over time)
    pop_growth_path = np.linspace(0.02, 0.02, T_transition)
 
    start = time.time()
    results = economy.simulate_transition(
        r_path=r_path,
        w_path=None,
        tau_c_path=tau_c_path,
        tau_l_path=tau_l_path,
        tau_p_path=tau_p_path,
        tau_k_path=tau_k_path,
        pension_replacement_path=pension_replacement_path,
        n_sim=N_SIM_TEST,  # <-- single knob
        pop_growth_path=pop_growth_path,
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
        birth_periods=[0, 5],
        edu_type='medium',
        n_sim=None,  # reuse economy._last_n_sim (== N_SIM_TEST)
        save=True,
        show=False,
        filename='test_lifecycle_comparison.png'
    )
    
    # Add government budget plot
    economy.compute_government_budget_path(n_sim=None, verbose=True)  # reuse economy._last_n_sim
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

    # Single MC knob for this run
    N_SIM_FULL = 15000

    print("=" * 60)
    print("RUNNING FULL SIMULATION")
    print("=" * 60)

    config = LifecycleConfig(
        T=40,
        beta=0.99,
        gamma=2.0,
        n_a=100,
        n_y=2,
        n_h=2,
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
    
    T_transition = 60  # Longer transition period
    r_initial = 0.04
    r_final = 0.03

    # Tax rates and pension
    tau_c_path = np.ones(T_transition) * 0.18
    tau_l_path = np.ones(T_transition) * 0.15
    tau_p_path = np.ones(T_transition) * 0.2
    tau_k_path = np.ones(T_transition) * 0.2
    pension_replacement_path = np.ones(T_transition) * 0.8
      
    periods = np.arange(T_transition)
    r_path = r_initial + (r_final - r_initial) * (1 - np.exp(-periods / 5))
    
    # NEW: ageing population experiment (declining pop growth over time)
    pop_growth_path = np.linspace(0.01, 0.01, T_transition)
    
    print(f"\nSimulating transition from r={r_initial:.3f} to r={r_final:.3f}")
    
    start = time.time()
    results = economy.simulate_transition(
        r_path=r_path,
        w_path=None,
        tau_c_path=tau_c_path,
        tau_l_path=tau_l_path,
        tau_p_path=tau_p_path,
        tau_k_path=tau_k_path,
        pension_replacement_path=pension_replacement_path,
        n_sim=N_SIM_FULL,  # <-- single knob
        verbose=True,
        pop_growth_path=pop_growth_path,
    )
    end = time.time()
    
    print(f"\nTotal simulation time: {end - start:.2f} seconds")
    
    economy.plot_transition(save=True, show=False)
    
    for edu_type in ['low', 'medium', 'high']:
        economy.plot_lifecycle_comparison(
            birth_periods=[0, 5],
            edu_type=edu_type,
            n_sim=None,  # reuse economy._last_n_sim (== N_SIM_FULL)
            save=True,
            show=False
        )
    
    economy.compute_government_budget_path(n_sim=None, verbose=True)  # reuse economy._last_n_sim
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