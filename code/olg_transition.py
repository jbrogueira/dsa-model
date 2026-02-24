import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from lifecycle_perfect_foresight import LifecycleModelPerfectForesight, LifecycleConfig
import os
from datetime import datetime
from numba import njit
from typing import Optional

def _get_lifecycle_model_class(backend: str):
    """Return the lifecycle model class for the given backend."""
    if backend == 'numpy':
        return LifecycleModelPerfectForesight
    elif backend == 'jax':
        from lifecycle_jax import LifecycleModelJAX
        return LifecycleModelJAX
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'numpy' or 'jax'.")

# Suppress RuntimeWarning from numpy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _extend_path(path, n_extra):
    """Extend a path by padding with copies of the last value, or return None."""
    if path is None:
        return None
    path = np.array(path)
    return np.concatenate([path, np.ones(n_extra) * path[-1]])


def _extract_cohort_path(path, birth_period, T, default=0.0):
    """Extract a cohort-length slice from a path, handling pre-transition cohorts."""
    if birth_period < 0:
        pre = -birth_period
        if path is not None:
            return np.concatenate([np.ones(pre) * path[0], path[0:T - pre]])
        else:
            return np.full(T, default)
    else:
        if path is not None:
            return path[birth_period:birth_period + T]
        else:
            return np.full(T, default)


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
                 # Public capital in production (Feature #8)
                 eta_g=0.0,
                 K_g_initial=0.0,
                 delta_g=0.05,
                 # Public investment path (Feature #10)
                 I_g_path=None,
                 # SOE / sovereign debt (Feature #9)
                 economy_type='soe',
                 r_star=None,
                 B_path=None,
                 # Demographic parameters
                 pop_growth=0.01,
                 birth_year=1960,
                 current_year=2020,
                 # Education distribution
                 education_shares=None,
                 # Output settings
                 output_dir='output',
                 # Government spending on goods (Feature #17)
                 govt_spending_path=None,
                 # Pension trust fund (Feature #18)
                 S_pens_initial=0.0,
                 # Defense spending (Feature #19, simplified)
                 defense_spending_path=None,
                 # Population aging (Feature #21)
                 fertility_path=None,              # (T + T_transition,) relative entering cohort sizes
                 survival_improvement_rate=0.0,    # annual multiplicative improvement in survival probs
                 # Initial asset distribution opt-in
                 use_initial_distribution=False,   # use ss asset distribution as initial conditions
                 # Backend selection
                 backend='numpy',
                 # JAX simulation chunk size (None = all cohorts at once; set e.g. 10 to avoid GPU OOM)
                 jax_sim_chunk_size=None):
        
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

        # Public capital (Feature #8)
        self.eta_g = eta_g
        self.K_g_initial = K_g_initial
        self.delta_g = delta_g

        # Public investment (Feature #10)
        if I_g_path is not None:
            self.I_g_path = np.asarray(I_g_path, dtype=float)
        else:
            self.I_g_path = None

        # SOE / sovereign debt (Feature #9)
        self.economy_type = economy_type
        self.r_star = r_star
        if B_path is not None:
            self.B_path = np.asarray(B_path, dtype=float)
        else:
            self.B_path = None

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
        
        # Government spending on goods (Feature #17)
        if govt_spending_path is not None:
            self.govt_spending_path = np.asarray(govt_spending_path, dtype=float)
        else:
            self.govt_spending_path = None

        # Pension trust fund (Feature #18)
        self.S_pens_initial = S_pens_initial

        # Defense spending (Feature #19, simplified)
        if defense_spending_path is not None:
            self.defense_spending_path = np.asarray(defense_spending_path, dtype=float)
        else:
            self.defense_spending_path = None

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
        self.K_g_path = None  # Public capital path (Feature #8)
        self.NFA_path = None  # Net foreign assets (Feature #9)
        self.S_pens_path = None  # Pension trust fund balance (Feature #18)
        self.birth_cohort_solutions = None

        # Backend selection ('numpy' or 'jax')
        self.backend = backend
        self._lifecycle_model_class = _get_lifecycle_model_class(backend)
        self.jax_sim_chunk_size = jax_sim_chunk_size

        # Population aging parameters (Feature #21)
        self.fertility_path = np.asarray(fertility_path, dtype=float) if fertility_path is not None else None
        self.survival_improvement_rate = float(survival_improvement_rate)

        # Initial asset distribution opt-in
        self.use_initial_distribution = bool(use_initial_distribution)

        # NEW: remember last Monte Carlo size used in simulate_transition()
        self._last_n_sim: Optional[int] = None

    @staticmethod
    def _seed_u32(x: int) -> int:
        """Map any integer (incl. negative/large) into NumPy's allowed seed range."""
        return int(x % (2**32))

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

    @staticmethod
    @njit
    def _slice_mean_single_age_njit(a_sim, effective_y_sim, tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim,
                                    ui_sim, pension_sim, gov_m_sim, age: int):
        """
        Compute means for a SINGLE age from cohort simulation arrays (shape (T, n_sim)).
        Returns 9 scalar values. O(n_sim) instead of O(T × n_sim).
        """
        n_sim = a_sim.shape[1]
        inv = 1.0 / n_sim

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

        return (sa * inv, sl * inv,
                stc * inv, stl * inv, stp * inv, stk * inv,
                sui * inv, spen * inv, sg * inv)

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

    def _solve_cohorts_jax_batched(self, birth_cohort_solutions, verbose=False):
        """Batch-solve all cohort lifecycle problems in one vmapped XLA call per education type."""
        import jax.numpy as jnp
        from lifecycle_jax import _solve_lifecycle_jax_batched

        for edu_type in self.education_shares.keys():
            models_dict = birth_cohort_solutions[edu_type]
            birth_periods = sorted(models_dict.keys())
            model_list = [models_dict[b] for b in birth_periods]

            if verbose:
                print(f"  JAX batched solve: {edu_type} ({len(model_list)} cohorts)")

            ref = model_list[0]

            # Stack per-cohort paths (already JAX arrays from LifecycleModelJAX.__init__)
            r_paths = jnp.stack([m.r_path for m in model_list])
            w_paths = jnp.stack([m.w_path for m in model_list])
            tau_c_paths = jnp.stack([m.tau_c_path for m in model_list])
            tau_l_paths = jnp.stack([m.tau_l_path for m in model_list])
            tau_p_paths = jnp.stack([m.tau_p_path for m in model_list])
            tau_k_paths = jnp.stack([m.tau_k_path for m in model_list])
            pension_paths = jnp.stack([m.pension_replacement_path for m in model_list])
            w_at_rets = jnp.array([m.w_at_retirement for m in model_list])

            # Pass all args positionally to match vmap in_axes
            P_y_4d_arg = ref.P_y_4d if ref.P_y_age_health else None
            bequest_lumpsums = jnp.array([float(models_dict[b].bequest_lumpsum)
                                          for b in birth_periods])
            V_batch, a_pol_batch, c_pol_batch, l_pol_batch = _solve_lifecycle_jax_batched(
                ref.a_grid, ref.y_grid, ref.h_grid, ref.m_grid,
                ref.P_y_2d, ref.P_h,
                w_at_rets,
                r_paths, w_paths,
                tau_c_paths, tau_l_paths, tau_p_paths, tau_k_paths,
                pension_paths,
                ref.ui_replacement_rate, ref.kappa,
                ref.beta, ref.gamma,
                ref.T, ref.retirement_age,
                ref.pension_min_floor, ref.tax_progressive,
                ref.tax_kappa_hsv, ref.tax_eta,
                ref.transfer_floor, ref.education_subsidy_rate,
                ref.child_cost_profile, ref.schooling_years,
                ref.survival_probs, P_y_4d_arg,
                ref.labor_supply, ref.nu, ref.phi,
                bequest_lumpsums,
            )

            # Inject results back into individual model objects
            for ci, b in enumerate(birth_periods):
                model = models_dict[b]
                model.V = np.asarray(V_batch[ci])
                model.a_policy = np.asarray(a_pol_batch[ci])
                model.c_policy = np.asarray(c_pol_batch[ci])
                model.l_policy = np.asarray(l_pol_batch[ci])

    def _simulate_cohorts_jax_batched(self, n_sim, seed_base, verbose=False):
        """Batched simulation of all cohorts in one vmapped XLA call per education type."""
        import jax
        import jax.numpy as jnp
        from scipy.linalg import eig
        from lifecycle_jax import _simulate_lifecycle_jax_batched

        education_types = list(self.education_shares.keys())
        min_birth_period = -(self.T - 1)
        max_birth_period = self.T_transition - 1
        birth_periods = list(range(min_birth_period, max_birth_period + 1))
        n_cohorts = len(birth_periods)

        if not hasattr(self, "_birth_sim_cache"):
            self._birth_sim_cache = {}

        panels = {edu_type: {} for edu_type in education_types}

        for edu_idx, edu_type in enumerate(education_types):
            model_list = [self.birth_cohort_solutions[edu_type][b] for b in birth_periods]
            ref = model_list[0]
            n_y = ref.n_y

            # Stationary distribution for initial income draws — use 2D P_y
            edu_unemployment_rate = ref.config.edu_params[ref.config.education_type]['unemployment_rate']
            if edu_unemployment_rate < 1e-10:
                stationary = None
            else:
                eigenvalues, eigenvectors = eig(np.asarray(ref.P_y_2d).T)
                stationary_idx = np.argmax(eigenvalues.real)
                stationary_dist = eigenvectors[:, stationary_idx].real
                stationary_dist = stationary_dist / stationary_dist.sum()
                stationary = jnp.array(stationary_dist)

            # Stack per-cohort policies and paths
            a_policies = jnp.stack([jnp.array(m.a_policy) for m in model_list])
            c_policies = jnp.stack([jnp.array(m.c_policy) for m in model_list])
            l_policies = jnp.stack([jnp.array(m.l_policy) for m in model_list])
            w_paths = jnp.stack([m.w_path for m in model_list])
            w_at_rets = jnp.array([m.w_at_retirement for m in model_list])
            r_paths = jnp.stack([m.r_path for m in model_list])
            tau_c_paths = jnp.stack([m.tau_c_path for m in model_list])
            tau_l_paths = jnp.stack([m.tau_l_path for m in model_list])
            tau_p_paths = jnp.stack([m.tau_p_path for m in model_list])
            tau_k_paths = jnp.stack([m.tau_k_path for m in model_list])
            pension_paths = jnp.stack([m.pension_replacement_path for m in model_list])

            # Pre-compute per-cohort initial conditions and PRNG keys
            # (replicates LifecycleModelJAX.simulate() setup per cohort)
            all_initial_i_a = []
            all_initial_i_y = []
            all_initial_avg_earnings = []
            all_initial_n_years = []
            all_sim_keys = []
            all_seeds_u32 = []

            for ci, (b, model) in enumerate(zip(birth_periods, model_list)):
                seed = self._crn_seed(edu_idx=edu_idx, birth_period=int(b), base=int(seed_base))
                seed = self._seed_u32(seed)
                all_seeds_u32.append(seed)

                key = jax.random.PRNGKey(seed)

                # 1st split: draw initial income state
                key, subkey = jax.random.split(key)
                if stationary is None:
                    initial_i_y = jax.random.choice(
                        subkey, jnp.arange(1, n_y), shape=(n_sim,)
                    ).astype(jnp.int32)
                else:
                    initial_i_y = jax.random.choice(
                        subkey, n_y, shape=(n_sim,), p=stationary
                    ).astype(jnp.int32)

                # 2nd split: key for simulation random draws
                key, subkey = jax.random.split(key)

                # Initial assets
                if model.config.initial_assets is not None:
                    i_a_init = int(jnp.argmin(jnp.abs(ref.a_grid - model.config.initial_assets)))
                    initial_i_a = jnp.full(n_sim, i_a_init, dtype=jnp.int32)
                else:
                    initial_i_a = jnp.zeros(n_sim, dtype=jnp.int32)

                # Initial earnings
                if model.config.initial_avg_earnings is not None:
                    initial_avg = jnp.ones(n_sim) * model.config.initial_avg_earnings
                    initial_n = jnp.full(n_sim, model.current_age, dtype=jnp.float64)
                else:
                    initial_avg = jnp.zeros(n_sim)
                    initial_n = jnp.zeros(n_sim, dtype=jnp.float64)

                all_initial_i_a.append(initial_i_a)
                all_initial_i_y.append(initial_i_y)
                all_initial_avg_earnings.append(initial_avg)
                all_initial_n_years.append(initial_n)
                all_sim_keys.append(subkey)

            # Stack into batched arrays
            batch_i_a = jnp.stack(all_initial_i_a)
            batch_i_y = jnp.stack(all_initial_i_y)
            batch_i_h = jnp.zeros((n_cohorts, n_sim), dtype=jnp.int32)
            batch_i_y_last = jnp.stack(all_initial_i_y)  # copy of initial_i_y
            batch_avg_earn = jnp.stack(all_initial_avg_earnings)
            batch_n_years = jnp.stack(all_initial_n_years)
            batch_keys = jnp.stack(all_sim_keys)

            chunk_size = self.jax_sim_chunk_size if self.jax_sim_chunk_size is not None else n_cohorts

            if verbose:
                n_chunks = (n_cohorts + chunk_size - 1) // chunk_size
                print(f"  JAX batched simulate: {edu_type} ({n_cohorts} cohorts, n_sim={n_sim}, chunk_size={chunk_size}, n_chunks={n_chunks})")

            P_y_4d_sim = ref.P_y_4d if ref.P_y_age_health else None

            # Per-cohort arrays indexed along axis 0 — group them for easy slicing.
            per_cohort_arrs = (
                a_policies, c_policies, l_policies,
                w_paths, w_at_rets,
                tau_c_paths, tau_l_paths, tau_p_paths, tau_k_paths,
                r_paths, pension_paths,
                batch_keys,
                batch_i_a, batch_i_y, batch_i_h, batch_i_y_last,
                batch_avg_earn, batch_n_years,
            )

            for chunk_start in range(0, n_cohorts, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_cohorts)
                chunk_actual = chunk_end - chunk_start

                # Pad last chunk to keep shape static (same JIT-compiled kernel reused)
                if chunk_actual < chunk_size:
                    pad = chunk_size - chunk_actual
                    idx = jnp.array(list(range(chunk_start, chunk_end)) + [chunk_end - 1] * pad)
                else:
                    idx = jnp.arange(chunk_start, chunk_end)

                def s(arr):
                    return arr[idx]

                (ca_pol, cc_pol, cl_pol,
                 cw, cwret, ctau_c, ctau_l, ctau_p, ctau_k, cr, cpen,
                 ckeys,
                 ci_a, ci_y, ci_h, ci_y_last, cavg, cn_yr) = (s(a) for a in per_cohort_arrs)

                chunk_results = _simulate_lifecycle_jax_batched(
                    ca_pol, cc_pol, cl_pol,
                    ref.a_grid, ref.y_grid, ref.h_grid, ref.m_grid,
                    ref.P_y_2d, ref.P_h,
                    cw, cwret,
                    ctau_c, ctau_l, ctau_p, ctau_k,
                    cr, cpen,
                    ref.ui_replacement_rate, ref.kappa,
                    ref.retirement_age, ref.T, ref.current_age,
                    n_sim, ckeys,
                    ci_a, ci_y, ci_h, ci_y_last,
                    cavg, cn_yr,
                    ref.pension_min_floor, ref.tax_progressive,
                    ref.tax_kappa_hsv, ref.tax_eta,
                    ref.P_y_age_health, P_y_4d_sim,
                    ref.survival_probs,
                )

                # Store only actual (non-padded) cohorts
                for ci_local in range(chunk_actual):
                    ci = chunk_start + ci_local
                    b = birth_periods[ci]
                    panel = tuple(np.asarray(arr[ci_local]) for arr in chunk_results)
                    panels[edu_type][int(b)] = panel
                    cache_key = (edu_type, int(b), n_sim, all_seeds_u32[ci])
                    self._birth_sim_cache[cache_key] = panel

        return panels

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
    def _production_function_njit(K, L, alpha, A, K_g=1.0, eta_g=0.0):
        """JIT-compiled production function: Y = A * K_g^eta_g * K^alpha * L^(1-alpha)."""
        K_g_factor = K_g ** eta_g if eta_g != 0.0 else 1.0
        return A * K_g_factor * (K ** alpha) * (L ** (1 - alpha))

    @staticmethod
    @njit
    def _marginal_products_njit(K, L, alpha, delta, A, K_g=1.0, eta_g=0.0):
        """JIT-compiled marginal products and factor prices with public capital."""
        K_g_factor = K_g ** eta_g if eta_g != 0.0 else 1.0
        MPK = alpha * A * K_g_factor * (K ** (alpha - 1)) * (L ** (1 - alpha))
        MPL = (1 - alpha) * A * K_g_factor * (K ** alpha) * (L ** (-alpha))
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
    def _compute_output_path_njit(K_path, L_path, alpha, A, K_g_path=None, eta_g=0.0):
        """JIT-compiled computation of output path with optional public capital."""
        T = len(K_path)
        Y_path = np.zeros(T)
        for t in range(T):
            K_g_factor = 1.0
            if eta_g != 0.0 and K_g_path is not None:
                K_g_factor = K_g_path[t] ** eta_g
            Y_path[t] = A * K_g_factor * (K_path[t] ** alpha) * (L_path[t] ** (1 - alpha))
        return Y_path

    @staticmethod
    @njit
    def _compute_wage_path_njit(K_path, L_path, alpha, delta, A, K_g_path=None, eta_g=0.0):
        """JIT-compiled computation of wage path from aggregates with optional public capital."""
        T = len(K_path)
        w_path = np.zeros(T)
        for t in range(T):
            K_g_factor = 1.0
            if eta_g != 0.0 and K_g_path is not None:
                K_g_factor = K_g_path[t] ** eta_g
            MPL = (1 - alpha) * A * K_g_factor * (K_path[t] ** alpha) * (L_path[t] ** (-alpha))
            w_path[t] = MPL
        return w_path
    
    def production_function(self, K, L, K_g=None):
        """Cobb-Douglas production function with optional public capital."""
        K_g_val = K_g if K_g is not None else 1.0
        return self._production_function_njit(K, L, self.alpha, self.A, K_g_val, self.eta_g)

    def factor_prices(self, K, L, K_g=None):
        """Compute factor prices from production function with optional public capital."""
        K_g_val = K_g if K_g is not None else 1.0
        return self._marginal_products_njit(K, L, self.alpha, self.delta, self.A, K_g_val, self.eta_g)

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

    def _survival_schedule_at_year(self, cal_year):
        """Return survival probabilities adjusted for longevity improvement at a given calendar year."""
        base = self.lifecycle_config.survival_probs
        if base is None:
            return None
        improvement = (1.0 + self.survival_improvement_rate) ** (cal_year - self.birth_year)
        return np.clip(base * improvement, 0.0, 1.0)

    def _cohort_survival_schedule(self, birth_period):
        """
        Build age-varying survival schedule for a cohort born at `birth_period`.

        Returns shape (T, n_h): entry [j, :] is the survival probability at age j
        using the calendar-year-adjusted schedule for that cohort at that age.
        """
        base = self.lifecycle_config.survival_probs
        if base is None:
            return None
        T = self.T
        n_h = self.n_h
        sched = np.zeros((T, n_h))
        for j in range(T):
            cal_year = self.birth_year + birth_period + j
            age_row = self._survival_schedule_at_year(cal_year)
            if age_row is not None:
                sched[j, :] = age_row[j, :]
            else:
                sched[j, :] = 1.0
        return sched

    def _build_population_weights(self):
        """
        Compute time-varying cohort size weights from fertility path and survival schedules.

        Sets self.cohort_sizes_path of shape (T_transition, T).
        """
        if self.T_transition is None:
            raise ValueError("T_transition must be set before building population weights.")

        T_trans = int(self.T_transition)
        T = int(self.T)

        # fertility_path: relative sizes of entering cohorts by birth period index
        # birth period index 0 = transition start; negative = pre-transition
        # We need fertility for birth periods -(T-1) ... T_trans-1
        # Mapped into fertility_path array as fertility_path[bp + (T-1)]
        if self.fertility_path is not None:
            fert = np.asarray(self.fertility_path, dtype=float)
        else:
            fert = np.ones(T + T_trans)

        cohort_sizes_path = np.zeros((T_trans, T), dtype=float)

        for t_cal in range(T_trans):
            for age in range(T):
                birth_period = t_cal - age
                fert_idx = birth_period + (T - 1)
                if 0 <= fert_idx < len(fert):
                    fert_val = fert[fert_idx]
                else:
                    fert_val = fert[0] if len(fert) > 0 else 1.0

                # Cumulative survival from birth to age
                surv_sched = self._cohort_survival_schedule(birth_period)
                if surv_sched is not None:
                    cum_surv = 1.0
                    for j in range(age):
                        cum_surv *= np.mean(surv_sched[j, :])
                else:
                    cum_surv = 1.0

                cohort_sizes_path[t_cal, age] = fert_val * cum_surv

            # Normalize
            row_sum = cohort_sizes_path[t_cal, :].sum()
            if row_sum > 0:
                cohort_sizes_path[t_cal, :] /= row_sum

        self.cohort_sizes_path = cohort_sizes_path

    def solve_cohort_problems(self, r_path, w_path,
                          tau_c_path=None, tau_l_path=None,
                          tau_p_path=None, tau_k_path=None,
                          pension_replacement_path=None,
                          bequest_lumpsum_path=None,
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

        # Feature flags from lifecycle_config — forwarded to all per-cohort configs
        _lc = self.lifecycle_config
        _feature_kwargs = dict(
            pension_min_floor=_lc.pension_min_floor,
            tax_progressive=_lc.tax_progressive,
            tax_kappa=_lc.tax_kappa,
            tax_eta=_lc.tax_eta,
            transfer_floor=_lc.transfer_floor,
            survival_probs=_lc.survival_probs,
            m_age_profile=_lc.m_age_profile,
            P_y_by_age_health=_lc.P_y_by_age_health,
            retirement_window=_lc.retirement_window,
            schooling_years=_lc.schooling_years,
            child_cost_profile=_lc.child_cost_profile,
            labor_supply=_lc.labor_supply,
            nu=_lc.nu,
            phi=_lc.phi,
            tau_beq=_lc.tau_beq,
        )

        # Check if per-cohort survival schedules are needed
        _use_per_cohort_survival = (
            self.survival_improvement_rate != 0.0 and
            self.lifecycle_config.survival_probs is not None
        )

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
                pension_replacement_path=np.ones(self.T) * (pension_replacement_path[0] if pension_replacement_path is not None else 0.4),
                **_feature_kwargs,
            )
            ss_model = self._lifecycle_model_class(ss_config, verbose=False)
            ss_model.solve(verbose=False)
            results = ss_model.simulate(T_sim=self.T, n_sim=100, seed=42)
            self.ss_asset_profiles[edu_type] = np.mean(results[0], axis=1)
            self.ss_earnings_profiles[edu_type] = np.mean(results[15], axis=1)
            if not hasattr(self, 'ss_asset_distributions'):
                self.ss_asset_distributions = {}
            self.ss_asset_distributions[edu_type] = results[0]  # (T, n_ss_sim)
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

                # Extract cohort-specific price/policy paths
                cohort_r = _extract_cohort_path(r_path, birth_period, self.T)
                cohort_w = _extract_cohort_path(w_path, birth_period, self.T)
                cohort_tau_c = _extract_cohort_path(tau_c_path, birth_period, self.T, default=0.0)
                cohort_tau_l = _extract_cohort_path(tau_l_path, birth_period, self.T, default=0.0)
                cohort_tau_p = _extract_cohort_path(tau_p_path, birth_period, self.T, default=0.0)
                cohort_tau_k = _extract_cohort_path(tau_k_path, birth_period, self.T, default=0.0)
                cohort_pension = _extract_cohort_path(pension_replacement_path, birth_period, self.T, default=0.4)

                # Create and solve the model for this birth cohort
                cohort_feature_kwargs = dict(_feature_kwargs)
                if _use_per_cohort_survival:
                    cohort_surv = self._cohort_survival_schedule(birth_period)
                    cohort_feature_kwargs['survival_probs'] = cohort_surv
                bequest_ls = (
                    float(bequest_lumpsum_path[birth_period])
                    if bequest_lumpsum_path is not None and birth_period in bequest_lumpsum_path
                    else 0.0
                )
                config = LifecycleConfig(
                    T=self.T, beta=self.beta, gamma=self.gamma, current_age=0,
                    education_type=edu_type, n_a=self.n_a, n_y=self.n_y, n_h=self.n_h,
                    retirement_age=self.retirement_age,
                    r_path=cohort_r, w_path=cohort_w,
                    tau_c_path=cohort_tau_c, tau_l_path=cohort_tau_l,
                    tau_p_path=cohort_tau_p, tau_k_path=cohort_tau_k,
                    pension_replacement_path=cohort_pension,
                    bequest_lumpsum=bequest_ls,
                    **cohort_feature_kwargs,
                )
                
                model = self._lifecycle_model_class(config, verbose=False)
                if self.backend != 'jax':
                    model.solve(verbose=False)

                # DEBUG: Print asset policy for cohorts born during transition
                # (skipped for JAX batched mode — policies not yet available)
                if self.backend != 'jax' and verbose and birth_period >= 0 and birth_period < 5:
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

        # JAX batched solve: all cohorts in one vmapped XLA call per education type
        if self.backend == 'jax':
            self._solve_cohorts_jax_batched(birth_cohort_solutions, verbose)

        # Store birth cohort solutions for later cohort-level simulation/slicing
        self.birth_cohort_solutions = birth_cohort_solutions

        # --- INITIAL CONDITIONS FOR OLD COHORTS ---
        # Old cohorts (birth_period < 0) simulate from age 0 with a=0 and avg_earnings=0,
        # exactly like new cohorts.  Their price path is padded with r_path[0] for ages
        # 0…(k-1), so the correct SS policy functions apply during those years and the
        # cohort arrives at age k (= calendar t=0) in the proper steady-state distribution.
        # Setting initial conditions to SS values at age k (as was done previously) placed
        # age-k wealth at age 0 of the simulation — wrong initial state, wrong trajectory,
        # and constant aggregate cross-sections because all cohorts looked like the SS.
        if verbose: print("\n  Setting initial conditions for pre-transition cohorts...")

        if verbose:
            print("All cohort problems ready!")
    
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

        if self.backend == 'jax':
            panels = self._simulate_cohorts_jax_batched(int(n_sim), int(seed_base), verbose)
        else:
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
        # Mortality is already encoded in simulation arrays: dead agents have zero assets/income
        # in all periods after death (NumPy: loop skips dead agents; JAX: jnp.where(alive, val, 0.0)).
        # Multiplying cohort weights by cum_surv would double-count mortality, so no adjustment here.

        assets_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        labor_by_age_edu = np.zeros((n_edu, self.T), dtype=float)

        tax_c_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_l_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_p_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_k_by_age_edu = np.zeros((n_edu, self.T), dtype=float)

        ui_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        pension_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        gov_health_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        bequest_by_age_edu = np.zeros((n_edu, self.T), dtype=float)

        panels = self._cohort_panel_cache[(n_sim, int(seed_base))]

        for edu_idx, edu_type in enumerate(education_types):
            edu_panels = panels[edu_type]
            for age in range(self.T):
                birth_period = t - age

                panel_data = edu_panels[int(birth_period)]
                if len(panel_data) == 21:
                    (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
                     ui_sim, m_sim, oop_m_sim, gov_m_sim,
                     tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                     pension_sim, retired_sim, l_sim, alive_sim, bequest_sim) = panel_data
                else:
                    (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
                     ui_sim, m_sim, oop_m_sim, gov_m_sim,
                     tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                     pension_sim, retired_sim, l_sim) = panel_data
                    alive_sim = np.ones_like(a_sim, dtype=bool)
                    bequest_sim = np.zeros_like(a_sim)

                # Fast single-age means using Numba - O(n_sim) instead of O(T × n_sim)
                (a_mean, labor_mean,
                 tax_c_mean, tax_l_mean, tax_p_mean, tax_k_mean,
                 ui_mean, pension_mean, gov_health_mean) = self._slice_mean_single_age_njit(
                    a_sim, effective_y_sim,
                    tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim,
                    ui_sim, pension_sim, gov_m_sim,
                    int(age)
                )

                assets_by_age_edu[edu_idx, age] = a_mean
                labor_by_age_edu[edu_idx, age] = labor_mean

                tax_c_by_age_edu[edu_idx, age] = tax_c_mean
                tax_l_by_age_edu[edu_idx, age] = tax_l_mean
                tax_p_by_age_edu[edu_idx, age] = tax_p_mean
                tax_k_by_age_edu[edu_idx, age] = tax_k_mean

                ui_by_age_edu[edu_idx, age] = ui_mean
                pension_by_age_edu[edu_idx, age] = pension_mean
                gov_health_by_age_edu[edu_idx, age] = gov_health_mean
                bequest_by_age_edu[edu_idx, age] = float(np.mean(bequest_sim[age, :]))

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
            "bequest_by_age_edu": bequest_by_age_edu,
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

        # Feature #16: Bequest taxation
        total_bequests = 0.0
        if 'bequest_by_age_edu' in px:
            for edu_idx in range(n_edu):
                for age in range(self.T):
                    weight = float(px["cohort_sizes_t"][age]) * float(px["education_shares_array"][edu_idx])
                    total_bequests += weight * float(px["bequest_by_age_edu"][edu_idx, age])

        tau_beq = float(getattr(self.lifecycle_config, 'tau_beq', 0.0))
        bequest_tax_revenue = tau_beq * total_bequests
        bequest_transfers = (1.0 - tau_beq) * total_bequests

        total_revenue = total_tax_c + total_tax_l + total_tax_p + total_tax_k + bequest_tax_revenue

        t_idx = int(t)
        _at = lambda path, default=0.0: float(path[t_idx]) if path is not None and t_idx < len(path) else default

        G_t = _at(self.govt_spending_path)
        I_g_t = _at(self.I_g_path)
        defense_t = _at(self.defense_spending_path)

        # Feature #9: Sovereign debt service
        debt_service = 0.0
        new_borrowing = 0.0
        if self.B_path is not None:
            r_t = float(self.r_path[t_idx]) if self.r_path is not None else 0.0
            B_t = float(self.B_path[t_idx]) if t_idx < len(self.B_path) else float(self.B_path[-1])
            B_next = float(self.B_path[t_idx + 1]) if t_idx + 1 < len(self.B_path) else float(self.B_path[-1])
            debt_service = r_t * B_t
            new_borrowing = B_next - B_t

        total_spending = (total_ui + total_pension + total_gov_health
                          + G_t + I_g_t + debt_service + defense_t)
        total_revenue_with_borrowing = total_revenue + new_borrowing
        primary_deficit = total_spending - total_revenue
        fiscal_deficit = total_spending - total_revenue_with_borrowing

        return {
            "tax_c": total_tax_c,
            "tax_l": total_tax_l,
            "tax_p": total_tax_p,
            "tax_k": total_tax_k,
            "total_revenue": total_revenue,
            "ui": total_ui,
            "pension": total_pension,
            "gov_health": total_gov_health,
            "govt_spending": G_t,
            "public_investment": I_g_t,
            "defense_spending": defense_t,
            "debt_service": debt_service,
            "new_borrowing": new_borrowing,
            "total_spending": total_spending,
            "primary_deficit": primary_deficit,
            "fiscal_deficit": fiscal_deficit,
            "bequest_tax": bequest_tax_revenue,
            "bequest_transfers": bequest_transfers,
            "total_bequests": total_bequests,
        }

    def _compute_bequest_lumpsum_path(self, n_sim: Optional[int] = None) -> dict:
        """Compute per-capita after-tax bequest transfer for each birth period.

        Returns a dict {birth_period: bequest_lumpsum} mapping each birth cohort
        to the per-individual lump-sum transfer received at age 0.  Must be called
        after simulate_transition() so that the cohort panel cache is populated.

        The transfer at birth period b = (1 - tau_beq) * total_bequests(b) / newborn_cohort_weight(b),
        where total_bequests(b) is the aggregate bequest pool at calendar time b (drawn from
        all cohorts alive at b who die that period).
        """
        if n_sim is None:
            n_sim = int(self._last_n_sim)
        tau_beq = float(getattr(self.lifecycle_config, 'tau_beq', 0.0))
        min_bp = 1 - self.T
        max_bp = self.T_transition - 1
        result = {}
        for t in range(self.T_transition):
            px = self._period_cross_section(t=t, n_sim=int(n_sim))
            n_edu = len(px["education_types"])
            total_bequests = 0.0
            if 'bequest_by_age_edu' in px:
                for edu_idx in range(n_edu):
                    for age in range(self.T):
                        weight = float(px["cohort_sizes_t"][age]) * float(px["education_shares_array"][edu_idx])
                        total_bequests += weight * float(px["bequest_by_age_edu"][edu_idx, age])
            after_tax_bequest = (1.0 - tau_beq) * total_bequests
            # Newborn cohort weight at calendar time t (age=0)
            newborn_weight = float(px["cohort_sizes_t"][0])
            if newborn_weight > 0.0:
                result[t] = after_tax_bequest / newborn_weight
            else:
                result[t] = 0.0
        return result

    def simulate_transition(self, r_path, w_path=None,
                           tau_c_path=None, tau_l_path=None,
                           tau_p_path=None, tau_k_path=None,
                           pension_replacement_path=None,
                           n_sim=10000, verbose=True,
                           pop_growth_path=None,
                           bequest_lumpsum_path=None):
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

        # Feature #8/#10: Compute public capital path before wages (wages depend on K_g)
        K_g_path = None
        if self.eta_g != 0.0 and self.I_g_path is not None:
            K_g_path = np.zeros(self.T_transition)
            K_g_path[0] = self.K_g_initial
            for t in range(1, self.T_transition):
                I_g_t = self.I_g_path[t - 1] if t - 1 < len(self.I_g_path) else self.I_g_path[-1]
                K_g_path[t] = (1 - self.delta_g) * K_g_path[t - 1] + I_g_t
            if verbose:
                print(f"\nPublic capital path: K_g[0]={K_g_path[0]:.4f} → K_g[-1]={K_g_path[-1]:.4f}")

        # Extend r_path for cohorts born before transition
        r_path_full = np.concatenate([r_path, np.ones(self.T) * r_path[-1]])

        # Compute wage path from production function
        if w_path is None:
            if verbose:
                print("\nComputing wage path from production function...")

            # With public capital: r + δ = α·A·K_g^{η_g}·(K/L)^{α-1}
            K_g_factor = np.ones(self.T_transition)
            if K_g_path is not None and self.eta_g != 0.0:
                K_g_factor = K_g_path ** self.eta_g

            K_over_L = np.power((r_path + self.delta) / (self.alpha * self.A * K_g_factor),
                                1.0 / (self.alpha - 1.0))
            w_path = (1 - self.alpha) * self.A * K_g_factor * np.power(K_over_L, self.alpha)

            if verbose:
                print(f"  Initial: r={r_path[0]:.4f} → K/L={K_over_L[0]:.4f} → w={w_path[0]:.4f}")
                print(f"  Final:   r={r_path[-1]:.4f} → K/L={K_over_L[-1]:.4f} → w={w_path[-1]:.4f}")
        else:
            w_path = np.array(w_path)
            if verbose:
                print("\nUsing provided wage path")
        
        # Extend paths for cohorts born before transition (pad with last value)
        w_path_full = _extend_path(w_path, self.T)
        tau_c_path_full = _extend_path(tau_c_path, self.T)
        tau_l_path_full = _extend_path(tau_l_path, self.T)
        tau_p_path_full = _extend_path(tau_p_path, self.T)
        tau_k_path_full = _extend_path(tau_k_path, self.T)
        pension_path_full = _extend_path(pension_replacement_path, self.T)
        
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
            bequest_lumpsum_path=bequest_lumpsum_path,
            verbose=verbose
        )

        # Precompute cohort panels ONCE (requires birth_cohort_solutions from solve_cohort_problems)
        self._ensure_cohort_panel_cache(n_sim=int(n_sim), seed_base=42, verbose=verbose)

        # Feature #21: Build population weights from fertility + survival
        if self.fertility_path is not None or self.survival_improvement_rate != 0.0:
            self._build_population_weights()

        if verbose:
            print("\nComputing aggregate quantities...")
        
        # Compute aggregates from household decisions
        K_path = np.zeros(self.T_transition)
        L_path = np.zeros(self.T_transition)
        
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            K_path[t], L_path[t] = self.compute_aggregates(t, n_sim=None)  # reuse stored n_sim
        
        # Compute output (with public capital if present)
        Y_path = self._compute_output_path_njit(K_path, L_path, self.alpha, self.A,
                                                 K_g_path, self.eta_g)

        # Verify consistency: check if implied r from aggregates matches exogenous r
        if verbose:
            print("\nVerifying consistency with production function...")
            K_g_0 = K_g_path[0] if K_g_path is not None else 1.0
            K_g_end = K_g_path[-1] if K_g_path is not None else 1.0
            r_implied, w_implied = self._marginal_products_njit(
                K_path[0], L_path[0], self.alpha, self.delta, self.A, K_g_0, self.eta_g
            )
            print(f"  Period 0:")
            print(f"    Exogenous r: {r_path[0]:.4f}, Implied r: {r_implied:.4f}")
            print(f"    Computed w:  {w_path[0]:.4f}, Implied w: {w_implied:.4f}")

            if self.T_transition > 1:
                r_implied_end, w_implied_end = self._marginal_products_njit(
                    K_path[-1], L_path[-1], self.alpha, self.delta, self.A, K_g_end, self.eta_g
                )
                print(f"  Period {self.T_transition-1}:")
                print(f"    Exogenous r: {r_path[-1]:.4f}, Implied r: {r_implied_end:.4f}")
                print(f"    Computed w:  {w_path[-1]:.4f}, Implied w: {w_implied_end:.4f}")

        # Feature #9: Compute net foreign assets in SOE mode
        NFA_path = None
        if self.economy_type == 'soe':
            # In SOE: domestic capital demand K is determined by production function
            # Household savings supply K_hh = K_path (from aggregation)
            # NFA = K_hh - K_domestic (positive = net creditor)
            # With public capital: K_domestic = K_over_L * L
            K_g_factor_arr = np.ones(self.T_transition)
            if K_g_path is not None and self.eta_g != 0.0:
                K_g_factor_arr = K_g_path ** self.eta_g
            K_over_L_implied = np.power(
                (r_path + self.delta) / (self.alpha * self.A * K_g_factor_arr),
                1.0 / (self.alpha - 1.0)
            )
            K_domestic = K_over_L_implied * L_path
            NFA_path = K_path - K_domestic
            if verbose:
                print(f"\n  SOE: NFA[0]={NFA_path[0]:.4f}, NFA[-1]={NFA_path[-1]:.4f}")

        # Store results
        self.r_path = r_path
        self.w_path = w_path
        self.K_path = K_path
        self.L_path = L_path
        self.Y_path = Y_path
        self.K_g_path = K_g_path
        self.NFA_path = NFA_path

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
            if K_g_path is not None:
                print(f"  K_g range: [{np.min(K_g_path):.4f}, {np.max(K_g_path):.4f}]")
            if NFA_path is not None:
                print(f"  NFA range: [{np.min(NFA_path):.4f}, {np.max(NFA_path):.4f}]")

        result = {'r': self.r_path, 'w': self.w_path, 'K': self.K_path,
                  'L': self.L_path, 'Y': self.Y_path}
        if K_g_path is not None:
            result['K_g'] = K_g_path
        if NFA_path is not None:
            result['NFA'] = NFA_path
        return result
    
    def compute_government_budget_path(self, n_sim: Optional[int] = None, verbose=True):
        """Compute government budget for all transition periods."""
        if self.birth_cohort_solutions is None:
            raise ValueError("Must simulate transition first")

        if n_sim is None:
            if self._last_n_sim is None:
                raise ValueError("n_sim is None and no previous simulate_transition() n_sim is stored.")
            n_sim = int(self._last_n_sim)

        if verbose:
            print("\nComputing government budget path...")
        
        # Initialize storage — keys match compute_government_budget() output
        budget_keys = [
            'tax_c', 'tax_l', 'tax_p', 'tax_k', 'total_revenue',
            'ui', 'pension', 'gov_health', 'govt_spending',
            'public_investment', 'defense_spending',
            'debt_service', 'new_borrowing',
            'total_spending', 'primary_deficit', 'fiscal_deficit',
            'bequest_tax', 'bequest_transfers', 'total_bequests',
        ]
        budget_path = {k: np.zeros(self.T_transition) for k in budget_keys}

        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")

            budget_t = self.compute_government_budget(t, n_sim=int(n_sim))

            for key in budget_path.keys():
                budget_path[key][t] = budget_t[key]

        self.budget_path = budget_path

        # Feature #18: Compute pension trust fund path
        # S[t+1] = (1+r[t]) * S[t] + payroll_tax[t] - pension_spending[t]
        S_pens = np.zeros(self.T_transition + 1)
        S_pens[0] = self.S_pens_initial
        for t in range(self.T_transition):
            r_t = float(self.r_path[t]) if self.r_path is not None else 0.0
            S_pens[t + 1] = (1 + r_t) * S_pens[t] + budget_path['tax_p'][t] - budget_path['pension'][t]
        self.S_pens_path = S_pens
        budget_path['S_pens'] = S_pens[:-1]  # Store balance at start of each period

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
            print(f"    Govt spending (G): {np.mean(budget_path['govt_spending']):.2f}")
            if np.any(budget_path['public_investment'] != 0):
                print(f"    Public invest (Ig):{np.mean(budget_path['public_investment']):.2f}")
            if np.any(budget_path['defense_spending'] != 0):
                print(f"    Defense spending:   {np.mean(budget_path['defense_spending']):.2f}")
            if np.any(budget_path['debt_service'] != 0):
                print(f"    Debt service:      {np.mean(budget_path['debt_service']):.2f}")
                print(f"    New borrowing:     {np.mean(budget_path['new_borrowing']):.2f}")
            print(f"  Average deficit:     {np.mean(budget_path['primary_deficit']):.2f}")
            print(f"  Deficit/GDP:         {np.mean(budget_path['primary_deficit'] / self.Y_path):.5%}")
            if self.S_pens_initial != 0.0 or np.any(S_pens != 0):
                print(f"  Pension trust fund:  S[0]={S_pens[0]:.2f} → S[-1]={S_pens[-1]:.2f}")

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
        if self.birth_cohort_solutions is None:
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
             pension_sim, retired_sim, l_sim, *_) = results

            max_age = min(self.T - 1, self.T_transition - 1 - b)
            ages = np.arange(0, max_age + 1, dtype=int)

            mean_a = np.mean(a_sim[ages, :], axis=1)
            mean_c = np.mean(c_sim[ages, :], axis=1)
            mean_y_eff = np.mean(effective_y_sim[ages, :], axis=1)
            emp_rate = np.mean(employed_sim[ages, :], axis=1)
            ui_rate = np.mean(ui_sim[ages, :] > 0, axis=1).astype(float)
            mean_pension = np.mean(pension_sim[ages, :], axis=1)
            mean_l = np.mean(l_sim[ages, :], axis=1)

            series.append(
                {"b": b, "ages": ages, "a": mean_a, "c": mean_c, "y_eff": mean_y_eff,
                 "emp": emp_rate, "ui": ui_rate, "pension": mean_pension, "l": mean_l}
            )

        # --- Plot: 7 panels (2x4, last slot empty) ---
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
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
        _plot_panel(axes[3], "l", f"Mean labor supply by age (edu={edu_type})", "Mean labor hours")
        _plot_panel(axes[4], "emp", f"Employment rate by age (edu={edu_type})", "Employment rate")
        _plot_panel(axes[5], "ui", f"UI recipiency by age (edu={edu_type})", "UI recipiency")
        _plot_panel(axes[6], "pension", f"Mean pension by age (edu={edu_type})", "Mean pension")
        axes[7].set_visible(False)

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
    T, n_h = 20, 1
    config = LifecycleConfig(
        T=T,
        beta=0.99,
        gamma=1.0,
        n_a=100,
        n_y=2,
        n_h=n_h,
        retirement_age=15,
        education_type='medium',
        labor_supply=True,
        nu=1.0,
        phi=2.0,
        survival_probs=np.linspace(0.995, 0.90, T).reshape(T, n_h),
    )
    return config


def run_fast_test(backend='numpy'):
    """Run OLG transition with minimal parameters for fast testing."""
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
        output_dir='output/test',
        backend=backend,
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
        birth_periods=[0, 20],
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


def run_full_simulation(backend='numpy'):
    """Run full OLG transition simulation."""
    # Single MC knob for this run
    N_SIM_FULL = 15000

    print("=" * 60)
    print("RUNNING FULL SIMULATION")
    print("=" * 60)

    # Lifecycle: age 20–79 (60 annual periods). Retirement at 70 = period 50.
    _T, _n_h = 60, 2

    # Survival probabilities π(j, h): piecewise linear, ages 20–79
    #   Ages 20–39 (j=0–19):  low mortality, 99.9% → 99.5%
    #   Ages 40–64 (j=20–44): slowly declining, 99.5% → 97.0%
    #   Ages 65–79 (j=45–59): steeper decline, 97.0% → 88.0%
    _surv_base = np.concatenate([
        np.linspace(0.999, 0.995, 20),
        np.linspace(0.995, 0.970, 25),
        np.linspace(0.970, 0.880, 15),
    ])                                                       # shape (60,)
    _surv_good = np.clip(_surv_base + 0.005, 0.0, 1.0)      # healthy: +0.5 pp
    _surv_bad  = np.clip(_surv_base - 0.005, 0.0, 1.0)      # unhealthy: -0.5 pp

    config = LifecycleConfig(
        T=_T,
        beta=0.99,
        gamma=2.0,
        n_a=100,
        n_y=2,
        n_h=_n_h,
        retirement_age=50,   # age 70 = period 50 (age 20 + 50)
        education_type='medium',
        labor_supply=True,
        nu=10.0,   # calibrated so FOC gives l≈0.1–0.3 for most agents (clamp l≤1 enforced)
        phi=2.0,
        survival_probs=np.column_stack([_surv_good, _surv_bad]),  # (60, 2)
    )

    economy = OLGTransition(
        lifecycle_config=config,
        alpha=0.33,
        delta=0.05,
        A=1.0,
        pop_growth=0.01,
        birth_year=1940,
        current_year=2020,
        education_shares={'low': 0.3, 'medium': 0.5, 'high': 0.2},
        output_dir='output',
        backend=backend,
        jax_sim_chunk_size=10 if backend == 'jax' else None,
    )

    T_transition = 80    # > one full lifecycle (T=60), ensures cohorts born early complete
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
            birth_periods=[0, 30],   # cohort born at t=0 vs t=30 (mid-transition)
            edu_type=edu_type,
            n_sim=None,  # reuse economy._last_n_sim (== N_SIM_FULL)
            save=True,
            show=False
        )
    
    economy.compute_government_budget_path(n_sim=None, verbose=True)  # reuse economy._last_n_sim
    economy.plot_government_budget(save=True, show=False)
    
    print("\nAll plots saved to 'output' directory")
    
    return economy, results


def _parse_backend():
    """Parse --backend flag from sys.argv."""
    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return 'numpy'


def main():
    """
    Main entry point. Check for --test flag and run accordingly.
    """
    backend = _parse_backend()
    if backend != 'numpy':
        print(f"Using backend: {backend}")

    if '--test' in sys.argv:
        economy, results = run_fast_test(backend=backend)
    else:
        economy, results = run_full_simulation(backend=backend)

    return economy, results


if __name__ == "__main__":
    economy, results = main()