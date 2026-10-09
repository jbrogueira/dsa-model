import sys
import time
import hashlib
import pickle
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from lifecycle_perfect_foresight import (LifecycleModelPerfectForesight, LifecycleConfig,
                                         income_matrices_by_age, employed_transition_matrix)
from firm_conditions import firm_conditions, marginal_products
import os
from datetime import datetime
from numba import njit
from typing import Optional
import warnings

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
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Indices into the raw simulation output; both backends return 23-tuples since
# 2026-10-02 (22 since Phase 8), and _panel_to_age_means accepts 21, 22 or 23.
# Order matches the return of _slice_mean_single_age_njit + bequest as 11th
# element + the means-tested transfer as 12th:
#   a, c, eff_y, tax_c, tax_l, tax_p, tax_k, ui, pension, gov_h, bequest, transfer
_PANEL_MEANS_IDX = (0, 1, 5, 11, 12, 13, 14, 7, 16, 10, 20, 22)
N_AGE_MEANS = len(_PANEL_MEANS_IDX)


def _panel_to_age_means(panel_data):
    """Reduce a raw 23-tuple (or 21/22, or legacy 19) of (T, n) simulation arrays
    to a 12-tuple of (T,) per-age means.  Memory-efficient: intermediate (T, n)
    slices are never retained after the mean is taken.

    Index 21 is ``alpha_idx_sim`` (Phase 8), never averaged; index 22 is
    ``transfer_sim`` (2026-10-02). A 21- or 22-tuple has no transfer column and
    gets zeros; a legacy 19-tuple also lacks bequests."""
    T_len = panel_data[0].shape[0]
    if len(panel_data) >= 23:
        return tuple(np.mean(panel_data[i], axis=1) for i in _PANEL_MEANS_IDX)
    elif len(panel_data) in (21, 22):
        out = tuple(np.mean(panel_data[i], axis=1) for i in _PANEL_MEANS_IDX[:11])
        return out + (np.zeros(T_len),)
    elif len(panel_data) >= 19:
        # legacy: no alive_sim / bequest_sim / transfer_sim
        out = tuple(np.mean(panel_data[i], axis=1) for i in _PANEL_MEANS_IDX[:10])
        return out + (np.zeros(T_len), np.zeros(T_len))
    else:
        raise ValueError(f"Unexpected panel_data length {len(panel_data)}")


def _extend_path(path, n_extra):
    """Extend a path by padding with copies of the last value, or return None."""
    if path is None:
        return None
    path = np.array(path)
    return np.concatenate([path, np.ones(n_extra) * path[-1]])


def _extract_cohort_path(path, birth_period, T, default=0.0, pre_value=None):
    """Extract a cohort-length slice from a path, handling pre-transition cohorts.

    Parameters
    ----------
    pre_value : float or None
        If provided, used as the fill value for pre-transition periods instead of
        ``path[0]``.  Pass the baseline (steady-state) value to enforce MIT-shock
        convention: pre-transition cohorts always experience baseline policy before
        t=0, regardless of which counterfactual scenario is being run.
    """
    if birth_period < 0:
        pre = -birth_period
        fill = pre_value if pre_value is not None else (path[0] if path is not None else default)
        if path is not None:
            return np.concatenate([np.ones(pre) * fill, path[0:T - pre]])
        else:
            return np.full(T, fill)
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
                 # Sovereign debt service rate (defaults to r_star = private capital return).
                 # Setting r_B<r captures the wedge between Eurozone sovereign yields and the
                 # firm-FOC marginal product of capital.
                 r_B=None,
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
                 # Other net primary spending residual (baseline fiscal closure):
                 # (other expenditure - other revenue) not modelled elsewhere.
                 other_net_spending_path=None,
                 # Tax on gross output paid by firms: a scalar or a path by
                 # transition period. It enters the firm conditions
                 # (firm_conditions.py) and the budget as revenue tau_y Y.
                 tau_y=0.0,
                 # Real sovereign rate by transition period; overrides the
                 # scalar r_B when given.
                 r_B_path=None,
                 # Education spending, a level path in the budget:
                 # e_0 Y_ref (w_t / w_0) s_t, with e_0 the base-year share of
                 # output, Y_ref base-year output (the run's own when None),
                 # w the detrended wage and s_t the school-age population
                 # relative to the model's population (education_index_path,
                 # by transition period, one in the base year).
                 education_over_Y0=0.0,
                 education_index_path=None,
                 education_Y0=None,
                 # Lump-sum transfer per adult by transition period (a level
                 # in detrended units), received by every living household
                 # and booked as spending.
                 lump_sum_path=None,
                 # Transfer from abroad (the EU net flow), a share of Y(t),
                 # scalar or path; booked as revenue, a flow from abroad in
                 # the resource constraint.
                 foreign_transfer_over_Y=None,
                 # Index of the unemployment rate by transition period (one in
                 # the base year): each cohort's income transition matrix by
                 # age follows u_e,t = u_e,base x index_t along its diagonal.
                 unemployment_index_path=None,
                 # Population aging (Feature #21)
                 fertility_path=None,              # (T + T_transition,) relative entering cohort sizes
                 survival_improvement_rate=0.0,    # annual multiplicative improvement in survival probs
                 # Data-driven cohort survival: period life tables by calendar year.
                 # Tuple (years, px) with years (Ny,) ascending and px (Ny, T) [or (Ny, T, n_h)],
                 # px[i, j] = survival prob at model age j (real age entry+j) in calendar years[i].
                 # When set, overrides survival_improvement_rate: each cohort uses the data
                 # period tables along its calendar diagonal, clamped to [years[0], years[-1]].
                 survival_table=None,
                 demography=None,
                 # Cohort-specific retirement: {entry_year: ((retirement_age,
                 # pension_avg_weight, share), ...)} for the cohort entering at model
                 # age 0 in entry_year, one entry or two when the cohort is split
                 # between consecutive ages. None keeps lifecycle_config.retirement_age.
                 cohort_retirement=None,
                 # Backend selection
                 backend='numpy',
                 # JAX simulation chunk size (None = all cohorts at once; set e.g. 10 to avoid GPU OOM)
                 jax_sim_chunk_size=None,
                 # Agent simulation batch size: agents simulated at once per cohort (controls RAM)
                 sim_agent_batch_size=10_000,
                 # Number of simulate_transition() calls whose per-cohort age means are
                 # kept for reuse by later calls with the same household inputs (0 = off)
                 household_cache_size=0,
                 # JAX backend: keep the cohort policy functions on the device between
                 # the solve and the simulation instead of copying them to the host and
                 # back (and do not keep the value functions). Holds every cohort's
                 # policies in device memory at once.
                 jax_policies_on_device=False,
                 # How cohort age means are obtained from the decision rules:
                 # 'simulation' draws n_sim household histories per cohort;
                 # 'exact' carries the distribution over states forward by age
                 # (no draws; n_sim and the seeds then play no role)
                 aggregation='simulation'):
        
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
        self.cohort_retirement = cohort_retirement
        
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
        self.r_B = float(r_B) if r_B is not None else None
        self.r_B_path = None   # built per-call in simulate_transition()
        if B_path is not None:
            self.B_path = np.asarray(B_path, dtype=float)
        else:
            self.B_path = None

        # Demographics
        self.pop_growth = pop_growth
        # Balanced growth. g is read from the lifecycle config (its single
        # source); n is the demographic rate above. Every per-capita detrended
        # stock — sovereign debt, public capital, the pension fund, net foreign
        # assets — loses growth_factor per period.
        self.trend_growth = float(getattr(self.lifecycle_config, 'trend_growth', 0.0))
        self.growth_factor = (1.0 + self.trend_growth) * (1.0 + float(pop_growth))
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

        # Other net primary spending residual (baseline fiscal closure).
        # Represents (other expenditure - other revenue) absent from the
        # explicit tax/transfer/spending lines; a constant share of Y is the
        # single knob used to pin the baseline primary balance to a target.
        if other_net_spending_path is not None:
            self.other_net_spending_path = np.asarray(other_net_spending_path, dtype=float)
        else:
            self.other_net_spending_path = None

        # Output tax, sovereign-rate path and the budget lines added on
        # 2026-10-07 (BUDGET_ALIGNMENT_PLAN.md, EC_ALIGNMENT_PLAN.md)
        self.tau_y = (float(tau_y) if np.isscalar(tau_y) or np.ndim(tau_y) == 0
                      else np.asarray(tau_y, dtype=float))
        self.r_B_path_input = (None if r_B_path is None
                               else np.asarray(r_B_path, dtype=float))
        self.education_over_Y0 = float(education_over_Y0 or 0.0)
        self.education_index_path = (None if education_index_path is None
                                     else np.asarray(education_index_path, dtype=float))
        self.education_Y0 = None if education_Y0 is None else float(education_Y0)
        self.lump_sum_path = (None if lump_sum_path is None
                              else np.asarray(lump_sum_path, dtype=float))
        self.foreign_transfer_over_Y = foreign_transfer_over_Y
        self.unemployment_index_path = (None if unemployment_index_path is None
                                        else np.asarray(unemployment_index_path, dtype=float))
        self._P_employed_cache = {}
        self._cohort_P_y_cache = {}

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
        self.NFA_path = None          # Net foreign assets: A - K_domestic - B (Feature #9)
        self.K_domestic_path = None   # Domestic physical capital from firm's FOC (SOE only)
        self.S_pens_path = None  # Pension trust fund balance (Feature #18)
        self.birth_cohort_solutions = None
        self.birth_cohort_later = {}
        self.later_share = {}
        self._active_I_g_path = None           # effective I_g used in last simulate_transition
        self._active_govt_spending_path = None  # effective G used in last simulate_transition
        self._active_defense_spending_path = None    # effective defense used in last simulate_transition
        self._active_other_net_spending_path = None  # effective other-net spending used in last simulate_transition
        # GDP-share spending mode: when set (scalar or (T,) array), the budget
        # uses level = ratio * Y_path[t] for that line, taking precedence over the
        # level path above. Lets exogenous fiscal lines be fixed shares of Y(t).
        self._active_G_over_Y = None
        self._active_I_g_over_Y = None
        self._active_defense_over_Y = None
        self._active_other_net_over_Y = None
        self._active_tau_y_path = None
        self._active_lump_sum_path = None
        self._active_kappa_path = None
        self._active_m_scale_path = None
        self._active_shock_period = 0
        self._active_foreign_transfer_over_Y = None
        self._active_education = (0.0, None, None)   # (e_0, index path, Y_ref)

        # Backend selection ('numpy' or 'jax')
        self.backend = backend
        self._lifecycle_model_class = _get_lifecycle_model_class(backend)
        self.jax_sim_chunk_size = jax_sim_chunk_size
        self.sim_agent_batch_size = int(sim_agent_batch_size)
        self.household_cache_size = int(household_cache_size)
        self.jax_policies_on_device = bool(jax_policies_on_device)
        if aggregation not in ('simulation', 'exact'):
            raise ValueError(f"aggregation must be 'simulation' or 'exact', got {aggregation!r}")
        self.aggregation = aggregation
        self._household_cache = OrderedDict()
        self._household_cache_hits = 0
        # [distinct cohort problems solved, cohort problems] by the batched JAX solve
        self._cohort_solve_counts = [0, 0]

        # Population aging parameters (Feature #21)
        if fertility_path is not None:
            raise NotImplementedError(
                "fertility_path was removed on 2026-10-01. It drove "
                "_build_population_weights, which double-counted survival and "
                "discarded the measured entrant path. Supply an alternative "
                "entrant series through the demographic sidecar instead -- see "
                "code/build_demography_GR.py and transition.demography_file.")
        self.survival_improvement_rate = float(survival_improvement_rate)

        # Data-driven cohort survival (period life tables by calendar year).
        # Stored as ascending years and px reshaped to (Ny, T, n_h).
        self._surv_years = None
        self._surv_px = None
        if survival_table is not None:
            yrs, px = survival_table
            yrs = np.asarray(yrs, dtype=int).ravel()
            px = np.asarray(px, dtype=float)
            if px.ndim == 2:                       # (Ny, T) -> (Ny, T, n_h)
                px = np.repeat(px[:, :, None], self.n_h, axis=2)
            if px.shape[0] != yrs.shape[0] or px.shape[1] != self.T:
                raise ValueError(
                    f"survival_table px shape {px.shape} incompatible with "
                    f"(n_years={yrs.shape[0]}, T={self.T}, n_h={self.n_h})")
            order = np.argsort(yrs)
            self._surv_years = yrs[order]
            self._surv_px = px[order]

        # Demographic path (entering-cohort sizes and the population growth
        # rate by calendar year), built by build_demography_GR.py. When absent
        # the cohort weights stay at the constant-n form and Gamma is scalar.
        self._demog = None
        if demography is not None:
            self._demog = {k: np.asarray(demography[k]).ravel()
                           for k in ('entrant_years', 'entrants',
                                     'pop_years', 'n_path')}
            self._demog['entrant_years'] = self._demog['entrant_years'].astype(int)
            self._demog['pop_years'] = self._demog['pop_years'].astype(int)

        # NEW: remember last Monte Carlo size used in simulate_transition()
        self._last_n_sim: Optional[int] = None

        # Policy version counter: incremented each time solve_cohort_problems() runs.
        # Used as part of the cohort-panel cache key so stale panels are not reused.
        self._policy_version: int = 0

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
    def _slice_mean_single_age_njit(a_sim, c_sim, effective_y_sim,
                                    tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim,
                                    ui_sim, pension_sim, gov_m_sim, age: int):
        """
        Compute means for a SINGLE age from cohort simulation arrays (shape (T, n_sim)).
        Returns 10 scalar values. O(n_sim) instead of O(T × n_sim).
        """
        n_sim = a_sim.shape[1]
        inv = 1.0 / n_sim

        sa = 0.0
        sc = 0.0
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
            sc += c_sim[age, j]
            sl += effective_y_sim[age, j]
            stc += tax_c_sim[age, j]
            stl += tax_l_sim[age, j]
            stp += tax_p_sim[age, j]
            stk += tax_k_sim[age, j]
            sui += ui_sim[age, j]
            spen += pension_sim[age, j]
            sg += gov_m_sim[age, j]

        return (sa * inv, sc * inv, sl * inv,
                stc * inv, stl * inv, stp * inv, stk * inv,
                sui * inv, spen * inv, sg * inv)

    def _simulate_birth_cohort_cached(self, edu_type, birth_period, n_sim, seed,
                                      later=False):
        """Simulate a birth cohort in agent batches and cache per-age means.

        Returns an 11-tuple of (T,) arrays (per-age means), not the raw panel.
        Batching keeps peak RAM at O(batch_size) regardless of n_sim. *later*
        simulates the later-retiring part of a split cohort, with the same seed.
        """
        if not hasattr(self, "_birth_sim_cache"):
            self._birth_sim_cache = {}

        seed = self._seed_u32(seed)
        # Keyed by _policy_version: without it a re-solve inside one
        # simulate_transition call -- which is what the bequest fixed point does --
        # is served the previous policy's panels, so iteration 2 reports a
        # bit-exact zero change and the loop "converges" after one step.
        key = (edu_type, int(birth_period), n_sim, seed,
               getattr(self, '_policy_version', 0)) + (('later',) if later else ())
        if key in self._birth_sim_cache:
            return self._birth_sim_cache[key]

        source = self.birth_cohort_later if later else self.birth_cohort_solutions
        model = source[edu_type][int(birth_period)]
        T_sim = self.T
        batch = self.sim_agent_batch_size

        sums = [np.zeros(T_sim) for _ in range(N_AGE_MEANS)]
        agents_done = 0
        batch_idx = 0
        while agents_done < n_sim:
            n_b = min(batch, n_sim - agents_done)
            b_seed = self._seed_u32(seed + batch_idx * 7919)
            panel = model.simulate(T_sim=T_sim, n_sim=n_b, seed=int(b_seed))
            b_means = _panel_to_age_means(panel)
            for k in range(N_AGE_MEANS):
                sums[k] += b_means[k] * n_b
            agents_done += n_b
            batch_idx += 1

        res = tuple(s / n_sim for s in sums)
        self._birth_sim_cache[key] = res
        return res

    @staticmethod
    def _retirement_groups(models_by_bp):
        """Birth periods grouped by the retirement-dependent scalars of their models.

        The batched JAX solve and simulation take retirement_age as a static
        argument and pension_avg_weight and mean_kappa_working as scalars shared
        across the batch, so cohorts that differ in them are batched separately.
        Groups come in order of their first birth period.
        """
        groups = {}
        for b in sorted(models_by_bp):
            m = models_by_bp[b]
            key = (int(m.retirement_age), float(m.pension_avg_weight),
                   float(m.mean_kappa_working))
            groups.setdefault(key, []).append(b)
        return list(groups.values())

    def _solve_cohorts_jax_batched(self, birth_cohort_solutions, verbose=False):
        """Batch-solve the cohort lifecycle problems, one vmapped XLA call per
        education type and retirement group."""
        for edu_type in self.education_shares.keys():
            models_dict = birth_cohort_solutions[edu_type]
            for group in self._retirement_groups(models_dict):
                self._solve_jax_group(edu_type, models_dict, group, verbose)

    def _solve_jax_group(self, edu_type, models_dict, birth_periods, verbose=False):
        """Batch-solve the cohorts in `birth_periods`, which share retirement_age,
        pension_avg_weight and mean_kappa_working."""
        import jax.numpy as jnp
        from lifecycle_jax import _solve_lifecycle_jax_batched

        # Cohorts with identical inputs have identical solutions: solve one of
        # each and let the others share its arrays. With constant prices and
        # policies, cohorts differ only in their survival schedule, which is
        # the same for every cohort entering after the mortality projection ends.
        group_birth_periods = birth_periods
        solved_as, first_with = {}, {}
        for b in group_birth_periods:
            solved_as[b] = first_with.setdefault(self._solve_inputs_key(models_dict[b]), b)
        birth_periods = [b for b in group_birth_periods if solved_as[b] == b]
        self._cohort_solve_counts[0] += len(birth_periods)
        self._cohort_solve_counts[1] += len(group_birth_periods)

        model_list = [models_dict[b] for b in birth_periods]

        if verbose:
            print(f"  JAX batched solve: {edu_type} ({len(group_birth_periods)} cohorts, "
                  f"{len(model_list)} distinct)")

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
        ls_paths = jnp.stack([m.lump_sum_path for m in model_list])
        # Coverage and medical spending by age, per cohort (they move with
        # calendar time in the health experiments)
        kappa_paths = jnp.stack([m.kappa_path for m in model_list])
        m_grids = jnp.stack([m.m_grid for m in model_list])

        # Pass all args positionally to match vmap in_axes. With age-dependent
        # income matrices every cohort carries its own (the unemployment path),
        # so they are stacked and batched over cohorts.
        per_cohort_py = bool(ref.P_y_age_health)
        P_y_4d_arg = None
        py_stack = jnp.stack([m.P_y_4d for m in model_list]) if per_cohort_py else None
        from lifecycle_jax import _solve_lifecycle_jax_batched_tr, _solve_lifecycle_jax_batched_tr_pyc
        solve_batched = (_solve_lifecycle_jax_batched_tr_pyc if per_cohort_py
                         else _solve_lifecycle_jax_batched_tr)
        bequest_lumpsums = jnp.array([float(models_dict[b].bequest_lumpsum)
                                      for b in birth_periods])
        # Per-cohort survival schedules (in_axes=0). Cohorts may have distinct
        # schedules (data cohort-historical survival or survival_improvement_rate);
        # fall back to ones (no mortality) where a model has none.
        _ones_surv = jnp.ones((ref.T, self.n_h))
        surv_paths = jnp.stack([
            (m.survival_probs if getattr(m, 'survival_probs', None) is not None
             else _ones_surv) for m in model_list])
        n_cohorts = len(model_list)
        chunk_size = self.jax_sim_chunk_size if self.jax_sim_chunk_size is not None else n_cohorts

        # Phase 8: alpha_mult is shared across cohorts within one solve sweep,
        # so the permanent-FE grid is handled by an outer loop over alpha
        # nodes — one batched solve sweep per node — mirroring the per-alpha
        # loops in LifecycleModelPerfectForesight.solve and
        # LifecycleModelJAX.solve. The per-agent simulation side draws alpha
        # indices over the full grid (Phase 8.5b), so the solve must supply
        # matching per-alpha policies.
        def _solve_chunk(alpha_mult_jax, w_at_c, r_c, w_c, tc_c, tl_c, tp_c, tk_c, pen_c, beq_c,
                         surv_c, ls_c, kap_c, mg_c, *py_c):
            return solve_batched(
                ref.a_grid, ref.y_grid, ref.h_grid, mg_c,
                ref.P_y_2d, ref.P_h,
                w_at_c, r_c, w_c, tc_c, tl_c, tp_c, tk_c, pen_c,
                ref.ui_replacement_rate, kap_c,
                ref.beta, ref.gamma,
                ref.T, ref.retirement_age,
                ref.pension_min_floor, ref.tax_progressive,
                ref.tax_kappa_hsv, ref.tax_eta,
                ref.transfer_floor, ref.education_subsidy_rate,
                ref.child_cost_profile, ref.schooling_years,
                surv_c, (py_c[0] if per_cohort_py else P_y_4d_arg),
                ref.labor_supply, ref.nu, ref.phi, ref.trend_growth,
                beq_c,
                ref.wage_age_profile,
                ref.pension_avg_weight, ref.mean_kappa_working, ref.mean_y_employed,
                alpha_mult_jax,
                ls_c,
                ref.ui_eligibility_prob,
                ref.minimum_income,
            )

        batched_arrays = (w_at_rets, r_paths, w_paths,
                          tau_c_paths, tau_l_paths, tau_p_paths, tau_k_paths,
                          pension_paths, bequest_lumpsums, surv_paths, ls_paths,
                          kappa_paths, m_grids) \
            + ((py_stack,) if per_cohort_py else ())

        # Where the results are kept: host arrays (the default; frees device
        # memory chunk by chunk), or device arrays with jax_policies_on_device,
        # in which case the value functions are dropped.
        on_device = self.jax_policies_on_device
        xp = jnp if on_device else np
        keep = (lambda x: x) if on_device else np.asarray

        n_alpha = ref.n_alpha
        V_alpha_sweeps, a_alpha_sweeps, c_alpha_sweeps, l_alpha_sweeps = [], [], [], []
        for alpha_idx in range(n_alpha):
            alpha_mult_jax = float(np.exp(np.asarray(ref.alpha_grid)[alpha_idx]))
            if chunk_size >= n_cohorts:
                V_b, a_b, c_b, l_b = _solve_chunk(alpha_mult_jax, *batched_arrays)
                V_batch = None if on_device else np.asarray(V_b)
                a_pol_batch = keep(a_b)
                c_pol_batch = keep(c_b)
                l_pol_batch = keep(l_b)
            else:
                V_chunks, a_chunks, c_chunks, l_chunks = [], [], [], []
                for start in range(0, n_cohorts, chunk_size):
                    end = min(start + chunk_size, n_cohorts)
                    actual = end - start
                    pad = chunk_size - actual
                    sliced = tuple(arr[start:end] for arr in batched_arrays)
                    if pad > 0:
                        sliced = tuple(
                            jnp.concatenate([s, jnp.repeat(s[-1:], pad, axis=0)])
                            for s in sliced
                        )
                    V_b, a_b, c_b, l_b = _solve_chunk(alpha_mult_jax, *sliced)
                    if not on_device:
                        V_chunks.append(np.asarray(V_b[:actual]))
                    a_chunks.append(keep(a_b[:actual]))
                    c_chunks.append(keep(c_b[:actual]))
                    l_chunks.append(keep(l_b[:actual]))
                V_batch = None if on_device else np.concatenate(V_chunks)
                a_pol_batch = xp.concatenate(a_chunks)
                c_pol_batch = xp.concatenate(c_chunks)
                l_pol_batch = xp.concatenate(l_chunks)
            V_alpha_sweeps.append(V_batch)
            a_alpha_sweeps.append(a_pol_batch)
            c_alpha_sweeps.append(c_pol_batch)
            l_alpha_sweeps.append(l_pol_batch)

        # Inject results into individual model objects.
        # Per-alpha policies on a leading (n_alpha, T, ...) axis; scalar
        # attributes alias alpha=0, matching LifecycleModelJAX.solve and
        # LifecycleModelPerfectForesight.solve conventions.
        for ci, b in enumerate(birth_periods):
            model = models_dict[b]
            model.V_alpha = (None if on_device
                             else np.stack([Vb[ci] for Vb in V_alpha_sweeps], axis=0))
            model.a_policy_alpha = xp.stack([ab[ci] for ab in a_alpha_sweeps], axis=0)
            model.c_policy_alpha = xp.stack([cb[ci] for cb in c_alpha_sweeps], axis=0)
            model.l_policy_alpha = xp.stack([lb[ci] for lb in l_alpha_sweeps], axis=0)
            model.V = None if on_device else model.V_alpha[0]
            model.a_policy = model.a_policy_alpha[0]
            model.c_policy = model.c_policy_alpha[0]
            model.l_policy = model.l_policy_alpha[0]

        # Cohorts not solved themselves take the arrays of their duplicate.
        # (The MIT stitching copies an array before writing into it.)
        for b in group_birth_periods:
            if solved_as[b] != b:
                solved, model = models_dict[solved_as[b]], models_dict[b]
                for attr in ('V_alpha', 'a_policy_alpha', 'c_policy_alpha', 'l_policy_alpha',
                             'V', 'a_policy', 'c_policy', 'l_policy'):
                    setattr(model, attr, getattr(solved, attr))

    @staticmethod
    def _solve_inputs_key(model):
        """Bytes of the per-cohort inputs to the batched solve (the paths along
        the cohort's diagonal, the wage at retirement, the bequest receipt and
        the survival schedule). Everything else is shared within a group."""
        surv = getattr(model, 'survival_probs', None)
        parts = (model.r_path, model.w_path, model.tau_c_path, model.tau_l_path,
                 model.tau_p_path, model.tau_k_path, model.pension_replacement_path,
                 model.w_at_retirement, model.bequest_lumpsum,
                 surv if surv is not None else (),
                 getattr(model, 'lump_sum_path', ()),
                 # coverage and medical spending by age
                 model.kappa_path, model.m_grid,
                 # the income matrices, which differ across cohorts under an
                 # unemployment path
                 model.P_y)
        return b'|'.join(np.asarray(x, dtype=float).tobytes() for x in parts)

    def _simulate_cohorts_jax_batched(self, n_sim, seed_base, verbose=False,
                                      age_means=False, solutions=None):
        """Batched simulation of all cohorts in one vmapped XLA call per education type.

        Returns dict[edu_type][birth_period] -> the 23-tuple panel of (T, n_sim)
        arrays, or with age_means=True the 12-tuple of (T,) per-age means of
        _panel_to_age_means. The means are taken on the device, so the panels
        (18.6 MB per cohort at n_sim = 2000) are not copied to the host and not
        kept in _birth_sim_cache.

        *solutions* ({edu_type: {birth_period: model}}) simulates those models
        instead of birth_cohort_solutions, with the same seeds per birth period;
        the later-retiring parts of split cohorts are simulated this way.
        """
        import jax
        import jax.numpy as jnp
        from scipy.linalg import eig
        from lifecycle_jax import _simulate_lifecycle_jax_batched

        education_types = list(self.education_shares.keys())
        min_birth_period = -(self.T - 1)
        max_birth_period = self.T_transition - 1
        source = self.birth_cohort_solutions if solutions is None else solutions

        if not hasattr(self, "_birth_sim_cache"):
            self._birth_sim_cache = {}

        panels = {edu_type: {} for edu_type in education_types}

        for edu_idx, edu_type in enumerate(education_types):
            birth_periods = (list(range(min_birth_period, max_birth_period + 1))
                             if solutions is None else sorted(solutions.get(edu_type, {})))
            n_cohorts = len(birth_periods)
            if n_cohorts == 0:
                continue
            model_list = [source[edu_type][b] for b in birth_periods]
            ref = model_list[0]
            n_y = ref.n_y

            # Stationary distribution of each cohort's entry-year income matrix
            # for the initial income draws (the matrices differ across cohorts
            # under an unemployment path). Cached by the matrix's bytes.
            if not hasattr(self, '_stationary_dist_cache'):
                self._stationary_dist_cache = {}

            def stationary_for(m):
                u = m.config.edu_params[m.config.education_type]['unemployment_rate']
                if u < 1e-10:
                    return None
                P2 = np.asarray(m.P_y_2d)
                k = P2.tobytes()
                if k not in self._stationary_dist_cache:
                    eigenvalues, eigenvectors = eig(P2.T)
                    st = eigenvectors[:, np.argmax(eigenvalues.real)].real
                    self._stationary_dist_cache[k] = jnp.array(st / st.sum())
                return self._stationary_dist_cache[k]
            stationary = stationary_for(ref)

            # Stack per-cohort 6-D policies on CPU; upload per chunk during
            # simulation to avoid holding all cohorts' policies on GPU at once.
            # If a_policy_alpha is missing (older code path), wrap the 5-D scalar
            # policy on a singleton leading axis to keep the shape uniform.
            on_device = self.jax_policies_on_device

            def _as_alpha_indexed(m, attr_alpha, attr_scalar):
                arr = getattr(m, attr_alpha, None)
                if arr is None:
                    arr = getattr(m, attr_scalar)[None, ...]
                return arr if on_device else np.asarray(arr)

            def _policy_stack(cohorts, attr_alpha, attr_scalar):
                """Policies of the cohorts of one chunk, stacked where they live."""
                arrs = [_as_alpha_indexed(model_list[ci], attr_alpha, attr_scalar)
                        for ci in cohorts]
                return jnp.stack(arrs) if on_device else np.stack(arrs)
            w_paths = jnp.stack([m.w_path for m in model_list])
            w_at_rets = jnp.array([m.w_at_retirement for m in model_list])
            r_paths = jnp.stack([m.r_path for m in model_list])
            tau_c_paths = jnp.stack([m.tau_c_path for m in model_list])
            tau_l_paths = jnp.stack([m.tau_l_path for m in model_list])
            tau_p_paths = jnp.stack([m.tau_p_path for m in model_list])
            tau_k_paths = jnp.stack([m.tau_k_path for m in model_list])
            pension_paths = jnp.stack([m.pension_replacement_path for m in model_list])
            beq_lumps = jnp.array([float(getattr(m, 'bequest_lumpsum', 0.0)) for m in model_list])
            ls_paths = jnp.stack([m.lump_sum_path for m in model_list])
            kappa_paths = jnp.stack([m.kappa_path for m in model_list])
            m_grids = jnp.stack([m.m_grid for m in model_list])

            # Pre-compute per-cohort initial conditions and PRNG keys
            # (replicates LifecycleModelJAX.simulate() setup per cohort)
            # Cache: keyed by (edu_type, n_sim, seed_base, birth periods). Independent of the
        # policies, but it does depend on the config's initial assets and earnings.
            if not hasattr(self, '_sim_init_cache'):
                self._sim_init_cache = {}
            _init_key = (edu_type, int(n_sim), int(seed_base), tuple(birth_periods))
            if _init_key in self._sim_init_cache:
                (_, batch_i_a, batch_i_y, batch_i_h,
                 batch_i_y_last, batch_avg_earn, batch_n_years,
                 batch_keys, all_seeds_u32) = self._sim_init_cache[_init_key]
            else:
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
                    stationary = stationary_for(model)

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

                batch_i_a = jnp.stack(all_initial_i_a)
                batch_i_y = jnp.stack(all_initial_i_y)
                batch_i_h = jnp.zeros((n_cohorts, n_sim), dtype=jnp.int32)
                batch_i_y_last = jnp.stack(all_initial_i_y)
                batch_avg_earn = jnp.stack(all_initial_avg_earnings)
                batch_n_years = jnp.stack(all_initial_n_years)
                batch_keys = jnp.stack(all_sim_keys)

                self._sim_init_cache[_init_key] = (
                    stationary, batch_i_a, batch_i_y, batch_i_h,
                    batch_i_y_last, batch_avg_earn, batch_n_years,
                    batch_keys, all_seeds_u32,
                )

            # Phase 8: per-cohort permanent-FE draws. With n_alpha=1 the draws
            # collapse to zeros / ones and the leading singleton axis of the
            # 6-D policies makes the simulation identical to the pre-Phase-8 path.
            n_alpha = ref.n_alpha
            if n_alpha > 1:
                alpha_probs_jax = jnp.array(ref.alpha_probs)
                alpha_idx_per_cohort = []
                for ci, seed in enumerate(all_seeds_u32):
                    fe_key = jax.random.PRNGKey(self._seed_u32(seed + 0x9E3779B1))
                    alpha_idx_per_cohort.append(
                        jax.random.choice(fe_key, n_alpha, shape=(n_sim,),
                                          p=alpha_probs_jax).astype(jnp.int32)
                    )
                batch_alpha_idx = jnp.stack(alpha_idx_per_cohort)
            else:
                batch_alpha_idx = jnp.zeros((n_cohorts, n_sim), dtype=jnp.int32)
            batch_alpha_mult = jnp.exp(jnp.array(ref.alpha_grid)[batch_alpha_idx])

            chunk_size = self.jax_sim_chunk_size if self.jax_sim_chunk_size is not None else n_cohorts

            if verbose:
                n_chunks = (n_cohorts + chunk_size - 1) // chunk_size
                print(f"  JAX batched simulate: {edu_type} ({n_cohorts} cohorts, n_sim={n_sim}, chunk_size={chunk_size}, n_chunks={n_chunks})")

            # Per-cohort income matrices (the unemployment path) are batched
            # over cohorts; the kernel variant with that axis is used then.
            per_cohort_py = bool(ref.P_y_age_health)
            P_y_4d_sim = None
            py_stack_sim = jnp.stack([m.P_y_4d for m in model_list]) if per_cohort_py else None
            from lifecycle_jax import (_simulate_lifecycle_jax_batched_tr,
                                       _simulate_lifecycle_jax_batched_tr_pyc)
            simulate_batched = (_simulate_lifecycle_jax_batched_tr_pyc if per_cohort_py
                                else _simulate_lifecycle_jax_batched_tr)

            # Per-cohort survival schedules (in_axes=0 in the batched simulate kernel).
            _ones_surv = jnp.ones((ref.T, self.n_h))
            surv_paths_sim = jnp.stack([
                (m.survival_probs if getattr(m, 'survival_probs', None) is not None
                 else _ones_surv) for m in model_list])

            # Per-cohort arrays indexed along axis 0 — group them for easy slicing.
            per_cohort_arrs = (
                w_paths, w_at_rets,
                tau_c_paths, tau_l_paths, tau_p_paths, tau_k_paths,
                r_paths, pension_paths,
                batch_keys,
                batch_i_a, batch_i_y, batch_i_h, batch_i_y_last,
                batch_avg_earn, batch_n_years,
                batch_alpha_idx, batch_alpha_mult,  # Phase 8 per-cohort FE arrays
                surv_paths_sim,
                beq_lumps,
                ls_paths,
                kappa_paths, m_grids,
            ) + ((py_stack_sim,) if per_cohort_py else ())

            # Cohorts with different retirement ages cannot share a batch (the
            # age is a static argument), so simulate each retirement group in
            # its own chunks, each padded to the group's chunk length.
            bp_index = {int(b): ci for ci, b in enumerate(birth_periods)}
            groups = self._retirement_groups(
                {int(b): m for b, m in zip(birth_periods, model_list)})
            for group in groups:
                gidx = [bp_index[b] for b in group]
                gref = model_list[gidx[0]]
                g_chunk = min(chunk_size, len(gidx))
                for start in range(0, len(gidx), g_chunk):
                    sel = gidx[start:start + g_chunk]
                    chunk_actual = len(sel)
                    padded = sel + [sel[-1]] * (g_chunk - chunk_actual)
                    idx = jnp.array(padded)

                    def s(arr):
                        return arr[idx]

                    ca_pol = _policy_stack(padded, 'a_policy_alpha', 'a_policy')
                    cc_pol = _policy_stack(padded, 'c_policy_alpha', 'c_policy')
                    cl_pol = _policy_stack(padded, 'l_policy_alpha', 'l_policy')
                    sliced = [s(a) for a in per_cohort_arrs]
                    (cw, cwret, ctau_c, ctau_l, ctau_p, ctau_k, cr, cpen,
                     ckeys,
                     ci_a, ci_y, ci_h, ci_y_last, cavg, cn_yr,
                     calpha_idx, calpha_mult, csurv, cbeq, cls,
                     ckappa, cmg, *cpy_l) = sliced
                    cpy = cpy_l[0] if per_cohort_py else P_y_4d_sim

                    chunk_results = simulate_batched(
                        ca_pol, cc_pol, cl_pol,
                        ref.a_grid, ref.y_grid, ref.h_grid, cmg,
                        ref.P_y_2d, ref.P_h,
                        cw, cwret,
                        ctau_c, ctau_l, ctau_p, ctau_k,
                        cr, cpen,
                        ref.ui_replacement_rate, ckappa,
                        gref.retirement_age, ref.T, ref.current_age,
                        n_sim, ckeys,
                        ci_a, ci_y, ci_h, ci_y_last,
                        cavg, cn_yr,
                        ref.pension_min_floor, ref.tax_progressive,
                        ref.tax_kappa_hsv, ref.tax_eta,
                        ref.P_y_age_health, cpy,
                        csurv,
                        ref.wage_age_profile,
                        gref.pension_avg_weight, gref.mean_kappa_working, ref.mean_y_employed,
                        calpha_idx, calpha_mult,
                        ref.trend_growth,
                        ref.transfer_floor,
                        cbeq,
                        cls,
                        ref.ui_eligibility_prob,
                        ref.minimum_income,
                    )

                    # Store only actual (non-padded) cohorts
                    if age_means:
                        # (N_AGE_MEANS, chunk, T): one mean over households per
                        # aggregated panel element, as _panel_to_age_means takes.
                        chunk_means = np.asarray(jnp.stack(
                            [jnp.mean(chunk_results[i], axis=2) for i in _PANEL_MEANS_IDX]))
                        for ci_local, ci in enumerate(sel):
                            panels[edu_type][int(birth_periods[ci])] = tuple(
                                chunk_means[k, ci_local] for k in range(N_AGE_MEANS))
                        continue
                    for ci_local, ci in enumerate(sel):
                        b = birth_periods[ci]
                        panel = tuple(np.asarray(arr[ci_local]) for arr in chunk_results)
                        panels[edu_type][int(b)] = panel
                        cache_key = (edu_type, int(b), n_sim, all_seeds_u32[ci],
                                     getattr(self, '_policy_version', 0))
                        if solutions is None:
                            self._birth_sim_cache[cache_key] = panel

        return panels

    def _exact_cohort_age_means(self, verbose=False, solutions=None):
        """Per-cohort age means from the exact distribution over states.

        Returns dict[edu_type][birth_period] -> 12-tuple of (T,) arrays, the
        layout _panel_to_age_means gives for a simulated panel. Cohorts with
        the same inputs and the same policy arrays have the same means and are
        computed once. *solutions* replaces birth_cohort_solutions (the
        later-retiring parts of split cohorts).
        """
        source = self.birth_cohort_solutions if solutions is None else solutions
        panels = {edu_type: {} for edu_type in self.education_shares}
        for edu_type in self.education_shares:
            models = source.get(edu_type, {})
            if not models:
                continue
            computed_as, first_with = {}, {}
            for b, m in models.items():
                key = (self._solve_inputs_key(m), id(m.a_policy_alpha),
                       id(m.c_policy_alpha), id(m.l_policy_alpha))
                computed_as[b] = first_with.setdefault(key, b)
            todo = {b: models[b] for b in models if computed_as[b] == b}
            if verbose:
                print(f"  Exact aggregation: {edu_type} ({len(models)} cohorts, "
                      f"{len(todo)} distinct)")
            if self.backend == 'jax':
                means = self._exact_age_means_jax_batched(todo)
            else:
                means = {b: m.exact_age_means() for b, m in todo.items()}
            for b in models:
                full = means[computed_as[b]]
                panels[edu_type][int(b)] = tuple(full[:, i] for i in _PANEL_MEANS_IDX)
        return panels

    def _exact_age_means_jax_batched(self, models):
        """exact_age_means for the cohort models in `models` ({birth_period:
        model}), one vmapped call per chunk of each retirement group.
        Returns {birth_period: (T, 23) array}."""
        import jax.numpy as jnp
        from lifecycle_jax import _exact_age_means_jax_batched_tr, _exact_age_means_jax_batched_tr_pyc

        out = {}
        for group in self._retirement_groups(models):
            ref = models[group[0]]
            # Per-cohort income matrices and initial distributions under an
            # unemployment path; shared otherwise.
            per_cohort_py = bool(ref.P_y_age_health)
            exact_batched = (_exact_age_means_jax_batched_tr_pyc if per_cohort_py
                             else _exact_age_means_jax_batched_tr)
            if (float(ref.transfer_floor) > 0.0
                    and int(getattr(ref.config, 'schooling_years', 0) or 0) > 0):
                raise NotImplementedError(
                    "the recorded transfer replicates the solve's budget without "
                    "child costs; a positive transfer_floor with schooling_years > 0 "
                    "would record the wrong transfer")
            initial_dist = jnp.array(ref._np_model._initial_distribution())
            P_y_4d = ref.P_y_4d if ref.P_y_age_health else None
            ones_surv = np.ones((ref.T, self.n_h))
            chunk = min(self.jax_sim_chunk_size or len(group), len(group))
            for start in range(0, len(group), chunk):
                sel = group[start:start + chunk]
                # Pad the last chunk to the chunk length (same compiled kernel).
                padded = sel + [sel[-1]] * (chunk - len(sel))
                ms = [models[b] for b in padded]
                stack = lambda f: jnp.stack([jnp.asarray(f(m)) for m in ms])
                if per_cohort_py:
                    init_arg = stack(lambda m: m._np_model._initial_distribution())
                    py_arg = stack(lambda m: m.P_y_4d)
                else:
                    init_arg, py_arg = initial_dist, P_y_4d
                res = exact_batched(
                    stack(lambda m: m.a_policy_alpha), stack(lambda m: m.c_policy_alpha),
                    stack(lambda m: m.l_policy_alpha),
                    ref.a_grid, ref.y_grid, ref.h_grid, stack(lambda m: m.m_grid),
                    ref.P_y_2d, ref.P_h,
                    stack(lambda m: m.w_path), jnp.array([m.w_at_retirement for m in ms]),
                    stack(lambda m: m.tau_c_path), stack(lambda m: m.tau_l_path),
                    stack(lambda m: m.tau_p_path), stack(lambda m: m.tau_k_path),
                    stack(lambda m: m.r_path), stack(lambda m: m.pension_replacement_path),
                    ref.ui_replacement_rate, stack(lambda m: m.kappa_path),
                    ref.retirement_age, ref.T, ref.current_age,
                    init_arg, ref.alpha_grid,
                    ref.pension_min_floor, ref.tax_progressive,
                    ref.tax_kappa_hsv, ref.tax_eta,
                    ref.P_y_age_health, py_arg,
                    stack(lambda m: m.survival_probs if m.survival_probs is not None
                          else ones_surv),
                    ref.wage_age_profile,
                    ref.pension_avg_weight, ref.mean_kappa_working, ref.mean_y_employed,
                    ref.trend_growth,
                    ref.transfer_floor,
                    jnp.array([float(m.bequest_lumpsum) for m in ms]),
                    stack(lambda m: m.lump_sum_path),
                    False,
                    None,
                    ref.ui_eligibility_prob,
                    ref.minimum_income,
                )
                res = np.asarray(res)
                for i, b in enumerate(sel):
                    out[b] = res[i]
        return out

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
            cohort_sizes[i] = (1.0 + pop_growth) ** years_since_base
        return cohort_sizes
    
    @staticmethod
    @njit
    def _production_function_njit(K, L, alpha, A, K_g=1.0, eta_g=0.0):
        """JIT-compiled production function: Y = A * K_g^eta_g * K^alpha * L^(1-alpha)."""
        K_g_factor = K_g ** eta_g if eta_g != 0.0 else 1.0
        return A * K_g_factor * (K ** alpha) * (L ** (1 - alpha))

    @staticmethod
    @njit
    def _marginal_products_njit(K, L, alpha, delta, A, K_g=1.0, eta_g=0.0, tau_y=0.0):
        """JIT-compiled factor prices with public capital and the output tax:
        r = (1 - tau_y) MPK - delta, w = (1 - tau_y) MPL (firm_conditions.py)."""
        K_g_factor = K_g ** eta_g if eta_g != 0.0 else 1.0
        MPK = alpha * A * K_g_factor * (K ** (alpha - 1)) * (L ** (1 - alpha))
        MPL = (1 - alpha) * A * K_g_factor * (K ** alpha) * (L ** (-alpha))
        r = (1.0 - tau_y) * MPK - delta
        w = (1.0 - tau_y) * MPL
        return r, w
    
    @staticmethod
    @njit
    def _aggregate_capital_labor_njit(assets_by_age_edu, consumption_by_age_edu,
                                      labor_by_age_edu,
                                      cohort_sizes, education_shares_array):
        """JIT-compiled aggregation of capital, consumption, and labor across age and education."""
        n_edu, T = assets_by_age_edu.shape
        K = 0.0
        C = 0.0
        L = 0.0

        for edu in range(n_edu):
            for age in range(T):
                weight = cohort_sizes[age] * education_shares_array[edu]
                K += weight * assets_by_age_edu[edu, age]
                C += weight * consumption_by_age_edu[edu, age]
                L += weight * labor_by_age_edu[edu, age]

        return K, C, L
    
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

    def factor_prices(self, K, L, K_g=None, tau_y=None):
        """Compute factor prices from production function with optional public capital."""
        K_g_val = K_g if K_g is not None else 1.0
        if tau_y is None:
            tau_y = self.tau_y if np.isscalar(self.tau_y) else float(np.asarray(self.tau_y)[0])
        return self._marginal_products_njit(K, L, self.alpha, self.delta, self.A, K_g_val,
                                            self.eta_g, float(tau_y))

    @staticmethod
    def _as_period_path(x, n):
        """A scalar or array as a (n,) path: a scalar is repeated, a shorter
        array is padded with its last value, a longer one truncated."""
        if x is None:
            return None
        if np.isscalar(x) or np.ndim(x) == 0:
            return np.full(n, float(x))
        a = np.asarray(x, dtype=float).ravel()
        if len(a) >= n:
            return a[:n].copy()
        return np.concatenate([a, np.full(n - len(a), a[-1])])

    # --- Unemployment path: per-cohort income matrices ------------------------

    def _employed_matrix(self, edu_type):
        """Tauchen matrix of the employed income states of an education group."""
        if edu_type not in self._P_employed_cache:
            self._P_employed_cache[edu_type] = employed_transition_matrix(
                self.lifecycle_config, edu_type)
        return self._P_employed_cache[edu_type]

    def _cohort_unemployment(self, edu_type, birth_period, index_full):
        """(rates by age (T,), P_y by age (T, n_h, n_y, n_y)) of the cohort
        born at birth_period, from the unemployment index along its diagonal
        (one before the base year), or (None, None) without an index path."""
        if index_full is None:
            return None, None
        u0 = float(self.lifecycle_config.edu_params[edu_type]['unemployment_rate'])
        idx = _extract_cohort_path(index_full, birth_period, self.T, default=1.0, pre_value=1.0)
        rates = u0 * np.asarray(idx, dtype=float)
        key = (edu_type, rates.tobytes())
        if key not in self._cohort_P_y_cache:
            lc = self.lifecycle_config
            self._cohort_P_y_cache[key] = income_matrices_by_age(
                self._employed_matrix(edu_type), lc.n_y, lc.n_h, rates,
                lc.job_finding_rate, lc.max_job_separation_rate)
        return rates, self._cohort_P_y_cache[key]

    def _edu_params_at_entry(self, edu_type, u_entry):
        """edu_params with the group's unemployment rate set to the rate at
        the cohort's entry (the rate behind its initial income draw)."""
        ep = dict(self.lifecycle_config.edu_params)
        ep[edu_type] = dict(ep[edu_type], unemployment_rate=float(u_entry))
        return ep

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
                cohort_sizes_path[t, age] = (1.0 + g) ** years_since_base

            s = float(np.sum(cohort_sizes_path[t, :]))
            if s > 0:
                cohort_sizes_path[t, :] /= s
            else:
                cohort_sizes_path[t, :] = 0.0

        self.cohort_sizes_path = cohort_sizes_path

    def _growth_at(self, t):
        """Gamma_t = (1+g)(1+n_t) for transition period t.

        Falls back to the scalar terminal Gamma when no demographic path has
        been built, which is what every fixture without one relies on.
        """
        gp = getattr(self, 'growth_factor_path', None)
        if gp is None:
            return float(self.growth_factor)
        return float(gp[int(np.clip(t, 0, len(gp) - 1))])

    def growth_factors(self, T_tr):
        """Gamma_t = (1+g)(1+n_t) for a transition of T_tr periods.

        Public because the callers that size the baseline public-investment
        path need Gamma_t before a transition has been run.
        """
        T_tr = int(T_tr)
        if self._demog is None:
            return np.full(T_tr, self.growth_factor)
        yrs, n = self._demog['pop_years'], self._demog['n_path']
        want = int(self.current_year) + np.arange(T_tr)
        if want[0] < yrs[0]:
            raise ValueError(
                f"demography starts in {yrs[0]} but the transition starts in "
                f"{want[0]}")
        # Past the end of the table the population is stable by construction,
        # so holding the terminal Gamma is exact. The fiscal layer extends its
        # recursions beyond the simulated horizon with every path frozen, and
        # asks for Gamma over that longer span.
        idx = np.searchsorted(yrs, np.clip(want, yrs[0], yrs[-1]))
        return (1.0 + self.trend_growth) * (1.0 + n[idx])

    def _entrant_weights(self, t):
        """Cross-sectional weights for period t from cohort sizes at entry.

        The transition's per-age means run over all simulated agents with the
        dead holding zero, so survival is already inside the mean and the
        weight must be the entering size rather than the living count. That is
        the opposite convention from the calibration, whose means are taken
        among the alive.
        """
        T = int(self.T)
        yrs, B = self._demog['entrant_years'], self._demog['entrants']
        year = int(self.current_year) + int(t)
        if year - (T - 1) < yrs[0]:
            raise ValueError(
                f"entering cohorts cover {yrs[0]}..{yrs[-1]} but period "
                f"t={t} needs {year - (T - 1)}..{year}")
        need = year - np.arange(T)
        # Past the end of the table the entering cohort grows at the terminal
        # rate n_inf (the table ends after the ramp has brought it there), so
        # extrapolating at that rate reproduces the series the builder would
        # have written. Needed when a fiscal run extends the horizon past the
        # table (n_post), which raised here until 2026-10-02 while
        # growth_factors clipped the same overrun.
        n_inf = float(self._demog['n_path'][-1])
        i = np.searchsorted(yrs, np.minimum(need, yrs[-1]))
        w = np.asarray(B[i], dtype=float) * (1.0 + n_inf) ** np.maximum(need - yrs[-1], 0)
        return w / w.sum()

    def _build_cohort_sizes_from_entrants(self):
        """Set cohort_sizes_path from the entering-cohort sizes."""
        self.cohort_sizes_path = np.array(
            [self._entrant_weights(t) for t in range(int(self.T_transition))])

    def _cohort_weights(self, t):
        """Return age weights for calendar period t (time-varying if cohort_sizes_path exists)."""
        if hasattr(self, "cohort_sizes_path") and self.cohort_sizes_path is not None:
            return self.cohort_sizes_path[int(t), :]
        if self._demog is not None:
            # The path is built inside simulate_transition; answer correctly for
            # callers that ask before one has been run.
            return self._entrant_weights(t)
        return self.cohort_sizes

    def _alive_fraction(self, t):
        """Living share of the cohorts present in period t.

        Each cohort's simulated means already hold zero for the dead, so the
        weights have to be sizes at entry or mortality is counted twice. That
        makes a weighted sum a total per person *ever entered*, which is not a
        per-capita quantity: the dead sit in the denominator. This returns
        sum_j w_j S_j, the factor between the two, so dividing by it gives a
        total per person alive -- which is what the detrending assumes, since
        Gamma_t is built from the growth of the living population.

        Returns 1.0 when the model has no mortality, so fixtures without a
        survival schedule are unaffected.
        """
        key = int(t)
        cache = getattr(self, '_alive_frac_cache', None)
        if cache is None:
            cache = self._alive_frac_cache = {}
        if key in cache:
            return cache[key]
        if int(self.n_h) > 1:
            raise NotImplementedError(
                f"_alive_fraction averages survival over health states "
                f"unweighted, which is only the probability of being alive at "
                f"n_h == 1 (this model has n_h == {self.n_h}). The correct S_j "
                f"weights each health state by its share of the age-j "
                f"population. Demonstrated error with two health states: a "
                f"24.7% level error in every aggregate. Implement the "
                f"distribution-weighted product before enabling health states.")
        w = self._cohort_weights(t)
        frac = 0.0
        for j in range(int(self.T)):
            sched = self._cohort_survival_schedule(int(t) - j)
            if sched is None or j == 0:
                S = 1.0
            else:
                S = float(np.prod(np.mean(np.asarray(sched), axis=1)[:j]))
            frac += float(w[j]) * S
        cache[key] = frac if frac > 0 else 1.0
        return cache[key]

    def _aggregation_weights(self, t):
        """Cohort weights scaled so aggregates come out per person alive.

        The defining property is sum_j w_j S_j == 1: the implied living
        population is one, so a weighted sum of per-cohort means (which carry
        zeros for the dead) is a per-capita average.
        """
        return np.asarray(self._cohort_weights(t), dtype=float) / self._alive_fraction(t)

    def _survival_schedule_at_year(self, cal_year):
        """Return the survival-prob age profile (T, n_h) for a given internal-clock year.

        `cal_year` is the birth_year-anchored clock used by `_cohort_survival_schedule`
        (cal_year = birth_year + birth_period + j), NOT the true calendar year.

        Data path (`survival_table` set): convert to the true calendar year
        true_cal = cal_year + (current_year - birth_year), clamp to the data range,
        and return that period life table. This is cohort-historical for past years
        and holds at the last data year for the future.

        Legacy path: scale the base schedule by the longevity-improvement factor.
        """
        if self._surv_px is not None:
            true_cal = int(cal_year) + (int(self.current_year) - int(self.birth_year))
            true_cal = int(np.clip(true_cal, int(self._surv_years[0]), int(self._surv_years[-1])))
            idx = int(np.searchsorted(self._surv_years, true_cal))
            return self._surv_px[idx]                       # (T, n_h)
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
        if self._surv_px is None and self.lifecycle_config.survival_probs is None:
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

    def _cohort_retirement_parts(self, birth_period):
        """[(config overrides, share), ...] fixing the retirement of one birth cohort.

        One part, or two when the cohort is split between consecutive retirement
        ages; the shares sum to one. The cohort with birth period b enters in
        year current_year + b. Entry years outside the table take its nearest
        end, where the retirement age is constant.
        """
        if not self.cohort_retirement:
            return [({}, 1.0)]
        years = sorted(self.cohort_retirement)
        k = int(np.clip(int(self.current_year) + int(birth_period), years[0], years[-1]))
        parts = []
        for J_R, lam, share in self.cohort_retirement[k]:
            kw = {'retirement_age': int(J_R)}
            if lam is not None:
                kw['pension_avg_weight'] = float(lam)
            parts.append((kw, float(share)))
        return parts

    def solve_cohort_problems(self, r_path, w_path,
                          tau_c_path=None, tau_l_path=None,
                          tau_p_path=None, tau_k_path=None,
                          pension_replacement_path=None,
                          bequest_lumpsum_path=None,
                          pre_transition_paths=None,
                          verbose=False,
                          lump_sum_path=None,
                          unemployment_index_path=None,
                          kappa_path=None,
                          m_scale_path=None,
                          shock_period=0):
        """
        Solve lifecycle problems for all cohorts given full price paths.

        kappa_path, m_scale_path : health coverage and the multiplier on the
        level of medical spending by calendar period (full length, like
        lump_sum_path); None keeps the configured scalar kappa and m.

        shock_period : the period t_s in which the counterfactual paths become
        known (MIT shock). Every cohort alive at t_s keeps the policies of the
        baseline (pre_transition_paths) at the ages it lived before t_s and is
        re-solved from age t_s - birth_period on. 0 is the shock at the start
        of the transition.
        
        Key indexing:
        - r_path, w_path, etc. are indexed by CALENDAR TIME (0 to T_transition + T - 1)
        - Each cohort born at calendar time t faces prices from t to t+T-1
        - Policy functions are indexed by LIFECYCLE AGE (0 to T-1)
        """
        # Each call produces new policy functions → bump the version counter so that
        # _ensure_cohort_panel_cache() treats the upcoming simulation as distinct.
        self._policy_version = getattr(self, '_policy_version', 0) + 1

        if verbose:
            print("\nSolving cohort lifecycle problems with perfect foresight...")
            print(f"  Education types: {list(self.education_shares.keys())}")

        # Feature flags from lifecycle_config — forwarded to all per-cohort configs
        _lc = self.lifecycle_config
        _feature_kwargs = dict(
            pension_min_floor=_lc.pension_min_floor,
            pension_floor_indexed=_lc.pension_floor_indexed,
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
            self._surv_px is not None or
            (self.survival_improvement_rate != 0.0 and
             self.lifecycle_config.survival_probs is not None)
        )

        # --- SOLVE FOR UNIQUE BIRTH COHORTS ---
        birth_cohort_solutions = {}
        birth_cohort_later = {}      # {edu_type: {bp: model}} — the later-retiring part
        later_share = {}             # {bp: share of the cohort retiring a year later}
        _mit_baseline_to_solve = {}  # {edu_type: {bp: model}} — JAX batch-solve deferred
        _mit_later_to_solve = {}     # the same for the later-retiring parts

        if verbose:
            print("\n  Solving for unique birth cohorts...")

        # Safety: ensure MIT baseline cache exists (handles standalone calls).
        if not hasattr(self, '_mit_baseline_cache'):
            self._mit_baseline_cache = {}

        # Helper: get baseline (steady-state) scalar for a given instrument key.
        # Returns None when pre_transition_paths is not set (preserves old behaviour).
        def _pv(key):
            if pre_transition_paths is None:
                return None
            arr = pre_transition_paths.get(key)
            return float(arr[0]) if arr is not None else None

        # Pre-compute extended baseline paths for MIT shock stitching.
        # These represent the FULL baseline lifecycle path (all ages at baseline values),
        # needed to solve the pure-baseline model that supplies pre-transition policy functions.
        if pre_transition_paths is not None:
            _base_tau_c_ext = _extend_path(pre_transition_paths.get('tau_c_path'), self.T)
            _base_tau_l_ext = _extend_path(pre_transition_paths.get('tau_l_path'), self.T)
            _base_tau_p_ext = _extend_path(pre_transition_paths.get('tau_p_path'), self.T)
            _base_tau_k_ext = _extend_path(pre_transition_paths.get('tau_k_path'), self.T)
            _base_pension_ext = _extend_path(
                pre_transition_paths.get('pension_replacement_path'), self.T)
            _base_r_ext = _extend_path(pre_transition_paths.get('r_path'), self.T)
            _base_w_ext = _extend_path(pre_transition_paths.get('w_path'), self.T)
            _base_ls_ext = _extend_path(pre_transition_paths.get('lump_sum_path'), self.T)
            _base_kappa_ext = _extend_path(pre_transition_paths.get('kappa_path'), self.T)
            _base_ms_ext = _extend_path(pre_transition_paths.get('m_scale_path'), self.T)
            if _base_w_ext is None or _base_r_ext is None:
                warnings.warn(
                    "pre_transition_paths has no 'w_path' or 'r_path': the MIT "
                    "baseline models fall back to the counterfactual prices, so "
                    "A[0] is not predetermined for a shock that moves them.",
                    RuntimeWarning, stacklevel=2)
        else:
            _base_tau_c_ext = _base_tau_l_ext = _base_tau_p_ext = \
                _base_tau_k_ext = _base_pension_ext = _base_r_ext = _base_w_ext = None
            _base_ls_ext = _base_kappa_ext = _base_ms_ext = None
        shock_period = int(shock_period)
        if shock_period < 0:
            raise ValueError(f"shock_period = {shock_period} < 0")
        if shock_period > 0 and pre_transition_paths is None:
            raise ValueError("shock_period > 0 needs pre_transition_paths (the baseline "
                             "that households follow before the shock)")
        # The unemployment index over the cohorts' horizon (one before t = 0),
        # the same in the baseline and in every counterfactual.
        _unemp_full = (self._as_period_path(unemployment_index_path, self.T_transition + self.T)
                       if unemployment_index_path is not None else None)

        # Define the range of birth cohorts we need to solve for
        min_birth_period = 1 - self.T  # Oldest cohort alive at t=0
        max_birth_period = self.T_transition - 1  # Last cohort born during transition

        for edu_type in self.education_shares.keys():
            birth_cohort_solutions[edu_type] = {}
            birth_cohort_later[edu_type] = {}

            for birth_period in range(min_birth_period, max_birth_period + 1):
                if verbose and birth_period % 10 == 0:
                    print(f"    Solving for cohort born at t={birth_period}...")

                # Extract cohort-specific price/policy paths.
                # When pre_transition_paths is provided (MIT shock convention),
                # pre-transition years are padded with baseline values so that
                # K_0 is identical across all counterfactual scenarios.
                cohort_r = _extract_cohort_path(r_path, birth_period, self.T, pre_value=_pv('r_path'))
                cohort_w = _extract_cohort_path(w_path, birth_period, self.T, pre_value=_pv('w_path'))
                cohort_tau_c = _extract_cohort_path(tau_c_path, birth_period, self.T, default=0.0, pre_value=_pv('tau_c_path'))
                cohort_tau_l = _extract_cohort_path(tau_l_path, birth_period, self.T, default=0.0, pre_value=_pv('tau_l_path'))
                cohort_tau_p = _extract_cohort_path(tau_p_path, birth_period, self.T, default=0.0, pre_value=_pv('tau_p_path'))
                cohort_tau_k = _extract_cohort_path(tau_k_path, birth_period, self.T, default=0.0, pre_value=_pv('tau_k_path'))
                cohort_pension = _extract_cohort_path(pension_replacement_path, birth_period, self.T, default=0.4, pre_value=_pv('pension_replacement_path'))
                cohort_ls = _extract_cohort_path(lump_sum_path, birth_period, self.T, default=0.0, pre_value=_pv('lump_sum_path'))
                cohort_kappa = (None if kappa_path is None else _extract_cohort_path(
                    kappa_path, birth_period, self.T, pre_value=_pv('kappa_path')))
                cohort_ms = (None if m_scale_path is None else _extract_cohort_path(
                    m_scale_path, birth_period, self.T, pre_value=_pv('m_scale_path')))
                u_rates, cohort_P_y = self._cohort_unemployment(edu_type, birth_period, _unemp_full)

                # A cohort split between two retirement ages solves both problems;
                # part 0 retires first, part 1 a year later with share `share`.
                for part, (ret_kw, share) in enumerate(
                        self._cohort_retirement_parts(birth_period)):
                    # Create and solve the model for this birth cohort
                    cohort_feature_kwargs = dict(_feature_kwargs)
                    cohort_feature_kwargs.update(ret_kw)
                    if cohort_P_y is not None:
                        cohort_feature_kwargs['P_y_by_age_health'] = cohort_P_y
                        cohort_feature_kwargs['edu_params'] = self._edu_params_at_entry(
                            edu_type, u_rates[0])
                    if _use_per_cohort_survival:
                        cohort_surv = self._cohort_survival_schedule(birth_period)
                        cohort_feature_kwargs['survival_probs'] = cohort_surv
                    bequest_ls = (
                        float(bequest_lumpsum_path[birth_period])
                        if bequest_lumpsum_path is not None and birth_period in bequest_lumpsum_path
                        else 0.0
                    )
                    # Cohort config: _replace preserves all fields on lifecycle_config
                    # (edu_params, n_alpha, wage_age_profile, kappa, m_good, ...);
                    # cohort_feature_kwargs only carries fields possibly overridden by
                    # per-cohort survival (transfer_floor mutation, etc.).
                    config = self.lifecycle_config._replace(
                        education_type=edu_type, current_age=0,
                        r_path=cohort_r, w_path=cohort_w,
                        tau_c_path=cohort_tau_c, tau_l_path=cohort_tau_l,
                        tau_p_path=cohort_tau_p, tau_k_path=cohort_tau_k,
                        pension_replacement_path=cohort_pension,
                        bequest_lumpsum=bequest_ls,
                        lump_sum_path=cohort_ls,
                        kappa_path=cohort_kappa,
                        m_scale_path=cohort_ms,
                        **cohort_feature_kwargs,
                    )
                
                    model = self._lifecycle_model_class(config, verbose=False)
                    if self.backend != 'jax':
                        model.solve(verbose=False)

                    # MIT shock stitching: pre-transition policy functions must equal baseline.
                    # Even with baseline-padded paths, backward induction propagates the
                    # post-t=0 counterfactual tax into ages 0…pre-1.  Fix: solve a pure
                    # baseline lifecycle model (NumPy, cached) and copy its policy functions
                    # for ages 0…pre-1 so that simulated assets at t=0 equal the baseline.
                    if pre_transition_paths is not None and birth_period < shock_period:
                        pre = shock_period - birth_period
                        bcs_key = ((edu_type, birth_period) if part == 0
                                   else (edu_type, birth_period, 'later'))
                        if bcs_key not in self._mit_baseline_cache:
                            base_cohort_tau_c = _extract_cohort_path(
                                _base_tau_c_ext, birth_period, self.T, default=0.0)
                            base_cohort_tau_l = _extract_cohort_path(
                                _base_tau_l_ext, birth_period, self.T, default=0.0)
                            base_cohort_tau_p = _extract_cohort_path(
                                _base_tau_p_ext, birth_period, self.T, default=0.0)
                            base_cohort_tau_k = _extract_cohort_path(
                                _base_tau_k_ext, birth_period, self.T, default=0.0)
                            base_cohort_pension = _extract_cohort_path(
                                _base_pension_ext, birth_period, self.T, default=0.4)
                            base_cohort_r = _extract_cohort_path(
                                _base_r_ext if _base_r_ext is not None else r_path,
                                birth_period, self.T)
                            base_cohort_w = _extract_cohort_path(
                                _base_w_ext if _base_w_ext is not None else w_path,
                                birth_period, self.T)
                            base_cohort_ls = _extract_cohort_path(
                                _base_ls_ext if _base_ls_ext is not None else lump_sum_path,
                                birth_period, self.T, default=0.0)
                            # Coverage and medical spending of the baseline; the
                            # configured scalars when the baseline has no path.
                            base_cohort_kappa = (None if _base_kappa_ext is None else
                                                 _extract_cohort_path(_base_kappa_ext,
                                                                      birth_period, self.T))
                            base_cohort_ms = (None if _base_ms_ext is None else
                                              _extract_cohort_path(_base_ms_ext,
                                                                   birth_period, self.T))
                            # MIT baseline must use baseline feature values, not the
                            # (possibly mutated) counterfactual ones.  Currently only
                            # transfer_floor can be mutated on lifecycle_config by
                            # simulate_transition(); restore it from pre_transition_paths.
                            base_feature_kwargs = dict(cohort_feature_kwargs)
                            _base_tf = pre_transition_paths.get('transfer_floor')
                            if _base_tf is not None:
                                base_feature_kwargs['transfer_floor'] = float(_base_tf)
                            # MIT baseline config: _replace also preserves edu_params,
                            # n_alpha, etc. base_feature_kwargs carries the
                            # transfer_floor override for the MIT baseline case.
                            base_config = self.lifecycle_config._replace(
                                education_type=edu_type, current_age=0,
                                r_path=base_cohort_r, w_path=base_cohort_w,
                                tau_c_path=base_cohort_tau_c, tau_l_path=base_cohort_tau_l,
                                tau_p_path=base_cohort_tau_p, tau_k_path=base_cohort_tau_k,
                                pension_replacement_path=base_cohort_pension,
                                bequest_lumpsum=bequest_ls,
                                lump_sum_path=base_cohort_ls,
                                kappa_path=base_cohort_kappa,
                                m_scale_path=base_cohort_ms,
                                **base_feature_kwargs,
                            )
                            if self.backend == 'jax':
                                # Defer to JAX batch-solve (collected, solved after the loop)
                                base_model = self._lifecycle_model_class(base_config, verbose=False)
                                (_mit_baseline_to_solve if part == 0 else _mit_later_to_solve) \
                                    .setdefault(edu_type, {})[birth_period] = base_model
                            else:
                                base_model = LifecycleModelPerfectForesight(base_config, verbose=False)
                                base_model.solve(verbose=False)
                                self._mit_baseline_cache[bcs_key] = base_model
                        # Stitch: overwrite pre-transition ages with baseline policy arrays.
                        # (JAX stitching happens post batch-solve at lines below;
                        #  NumPy stitching happens here immediately.)
                        if self.backend != 'jax' and bcs_key in self._mit_baseline_cache:
                            base_model = self._mit_baseline_cache[bcs_key]
                            # Stitch the per-alpha arrays too: both simulate paths
                            # read *_policy_alpha (the scalar arrays alias alpha=0
                            # only), so stitching the scalars alone never reaches
                            # the simulation.
                            for attr in ('a_policy', 'c_policy', 'l_policy',
                                         'a_policy_alpha', 'c_policy_alpha', 'l_policy_alpha'):
                                base_arr = getattr(base_model, attr, None)
                                cf_arr   = getattr(model, attr, None)
                                if base_arr is None or cf_arr is None:
                                    continue
                                arr = np.asarray(cf_arr).copy()
                                if attr.endswith('_alpha'):
                                    arr[:, :pre] = np.asarray(base_arr)[:, :pre]
                                else:
                                    arr[:pre] = np.asarray(base_arr)[:pre]
                                setattr(model, attr, arr)

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
                            print("       ⚠️  WARNING: Near-zero savings for this cohort!")

                    if part == 0:
                        birth_cohort_solutions[edu_type][birth_period] = model
                    else:
                        birth_cohort_later[edu_type][birth_period] = model
                        later_share[birth_period] = share

        # JAX batched solve: MIT baseline models (deferred from the loop above)
        if self.backend == 'jax':
            for todo, tag in ((_mit_baseline_to_solve, ()), (_mit_later_to_solve, ('later',))):
                if not todo:
                    continue
                if verbose:
                    print(f"  JAX batched solve: {sum(len(d) for d in todo.values())} "
                          f"MIT baseline models{' (later-retiring parts)' if tag else ''}...")
                self._solve_cohorts_jax_batched(todo, verbose=False)
                for edu_mit, models_mit in todo.items():
                    for bp_mit, model_mit in models_mit.items():
                        self._mit_baseline_cache[(edu_mit, bp_mit) + tag] = model_mit

        # JAX batched solve: all cohorts in one vmapped XLA call per education
        # type and retirement group, then the later-retiring parts
        if self.backend == 'jax':
            self._solve_cohorts_jax_batched(birth_cohort_solutions, verbose)
            if any(birth_cohort_later.values()):
                self._solve_cohorts_jax_batched(birth_cohort_later, verbose)

        # MIT shock stitching for JAX backend (post batch-solve).
        # Baseline models were built during the loop above; on the JAX backend they
        # are LifecycleModelJAX instances deferred to _solve_cohorts_jax_batched.
        if self.backend == 'jax' and pre_transition_paths is not None:
            stitch = [((e, bp), birth_cohort_solutions[e][bp])
                      for e in self.education_shares
                      for bp in range(min_birth_period, min(shock_period, max_birth_period + 1))]
            stitch += [((e, bp, 'later'), m) for e in self.education_shares
                       for bp, m in birth_cohort_later[e].items() if bp < shock_period]
            for bcs_key, jax_m in stitch:
                if bcs_key not in self._mit_baseline_cache:
                    raise RuntimeError(
                        f"no baseline model for cohort {bcs_key}: its ages before the "
                        f"shock (t_s = {shock_period}) cannot be held at the baseline")
                base_m = self._mit_baseline_cache[bcs_key]
                pre = shock_period - bcs_key[1]
                # Stitch the per-alpha arrays too: the batched simulate
                # reads *_policy_alpha, so stitching the scalar arrays
                # alone never reaches the simulation.
                for attr in ('a_policy', 'c_policy', 'l_policy',
                             'a_policy_alpha', 'c_policy_alpha', 'l_policy_alpha'):
                    base_arr = getattr(base_m, attr, None)
                    jax_arr  = getattr(jax_m, attr, None)
                    if base_arr is None or jax_arr is None:
                        continue
                    if self.jax_policies_on_device:
                        import jax.numpy as jnp
                        ages = ((slice(None), slice(None, pre)) if attr.endswith('_alpha')
                                else (slice(None, pre),))
                        arr = jnp.asarray(jax_arr).at[ages].set(jnp.asarray(base_arr)[ages])
                    else:
                        arr = np.asarray(jax_arr).copy()
                        if attr.endswith('_alpha'):
                            arr[:, :pre] = np.asarray(base_arr)[:, :pre]
                        else:
                            arr[:pre] = np.asarray(base_arr)[:pre]
                    setattr(jax_m, attr, arr)
        # Store birth cohort solutions for later cohort-level simulation/slicing
        self.birth_cohort_solutions = birth_cohort_solutions
        self.birth_cohort_later = birth_cohort_later
        self.later_share = later_share

        # --- INITIAL CONDITIONS FOR OLD COHORTS ---
        # Old cohorts (birth_period < 0) simulate from age 0 with a=0 and avg_earnings=0,
        # exactly like new cohorts.  Their price path is padded with r_path[0] for ages
        # 0…(k-1), so the correct SS policy functions apply during those years and the
        # cohort arrives at age k (= calendar t=0) in the proper steady-state distribution.
        # Setting initial conditions to SS values at age k (as was done previously) placed
        # age-k wealth at age 0 of the simulation — wrong initial state, wrong trajectory,
        # and constant aggregate cross-sections because all cohorts looked like the SS.
        if verbose:
            print("\n  Setting initial conditions for pre-transition cohorts...")

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

    def _cohort_age_means(self, n_sim, seed_base, verbose=False, solutions=None):
        """{edu_type: {birth_period: N_AGE_MEANS-tuple of (T,) age means}} for
        the cohort models in *solutions* (default birth_cohort_solutions), by
        exact aggregation or by simulation on the configured backend."""
        education_types = list(self.education_shares.keys())
        min_birth_period = -(self.T - 1)
        max_birth_period = self.T_transition - 1
        if self.aggregation == 'exact':
            panels = self._exact_cohort_age_means(verbose, solutions=solutions)
        elif self.backend == 'jax':
            agent_batch = self.sim_agent_batch_size
            n_agent_batches = max(1, (n_sim + agent_batch - 1) // agent_batch)
            if n_agent_batches == 1:
                panels = self._simulate_cohorts_jax_batched(int(n_sim), int(seed_base), verbose,
                                                            age_means=True, solutions=solutions)
            else:
                edu_types_ab = list(self.education_shares.keys())
                birth_periods_ab = (list(range(min_birth_period, max_birth_period + 1))
                                    if solutions is None else
                                    sorted({b for d in solutions.values() for b in d}))
                sums = {edu: {b: [np.zeros(self.T) for _ in range(N_AGE_MEANS)]
                              for b in birth_periods_ab}
                        for edu in edu_types_ab}
                agents_done = 0
                for ab_idx in range(n_agent_batches):
                    n_ab = min(agent_batch, n_sim - agents_done)
                    ab_seed = int((seed_base + ab_idx * 999983) & 0xFFFFFFFF)
                    raw_b = self._simulate_cohorts_jax_batched(
                        n_ab, ab_seed, verbose=(verbose and ab_idx == 0), age_means=True,
                        solutions=solutions)
                    for edu in edu_types_ab:
                        for b, bm in raw_b.get(edu, {}).items():
                            for k in range(N_AGE_MEANS):
                                sums[edu][b][k] += bm[k] * n_ab
                    # Free cached init conditions to bound memory
                    if hasattr(self, '_sim_init_cache'):
                        self._sim_init_cache = {}
                    agents_done += n_ab
                panels = {edu: {b: tuple(sums[edu][b][k] / n_sim for k in range(N_AGE_MEANS))
                                for b in birth_periods_ab
                                if solutions is None or b in solutions.get(edu, {})}
                          for edu in edu_types_ab}
        else:
            panels = {edu_type: {} for edu_type in education_types}

            for edu_idx, edu_type in enumerate(education_types):
                bps = (range(min_birth_period, max_birth_period + 1) if solutions is None
                       else sorted(solutions.get(edu_type, {})))
                for b in bps:
                    seed = self._crn_seed(edu_idx=edu_idx, birth_period=int(b), base=int(seed_base))
                    panels[edu_type][int(b)] = self._simulate_birth_cohort_cached(
                        edu_type=edu_type,
                        birth_period=int(b),
                        n_sim=int(n_sim),
                        seed=int(seed),
                        later=solutions is not None,
                    )

        return panels

    def _ensure_cohort_panel_cache(self, n_sim: Optional[int] = None, seed_base: int = 42, verbose: bool = False):
        """
        Precompute (once) all cohort Monte Carlo panels needed to build any (t,age) slice
        during the transition. This eliminates repeated cohort simulations inside
        per-period aggregation routines.

        Cache key: (n_sim, seed_base, _policy_version); the cached object is a
        dict[edu_type][birth_period] of 11-tuples of per-age MEAN arrays, not panels.
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

        pv = getattr(self, '_policy_version', 0)
        cache_key = (int(n_sim), int(seed_base), int(pv))
        if cache_key in self._cohort_panel_cache:
            return

        education_types = list(self.education_shares.keys())
        min_birth_period = -(self.T - 1)                 # cohorts already alive at t=0
        max_birth_period = self.T_transition - 1          # cohorts born during transition

        if verbose:
            print(f"Precomputing cohort panels for birth_period in [{min_birth_period}, {max_birth_period}] "
                  f"(n_sim={n_sim}, seed_base={seed_base}) ...")

        panels = self._cohort_age_means(n_sim, seed_base, verbose)
        # A cohort split between two retirement ages: its age means are the
        # mixture of its two parts' means, weighted by their shares.
        later = getattr(self, 'birth_cohort_later', None) or {}
        if any(later.values()):
            later_means = self._cohort_age_means(n_sim, seed_base, verbose, solutions=later)
            for edu, by_bp in later_means.items():
                for b, q in by_bp.items():
                    w = float(self.later_share[b])
                    panels[edu][b] = tuple((1.0 - w) * np.asarray(p_) + w * np.asarray(q_)
                                           for p_, q_ in zip(panels[edu][b], q))

        self._cohort_panel_cache[cache_key] = panels
        # Evict entries with a different policy_version to bound memory use
        stale_keys = [k for k in self._cohort_panel_cache if k[2] != pv]
        for k in stale_keys:
            del self._cohort_panel_cache[k]

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
        pv = getattr(self, '_policy_version', 0)
        return self._cohort_panel_cache[(int(n_sim), int(seed_base), int(pv))][edu_type][int(birth_period)]

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
        # Same policy-version key as the writer in _compute_all_cross_sections:
        # a re-solve within one simulate_transition must not be served the
        # previous policy's cross-sections.
        key = (t, n_sim, int(seed_base), getattr(self, '_policy_version', 0))

        if key in self._period_cache:
            return self._period_cache[key]

        # Ensure we have all cohort panels in memory for this (n_sim, seed_base)
        self._ensure_cohort_panel_cache(n_sim=n_sim, seed_base=seed_base, verbose=False)

        education_types = list(self.education_shares.keys())
        n_edu = len(education_types)
        education_shares_array = np.array([self.education_shares[edu] for edu in education_types], dtype=float)
        cohort_sizes_t = self._aggregation_weights(t)
        # Mortality is already encoded in simulation arrays: dead agents have zero assets/income
        # in all periods after death (NumPy: loop skips dead agents; JAX: jnp.where(alive, val, 0.0)),
        # so the weights are sizes at entry and must NOT be multiplied by cumulative survival.
        # _aggregation_weights divides them by the living share instead, which makes the result
        # per person alive rather than per person ever entered.

        assets_by_age_edu      = np.zeros((n_edu, self.T), dtype=float)
        consumption_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        labor_by_age_edu       = np.zeros((n_edu, self.T), dtype=float)

        tax_c_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_l_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_p_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        tax_k_by_age_edu = np.zeros((n_edu, self.T), dtype=float)

        ui_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        pension_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        gov_health_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        bequest_by_age_edu = np.zeros((n_edu, self.T), dtype=float)
        transfer_by_age_edu = np.zeros((n_edu, self.T), dtype=float)

        pv = getattr(self, '_policy_version', 0)
        panels = self._cohort_panel_cache[(n_sim, int(seed_base), int(pv))]

        for edu_idx, edu_type in enumerate(education_types):
            edu_panels = panels[edu_type]
            for age in range(self.T):
                birth_period = t - age

                panel_data = edu_panels[int(birth_period)]
                if len(panel_data) == N_AGE_MEANS:
                    # Means format: 12-tuple of (T,) per-age mean arrays
                    (a_mean, c_mean, labor_mean,
                     tax_c_mean, tax_l_mean, tax_p_mean, tax_k_mean,
                     ui_mean, pension_mean, gov_health_mean,
                     bequest_mean, transfer_mean) = (float(panel_data[k][age]) for k in range(N_AGE_MEANS))
                else:
                    if len(panel_data) >= 21:
                        # 21-tuple (pre Phase-8) and 22-tuple (Phase 8+, with
                        # trailing alpha_idx_sim) handled identically here
                        (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
                         ui_sim, m_sim, oop_m_sim, gov_m_sim,
                         tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                         pension_sim, retired_sim, l_sim, alive_sim, bequest_sim, *_) = panel_data
                    else:
                        (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
                         ui_sim, m_sim, oop_m_sim, gov_m_sim,
                         tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                         pension_sim, retired_sim, l_sim) = panel_data
                        bequest_sim = np.zeros_like(a_sim)

                    (a_mean, c_mean, labor_mean,
                     tax_c_mean, tax_l_mean, tax_p_mean, tax_k_mean,
                     ui_mean, pension_mean, gov_health_mean) = self._slice_mean_single_age_njit(
                        a_sim, c_sim, effective_y_sim,
                        tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim,
                        ui_sim, pension_sim, gov_m_sim,
                        int(age)
                    )
                    bequest_mean = float(np.mean(bequest_sim[age, :]))
                    transfer_mean = (float(np.mean(panel_data[22][age, :]))
                                     if len(panel_data) >= 23 else 0.0)

                assets_by_age_edu[edu_idx, age]      = a_mean
                consumption_by_age_edu[edu_idx, age] = c_mean
                labor_by_age_edu[edu_idx, age]        = labor_mean

                tax_c_by_age_edu[edu_idx, age] = tax_c_mean
                tax_l_by_age_edu[edu_idx, age] = tax_l_mean
                tax_p_by_age_edu[edu_idx, age] = tax_p_mean
                tax_k_by_age_edu[edu_idx, age] = tax_k_mean

                ui_by_age_edu[edu_idx, age] = ui_mean
                pension_by_age_edu[edu_idx, age] = pension_mean
                gov_health_by_age_edu[edu_idx, age] = gov_health_mean
                bequest_by_age_edu[edu_idx, age] = bequest_mean
                transfer_by_age_edu[edu_idx, age] = transfer_mean

        out = {
            "education_types": education_types,
            "education_shares_array": education_shares_array,
            "cohort_sizes_t": cohort_sizes_t,
            "assets_by_age_edu": assets_by_age_edu,
            "consumption_by_age_edu": consumption_by_age_edu,
            "labor_by_age_edu": labor_by_age_edu,
            "tax_c_by_age_edu": tax_c_by_age_edu,
            "tax_l_by_age_edu": tax_l_by_age_edu,
            "tax_p_by_age_edu": tax_p_by_age_edu,
            "tax_k_by_age_edu": tax_k_by_age_edu,
            "ui_by_age_edu": ui_by_age_edu,
            "pension_by_age_edu": pension_by_age_edu,
            "gov_health_by_age_edu": gov_health_by_age_edu,
            "bequest_by_age_edu": bequest_by_age_edu,
            "transfer_by_age_edu": transfer_by_age_edu,
        }
        self._period_cache[key] = out
        return out

    def _compute_all_cross_sections(self, n_sim: int):
        """
        Compute per-(edu,age) means for ALL T_transition periods in one pass.

        Iterates (edu_type, age) in the outer loops — one panel lookup per cohort —
        then writes the result into every transition period t for which the cohort is alive.
        This is more cache-friendly than the per-period loop in simulate_transition(),
        which repeated the panel lookup T times per cohort.

        Returns (T_transition, n_edu, T) arrays in the order assets, consumption,
        labor -- matching the return statement, NOT the historical docstring order.
        Following the old wording swapped C and L, which is a bug this repo has
        already had once. plus
        budget components, matching what _period_cross_section() computes individually.
        """
        T_tr = int(self.T_transition)
        education_types = list(self.education_shares.keys())
        n_edu = len(education_types)
        T = int(self.T)
        seed_base = 42
        n_sim = int(n_sim)
        pv = getattr(self, '_policy_version', 0)
        panels = self._cohort_panel_cache[(n_sim, int(seed_base), int(pv))]

        # Allocate output arrays
        assets  = np.zeros((T_tr, n_edu, T), dtype=float)
        consum  = np.zeros((T_tr, n_edu, T), dtype=float)
        labor   = np.zeros((T_tr, n_edu, T), dtype=float)
        tax_c   = np.zeros((T_tr, n_edu, T), dtype=float)
        tax_l   = np.zeros((T_tr, n_edu, T), dtype=float)
        tax_p   = np.zeros((T_tr, n_edu, T), dtype=float)
        tax_k   = np.zeros((T_tr, n_edu, T), dtype=float)
        ui_arr  = np.zeros((T_tr, n_edu, T), dtype=float)
        pension = np.zeros((T_tr, n_edu, T), dtype=float)
        gov_h   = np.zeros((T_tr, n_edu, T), dtype=float)
        bequest = np.zeros((T_tr, n_edu, T), dtype=float)
        transf  = np.zeros((T_tr, n_edu, T), dtype=float)

        min_birth = -(T - 1)
        max_birth = T_tr - 1

        for edu_idx, edu_type in enumerate(education_types):
            edu_panels = panels[edu_type]
            for age in range(T):
                for t in range(T_tr):
                    b = t - age
                    if b < min_birth or b > max_birth:
                        continue
                    if b not in edu_panels:
                        continue
                    panel_data = edu_panels[int(b)]
                    if len(panel_data) == N_AGE_MEANS:
                        # Means format: 12-tuple of (T,) per-age mean arrays
                        (a_m, c_m, l_m, tc_m, tl_m, tp_m, tk_m,
                         ui_m, pen_m, gh_m, beq_m, tr_m) = (
                            float(panel_data[k][age]) for k in range(N_AGE_MEANS))
                    else:
                        if len(panel_data) >= 21:
                            (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
                             ui_sim, m_sim, oop_m_sim, gov_m_sim,
                             tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                             pension_sim, retired_sim, l_sim, alive_sim, bequest_sim, *_) = panel_data
                        else:
                            (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
                             ui_sim, m_sim, oop_m_sim, gov_m_sim,
                             tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                             pension_sim, retired_sim, l_sim) = panel_data
                            bequest_sim = np.zeros_like(a_sim)

                        (a_m, c_m, l_m, tc_m, tl_m, tp_m, tk_m,
                         ui_m, pen_m, gh_m) = self._slice_mean_single_age_njit(
                            a_sim, c_sim, effective_y_sim,
                            tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim,
                            ui_sim, pension_sim, gov_m_sim,
                            int(age),
                        )
                        beq_m = float(np.mean(bequest_sim[age, :]))
                        tr_m = (float(np.mean(panel_data[22][age, :]))
                                if len(panel_data) >= 23 else 0.0)

                    assets[t, edu_idx, age]  = a_m
                    consum[t, edu_idx, age]  = c_m
                    labor[t, edu_idx, age]   = l_m
                    tax_c[t, edu_idx, age]   = tc_m
                    tax_l[t, edu_idx, age]   = tl_m
                    tax_p[t, edu_idx, age]   = tp_m
                    tax_k[t, edu_idx, age]   = tk_m
                    ui_arr[t, edu_idx, age]  = ui_m
                    pension[t, edu_idx, age] = pen_m
                    gov_h[t, edu_idx, age]   = gh_m
                    bequest[t, edu_idx, age] = beq_m
                    transf[t, edu_idx, age]  = tr_m

        # Populate _period_cache so compute_government_budget_path() can reuse
        # these results instead of recomputing via _period_cross_section().
        education_types = list(self.education_shares.keys())
        education_shares_array = np.array([self.education_shares[e] for e in education_types], dtype=float)
        for t in range(T_tr):
            cohort_sizes_t = self._aggregation_weights(t)
            cache_key = (t, n_sim, seed_base, getattr(self, '_policy_version', 0))
            self._period_cache[cache_key] = {
                "education_types": education_types,
                "education_shares_array": education_shares_array,
                "cohort_sizes_t": cohort_sizes_t,
                "assets_by_age_edu": assets[t],
                "consumption_by_age_edu": consum[t],
                "labor_by_age_edu": labor[t],
                "tax_c_by_age_edu": tax_c[t],
                "tax_l_by_age_edu": tax_l[t],
                "tax_p_by_age_edu": tax_p[t],
                "tax_k_by_age_edu": tax_k[t],
                "ui_by_age_edu": ui_arr[t],
                "pension_by_age_edu": pension[t],
                "gov_health_by_age_edu": gov_h[t],
                "bequest_by_age_edu": bequest[t],
                "transfer_by_age_edu": transf[t],
            }

        return (assets, consum, labor, tax_c, tax_l, tax_p, tax_k, ui_arr, pension,
                gov_h, bequest, transf)

    def compute_aggregates(self, t, n_sim: Optional[int] = None):
        """Compute aggregate household wealth (A), labor (L), consumption (C).

        The return order is (K, L, C) -- the njit it wraps returns (K, C, L), so
        the two differ and the docstring used to state the njit's order. No
        caller remains in the repo. for period t."""
        if n_sim is None:
            if self._last_n_sim is None:
                raise ValueError("n_sim is None and no previous simulate_transition() n_sim is stored.")
            n_sim = int(self._last_n_sim)

        px = self._period_cross_section(t=int(t), n_sim=int(n_sim))

        K, C, L = self._aggregate_capital_labor_njit(
            px["assets_by_age_edu"],
            px["consumption_by_age_edu"],
            px["labor_by_age_edu"],
            px["cohort_sizes_t"],
            px["education_shares_array"],
        )
        # Same units as simulate_transition's L_path: the labor mean is
        # wage-valued (effective_y_sim = wages + UI), so net out UI and
        # convert to efficiency units.
        if self.w_path is None:
            raise ValueError("w_path is not set — run simulate_transition() first.")
        UI = float(np.sum(px["cohort_sizes_t"][None, :] * px["education_shares_array"][:, None]
                          * px["ui_by_age_edu"]))
        L = (L - UI) / float(np.asarray(self.w_path)[int(t)])
        return K, L, C

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
        total_transfers = 0.0   # means-tested consumption-floor top-ups

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
                if 'transfer_by_age_edu' in px:
                    total_transfers += weight * float(px["transfer_by_age_edu"][edu_idx, age])

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
        def _at(path, default=0.0):
            return float(path[t_idx]) if path is not None and t_idx < len(path) else default

        # GDP-share spending mode: if a ratio is set, level = ratio * Y_path[t]
        # (ratio may be a scalar or a (T,) array). Otherwise read the level path.
        Y_t = _at(self.Y_path)
        def _ratio_at(ratio):
            if ratio is None:
                return None
            if np.isscalar(ratio):
                return float(ratio)
            arr = np.asarray(ratio)
            return float(arr[t_idx]) if t_idx < len(arr) else float(arr[-1])
        def _spend(ratio, active_level, obj_level):
            r = _ratio_at(ratio)
            if r is not None:
                return r * Y_t
            return _at(active_level if active_level is not None else obj_level)

        G_t   = _spend(self._active_G_over_Y, self._active_govt_spending_path, self.govt_spending_path)
        I_g_t = _spend(self._active_I_g_over_Y, self._active_I_g_path, self.I_g_path)
        defense_t = _spend(self._active_defense_over_Y, self._active_defense_spending_path, self.defense_spending_path)
        other_t   = _spend(self._active_other_net_over_Y, self._active_other_net_spending_path, self.other_net_spending_path)
        # Lines added on 2026-10-07: the output tax and the transfer from
        # abroad as revenue, education and the lump-sum transfer as spending.
        tau_y_t = _at(self._active_tau_y_path)
        tax_y_t = tau_y_t * Y_t
        ft_ratio = _ratio_at(self._active_foreign_transfer_over_Y)
        foreign_t = (ft_ratio or 0.0) * Y_t
        lump_t = _at(self._active_lump_sum_path)      # per living person
        education_t = self._education_at(t_idx)

        # Feature #9: Sovereign debt service
        debt_service = 0.0
        new_borrowing = 0.0
        if self.B_path is not None:
            r_debt = float(self.r_B_path[t_idx]) if self.r_B_path is not None else (
                float(self.r_path[t_idx]) if self.r_path is not None else 0.0)
            B_t = float(self.B_path[t_idx]) if t_idx < len(self.B_path) else float(self.B_path[-1])
            B_next = float(self.B_path[t_idx + 1]) if t_idx + 1 < len(self.B_path) else float(self.B_path[-1])
            debt_service = r_debt * B_t
            # In detrended units the stock carried into t+1 is worth
            # Gamma_t times its per-capita value next period.
            new_borrowing = self._growth_at(t_idx) * B_next - B_t

        # The means-tested transfer is an outlay: the simulation records the
        # top-up each household received (transfer_sim) and it is aggregated
        # above, so a positive floor is a closed circuit (since 2026-10-02).
        total_spending = (total_ui + total_pension + total_gov_health + total_transfers
                          + G_t + I_g_t + defense_t + other_t + education_t + lump_t)
        # Memo lines (not outlays): total medical spending and the households'
        # part of it. Every household alive at t faces the coverage of t, so
        # M_t = gov_health_t / kappa_t.
        kappa_t = _at(self._active_kappa_path,
                      default=float(getattr(self.lifecycle_config, 'kappa', 1.0)))
        medical_total = total_gov_health / kappa_t if kappa_t > 0.0 else float('nan')
        oop_health = medical_total - total_gov_health
        total_revenue = total_revenue + tax_y_t + foreign_t
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
            "oop_health": oop_health,
            "medical_total": medical_total,
            "kappa": kappa_t,
            "transfers": total_transfers,
            "govt_spending": G_t,
            "public_investment": I_g_t,
            "defense_spending": defense_t,
            "other_net_spending": other_t,
            "tax_y": tax_y_t,
            "foreign_transfer": foreign_t,
            "education": education_t,
            "lump_sum": lump_t,
            "debt_service": debt_service,
            "new_borrowing": new_borrowing,
            "total_spending": total_spending,
            "primary_deficit": primary_deficit,
            "fiscal_deficit": fiscal_deficit,
            "bequest_tax": bequest_tax_revenue,
            "bequest_transfers": bequest_transfers,
            "total_bequests": total_bequests,
        }

    def _education_at(self, t_idx):
        """Education spending of period t: e_0 Y_ref (w_t / w_0) s_t (the
        constructor's description), zero without a base-year share."""
        e0, index_path, Y_ref = self._active_education
        if not e0:
            return 0.0
        if Y_ref is None:
            Y_ref = float(self.Y_path[0])
        s_t = 1.0
        if index_path is not None:
            arr = np.asarray(index_path, dtype=float)
            s_t = float(arr[t_idx]) if t_idx < len(arr) else float(arr[-1])
        w_t = float(self.w_path[t_idx]) if t_idx < len(self.w_path) else float(self.w_path[-1])
        return float(e0) * Y_ref * (w_t / float(self.w_path[0])) * s_t

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

    def _household_inputs_key(self, paths, bequest_lumpsum_path,
                              pre_transition_paths, n_sim):
        """Digest of the inputs to the cohort solves and simulations of one
        simulate_transition() call.

        Covers the price and policy paths faced by households, the bequest
        receipts, the baseline paths that govern pre-transition ages, the
        lifecycle configuration (including the active transfer floor), n_sim
        and the horizon. Government purchases are not inputs to the household
        side. The demographic and survival tables and the cohort retirement
        table are fixed at construction and are not part of the key.
        """
        h = hashlib.sha1()

        def feed(x):
            if x is None:
                h.update(b'<None>')
            elif isinstance(x, dict):
                for k in sorted(x, key=str):
                    h.update(str(k).encode())
                    feed(x[k])
            else:
                a = np.ascontiguousarray(np.asarray(x, dtype=float))
                h.update(str(a.shape).encode())
                h.update(a.tobytes())

        for path in paths:
            feed(path)
        feed(bequest_lumpsum_path)
        feed(pre_transition_paths)
        h.update(pickle.dumps(self.lifecycle_config))
        h.update(repr((int(n_sim), int(self.T_transition), self.backend,
                       int(self.sim_agent_batch_size), self.aggregation)).encode())
        return h.hexdigest()

    def simulate_transition(self, r_path, w_path=None,
                           tau_c_path=None, tau_l_path=None,
                           tau_p_path=None, tau_k_path=None,
                           pension_replacement_path=None,
                           I_g_path=None, govt_spending_path=None,
                           defense_spending_path=None,
                           other_net_spending_path=None,
                           G_over_Y=None, I_g_over_Y=None,
                           defense_over_Y=None, other_net_over_Y=None,
                           transfer_floor=None,
                           tau_y_path=None, r_B_path=None,
                           lump_sum_path=None,
                           education_over_Y0=None, education_index_path=None,
                           education_Y0=None,
                           foreign_transfer_over_Y=None,
                           unemployment_index_path=None,
                           n_sim=10000, verbose=True,
                           pop_growth_path=None,
                           bequest_lumpsum_path=None,
                           recompute_bequests=False,
                           bequest_tol=1e-4,
                           max_bequest_iters=5,
                           pre_transition_paths=None,
                           kappa_path=None,
                           m_scale_path=None,
                           shock_period=0):
        """
        Simulate transition dynamics with exogenous interest rate path.
        
        With exogenous r, the capital-labor ratio K/L is pinned down by the
        firm's condition with the tax on gross output (firm_conditions.py):
            r + δ = (1 - τ_y) α A K_g^η (K/L)^(α-1)
        
        This determines the wage:
            w = (1 - τ_y)(1-α) A K_g^η (K/L)^α
        
        Parameters
        ----------
        r_path : array_like
            Exogenous interest rate path
        w_path : array_like, optional
            If provided, uses this wage path (for testing)
            If None, computes wage from production function given r
        pop_growth_path : array_like, optional
            If provided, creates time-varying cohort weights (ageing over time).
        kappa_path, m_scale_path : array_like, optional
            Health coverage and the multiplier on the level of medical spending
            by calendar period; None keeps the configured kappa and m.
        shock_period : int
            The period t_s in which households learn the paths of this run
            (an unanticipated shock in t_s; 0 is the start of the transition).
            Cohorts alive at t_s follow the baseline (pre_transition_paths)
            before t_s. Requires pre_transition_paths when positive.
        """
        r_path = np.array(r_path)
        self.T_transition = len(r_path)
        shock_period = int(shock_period)
        if shock_period > 0 and recompute_bequests:
            # Bequests received before t_s would differ from the baseline's.
            raise ValueError("shock_period > 0 is not supported with recompute_bequests=True")
        self._active_shock_period = shock_period
        # Health coverage and the medical-spending multiplier by period
        self._active_kappa_path = (None if kappa_path is None
                                   else self._as_period_path(kappa_path, self.T_transition))
        self._active_m_scale_path = (None if m_scale_path is None
                                     else self._as_period_path(m_scale_path, self.T_transition))
        kappa_path_full = _extend_path(self._active_kappa_path, self.T)
        m_scale_path_full = _extend_path(self._active_m_scale_path, self.T)

        # Demography: Gamma_t and, with a demographic path, the entering-cohort
        # weights. An explicit pop_growth_path argument still overrides below.
        self.growth_factor_path = self.growth_factors(self.T_transition)
        if self._demog is not None:
            self._build_cohort_sizes_from_entrants()

        # Resolve effective paths for this run (explicit args override object attributes)
        _I_g = np.asarray(I_g_path, dtype=float) if I_g_path is not None else self.I_g_path
        _G   = np.asarray(govt_spending_path, dtype=float) if govt_spending_path is not None else self.govt_spending_path
        _defense = (np.asarray(defense_spending_path, dtype=float)
                    if defense_spending_path is not None else self.defense_spending_path)
        _other   = (np.asarray(other_net_spending_path, dtype=float)
                    if other_net_spending_path is not None else self.other_net_spending_path)
        self._active_I_g_path = _I_g
        self._active_govt_spending_path = _G
        self._active_defense_spending_path = _defense
        self._active_other_net_spending_path = _other

        # GDP-share spending mode (level = ratio * Y_path[t], computed in the
        # budget). A set ratio takes precedence over the level path for that line.
        self._active_G_over_Y         = G_over_Y
        self._active_I_g_over_Y       = I_g_over_Y
        self._active_defense_over_Y   = defense_over_Y
        self._active_other_net_over_Y = other_net_over_Y
        # Output tax by period; the lump-sum transfer, the education line and
        # the transfer from abroad (explicit arguments override the object's).
        self._active_tau_y_path = self._as_period_path(
            tau_y_path if tau_y_path is not None else self.tau_y, self.T_transition)
        _ls = lump_sum_path if lump_sum_path is not None else self.lump_sum_path
        if _ls is None:
            # No level path given: the household's configured transfer by age
            # (lump_sum_over_Y times base-year output, constant) applies in
            # every period, so a transition run from the configuration pays
            # the same transfer the calibration cross-section does.
            _cfg_ls = getattr(self.lifecycle_config, 'lump_sum_path', None)
            _cfg_ls = np.zeros(1) if _cfg_ls is None else np.asarray(_cfg_ls, dtype=float)
            if np.any(_cfg_ls != 0.0):
                _ls = float(_cfg_ls[0])
        self._active_lump_sum_path = self._as_period_path(_ls, self.T_transition)
        self._active_foreign_transfer_over_Y = (foreign_transfer_over_Y
                                                if foreign_transfer_over_Y is not None
                                                else self.foreign_transfer_over_Y)
        self._active_education = (
            float(education_over_Y0 if education_over_Y0 is not None else self.education_over_Y0),
            (education_index_path if education_index_path is not None
             else self.education_index_path),
            education_Y0 if education_Y0 is not None else self.education_Y0)
        _unemp = (unemployment_index_path if unemployment_index_path is not None
                  else self.unemployment_index_path)
        # I_g feeds public capital (K_g) BEFORE Y exists, so a GDP-share I_g would
        # create a simultaneity (I_g level needs Y; K_g→Y needs I_g level). Only
        # safe when public capital has no production feedback.
        if I_g_over_Y is not None and self.eta_g != 0.0:
            raise ValueError(
                "I_g_over_Y (GDP-share public investment) is incompatible with "
                "eta_g != 0: I_g feeds K_g feeds Y, requiring a fixed point. "
                "Pass I_g_path as a level instead.")

        # Handle transfer_floor override (used in lifecycle config during cohort solves)
        _orig_tf = None
        if transfer_floor is not None:
            _orig_tf = getattr(self.lifecycle_config, 'transfer_floor', 0.0)
            self.lifecycle_config.transfer_floor = float(transfer_floor)

        # NEW: build time-varying cohort sizes if requested
        if pop_growth_path is not None:
            self.set_cohort_sizes_path_from_pop_growth(pop_growth_path)

        # Feature #8/#10: Compute public capital path before wages (wages depend on K_g)
        K_g_path = None
        if self.eta_g != 0.0 and _I_g is not None:
            K_g_path = np.zeros(self.T_transition)
            K_g_path[0] = self.K_g_initial
            for t in range(1, self.T_transition):
                I_g_t = _I_g[t - 1] if t - 1 < len(_I_g) else _I_g[-1]
                # Per-capita detrended stock: the whole period-(t-1) right-hand
                # side is divided by Gamma_t = (1+g)(1+n_t).
                K_g_path[t] = ((1 - self.delta_g) * K_g_path[t - 1]
                               + I_g_t) / self._growth_at(t - 1)
            if verbose:
                print(f"\nPublic capital path: K_g[0]={K_g_path[0]:.4f} → K_g[-1]={K_g_path[-1]:.4f}")

        # Extend r_path for cohorts born before transition
        r_path_full = np.concatenate([r_path, np.ones(self.T) * r_path[-1]])

        # Sovereign rate path: the configured path by period when there is one
        # (a real rate: the data's to 2025, the Commission's projection to 2060,
        # 2% from 2070), else the scalar r_B broadcast, else the capital path.
        _rb = r_B_path if r_B_path is not None else self.r_B_path_input
        if _rb is not None:
            self.r_B_path = self._as_period_path(_rb, self.T_transition)
        else:
            self.r_B_path = (np.full(self.T_transition, float(self.r_B))
                             if self.r_B is not None
                             else np.asarray(r_path, dtype=float))

        # Compute wage path from production function
        if w_path is None:
            if verbose:
                print("\nComputing wage path from production function...")

            # With public capital: r + δ = α·A·K_g^{η_g}·(K/L)^{α-1}
            K_g_factor = np.ones(self.T_transition)
            if K_g_path is not None and self.eta_g != 0.0:
                K_g_factor = K_g_path ** self.eta_g

            # The firm's conditions with the output tax (firm_conditions.py).
            K_over_L, w_path, _ = firm_conditions(r_path, self.A, K_g_factor, self.alpha,
                                                  self.delta, self._active_tau_y_path)

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
        lump_path_full = _extend_path(self._active_lump_sum_path, self.T)
        
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
        # The living share depends on _cohort_weights, so it must be dropped with
        # them: it was previously keyed on t alone and never cleared, so a second
        # call with different cohort weights reused shares 6-8% wrong.
        self._alive_frac_cache = {}
        # _cohort_panel_cache is intentionally NOT cleared here — it is keyed by
        # (n_sim, seed_base, _policy_version); solve_cohort_problems() increments
        # _policy_version so stale panels are never reused.
        if not hasattr(self, '_cohort_panel_cache'):
            self._cohort_panel_cache = {}

        # MIT shock baseline-solution cache: keyed by (edu_type, birth_period).
        # Valid while pre_transition_paths is the same object across calls (bisection
        # iterations within a single run all share the same pre_tp dict).
        # Invalidated whenever pre_transition_paths changes (new experiment or new run).
        # A call without pre_transition_paths (a baseline run) does not use the
        # cache and leaves it in place: a baseline rerun served from the
        # household cache keeps no cohort models to refill it from.
        if pre_transition_paths is not None:
            _new_pre_tp_id = id(pre_transition_paths)
            if getattr(self, '_mit_pre_tp_id', None) != _new_pre_tp_id:
                self._mit_baseline_cache = {}
                self._mit_pre_tp_id = _new_pre_tp_id

        # Feature A: Bequest redistribution loop (closed-circuit bequests)
        # When recompute_bequests=True and survival_probs is set, iterate until
        # the bequest transfer received by newborns matches the bequests generated
        # by dying agents.  Convergence is fast (1-2 iterations) because bequests
        # are a small fraction of newborn wealth.
        # When survival_probs is None the loop exits after one pass with zero
        # transfers — fully backward compatible.
        _do_bequest_loop = (
            recompute_bequests
            and self.lifecycle_config.survival_probs is not None
        )
        if _do_bequest_loop:
            current_bequest_path = bequest_lumpsum_path  # caller value as initial guess
            self._bequest_converged = False
            self._bequest_iter_count = 0
            if verbose:
                print(f"\nBequest redistribution loop (max {max_bequest_iters} iterations, tol={bequest_tol})...")
            for _bequest_iter in range(max_bequest_iters):
                self._bequest_iter_count = _bequest_iter + 1
                if verbose:
                    print(f"\n  Bequest iteration {self._bequest_iter_count}/{max_bequest_iters}: solving cohort problems...")
                self.solve_cohort_problems(
                    r_path_full, w_path_full,
                    tau_c_path=tau_c_path_full,
                    tau_l_path=tau_l_path_full,
                    tau_p_path=tau_p_path_full,
                    tau_k_path=tau_k_path_full,
                    pension_replacement_path=pension_path_full,
                    bequest_lumpsum_path=current_bequest_path,
                    pre_transition_paths=pre_transition_paths,
                    verbose=False,
                    lump_sum_path=lump_path_full,
                    unemployment_index_path=_unemp,
                    kappa_path=kappa_path_full,
                    m_scale_path=m_scale_path_full,
                    shock_period=shock_period,
                )
                if verbose:
                    print(f"  Bequest iteration {self._bequest_iter_count}/{max_bequest_iters}: simulating panels (n_sim={n_sim})...")
                self._ensure_cohort_panel_cache(n_sim=int(n_sim), seed_base=42, verbose=False)
                new_bequest_path = self._compute_bequest_lumpsum_path(n_sim=int(n_sim))
                old_vals = (
                    np.array([current_bequest_path.get(t, 0.0) for t in range(self.T_transition)])
                    if current_bequest_path is not None
                    else np.zeros(self.T_transition)
                )
                new_vals = np.array([new_bequest_path.get(t, 0.0) for t in range(self.T_transition)])
                max_change = float(np.max(np.abs(new_vals - old_vals)))
                if verbose:
                    print(f"  Bequest iteration {self._bequest_iter_count}/{max_bequest_iters}: max change = {max_change:.2e} (tol={bequest_tol:.2e})")
                if max_change < bequest_tol:
                    current_bequest_path = new_bequest_path
                    self._bequest_converged = True
                    break
                current_bequest_path = new_bequest_path
            # cohort panel cache is up to date from the last loop iteration
            bequest_lumpsum_path = current_bequest_path
            if verbose:
                status = "converged" if self._bequest_converged else "did not converge"
                print(f"\nBequest loop {status} in {self._bequest_iter_count} iteration(s).")
        else:
            self._bequest_converged = True
            self._bequest_iter_count = 0
            # With household_cache_size > 0, a call whose household inputs equal
            # those of an earlier call takes that call's per-cohort age means
            # instead of solving and simulating again; the aggregates and the
            # government budget below are then recomputed from them.
            cache_key = cached_panels = None
            if self.household_cache_size > 0:
                cache_key = self._household_inputs_key(
                    (r_path_full, w_path_full, tau_c_path_full, tau_l_path_full,
                     tau_p_path_full, tau_k_path_full, pension_path_full,
                     lump_path_full, _unemp, kappa_path_full, m_scale_path_full,
                     np.array([shock_period])),
                    bequest_lumpsum_path, pre_transition_paths, n_sim)
                cached_panels = self._household_cache.get(cache_key)
            if cached_panels is not None:
                self._household_cache.move_to_end(cache_key)
                self._household_cache_hits += 1
                self._policy_version = getattr(self, '_policy_version', 0) + 1
                self._cohort_panel_cache = {
                    (int(n_sim), 42, int(self._policy_version)): cached_panels}
                # The cohort models of that call were not kept, so there are no
                # policy functions to read after this one.
                self.birth_cohort_solutions = None
                self.birth_cohort_later = {}
                if verbose:
                    print("\nHousehold inputs unchanged from an earlier call: "
                          "reusing its cohort age means.")
            else:
                # Solve all cohort problems with perfect foresight of r and w
                self.solve_cohort_problems(
                    r_path_full, w_path_full,
                    tau_c_path=tau_c_path_full,
                    tau_l_path=tau_l_path_full,
                    tau_p_path=tau_p_path_full,
                    tau_k_path=tau_k_path_full,
                    pension_replacement_path=pension_path_full,
                    bequest_lumpsum_path=bequest_lumpsum_path,
                    pre_transition_paths=pre_transition_paths,
                    verbose=verbose,
                    lump_sum_path=lump_path_full,
                    unemployment_index_path=_unemp,
                    kappa_path=kappa_path_full,
                    m_scale_path=m_scale_path_full,
                    shock_period=shock_period,
                )
                # Precompute cohort panels ONCE (requires birth_cohort_solutions from solve_cohort_problems)
                self._ensure_cohort_panel_cache(n_sim=int(n_sim), seed_base=42, verbose=verbose)
                if cache_key is not None:
                    self._household_cache[cache_key] = self._cohort_panel_cache[
                        (int(n_sim), 42, int(self._policy_version))]
                    while len(self._household_cache) > self.household_cache_size:
                        self._household_cache.popitem(last=False)

        # Feature #21: Build population weights from fertility + survival.
        # Population weights are births only: per-cohort means divide by n_sim with
        # the dead at zero, so survival is already inside every mean and putting it in
        # the weights too would double-count it. _aggregation_weights then divides the
        # entry weights by the living share, which is what makes the result per living
        # person rather than per person ever entered.
        #
        # There used to be a second mechanism here, _build_population_weights, which
        # built fertility x cumulative survival -- the double count -- and discarded
        # the measured entrant path. It was removed on 2026-10-01: the demographic
        # sidecar supersedes it, since the EUROPOP2023 entrant series IS the fertility
        # path and the projected tables ARE the longevity improvement. A counterfactual
        # fertility path belongs in build_demography_GR.py as an alternative entrant
        # series, not as a parallel weighting rule.

        if verbose:
            print("\nComputing aggregate quantities...")
        
        # Compute aggregates from household decisions
        K_path = np.zeros(self.T_transition)
        C_path = np.zeros(self.T_transition)
        L_path = np.zeros(self.T_transition)

        # Compute all cross-sections in one pass (fewer panel lookups than T_transition × T × n_edu)
        _edu_types_ordered = list(self.education_shares.keys())
        _edu_shares_arr = np.array([self.education_shares[e] for e in _edu_types_ordered], dtype=float)
        (assets_all, consum_all, labor_all,
         _tax_c_all, _tax_l_all, _tax_p_all, _tax_k_all,
         ui_all, _pension_all, _gov_h_all, _bequest_all,
         _transfer_all) = self._compute_all_cross_sections(int(n_sim))

        UI_path = np.zeros(self.T_transition)
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            cohort_sizes_t = self._aggregation_weights(t)
            # njit returns (K, C, L) — keep the unpack order aligned with that return
            K_path[t], C_path[t], L_path[t] = self._aggregate_capital_labor_njit(
                assets_all[t], consum_all[t], labor_all[t], cohort_sizes_t, _edu_shares_arr
            )
            UI_path[t] = float(np.sum(cohort_sizes_t[None, :] * _edu_shares_arr[:, None] * ui_all[t]))

        # L is aggregated from effective_y_sim, which is wage-valued and carries
        # UI as well (w·κ(j)·y·l·exp(α) + UI). UI is a transfer, not labour, so
        # it is netted out before dividing by the wage; the production-function
        # input is then hours in efficiency units -- same convention as
        # calibrate.py since 2026-10-02 (L = (labor_income - ui) / w). Until
        # then the UI stayed in, overstating L, K and Y by UI/(wL).
        L_path = (L_path - UI_path) / w_path

        # Feature #9: Compute K_domestic before Y — Y must use domestic capital, not household wealth.
        # K_path = A = total household wealth (aggregated from simulation).
        # K_domestic = domestic capital demand, pinned by firm's FOC given exogenous r and L.
        # NFA = A - K_domestic  (partial; B is subtracted by fiscal_experiments callers).
        NFA_path = None
        K_domestic = None
        if self.economy_type == 'soe':
            K_g_factor_arr = np.ones(self.T_transition)
            if K_g_path is not None and self.eta_g != 0.0:
                K_g_factor_arr = K_g_path ** self.eta_g
            K_over_L_implied, _, _ = firm_conditions(r_path, self.A, K_g_factor_arr, self.alpha,
                                                     self.delta, self._active_tau_y_path)
            K_domestic = K_over_L_implied * L_path
            NFA_path = K_path - K_domestic   # = A - K_domestic; B not yet subtracted
            if verbose:
                print(f"\n  SOE: NFA[0]={NFA_path[0]:.4f}, NFA[-1]={NFA_path[-1]:.4f}")

        # In SOE, firms hire K_domestic (pinned by FOC); in closed economy K_domestic = A = K_path.
        K_for_Y = K_domestic if K_domestic is not None else K_path

        # Compute output (with public capital if present)
        Y_path = self._compute_output_path_njit(K_for_Y, L_path, self.alpha, self.A,
                                                 K_g_path, self.eta_g)

        # Verify consistency: check if implied r from aggregates matches exogenous r
        if verbose:
            print("\nVerifying consistency with production function...")
            K_g_0 = K_g_path[0] if K_g_path is not None else 1.0
            K_g_end = K_g_path[-1] if K_g_path is not None else 1.0
            r_implied, w_implied = self._marginal_products_njit(
                K_for_Y[0], L_path[0], self.alpha, self.delta, self.A, K_g_0, self.eta_g,
                float(self._active_tau_y_path[0])
            )
            print("  Period 0:")
            print(f"    Exogenous r: {r_path[0]:.4f}, Implied r: {r_implied:.4f}")
            print(f"    Computed w:  {w_path[0]:.4f}, Implied w: {w_implied:.4f}")

            if self.T_transition > 1:
                r_implied_end, w_implied_end = self._marginal_products_njit(
                    K_for_Y[-1], L_path[-1], self.alpha, self.delta, self.A, K_g_end, self.eta_g,
                    float(self._active_tau_y_path[-1])
                )
                print(f"  Period {self.T_transition-1}:")
                print(f"    Exogenous r: {r_path[-1]:.4f}, Implied r: {r_implied_end:.4f}")
                print(f"    Computed w:  {w_path[-1]:.4f}, Implied w: {w_implied_end:.4f}")

        # C_path is aggregated directly from individual consumption decisions above.

        # Store results
        self.r_path = r_path
        self.w_path = w_path
        self.K_path          = K_path      # = A: total household wealth
        self.K_domestic_path = K_domestic  # domestic physical capital (SOE only, else None)
        self.L_path          = L_path
        self.Y_path          = Y_path
        self.C_path          = C_path
        self.K_g_path        = K_g_path
        self.NFA_path        = NFA_path    # = A - K_domestic; B subtracted by fiscal callers

        if verbose:
            print("\n" + "=" * 60)
            print("Transition Simulation Complete")
            print("=" * 60)
            print("\nSummary Statistics:")
            print(f"  Average A (hh wealth): {np.mean(K_path):.4f}")
            print(f"  Average L: {np.mean(L_path):.4f}")
            print(f"  Average Y: {np.mean(Y_path):.4f}")
            print(f"  Average w: {np.mean(w_path):.4f}")
            print(f"  Average A/Y: {np.mean(K_path/Y_path):.4f}")
            print(f"  A range: [{np.min(K_path):.4f}, {np.max(K_path):.4f}]")
            print(f"  L range: [{np.min(L_path):.4f}, {np.max(L_path):.4f}]")
            if K_g_path is not None:
                print(f"  K_g range: [{np.min(K_g_path):.4f}, {np.max(K_g_path):.4f}]")
            if NFA_path is not None:
                print(f"  NFA range: [{np.min(NFA_path):.4f}, {np.max(NFA_path):.4f}]")

        if _orig_tf is not None:
            self.lifecycle_config.transfer_floor = _orig_tf

        # 'K' kept for backward compatibility (= A, household wealth).
        # 'A' is the semantically correct name for total household wealth.
        # 'K_domestic' is the domestic physical capital stock (SOE only).
        result = {'r': self.r_path, 'w': self.w_path,
                  'K': self.K_path, 'A': self.K_path,
                  'L': self.L_path, 'Y': self.Y_path, 'C': self.C_path}
        if K_g_path is not None:
            result['K_g'] = K_g_path
        if K_domestic is not None:
            result['K_domestic'] = K_domestic
        if NFA_path is not None:
            result['NFA'] = NFA_path
        return result
    
    def compute_government_budget_path(self, n_sim: Optional[int] = None, verbose=True):
        """Compute government budget for all transition periods."""
        # Y_path, not birth_cohort_solutions: a run served from the household
        # cache has aggregates and age means but no cohort models.
        if self.Y_path is None:
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
            'ui', 'pension', 'gov_health', 'oop_health', 'medical_total', 'kappa',
            'transfers', 'govt_spending',
            'public_investment', 'defense_spending', 'other_net_spending',
            'tax_y', 'foreign_transfer', 'education', 'lump_sum',
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
            # The fund holds government liabilities, so it accrues at the
            # sovereign rate, not the return on capital. It used r_path (4%)
            # against r_B (1.9%) until 2026-10-01.
            r_t = (float(self.r_B_path[t]) if getattr(self, 'r_B_path', None) is not None
                   else (float(self.r_path[t]) if self.r_path is not None else 0.0))
            # Per-capita detrended stock: divide the whole right-hand side by
            # Gamma_t = (1+g)(1+n_t).
            S_pens[t + 1] = ((1 + r_t) * S_pens[t] + budget_path['tax_p'][t]
                             - budget_path['pension'][t]) / self._growth_at(t)
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
        axes[2].set_title("Household wealth A")
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
        axes[5].set_title("A/Y")
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

            # _simulate_birth_cohort_cached returns an 11-tuple of per-age means
            # (memory-optimized for aggregation), which drops fields like
            # employed_sim and l_sim that the plot wants. For just two cohorts
            # the raw panels are cheap, so call the cohort model directly.
            model = self.birth_cohort_solutions[edu_type][b]
            results = model.simulate(T_sim=self.T, n_sim=int(n_sim), seed=seed)

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

def get_test_config(trend_growth=0.0):
    """Return a minimal LifecycleConfig for fast testing.

    trend_growth defaults to 0.0 so existing tests are unaffected; the
    balanced-growth tests pass g explicitly.
    """
    T, n_h = 20, 1
    config = LifecycleConfig(
        T=T,
        beta=0.99,
        gamma=1.0,
        trend_growth=trend_growth,
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


def run_fast_test(backend='numpy', recompute_bequests=False):
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
    pension_replacement_path = np.ones(T_transition) * 0.3
    
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
        verbose=True,
        recompute_bequests=recompute_bequests,
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


def run_full_simulation(backend='numpy', recompute_bequests=True):
    """Run full OLG transition simulation."""
    # Single MC knob for this run
    N_SIM_FULL = 5000

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
    pension_replacement_path = np.ones(T_transition) * 0.3

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
        recompute_bequests=recompute_bequests,
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


def run_from_config(config_path, backend='numpy', recompute_bequests=False, n_sim=None):
    """Run OLG transition from a JSON config file.

    Uses build_olg_transition from calibrate.py to construct the economy
    and transition paths from the same JSON used for calibration.
    """
    import json
    from calibrate import build_olg_transition

    with open(config_path) as f:
        config_data = json.load(f)

    economy, paths, T_tr = build_olg_transition(config_data, backend=backend)
    sim_n = n_sim or config_data.get('transition', {}).get('n_sim', 2000)

    # Run baseline simulation (T_transition determined by len(r_path)) with
    # the budget lines of the configuration: the spending shares, public
    # investment at its stationary level, the output tax at its base-year
    # rate and the lump-sum transfer at lambda times base-year output (one);
    # the fixed point over the lump-sum path and the terminal tax rate is the
    # drivers' (baseline_closure.solve_baseline), not this quick route's.
    prod = config_data.get('production', {})
    fisc = config_data.get('fiscal', {})
    I_g = ((prod.get('delta_g', 0.05) + economy.growth_factors(T_tr) - 1.0)
           * prod.get('K_g', 0.0)) if prod.get('eta_g', 0.0) != 0.0 else None
    results = economy.simulate_transition(
        r_path=paths['r_path'],
        tau_c_path=paths['tau_c_path'],
        tau_l_path=paths['tau_l_path'],
        tau_p_path=paths['tau_p_path'],
        tau_k_path=paths['tau_k_path'],
        pension_replacement_path=paths['pension_replacement_path'],
        I_g_path=I_g,
        G_over_Y=fisc.get('G_over_Y', 0.0),
        defense_over_Y=fisc.get('defense_over_Y', 0.0),
        tau_y_path=np.full(T_tr, float(paths.get('tau_y', 0.0) or 0.0)),
        lump_sum_path=np.full(T_tr, float(paths.get('lump_sum_over_Y', 0.0) or 0.0)),
        education_over_Y0=paths.get('education_over_Y0', 0.0),
        education_index_path=paths.get('education_index_path'),
        foreign_transfer_over_Y=paths.get('foreign_transfer_over_Y'),
        unemployment_index_path=paths.get('unemployment_index_path'),
        r_B_path=paths.get('r_B_path'),
        n_sim=sim_n,
        recompute_bequests=recompute_bequests,
    )

    print(f"\nTransition from {config_path}: T_transition={T_tr}, n_sim={sim_n}")
    print(f"  Y[0]={results['Y'][0]:.4f}, Y[-1]={results['Y'][-1]:.4f}")
    print(f"  K[0]={results['K'][0]:.4f}, L[0]={results['L'][0]:.4f}")

    return economy, results


def main():
    """
    Main entry point. Check for --test, --config flags and run accordingly.
    """
    backend = _parse_backend()
    if backend != 'numpy':
        print(f"Using backend: {backend}")

    # Check for --config
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    if config_path:
        recompute_bequests = '--recompute-bequests' in sys.argv
        n_sim = None
        for i, arg in enumerate(sys.argv):
            if arg == '--n-sim' and i + 1 < len(sys.argv):
                n_sim = int(sys.argv[i + 1])
        economy, results = run_from_config(
            config_path, backend=backend,
            recompute_bequests=recompute_bequests, n_sim=n_sim)
    elif '--test' in sys.argv:
        recompute_bequests = '--recompute-bequests' in sys.argv
        economy, results = run_fast_test(backend=backend, recompute_bequests=recompute_bequests)
    else:
        recompute_bequests = '--no-recompute-bequests' not in sys.argv
        economy, results = run_full_simulation(backend=backend, recompute_bequests=recompute_bequests)

    return economy, results


if __name__ == "__main__":
    economy, results = main()
