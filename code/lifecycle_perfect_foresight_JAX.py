"""
JAX-accelerated Lifecycle Model with Perfect Foresight

This module implements a GPU-optimized version of the lifecycle consumption-savings model
using JAX for automatic differentiation and JIT compilation.

Key features:
- JAX JIT compilation for fast execution
- GPU acceleration (automatically uses GPU if available)
- Vectorized operations for parallel state-space solving
- Batched simulation for thousands of agents
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.scipy.special import logsumexp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple
import time
from functools import partial

# Check if GPU is available
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


@dataclass
class LifecycleConfigJAX:
    """Configuration class for JAX lifecycle model parameters."""
    
    # === Life cycle parameters ===
    T: int = 60
    beta: float = 0.96
    gamma: float = 2.0
    current_age: int = 0
    
    # === Retirement parameters ===
    retirement_age: int = 45
    pension_replacement_path: Optional[np.ndarray] = None
    pension_replacement_default: float = 0.60
    
    # === Asset grid parameters ===
    a_min: float = 0.0
    a_max: float = 50.0
    n_a: int = 100
    
    # === Income process parameters ===
    n_y: int = 5
    
    # === Education parameters ===
    education_type: str = 'medium'
    
    edu_params: dict = field(default_factory=lambda: {
        'low': {
            'mu_y': 0.05,
            'sigma_y': 0.03,
            'rho_y': 0.97,
            'unemployment_rate': 0.10,
        },
        'medium': {
            'mu_y': 0.1,
            'sigma_y': 0.03,
            'rho_y': 0.97,
            'unemployment_rate': 0.06,
        },
        'high': {
            'mu_y': 0.12,
            'sigma_y': 0.03,
            'rho_y': 0.97,
            'unemployment_rate': 0.03,
        }
    })
    
    # === Unemployment parameters ===
    job_finding_rate: float = 0.5
    max_job_separation_rate: float = 0.1
    ui_replacement_rate: float = 0.4
    
    # === Health process parameters ===
    n_h: int = 3
    h_good: float = 1.0
    h_moderate: float = 0.7
    h_poor: float = 0.3
    
    m_good: float = 0.05
    m_moderate: float = 0.15
    m_poor: float = 0.30
    
    kappa: float = 0.7
    
    P_h_young: list = field(default_factory=lambda: [
        [0.95, 0.04, 0.01],
        [0.30, 0.60, 0.10],
        [0.10, 0.30, 0.60]
    ])
    
    P_h_middle: list = field(default_factory=lambda: [
        [0.85, 0.12, 0.03],
        [0.20, 0.60, 0.20],
        [0.05, 0.25, 0.70]
    ])
    
    P_h_old: list = field(default_factory=lambda: [
        [0.70, 0.20, 0.10],
        [0.10, 0.50, 0.40],
        [0.02, 0.18, 0.80]
    ])
    
    P_h: Optional[np.ndarray] = None
    
    # === Price paths ===
    r_path: Optional[np.ndarray] = None
    w_path: Optional[np.ndarray] = None
    
    r_default: float = 0.03
    w_default: float = 1.0
    
    # === Tax rate paths ===
    tau_c_path: Optional[np.ndarray] = None
    tau_l_path: Optional[np.ndarray] = None
    tau_p_path: Optional[np.ndarray] = None
    tau_k_path: Optional[np.ndarray] = None
    
    tau_c_default: float = 0.0
    tau_l_default: float = 0.0
    tau_p_default: float = 0.0
    tau_k_default: float = 0.0
    
    # === Initial conditions ===
    initial_assets: Optional[float] = None
    initial_avg_earnings: Optional[float] = None
    
    def _replace(self, **changes):
        """Return a new instance with specified fields replaced."""
        return replace(self, **changes)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Setup income process parameters
        if self.education_type in ['low', 'medium', 'high']:
            params = self.edu_params[self.education_type]
            self.mu_y = params['mu_y']
            self.sigma_y = params['sigma_y']
            self.rho_y = params['rho_y']
        
        # Create health transition matrix if not provided
        if self.P_h is None:
            if self.n_h == 3:
                P_base = np.array([
                    [0.90, 0.08, 0.02],
                    [0.10, 0.80, 0.10],
                    [0.05, 0.15, 0.80]
                ])
            elif self.n_h == 2:
                P_base = np.array([
                    [0.95, 0.05],
                    [0.10, 0.90]
                ])
            elif self.n_h == 1:
                P_base = np.array([[1.0]])
            else:
                P_base = np.eye(self.n_h)
            
            self.P_h = np.tile(P_base, (self.T, 1, 1))
        else:
            if self.P_h.ndim == 2:
                self.P_h = np.tile(self.P_h, (self.T, 1, 1))


class LifecycleModelJAX:
    """
    JAX-accelerated lifecycle model with GPU support.
    
    This implementation uses:
    - JIT compilation for fast execution
    - Vectorized operations across state space
    - Batched simulation for parallel agent execution
    - GPU acceleration when available
    """
    
    def __init__(self, config: LifecycleConfigJAX, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        
        # Extract parameters
        self.T = config.T
        self.beta = config.beta
        self.gamma = config.gamma
        self.n_a = config.n_a
        self.n_y = config.n_y
        self.n_h = config.n_h
        self.current_age = config.current_age
        self.retirement_age = config.retirement_age
        self.ui_replacement_rate = config.ui_replacement_rate
        self.kappa = config.kappa
        
        # Create grids (as JAX arrays)
        self.a_grid = self._create_asset_grid()
        self.y_grid, self.P_y = self._income_process()
        self.h_grid, self.P_h = self._health_process()
        self.m_grid = self._medical_cost_grid()
        
        # Setup paths
        self.r_path = self._setup_path(config.r_path, config.r_default)
        self.w_path = self._setup_path(config.w_path, config.w_default)
        self.tau_c_path = self._setup_path(config.tau_c_path, config.tau_c_default)
        self.tau_l_path = self._setup_path(config.tau_l_path, config.tau_l_default)
        self.tau_p_path = self._setup_path(config.tau_p_path, config.tau_p_default)
        self.tau_k_path = self._setup_path(config.tau_k_path, config.tau_k_default)
        self.pension_replacement_path = self._setup_path(
            config.pension_replacement_path, 
            config.pension_replacement_default
        )
        
        # Policy functions (will be computed)
        self.V = None
        self.a_policy = None
        self.c_policy = None
        
        if verbose:
            print(f"JAX Lifecycle Model initialized")
            print(f"  Backend: {jax.default_backend()}")
            print(f"  Education: {config.education_type}")
            print(f"  State space: {self.n_a} × {self.n_y} × {self.n_h} × {self.n_y}")
            print(f"  Total states per period: {self.n_a * self.n_y * self.n_h * self.n_y:,}")
    
    def _create_asset_grid(self):
        """Create asset grid with exponential spacing."""
        grid_linear = jnp.linspace(0, 1, self.n_a)
        curvature = 1.5
        grid_transformed = grid_linear ** curvature
        a_grid = self.config.a_min + (self.config.a_max - self.config.a_min) * grid_transformed
        return a_grid
    
    def _setup_path(self, path, default_value):
        """Setup time-varying path as JAX array."""
        if path is None:
            return jnp.ones(self.T) * default_value
        else:
            path = jnp.array(path)
            required_length = self.T - self.current_age
            if len(path) < required_length:
                padding = jnp.ones(required_length - len(path)) * path[-1]
                path = jnp.concatenate([path, padding])
            
            full_path = jnp.ones(self.T) * default_value
            full_path = full_path.at[self.current_age:self.current_age + len(path)].set(
                path[:required_length]
            )
            return full_path
    
    def _income_process(self):
        """Discretize income process using Tauchen method."""
        from quantecon.markov import tauchen
        
        edu_params = self.config.edu_params[self.config.education_type]
        rho_y = edu_params['rho_y']
        sigma_y = edu_params['sigma_y']
        mu_y = edu_params['mu_y']
        unemployment_rate = edu_params['unemployment_rate']
        
        n_employed = self.n_y - 1
        mc = tauchen(n_employed, rho_y, sigma_y, mu_y, n_std=2)
        y_employed = np.exp(mc.state_values)
        P_employed = mc.P
        
        # Create full grid with unemployment
        y_grid_np = np.zeros(self.n_y)
        y_grid_np[0] = 0.0
        y_grid_np[1:] = y_employed
        
        # Create transition matrix
        P_y_np = np.zeros((self.n_y, self.n_y))
        
        if unemployment_rate < 1e-10:
            P_y_np[0, 0] = 0.0
            P_y_np[0, 1:] = 1.0 / n_employed
            for i in range(n_employed):
                P_y_np[i + 1, 0] = 0.0
                P_y_np[i + 1, 1:] = P_employed[i, :]
        else:
            job_finding_rate = self.config.job_finding_rate
            max_job_sep = self.config.max_job_separation_rate
            
            P_y_np[0, 0] = 1 - job_finding_rate
            P_y_np[0, 1:] = job_finding_rate / n_employed
            
            for i in range(n_employed):
                job_sep = unemployment_rate / (1 - unemployment_rate) * job_finding_rate
                job_sep = min(job_sep, max_job_sep)
                P_y_np[i + 1, 0] = job_sep
                P_y_np[i + 1, 1:] = (1 - job_sep) * P_employed[i, :]
        
        # Normalize rows
        for i in range(self.n_y):
            P_y_np[i, :] = P_y_np[i, :] / P_y_np[i, :].sum()
        
        return jnp.array(y_grid_np), jnp.array(P_y_np)
    
    def _health_process(self):
        """Create health process."""
        if self.n_h == 1:
            h_grid = jnp.array([self.config.h_good])
            P_h = jnp.ones((self.T, 1, 1))
            return h_grid, P_h
        elif self.n_h == 3:
            h_grid = jnp.array([
                self.config.h_good,
                self.config.h_moderate,
                self.config.h_poor
            ])
        elif self.n_h == 2:
            h_grid = jnp.array([self.config.h_good, self.config.h_poor])
        else:
            h_grid = jnp.linspace(1.0, 0.3, self.n_h)
        
        # Convert P_h to JAX array
        P_h = jnp.array(self.config.P_h)
        
        return h_grid, P_h
    
    def _medical_cost_grid(self):
        """Create medical cost grid."""
        if self.n_h == 1:
            return jnp.array([self.config.m_good])
        elif self.n_h == 3:
            return jnp.array([
                self.config.m_good,
                self.config.m_moderate,
                self.config.m_poor
            ])
        elif self.n_h == 2:
            return jnp.array([self.config.m_good, self.config.m_poor])
        else:
            return jnp.linspace(self.config.m_good, self.config.m_poor, self.n_h)
    
    @staticmethod
    @jit
    def utility(c, gamma):
        """CRRA utility function (JIT-compiled)."""
        return jnp.where(
            gamma == 1.0,
            jnp.log(jnp.maximum(c, 1e-10)),
            (jnp.maximum(c, 1e-10) ** (1 - gamma)) / (1 - gamma)
        )
    
    @partial(jit, static_argnums=(0,))
    def _solve_terminal_period(self, params):
        """
        Solve terminal period (vectorized across entire state space).
        
        Returns V[T-1], a_policy[T-1], c_policy[T-1]
        """
        t = self.T - 1
        r_t = self.r_path[t]
        w_t = self.w_path[t]
        tau_c_t = self.tau_c_path[t]
        tau_l_t = self.tau_l_path[t]
        tau_p_t = self.tau_p_path[t]
        tau_k_t = self.tau_k_path[t]
        
        # Vectorized computation across (n_a, n_y, n_h, n_y_last)
        a_grid = self.a_grid[:, None, None, None]  # (n_a, 1, 1, 1)
        y_grid = self.y_grid[None, :, None, None]  # (1, n_y, 1, 1)
        h_grid = self.h_grid[None, None, :, None]  # (1, 1, n_h, 1)
        y_last_grid = self.y_grid[None, None, None, :]  # (1, 1, 1, n_y)
        
        # UI benefits (if unemployed)
        is_unemployed = (y_grid == 0.0)
        ui_benefit = jnp.where(
            is_unemployed,
            self.ui_replacement_rate * w_t * y_last_grid,
            0.0
        )
        
        # Labor income
        gross_labor = w_t * y_grid * h_grid + ui_benefit
        payroll_tax = tau_p_t * (w_t * y_grid * h_grid)
        income_tax = tau_l_t * (gross_labor - payroll_tax)
        after_tax_labor = gross_labor - payroll_tax - income_tax
        
        # Capital income
        gross_capital = r_t * a_grid
        capital_tax = tau_k_t * gross_capital
        after_tax_capital = gross_capital - capital_tax
        
        # Health expenditure
        m_cost = self.m_grid[None, None, :, None]  # (1, 1, n_h, 1)
        oop_health = (1 - self.kappa) * m_cost
        
        # Budget and consumption
        budget = a_grid + after_tax_capital + after_tax_labor - oop_health
        c = jnp.maximum(budget / (1 + tau_c_t), 1e-10)
        
        # Value function
        V = self.utility(c, self.gamma)
        
        # Policy functions
        a_policy = jnp.zeros((self.n_a, self.n_y, self.n_h, self.n_y), dtype=jnp.int32)
        c_policy = c
        
        return V, a_policy, c_policy
    
    @partial(jit, static_argnums=(0,))
    def _bellman_operator(self, t, V_next, params):
        """
        Bellman operator for period t (fully vectorized).
        
        Args:
            t: Period index
            V_next: Value function for t+1, shape (n_a, n_y, n_h, n_y)
            params: Tuple of (r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t, is_retired)
        
        Returns:
            V[t], a_policy[t], c_policy[t]
        """
        r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t, is_retired = params
        
        # Current state grids (broadcasting dimensions)
        a = self.a_grid[:, None, None, None]  # (n_a, 1, 1, 1)
        y = self.y_grid[None, :, None, None]  # (1, n_y, 1, 1)
        h = self.h_grid[None, None, :, None]  # (1, 1, n_h, 1)
        y_last = self.y_grid[None, None, None, :]  # (1, 1, 1, n_y)
        
        # Next period assets (choice variable)
        a_next = self.a_grid[None, None, None, None, :]  # (1, 1, 1, 1, n_a)
        
        # Compute income and budget
        is_unemployed = (y == 0.0)
        ui_benefit = jnp.where(
            is_unemployed,
            self.ui_replacement_rate * w_t * y_last,
            0.0
        )
        
        gross_labor = w_t * y * h + ui_benefit
        payroll_tax = tau_p_t * (w_t * y * h)
        income_tax = tau_l_t * (gross_labor - payroll_tax)
        after_tax_labor = jnp.where(is_retired, 0.0, gross_labor - payroll_tax - income_tax)
        
        gross_capital = r_t * a
        capital_tax = tau_k_t * gross_capital
        after_tax_capital = gross_capital - capital_tax
        
        m_cost = self.m_grid[None, None, :, None]
        oop_health = (1 - self.kappa) * m_cost
        
        budget = a + after_tax_capital + after_tax_labor - oop_health
        
        # Consumption for each choice of a_next
        # Shape: (n_a, n_y, n_h, n_y, n_a_next)
        c = (budget[..., None] - a_next) / (1 + tau_c_t)
        c = jnp.maximum(c, 1e-10)
        
        # Current utility
        u = self.utility(c, self.gamma)
        
        # Expected continuation value - use lax.cond instead of if/else
        def compute_EV_retired():
            """Compute EV for retired agents (only health transitions)."""
            # V_next shape: (n_a, n_y, n_h, n_y)
            # In retirement: y=0, y_last=0
            # P_h[t] shape: (n_h, n_h_next)
            
            # Extract retirement slice: (n_a_next, n_h_next)
            V_next_ret = V_next[:, 0, :, 0]  # shape: (n_a, n_h)
            
            # Compute expected value over health transitions
            # For each current health h, compute E[V(a', h') | h]
            # P_h[t]: (n_h_curr, n_h_next)
            # V_next_ret: (n_a_next, n_h_next)
            # We want: (n_h_curr, n_a_next)
            
            # Matrix multiply: (n_h, n_h_next) @ (n_h_next, n_a) = (n_h, n_a)
            EV_temp = jnp.matmul(self.P_h[t], V_next_ret.T).T  # (n_a, n_h)
            
            # Now EV_temp is (n_a_next, n_h_curr)
            # We need to broadcast to (n_a_curr, n_y, n_h_curr, n_y_last, n_a_next)
            # Reshape: (n_a_next, n_h) -> (1, 1, n_h, 1, n_a_next)
            EV = EV_temp.T[None, None, :, None, :]  # Transpose to (n_h, n_a), then add dims
            
            # Broadcast to full shape: (n_a, n_y, n_h, n_y, n_a_next)
            return jnp.broadcast_to(EV, (self.n_a, self.n_y, self.n_h, self.n_y, self.n_a))
        
        def compute_EV_working():
            """Compute EV for working agents (income and health transitions)."""
            
            def compute_EV_for_y(y_idx):
                """Compute EV for agents with current income y_idx."""
                # V_next indexed by current y as y_last: (n_a, n_y_next, n_h_next)
                V_next_y = V_next[:, :, :, y_idx]
                
                # Transition probabilities
                # P_y[y_idx]: (n_y_next,)
                # P_h[t]: (n_h, n_h_next)
                
                # Joint transition: (n_h, n_y_next, n_h_next)
                P_y_broadcast = self.P_y[y_idx][None, :, None]  # (1, n_y_next, 1)
                P_h_broadcast = self.P_h[t][:, None, :]  # (n_h, 1, n_h_next)
                P_joint = P_y_broadcast * P_h_broadcast  # (n_h, n_y_next, n_h_next)
                
                # Expected value computation
                # We want: E[V(a', y', h') | h, y] for each a'
                # V_next_y: (n_a, n_y_next, n_h_next)
                # P_joint: (n_h, n_y_next, n_h_next)
                
                # For each (h, a), sum over (y_next, h_next)
                # Reshape for broadcasting
                V_expanded = V_next_y[None, :, :, :]  # (1, n_a, n_y_next, n_h_next)
                P_expanded = P_joint[:, None, :, :]  # (n_h, 1, n_y_next, n_h_next)
                
                # Element-wise multiply and sum over y_next and h_next
                EV_y = jnp.sum(P_expanded * V_expanded, axis=(2, 3))  # (n_h, n_a)
                
                return EV_y  # (n_h, n_a)
            
            # Vectorize over all y values: (n_y, n_h, n_a)
            EV_all_y = vmap(compute_EV_for_y)(jnp.arange(self.n_y))
            
            # Reshape: (n_y, n_h, n_a) -> (1, n_y, n_h, 1, n_a)
            EV = EV_all_y[None, :, :, None, :]
            
            # Broadcast to: (n_a, n_y, n_h, n_y, n_a)
            return jnp.broadcast_to(EV, (self.n_a, self.n_y, self.n_h, self.n_y, self.n_a))
        
        # Use lax.cond for JIT-compatible conditional
        EV = lax.cond(
            is_retired,
            compute_EV_retired,
            compute_EV_working
        )
        
        # Total value for each choice
        total_value = u + self.beta * EV  # (n_a, n_y, n_h, n_y, n_a_next)
        
        # Find optimal a_next
        V = jnp.max(total_value, axis=-1)  # (n_a, n_y, n_h, n_y)
        a_policy = jnp.argmax(total_value, axis=-1).astype(jnp.int32)  # (n_a, n_y, n_h, n_y)
        
        # Optimal consumption
        a_policy_idx = a_policy[:, :, :, :, None]  # Add dimension for gather
        c_policy = jnp.take_along_axis(c, a_policy_idx, axis=-1).squeeze(-1)
        
        return V, a_policy, c_policy    
    def solve(self, verbose=False):
        """
        Solve the lifecycle problem using backward induction (GPU-accelerated).
        """
        if verbose:
            print(f"Solving lifecycle model with JAX...")
            start_time = time.time()
        
        # Initialize arrays
        V_all = jnp.zeros((self.T, self.n_a, self.n_y, self.n_h, self.n_y))
        a_policy_all = jnp.zeros((self.T, self.n_a, self.n_y, self.n_h, self.n_y), dtype=jnp.int32)
        c_policy_all = jnp.zeros((self.T, self.n_a, self.n_y, self.n_h, self.n_y))
        
        # Terminal period
        V_terminal, a_policy_terminal, c_policy_terminal = self._solve_terminal_period(None)
        V_all = V_all.at[self.T - 1].set(V_terminal)
        a_policy_all = a_policy_all.at[self.T - 1].set(a_policy_terminal)
        c_policy_all = c_policy_all.at[self.T - 1].set(c_policy_terminal)
        
        # Backward induction
        for t in range(self.T - 2, self.current_age - 1, -1):
            if verbose and t % 10 == 0:
                print(f"  Solving period {t}/{self.T-1}")
            
            is_retired = (t >= self.retirement_age)
            params = (
                self.r_path[t],
                self.w_path[t],
                self.tau_c_path[t],
                self.tau_l_path[t],
                self.tau_p_path[t],
                self.tau_k_path[t],
                is_retired
            )
            
            V_next = V_all[t + 1]
            V_t, a_policy_t, c_policy_t = self._bellman_operator(t, V_next, params)
            
            V_all = V_all.at[t].set(V_t)
            a_policy_all = a_policy_all.at[t].set(a_policy_t)
            c_policy_all = c_policy_all.at[t].set(c_policy_t)
        
        self.V = V_all
        self.a_policy = a_policy_all
        self.c_policy = c_policy_all
        
        # Block until computation is done (for accurate timing)
        self.V[0, 0, 0, 0, 0].block_until_ready()
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"Done! Time: {elapsed:.2f} seconds")
            print(f"  Per-period average: {elapsed / (self.T - self.current_age):.3f} seconds")
    
    def simulate(self, T_sim=None, n_sim=10000, seed=42):
        """
        Simulate lifecycle paths (batched on GPU).
        
        Uses JAX's random number generation and vectorized operations.
        
        Args:
            T_sim: Number of periods to simulate (default: full lifecycle T)
            n_sim: Number of agents to simulate
            seed: Random seed
        """
        if T_sim is None:
            T_sim = self.T  # Changed from: self.T - self.current_age
        
        key = jax.random.PRNGKey(seed)
        
        # Initialize states (vectorized across n_sim)
        key, subkey = jax.random.split(key)
        
        # Initial income distribution
        edu_unemp_rate = self.config.edu_params[self.config.education_type]['unemployment_rate']
        
        if edu_unemp_rate < 1e-10:
            # All start employed
            i_y_init = jax.random.choice(
                subkey,
                jnp.arange(1, self.n_y),
                shape=(n_sim,)
            )
        else:
            # Use stationary distribution
            from scipy.linalg import eig
            P_y_np = np.array(self.P_y)
            eigenvalues, eigenvectors = eig(P_y_np.T)
            stationary_idx = np.argmax(eigenvalues.real)
            stationary = eigenvectors[:, stationary_idx].real
            stationary = stationary / stationary.sum()
            
            i_y_init = jax.random.choice(
                subkey,
                jnp.arange(self.n_y),
                shape=(n_sim,),
                p=jnp.array(stationary)
            )
        
        # Compile simulation function
        sim_func = jit(partial(self._simulate_jax, T_sim=T_sim))
        
        # Run simulation
        key, subkey = jax.random.split(key)
        results = sim_func(subkey, i_y_init)
        
        # Block until done
        results[0].block_until_ready()
        
        return results
    
    def _simulate_jax(self, key, i_y_init, T_sim):
        """
        JIT-compiled simulation function (fully vectorized).
        
        This function simulates all agents in parallel using JAX's scan.
        """
        n_sim = i_y_init.shape[0]
        
        # Initialize state arrays
        # Handle initial_avg_earnings outside of JAX context
        if self.config.initial_avg_earnings is not None:
            initial_avg = self.config.initial_avg_earnings
        else:
            initial_avg = 0.0
        
        init_state = {
            'i_a': jnp.zeros(n_sim, dtype=jnp.int32),
            'i_y': i_y_init,
            'i_y_last': i_y_init,
            'i_h': jnp.zeros(n_sim, dtype=jnp.int32),
            'avg_earnings': jnp.ones(n_sim) * initial_avg,
            'avg_earnings_at_ret': jnp.zeros(n_sim),
            'key': key
        }
        
        def scan_body(state, t):
            """Single time step of simulation."""
            lifecycle_age = self.current_age + t
            is_retired = (lifecycle_age >= self.retirement_age)
            
            # Extract current states
            i_a = state['i_a']
            i_y = state['i_y']
            i_y_last = state['i_y_last']
            i_h = state['i_h']
            avg_earnings = state['avg_earnings']
            avg_earnings_at_ret = state['avg_earnings_at_ret']
            key = state['key']
            
            # Lookup current values
            a = self.a_grid[i_a]
            y = self.y_grid[i_y]
            h = self.h_grid[i_h]
            
            # Compute period quantities
            w_t = self.w_path[lifecycle_age]
            r_t = self.r_path[lifecycle_age]
            
            # UI benefits
            ui = jnp.where(
                i_y == 0,
                self.ui_replacement_rate * w_t * self.y_grid[i_y_last],
                0.0
            )
            
            # Pension (if retired)
            # Lock in avg_earnings at retirement
            avg_earnings_at_ret = jnp.where(
                (lifecycle_age == self.retirement_age),
                avg_earnings,
                avg_earnings_at_ret
            )
            
            pension = jnp.where(
                is_retired,
                self.pension_replacement_path[lifecycle_age] * avg_earnings_at_ret,
                0.0
            )
            
            # Income and employment
            employed = (i_y > 0) & ~is_retired
            effective_y = w_t * y * h + ui
            
            # Update average earnings (if working)
            gross_labor_income = w_t * y * h
            total_work_years = jnp.maximum(lifecycle_age - self.current_age + 1, 1)
            avg_earnings = jnp.where(
                ~is_retired,
                (avg_earnings * (total_work_years - 1) + gross_labor_income) / total_work_years,
                avg_earnings
            )
            
            # Policy functions - use lifecycle_age to index
            c = self.c_policy[lifecycle_age, i_a, i_y, i_h, i_y_last]
            
            # Medical costs
            m = self.m_grid[i_h]
            oop_m = (1 - self.kappa) * m
            gov_m = self.kappa * m
            
            # Taxes
            tax_c = self.tau_c_path[lifecycle_age] * c
            tax_l = self.tau_l_path[lifecycle_age] * effective_y
            tax_p = self.tau_p_path[lifecycle_age] * pension
            tax_k = self.tau_k_path[lifecycle_age] * r_t * a
            
            # Next period states
            # Asset policy - use lifecycle_age to index
            i_a_next = jnp.where(
                t < T_sim - 1,
                self.a_policy[lifecycle_age, i_a, i_y, i_h, i_y_last],
                i_a
            )
            
            # Income transitions (only if working and not terminal)
            key, subkey = jax.random.split(key)
            # Sample from categorical distribution using cumulative sum method
            uniform_draw = jax.random.uniform(subkey, shape=i_y.shape)
            cumsum_P_y = jnp.cumsum(self.P_y[i_y], axis=-1)
            i_y_sampled = jnp.sum(uniform_draw[:, None] > cumsum_P_y, axis=-1)
            
            i_y_next = jnp.where(
                (t < T_sim - 1) & ~is_retired,
                i_y_sampled,
                jnp.where(is_retired, 0, i_y)
            )
            
            i_y_last_next = jnp.where(
                (t < T_sim - 1) & ~is_retired,
                i_y,
                i_y_last
            )
            
            # Health transitions
            key, subkey = jax.random.split(key)
            uniform_draw = jax.random.uniform(subkey, shape=i_h.shape)
            cumsum_P_h = jnp.cumsum(self.P_h[lifecycle_age, i_h], axis=-1)
            i_h_sampled = jnp.sum(uniform_draw[:, None] > cumsum_P_h, axis=-1)
            
            i_h_next = jnp.where(
                t < T_sim - 1,
                i_h_sampled,
                i_h
            )
            
            # Create retired array (broadcast scalar to match n_sim)
            retired_array = jnp.ones(n_sim, dtype=jnp.bool_) * is_retired
            
            # Output for this period
            output = {
                'a': a,
                'c': c,
                'y': y,
                'h': h,
                'h_idx': i_h,
                'effective_y': effective_y,
                'employed': employed,
                'ui': ui,
                'm': m,
                'oop_m': oop_m,
                'gov_m': gov_m,
                'tax_c': tax_c,
                'tax_l': tax_l,
                'tax_p': tax_p,
                'tax_k': tax_k,
                'avg_earnings': avg_earnings,
                'pension': pension,
                'retired': retired_array
            }
            
            # Updated state
            new_state = {
                'i_a': i_a_next,
                'i_y': i_y_next,
                'i_y_last': i_y_last_next,
                'i_h': i_h_next,
                'avg_earnings': avg_earnings,
                'avg_earnings_at_ret': avg_earnings_at_ret,
                'key': key
            }
            
            return new_state, output
        
        # Run scan over time periods
        _, outputs = lax.scan(scan_body, init_state, jnp.arange(T_sim))
        
        # outputs has shape (T_sim, n_sim) for each key
        # Return as tuple - no need to transpose since scan already gives us (T_sim, n_sim)
        results = tuple(
            outputs[key] for key in [
                'a', 'c', 'y', 'h', 'h_idx', 'effective_y', 'employed',
                'ui', 'm', 'oop_m', 'gov_m', 'tax_c', 'tax_l', 'tax_p', 'tax_k',
                'avg_earnings', 'pension', 'retired'
            ]
        )
        
        return results

# Test function
if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        print("\n" + "="*70)
        print("TESTING JAX LIFECYCLE MODEL (GPU-ACCELERATED)")
        print("="*70)
        
        # Create minimal config for quick testing
        config = LifecycleConfigJAX(
            T=10,
            beta=0.96,
            gamma=2.0,
            current_age=0,
            retirement_age=8,
            pension_replacement_default=0.40,
            education_type='medium',
            n_a=10,
            n_y=2,
            n_h=1,
            m_good=0.0
        )
        
        n_sim = 5000
        
    else:
        # PRODUCTION MODE - larger state space
        print("\n" + "="*70)
        print("PRODUCTION RUN - JAX LIFECYCLE MODEL")
        print("="*70)
        
        # Production configuration with high resolution
        config = LifecycleConfigJAX(
            T=60,                      # Full lifecycle (age 20-80)
            beta=0.97,                 # Higher discount factor (more patient)
            gamma=2.0,
            current_age=20,
            retirement_age=65,
            pension_replacement_default=0.60,  # More generous pension (60% replacement)
            education_type='medium',
            n_a=100,                   # High resolution for assets
            n_y=7,                     # More income states
            n_h=3,                     # Full health states
            a_max=100.0,               # Larger asset range
            m_good=0.05,
            m_moderate=0.15,
            m_poor=0.30,
            # Adjust income process for realistic lifecycle profile
            edu_params={
                'low': {
                    'mu_y': 0.0,       # Mean of log income (will be exp(0.0) = 1.0)
                    'sigma_y': 0.15,   # Standard deviation
                    'rho_y': 0.95,     # High persistence
                    'unemployment_rate': 0.08,
                },
                'medium': {
                    'mu_y': 0.3,       # Higher mean (exp(0.3) ≈ 1.35)
                    'sigma_y': 0.20,   # More variance
                    'rho_y': 0.96,     # High persistence
                    'unemployment_rate': 0.05,
                },
                'high': {
                    'mu_y': 0.5,       # Even higher mean (exp(0.5) ≈ 1.65)
                    'sigma_y': 0.25,   # More variance
                    'rho_y': 0.97,     # Very high persistence
                    'unemployment_rate': 0.03,
                }
            },
            # More realistic initial conditions
            initial_assets=0.5,        # Start with some assets
            initial_avg_earnings=0.8,  # Start below average earnings
            # Adjust wage path to show lifecycle earnings growth
            w_path=np.concatenate([
                # Ages 20-35: Rapid growth (entry to mid-career)
                np.linspace(0.7, 1.2, 15),
                # Ages 35-50: Continued growth (mid to senior)
                np.linspace(1.2, 1.5, 15),
                # Ages 50-65: Peak and slight decline (senior to retirement)
                np.linspace(1.5, 1.45, 15),
                # Ages 65-80: Retirement (wage irrelevant but set for consistency)
                np.ones(15) * 1.45
            ])
        )
        
        n_sim = 50000  # Large simulation
    
    # ============================================================
    # Common code for both test and production
    # ============================================================
    
    # Create model
    print("\nCreating model...")
    model = LifecycleModelJAX(config, verbose=True)
    
    # ============================================================
    # Solve and simulate (WITH warm-up for production only)
    # ============================================================
    
    if "--test" not in sys.argv:
        # PRODUCTION: Run with warm-up to show compilation overhead
        print("\nSolving model...")
        start = time.time()
        model.solve(verbose=True)
        warmup_solve_time = time.time() - start
        
        print(f"\n{'='*70}")
        print(f"SOLVE TIME: {warmup_solve_time:.2f} seconds")
        print(f"{'='*70}")
        
        print("\nSimulating...")
        start = time.time()
        results = model.simulate(n_sim=n_sim, seed=42)
        warmup_sim_time = time.time() - start
        
        print(f"\n{'='*70}")
        print(f"SIMULATION TIME: {warmup_sim_time:.2f} seconds ({n_sim:,} agents)")
        print(f"{'='*70}")
        
        # Run again to show speedup from compiled code
        print("\nRunning compiled version...")
        start = time.time()
        model.solve(verbose=False)
        actual_solve_time = time.time() - start
        
        start = time.time()
        results = model.simulate(n_sim=n_sim, seed=42)
        actual_sim_time = time.time() - start
        
        # Show performance improvement
        solve_speedup = warmup_solve_time / actual_solve_time
        sim_speedup = warmup_sim_time / actual_sim_time
        
        if solve_speedup > 1.5 or sim_speedup > 1.5:
            print("\n" + "="*70)
            print("PERFORMANCE NOTE")
            print("="*70)
            if solve_speedup > 1.5:
                print(f"Solve: JIT compilation overhead was {warmup_solve_time - actual_solve_time:.2f}s")
                print(f"       ({solve_speedup:.1f}x speedup after compilation)")
            if sim_speedup > 1.5:
                print(f"Simulation: JIT compilation overhead was {warmup_sim_time - actual_sim_time:.2f}s")
                print(f"            ({sim_speedup:.1f}x speedup after compilation)")
    
    else:
        # TEST: Just run once (no warm-up comparison)
        print("\nSolving model...")
        start = time.time()
        model.solve(verbose=True)
        solve_time = time.time() - start
        
        print(f"\n{'='*70}")
        print(f"SOLVE TIME: {solve_time:.2f} seconds")
        print(f"{'='*70}")
        
        print("\nSimulating...")
        start = time.time()
        results = model.simulate(n_sim=n_sim, seed=42)
        sim_time = time.time() - start
        
        print(f"\n{'='*70}")
        print(f"SIMULATION TIME: {sim_time:.2f} seconds ({n_sim:,} agents)")
        print(f"{'='*70}")
    
    # ============================================================
    # Extract and display results (common for both modes)
    # ============================================================
    
    (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim,
     ui_sim, m_sim, oop_m_sim, gov_m_sim, tax_c_sim, tax_l_sim, tax_p_sim, 
     tax_k_sim, avg_earnings_sim, pension_sim, retired_sim) = results
    
    # Convert to numpy for plotting
    a_sim_np = np.array(a_sim)
    c_sim_np = np.array(c_sim)
    effective_y_sim_np = np.array(effective_y_sim)
    pension_sim_np = np.array(pension_sim)
    
    # Print statistics
    print(f"\nRESULTS:")
    print(f"  Mean assets:       {np.mean(a_sim_np):.2f}")
    print(f"  Mean consumption:  {np.mean(c_sim_np):.3f}")
    print(f"  Mean income:       {np.mean(effective_y_sim_np):.3f}")
    
    # For pension, only compute mean for retired periods
    retired_mask = np.array(retired_sim).flatten()
    pension_flat = pension_sim_np.flatten()
    if np.any(retired_mask):
        mean_pension = np.mean(pension_flat[retired_mask])
        print(f"  Mean pension:      {mean_pension:.3f}")
    else:
        print(f"  Mean pension:      N/A (no retirement periods)")
    
    # Simple plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get actual simulation length from data and create age array
    T_sim = a_sim_np.shape[0]
    ages = np.arange(config.current_age, config.current_age + T_sim)
    
    axes[0, 0].plot(ages, a_sim_np.mean(axis=1))
    axes[0, 0].set_title('Assets')
    axes[0, 0].axvline(x=config.retirement_age, color='r', linestyle='--', label='Retirement')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Age')
    
    axes[0, 1].plot(ages, c_sim_np.mean(axis=1))
    axes[0, 1].set_title('Consumption')
    axes[0, 1].axvline(x=config.retirement_age, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Age')
    
    axes[1, 0].plot(ages, effective_y_sim_np.mean(axis=1))
    axes[1, 0].set_title('Income')
    axes[1, 0].axvline(x=config.retirement_age, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Age')
    
    axes[1, 1].plot(ages, pension_sim_np.mean(axis=1))
    axes[1, 1].set_title('Pension')
    axes[1, 1].axvline(x=config.retirement_age, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Age')
    
    plt.tight_layout()
    
    # Save with different names for test vs production
    if "--test" in sys.argv:
        filename = 'jax_test_results.png'
        print(f"\n{'='*70}")
        print("JAX TEST COMPLETE")
        print(f"{'='*70}")
    else:
        filename = 'jax_production_results.png'
        print(f"\n{'='*70}")
        print("JAX PRODUCTION RUN COMPLETE")
        print(f"{'='*70}")
    
    plt.savefig(filename, dpi=300)
    print(f"\nPlot saved: {filename}")
    
    plt.show()