import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from quantecon.markov import tauchen
from dataclasses import dataclass, field, replace
from typing import Optional
from multiprocessing import Pool, cpu_count
from functools import partial


@dataclass
class LifecycleConfig:
    """Configuration class for lifecycle model parameters."""
    
    # === Life cycle parameters ===
    T: int = 60                          # Life cycle length (periods)
    beta: float = 0.96                   # Discount factor
    gamma: float = 2.0                   # CRRA coefficient
    current_age: int = 0                 # Starting age (0 = age 20)
    
    # === Retirement parameters ===
    retirement_age: int = 45             # Mandatory retirement age (period index, e.g., 45 = age 65)
    pension_replacement_path: Optional[np.ndarray] = None  # Pension replacement rate path (fraction of avg earnings)
    pension_replacement_default: float = 0.60  # Default pension replacement rate (60% of avg earnings)
    
    # === Asset grid parameters ===
    a_min: float = 0.0                   # Borrowing constraint
    a_max: float = 50.0                  # Maximum assets
    n_a: int = 100                       # Asset grid points
    
    # === Income process parameters ===
    n_y: int = 5                         # Number of income states (including unemployment)
    
    # === Pension parameters ===
    N_earnings_history: int = 35         # Number of years for earnings history computation (moving average)
    
    # === Education parameters ===
    education_type: str = 'medium'       # 'low', 'medium', 'high'
    
    # Education-specific income parameters
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
    
    # Health expenditure by health state
    m_good: float = 0.05
    m_moderate: float = 0.15
    m_poor: float = 0.30
    
    # Government health coverage rate
    kappa: float = 0.7
    
    # Health transition probabilities by age group
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
    
    # Health transition matrix (will be created in __post_init__ if None)
    P_h: Optional[np.ndarray] = None
    
    # === Price paths (time-varying) ===
    r_path: Optional[np.ndarray] = None
    w_path: Optional[np.ndarray] = None
    
    # Default constant prices if paths not provided
    r_default: float = 0.03
    w_default: float = 1.0
    
    # === Tax rate paths (time-varying) ===
    tau_c_path: Optional[np.ndarray] = None
    tau_l_path: Optional[np.ndarray] = None
    tau_p_path: Optional[np.ndarray] = None
    tau_k_path: Optional[np.ndarray] = None
    
    # Default tax rates if paths not provided
    tau_c_default: float = 0.0
    tau_l_default: float = 0.0
    tau_p_default: float = 0.0
    tau_k_default: float = 0.0
    
    # === Initial conditions ===
    initial_assets: Optional[float] = None   # Initial asset level (default: a_min)
    initial_avg_earnings: Optional[float] = None  # ← ADD THIS LINE
    
    def _replace(self, **changes): # <-- 2. Add this method
        """Return a new instance with specified fields replaced."""
        return replace(self, **changes) 
    
    def __post_init__(self):
        """Post-initialization setup and validation."""
        
        # Setup income process parameters based on education type
        if self.education_type == 'low':
            self.mu_y = 0.05
            self.sigma_y = 0.03
            self.rho_y = 0.97
        elif self.education_type == 'medium':
            self.mu_y = 0.1
            self.sigma_y = 0.03
            self.rho_y = 0.97
        elif self.education_type == 'high':
            self.mu_y = 0.12
            self.sigma_y = 0.03
            self.rho_y = 0.97
        
        # Create health transition matrix if not provided
        if self.P_h is None:
            # Age-dependent health transitions (T x n_h x n_h)
            # For now, use constant transitions across ages
            
            if self.n_h == 3:
                # Default 3-state health transition
                P_base = np.array([
                    [0.90, 0.08, 0.02],  # Good -> Good, Moderate, Poor
                    [0.10, 0.80, 0.10],  # Moderate -> Good, Moderate, Poor
                    [0.05, 0.15, 0.80]   # Poor -> Good, Moderate, Poor
                ])
            elif self.n_h == 2:
                # 2-state health transition (Good, Poor)
                P_base = np.array([
                    [0.95, 0.05],  # Good -> Good, Poor
                    [0.10, 0.90]   # Poor -> Good, Poor
                ])
            else:
                # Fallback: identity matrix
                P_base = np.eye(self.n_h)
            
            # Replicate across all ages
            self.P_h = np.tile(P_base, (self.T, 1, 1))
        else:
            # Validate provided P_h
            if self.P_h.ndim == 2:
                # If 2D, replicate across ages
                P_arr = self.P_h
                assert P_arr.shape == (self.n_h, self.n_h), \
                    f"Health transition matrix wrong shape: expected {(self.n_h, self.n_h)}, got {P_arr.shape}"
                self.P_h = np.tile(P_arr, (self.T, 1, 1))
            elif self.P_h.ndim == 3:
                # Already age-dependent
                assert self.P_h.shape == (self.T, self.n_h, self.n_h), \
                    f"Health transition matrix wrong shape: expected {(self.T, self.n_h, self.n_h)}, got {self.P_h.shape}"
            else:
                raise ValueError(f"P_h must be 2D or 3D array, got {self.P_h.ndim}D")
    

class LifecycleModelPerfectForesight:
    """
    Lifecycle consumption-savings model with:
    - Time-varying interest rates and wages (perfect foresight)
    - Time-varying tax rates: consumption, labor income, payroll, capital income (perfect foresight)
    - Unemployment insurance with replacement rate
    - Health expenditures with government coverage
    - Heterogeneous agents (income, unemployment, and health risk)
    - Education heterogeneity (different income processes by education type)
    - Incomplete markets (borrowing constraint)
    - Moving average of gross labor income for earnings history (continuous state)
    - Parallel processing support for faster computation
    """
    
    def __init__(self, config: LifecycleConfig, verbose: bool = True):
        """Initialize model with configuration."""
        self.config = config
        self.verbose = verbose
        
        # Extract frequently used parameters
        self.T = config.T
        self.beta = config.beta
        self.gamma = config.gamma
        self.a_min = config.a_min
        self.a_max = config.a_max
        self.n_a = config.n_a
        self.n_y = config.n_y
        self.n_h = config.n_h
        self.current_age = config.current_age
        self.ui_replacement_rate = config.ui_replacement_rate
        self.kappa = config.kappa
        self.N_earnings_history = config.N_earnings_history
        self.retirement_age = config.retirement_age
        
        # Set up price paths
        self.r_path = self._setup_path(config.r_path, config.r_default, "r_path")
        self.w_path = self._setup_path(config.w_path, config.w_default, "w_path")
        
        # Set up tax paths
        self.tau_c_path = self._setup_path(config.tau_c_path, config.tau_c_default, "tau_c_path")
        self.tau_l_path = self._setup_path(config.tau_l_path, config.tau_l_default, "tau_l_path")
        self.tau_p_path = self._setup_path(config.tau_p_path, config.tau_p_default, "tau_p_path")
        self.tau_k_path = self._setup_path(config.tau_k_path, config.tau_k_default, "tau_k_path")
        
        # Set up pension replacement rate path
        self.pension_replacement_path = self._setup_path(
            config.pension_replacement_path, 
            config.pension_replacement_default, 
            "pension_replacement_path"
        )
        
        # Create grids and processes
        self.a_grid = self._create_asset_grid()
        self.y_grid, self.P_y = self._income_process()
        
        # COMPREHENSIVE DIAGNOSTIC
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"INCOME PROCESS DIAGNOSTIC - {self.config.education_type.upper()} EDUCATION")
            print('='*70)
            edu_params = self.config.edu_params[self.config.education_type]
            print(f"Input parameters:")
            print(f"  mu_y:    {edu_params['mu_y']}")
            print(f"  sigma_y: {edu_params['sigma_y']}")
            print(f"  rho_y:   {edu_params['rho_y']}")
            print(f"\nIncome grid (y_grid):")
            print(f"  {self.y_grid}")
            print(f"\nIncome grid details:")
            for i, y in enumerate(self.y_grid):
                state_name = "UNEMPLOYED" if i == 0 else f"Employed {i}"
                print(f"  State {i} ({state_name}): y = {y:.6f}")
            
            # Check stationary distribution
            from scipy.linalg import eig
            eigenvalues, eigenvectors = eig(self.P_y.T)
            stationary_idx = np.argmax(eigenvalues.real)
            stationary = eigenvectors[:, stationary_idx].real
            stationary = stationary / stationary.sum()
            
            print(f"\nTransition matrix P_y (first 3 rows):")
            print(self.P_y[:3])
            
            print(f"\nStationary distribution:")
            for i in range(self.n_y):
                state_name = "UNEMPLOYED" if i == 0 else f"Employed {i}"
                print(f"  State {i} ({state_name}, y={self.y_grid[i]:.4f}): {stationary[i]:.4%}")
            
            expected_income = np.dot(stationary, self.y_grid)
            print(f"\nExpected steady-state income: {expected_income:.6f}")
            
            if expected_income < 0.01:
                print(f"\n⚠️  WARNING: Expected income is nearly zero!")
                print(f"    This will cause zero average income in simulations!")
            
            print('='*70)
        
        self.h_grid, self.P_h = self._health_process()
        
        # Health expenditure grid (by health state)
        if self.n_h == 3:
            self.m_grid = np.array([
                config.m_good,
                config.m_moderate,  # ✅ FIXED
                config.m_poor
            ])
        elif self.n_h == 2:
            self.m_grid = np.array([
                config.m_good,
                config.m_poor
            ])
        else:
            # Fallback: linearly spaced from m_good to m_poor
            self.m_grid = np.linspace(config.m_good, config.m_poor, self.n_h)
        
        # Value and policy functions (earnings history stored as continuous value, not gridded)
        # Dimensions: (T, n_a, n_y, n_h, n_y_last)
        # avg_earnings is tracked separately in simulation
        self.V = None
        self.a_policy = None
        self.c_policy = None
        self.pension_avg_policy = None  # Stores next period's pension average
    
    def _create_asset_grid(self):
        """Create non-linear asset grid with more points near borrowing constraint."""
        # Use exponential spacing for better resolution at low assets
        grid_linear = np.linspace(0, 1, self.n_a)
        
        # Transform to get more points near zero
        curvature = 1.5  # Higher values = more points near zero
        grid_transformed = grid_linear ** curvature
        
        # Scale to [a_min, a_max]
        a_grid = self.a_min + (self.a_max - self.a_min) * grid_transformed
        
        return a_grid
    
    def _setup_path(self, path, default_value, name):
        """Set up time-varying path or use default constant."""
        if path is None:
            # Create path for full lifecycle (T periods total)
            return np.ones(self.T) * default_value
        else:
            path = np.array(path)
            # Path must be at least as long as the agent's remaining lifetime
            required_length = self.T - self.current_age
            if len(path) < required_length:
                # If path is too short, pad with the last value
                padding_needed = required_length - len(path)
                path = np.concatenate([path, np.ones(padding_needed) * path[-1]])
            
            # Create a full-length array initialized with default value
            full_path = np.ones(self.T) * default_value
            # Fill in the portion from current_age onwards with the provided path
            full_path[self.current_age:self.current_age + len(path)] = path[:required_length]
            
            return full_path
    
    def _income_process(self):
        """
        Discretize income process using Tauchen method with unemployment state.
        Uses education-specific parameters.
        
        The first state (index 0) represents unemployment with y=0.
        The remaining states represent employed income levels.
        """
        # Get education-specific parameters
        edu_type = self.config.education_type
        edu_params = self.config.edu_params[edu_type]
        
        rho_y = edu_params['rho_y']
        sigma_y = edu_params['sigma_y']
        mu_y = edu_params['mu_y']
        unemployment_rate = edu_params['unemployment_rate']
        
        # Extract unemployment parameters
        job_finding_rate = self.config.job_finding_rate
        max_job_separation_rate = self.config.max_job_separation_rate
        
        # Discretize employed income states using Tauchen (excluding unemployment)
        n_employed = self.n_y - 1
        
        mc = tauchen(n_employed, rho_y, sigma_y, mu_y, n_std=2)  # Changed from n_std=3
        y_employed = np.exp(mc.state_values)
        P_employed = mc.P
        
        # Create full income grid with unemployment state first
        y_grid = np.zeros(self.n_y)
        y_grid[0] = 0.0  # Unemployment state
        y_grid[1:] = y_employed
        
        # Create full transition matrix
        P_y = np.zeros((self.n_y, self.n_y))
        
        # If unemployment rate is zero, everyone stays employed
        if unemployment_rate == 0.0 or unemployment_rate < 1e-10:
            # From unemployment (should never happen, but set to go to employment)
            P_y[0, 0] = 0.0
            P_y[0, 1:] = 1.0 / n_employed  # Equal probability to any employed state
            
            # From employed states: no job separation, just normal employed transitions
            for i in range(n_employed):
                P_y[i + 1, 0] = 0.0  # Never become unemployed
                P_y[i + 1, 1:] = P_employed[i, :]  # Normal employed transitions
        else:
            # Original code for non-zero unemployment
            # Transition probabilities from unemployment (state 0)
            P_y[0, 0] = 1 - job_finding_rate  # Stay unemployed
            P_y[0, 1:] = job_finding_rate / n_employed  # Equal probability to any employed state
            
            # Transition probabilities from employed states (states 1 to n_y-1)
            for i in range(n_employed):
                job_separation_rate = unemployment_rate / (1 - unemployment_rate) * job_finding_rate
                job_separation_rate = min(job_separation_rate, max_job_separation_rate)
                
                P_y[i + 1, 0] = job_separation_rate  # Become unemployed
                P_y[i + 1, 1:] = (1 - job_separation_rate) * P_employed[i, :]
        
        # Ensure rows sum to 1 (numerical stability)
        for i in range(self.n_y):
            P_y[i, :] = P_y[i, :] / P_y[i, :].sum()
        
        return y_grid, P_y
    
    def _health_process(self):
        """Age-dependent health transition matrix."""
        # Create health grid based on n_h
        if self.n_h == 1:
            # If only one health state, it's a single good state with no medical costs
            m_grid = np.array([self.config.m_good])
            # The transition matrix is trivial: always stay in the single state
            P_h = np.ones((self.T, 1, 1))
            return m_grid, P_h
        if self.n_h == 3:
            h_grid = np.array([
                self.config.h_good,
                self.config.h_moderate,
                self.config.h_poor
            ])
        elif self.n_h == 2:
            h_grid = np.array([
                self.config.h_good,
                self.config.h_poor
            ])
        else:
            # Fallback: equally spaced from 1.0 to 0.3
            h_grid = np.linspace(1.0, 0.3, self.n_h)
        
        # Age-dependent transition probabilities
        P_h = np.zeros((self.T, self.n_h, self.n_h))
        
        if self.n_h == 3:
            # Use the detailed 3-state transitions from config
            P_h_young = np.array(self.config.P_h_young)
            P_h_middle = np.array(self.config.P_h_middle)
            P_h_old = np.array(self.config.P_h_old)
            
            for t in range(self.T):
                age = 20 + t
                
                if age < 40:
                    P_h[t, :, :] = P_h_young
                elif age < 60:
                    P_h[t, :, :] = P_h_middle
                else:
                    P_h[t, :, :] = P_h_old
        
        elif self.n_h == 2:
            # Simple 2-state transitions (Good, Poor)
            # Age-dependent: health deteriorates with age
            for t in range(self.T):
                age = 20 + t
                
                if age < 40:
                    # Young: mostly stay in good health
                    P_h[t, :, :] = np.array([
                        [0.98, 0.02],  # Good -> Good, Poor
                        [0.20, 0.80]   # Poor -> Good, Poor
                    ])
                elif age < 60:
                    # Middle age: more health deterioration
                    P_h[t, :, :] = np.array([
                        [0.95, 0.05],  # Good -> Good, Poor
                        [0.15, 0.85]   # Poor -> Good, Poor
                    ])
                else:
                    # Old: significant health deterioration
                    P_h[t, :, :] = np.array([
                        [0.90, 0.10],  # Good -> Good, Poor
                        [0.10, 0.90]   # Poor -> Good, Poor
                    ])
        
        else:
            # Fallback for other n_h values: use simple persistence
            for t in range(self.T):
                persistence = 0.90  # Probability of staying in same state
                P_h[t, :, :] = (1 - persistence) / (self.n_h - 1) * np.ones((self.n_h, self.n_h))
                np.fill_diagonal(P_h[t, :, :], persistence)
        
        return h_grid, P_h
    
    @staticmethod
    @njit
    def utility(c, gamma):
        """CRRA utility function."""
        if gamma == 1.0:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    def solve(self, verbose=False, parallel=False, n_jobs=None):
        """
        Solve the lifecycle problem using backward induction.
        
        Parameters:
        -----------
        verbose : bool
            Print progress messages
        parallel : bool
            Use parallel processing for period-by-period solving
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available CPUs.
        """
        if verbose:
            print(f"Solving lifecycle model for {self.config.education_type} education...")
            print(f"  Solving for periods {self.current_age} to {self.T-1}")
            if parallel:
                n_jobs_actual = n_jobs if n_jobs is not None else cpu_count()
                print(f"  Using parallel processing with {n_jobs_actual} workers")
        
        # Initialize value and policy functions
        # Dimensions: (T, n_a, n_y, n_h, n_y_last)
        self.V = np.zeros((self.T, self.n_a, self.n_y, self.n_h, self.n_y))
        self.a_policy = np.zeros((self.T, self.n_a, self.n_y, self.n_h, self.n_y), dtype=np.int32)
        self.c_policy = np.zeros((self.T, self.n_a, self.n_y, self.n_h, self.n_y))
        
        # Terminal period (T-1)
        self._solve_terminal_period(verbose)
        
        # Backward induction
        if parallel:
            self._solve_backward_parallel(verbose, n_jobs)
        else:
            self._solve_backward_sequential(verbose)
        
        if verbose:
            print("Done!")
    
    def _solve_terminal_period(self, verbose=False):
        """Solve the terminal period."""
        for i_a, a in enumerate(self.a_grid):
            for i_y, y in enumerate(self.y_grid):
                for i_h, h in enumerate(self.h_grid):
                    for i_y_last in range(self.n_y):
                        # Compute UI benefits if unemployed
                        if i_y == 0:  # Unemployed
                            ui_benefit = self.ui_replacement_rate * self.w_path[self.T - 1] * self.y_grid[i_y_last]
                        else:
                            ui_benefit = 0.0
                        
                        # After-tax labor income
                        gross_labor_income = self.w_path[self.T - 1] * y * h + ui_benefit
                        payroll_tax = self.tau_p_path[self.T - 1] * (self.w_path[self.T - 1] * y * h)
                        income_tax = self.tau_l_path[self.T - 1] * (gross_labor_income - payroll_tax)
                        after_tax_labor_income = gross_labor_income - payroll_tax - income_tax
                        
                        # After-tax capital income
                        gross_capital_income = self.r_path[self.T - 1] * a
                        capital_income_tax = self.tau_k_path[self.T - 1] * gross_capital_income
                        after_tax_capital_income = gross_capital_income - capital_income_tax
                        
                        # Out-of-pocket health expenditure
                        oop_health_exp = (1 - self.kappa) * self.m_grid[i_h]
                        
                        # Budget constraint
                        budget = a + after_tax_capital_income + after_tax_labor_income - oop_health_exp
                        c = budget / (1 + self.tau_c_path[self.T - 1])
                        c = max(c, 1e-10)
                        
                        self.V[self.T - 1, i_a, i_y, i_h, i_y_last] = self.utility(c, self.gamma)
                        self.a_policy[self.T - 1, i_a, i_y, i_h, i_y_last] = 0
                        self.c_policy[self.T - 1, i_a, i_y, i_h, i_y_last] = c
    
    def _solve_period(self, t, r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t):
        """Solve single period with time-varying prices, taxes, UI, health expenditures, and retirement."""
        
        # Check if this is a retirement period
        is_retired = (t >= self.retirement_age)
        
        for i_a, a in enumerate(self.a_grid):
            for i_y, y in enumerate(self.y_grid):
                for i_h, h in enumerate(self.h_grid):
                    for i_y_last in range(self.n_y):
                        
                        if is_retired:
                            # RETIREMENT PERIOD: No labor income, receive pension based on avg_earnings
                            # Note: We can't use avg_earnings directly in value function
                            # Instead, we'll need to track it separately during simulation
                            # For the value function, we assume pension = 0 (conservative)
                            # The actual pension will be computed during simulation
                            
                            gross_labor_income = 0.0
                            ui_benefit = 0.0
                            after_tax_labor_income = 0.0
                            
                        else:
                            # WORKING PERIOD: Labor income with potential UI
                            
                            # Compute UI benefits if unemployed
                            if i_y == 0:  # Unemployed
                                ui_benefit = self.ui_replacement_rate * w_t * self.y_grid[i_y_last]
                            else:
                                ui_benefit = 0.0
                            
                            # Compute after-tax labor income
                            gross_wage_income = w_t * y * h
                            gross_labor_income = gross_wage_income + ui_benefit
                            payroll_tax = tau_p_t * gross_wage_income
                            income_tax = tau_l_t * (gross_labor_income - payroll_tax)
                            after_tax_labor_income = gross_labor_income - payroll_tax - income_tax
                        
                        # Compute after-tax capital income (same for working and retired)
                        gross_capital_income = r_t * a
                        capital_income_tax = tau_k_t * gross_capital_income
                        after_tax_capital_income = gross_capital_income - capital_income_tax
                        
                        # Out-of-pocket health expenditure
                        oop_health_exp = (1 - self.kappa) * self.m_grid[i_h]
                        
                        # Budget available for consumption and savings
                        budget = a + after_tax_capital_income + after_tax_labor_income - oop_health_exp
                        
                        # Find optimal next period assets
                        max_val = -np.inf
                        best_a_idx = 0
                        best_c = 1e-10
                        
                        for i_a_next, a_next in enumerate(self.a_grid):
                            # Consumption from budget constraint
                            c = (budget - a_next) / (1 + tau_c_t)
                            
                            if c <= 0:
                                continue
                            
                            # Expected continuation value
                            EV = 0.0
                            
                            if is_retired:
                                # In retirement: only health transitions (no income transitions)
                                for i_h_next in range(self.n_h):
                                    prob = self.P_h[t, i_h, i_h_next]
                                    # In retirement, income state is irrelevant (set to 0 = unemployed)
                                    next_val = self.V[t + 1, i_a_next, 0, i_h_next, 0]
                                    
                                    if np.isfinite(prob) and np.isfinite(next_val):
                                        EV += prob * next_val
                            else:
                                # While working: both income and health transitions
                                for i_y_next in range(self.n_y):
                                    for i_h_next in range(self.n_h):
                                        prob = self.P_y[i_y, i_y_next] * self.P_h[t, i_h, i_h_next]
                                        next_val = self.V[t + 1, i_a_next, i_y_next, i_h_next, i_y]
                                        
                                        if np.isfinite(prob) and np.isfinite(next_val):
                                            EV += prob * next_val
                            
                            # Check if EV is valid
                            if not np.isfinite(EV):
                                continue
                            
                            # Total value
                            val = self.utility(c, self.gamma) + self.beta * EV
                            
                            if val > max_val:
                                max_val = val
                                best_a_idx = i_a_next
                                best_c = c
                        
                        # Store results (use fallback if no valid choice found)
                        if np.isfinite(max_val):
                            self.V[t, i_a, i_y, i_h, i_y_last] = max_val
                            self.a_policy[t, i_a, i_y, i_h, i_y_last] = best_a_idx
                            self.c_policy[t, i_a, i_y, i_h, i_y_last] = best_c
                        else:
                            # Fallback: consume everything if no valid choice
                            c_fallback = max(budget / (1 + tau_c_t), 1e-10)
                            self.V[t, i_a, i_y, i_h, i_y_last] = self.utility(c_fallback, self.gamma)
                            self.a_policy[t, i_a, i_y, i_h, i_y_last] = 0
                            self.c_policy[t, i_a, i_y, i_h, i_y_last] = c_fallback

    def _solve_backward_sequential(self, verbose=False):
        """Solve backward induction sequentially."""
        for t in range(self.T - 2, self.current_age - 1, -1):
            if verbose and t % 10 == 0:
                print(f"  Solving period {t}/{self.T-1}")
            
            r_t = self.r_path[t]
            w_t = self.w_path[t]
            tau_c_t = self.tau_c_path[t]
            tau_l_t = self.tau_l_path[t]
            tau_p_t = self.tau_p_path[t]
            tau_k_t = self.tau_k_path[t]
            
            self._solve_period(t, r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t)
    
    def _solve_backward_parallel(self, verbose=False, n_jobs=None):
        """Solve backward induction using parallel processing."""
        if n_jobs is None:
            n_jobs = cpu_count()
        
        # Create partial function with model parameters
        solve_func = partial(
            _solve_period_wrapper,
            model=self
        )
        
        # Prepare period data
        periods = []
        for t in range(self.T - 2, self.current_age - 1, -1):
            periods.append((
                t,
                self.r_path[t],
                self.w_path[t],
                self.tau_c_path[t],
                self.tau_l_path[t],
                self.tau_p_path[t],
                self.tau_k_path[t]
            ))
        
        # Process periods in batches to maintain sequential dependency
        # Note: We can't fully parallelize backward induction since t depends on t+1
        # But we can parallelize across state space dimensions within each period
        for i, period_data in enumerate(periods):
            if verbose and period_data[0] % 10 == 0:
                print(f"  Solving period {period_data[0]}/{self.T-1}")
            
            # For each period, parallelize across asset grid points
            state_combinations = []
            for i_a in range(self.n_a):
                state_combinations.append((period_data, i_a))
            
            # Solve all asset states in parallel
            with Pool(processes=n_jobs) as pool:
                results = pool.map(solve_func, state_combinations)
            
            # Update value and policy functions
            t = period_data[0]
            for i_a, (V_slice, a_pol_slice, c_pol_slice) in enumerate(results):
                self.V[t, i_a, :, :, :] = V_slice
                self.a_policy[t, i_a, :, :, :] = a_pol_slice
                self.c_policy[t, i_a, :, :, :] = c_pol_slice

    def _simulate_sequential(self, T_sim, n_sim):
        """Sequential simulation with correct indexing."""
        
        a_sim = np.zeros((T_sim, n_sim))
        c_sim = np.zeros((T_sim, n_sim))
        y_sim = np.zeros((T_sim, n_sim))
        h_sim = np.zeros((T_sim, n_sim))
        h_idx_sim = np.zeros((T_sim, n_sim), dtype=int)
        effective_y_sim = np.zeros((T_sim, n_sim))  # ← INITIALIZE AS ARRAY
        ui_sim = np.zeros((T_sim, n_sim))
        m_sim = np.zeros((T_sim, n_sim))
        oop_m_sim = np.zeros((T_sim, n_sim))
        gov_m_sim = np.zeros((T_sim, n_sim))
        avg_earnings_sim = np.zeros((T_sim, n_sim))
        pension_sim = np.zeros((T_sim, n_sim))
        retired_sim = np.zeros((T_sim, n_sim), dtype=bool)
        
        employed_sim = np.zeros((T_sim, n_sim), dtype=bool)
        
        tax_c_sim = np.zeros((T_sim, n_sim))
        tax_l_sim = np.zeros((T_sim, n_sim))
        tax_p_sim = np.zeros((T_sim, n_sim))
        tax_k_sim = np.zeros((T_sim, n_sim))
        
        avg_earnings_at_retirement = np.zeros(n_sim)
        
        # If the cohort starts retired, their "earnings at retirement" is their initial average earnings.
        if self.current_age >= self.retirement_age and hasattr(self.config, 'initial_avg_earnings') and self.config.initial_avg_earnings is not None:
            avg_earnings_at_retirement.fill(self.config.initial_avg_earnings)

        # Initial conditions
        edu_unemployment_rate = self.config.edu_params[self.config.education_type]['unemployment_rate']
        
        if edu_unemployment_rate < 1e-10:
            n_employed = self.n_y - 1
            i_y = np.random.choice(range(1, self.n_y), size=n_sim, 
                                  p=np.ones(n_employed) / n_employed)
        else:
            from scipy.linalg import eig
            eigenvalues, eigenvectors = eig(self.P_y.T)
            stationary_idx = np.argmax(eigenvalues.real)
            stationary = eigenvectors[:, stationary_idx].real
            stationary = stationary / stationary.sum()
            
            i_y = np.random.choice(self.n_y, size=n_sim, p=stationary)
        
        i_y_last = i_y.copy()
        i_h = np.zeros(n_sim, dtype=int)
        i_a = np.zeros(n_sim, dtype=int)
        if hasattr(self.config, 'initial_avg_earnings') and self.config.initial_avg_earnings is not None:
            avg_earnings = np.ones(n_sim) * self.config.initial_avg_earnings
        else:
            avg_earnings = np.zeros(n_sim)
        
        for t_sim in range(T_sim):
            lifecycle_age = self.current_age + t_sim
            is_retired = (lifecycle_age >= self.retirement_age)
            
            for i in range(n_sim):
                a_sim[t_sim, i] = self.a_grid[i_a[i]]
                h_sim[t_sim, i] = self.h_grid[i_h[i]]
                h_idx_sim[t_sim, i] = i_h[i]
                avg_earnings_sim[t_sim, i] = avg_earnings[i]
                retired_sim[t_sim, i] = is_retired
                
                if is_retired:
                    # Lock in average earnings at retirement
                    if t_sim == self.retirement_age - self.current_age:
                        avg_earnings_at_retirement[i] = avg_earnings[i]
                    
                    pension_replacement = self.pension_replacement_path[lifecycle_age]
                    pension_sim[t_sim, i] = pension_replacement * avg_earnings_at_retirement[i]
                    
                    y_sim[t_sim, i] = 0.0
                    employed_sim[t_sim, i] = False
                    ui_sim[t_sim, i] = 0.0
                    
                    i_y[i] = 0
                    i_y_last[i] = 0
                    
                else:
                    # WORKING PERIOD
                    y_sim[t_sim, i] = self.y_grid[i_y[i]]
                    employed_sim[t_sim, i] = (i_y[i] > 0)
                    pension_sim[t_sim, i] = 0.0
                    
                    if i_y[i] == 0:
                        ui_sim[t_sim, i] = self.ui_replacement_rate * self.w_path[lifecycle_age] * self.y_grid[i_y_last[i]]
                    else:
                        ui_sim[t_sim, i] = 0.0
                    
                    # --- FIX: UPDATE AVERAGE EARNINGS DURING WORKING YEARS ---
                    # Compute gross labor income (w * y * h, before taxes, excluding UI)
                    gross_labor_income = self.w_path[lifecycle_age] * y_sim[t_sim, i] * h_sim[t_sim, i]
                    
                    # Update average earnings using expanding average
                    # Total working years = lifecycle_age + 1 (ages 0, 1, ..., lifecycle_age)
                    total_work_years = lifecycle_age + 1
                    
                    # Expanding average formula: new_avg = (old_avg * (n-1) + new_value) / n
                    avg_earnings[i] = (avg_earnings[i] * (total_work_years - 1) + gross_labor_income) / total_work_years
                
                c_sim[t_sim, i] = self.c_policy[t_sim, i_a[i], i_y[i], i_h[i], i_y_last[i]]
                
                m_sim[t_sim, i] = self.m_grid[i_h[i]]
                oop_m_sim[t_sim, i] = (1 - self.kappa) * m_sim[t_sim, i]
                gov_m_sim[t_sim, i] = self.kappa * m_sim[t_sim, i]

                # --- FIX: Include health multiplier h in effective_y_sim ---
                effective_y_sim[t_sim, i] = self.w_path[lifecycle_age] * y_sim[t_sim, i] * h_sim[t_sim, i] + ui_sim[t_sim, i]
                
                gross_labor_income = effective_y_sim[t_sim, i]
                tax_c_sim[t_sim, i] = self.tau_c_path[lifecycle_age] * c_sim[t_sim, i]
                tax_l_sim[t_sim, i] = self.tau_l_path[lifecycle_age] * gross_labor_income
                tax_p_sim[t_sim, i] = self.tau_p_path[lifecycle_age] * pension_sim[t_sim, i]
                
                gross_capital_income = self.r_path[lifecycle_age] * a_sim[t_sim, i]
                tax_k_sim[t_sim, i] = self.tau_k_path[lifecycle_age] * gross_capital_income
                
                if t_sim < T_sim - 1:
                    i_a[i] = self.a_policy[t_sim, i_a[i], i_y[i], i_h[i], i_y_last[i]]
                    
                    if not is_retired:
                        i_y_last[i] = i_y[i]
                        i_y[i] = np.searchsorted(np.cumsum(self.P_y[i_y[i], :]), np.random.random())
                    
                    i_h[i] = np.searchsorted(np.cumsum(self.P_h[lifecycle_age, i_h[i], :]), np.random.random())
        
        return (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
                ui_sim, m_sim, oop_m_sim, gov_m_sim,
                tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                pension_sim, retired_sim)
    
    def simulate(self, T_sim=None, n_sim=10000, seed=42, parallel=False, n_jobs=None):
        """
        Simulate lifecycle paths with earnings history tracking and retirement.
        
        Parameters:
        -----------
        T_sim : int, optional
            Number of periods to simulate
        n_sim : int
            Number of agents to simulate
        seed : int
            Random seed
        parallel : bool
            Use parallel processing for simulation
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available CPUs.
        """
        if T_sim is None:
            T_sim = self.T - self.current_age
        
        np.random.seed(seed)
        
        if parallel:
            return self._simulate_parallel(T_sim, n_sim, seed, n_jobs)
        else:
            return self._simulate_sequential(T_sim, n_sim)
    
    def _simulate_parallel(self, T_sim, n_sim, seed, n_jobs=None):
        """Parallel simulation by splitting agents across workers."""
        if n_jobs is None:
            n_jobs = cpu_count()
        
        # Split agents across workers
        agents_per_job = n_sim // n_jobs
        remaining = n_sim % n_jobs
        
        agent_splits = []
        start_idx = 0
        for i in range(n_jobs):
            n_agents = agents_per_job + (1 if i < remaining else 0)
            agent_splits.append((start_idx, n_agents, seed + i))
            start_idx += n_agents
        
        # Create partial function
        sim_func = partial(_simulate_agent_batch, model=self, T_sim=T_sim)
        
        # Run simulations in parallel
        with Pool(processes=n_jobs) as pool:
            results = pool.map(sim_func, agent_splits)
        
        # Combine results
        return self._combine_simulation_results(results)
    
    def _combine_simulation_results(self, results):
        """Combine results from parallel simulation."""
        # Concatenate all results along agent dimension
        combined = tuple(
            np.concatenate([r[i] for r in results], axis=1)
            for i in range(len(results[0]))
        )
        return combined


# Helper functions for parallel processing (must be at module level for pickling)

def _solve_period_wrapper(args, model):
    """Wrapper for parallel period solving."""
    period_data, i_a = args
    t, r_t, w_t, tau_c_t, tau_l_t, tau_p_t, tau_k_t = period_data
    
    a = model.a_grid[i_a]
    is_retired = (t >= model.retirement_age)
    
    V_slice = np.zeros((model.n_y, model.n_h, model.n_y))
    a_pol_slice = np.zeros((model.n_y, model.n_h, model.n_y), dtype=np.int32)
    c_pol_slice = np.zeros((model.n_y, model.n_h, model.n_y))
    
    for i_y, y in enumerate(model.y_grid):
        for i_h, h in enumerate(model.h_grid):
            for i_y_last in range(model.n_y):
                
                if is_retired:
                    gross_labor_income = 0.0
                    ui_benefit = 0.0
                    after_tax_labor_income = 0.0
                else:
                    if i_y == 0:
                        ui_benefit = model.ui_replacement_rate * w_t * model.y_grid[i_y_last]
                    else:
                        ui_benefit = 0.0
                    
                    gross_wage_income = w_t * y * h
                    gross_labor_income = gross_wage_income + ui_benefit
                    payroll_tax = tau_p_t * gross_wage_income
                    income_tax = tau_l_t * (gross_labor_income - payroll_tax)
                    after_tax_labor_income = gross_labor_income - payroll_tax - income_tax
                
                gross_capital_income = r_t * a
                capital_income_tax = tau_k_t * gross_capital_income
                after_tax_capital_income = gross_capital_income - capital_income_tax
                
                oop_health_exp = (1 - model.kappa) * model.m_grid[i_h]
                budget = a + after_tax_capital_income + after_tax_labor_income - oop_health_exp
                
                max_val = -np.inf
                best_a_idx = 0
                best_c = 1e-10
                
                for i_a_next, a_next in enumerate(model.a_grid):
                    c = (budget - a_next) / (1 + tau_c_t)
                    
                    if c <= 0:
                        continue
                    
                    EV = 0.0
                    
                    if is_retired:
                        for i_h_next in range(model.n_h):
                            prob = model.P_h[t, i_h, i_h_next]
                            next_val = model.V[t + 1, i_a_next, 0, i_h_next, 0]
                            
                            if np.isfinite(prob) and np.isfinite(next_val):
                                EV += prob * next_val
                    else:
                        for i_y_next in range(model.n_y):
                            for i_h_next in range(model.n_h):
                                prob = model.P_y[i_y, i_y_next] * model.P_h[t, i_h, i_h_next]
                                next_val = model.V[t + 1, i_a_next, i_y_next, i_h_next, i_y]
                                
                                if np.isfinite(prob) and np.isfinite(next_val):
                                    EV += prob * next_val
                    
                    if not np.isfinite(EV):
                        continue
                    
                    val = model.utility(c, model.gamma) + model.beta * EV
                    
                    if val > max_val:
                        max_val = val
                        best_a_idx = i_a_next
                        best_c = c
                
                if np.isfinite(max_val):
                    V_slice[i_y, i_h, i_y_last] = max_val
                    a_pol_slice[i_y, i_h, i_y_last] = best_a_idx
                    c_pol_slice[i_y, i_h, i_y_last] = best_c
                else:
                    c_fallback = max(budget / (1 + tau_c_t), 1e-10)
                    V_slice[i_y, i_h, i_y_last] = model.utility(c_fallback, model.gamma)
                    a_pol_slice[i_y, i_h, i_y_last] = 0
                    c_pol_slice[i_y, i_h, i_y_last] = c_fallback
    
    return V_slice, a_pol_slice, c_pol_slice


def _simulate_agent_batch(args, model, T_sim):
    """Simulate a batch of agents."""
    start_idx, n_agents, seed = args
    
    np.random.seed(seed)
    
    # Run sequential simulation for this batch
    result = model._simulate_sequential(T_sim, n_agents)
    
    return result


def _solve_and_simulate_education_type(args):
    """Solve and simulate a single education type (for parallel execution)."""
    edu_type, n_sim, use_parallel_internal = args
    
    print(f"\n{'='*70}")
    print(f"Education type: {edu_type.upper()}")
    print('='*70)
    
    # Create configuration
    config = LifecycleConfig(
        T=60,
        beta=0.96,
        gamma=2.0,
        current_age=0,
        retirement_age=45,
        pension_replacement_default=0.60,
        education_type=edu_type,
        n_a=50,
        n_y=5,
    )
    
    # Create and solve model
    model = LifecycleModelPerfectForesight(config)
    print(f"\nSolving model for {edu_type}...")
    model.solve(verbose=True, parallel=use_parallel_internal)
    
    # Simulate
    print(f"\nSimulating {n_sim} agents for {edu_type}...")
    results = model.simulate(n_sim=n_sim, seed=42, parallel=use_parallel_internal)
    (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
     ui_sim, m_sim, oop_m_sim, gov_m_sim,
     tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
     pension_sim, retired_sim) = results
    
    # Compute statistics
    unemployment_rate = np.mean(~employed_sim & ~retired_sim)
    retirement_rate = np.mean(retired_sim)
    total_tax = tax_c_sim + tax_l_sim + tax_p_sim + tax_k_sim
    total_gov_spending = gov_m_sim + ui_sim + pension_sim
    
    print(f"\n{'-'*70}")
    print(f"RESULTS FOR {edu_type.upper()} EDUCATION:")
    print(f"{'-'*70}")
    print(f"  Mean assets:              {np.mean(a_sim):.2f}")
    print(f"  Mean consumption:         {np.mean(c_sim):.3f}")
    print(f"  Mean effective income:    {np.mean(effective_y_sim):.3f}")
    print(f"  Mean avg earnings:        {np.mean(avg_earnings_sim):.3f}")
    
    if np.any(retired_sim):
        print(f"  Mean pension (retired):   {np.mean(pension_sim[retired_sim]):.3f}")
    else:
        print(f"  Mean pension (retired):   N/A (no retirement periods)")
    
    print(f"  Unemployment rate:        {unemployment_rate:.2%}")
    print(f"  Retirement rate:          {retirement_rate:.2%}")
    print(f"  Mean total taxes:         {np.mean(total_tax):.4f}")
    print(f"  Mean gov spending:        {np.mean(total_gov_spending):.4f}")
    print(f"  Mean UI benefits:         {np.mean(ui_sim):.4f}")
    print(f"  Mean gov health spending: {np.mean(gov_m_sim):.4f}")
    
    # Age-specific statistics
    print(f"\n  Age-specific means:")
    ages_to_check = [0, 20, 40, 45, 50, 59]
    for age_idx in ages_to_check:
        if age_idx < len(a_sim):
            is_retired_age = age_idx >= (config.retirement_age - config.current_age)
            status = "RETIRED" if is_retired_age else "WORKING"
            print(f"    Age {20+age_idx} ({status}):")
            print(f"      Assets:      {np.mean(a_sim[age_idx, :]):.2f}")
            print(f"      Consumption: {np.mean(c_sim[age_idx, :]):.3f}")
            if is_retired_age and np.any(pension_sim[age_idx, :] > 0):
                print(f"      Pension:     {np.mean(pension_sim[age_idx, :]):.3f}")
            else:
                print(f"      Income:      {np.mean(effective_y_sim[age_idx, :]):.3f}")
    
    # Store results for plotting
    result_dict = {
        'a_sim': a_sim,
        'c_sim': c_sim,
        'effective_y_sim': effective_y_sim,
        'pension_sim': pension_sim,
        'retired_sim': retired_sim,
        'avg_earnings_sim': avg_earnings_sim,
        'employed_sim': employed_sim,
        'ui_sim': ui_sim,
        'total_tax': total_tax,
        'total_gov_spending': total_gov_spending,
        'config': config
    }
    
    return edu_type, result_dict

if __name__ == "__main__":
    import sys
    
    # Check if --test flag is provided
    if "--test" in sys.argv:
        # Check for parallel flags
        use_parallel = "--parallel" in sys.argv
        use_parallel_all = "--parallel-all" in sys.argv
        
        print("\n" + "="*70)
        print("TESTING LIFECYCLE MODEL WITH MINIMAL STATE CONFIGURATION")
        print("  n_h = 1 (single health state)")
        print("  n_y = 2 (unemployment + 1 employed state)")
        print("  Single education group (medium)")
        print("="*70)
        if use_parallel_all:
            print(f"FULL PARALLEL MODE: Using {cpu_count()} CPU cores")
        elif use_parallel:
            print(f"PARALLEL MODE: Using {cpu_count()} CPU cores")
        print("="*70)
        
        # Test with minimal configuration - only medium education
        education_types = ['medium']  # Only one education type
        n_sim = 500  # Reduced for faster testing
        
        # Store results for plotting
        all_results = {}
        
        # Run sequentially (only one education type)
        for edu_type in education_types:
            print(f"\n{'='*70}")
            print(f"Education type: {edu_type.upper()}")
            print('='*70)
            
            # Create minimal configuration
            config = LifecycleConfig(
                T=10,                      # Shorter lifecycle for testing
                beta=0.96,
                gamma=2.0,
                current_age=0,
                retirement_age=8,          # Retire at age 8 (= real age 28)
                pension_replacement_default=0.40,
                education_type=edu_type,
                n_a=10,                    # Minimal asset grid
                n_y=2,                     # Just unemployment + 1 employed state
                n_h=1,                     # Single health state (no health risk)
                # Override health parameters for n_h=1
                m_good=0.0,                # No medical costs with single health state
            )
            
            # Create and solve model
            model = LifecycleModelPerfectForesight(config, verbose=True)
            print(f"\nSolving model for {edu_type}...")
            model.solve(verbose=True, parallel=use_parallel or use_parallel_all)
            
            # Simulate
            print(f"\nSimulating {n_sim} agents for {edu_type}...")
            results = model.simulate(n_sim=n_sim, seed=42, parallel=use_parallel or use_parallel_all)
            (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
             ui_sim, m_sim, oop_m_sim, gov_m_sim,
             tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
             pension_sim, retired_sim) = results
            
            # Compute statistics
            unemployment_rate = np.mean(~employed_sim & ~retired_sim)
            retirement_rate = np.mean(retired_sim)
            total_tax = tax_c_sim + tax_l_sim + tax_p_sim + tax_k_sim
            total_gov_spending = gov_m_sim + ui_sim + pension_sim
            
            print(f"\n{'-'*70}")
            print(f"RESULTS FOR {edu_type.upper()} EDUCATION (MINIMAL CONFIG):")
            print(f"{'-'*70}")
            print(f"  Mean assets:              {np.mean(a_sim):.2f}")
            print(f"  Mean consumption:         {np.mean(c_sim):.3f}")
            print(f"  Mean effective income:    {np.mean(effective_y_sim):.3f}")
            print(f"  Mean avg earnings:        {np.mean(avg_earnings_sim):.3f}")
            
            if np.any(retired_sim):
                print(f"  Mean pension (retired):   {np.mean(pension_sim[retired_sim]):.3f}")
            else:
                print(f"  Mean pension (retired):   N/A (no retirement periods)")
            
            print(f"  Unemployment rate:        {unemployment_rate:.2%}")
            print(f"  Retirement rate:          {retirement_rate:.2%}")
            print(f"  Mean total taxes:         {np.mean(total_tax):.4f}")
            print(f"  Mean gov spending:        {np.mean(total_gov_spending):.4f}")
            print(f"  Mean UI benefits:         {np.mean(ui_sim):.4f}")
            print(f"  Mean gov health spending: {np.mean(gov_m_sim):.4f}")
            
            # Age-specific statistics
            print(f"\n  Age-specific means (age 20 = lifecycle age 0):")
            ages_to_check = list(range(config.T))  # All ages in short lifecycle
            for age_idx in ages_to_check:
                if age_idx < len(a_sim):
                    is_retired_age = age_idx >= (config.retirement_age - config.current_age)
                    status = "RETIRED" if is_retired_age else "WORKING"
                    print(f"    Age {20+age_idx} (lifecycle {age_idx}, {status}):")
                    print(f"      Assets:      {np.mean(a_sim[age_idx, :]):.2f}")
                    print(f"      Consumption: {np.mean(c_sim[age_idx, :]):.3f}")
                    print(f"      Employment:  {np.mean(employed_sim[age_idx, :]):.2%}")
                    if is_retired_age and np.any(pension_sim[age_idx, :] > 0):
                        print(f"      Pension:     {np.mean(pension_sim[age_idx, :]):.3f}")
                    else:
                        print(f"      Income:      {np.mean(effective_y_sim[age_idx, :]):.3f}")
            
            # Store results for plotting
            result_dict = {
                'a_sim': a_sim,
                'c_sim': c_sim,
                'effective_y_sim': effective_y_sim,
                'pension_sim': pension_sim,
                'retired_sim': retired_sim,
                'avg_earnings_sim': avg_earnings_sim,
                'employed_sim': employed_sim,
                'ui_sim': ui_sim,
                'total_tax': total_tax,
                'total_gov_spending': total_gov_spending,
                'config': config
            }
            
            all_results[edu_type] = result_dict
        
        print(f"\n{'='*70}")
        print("GENERATING PLOTS (MINIMAL CONFIG)")
        print('='*70)
        
        # Create comprehensive plots (simplified for single education type)
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Lifecycle Profiles - Minimal Configuration Test', fontsize=16, fontweight='bold')
        
        edu_type = 'medium'
        res = all_results[edu_type]
        T_sim = res['a_sim'].shape[0]
        ages = np.arange(20, 20 + T_sim)
        retirement_age = res['config'].retirement_age + 20
        
        # Plot mean paths (no multiple education types, so simpler)
        axes[0, 0].plot(ages, res['a_sim'].mean(axis=1), 
                       label='Mean', color='C0', linewidth=2)
        axes[0, 0].fill_between(ages, 
                               res['a_sim'].mean(axis=1) - res['a_sim'].std(axis=1),
                               res['a_sim'].mean(axis=1) + res['a_sim'].std(axis=1),
                               alpha=0.2, color='C0', label='±1 Std Dev')
        
        axes[0, 1].plot(ages, res['c_sim'].mean(axis=1), 
                       label='Mean', color='C0', linewidth=2)
        axes[0, 1].fill_between(ages,
                               res['c_sim'].mean(axis=1) - res['c_sim'].std(axis=1),
                               res['c_sim'].mean(axis=1) + res['c_sim'].std(axis=1),
                               alpha=0.2, color='C0')
        
        axes[1, 0].plot(ages, res['effective_y_sim'].mean(axis=1), 
                       label='Mean', color='C0', linewidth=2)
        
        axes[1, 1].plot(ages, res['pension_sim'].mean(axis=1), 
                       label='Mean', color='C0', linewidth=2)
        
        axes[2, 0].plot(ages, res['total_tax'].mean(axis=1), 
                       label='Mean', color='C0', linewidth=2)
        
        axes[2, 1].plot(ages, res['total_gov_spending'].mean(axis=1), 
                       label='Mean', color='C0', linewidth=2)
        
        # Add retirement age vertical line to all plots
        for ax in axes.flat:
            ax.axvline(x=retirement_age, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label='Retirement Age' if ax == axes[0, 0] else '')
        
        # Formatting
        axes[0, 0].set_title('Assets', fontweight='bold')
        axes[0, 0].set_ylabel('Assets')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Consumption', fontweight='bold')
        axes[0, 1].set_ylabel('Consumption')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Effective Income (Labor)', fontweight='bold')
        axes[1, 0].set_ylabel('Income')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Pension Benefits', fontweight='bold')
        axes[1, 1].set_ylabel('Pension')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_title('Total Taxes Paid', fontweight='bold')
        axes[2, 0].set_ylabel('Taxes')
        axes[2, 0].set_xlabel('Age')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].set_title('Total Government Spending', fontweight='bold')
        axes[2, 1].set_ylabel('Gov. Spending')
        axes[2, 1].set_xlabel('Age')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lifecycle_profiles_minimal_test.png', dpi=300, bbox_inches='tight')
        print("  Saved: lifecycle_profiles_minimal_test.png")
        
        # Additional plot: Employment and earnings details
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        fig2.suptitle('Employment and Earnings Details - Minimal Config', fontsize=16, fontweight='bold')
        
        # Average earnings history
        axes2[0, 0].plot(ages, res['avg_earnings_sim'].mean(axis=1),
                       label='Mean', color='C0', linewidth=2)
        axes2[0, 0].axvline(x=retirement_age, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7, label='Retirement Age')
        
        # Employment rate
        employment_rate = res['employed_sim'].mean(axis=1)
        axes2[0, 1].plot(ages, employment_rate,
                       label='Employment Rate', color='C0', linewidth=2)
        axes2[0, 1].axvline(x=retirement_age, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
        
        # UI benefits
        axes2[1, 0].plot(ages, res['ui_sim'].mean(axis=1),
                       label='UI Benefits', color='C0', linewidth=2)
        axes2[1, 0].axvline(x=retirement_age, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
        
        # Savings rate
        capital_income = res['config'].r_default * res['a_sim']
        labor_income_flow = res['effective_y_sim'] + res['pension_sim']
        total_income = capital_income + labor_income_flow
        savings = total_income - res['c_sim']
        savings_rate = savings / (total_income + 1e-10)
        axes2[1, 1].plot(ages, savings_rate.mean(axis=1),
                       label='Savings Rate', color='C0', linewidth=2)
        axes2[1, 1].axvline(x=retirement_age, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
        
        axes2[0, 0].set_title('Average Earnings History', fontweight='bold')
        axes2[0, 0].set_ylabel('Avg Earnings')
        axes2[0, 0].legend()
        axes2[0, 0].grid(True, alpha=0.3)
        
        axes2[0, 1].set_title('Employment Rate', fontweight='bold')
        axes2[0, 1].set_ylabel('Employment Rate')
        axes2[0, 1].legend()
        axes2[0, 1].grid(True, alpha=0.3)
        
        axes2[1, 0].set_title('UI Benefits', fontweight='bold')
        axes2[1, 0].set_ylabel('UI Benefits')
        axes2[1, 0].set_xlabel('Age')
        axes2[1, 0].legend()
        axes2[1, 0].grid(True, alpha=0.3)
        
        axes2[1, 1].set_title('Savings Rate', fontweight='bold')
        axes2[1, 1].set_ylabel('Savings Rate')
        axes2[1, 1].set_xlabel('Age')
        axes2[1, 1].legend()
        axes2[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('employment_earnings_minimal_test.png', dpi=300, bbox_inches='tight')
        print("  Saved: employment_earnings_minimal_test.png")
        
        # Show plots
        plt.show()
        
        print(f"\n{'='*70}")
        print("MINIMAL CONFIGURATION TESTING COMPLETE")
        print('='*70)
    
    else:
        print("Usage: python lifecycle_perfect_foresight.py --test [--parallel] [--parallel-all]")
        print("  --test         Run minimal configuration testing suite")
        print("                 (n_h=1, n_y=2, T=10, single education group)")
        print("  --parallel     Use parallel processing within education type")
        print("  --parallel-all Maximum parallelism (same as --parallel for single education type)")