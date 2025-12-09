import numpy as np
import matplotlib.pyplot as plt
from lifecycle_perfect_foresight import LifecycleModelPerfectForesight
import os
from datetime import datetime
from numba import njit


class OLGTransition:
    """
    Overlapping Generations Economy for Transition Dynamics with Perfect Foresight.
    
    Takes exogenous interest rate path and simulates the economy's response.
    All agents know the entire future path of interest rates and wages.
    """
    
    def __init__(self,
                 # Individual parameters
                 T=60,
                 beta=0.96,
                 gamma=2.0,
                 a_min=0.0,
                 a_max=50.0,
                 n_a=100,
                 n_y=5,
                 n_h=3,
                 # Production parameters
                 alpha=0.33,
                 delta=0.05,
                 A=1.0,
                 # Demographic parameters
                 pop_growth=0.01,
                 birth_year=1960,
                 current_year=2020,
                 # Output settings
                 output_dir='output'):
        
        # Store individual parameters
        self.T = T
        self.beta = beta
        self.gamma = gamma
        self.a_min = a_min
        self.a_max = a_max
        self.n_a = n_a
        self.n_y = n_y
        self.n_h = n_h
        
        # Production parameters
        self.alpha = alpha
        self.delta = delta
        self.A = A
        
        # Demographics
        self.pop_growth = pop_growth
        self.birth_year = birth_year
        self.current_year = current_year
        self.n_cohorts = T
        
        # Output directory
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create cohort sizes (demographic structure)
        self.cohort_sizes = self._create_cohort_sizes()
        
        # Age-efficiency profile
        self.efficiency_profile = self._create_efficiency_profile()
        
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
    
    def _create_efficiency_profile(self):
        """Create age-efficiency profile for labor."""
        return self._efficiency_profile_njit(self.T)
    
    @staticmethod
    @njit
    def _efficiency_profile_njit(T):
        """JIT-compiled efficiency profile calculation."""
        efficiency = np.zeros(T)
        for age in range(T):
            actual_age = 20 + age
            efficiency[age] = np.exp(-((actual_age - 50) / 20) ** 2)
        return efficiency
    
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
    def _aggregate_capital_labor_njit(assets_by_age, labor_by_age, cohort_sizes, efficiency_profile):
        """
        JIT-compiled aggregation of capital and labor.
        
        Parameters:
        -----------
        assets_by_age : array (T,)
            Average assets for each age group
        labor_by_age : array (T,)
            Average effective labor for each age group
        cohort_sizes : array (T,)
            Population weights
        efficiency_profile : array (T,)
            Age-efficiency profile
        """
        T = len(cohort_sizes)
        K = 0.0
        L = 0.0
        
        for age in range(T):
            K += cohort_sizes[age] * assets_by_age[age]
            effective_labor = labor_by_age[age] * efficiency_profile[age]
            L += cohort_sizes[age] * effective_labor
        
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
    
    def solve_cohort_problems(self, r_path, w_path, verbose=False):
        """
        Solve lifecycle problems for all cohorts given full price paths.
        
        Parameters:
        -----------
        r_path : array
            Full interest rate path (length >= T_transition + T)
        w_path : array
            Full wage path (length >= T_transition + T)
        """
        if verbose:
            print("\nSolving cohort lifecycle problems with perfect foresight...")
        
        self.cohort_models = {}
        
        # For each period in transition
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            
            self.cohort_models[t] = {}
            
            # For each age group alive in period t
            for age in range(self.T):
                birth_period = t - age
                
                # Extract the price path this cohort faces
                remaining_life = self.T - age
                
                if birth_period >= 0:
                    # Cohort born during or after transition starts
                    cohort_r = r_path[birth_period:birth_period + self.T]
                    cohort_w = w_path[birth_period:birth_period + self.T]
                else:
                    # Cohort born before transition
                    pre_periods = -birth_period
                    r_initial = r_path[0]
                    w_initial = w_path[0]
                    
                    cohort_r = np.concatenate([
                        np.ones(pre_periods) * r_initial,
                        r_path[:remaining_life]
                    ])
                    cohort_w = np.concatenate([
                        np.ones(pre_periods) * w_initial,
                        w_path[:remaining_life]
                    ])
                
                # Solve this cohort's problem
                model = LifecycleModelPerfectForesight(
                    T=self.T,
                    beta=self.beta,
                    gamma=self.gamma,
                    r_path=cohort_r,
                    w_path=cohort_w,
                    a_min=self.a_min,
                    a_max=self.a_max,
                    n_a=self.n_a,
                    n_y=self.n_y,
                    n_h=self.n_h,
                    current_age=age
                )
                
                model.solve(verbose=False)
                self.cohort_models[t][age] = model
        
        if verbose:
            print("All cohort problems solved!")
    
    def compute_aggregates(self, t, n_sim=10000):
        """
        Compute aggregate capital and labor for period t.
        Uses cohort-specific models and JIT-compiled aggregation.
        """
        # Preallocate arrays for JIT aggregation
        assets_by_age = np.zeros(self.T)
        labor_by_age = np.zeros(self.T)
        
        for age in range(self.T):
            model = self.cohort_models[t][age]
            
            # Simulate just one period for this cohort
            a_sim, c_sim, y_sim, h_sim, effective_y_sim = model.simulate(
                T_sim=1,
                n_sim=n_sim,
                seed=42 + t * 100 + age
            )
            
            # Average across simulations
            assets_by_age[age] = np.mean(a_sim[0, :])
            labor_by_age[age] = np.mean(effective_y_sim[0, :])
        
        # Use JIT-compiled aggregation
        K, L = self._aggregate_capital_labor_njit(
            assets_by_age,
            labor_by_age,
            self.cohort_sizes,
            self.efficiency_profile
        )
        
        return K, L
    
    def simulate_transition(self, 
                           r_path,
                           w_path=None,
                           n_sim=10000,
                           verbose=True):
        """
        Simulate transition dynamics with exogenous interest rate path and perfect foresight.
        
        Parameters:
        -----------
        r_path : array-like
            Exogenous interest rate path over transition
        w_path : array-like, optional
            Exogenous wage path. If None, computed from production function
            given implied aggregates
        n_sim : int
            Number of simulations per cohort
        verbose : bool
            Print progress
        """
        r_path = np.array(r_path)
        self.T_transition = len(r_path)
        
        # Extend paths to cover all cohorts (pad with final values)
        r_path_full = np.concatenate([r_path, np.ones(self.T) * r_path[-1]])
        
        if w_path is None:
            # Initial guess for wage path from production function
            K_init = 10.0
            L_init = 1.0
            _, w_init = self.factor_prices(K_init, L_init)
            w_path_full = np.ones(len(r_path_full)) * w_init
            compute_wages = True
        else:
            w_path = np.array(w_path)
            w_path_full = np.concatenate([w_path, np.ones(self.T) * w_path[-1]])
            compute_wages = False
        
        if verbose:
            print("=" * 60)
            print("Simulating OLG Transition with Exogenous Interest Rates")
            print("=" * 60)
            print(f"Transition periods: {self.T_transition}")
            print(f"Initial r: {r_path[0]:.4f}")
            print(f"Final r: {r_path[-1]:.4f}")
        
        # Solve all cohort problems with current price paths
        self.solve_cohort_problems(r_path_full, w_path_full, verbose=verbose)
        
        # Compute implied aggregates for each period
        if verbose:
            print("\nComputing aggregate quantities...")
        
        K_path = np.zeros(self.T_transition)
        L_path = np.zeros(self.T_transition)
        
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            K_path[t], L_path[t] = self.compute_aggregates(t, n_sim=n_sim)
        
        # If wages were not provided, compute them from production function using JIT
        if compute_wages:
            if verbose:
                print("\nComputing implied wages from production function...")
            w_path_computed = self._compute_wage_path_njit(
                K_path, L_path, self.alpha, self.delta, self.A
            )
            w_path_full[:self.T_transition] = w_path_computed
        
        # Compute output using JIT
        Y_path = self._compute_output_path_njit(K_path, L_path, self.alpha, self.A)
        
        # Store results
        self.r_path = r_path
        self.w_path = w_path_full[:self.T_transition]
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
            print(f"  Average K/Y: {np.mean(K_path/Y_path):.4f}")
        
        return {
            'r': self.r_path,
            'w': self.w_path,
            'K': self.K_path,
            'L': self.L_path,
            'Y': self.Y_path
        }
    
    def plot_transition(self, save=True, show=True, filename=None):
        """Plot transition dynamics."""
        if self.r_path is None:
            raise ValueError("Must simulate transition first")
        
        fig = plt.figure(figsize=(15, 10))
        periods = np.arange(self.T_transition)
        
        # 1. Interest rate
        plt.subplot(3, 3, 1)
        plt.plot(periods, self.r_path * 100, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Interest Rate (%)')
        plt.title('Interest Rate Path (Exogenous)')
        plt.grid(True, alpha=0.3)
        
        # 2. Wage
        plt.subplot(3, 3, 2)
        plt.plot(periods, self.w_path, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Wage')
        plt.title('Wage Path')
        plt.grid(True, alpha=0.3)
        
        # 3. Capital
        plt.subplot(3, 3, 3)
        plt.plot(periods, self.K_path, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Capital')
        plt.title('Aggregate Capital')
        plt.grid(True, alpha=0.3)
        
        # 4. Labor
        plt.subplot(3, 3, 4)
        plt.plot(periods, self.L_path, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Labor')
        plt.title('Aggregate Labor')
        plt.grid(True, alpha=0.3)
        
        # 5. Output
        plt.subplot(3, 3, 5)
        plt.plot(periods, self.Y_path, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Output')
        plt.title('Aggregate Output')
        plt.grid(True, alpha=0.3)
        
        # 6. Capital-Output ratio
        plt.subplot(3, 3, 6)
        K_Y_ratio = self.K_path / self.Y_path
        plt.plot(periods, K_Y_ratio, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('K/Y Ratio')
        plt.title('Capital-Output Ratio')
        plt.grid(True, alpha=0.3)
        
        # 7. Capital growth rate
        plt.subplot(3, 3, 7)
        K_growth = np.diff(self.K_path) / self.K_path[:-1] * 100
        plt.plot(periods[1:], K_growth, linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Period')
        plt.ylabel('Growth Rate (%)')
        plt.title('Capital Growth Rate')
        plt.grid(True, alpha=0.3)
        
        # 8. Output growth rate
        plt.subplot(3, 3, 8)
        Y_growth = np.diff(self.Y_path) / self.Y_path[:-1] * 100
        plt.plot(periods[1:], Y_growth, linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Period')
        plt.ylabel('Growth Rate (%)')
        plt.title('Output Growth Rate')
        plt.grid(True, alpha=0.3)
        
        # 9. Factor price ratio
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
    
    def plot_lifecycle_comparison(self, periods_to_plot, n_sim=5000, 
                                  save=True, show=True, filename=None):
        """Compare lifecycle profiles at different points in transition."""
        if self.cohort_models is None:
            raise ValueError("Must simulate transition first")
        
        fig = plt.figure(figsize=(15, 8))
        ages = np.arange(self.T) + 20
        
        for t in periods_to_plot:
            if t >= self.T_transition:
                continue
            
            model = self.cohort_models[t][0]  # Age 0 cohort in period t
            a_sim, c_sim, _, _, _ = model.simulate(T_sim=self.T, n_sim=n_sim, seed=42)
            
            # Assets
            plt.subplot(2, 3, 1)
            plt.plot(ages, np.mean(a_sim, axis=1), 
                    label=f't={t}, r={self.r_path[t]:.3f}', linewidth=2)
            
            # Consumption
            plt.subplot(2, 3, 2)
            plt.plot(ages, np.mean(c_sim, axis=1), 
                    label=f't={t}', linewidth=2)
            
            # Savings rate
            plt.subplot(2, 3, 3)
            income_flow = model.simulate(T_sim=self.T, n_sim=n_sim, seed=42)[4] + \
                         self.r_path[t] * a_sim
            savings_rate = (income_flow - c_sim) / (income_flow + 1e-10)
            plt.plot(ages, np.mean(savings_rate, axis=1), 
                    label=f't={t}', linewidth=2)
        
        plt.subplot(2, 3, 1)
        plt.xlabel('Age')
        plt.ylabel('Assets')
        plt.title('Asset Profiles Over Transition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.xlabel('Age')
        plt.ylabel('Consumption')
        plt.title('Consumption Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.xlabel('Age')
        plt.ylabel('Savings Rate')
        plt.title('Savings Rate Profiles')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lifecycle_comparison_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()


# Example usage
if __name__ == "__main__":
    print("Initializing OLG Transition Model...")
    
    # Create economy
    economy = OLGTransition(
        T=40,
        beta=0.98,
        gamma=2.0,
        alpha=0.33,
        delta=0.05,
        A=1.0,
        a_min=-2.0,
        a_max=50.0,
        n_a=30,
        n_y=3,
        n_h=3,
        pop_growth=0.01,
        birth_year=1960,
        current_year=2020,
        output_dir='output'
    )
    
    # Define exogenous interest rate path
    T_transition = 15
    r_initial = 0.04
    r_final = 0.02
    
    # Smooth transition
    periods = np.arange(T_transition)
    r_path = r_initial + (r_final - r_initial) * (1 - np.exp(-periods / 5))
    
    print(f"\nSimulating transition from r={r_initial:.3f} to r={r_final:.3f}")
    
    # Simulate transition
    import time
    start = time.time()
    results = economy.simulate_transition(
        r_path=r_path,
        w_path=None,
        n_sim=500,
        verbose=True
    )
    end = time.time()
    
    print(f"\nTotal simulation time: {end - start:.2f} seconds")
    
    # Plot results
    economy.plot_transition(save=True, show=True)
    
    # Compare lifecycle profiles
    economy.plot_lifecycle_comparison(
        periods_to_plot=[0, 5, 10, 14],
        n_sim=500,
        save=True,
        show=True
    )