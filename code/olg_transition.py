import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from lifecycle_perfect_foresight import LifecycleModelPerfectForesight, LifecycleConfig
import os
from datetime import datetime
from numba import njit


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
        """Solve lifecycle problems for all cohorts given full price paths."""
        if verbose:
            print("\nSolving cohort lifecycle problems with perfect foresight...")
            print(f"  Education types: {list(self.education_shares.keys())}")
        
        self.cohort_models = {}
        
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            
            self.cohort_models[t] = {}
            
            for edu_type in self.education_shares.keys():
                self.cohort_models[t][edu_type] = {}
                
                for age in range(self.T):
                    birth_period = t - age
                    remaining_life = self.T - age
                    
                    if birth_period >= 0:
                        cohort_r = r_path[birth_period:birth_period + self.T]
                        cohort_w = w_path[birth_period:birth_period + self.T]
                        cohort_tau_c = tau_c_path[birth_period:birth_period + self.T] if tau_c_path is not None else None
                        cohort_tau_l = tau_l_path[birth_period:birth_period + self.T] if tau_l_path is not None else None
                        cohort_tau_p = tau_p_path[birth_period:birth_period + self.T] if tau_p_path is not None else None
                        cohort_tau_k = tau_k_path[birth_period:birth_period + self.T] if tau_k_path is not None else None
                        cohort_pension = pension_replacement_path[birth_period:birth_period + self.T] if pension_replacement_path is not None else None
                    else:
                        pre_periods = -birth_period
                        r_initial = r_path[0]
                        w_initial = w_path[0]
                        
                        cohort_r = np.concatenate([np.ones(pre_periods) * r_initial, r_path[:remaining_life]])
                        cohort_w = np.concatenate([np.ones(pre_periods) * w_initial, w_path[:remaining_life]])
                        
                        if tau_c_path is not None:
                            cohort_tau_c = np.concatenate([np.ones(pre_periods) * tau_c_path[0], tau_c_path[:remaining_life]])
                        else:
                            cohort_tau_c = None
                        
                        if tau_l_path is not None:
                            cohort_tau_l = np.concatenate([np.ones(pre_periods) * tau_l_path[0], tau_l_path[:remaining_life]])
                        else:
                            cohort_tau_l = None
                        
                        if tau_p_path is not None:
                            cohort_tau_p = np.concatenate([np.ones(pre_periods) * tau_p_path[0], tau_p_path[:remaining_life]])
                        else:
                            cohort_tau_p = None
                        
                        if tau_k_path is not None:
                            cohort_tau_k = np.concatenate([np.ones(pre_periods) * tau_k_path[0], tau_k_path[:remaining_life]])
                        else:
                            cohort_tau_k = None
                        
                        if pension_replacement_path is not None:
                            cohort_pension = np.concatenate([np.ones(pre_periods) * pension_replacement_path[0], pension_replacement_path[:remaining_life]])
                        else:
                            cohort_pension = None
                    
                    config = LifecycleConfig(
                        T=self.T,
                        beta=self.beta,
                        gamma=self.gamma,
                        current_age=age,
                        education_type=edu_type,
                        n_a=self.n_a,
                        n_y=self.n_y,
                        n_h=self.n_h,
                        retirement_age=self.retirement_age,
                        r_path=cohort_r,
                        w_path=cohort_w,
                        tau_c_path=cohort_tau_c,
                        tau_l_path=cohort_tau_l,
                        tau_p_path=cohort_tau_p,
                        tau_k_path=cohort_tau_k,
                        pension_replacement_path=cohort_pension
                    )
                    
                    model = LifecycleModelPerfectForesight(config)
                    model.solve(verbose=False)
                    
                    self.cohort_models[t][edu_type][age] = model
        
        if verbose:
            print("All cohort problems solved!")
    
    def compute_aggregates(self, t, n_sim=10000):
        """Compute aggregate capital and labor for period t."""
        n_edu = len(self.education_shares)
        education_types = list(self.education_shares.keys())
        
        assets_by_age_edu = np.zeros((n_edu, self.T))
        labor_by_age_edu = np.zeros((n_edu, self.T))
        education_shares_array = np.array([self.education_shares[edu] for edu in education_types])
        
        for edu_idx, edu_type in enumerate(education_types):
            for age in range(self.T):
                model = self.cohort_models[t][edu_type][age]
                
                results = model.simulate(T_sim=1, n_sim=n_sim, seed=42 + t * 100 + age)
                
                (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
                 ui_sim, m_sim, oop_m_sim, gov_m_sim,
                 tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
                 pension_sim, retired_sim) = results
                
                assets_by_age_edu[edu_idx, age] = np.mean(a_sim[0, :])
                labor_by_age_edu[edu_idx, age] = np.mean(effective_y_sim[0, :])
        
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
        """Simulate transition dynamics with exogenous interest rate path."""
        r_path = np.array(r_path)
        self.T_transition = len(r_path)
        
        r_path_full = np.concatenate([r_path, np.ones(self.T) * r_path[-1]])
        
        if w_path is None:
            K_init = 10.0
            L_init = 1.0
            _, w_init = self.factor_prices(K_init, L_init)
            w_path_full = np.ones(len(r_path_full)) * w_init
            compute_wages = True
        else:
            w_path = np.array(w_path)
            w_path_full = np.concatenate([w_path, np.ones(self.T) * w_path[-1]])
            compute_wages = False
        
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
            print("=" * 60)
            print("Simulating OLG Transition with Exogenous Interest Rates")
            print("=" * 60)
            print(f"Transition periods: {self.T_transition}")
            print(f"Initial r: {r_path[0]:.4f}")
            print(f"Final r: {r_path[-1]:.4f}")
            print(f"Retirement age: {self.retirement_age}")
            print(f"Education groups: {list(self.education_shares.keys())}")
        
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
        
        K_path = np.zeros(self.T_transition)
        L_path = np.zeros(self.T_transition)
        
        for t in range(self.T_transition):
            if verbose and (t % 10 == 0 or t == self.T_transition - 1):
                print(f"  Period {t + 1}/{self.T_transition}")
            K_path[t], L_path[t] = self.compute_aggregates(t, n_sim=n_sim)
        
        if compute_wages:
            if verbose:
                print("\nComputing implied wages from production function...")
            w_path_computed = self._compute_wage_path_njit(
                K_path, L_path, self.alpha, self.delta, self.A
            )
            w_path_full[:self.T_transition] = w_path_computed
        
        Y_path = self._compute_output_path_njit(K_path, L_path, self.alpha, self.A)
        
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
        
        return {'r': self.r_path, 'w': self.w_path, 'K': self.K_path, 'L': self.L_path, 'Y': self.Y_path}
    
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
                                  n_sim=5000, save=True, show=True, filename=None):
        """Compare lifecycle profiles at different points in transition."""
        if self.cohort_models is None:
            raise ValueError("Must simulate transition first")
        
        fig = plt.figure(figsize=(15, 12))
        ages = np.arange(self.T) + 20
        
        for t in periods_to_plot:
            if t >= self.T_transition:
                continue
            
            model = self.cohort_models[t][edu_type][0]
            results = model.simulate(T_sim=self.T, n_sim=n_sim, seed=42)
            
            (a_sim, c_sim, y_sim, h_sim, h_idx_sim, effective_y_sim, employed_sim, 
             ui_sim, m_sim, oop_m_sim, gov_m_sim,
             tax_c_sim, tax_l_sim, tax_p_sim, tax_k_sim, avg_earnings_sim,
             pension_sim, retired_sim) = results
            
            plt.subplot(3, 3, 1)
            plt.plot(ages, np.mean(a_sim, axis=1), 
                    label=f't={t}, r={self.r_path[t]:.3f}', linewidth=2)
            
            plt.subplot(3, 3, 2)
            plt.plot(ages, np.mean(c_sim, axis=1), 
                    label=f't={t}', linewidth=2)
            
            plt.subplot(3, 3, 3)
            plt.plot(ages, np.mean(effective_y_sim, axis=1), 
                    label=f't={t}', linewidth=2)
            
            plt.subplot(3, 3, 4)
            plt.plot(ages, np.mean(pension_sim, axis=1), 
                    label=f't={t}', linewidth=2)
            
            plt.subplot(3, 3, 5)
            plt.plot(ages, np.mean(employed_sim, axis=1), 
                    label=f't={t}', linewidth=2)
            
            plt.subplot(3, 3, 6)
            total_tax = tax_c_sim + tax_l_sim + tax_p_sim + tax_k_sim
            plt.plot(ages, np.mean(total_tax, axis=1), 
                    label=f't={t}', linewidth=2)
            
            plt.subplot(3, 3, 7)
            capital_income = self.r_path[t] * a_sim
            labor_income_flow = effective_y_sim * self.w_path[t] + pension_sim
            total_income = capital_income + labor_income_flow
            savings = total_income - c_sim
            savings_rate = savings / (total_income + 1e-10)
            plt.plot(ages, np.mean(savings_rate, axis=1), 
                    label=f't={t}', linewidth=2)
            
            plt.subplot(3, 3, 8)
            plt.plot(ages, np.mean(oop_m_sim, axis=1), 
                    label=f't={t} (OOP)', linewidth=2, linestyle='-')
            
            plt.subplot(3, 3, 9)
            plt.plot(ages, np.mean(avg_earnings_sim, axis=1), 
                    label=f't={t}', linewidth=2)
        
        retirement_age_actual = 20 + self.retirement_age
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            plt.axvline(x=retirement_age_actual, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.5)
        
        plt.subplot(3, 3, 1)
        plt.xlabel('Age')
        plt.ylabel('Assets')
        plt.title('Asset Profiles Over Transition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        plt.xlabel('Age')
        plt.ylabel('Consumption')
        plt.title('Consumption Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        plt.xlabel('Age')
        plt.ylabel('Labor Income')
        plt.title('Labor Income Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 4)
        plt.xlabel('Age')
        plt.ylabel('Pension')
        plt.title('Pension Benefits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5)
        plt.xlabel('Age')
        plt.ylabel('Employment Rate')
        plt.title('Employment Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 6)
        plt.xlabel('Age')
        plt.ylabel('Total Taxes')
        plt.title('Total Tax Payments')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 7)
        plt.xlabel('Age')
        plt.ylabel('Savings Rate')
        plt.title('Savings Rate Profiles')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        plt.xlabel('Age')
        plt.ylabel('OOP Health Exp.')
        plt.title('Out-of-Pocket Health Expenditures')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 9)
        plt.xlabel('Age')
        plt.ylabel('Avg Earnings')
        plt.title('Average Earnings History')
        plt.legend()
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
    """Return minimal configuration for fast testing."""
    config = LifecycleConfig(
        T=10,
        beta=0.98,
        gamma=2.0,
        n_a=10,
        n_y=2,
        n_h=2,
        retirement_age=7,
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
    print("\nTest configuration:")
    print("  T = 10 periods (vs 40 in full run)")
    print("  n_a = 10 asset grid points (vs 30 in full run)")
    print("  n_y = 2 income states (vs 3 in full run)")
    print("  n_h = 2 health states (vs 3 in full run)")
    print("  T_transition = 5 periods (vs 15 in full run)")
    print("  n_sim = 100 simulations (vs 500 in full run)")
    print("  Education groups = 1 (medium only)")
    print()
    
    config = get_test_config()
    
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
    
    T_transition = 5
    r_initial = 0.04
    r_final = 0.02
    
    periods = np.arange(T_transition)
    r_path = r_initial + (r_final - r_initial) * (1 - np.exp(-periods / 2))
    
    print(f"Simulating transition from r={r_initial:.3f} to r={r_final:.3f}")
    print(f"Transition periods: {T_transition}")
    
    start = time.time()
    results = economy.simulate_transition(
        r_path=r_path,
        w_path=None,
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
        periods_to_plot=[0, 2, 4],
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