# Code Issues and Bugs

This document tracks identified issues in `olg_transition.py` and `lifecycle_perfect_foresight.py`.

---

## Critical Issues

### 1. Pensions NOT included in value function

**Status**: [x] Complete

**Location**: `lifecycle_perfect_foresight.py:571-590` (`_solve_period` method)

**Description**: When solving the retirement periods, pension income is not included in the budget constraint. The value function is solved assuming retirees have zero labor income:

```python
if is_retired:
    gross_labor_income = 0.0
    ui_benefit = 0.0
    after_tax_labor_income = 0.0  # NO PENSION ADDED!
```

However, in simulation (`lifecycle_perfect_foresight.py:816-817`), pensions ARE paid:

```python
pension_sim[t_sim, i] = pension_replacement * avg_earnings_at_retirement[i]
```

**Impact**: Agents solve their optimal savings problem thinking they will have zero income in retirement, causing systematic over-saving. The policy functions are inconsistent with the actual income process.

**Fix**: Redefine pensions to be based on **final-period earnings** using `y_last` (already in state space) instead of career-average earnings.

**Design change**:
```
OLD: pension = replacement_rate × avg_earnings_at_retirement  (requires tracking avg_earnings)
NEW: pension = replacement_rate × w[retirement_age-1] × y_grid[y_last]  (uses existing state)
```

**Why this works**: The state variable `y_last` (last period's income state) is already tracked for UI benefit calculation. At retirement, `y_last` equals the income state in the final working period. Due to income persistence (ρ ≈ 0.97), this correlates with lifetime earnings.

**Changes required**:

1. `_solve_terminal_period`: Add pension to budget if T-1 is a retirement period
2. `_solve_period` (retirement periods): Add pension = replacement_rate × w_retirement × y_grid[y_last] to budget
3. `_solve_period_wrapper`: Same changes for parallel case
4. `_simulate_sequential`:
   - Compute pension from y_last instead of avg_earnings
   - Keep y_last frozen during retirement (don't set to 0)
5. Remove avg_earnings tracking for pension purposes (can keep for diagnostics)

**How expectations work for working periods**: No explicit expected pension calculation needed. Backward induction automatically embeds pension expectations:
- Retirement periods have pension in budget, making V[retired, y_last=high] > V[retired, y_last=low]
- At retirement_age-1, continuation value uses current y as next period's y_last
- Earlier periods: high y today → likely high y at retirement (persistence) → higher continuation value

---

### 2. Policy indexing bug (latent)

**Status**: [x] Complete

**Location**: `lifecycle_perfect_foresight.py:862,891` (`_simulate_sequential` method)

**Description**: Policy functions were indexed by simulation time `t_sim` instead of `lifecycle_age`:

```python
c_sim[t_sim, i] = self.c_policy[t_sim, i_a[i], i_y[i], i_h[i], i_y_last[i]]
# ...
i_a[i] = self.a_policy[t_sim, i_a[i], i_y[i], i_h[i], i_y_last[i]]
```

But `lifecycle_age = self.current_age + t_sim` (line 816). When `current_age > 0`, the policy index should be `lifecycle_age`, not `t_sim`.

**Impact**: Currently latent because the main aggregation uses `birth_cohort_solutions` with `current_age=0`. But any code using `cohort_models[t][edu][age]` with `age > 0` would get wrong policies.

**Fix**: Changed indexing to use `lifecycle_age`:
```python
c_sim[t_sim, i] = self.c_policy[lifecycle_age, i_a[i], i_y[i], i_h[i], i_y_last[i]]
i_a[i] = self.a_policy[lifecycle_age, i_a[i], i_y[i], i_h[i], i_y_last[i]]
```

---

### 3. Payroll tax applied to pensions

**Status**: [x] Complete

**Location**: `lifecycle_perfect_foresight.py:876`

**Description**: Payroll tax (`tau_p`) was incorrectly applied to pension benefits:

```python
tax_p_sim[t_sim, i] = self.tau_p_path[lifecycle_age] * pension_sim[t_sim, i]
```

Payroll taxes are typically levied on wages/earnings, not on pension benefits received.

**Impact**: Artificially reduces net pension income and overstates payroll tax revenue.

**Fix**: Payroll tax now only applies to wage income:
```python
wage_income = self.w_path[lifecycle_age] * y_sim[t_sim, i] * h_sim[t_sim, i]
tax_p_sim[t_sim, i] = self.tau_p_path[lifecycle_age] * wage_income  # wages only, not pensions
```

This fix is consistently applied in:
- Terminal period solving (line 563)
- Period solving (line 619)
- Simulation (line 876)
- Parallel wrapper (line 1004)

---

## Moderate Issues

### 4. Inefficient `_slice_means_njit` usage

**Status**: [x] Complete

**Location**: `olg_transition.py:703-724` (`_period_cross_section` method)

**Description**: Inside a loop over ages, `_slice_means_njit` was called which computes means for ALL T ages, but only one age's mean was used:

```python
for age in range(self.T):
    # ...
    (a_mean, labor_mean, ...) = self._slice_means_njit(...)  # computes T means
    assets_by_age_edu[edu_idx, age] = float(a_mean[age])     # uses 1 mean
```

**Impact**: O(T² × n_edu × n_sim) complexity when O(T × n_edu × n_sim) is sufficient. Performance penalty.

**Fix**: Created new `_slice_mean_single_age_njit` function (lines 189-222) that computes means for a single age only. The loop now uses this efficient function:

```python
for age in range(self.T):
    # ...
    (a_mean, labor_mean, ...) = self._slice_mean_single_age_njit(..., int(age))  # O(n_sim)
    assets_by_age_edu[edu_idx, age] = a_mean
```

Complexity reduced from O(T² × n_edu × n_sim) to O(T × n_edu × n_sim)

---

### 5. Average earnings formula incorrect for mid-life agents

**Status**: [x] Complete

**Location**: `lifecycle_perfect_foresight.py:810-817, 855-863`

**Description**: The expanding average formula assumed the agent has been working since age 0:

```python
total_work_years = lifecycle_age + 1
avg_earnings[i] = (avg_earnings[i] * (total_work_years - 1) + gross_labor_income) / total_work_years
```

If an agent starts at `current_age=20` with `initial_avg_earnings` representing 20 years of prior earnings history, this formula incorrectly treats them as having `lifecycle_age + 1` years total, not `lifecycle_age + 1 + prior_years`.

**Impact**: For cohorts born before the transition (who start mid-life), the average earnings calculation is wrong, affecting pension amounts.

**Fix**: Now tracks the actual number of years of earnings history with `n_earnings_years` array:
```python
# Initialize: n_earnings_years = current_age if initial_avg_earnings provided, else 0
if hasattr(self.config, 'initial_avg_earnings') and self.config.initial_avg_earnings is not None:
    n_earnings_years = np.full(n_sim, self.current_age, dtype=int)
else:
    n_earnings_years = np.zeros(n_sim, dtype=int)

# In working years:
n_earnings_years[i] += 1
avg_earnings[i] = (avg_earnings[i] * (n_earnings_years[i] - 1) + gross_labor_income) / n_earnings_years[i]
```

---

## Minor Issues

### 6. Dead/vestigial code

**Status**: [x] Complete

**Location**:
- `olg_transition.py` (`_simulate_cached` method) - REMOVED
- `olg_transition.py` (`cohort_models` structure) - REMOVED

**Description**:
- `_simulate_cached()` was defined but never called
- `cohort_models[t][edu_type][age]` entries were created but unused (aggregation uses `birth_cohort_solutions` instead)

**Impact**: Code bloat, potential confusion, maintenance burden.

**Fix**: Removed dead code:
1. Removed `_simulate_cached()` method entirely
2. Removed `cohort_models` data structure (was T_transition × n_edu × T model instances)
3. Extracted the essential initial conditions logic into a simpler loop that directly updates `birth_cohort_solutions` for pre-transition cohorts
4. Changed state checks from `cohort_models is None` to `birth_cohort_solutions is None`
5. Removed unused `_sim_cache` initialization

---

### 7. UI benefits taxed as labor income

**Status**: [x] Complete (by design)

**Location**: `lifecycle_perfect_foresight.py:886-889`

**Description**: UI benefits are included in gross labor income and taxed at the full labor income tax rate:

```python
effective_y_sim[t_sim, i] = w * y * h + ui_sim[t_sim, i]
gross_labor_income = effective_y_sim[t_sim, i]
tax_l_sim[t_sim, i] = self.tau_l_path[lifecycle_age] * (gross_labor_income - tax_p_sim[t_sim, i])
```

**Resolution**: This is **intended behavior**. In most tax systems, unemployment insurance benefits are taxable income. The current implementation correctly:
1. Includes UI in `effective_y_sim` for aggregation purposes
2. Taxes UI at the labor income rate (after deducting payroll tax, which only applies to wages)
3. Does NOT apply payroll tax to UI (fixed in issue #3)

---

### 8. Inconsistent `initial_assets` default handling

**Status**: [x] Complete

**Location**: `lifecycle_perfect_foresight.py:124, 802-808`

**Description**: The `else` branch sets `i_a=0` (lowest grid point), but `LifecycleConfig` had `initial_assets: Optional[float] = 1` as default.

**Impact**: Minor inconsistency; the else branch rarely executed.

**Fix**: Changed `initial_assets` default from `1` to `None`:
```python
# Before:
initial_assets: Optional[float] = 1   # Initial asset level (default: a_min)

# After:
initial_assets: Optional[float] = None  # Initial asset level (default: a_min if None)
```

Now when `initial_assets` is not specified, agents correctly start at `a_min` (the borrowing constraint, first grid point). Also clarified the simulation code comments.

---

### 9. `_health_process` returns m_grid instead of h_grid for n_h=1

**Status**: [x] Complete

**Location**: `lifecycle_perfect_foresight.py:414-419` (`_health_process` method)

**Description**: When `n_h=1`, the function returned `m_grid` (medical expenditure) instead of `h_grid` (health productivity):

```python
if self.n_h == 1:
    m_grid = np.array([self.config.m_good])  # BUG: m_good=0.05, not h_good=1.0
    P_h = np.ones((self.T, 1, 1))
    return m_grid, P_h
```

This caused `h_grid = [0.05]` instead of `[1.0]`, making effective labor income 20× lower than intended.

**Impact**: With `h=0.05`, working income was extremely low while pensions (based on `y_grid` without `h`) were very high. This caused agents to rationally not save during working years, producing unrealistic lifecycle profiles.

**Fix**: Changed to return `h_grid` with `h_good`:

```python
if self.n_h == 1:
    h_grid = np.array([self.config.h_good])  # h_good=1.0
    P_h = np.ones((self.T, 1, 1))
    return h_grid, P_h
```

---

## Progress Tracker

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Pensions missing from value function | Critical | [x] |
| 2 | Policy indexing bug (latent) | Critical | [x] |
| 3 | Payroll tax on pensions | Critical | [x] |
| 9 | `_health_process` returns m_grid instead of h_grid for n_h=1 | Critical | [x] |
| 4 | O(T²) inefficiency in `_slice_means_njit` | Moderate | [x] |
| 5 | Avg earnings formula for mid-life | Moderate | [x] |
| 6 | Dead code | Minor | [x] |
| 7 | UI taxed as labor income | Minor | [x] |
| 8 | `initial_assets` default inconsistency | Minor | [x] |
