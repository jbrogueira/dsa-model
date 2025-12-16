# Critical Bug Fixes and Known Issues

## 1. K→0 Bug in OLG Transition (December 2024)

### Summary
Aggregate capital K would collapse to zero during OLG transition simulations, even with constant exogenous interest rates and stable economic fundamentals.

### Symptom Details
- **Observable behavior**: K_path shows monotonic decrease to zero over transition periods
- **Aggregate paths**: 
  - Capital (K): Decreases from initial positive value → 0
  - Labor (L): Remains stable (no collapse)
  - Output (Y): Decreases proportionally with K
  - Coefficient of Variation: Unable to compute (CV = nan) when K→0
- **Test failure**: `test_constant_environment_implies_constant_aggregates` failed with "Capital should be nearly constant (CV=nan% > 5%)"

### Root Cause Analysis

**Primary Issue**: Asset grid (a_grid) too sparse for steady-state initialization

The bug resulted from a chain of related issues:

1. **Grid Construction** (`lifecycle_perfect_foresight.py`, line ~213)
   ```python
   self.n_a = config.n_a  # With n_a=10
   # Grid constructed with only 10 points from 0 to 50
   # Example grid: [0, 1.85, 5.24, 9.62, 14.81, ...]
   ```

2. **Steady-State Profile Computation** (`olg_transition.py`, line ~268-285)
   ```python
   # Computed steady-state assets for old cohorts:
   ss_asset_profiles['medium'] = [0.0000, 0.1519, 0.2881, 0.3928, ...]
   # These values are SMALL (< 1.0) for young/middle-aged agents
   ```

3. **Grid Point Mapping** (`lifecycle_perfect_foresight.py`, line ~787-800)
   ```python
   if hasattr(self.config, 'initial_assets') and self.config.initial_assets is not None:
       # PROBLEM: Find closest grid point
       i_a_initial = np.argmin(np.abs(self.a_grid - self.config.initial_assets))
       # With sparse grid [0, 1.85, ...], ALL steady-state values < 1.0
       # map to index 0 (distance to 0 < distance to 1.85)
       i_a = np.full(n_sim, i_a_initial, dtype=int)
   ```

4. **Initialization of Old Cohorts** (`olg_transition.py`, line ~340-360)
   ```python
   if birth_period < 0:
       # Cohort born before transition
       cohort_age_at_transition = -birth_period
       initial_assets = self.ss_asset_profiles[edu_type][cohort_age_at_transition]
       # Example: initial_assets = 0.1519 (age 2)
       # But this gets mapped to a_grid[0] = 0.0 !!!
       
       solved_model.config = solved_model.config._replace(
           initial_assets=initial_assets,  # Set to 0.1519
           initial_avg_earnings=initial_avg_earnings
       )
   ```

5. **Behavioral Consequence**
   ```python
   # Agent initialized at a_grid[0] = 0.0 (not 0.1519!)
   # Policy function at a=0: save nothing (borrowing constraint binds)
   # Agent never accumulates assets
   # As old cohorts (with positive assets) die out → K→0
   ```

### Detailed Diagnostic Output

**Before Fix (n_a=10):**
```
DEBUG _simulate_sequential: age=2
  config.initial_assets = 0.1519  ✓ Config has correct value
  a_grid range: [0.0000, 50.0000]
  a_grid length: 10               ✗ Too few points!
  a_grid first 5 points: [0.0, 1.85, 5.24, 9.62, 14.81]
  i_a_initial = 0                 ✗ Maps to zero!
  a_grid[i_a_initial] = 0.0000    ✗ Agent starts with zero assets!
```

**After Fix (n_a=50):**
```
DEBUG _simulate_sequential: age=2
  config.initial_assets = 0.1519  ✓ Config has correct value
  a_grid range: [0.0000, 50.0000]
  a_grid length: 50               ✓ Sufficient density
  a_grid first 5 points: [0.0, 0.146, 0.412, 0.757, 1.166]
  i_a_initial = 1                 ✓ Maps to positive grid point!
  a_grid[i_a_initial] = 0.146     ✓ Close to target 0.1519
```

### Solution

**Change**: Increase minimum asset grid density

```python
# In test_olg_transition.py or any OLG simulation:
config = LifecycleConfig(
    T=100,
    beta=0.96,
    gamma=2.0,
    n_a=50,  # CHANGED: from 10 to 50
    n_y=2,
    n_h=1,
    retirement_age=6,
    education_type='medium'
)
```

**Why This Works**:
- With n_a=50, grid spacing near zero is approximately 0.146
- Steady-state assets (0.15, 0.29, 0.39, ...) now map to **positive** grid indices
- Old cohorts properly initialized with positive assets
- Borrowing constraint does not bind inappropriately
- Agents accumulate and decumulate assets normally
- Aggregate K remains stable

### Implementation Details

**File**: `lifecycle_perfect_foresight.py`, line ~787-800
```python
# In _simulate_sequential() method:
if hasattr(self.config, 'initial_assets') and self.config.initial_assets is not None:
    # CRITICAL: Map initial_assets to closest grid point
    # NOTE: Requires sufficient grid density (n_a >= 50) to avoid mapping
    #       small positive assets to index 0 (borrowing constraint).
    #       See BUGFIX.md for full details of K→0 bug.
    i_a_initial = np.argmin(np.abs(self.a_grid - self.config.initial_assets))
    i_a = np.full(n_sim, i_a_initial, dtype=int)
else:
    # Default: start with zero assets
    i_a = np.zeros(n_sim, dtype=int)
```

**File**: `olg_transition.py`, line ~268-285
```python
# In solve_cohort_problems() method:

# CRITICAL: Compute steady-state asset profiles for old cohorts
# These initial conditions are essential to prevent K→0 bug.
# Old cohorts (born before transition) must start with positive assets
# from their steady-state lifecycle accumulation.
# 
# IMPORTANT: This requires n_a >= 50 to properly map small steady-state
#            asset values (typically 0.1-0.5 for young cohorts) to 
#            positive grid indices. With sparse grids (n_a=10), all
#            values < 1.0 map to a_grid[0]=0, triggering borrowing
#            constraint and causing K→0.

for edu_type in self.education_shares.keys():
    ss_config = LifecycleConfig(
        T=self.T, beta=self.beta, gamma=self.gamma, current_age=0,
        education_type=edu_type, 
        n_a=self.n_a,  # Must be >= 50!
        n_y=self.n_y, n_h=self.n_h,
        retirement_age=self.retirement_age,
        # ... (rest of config)
    )
    ss_model = LifecycleModelPerfectForesight(ss_config, verbose=False)
    ss_model.solve(verbose=False)
    results = ss_model.simulate(T_sim=self.T, n_sim=1000, seed=42)
    
    # Store mean assets by age as initial conditions
    self.ss_asset_profiles[edu_type] = np.mean(results[0], axis=1)
    self.ss_earnings_profiles[edu_type] = np.mean(results[15], axis=1)
```

### Minimum Requirements

For typical calibrations:
- **β = 0.96** (discount factor)
- **γ = 2.0** (risk aversion)
- **r = 0.04** (interest rate)
- **Retirement age = 6-30** (periods)

**Required grid parameters**:
- **Minimum n_a**: 50 (absolute minimum)
- **Recommended n_a**: 100 (for production runs)
- **Grid spacing**: ≤ 0.5 around steady-state asset range [0, 5]

**Grid spacing formula**:
```python
# For linear grid: spacing = a_max / (n_a - 1)
# For n_a=50, a_max=50: spacing = 50/49 ≈ 1.02 (TOO LARGE near zero)
# For n_a=50 with better spacing, use exponential or custom grid
```

**Better grid construction** (recommended for production):
```python
# Option 1: Exponential spacing (more points near zero)
a_max = 50
n_a = 50
a_grid = np.linspace(0, a_max**(1/3), n_a)**3
# Gives: [0, 0.018, 0.145, 0.489, 1.16, 2.26, ...]

# Option 2: Piecewise-linear (fine near zero, coarse at high values)
n_fine = 30  # Points from 0 to 5
n_coarse = 20  # Points from 5 to 50
a_grid = np.concatenate([
    np.linspace(0, 5, n_fine),
    np.linspace(5, 50, n_coarse)[1:]  # Exclude duplicate 5
])
```

### Testing

**Primary test**: `test_olg_transition.py::TestConstantInterestRate::test_constant_environment_implies_constant_aggregates`

**Purpose**: Verify that with constant exogenous prices (r, w, taxes), aggregate capital remains stable

**Pass criteria**:
```python
# After burn-in period:
K_cv < 0.05  # Coefficient of variation < 5%
|K_trend| < 0.5%  # per period
```

**Example output** (passing test):
```
Aggregate stability (Coefficient of Variation):
  Capital:  1.05% (< 5% ✓)
  Labor:    0.94% (< 5% ✓)
  Output:   0.69% (< 5% ✓)

Aggregate trends (% change per period):
  Capital:  -0.42% (|·| < 0.5% ✓)
  Labor:    +0.12% (|·| < 0.5% ✓)
  Output:   -0.06% (|·| < 0.5% ✓)

Mean values:
  K = 4.1309 ± 0.0532  ← Stable positive capital!
  L = 0.0783 ± 0.0008
  Y = 0.2899 ± 0.0020
```

**Supporting test**: `test_olg_transition.py::TestBorrowingConstraint::test_policy_at_different_asset_levels`

**Purpose**: Verify that borrowing constraint only binds at a=0

**Pass criteria**:
```python
# Policy functions:
a_policy[a=0] == 0  # Borrowing constraint binds
a_policy[a>0] > 0   # Agents with positive assets DO save
```

### Performance Impact

**Computational cost increase**: Approximately 5x (n_a: 10 → 50)

**Time comparison** (test suite, T=100, T_transition=10, n_sim=100):
- **Before (n_a=10)**: ~70 seconds
- **After (n_a=50)**: ~230 seconds (3-4 minutes)

**Memory impact**: Negligible (policy functions stored per cohort-age-education)

**Recommendation**: Use n_a=50 for tests, n_a=100 for production simulations

### Files Modified

1. **lifecycle_perfect_foresight.py** (line ~787):
   - Added inline comment explaining grid density requirement
   - No code changes needed (logic was correct)

2. **olg_transition.py** (line ~268-285):
   - Added extensive documentation block explaining steady-state initialization
   - No code changes needed (logic was correct)

3. **test_olg_transition.py**:
   - Changed `n_a=10` to `n_a=50` in `TestConstantInterestRate` class

4. **lifecycle_config.py**:
   - Added `_replace()` method to LifecycleConfig dataclass

### Prevention Checklist

When setting up new OLG simulations, verify:

- [ ] `n_a >= 50` in lifecycle config
- [ ] Grid spacing near zero is < 0.5 (check `a_grid[1] - a_grid[0]`)
- [ ] Steady-state asset profiles are positive for ages > 0
- [ ] Initial asset mapping produces positive grid indices: `i_a_initial > 0`
- [ ] Run `test_constant_environment_implies_constant_aggregates` before production
- [ ] Monitor K_path for unexpected trends or collapse in early periods

### References

- **Git commit**: [Add commit hash after committing this fix]
- **Date discovered**: December 16, 2024
- **Date fixed**: December 16, 2024
- **Time to diagnose**: ~2-3 hours of debugging with debug prints
- **Fixed by**: João B. Sousa

### Lessons Learned

1. **Grid density matters**: Discretization errors can cause catastrophic failures
2. **Test initialization**: Always verify initial conditions map correctly to discrete states
3. **Debug visibility**: Comprehensive debug output was essential for diagnosis
4. **Test design**: Both high-level and low-level tests helped isolate the issue
5. **Documentation**: Inline comments at critical operations prevent regression

---

## 2. [Future bugs go here]

...

---