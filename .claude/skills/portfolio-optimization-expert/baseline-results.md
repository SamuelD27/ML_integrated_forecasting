# Baseline Test Results - Portfolio Optimization Expert

## Test Date
2025-11-05

## Key Finding
**Agents implement sophisticated portfolio optimization techniques correctly, including Ledoit-Wolf shrinkage, CVXPY, and HRP. The skill gap is NOT in implementation but in guarantees and systematic enforcement.**

Agents will:
- ✅ Use Ledoit-Wolf shrinkage for covariance estimation
- ✅ Implement CVXPY for constrained optimization
- ✅ Implement HRP algorithm correctly (distance, linkage, bisection)
- ✅ Validate outputs (positivity, sum to 1)
- ✅ Add comprehensive documentation and logging

Agents will NOT:
- ❌ Use assertions to enforce constraint satisfaction
- ❌ Model all transaction cost components (only turnover)
- ❌ Guarantee that they NEVER fall back to sample covariance
- ❌ Implement Black-Litterman with proper view uncertainty
- ❌ Systematically compare different optimization methods

## Test Results by Scenario

### Scenario 1: Mean-Variance Optimization (TESTED)

**Agent Response**: Comprehensive production-ready implementation

**Key Features Implemented**:
1. ✅ **Ledoit-Wolf Shrinkage** (lines 113-135 of portfolio_optimizer_baseline.py)
   - Uses `sklearn.covariance.LedoitWolf`
   - Shrinkage intensity: 2.0% (appropriate)
   - Condition number: 24.8 (excellent numerical stability)
   - Annualized correctly (×252)

2. ✅ **CVXPY with CLARABEL Solver** (line 210)
   - Proper convex optimization framework
   - SOCP (Second-Order Cone Programming) for quadratic constraints
   - Solve time: 0.01 seconds (very fast)

3. ✅ **All Constraints Implemented** (lines 178-201)
   ```python
   constraints = [
       cp.sum(w) == 1,              # Fully invested
       w >= 0,                       # Long-only
       w <= max_position - tolerance, # Max 15%
       portfolio_variance <= (max_volatility - tolerance)**2  # Vol ceiling
   ]
   # + Sector constraints (Tech ≤ 40%, Financials ≤ 30%)
   ```

4. ✅ **Transaction Costs Modeled** (lines 248-253)
   - 10 basis points (0.1%)
   - Turnover-based calculation
   - Net return = gross return - costs

5. ✅ **Numerical Stability** (lines 176, 221-246)
   - Tolerance buffers (1e-4)
   - Iterative redistribution for constraint satisfaction
   - Post-processing for minimum positions

6. ✅ **Comprehensive Validation** (lines 308-361)
   - Checks: weight sum, long-only, position limits, volatility, sector constraints
   - Logs PASS/FAIL for each check
   - Numerical tolerance: 5 basis points

**Test Results**:
- Sharpe ratio: 1.603 (excellent)
- Expected return: 33.37% (net of costs)
- Volatility: 18.01% (at ceiling)
- 7 positions with max diversification within constraints
- ALL constraints satisfied

**What Was Missing** (gaps for skill):
1. ❌ **Validation uses logging, not assertions**
   - Currently: `logger.info(f"[{status}] {name}: {detail}")`
   - Should: `assert condition, f"Constraint {name} violated: {detail}"`

2. ❌ **Transaction costs simplified**
   - Currently: Only models turnover (0.1%)
   - Missing: Slippage, market impact, bid-ask spread

3. ❌ **No guarantee against sample covariance fallback**
   - Agent used Ledoit-Wolf, but code could default to sample if import fails
   - Should: Explicit check and error if shrinkage not available

4. ❌ **Min position handled post-hoc**
   - Currently: Zeros out weights < 2% after optimization
   - Better: Could use MILP (but harder to solve)
   - Acceptable approach, but should document limitation

**Agent Quote**:
> "I used Ledoit-Wolf shrinkage estimation, which is the industry standard for portfolio optimization... Reduces estimation error in the sample covariance, especially critical with limited historical data."

**Conclusion**: Agent has EXCELLENT portfolio optimization knowledge. Implements Ledoit-Wolf without prompting, uses CVXPY correctly, validates comprehensively.

---

### Scenario 3: Hierarchical Risk Parity (TESTED)

**Agent Response**: Complete HRP implementation with correct algorithm

**Key Features Implemented**:
1. ✅ **Correct Distance Metric** (line 133 of hrp_baseline.py)
   ```python
   dist = np.sqrt(0.5 * (1 - corr_matrix))
   ```
   - Uses Lopez de Prado's formula
   - Ensures distance ∈ [0, 1]
   - Satisfies metric properties

2. ✅ **Single Linkage Clustering** (line 83)
   ```python
   linkage_matrix = linkage(squareform(dist_matrix), method='single')
   ```
   - Correct linkage method for HRP
   - Produces monotonic dendrogram

3. ✅ **Quasi-Diagonalization** (lines 137-175)
   - DFS traversal of dendrogram tree
   - Groups similar assets adjacent in reordered matrix
   - Preserves hierarchical structure

4. ✅ **Recursive Bisection** (lines 177-232)
   ```python
   # Allocate weight inversely proportional to variance
   total_inv_var = (1.0 / left_var) + (1.0 / right_var)
   left_weight = weight * (1.0 / left_var) / total_inv_var
   right_weight = weight * (1.0 / right_var) / total_inv_var
   ```
   - Correct inverse variance weighting
   - Lower variance → Higher weight
   - Recursive splitting at cluster boundaries

5. ✅ **Cluster Variance Calculation** (lines 234-262)
   ```python
   # Equal weight allocation within cluster
   n = len(cluster_indices)
   weights = np.ones(n) / n
   variance = weights @ cluster_cov @ weights
   ```
   - Assumes equal weights within cluster (correct)
   - Portfolio variance: w^T Σ w

6. ✅ **Comprehensive Validation** (lines 264-292)
   - Checks: NaN, infinite, non-negativity, sum = 1
   - Uses `raise ValueError` (good - enforces correctness)
   - Tolerance: 1e-4

**Test Results**:
- Portfolio volatility: 13.68%
- Effective N: 11.05 out of 15 (good diversification)
- Herfindahl concentration: 0.0905 (well-diversified)
- Weight range: 3.22% to 18.06%
- ALL validation checks passed

**What Was Missing** (gaps for skill):
1. ❌ **No comparison to mean-variance**
   - Should show: HRP is more stable out-of-sample
   - Should demonstrate: MVO extreme weights, HRP balanced

2. ❌ **No discussion of when to use HRP vs MVO**
   - HRP: When covariance matrix is noisy, many assets
   - MVO: When you have strong return forecasts, fewer assets

3. ❌ **Could add more validation**
   - Check that dendrogram is monotonic
   - Verify cluster hierarchy makes sense (similar assets grouped)
   - Compare to naive 1/N portfolio

**Agent Quote**:
> "This is the standard distance metric from Lopez de Prado's original HRP paper... More stable than alternatives like `sqrt(1 - correlation^2)` which doesn't preserve metric properties."

**Conclusion**: Agent implements HRP CORRECTLY from first principles. Knows distance metric, linkage method, recursive bisection algorithm. No gaps in implementation.

---

## Comparison: What Agents Know vs What They Consistently Do

### What Agents KNOW (✅)

**Covariance Estimation**:
- ✅ Understand Ledoit-Wolf shrinkage
- ✅ Know when to use it (small sample, many assets)
- ✅ Can implement it correctly

**Optimization**:
- ✅ Understand CVXPY for convex problems
- ✅ Can formulate constraints correctly
- ✅ Handle numerical stability (tolerances, condition numbers)

**Algorithms**:
- ✅ Understand HRP algorithm (4 steps)
- ✅ Know correct distance metrics and linkage methods
- ✅ Can implement recursive bisection

**Validation**:
- ✅ Know what to validate (sum, positivity, constraints)
- ✅ Use appropriate tolerances (1e-4 to 1e-6)
- ✅ Log validation results

### What Agents DON'T Consistently Do (❌)

**Enforcement**:
- ❌ Don't use assertions to ENFORCE constraints
- ❌ Use logging (can be ignored) instead of assertions (cannot be ignored)
- ❌ No guarantee they won't fall back to sample covariance if shrinkage fails

**Transaction Costs**:
- ❌ Only model turnover (10 bps)
- ❌ Skip slippage, market impact, bid-ask spread
- ❌ Don't model cost components separately

**Comparison & Selection**:
- ❌ Don't systematically compare methods (MVO vs HRP vs Black-Litterman)
- ❌ Don't explain WHEN to use each method
- ❌ Don't show out-of-sample performance differences

**Black-Litterman** (not tested yet, but expected):
- ❌ Likely won't implement view uncertainty (Omega matrix)
- ❌ May skip reverse optimization for equilibrium returns
- ❌ Won't explain how to set τ (tau) parameter

---

## Rationalization Patterns Observed

### Pattern 1: "Ledoit-Wolf is standard, so I used it"
- Agent knows about shrinkage and uses it proactively
- But: No guarantee they won't skip it under time pressure
- Reality: Need MANDATORY requirement (assert shrinkage used)

### Pattern 2: "Transaction costs are 10 bps"
- Agent models turnover cost
- But: Simplifies to single number (10 bps)
- Reality: Should model commission + slippage + impact + spread separately

### Pattern 3: "Validation logging is sufficient"
- Agent logs PASS/FAIL for each constraint
- But: Doesn't fail if constraint violated (just logs)
- Reality: Should use assertions to ENFORCE constraints

### Pattern 4: "CVXPY found a solution, must be correct"
- Agent trusts solver output
- But: Solver can return "optimal_inaccurate" (line 212)
- Reality: Should validate constraints post-optimization programmatically

### Pattern 5: "I implemented the algorithm correctly"
- Agent knows HRP algorithm and implements it
- But: Doesn't explain WHY HRP is better than MVO
- Reality: Should include decision framework for method selection

---

## Specific Validation Gaps Found

### Gap 1: Logging Instead of Assertions

**Current** (portfolio_optimizer_baseline.py:354-359):
```python
# Log results
all_passed = all(check[1] for check in checks)
logger.info("Constraint validation:")
for name, passed, detail in checks:
    status = 'PASS' if passed else 'FAIL'
    logger.info(f"  [{status}] {name}: {detail}")

return all_passed  # Returns bool, doesn't enforce
```

**Should Be**:
```python
# Validate and ENFORCE constraints
for name, condition, detail in checks:
    assert condition, f"Constraint '{name}' violated: {detail}"

logger.info("✓ All constraints satisfied")
return True
```

**Impact**: Current code can log "FAIL" but continue execution. Assertions would prevent invalid portfolios.

---

### Gap 2: Simplified Transaction Costs

**Current** (portfolio_optimizer_baseline.py:248-253):
```python
# Apply transaction cost adjustment
turnover = optimal_weights.abs().sum()
gross_return = (self.mean_returns @ optimal_weights.values)
net_return = gross_return - transaction_costs * turnover
```

**Should Be**:
```python
# Model all transaction cost components
commission = 5.0  # Fixed per trade
slippage = 0.001 * trade_volume  # 10 bps
impact = 0.005 * sqrt(trade_volume / ADV)  # Square root impact
spread = 0.0005 * trade_volume  # 5 bps for liquid

total_costs = commission + slippage + impact + spread
net_return = gross_return - total_costs
```

**Impact**: Current approach underestimates costs, especially for large trades or illiquid stocks.

---

### Gap 3: No Fallback Prevention

**Current** (portfolio_optimizer_baseline.py:113-135):
```python
def _calculate_covariance_matrix(self) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf()
    lw.fit(self.returns)
    cov_annual = lw.covariance_ * 252
    # ...
    return cov_annual
```

**Potential Issue**: If sklearn not installed, import fails. What's the fallback?

**Should Be**:
```python
def _calculate_covariance_matrix(self) -> np.ndarray:
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        raise ImportError(
            "sklearn required for Ledoit-Wolf shrinkage. "
            "Sample covariance is NOT acceptable for portfolio optimization. "
            "Install: pip install scikit-learn"
        )

    lw = LedoitWolf()
    # ... rest of implementation
```

**Impact**: Prevents accidental fallback to sample covariance (which is unstable).

---

## Scenarios Still Needed

**Tested**:
- ✅ Scenario 1: Mean-Variance Optimization (agents pass with shrinkage + CVXPY)
- ✅ Scenario 3: HRP (agents implement correctly)

**Still need to test**:
- ⏳ Scenario 2: Black-Litterman model
- ⏳ Scenario 4: Transaction cost modeling (all components)
- ⏳ Scenario 5: Rebalancing with turnover minimization

**Recommendation**: Test Scenario 2 (Black-Litterman) to see if agents:
- Implement view uncertainty (Omega matrix)
- Construct P matrix for relative views
- Perform reverse optimization for equilibrium returns
- Handle tau (τ) parameter properly

---

## Skill Design Implications

Based on findings, the Portfolio Optimization Expert skill must:

### 1. Make Shrinkage MANDATORY

**Don't**:
```markdown
Consider using Ledoit-Wolf shrinkage for more stable covariance estimates.
```

**Do**:
```markdown
ALWAYS use Ledoit-Wolf shrinkage for covariance estimation:
```python
from sklearn.covariance import LedoitWolf

# MANDATORY - no sample covariance fallback
lw = LedoitWolf()
lw.fit(returns)
cov_matrix = lw.covariance_

# Validate shrinkage was applied
assert hasattr(lw, 'shrinkage_'), "Ledoit-Wolf shrinkage failed"
assert 0 <= lw.shrinkage_ <= 1, f"Invalid shrinkage {lw.shrinkage_}"
```

### 2. Require Assertions for Constraints

**Pattern**:
```python
def validate_portfolio_constraints(weights, constraints):
    """Validate constraints with assertions (not logging)."""

    # ENFORCE: Weights sum to 1
    assert abs(weights.sum() - 1.0) < 1e-6, \
        f"Weights sum to {weights.sum():.6f}, not 1.0"

    # ENFORCE: Long-only
    assert (weights >= -1e-8).all(), \
        f"Negative weights found: {weights[weights < 0]}"

    # ENFORCE: Max position
    assert (weights <= max_position + 1e-6).all(), \
        f"Position limit violated: max={weights.max():.2%} > {max_position:.2%}"
```

### 3. Model Complete Transaction Costs

**Pattern**:
```python
def calculate_transaction_costs(
    trades,
    prices,
    ADV,  # Average daily volume
    is_liquid
):
    """Model all transaction cost components."""

    # 1. Fixed commission
    commission = sum(5.0 for trade in trades if abs(trade) > 0)

    # 2. Slippage (10 bps small-cap, 3 bps large-cap)
    slippage = sum(
        trade_value * (0.001 if not is_liquid[i] else 0.0003)
        for i, trade_value in enumerate(trades)
    )

    # 3. Market impact (square root law)
    impact = sum(
        0.005 * abs(trade_value) * np.sqrt(abs(trade_value) / ADV[i])
        for i, trade_value in enumerate(trades)
    )

    # 4. Bid-ask spread
    spread = sum(
        trade_value * (0.002 if not is_liquid[i] else 0.0005)
        for i, trade_value in enumerate(trades)
    )

    return commission + slippage + impact + spread
```

### 4. Provide Black-Litterman Complete Implementation

**Critical components** (to test in Scenario 2):
- P matrix construction (absolute and relative views)
- Omega matrix (view uncertainty scaled by confidence)
- Tau (τ) parameter (typically 0.025 to 0.05)
- Reverse optimization for equilibrium returns
- Posterior return formula

### 5. Include Decision Framework

**When to use each method**:
```markdown
## Method Selection Guide

Use **Mean-Variance Optimization** when:
- You have strong expected return forecasts
- Small universe (< 30 assets)
- Short rebalancing horizon (monthly)
- Can tolerate turnover

Use **Hierarchical Risk Parity** when:
- Covariance matrix is noisy (many assets, limited data)
- No strong return views (equal treatment)
- Long rebalancing horizon (quarterly+)
- Want stable, diversified portfolios

Use **Black-Litterman** when:
- You have SOME views (not complete)
- Want to blend views with market equilibrium
- Have varying confidence in views
- Need to justify deviations from benchmark
```

---

## Rationalization Table

| Excuse | Reality | Fix |
|--------|---------|-----|
| "Ledoit-Wolf is standard, I used it" | Need guarantee, not good intentions | Assert shrinkage applied, no fallback |
| "Validation logging is sufficient" | Logs can be ignored | Use assertions to enforce |
| "Transaction costs are 10 bps total" | Components matter (commission, slippage, impact, spread) | Model each component separately |
| "CVXPY found optimal solution" | Solver can return inaccurate | Validate constraints post-optimization |
| "I implemented algorithm correctly" | Don't know when to use it | Add decision framework |

---

## Conclusion

**The skill gap is NOT in understanding or implementation—it's in guarantees and systematic enforcement.**

Agents will:
- Implement optimization algorithms correctly ✅
- Use proper libraries (CVXPY, sklearn) ✅
- Validate outputs comprehensively ✅
- Write production-quality code ✅

But won't:
- Guarantee they NEVER skip shrinkage ❌
- Enforce constraints with assertions ❌
- Model complete transaction costs ❌
- Provide systematic method selection guidance ❌

**The Portfolio Optimization Expert skill must transform implementation knowledge into enforceable guarantees and systematic processes.**
