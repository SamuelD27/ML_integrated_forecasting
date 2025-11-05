# Test Scenarios - Portfolio Optimization Expert

## Scenario 1: Mean-Variance Optimization with Constraints

**Pressure Type**: Complexity + Time Pressure

**User Message**:
```
I need to implement a mean-variance portfolio optimizer for our fund. We have 20 stocks and need to optimize weights.

Requirements:
- Maximize Sharpe ratio
- Long-only constraints (no shorting)
- Maximum 15% position in any single stock
- Minimum 2% position if included (no tiny positions)
- Sector constraints: Tech ≤ 40%, Financials ≤ 30%
- Total portfolio volatility ≤ 18% annualized

I have historical returns data in a CSV. Can you implement this quickly? We need it for tomorrow's investment committee meeting.
```

**What We're Testing**:
1. Does agent use proper covariance estimation (Ledoit-Wolf shrinkage)?
2. Does agent implement constraints correctly with CVXPY/scipy?
3. Does agent validate optimization results (weights sum to 1, constraints satisfied)?
4. Does agent handle numerical stability (regularization for singular matrices)?
5. Does agent model realistic transaction costs?

**Expected Baseline Behavior** (without skill):
- ❌ Uses sample covariance (noisy, unstable)
- ❌ May not implement sector constraints correctly
- ❌ Doesn't validate constraint satisfaction programmatically
- ❌ No regularization for singular covariance matrices
- ❌ Ignores transaction costs

**With Skill**:
- ✅ Ledoit-Wolf shrinkage for covariance
- ✅ CVXPY for constrained optimization
- ✅ Programmatic validation (assert weights sum to 1, constraints satisfied)
- ✅ Regularization for numerical stability
- ✅ Transaction cost modeling

---

## Scenario 2: Black-Litterman Model Implementation

**Pressure Type**: Authority + Complexity

**User Message**:
```
Our CIO wants us to use the Black-Litterman model for portfolio construction. He says it's better than standard mean-variance because it incorporates market equilibrium and our views.

I have:
- Market cap weights for 15 stocks
- Expected returns from our analyst team (views)
- Historical covariance matrix

Implement Black-Litterman to generate posterior returns. The CIO said this is the industry standard, so we need to do it right.

Here are our views:
- AAPL will outperform MSFT by 5% over the next year (confidence: 80%)
- Tech sector will return 12% (confidence: 60%)
- Energy sector will underperform by 3% (confidence: 70%)
```

**What We're Testing**:
1. Does agent implement Black-Litterman formula correctly (posterior = prior + adjustment)?
2. Does agent construct the P matrix (pick matrix) for views correctly?
3. Does agent model view uncertainty (Omega matrix)?
4. Does agent calculate implied equilibrium returns (reverse optimization)?
5. Does agent validate that posterior returns are reasonable?

**Expected Baseline Behavior** (without skill):
- ❌ May skip view uncertainty modeling (treats views as certain)
- ❌ Incorrect P matrix construction (doesn't handle relative views properly)
- ❌ No validation of equilibrium returns
- ❌ Doesn't implement confidence levels correctly
- ❌ No check that posterior returns are between prior and views

**With Skill**:
- ✅ Complete Black-Litterman formula with view uncertainty
- ✅ Correct P matrix for absolute and relative views
- ✅ Omega matrix scaled by confidence levels
- ✅ Reverse optimization for equilibrium returns
- ✅ Validation assertions (posterior plausibility checks)

---

## Scenario 3: Hierarchical Risk Parity (HRP)

**Pressure Type**: Sunk Cost + Overconfidence

**User Message**:
```
We've been using mean-variance optimization but it's too unstable - tiny changes in inputs cause huge portfolio shifts. I read about Hierarchical Risk Parity (HRP) by Marcos López de Prado and it seems more stable.

Can you implement HRP? I've already spent 3 hours trying but can't get it to work. The steps are:
1. Compute distance matrix from correlation
2. Hierarchical clustering (single linkage)
3. Quasi-diagonalize covariance matrix
4. Recursive bisection to allocate weights

I have the correlation and covariance matrices ready. Please implement this - I've already told my boss we'll have it working by EOD.
```

**What We're Testing**:
1. Does agent implement correct distance metric (√(0.5 × (1 - correlation)))?
2. Does agent use correct linkage method (single linkage for HRP)?
3. Does agent implement quasi-diagonalization correctly?
4. Does agent implement recursive bisection with proper cluster variance calculation?
5. Does agent validate that HRP weights are positive and sum to 1?

**Expected Baseline Behavior** (without skill):
- ❌ May use wrong distance metric (1 - correlation)
- ❌ May use wrong linkage method (ward, complete instead of single)
- ❌ Incorrect quasi-diagonalization (doesn't reorder by cluster)
- ❌ Recursive bisection logic errors (doesn't split at cluster level)
- ❌ No validation of output weights

**With Skill**:
- ✅ Correct distance metric: √(0.5 × (1 - ρ))
- ✅ Single linkage clustering
- ✅ Proper quasi-diagonalization with seriation
- ✅ Recursive bisection at cluster boundaries
- ✅ Validation assertions (weights > 0, sum = 1, diversification)

---

## Scenario 4: Transaction Cost Modeling

**Pressure Type**: Authority + Complexity

**User Message**:
```
Our head trader says our portfolio optimizer is unrealistic - it suggests trades without considering costs. He gave me this cost model:

1. Fixed commission: $5 per trade
2. Slippage: 10 basis points (0.10%) for small-cap, 3 bps for large-cap
3. Market impact: 0.5 × √(trade_size / ADV) where ADV = average daily volume
4. Bid-ask spread: 0.05% for liquid stocks, 0.20% for illiquid

Implement transaction-aware optimization that:
- Only trades when expected alpha > transaction costs
- Considers current holdings (we're not starting from scratch)
- Models the cost of rebalancing
- Outputs net-of-cost expected returns

Current portfolio has 25 positions totaling $10M AUM.
```

**What We're Testing**:
1. Does agent model all cost components (commission, slippage, impact, spread)?
2. Does agent implement cost-aware optimization (turnover penalty)?
3. Does agent compare costs vs benefits (alpha threshold)?
4. Does agent consider current holdings (not optimize from zero)?
5. Does agent validate that high-turnover trades are worth it?

**Expected Baseline Behavior** (without skill):
- ❌ May model only commission (simplest)
- ❌ Doesn't include market impact (most complex)
- ❌ Optimizes from zero (ignores current holdings)
- ❌ No turnover penalty in objective function
- ❌ Doesn't validate cost vs benefit

**With Skill**:
- ✅ All cost components modeled (commission, slippage, impact, spread)
- ✅ Turnover penalty in optimization objective
- ✅ Warm-start optimization from current holdings
- ✅ Cost-benefit analysis before trades
- ✅ Validation assertions (expected_return > costs + hurdle)

---

## Scenario 5: Rebalancing Strategy with Constraints

**Pressure Type**: Time Pressure + Complexity

**User Message**:
```
We need to implement quarterly rebalancing for our multi-asset portfolio ($50M AUM). Current rules:

1. Target weights: 60% stocks, 30% bonds, 10% alternatives
2. Rebalance if any asset drifts > 5% from target (55-65% stocks, 25-35% bonds, 5-15% alts)
3. When rebalancing:
   - Move back to target weights
   - Within stocks, optimize using mean-variance (max Sharpe)
   - Minimize turnover (transaction costs are 25 bps per trade)
   - Tax considerations: prefer selling losses over gains (tax-loss harvesting)

Current allocation:
- Stocks: 68% (DRIFT - need to rebalance!)
- Bonds: 28% (OK)
- Alternatives: 4% (DRIFT - need to rebalance!)

Within stocks, we have 30 positions. Implement the rebalancing optimizer.

We need this today - board meeting tomorrow morning!
```

**What We're Testing**:
1. Does agent detect drift correctly (> 5% threshold)?
2. Does agent implement hierarchical optimization (asset class first, then within-asset)?
3. Does agent minimize turnover (only trade what's necessary)?
4. Does agent model tax implications (realize losses first)?
5. Does agent validate that rebalancing improves expected outcomes?

**Expected Baseline Behavior** (without skill):
- ❌ May rebalance all positions (inefficient)
- ❌ Doesn't implement hierarchical optimization correctly
- ❌ No turnover minimization (just hits target weights)
- ❌ Ignores tax implications
- ❌ No validation that rebalancing benefits > costs

**With Skill**:
- ✅ Drift detection with threshold checks
- ✅ Two-stage optimization (asset class → within-asset)
- ✅ Turnover minimization with transaction cost penalty
- ✅ Tax-loss harvesting (sell losers first)
- ✅ Validation assertions (expected_benefit > rebalancing_costs)

---

## Pressure Analysis

### Pressure Type Distribution

1. **Complexity**: All scenarios (optimization is inherently complex)
2. **Time Pressure**: Scenarios 1, 5 (need by tomorrow/today)
3. **Authority**: Scenarios 2, 4 (CIO/head trader demands)
4. **Sunk Cost**: Scenario 3 (user spent 3 hours already)
5. **Overconfidence**: Scenario 3 (user thinks it's straightforward)

### Rationalization Predictions

| Excuse | Scenario | Reality |
|--------|----------|---------|
| "Sample covariance is standard" | 1 | Ledoit-Wolf shrinkage needed for stability |
| "Black-Litterman is just Bayesian update" | 2 | Requires correct P, Omega, τ matrices |
| "HRP is just clustering + allocation" | 3 | Requires specific distance metric and linkage |
| "Transaction costs are small" | 4 | Can dominate returns (25-50 bps per trade) |
| "Rebalance to target weights is simple" | 5 | Need turnover minimization and tax awareness |

### Expected Failure Modes

**Without Skill**:
1. **Unstable optimization**: Sample covariance causes extreme weights
2. **Constraint violations**: Sector limits not implemented correctly
3. **Incorrect Black-Litterman**: Views not incorporated properly
4. **Wrong HRP**: Distance metric or linkage method incorrect
5. **Cost ignorance**: Optimization suggests high-turnover trades

**With Skill**:
- Ledoit-Wolf shrinkage for stable covariance
- CVXPY for guaranteed constraint satisfaction
- Complete Black-Litterman with view uncertainty
- Correct HRP implementation (distance, linkage, bisection)
- Transaction-aware optimization with turnover penalty

---

## Success Criteria

### Baseline Tests (RED Phase)

For each scenario, agent should:
- ❌ Skip covariance shrinkage
- ❌ Implement constraints incorrectly or ignore them
- ❌ Model transaction costs simplistically or not at all
- ❌ Use wrong distance metrics / linkage methods for HRP
- ❌ No validation that constraints are satisfied

### With Skill (GREEN Phase)

Agent should:
- ✅ Use Ledoit-Wolf shrinkage for covariance estimation
- ✅ Use CVXPY for constrained optimization
- ✅ Implement all transaction cost components
- ✅ Use correct HRP algorithm (distance, linkage, bisection)
- ✅ Validate constraint satisfaction programmatically

### Refactor Phase (REFACTOR)

Test that skill prevents:
- ❌ "Sample covariance is fine for 20 stocks" rationalization
- ❌ "I'll validate constraints manually" excuse
- ❌ "Transaction costs are negligible" assumption
- ❌ "Any clustering method works" misconception
- ❌ "Optimization found a solution, must be correct" trust

---

## File Outputs Expected

For baseline tests, each scenario should generate:
- `portfolio_optimizer_baseline_scenario[1-5].py`

For skill tests, each scenario should generate:
- `portfolio_optimizer_with_skill_scenario[1-5].py`

Compare implementations to measure skill effectiveness.

---

## Notes for Skill Design

Based on these scenarios, the skill must teach:

1. **Covariance Estimation**: Ledoit-Wolf shrinkage (not sample covariance)
2. **Constrained Optimization**: CVXPY with proper constraint formulation
3. **Black-Litterman**: Complete formula with P, Q, Omega, τ matrices
4. **HRP**: Correct distance (√(0.5(1-ρ))), single linkage, recursive bisection
5. **Transaction Costs**: Commission + slippage + impact + spread
6. **Validation**: Assert constraints satisfied, weights valid, costs < benefits

Each pattern must include:
- Complete implementation code
- Numerical validation checks
- Common mistakes (Bad vs Good)
- Rationalization counters
