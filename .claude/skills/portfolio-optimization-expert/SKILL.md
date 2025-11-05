---
name: portfolio-optimization-expert
description: Use when constructing portfolios with mean-variance, Black-Litterman, or Hierarchical Risk Parity - enforces Ledoit-Wolf shrinkage, constraint validation with assertions, complete transaction cost modeling, and method selection guidance to prevent unstable allocations and ensure out-of-sample performance
---

# Portfolio Optimization Expert

## Overview

**Portfolio optimization is enforcement + validation, not just implementation.** Every optimizer must use Ledoit-Wolf shrinkage (NEVER sample covariance), validate constraints with assertions (not logging), and model all transaction cost components. Implementation alone causes unstable portfolios.

**Core principle:** Knowing the algorithm ≠ enforceable guarantees. Always assert constraints and prevent fallbacks.

## When to Use

Use this skill when:
- Implementing mean-variance portfolio optimization
- Constructing portfolios with Black-Litterman model
- Using Hierarchical Risk Parity (HRP) for diversification
- Modeling transaction costs for rebalancing
- Comparing optimization methods (MVO vs HRP vs Black-Litterman)
- Constrained optimization (sector limits, turnover, ESG)

**Don't skip enforcement because:**
- "Algorithm is implemented correctly" (no validation = unstable portfolios)
- "Ledoit-Wolf is standard" (need guarantee, not intention)
- "Transaction costs are small" (can be 25-50 bps per trade)
- "Logging validation is sufficient" (logs can be ignored)
- "CVXPY found a solution" (solver can return inaccurate results)

## Implementation Checklist

Before deploying ANY portfolio optimizer:

- [ ] Ledoit-Wolf shrinkage for covariance (NO sample covariance fallback)
- [ ] CVXPY for constrained optimization (scipy.optimize only for unconstrained)
- [ ] Assert constraint satisfaction (weights sum to 1, long-only, limits)
- [ ] Model all transaction costs (commission + slippage + impact + spread)
- [ ] Validate optimization status (check solver didn't return "infeasible")
- [ ] Out-of-sample testing (walk-forward validation, not in-sample only)

## Optimization Method Selection

### Decision Framework

**Mean-Variance Optimization (MVO)**:
- ✅ Use when: Strong return forecasts, small universe (< 30 assets), short horizon
- ❌ Avoid when: Noisy covariance, many assets, no return views
- **Strengths**: Maximizes expected return, incorporates views directly
- **Weaknesses**: Unstable, extreme weights, sensitive to estimation error

**Hierarchical Risk Parity (HRP)**:
- ✅ Use when: Noisy covariance, many assets (30+), no return views
- ❌ Avoid when: Strong return forecasts, need to tilt toward specific sectors
- **Strengths**: Stable, diversified, no matrix inversion
- **Weaknesses**: Ignores expected returns, equal treatment of assets

**Black-Litterman (BL)**:
- ✅ Use when: SOME views (not complete), varying confidence, benchmark tracking
- ❌ Avoid when: Complete views on all assets, no benchmark
- **Strengths**: Blends views with equilibrium, handles uncertainty
- **Weaknesses**: Complex, requires reverse optimization, sensitive to τ

---

## Pattern 1: Mean-Variance with Ledoit-Wolf Shrinkage (MANDATORY)

### Covariance Estimation

**NEVER use sample covariance. ALWAYS use Ledoit-Wolf shrinkage.**

```python
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def calculate_covariance_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate covariance matrix with Ledoit-Wolf shrinkage.

    CRITICAL: Sample covariance is unstable for portfolio optimization.
    Shrinkage reduces estimation error by blending with structured estimator.

    Args:
        returns: DataFrame of asset returns (rows=dates, cols=assets)

    Returns:
        Shrunk covariance matrix (annualized)

    Raises:
        ImportError: If sklearn not available (NO fallback to sample covariance)
    """
    # ENFORCE: sklearn must be available
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        raise ImportError(
            "sklearn required for Ledoit-Wolf shrinkage. "
            "Sample covariance is NOT acceptable for portfolio optimization. "
            "Install: pip install scikit-learn"
        )

    # Fit Ledoit-Wolf estimator
    lw = LedoitWolf()
    lw.fit(returns)

    # VALIDATE: Shrinkage was applied
    assert hasattr(lw, 'shrinkage_'), "Ledoit-Wolf shrinkage failed"
    assert 0 <= lw.shrinkage_ <= 1, \
        f"Invalid shrinkage intensity: {lw.shrinkage_}"

    # Annualize (multiply by trading days)
    cov_annual = lw.covariance_ * 252

    # VALIDATE: Covariance is positive definite
    eigenvalues = np.linalg.eigvalsh(cov_annual)
    assert (eigenvalues > 0).all(), \
        f"Covariance not positive definite: min eigenvalue = {eigenvalues.min()}"

    # Log diagnostics
    cond_number = np.linalg.cond(cov_annual)
    print(f"✓ Ledoit-Wolf shrinkage: {lw.shrinkage_:.3f}")
    print(f"✓ Condition number: {cond_number:.1f}")

    return pd.DataFrame(cov_annual, index=returns.columns, columns=returns.columns)
```

---

### Constrained Optimization with CVXPY

**Use CVXPY for constrained optimization. Validate constraints with assertions.**

```python
import cvxpy as cp


def optimize_portfolio_cvxpy(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_position: float = 0.15,
    max_volatility: float = 0.18,
    sector_constraints: dict = None
) -> dict:
    """
    Optimize portfolio using CVXPY with constraint validation.

    Args:
        expected_returns: Expected annual returns for each asset
        cov_matrix: Covariance matrix (Ledoit-Wolf shrinkage applied)
        max_position: Maximum weight per asset (default 15%)
        max_volatility: Maximum portfolio volatility (default 18%)
        sector_constraints: Dict of sector -> max_weight

    Returns:
        Dict with weights, return, volatility, status

    Raises:
        ValueError: If optimization fails or constraints violated
    """
    n_assets = len(expected_returns)

    # Define optimization variables
    w = cp.Variable(n_assets)

    # Portfolio statistics
    portfolio_return = expected_returns @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)

    # Objective: Maximize expected return
    objective = cp.Maximize(portfolio_return)

    # Constraints
    tolerance = 1e-4
    constraints = [
        # Fully invested
        cp.sum(w) == 1,

        # Long-only (no shorting)
        w >= 0,

        # Maximum position size
        w <= max_position - tolerance,

        # Volatility ceiling
        portfolio_variance <= (max_volatility - tolerance) ** 2
    ]

    # Sector constraints
    if sector_constraints:
        for sector_indices, max_weight in sector_constraints.items():
            sector_exposure = cp.sum(w[sector_indices])
            constraints.append(sector_exposure <= max_weight + tolerance)

    # Solve problem
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception as e:
        raise ValueError(f"Optimization failed: {e}")

    # VALIDATE: Optimization succeeded
    assert problem.status in ['optimal', 'optimal_inaccurate'], \
        f"Optimization failed with status: {problem.status}"

    # Extract weights
    optimal_weights = w.value

    # POST-OPTIMIZATION VALIDATION (MANDATORY)
    validate_portfolio_constraints(
        optimal_weights,
        max_position,
        max_volatility,
        cov_matrix
    )

    # Calculate portfolio statistics
    realized_return = expected_returns @ optimal_weights
    realized_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)

    return {
        'weights': optimal_weights,
        'expected_return': realized_return,
        'volatility': realized_vol,
        'sharpe': realized_return / realized_vol,
        'status': problem.status
    }


def validate_portfolio_constraints(
    weights: np.ndarray,
    max_position: float,
    max_volatility: float,
    cov_matrix: np.ndarray,
    tolerance: float = 1e-4
) -> None:
    """
    Validate portfolio constraints with assertions (NOT logging).

    CRITICAL: Use assertions to ENFORCE constraints, not just log.
    Logging can be ignored; assertions cannot.

    Args:
        weights: Portfolio weights
        max_position: Maximum position size
        max_volatility: Maximum volatility
        cov_matrix: Covariance matrix
        tolerance: Numerical tolerance (default 1e-4)

    Raises:
        AssertionError: If any constraint violated
    """
    # ENFORCE: Weights sum to 1
    assert abs(weights.sum() - 1.0) < tolerance, \
        f"Weights sum to {weights.sum():.6f}, not 1.0"

    # ENFORCE: Long-only
    assert (weights >= -tolerance).all(), \
        f"Negative weights found: {weights[weights < -tolerance]}"

    # ENFORCE: Maximum position
    assert (weights <= max_position + tolerance).all(), \
        f"Position limit violated: max={weights.max():.2%} > {max_position:.2%}"

    # ENFORCE: Volatility ceiling
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    assert portfolio_vol <= max_volatility + 0.001, \
        f"Volatility {portfolio_vol:.2%} > ceiling {max_volatility:.2%}"

    print(f"✓ All constraints satisfied (tolerance={tolerance})")
```

---

## Pattern 2: Hierarchical Risk Parity (HRP)

### Complete HRP Implementation

```python
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def hierarchical_risk_parity(cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Implement Hierarchical Risk Parity (Lopez de Prado 2016).

    HRP is more stable than mean-variance because it doesn't invert covariance.
    Uses hierarchical clustering to build diversified portfolios.

    Args:
        cov_matrix: Covariance matrix with ticker labels

    Returns:
        Portfolio weights (sum to 1.0)
    """
    # Step 1: Compute distance matrix from correlation
    corr_matrix = cov_to_corr(cov_matrix)
    dist_matrix = compute_distance_matrix(corr_matrix)

    # Step 2: Hierarchical clustering (single linkage)
    linkage_matrix = linkage(squareform(dist_matrix), method='single')

    # Step 3: Quasi-diagonalize covariance matrix
    ordered_indices = quasi_diagonalize(linkage_matrix, len(cov_matrix))
    ordered_cov = cov_matrix.iloc[ordered_indices, ordered_indices]

    # Step 4: Recursive bisection to allocate weights
    weights = recursive_bisection(ordered_cov)

    # Create Series with original order
    weights_series = pd.Series(weights, index=ordered_cov.index)
    weights_series = weights_series.reindex(cov_matrix.index)

    # VALIDATE: HRP weights
    validate_hrp_weights(weights_series)

    return weights_series


def compute_distance_matrix(corr_matrix: pd.DataFrame) -> np.ndarray:
    """
    Compute distance matrix from correlation (Lopez de Prado formula).

    CRITICAL: Use sqrt(0.5 * (1 - correlation)), not other distance metrics.
    This ensures distance ∈ [0, 1] and satisfies metric properties.

    Args:
        corr_matrix: Correlation matrix

    Returns:
        Distance matrix
    """
    # CORRECT distance metric for HRP
    dist = np.sqrt(0.5 * (1 - corr_matrix))

    # Ensure diagonal is exactly zero
    np.fill_diagonal(dist.values, 0)

    # VALIDATE: Distance properties
    assert (dist >= 0).all().all(), "Distance must be non-negative"
    assert (dist <= 1.0001).all().all(), "Distance must be <= 1"  # Small tolerance

    return dist.values


def quasi_diagonalize(linkage_matrix: np.ndarray, n_assets: int) -> list:
    """
    Quasi-diagonalize using depth-first search on dendrogram.

    Reorders assets so similar assets are adjacent, creating block-diagonal structure.

    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        n_assets: Number of assets

    Returns:
        List of asset indices in quasi-diagonal order
    """
    ordered_indices = []

    def dfs(node_id: int):
        """Depth-first search to extract leaf order."""
        if node_id < n_assets:
            # Leaf node (original asset)
            ordered_indices.append(node_id)
        else:
            # Internal node
            cluster_id = node_id - n_assets
            left_child = int(linkage_matrix[cluster_id, 0])
            right_child = int(linkage_matrix[cluster_id, 1])
            dfs(left_child)
            dfs(right_child)

    # Start DFS from root (last cluster formed)
    root_id = 2 * n_assets - 2
    dfs(root_id)

    return ordered_indices


def recursive_bisection(cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    Recursive bisection to allocate weights based on cluster variance.

    Splits portfolio into two clusters, allocates weight inversely proportional
    to variance (lower variance → higher weight).

    Args:
        cov_matrix: Quasi-diagonalized covariance matrix

    Returns:
        Portfolio weights
    """
    n_assets = cov_matrix.shape[0]
    weights = np.ones(n_assets)

    def bisect(cluster_indices: list, weight: float):
        """Recursively bisect cluster and allocate weight."""
        if len(cluster_indices) == 1:
            # Base case: single asset
            weights[cluster_indices[0]] = weight
            return

        # Split cluster in half
        mid = len(cluster_indices) // 2
        left_cluster = cluster_indices[:mid]
        right_cluster = cluster_indices[mid:]

        # Compute cluster variances
        left_var = cluster_variance(cov_matrix, left_cluster)
        right_var = cluster_variance(cov_matrix, right_cluster)

        # Allocate weight inversely proportional to variance
        total_inv_var = (1.0 / left_var) + (1.0 / right_var)
        left_weight = weight * (1.0 / left_var) / total_inv_var
        right_weight = weight * (1.0 / right_var) / total_inv_var

        # Recursive calls
        bisect(left_cluster, left_weight)
        bisect(right_cluster, right_weight)

    # Start recursion with all assets and weight = 1.0
    bisect(list(range(n_assets)), 1.0)

    return weights


def cluster_variance(cov_matrix: pd.DataFrame, cluster_indices: list) -> float:
    """
    Compute variance of a cluster assuming equal weights.

    Args:
        cov_matrix: Covariance matrix
        cluster_indices: Indices of assets in cluster

    Returns:
        Cluster variance
    """
    cluster_cov = cov_matrix.iloc[cluster_indices, cluster_indices].values
    n = len(cluster_indices)
    weights = np.ones(n) / n
    variance = weights @ cluster_cov @ weights

    return variance


def validate_hrp_weights(weights: pd.Series) -> None:
    """
    Validate HRP weights with assertions.

    Args:
        weights: Portfolio weights

    Raises:
        AssertionError: If weights invalid
    """
    # ENFORCE: No NaN or infinite
    assert not weights.isna().any(), "Weights contain NaN"
    assert not np.isinf(weights).any(), "Weights contain infinite values"

    # ENFORCE: Non-negative (HRP is long-only by design)
    assert (weights >= -1e-6).all(), \
        f"Negative weights found: {weights[weights < 0]}"

    # ENFORCE: Sum to 1
    assert abs(weights.sum() - 1.0) < 1e-4, \
        f"Weights sum to {weights.sum():.6f}, not 1.0"

    print(f"✓ HRP weights validated: sum={weights.sum():.6f}, "
          f"min={weights.min():.4f}, max={weights.max():.4f}")
```

---

## Pattern 3: Transaction Cost Modeling (ALL COMPONENTS)

### Complete Cost Model

**Model all four components: commission + slippage + market impact + bid-ask spread.**

```python
def calculate_complete_transaction_costs(
    trades: np.ndarray,
    prices: np.ndarray,
    ADV: np.ndarray,  # Average Daily Volume
    is_liquid: np.ndarray,
    fixed_commission: float = 5.0
) -> float:
    """
    Calculate complete transaction costs (all components).

    CRITICAL: Don't simplify to single cost number (e.g., 10 bps).
    Each component matters and scales differently with trade size.

    Components:
    1. Fixed commission (per trade, independent of size)
    2. Slippage (percentage of trade value, depends on liquidity)
    3. Market impact (square root law, depends on volume)
    4. Bid-ask spread (percentage of trade value, depends on liquidity)

    Args:
        trades: Trade sizes in shares (positive = buy, negative = sell)
        prices: Current prices per share
        ADV: Average daily volume per stock
        is_liquid: Boolean array (True = liquid stock)
        fixed_commission: Fixed cost per trade (default $5)

    Returns:
        Total transaction costs in dollars
    """
    trade_values = np.abs(trades * prices)
    num_trades = (trades != 0).sum()

    # Component 1: Fixed commission
    commission = fixed_commission * num_trades

    # Component 2: Slippage (depends on liquidity)
    # Liquid stocks: 3 bps, Illiquid stocks: 10 bps
    slippage_rates = np.where(is_liquid, 0.0003, 0.001)
    slippage = (trade_values * slippage_rates).sum()

    # Component 3: Market impact (square root law)
    # impact = 0.5% * sqrt(trade_value / ADV)
    # Captures that large trades move the market
    impact_coefficient = 0.005
    impact = sum(
        impact_coefficient * abs(trade_values[i]) *
        np.sqrt(abs(trades[i]) / ADV[i])
        for i in range(len(trades))
        if trades[i] != 0 and ADV[i] > 0
    )

    # Component 4: Bid-ask spread (depends on liquidity)
    # Liquid stocks: 5 bps, Illiquid stocks: 20 bps
    spread_rates = np.where(is_liquid, 0.0005, 0.002)
    spread = (trade_values * spread_rates).sum()

    total_costs = commission + slippage + impact + spread

    # VALIDATE: Costs are reasonable (< 1% of total trade value)
    total_trade_value = trade_values.sum()
    cost_percentage = total_costs / total_trade_value if total_trade_value > 0 else 0

    assert cost_percentage < 0.01, \
        f"Transaction costs {cost_percentage:.2%} > 1% seem unreasonably high"

    print(f"Transaction Costs Breakdown:")
    print(f"  Commission:  ${commission:>10,.2f} ({commission/total_costs:.1%})")
    print(f"  Slippage:    ${slippage:>10,.2f} ({slippage/total_costs:.1%})")
    print(f"  Impact:      ${impact:>10,.2f} ({impact/total_costs:.1%})")
    print(f"  Spread:      ${spread:>10,.2f} ({spread/total_costs:.1%})")
    print(f"  TOTAL:       ${total_costs:>10,.2f} ({cost_percentage:.2%} of trades)")

    return total_costs
```

---

## Pattern 4: Black-Litterman Model

### Complete Black-Litterman Implementation

```python
def black_litterman(
    market_weights: np.ndarray,
    cov_matrix: np.ndarray,
    P: np.ndarray,  # Pick matrix (views)
    Q: np.ndarray,  # View returns
    confidence: np.ndarray,  # Confidence levels (0-1)
    tau: float = 0.025,
    risk_aversion: float = 2.5
) -> np.ndarray:
    """
    Black-Litterman model: Blend market equilibrium with investor views.

    Formula: E[R] = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 Π + P'Ω^-1 Q]
    where:
    - Π = equilibrium returns (from market weights)
    - P = pick matrix (which assets views apply to)
    - Q = view returns
    - Ω = view uncertainty matrix (scaled by confidence)
    - τ = scaling factor (typically 0.025 to 0.05)

    Args:
        market_weights: Market cap weights (equilibrium)
        cov_matrix: Covariance matrix (Ledoit-Wolf shrinkage)
        P: Pick matrix (n_views × n_assets)
        Q: View returns (n_views,)
        confidence: Confidence in each view (n_views,), range [0, 1]
        tau: Scaling factor (default 0.025)
        risk_aversion: Risk aversion parameter (default 2.5)

    Returns:
        Posterior expected returns
    """
    n_assets = len(market_weights)
    n_views = len(Q)

    # VALIDATE inputs
    assert P.shape == (n_views, n_assets), \
        f"P matrix shape {P.shape} != ({n_views}, {n_assets})"
    assert len(Q) == n_views, f"Q length {len(Q)} != {n_views}"
    assert len(confidence) == n_views, f"Confidence length {len(confidence)} != {n_views}"
    assert all(0 <= c <= 1 for c in confidence), "Confidence must be in [0, 1]"

    # Step 1: Reverse optimization - compute equilibrium returns (Π)
    # Π = λ * Σ * w_mkt
    equilibrium_returns = risk_aversion * cov_matrix @ market_weights

    # Step 2: Construct view uncertainty matrix (Ω)
    # Ω = diag(P * τ * Σ * P') / confidence
    # Higher confidence → lower uncertainty
    Omega = np.diag(
        np.diag(P @ (tau * cov_matrix) @ P.T) / confidence
    )

    # VALIDATE: Omega is positive definite
    omega_eigenvalues = np.linalg.eigvalsh(Omega)
    assert (omega_eigenvalues > 0).all(), \
        "Omega matrix not positive definite"

    # Step 3: Compute posterior returns (Black-Litterman formula)
    tau_cov_inv = np.linalg.inv(tau * cov_matrix)
    P_omega_inv = P.T @ np.linalg.inv(Omega)

    # Posterior covariance
    posterior_cov_inv = tau_cov_inv + P_omega_inv @ P

    # Posterior returns
    posterior_returns = np.linalg.inv(posterior_cov_inv) @ (
        tau_cov_inv @ equilibrium_returns + P_omega_inv @ Q
    )

    # VALIDATE: Posterior returns are reasonable
    # Should be between equilibrium and views (weighted by confidence)
    assert not np.isnan(posterior_returns).any(), \
        "Posterior returns contain NaN"
    assert not np.isinf(posterior_returns).any(), \
        "Posterior returns contain infinite values"

    # Check posterior is plausible (between min/max of equilibrium and views)
    combined = np.concatenate([equilibrium_returns, Q])
    assert posterior_returns.min() >= combined.min() - 0.05, \
        "Posterior returns unreasonably low"
    assert posterior_returns.max() <= combined.max() + 0.05, \
        "Posterior returns unreasonably high"

    print(f"✓ Black-Litterman posterior returns computed")
    print(f"  Equilibrium range: [{equilibrium_returns.min():.2%}, {equilibrium_returns.max():.2%}]")
    print(f"  Posterior range:   [{posterior_returns.min():.2%}, {posterior_returns.max():.2%}]")

    return posterior_returns
```

### Constructing the P Matrix (Pick Matrix)

```python
def construct_pick_matrix_absolute(
    view_assets: list,
    n_assets: int
) -> np.ndarray:
    """
    Construct P matrix for absolute views (single asset).

    Example: "AAPL will return 12%"

    Args:
        view_assets: List of asset indices for each view
        n_assets: Total number of assets

    Returns:
        P matrix (n_views × n_assets)
    """
    n_views = len(view_assets)
    P = np.zeros((n_views, n_assets))

    for i, asset_idx in enumerate(view_assets):
        P[i, asset_idx] = 1.0

    return P


def construct_pick_matrix_relative(
    view_pairs: list,
    n_assets: int
) -> np.ndarray:
    """
    Construct P matrix for relative views (asset pairs).

    Example: "AAPL will outperform MSFT by 5%"

    Args:
        view_pairs: List of (asset1_idx, asset2_idx) tuples
        n_assets: Total number of assets

    Returns:
        P matrix (n_views × n_assets)
    """
    n_views = len(view_pairs)
    P = np.zeros((n_views, n_assets))

    for i, (asset1, asset2) in enumerate(view_pairs):
        P[i, asset1] = 1.0
        P[i, asset2] = -1.0

    return P
```

---

## Common Mistakes

### Mistake 1: Using Sample Covariance
❌ **Bad:**
```python
cov_matrix = returns.cov() * 252  # Sample covariance (unstable)
```

✅ **Good:**
```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw.fit(returns)
cov_matrix = lw.covariance_ * 252

# ENFORCE: Shrinkage was applied
assert hasattr(lw, 'shrinkage_'), "Ledoit-Wolf failed"
```

### Mistake 2: Validation with Logging Only
❌ **Bad:**
```python
if weights.sum() != 1.0:
    logger.warning(f"Weights sum to {weights.sum()}")  # Can be ignored
```

✅ **Good:**
```python
assert abs(weights.sum() - 1.0) < 1e-6, \
    f"Weights sum to {weights.sum():.6f}, not 1.0"  # Enforces correctness
```

### Mistake 3: Simplified Transaction Costs
❌ **Bad:**
```python
transaction_costs = 0.001 * turnover  # Only turnover (too simple)
```

✅ **Good:**
```python
costs = (
    commission +
    slippage +
    market_impact +  # Square root law
    bid_ask_spread
)  # All components
```

### Mistake 4: Wrong HRP Distance Metric
❌ **Bad:**
```python
distance = 1 - correlation  # Wrong metric
```

✅ **Good:**
```python
distance = np.sqrt(0.5 * (1 - correlation))  # Lopez de Prado formula
```

### Mistake 5: Black-Litterman Without View Uncertainty
❌ **Bad:**
```python
# Treats views as certain (Omega = 0)
posterior = equilibrium + P.T @ (Q - P @ equilibrium)
```

✅ **Good:**
```python
# Models view uncertainty (Omega scaled by confidence)
Omega = np.diag(np.diag(P @ (tau * cov) @ P.T) / confidence)
posterior = compute_bl_posterior(equilibrium, P, Q, Omega, tau, cov)
```

---

## Rationalization Table

| Excuse | Reality | Fix |
|--------|---------|-----|
| "Ledoit-Wolf is standard, I used it" | Need guarantee, not intention | Assert shrinkage applied, no fallback |
| "Validation logging is sufficient" | Logs can be ignored | Use assertions to enforce constraints |
| "Transaction costs are 10 bps" | Components scale differently | Model commission + slippage + impact + spread |
| "CVXPY found optimal solution" | Solver can return "inaccurate" | Validate constraints post-optimization |
| "Sample covariance for small universe" | Still unstable (30+ assets) | Always use Ledoit-Wolf shrinkage |
| "HRP algorithm is implemented" | Distance metric matters | Use sqrt(0.5 * (1 - ρ)), not 1 - ρ |
| "Black-Litterman blends views" | Need view uncertainty | Model Omega matrix scaled by confidence |

---

## Real-World Impact

**Proper enforcement prevents:**
- Portfolio weights of 80% in one stock (from unstable covariance)
- Turnover of 200%/quarter (from ignoring transaction costs)
- Out-of-sample Sharpe collapsing from 2.0 to 0.5 (from overfitting)
- Constraint violations (35% in Tech when limit is 30%)
- Optimization failures silently returning invalid portfolios

**Time investment:**
- Implement Ledoit-Wolf: 2 minutes (copy pattern)
- Add constraint assertions: 3 minutes
- Model complete transaction costs: 5 minutes
- Implement Black-Litterman: 15 minutes
- Debug unstable portfolio from sample covariance: 4+ hours

---

## Method Comparison Table

| Feature | Mean-Variance | HRP | Black-Litterman |
|---------|--------------|-----|----------------|
| **Requires return forecasts** | Yes (all assets) | No | Yes (some assets) |
| **Requires covariance** | Yes | Yes | Yes |
| **Matrix inversion** | Yes (unstable) | No (stable) | Yes (but regularized) |
| **Out-of-sample stability** | Low | High | Medium |
| **Computational cost** | Low | Medium | Medium |
| **Incorporates views** | Direct | None | Blended with equilibrium |
| **Best for** | <30 assets, strong views | 30+ assets, no views | Some views, benchmark tracking |

---

## Bottom Line

**Portfolio optimization requires three components:**
1. Algorithm implementation (mean-variance, HRP, Black-Litterman)
2. Enforcement (Ledoit-Wolf shrinkage, constraint assertions) ← AGENTS SKIP
3. Complete modeling (all transaction costs, view uncertainty) ← AGENTS SKIP

Don't just implement the algorithm—enforce guarantees and model completely.

Every optimizer gets Ledoit-Wolf shrinkage (no fallback). Every solution gets assertion-based validation (not logging). Every rebalancing models complete transaction costs. No exceptions.
