---
name: financial-knowledge-validator
description: Use when implementing financial calculations, risk metrics, or portfolio optimization - validates all formulas with assertions and bounds checking before deployment, preventing calculation errors in Sharpe ratios, Kelly criterion, Greeks, correlation matrices, and other quantitative finance metrics
---

# Financial Knowledge Validator

## Overview

**Validation as code, not advice.** Every financial calculation must include programmatic validation: input assertions, edge case handling, output bounds checking, and mathematical property verification.

**Core principle:** Warnings get ignored under pressure. Assertions prevent disasters.

## When to Use

Use this skill when implementing:
- Risk metrics (Sharpe, Sortino, VaR, CVaR, max drawdown)
- Position sizing (Kelly criterion, volatility targeting)
- Portfolio optimization (mean-variance, Black-Litterman, HRP)
- Options pricing (Greeks, Black-Scholes, volatility surfaces)
- Correlation/covariance matrices
- Any quantitative finance formula

**Don't skip validation because:**
- "Formula is mathematically correct" (correct ≠ robust)
- "User is an expert" (experts make input errors)
- "Just a quick calculation" (quick bugs cause real losses)
- "I'll add validation later" (you won't, and pressure will skip it)

## Validation Checklist

Before implementing ANY financial formula:

- [ ] Assert input types and ranges (e.g., `assert 0 < win_rate < 1`)
- [ ] Handle edge cases (zero division, negative roots, infinities)
- [ ] Validate output against realistic bounds (e.g., Sharpe in [-3, 5])
- [ ] Check mathematical properties (symmetry, positive definiteness)
- [ ] Add interpretive comments (what bounds mean financially)

## Validation Patterns by Context

### Development vs Production Code

**CRITICAL:** Python assertions can be disabled with `python -O` flag. Choose validation pattern based on context:

**Development/Scripts (Assertions OK):**
```python
def sharpe_ratio(returns, risk_free_rate=0.02):
    assert isinstance(returns, np.ndarray), "Returns must be numpy array"
    assert len(returns) > 0, "Returns cannot be empty"
    # ... calculation
```

**Production/Libraries (Explicit Checks Required):**
```python
def sharpe_ratio(returns, risk_free_rate=0.02):
    if not isinstance(returns, np.ndarray):
        raise TypeError("Returns must be numpy array")
    if len(returns) == 0:
        raise ValueError("Returns cannot be empty")
    # ... calculation
```

**When to use which:**
- Scripts/Notebooks: Assertions are fine (fast, clear)
- Libraries/APIs: Explicit checks (cannot be disabled)
- Production systems: Explicit checks always
- Tests: Both (assertions for test logic, explicit checks for validation)

## Implementation Patterns

### Pattern 1: Input Validation (Development)

```python
def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                   fractional: float = 0.25) -> float:
    """Calculate Kelly position size with validation."""

    # INPUT VALIDATION (MANDATORY)
    assert 0 < win_rate < 1, f"Win rate {win_rate} must be in (0, 1)"
    assert avg_win > 0, f"Average win {avg_win} must be positive"
    assert avg_loss > 0, f"Average loss {avg_loss} must be positive"
    assert 0 < fractional <= 1, f"Fractional {fractional} must be in (0, 1]"

    loss_rate = 1 - win_rate
    full_kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
    position_size = full_kelly * fractional

    # OUTPUT VALIDATION (MANDATORY)
    assert 0 <= position_size <= 0.50, \
        f"Position size {position_size:.1%} exceeds 50% safety limit"

    return position_size
```

### Pattern 1b: Input Validation (Production)

```python
def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                   fractional: float = 0.25) -> float:
    """Calculate Kelly position size with production-grade validation."""

    # INPUT VALIDATION (PRODUCTION - cannot be disabled)
    if not (0 < win_rate < 1):
        raise ValueError(f"Win rate {win_rate:.1%} must be in (0, 1)")
    if avg_win <= 0:
        raise ValueError(f"Average win ${avg_win:.2f} must be positive")
    if avg_loss <= 0:
        raise ValueError(f"Average loss ${avg_loss:.2f} must be positive")
    if not (0 < fractional <= 1):
        raise ValueError(f"Fractional {fractional:.1%} must be in (0, 1]")

    loss_rate = 1 - win_rate
    full_kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
    position_size = full_kelly * fractional

    # OUTPUT VALIDATION (PRODUCTION)
    if position_size > 0.50:
        raise ValueError(
            f"Position size {position_size:.1%} exceeds 50% safety limit. "
            f"Risk of ruin is unacceptable."
        )

    return position_size
```

### Pattern 2: Edge Case Handling

```python
def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio with edge case handling."""

    # TYPE AND SHAPE VALIDATION
    assert isinstance(returns, np.ndarray), "Returns must be numpy array"
    assert len(returns) > 0, "Returns array cannot be empty"

    excess_returns = returns - risk_free_rate

    # EDGE CASE: Zero standard deviation
    std = excess_returns.std()
    if std < 1e-10:
        return 0.0  # Constant returns = no risk-adjusted return

    sharpe = excess_returns.mean() / std

    # OUTPUT BOUNDS CHECK
    assert -3 < sharpe < 5, \
        f"Sharpe {sharpe:.2f} outside realistic bounds [-3, 5] - likely calculation error"

    return sharpe
```

### Pattern 3: Mathematical Property Validation

```python
def validate_correlation_matrix(corr_matrix: np.ndarray) -> bool:
    """Validate correlation matrix properties."""

    # SYMMETRY
    assert np.allclose(corr_matrix, corr_matrix.T), \
        "Correlation matrix not symmetric"

    # DIAGONAL = 1
    assert np.allclose(np.diag(corr_matrix), 1.0), \
        "Diagonal elements not 1.0"

    # VALUES IN [-1, 1]
    assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0), \
        "Correlation values outside [-1, 1]"

    # POSITIVE SEMI-DEFINITE
    eigenvalues = np.linalg.eigvals(corr_matrix)
    assert np.all(eigenvalues > -1e-10), \
        f"Correlation matrix not positive semi-definite (min eigenvalue: {eigenvalues.min():.6f})"

    return True
```

### Pattern 4: Portfolio Weights Validation

```python
def validate_portfolio_weights(weights: np.ndarray,
                               allow_short: bool = False) -> bool:
    """Validate portfolio weights sum to 1.0 and respect constraints."""

    # SUM TO 1.0 (with tolerance)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6), \
        f"Weights sum to {weights.sum():.6f}, not 1.0"

    # LONG-ONLY OR LONG/SHORT BOUNDS
    if allow_short:
        assert np.all(weights >= -1.0) and np.all(weights <= 1.0), \
            f"Weights outside [-1, 1] range: min={weights.min():.3f}, max={weights.max():.3f}"
    else:
        assert np.all(weights >= 0) and np.all(weights <= 1.0), \
            f"Weights outside [0, 1] range (long-only): min={weights.min():.3f}"

    return True
```

## Reusable Validator Library

Store validators in `validators.py`:

```python
class FinancialValidators:
    """Centralized validation functions for financial calculations."""

    @staticmethod
    def validate_returns(returns, min_samples=20):
        assert isinstance(returns, (np.ndarray, pd.Series))
        assert len(returns) >= min_samples, f"Need ≥{min_samples} samples"
        assert not np.any(np.isnan(returns)), "Returns contain NaN"
        assert not np.any(np.isinf(returns)), "Returns contain infinity"

    @staticmethod
    def validate_sharpe(sharpe, max_value=5.0):
        assert -3 < sharpe < max_value, f"Sharpe {sharpe:.2f} suspicious"

    @staticmethod
    def validate_greeks(delta, gamma, option_type='call'):
        if option_type == 'call':
            assert 0 <= delta <= 1, f"Call delta {delta:.3f} ∉ [0,1]"
        else:
            assert -1 <= delta <= 0, f"Put delta {delta:.3f} ∉ [-1,0]"
        assert gamma >= 0, f"Gamma {gamma:.6f} must be non-negative"
```

## Common Mistakes

### Mistake 1: Advice Instead of Assertions
❌ **Bad:**
```python
# WARNING: Sharpe > 3 is suspicious
sharpe = calculate_sharpe(returns)
print(f"Sharpe: {sharpe:.2f}")
```

✅ **Good:**
```python
sharpe = calculate_sharpe(returns)
assert -3 < sharpe < 5, f"Sharpe {sharpe:.2f} outside bounds"
```

### Mistake 2: Missing Edge Cases
❌ **Bad:**
```python
sharpe = excess_returns.mean() / excess_returns.std()
```

✅ **Good:**
```python
std = excess_returns.std()
sharpe = 0.0 if std < 1e-10 else excess_returns.mean() / std
```

### Mistake 3: No Input Validation
❌ **Bad:**
```python
def kelly(win_rate, avg_win, avg_loss):
    return (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
```

✅ **Good:**
```python
def kelly(win_rate, avg_win, avg_loss):
    assert 0 < win_rate < 1, f"Invalid win_rate: {win_rate}"
    assert avg_win > 0 and avg_loss > 0, "Wins/losses must be positive"
    # ... rest of calculation
```

## Rationalization Table

| Excuse | Reality | Fix |
|--------|---------|-----|
| "Formula is mathematically correct" | Correct ≠ robust | Add input/output validation |
| "User is an expert" | Experts make input errors | Validate anyway |
| "I warned about the risks" | Warnings get ignored | Enforce with assertions |
| "Output looks reasonable" | Reasonable ≠ validated | Check against known bounds |
| "Just a quick calculation" | Quick bugs cause real losses | Takes 30 seconds to validate |
| "I'll add validation later" | Never happens, pressure skips it | Validate now or delete code |
| "Assertions are fine for validation" | Can be disabled with -O flag | Use explicit checks in production |
| "I'll follow the spirit not letter" | Both are required | Use pattern appropriate to context |

## Financial Formulas Reference

For detailed formulas with interpretive ranges, see [financial-formulas-reference.md](financial-formulas-reference.md):
- Valuation metrics (P/E, PEG, EV/EBITDA, P/B, FCF Yield)
- Profitability metrics (ROE, ROA, Profit Margin)
- Risk metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
- Technical indicators (RSI, MACD, Bollinger Bands, ADX)
- Options Greeks (Delta, Gamma, Vega, Theta, Rho)
- Portfolio optimization (Black-Litterman, Kelly Criterion)

## Real-World Impact

**Validation prevents:**
- Portfolio weights summing to 1.08 (10% position size error)
- Sharpe ratio of 12.5 (data error undetected for months)
- Negative gamma causing $50K loss (wrong hedge direction)
- Kelly criterion suggesting 85% position (account ruin)
- Singular correlation matrix crashing optimizer

**Time investment:**
- Add validation: 30 seconds per function
- Debug production bug: 3-8 hours + client trust damage

## Bottom Line

**Transform advice into enforceable code.**

Don't explain why values are wrong—prevent wrong values with assertions.

Every financial calculation gets validation. No exceptions.
