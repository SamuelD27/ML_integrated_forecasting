# Financial Formulas Reference

Comprehensive reference for financial calculations with interpretive ranges and validation bounds.

---

## VALUATION METRICS

### P/E Ratio (Price-to-Earnings)
```
Formula: Market Price / EPS

Interpretation:
  <10:    Undervalued (or distressed company)
  15-25:  Fair value (market average)
  >25:    Expensive (growth premium or bubble)

Validation:
  assert P_E > 0, "Negative P/E indicates losses"
  assert P_E < 100, f"P/E {P_E} suspiciously high"
```

### PEG Ratio (P/E to Growth)
```
Formula: (P/E Ratio) / (Annual EPS Growth %)

Interpretation:
  <1:   Undervalued relative to growth
  1-2:  Fair value
  >2:   Expensive relative to growth

Validation:
  assert growth_rate > 0, "Negative growth makes PEG invalid"
  assert 0 < PEG < 5, f"PEG {PEG:.2f} outside normal range"
```

### EV/EBITDA (Enterprise Value to EBITDA)
```
Formula: (Market Cap + Total Debt - Cash) / EBITDA

Interpretation:
  Industry-specific typical ranges:
  Tech:       10-20x
  Industrial: 8-15x
  Utilities:  6-12x
  Retail:     5-10x

Validation:
  assert EBITDA > 0, "Negative EBITDA makes ratio invalid"
  assert 2 < EV_EBITDA < 50, f"EV/EBITDA {EV_EBITDA:.1f} extreme"
```

### P/B Ratio (Price-to-Book)
```
Formula: Market Price / Book Value Per Share

Interpretation:
  <1:   Value play (trading below book value)
  1-2:  Fair value
  >3:   Growth premium (intangibles, brand value)

Validation:
  assert book_value > 0, "Negative book value problematic"
  assert P_B > 0, "Negative P/B indicates accounting issues"
```

### FCF Yield (Free Cash Flow Yield)
```
Formula: Free Cash Flow / Market Cap

Interpretation:
  <3%:  Expensive (low cash generation)
  3-7%: Fair value
  >7%:  Deep value (high cash generation or distressed)

Validation:
  assert -0.10 < FCF_Yield < 0.30, f"FCF Yield {FCF_Yield:.1%} extreme"
```

---

## PROFITABILITY METRICS

### ROE (Return on Equity)
```
Formula: Net Income / Shareholders' Equity

Interpretation:
  >15%:  Exceptional (competitive moat)
  10-15%: Good
  <10%:  Weak (poor capital efficiency)

Validation:
  assert shareholders_equity > 0, "Negative equity problematic"
  assert -0.50 < ROE < 1.50, f"ROE {ROE:.1%} outside normal range"
```

### ROA (Return on Assets)
```
Formula: Net Income / Total Assets

Interpretation:
  >10%:  Excellent asset utilization
  5-10%: Good
  <5%:   Weak asset efficiency

Validation:
  assert total_assets > 0, "Total assets must be positive"
  assert -0.20 < ROA < 0.50, f"ROA {ROA:.1%} extreme"
```

### Profit Margin
```
Formula: Net Income / Revenue

Interpretation:
  >20%:  Excellent (pricing power)
  10-20%: Good
  <5%:   Weak (commodity business)

Validation:
  assert revenue > 0, "Revenue must be positive"
  assert -0.50 < profit_margin < 1.0, f"Margin {profit_margin:.1%} extreme"
```

---

## RISK METRICS

### Sharpe Ratio
```
Formula: (Portfolio Return - Risk-Free Rate) / Standard Deviation of Returns

Interpretation:
  <0:     Negative risk-adjusted return
  0-1:    Poor (not beating risk-free rate efficiently)
  1-2:    Good
  2-3:    Excellent
  >3:     Suspicious (likely data error or overfitting)

Validation:
  assert std_dev > 1e-10, "Zero std dev requires special handling"
  assert -3 < sharpe < 5, f"Sharpe {sharpe:.2f} outside realistic bounds"

Annualization:
  Daily returns:   multiply by sqrt(252)
  Monthly returns: multiply by sqrt(12)
  Annual returns:  no adjustment needed
```

### Sortino Ratio
```
Formula: (Portfolio Return - Risk-Free Rate) / Downside Standard Deviation

Interpretation:
  Typically 20-50% higher than Sharpe (only penalizes downside)
  Same thresholds as Sharpe, but expect higher values

Validation:
  assert downside_std > 1e-10, "Zero downside std requires handling"
  assert -2 < sortino < 10, f"Sortino {sortino:.2f} suspicious"

Calculation Note:
  Downside returns = returns[returns < 0]
  Downside std = sqrt(mean(downside_returns^2))
```

### Calmar Ratio
```
Formula: Annual Return / Maximum Drawdown

Interpretation:
  >1:   Good (return exceeds worst drawdown)
  1-3:  Excellent
  >3:   Suspicious (too good to be true)

Validation:
  assert max_drawdown < 0, "Max drawdown must be negative"
  assert 0 < calmar < 10, f"Calmar {calmar:.2f} outside realistic range"
```

### Maximum Drawdown
```
Formula: (Trough Value - Peak Value) / Peak Value

Interpretation:
  10-20%: Good (low risk)
  20-40%: Acceptable (moderate risk)
  >40%:   High risk (significant capital erosion)

Validation:
  assert -0.99 < max_dd < 0, f"Max DD {max_dd:.1%} outside bounds"

Calculation:
  running_max = cumulative_returns.expanding().max()
  drawdowns = (cumulative_returns - running_max) / running_max
  max_dd = drawdowns.min()
```

### Value at Risk (VaR)
```
Formula: Xth percentile of returns distribution
  95% VaR: 5th percentile
  99% VaR: 1st percentile

Interpretation:
  95% VaR of -0.05: 5% chance of losing >5% in given period
  99% VaR of -0.10: 1% chance of losing >10%

Validation:
  assert 0.90 < confidence < 0.99, "Confidence must be 90-99%"
  assert -0.50 < VaR < 0, f"VaR {VaR:.1%} outside typical range"

Calculation:
  VaR = np.percentile(returns, (1 - confidence) * 100)
```

### Conditional VaR (CVaR / Expected Shortfall)
```
Formula: Mean of all returns worse than VaR threshold

Interpretation:
  Better than VaR for tail risk (average loss in worst cases)
  Typically 20-50% worse than VaR

Validation:
  assert CVaR < VaR < 0, "CVaR must be worse than VaR"
  assert -0.70 < CVaR, f"CVaR {CVaR:.1%} extremely pessimistic"

Calculation:
  threshold = np.percentile(returns, (1 - confidence) * 100)
  CVaR = returns[returns < threshold].mean()
```

---

## TECHNICAL INDICATORS

### RSI (Relative Strength Index)
```
Formula: 100 - (100 / (1 + RS))
  where RS = Average Gain / Average Loss (14 periods)

Interpretation:
  >70:  Overbought (potential reversal down)
  30-70: Neutral range
  <30:  Oversold (potential reversal up)

Validation:
  assert 0 <= RSI <= 100, f"RSI {RSI:.1f} outside [0, 100]"

Calculation:
  gains = returns[returns > 0].mean()
  losses = abs(returns[returns < 0].mean())
  RS = gains / (losses + 1e-10)
  RSI = 100 - (100 / (1 + RS))
```

### MACD (Moving Average Convergence Divergence)
```
Formula:
  MACD Line = 12-period EMA - 26-period EMA
  Signal Line = 9-period EMA of MACD
  Histogram = MACD Line - Signal Line

Interpretation:
  MACD crosses above Signal: Bullish signal
  MACD crosses below Signal: Bearish signal
  Histogram magnitude: Momentum strength

Validation:
  # MACD is price-scale dependent, no absolute bounds
  # Validate signal crossings make sense
```

### Bollinger Bands
```
Formula:
  Middle Band = 20-period SMA
  Upper Band = SMA + (2 × 20-period Std Dev)
  Lower Band = SMA - (2 × 20-period Std Dev)

Interpretation:
  Price > Upper Band: Overbought condition
  Price < Lower Band: Oversold condition
  Band width: Volatility measure

Validation:
  assert upper_band > middle_band > lower_band, "Bands misordered"
  position = (price - lower) / (upper - lower + 1e-10)
  assert -0.5 < position < 1.5, f"Position {position:.2f} too extreme"
```

### ADX (Average Directional Index)
```
Formula: Smoothed average of directional movement indicators (complex)

Interpretation:
  <20:   No trend (choppy market)
  20-40: Moderate trend
  >40:   Strong trend

Validation:
  assert 0 <= ADX <= 100, f"ADX {ADX:.1f} outside [0, 100]"
```

---

## OPTIONS GREEKS

### Delta (Δ)
```
Formula: ∂V/∂S (rate of option value change with underlying price)

Interpretation:
  Call Options:
    ATM: ~0.50 (50% probability of finishing ITM)
    ITM: approaching 1.0
    OTM: approaching 0.0

  Put Options:
    ATM: ~-0.50
    ITM: approaching -1.0
    OTM: approaching 0.0

Validation:
  if option_type == 'call':
      assert 0 <= delta <= 1, f"Call delta {delta:.3f} outside [0, 1]"
  else:
      assert -1 <= delta <= 0, f"Put delta {delta:.3f} outside [-1, 0]"
```

### Gamma (Γ)
```
Formula: ∂²V/∂S² = ∂Δ/∂S (delta sensitivity to price changes)

Interpretation:
  Always positive for long options (convexity)
  Highest for ATM options near expiration
  Long gamma: Profit from volatility
  Short gamma: Lose from volatility

Validation:
  assert gamma >= 0, f"Gamma {gamma:.6f} must be non-negative"
  assert gamma < 1.0, f"Gamma {gamma:.6f} unrealistically high"
```

### Vega (ν)
```
Formula: ∂V/∂σ (sensitivity to implied volatility changes)

Interpretation:
  Always positive for long options
  Highest for ATM options far from expiration
  Vega of 0.15: $15 gain per 1% IV increase (on 100-share contract)

Validation:
  assert vega >= 0, f"Vega {vega:.4f} should be non-negative (long options)"
  # Vega can be large for long-dated options, no strict upper bound
```

### Theta (Θ)
```
Formula: ∂V/∂t (time decay per day)

Interpretation:
  Negative for long options (lose value daily)
  Positive for short options (collect time decay)
  Accelerates near expiration (exponential decay)
  Theta of -0.05: Lose $5/day per contract

Validation:
  if position == 'long':
      assert theta < 0.5, f"Long theta {theta:.4f} should be negative"
  # No strict bounds (theta accelerates near expiry)
```

### Rho (ρ)
```
Formula: ∂V/∂r (sensitivity to interest rate changes)

Interpretation:
  Small for equity options (rates stable)
  Important for bond/currency options
  Calls: positive rho
  Puts: negative rho

Validation:
  # Rho typically very small for equity options
  if asset_class == 'equity':
      assert abs(rho) < 1.0, f"Equity rho {rho:.4f} unexpectedly large"
```

---

## PORTFOLIO OPTIMIZATION

### Black-Litterman Model
```
Market Equilibrium Returns:
  π = λ × Σ × w_market

  λ = risk aversion parameter (typically 2.5-3.5)
  Σ = covariance matrix
  w_market = market-cap weighted portfolio

ML Views Integration:
  v = model predictions (expected returns vector)
  confidence = 1 / prediction_variance
  Ω = diag(1 / confidence) (view uncertainty matrix)

Posterior Expected Returns:
  P = pick matrix (which assets have views)
  E[R] = π + Σ × P^T × (P × Σ × P^T + Ω)^(-1) × (v - P × π)

Validation:
  assert np.all(np.isfinite(posterior_returns)), "Non-finite posteriors"
  assert np.all(eigenvalues(cov_posterior) > -1e-10), "Non-PSD covariance"
```

### Kelly Criterion
```
Formula (Binary Outcomes):
  f* = (p × b - q) / b

  p = win probability
  q = 1 - p (loss probability)
  b = win/loss ratio (avg_win / avg_loss)

Formula (General):
  f* = (Win_Rate × Avg_Win - Loss_Rate × Avg_Loss) / Avg_Win

Fractional Kelly (RECOMMENDED):
  Half-Kelly: f = f* / 2
  Quarter-Kelly: f = f* / 4

Interpretation:
  Full Kelly: 41.7% → DANGEROUS (high drawdowns)
  Half Kelly: 20.9% → Aggressive
  Quarter Kelly: 10.4% → Conservative (recommended)

Validation:
  assert 0 < win_rate < 1, f"Win rate {win_rate:.1%} invalid"
  assert avg_win > 0 and avg_loss > 0, "Win/loss must be positive"
  assert 0 < kelly_fraction <= 1, "Fractional Kelly must be in (0, 1]"

  position_size = kelly * fractional
  assert 0 <= position_size <= 0.50, \
      f"Position {position_size:.1%} exceeds 50% safety limit"

Safety Warning:
  NEVER use full Kelly in production
  Estimation error + market regime change = ruin
  Minimum 500 trades before trusting win rate
```

### Mean-Variance Optimization
```
Objective: Minimize variance for target return

  min w^T Σ w
  subject to:
    w^T μ >= R_target (return constraint)
    w^T 1 = 1 (weights sum to 1)
    w_i >= 0 (long-only) OR w_i >= -1 (allow shorting)

Validation:
  assert np.isclose(weights.sum(), 1.0, atol=1e-6), "Weights must sum to 1"
  assert np.all(eigenvalues(cov_matrix) > -1e-10), "Covariance not PSD"

  portfolio_return = weights @ expected_returns
  assert portfolio_return >= target_return - 1e-6, "Return constraint violated"
```

---

## VALIDATION TEMPLATES

### Returns Series Validation
```python
def validate_returns_series(returns: np.ndarray, min_samples: int = 20):
    assert isinstance(returns, (np.ndarray, pd.Series)), \
        "Returns must be numpy array or pandas Series"
    assert len(returns) >= min_samples, \
        f"Need at least {min_samples} samples, got {len(returns)}"
    assert not np.any(np.isnan(returns)), \
        "Returns contain NaN values"
    assert not np.any(np.isinf(returns)), \
        "Returns contain infinite values"
    assert np.all(np.abs(returns) < 1.0), \
        f"Daily returns > 100% suspicious: max={returns.max():.1%}"
```

### Correlation Matrix Validation
```python
def validate_correlation_matrix(corr: np.ndarray):
    # Symmetry
    assert np.allclose(corr, corr.T, atol=1e-8), "Not symmetric"

    # Diagonal = 1
    assert np.allclose(np.diag(corr), 1.0, atol=1e-8), "Diagonal != 1"

    # Range [-1, 1]
    assert np.all(corr >= -1.0) and np.all(corr <= 1.0), "Values outside [-1, 1]"

    # Positive semi-definite
    eigenvalues = np.linalg.eigvals(corr)
    min_eig = eigenvalues.min()
    assert min_eig > -1e-10, f"Not PSD: min eigenvalue = {min_eig:.6f}"
```

### Portfolio Weights Validation
```python
def validate_portfolio_weights(weights: np.ndarray,
                               allow_short: bool = False,
                               max_single_position: float = 0.30):
    # Sum to 1
    assert np.isclose(weights.sum(), 1.0, atol=1e-6), \
        f"Weights sum to {weights.sum():.6f}, not 1.0"

    # Bounds
    if allow_short:
        assert np.all(weights >= -1.0) and np.all(weights <= 1.0), \
            "Weights outside [-1, 1]"
    else:
        assert np.all(weights >= 0.0) and np.all(weights <= 1.0), \
            "Weights outside [0, 1] (long-only)"

    # Concentration
    assert np.all(np.abs(weights) <= max_single_position), \
        f"Position exceeds {max_single_position:.0%} limit: max={np.abs(weights).max():.1%}"
```

---

## COMMON PITFALLS

### Pitfall 1: Annualization Errors
```python
# ❌ WRONG: Mixing annualized and daily rates
sharpe = (daily_returns.mean() - 0.02) / daily_returns.std()

# ✅ CORRECT: Convert risk-free rate to daily
rf_daily = 0.02 / 252
sharpe = (daily_returns.mean() - rf_daily) / daily_returns.std()
sharpe_annualized = sharpe * np.sqrt(252)
```

### Pitfall 2: Zero Division
```python
# ❌ WRONG: No edge case handling
sharpe = excess_returns.mean() / excess_returns.std()

# ✅ CORRECT: Handle zero std dev
std = excess_returns.std()
sharpe = 0.0 if std < 1e-10 else excess_returns.mean() / std
```

### Pitfall 3: Trusting User Inputs
```python
# ❌ WRONG: Assume inputs are valid
kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

# ✅ CORRECT: Validate first
assert 0 < win_rate < 1, f"Invalid win_rate: {win_rate}"
assert avg_win > 0 and avg_loss > 0, "Win/loss must be positive"
kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
assert 0 <= kelly <= 1, f"Kelly {kelly:.1%} outside [0, 100%]"
```

---

## REFERENCES

- **Sharpe (1994):** "The Sharpe Ratio" - original paper
- **Black & Litterman (1992):** "Global Portfolio Optimization" - BL model
- **Kelly (1956):** "A New Interpretation of Information Rate" - Kelly criterion
- **Ledoit & Wolf (2004):** "Honey, I Shrunk the Sample Covariance Matrix" - covariance estimation
- **López de Prado (2018):** "Advances in Financial Machine Learning" - walk-forward validation, purged k-fold

---

**Use this reference when implementing any financial calculation to ensure correct formulas and realistic bounds.**
