---
name: time-series-validation-specialist
description: Use when backtesting trading strategies or validating ML models on time series data - prevents look-ahead bias, data leakage, and overfitting through walk-forward validation, embargo periods, and temporal splitting to ensure models generalize beyond training data
---

# Time Series Validation Specialist

## Overview

**Time series validation is systematic prevention of leakage, not ad-hoc review.** Every backtest must implement walk-forward validation with embargo periods, check for look-ahead bias programmatically, and validate across multiple market regimes.

**Core principle:** Understanding validation concepts ≠ implementing validation code. Always implement systematic checks.

## When to Use

Use this skill when:
- Backtesting trading strategies
- Training ML models on financial time series
- Validating predictive models (price, volatility, regime)
- Splitting data for time series analysis
- Reviewing features for data leakage
- Conducting model selection or hyperparameter tuning

**Don't skip validation because:**
- "Code runs without errors" (leakage is silent)
- "Obvious leakage is caught" (subtle leakage is more dangerous)
- "Using existing patterns" (need to verify correct usage)
- "High performance questioned" (need automated validation, not judgment)
- "Standard ML practices apply" (time series is special)

## Validation Checklist

Before deploying ANY time series model or backtest:

- [ ] Walk-forward validation (not random split)
- [ ] Embargo period between train and test (minimum T+2 for daily)
- [ ] Look-ahead bias check (features use only t-1 or earlier)
- [ ] Target leakage check (features don't contain target)
- [ ] Regime validation (backtest spans bull + bear markets)
- [ ] Performance degradation monitored (out-of-sample vs in-sample)

## Implementation Patterns

### Pattern 1: Walk-Forward Validation (MANDATORY)

**NEVER use random train_test_split for time series. ALWAYS use walk-forward.**

```python
def walk_forward_split(
    data: pd.DataFrame,
    lookback_days: int = 252,  # 1 year training
    test_days: int = 63,        # 3 months testing
    step_days: int = 21,        # 1 month step
    embargo_days: int = 2       # T+2 settlement
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward split preserving temporal order.

    Args:
        data: Time series dataframe with DatetimeIndex
        lookback_days: Training window size
        test_days: Test window size
        step_days: How far to move forward each iteration
        embargo_days: Gap between train and test (prevents overlap)

    Yields:
        train_idx, test_idx: Arrays of integer positions
    """
    # Validate inputs
    assert isinstance(data.index, pd.DatetimeIndex), \
        "Data must have DatetimeIndex"
    assert lookback_days > 0, "lookback_days must be positive"
    assert test_days > 0, "test_days must be positive"
    assert embargo_days >= 0, "embargo_days must be non-negative"

    n = len(data)
    train_end = lookback_days

    while train_end + embargo_days + test_days <= n:
        train_start = max(0, train_end - lookback_days)
        train_idx = np.arange(train_start, train_end)

        test_start = train_end + embargo_days
        test_end = test_start + test_days
        test_idx = np.arange(test_start, test_end)

        # VALIDATE: No overlap between train and test
        assert train_idx[-1] + embargo_days < test_idx[0], \
            f"Train/test overlap detected"

        yield train_idx, test_idx

        # Move forward
        train_end += step_days
```

**Usage:**
```python
# For backtesting
for train_idx, test_idx in walk_forward_split(data):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    model.fit(train_data)
    predictions = model.predict(test_data)

    # Evaluate on this fold
    sharpe = calculate_sharpe(predictions, test_data['returns'])
    print(f"Fold Sharpe: {sharpe:.2f}")
```

### Pattern 2: Look-Ahead Bias Detection

**Check that ALL features use only past data.**

```python
def check_look_ahead_bias(
    features: pd.DataFrame,
    target: pd.Series,
    feature_timestamps: Dict[str, pd.Series],
    target_timestamps: pd.Series
) -> None:
    """
    Validate no look-ahead bias in features.

    Args:
        features: Feature dataframe
        target: Target series
        feature_timestamps: Timestamp for each feature
        target_timestamps: Timestamp for each target value

    Raises:
        AssertionError: If look-ahead bias detected
    """
    for col in features.columns:
        if col not in feature_timestamps:
            print(f"WARNING: No timestamp info for {col}, skipping check")
            continue

        feat_times = feature_timestamps[col]

        # CRITICAL: Feature timestamp must be <= target timestamp
        assert (feat_times <= target_timestamps).all(), \
            f"Feature '{col}' contains future data (look-ahead bias)"

        # Additional check: Feature should use t-1 or earlier
        # Allow small tolerance for timestamp alignment (1 second)
        time_diff = (target_timestamps - feat_times).dt.total_seconds()
        assert (time_diff >= -1).all(), \
            f"Feature '{col}' timestamp is after target timestamp"

        print(f"✓ {col}: No look-ahead bias detected")
```

### Pattern 3: Target Leakage Detection

**Check that features don't contain the target.**

```python
def check_target_leakage(
    features: pd.DataFrame,
    target: pd.Series,
    correlation_threshold: float = 0.90
) -> None:
    """
    Detect target leakage via correlation.

    High correlation (>0.90) often indicates leakage.

    Args:
        features: Feature dataframe
        target: Target series
        correlation_threshold: Threshold for suspicious correlation

    Raises:
        AssertionError: If leakage detected
    """
    for col in features.columns:
        # Skip if feature is constant
        if features[col].std() < 1e-10:
            print(f"WARNING: {col} is constant, skipping")
            continue

        # Calculate correlation
        corr = features[col].corr(target)

        # HIGH correlation is suspicious
        if abs(corr) > correlation_threshold:
            raise ValueError(
                f"LEAKAGE DETECTED: Feature '{col}' has correlation "
                f"{corr:.3f} with target (threshold: {correlation_threshold}). "
                f"This usually indicates target leakage."
            )

        # MODERATE correlation deserves investigation
        if abs(corr) > 0.75:
            print(f"WARNING: {col} correlation {corr:.3f} is high. "
                  f"Verify no leakage.")

        print(f"✓ {col}: Correlation {corr:.3f} (OK)")
```

### Pattern 4: Embargo Period Calculation

**Always include embargo period between train and test.**

```python
def calculate_embargo_period(
    data_frequency: str,
    settlement_days: int = 2,
    market_hours_per_day: float = 6.5
) -> int:
    """
    Calculate minimum embargo period.

    Accounts for settlement lag and information propagation.

    Args:
        data_frequency: 'daily', 'hourly', '5min', '1min'
        settlement_days: T+N settlement (default T+2 for US stocks)
        market_hours_per_day: Trading hours per day

    Returns:
        Embargo period in data frequency units
    """
    if data_frequency == 'daily':
        return settlement_days

    elif data_frequency == 'hourly':
        return int(settlement_days * market_hours_per_day)

    elif data_frequency in ['5min', '1min']:
        minutes_per_day = int(market_hours_per_day * 60)

        if data_frequency == '5min':
            periods_per_day = minutes_per_day // 5
        else:  # 1min
            periods_per_day = minutes_per_day

        return settlement_days * periods_per_day

    else:
        raise ValueError(f"Unknown frequency: {data_frequency}")
```

**Usage:**
```python
embargo = calculate_embargo_period('daily', settlement_days=2)  # Returns 2

# Use in walk-forward split
for train_idx, test_idx in walk_forward_split(data, embargo_days=embargo):
    # ... train and test
```

### Pattern 5: Regime Validation

**Validate performance across multiple market conditions.**

```python
def validate_across_regimes(
    backtest_results: pd.DataFrame,
    regime_labels: pd.Series,
    min_sharpe_ratio: float = 0.3
) -> Dict[str, float]:
    """
    Validate model performance across market regimes.

    Args:
        backtest_results: DataFrame with returns and predictions
        regime_labels: Series with regime for each timestamp
        min_sharpe_ratio: Minimum acceptable Sharpe per regime

    Returns:
        Dict of Sharpe ratios by regime

    Raises:
        AssertionError: If performance degrades too much in any regime
    """
    regimes = regime_labels.unique()

    sharpe_by_regime = {}

    for regime in regimes:
        # Get returns for this regime
        regime_mask = regime_labels == regime
        regime_returns = backtest_results.loc[regime_mask, 'returns']

        if len(regime_returns) < 20:
            print(f"WARNING: Only {len(regime_returns)} samples in {regime}")
            continue

        # Calculate Sharpe
        sharpe = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252)
        sharpe_by_regime[regime] = sharpe

        print(f"{regime}: Sharpe = {sharpe:.2f}")

        # VALIDATE: Performance should be positive in all regimes
        assert sharpe > min_sharpe_ratio, \
            f"Sharpe {sharpe:.2f} in {regime} below minimum {min_sharpe_ratio}"

    # VALIDATE: Performance consistency across regimes
    sharpe_values = list(sharpe_by_regime.values())
    sharpe_std = np.std(sharpe_values)
    sharpe_mean = np.mean(sharpe_values)

    if sharpe_mean > 0:
        cv = sharpe_std / sharpe_mean  # Coefficient of variation

        # High CV indicates regime-specific overfitting
        if cv > 1.0:
            print(f"WARNING: High Sharpe variation across regimes (CV={cv:.2f})")
            print("This suggests regime-specific overfitting")

    return sharpe_by_regime
```

## Common Mistakes

### Mistake 1: Using Random Split
❌ **Bad:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

✅ **Good:**
```python
# Walk-forward preserving temporal order
for train_idx, test_idx in walk_forward_split(data):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
```

### Mistake 2: No Embargo Period
❌ **Bad:**
```python
train_data = data[:1000]
test_data = data[1000:1250]  # Starts immediately after train
```

✅ **Good:**
```python
embargo = 2  # T+2 settlement
train_data = data[:1000]
test_data = data[1000 + embargo:1250]  # Gap enforced
```

### Mistake 3: Features Use Future Data
❌ **Bad:**
```python
# Volume spike using FUTURE 5-day average
features['vol_spike'] = current_vol / future_5day_avg_vol
```

✅ **Good:**
```python
# Volume spike using PAST 5-day average
features['vol_spike'] = current_vol / past_5day_avg_vol.shift(1)
```

## Rationalization Table

| Excuse | Reality | Fix |
|--------|---------|-----|
| "Code runs without errors" | Leakage is silent | Add systematic validation checks |
| "Obvious leakage is caught" | Subtle leakage is more dangerous | Implement automated checks |
| "Random split is standard ML" | Wrong for time series | Use walk-forward always |
| "Existing patterns are used" | Need to verify correct usage | Check implementation, not existence |
| "High performance questioned" | Need automated validation | Add `assert sharpe < 5` as code |
| "train_test_split works fine" | Causes 30-50% overstatement | Walk-forward only |

## Validation Utilities Reference

For complete implementation examples, see [validation-utilities.md](validation-utilities.md):
- Walk-forward splitter (expanding/rolling window)
- Purged k-fold for time series
- Look-ahead bias detector
- Target leakage checker
- Embargo period calculator
- Regime validator

## Real-World Impact

**Proper validation prevents:**
- Sharpe ratio overstatement by 30-50% (from random split)
- 95% accuracy that becomes 50% in production (from leakage)
- Strategies that work in bull markets, fail in bear markets
- Models using tomorrow's price to predict tomorrow's price
- Backtests that can't be replicated in live trading

**Time investment:**
- Implement walk-forward: 5 minutes (copy template)
- Add leakage checks: 3 minutes per feature set
- Validate across regimes: 2 minutes
- Debug production failure from leakage: 8+ hours + losses

## Bottom Line

**Validation is systematic prevention, not ad-hoc review.**

Don't just understand concepts—implement checks as code.

Every time series model gets walk-forward validation. Every feature set gets leakage checks. Every backtest validates across regimes. No exceptions.
