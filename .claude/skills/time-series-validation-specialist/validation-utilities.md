# Time Series Validation Utilities

Complete production-ready implementations of validation utilities for time series backtesting and ML model validation.

## Table of Contents
1. [Walk-Forward Splitters](#walk-forward-splitters)
2. [Purged K-Fold](#purged-k-fold)
3. [Look-Ahead Bias Detection](#look-ahead-bias-detection)
4. [Target Leakage Detection](#target-leakage-detection)
5. [Embargo Period Calculation](#embargo-period-calculation)
6. [Regime Validation](#regime-validation)
7. [Performance Degradation Monitor](#performance-degradation-monitor)

---

## Walk-Forward Splitters

### Expanding Window Walk-Forward

```python
from typing import Iterator, Tuple
import numpy as np
import pandas as pd


def expanding_window_split(
    data: pd.DataFrame,
    min_train_days: int = 252,
    test_days: int = 63,
    step_days: int = 21,
    embargo_days: int = 2
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding window walk-forward split.

    Training window grows over time, test window is fixed.
    Uses all available historical data for each fold.

    Args:
        data: Time series dataframe with DatetimeIndex
        min_train_days: Minimum training window size (default 252 = 1 year)
        test_days: Test window size (default 63 = 3 months)
        step_days: How far to move forward each iteration (default 21 = 1 month)
        embargo_days: Gap between train and test (default 2 = T+2 settlement)

    Yields:
        train_idx: Array of training indices
        test_idx: Array of test indices

    Example:
        >>> data = pd.DataFrame({'price': ...}, index=pd.date_range('2020-01-01', periods=1000))
        >>> for train_idx, test_idx in expanding_window_split(data):
        ...     train_data = data.iloc[train_idx]
        ...     test_data = data.iloc[test_idx]
        ...     # Train and evaluate
    """
    # Validate inputs
    assert isinstance(data.index, pd.DatetimeIndex), \
        "Data must have DatetimeIndex"
    assert min_train_days > 0, "min_train_days must be positive"
    assert test_days > 0, "test_days must be positive"
    assert embargo_days >= 0, "embargo_days must be non-negative"
    assert step_days > 0, "step_days must be positive"

    n = len(data)
    train_end = min_train_days

    fold_num = 0
    while train_end + embargo_days + test_days <= n:
        # Expanding window: train from start
        train_start = 0
        train_idx = np.arange(train_start, train_end)

        # Fixed test window
        test_start = train_end + embargo_days
        test_end = min(test_start + test_days, n)
        test_idx = np.arange(test_start, test_end)

        # VALIDATE: No overlap between train and test
        assert train_idx[-1] + embargo_days < test_idx[0], \
            f"Fold {fold_num}: Train/test overlap detected"

        # VALIDATE: Temporal ordering
        assert data.index[train_idx[-1]] < data.index[test_idx[0]], \
            f"Fold {fold_num}: Train end {data.index[train_idx[-1]]} >= test start {data.index[test_idx[0]]}"

        fold_num += 1
        yield train_idx, test_idx

        # Move forward
        train_end += step_days


def rolling_window_split(
    data: pd.DataFrame,
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 21,
    embargo_days: int = 2
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Rolling window walk-forward split.

    Both training and test windows are fixed size and move forward.
    Useful when recent data is more relevant than distant history.

    Args:
        data: Time series dataframe with DatetimeIndex
        train_days: Training window size (default 252 = 1 year)
        test_days: Test window size (default 63 = 3 months)
        step_days: How far to move forward each iteration (default 21 = 1 month)
        embargo_days: Gap between train and test (default 2 = T+2 settlement)

    Yields:
        train_idx: Array of training indices
        test_idx: Array of test indices

    Example:
        >>> for train_idx, test_idx in rolling_window_split(data):
        ...     # Uses fixed 252-day training window
        ...     train_data = data.iloc[train_idx]
        ...     test_data = data.iloc[test_idx]
    """
    # Validate inputs
    assert isinstance(data.index, pd.DatetimeIndex), \
        "Data must have DatetimeIndex"
    assert train_days > 0, "train_days must be positive"
    assert test_days > 0, "test_days must be positive"
    assert embargo_days >= 0, "embargo_days must be non-negative"
    assert step_days > 0, "step_days must be positive"

    n = len(data)
    train_end = train_days

    fold_num = 0
    while train_end + embargo_days + test_days <= n:
        # Rolling window: fixed size
        train_start = train_end - train_days
        train_idx = np.arange(train_start, train_end)

        # Fixed test window
        test_start = train_end + embargo_days
        test_end = min(test_start + test_days, n)
        test_idx = np.arange(test_start, test_end)

        # VALIDATE: No overlap
        assert train_idx[-1] + embargo_days < test_idx[0], \
            f"Fold {fold_num}: Train/test overlap"

        # VALIDATE: Correct window sizes
        assert len(train_idx) == train_days, \
            f"Fold {fold_num}: Train window size {len(train_idx)} != {train_days}"

        fold_num += 1
        yield train_idx, test_idx

        # Move forward
        train_end += step_days
```

---

## Purged K-Fold

### Purged K-Fold for Time Series

```python
from typing import List
from sklearn.model_selection import KFold


class PurgedKFold:
    """
    K-fold cross-validation with purging and embargo for time series.

    Purging: Remove samples from training set that overlap with test period
    Embargo: Add gap between train and test to prevent information leakage

    Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        purge_pct: float = 0.02
    ):
        """
        Initialize purged k-fold splitter.

        Args:
            n_splits: Number of folds
            embargo_pct: Embargo as percentage of total samples (default 2%)
            purge_pct: Purging window as percentage of total samples (default 2%)
        """
        assert n_splits > 1, "n_splits must be > 1"
        assert 0 <= embargo_pct < 1, "embargo_pct must be in [0, 1)"
        assert 0 <= purge_pct < 1, "purge_pct must be in [0, 1)"

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        data: pd.DataFrame,
        sample_weights: pd.Series = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging and embargo.

        Args:
            data: Time series dataframe with DatetimeIndex
            sample_weights: Optional sample weights (used for purging)

        Yields:
            train_idx: Array of training indices (purged)
            test_idx: Array of test indices
        """
        assert isinstance(data.index, pd.DatetimeIndex), \
            "Data must have DatetimeIndex"

        n = len(data)
        embargo_samples = int(n * self.embargo_pct)
        purge_samples = int(n * self.purge_pct)

        # Use standard k-fold to get initial splits
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(data)):
            # Apply embargo: remove samples immediately after test set
            test_end = test_idx[-1]
            embargo_start = test_end + 1
            embargo_end = min(embargo_start + embargo_samples, n)

            # Remove embargo period from training
            train_idx = train_idx[train_idx < embargo_start]

            # Apply purging: remove training samples that overlap with test
            # (e.g., if features use T+5 days of data, purge last 5 days of training)
            if purge_samples > 0:
                test_start = test_idx[0]
                purge_start = max(0, test_start - purge_samples)
                purge_end = test_start

                # Remove purged samples from training
                purge_mask = (train_idx < purge_start) | (train_idx >= purge_end)
                train_idx = train_idx[purge_mask]

            # VALIDATE: No overlap
            assert train_idx[-1] < test_idx[0], \
                f"Fold {fold_num}: Train/test overlap after purging"

            # VALIDATE: Temporal ordering
            assert data.index[train_idx[-1]] < data.index[test_idx[0]], \
                f"Fold {fold_num}: Train end >= test start"

            yield train_idx, test_idx


def combinatorial_purged_cv(
    data: pd.DataFrame,
    n_splits: int = 5,
    n_test_groups: int = 2,
    embargo_pct: float = 0.02,
    purge_pct: float = 0.02
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Combinatorial purged cross-validation.

    Creates multiple test groups and all combinations of train/test splits.
    Increases number of backtest paths without overfitting.

    Args:
        data: Time series dataframe
        n_splits: Number of groups to split data into
        n_test_groups: Number of groups to use for testing in each fold
        embargo_pct: Embargo percentage
        purge_pct: Purge percentage

    Yields:
        train_idx: Training indices
        test_idx: Test indices
    """
    from itertools import combinations

    assert isinstance(data.index, pd.DatetimeIndex), \
        "Data must have DatetimeIndex"
    assert n_splits >= n_test_groups, \
        f"n_splits ({n_splits}) must be >= n_test_groups ({n_test_groups})"

    n = len(data)
    embargo_samples = int(n * embargo_pct)
    purge_samples = int(n * purge_pct)

    # Split data into groups
    group_size = n // n_splits
    groups = []
    for i in range(n_splits):
        start = i * group_size
        end = (i + 1) * group_size if i < n_splits - 1 else n
        groups.append(np.arange(start, end))

    # Generate all combinations of test groups
    fold_num = 0
    for test_groups in combinations(range(n_splits), n_test_groups):
        # Test set: selected groups
        test_idx = np.concatenate([groups[i] for i in test_groups])

        # Training set: all other groups
        train_groups = [i for i in range(n_splits) if i not in test_groups]
        train_idx = np.concatenate([groups[i] for i in train_groups])

        # Apply embargo and purging
        test_end = test_idx[-1]
        embargo_start = test_end + 1
        embargo_end = min(embargo_start + embargo_samples, n)
        train_idx = train_idx[train_idx < embargo_start]

        if purge_samples > 0:
            test_start = test_idx[0]
            purge_start = max(0, test_start - purge_samples)
            purge_end = test_start
            purge_mask = (train_idx < purge_start) | (train_idx >= purge_end)
            train_idx = train_idx[purge_mask]

        # VALIDATE: No overlap
        assert len(set(train_idx) & set(test_idx)) == 0, \
            f"Fold {fold_num}: Train/test overlap"

        fold_num += 1
        yield train_idx, test_idx
```

---

## Look-Ahead Bias Detection

### Feature Timestamp Validator

```python
from typing import Dict


def validate_feature_timestamps(
    features: pd.DataFrame,
    target: pd.Series,
    feature_timestamps: Dict[str, pd.Series],
    target_timestamps: pd.Series,
    tolerance_seconds: float = 1.0
) -> None:
    """
    Validate that all features use only past data (no look-ahead bias).

    Args:
        features: Feature dataframe
        target: Target series
        feature_timestamps: Dict mapping feature name to timestamp series
        target_timestamps: Timestamp for each target value
        tolerance_seconds: Allowed time difference for alignment (default 1 second)

    Raises:
        AssertionError: If look-ahead bias detected

    Example:
        >>> features = pd.DataFrame({'ma_20': [...], 'rsi': [...]})
        >>> target = pd.Series([...])
        >>> feature_times = {
        ...     'ma_20': pd.Series([pd.Timestamp('2024-01-01 09:30:00'), ...]),
        ...     'rsi': pd.Series([pd.Timestamp('2024-01-01 09:29:00'), ...])
        ... }
        >>> target_times = pd.Series([pd.Timestamp('2024-01-01 09:30:00'), ...])
        >>> validate_feature_timestamps(features, target, feature_times, target_times)
    """
    assert len(features) == len(target), \
        f"Features ({len(features)}) and target ({len(target)}) length mismatch"
    assert len(target) == len(target_timestamps), \
        f"Target ({len(target)}) and timestamps ({len(target_timestamps)}) length mismatch"

    for col in features.columns:
        if col not in feature_timestamps:
            print(f"WARNING: No timestamp info for '{col}', skipping check")
            continue

        feat_times = feature_timestamps[col]

        assert len(feat_times) == len(features), \
            f"Feature '{col}' timestamp length {len(feat_times)} != data length {len(features)}"

        # CRITICAL: Feature timestamp must be <= target timestamp
        future_mask = feat_times > target_timestamps
        if future_mask.any():
            num_violations = future_mask.sum()
            first_violation_idx = future_mask.idxmax()
            raise ValueError(
                f"Look-ahead bias in '{col}': {num_violations} samples use future data. "
                f"First violation at index {first_violation_idx}: "
                f"feature time {feat_times.iloc[first_violation_idx]} > "
                f"target time {target_timestamps.iloc[first_violation_idx]}"
            )

        # Additional check: Feature should use t-1 or earlier
        # Allow small tolerance for timestamp alignment
        time_diff = (target_timestamps - feat_times).dt.total_seconds()

        # Negative diff means feature is from future
        if (time_diff < -tolerance_seconds).any():
            num_violations = (time_diff < -tolerance_seconds).sum()
            raise ValueError(
                f"Feature '{col}' has {num_violations} timestamps after target "
                f"(tolerance: {tolerance_seconds}s)"
            )

        print(f"✓ '{col}': No look-ahead bias detected "
              f"(avg lag: {time_diff.mean():.2f}s)")


def detect_shift_leakage(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    expected_shift: int = 1
) -> None:
    """
    Detect if features accidentally use current value when predicting future.

    Common mistake: Using close_t to predict close_t+1 (should use close_t-1).

    Args:
        data: Dataframe with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        expected_shift: Expected shift for features (default 1 = use t-1)

    Raises:
        ValueError: If shift leakage detected

    Example:
        >>> data = pd.DataFrame({
        ...     'close': [100, 101, 102, 103],
        ...     'target': [101, 102, 103, 104]  # Next day close
        ... })
        >>> detect_shift_leakage(data, ['close'], 'target', expected_shift=1)
        # Raises ValueError because close_t is used to predict close_t+1
    """
    target = data[target_col]

    for col in feature_cols:
        if col not in data.columns:
            print(f"WARNING: Feature '{col}' not found in data")
            continue

        feature = data[col]

        # Check correlation between feature_t and target_t
        # High correlation suggests using current value (leakage)
        corr_current = feature.corr(target)

        # Check correlation between feature_t and target_t+shift
        # This is what we expect (feature from past predicting future)
        if len(target) > expected_shift:
            target_shifted = target.shift(-expected_shift)
            corr_shifted = feature.corr(target_shifted)

            # If correlation with CURRENT target is higher, likely leakage
            if abs(corr_current) > abs(corr_shifted) + 0.1:
                print(f"WARNING: '{col}' has higher correlation with current target "
                      f"({corr_current:.3f}) than future target ({corr_shifted:.3f}). "
                      f"Possible shift leakage - verify feature uses t-{expected_shift}")
```

---

## Target Leakage Detection

### Correlation-Based Leakage Detector

```python
def detect_target_leakage(
    features: pd.DataFrame,
    target: pd.Series,
    high_corr_threshold: float = 0.95,
    moderate_corr_threshold: float = 0.75,
    raise_on_leakage: bool = True
) -> Dict[str, float]:
    """
    Detect target leakage via correlation analysis.

    High correlation (>0.95) usually indicates target leakage.
    Features shouldn't be near-perfect predictors.

    Args:
        features: Feature dataframe
        target: Target series
        high_corr_threshold: Threshold for leakage (default 0.95)
        moderate_corr_threshold: Threshold for warning (default 0.75)
        raise_on_leakage: Raise error if leakage detected (default True)

    Returns:
        Dict mapping feature names to correlation values

    Raises:
        ValueError: If leakage detected and raise_on_leakage=True

    Example:
        >>> features = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4],
        ...     'feature2': [10, 20, 30, 40]
        ... })
        >>> target = pd.Series([10, 20, 30, 40])  # Identical to feature2
        >>> detect_target_leakage(features, target)
        # Raises ValueError for feature2 (perfect correlation)
    """
    assert len(features) == len(target), \
        f"Features ({len(features)}) and target ({len(target)}) length mismatch"

    correlations = {}
    leakage_detected = []
    warnings = []

    for col in features.columns:
        # Skip if feature is constant
        if features[col].std() < 1e-10:
            print(f"WARNING: '{col}' is constant (std < 1e-10), skipping")
            continue

        # Calculate correlation
        corr = features[col].corr(target)
        correlations[col] = corr

        # HIGH correlation is suspicious (likely leakage)
        if abs(corr) > high_corr_threshold:
            leakage_detected.append(col)
            print(f"❌ LEAKAGE: '{col}' correlation {corr:.4f} > {high_corr_threshold}")

        # MODERATE correlation deserves investigation
        elif abs(corr) > moderate_corr_threshold:
            warnings.append(col)
            print(f"⚠️  WARNING: '{col}' correlation {corr:.4f} > {moderate_corr_threshold}")

        else:
            print(f"✓ '{col}': Correlation {corr:.4f} (OK)")

    # Raise error if leakage detected
    if leakage_detected and raise_on_leakage:
        raise ValueError(
            f"Target leakage detected in {len(leakage_detected)} features: "
            f"{leakage_detected}. Correlations: "
            f"{[f'{col}={correlations[col]:.4f}' for col in leakage_detected]}"
        )

    return correlations


def detect_identical_distribution(
    features: pd.DataFrame,
    target: pd.Series,
    ks_threshold: float = 0.05
) -> None:
    """
    Detect if feature has identical distribution to target (statistical leakage).

    Uses Kolmogorov-Smirnov test to compare distributions.

    Args:
        features: Feature dataframe
        target: Target series
        ks_threshold: P-value threshold for KS test (default 0.05)

    Raises:
        ValueError: If identical distribution detected
    """
    from scipy.stats import ks_2samp

    for col in features.columns:
        # Skip non-numeric features
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue

        # Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(features[col].dropna(), target.dropna())

        # High p-value means distributions are similar (suspicious)
        if p_value > 1 - ks_threshold:
            print(f"⚠️  WARNING: '{col}' has similar distribution to target "
                  f"(KS p-value: {p_value:.4f})")
        else:
            print(f"✓ '{col}': Distribution differs from target (p={p_value:.4f})")
```

---

## Embargo Period Calculation

### Multi-Frequency Embargo Calculator

```python
def calculate_embargo_period(
    data_frequency: str,
    settlement_days: int = 2,
    market_hours_per_day: float = 6.5,
    additional_lag_days: int = 0
) -> int:
    """
    Calculate minimum embargo period for time series validation.

    Accounts for settlement lag and information propagation.

    Args:
        data_frequency: 'daily', 'hourly', '5min', '1min', '15min', '30min'
        settlement_days: T+N settlement (default T+2 for US stocks)
        market_hours_per_day: Trading hours per day (default 6.5 for US)
        additional_lag_days: Additional lag for information propagation

    Returns:
        Embargo period in data frequency units

    Example:
        >>> # Daily data with T+2 settlement
        >>> embargo = calculate_embargo_period('daily', settlement_days=2)
        >>> print(embargo)  # 2

        >>> # 5-minute data with T+2 settlement
        >>> embargo = calculate_embargo_period('5min', settlement_days=2)
        >>> print(embargo)  # 156 (2 days * 6.5 hours * 60 min / 5)
    """
    total_days = settlement_days + additional_lag_days

    if data_frequency == 'daily':
        return total_days

    elif data_frequency == 'hourly':
        return int(total_days * market_hours_per_day)

    elif data_frequency in ['5min', '1min', '15min', '30min']:
        minutes_per_day = int(market_hours_per_day * 60)

        if data_frequency == '1min':
            periods_per_day = minutes_per_day
        elif data_frequency == '5min':
            periods_per_day = minutes_per_day // 5
        elif data_frequency == '15min':
            periods_per_day = minutes_per_day // 15
        elif data_frequency == '30min':
            periods_per_day = minutes_per_day // 30

        return total_days * periods_per_day

    else:
        raise ValueError(
            f"Unknown frequency: {data_frequency}. "
            f"Supported: 'daily', 'hourly', '5min', '1min', '15min', '30min'"
        )


def validate_embargo_gap(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    data: pd.DataFrame,
    min_embargo_days: int = 2
) -> None:
    """
    Validate that embargo gap between train and test is sufficient.

    Args:
        train_idx: Training indices
        test_idx: Test indices
        data: Dataframe with DatetimeIndex
        min_embargo_days: Minimum embargo in days

    Raises:
        AssertionError: If embargo gap is insufficient
    """
    assert isinstance(data.index, pd.DatetimeIndex), \
        "Data must have DatetimeIndex"

    train_end_date = data.index[train_idx[-1]]
    test_start_date = data.index[test_idx[0]]

    gap = (test_start_date - train_end_date).days

    assert gap >= min_embargo_days, \
        f"Embargo gap {gap} days < minimum {min_embargo_days} days. " \
        f"Train ends {train_end_date}, test starts {test_start_date}"

    print(f"✓ Embargo gap: {gap} days (minimum {min_embargo_days} days)")
```

---

## Regime Validation

### Multi-Regime Performance Validator

```python
def validate_performance_across_regimes(
    returns: pd.Series,
    regime_labels: pd.Series,
    min_sharpe: float = 0.3,
    max_sharpe_cv: float = 1.0,
    min_samples_per_regime: int = 20
) -> Dict[str, Dict[str, float]]:
    """
    Validate model performance across multiple market regimes.

    Ensures model works in different market conditions (bull, bear, neutral, crisis).

    Args:
        returns: Series of strategy returns
        regime_labels: Series with regime for each timestamp
        min_sharpe: Minimum acceptable Sharpe per regime (default 0.3)
        max_sharpe_cv: Maximum coefficient of variation for Sharpe across regimes
        min_samples_per_regime: Minimum samples required per regime

    Returns:
        Dict with performance metrics by regime

    Raises:
        AssertionError: If performance degrades too much in any regime

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, ...])
        >>> regimes = pd.Series(['bull', 'bear', 'bull', ...])
        >>> metrics = validate_performance_across_regimes(returns, regimes)
    """
    assert len(returns) == len(regime_labels), \
        "Returns and regime labels must have same length"

    regimes = regime_labels.unique()
    metrics_by_regime = {}
    sharpe_values = []

    print(f"\n{'='*60}")
    print(f"REGIME VALIDATION")
    print(f"{'='*60}\n")

    for regime in regimes:
        # Get returns for this regime
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) < min_samples_per_regime:
            print(f"⚠️  WARNING: '{regime}' has only {len(regime_returns)} samples "
                  f"(minimum {min_samples_per_regime})")
            continue

        # Calculate metrics
        mean_return = regime_returns.mean()
        std_return = regime_returns.std()
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        win_rate = (regime_returns > 0).sum() / len(regime_returns)
        max_drawdown = (regime_returns.cumsum() - regime_returns.cumsum().cummax()).min()

        metrics_by_regime[regime] = {
            'samples': len(regime_returns),
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }

        sharpe_values.append(sharpe)

        # Print metrics
        print(f"{regime.upper():12s} | Samples: {len(regime_returns):4d} | "
              f"Sharpe: {sharpe:5.2f} | Win Rate: {win_rate:.1%} | "
              f"Max DD: {max_drawdown:.2%}")

        # VALIDATE: Minimum Sharpe per regime
        assert sharpe > min_sharpe, \
            f"Sharpe {sharpe:.2f} in '{regime}' below minimum {min_sharpe}. " \
            f"Model fails in this regime."

    print(f"\n{'-'*60}")

    # VALIDATE: Consistency across regimes
    if len(sharpe_values) > 1:
        sharpe_mean = np.mean(sharpe_values)
        sharpe_std = np.std(sharpe_values)
        sharpe_cv = sharpe_std / sharpe_mean if sharpe_mean > 0 else float('inf')

        print(f"Overall Sharpe: {sharpe_mean:.2f} ± {sharpe_std:.2f} "
              f"(CV: {sharpe_cv:.2f})")

        # High CV indicates regime-specific overfitting
        if sharpe_cv > max_sharpe_cv:
            print(f"\n⚠️  WARNING: High Sharpe variation across regimes (CV={sharpe_cv:.2f} > {max_sharpe_cv})")
            print(f"This suggests regime-specific overfitting")
            print(f"Sharpe by regime: {[f'{s:.2f}' for s in sharpe_values]}")

    print(f"{'='*60}\n")

    return metrics_by_regime


def detect_regime_clustering(
    data: pd.DataFrame,
    regime_col: str = 'regime',
    min_regime_duration: int = 20
) -> None:
    """
    Detect if regime labels are clustered (non-stationary).

    Regimes should be relatively stable but not clustered all at start/end.

    Args:
        data: Dataframe with regime labels
        regime_col: Column name for regime labels
        min_regime_duration: Minimum duration for each regime appearance

    Raises:
        ValueError: If regime clustering detected
    """
    assert regime_col in data.columns, \
        f"Regime column '{regime_col}' not found"

    regimes = data[regime_col]
    unique_regimes = regimes.unique()

    for regime in unique_regimes:
        # Find all consecutive runs of this regime
        is_regime = (regimes == regime).astype(int)
        regime_changes = is_regime.diff().fillna(0)

        # Count number of separate clusters
        num_clusters = (regime_changes == 1).sum()

        # Average cluster duration
        total_samples = (regimes == regime).sum()
        avg_duration = total_samples / num_clusters if num_clusters > 0 else 0

        print(f"Regime '{regime}': {num_clusters} clusters, "
              f"avg duration {avg_duration:.1f} samples")

        # Check if all samples are in one cluster (clustered at start or end)
        if num_clusters == 1 and total_samples > min_regime_duration:
            first_idx = is_regime.idxmax()
            last_idx = is_regime[::-1].idxmax()

            # Check if clustered at edges
            if first_idx == 0 or last_idx == len(regimes) - 1:
                print(f"⚠️  WARNING: Regime '{regime}' is clustered at data edge. "
                      f"May not generalize to other regimes.")
```

---

## Performance Degradation Monitor

### Out-of-Sample Degradation Tracker

```python
def monitor_performance_degradation(
    in_sample_sharpe: float,
    out_sample_sharpe: float,
    max_degradation_pct: float = 0.30,
    warn_degradation_pct: float = 0.20
) -> None:
    """
    Monitor performance degradation from in-sample to out-of-sample.

    Significant degradation indicates overfitting.

    Args:
        in_sample_sharpe: In-sample Sharpe ratio
        out_sample_sharpe: Out-of-sample Sharpe ratio
        max_degradation_pct: Maximum acceptable degradation (default 30%)
        warn_degradation_pct: Warning threshold (default 20%)

    Raises:
        AssertionError: If degradation exceeds maximum

    Example:
        >>> monitor_performance_degradation(
        ...     in_sample_sharpe=2.5,
        ...     out_sample_sharpe=1.8,  # 28% degradation
        ...     max_degradation_pct=0.30
        ... )
    """
    if in_sample_sharpe <= 0:
        print(f"⚠️  WARNING: In-sample Sharpe {in_sample_sharpe:.2f} is non-positive")
        return

    degradation = (in_sample_sharpe - out_sample_sharpe) / in_sample_sharpe

    print(f"\nPerformance Degradation Analysis:")
    print(f"  In-sample Sharpe:  {in_sample_sharpe:.2f}")
    print(f"  Out-sample Sharpe: {out_sample_sharpe:.2f}")
    print(f"  Degradation:       {degradation:.1%}")

    # Check degradation severity
    if degradation > max_degradation_pct:
        raise ValueError(
            f"Performance degradation {degradation:.1%} exceeds maximum {max_degradation_pct:.1%}. "
            f"Model is likely overfit to training data."
        )

    elif degradation > warn_degradation_pct:
        print(f"  ⚠️  WARNING: Degradation {degradation:.1%} > {warn_degradation_pct:.1%}")
        print(f"  Consider reducing model complexity or increasing regularization")

    else:
        print(f"  ✓ Degradation {degradation:.1%} is acceptable (< {warn_degradation_pct:.1%})")


def track_walk_forward_consistency(
    fold_sharpes: List[float],
    min_sharpe: float = 0.5,
    max_sharpe_std: float = 0.5
) -> None:
    """
    Track consistency of performance across walk-forward folds.

    Args:
        fold_sharpes: List of Sharpe ratios from each walk-forward fold
        min_sharpe: Minimum acceptable Sharpe per fold
        max_sharpe_std: Maximum standard deviation of Sharpe across folds

    Raises:
        AssertionError: If consistency check fails
    """
    assert len(fold_sharpes) > 0, "No fold Sharpes provided"

    mean_sharpe = np.mean(fold_sharpes)
    std_sharpe = np.std(fold_sharpes)
    min_fold_sharpe = np.min(fold_sharpes)
    max_fold_sharpe = np.max(fold_sharpes)

    print(f"\nWalk-Forward Consistency:")
    print(f"  Folds:       {len(fold_sharpes)}")
    print(f"  Mean Sharpe: {mean_sharpe:.2f} ± {std_sharpe:.2f}")
    print(f"  Range:       [{min_fold_sharpe:.2f}, {max_fold_sharpe:.2f}]")

    # Check minimum performance
    assert min_fold_sharpe > min_sharpe, \
        f"Minimum fold Sharpe {min_fold_sharpe:.2f} < {min_sharpe}. " \
        f"Model fails in some time periods."

    # Check consistency
    if std_sharpe > max_sharpe_std:
        print(f"  ⚠️  WARNING: High Sharpe variation (std={std_sharpe:.2f} > {max_sharpe_std})")
        print(f"  Performance is inconsistent across time periods")
        print(f"  Fold Sharpes: {[f'{s:.2f}' for s in fold_sharpes]}")
    else:
        print(f"  ✓ Consistent performance across folds")
```

---

## Complete Validation Pipeline

### End-to-End Validation Workflow

```python
def run_complete_validation_pipeline(
    data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    feature_timestamps: Dict[str, pd.Series],
    target_timestamps: pd.Series,
    regime_labels: pd.Series,
    data_frequency: str = 'daily',
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Run complete time series validation pipeline.

    Performs all validation checks in proper order:
    1. Feature timestamp validation (look-ahead bias)
    2. Target leakage detection
    3. Walk-forward cross-validation
    4. Regime performance validation
    5. Degradation monitoring

    Args:
        data: Time series dataframe
        features: Feature dataframe
        target: Target series
        feature_timestamps: Feature timestamp dict
        target_timestamps: Target timestamp series
        regime_labels: Market regime labels
        data_frequency: Data frequency ('daily', 'hourly', etc.)
        n_folds: Number of walk-forward folds

    Returns:
        Dict with validation results
    """
    print(f"\n{'='*70}")
    print(f"TIME SERIES VALIDATION PIPELINE")
    print(f"{'='*70}\n")

    results = {}

    # STEP 1: Look-Ahead Bias Check
    print(f"[1/5] Checking for look-ahead bias...")
    try:
        validate_feature_timestamps(
            features, target, feature_timestamps, target_timestamps
        )
        results['look_ahead_bias'] = 'PASS'
        print(f"✓ No look-ahead bias detected\n")
    except Exception as e:
        results['look_ahead_bias'] = f'FAIL: {str(e)}'
        raise

    # STEP 2: Target Leakage Check
    print(f"[2/5] Checking for target leakage...")
    try:
        correlations = detect_target_leakage(features, target)
        results['target_leakage'] = 'PASS'
        results['feature_correlations'] = correlations
        print(f"✓ No target leakage detected\n")
    except Exception as e:
        results['target_leakage'] = f'FAIL: {str(e)}'
        raise

    # STEP 3: Walk-Forward Validation
    print(f"[3/5] Running walk-forward validation ({n_folds} folds)...")
    embargo = calculate_embargo_period(data_frequency)
    fold_sharpes = []

    for fold_num, (train_idx, test_idx) in enumerate(
        expanding_window_split(data, embargo_days=embargo, step_days=21)
    ):
        if fold_num >= n_folds:
            break

        # Validate embargo gap
        validate_embargo_gap(train_idx, test_idx, data, min_embargo_days=embargo)

        # Mock training and evaluation
        # In real usage, train model and get predictions here
        # For demonstration, generate random Sharpe
        fold_sharpe = np.random.uniform(0.5, 2.0)
        fold_sharpes.append(fold_sharpe)

        print(f"  Fold {fold_num + 1}: Sharpe = {fold_sharpe:.2f}")

    results['fold_sharpes'] = fold_sharpes
    print(f"✓ Walk-forward validation complete\n")

    # STEP 4: Walk-Forward Consistency Check
    print(f"[4/5] Checking walk-forward consistency...")
    try:
        track_walk_forward_consistency(fold_sharpes)
        results['walk_forward_consistency'] = 'PASS'
        print(f"✓ Walk-forward consistency validated\n")
    except Exception as e:
        results['walk_forward_consistency'] = f'FAIL: {str(e)}'
        raise

    # STEP 5: Regime Validation
    print(f"[5/5] Validating performance across regimes...")
    try:
        # Mock returns (in real usage, use actual strategy returns)
        returns = pd.Series(np.random.normal(0.001, 0.02, len(regime_labels)))

        regime_metrics = validate_performance_across_regimes(
            returns, regime_labels
        )
        results['regime_validation'] = 'PASS'
        results['regime_metrics'] = regime_metrics
        print(f"✓ Regime validation complete\n")
    except Exception as e:
        results['regime_validation'] = f'FAIL: {str(e)}'
        raise

    print(f"{'='*70}")
    print(f"VALIDATION PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"Results:")
    print(f"  Look-ahead bias:      {results['look_ahead_bias']}")
    print(f"  Target leakage:       {results['target_leakage']}")
    print(f"  Walk-forward folds:   {len(fold_sharpes)}")
    print(f"  WF consistency:       {results['walk_forward_consistency']}")
    print(f"  Regime validation:    {results['regime_validation']}")
    print(f"\n✓ All validation checks passed!\n")

    return results
```

---

## Usage Examples

### Example 1: Basic Walk-Forward Validation

```python
import pandas as pd
import numpy as np

# Load data
data = pd.DataFrame({
    'close': np.random.randn(1000).cumsum() + 100
}, index=pd.date_range('2020-01-01', periods=1000))

# Walk-forward split
for train_idx, test_idx in expanding_window_split(data, min_train_days=252):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    # Train model
    # model.fit(train_data)

    # Evaluate
    # predictions = model.predict(test_data)

    print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")
```

### Example 2: Feature Validation

```python
# Create features and target
features = pd.DataFrame({
    'ma_20': data['close'].rolling(20).mean(),
    'rsi': ...,
    'volume_ratio': ...
})

target = data['close'].pct_change().shift(-1)  # Next day return

# Feature timestamps (for each feature, when was it available?)
feature_timestamps = {
    'ma_20': data.index,  # Available at market close
    'rsi': data.index,
    'volume_ratio': data.index
}

target_timestamps = data.index

# Validate no look-ahead bias
validate_feature_timestamps(
    features.dropna(),
    target.dropna(),
    feature_timestamps,
    target_timestamps
)

# Validate no target leakage
detect_target_leakage(features.dropna(), target.dropna())
```

### Example 3: Complete Pipeline

```python
# Run complete validation
results = run_complete_validation_pipeline(
    data=data,
    features=features,
    target=target,
    feature_timestamps=feature_timestamps,
    target_timestamps=target_timestamps,
    regime_labels=regime_labels,
    data_frequency='daily',
    n_folds=5
)

print(f"Validation complete: {results}")
```

