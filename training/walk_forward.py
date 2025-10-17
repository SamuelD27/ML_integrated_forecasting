"""
Walk-Forward Validation with Purging and Embargo
=================================================
Proper time series cross-validation to prevent look-ahead bias.

Key Concepts:
1. Walk-Forward: Rolling window (train → test → retrain → test → ...)
2. Purging: Remove overlapping samples between train/test
3. Embargo: Add gap after test period before next train
4. Expanding Window: Use all historical data (vs fixed window)

Reference:
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
- Chapter 7: Cross-Validation in Finance
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    # Window sizes (in days or samples)
    train_window: int = 252 * 2  # 2 years
    test_window: int = 63  # 3 months
    step_size: int = 21  # Move forward by 1 month

    # Purging and embargo
    purge_pct: float = 0.02  # Purge 2% of train samples near test
    embargo_pct: float = 0.01  # Embargo 1% after test

    # Window type
    expanding_window: bool = True  # Use all history (vs fixed window)
    min_train_size: int = 252  # Minimum training samples

    # Validation
    min_test_samples: int = 20  # Minimum test samples per fold


class WalkForwardSplitter:
    """
    Walk-forward cross-validation splitter with purging and embargo.

    Example:
        >>> splitter = WalkForwardSplitter(config)
        >>> for train_idx, test_idx in splitter.split(data):
        ...     train_data = data.iloc[train_idx]
        ...     test_data = data.iloc[test_idx]
        ...     # Train and evaluate
    """

    def __init__(self, config: WalkForwardConfig):
        """
        Initialize walk-forward splitter.

        Args:
            config: Walk-forward configuration
        """
        self.config = config

    def split(
        self,
        data: pd.DataFrame,
        time_col: Optional[str] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Args:
            data: Data with time index or time column
            time_col: Name of time column (if not in index)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        # Get time index
        if time_col is not None:
            times = data[time_col].values
        elif isinstance(data.index, pd.DatetimeIndex):
            times = data.index.values
        else:
            times = np.arange(len(data))

        n_samples = len(data)

        # Starting position
        if self.config.expanding_window:
            train_start = 0
        else:
            train_start = 0

        test_start = self.config.train_window

        fold = 0

        while test_start + self.config.test_window <= n_samples:
            # Train indices
            if self.config.expanding_window:
                train_end = test_start
                train_idx = np.arange(train_start, train_end)
            else:
                train_end = test_start
                train_start_current = max(0, train_end - self.config.train_window)
                train_idx = np.arange(train_start_current, train_end)

            # Test indices
            test_end = test_start + self.config.test_window
            test_idx = np.arange(test_start, test_end)

            # Apply purging
            train_idx = self._apply_purging(train_idx, test_idx)

            # Apply embargo
            test_idx, embargo_idx = self._apply_embargo(test_idx, n_samples)

            # Validate
            if len(train_idx) < self.config.min_train_size:
                logger.warning(f"Fold {fold}: Insufficient training samples "
                             f"({len(train_idx)} < {self.config.min_train_size})")
                break

            if len(test_idx) < self.config.min_test_samples:
                logger.warning(f"Fold {fold}: Insufficient test samples "
                             f"({len(test_idx)} < {self.config.min_test_samples})")
                break

            logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}, "
                       f"purged={len(test_idx) + len(embargo_idx) - (test_end - test_start)}")

            yield train_idx, test_idx

            # Move to next fold
            test_start += self.config.step_size
            fold += 1

    def _apply_purging(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray
    ) -> np.ndarray:
        """
        Purge training samples that overlap with test period.

        Removes samples from end of training set that might leak information
        into test set (e.g., due to label calculation window).

        Args:
            train_idx: Training indices
            test_idx: Test indices

        Returns:
            Purged training indices
        """
        if self.config.purge_pct <= 0:
            return train_idx

        # Calculate purge size
        n_purge = int(len(train_idx) * self.config.purge_pct)

        if n_purge > 0:
            # Remove last n_purge samples from training
            train_idx = train_idx[:-n_purge]

        return train_idx

    def _apply_embargo(
        self,
        test_idx: np.ndarray,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply embargo period after test set.

        Adds gap between test and next train to prevent leakage.

        Args:
            test_idx: Test indices
            n_samples: Total number of samples

        Returns:
            Tuple of (test_indices, embargo_indices)
        """
        if self.config.embargo_pct <= 0:
            return test_idx, np.array([])

        # Calculate embargo size
        n_embargo = int(len(test_idx) * self.config.embargo_pct)

        if n_embargo > 0:
            embargo_start = test_idx[-1] + 1
            embargo_end = min(embargo_start + n_embargo, n_samples)
            embargo_idx = np.arange(embargo_start, embargo_end)
        else:
            embargo_idx = np.array([])

        return test_idx, embargo_idx

    def get_n_splits(self, data: pd.DataFrame) -> int:
        """
        Calculate number of splits.

        Args:
            data: Data

        Returns:
            Number of folds
        """
        n_samples = len(data)
        test_start = self.config.train_window

        n_splits = 0
        while test_start + self.config.test_window <= n_samples:
            n_splits += 1
            test_start += self.config.step_size

        return n_splits


class WalkForwardValidator:
    """
    Complete walk-forward validation with model training.

    Example:
        >>> validator = WalkForwardValidator(model_class, config)
        >>> results = validator.validate(data, features, target)
        >>> validator.plot_results()
    """

    def __init__(
        self,
        model_factory: callable,
        config: WalkForwardConfig
    ):
        """
        Initialize walk-forward validator.

        Args:
            model_factory: Function that creates model instance
            config: Walk-forward configuration
        """
        self.model_factory = model_factory
        self.config = config
        self.splitter = WalkForwardSplitter(config)

        self.results = {
            'fold': [],
            'train_score': [],
            'test_score': [],
            'test_start_date': [],
            'test_end_date': [],
            'n_train': [],
            'n_test': []
        }

    def validate(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str,
        metric_fn: callable = None
    ) -> Dict:
        """
        Run walk-forward validation.

        Args:
            data: Complete dataset with features and target
            features: List of feature column names
            target: Target column name
            metric_fn: Metric function (higher = better)

        Returns:
            Dictionary with validation results
        """
        if metric_fn is None:
            # Default: negative MSE (higher = better)
            metric_fn = lambda y_true, y_pred: -np.mean((y_true - y_pred) ** 2)

        for fold, (train_idx, test_idx) in enumerate(self.splitter.split(data)):
            # Split data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            X_train = train_data[features].values
            y_train = train_data[target].values
            X_test = test_data[features].values
            y_test = test_data[target].values

            # Create and train model
            model = self.model_factory()

            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Fold {fold}: Training failed - {e}")
                continue

            # Evaluate
            try:
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_score = metric_fn(y_train, train_pred)
                test_score = metric_fn(y_test, test_pred)
            except Exception as e:
                logger.error(f"Fold {fold}: Prediction failed - {e}")
                continue

            # Store results
            self.results['fold'].append(fold)
            self.results['train_score'].append(train_score)
            self.results['test_score'].append(test_score)
            self.results['test_start_date'].append(data.index[test_idx[0]])
            self.results['test_end_date'].append(data.index[test_idx[-1]])
            self.results['n_train'].append(len(train_idx))
            self.results['n_test'].append(len(test_idx))

            logger.info(
                f"Fold {fold}: "
                f"train_score={train_score:.4f}, "
                f"test_score={test_score:.4f}, "
                f"train_samples={len(train_idx)}, "
                f"test_samples={len(test_idx)}"
            )

        # Convert to DataFrame
        self.results_df = pd.DataFrame(self.results)

        # Summary statistics
        summary = {
            'mean_train_score': np.mean(self.results['train_score']),
            'mean_test_score': np.mean(self.results['test_score']),
            'std_test_score': np.std(self.results['test_score']),
            'n_folds': len(self.results['fold'])
        }

        logger.info(f"\nValidation Summary:")
        logger.info(f"  Mean Train Score: {summary['mean_train_score']:.4f}")
        logger.info(f"  Mean Test Score: {summary['mean_test_score']:.4f}")
        logger.info(f"  Std Test Score: {summary['std_test_score']:.4f}")
        logger.info(f"  Number of Folds: {summary['n_folds']}")

        return summary

    def plot_results(self, save_path: str = 'reports/walk_forward_results.png'):
        """Plot walk-forward validation results."""
        if not hasattr(self, 'results_df'):
            logger.error("No results to plot. Run validate() first.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot scores over time
        ax1 = axes[0]
        ax1.plot(
            self.results_df['test_start_date'],
            self.results_df['train_score'],
            'o-',
            label='Train Score',
            alpha=0.7
        )
        ax1.plot(
            self.results_df['test_start_date'],
            self.results_df['test_score'],
            'o-',
            label='Test Score',
            alpha=0.7
        )
        ax1.axhline(
            self.results_df['test_score'].mean(),
            color='red',
            linestyle='--',
            label=f'Mean Test Score: {self.results_df["test_score"].mean():.4f}'
        )
        ax1.set_xlabel('Test Start Date')
        ax1.set_ylabel('Score')
        ax1.set_title('Walk-Forward Validation: Scores Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot sample sizes
        ax2 = axes[1]
        ax2.bar(
            self.results_df['fold'],
            self.results_df['n_train'],
            alpha=0.6,
            label='Training Samples'
        )
        ax2.bar(
            self.results_df['fold'],
            self.results_df['n_test'],
            alpha=0.6,
            label='Test Samples'
        )
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Sizes per Fold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Walk-forward plot saved to {save_path}")


def create_walk_forward_splitter(
    train_years: float = 2.0,
    test_months: float = 3.0,
    step_months: float = 1.0,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
    expanding_window: bool = True
) -> WalkForwardSplitter:
    """
    Convenience function to create walk-forward splitter.

    Args:
        train_years: Training window in years
        test_months: Test window in months
        step_months: Step size in months
        purge_pct: Purging percentage
        embargo_pct: Embargo percentage
        expanding_window: Use expanding window

    Returns:
        WalkForwardSplitter instance

    Example:
        >>> splitter = create_walk_forward_splitter(
        ...     train_years=2.0,
        ...     test_months=3.0,
        ...     step_months=1.0
        ... )
    """
    config = WalkForwardConfig(
        train_window=int(252 * train_years),
        test_window=int(21 * test_months),
        step_size=int(21 * step_months),
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        expanding_window=expanding_window
    )

    return WalkForwardSplitter(config)


if __name__ == '__main__':
    # Example usage
    print("Walk-Forward Validation Example")
    print("=" * 60)

    # Create synthetic data
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'feature1': np.random.randn(len(dates)),
        'feature2': np.random.randn(len(dates)),
        'feature3': np.random.randn(len(dates)),
        'target': np.random.randn(len(dates))
    }, index=dates)

    print(f"Data: {len(data)} days ({data.index[0]} to {data.index[-1]})")

    # Create splitter
    splitter = create_walk_forward_splitter(
        train_years=2.0,
        test_months=3.0,
        step_months=1.0,
        purge_pct=0.02,
        embargo_pct=0.01
    )

    # Count splits
    n_splits = splitter.get_n_splits(data)
    print(f"\nNumber of folds: {n_splits}")

    # Iterate through splits
    print("\nFirst 3 folds:")
    for i, (train_idx, test_idx) in enumerate(splitter.split(data)):
        if i >= 3:
            break

        train_dates = data.index[train_idx]
        test_dates = data.index[test_idx]

        print(f"\nFold {i}:")
        print(f"  Train: {len(train_idx)} samples "
              f"({train_dates[0]} to {train_dates[-1]})")
        print(f"  Test: {len(test_idx)} samples "
              f"({test_dates[0]} to {test_dates[-1]})")

    # Validate with dummy model
    print("\n" + "=" * 60)
    print("Running walk-forward validation...")

    from sklearn.linear_model import Ridge

    def model_factory():
        return Ridge(alpha=1.0)

    validator = WalkForwardValidator(model_factory, splitter.config)

    results = validator.validate(
        data,
        features=['feature1', 'feature2', 'feature3'],
        target='target'
    )

    print("\n✓ Walk-forward module ready")
