"""
Unit tests for critical data processing bug fixes.

Tests:
1. Log-return index preservation (data_fetching.py)
2. CVaR graceful fallback for underdetermined cases (cvar_allocator.py)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching import compute_log_returns
from portfolio.cvar_allocator import CVaRAllocator


class TestLogReturnIndexPreservation:
    """Test that log returns preserve datetime index."""

    def test_single_ticker_preserves_datetime_index(self):
        """Log returns should preserve DatetimeIndex for single ticker."""
        # Create price DataFrame with datetime index
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'Adj Close': np.random.uniform(100, 200, 100)
        }, index=dates)

        returns = compute_log_returns(prices, field='Adj Close')

        # CRITICAL: Index must be DatetimeIndex, not RangeIndex
        assert isinstance(returns.index, pd.DatetimeIndex), \
            f"Expected DatetimeIndex, got {type(returns.index)}"

        # Verify dates are preserved (minus first row dropped by pct_change)
        expected_dates = dates[1:]  # First date dropped due to NaN
        assert len(returns) == len(expected_dates), \
            f"Expected {len(expected_dates)} rows, got {len(returns)}"

    def test_multi_ticker_preserves_datetime_index(self):
        """Log returns should preserve DatetimeIndex for multi-ticker."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        # Create multi-level column DataFrame
        arrays = [
            ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            ['Adj Close', 'Volume', 'Adj Close', 'Volume']
        ]
        columns = pd.MultiIndex.from_arrays(arrays)

        data = np.random.uniform(100, 200, (100, 4))
        prices = pd.DataFrame(data, index=dates, columns=columns)

        returns = compute_log_returns(prices, field='Adj Close')

        # CRITICAL: Index must be DatetimeIndex
        assert isinstance(returns.index, pd.DatetimeIndex), \
            f"Expected DatetimeIndex, got {type(returns.index)}"

        # Verify both tickers present
        assert 'AAPL' in returns.columns
        assert 'MSFT' in returns.columns

    def test_returns_are_log_returns(self):
        """Verify returns are actually log returns, not simple returns."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'Adj Close': [100, 110, 105, 115, 120, 118, 125, 130, 128, 135]
        }, index=dates)

        returns = compute_log_returns(prices, field='Adj Close')

        # First return should be log(110/100) = log(1.1) ≈ 0.0953
        expected_first = np.log1p((110 - 100) / 100)
        actual_first = returns.iloc[0, 0]

        assert abs(actual_first - expected_first) < 1e-10, \
            f"Expected log return {expected_first:.6f}, got {actual_first:.6f}"

    def test_index_alignment_for_backtesting(self):
        """Verify returns can be aligned with original prices for backtesting."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = pd.DataFrame({
            'Adj Close': np.random.uniform(100, 200, 50)
        }, index=dates)

        returns = compute_log_returns(prices, field='Adj Close')

        # Returns index should be a subset of prices index
        assert returns.index.isin(prices.index).all(), \
            "Returns index not a subset of prices index - alignment will fail"

        # Joining should work without errors
        combined = prices.join(returns, rsuffix='_return')
        assert len(combined) == len(prices)


class TestCVaRGracefulFallback:
    """Test CVaR allocator gracefully handles underdetermined cases."""

    def test_more_assets_than_observations_warns(self):
        """When n_assets > n_obs, should warn and reduce universe."""
        # Create underdetermined case: 5 observations, 10 assets
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        tickers = [f'STOCK{i}' for i in range(10)]

        returns = pd.DataFrame(
            np.random.randn(5, 10) * 0.02,
            index=dates,
            columns=tickers
        )

        allocator = CVaRAllocator(risk_aversion=5.0, max_weight=0.25)

        # Should warn but not raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = allocator.optimize(returns)

            # Check that warning was issued
            assert len(w) >= 1, "Expected warning for underdetermined case"
            warning_messages = [str(warning.message) for warning in w]
            assert any('Insufficient data' in msg or 'auto-reducing' in msg.lower()
                      for msg in warning_messages), \
                f"Expected 'Insufficient data' warning, got: {warning_messages}"

        # Should still return valid weights
        assert 'weights' in result
        assert abs(result['weights'].sum() - 1.0) < 1e-6, \
            f"Weights should sum to 1.0, got {result['weights'].sum()}"

    def test_equal_assets_and_observations_warns(self):
        """When n_assets == n_obs, should warn and reduce universe."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        tickers = [f'STOCK{i}' for i in range(10)]

        returns = pd.DataFrame(
            np.random.randn(10, 10) * 0.02,
            index=dates,
            columns=tickers
        )

        allocator = CVaRAllocator(risk_aversion=5.0, max_weight=0.25)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = allocator.optimize(returns)

        # Should return valid result
        assert 'weights' in result
        assert result['weights'].sum() > 0

    def test_sufficient_observations_no_warning(self):
        """When n_obs > n_assets, should not warn."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

        returns = pd.DataFrame(
            np.random.randn(100, 4) * 0.02,
            index=dates,
            columns=tickers
        )

        allocator = CVaRAllocator(risk_aversion=5.0, max_weight=0.25)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = allocator.optimize(returns)

            # Filter only our warnings
            our_warnings = [warning for warning in w
                           if 'Insufficient' in str(warning.message)]
            assert len(our_warnings) == 0, \
                f"Unexpected warning for well-determined case: {our_warnings}"

        # Should have all 4 tickers in weights
        assert len(result['weights']) == 4

    def test_fallback_weights_are_valid(self):
        """Fallback weights should satisfy basic portfolio constraints."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        tickers = [f'STOCK{i}' for i in range(15)]

        returns = pd.DataFrame(
            np.random.randn(10, 15) * 0.02,
            index=dates,
            columns=tickers
        )

        allocator = CVaRAllocator(
            risk_aversion=5.0,
            max_weight=0.50,  # Higher max to allow valid solution with reduced assets
            min_weight=0.0
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = allocator.optimize(returns)

        weights = result['weights']

        # Weights sum to 1
        assert abs(weights.sum() - 1.0) < 1e-6

        # No negative weights (long-only)
        assert (weights >= -1e-6).all(), f"Negative weights: {weights[weights < 0]}"

        # Reduced universe should be smaller than original
        assert len(weights) < 15, \
            f"Expected reduced universe, got {len(weights)} assets"

    def test_reduced_universe_has_highest_variance_assets(self):
        """When reducing universe, should keep highest variance assets."""
        # 5 observations, 8 assets → must reduce to 4 assets
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')

        # Create returns with clearly different variances
        # Set random seed for reproducibility
        np.random.seed(42)
        returns = pd.DataFrame({
            'LOW_VAR1': np.random.randn(5) * 0.001,   # Very low variance
            'LOW_VAR2': np.random.randn(5) * 0.002,   # Very low variance
            'MED_VAR1': np.random.randn(5) * 0.01,    # Medium variance
            'MED_VAR2': np.random.randn(5) * 0.015,   # Medium variance
            'HIGH_VAR1': np.random.randn(5) * 0.10,   # High variance
            'HIGH_VAR2': np.random.randn(5) * 0.08,   # High variance
            'HIGH_VAR3': np.random.randn(5) * 0.09,   # High variance
            'HIGH_VAR4': np.random.randn(5) * 0.07,   # High variance
        }, index=dates)

        allocator = CVaRAllocator(risk_aversion=5.0, max_weight=0.50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = allocator.optimize(returns)

        # Should have reduced to fewer assets (5 obs → max 4 assets)
        assert len(result['weights']) <= 4, \
            f"Expected reduced universe to ≤4 assets, got {len(result['weights'])}"

        # High variance assets should be in the kept set
        kept_assets = set(result['weights'].index)
        high_var_assets = {'HIGH_VAR1', 'HIGH_VAR2', 'HIGH_VAR3', 'HIGH_VAR4'}

        # At least some high-variance assets should be kept
        high_var_kept = kept_assets & high_var_assets
        assert len(high_var_kept) >= 2, \
            f"Expected at least 2 high-variance assets kept, got {high_var_kept}"


class TestCVaRConstraintValidation:
    """Test that CVaR optimizer validates constraints properly."""

    def test_weights_sum_to_one(self):
        """Portfolio weights must sum to 1.0."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns = pd.DataFrame(
            np.random.randn(100, 5) * 0.02,
            index=dates,
            columns=['A', 'B', 'C', 'D', 'E']
        )

        allocator = CVaRAllocator(risk_aversion=5.0)
        result = allocator.optimize(returns)

        assert abs(result['weights'].sum() - 1.0) < 1e-6, \
            f"Weights sum to {result['weights'].sum()}, not 1.0"

    def test_sharpe_ratio_is_reasonable(self):
        """Sharpe ratio should be within realistic bounds."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        returns = pd.DataFrame(
            np.random.randn(252, 4) * 0.02,
            index=dates,
            columns=['A', 'B', 'C', 'D']
        )

        allocator = CVaRAllocator(risk_aversion=5.0)
        result = allocator.optimize(returns)

        sharpe = result['sharpe']
        assert -3 < sharpe < 5, \
            f"Sharpe {sharpe:.2f} outside realistic bounds [-3, 5]"

    def test_cvar_is_non_negative(self):
        """CVaR (annualized) should be non-negative."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns = pd.DataFrame(
            np.random.randn(100, 4) * 0.02,
            index=dates,
            columns=['A', 'B', 'C', 'D']
        )

        allocator = CVaRAllocator(risk_aversion=5.0, cvar_alpha=0.95)
        result = allocator.optimize(returns)

        assert result['cvar'] >= 0, f"CVaR should be non-negative, got {result['cvar']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
