"""
Tests for risk metric calculations with proper validation.

These tests verify:
1. Sharpe ratio calculations with risk-free rate
2. VaR/CVaR calculations with horizon scaling
3. Input validation for all risk metrics
4. Edge case handling

Per financial-knowledge-validator skill guidelines.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the modules we're testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from single_stock.risk_metrics import (
    compute_var,
    compute_sharpe_ratio,
    compute_beta,
    compute_cvar,
    SHARPE_BOUNDS,
    BETA_BOUNDS,
    VAR_CONFIDENCE_BOUNDS,
)
from utils.data_processing import (
    compute_descriptive_stats,
    compute_risk_metrics,
)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        # Simulate ~10% annual return with 20% volatility
        daily_mean = 0.10 / 252
        daily_std = 0.20 / np.sqrt(252)
        returns = pd.Series(
            np.random.normal(daily_mean, daily_std, 252),
            index=pd.date_range('2024-01-01', periods=252, freq='B')
        )
        return returns

    def test_sharpe_with_risk_free_rate(self, sample_returns):
        """Test that Sharpe ratio correctly incorporates risk-free rate."""
        # With Rf=0
        sharpe_rf0 = compute_sharpe_ratio(sample_returns, risk_free_rate=0.0)

        # With Rf=5%
        sharpe_rf5 = compute_sharpe_ratio(sample_returns, risk_free_rate=0.05)

        # Sharpe with higher Rf should be lower
        assert sharpe_rf5 < sharpe_rf0, "Higher Rf should reduce Sharpe ratio"

        # Difference should be approximately Rf / volatility
        vol = sample_returns.std() * np.sqrt(252)
        expected_diff = 0.05 / vol
        actual_diff = sharpe_rf0 - sharpe_rf5
        assert abs(actual_diff - expected_diff) < 0.1, \
            f"Sharpe difference {actual_diff:.3f} should be ~{expected_diff:.3f}"

    def test_sharpe_input_validation(self, sample_returns):
        """Test input validation for Sharpe ratio."""
        # Invalid risk-free rate (too high)
        with pytest.raises(ValueError, match="risk_free_rate.*outside realistic bounds"):
            compute_sharpe_ratio(sample_returns, risk_free_rate=0.25)

        # Invalid risk-free rate (negative)
        with pytest.raises(ValueError, match="risk_free_rate.*outside realistic bounds"):
            compute_sharpe_ratio(sample_returns, risk_free_rate=-0.01)

        # Insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            compute_sharpe_ratio(sample_returns.head(10))

        # Wrong type
        with pytest.raises(TypeError, match="must be a pandas Series"):
            compute_sharpe_ratio(sample_returns.values)

    def test_sharpe_zero_volatility(self):
        """Test edge case: zero volatility returns."""
        constant_returns = pd.Series([0.001] * 100)
        result = compute_sharpe_ratio(constant_returns)
        assert np.isnan(result), "Zero volatility should return NaN"

    def test_sharpe_bounds_reasonable(self, sample_returns):
        """Test that Sharpe ratios fall within realistic bounds."""
        sharpe = compute_sharpe_ratio(sample_returns, risk_free_rate=0.05)
        assert SHARPE_BOUNDS[0] < sharpe < SHARPE_BOUNDS[1], \
            f"Sharpe {sharpe:.2f} outside bounds {SHARPE_BOUNDS}"


class TestVaRCalculation:
    """Tests for Value at Risk calculation."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        daily_std = 0.02  # 2% daily volatility
        returns = pd.Series(
            np.random.normal(0, daily_std, 252),
            index=pd.date_range('2024-01-01', periods=252, freq='B')
        )
        return returns

    def test_var_horizon_scaling(self, sample_returns):
        """Test that VaR scales correctly with horizon (sqrt rule)."""
        var_1d = compute_var(sample_returns, confidence_level=0.95, horizon_days=1)
        var_10d = compute_var(sample_returns, confidence_level=0.95, horizon_days=10)

        # VaR_10d should be approximately sqrt(10) * VaR_1d
        expected_ratio = np.sqrt(10)
        actual_ratio = var_10d / var_1d

        assert abs(actual_ratio - expected_ratio) < 0.5, \
            f"VaR ratio {actual_ratio:.2f} should be ~{expected_ratio:.2f}"

    def test_var_confidence_level(self, sample_returns):
        """Test that higher confidence = larger (more negative) VaR."""
        var_95 = compute_var(sample_returns, confidence_level=0.95)
        var_99 = compute_var(sample_returns, confidence_level=0.99)

        # 99% VaR should be more negative (larger loss) than 95% VaR
        assert var_99 < var_95, "99% VaR should be more negative than 95% VaR"

    def test_var_methods_consistency(self, sample_returns):
        """Test that different VaR methods give similar results."""
        np.random.seed(42)  # For Monte Carlo reproducibility

        var_hist = compute_var(sample_returns, method='historical')
        var_param = compute_var(sample_returns, method='parametric')
        var_mc = compute_var(sample_returns, method='monte_carlo', num_simulations=50000)

        # All methods should give similar VaR (within 50% of each other)
        assert abs(var_hist - var_param) / abs(var_hist) < 0.5, \
            f"Historical ({var_hist:.4f}) vs Parametric ({var_param:.4f}) differ too much"
        assert abs(var_hist - var_mc) / abs(var_hist) < 0.5, \
            f"Historical ({var_hist:.4f}) vs Monte Carlo ({var_mc:.4f}) differ too much"

    def test_var_input_validation(self, sample_returns):
        """Test input validation for VaR."""
        # Invalid confidence level
        with pytest.raises(ValueError, match="confidence_level"):
            compute_var(sample_returns, confidence_level=0.3)

        with pytest.raises(ValueError, match="confidence_level"):
            compute_var(sample_returns, confidence_level=1.0)

        # Invalid horizon
        with pytest.raises(ValueError, match="horizon_days"):
            compute_var(sample_returns, horizon_days=0)

        with pytest.raises(ValueError, match="horizon_days"):
            compute_var(sample_returns, horizon_days=500)

        # Insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            compute_var(sample_returns.head(10))

        # Invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            compute_var(sample_returns, method='invalid')


class TestCVaRCalculation:
    """Tests for Conditional Value at Risk (Expected Shortfall)."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns with some fat tails."""
        np.random.seed(42)
        # Use t-distribution for fat tails
        returns = pd.Series(
            np.random.standard_t(df=5, size=500) * 0.01,
            index=pd.date_range('2024-01-01', periods=500, freq='B')
        )
        return returns

    def test_cvar_worse_than_var(self, sample_returns):
        """CVaR should always be more negative than VaR."""
        var = compute_var(sample_returns, confidence_level=0.95)
        cvar = compute_cvar(sample_returns, confidence_level=0.95)

        assert cvar < var, f"CVaR ({cvar:.4f}) should be worse than VaR ({var:.4f})"

    def test_cvar_horizon_scaling(self, sample_returns):
        """Test that CVaR scales correctly with horizon."""
        cvar_1d = compute_cvar(sample_returns, confidence_level=0.95, horizon_days=1)
        cvar_10d = compute_cvar(sample_returns, confidence_level=0.95, horizon_days=10)

        expected_ratio = np.sqrt(10)
        actual_ratio = cvar_10d / cvar_1d

        assert abs(actual_ratio - expected_ratio) < 0.5, \
            f"CVaR ratio {actual_ratio:.2f} should be ~{expected_ratio:.2f}"


class TestBetaCalculation:
    """Tests for CAPM beta calculation."""

    @pytest.fixture
    def market_returns(self):
        """Generate market returns."""
        np.random.seed(42)
        return pd.Series(
            np.random.normal(0.0004, 0.01, 252),
            index=pd.date_range('2024-01-01', periods=252, freq='B')
        )

    def test_beta_of_market(self, market_returns):
        """Beta of market with itself should be 1.0."""
        beta = compute_beta(market_returns, market_returns)
        assert abs(beta - 1.0) < 0.001, f"Beta of market should be 1.0, got {beta:.4f}"

    def test_beta_high_correlation(self, market_returns):
        """High beta stock should have beta > 1."""
        np.random.seed(43)
        # High beta stock: 1.5x market moves + noise
        asset_returns = market_returns * 1.5 + pd.Series(
            np.random.normal(0, 0.002, 252),
            index=market_returns.index
        )
        beta = compute_beta(asset_returns, market_returns)
        assert 1.2 < beta < 1.8, f"High beta stock should have beta ~1.5, got {beta:.4f}"

    def test_beta_low_correlation(self, market_returns):
        """Low beta stock should have beta < 1."""
        np.random.seed(44)
        # Low beta stock: 0.5x market moves + noise
        asset_returns = market_returns * 0.5 + pd.Series(
            np.random.normal(0, 0.002, 252),
            index=market_returns.index
        )
        beta = compute_beta(asset_returns, market_returns)
        assert 0.3 < beta < 0.7, f"Low beta stock should have beta ~0.5, got {beta:.4f}"

    def test_beta_input_validation(self, market_returns):
        """Test input validation for beta."""
        asset_returns = market_returns * 1.2

        # Wrong types
        with pytest.raises(TypeError):
            compute_beta(asset_returns.values, market_returns)

        with pytest.raises(TypeError):
            compute_beta(asset_returns, market_returns.values)

        # Insufficient data
        with pytest.raises(ValueError, match="Insufficient"):
            compute_beta(asset_returns.head(10), market_returns.head(10))


class TestDataProcessingRiskMetrics:
    """Tests for risk metrics in data_processing.py."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        daily_mean = 0.10 / 252
        daily_std = 0.20 / np.sqrt(252)
        returns = pd.Series(
            np.random.normal(daily_mean, daily_std, 252),
            index=pd.date_range('2024-01-01', periods=252, freq='B')
        )
        return returns

    @pytest.fixture
    def sample_prices(self, sample_returns):
        """Generate prices from returns."""
        return (1 + sample_returns).cumprod() * 100

    def test_descriptive_stats_rf_parameter(self, sample_returns, sample_prices):
        """Test that risk_free_rate parameter is used."""
        stats_rf0 = compute_descriptive_stats(sample_returns, sample_prices, risk_free_rate=0.0)
        stats_rf5 = compute_descriptive_stats(sample_returns, sample_prices, risk_free_rate=0.05)

        assert stats_rf5['sharpe_ratio'] < stats_rf0['sharpe_ratio'], \
            "Higher Rf should reduce Sharpe"
        assert stats_rf5['risk_free_rate_used'] == 0.05, \
            "Should document the Rf used"

    def test_risk_metrics_horizon_keys(self, sample_returns):
        """Test that horizon-scaled VaR/CVaR are computed."""
        # Default horizon = 1
        metrics_1d = compute_risk_metrics(sample_returns, horizon_days=1)
        assert 'var_95_1d' in metrics_1d
        assert 'cvar_95_1d' in metrics_1d

        # 10-day horizon
        metrics_10d = compute_risk_metrics(sample_returns, horizon_days=10)
        assert 'var_95_10d' in metrics_10d
        assert 'cvar_95_10d' in metrics_10d

        # 10-day should be larger magnitude
        assert abs(metrics_10d['var_95_10d']) > abs(metrics_1d['var_95_1d'])

    def test_backwards_compatible_keys(self, sample_returns):
        """Test that backwards-compatible keys are present."""
        metrics = compute_risk_metrics(sample_returns)

        # Old keys should still work
        assert 'var_95' in metrics
        assert 'var_99' in metrics
        assert 'cvar_95' in metrics
        assert 'cvar_99' in metrics
        assert 'max_drawdown' in metrics


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_positive_returns(self):
        """Test with all positive returns (no losses)."""
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.008, 0.012] * 10)

        # VaR should still be computed (could be positive or small negative)
        var = compute_var(positive_returns)
        assert isinstance(var, float)

        # Sharpe should be very high
        sharpe = compute_sharpe_ratio(positive_returns, risk_free_rate=0.0)
        assert sharpe > 0

    def test_all_negative_returns(self):
        """Test with all negative returns."""
        negative_returns = pd.Series([-0.01, -0.02, -0.015, -0.008, -0.012] * 10)

        # VaR should be very negative
        var = compute_var(negative_returns)
        assert var < 0

        # Sharpe should be very negative
        sharpe = compute_sharpe_ratio(negative_returns, risk_free_rate=0.0)
        assert sharpe < 0

    def test_high_volatility_returns(self):
        """Test with very high volatility returns."""
        np.random.seed(42)
        high_vol_returns = pd.Series(np.random.normal(0, 0.10, 100))  # 10% daily vol

        # Should still compute without errors
        var = compute_var(high_vol_returns)
        sharpe = compute_sharpe_ratio(high_vol_returns)

        assert isinstance(var, float)
        assert isinstance(sharpe, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
