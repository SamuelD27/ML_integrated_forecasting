"""
Tests for Practical ML Ensemble System
========================================
Comprehensive tests for practical_ensemble.py

Run: pytest tests/test_practical_ensemble.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ml_models.practical_ensemble import StockEnsemble, generate_trading_signal


class TestEnsembleBasic:
    """Basic ensemble functionality tests."""

    def test_ensemble_initialization(self):
        """Test ensemble initializes correctly."""
        ensemble = StockEnsemble()

        assert ensemble.models == {}
        assert ensemble.weights == {}
        assert ensemble.is_fitted is False
        assert ensemble.feature_names == []

    def test_ensemble_fit_with_synthetic_data(self):
        """Test ensemble training with synthetic price data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(500) * 0.02 + 0.0005)),
            index=dates
        )

        ensemble = StockEnsemble()
        results = ensemble.fit(prices, verbose=False)

        # Check results structure
        assert 'lgb_rmse' in results
        assert 'ridge_rmse' in results
        assert 'momentum_rmse' in results
        assert 'weights' in results
        assert ensemble.is_fitted is True

        # Check weights sum to 1
        weight_sum = sum(results['weights'].values())
        assert abs(weight_sum - 1.0) < 1e-6

    def test_ensemble_prediction(self):
        """Test prediction returns correct structure."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(500) * 0.02)),
            index=dates
        )

        ensemble = StockEnsemble()
        ensemble.fit(prices, verbose=False)
        forecast = ensemble.predict(prices)

        # Check all required keys exist
        required_keys = [
            'current_price', 'forecast_price', 'forecast_return',
            'lower_bound', 'upper_bound', 'confidence'
        ]
        for key in required_keys:
            assert key in forecast

        # Check bounds logic
        assert forecast['lower_bound'] < forecast['forecast_price']
        assert forecast['forecast_price'] < forecast['upper_bound']

        # Check confidence range
        assert 0 <= forecast['confidence'] <= 1


class TestTradingSignals:
    """Tests for trading signal generation."""

    def test_strong_buy_signal(self):
        """Test strong buy signal."""
        forecast = {
            'forecast_return': 0.10,
            'confidence': 0.8
        }
        signal = generate_trading_signal(forecast, buy_threshold=0.02)
        assert "STRONG BUY" in signal

    def test_hold_low_confidence(self):
        """Test hold signal with low confidence."""
        forecast = {
            'forecast_return': 0.10,
            'confidence': 0.3
        }
        signal = generate_trading_signal(forecast, min_confidence=0.5)
        assert "HOLD" in signal


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
