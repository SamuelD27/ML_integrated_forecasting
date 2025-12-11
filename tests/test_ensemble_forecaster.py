"""
Tests for Ensemble Forecaster (Phase 3)
=======================================
Tests for the ensemble forecaster with N-BEATS, LSTM, and TFT wrappers.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ml_models.ensemble_forecaster import (
    EnsembleForecaster,
    NBeatsForecaster,
    LSTMForecaster,
    TFTForecaster,
    ForecastResult,
    BaseForecaster,
    forecast_ticker,
    get_forecaster,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_price_history():
    """Create mock price history DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(120) * 2)
    prices = np.abs(prices) + 50

    return pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 120),
    }, index=dates)


@pytest.fixture
def sample_config():
    """Sample forecaster configuration."""
    return {
        'forecaster': {
            'horizon_days': 10,
            'models': [
                {'name': 'nbeats', 'weight': 0.35, 'enabled': True},
                {'name': 'lstm', 'weight': 0.35, 'enabled': True},
                {'name': 'tft', 'weight': 0.30, 'enabled': True},
            ],
            'ensemble_method': 'weighted_average',
        }
    }


# =============================================================================
# Test ForecastResult
# =============================================================================

class TestForecastResult:
    """Tests for ForecastResult dataclass."""

    def test_create_forecast_result(self):
        """Test creating a ForecastResult."""
        result = ForecastResult(
            mean_return=0.05,
            volatility=0.20,
            p10=-0.03,
            p90=0.12,
            confidence_score=0.7,
            model_name="test_model"
        )

        assert result.mean_return == 0.05
        assert result.volatility == 0.20
        assert result.p10 == -0.03
        assert result.p90 == 0.12
        assert result.confidence_score == 0.7
        assert result.model_name == "test_model"

    def test_forecast_result_optional_fields(self):
        """Test ForecastResult with optional fields."""
        result = ForecastResult(
            mean_return=0.05,
            volatility=0.20,
            p10=-0.03,
            p90=0.12,
            confidence_score=0.7,
        )

        assert result.raw_predictions is None
        assert result.model_name == ""
        assert result.metadata == {}


# =============================================================================
# Test BaseForecaster
# =============================================================================

class TestBaseForecaster:
    """Tests for BaseForecaster functionality."""

    def test_prepare_returns(self, mock_price_history):
        """Test returns preparation from price data."""
        # Create a concrete implementation for testing
        class ConcreteForecaster(BaseForecaster):
            def load_model(self):
                return True

            def forecast(self, price_history, horizon_days=10):
                return ForecastResult(0.0, 0.0, 0.0, 0.0, 0.0)

        forecaster = ConcreteForecaster()
        returns = forecaster._prepare_returns(mock_price_history)

        assert isinstance(returns, np.ndarray)
        assert len(returns) == len(mock_price_history) - 1  # One less due to diff
        assert np.isfinite(returns).all()


# =============================================================================
# Test NBeatsForecaster
# =============================================================================

class TestNBeatsForecaster:
    """Tests for N-BEATS forecaster wrapper."""

    def test_initialization(self):
        """Test N-BEATS forecaster initialization."""
        forecaster = NBeatsForecaster(device='cpu')

        assert forecaster.input_size == 60
        assert forecaster.output_size == 21
        assert forecaster.model_name == "nbeats"
        assert not forecaster.is_loaded

    def test_load_model_without_checkpoint(self):
        """Test loading N-BEATS without checkpoint."""
        forecaster = NBeatsForecaster(device='cpu')
        result = forecaster.load_model()

        assert result == True
        assert forecaster.is_loaded
        # Model is None when torch isn't available (fallback mode)
        if HAS_TORCH:
            assert forecaster.model is not None

    def test_forecast_generates_result(self, mock_price_history):
        """Test N-BEATS forecast generation."""
        forecaster = NBeatsForecaster(device='cpu')

        result = forecaster.forecast(mock_price_history, horizon_days=10)

        assert isinstance(result, ForecastResult)
        assert result.model_name == "nbeats"
        assert isinstance(result.mean_return, float)
        assert isinstance(result.volatility, float)
        assert result.raw_predictions is not None
        assert 0 <= result.confidence_score <= 1

    def test_forecast_handles_short_history(self):
        """Test N-BEATS handles short price history."""
        forecaster = NBeatsForecaster(device='cpu')

        # Create short history (less than input_size)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        prices = np.linspace(100, 110, 30)
        short_history = pd.DataFrame({
            'Close': prices
        }, index=dates)

        result = forecaster.forecast(short_history, horizon_days=10)

        assert isinstance(result, ForecastResult)
        assert np.isfinite(result.mean_return)


# =============================================================================
# Test LSTMForecaster
# =============================================================================

class TestLSTMForecaster:
    """Tests for LSTM forecaster wrapper."""

    def test_initialization(self):
        """Test LSTM forecaster initialization."""
        forecaster = LSTMForecaster(device='cpu')

        assert forecaster.input_dim == 1
        assert forecaster.hidden_dim == 256
        assert forecaster.seq_length == 60
        assert forecaster.model_name == "lstm"

    def test_load_model_without_checkpoint(self):
        """Test loading LSTM without checkpoint."""
        forecaster = LSTMForecaster(device='cpu')
        result = forecaster.load_model()

        assert result == True
        assert forecaster.is_loaded

    def test_forecast_generates_result(self, mock_price_history):
        """Test LSTM forecast generation."""
        forecaster = LSTMForecaster(device='cpu')

        result = forecaster.forecast(mock_price_history, horizon_days=10)

        assert isinstance(result, ForecastResult)
        assert result.model_name == "lstm"
        assert isinstance(result.mean_return, float)
        assert np.isfinite(result.mean_return)


# =============================================================================
# Test TFTForecaster
# =============================================================================

class TestTFTForecaster:
    """Tests for TFT forecaster wrapper."""

    def test_initialization(self):
        """Test TFT forecaster initialization."""
        forecaster = TFTForecaster(device='cpu')

        assert forecaster.seq_length == 60
        assert forecaster.model_name == "tft"

    def test_load_model_without_checkpoint(self):
        """Test loading TFT without checkpoint (fallback mode)."""
        forecaster = TFTForecaster(device='cpu')
        result = forecaster.load_model()

        assert result == True
        assert forecaster.is_loaded

    def test_forecast_fallback(self, mock_price_history):
        """Test TFT forecast with fallback method."""
        forecaster = TFTForecaster(device='cpu')

        result = forecaster.forecast(mock_price_history, horizon_days=10)

        assert isinstance(result, ForecastResult)
        assert result.model_name == "tft"
        # Fallback has lower confidence
        assert result.confidence_score <= 0.5
        assert result.metadata.get('fallback', False) == True


# =============================================================================
# Test EnsembleForecaster
# =============================================================================

class TestEnsembleForecaster:
    """Tests for ensemble forecaster."""

    def test_initialization(self, sample_config):
        """Test ensemble initialization."""
        ensemble = EnsembleForecaster(config=sample_config, device='cpu')

        assert 'nbeats' in ensemble.models
        assert 'lstm' in ensemble.models
        assert 'tft' in ensemble.models
        assert ensemble.weights['nbeats'] == 0.35
        assert ensemble.weights['lstm'] == 0.35
        assert ensemble.weights['tft'] == 0.30

    def test_initialization_default_weights(self):
        """Test ensemble with default weights."""
        ensemble = EnsembleForecaster(config=None, device='cpu')

        assert sum(ensemble.weights.values()) == pytest.approx(1.0)

    def test_forecast_sequential(self, mock_price_history, sample_config):
        """Test ensemble forecast in sequential mode."""
        ensemble = EnsembleForecaster(config=sample_config, device='cpu')

        result = ensemble.forecast(
            mock_price_history,
            horizon_days=10,
            parallel=False
        )

        assert 'mean_return' in result
        assert 'volatility' in result
        assert 'p10' in result
        assert 'p90' in result
        assert 'confidence_score' in result
        assert 'model_disagreement' in result
        assert 'individual_forecasts' in result
        assert 'weights_used' in result

        assert isinstance(result['mean_return'], float)
        assert isinstance(result['model_disagreement'], float)

    def test_forecast_parallel(self, mock_price_history, sample_config):
        """Test ensemble forecast in parallel mode."""
        ensemble = EnsembleForecaster(config=sample_config, device='cpu')

        result = ensemble.forecast(
            mock_price_history,
            horizon_days=10,
            parallel=True
        )

        assert 'mean_return' in result
        assert 'individual_forecasts' in result

    def test_individual_forecasts_included(self, mock_price_history, sample_config):
        """Test that individual model forecasts are included."""
        ensemble = EnsembleForecaster(config=sample_config, device='cpu')

        result = ensemble.forecast(mock_price_history, horizon_days=10)

        individual = result['individual_forecasts']
        assert len(individual) > 0

        for model_name, forecast in individual.items():
            assert 'mean_return' in forecast
            assert 'volatility' in forecast
            assert 'confidence' in forecast

    def test_weights_normalization(self, mock_price_history, sample_config):
        """Test that weights are normalized."""
        ensemble = EnsembleForecaster(config=sample_config, device='cpu')

        result = ensemble.forecast(mock_price_history, horizon_days=10)

        weights = result['weights_used']
        total = sum(weights.values())

        assert total == pytest.approx(1.0, rel=0.01)

    def test_model_disagreement_calculation(self, mock_price_history, sample_config):
        """Test model disagreement is calculated correctly."""
        ensemble = EnsembleForecaster(config=sample_config, device='cpu')

        result = ensemble.forecast(mock_price_history, horizon_days=10)

        # Model disagreement should be non-negative
        assert result['model_disagreement'] >= 0

        # If we have multiple forecasts, there should be some disagreement
        if len(result['individual_forecasts']) > 1:
            returns = [
                f['mean_return']
                for f in result['individual_forecasts'].values()
            ]
            expected_std = np.std(returns)
            assert result['model_disagreement'] == pytest.approx(expected_std, rel=0.01)

    def test_to_pipeline_forecast(self, mock_price_history, sample_config):
        """Test conversion to pipeline Forecast type."""
        ensemble = EnsembleForecaster(config=sample_config, device='cpu')

        result = ensemble.forecast(mock_price_history, horizon_days=10)

        forecast = ensemble.to_pipeline_forecast(
            ticker='AAPL',
            result=result,
            as_of_date=pd.Timestamp.now(),
            horizon_days=10
        )

        from pipeline.core_types import Forecast
        assert isinstance(forecast, Forecast)
        assert forecast.ticker == 'AAPL'
        assert forecast.expected_return == result['mean_return']
        assert forecast.model_disagreement == result['model_disagreement']

    def test_default_forecast_on_failure(self, sample_config):
        """Test default forecast when all models fail."""
        ensemble = EnsembleForecaster(config=sample_config, device='cpu')

        # Force all models to fail by passing invalid data
        result = ensemble._default_forecast()

        assert result['mean_return'] == 0.0
        assert result['confidence_score'] == 0.1
        assert 'error' in result


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_forecast_ticker(self, mock_price_history):
        """Test forecast_ticker convenience function."""
        result = forecast_ticker(
            ticker='AAPL',
            price_history=mock_price_history,
            horizon_days=10
        )

        assert result['ticker'] == 'AAPL'
        assert 'mean_return' in result
        assert 'volatility' in result

    def test_get_forecaster_singleton(self):
        """Test get_forecaster returns singleton."""
        forecaster1 = get_forecaster()
        forecaster2 = get_forecaster()

        assert forecaster1 is forecaster2

    def test_get_forecaster_reload(self):
        """Test get_forecaster with reload."""
        forecaster1 = get_forecaster()
        forecaster2 = get_forecaster(reload=True)

        assert forecaster1 is not forecaster2


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_price_history(self):
        """Test handling of empty price history."""
        ensemble = EnsembleForecaster(device='cpu')

        empty_df = pd.DataFrame(columns=['Close'])

        # Should return default or handle gracefully
        result = ensemble.forecast(empty_df, horizon_days=10)

        # Should not crash, returns some default
        assert 'mean_return' in result

    def test_single_price_point(self):
        """Test handling of single price point."""
        forecaster = NBeatsForecaster(device='cpu')

        single_df = pd.DataFrame({
            'Close': [100.0]
        }, index=[datetime.now()])

        result = forecaster.forecast(single_df, horizon_days=10)

        assert isinstance(result, ForecastResult)

    def test_nan_in_prices(self, mock_price_history):
        """Test handling of NaN in prices."""
        df = mock_price_history.copy()
        df.iloc[10, df.columns.get_loc('Close')] = np.nan

        forecaster = NBeatsForecaster(device='cpu')
        result = forecaster.forecast(df, horizon_days=10)

        assert isinstance(result, ForecastResult)
        assert np.isfinite(result.mean_return)

    def test_different_column_names(self):
        """Test handling of different column name cases."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.linspace(100, 120, 100)

        # Test lowercase
        df_lower = pd.DataFrame({'close': prices}, index=dates)
        forecaster = NBeatsForecaster(device='cpu')
        result = forecaster.forecast(df_lower, horizon_days=10)
        assert isinstance(result, ForecastResult)

        # Test with Adj Close
        df_adj = pd.DataFrame({'Adj Close': prices}, index=dates)
        result = forecaster.forecast(df_adj, horizon_days=10)
        assert isinstance(result, ForecastResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
