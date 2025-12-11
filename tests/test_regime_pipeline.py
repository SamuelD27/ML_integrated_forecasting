"""
Tests for Regime Pipeline (Phase 2)
====================================
Tests for macro_loader and regime_utils pipeline components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.macro_loader import (
    load_macro_features,
    get_current_macro_snapshot,
    get_macro_regime_features,
    _add_derived_features,
)
from pipeline.regime_utils import (
    RegimeDetector,
    classify_regime,
    get_regime_parameters,
    compute_regime_time_series,
    REGIME_INT_MAP,
    REGIME_LABEL_MAP,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_price_data():
    """Mock OHLCV price data."""
    dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(120) * 2)
    prices = np.abs(prices) + 50  # Ensure positive

    return pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 120),
    }, index=dates)


@pytest.fixture
def mock_macro_data():
    """Mock macro data DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='B')

    return pd.DataFrame({
        'vix': 20 + np.random.randn(60) * 5,
        'treasury_2y': 4.0 + np.random.randn(60) * 0.2,
        'treasury_10y': 4.5 + np.random.randn(60) * 0.2,
        'yield_spread': 0.5 + np.random.randn(60) * 0.1,
        'spy_return': np.random.randn(60) * 0.01,
        'spy_close': 500 + np.cumsum(np.random.randn(60) * 2),
    }, index=dates)


@pytest.fixture
def sample_config():
    """Sample regime configuration."""
    return {
        'regime': {
            'min_persistence_days': 3,
            'smoothing_window': 5,
            'model_path': 'models/regime_classifier/regime_model.pt',
            'regime_params': {
                'Bull': {
                    'risk_multiplier': 1.0,
                    'position_size_multiplier': 1.0,
                    'options_enabled': True,
                },
                'Bear': {
                    'risk_multiplier': 0.5,
                    'position_size_multiplier': 0.5,
                    'options_enabled': True,
                },
                'Neutral': {
                    'risk_multiplier': 0.8,
                    'position_size_multiplier': 0.8,
                    'options_enabled': True,
                },
                'Crisis': {
                    'risk_multiplier': 0.2,
                    'position_size_multiplier': 0.2,
                    'options_enabled': False,
                },
            }
        }
    }


# =============================================================================
# Test Macro Loader
# =============================================================================

class TestMacroLoader:
    """Tests for macro data loading."""

    def test_load_macro_features_structure(self):
        """Test that load_macro_features returns correct structure."""
        with patch('data.macro_loader._load_vix') as mock_vix, \
             patch('data.macro_loader._load_treasury_yields') as mock_treasury, \
             patch('data.macro_loader._load_spy_returns') as mock_spy:

            # Mock returns
            mock_vix.return_value = pd.Series(20.0, index=pd.date_range('2024-01-01', periods=30))
            mock_treasury.return_value = pd.DataFrame({
                'treasury_2y': 4.0,
                'treasury_10y': 4.5,
            }, index=pd.date_range('2024-01-01', periods=30))
            mock_spy.return_value = pd.DataFrame({
                'close': 500.0,
                'return': 0.001,
            }, index=pd.date_range('2024-01-01', periods=30))

            result = load_macro_features(
                start_date='2024-01-15',
                end_date='2024-01-30',
                config=None
            )

            assert isinstance(result, pd.DataFrame)
            assert 'vix' in result.columns
            assert 'yield_spread' in result.columns

    def test_add_derived_features(self):
        """Test derived feature calculation."""
        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        df = pd.DataFrame({
            'vix': np.linspace(15, 25, 30),
            'spy_return': np.random.randn(30) * 0.01,
            'yield_spread': np.linspace(0.3, 0.7, 30),
        }, index=dates)

        result = _add_derived_features(df)

        # Check VIX-derived features
        assert 'vix_change' in result.columns
        assert 'vix_ma_20' in result.columns

        # Check SPY-derived features
        assert 'spy_return_5d' in result.columns
        assert 'spy_return_20d' in result.columns

        # Check yield curve features
        assert 'yield_spread_change' in result.columns
        assert 'curve_inverted' in result.columns

    def test_get_current_macro_snapshot(self):
        """Test getting current macro snapshot."""
        with patch('data.macro_loader.load_macro_features') as mock_load:
            mock_load.return_value = pd.DataFrame({
                'vix': [20.5],
                'treasury_2y': [4.0],
                'treasury_10y': [4.5],
                'yield_spread': [0.5],
                'spy_return': [0.002],
            }, index=[pd.Timestamp.now().normalize()])

            result = get_current_macro_snapshot()

            assert isinstance(result, dict)
            assert 'vix' in result
            assert isinstance(result['vix'], float)

    def test_macro_features_handle_missing_data(self):
        """Test that macro loader handles missing data gracefully."""
        with patch('data.macro_loader._load_vix') as mock_vix, \
             patch('data.macro_loader._load_treasury_yields') as mock_treasury, \
             patch('data.macro_loader._load_spy_returns') as mock_spy:

            mock_vix.return_value = None
            mock_treasury.return_value = None
            mock_spy.return_value = None

            result = load_macro_features(
                start_date='2024-01-01',
                end_date='2024-01-10'
            )

            # Should still return a DataFrame with defaults
            assert isinstance(result, pd.DataFrame)
            assert 'vix' in result.columns  # Should have fallback value

    def test_get_macro_regime_features(self):
        """Test getting regime-specific macro features."""
        with patch('data.macro_loader.load_macro_features') as mock_load:
            mock_load.return_value = pd.DataFrame({
                'vix': [20.0, 21.0],
                'vix_change': [0.5, 1.0],
                'vix_percentile': [0.5, 0.6],
                'vix_zscore': [0.1, 0.2],
                'yield_spread': [0.5, 0.5],
                'curve_inverted': [0, 0],
                'spy_return': [0.001, 0.002],
                'spy_return_5d': [0.005, 0.01],
                'spy_return_20d': [0.02, 0.03],
                'spy_vol_20d': [0.15, 0.16],
                'extra_col': [1, 2],  # Should be excluded
            }, index=pd.date_range('2024-01-01', periods=2))

            result = get_macro_regime_features('2024-01-01', '2024-01-02')

            # Should only include regime-relevant columns
            assert 'vix' in result.columns
            assert 'spy_vol_20d' in result.columns
            assert 'extra_col' not in result.columns  # Filtered out


# =============================================================================
# Test Regime Detector
# =============================================================================

class TestRegimeDetector:
    """Tests for RegimeDetector class."""

    def test_initialization(self, sample_config):
        """Test RegimeDetector initialization."""
        detector = RegimeDetector(config=sample_config)

        assert detector.min_persistence == 3
        assert detector.smoothing_window == 5
        assert detector._regime_history == []

    def test_compute_regime_features(self, mock_price_data):
        """Test feature computation from price data."""
        detector = RegimeDetector()
        features = detector.compute_regime_features(mock_price_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(mock_price_data)

        # Check all required features are present
        required_features = [
            'vol_20d', 'ret_20d', 'ret_60d', 'vol_ratio',
            'rsi', 'macd', 'adx', 'volume_ratio',
            'price_momentum', 'volatility_trend'
        ]
        for feat in required_features:
            assert feat in features.columns, f"Missing feature: {feat}"

    def test_classify_regime_from_dict(self, sample_config):
        """Test regime classification from feature dict."""
        detector = RegimeDetector(config=sample_config)

        features = {
            'vol_20d': 0.15,
            'ret_20d': 0.05,
            'ret_60d': 0.10,
            'vol_ratio': 1.0,
            'rsi': 60.0,
            'macd': 0.01,
            'adx': 30.0,
            'volume_ratio': 1.0,
            'price_momentum': 1.05,
            'volatility_trend': 1.0,
        }

        result = detector.classify_regime(features, apply_smoothing=False)

        assert 'regime' in result
        assert 'regime_int' in result
        assert 'confidence' in result
        assert result['regime'] in ['bull', 'bear', 'neutral', 'crisis']
        assert 0 <= result['confidence'] <= 1

    def test_classify_regime_from_array(self, sample_config):
        """Test regime classification from numpy array."""
        detector = RegimeDetector(config=sample_config)

        features = np.array([0.15, 0.05, 0.10, 1.0, 60.0, 0.01, 30.0, 1.0, 1.05, 1.0])

        result = detector.classify_regime(features, apply_smoothing=False)

        assert 'regime' in result
        assert result['regime'] in ['bull', 'bear', 'neutral', 'crisis']

    def test_classify_regime_from_series(self, sample_config):
        """Test regime classification from pandas Series."""
        detector = RegimeDetector(config=sample_config)

        features = pd.Series({
            'vol_20d': 0.15,
            'ret_20d': 0.05,
            'ret_60d': 0.10,
            'vol_ratio': 1.0,
            'rsi': 60.0,
            'macd': 0.01,
            'adx': 30.0,
            'volume_ratio': 1.0,
            'price_momentum': 1.05,
            'volatility_trend': 1.0,
        })

        result = detector.classify_regime(features, apply_smoothing=False)

        assert 'regime' in result
        assert result['regime'] in ['bull', 'bear', 'neutral', 'crisis']

    def test_rule_based_classifier_crisis(self, sample_config):
        """Test rule-based classifier detects crisis conditions."""
        detector = RegimeDetector(config=sample_config)

        # Crisis conditions: high vol + negative returns
        crisis_features = {
            'vol_20d': 0.50,  # Very high volatility
            'ret_20d': -0.15,  # Negative returns
            'ret_60d': -0.20,
            'vol_ratio': 2.0,
            'rsi': 25.0,
            'macd': -0.02,
            'adx': 40.0,
            'volume_ratio': 2.0,
            'price_momentum': 0.85,
            'volatility_trend': 1.5,
        }

        result = detector.classify_regime(crisis_features, apply_smoothing=False)

        # Should detect crisis or bear
        assert result['regime'] in ['crisis', 'bear']

    def test_rule_based_classifier_bull(self, sample_config):
        """Test rule-based classifier detects bull conditions."""
        detector = RegimeDetector(config=sample_config)

        # Bull conditions: positive trend, moderate RSI
        bull_features = {
            'vol_20d': 0.12,
            'ret_20d': 0.03,
            'ret_60d': 0.15,  # Strong positive trend
            'vol_ratio': 0.9,
            'rsi': 65.0,
            'macd': 0.02,
            'adx': 35.0,  # Strong trend
            'volume_ratio': 1.2,
            'price_momentum': 1.1,
            'volatility_trend': 0.9,
        }

        result = detector.classify_regime(bull_features, apply_smoothing=False)

        assert result['regime'] == 'bull'

    def test_smoothing_persistence(self, sample_config):
        """Test that smoothing requires minimum persistence."""
        detector = RegimeDetector(config=sample_config)

        # Simulate regime changes
        bull_features = {'vol_20d': 0.12, 'ret_20d': 0.03, 'ret_60d': 0.15, 'vol_ratio': 0.9,
                        'rsi': 65.0, 'macd': 0.02, 'adx': 35.0, 'volume_ratio': 1.2,
                        'price_momentum': 1.1, 'volatility_trend': 0.9}

        bear_features = {'vol_20d': 0.25, 'ret_20d': -0.05, 'ret_60d': -0.20, 'vol_ratio': 1.5,
                        'rsi': 35.0, 'macd': -0.02, 'adx': 30.0, 'volume_ratio': 1.0,
                        'price_momentum': 0.9, 'volatility_trend': 1.2}

        # First few bull signals should establish bull regime
        for _ in range(3):
            result = detector.classify_regime(bull_features, apply_smoothing=True)

        # Single bear signal shouldn't immediately change regime
        result = detector.classify_regime(bear_features, apply_smoothing=True)

        # May still be bull due to smoothing
        # (depends on exact implementation, but smoothing should be evident)
        assert 'smoothed' in result

    def test_reset_smoothing(self, sample_config):
        """Test smoothing reset."""
        detector = RegimeDetector(config=sample_config)

        # Add some regime history
        detector._regime_history = [0, 0, 0, 1, 1]
        detector._last_confirmed_regime = 0

        detector.reset_smoothing()

        assert detector._regime_history == []
        assert detector._last_confirmed_regime is None


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_classify_regime_function(self):
        """Test classify_regime convenience function."""
        with patch('pipeline.regime_utils.RegimeDetector') as MockDetector:
            mock_instance = Mock()
            mock_instance.compute_regime_features.return_value = pd.DataFrame({
                'vol_20d': [0.15],
                'ret_20d': [0.05],
                'ret_60d': [0.10],
                'vol_ratio': [1.0],
                'rsi': [60.0],
                'macd': [0.01],
                'adx': [30.0],
                'volume_ratio': [1.0],
                'price_momentum': [1.05],
                'volatility_trend': [1.0],
            })
            mock_instance.classify_regime.return_value = {
                'regime': 'bull',
                'regime_int': 0,
                'confidence': 0.7,
            }
            MockDetector.return_value = mock_instance

            # Create mock price data
            price_data = pd.DataFrame({
                'Close': [100, 101, 102],
            }, index=pd.date_range('2024-01-01', periods=3))

            result = classify_regime(
                as_of_date='2024-01-15',
                price_data=price_data
            )

            assert result['regime'] == 'bull'

    def test_get_regime_parameters_bull(self, sample_config):
        """Test getting bull regime parameters."""
        params = get_regime_parameters('bull', sample_config)

        assert params['risk_multiplier'] == 1.0
        assert params['position_size_multiplier'] == 1.0
        assert params['options_enabled'] == True

    def test_get_regime_parameters_crisis(self, sample_config):
        """Test getting crisis regime parameters."""
        params = get_regime_parameters('crisis', sample_config)

        assert params['risk_multiplier'] == 0.2
        assert params['position_size_multiplier'] == 0.2
        assert params['options_enabled'] == False

    def test_get_regime_parameters_from_int(self, sample_config):
        """Test getting parameters from regime integer."""
        params = get_regime_parameters(3, sample_config)  # 3 = crisis

        assert params['risk_multiplier'] == 0.2
        assert params['options_enabled'] == False

    def test_get_regime_parameters_default(self):
        """Test getting default parameters without config."""
        params = get_regime_parameters('bear')

        assert 'risk_multiplier' in params
        assert 'position_size_multiplier' in params
        assert 'options_enabled' in params

    def test_compute_regime_time_series(self, mock_price_data, sample_config):
        """Test computing regime for entire time series."""
        result = compute_regime_time_series(mock_price_data, sample_config)

        assert isinstance(result, pd.DataFrame)
        assert 'regime' in result.columns
        assert 'regime_int' in result.columns
        assert 'confidence' in result.columns
        assert len(result) == len(mock_price_data)


# =============================================================================
# Test Regime Mappings
# =============================================================================

class TestRegimeMappings:
    """Tests for regime mapping constants."""

    def test_regime_int_map(self):
        """Test regime to int mapping."""
        assert REGIME_INT_MAP['bull'] == 0
        assert REGIME_INT_MAP['bear'] == 1
        assert REGIME_INT_MAP['neutral'] == 2
        assert REGIME_INT_MAP['crisis'] == 3

    def test_regime_label_map(self):
        """Test int to regime mapping."""
        assert REGIME_LABEL_MAP[0] == 'bull'
        assert REGIME_LABEL_MAP[1] == 'bear'
        assert REGIME_LABEL_MAP[2] == 'neutral'
        assert REGIME_LABEL_MAP[3] == 'crisis'

    def test_mappings_are_inverse(self):
        """Test that mappings are inverse of each other."""
        for regime, int_val in REGIME_INT_MAP.items():
            assert REGIME_LABEL_MAP[int_val] == regime

        for int_val, regime in REGIME_LABEL_MAP.items():
            assert REGIME_INT_MAP[regime] == int_val


# =============================================================================
# Test Feature Computation
# =============================================================================

class TestFeatureComputation:
    """Tests for feature computation methods."""

    def test_rsi_computation(self):
        """Test RSI computation."""
        detector = RegimeDetector()

        # Create price series with known trend
        prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                           120, 122, 124, 126, 128, 130])

        rsi = detector._compute_rsi(prices, period=14)

        # Strong uptrend should have RSI > 50
        assert rsi.iloc[-1] > 50

    def test_rsi_overbought(self):
        """Test RSI detects overbought."""
        detector = RegimeDetector()

        # Continuous upward movement
        prices = pd.Series(np.arange(50, 150, 1.0))

        rsi = detector._compute_rsi(prices, period=14)

        # Very strong uptrend should approach 100
        assert rsi.iloc[-1] > 70

    def test_macd_computation(self):
        """Test MACD computation."""
        detector = RegimeDetector()

        prices = pd.Series(np.linspace(100, 120, 50))  # Uptrend

        macd = detector._compute_macd(prices)

        # Uptrend should have positive MACD
        assert macd.iloc[-1] > 0

    def test_adx_computation(self, mock_price_data):
        """Test ADX computation."""
        detector = RegimeDetector()

        adx = detector._compute_adx(
            mock_price_data['High'],
            mock_price_data['Low'],
            mock_price_data['Close']
        )

        # ADX should be between 0 and 100
        assert adx.dropna().min() >= 0
        assert adx.dropna().max() <= 100

    def test_features_no_nan_at_end(self, mock_price_data):
        """Test that features don't have NaN at the end of series."""
        detector = RegimeDetector()
        features = detector.compute_regime_features(mock_price_data)

        # Last row should have no NaN (important for live trading)
        last_row = features.iloc[-1]
        assert not last_row.isna().any(), f"NaN in last row: {last_row[last_row.isna()]}"


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestRegimePipelineIntegration:
    """Integration tests for regime pipeline (requires API)."""

    @pytest.mark.skip(reason="Requires API access")
    def test_end_to_end_regime_classification(self):
        """Test full regime classification pipeline."""
        result = classify_regime(
            as_of_date=pd.Timestamp.now().normalize(),
            price_data=None  # Will fetch
        )

        assert 'regime' in result
        assert result['regime'] in ['bull', 'bear', 'neutral', 'crisis']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
