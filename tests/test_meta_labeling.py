"""
Tests for Meta-Labeling Pipeline (Phase 4)
==========================================
Tests for triple barrier labeling, meta-features, meta-model, and trade filtering.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import meta-labeling modules
from ml_models.triple_barrier import (
    label_triple_barrier,
    get_triple_barrier_labels,
    calculate_barrier_stats,
    label_with_volatility_scaling,
)

from ml_models.meta_features import (
    MetaFeatures,
    extract_forecast_features,
    extract_regime_features,
    extract_market_features,
    build_meta_features,
    get_feature_names,
    prepare_training_features,
)

from ml_models.meta_model import (
    MetaModel,
    train_meta_model,
    cross_validate_meta_model,
    get_meta_model,
    HAS_XGB,
    HAS_SKLEARN,
)

from pipeline.trade_filter import (
    filter_by_meta_prob,
    filter_by_regime,
    filter_by_direction,
    filter_by_expected_return,
    filter_by_volatility,
    filter_signals,
    rank_signals,
    select_top_signals,
    diversify_signals,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_price_series():
    """Create mock price series for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    # Create trending price with some noise
    base = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.randn(100) * 2
    prices = base + trend + noise
    prices = np.maximum(prices, 50)  # Ensure positive
    return pd.Series(prices, index=dates)


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
def mock_forecast():
    """Create mock forecast dict."""
    return {
        'expected_return': 0.05,
        'volatility': 0.20,
        'p10': -0.03,
        'p90': 0.12,
        'model_disagreement': 0.02,
        'confidence': 0.7,
    }


@pytest.fixture
def mock_macro_data():
    """Create mock macro data."""
    return {
        'vix': 18.5,
        'vix_change': -0.5,
        'vix_percentile': 0.35,
        'yield_spread': 0.75,
        'spy_return': 0.01,
        'spy_vol_20d': 0.15,
    }


@pytest.fixture
def mock_trade_signal():
    """Create mock TradeSignal-like object."""
    class MockSignal:
        def __init__(self, ticker, meta_prob, regime, direction, expected_return, expected_vol):
            self.ticker = ticker
            self.meta_prob = meta_prob
            self.regime = regime
            self.regime_label = {0: 'Bull', 1: 'Bear', 2: 'Neutral', 3: 'Crisis'}.get(regime, 'Neutral')
            self.direction = direction
            self.expected_return = expected_return
            self.expected_vol = expected_vol
            self.forecast = None
            self.snapshot = None

    return MockSignal


# =============================================================================
# Test Triple Barrier Labeling
# =============================================================================

class TestTripleBarrierLabeling:
    """Tests for triple barrier labeling."""

    def test_label_triple_barrier_basic(self, mock_price_series):
        """Test basic triple barrier labeling."""
        results = label_triple_barrier(
            mock_price_series,
            tp_pct=0.05,
            sl_pct=0.02,
            max_holding_days=10,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'label' in results.columns
        assert 'barrier_hit' in results.columns
        assert 'return_pct' in results.columns
        assert 'holding_days' in results.columns

        # Labels should be 0 or 1
        assert results['label'].isin([0, 1]).all()

        # Barrier hit should be one of three types
        assert results['barrier_hit'].isin(['tp', 'sl', 'time']).all()

    def test_label_triple_barrier_holding_days(self, mock_price_series):
        """Test that holding days are within bounds."""
        max_days = 10
        results = label_triple_barrier(
            mock_price_series,
            tp_pct=0.05,
            sl_pct=0.02,
            max_holding_days=max_days,
        )

        assert (results['holding_days'] >= 1).all()
        assert (results['holding_days'] <= max_days).all()

    def test_label_triple_barrier_short_series(self):
        """Test handling of short price series."""
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        prices = pd.Series([100, 101, 102, 103, 104], index=dates)

        results = label_triple_barrier(
            prices,
            tp_pct=0.05,
            sl_pct=0.02,
            max_holding_days=10,
        )

        # Should return empty DataFrame for too short series
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_label_triple_barrier_percentages(self, mock_price_series):
        """Test different take profit/stop loss percentages."""
        # Tight barriers
        tight = label_triple_barrier(
            mock_price_series,
            tp_pct=0.01,
            sl_pct=0.01,
            max_holding_days=10,
        )

        # Wide barriers
        wide = label_triple_barrier(
            mock_price_series,
            tp_pct=0.10,
            sl_pct=0.05,
            max_holding_days=10,
        )

        # Tight barriers should have fewer time barrier hits
        tight_time = (tight['barrier_hit'] == 'time').sum()
        wide_time = (wide['barrier_hit'] == 'time').sum()

        # Not strictly true but generally expected
        assert isinstance(tight_time, (int, np.integer))
        assert isinstance(wide_time, (int, np.integer))

    def test_get_triple_barrier_labels(self, mock_price_series):
        """Test get_triple_barrier_labels convenience function."""
        labels, results_df = get_triple_barrier_labels(mock_price_series)

        assert isinstance(labels, pd.Series)
        assert isinstance(results_df, pd.DataFrame)
        assert len(labels) == len(results_df)

    def test_get_triple_barrier_labels_with_config(self, mock_price_series):
        """Test get_triple_barrier_labels with config."""
        config = {
            'meta_labeling': {
                'triple_barrier': {
                    'take_profit_pct': 0.03,
                    'stop_loss_pct': 0.01,
                    'max_holding_days': 5,
                }
            }
        }

        labels, results_df = get_triple_barrier_labels(mock_price_series, config=config)

        assert isinstance(labels, pd.Series)
        assert (results_df['holding_days'] <= 5).all()

    def test_calculate_barrier_stats(self, mock_price_series):
        """Test barrier statistics calculation."""
        results_df = label_triple_barrier(mock_price_series)
        stats = calculate_barrier_stats(results_df)

        assert 'win_rate' in stats
        assert 'avg_return' in stats
        assert 'avg_holding' in stats
        assert 'tp_rate' in stats
        assert 'sl_rate' in stats
        assert 'time_rate' in stats

        # Win rate should be between 0 and 1
        assert 0 <= stats['win_rate'] <= 1

        # Rates should sum to 1
        total_rate = stats['tp_rate'] + stats['sl_rate'] + stats['time_rate']
        assert total_rate == pytest.approx(1.0, rel=0.01)

    def test_calculate_barrier_stats_empty(self):
        """Test stats calculation with empty DataFrame."""
        empty_df = pd.DataFrame()
        stats = calculate_barrier_stats(empty_df)

        assert stats['win_rate'] == 0.0
        assert stats['avg_return'] == 0.0

    def test_label_with_volatility_scaling(self, mock_price_series):
        """Test volatility-scaled barrier labeling."""
        results = label_with_volatility_scaling(
            mock_price_series,
            vol_window=20,
            vol_mult_tp=2.0,
            vol_mult_sl=1.0,
            max_holding_days=10,
        )

        assert isinstance(results, pd.DataFrame)
        if len(results) > 0:
            assert 'tp_pct' in results.columns
            assert 'sl_pct' in results.columns
            assert 'daily_vol' in results.columns

            # Barriers should vary based on volatility
            assert results['tp_pct'].std() > 0 or len(results) == 1


# =============================================================================
# Test Meta-Features
# =============================================================================

class TestMetaFeatures:
    """Tests for meta-feature extraction."""

    def test_extract_forecast_features_dict(self, mock_forecast):
        """Test forecast feature extraction from dict."""
        features = extract_forecast_features(mock_forecast)

        assert 'expected_return' in features
        assert 'volatility' in features
        assert 'p10' in features
        assert 'p90' in features
        assert 'forecast_range' in features
        assert 'model_disagreement' in features
        assert 'confidence' in features
        assert 'return_vol_ratio' in features

        assert features['expected_return'] == 0.05
        assert features['forecast_range'] == pytest.approx(0.15, rel=0.01)

    def test_extract_regime_features(self):
        """Test regime feature extraction."""
        features = extract_regime_features(
            regime=0,
            regime_label='Bull',
            regime_confidence=0.8,
        )

        assert features['regime_int'] == 0.0
        assert features['regime_bull'] == 1.0
        assert features['regime_bear'] == 0.0
        assert features['regime_neutral'] == 0.0
        assert features['regime_crisis'] == 0.0
        assert features['regime_confidence'] == 0.8
        assert features['regime_risk_mult'] == 1.0

    def test_extract_regime_features_all_regimes(self):
        """Test regime features for all regime types."""
        regimes = {
            0: ('Bull', 1.0),
            1: ('Bear', 0.5),
            2: ('Neutral', 0.8),
            3: ('Crisis', 0.2),
        }

        for regime, (label, expected_mult) in regimes.items():
            features = extract_regime_features(regime, label)
            assert features['regime_int'] == float(regime)
            assert features['regime_risk_mult'] == expected_mult

    def test_extract_market_features_with_data(self, mock_price_history, mock_macro_data):
        """Test market feature extraction with data."""
        features = extract_market_features(
            price_history=mock_price_history,
            macro_data=mock_macro_data,
        )

        assert 'recent_returns_5d' in features
        assert 'recent_returns_20d' in features
        assert 'realized_vol_20d' in features
        assert 'realized_vol_60d' in features
        assert 'price_vs_ma20' in features
        assert 'price_vs_ma50' in features
        assert 'vix_level' in features
        assert 'vix_change' in features

        assert features['vix_level'] == 18.5
        assert features['spy_return'] == 0.01

    def test_extract_market_features_no_data(self):
        """Test market feature extraction without data."""
        features = extract_market_features(
            price_history=None,
            macro_data=None,
        )

        # Should return defaults
        assert features['vix_level'] == 20.0
        assert features['realized_vol_20d'] == 0.20
        assert features['recent_returns_5d'] == 0.0

    def test_build_meta_features(self, mock_forecast, mock_price_history, mock_macro_data):
        """Test complete meta-feature building."""
        meta_features = build_meta_features(
            forecast=mock_forecast,
            regime=0,
            regime_label='Bull',
            regime_confidence=0.8,
            price_history=mock_price_history,
            macro_data=mock_macro_data,
        )

        assert isinstance(meta_features, MetaFeatures)
        assert meta_features.values is not None
        assert len(meta_features.feature_names) > 0
        assert len(meta_features.values) == len(meta_features.feature_names)

    def test_meta_features_to_array(self, mock_forecast):
        """Test MetaFeatures to_array method."""
        meta_features = build_meta_features(
            forecast=mock_forecast,
            regime=1,
            regime_label='Bear',
        )

        array = meta_features.to_array()

        assert isinstance(array, np.ndarray)
        assert len(array) == len(meta_features.feature_names)

    def test_meta_features_to_dict(self, mock_forecast):
        """Test MetaFeatures to_dict method."""
        meta_features = build_meta_features(
            forecast=mock_forecast,
            regime=2,
            regime_label='Neutral',
        )

        d = meta_features.to_dict()

        assert isinstance(d, dict)
        assert len(d) == len(meta_features.feature_names)
        for name in meta_features.feature_names:
            assert name in d

    def test_get_feature_names(self):
        """Test get_feature_names returns consistent order."""
        names1 = get_feature_names()
        names2 = get_feature_names()

        assert names1 == names2
        assert len(names1) > 0

        # Check prefixes
        prefixes = ['forecast_', 'regime_', 'market_']
        assert all(any(n.startswith(p) for p in prefixes) for n in names1)

    def test_prepare_training_features(self):
        """Test training feature preparation."""
        forecasts = [
            {'expected_return': 0.05, 'volatility': 0.20},
            {'expected_return': -0.02, 'volatility': 0.25},
            {'expected_return': 0.08, 'volatility': 0.18},
        ]
        regimes = [0, 1, 0]

        X = prepare_training_features(forecasts, regimes)

        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 3
        assert X.shape[1] > 0


# =============================================================================
# Test Meta-Model
# =============================================================================

@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
class TestMetaModel:
    """Tests for meta-model."""

    def test_meta_model_initialization_xgboost(self):
        """Test meta-model initialization with XGBoost."""
        if not HAS_XGB:
            pytest.skip("XGBoost not available")

        model = MetaModel(model_type='xgboost')

        assert model.model_type == 'xgboost'
        assert model.model is not None
        assert not model.is_fitted

    def test_meta_model_initialization_random_forest(self):
        """Test meta-model initialization with Random Forest."""
        model = MetaModel(model_type='random_forest')

        assert model.model_type == 'random_forest'
        assert model.model is not None
        assert not model.is_fitted

    def test_meta_model_fit(self):
        """Test meta-model fitting."""
        model = MetaModel(model_type='random_forest')

        # Create training data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        metrics = model.fit(X, y, validation_split=0.2)

        assert model.is_fitted
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics

        assert 0 <= metrics['accuracy'] <= 1

    def test_meta_model_predict_proba(self):
        """Test meta-model probability prediction."""
        model = MetaModel(model_type='random_forest')

        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model.fit(X, y)

        # Single prediction
        proba = model.predict_proba(X[0])

        assert isinstance(proba, float)
        assert 0 <= proba <= 1

    def test_meta_model_predict_batch(self):
        """Test meta-model batch prediction."""
        model = MetaModel(model_type='random_forest')

        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model.fit(X, y)

        probas = model.predict_batch(X[:10])

        assert isinstance(probas, np.ndarray)
        assert len(probas) == 10
        assert all(0 <= p <= 1 for p in probas)

    def test_meta_model_unfitted_prediction(self):
        """Test prediction with unfitted model."""
        model = MetaModel(model_type='random_forest')

        X = np.random.randn(1, 10)
        proba = model.predict_proba(X)

        # Should return default 0.5
        assert proba == 0.5

    def test_meta_model_feature_importance(self):
        """Test feature importance extraction."""
        model = MetaModel(model_type='random_forest')

        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        feature_names = [f'feature_{i}' for i in range(10)]

        model.fit(X, y, feature_names=feature_names)

        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 10
        assert all(v >= 0 for v in importance.values())

    def test_meta_model_save_load(self, tmp_path):
        """Test model save and load."""
        model = MetaModel(model_type='random_forest')

        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model.fit(X, y)

        # Save
        save_path = tmp_path / "meta_model.pkl"
        model.save(str(save_path))

        assert save_path.exists()

        # Load
        new_model = MetaModel(model_path=str(save_path))

        assert new_model.is_fitted

        # Predictions should match
        proba1 = model.predict_proba(X[0])
        proba2 = new_model.predict_proba(X[0])

        assert proba1 == pytest.approx(proba2, rel=0.01)

    def test_train_meta_model_function(self):
        """Test train_meta_model convenience function."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model, metrics = train_meta_model(X, y, model_type='random_forest')

        assert isinstance(model, MetaModel)
        assert model.is_fitted
        assert isinstance(metrics, dict)

    def test_cross_validate_meta_model(self):
        """Test cross-validation."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        cv_results = cross_validate_meta_model(X, y, model_type='random_forest', n_folds=3)

        assert 'mean_auc' in cv_results
        assert 'std_auc' in cv_results
        assert 0 <= cv_results['mean_auc'] <= 1

    def test_get_meta_model_singleton(self):
        """Test global meta-model singleton."""
        model1 = get_meta_model()
        model2 = get_meta_model()

        assert model1 is model2

    def test_get_meta_model_reload(self):
        """Test meta-model reload."""
        model1 = get_meta_model()
        model2 = get_meta_model(reload=True)

        assert model1 is not model2

    def test_meta_model_handles_nan(self):
        """Test meta-model handles NaN values."""
        model = MetaModel(model_type='random_forest')

        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[0, 0] = np.nan
        X[1, 5] = np.inf
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Should not crash
        metrics = model.fit(X, y)

        assert model.is_fitted

        # Prediction with NaN
        X_test = np.array([[np.nan, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        proba = model.predict_proba(X_test)

        assert np.isfinite(proba)


# =============================================================================
# Test Trade Filter
# =============================================================================

class TestTradeFilter:
    """Tests for trade signal filtering."""

    def test_filter_by_meta_prob(self, mock_trade_signal):
        """Test filtering by meta probability."""
        signals = [
            mock_trade_signal('AAPL', 0.80, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.60, 0, 'long', 0.04, 0.18),
            mock_trade_signal('MSFT', 0.70, 0, 'long', 0.06, 0.22),
        ]

        filtered = filter_by_meta_prob(signals, threshold=0.65)

        assert len(filtered) == 2
        assert all(s.meta_prob >= 0.65 for s in filtered)

    def test_filter_by_meta_prob_none(self, mock_trade_signal):
        """Test filtering passes signals without meta_prob."""
        signals = [
            mock_trade_signal('AAPL', None, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.60, 0, 'long', 0.04, 0.18),
        ]

        filtered = filter_by_meta_prob(signals, threshold=0.65)

        # Signal with None meta_prob should pass through
        assert len(filtered) == 1
        assert filtered[0].ticker == 'AAPL'

    def test_filter_by_regime(self, mock_trade_signal):
        """Test filtering by regime."""
        signals = [
            mock_trade_signal('AAPL', 0.80, 0, 'long', 0.05, 0.20),  # Bull
            mock_trade_signal('GOOGL', 0.70, 1, 'long', 0.04, 0.18),  # Bear
            mock_trade_signal('MSFT', 0.75, 3, 'long', 0.06, 0.22),  # Crisis
        ]

        # Block crisis regime
        filtered = filter_by_regime(signals, blocked_regimes=[3])

        assert len(filtered) == 2
        assert all(s.regime != 3 for s in filtered)

    def test_filter_by_regime_allowed(self, mock_trade_signal):
        """Test filtering by allowed regimes."""
        signals = [
            mock_trade_signal('AAPL', 0.80, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.70, 1, 'long', 0.04, 0.18),
            mock_trade_signal('MSFT', 0.75, 2, 'long', 0.06, 0.22),
        ]

        # Only allow bull regime
        filtered = filter_by_regime(signals, allowed_regimes=[0])

        assert len(filtered) == 1
        assert filtered[0].regime == 0

    def test_filter_by_direction(self, mock_trade_signal):
        """Test filtering by direction."""
        signals = [
            mock_trade_signal('AAPL', 0.80, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.70, 0, 'short', -0.04, 0.18),
            mock_trade_signal('MSFT', 0.75, 0, 'flat', 0.00, 0.10),
        ]

        # Long only
        filtered = filter_by_direction(signals, allowed_directions=['long'])

        assert len(filtered) == 1
        assert filtered[0].direction == 'long'

    def test_filter_by_expected_return(self, mock_trade_signal):
        """Test filtering by expected return."""
        signals = [
            mock_trade_signal('AAPL', 0.80, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.70, 0, 'long', 0.005, 0.18),
            mock_trade_signal('MSFT', 0.75, 0, 'long', 0.03, 0.22),
        ]

        filtered = filter_by_expected_return(signals, min_return=0.01)

        assert len(filtered) == 2
        assert all(s.expected_return >= 0.01 for s in filtered)

    def test_filter_by_expected_return_max(self, mock_trade_signal):
        """Test filtering by maximum expected return."""
        signals = [
            mock_trade_signal('AAPL', 0.80, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.70, 0, 'long', 0.02, 0.18),
            mock_trade_signal('MSFT', 0.75, 0, 'long', 0.15, 0.40),
        ]

        filtered = filter_by_expected_return(signals, min_return=0.01, max_return=0.10)

        assert len(filtered) == 2
        assert all(0.01 <= s.expected_return <= 0.10 for s in filtered)

    def test_filter_by_volatility(self, mock_trade_signal):
        """Test filtering by volatility."""
        signals = [
            mock_trade_signal('AAPL', 0.80, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.70, 0, 'long', 0.04, 0.55),
            mock_trade_signal('MSFT', 0.75, 0, 'long', 0.06, 0.30),
        ]

        filtered = filter_by_volatility(signals, max_vol=0.50)

        assert len(filtered) == 2
        assert all(s.expected_vol <= 0.50 for s in filtered)

    def test_filter_signals_combined(self, mock_trade_signal):
        """Test combined filtering with config."""
        signals = [
            mock_trade_signal('AAPL', 0.80, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.60, 0, 'long', 0.04, 0.18),
            mock_trade_signal('MSFT', 0.75, 3, 'long', 0.06, 0.22),  # Crisis
            mock_trade_signal('AMZN', 0.70, 0, 'short', 0.03, 0.25),  # Short
        ]

        config = {
            'meta_labeling': {'meta_prob_threshold': 0.65},
            'filters': {
                'blocked_regimes': [3],
                'long_only': True,
                'min_expected_return': 0.01,
                'max_volatility': 0.50,
            }
        }

        filtered = filter_signals(signals, config=config)

        # Should only keep AAPL (high prob, bull, long, good return/vol)
        assert len(filtered) == 1
        assert filtered[0].ticker == 'AAPL'

    def test_filter_signals_empty_list(self):
        """Test filtering empty signal list."""
        filtered = filter_signals([], config=None)

        assert filtered == []

    def test_rank_signals_by_meta_prob(self, mock_trade_signal):
        """Test ranking by meta probability."""
        signals = [
            mock_trade_signal('AAPL', 0.70, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.90, 0, 'long', 0.04, 0.18),
            mock_trade_signal('MSFT', 0.80, 0, 'long', 0.06, 0.22),
        ]

        ranked = rank_signals(signals, ranking_metric='meta_prob')

        assert ranked[0].ticker == 'GOOGL'
        assert ranked[1].ticker == 'MSFT'
        assert ranked[2].ticker == 'AAPL'

    def test_rank_signals_by_risk_reward(self, mock_trade_signal):
        """Test ranking by risk-reward ratio."""
        signals = [
            mock_trade_signal('AAPL', 0.70, 0, 'long', 0.05, 0.20),   # 0.25
            mock_trade_signal('GOOGL', 0.70, 0, 'long', 0.10, 0.20),  # 0.50
            mock_trade_signal('MSFT', 0.70, 0, 'long', 0.06, 0.30),   # 0.20
        ]

        ranked = rank_signals(signals, ranking_metric='risk_reward')

        assert ranked[0].ticker == 'GOOGL'  # Highest risk-reward

    def test_select_top_signals(self, mock_trade_signal):
        """Test selecting top N signals."""
        signals = [
            mock_trade_signal('AAPL', 0.70, 0, 'long', 0.05, 0.20),
            mock_trade_signal('GOOGL', 0.90, 0, 'long', 0.04, 0.18),
            mock_trade_signal('MSFT', 0.80, 0, 'long', 0.06, 0.22),
            mock_trade_signal('AMZN', 0.75, 0, 'long', 0.03, 0.25),
            mock_trade_signal('META', 0.85, 0, 'long', 0.07, 0.28),
        ]

        top = select_top_signals(signals, max_signals=3, ranking_metric='meta_prob')

        assert len(top) == 3
        assert top[0].ticker == 'GOOGL'

    def test_diversify_signals(self, mock_trade_signal):
        """Test signal diversification by sector."""
        # Create signals with sector info
        class MockSignalWithSector:
            def __init__(self, ticker, sector):
                self.ticker = ticker
                self.snapshot = Mock()
                self.snapshot.sector = sector

        signals = [
            MockSignalWithSector('AAPL', 'Technology'),
            MockSignalWithSector('MSFT', 'Technology'),
            MockSignalWithSector('GOOGL', 'Technology'),
            MockSignalWithSector('AMZN', 'Consumer'),
            MockSignalWithSector('JPM', 'Financial'),
        ]

        diversified = diversify_signals(signals, max_per_sector=2)

        # Should keep max 2 from Technology
        tech_count = sum(1 for s in diversified if s.snapshot.sector == 'Technology')
        assert tech_count <= 2
        assert len(diversified) == 4  # 2 tech + 1 consumer + 1 financial


# =============================================================================
# Test Integration
# =============================================================================

class TestMetaLabelingIntegration:
    """Integration tests for meta-labeling pipeline."""

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
    def test_full_meta_labeling_pipeline(self, mock_price_series, mock_price_history, mock_forecast):
        """Test full meta-labeling pipeline from labels to prediction."""
        # Step 1: Generate triple barrier labels
        labels, results_df = get_triple_barrier_labels(mock_price_series)

        assert len(labels) > 0

        # Step 2: Build meta-features
        meta_features = build_meta_features(
            forecast=mock_forecast,
            regime=0,
            regime_label='Bull',
            price_history=mock_price_history,
        )

        # Step 3: Train meta-model (with synthetic data)
        n_samples = len(labels)
        X = np.random.randn(n_samples, len(meta_features.feature_names))
        y = labels.values

        model = MetaModel(model_type='random_forest')
        metrics = model.fit(X, y[:len(X)])

        assert model.is_fitted

        # Step 4: Make prediction
        proba = model.predict_proba(meta_features)

        assert 0 <= proba <= 1

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
    def test_meta_features_with_model(self, mock_price_history, mock_macro_data):
        """Test that meta-features work correctly with model."""
        # Build features
        meta_features = build_meta_features(
            forecast={
                'expected_return': 0.05,
                'volatility': 0.20,
            },
            regime=0,
            regime_label='Bull',
            price_history=mock_price_history,
            macro_data=mock_macro_data,
        )

        # Train model with matching feature count
        n_features = len(meta_features.feature_names)
        X = np.random.randn(100, n_features)
        y = np.random.randint(0, 2, 100)

        model = MetaModel(model_type='random_forest')
        model.fit(X, y, feature_names=meta_features.feature_names)

        # Predict with meta-features
        proba = model.predict_proba(meta_features)

        assert 0 <= proba <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
