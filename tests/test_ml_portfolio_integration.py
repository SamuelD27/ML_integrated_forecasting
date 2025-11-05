"""
Integration Tests: ML-Portfolio Bridge
======================================
Tests the end-to-end pipeline from ML forecasts to portfolio optimization.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from integration.ml_portfolio_bridge import (
    MLPortfolioBridge,
    create_bridge,
    ForecastHorizon,
    OptimizerType,
    ForecastView,
    PortfolioAllocation
)
from integration.signal_generator import (
    SignalGenerator,
    create_signal_generator,
    TradingSignal,
    SignalType
)


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample historical price data."""
    np.random.seed(42)

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        freq='D'
    )

    data = {}
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    for ticker in tickers:
        # Generate realistic price data
        returns = np.random.randn(len(dates)) * 0.02  # 2% daily vol
        price = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': price * (1 + np.random.randn(len(dates)) * 0.01),
            'high': price * (1 + abs(np.random.randn(len(dates))) * 0.02),
            'low': price * (1 - abs(np.random.randn(len(dates))) * 0.02),
            'close': price,
            'volume': np.random.randint(1e6, 1e8, len(dates))
        }, index=dates)

        data[ticker] = df

    return data


@pytest.fixture
def ml_bridge():
    """Create ML-Portfolio bridge with ensemble model."""
    return create_bridge(
        model_type="ensemble",
        optimizer_type="black_litterman",
        risk_aversion=2.5
    )


@pytest.fixture
def signal_generator():
    """Create signal generator."""
    return create_signal_generator(strategy="moderate")


# Tests
class TestMLPortfolioBridge:
    """Test ML-Portfolio Bridge functionality."""

    def test_bridge_initialization(self):
        """Test bridge can be initialized with different configs."""
        # Ensemble + Black-Litterman
        bridge1 = create_bridge(
            model_type="ensemble",
            optimizer_type="black_litterman"
        )
        assert bridge1.model_type == "ensemble"
        assert bridge1.optimizer_type == OptimizerType.BLACK_LITTERMAN

        # Ensemble + Mean-Variance
        bridge2 = create_bridge(
            model_type="ensemble",
            optimizer_type="mean_variance"
        )
        assert bridge2.optimizer_type == OptimizerType.MEAN_VARIANCE

    def test_model_loading(self, ml_bridge):
        """Test ML model can be loaded."""
        assert ml_bridge.model is None  # Not loaded yet

        ml_bridge.load_model()

        assert ml_bridge.model is not None
        assert ml_bridge.model_type == "ensemble"

    def test_optimizer_loading(self, ml_bridge):
        """Test portfolio optimizer can be loaded."""
        assert ml_bridge.optimizer is None  # Not loaded yet

        ml_bridge.load_optimizer()

        assert ml_bridge.optimizer is not None

    def test_generate_forecast_view(self, ml_bridge, sample_data):
        """Test single stock forecast generation."""
        ticker = 'AAPL'
        historical_data = sample_data[ticker]

        view = ml_bridge.generate_forecast_view(
            ticker=ticker,
            historical_data=historical_data,
            horizon=ForecastHorizon.MEDIUM_TERM
        )

        assert isinstance(view, ForecastView)
        assert view.ticker == ticker
        assert isinstance(view.expected_return, float)
        assert 0 <= view.confidence <= 1
        assert view.forecast_price > 0
        assert view.current_price > 0

    def test_generate_views_for_universe(self, ml_bridge, sample_data):
        """Test forecast generation for multiple stocks."""
        tickers = list(sample_data.keys())

        views = ml_bridge.generate_views_for_universe(
            tickers=tickers,
            historical_data=sample_data,
            horizon=ForecastHorizon.MEDIUM_TERM,
            min_confidence=0.0  # Accept all views
        )

        assert len(views) == len(tickers)
        assert all(isinstance(v, ForecastView) for v in views)
        assert all(v.ticker in tickers for v in views)

    def test_generate_allocation(self, ml_bridge, sample_data):
        """Test end-to-end portfolio allocation."""
        tickers = list(sample_data.keys())

        allocation = ml_bridge.generate_allocation(
            tickers=tickers,
            historical_data=sample_data,
            horizon=ForecastHorizon.MEDIUM_TERM,
            min_confidence=0.0
        )

        assert isinstance(allocation, PortfolioAllocation)

        # Check weights
        assert len(allocation.weights) > 0
        assert all(ticker in allocation.weights for ticker in tickers)
        assert abs(sum(allocation.weights.values()) - 1.0) < 1e-6  # Sum to 1

        # Check metrics
        assert isinstance(allocation.expected_return, float)
        assert isinstance(allocation.expected_volatility, float)
        assert isinstance(allocation.sharpe_ratio, float)
        assert allocation.expected_volatility >= 0

        # Check views
        assert len(allocation.views) > 0
        assert all(isinstance(v, ForecastView) for v in allocation.views)

    def test_allocation_with_constraints(self, ml_bridge, sample_data):
        """Test portfolio optimization with constraints."""
        tickers = list(sample_data.keys())

        constraints = {
            'long_only': True,
            'max_weight': 0.5,
            'min_weight': 0.1
        }

        allocation = ml_bridge.generate_allocation(
            tickers=tickers,
            historical_data=sample_data,
            constraints=constraints
        )

        # Verify constraints
        for weight in allocation.weights.values():
            assert weight >= 0  # Long only
            assert weight <= 0.5 + 1e-6  # Max weight
            assert weight >= 0.1 - 1e-6 or weight == 0  # Min weight (if nonzero)

    def test_different_horizons(self, ml_bridge, sample_data):
        """Test forecasts with different time horizons."""
        ticker = 'AAPL'
        historical_data = sample_data[ticker]

        horizons = [
            ForecastHorizon.SHORT_TERM,
            ForecastHorizon.MEDIUM_TERM,
            ForecastHorizon.LONG_TERM
        ]

        views = []
        for horizon in horizons:
            view = ml_bridge.generate_forecast_view(
                ticker=ticker,
                historical_data=historical_data,
                horizon=horizon
            )
            views.append(view)

        assert len(views) == 3
        assert all(isinstance(v, ForecastView) for v in views)


class TestSignalGenerator:
    """Test Signal Generator functionality."""

    def test_generator_initialization(self):
        """Test signal generator can be initialized with strategies."""
        strategies = ['conservative', 'moderate', 'aggressive']

        for strategy in strategies:
            gen = create_signal_generator(strategy=strategy)
            assert isinstance(gen, SignalGenerator)

    def test_generate_signal_buy(self, signal_generator):
        """Test BUY signal generation."""
        signal = signal_generator.generate_signal(
            ticker="AAPL",
            forecast_return=0.15,  # 15% expected return
            confidence=0.8,
            forecast_price=180.0,
            current_price=150.0,
            downside_risk=-0.05
        )

        assert isinstance(signal, TradingSignal)
        assert signal.ticker == "AAPL"
        assert signal.action in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.strength > 0
        assert 0 <= signal.confidence <= 1

    def test_generate_signal_sell(self, signal_generator):
        """Test SELL signal generation."""
        signal = signal_generator.generate_signal(
            ticker="MSFT",
            forecast_return=-0.12,  # -12% expected return
            confidence=0.75,
            forecast_price=280.0,
            current_price=320.0,
            downside_risk=-0.15
        )

        assert signal.action in [SignalType.SELL, SignalType.STRONG_SELL]
        assert signal.strength > 0

    def test_generate_signal_hold(self, signal_generator):
        """Test HOLD signal generation."""
        # Low expected return
        signal1 = signal_generator.generate_signal(
            ticker="GOOGL",
            forecast_return=0.02,  # Only 2% expected return
            confidence=0.8,
            forecast_price=152.0,
            current_price=150.0
        )
        assert signal1.action == SignalType.HOLD

        # Low confidence
        signal2 = signal_generator.generate_signal(
            ticker="GOOGL",
            forecast_return=0.15,
            confidence=0.3,  # Low confidence
            forecast_price=180.0,
            current_price=150.0
        )
        assert signal2.action == SignalType.HOLD

    def test_signal_is_actionable(self, signal_generator):
        """Test actionable signal detection."""
        # Actionable signal
        signal1 = signal_generator.generate_signal(
            ticker="AAPL",
            forecast_return=0.15,
            confidence=0.8,
            forecast_price=180.0,
            current_price=150.0
        )
        assert signal1.is_actionable

        # Non-actionable signal
        signal2 = signal_generator.generate_signal(
            ticker="MSFT",
            forecast_return=0.02,
            confidence=0.8,
            forecast_price=152.0,
            current_price=150.0
        )
        assert not signal2.is_actionable

    def test_filter_signals(self, signal_generator):
        """Test signal filtering."""
        signals = [
            TradingSignal(
                ticker="AAPL",
                action=SignalType.BUY,
                strength=75,
                confidence=0.8,
                forecast_return=0.15,
                forecast_price=180.0,
                current_price=150.0,
                risk_reward_ratio=3.0,
                timestamp=datetime.now(),
                metadata={}
            ),
            TradingSignal(
                ticker="MSFT",
                action=SignalType.HOLD,
                strength=20,
                confidence=0.5,
                forecast_return=0.02,
                forecast_price=152.0,
                current_price=150.0,
                risk_reward_ratio=0.5,
                timestamp=datetime.now(),
                metadata={}
            )
        ]

        # Filter by strength
        filtered = signal_generator.filter_signals(signals, min_strength=50)
        assert len(filtered) == 1
        assert filtered[0].ticker == "AAPL"

        # Filter by action
        filtered = signal_generator.filter_signals(
            signals,
            action_types=[SignalType.BUY, SignalType.STRONG_BUY]
        )
        assert len(filtered) == 1
        assert filtered[0].action == SignalType.BUY

    def test_rank_signals(self, signal_generator):
        """Test signal ranking."""
        signals = [
            TradingSignal(
                ticker="AAPL",
                action=SignalType.BUY,
                strength=75,
                confidence=0.8,
                forecast_return=0.15,
                forecast_price=180.0,
                current_price=150.0,
                risk_reward_ratio=3.0,
                timestamp=datetime.now(),
                metadata={}
            ),
            TradingSignal(
                ticker="MSFT",
                action=SignalType.BUY,
                strength=60,
                confidence=0.7,
                forecast_return=0.12,
                forecast_price=336.0,
                current_price=300.0,
                risk_reward_ratio=2.5,
                timestamp=datetime.now(),
                metadata={}
            )
        ]

        # Rank by strength
        ranked = signal_generator.rank_signals(signals, by="strength")
        assert ranked[0].ticker == "AAPL"  # Higher strength

        # Rank by confidence
        ranked = signal_generator.rank_signals(signals, by="confidence")
        assert ranked[0].ticker == "AAPL"  # Higher confidence

    def test_to_dataframe(self, signal_generator):
        """Test signal conversion to DataFrame."""
        signals = [
            TradingSignal(
                ticker="AAPL",
                action=SignalType.BUY,
                strength=75,
                confidence=0.8,
                forecast_return=0.15,
                forecast_price=180.0,
                current_price=150.0,
                risk_reward_ratio=3.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]

        df = signal_generator.to_dataframe(signals)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'ticker' in df.columns
        assert 'action' in df.columns
        assert 'strength' in df.columns
        assert df.iloc[0]['ticker'] == 'AAPL'


class TestEndToEndIntegration:
    """Test complete end-to-end workflow."""

    def test_full_pipeline(self, sample_data):
        """Test complete ML → Portfolio → Signals pipeline."""
        # Step 1: Generate portfolio allocation
        bridge = create_bridge(
            model_type="ensemble",
            optimizer_type="black_litterman"
        )

        allocation = bridge.generate_allocation(
            tickers=list(sample_data.keys()),
            historical_data=sample_data,
            horizon=ForecastHorizon.MEDIUM_TERM
        )

        assert len(allocation.weights) > 0
        assert len(allocation.views) > 0

        # Step 2: Generate trading signals from views
        signal_gen = create_signal_generator(strategy="moderate")

        current_prices = {
            ticker: data['close'].iloc[-1]
            for ticker, data in sample_data.items()
        }

        signals = []
        for view in allocation.views:
            signal = signal_gen.generate_signal(
                ticker=view.ticker,
                forecast_return=view.expected_return,
                confidence=view.confidence,
                forecast_price=view.forecast_price,
                current_price=view.current_price,
                downside_risk=view.downside_risk
            )
            signals.append(signal)

        assert len(signals) > 0
        assert all(isinstance(s, TradingSignal) for s in signals)

        # Step 3: Filter to actionable signals
        actionable = [s for s in signals if s.is_actionable]

        # Verify we got some actionable signals (probabilistic, so may be 0)
        print(f"Generated {len(signals)} signals, {len(actionable)} actionable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
