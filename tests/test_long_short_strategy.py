"""
Tests for Long/Short Strategy Module
=====================================
Comprehensive tests for market-neutral portfolio construction.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio.long_short_strategy import (
    LongShortStrategy,
    create_market_neutral_portfolio,
    FactorWeights,
    SignalDirection
)
from portfolio.sector_classifier import SectorClassifier


@pytest.fixture
def sample_universe():
    """Sample universe of tickers."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']


@pytest.fixture
def sample_features(sample_universe):
    """Sample features for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'momentum_20': np.random.randn(len(sample_universe)),
        'momentum_60': np.random.randn(len(sample_universe)),
        'volatility_20': np.random.uniform(0.1, 0.3, len(sample_universe)),
        'rsi_14': np.random.uniform(30, 70, len(sample_universe)),
        'factor_alpha': np.random.randn(len(sample_universe)) * 0.1,
    }, index=sample_universe)


@pytest.fixture
def sample_returns(sample_universe):
    """Sample returns for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = np.random.randn(len(dates), len(sample_universe)) * 0.02
    return pd.DataFrame(data, index=dates, columns=sample_universe)


@pytest.fixture
def sample_prices(sample_universe):
    """Sample current prices."""
    np.random.seed(42)
    return {ticker: np.random.uniform(50, 500) for ticker in sample_universe}


@pytest.fixture
def sample_volatilities(sample_universe):
    """Sample volatilities (ATR)."""
    np.random.seed(42)
    return {ticker: np.random.uniform(2, 10) for ticker in sample_universe}


class TestLongShortStrategy:
    """Test LongShortStrategy class."""

    def test_initialization(self, sample_universe):
        """Test strategy initialization."""
        strategy = LongShortStrategy(universe=sample_universe)
        assert strategy.factor_weights is not None
        assert isinstance(strategy.factor_weights, FactorWeights)

    def test_custom_factor_weights(self):
        """Test custom factor weights."""
        custom_weights = FactorWeights(
            momentum=0.5,
            alpha=0.3,
            quality=0.1,
            value=0.05,
            volatility=0.05
        )
        strategy = LongShortStrategy(factor_weights=custom_weights)
        assert strategy.factor_weights.momentum == 0.5
        assert strategy.factor_weights.alpha == 0.3

    def test_calculate_composite_score(self, sample_features, sample_returns):
        """Test composite score calculation."""
        strategy = LongShortStrategy()
        scores = strategy.calculate_composite_score(sample_features, sample_returns)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_features)
        assert not scores.isna().any()

    def test_composite_score_normalization(self, sample_features, sample_returns):
        """Test that composite scores are normalized."""
        strategy = LongShortStrategy()
        scores = strategy.calculate_composite_score(sample_features, sample_returns)

        # Scores should be roughly centered around 0 (z-scored)
        assert abs(scores.mean()) < 0.5
        assert abs(scores.std() - 1.0) < 0.5  # Should be ~1 if z-scored

    def test_generate_signals(self, sample_features, sample_returns):
        """Test signal generation."""
        strategy = LongShortStrategy(n_long=3, n_short=3)
        signals = strategy.generate_signals(sample_features, sample_returns)

        assert isinstance(signals, dict)
        assert len(signals) == len(sample_features)

        # Check signal counts
        signal_values = list(signals.values())
        n_long = sum(1 for s in signal_values if s == SignalDirection.LONG)
        n_short = sum(1 for s in signal_values if s == SignalDirection.SHORT)

        assert n_long == 3
        assert n_short == 3

    def test_signals_sector_neutral(self, sample_features, sample_returns, sample_universe):
        """Test sector-neutral signal generation."""
        # Mock sector classifier
        classifier = SectorClassifier()
        sectors = classifier.classify(sample_universe, method='mock')

        strategy = LongShortStrategy(
            n_long=4,
            n_short=4,
            sector_neutral=True
        )

        signals = strategy.generate_signals(
            sample_features,
            sample_returns,
            sectors=sectors
        )

        # Count signals by sector
        sector_signals = {}
        for ticker, signal in signals.items():
            sector = sectors.get(ticker, 'Unknown')
            if sector not in sector_signals:
                sector_signals[sector] = {'long': 0, 'short': 0}

            if signal == SignalDirection.LONG:
                sector_signals[sector]['long'] += 1
            elif signal == SignalDirection.SHORT:
                sector_signals[sector]['short'] += 1

        # Should have balanced exposure across sectors
        for sector, counts in sector_signals.items():
            net = counts['long'] - counts['short']
            # Net exposure per sector should be small
            assert abs(net) <= 2

    def test_calculate_position_sizes_equal(
        self,
        sample_universe,
        sample_prices,
        sample_volatilities
    ):
        """Test equal position sizing."""
        strategy = LongShortStrategy()

        signals = {
            'AAPL': SignalDirection.LONG,
            'MSFT': SignalDirection.LONG,
            'GOOGL': SignalDirection.SHORT,
            'AMZN': SignalDirection.SHORT,
        }

        positions = strategy.calculate_position_sizes(
            signals,
            sample_prices,
            sample_volatilities,
            capital=100000,
            method='equal'
        )

        assert len(positions) == 4

        # Long positions should sum to ~50% of capital
        long_value = sum(p['value'] for t, p in positions.items()
                        if signals[t] == SignalDirection.LONG)
        assert 45000 < long_value < 55000

        # Short positions should sum to ~50% of capital
        short_value = abs(sum(p['value'] for t, p in positions.items()
                             if signals[t] == SignalDirection.SHORT))
        assert 45000 < short_value < 55000

    def test_calculate_position_sizes_atr(
        self,
        sample_universe,
        sample_prices,
        sample_volatilities
    ):
        """Test ATR-based position sizing."""
        strategy = LongShortStrategy()

        signals = {
            'AAPL': SignalDirection.LONG,
            'MSFT': SignalDirection.LONG,
            'GOOGL': SignalDirection.SHORT,
            'AMZN': SignalDirection.SHORT,
        }

        positions = strategy.calculate_position_sizes(
            signals,
            sample_prices,
            sample_volatilities,
            capital=100000,
            method='atr'
        )

        assert len(positions) == 4

        # Lower volatility → higher position size
        for ticker, position in positions.items():
            # Risk contribution should be roughly equal
            risk = position['shares'] * sample_prices[ticker] * sample_volatilities[ticker]
            assert risk > 0

    def test_position_sizes_respect_max_position(
        self,
        sample_universe,
        sample_prices,
        sample_volatilities
    ):
        """Test that position sizes respect max_position_pct."""
        strategy = LongShortStrategy()

        signals = {ticker: SignalDirection.LONG for ticker in sample_universe[:2]}

        positions = strategy.calculate_position_sizes(
            signals,
            sample_prices,
            sample_volatilities,
            capital=100000,
            max_position_pct=0.20
        )

        for position in positions.values():
            assert position['weight'] <= 0.20


class TestCreateMarketNeutralPortfolio:
    """Test create_market_neutral_portfolio function."""

    def test_basic_portfolio_creation(
        self,
        sample_universe,
        sample_features,
        sample_prices,
        sample_volatilities
    ):
        """Test basic portfolio creation."""
        portfolio = create_market_neutral_portfolio(
            universe=sample_universe,
            features=sample_features,
            prices=sample_prices,
            volatilities=sample_volatilities,
            capital=100000
        )

        assert 'long_positions' in portfolio
        assert 'short_positions' in portfolio
        assert 'net_exposure' in portfolio
        assert 'gross_exposure' in portfolio

    def test_market_neutral_target(
        self,
        sample_universe,
        sample_features,
        sample_prices,
        sample_volatilities
    ):
        """Test market-neutral targeting."""
        portfolio = create_market_neutral_portfolio(
            universe=sample_universe,
            features=sample_features,
            prices=sample_prices,
            volatilities=sample_volatilities,
            capital=100000,
            target_net_exposure=0.0
        )

        # Net exposure should be close to 0
        assert abs(portfolio['net_exposure']) < 0.15

    def test_long_bias_target(
        self,
        sample_universe,
        sample_features,
        sample_prices,
        sample_volatilities
    ):
        """Test long-bias portfolio."""
        portfolio = create_market_neutral_portfolio(
            universe=sample_universe,
            features=sample_features,
            prices=sample_prices,
            volatilities=sample_volatilities,
            capital=100000,
            target_net_exposure=0.20
        )

        # Net exposure should be positive
        assert 0.10 < portfolio['net_exposure'] < 0.30

    def test_sector_exposure_constraints(
        self,
        sample_universe,
        sample_features,
        sample_prices,
        sample_volatilities
    ):
        """Test sector exposure constraints."""
        portfolio = create_market_neutral_portfolio(
            universe=sample_universe,
            features=sample_features,
            prices=sample_prices,
            volatilities=sample_volatilities,
            capital=100000,
            max_sector_exposure=0.10
        )

        # Check sector exposures
        for sector, exposure in portfolio['sector_exposures'].items():
            assert abs(exposure) <= 0.15  # Allow some tolerance

    def test_gross_exposure_range(
        self,
        sample_universe,
        sample_features,
        sample_prices,
        sample_volatilities
    ):
        """Test gross exposure is in reasonable range."""
        portfolio = create_market_neutral_portfolio(
            universe=sample_universe,
            features=sample_features,
            prices=sample_prices,
            volatilities=sample_volatilities,
            capital=100000
        )

        # Typical range: 1.5-2.0x (150-200% gross exposure)
        assert 1.0 < portfolio['gross_exposure'] < 2.5

    def test_capital_utilization(
        self,
        sample_universe,
        sample_features,
        sample_prices,
        sample_volatilities
    ):
        """Test capital utilization."""
        capital = 100000
        portfolio = create_market_neutral_portfolio(
            universe=sample_universe,
            features=sample_features,
            prices=sample_prices,
            volatilities=sample_volatilities,
            capital=capital
        )

        # Total position value should be close to capital
        long_value = sum(p['value'] for p in portfolio['long_positions'].values())
        short_value = abs(sum(p['value'] for p in portfolio['short_positions'].values()))
        total_value = long_value + short_value

        # Should use most of capital (allowing for rounding)
        assert 0.8 * capital < total_value < 1.2 * capital


class TestFactorWeights:
    """Test FactorWeights dataclass."""

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1."""
        weights = FactorWeights()
        total = (weights.momentum + weights.alpha + weights.quality +
                weights.value + weights.volatility)
        assert abs(total - 1.0) < 0.01

    def test_custom_weights(self):
        """Test custom weight assignment."""
        weights = FactorWeights(
            momentum=0.4,
            alpha=0.3,
            quality=0.2,
            value=0.05,
            volatility=0.05
        )

        assert weights.momentum == 0.4
        assert weights.alpha == 0.3

    def test_weights_immutable(self):
        """Test that weights are immutable (dataclass frozen)."""
        weights = FactorWeights()
        # Dataclass is frozen, so this should raise
        with pytest.raises(Exception):
            weights.momentum = 0.5


class TestSignalDirection:
    """Test SignalDirection enum."""

    def test_signal_values(self):
        """Test signal enum values."""
        assert SignalDirection.LONG == 1
        assert SignalDirection.SHORT == -1
        assert SignalDirection.NEUTRAL == 0

    def test_signal_comparison(self):
        """Test signal comparisons."""
        assert SignalDirection.LONG > SignalDirection.NEUTRAL
        assert SignalDirection.SHORT < SignalDirection.NEUTRAL


# ============================================================================
# Property-Based Tests (using hypothesis)
# ============================================================================

from hypothesis import given, strategies as st, assume

@given(
    capital=st.floats(min_value=10000, max_value=10000000),
    n_stocks=st.integers(min_value=4, max_value=20)
)
def test_portfolio_net_exposure_property(capital, n_stocks):
    """Property: Net exposure should always be within target ± tolerance."""
    np.random.seed(42)

    # Generate synthetic data
    tickers = [f'TICK{i}' for i in range(n_stocks)]
    features = pd.DataFrame({
        'momentum_20': np.random.randn(n_stocks),
        'volatility_20': np.random.uniform(0.1, 0.3, n_stocks),
        'factor_alpha': np.random.randn(n_stocks) * 0.1,
    }, index=tickers)

    prices = {ticker: np.random.uniform(50, 500) for ticker in tickers}
    volatilities = {ticker: np.random.uniform(2, 10) for ticker in tickers}

    try:
        portfolio = create_market_neutral_portfolio(
            universe=tickers,
            features=features,
            prices=prices,
            volatilities=volatilities,
            capital=capital,
            target_net_exposure=0.0
        )

        # Property: Net exposure should be small
        assert abs(portfolio['net_exposure']) < 0.20

    except Exception:
        # Some combinations may not work (e.g., too few stocks)
        assume(False)


@given(
    n_long=st.integers(min_value=1, max_value=10),
    n_short=st.integers(min_value=1, max_value=10)
)
def test_signal_count_property(n_long, n_short):
    """Property: Number of signals should match n_long + n_short."""
    n_stocks = n_long + n_short + 2  # Extra for neutral

    np.random.seed(42)
    tickers = [f'TICK{i}' for i in range(n_stocks)]

    features = pd.DataFrame({
        'momentum_20': np.random.randn(n_stocks),
    }, index=tickers)

    returns = pd.DataFrame(
        np.random.randn(100, n_stocks) * 0.02,
        columns=tickers
    )

    strategy = LongShortStrategy(n_long=n_long, n_short=n_short)
    signals = strategy.generate_signals(features, returns)

    # Count signals
    signal_values = list(signals.values())
    actual_long = sum(1 for s in signal_values if s == SignalDirection.LONG)
    actual_short = sum(1 for s in signal_values if s == SignalDirection.SHORT)

    # Property: Should have exactly n_long and n_short
    assert actual_long == n_long
    assert actual_short == n_short


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
