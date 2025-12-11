"""
Tests for Universe Builder (Phase 1)
=====================================
Tests for the pipeline universe builder module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.core_types import StockSnapshot
from pipeline.universe_builder import (
    build_universe,
    process_single_ticker,
    get_base_universe,
    universe_to_dataframe,
    fetch_ticker_data
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_ticker_info():
    """Mock Yahoo Finance info dict."""
    return {
        'currentPrice': 150.0,
        'marketCap': 2.5e12,
        'averageVolume': 50000000,
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'trailingPE': 25.0,
        'forwardPE': 22.0,
        'priceToBook': 40.0,
        'returnOnEquity': 0.45,
        'returnOnAssets': 0.20,
        'profitMargins': 0.25,
        'grossMargins': 0.43,
        'revenueGrowth': 0.08,
        'earningsGrowth': 0.10,
        'trailingEps': 6.0,
        'freeCashflow': 100e9,
        'sharesOutstanding': 16e9,
        'bookValue': 3.5,
        'dividendYield': 0.005,
    }


@pytest.fixture
def mock_price_history():
    """Mock price history DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
    prices = 100 + np.cumsum(np.random.randn(300) * 2)
    prices = np.abs(prices)  # Ensure positive

    return pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)


@pytest.fixture
def mock_ticker_data(mock_ticker_info, mock_price_history):
    """Complete mock ticker data."""
    return {
        'ticker': 'AAPL',
        'info': mock_ticker_info,
        'history': mock_price_history,
        'current_price': 150.0,
        'error': None
    }


@pytest.fixture
def sample_config():
    """Sample trading configuration."""
    return {
        'universe': {
            'source': 'custom',
            'custom_tickers': ['AAPL', 'MSFT', 'GOOGL'],
            'filters': {
                'min_upside_pct': 0.10,  # Lower threshold for testing
                'min_quality_score': -0.5,
                'min_momentum_score': -0.5,
                'min_market_cap': 1e9,
                'min_avg_volume': 100000,
                'max_universe_size': 50
            },
            'valuation': {
                'wacc': 0.10,
                'terminal_growth': 0.025,
                'projection_years': 5
            }
        }
    }


# =============================================================================
# Test Core Types
# =============================================================================

class TestStockSnapshot:
    """Tests for StockSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a basic snapshot."""
        snapshot = StockSnapshot(
            ticker='AAPL',
            date=pd.Timestamp('2024-01-15'),
            fundamentals={'pe_ratio': 25.0},
            factor_scores={'value': 0.3, 'quality': 0.5, 'momentum': 0.4},
            current_price=150.0,
            intrinsic_value=180.0,
            upside_pct=0.20
        )

        assert snapshot.ticker == 'AAPL'
        assert snapshot.current_price == 150.0
        assert snapshot.upside_pct == 0.20
        assert snapshot.is_quality_stock == True
        assert snapshot.is_momentum_stock == True

    def test_snapshot_date_conversion(self):
        """Test that date is converted to Timestamp."""
        snapshot = StockSnapshot(
            ticker='AAPL',
            date='2024-01-15',  # String instead of Timestamp
            factor_scores={}
        )

        assert isinstance(snapshot.date, pd.Timestamp)

    def test_snapshot_to_dict(self):
        """Test conversion to dictionary."""
        snapshot = StockSnapshot(
            ticker='AAPL',
            date=pd.Timestamp('2024-01-15'),
            factor_scores={'value': 0.3},
            current_price=150.0
        )

        d = snapshot.to_dict()
        assert d['ticker'] == 'AAPL'
        assert d['factor_scores'] == {'value': 0.3}
        assert isinstance(d['date'], str)  # ISO format

    def test_snapshot_from_dict(self):
        """Test creation from dictionary."""
        d = {
            'ticker': 'AAPL',
            'date': '2024-01-15T00:00:00',
            'factor_scores': {'value': 0.3},
            'fundamentals': {},
            'current_price': 150.0,
            'intrinsic_value': None,
            'upside_pct': None,
            'sector': None,
            'market_cap': None,
            'metadata': {}
        }

        snapshot = StockSnapshot.from_dict(d)
        assert snapshot.ticker == 'AAPL'
        assert isinstance(snapshot.date, pd.Timestamp)


# =============================================================================
# Test Universe Builder Functions
# =============================================================================

class TestGetBaseUniverse:
    """Tests for get_base_universe function."""

    def test_dow30_source(self):
        """Test DOW 30 returns correct tickers."""
        tickers = get_base_universe('dow30')
        assert len(tickers) == 30
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers

    def test_custom_source(self):
        """Test custom tickers are returned."""
        custom = ['AAPL', 'NVDA', 'AMD']
        tickers = get_base_universe('custom', custom)
        assert tickers == custom

    def test_unknown_source_fallback(self):
        """Test unknown source falls back to S&P 500."""
        with patch('pipeline.universe_builder.get_sp500_tickers') as mock:
            mock.return_value = ['AAPL', 'MSFT']
            tickers = get_base_universe('unknown_source')
            assert len(tickers) > 0


class TestProcessSingleTicker:
    """Tests for process_single_ticker function."""

    def test_process_valid_ticker(self, mock_ticker_data, sample_config):
        """Test processing a valid ticker."""
        # Mock the valuation and factor functions
        with patch('single_stock.valuation.compute_intrinsic_value') as mock_val, \
             patch('portfolio.factor_models.compute_factor_scores') as mock_factors:

            mock_val.return_value = {
                'intrinsic_value': 200.0,
                'upside_pct': 0.33,
                'pe_ratio': 25.0,
                'pb_ratio': 5.0,
                'ev_ebitda': 15.0,
                'valuation_method': 'dcf',
                'confidence': 0.7
            }

            mock_factors.return_value = {
                'value': 0.2,
                'quality': 0.6,
                'momentum': 0.4,
                'growth': 0.3,
                'low_volatility': 0.5,
                'composite': 0.4
            }

            snapshot = process_single_ticker(
                ticker='AAPL',
                as_of_date=pd.Timestamp('2024-01-15'),
                config=sample_config,
                ticker_data=mock_ticker_data
            )

            assert snapshot is not None
            assert snapshot.ticker == 'AAPL'
            assert snapshot.upside_pct == 0.33
            assert snapshot.factor_scores['quality'] == 0.6

    def test_process_ticker_with_error(self, sample_config):
        """Test that ticker with error returns None."""
        ticker_data = {
            'ticker': 'INVALID',
            'info': {},
            'history': pd.DataFrame(),
            'current_price': None,
            'error': 'No data available'
        }

        snapshot = process_single_ticker(
            ticker='INVALID',
            as_of_date=pd.Timestamp('2024-01-15'),
            config=sample_config,
            ticker_data=ticker_data
        )

        assert snapshot is None

    def test_process_ticker_below_market_cap(self, mock_ticker_data, sample_config):
        """Test that ticker below market cap threshold is filtered."""
        # Set market cap below threshold
        mock_ticker_data['info']['marketCap'] = 500e6  # $500M < $1B threshold

        snapshot = process_single_ticker(
            ticker='SMALL',
            as_of_date=pd.Timestamp('2024-01-15'),
            config=sample_config,
            ticker_data=mock_ticker_data
        )

        assert snapshot is None

    def test_process_ticker_below_quality_threshold(self, mock_ticker_data, sample_config):
        """Test that ticker below quality threshold is filtered."""
        with patch('single_stock.valuation.compute_intrinsic_value') as mock_val, \
             patch('portfolio.factor_models.compute_factor_scores') as mock_factors:

            mock_val.return_value = {
                'intrinsic_value': 200.0,
                'upside_pct': 0.50,  # High upside
                'pe_ratio': 25.0,
                'pb_ratio': 5.0,
                'ev_ebitda': 15.0,
                'valuation_method': 'dcf',
                'confidence': 0.7
            }

            # Quality below threshold
            mock_factors.return_value = {
                'value': 0.2,
                'quality': -0.8,  # Below -0.5 threshold
                'momentum': 0.4,
                'growth': 0.3,
                'low_volatility': 0.5,
                'composite': 0.1
            }

            snapshot = process_single_ticker(
                ticker='LOW_QUALITY',
                as_of_date=pd.Timestamp('2024-01-15'),
                config=sample_config,
                ticker_data=mock_ticker_data
            )

            assert snapshot is None


class TestBuildUniverse:
    """Tests for build_universe function."""

    def test_build_universe_empty_tickers(self, sample_config):
        """Test building universe with empty ticker list."""
        universe = build_universe(
            as_of_date=pd.Timestamp('2024-01-15'),
            config=sample_config,
            tickers=[],
            max_workers=1
        )

        assert universe == []

    def test_build_universe_handles_missing_data(self, sample_config):
        """Test that universe builder handles missing data gracefully."""
        with patch('pipeline.universe_builder.fetch_ticker_data') as mock_fetch:
            mock_fetch.return_value = {
                'ticker': 'MISSING',
                'info': {},
                'history': pd.DataFrame(),
                'current_price': None,
                'error': 'No data'
            }

            universe = build_universe(
                as_of_date=pd.Timestamp('2024-01-15'),
                config=sample_config,
                tickers=['MISSING'],
                max_workers=1
            )

            # Should return empty list, not crash
            assert universe == []

    def test_build_universe_respects_max_size(self, sample_config):
        """Test that universe is limited to max_universe_size."""
        sample_config['universe']['filters']['max_universe_size'] = 5

        # Create a mock that returns valid data
        def mock_process(ticker, as_of_date, config, ticker_data):
            return StockSnapshot(
                ticker=ticker,
                date=as_of_date,
                factor_scores={'composite': np.random.random()},
                current_price=100.0,
                upside_pct=0.30
            )

        with patch('pipeline.universe_builder.fetch_ticker_data') as mock_fetch, \
             patch('pipeline.universe_builder.process_single_ticker', side_effect=mock_process):

            mock_fetch.return_value = {
                'ticker': 'TEST',
                'info': {'marketCap': 100e9},
                'history': pd.DataFrame(),
                'current_price': 100.0,
                'error': None
            }

            universe = build_universe(
                as_of_date=pd.Timestamp('2024-01-15'),
                config=sample_config,
                tickers=[f'TEST{i}' for i in range(20)],  # 20 tickers
                max_workers=1
            )

            assert len(universe) <= 5


class TestUniverseToDataframe:
    """Tests for universe_to_dataframe function."""

    def test_convert_to_dataframe(self):
        """Test converting universe list to DataFrame."""
        universe = [
            StockSnapshot(
                ticker='AAPL',
                date=pd.Timestamp('2024-01-15'),
                fundamentals={'pe_ratio': 25.0, 'roe': 0.45},
                factor_scores={'value': 0.3, 'quality': 0.5},
                current_price=150.0,
                intrinsic_value=180.0,
                upside_pct=0.20,
                sector='Technology',
                market_cap=2.5e12
            ),
            StockSnapshot(
                ticker='MSFT',
                date=pd.Timestamp('2024-01-15'),
                fundamentals={'pe_ratio': 30.0, 'roe': 0.35},
                factor_scores={'value': 0.2, 'quality': 0.6},
                current_price=380.0,
                intrinsic_value=420.0,
                upside_pct=0.10,
                sector='Technology',
                market_cap=2.8e12
            )
        ]

        df = universe_to_dataframe(universe)

        assert len(df) == 2
        assert 'ticker' in df.columns
        assert 'factor_value' in df.columns
        assert 'fund_pe_ratio' in df.columns
        assert df.loc[0, 'ticker'] == 'AAPL'

    def test_empty_universe(self):
        """Test converting empty universe."""
        df = universe_to_dataframe([])
        assert len(df) == 0


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestUniverseBuilderIntegration:
    """Integration tests for universe builder (requires API)."""

    @pytest.mark.skip(reason="Requires API access")
    def test_build_small_universe(self, sample_config):
        """Test building a small universe with real data."""
        universe = build_universe(
            as_of_date=pd.Timestamp.now().normalize(),
            config=sample_config,
            tickers=['AAPL', 'MSFT'],  # Small list
            max_workers=2
        )

        # Should process without errors
        # May or may not have results depending on filters
        assert isinstance(universe, list)


# =============================================================================
# Filter Tests
# =============================================================================

class TestFiltering:
    """Tests for filtering logic."""

    def test_filter_by_upside(self, mock_ticker_data, sample_config):
        """Test filtering by upside percentage."""
        # Set a high min_upside threshold
        sample_config['universe']['filters']['min_upside_pct'] = 0.50

        with patch('single_stock.valuation.compute_intrinsic_value') as mock_val, \
             patch('portfolio.factor_models.compute_factor_scores') as mock_factors:

            # Upside below 50%
            mock_val.return_value = {
                'intrinsic_value': 165.0,
                'upside_pct': 0.10,  # 10% < 50%
                'pe_ratio': 25.0,
                'pb_ratio': 5.0,
                'ev_ebitda': 15.0,
                'valuation_method': 'dcf',
                'confidence': 0.7
            }

            mock_factors.return_value = {
                'value': 0.5,
                'quality': 0.5,
                'momentum': 0.5,
                'growth': 0.5,
                'low_volatility': 0.5,
                'composite': 0.5
            }

            snapshot = process_single_ticker(
                ticker='LOW_UPSIDE',
                as_of_date=pd.Timestamp('2024-01-15'),
                config=sample_config,
                ticker_data=mock_ticker_data
            )

            assert snapshot is None

    def test_filter_by_momentum(self, mock_ticker_data, sample_config):
        """Test filtering by momentum score."""
        sample_config['universe']['filters']['min_momentum_score'] = 0.3

        with patch('single_stock.valuation.compute_intrinsic_value') as mock_val, \
             patch('portfolio.factor_models.compute_factor_scores') as mock_factors:

            mock_val.return_value = {
                'intrinsic_value': 200.0,
                'upside_pct': 0.50,
                'pe_ratio': 25.0,
                'pb_ratio': 5.0,
                'ev_ebitda': 15.0,
                'valuation_method': 'dcf',
                'confidence': 0.7
            }

            # Momentum below threshold
            mock_factors.return_value = {
                'value': 0.5,
                'quality': 0.5,
                'momentum': 0.1,  # Below 0.3 threshold
                'growth': 0.5,
                'low_volatility': 0.5,
                'composite': 0.4
            }

            snapshot = process_single_ticker(
                ticker='LOW_MOMENTUM',
                as_of_date=pd.Timestamp('2024-01-15'),
                config=sample_config,
                ticker_data=mock_ticker_data
            )

            assert snapshot is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
