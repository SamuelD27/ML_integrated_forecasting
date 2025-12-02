"""Pytest configuration and fixtures for stock_analysis tests."""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import pytest

# Add parent directory to Python path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_price_data():
    """Generate deterministic mock price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    initial_price = 150.0
    returns = np.random.normal(0.0005, 0.02, 252)
    prices = initial_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 252)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 252)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 252)),
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(50000000, 150000000, 252),
    }, index=dates)
    return df


@pytest.fixture
def mock_multi_ticker_data():
    """Generate mock price data for multiple tickers."""
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    data = {}
    initial_prices = {'AAPL': 150, 'MSFT': 380, 'GOOGL': 140, 'SPY': 450}

    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)
        returns = np.random.normal(0.0005, 0.02, 252)
        prices = initial_prices[ticker] * np.exp(np.cumsum(returns))
        for field in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            data[(ticker, field)] = prices
        data[(ticker, 'Volume')] = np.random.randint(10000000, 100000000, 252)

    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker object."""
    mock_ticker = Mock()
    mock_ticker.info = {
        'symbol': 'AAPL',
        'longName': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'country': 'United States',
        'exchange': 'NMS',
        'marketCap': 2800000000000,
        'trailingPE': 28.5,
        'dividendYield': 0.005,
        'beta': 1.25,
        'averageVolume': 60000000,
    }
    mock_ticker.dividends = pd.Series([0.24, 0.24], index=pd.DatetimeIndex(['2024-02-09', '2024-05-10']))
    mock_ticker.splits = pd.Series(dtype=float)
    mock_ticker.actions = pd.DataFrame({'Dividends': [0.24], 'Stock Splits': [0.0]},
                                       index=pd.DatetimeIndex(['2024-02-09']))
    return mock_ticker


@pytest.fixture
def sample_portfolio_config():
    """Sample portfolio configuration."""
    return {
        'ticker': 'AAPL',
        'capital': 100000.0,
        'rrr': 0.6,
        'enable_ml': False,
        'ml_top_n': 10,
        'max_peers': 15,
        'target_dte': 90,
    }


@pytest.fixture
def sample_portfolio_weights():
    """Sample portfolio weights."""
    return pd.Series({'AAPL': 0.30, 'MSFT': 0.25, 'GOOGL': 0.20, 'SPY': 0.15, 'USMV': 0.10})


@pytest.fixture
def sample_returns():
    """Sample returns DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'USMV']
    return pd.DataFrame(np.random.normal(0.0005, 0.02, (252, 5)), index=dates, columns=tickers)


@pytest.fixture
def mock_peer_discovery_result():
    """Mock peer discovery result."""
    return {
        'primary_meta': {
            'symbol': 'AAPL', 'sector': 'Technology', 'industry': 'Consumer Electronics',
            'exchange': 'NMS', 'country': 'United States', 'marketCap': 2800000000000,
        },
        'peers': ['MSFT', 'GOOGL', 'META', 'NVDA'],
        'peer_metadata': {
            'MSFT': {'symbol': 'MSFT', 'sector': 'Technology', 'liquidity_score': 15.5},
            'GOOGL': {'symbol': 'GOOGL', 'sector': 'Technology', 'liquidity_score': 14.8},
        },
        'diagnostics': {'candidates_found': 20, 'passed_liquidity': 4}
    }


@pytest.fixture
def mock_etf_discovery_result():
    """Mock ETF discovery result."""
    return {
        'sector_etf': {'symbol': 'XLK', 'name': 'Technology Select Sector SPDR', 'current_price': 180.50},
        'low_vol_etf': {'symbol': 'USMV', 'name': 'iShares MSCI USA Min Vol Factor ETF', 'current_price': 78.25},
        'market_etf': {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF Trust', 'current_price': 450.00},
        'all_etfs': ['XLK', 'USMV', 'SPY'],
        'etf_metadata': {},
        'diagnostics': {'sector': 'Technology', 'total_etfs': 3}
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for test data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_reports_dir(tmp_path):
    """Temporary directory for test report files."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    return reports_dir


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_api: marks tests that require external API access")
