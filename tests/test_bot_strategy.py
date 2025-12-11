"""
Unit Tests for Bot Strategy and Sizing Logic
=============================================
Tests for TradingStrategy class covering:
- Rebalance threshold logic
- Trade calculation (sizing)
- Regime detection
- Feature scoring
- Market hours check
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy import TradingStrategy, StrategyConfig, PortfolioSignal


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StrategyConfig()
        assert config.symbols == ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        assert config.rrr == 0.5
        assert config.max_position_pct == 0.25
        assert config.rebalance_threshold == 0.05
        assert config.enable_ml is False
        assert config.enable_regime_detection is True
        assert config.enable_feature_scoring is True

    def test_config_from_env(self):
        """Test loading config from environment variables."""
        with patch.dict('os.environ', {
            'TRADING_SYMBOLS': 'AAPL,TSLA',
            'TRADING_RRR': '0.7',
            'RISK_MAX_POSITION_PCT': '0.15',
            'TRADING_REBALANCE_THRESHOLD': '0.03',
            'ENABLE_ML': 'true',
        }):
            config = StrategyConfig.from_env()
            assert config.symbols == ['AAPL', 'TSLA']
            assert config.rrr == 0.7
            assert config.max_position_pct == 0.15
            assert config.rebalance_threshold == 0.03
            assert config.enable_ml is True


class TestRebalanceLogic:
    """Test rebalance threshold logic."""

    def test_no_target_weights_triggers_rebalance(self):
        """Should rebalance when no target weights are set."""
        strategy = TradingStrategy(StrategyConfig(
            symbols=['AAPL', 'MSFT'],
            rebalance_interval_hours=0,  # Allow immediate rebalance
        ))
        strategy._current_weights = {}  # No targets set

        should_rebal, reason = strategy.should_rebalance(
            current_positions={'AAPL': 50000},
            portfolio_value=100000
        )
        assert should_rebal is True
        assert "No target weights" in reason

    def test_drift_above_threshold_triggers_rebalance(self):
        """Should rebalance when drift exceeds threshold."""
        strategy = TradingStrategy(StrategyConfig(
            symbols=['AAPL', 'MSFT'],
            rebalance_threshold=0.05,  # 5% threshold
            rebalance_interval_hours=0,
        ))
        strategy._current_weights = {'AAPL': 0.50, 'MSFT': 0.50}  # Target: 50/50

        # Current: AAPL at 60% (10% drift)
        should_rebal, reason = strategy.should_rebalance(
            current_positions={'AAPL': 60000, 'MSFT': 40000},
            portfolio_value=100000
        )
        assert should_rebal is True
        assert "AAPL" in reason
        assert "drifted" in reason

    def test_drift_within_threshold_no_rebalance(self):
        """Should not rebalance when drift is within threshold."""
        strategy = TradingStrategy(StrategyConfig(
            symbols=['AAPL', 'MSFT'],
            rebalance_threshold=0.05,  # 5% threshold
            rebalance_interval_hours=0,
        ))
        strategy._current_weights = {'AAPL': 0.50, 'MSFT': 0.50}

        # Current: AAPL at 52% (2% drift, within threshold)
        should_rebal, reason = strategy.should_rebalance(
            current_positions={'AAPL': 52000, 'MSFT': 48000},
            portfolio_value=100000
        )
        assert should_rebal is False
        assert "within threshold" in reason

    def test_time_restriction_prevents_rebalance(self):
        """Should not rebalance if recently rebalanced."""
        strategy = TradingStrategy(StrategyConfig(
            symbols=['AAPL', 'MSFT'],
            rebalance_interval_hours=24,
        ))
        strategy._current_weights = {'AAPL': 0.50, 'MSFT': 0.50}
        strategy._last_rebalance = datetime.now() - timedelta(hours=1)  # 1 hour ago

        # Large drift but too soon
        should_rebal, reason = strategy.should_rebalance(
            current_positions={'AAPL': 80000, 'MSFT': 20000},
            portfolio_value=100000
        )
        assert should_rebal is False
        assert "since last rebalance" in reason


class TestTradeCalculation:
    """Test trade size calculation."""

    def test_buy_trade_calculation(self):
        """Test buying new position calculation."""
        strategy = TradingStrategy(StrategyConfig(symbols=['AAPL', 'MSFT']))

        # Create mock price data
        prices_df = pd.DataFrame({
            'AAPL': [150.0] * 100,
            'MSFT': [300.0] * 100,
        })

        trades = strategy._calculate_trades(
            target_weights={'AAPL': 0.50, 'MSFT': 0.50},
            current_weights={},  # No current positions
            portfolio_value=100000,
            prices_df=prices_df,
        )

        assert len(trades) == 2
        assert all(t['side'] == 'buy' for t in trades)

        aapl_trade = next(t for t in trades if t['symbol'] == 'AAPL')
        msft_trade = next(t for t in trades if t['symbol'] == 'MSFT')

        # AAPL: $50k target / $150 = 333 shares
        assert aapl_trade['qty'] == 333
        # MSFT: $50k target / $300 = 166 shares
        assert msft_trade['qty'] == 166

    def test_sell_trade_calculation(self):
        """Test selling position calculation."""
        strategy = TradingStrategy(StrategyConfig(symbols=['AAPL']))

        prices_df = pd.DataFrame({
            'AAPL': [150.0] * 100,
        })

        trades = strategy._calculate_trades(
            target_weights={'AAPL': 0.25},  # Reduce to 25%
            current_weights={'AAPL': 0.50},  # Currently 50%
            portfolio_value=100000,
            prices_df=prices_df,
        )

        assert len(trades) == 1
        assert trades[0]['side'] == 'sell'
        assert trades[0]['symbol'] == 'AAPL'
        # Sell $25k / $150 = 166 shares
        assert trades[0]['qty'] == 166

    def test_live_quotes_override_historical(self):
        """Test that live quotes are used when available."""
        strategy = TradingStrategy(StrategyConfig(symbols=['AAPL']))

        prices_df = pd.DataFrame({
            'AAPL': [100.0] * 100,  # Old price: $100
        })

        live_quotes = {
            'AAPL': {'bid': 149.0, 'ask': 151.0, 'mid': 150.0}  # Live: $150
        }

        trades = strategy._calculate_trades(
            target_weights={'AAPL': 0.50},
            current_weights={},
            portfolio_value=100000,
            prices_df=prices_df,
            live_quotes=live_quotes,
        )

        # Should use $150 (live), not $100 (historical)
        # $50k target / $150 = 333 shares
        assert trades[0]['qty'] == 333

    def test_small_trades_filtered(self):
        """Test that trades < $100 are filtered out."""
        strategy = TradingStrategy(StrategyConfig(symbols=['AAPL']))

        prices_df = pd.DataFrame({
            'AAPL': [150.0] * 100,
        })

        trades = strategy._calculate_trades(
            target_weights={'AAPL': 0.501},  # 0.1% difference
            current_weights={'AAPL': 0.500},
            portfolio_value=100000,
            prices_df=prices_df,
        )

        # $100 difference is too small
        assert len(trades) == 0

    def test_sells_sorted_before_buys(self):
        """Test that sell orders come before buy orders."""
        strategy = TradingStrategy(StrategyConfig(symbols=['AAPL', 'MSFT']))

        prices_df = pd.DataFrame({
            'AAPL': [150.0] * 100,
            'MSFT': [300.0] * 100,
        })

        trades = strategy._calculate_trades(
            target_weights={'MSFT': 0.60},  # Buy MSFT
            current_weights={'AAPL': 0.60},  # Sell AAPL
            portfolio_value=100000,
            prices_df=prices_df,
        )

        # Sells should come first to free up cash
        assert trades[0]['side'] == 'sell'
        assert trades[0]['symbol'] == 'AAPL'
        assert trades[1]['side'] == 'buy'
        assert trades[1]['symbol'] == 'MSFT'


class TestFeatureScoring:
    """Test feature-based scoring."""

    def test_feature_scores_calculation(self):
        """Test that feature scores are calculated correctly."""
        strategy = TradingStrategy(StrategyConfig(
            symbols=['AAPL', 'MSFT'],
            enable_feature_scoring=True,
        ))

        # Create price data with different characteristics
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # AAPL: uptrend (good momentum)
        aapl_prices = 150 * (1 + np.cumsum(np.random.randn(100) * 0.01 + 0.001))
        # MSFT: sideways/volatile (lower momentum)
        msft_prices = 300 * (1 + np.cumsum(np.random.randn(100) * 0.02))

        prices_df = pd.DataFrame({
            'AAPL': aapl_prices,
            'MSFT': msft_prices,
        }, index=dates)

        scores = strategy._calculate_feature_scores(prices_df)

        assert scores is not None
        assert 'AAPL' in scores.index
        assert 'MSFT' in scores.index
        # AAPL should have higher score due to momentum
        assert scores['AAPL'] > scores['MSFT']


class TestMarketHours:
    """Test market hours checking."""

    def test_market_open_during_trading_hours(self):
        """Test that market is detected as open during trading hours."""
        from bot.main import is_market_open
        import pytz

        ny_tz = pytz.timezone('America/New_York')

        # Mock a Wednesday at 10:00 AM ET
        mock_time = datetime(2024, 1, 3, 10, 0, 0, tzinfo=ny_tz)  # Wednesday

        with patch('bot.main.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time

            is_open, status = is_market_open()
            assert is_open is True
            assert "Market open" in status

    def test_market_closed_on_weekend(self):
        """Test that market is detected as closed on weekend."""
        from bot.main import is_market_open
        import pytz

        ny_tz = pytz.timezone('America/New_York')

        # Mock a Saturday at 10:00 AM ET
        mock_time = datetime(2024, 1, 6, 10, 0, 0, tzinfo=ny_tz)  # Saturday

        with patch('bot.main.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time

            is_open, status = is_market_open()
            assert is_open is False
            assert "Weekend" in status

    def test_market_closed_premarket(self):
        """Test that market is detected as closed before open."""
        from bot.main import is_market_open
        import pytz

        ny_tz = pytz.timezone('America/New_York')

        # Mock a Wednesday at 8:00 AM ET (before 9:30 open)
        mock_time = datetime(2024, 1, 3, 8, 0, 0, tzinfo=ny_tz)

        with patch('bot.main.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time

            is_open, status = is_market_open()
            assert is_open is False
            assert "Pre-market" in status


class TestPriceCaching:
    """Test price data caching."""

    def test_cache_used_when_valid(self):
        """Test that cache is used when still valid."""
        strategy = TradingStrategy(StrategyConfig(symbols=['AAPL']))

        # Set up cache
        cached_prices = pd.DataFrame({'AAPL': [150.0] * 100})
        strategy._price_cache = cached_prices
        strategy._price_cache_time = datetime.now() - timedelta(minutes=30)  # 30 min old

        # Should return cached data without fetching
        with patch('yfinance.download') as mock_download:
            prices = strategy._fetch_prices()
            mock_download.assert_not_called()
            assert prices is cached_prices

    def test_cache_refreshed_when_expired(self):
        """Test that cache is refreshed when expired."""
        strategy = TradingStrategy(StrategyConfig(symbols=['AAPL']))

        # Set up expired cache
        strategy._price_cache = pd.DataFrame({'AAPL': [150.0] * 100})
        strategy._price_cache_time = datetime.now() - timedelta(hours=2)  # 2 hours old (expired)

        # Should fetch new data
        with patch('yfinance.download') as mock_download:
            mock_data = MagicMock()
            mock_data.columns = pd.Index(['Adj Close'])
            mock_data.__getitem__ = lambda self, key: pd.DataFrame({'AAPL': [160.0] * 100})
            mock_download.return_value = mock_data

            prices = strategy._fetch_prices()
            mock_download.assert_called_once()

    def test_force_refresh_bypasses_cache(self):
        """Test that force_refresh bypasses valid cache."""
        strategy = TradingStrategy(StrategyConfig(symbols=['AAPL']))

        # Set up valid cache
        strategy._price_cache = pd.DataFrame({'AAPL': [150.0] * 100})
        strategy._price_cache_time = datetime.now()

        with patch('yfinance.download') as mock_download:
            mock_data = MagicMock()
            mock_data.columns = pd.Index(['Adj Close'])
            mock_data.__getitem__ = lambda self, key: pd.DataFrame({'AAPL': [160.0] * 100})
            mock_download.return_value = mock_data

            prices = strategy._fetch_prices(force_refresh=True)
            mock_download.assert_called_once()


class TestPositionReconciliation:
    """Test position reconciliation logic."""

    def test_reconcile_matching_positions(self):
        """Test reconciliation when positions match."""
        import tempfile
        from bot.trade_store import TradeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeStore(Path(tmpdir) / 'test.db')

            # Add position to DB
            store.track_position_open(
                symbol='AAPL',
                side='long',
                qty=100,
                entry_price=150.0,
                entry_order_id='test-123',
            )

            # Reconcile with matching Alpaca position
            alpaca_positions = [
                {'symbol': 'AAPL', 'qty': 100, 'avg_entry_price': 150.0, 'market_value': 15000},
            ]

            result = store.reconcile_positions(alpaca_positions)

            assert result['needs_sync'] is False
            assert 'AAPL' in result['matched']
            assert len(result['missing_in_db']) == 0
            assert len(result['missing_in_alpaca']) == 0

    def test_reconcile_missing_in_db(self):
        """Test reconciliation when position exists in Alpaca but not DB."""
        import tempfile
        from bot.trade_store import TradeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeStore(Path(tmpdir) / 'test.db')

            # Alpaca has position, DB is empty
            alpaca_positions = [
                {'symbol': 'AAPL', 'qty': 100, 'avg_entry_price': 150.0, 'market_value': 15000},
            ]

            result = store.reconcile_positions(alpaca_positions)

            assert result['needs_sync'] is True
            assert len(result['missing_in_db']) == 1
            assert result['missing_in_db'][0]['symbol'] == 'AAPL'

    def test_reconcile_missing_in_alpaca(self):
        """Test reconciliation when position exists in DB but not Alpaca."""
        import tempfile
        from bot.trade_store import TradeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeStore(Path(tmpdir) / 'test.db')

            # Add position to DB
            store.track_position_open(
                symbol='AAPL',
                side='long',
                qty=100,
                entry_price=150.0,
                entry_order_id='test-123',
            )

            # Alpaca has no positions
            alpaca_positions = []

            result = store.reconcile_positions(alpaca_positions)

            assert result['needs_sync'] is True
            assert len(result['missing_in_alpaca']) == 1
            assert result['missing_in_alpaca'][0]['symbol'] == 'AAPL'

    def test_sync_from_alpaca(self):
        """Test syncing positions from Alpaca."""
        import tempfile
        from bot.trade_store import TradeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TradeStore(Path(tmpdir) / 'test.db')

            # Sync positions from Alpaca
            alpaca_positions = [
                {'symbol': 'AAPL', 'qty': 100, 'avg_entry_price': 150.0, 'market_value': 15000},
                {'symbol': 'MSFT', 'qty': 50, 'avg_entry_price': 300.0, 'market_value': 15000},
            ]

            result = store.sync_from_alpaca(alpaca_positions)

            assert result['action'] == 'synced'
            assert 'AAPL' in result['added']
            assert 'MSFT' in result['added']

            # Verify positions in DB
            db_positions = store.get_open_positions()
            assert len(db_positions) == 2


class TestKillSwitchConfig:
    """Test kill switch configuration."""

    def test_kill_switch_defaults(self):
        """Test default kill switch configuration."""
        from bot.config import RiskConfig

        config = RiskConfig()

        assert config.kill_switch_enabled is True
        assert config.kill_switch_daily_loss_pct == 0.03  # 3%
        assert config.kill_switch_max_drawdown_pct == 0.10  # 10%
        assert config.position_reconcile_on_start is True

    def test_kill_switch_from_env(self):
        """Test kill switch configuration from environment."""
        from bot.config import RiskConfig

        with patch.dict('os.environ', {
            'RISK_KILL_DAILY_LOSS_PCT': '0.05',
            'RISK_KILL_MAX_DRAWDOWN_PCT': '0.15',
            'RISK_KILL_SWITCH_ENABLED': 'false',
        }):
            config = RiskConfig.from_env()

            assert config.kill_switch_daily_loss_pct == 0.05
            assert config.kill_switch_max_drawdown_pct == 0.15
            assert config.kill_switch_enabled is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
