#!/usr/bin/env python3
"""
Trading Bot Main Entry Point
=============================
Production-ready trading bot with:
- Alpaca paper trading
- Robust error handling and reconnection
- Trade logging to SQLite
- Optional Telegram notifications

Usage:
    python -m bot.main
    python bot/main.py

Environment Variables:
    See .env.example for required configuration.
"""

from __future__ import annotations

import os
import sys
import time
import signal
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure we can import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from bot.config import load_config, BotConfig
from bot.logging_config import setup_logging, get_logger
from bot.trader import EnhancedTrader
from bot.trade_store import TradeStore, EquitySnapshot
from bot.strategy import TradingStrategy, StrategyConfig

import pytz

logger = get_logger(__name__)


def is_market_open() -> tuple[bool, str]:
    """
    Check if US stock market is currently open.

    Returns:
        Tuple of (is_open: bool, reason: str)
    """
    try:
        ny_tz = pytz.timezone('America/New_York')
        now = datetime.now(ny_tz)

        # Check weekday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:
            return False, f"Weekend ({now.strftime('%A')})"

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now < market_open:
            return False, f"Pre-market ({now.strftime('%H:%M')} ET, opens 9:30)"
        if now >= market_close:
            return False, f"After-hours ({now.strftime('%H:%M')} ET, closed at 16:00)"

        return True, f"Market open ({now.strftime('%H:%M')} ET)"

    except Exception as e:
        logger.warning(f"Market hours check failed: {e}, assuming open")
        return True, "Unknown (assuming open)"


class TradingBot:
    """
    Main trading bot orchestrator.

    Manages:
    - Connection to Alpaca
    - Trade execution
    - Position tracking
    - Equity snapshots
    - Telegram notifications
    """

    EQUITY_SNAPSHOT_INTERVAL = 300  # 5 minutes
    RECONCILIATION_INTERVAL = 3600  # 1 hour

    def __init__(self, config: Optional[BotConfig] = None):
        """
        Initialize trading bot.

        Args:
            config: Bot configuration (loads from env if not provided)
        """
        self.config = config or load_config()
        self._shutdown_requested = False
        self._trader: Optional[EnhancedTrader] = None
        self._trade_store: Optional[TradeStore] = None
        self._telegram: Optional['TelegramNotifier'] = None
        self._strategy: Optional[TradingStrategy] = None
        self._last_equity_snapshot = datetime.min
        self._last_strategy_check = datetime.min
        self._last_reconciliation = datetime.min

        # Kill switch state
        self._kill_switch_triggered = False
        self._kill_switch_reason: Optional[str] = None
        self._peak_equity: float = 0.0
        self._day_start_equity: float = 0.0

        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if initialization successful
        """
        logger.info("=" * 60)
        logger.info("TRADING BOT STARTING")
        logger.info("=" * 60)

        try:
            # Initialize trade store
            logger.info("Initializing trade store...")
            self._trade_store = TradeStore(self.config.trades_db)

            # Initialize trader
            logger.info("Connecting to Alpaca...")
            self._trader = EnhancedTrader(config=self.config)

            if not self._trader.is_connected:
                logger.error("Failed to connect to Alpaca")
                return False

            # Telegram notifications disabled for now
            # TODO: Re-enable when bot strategy is production-ready
            logger.info("Telegram notifications disabled")
            self._telegram = None

            # Log initial account state
            account = self._trader.get_account()
            logger.info(f"Account equity: ${account.equity:,.2f}")
            logger.info(f"Cash: ${account.cash:,.2f}")
            logger.info(f"Buying power: ${account.buying_power:,.2f}")

            # Initialize kill switch tracking
            self._day_start_equity = account.equity
            self._peak_equity = account.equity
            logger.info(f"Kill switch tracking: day_start=${self._day_start_equity:,.2f}, "
                       f"daily_loss_limit={self.config.risk.kill_switch_daily_loss_pct:.1%}, "
                       f"max_drawdown_limit={self.config.risk.kill_switch_max_drawdown_pct:.1%}")

            # Reconcile positions with Alpaca on startup
            if self.config.risk.position_reconcile_on_start:
                self._reconcile_positions()

            # Record initial equity snapshot
            self._record_equity_snapshot()

            # Initialize trading strategy
            logger.info("Initializing trading strategy...")
            self._strategy = TradingStrategy()
            logger.info(f"Strategy config: {self._strategy.get_status()['config']}")

            logger.info("Bot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False

    def _record_equity_snapshot(self) -> None:
        """Record equity snapshot to database."""
        try:
            account = self._trader.get_account()

            snapshot = EquitySnapshot(
                timestamp=datetime.now().isoformat(),
                equity=account.equity,
                cash=account.cash,
                positions_value=account.positions_value,
                daily_pnl=account.daily_pnl,
                daily_pnl_pct=account.daily_pnl_pct,
            )

            self._trade_store.record_equity_snapshot(snapshot)
            self._last_equity_snapshot = datetime.now()

        except Exception as e:
            logger.error(f"Failed to record equity snapshot: {e}")

    def _reconcile_positions(self) -> None:
        """Reconcile positions between Alpaca and SQLite database."""
        try:
            logger.info("Reconciling positions with Alpaca...")

            # Get Alpaca positions
            alpaca_positions = self._trader.get_positions()

            # Convert to dict format expected by trade_store
            pos_list = []
            for pos in alpaca_positions:
                pos_list.append({
                    'symbol': pos.symbol,
                    'qty': pos.qty,
                    'avg_entry_price': pos.avg_entry_price,
                    'market_value': pos.market_value,
                })

            # Reconcile and sync
            result = self._trade_store.reconcile_positions(pos_list)

            if result['needs_sync']:
                logger.warning(f"Position discrepancies found, syncing from Alpaca...")
                sync_result = self._trade_store.sync_from_alpaca(pos_list)
                logger.info(f"Sync complete: added={sync_result['added']}, "
                           f"removed={sync_result['removed']}, "
                           f"updated={sync_result['updated']}")

                # Alert via Telegram
                if self._telegram:
                    self._telegram.send_error_alert(
                        f"Position sync performed",
                        f"Added: {sync_result['added']}, Removed: {sync_result['removed']}"
                    )
            else:
                logger.info(f"Positions in sync: {len(result['matched'])} positions matched")

            self._last_reconciliation = datetime.now()

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")

    def _check_kill_switch(self) -> tuple[bool, str]:
        """
        Check if kill switch should be triggered.

        Returns:
            Tuple of (should_kill: bool, reason: str)
        """
        if not self.config.risk.kill_switch_enabled:
            return False, "Kill switch disabled"

        if self._kill_switch_triggered:
            return True, self._kill_switch_reason or "Previously triggered"

        try:
            account = self._trader.get_account()
            current_equity = account.equity

            # Update peak equity
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity

            # Check daily loss
            daily_loss_pct = (self._day_start_equity - current_equity) / self._day_start_equity
            if daily_loss_pct >= self.config.risk.kill_switch_daily_loss_pct:
                reason = (f"Daily loss limit hit: {daily_loss_pct:.2%} loss "
                         f"(limit: {self.config.risk.kill_switch_daily_loss_pct:.1%})")
                self._trigger_kill_switch(reason)
                return True, reason

            # Check drawdown from peak
            drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown_pct >= self.config.risk.kill_switch_max_drawdown_pct:
                reason = (f"Max drawdown limit hit: {drawdown_pct:.2%} drawdown "
                         f"(limit: {self.config.risk.kill_switch_max_drawdown_pct:.1%})")
                self._trigger_kill_switch(reason)
                return True, reason

            return False, "OK"

        except Exception as e:
            logger.error(f"Kill switch check failed: {e}")
            return False, f"Check failed: {e}"

    def _trigger_kill_switch(self, reason: str) -> None:
        """Trigger the kill switch - no new buys allowed."""
        self._kill_switch_triggered = True
        self._kill_switch_reason = reason

        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")

        # Alert via Telegram
        if self._telegram:
            self._telegram.send_error_alert(
                f"KILL SWITCH TRIGGERED",
                reason
            )

    def _reset_kill_switch(self) -> None:
        """Reset kill switch for new trading day."""
        if self._kill_switch_triggered:
            logger.info("Resetting kill switch for new trading day")

        self._kill_switch_triggered = False
        self._kill_switch_reason = None

        # Reset day start equity
        try:
            account = self._trader.get_account()
            self._day_start_equity = account.equity
            logger.info(f"New day start equity: ${self._day_start_equity:,.2f}")
        except Exception as e:
            logger.error(f"Failed to reset day start equity: {e}")

    def _run_strategy(self) -> None:
        """Run the trading strategy and execute signals."""
        try:
            logger.info("Running strategy check...")

            # Get current account state
            account = self._trader.get_account()
            positions = self._trader.get_positions()

            # Build current positions dict (symbol -> market value)
            current_positions = {}
            for pos in positions:
                current_positions[pos.symbol] = pos.market_value

            # Fetch live quotes for accurate order sizing
            live_quotes = None
            try:
                symbols_to_quote = list(set(
                    list(current_positions.keys()) +
                    self._strategy.config.symbols
                ))
                if symbols_to_quote:
                    live_quotes = self._trader.get_quotes(symbols_to_quote)
                    logger.info(f"Fetched live quotes for {len(live_quotes)} symbols")
            except Exception as e:
                logger.warning(f"Could not fetch live quotes, will use historical prices: {e}")

            # Generate signals (pass live quotes for accurate sizing)
            signal = self._strategy.generate_signals(
                current_positions=current_positions,
                portfolio_value=account.equity,
                cash=account.cash,
                live_quotes=live_quotes,
            )

            if signal is None:
                logger.info("No trading signals generated")
                return

            # Execute trades
            logger.info(f"Executing {len(signal.trades_needed)} trades...")
            logger.info(f"Regime: {signal.regime}, Confidence: {signal.confidence:.0%}")

            for trade in signal.trades_needed:
                try:
                    # Check if dry run
                    if self.config.dry_run:
                        logger.info(f"[DRY RUN] Would {trade['side']} {trade['qty']} {trade['symbol']}")
                        continue

                    # Kill switch: block new buys but allow sells
                    if self._kill_switch_triggered and trade['side'] == 'buy':
                        logger.warning(f"KILL SWITCH ACTIVE - blocking buy: {trade['symbol']}")
                        continue

                    # Execute the trade
                    order = self._trader.submit_order(
                        symbol=trade['symbol'],
                        qty=trade['qty'],
                        side=trade['side'],
                        order_type='market',
                    )

                    logger.info(f"Order submitted: {order.order_id} - {trade['side']} {trade['qty']} {trade['symbol']}")

                    # Wait for fill to get actual fill price (market orders usually fill immediately)
                    filled_order = self._trader.wait_for_fill(order.order_id, timeout_seconds=30)

                    # Use actual fill price if available, fallback to estimated price
                    if filled_order and filled_order.filled_avg_price:
                        fill_price = filled_order.filled_avg_price
                        fill_qty = filled_order.filled_qty
                        logger.info(f"Order filled: {trade['symbol']} {fill_qty} @ ${fill_price:.2f}")
                    else:
                        fill_price = trade.get('price', 0)
                        fill_qty = trade['qty']
                        logger.warning(f"Using estimated price for {trade['symbol']} (fill not confirmed)")

                    # Track position open (for buy orders)
                    if trade['side'] == 'buy':
                        self._trade_store.track_position_open(
                            symbol=trade['symbol'],
                            side='long',  # Normalize to 'long' for consistency with trade_store
                            qty=fill_qty,
                            entry_price=fill_price,
                            entry_order_id=order.order_id,
                        )
                    # Track position close (for sell orders)
                    elif trade['side'] == 'sell':
                        # Get current equity for PnL calculation
                        account = self._trader.get_account()
                        closed_trade = self._trade_store.track_position_close(
                            symbol=trade['symbol'],
                            exit_price=fill_price,
                            equity_after=account.equity,
                            exit_order_id=order.order_id,
                        )
                        if closed_trade:
                            logger.info(f"Position closed: {trade['symbol']} PnL ${closed_trade.pnl_abs:,.2f} ({closed_trade.pnl_pct:.2f}%)")

                    # Notify via Telegram with actual fill price
                    if self._telegram:
                        self._telegram.send_trade_notification(
                            symbol=trade['symbol'],
                            side=trade['side'],
                            qty=fill_qty,
                            price=fill_price,
                        )

                except Exception as e:
                    logger.error(f"Failed to execute trade {trade}: {e}")
                    if self._telegram:
                        self._telegram.send_error_alert(str(e), f"Trade execution failed: {trade['symbol']}")

            # Log new target weights
            logger.info(f"New target weights: {signal.target_weights}")

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            logger.error(traceback.format_exc())

    def run(self) -> int:
        """
        Main bot loop.

        Returns:
            Exit code (0 for success, 1 for error)
        """
        if not self.initialize():
            return 1

        logger.info("Entering main loop...")
        logger.info("Press Ctrl+C to stop")

        try:
            while not self._shutdown_requested:
                try:
                    # Ensure connection is alive
                    if not self._trader.ensure_connected():
                        logger.error("Lost connection, attempting reconnect...")
                        time.sleep(60)
                        continue

                    # Record equity snapshot periodically (always, even outside market hours)
                    if (datetime.now() - self._last_equity_snapshot).seconds >= self.EQUITY_SNAPSHOT_INTERVAL:
                        self._record_equity_snapshot()

                    # Check market hours before running strategy
                    market_open, market_status = is_market_open()

                    # Check kill switch conditions
                    if market_open:
                        kill_triggered, kill_reason = self._check_kill_switch()
                        if kill_triggered and not self._kill_switch_triggered:
                            # Kill switch just triggered - already logged in _trigger_kill_switch
                            pass

                    # Periodic position reconciliation (every hour)
                    if (datetime.now() - self._last_reconciliation).seconds >= self.RECONCILIATION_INTERVAL:
                        if market_open:
                            self._reconcile_positions()

                    # Run trading strategy (check every 5 minutes, actual rebalance per config)
                    if (datetime.now() - self._last_strategy_check).seconds >= 300:
                        if market_open:
                            self._run_strategy()
                        else:
                            logger.info(f"Skipping strategy check - {market_status}")
                        self._last_strategy_check = datetime.now()

                    # Sleep to prevent busy-waiting
                    time.sleep(60)

                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    logger.error(traceback.format_exc())

                    if self._telegram:
                        self._telegram.send_error_alert(str(e), "Main loop error")

                    # Wait before retrying
                    time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        finally:
            self.shutdown()

        return 0

    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN")
        logger.info("=" * 60)

        try:
            # Record final equity snapshot
            if self._trader and self._trader.is_connected:
                self._record_equity_snapshot()

            # Send shutdown notification
            if self._telegram:
                self._telegram.send_shutdown_notification("Normal shutdown")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("Shutdown complete")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Alpaca Trading Bot")
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run without executing real orders'
    )
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Set up logging
    log_dir = Path(os.getenv('BOT_LOGS_DIR', 'logs'))
    setup_logging(log_dir, args.log_level)

    # Override dry run from command line
    if args.dry_run:
        os.environ['BOT_DRY_RUN'] = 'true'

    # Run bot
    bot = TradingBot()
    exit_code = bot.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
