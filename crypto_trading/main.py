"""
Main orchestrator for crypto trading system.
"""
import asyncio
import logging
import signal
import sys
import os
from typing import Dict
from datetime import datetime
import yaml
from dotenv import load_dotenv

from crypto_trading.database import DatabaseManager
from crypto_trading.websocket_manager import KrakenWebSocketClient, TickEvent
from crypto_trading.signal_engine import SignalEngine
from crypto_trading.risk_manager import RiskManager
from crypto_trading.paper_trader import PaperTradingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Application:
    """Main application orchestrator."""

    def __init__(self, config_path: str = "crypto_trading/config.yaml"):
        """Initialize application with configuration."""
        # Load environment variables
        load_dotenv()

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.db = self._init_database()
        self.signal_engine = self._init_signal_engine()
        self.risk_manager = self._init_risk_manager()
        self.paper_trader = self._init_paper_trader()
        self.ws_client = None

        # Shutdown flag
        self._shutdown = False

        # Track OHLCV data in memory for signal generation
        self.ohlcv_cache: Dict[str, None] = {}

    def _init_database(self) -> DatabaseManager:
        """Initialize database connection."""
        db_config = self.config['database']

        # Substitute environment variables
        host = os.getenv('DB_HOST', db_config['host'].replace('${DB_HOST:', '').replace('}', ''))
        user = os.getenv('DB_USER', db_config['user'].replace('${DB_USER:', '').replace('}', ''))
        password = os.getenv('DB_PASSWORD', db_config['password'].replace('${DB_PASSWORD}', ''))

        return DatabaseManager(
            host=host if not host.startswith('${') else 'localhost',
            port=db_config['port'],
            database=db_config['database'],
            user=user if not user.startswith('${') else 'postgres',
            password=password
        )

    def _init_signal_engine(self) -> SignalEngine:
        """Initialize signal engine."""
        strategy_config = self.config['strategy']
        return SignalEngine(
            fast_period=strategy_config['ema_fast_period'],
            slow_period=strategy_config['ema_slow_period'],
            min_bars_required=strategy_config['min_bars_required']
        )

    def _init_risk_manager(self) -> RiskManager:
        """Initialize risk manager."""
        risk_config = self.config['risk']
        return RiskManager(
            max_portfolio_heat=risk_config['max_portfolio_heat'],
            atr_period=risk_config['atr_period'],
            min_position_pct=risk_config['min_position_pct'],
            max_position_pct=risk_config['max_position_pct']
        )

    def _init_paper_trader(self) -> PaperTradingEngine:
        """Initialize paper trading engine."""
        trading_config = self.config['trading']
        execution_config = self.config['execution']

        return PaperTradingEngine(
            initial_balance=trading_config['initial_balance'],
            base_slippage=execution_config['base_slippage'],
            commission=execution_config['commission']
        )

    async def on_tick(self, tick: TickEvent) -> None:
        """
        Handle incoming tick event.

        Args:
            tick: Tick event from WebSocket
        """
        try:
            # Write tick to database
            await self.db.write_tick(
                exchange=tick.exchange,
                symbol=tick.symbol,
                timestamp=tick.timestamp,
                price=tick.price,
                volume=tick.volume,
                bid=tick.bid,
                ask=tick.ask
            )

            # Fetch OHLCV data for signal generation
            ohlcv_df = await self.db.get_ohlcv(
                symbol=tick.symbol,
                interval="1m",
                limit=self.config['strategy']['min_bars_required']
            )

            if ohlcv_df.empty:
                return

            # Generate signal
            signal = await self.signal_engine.process_tick(tick.symbol, ohlcv_df)

            if not signal:
                return

            # Validate signal with risk manager
            if not self.risk_manager.validate_signal(
                symbol=signal.symbol,
                balance=self.paper_trader.get_balance(),
                open_positions=len(self.paper_trader.get_positions()),
                ohlcv_data={tick.symbol: ohlcv_df}
            ):
                logger.warning(f"Signal rejected by risk manager: {signal.symbol}")
                return

            # Size position
            positions = self.risk_manager.size_position(
                symbols=[signal.symbol],
                balance=self.paper_trader.get_balance(),
                ohlcv_data={tick.symbol: ohlcv_df}
            )

            if not positions:
                logger.warning(f"No position sizing for {signal.symbol}")
                return

            amount_usd = positions[0].amount_usd

            # Execute trade
            execution = self.paper_trader.execute(signal, amount_usd)

            if execution:
                logger.info(
                    f"✅ Trade executed: {execution.direction} {execution.quantity:.6f} "
                    f"{execution.symbol} @ ${execution.execution_price:.2f}"
                )

                # Log portfolio status
                balance = self.paper_trader.get_balance()
                positions = self.paper_trader.get_positions()
                logger.info(
                    f"💰 Balance: ${balance:.2f}, "
                    f"Positions: {len(positions)}"
                )

        except Exception as e:
            logger.error(f"Error processing tick for {tick.symbol}: {e}", exc_info=True)

    async def run(self) -> None:
        """Run the trading system."""
        try:
            # Connect to database
            logger.info("Connecting to database...")
            await self.db.connect()
            logger.info("✅ Database connected")

            # Initialize WebSocket client
            pairs = self.config['trading']['pairs']
            logger.info(f"Subscribing to {len(pairs)} pairs: {pairs}")

            self.ws_client = KrakenWebSocketClient(
                url=self.config['kraken']['websocket_url'],
                pairs=pairs,
                on_tick=lambda tick: asyncio.create_task(self.on_tick(tick))
            )

            # Setup signal handlers
            def signal_handler(sig, frame):
                logger.info("Shutdown signal received")
                self._shutdown = True

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Log startup info
            logger.info("=" * 60)
            logger.info("🚀 Crypto Trading System Started")
            logger.info(f"💵 Initial Balance: ${self.config['trading']['initial_balance']:.2f}")
            logger.info(f"📊 Strategy: EMA{self.config['strategy']['ema_fast_period']}/"
                       f"EMA{self.config['strategy']['ema_slow_period']} Crossover")
            logger.info(f"⚠️  Max Portfolio Heat: {self.config['risk']['max_portfolio_heat']*100:.0f}%")
            logger.info("=" * 60)

            # Run WebSocket client
            await self.ws_client.connect()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")

        if self.ws_client:
            await self.ws_client.disconnect()

        if self.db:
            await self.db.disconnect()

        # Log final status
        logger.info("=" * 60)
        logger.info("📊 Final Status")
        logger.info(f"💰 Balance: ${self.paper_trader.get_balance():.2f}")
        logger.info(f"📈 P&L: ${self.paper_trader.get_balance() - self.config['trading']['initial_balance']:.2f}")
        logger.info(f"📊 Open Positions: {len(self.paper_trader.get_positions())}")
        logger.info(f"📝 Total Trades: {len(self.paper_trader.trade_history)}")
        logger.info("=" * 60)
        logger.info("✅ Shutdown complete")


def main():
    """Main entry point."""
    app = Application()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
