"""
Enhanced Alpaca Trading Bot
===========================
Production-ready trading client with:
- Automatic reconnection
- Robust error handling
- Trade logging
- Graceful shutdown
"""

from __future__ import annotations

import os
import signal
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .config import BotConfig, AlpacaConfig, RiskConfig
from .logging_config import get_logger, TradeLogger

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        OrderType,
        TimeInForce,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


logger = get_logger(__name__)
trade_logger = TradeLogger()


@dataclass
class Position:
    """Position representation."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str  # 'long' or 'short'


@dataclass
class TradeOrder:
    """Trade order representation."""
    symbol: str
    qty: float
    side: str
    order_type: str
    limit_price: Optional[float] = None
    order_id: Optional[str] = None
    status: Optional[str] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class AccountSnapshot:
    """Account state snapshot."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    positions_value: float
    daily_pnl: float
    daily_pnl_pct: float
    timestamp: datetime = field(default_factory=datetime.now)


class EnhancedTrader:
    """
    Enhanced Alpaca trading client with production features.

    Features:
    - Automatic reconnection with exponential backoff
    - Proper error handling and logging
    - Clean shutdown handling
    - Trade event callbacks
    """

    MAX_RECONNECT_ATTEMPTS = 10
    INITIAL_BACKOFF_SECONDS = 1
    MAX_BACKOFF_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        config: Optional[BotConfig] = None,
        alpaca_config: Optional[AlpacaConfig] = None,
        risk_config: Optional[RiskConfig] = None,
    ):
        """
        Initialize enhanced trader.

        Args:
            config: Complete bot configuration (preferred)
            alpaca_config: Alpaca API configuration (alternative)
            risk_config: Risk limits (alternative)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

        # Load configuration
        if config:
            self._alpaca_config = config.alpaca
            self._risk_config = config.risk
        else:
            self._alpaca_config = alpaca_config or AlpacaConfig.from_env()
            self._risk_config = risk_config or RiskConfig.from_env()

        # Connection state
        self._trading_client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None
        self._connected = False
        self._reconnect_count = 0
        self._last_reconnect_attempt = 0

        # Shutdown handling
        self._shutdown_event = threading.Event()
        self._setup_signal_handlers()

        # Trade tracking
        self._day_start_equity: Optional[float] = None
        self._trade_log: List[TradeOrder] = []

        # Callbacks
        self._on_trade_callbacks: List[Callable] = []

        # Initial connection
        self._connect()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _connect(self) -> bool:
        """
        Establish connection to Alpaca APIs.

        Returns:
            True if connected successfully
        """
        try:
            logger.info("Connecting to Alpaca...")

            # Trading client for orders and account
            self._trading_client = TradingClient(
                api_key=self._alpaca_config.api_key,
                secret_key=self._alpaca_config.api_secret,
                paper=True  # Paper trading only
            )

            # Data client for quotes
            self._data_client = StockHistoricalDataClient(
                api_key=self._alpaca_config.api_key,
                secret_key=self._alpaca_config.api_secret
            )

            # Verify connection by fetching account
            account = self._trading_client.get_account()
            self._day_start_equity = float(account.last_equity)
            self._connected = True
            self._reconnect_count = 0

            trade_logger.log_connection_status(
                True,
                f"Equity: ${float(account.equity):,.2f}"
            )

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False

    def _reconnect_with_backoff(self) -> bool:
        """
        Attempt reconnection with exponential backoff.

        Returns:
            True if reconnected successfully
        """
        if self._reconnect_count >= self.MAX_RECONNECT_ATTEMPTS:
            logger.error(f"Max reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) exceeded")
            return False

        # Calculate backoff delay
        backoff = min(
            self.INITIAL_BACKOFF_SECONDS * (2 ** self._reconnect_count),
            self.MAX_BACKOFF_SECONDS
        )

        self._reconnect_count += 1
        logger.info(f"Reconnection attempt {self._reconnect_count}/{self.MAX_RECONNECT_ATTEMPTS} "
                   f"in {backoff} seconds...")

        time.sleep(backoff)

        return self._connect()

    def ensure_connected(self) -> bool:
        """
        Ensure connection is active, reconnecting if necessary.

        Returns:
            True if connected
        """
        if self._connected:
            # Verify connection is still alive
            try:
                self._trading_client.get_account()
                return True
            except Exception:
                logger.warning("Connection lost, attempting reconnect...")
                self._connected = False

        return self._reconnect_with_backoff()

    @property
    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self._connected

    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()

    def shutdown(self) -> None:
        """Initiate graceful shutdown."""
        logger.info("Initiating shutdown...")
        self._shutdown_event.set()

    def get_account(self) -> AccountSnapshot:
        """Get current account snapshot."""
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Alpaca")

        account = self._trading_client.get_account()

        equity = float(account.equity)
        cash = float(account.cash)
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value)

        last_equity = float(account.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0

        positions_value = portfolio_value - cash

        snapshot = AccountSnapshot(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
        )

        trade_logger.log_account_snapshot(equity, cash, daily_pnl, daily_pnl_pct)

        return snapshot

    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Alpaca")

        positions = self._trading_client.get_all_positions()
        result = []

        for pos in positions:
            result.append(Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                side='long' if float(pos.qty) > 0 else 'short',
            ))

        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        try:
            pos = self._trading_client.get_open_position(symbol)
            return Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                side='long' if float(pos.qty) > 0 else 'short',
            )
        except Exception:
            return None

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get latest quote for a symbol."""
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Alpaca")

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self._data_client.get_stock_latest_quote(request)

        quote = quotes[symbol]
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)

        return {
            'bid': bid,
            'ask': ask,
            'mid': (bid + ask) / 2,
            'bid_size': int(quote.bid_size),
            'ask_size': int(quote.ask_size),
        }

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get latest quotes for multiple symbols."""
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Alpaca")

        request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = self._data_client.get_stock_latest_quote(request)

        result = {}
        for symbol, quote in quotes.items():
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            result[symbol] = {
                'bid': bid,
                'ask': ask,
                'mid': (bid + ask) / 2,
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size),
            }

        return result

    def check_risk(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Perform pre-trade risk checks.

        Returns:
            Tuple of (passed: bool, message: str)
        """
        try:
            account = self.get_account()
            positions = self.get_positions()

            # Get current price
            if limit_price:
                price = limit_price
            else:
                quote = self.get_quote(symbol)
                price = quote['ask'] if side == 'buy' else quote['bid']

            order_value = qty * price

            # Check order size
            if order_value > self._risk_config.max_order_value:
                msg = f"Order value ${order_value:,.2f} exceeds max ${self._risk_config.max_order_value:,.2f}"
                trade_logger.log_risk_check_failed(symbol, msg)
                return False, msg

            # Check buying power
            if side == 'buy':
                min_reserve = account.equity * self._risk_config.min_buying_power_reserve
                available = account.buying_power - min_reserve
                if order_value > available:
                    msg = f"Insufficient buying power. Need ${order_value:,.2f}, available ${available:,.2f}"
                    trade_logger.log_risk_check_failed(symbol, msg)
                    return False, msg

            # Check concentration
            if side == 'buy':
                existing_pos = self.get_position(symbol)
                existing_value = existing_pos.market_value if existing_pos else 0
                new_value = existing_value + order_value
                concentration = new_value / account.equity

                if concentration > self._risk_config.max_position_pct:
                    msg = f"Position would be {concentration:.1%}, exceeds max {self._risk_config.max_position_pct:.0%}"
                    trade_logger.log_risk_check_failed(symbol, msg)
                    return False, msg

            # Check position count
            if side == 'buy' and not self.get_position(symbol):
                if len(positions) >= self._risk_config.max_positions:
                    msg = f"Already at max positions ({self._risk_config.max_positions})"
                    trade_logger.log_risk_check_failed(symbol, msg)
                    return False, msg

            # Check daily loss
            if account.daily_pnl_pct < -self._risk_config.max_daily_loss_pct * 100:
                msg = f"Daily loss {account.daily_pnl_pct:.1f}% exceeds limit"
                trade_logger.log_risk_check_failed(symbol, msg)
                return False, msg

            return True, "All risk checks passed"

        except Exception as e:
            msg = f"Risk check error: {e}"
            logger.error(msg)
            return False, msg

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        time_in_force: str = 'day',
        skip_risk_check: bool = False,
    ) -> TradeOrder:
        """Submit an order to Alpaca."""
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Alpaca")

        # Validate
        if side not in ('buy', 'sell'):
            raise ValueError(f"Invalid side: {side}")
        if order_type not in ('market', 'limit'):
            raise ValueError(f"Invalid order_type: {order_type}")
        if order_type == 'limit' and limit_price is None:
            raise ValueError("limit_price required for limit orders")

        # Risk checks
        if not skip_risk_check:
            passed, msg = self.check_risk(symbol, qty, side, limit_price)
            if not passed:
                raise ValueError(f"Risk check failed: {msg}")

        # Build order
        order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
        tif_map = {
            'day': TimeInForce.DAY,
            'gtc': TimeInForce.GTC,
            'ioc': TimeInForce.IOC,
            'fok': TimeInForce.FOK,
        }
        tif = tif_map.get(time_in_force, TimeInForce.DAY)

        if order_type == 'market':
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
        else:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )

        # Submit
        order = self._trading_client.submit_order(request)

        trade_order = TradeOrder(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            order_id=str(order.id),
            status=order.status.value,
            filled_qty=float(order.filled_qty) if order.filled_qty else 0,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            submitted_at=order.submitted_at,
            filled_at=order.filled_at,
        )

        self._trade_log.append(trade_order)

        trade_logger.log_order_submitted(
            symbol, side, qty, order_type,
            trade_order.order_id,
            limit_price
        )

        # Trigger callbacks
        for callback in self._on_trade_callbacks:
            try:
                callback(trade_order)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

        return trade_order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self._trading_client.cancel_order_by_id(order_id)
            trade_logger.log_order_cancelled("", order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def close_position(self, symbol: str) -> Optional[TradeOrder]:
        """Close an entire position."""
        position = self.get_position(symbol)
        if not position:
            logger.warning(f"No position to close for {symbol}")
            return None

        side = 'sell' if position.side == 'long' else 'buy'

        return self.submit_order(
            symbol=symbol,
            qty=abs(position.qty),
            side=side,
            order_type='market',
            skip_risk_check=True,
        )

    def close_all_positions(self) -> List[TradeOrder]:
        """Close all open positions."""
        positions = self.get_positions()
        orders = []

        for pos in positions:
            order = self.close_position(pos.symbol)
            if order:
                orders.append(order)

        return orders

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        try:
            self._trading_client.cancel_orders()
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            pending = self._trading_client.get_orders(request)
            return len(pending)
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    def on_trade(self, callback: Callable[[TradeOrder], None]) -> None:
        """Register a callback for trade events."""
        self._on_trade_callbacks.append(callback)

    def get_trade_log(self) -> List[TradeOrder]:
        """Get log of all trades submitted this session."""
        return self._trade_log.copy()

    def get_order(self, order_id: str) -> Optional[TradeOrder]:
        """
        Get current status of an order.

        Args:
            order_id: The order ID to look up

        Returns:
            TradeOrder with current status, or None if not found
        """
        try:
            order = self._trading_client.get_order_by_id(order_id)

            return TradeOrder(
                symbol=order.symbol,
                qty=float(order.qty),
                side='buy' if order.side.value == 'buy' else 'sell',
                order_type=order.order_type.value,
                limit_price=float(order.limit_price) if order.limit_price else None,
                order_id=str(order.id),
                status=order.status.value,
                filled_qty=float(order.filled_qty) if order.filled_qty else 0,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
            )
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: int = 30,
        poll_interval: float = 0.5,
    ) -> Optional[TradeOrder]:
        """
        Wait for an order to fill.

        Args:
            order_id: The order ID to wait for
            timeout_seconds: Maximum time to wait
            poll_interval: How often to check order status

        Returns:
            TradeOrder with fill info, or None if timeout/error
        """
        import time

        start_time = time.time()
        logger.info(f"Waiting for order {order_id} to fill (timeout: {timeout_seconds}s)")

        while time.time() - start_time < timeout_seconds:
            order = self.get_order(order_id)
            if not order:
                return None

            if order.status == 'filled':
                logger.info(f"Order {order_id} filled: {order.filled_qty} @ ${order.filled_avg_price:.2f}")
                return order

            if order.status in ('canceled', 'expired', 'rejected'):
                logger.warning(f"Order {order_id} ended with status: {order.status}")
                return order

            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for order {order_id} to fill")
        return self.get_order(order_id)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from .logging_config import setup_logging

    setup_logging()

    try:
        trader = EnhancedTrader()

        print("\n=== Account Info ===")
        account = trader.get_account()
        print(f"Equity: ${account.equity:,.2f}")
        print(f"Daily P&L: ${account.daily_pnl:,.2f}")

        print("\n=== Quotes ===")
        for symbol in ['AAPL', 'MSFT']:
            quote = trader.get_quote(symbol)
            print(f"{symbol}: ${quote['mid']:.2f}")

        print("\nTrader ready!")

    except Exception as e:
        print(f"Error: {e}")
