"""
Alpaca Paper Trading Pipeline
==============================
Production-ready paper trading integration with Alpaca Markets API.

Features:
- Connect to Alpaca Paper Trading API
- Fetch real-time/delayed quotes
- Submit market/limit orders
- Track positions and P&L
- Pre-trade risk checks

Usage:
    >>> from deployment.alpaca_paper_trader import AlpacaPaperTrader
    >>> trader = AlpacaPaperTrader()
    >>> trader.submit_order('AAPL', qty=10, side='buy', order_type='market')
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

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
        OrderStatus,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

logger = logging.getLogger(__name__)


class RiskCheckResult(Enum):
    """Risk check result codes."""
    PASSED = "passed"
    FAILED_INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    FAILED_POSITION_LIMIT = "position_limit_exceeded"
    FAILED_ORDER_SIZE = "order_size_exceeded"
    FAILED_CONCENTRATION = "concentration_limit_exceeded"
    FAILED_DAILY_LOSS = "daily_loss_limit_exceeded"


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_pct: float = 0.25  # Max 25% of portfolio in single position
    max_order_value: float = 50000.0  # Max $50k per order
    max_daily_loss_pct: float = 0.05  # Max 5% daily drawdown
    min_buying_power_reserve: float = 0.10  # Keep 10% cash reserve
    max_positions: int = 20  # Max number of concurrent positions


@dataclass
class TradeOrder:
    """Trade order representation."""
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market' or 'limit'
    limit_price: Optional[float] = None
    time_in_force: str = 'day'

    # Filled after submission
    order_id: Optional[str] = None
    status: Optional[str] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


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


class AlpacaPaperTrader:
    """
    Alpaca Paper Trading client with risk management.

    Connects to Alpaca's paper trading API for testing strategies
    with simulated money.

    Example:
        >>> trader = AlpacaPaperTrader()
        >>> if trader.is_connected():
        ...     quote = trader.get_quote('AAPL')
        ...     result = trader.submit_order('AAPL', qty=10, side='buy')
        ...     print(f"Order {result.order_id}: {result.status}")
    """

    PAPER_BASE_URL = "https://paper-api.alpaca.markets"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize Alpaca paper trading client.

        Args:
            api_key: Alpaca API key (or ALPACA_API_KEY env var)
            api_secret: Alpaca API secret (or ALPACA_API_SECRET env var)
            risk_limits: Risk limit configuration

        Raises:
            ImportError: If alpaca-py not installed
            ValueError: If credentials missing
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            )

        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca credentials required. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET environment variables."
            )

        self.risk_limits = risk_limits or RiskLimits()
        self._trading_client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None
        self._connected = False

        # Track daily P&L
        self._day_start_equity: Optional[float] = None
        self._trade_log: List[TradeOrder] = []

        self._connect()

    def _connect(self) -> None:
        """Establish connection to Alpaca APIs."""
        try:
            # Trading client for orders and account
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=True  # IMPORTANT: Paper trading only
            )

            # Data client for quotes
            self._data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )

            # Verify connection by fetching account
            account = self._trading_client.get_account()
            self._day_start_equity = float(account.last_equity)
            self._connected = True

            logger.info(
                f"Connected to Alpaca Paper Trading. "
                f"Equity: ${float(account.equity):,.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self._connected = False
            raise

    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self._connected

    def get_account(self) -> AccountSnapshot:
        """
        Get current account snapshot.

        Returns:
            AccountSnapshot with equity, cash, positions value, P&L
        """
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        account = self._trading_client.get_account()

        equity = float(account.equity)
        cash = float(account.cash)
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value)

        # Calculate daily P&L
        last_equity = float(account.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0

        positions_value = portfolio_value - cash

        return AccountSnapshot(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
        )

    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        if not self._connected:
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
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Position or None if no position
        """
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
        """
        Get latest quote for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with 'bid', 'ask', 'mid', 'bid_size', 'ask_size'
        """
        if not self._connected:
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
        """
        Get latest quotes for multiple symbols.

        Args:
            symbols: List of stock tickers

        Returns:
            Dict mapping symbol to quote dict
        """
        if not self._connected:
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
    ) -> Tuple[RiskCheckResult, str]:
        """
        Perform pre-trade risk checks.

        Args:
            symbol: Stock ticker
            qty: Order quantity
            side: 'buy' or 'sell'
            limit_price: Limit price (or current market price used)

        Returns:
            Tuple of (RiskCheckResult, explanation message)
        """
        account = self.get_account()
        positions = self.get_positions()

        # Get current price for value calculations
        if limit_price:
            price = limit_price
        else:
            quote = self.get_quote(symbol)
            price = quote['ask'] if side == 'buy' else quote['bid']

        order_value = qty * price

        # Check 1: Order size limit
        if order_value > self.risk_limits.max_order_value:
            return (
                RiskCheckResult.FAILED_ORDER_SIZE,
                f"Order value ${order_value:,.2f} exceeds max "
                f"${self.risk_limits.max_order_value:,.2f}"
            )

        # Check 2: Buying power (for buys)
        if side == 'buy':
            min_reserve = account.equity * self.risk_limits.min_buying_power_reserve
            available = account.buying_power - min_reserve

            if order_value > available:
                return (
                    RiskCheckResult.FAILED_INSUFFICIENT_BUYING_POWER,
                    f"Insufficient buying power. Need ${order_value:,.2f}, "
                    f"available ${available:,.2f} (after {self.risk_limits.min_buying_power_reserve:.0%} reserve)"
                )

        # Check 3: Position concentration
        if side == 'buy':
            # Calculate what position would be after order
            existing_pos = self.get_position(symbol)
            existing_value = existing_pos.market_value if existing_pos else 0
            new_position_value = existing_value + order_value
            concentration = new_position_value / account.equity

            if concentration > self.risk_limits.max_position_pct:
                return (
                    RiskCheckResult.FAILED_CONCENTRATION,
                    f"Position would be {concentration:.1%} of portfolio, "
                    f"exceeds max {self.risk_limits.max_position_pct:.0%}"
                )

        # Check 4: Number of positions
        if side == 'buy' and not self.get_position(symbol):
            if len(positions) >= self.risk_limits.max_positions:
                return (
                    RiskCheckResult.FAILED_POSITION_LIMIT,
                    f"Already at max positions ({self.risk_limits.max_positions})"
                )

        # Check 5: Daily loss limit
        if account.daily_pnl_pct < -self.risk_limits.max_daily_loss_pct * 100:
            return (
                RiskCheckResult.FAILED_DAILY_LOSS,
                f"Daily loss {account.daily_pnl_pct:.1f}% exceeds limit "
                f"{-self.risk_limits.max_daily_loss_pct:.0%}"
            )

        return (RiskCheckResult.PASSED, "All risk checks passed")

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
        """
        Submit an order to Alpaca.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            limit_price: Required for limit orders
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            skip_risk_check: Bypass risk checks (use with caution)

        Returns:
            TradeOrder with order details

        Raises:
            ValueError: If risk checks fail or invalid parameters
        """
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        # Validate parameters
        if side not in ('buy', 'sell'):
            raise ValueError(f"Invalid side: {side}")
        if order_type not in ('market', 'limit'):
            raise ValueError(f"Invalid order_type: {order_type}")
        if order_type == 'limit' and limit_price is None:
            raise ValueError("limit_price required for limit orders")

        # Risk checks
        if not skip_risk_check:
            risk_result, risk_msg = self.check_risk(
                symbol, qty, side, limit_price
            )
            if risk_result != RiskCheckResult.PASSED:
                raise ValueError(f"Risk check failed: {risk_msg}")

        # Build order request
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

        # Submit order
        order = self._trading_client.submit_order(request)

        # Build response
        trade_order = TradeOrder(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            order_id=str(order.id),
            status=order.status.value,
            filled_qty=float(order.filled_qty) if order.filled_qty else 0,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            submitted_at=order.submitted_at,
            filled_at=order.filled_at,
        )

        self._trade_log.append(trade_order)

        logger.info(
            f"Order submitted: {side.upper()} {qty} {symbol} @ "
            f"{'MARKET' if order_type == 'market' else f'LIMIT ${limit_price}'} "
            f"[{trade_order.order_id[:8]}...]"
        )

        return trade_order

    def get_order(self, order_id: str) -> Optional[TradeOrder]:
        """
        Get order status by ID.

        Args:
            order_id: Alpaca order ID

        Returns:
            TradeOrder with current status
        """
        try:
            order = self._trading_client.get_order_by_id(order_id)

            return TradeOrder(
                symbol=order.symbol,
                qty=float(order.qty),
                side='buy' if order.side == OrderSide.BUY else 'sell',
                order_type='market' if order.type == OrderType.MARKET else 'limit',
                limit_price=float(order.limit_price) if order.limit_price else None,
                time_in_force=order.time_in_force.value,
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

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if cancelled successfully
        """
        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def close_position(self, symbol: str) -> Optional[TradeOrder]:
        """
        Close an entire position.

        Args:
            symbol: Stock ticker

        Returns:
            TradeOrder for the closing trade
        """
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
            skip_risk_check=True,  # Closing is always allowed
        )

    def close_all_positions(self) -> List[TradeOrder]:
        """
        Close all open positions.

        Returns:
            List of TradeOrders for closing trades
        """
        positions = self.get_positions()
        orders = []

        for pos in positions:
            order = self.close_position(pos.symbol)
            if order:
                orders.append(order)

        return orders

    def get_pending_orders(self) -> List[TradeOrder]:
        """
        Get all pending (open) orders.

        Returns:
            List of pending TradeOrders
        """
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
        )
        orders = self._trading_client.get_orders(request)

        result = []
        for order in orders:
            result.append(TradeOrder(
                symbol=order.symbol,
                qty=float(order.qty),
                side='buy' if order.side == OrderSide.BUY else 'sell',
                order_type='market' if order.type == OrderType.MARKET else 'limit',
                limit_price=float(order.limit_price) if order.limit_price else None,
                time_in_force=order.time_in_force.value,
                order_id=str(order.id),
                status=order.status.value,
                filled_qty=float(order.filled_qty) if order.filled_qty else 0,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
            ))

        return result

    def cancel_all_orders(self) -> int:
        """
        Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        self._trading_client.cancel_orders()
        pending = self.get_pending_orders()
        cancelled = len(pending)
        logger.info(f"Cancelled {cancelled} pending orders")
        return cancelled

    def get_trade_log(self) -> List[TradeOrder]:
        """Get log of all trades submitted this session."""
        return self._trade_log.copy()

    def execute_portfolio_allocation(
        self,
        target_allocation: Dict[str, float],
        portfolio_value: Optional[float] = None,
        rebalance: bool = True,
    ) -> List[TradeOrder]:
        """
        Execute trades to achieve target portfolio allocation.

        This is the main integration point with portfolio optimizers.

        Args:
            target_allocation: Dict of symbol -> weight (0.0 to 1.0)
            portfolio_value: Total portfolio value (uses account equity if None)
            rebalance: If True, also sell/reduce over-allocated positions

        Returns:
            List of TradeOrders executed

        Example:
            >>> allocation = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'AMZN': 0.25}
            >>> orders = trader.execute_portfolio_allocation(allocation)
        """
        account = self.get_account()
        pv = portfolio_value or account.equity

        # Validate allocation sums to <= 1.0
        total_weight = sum(target_allocation.values())
        if total_weight > 1.0 + 1e-6:
            raise ValueError(f"Allocation weights sum to {total_weight:.2%}, must be <= 100%")

        # Get current positions
        current_positions = {p.symbol: p for p in self.get_positions()}

        # Get quotes for all symbols
        all_symbols = list(set(target_allocation.keys()) | set(current_positions.keys()))
        quotes = self.get_quotes(all_symbols)

        orders = []

        # Calculate required trades
        for symbol, target_weight in target_allocation.items():
            target_value = pv * target_weight
            current_value = current_positions.get(symbol, Position(
                symbol=symbol, qty=0, avg_entry_price=0, current_price=0,
                market_value=0, unrealized_pnl=0, unrealized_pnl_pct=0, side='long'
            )).market_value

            diff_value = target_value - current_value
            price = quotes[symbol]['mid']

            if abs(diff_value) < 50:  # Skip tiny adjustments
                continue

            if diff_value > 0:
                # Need to buy
                qty = int(diff_value / price)
                if qty > 0:
                    try:
                        order = self.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            order_type='market',
                        )
                        orders.append(order)
                    except ValueError as e:
                        logger.warning(f"Skipping {symbol} buy: {e}")
            elif diff_value < 0 and rebalance:
                # Need to sell
                qty = int(abs(diff_value) / price)
                if qty > 0:
                    try:
                        order = self.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='sell',
                            order_type='market',
                            skip_risk_check=True,
                        )
                        orders.append(order)
                    except ValueError as e:
                        logger.warning(f"Skipping {symbol} sell: {e}")

        # Close positions not in target allocation
        if rebalance:
            for symbol in current_positions:
                if symbol not in target_allocation:
                    order = self.close_position(symbol)
                    if order:
                        orders.append(order)

        logger.info(f"Executed {len(orders)} orders for portfolio rebalancing")
        return orders


def create_trader_from_env() -> AlpacaPaperTrader:
    """
    Factory function to create trader from environment variables.

    Returns:
        Configured AlpacaPaperTrader instance

    Raises:
        ValueError: If required env vars not set
    """
    return AlpacaPaperTrader()


if __name__ == "__main__":
    # Quick test
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        trader = AlpacaPaperTrader()

        print("\n=== Account Info ===")
        account = trader.get_account()
        print(f"Equity:       ${account.equity:,.2f}")
        print(f"Cash:         ${account.cash:,.2f}")
        print(f"Buying Power: ${account.buying_power:,.2f}")
        print(f"Daily P&L:    ${account.daily_pnl:,.2f} ({account.daily_pnl_pct:+.2f}%)")

        print("\n=== Current Positions ===")
        positions = trader.get_positions()
        if positions:
            for pos in positions:
                print(
                    f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f} "
                    f"(P&L: ${pos.unrealized_pnl:,.2f})"
                )
        else:
            print("  No open positions")

        print("\n=== Sample Quotes ===")
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            try:
                quote = trader.get_quote(symbol)
                print(f"  {symbol}: ${quote['mid']:.2f} (bid: ${quote['bid']:.2f}, ask: ${quote['ask']:.2f})")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")

        print("\nPaper trading pipeline ready!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
