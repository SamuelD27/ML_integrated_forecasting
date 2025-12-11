"""
Paper Trading Broker
====================
Simulated broker for paper trading with the pipeline.

Provides:
- Order submission and tracking
- Position management
- P&L calculation
- Integration with Alpaca paper trading (if configured)

Usage:
    from execution.paper_broker import PaperBroker

    broker = PaperBroker(initial_capital=100000)
    broker.submit_order('AAPL', 'buy', 100)
    broker.get_positions()
    broker.get_portfolio_value()
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False
    logger.debug("Alpaca SDK not available")


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class OrderType(Enum):
    """Order type enum."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    ticker: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_avg_price: float = 0.0

    # Timestamps
    created_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    submitted_at: Optional[pd.Timestamp] = None
    filled_at: Optional[pd.Timestamp] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def fill_value(self) -> float:
        return self.filled_quantity * self.filled_avg_price


@dataclass
class Position:
    """Represents a position in a security."""
    ticker: str
    quantity: int
    avg_cost: float
    side: str = 'long'  # 'long' or 'short'

    # Current values
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # History
    realized_pnl: float = 0.0

    def update_price(self, price: float):
        """Update current price and calculate P&L."""
        self.current_price = price
        self.market_value = self.quantity * price

        cost_basis = self.quantity * self.avg_cost
        self.unrealized_pnl = self.market_value - cost_basis
        self.unrealized_pnl_pct = self.unrealized_pnl / cost_basis if cost_basis > 0 else 0.0


class PaperBroker:
    """
    Paper trading broker.

    Simulates order execution and position management.
    Can optionally connect to Alpaca for real paper trading.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        use_alpaca: bool = False,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        """
        Initialize paper broker.

        Args:
            initial_capital: Starting capital
            use_alpaca: Use Alpaca API for paper trading
            alpaca_api_key: Alpaca API key
            alpaca_secret_key: Alpaca secret key
            paper: Use paper trading (True) or live (False)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict[str, Any]] = []

        # Alpaca integration
        self.use_alpaca = use_alpaca and HAS_ALPACA
        self.alpaca_client = None

        if self.use_alpaca:
            self._init_alpaca(alpaca_api_key, alpaca_secret_key, paper)

        # Price cache for simulated fills
        self._price_cache: Dict[str, float] = {}

        # Counters
        self._order_counter = 0

        logger.info(f"Paper broker initialized with ${initial_capital:,.0f}")
        if self.use_alpaca:
            logger.info("Connected to Alpaca paper trading")

    def _init_alpaca(
        self,
        api_key: Optional[str],
        secret_key: Optional[str],
        paper: bool,
    ):
        """Initialize Alpaca client."""
        import os

        api_key = api_key or os.getenv('ALPACA_API_KEY')
        secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not secret_key:
            logger.warning("Alpaca credentials not found, using simulated mode")
            self.use_alpaca = False
            return

        try:
            self.alpaca_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
            )

            # Sync account state
            account = self.alpaca_client.get_account()
            self.cash = float(account.cash)
            self.initial_capital = float(account.equity)

            logger.info(f"Alpaca account: ${float(account.equity):,.0f} equity, ${self.cash:,.0f} cash")

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}")
            self.use_alpaca = False

    def submit_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """
        Submit an order.

        Args:
            ticker: Security ticker
            side: 'buy' or 'sell'
            quantity: Number of shares
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            limit_price: Limit price (for limit/stop_limit orders)
            stop_price: Stop price (for stop/stop_limit orders)
            metadata: Additional order metadata

        Returns:
            Order object
        """
        # Generate order ID
        self._order_counter += 1
        order_id = f"ORD_{self._order_counter:06d}"

        # Create order
        order = Order(
            order_id=order_id,
            ticker=ticker,
            side=side.lower(),
            quantity=quantity,
            order_type=OrderType(order_type.lower()),
            limit_price=limit_price,
            stop_price=stop_price,
            metadata=metadata or {},
        )

        # Submit via Alpaca or simulate
        if self.use_alpaca and order_type == 'market':
            self._submit_alpaca_order(order)
        else:
            self._simulate_order(order)

        # Store order
        self.orders[order_id] = order
        self.order_history.append(order)

        logger.info(f"Order submitted: {side.upper()} {quantity} {ticker} @ {order_type}")

        return order

    def _submit_alpaca_order(self, order: Order):
        """Submit order via Alpaca."""
        try:
            alpaca_side = OrderSide.BUY if order.side == 'buy' else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=order.ticker,
                qty=order.quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )

            alpaca_order = self.alpaca_client.submit_order(request)

            order.status = OrderStatus.SUBMITTED
            order.submitted_at = pd.Timestamp.now()
            order.metadata['alpaca_order_id'] = alpaca_order.id

            # Poll for fill
            self._wait_for_fill(order, alpaca_order.id)

        except Exception as e:
            logger.error(f"Alpaca order failed: {e}")
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = str(e)

    def _wait_for_fill(self, order: Order, alpaca_order_id: str, timeout: int = 30):
        """Wait for Alpaca order to fill."""
        import time

        start = time.time()

        while time.time() - start < timeout:
            try:
                alpaca_order = self.alpaca_client.get_order_by_id(alpaca_order_id)

                if alpaca_order.status == 'filled':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = int(alpaca_order.filled_qty)
                    order.filled_avg_price = float(alpaca_order.filled_avg_price)
                    order.filled_at = pd.Timestamp.now()

                    # Update positions
                    self._update_position_from_fill(order)
                    return

                elif alpaca_order.status in ['cancelled', 'expired', 'rejected']:
                    order.status = OrderStatus.REJECTED
                    return

                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error checking order: {e}")
                time.sleep(1)

        logger.warning(f"Order {order.order_id} timed out")

    def _simulate_order(self, order: Order):
        """Simulate order fill."""
        # Get price
        price = self._get_price(order.ticker)
        if price is None:
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = 'No price available'
            return

        # Check for limit orders
        if order.order_type == OrderType.LIMIT:
            if order.side == 'buy' and price > order.limit_price:
                order.status = OrderStatus.PENDING
                return
            if order.side == 'sell' and price < order.limit_price:
                order.status = OrderStatus.PENDING
                return

        # Simulate fill
        if order.side == 'buy':
            fill_price = price * 1.001  # 10 bps slippage
            cost = order.quantity * fill_price

            if cost > self.cash:
                # Reduce quantity to fit cash
                order.quantity = int(self.cash / fill_price)
                if order.quantity <= 0:
                    order.status = OrderStatus.REJECTED
                    order.metadata['error'] = 'Insufficient cash'
                    return
                cost = order.quantity * fill_price

            self.cash -= cost

        else:  # sell
            fill_price = price * 0.999  # 10 bps slippage
            proceeds = order.quantity * fill_price
            self.cash += proceeds

        # Mark as filled
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_avg_price = fill_price
        order.submitted_at = pd.Timestamp.now()
        order.filled_at = pd.Timestamp.now()

        # Update positions
        self._update_position_from_fill(order)

        # Record trade
        self.trade_history.append({
            'timestamp': order.filled_at,
            'ticker': order.ticker,
            'side': order.side,
            'quantity': order.filled_quantity,
            'price': order.filled_avg_price,
            'value': order.fill_value,
        })

    def _update_position_from_fill(self, order: Order):
        """Update positions after order fill."""
        ticker = order.ticker

        if order.side == 'buy':
            if ticker in self.positions:
                pos = self.positions[ticker]
                # Average up
                total_cost = pos.avg_cost * pos.quantity + order.filled_avg_price * order.filled_quantity
                pos.quantity += order.filled_quantity
                pos.avg_cost = total_cost / pos.quantity
            else:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=order.filled_quantity,
                    avg_cost=order.filled_avg_price,
                )

        else:  # sell
            if ticker in self.positions:
                pos = self.positions[ticker]
                # Calculate realized P&L
                realized = (order.filled_avg_price - pos.avg_cost) * order.filled_quantity
                pos.realized_pnl += realized
                pos.quantity -= order.filled_quantity

                if pos.quantity <= 0:
                    del self.positions[ticker]

    def _get_price(self, ticker: str) -> Optional[float]:
        """Get current price for ticker."""
        # Check cache
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        # Try Alpaca
        if self.use_alpaca:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestQuoteRequest
                import os

                data_client = StockHistoricalDataClient(
                    api_key=os.getenv('ALPACA_API_KEY'),
                    secret_key=os.getenv('ALPACA_SECRET_KEY'),
                )

                request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
                quote = data_client.get_stock_latest_quote(request)

                if ticker in quote:
                    price = float(quote[ticker].ask_price)
                    self._price_cache[ticker] = price
                    return price

            except Exception as e:
                logger.debug(f"Failed to get Alpaca price for {ticker}: {e}")

        # Try yfinance
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                self._price_cache[ticker] = price
                return price
        except Exception:
            pass

        # Return None if no price found
        return None

    def set_price(self, ticker: str, price: float):
        """Set price for ticker (for testing)."""
        self._price_cache[ticker] = price

    def update_prices(self, prices: Dict[str, float]):
        """Update prices for multiple tickers."""
        self._price_cache.update(prices)

        # Update position values
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.update_price(prices[ticker])

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for ticker."""
        return self.positions.get(ticker)

    def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    def get_cash(self) -> float:
        """Get available cash."""
        return self.cash

    def get_buying_power(self) -> float:
        """Get buying power (same as cash for now)."""
        return self.cash

    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary."""
        portfolio_value = self.get_portfolio_value()
        positions_value = sum(p.market_value for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        realized_pnl = sum(p.realized_pnl for p in self.positions.values())

        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': unrealized_pnl + realized_pnl,
            'return_pct': (portfolio_value - self.initial_capital) / self.initial_capital,
            'n_positions': len(self.positions),
        }

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]

        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False

        if self.use_alpaca and 'alpaca_order_id' in order.metadata:
            try:
                self.alpaca_client.cancel_order_by_id(order.metadata['alpaca_order_id'])
            except Exception as e:
                logger.warning(f"Failed to cancel Alpaca order: {e}")

        order.status = OrderStatus.CANCELLED
        return True

    def cancel_all_orders(self):
        """Cancel all pending orders."""
        for order_id in list(self.orders.keys()):
            self.cancel_order(order_id)

    def close_position(self, ticker: str) -> Optional[Order]:
        """Close position in ticker."""
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        return self.submit_order(
            ticker=ticker,
            side='sell',
            quantity=pos.quantity,
            order_type='market',
        )

    def close_all_positions(self) -> List[Order]:
        """Close all positions."""
        orders = []
        for ticker in list(self.positions.keys()):
            order = self.close_position(ticker)
            if order:
                orders.append(order)
        return orders

    def sync_from_alpaca(self):
        """Sync positions from Alpaca."""
        if not self.use_alpaca:
            return

        try:
            # Get account
            account = self.alpaca_client.get_account()
            self.cash = float(account.cash)

            # Get positions
            alpaca_positions = self.alpaca_client.get_all_positions()

            self.positions = {}
            for ap in alpaca_positions:
                self.positions[ap.symbol] = Position(
                    ticker=ap.symbol,
                    quantity=int(ap.qty),
                    avg_cost=float(ap.avg_entry_price),
                    current_price=float(ap.current_price),
                    market_value=float(ap.market_value),
                    unrealized_pnl=float(ap.unrealized_pl),
                )

            logger.info(f"Synced {len(self.positions)} positions from Alpaca")

        except Exception as e:
            logger.error(f"Failed to sync from Alpaca: {e}")

    def reset(self):
        """Reset broker to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.order_history = []
        self.trade_history = []
        self._order_counter = 0
        self._price_cache = {}

        if self.use_alpaca:
            self.sync_from_alpaca()


def create_broker(
    use_alpaca: bool = False,
    initial_capital: float = 100000,
) -> PaperBroker:
    """
    Factory function to create broker.

    Args:
        use_alpaca: Use Alpaca paper trading
        initial_capital: Initial capital (ignored if using Alpaca)

    Returns:
        PaperBroker instance
    """
    return PaperBroker(
        initial_capital=initial_capital,
        use_alpaca=use_alpaca,
    )
