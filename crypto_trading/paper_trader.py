"""
Paper trading engine with virtual balance simulation.
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from crypto_trading.signal_engine import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    entry_time: datetime

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity


@dataclass
class TradeExecution:
    """Trade execution details."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    amount_usd: float
    quantity: float
    execution_price: float
    timestamp: datetime
    commission: float
    slippage: float


class PaperTradingEngine:
    """Paper trading simulation with virtual balance."""

    def __init__(
        self,
        initial_balance: float,
        base_slippage: float = 0.001,
        commission: float = 0.0026
    ):
        """
        Initialize paper trading engine.

        Args:
            initial_balance: Starting balance in USD
            base_slippage: Base slippage (0.1%)
            commission: Commission rate (0.26% Kraken taker)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.base_slippage = base_slippage
        self.commission = commission
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeExecution] = []

    def get_balance(self) -> float:
        """Get current cash balance."""
        return self.balance

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def execute(
        self,
        signal: TradeSignal,
        amount_usd: Optional[float] = None
    ) -> Optional[TradeExecution]:
        """
        Execute trade based on signal.

        Args:
            signal: Trade signal
            amount_usd: Amount in USD to trade (required for BUY, optional for SELL)

        Returns:
            TradeExecution if successful, None otherwise
        """
        if signal.direction == "BUY":
            if amount_usd is None:
                logger.error(f"amount_usd required for BUY signal")
                return None
            return self._execute_buy(signal, amount_usd)
        elif signal.direction == "SELL":
            return self._execute_sell(signal, amount_usd)
        else:
            logger.error(f"Invalid signal direction: {signal.direction}")
            return None

    def _execute_buy(
        self,
        signal: TradeSignal,
        amount_usd: float
    ) -> Optional[TradeExecution]:
        """Execute buy order."""
        # Check if we have enough balance
        if amount_usd > self.balance:
            logger.warning(
                f"Insufficient balance for {signal.symbol}: "
                f"need ${amount_usd:.2f}, have ${self.balance:.2f}"
            )
            return None

        # Calculate slippage (higher for larger orders)
        slippage_pct = self.base_slippage + (amount_usd / 100000 * 0.001)
        slippage_pct = min(slippage_pct, 0.01)  # Cap at 1%

        # Apply slippage to price (buy at higher price)
        execution_price = signal.price * (1 + slippage_pct)

        # Calculate quantity (before commission)
        gross_quantity = amount_usd / execution_price

        # Apply commission
        commission_usd = amount_usd * self.commission
        net_amount = amount_usd - commission_usd
        quantity = net_amount / execution_price

        # Update balance
        self.balance -= amount_usd

        # Open or add to position
        if signal.symbol in self.positions:
            # Add to existing position (average price)
            pos = self.positions[signal.symbol]
            total_quantity = pos.quantity + quantity
            avg_price = (
                (pos.entry_price * pos.quantity + execution_price * quantity) /
                total_quantity
            )
            pos.quantity = total_quantity
            pos.entry_price = avg_price
        else:
            # Open new position
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                direction="LONG",
                entry_price=execution_price,
                quantity=quantity,
                entry_time=signal.timestamp
            )

        # Record execution
        execution = TradeExecution(
            symbol=signal.symbol,
            direction="BUY",
            amount_usd=amount_usd,
            quantity=quantity,
            execution_price=execution_price,
            timestamp=signal.timestamp,
            commission=commission_usd,
            slippage=slippage_pct
        )
        self.trade_history.append(execution)

        logger.info(
            f"BUY {quantity:.6f} {signal.symbol} @ ${execution_price:.2f} "
            f"(cost: ${amount_usd:.2f}, commission: ${commission_usd:.2f})"
        )

        return execution

    def _execute_sell(
        self,
        signal: TradeSignal,
        amount_usd: Optional[float] = None
    ) -> Optional[TradeExecution]:
        """Execute sell order (close position)."""
        # Check if we have a position
        if signal.symbol not in self.positions:
            logger.warning(f"No position to sell for {signal.symbol}")
            return None

        pos = self.positions[signal.symbol]

        # Calculate slippage
        position_value = pos.quantity * signal.price
        slippage_pct = self.base_slippage + (position_value / 100000 * 0.001)
        slippage_pct = min(slippage_pct, 0.01)

        # Apply slippage to price (sell at lower price)
        execution_price = signal.price * (1 - slippage_pct)

        # Calculate gross proceeds
        gross_proceeds = pos.quantity * execution_price

        # Apply commission
        commission_usd = gross_proceeds * self.commission
        net_proceeds = gross_proceeds - commission_usd

        # Calculate P&L
        cost_basis = pos.quantity * pos.entry_price
        realized_pnl = net_proceeds - cost_basis

        # Update balance
        self.balance += net_proceeds

        # Close position
        del self.positions[signal.symbol]

        # Record execution
        execution = TradeExecution(
            symbol=signal.symbol,
            direction="SELL",
            amount_usd=net_proceeds,
            quantity=pos.quantity,
            execution_price=execution_price,
            timestamp=signal.timestamp,
            commission=commission_usd,
            slippage=slippage_pct
        )
        self.trade_history.append(execution)

        logger.info(
            f"SELL {pos.quantity:.6f} {signal.symbol} @ ${execution_price:.2f} "
            f"(proceeds: ${net_proceeds:.2f}, P&L: ${realized_pnl:.2f})"
        )

        return execution

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Get total portfolio value (cash + positions).

        Args:
            current_prices: Dict mapping symbol to current price

        Returns:
            Total value in USD
        """
        position_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                position_value += pos.quantity * current_prices[symbol]

        return self.balance + position_value
