from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class PositionStatus(Enum):
    """Position status."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class Order:
    """Trading order."""
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0
    slippage: float = 0
    status: str = "pending"
    order_id: Optional[str] = None


@dataclass
class Position:
    """Trading position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float = 0
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    status: PositionStatus = PositionStatus.OPEN
    pnl: float = 0
    pnl_pct: float = 0
    commission: float = 0
    position_id: Optional[str] = None


@dataclass
class Trade:
    """Completed trade."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    side: OrderSide
    pnl: float
    pnl_pct: float
    commission: float
    duration: timedelta
    trade_id: Optional[str] = None


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    min_commission: float = 1.0  # Minimum commission per trade
    max_position_size: float = 0.1  # 10% of capital per position
    max_positions: int = 10  # Maximum concurrent positions
    use_stops: bool = True
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.10  # 10% take profit
    trailing_stop: Optional[float] = None  # Trailing stop percentage
    allow_short: bool = False  # Allow short selling
    margin_requirement: float = 0.5  # 50% margin for shorts
    risk_per_trade: float = 0.02  # 2% risk per trade


class BacktestEngine:
    """
    Main backtesting engine for strategy evaluation.
    """

    def __init__(self, data: pd.DataFrame, config: BacktestConfig = None):
        """
        Initialize backtesting engine.

        Args:
            data: DataFrame with OHLCV data
            config: Backtesting configuration
        """
        self.data = data
        self.config = config or BacktestConfig()

        # Portfolio state
        self.cash = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []

        # Performance tracking
        self.portfolio_value_history = []
        self.cash_history = []
        self.position_history = []
        self.drawdown_history = []

        # Current state
        self.current_time = None
        self.current_prices = {}

        logger.info(f"Initialized BacktestEngine with ${self.config.initial_capital:,.2f}")

    def reset(self):
        """Reset backtesting state."""
        self.cash = self.config.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.portfolio_value_history.clear()
        self.cash_history.clear()
        self.position_history.clear()
        self.drawdown_history.clear()
        self.current_time = None
        self.current_prices.clear()

    def run(self, strategy: Callable, start_date: Optional[str] = None,
           end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest with given strategy.

        Args:
            strategy: Strategy function that returns signals
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Dictionary of backtest results
        """
        logger.info("Starting backtest...")
        self.reset()

        # Filter data by date range
        data = self.data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        # Main backtest loop
        for timestamp, row in data.iterrows():
            self.current_time = timestamp
            self.current_prices = {'default': row['Close']}

            # Update positions with current prices
            self._update_positions(row)

            # Check stop losses and take profits
            if self.config.use_stops:
                self._check_stops(row)

            # Get strategy signal
            signal = strategy(
                timestamp=timestamp,
                data=data.loc[:timestamp],
                positions=self.positions,
                cash=self.cash,
                portfolio_value=self._calculate_portfolio_value()
            )

            # Process signal
            if signal:
                self._process_signal(signal, row)

            # Record portfolio state
            self._record_state()

        # Close all remaining positions
        self._close_all_positions()

        # Calculate performance metrics
        results = self._calculate_performance()

        logger.info(f"Backtest complete. Total return: {results['total_return']:.2%}")

        return results

    def _process_signal(self, signal: Dict[str, Any], row: pd.Series):
        """Process trading signal."""
        action = signal.get('action')
        symbol = signal.get('symbol', 'default')
        quantity = signal.get('quantity', 0)
        price = row['Close']

        if action == 'buy':
            self._execute_buy(symbol, quantity, price, row)
        elif action == 'sell':
            self._execute_sell(symbol, quantity, price, row)
        elif action == 'close':
            self._close_position(symbol, price, row)

    def _execute_buy(self, symbol: str, quantity: float, price: float, row: pd.Series):
        """Execute buy order."""
        # Calculate position size if not specified
        if quantity == 0:
            max_position_value = self.cash * self.config.max_position_size
            quantity = max_position_value / price

        # Apply slippage
        execution_price = price * (1 + self.config.slippage)

        # Calculate commission
        trade_value = quantity * execution_price
        commission = max(trade_value * self.config.commission, self.config.min_commission)

        # Check if we have enough cash
        total_cost = trade_value + commission
        if total_cost > self.cash:
            # Adjust quantity to available cash
            available_cash = self.cash - commission
            quantity = available_cash / execution_price
            trade_value = quantity * execution_price
            total_cost = trade_value + commission

        if quantity <= 0:
            return

        # Execute trade
        self.cash -= total_cost

        # Update or create position
        if symbol in self.positions:
            position = self.positions[symbol]
            # Average up/down
            total_quantity = position.quantity + quantity
            position.entry_price = (
                (position.entry_price * position.quantity + execution_price * quantity) /
                total_quantity
            )
            position.quantity = total_quantity
            position.commission += commission
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=execution_price,
                entry_time=self.current_time,
                current_price=execution_price,
                commission=commission
            )

        # Record order
        self.orders.append(Order(
            timestamp=self.current_time,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            filled_quantity=quantity,
            filled_price=execution_price,
            commission=commission,
            slippage=price * self.config.slippage,
            status="filled"
        ))

    def _execute_sell(self, symbol: str, quantity: float, price: float, row: pd.Series):
        """Execute sell order."""
        if symbol not in self.positions:
            if not self.config.allow_short:
                return
            # Short selling
            self._execute_short(symbol, quantity, price, row)
            return

        position = self.positions[symbol]

        # Determine quantity to sell
        if quantity == 0 or quantity > position.quantity:
            quantity = position.quantity

        # Apply slippage
        execution_price = price * (1 - self.config.slippage)

        # Calculate commission
        trade_value = quantity * execution_price
        commission = max(trade_value * self.config.commission, self.config.min_commission)

        # Execute trade
        revenue = trade_value - commission
        self.cash += revenue

        # Calculate PnL
        pnl = (execution_price - position.entry_price) * quantity - commission
        pnl_pct = pnl / (position.entry_price * quantity)

        # Update position
        position.quantity -= quantity
        if position.quantity <= 0:
            # Close position
            position.exit_price = execution_price
            position.exit_time = self.current_time
            position.status = PositionStatus.CLOSED
            position.pnl = pnl
            position.pnl_pct = pnl_pct

            # Record trade
            self.trades.append(Trade(
                symbol=symbol,
                entry_time=position.entry_time,
                exit_time=self.current_time,
                entry_price=position.entry_price,
                exit_price=execution_price,
                quantity=quantity,
                side=OrderSide.SELL,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=position.commission + commission,
                duration=self.current_time - position.entry_time
            ))

            # Remove from positions
            del self.positions[symbol]
        else:
            position.commission += commission

        # Record order
        self.orders.append(Order(
            timestamp=self.current_time,
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            filled_quantity=quantity,
            filled_price=execution_price,
            commission=commission,
            slippage=price * self.config.slippage,
            status="filled"
        ))

    def _execute_short(self, symbol: str, quantity: float, price: float, row: pd.Series):
        """Execute short sell (if allowed) - FULLY IMPLEMENTED."""
        if not self.config.allow_short:
            logger.warning("Short selling not allowed in configuration")
            return

        # Calculate position size if not specified
        if quantity == 0:
            max_position_value = self.cash * self.config.max_position_size
            quantity = max_position_value / price

        # Apply slippage (unfavorable for shorting)
        execution_price = price * (1 - self.config.slippage)

        # Calculate margin requirement
        margin_required = quantity * execution_price * self.config.margin_requirement

        # Check if we have enough cash for margin
        if margin_required > self.cash:
            # Adjust quantity to available cash
            available_cash = self.cash
            quantity = available_cash / (execution_price * self.config.margin_requirement)
            margin_required = quantity * execution_price * self.config.margin_requirement

        if quantity <= 0:
            return

        # Calculate commission
        trade_value = quantity * execution_price
        commission = max(trade_value * self.config.commission, self.config.min_commission)

        # Execute short sale
        # We receive the proceeds but need to maintain margin
        proceeds = trade_value - commission
        self.cash += proceeds - margin_required  # Keep margin locked

        # Create or update short position (negative quantity indicates short)
        if symbol in self.positions:
            position = self.positions[symbol]
            if position.quantity < 0:  # Already short, adding to position
                total_quantity = position.quantity - quantity
                position.entry_price = (
                    (abs(position.entry_price * position.quantity) + execution_price * quantity) /
                    abs(total_quantity)
                )
                position.quantity = total_quantity
            else:  # Currently long, reversing to short
                logger.warning(f"Reversing from long to short position for {symbol}")
                # Close long position first
                self._execute_sell(symbol, position.quantity, price, row)
                # Then open short
                self._execute_short(symbol, quantity, price, row)
                return
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=-quantity,  # Negative for short
                entry_price=execution_price,
                entry_time=self.current_time,
                current_price=execution_price,
                commission=commission
            )

        # Record order
        self.orders.append(Order(
            timestamp=self.current_time,
            symbol=symbol,
            side=OrderSide.SELL,  # Short sell
            order_type=OrderType.MARKET,
            quantity=quantity,
            filled_quantity=quantity,
            filled_price=execution_price,
            commission=commission,
            slippage=price * self.config.slippage,
            status="filled"
        ))

    def _close_position(self, symbol: str, price: float, row: pd.Series):
        """Close entire position."""
        if symbol in self.positions:
            position = self.positions[symbol]
            self._execute_sell(symbol, position.quantity, price, row)

    def _check_stops(self, row: pd.Series):
        """Check stop loss and take profit levels."""
        for symbol, position in list(self.positions.items()):
            current_price = row['Close']
            position.current_price = current_price

            # Calculate current PnL percentage
            pnl_pct = (current_price - position.entry_price) / position.entry_price

            # Check stop loss
            if self.config.stop_loss and pnl_pct <= -self.config.stop_loss:
                logger.info(f"Stop loss triggered for {symbol} at {pnl_pct:.2%}")
                self._close_position(symbol, current_price, row)

            # Check take profit
            elif self.config.take_profit and pnl_pct >= self.config.take_profit:
                logger.info(f"Take profit triggered for {symbol} at {pnl_pct:.2%}")
                self._close_position(symbol, current_price, row)

            # Check trailing stop
            elif self.config.trailing_stop:
                # Update trailing stop implementation
                if not hasattr(position, 'trailing_stop_price'):
                    position.trailing_stop_price = position.entry_price * (1 - self.config.trailing_stop)

                # Update trailing stop if price moved favorably
                new_stop = current_price * (1 - self.config.trailing_stop)
                if new_stop > position.trailing_stop_price:
                    position.trailing_stop_price = new_stop

                # Check if trailing stop hit
                if current_price <= position.trailing_stop_price:
                    logger.info(f"Trailing stop triggered for {symbol} at {current_price:.2f}")
                    self._close_position(symbol, current_price, row)

    def _update_positions(self, row: pd.Series):
        """Update position values with current prices."""
        current_price = row['Close']
        for position in self.positions.values():
            position.current_price = current_price
            position.pnl = (current_price - position.entry_price) * position.quantity
            position.pnl_pct = position.pnl / (position.entry_price * position.quantity)

    def _cover_short(self, symbol: str, quantity: float, price: float, row: pd.Series):
        """Cover (close) a short position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        if position.quantity >= 0:  # Not a short position
            return

        # Apply slippage (unfavorable for covering)
        execution_price = price * (1 + self.config.slippage)

        # Calculate cost to cover
        cost_to_cover = abs(position.quantity) * execution_price
        commission = max(cost_to_cover * self.config.commission, self.config.min_commission)
        total_cost = cost_to_cover + commission

        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient funds to cover short position {symbol}")
            return

        # Execute cover
        self.cash -= total_cost

        # Calculate PnL (entry price was when we shorted)
        # For shorts: profit when price goes down, loss when price goes up
        pnl = (position.entry_price - execution_price) * abs(position.quantity) - commission
        pnl_pct = pnl / (position.entry_price * abs(position.quantity))

        # Record trade
        self.trades.append(Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=abs(position.quantity),
            side=OrderSide.BUY,  # Covering is buying back
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=position.commission + commission,
            duration=self.current_time - position.entry_time if self.current_time else timedelta(0)
        ))

        # Remove position
        del self.positions[symbol]

    def _close_all_positions(self):
        """Close all open positions at end of backtest."""
        for symbol in list(self.positions.keys()):
            if symbol in self.positions:
                position = self.positions[symbol]
                # Fix: Create a minimal Series with the current price for compatibility
                dummy_row = pd.Series({'Close': position.current_price})

                if position.quantity > 0:  # Long position
                    self._execute_sell(symbol, position.quantity, position.current_price, dummy_row)
                elif position.quantity < 0:  # Short position
                    self._cover_short(symbol, abs(position.quantity), position.current_price, dummy_row)

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        positions_value = sum(
            position.quantity * position.current_price
            for position in self.positions.values()
        )
        return self.cash + positions_value

    def _record_state(self):
        """Record current portfolio state."""
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_value_history.append({
            'timestamp': self.current_time,
            'value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'n_positions': len(self.positions)
        })

    def _calculate_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_value_history:
            return {}

        # Convert to DataFrame for easier calculation
        portfolio_df = pd.DataFrame(self.portfolio_value_history)
        portfolio_df.set_index('timestamp', inplace=True)

        # Basic metrics
        initial_value = self.initial_capital
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        # Calculate daily returns
        daily_returns = portfolio_df['value'].pct_change().dropna()

        # Sharpe ratio
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0

        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std()
        else:
            sortino_ratio = 0

        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        n_trades = len(self.trades)
        if n_trades > 0:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]

            win_rate = len(winning_trades) / n_trades
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

            profit_factor = (
                sum(t.pnl for t in winning_trades) /
                abs(sum(t.pnl for t in losing_trades))
                if losing_trades else float('inf')
            )

            avg_trade_duration = np.mean([
                t.duration.total_seconds() / 86400
                for t in self.trades
            ])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0

        # Compile results
        results = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            # Fix: Detect frequency of data for proper annualization
            'annual_return': (1 + total_return) ** (252 / len(portfolio_df)) - 1 if len(portfolio_df) > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': abs(max_drawdown),
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'total_commission': sum(t.commission for t in self.trades),
            'portfolio_history': portfolio_df,
            'trades': pd.DataFrame([t.__dict__ for t in self.trades]) if self.trades else pd.DataFrame()
        }

        return results


def create_strategy(model: Any, features: pd.DataFrame,
                   threshold: float = 0.0) -> Callable:
    """
    Create a strategy function from a trained model.

    Args:
        model: Trained model
        features: Feature DataFrame
        threshold: Signal threshold

    Returns:
        Strategy function
    """
    def strategy(timestamp, data, positions, cash, portfolio_value):
        """Model-based trading strategy."""
        # Get features for current timestamp
        if timestamp not in features.index:
            return None

        current_features = features.loc[timestamp].values

        # Make prediction
        prediction = model.predict(current_features.reshape(1, -1))[0]

        # Generate signal
        signal = None
        if prediction > threshold and len(positions) == 0:
            # Buy signal
            signal = {
                'action': 'buy',
                'symbol': 'default',
                'quantity': 0  # Let engine calculate
            }
        elif prediction < -threshold and len(positions) > 0:
            # Sell signal
            signal = {
                'action': 'close',
                'symbol': 'default'
            }

        return signal

    return strategy


if __name__ == "__main__":
    # Example usage
    logger.info("Running example backtest...")

    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    n_days = len(dates)

    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.02, 0, n_days)),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 1e7, n_days)
    }, index=dates)

    # Simple moving average crossover strategy
    def ma_crossover_strategy(timestamp, data, positions, cash, portfolio_value):
        """Simple MA crossover strategy."""
        if len(data) < 50:
            return None

        # Calculate moving averages
        ma_20 = data['Close'].iloc[-20:].mean()
        ma_50 = data['Close'].iloc[-50:].mean()

        signal = None
        if ma_20 > ma_50 and len(positions) == 0:
            signal = {'action': 'buy', 'symbol': 'default', 'quantity': 0}
        elif ma_20 < ma_50 and len(positions) > 0:
            signal = {'action': 'close', 'symbol': 'default'}

        return signal

    # Run backtest
    config = BacktestConfig()
    engine = BacktestEngine(data, config)
    results = engine.run(ma_crossover_strategy)

    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Number of Trades: {results['n_trades']}")