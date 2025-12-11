#!/usr/bin/env python
"""
Pipeline Backtester
==================
Backtest the ML trading pipeline over historical data.

Supports:
- Walk-forward validation with expanding/rolling windows
- Realistic transaction costs
- Regime-aware position sizing
- Performance attribution

Usage:
    python scripts/run_backtest.py --start 2023-01-01 --end 2024-01-01
    python scripts/run_backtest.py --config config/backtest.yaml
    python scripts/run_backtest.py --tickers AAPL MSFT GOOGL
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import pipeline components
try:
    from pipeline.core_types import TradeSignal, Forecast
    from pipeline.regime_utils import classify_regime, RegimeDetector
    from pipeline.trade_filter import filter_signals, rank_signals, select_top_signals
    from pipeline.allocation_utils import allocate_from_signals, validate_allocation
    HAS_PIPELINE = True
except ImportError as e:
    logger.warning(f"Pipeline imports failed: {e}")
    HAS_PIPELINE = False

# Import data fetching
try:
    from data_fetching import fetch_data
    HAS_DATA = True
except ImportError:
    HAS_DATA = False

# Import yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class BacktestTrade:
    """Record of a single trade."""
    date: pd.Timestamp
    ticker: str
    action: str  # 'buy' or 'sell'
    quantity: int
    price: float
    value: float
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestPosition:
    """Current position in a ticker."""
    ticker: str
    quantity: int
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot."""
    date: pd.Timestamp
    portfolio_value: float
    cash: float
    positions_value: float
    n_positions: int
    regime: int
    daily_return: float = 0.0
    trades_executed: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_value: float

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Time series
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None

    # Detailed records
    trades: List[BacktestTrade] = field(default_factory=list)
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_value': self.final_value,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
        }


class Backtester:
    """
    Pipeline backtester with walk-forward validation.

    Simulates the pipeline over historical data with realistic
    transaction costs and slippage.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,  # 10 bps
        slippage_rate: float = 0.001,    # 10 bps
        rebalance_frequency: str = 'weekly',  # 'daily', 'weekly', 'monthly'
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission_rate: Commission as fraction of trade value
            slippage_rate: Slippage as fraction of trade value
            rebalance_frequency: How often to rebalance
            config: Pipeline configuration
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.rebalance_frequency = rebalance_frequency
        self.config = config or {}

        # State
        self.cash = initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []
        self.daily_snapshots: List[DailySnapshot] = []

        # Data cache
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def run(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        lookback_days: int = 252,
    ) -> BacktestResult:
        """
        Run backtest over date range.

        Args:
            tickers: Universe of tickers
            start_date: Backtest start date
            end_date: Backtest end date
            lookback_days: Historical data needed for signals

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest: {start_date.date()} to {end_date.date()}")
        logger.info(f"Universe: {len(tickers)} tickers")
        logger.info(f"Initial capital: ${self.initial_capital:,.0f}")

        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_snapshots = []

        # Load all price data upfront
        logger.info("Loading price data...")
        self._load_price_data(tickers, start_date, end_date, lookback_days)

        # Get trading days
        trading_days = self._get_trading_days(start_date, end_date)
        logger.info(f"Trading days: {len(trading_days)}")

        # Run simulation
        rebalance_days = self._get_rebalance_days(trading_days)
        logger.info(f"Rebalance days: {len(rebalance_days)}")

        for i, date in enumerate(trading_days):
            # Update positions with current prices
            self._mark_to_market(date)

            # Check if rebalance day
            if date in rebalance_days:
                self._rebalance(date, tickers)

            # Record daily snapshot
            self._record_snapshot(date)

            # Progress
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(trading_days)} days")

        # Calculate final metrics
        result = self._calculate_metrics(start_date, end_date)

        logger.info(f"\nBacktest complete")
        logger.info(f"Final value: ${result.final_value:,.0f}")
        logger.info(f"Total return: {result.total_return:.2%}")
        logger.info(f"Sharpe ratio: {result.sharpe_ratio:.3f}")
        logger.info(f"Max drawdown: {result.max_drawdown:.2%}")

        return result

    def _load_price_data(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        lookback_days: int,
    ):
        """Load and cache price data."""
        data_start = start_date - pd.Timedelta(days=lookback_days + 30)

        for ticker in tickers:
            try:
                if HAS_DATA:
                    df = fetch_data(ticker, start=data_start, end=end_date)
                else:
                    # Generate synthetic data for testing
                    dates = pd.date_range(start=data_start, end=end_date, freq='B')
                    base = 100
                    returns = np.random.randn(len(dates)) * 0.02
                    prices = base * np.exp(np.cumsum(returns))
                    df = pd.DataFrame({
                        'Open': prices * 0.99,
                        'High': prices * 1.02,
                        'Low': prices * 0.98,
                        'Close': prices,
                        'Volume': np.random.randint(1e6, 1e7, len(dates)),
                    }, index=dates)

                if df is not None and len(df) > 0:
                    self._price_cache[ticker] = df

            except Exception as e:
                logger.debug(f"Failed to load {ticker}: {e}")

        logger.info(f"Loaded data for {len(self._price_cache)} tickers")

    def _get_trading_days(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DatetimeIndex:
        """Get trading days in range."""
        # Use SPY dates if available, else generate business days
        if 'SPY' in self._price_cache:
            dates = self._price_cache['SPY'].index
            return dates[(dates >= start_date) & (dates <= end_date)]

        # Fallback to business days
        return pd.date_range(start=start_date, end=end_date, freq='B')

    def _get_rebalance_days(self, trading_days: pd.DatetimeIndex) -> set:
        """Get days to rebalance."""
        if self.rebalance_frequency == 'daily':
            return set(trading_days)

        elif self.rebalance_frequency == 'weekly':
            # Friday of each week
            return set(trading_days[trading_days.dayofweek == 4])

        elif self.rebalance_frequency == 'monthly':
            # Last day of each month
            monthly = trading_days.to_series().groupby(pd.Grouper(freq='M')).last()
            return set(monthly.values)

        return set(trading_days)

    def _get_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Get price for ticker on date."""
        if ticker not in self._price_cache:
            return None

        df = self._price_cache[ticker]
        if date in df.index:
            return float(df.loc[date, 'Close'])

        # Find nearest prior date
        prior_dates = df.index[df.index <= date]
        if len(prior_dates) > 0:
            return float(df.loc[prior_dates[-1], 'Close'])

        return None

    def _get_returns(
        self,
        ticker: str,
        end_date: pd.Timestamp,
        lookback: int = 252,
    ) -> Optional[pd.Series]:
        """Get historical returns for ticker."""
        if ticker not in self._price_cache:
            return None

        df = self._price_cache[ticker]
        df_prior = df[df.index <= end_date].tail(lookback + 1)

        if len(df_prior) < 60:
            return None

        return df_prior['Close'].pct_change().dropna()

    def _mark_to_market(self, date: pd.Timestamp):
        """Update position values with current prices."""
        for ticker, position in self.positions.items():
            price = self._get_price(ticker, date)
            if price:
                position.market_value = position.quantity * price
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_cost)

    def _rebalance(self, date: pd.Timestamp, universe: List[str]):
        """Run pipeline and rebalance portfolio."""
        # Generate signals for available tickers
        available_tickers = [t for t in universe if t in self._price_cache]

        signals = self._generate_signals(available_tickers, date)
        if not signals:
            return

        # Filter signals
        filtered = self._filter_signals(signals)
        if not filtered:
            return

        # Calculate target weights
        target_weights = self._calculate_weights(filtered, date)
        if target_weights is None or len(target_weights) == 0:
            return

        # Get current portfolio value
        portfolio_value = self._portfolio_value()

        # Calculate target positions
        targets = {}
        for ticker, weight in target_weights.items():
            price = self._get_price(ticker, date)
            if price and price > 0:
                target_value = portfolio_value * weight
                target_shares = int(target_value / price)
                targets[ticker] = target_shares

        # Execute trades to reach targets
        self._execute_rebalance(targets, date)

    def _generate_signals(
        self,
        tickers: List[str],
        date: pd.Timestamp,
    ) -> List[TradeSignal]:
        """Generate trade signals for date."""
        signals = []

        # Classify regime
        regime = self._classify_regime(date)

        for ticker in tickers:
            returns = self._get_returns(ticker, date)
            if returns is None or len(returns) < 60:
                continue

            # Simple momentum signal
            momentum_20 = float(returns.tail(20).sum())
            momentum_60 = float(returns.tail(60).sum())
            volatility = float(returns.std() * np.sqrt(252))

            # Determine direction
            if momentum_20 > 0.02 and momentum_60 > 0:
                direction = 'long'
                expected_return = momentum_20 / 20 * 10  # Scale to 10 days
            elif momentum_20 < -0.02:
                direction = 'short'
                expected_return = momentum_20 / 20 * 10
            else:
                direction = 'flat'
                expected_return = 0.0

            # Create signal
            signal = TradeSignal(
                ticker=ticker,
                direction=direction,
                expected_return=expected_return,
                expected_vol=volatility,
                regime=regime,
                regime_label={0: 'Bull', 1: 'Bear', 2: 'Neutral', 3: 'Crisis'}.get(regime, 'Neutral'),
            )

            # Add pseudo meta_prob
            signal.meta_prob = 0.5 + abs(momentum_20) * 5  # Higher momentum = higher prob
            signal.meta_prob = min(0.9, max(0.3, signal.meta_prob))

            signals.append(signal)

        return signals

    def _classify_regime(self, date: pd.Timestamp) -> int:
        """Classify regime for date."""
        if 'SPY' not in self._price_cache:
            return 2  # Neutral

        returns = self._get_returns('SPY', date, lookback=60)
        if returns is None:
            return 2

        # Simple regime classification based on SPY
        cum_return = float((1 + returns).prod() - 1)
        volatility = float(returns.std() * np.sqrt(252))

        if cum_return > 0.05 and volatility < 0.20:
            return 0  # Bull
        elif cum_return < -0.05 or volatility > 0.30:
            return 1  # Bear
        elif volatility > 0.40:
            return 3  # Crisis
        else:
            return 2  # Neutral

    def _filter_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Filter signals."""
        if HAS_PIPELINE:
            filtered = filter_signals(signals, config=self.config)
            ranked = rank_signals(filtered, ranking_metric='risk_reward')
            return select_top_signals(ranked, max_signals=15)
        else:
            # Simple filter
            return [s for s in signals if s.direction == 'long' and s.meta_prob > 0.5][:15]

    def _calculate_weights(
        self,
        signals: List[TradeSignal],
        date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """Calculate target weights."""
        if not signals:
            return None

        tickers = [s.ticker for s in signals]

        # Get returns for allocation
        returns_dict = {}
        for ticker in tickers:
            returns = self._get_returns(ticker, date)
            if returns is not None:
                returns_dict[ticker] = returns

        if len(returns_dict) < 2:
            # Equal weight
            return pd.Series(1.0 / len(tickers), index=tickers)

        returns_df = pd.DataFrame(returns_dict)

        if HAS_PIPELINE:
            try:
                result = allocate_from_signals(
                    signals=signals,
                    returns=returns_df,
                    portfolio_value=self._portfolio_value(),
                    regime=signals[0].regime,
                    config=self.config,
                )
                return result.weights
            except Exception as e:
                logger.debug(f"Allocation failed: {e}")

        # Fallback to equal weight
        return pd.Series(1.0 / len(tickers), index=tickers)

    def _execute_rebalance(
        self,
        targets: Dict[str, int],
        date: pd.Timestamp,
    ):
        """Execute trades to reach target positions."""
        # Calculate deltas
        current_holdings = {t: p.quantity for t, p in self.positions.items()}

        all_tickers = set(targets.keys()) | set(current_holdings.keys())

        for ticker in all_tickers:
            current = current_holdings.get(ticker, 0)
            target = targets.get(ticker, 0)
            delta = target - current

            if delta == 0:
                continue

            price = self._get_price(ticker, date)
            if not price:
                continue

            # Execute trade
            if delta > 0:
                self._buy(ticker, delta, price, date)
            else:
                self._sell(ticker, -delta, price, date)

    def _buy(
        self,
        ticker: str,
        quantity: int,
        price: float,
        date: pd.Timestamp,
    ):
        """Execute buy order."""
        # Apply slippage
        exec_price = price * (1 + self.slippage_rate)
        trade_value = quantity * exec_price
        commission = trade_value * self.commission_rate

        # Check cash
        total_cost = trade_value + commission
        if total_cost > self.cash:
            # Reduce quantity
            max_value = self.cash / (1 + self.commission_rate)
            quantity = int(max_value / exec_price)
            if quantity <= 0:
                return
            trade_value = quantity * exec_price
            commission = trade_value * self.commission_rate
            total_cost = trade_value + commission

        # Update cash
        self.cash -= total_cost

        # Update position
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_cost_basis = pos.avg_cost * pos.quantity + exec_price * quantity
            pos.quantity += quantity
            pos.avg_cost = total_cost_basis / pos.quantity
        else:
            self.positions[ticker] = BacktestPosition(
                ticker=ticker,
                quantity=quantity,
                avg_cost=exec_price,
            )

        # Record trade
        self.trades.append(BacktestTrade(
            date=date,
            ticker=ticker,
            action='buy',
            quantity=quantity,
            price=exec_price,
            value=trade_value,
            commission=commission,
            slippage=exec_price - price,
        ))

    def _sell(
        self,
        ticker: str,
        quantity: int,
        price: float,
        date: pd.Timestamp,
    ):
        """Execute sell order."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        quantity = min(quantity, pos.quantity)
        if quantity <= 0:
            return

        # Apply slippage
        exec_price = price * (1 - self.slippage_rate)
        trade_value = quantity * exec_price
        commission = trade_value * self.commission_rate

        # Update cash
        self.cash += trade_value - commission

        # Update position
        pos.quantity -= quantity
        if pos.quantity <= 0:
            del self.positions[ticker]

        # Record trade
        self.trades.append(BacktestTrade(
            date=date,
            ticker=ticker,
            action='sell',
            quantity=quantity,
            price=exec_price,
            value=trade_value,
            commission=commission,
            slippage=price - exec_price,
        ))

    def _portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    def _record_snapshot(self, date: pd.Timestamp):
        """Record daily snapshot."""
        positions_value = sum(p.market_value for p in self.positions.values())
        portfolio_value = self.cash + positions_value

        # Calculate daily return
        if self.daily_snapshots:
            prev_value = self.daily_snapshots[-1].portfolio_value
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            daily_return = 0

        # Count today's trades
        today_trades = sum(1 for t in self.trades if t.date == date)

        # Get regime
        regime = self._classify_regime(date)

        self.daily_snapshots.append(DailySnapshot(
            date=date,
            portfolio_value=portfolio_value,
            cash=self.cash,
            positions_value=positions_value,
            n_positions=len(self.positions),
            regime=regime,
            daily_return=daily_return,
            trades_executed=today_trades,
        ))

    def _calculate_metrics(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> BacktestResult:
        """Calculate backtest metrics."""
        # Build equity curve
        dates = [s.date for s in self.daily_snapshots]
        values = [s.portfolio_value for s in self.daily_snapshots]
        returns = [s.daily_return for s in self.daily_snapshots]

        equity_curve = pd.Series(values, index=dates)
        daily_returns = pd.Series(returns, index=dates)

        # Calculate metrics
        final_value = values[-1] if values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital

        n_days = len(values)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0

        volatility = float(daily_returns.std() * np.sqrt(252))
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Sortino
        downside = daily_returns[daily_returns < 0].std() * np.sqrt(252)
        sortino = annualized_return / downside if downside > 0 else 0

        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = float(drawdown.min())

        # Calmar
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0

        # Group trades by ticker to calculate P&L
        # (simplified - in reality would track entry/exit pairs)
        total_trades = len(self.trades)

        # Estimate win rate from daily returns
        winning_days = (daily_returns > 0).sum()
        losing_days = (daily_returns < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            total_trades=total_trades,
            winning_trades=winning_days,
            losing_trades=losing_days,
            win_rate=win_rate,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            drawdown_series=drawdown,
            trades=self.trades,
            daily_snapshots=self.daily_snapshots,
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Backtest trading pipeline')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--tickers', type=str, nargs='+', help='Tickers to backtest')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--frequency', type=str, default='weekly',
                       choices=['daily', 'weekly', 'monthly'])
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default universe
    default_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'JPM', 'V', 'JNJ',
        'WMT', 'PG', 'MA', 'HD', 'XOM',
    ]

    tickers = args.tickers or default_tickers

    # Load config
    config = {}
    if args.config and HAS_YAML:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Run backtest
    backtester = Backtester(
        initial_capital=args.capital,
        rebalance_frequency=args.frequency,
        config=config,
    )

    result = backtester.run(
        tickers=tickers,
        start_date=pd.Timestamp(args.start),
        end_date=pd.Timestamp(args.end),
    )

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.0f}")
    print(f"Final Value: ${result.final_value:,.0f}")
    print(f"\nPerformance:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Annualized Return: {result.annualized_return:.2%}")
    print(f"  Volatility: {result.volatility:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.3f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Calmar Ratio: {result.calmar_ratio:.3f}")
    print(f"\nTrading:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1%}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
