"""
Trading Strategy Module
========================
Integrates portfolio construction with live trading execution.

This module provides:
- Portfolio signal generation using CVaR optimization
- Rebalancing logic with threshold checks
- ML model toggle (on/off)
- Integration with existing portfolio tools
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioSignal:
    """Represents a portfolio rebalancing signal."""
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    trades_needed: List[Dict]  # [{symbol, side, qty, reason}]
    regime: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    # Universe
    symbols: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'])

    # Risk parameters
    rrr: float = 0.5  # Reward-to-risk ratio (0=aggressive, 1=defensive)
    max_position_pct: float = 0.25
    rebalance_threshold: float = 0.05  # 5% deviation triggers rebalance

    # Model toggles
    enable_ml: bool = False  # ML ensemble (not trained yet)
    enable_regime_detection: bool = True
    enable_feature_scoring: bool = True  # Heuristic scoring

    # Timing
    rebalance_interval_hours: int = 24  # How often to check for rebalance
    lookback_days: int = 252  # Historical data for optimization

    @classmethod
    def from_env(cls) -> 'StrategyConfig':
        """Load configuration from environment variables."""
        symbols_str = os.getenv('TRADING_SYMBOLS', 'AAPL,MSFT,GOOGL,AMZN,NVDA')
        symbols = [s.strip().upper() for s in symbols_str.split(',')]

        return cls(
            symbols=symbols,
            rrr=float(os.getenv('TRADING_RRR', '0.5')),
            max_position_pct=float(os.getenv('RISK_MAX_POSITION_PCT', '0.25')),
            rebalance_threshold=float(os.getenv('TRADING_REBALANCE_THRESHOLD', '0.05')),
            enable_ml=os.getenv('ENABLE_ML', 'false').lower() == 'true',
            enable_regime_detection=os.getenv('ENABLE_REGIME', 'true').lower() == 'true',
            enable_feature_scoring=os.getenv('ENABLE_FEATURE_SCORING', 'true').lower() == 'true',
            rebalance_interval_hours=int(os.getenv('REBALANCE_INTERVAL_HOURS', '24')),
            lookback_days=int(os.getenv('TRADING_LOOKBACK_DAYS', '252')),
        )


class TradingStrategy:
    """
    Portfolio-based trading strategy.

    Uses CVaR optimization with optional:
    - Regime detection (adjusts risk parameters)
    - Feature-based scoring (stock selection)
    - ML ensemble (when trained and enabled)
    """

    # Cache price data for 1 hour (no need to refetch 1y of data every 5 min)
    PRICE_CACHE_TTL_SECONDS = 3600

    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize strategy with configuration."""
        self.config = config or StrategyConfig.from_env()
        self._last_rebalance: Optional[datetime] = None
        self._current_weights: Dict[str, float] = {}
        self._regime_info: Optional[Dict] = None

        # Price data cache
        self._price_cache: Optional[pd.DataFrame] = None
        self._price_cache_time: Optional[datetime] = None

        logger.info("Strategy initialized:")
        logger.info(f"  Symbols: {self.config.symbols}")
        logger.info(f"  RRR: {self.config.rrr}")
        logger.info(f"  ML Enabled: {self.config.enable_ml}")
        logger.info(f"  Regime Detection: {self.config.enable_regime_detection}")
        logger.info(f"  Feature Scoring: {self.config.enable_feature_scoring}")

    def should_rebalance(self, current_positions: Dict[str, float],
                         portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if rebalancing is needed.

        Args:
            current_positions: Dict of symbol -> market value
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (should_rebalance, reason)
        """
        # Check time since last rebalance
        if self._last_rebalance:
            hours_since = (datetime.now() - self._last_rebalance).total_seconds() / 3600
            if hours_since < self.config.rebalance_interval_hours:
                return False, f"Only {hours_since:.1f}h since last rebalance"

        # If no target weights yet, need to rebalance
        if not self._current_weights:
            return True, "No target weights set"

        # Calculate current weights
        current_weights = {}
        for symbol, value in current_positions.items():
            current_weights[symbol] = value / portfolio_value if portfolio_value > 0 else 0

        # Check for drift from target
        max_drift = 0
        drifted_symbol = None
        for symbol, target in self._current_weights.items():
            current = current_weights.get(symbol, 0)
            drift = abs(current - target)
            if drift > max_drift:
                max_drift = drift
                drifted_symbol = symbol

        if max_drift > self.config.rebalance_threshold:
            return True, f"{drifted_symbol} drifted {max_drift:.1%} from target"

        return False, f"Max drift {max_drift:.1%} within threshold"

    def generate_signals(self,
                        current_positions: Dict[str, float],
                        portfolio_value: float,
                        cash: float,
                        live_quotes: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[PortfolioSignal]:
        """
        Generate portfolio rebalancing signals.

        Args:
            current_positions: Dict of symbol -> market value
            portfolio_value: Total portfolio value
            cash: Available cash
            live_quotes: Optional live quotes from Alpaca for accurate sizing

        Returns:
            PortfolioSignal if rebalancing needed, None otherwise
        """
        logger.info("Generating trading signals...")

        # Check if rebalance needed
        should_rebal, reason = self.should_rebalance(current_positions, portfolio_value)
        if not should_rebal:
            logger.info(f"No rebalance needed: {reason}")
            return None

        logger.info(f"Rebalance triggered: {reason}")

        try:
            # Fetch price data
            prices_df = self._fetch_prices()
            if prices_df is None or prices_df.empty:
                logger.error("Failed to fetch price data")
                return None

            # Detect regime (if enabled)
            regime = 'neutral'
            confidence = 0.5
            if self.config.enable_regime_detection:
                regime, confidence = self._detect_regime(prices_df)
                logger.info(f"Detected regime: {regime} (confidence: {confidence:.0%})")

            # Calculate target weights
            target_weights = self._optimize_portfolio(prices_df, regime)
            if not target_weights:
                logger.error("Portfolio optimization failed")
                return None

            # Calculate current weights
            current_weights = {}
            for symbol, value in current_positions.items():
                current_weights[symbol] = value / portfolio_value if portfolio_value > 0 else 0

            # Determine trades needed (use live quotes if available)
            trades = self._calculate_trades(
                target_weights,
                current_weights,
                portfolio_value,
                prices_df,
                live_quotes=live_quotes,
            )

            # Update state
            self._current_weights = target_weights
            self._last_rebalance = datetime.now()

            signal = PortfolioSignal(
                target_weights=target_weights,
                current_weights=current_weights,
                trades_needed=trades,
                regime=regime,
                confidence=confidence,
            )

            logger.info(f"Generated {len(trades)} trade signals")
            for trade in trades:
                logger.info(f"  {trade['side'].upper()} {trade['qty']} {trade['symbol']} ({trade['reason']})")

            return signal

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _fetch_prices(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch historical prices for universe with caching.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            DataFrame with historical prices
        """
        # Check cache validity
        if not force_refresh and self._price_cache is not None and self._price_cache_time:
            cache_age = (datetime.now() - self._price_cache_time).total_seconds()
            if cache_age < self.PRICE_CACHE_TTL_SECONDS:
                logger.info(f"Using cached price data ({cache_age/60:.0f}min old, TTL={self.PRICE_CACHE_TTL_SECONDS/60:.0f}min)")
                return self._price_cache

        try:
            import yfinance as yf

            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.lookback_days + 30)

            logger.info(f"Fetching fresh prices for {self.config.symbols}...")

            data = yf.download(
                self.config.symbols,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
            )

            # Handle different yfinance column formats
            if isinstance(data.columns, pd.MultiIndex):
                # New format: MultiIndex with ('Price', 'Ticker')
                # Try 'Adj Close' first, fall back to 'Close'
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close']
                elif 'Close' in data.columns.get_level_values(0):
                    prices = data['Close']
                else:
                    # Get first level (any price type)
                    first_level = data.columns.get_level_values(0)[0]
                    prices = data[first_level]
            else:
                # Single ticker or flat columns
                if 'Adj Close' in data.columns:
                    prices = data[['Adj Close']]
                elif 'Close' in data.columns:
                    prices = data[['Close']]
                else:
                    prices = data
                prices.columns = self.config.symbols[:len(prices.columns)]

            prices = prices.dropna()
            logger.info(f"Fetched {len(prices)} days of price data for {list(prices.columns)}")

            # Update cache
            self._price_cache = prices
            self._price_cache_time = datetime.now()
            logger.info(f"Price data cached (TTL={self.PRICE_CACHE_TTL_SECONDS/60:.0f}min)")

            return prices

        except Exception as e:
            logger.error(f"Price fetch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return stale cache if available (better than nothing)
            if self._price_cache is not None:
                logger.warning("Returning stale cache data due to fetch failure")
                return self._price_cache
            return None

    def _detect_regime(self, prices_df: pd.DataFrame) -> Tuple[str, float]:
        """Detect market regime from price data."""
        try:
            from ml_models.regime import RegimeDetector

            # Use SPY as market proxy if available, else first symbol
            market_col = 'SPY' if 'SPY' in prices_df.columns else prices_df.columns[0]
            market_returns = prices_df[market_col].pct_change().dropna()

            detector = RegimeDetector()
            regime_info = detector.detect_regime(market_returns, method='combined')

            self._regime_info = regime_info
            return regime_info['regime'], regime_info['confidence']

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return 'neutral', 0.5

    def _optimize_portfolio(self, prices_df: pd.DataFrame, regime: str) -> Optional[Dict[str, float]]:
        """Run portfolio optimization."""
        try:
            from portfolio.cvar_allocator import CVaRAllocator
            from ml_models.regime import RegimeDetector

            # Get regime-adjusted parameters
            detector = RegimeDetector()
            regime_params = detector.get_regime_parameters(regime)

            # Adjust RRR based on regime
            adjusted_rrr = self.config.rrr
            if regime in ['bear_market', 'crisis']:
                adjusted_rrr = min(1.0, self.config.rrr + 0.2)  # More defensive
            elif regime == 'bull_market':
                adjusted_rrr = max(0.0, self.config.rrr - 0.1)  # Slightly more aggressive

            # Calculate returns
            returns = prices_df.pct_change().dropna()

            # Feature-based scoring (if enabled)
            ml_scores = None
            if self.config.enable_feature_scoring:
                ml_scores = self._calculate_feature_scores(prices_df)

            # Initialize optimizer
            risk_aversion = 1.0 + (adjusted_rrr * 19.0)
            max_weight = min(self.config.max_position_pct, regime_params['max_weight'])

            allocator = CVaRAllocator(
                risk_aversion=risk_aversion,
                max_weight=max_weight,
            )

            # Run optimization
            result = allocator.optimize(
                returns=returns,
                ml_scores=ml_scores,
            )

            weights = result['weights']

            # Convert to dict, filter small weights
            weight_dict = {}
            for symbol, weight in weights.items():
                if weight > 0.01:  # Ignore < 1% allocations
                    weight_dict[symbol] = float(weight)

            # Renormalize
            total = sum(weight_dict.values())
            if total > 0:
                weight_dict = {k: v/total for k, v in weight_dict.items()}

            logger.info(f"Optimized weights: {weight_dict}")
            return weight_dict

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _calculate_feature_scores(self, prices_df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate feature-based scores for stock selection."""
        try:
            scores = {}

            for symbol in prices_df.columns:
                prices = prices_df[symbol].dropna()
                if len(prices) < 60:
                    continue

                returns = prices.pct_change().dropna()

                # Momentum (medium-term: 60 days)
                momentum_medium = (prices.iloc[-1] / prices.iloc[-60] - 1) if len(prices) >= 60 else 0

                # Momentum (short-term: 20 days)
                momentum_short = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0

                # Price stability (inverse of volatility)
                vol = returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else 1
                price_stability = 1 / (1 + vol)

                # Sharpe-like ratio
                mean_ret = returns.iloc[-60:].mean() * 252 if len(returns) >= 60 else 0
                sharpe = mean_ret / vol if vol > 0 else 0

                # Composite score (same weights as Base.md)
                score = (
                    0.30 * momentum_medium +
                    0.20 * momentum_short +
                    0.20 * price_stability +
                    0.30 * sharpe
                )

                scores[symbol] = score

            return pd.Series(scores)

        except Exception as e:
            logger.warning(f"Feature scoring failed: {e}")
            return None

    def _calculate_trades(self,
                         target_weights: Dict[str, float],
                         current_weights: Dict[str, float],
                         portfolio_value: float,
                         prices_df: pd.DataFrame,
                         live_quotes: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict]:
        """
        Calculate trades needed to reach target allocation.

        Args:
            target_weights: Target allocation weights
            current_weights: Current allocation weights
            portfolio_value: Total portfolio value
            prices_df: Historical price data (fallback)
            live_quotes: Live quotes from Alpaca (preferred for sizing)
        """
        trades = []

        # Get current prices - prefer live quotes over stale yfinance data
        if live_quotes:
            # Use mid price from live quotes
            current_prices = pd.Series({
                symbol: quote['mid']
                for symbol, quote in live_quotes.items()
            })
            logger.info("Using live Alpaca quotes for order sizing")
        else:
            # Fallback to historical prices
            current_prices = prices_df.iloc[-1]
            logger.warning("Using stale yfinance prices for order sizing (live quotes unavailable)")

        # All symbols to consider
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())

        for symbol in all_symbols:
            target = target_weights.get(symbol, 0)
            current = current_weights.get(symbol, 0)

            diff = target - current
            diff_value = diff * portfolio_value

            # Skip small adjustments
            if abs(diff_value) < 100:
                continue

            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            qty = int(abs(diff_value) / price)
            if qty <= 0:
                continue

            side = 'buy' if diff > 0 else 'sell'
            reason = f"Target: {target:.1%}, Current: {current:.1%}"

            trades.append({
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'price': float(price),
                'value': qty * float(price),
                'reason': reason,
            })

        # Sort: sells first (to free up cash), then buys
        trades.sort(key=lambda x: (0 if x['side'] == 'sell' else 1, -x['value']))

        return trades

    def get_status(self) -> Dict:
        """Get current strategy status."""
        return {
            'config': {
                'symbols': self.config.symbols,
                'rrr': self.config.rrr,
                'enable_ml': self.config.enable_ml,
                'enable_regime': self.config.enable_regime_detection,
                'enable_feature_scoring': self.config.enable_feature_scoring,
                'rebalance_threshold': self.config.rebalance_threshold,
            },
            'state': {
                'last_rebalance': self._last_rebalance.isoformat() if self._last_rebalance else None,
                'current_weights': self._current_weights,
                'regime': self._regime_info.get('regime') if self._regime_info else None,
            }
        }


if __name__ == "__main__":
    # Test the strategy
    from bot.logging_config import setup_logging
    setup_logging()

    strategy = TradingStrategy()

    # Simulate current positions (empty portfolio)
    current_positions = {}
    portfolio_value = 100000
    cash = 100000

    signal = strategy.generate_signals(current_positions, portfolio_value, cash)

    if signal:
        print("\n=== Portfolio Signal ===")
        print(f"Regime: {signal.regime}")
        print(f"Target weights: {signal.target_weights}")
        print(f"\nTrades needed:")
        for trade in signal.trades_needed:
            print(f"  {trade['side'].upper()} {trade['qty']} {trade['symbol']} @ ${trade['price']:.2f}")
