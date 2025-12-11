"""
Trading Strategy V2 - With Universe Screening & Enhanced Data Pipeline
========================================================================
Upgraded trading strategy that:
- Screens a broader market universe (S&P 500)
- Fetches comprehensive security profiles
- Uses standardized data structures
- Supports multiple selection/ranking methods

Key Improvements over V1:
- Dynamic universe (not hardcoded 5 symbols)
- Full fundamental + technical analysis
- Parallel data fetching from multiple sources
- Modular scoring system
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.logging_config import get_logger
from bot.data_pipeline import (
    UniverseScreener,
    UniverseConfig,
    SecurityProfileFetcher,
    SecurityProfile,
)
from bot.data_pipeline.universe_screener import UniverseType

logger = get_logger(__name__)


class SelectionMethod(Enum):
    """Available stock selection/scoring methods."""
    FEATURE_SCORING = "feature_scoring"  # Technical momentum/quality (current)
    MULTI_FACTOR = "multi_factor"  # Fundamental + technical factors
    ML_ENSEMBLE = "ml_ensemble"  # Trained ML model (requires training)
    MOMENTUM = "momentum"  # Pure momentum strategy
    VALUE = "value"  # Value investing
    QUALITY = "quality"  # Quality metrics
    COMBINED = "combined"  # Multi-factor combined


@dataclass
class PortfolioSignal:
    """Represents a portfolio rebalancing signal."""
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    trades_needed: List[Dict]
    regime: str
    confidence: float
    selected_symbols: List[str]
    selection_scores: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyConfigV2:
    """Enhanced strategy configuration."""

    # Universe Configuration
    universe_type: UniverseType = UniverseType.SP500
    custom_symbols: List[str] = field(default_factory=list)
    exclude_symbols: List[str] = field(default_factory=list)

    # Universe Filters
    min_market_cap: float = 10_000_000_000  # $10B minimum
    min_avg_volume: int = 500_000  # 500k shares/day
    min_price: float = 10.0  # Avoid penny stocks
    max_universe_size: int = 100  # Max stocks to analyze

    # Selection Configuration
    selection_method: SelectionMethod = SelectionMethod.MULTI_FACTOR
    num_stocks_to_select: int = 20  # Top N stocks for portfolio
    min_selection_score: float = 0.0  # Minimum score to include

    # Risk Parameters
    rrr: float = 0.5  # Reward-to-risk ratio (0=aggressive, 1=defensive)
    max_position_pct: float = 0.15  # Max 15% per stock (diversified)
    min_position_pct: float = 0.02  # Min 2% to avoid tiny positions
    rebalance_threshold: float = 0.05  # 5% drift triggers rebalance

    # Model Toggles
    enable_regime_detection: bool = True
    enable_fundamentals: bool = True
    enable_live_quotes: bool = True

    # Timing
    rebalance_interval_hours: int = 24
    lookback_days: int = 252
    data_cache_hours: int = 4  # Cache profile data

    @classmethod
    def from_env(cls) -> 'StrategyConfigV2':
        """Load configuration from environment variables."""
        universe_type_str = os.getenv('UNIVERSE_TYPE', 'sp500').lower()
        universe_map = {
            'sp500': UniverseType.SP500,
            'sp100': UniverseType.SP100,
            'nasdaq100': UniverseType.NASDAQ100,
            'dow30': UniverseType.DOW30,
            'custom': UniverseType.CUSTOM,
        }
        universe_type = universe_map.get(universe_type_str, UniverseType.SP500)

        selection_method_str = os.getenv('SELECTION_METHOD', 'multi_factor').lower()
        method_map = {
            'feature_scoring': SelectionMethod.FEATURE_SCORING,
            'multi_factor': SelectionMethod.MULTI_FACTOR,
            'momentum': SelectionMethod.MOMENTUM,
            'value': SelectionMethod.VALUE,
            'quality': SelectionMethod.QUALITY,
            'combined': SelectionMethod.COMBINED,
        }
        selection_method = method_map.get(selection_method_str, SelectionMethod.MULTI_FACTOR)

        custom_symbols_str = os.getenv('CUSTOM_SYMBOLS', '')
        custom_symbols = [s.strip().upper() for s in custom_symbols_str.split(',') if s.strip()]

        exclude_symbols_str = os.getenv('EXCLUDE_SYMBOLS', '')
        exclude_symbols = [s.strip().upper() for s in exclude_symbols_str.split(',') if s.strip()]

        return cls(
            universe_type=universe_type,
            custom_symbols=custom_symbols,
            exclude_symbols=exclude_symbols,
            min_market_cap=float(os.getenv('MIN_MARKET_CAP', '10000000000')),
            min_avg_volume=int(os.getenv('MIN_AVG_VOLUME', '500000')),
            min_price=float(os.getenv('MIN_PRICE', '10.0')),
            max_universe_size=int(os.getenv('MAX_UNIVERSE_SIZE', '100')),
            selection_method=selection_method,
            num_stocks_to_select=int(os.getenv('NUM_STOCKS_SELECT', '20')),
            rrr=float(os.getenv('TRADING_RRR', '0.5')),
            max_position_pct=float(os.getenv('RISK_MAX_POSITION_PCT', '0.15')),
            rebalance_threshold=float(os.getenv('TRADING_REBALANCE_THRESHOLD', '0.05')),
            enable_regime_detection=os.getenv('ENABLE_REGIME', 'true').lower() == 'true',
            enable_fundamentals=os.getenv('ENABLE_FUNDAMENTALS', 'true').lower() == 'true',
            enable_live_quotes=os.getenv('ENABLE_LIVE_QUOTES', 'true').lower() == 'true',
            rebalance_interval_hours=int(os.getenv('REBALANCE_INTERVAL_HOURS', '24')),
            lookback_days=int(os.getenv('TRADING_LOOKBACK_DAYS', '252')),
            data_cache_hours=int(os.getenv('DATA_CACHE_HOURS', '4')),
        )


class TradingStrategyV2:
    """
    Enhanced portfolio trading strategy with universe screening.

    Flow:
    1. Screen universe (S&P 500 or custom) based on filters
    2. Fetch comprehensive security profiles for candidates
    3. Score/rank stocks using selected method
    4. Select top N stocks for portfolio
    5. Optimize portfolio weights using CVaR
    6. Generate trade signals
    """

    def __init__(self, config: Optional[StrategyConfigV2] = None):
        """Initialize strategy with configuration."""
        self.config = config or StrategyConfigV2.from_env()
        self._last_rebalance: Optional[datetime] = None
        self._current_weights: Dict[str, float] = {}
        self._selected_symbols: List[str] = []
        self._profiles_cache: Dict[str, SecurityProfile] = {}
        self._profiles_cache_time: Optional[datetime] = None
        self._regime_info: Optional[Dict] = None

        # Initialize components
        self._universe_screener = UniverseScreener(cache_enabled=True)
        self._profile_fetcher = SecurityProfileFetcher(
            enable_alpaca=self.config.enable_live_quotes,
            enable_fundamentals=self.config.enable_fundamentals,
            enable_technicals=True,
            enable_risk_metrics=True,
            lookback_days=self.config.lookback_days,
        )

        logger.info("Strategy V2 initialized:")
        logger.info(f"  Universe: {self.config.universe_type.value}")
        logger.info(f"  Selection Method: {self.config.selection_method.value}")
        logger.info(f"  Stocks to Select: {self.config.num_stocks_to_select}")
        logger.info(f"  Max Position: {self.config.max_position_pct:.0%}")
        logger.info(f"  Min Market Cap: ${self.config.min_market_cap/1e9:.1f}B")

    def should_rebalance(
        self,
        current_positions: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """Check if rebalancing is needed."""
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

    def generate_signals(
        self,
        current_positions: Dict[str, float],
        portfolio_value: float,
        cash: float,
        live_quotes: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[PortfolioSignal]:
        """Generate portfolio rebalancing signals."""
        logger.info("Generating trading signals (V2)...")

        # Check if rebalance needed
        should_rebal, reason = self.should_rebalance(current_positions, portfolio_value)
        if not should_rebal:
            logger.info(f"No rebalance needed: {reason}")
            return None

        logger.info(f"Rebalance triggered: {reason}")

        try:
            # Step 1: Get screened universe
            universe = self._get_screened_universe()
            if not universe:
                logger.error("Failed to get universe")
                return None

            logger.info(f"Screened universe: {len(universe)} stocks")

            # Step 2: Fetch security profiles
            profiles = self._get_security_profiles(universe)
            if not profiles:
                logger.error("Failed to fetch security profiles")
                return None

            logger.info(f"Fetched profiles for {len(profiles)} stocks")

            # Step 3: Score and rank stocks
            scores = self._score_stocks(profiles)
            if not scores:
                logger.error("Stock scoring failed")
                return None

            # Step 4: Select top N stocks
            selected, selection_scores = self._select_top_stocks(scores)
            if not selected:
                logger.error("Stock selection failed")
                return None

            logger.info(f"Selected {len(selected)} stocks for portfolio")
            self._selected_symbols = selected

            # Step 5: Detect regime (optional)
            regime = 'neutral'
            confidence = 0.5
            if self.config.enable_regime_detection:
                regime, confidence = self._detect_regime(profiles)
                logger.info(f"Detected regime: {regime} (confidence: {confidence:.0%})")

            # Step 6: Optimize portfolio weights
            target_weights = self._optimize_portfolio(selected, profiles, regime)
            if not target_weights:
                logger.error("Portfolio optimization failed")
                return None

            # Step 7: Calculate current weights
            current_weights = {}
            for symbol, value in current_positions.items():
                current_weights[symbol] = value / portfolio_value if portfolio_value > 0 else 0

            # Step 8: Calculate trades needed
            trades = self._calculate_trades(
                target_weights,
                current_weights,
                portfolio_value,
                profiles,
                live_quotes,
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
                selected_symbols=selected,
                selection_scores=selection_scores,
            )

            logger.info(f"Generated {len(trades)} trade signals")
            for trade in trades[:10]:  # Log first 10
                logger.info(f"  {trade['side'].upper()} {trade['qty']} {trade['symbol']} ({trade['reason']})")
            if len(trades) > 10:
                logger.info(f"  ... and {len(trades) - 10} more")

            return signal

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _get_screened_universe(self) -> List[str]:
        """Get screened universe based on configuration."""
        config = UniverseConfig(
            universe_type=self.config.universe_type,
            custom_symbols=self.config.custom_symbols,
            exclude_symbols=self.config.exclude_symbols,
            min_price=self.config.min_price,
            min_avg_volume=self.config.min_avg_volume,
            min_market_cap=self.config.min_market_cap,
            max_symbols=self.config.max_universe_size,
        )

        return self._universe_screener.get_universe(config)

    def _get_security_profiles(self, symbols: List[str]) -> Dict[str, SecurityProfile]:
        """Fetch security profiles with caching."""
        # Check cache
        if self._profiles_cache and self._profiles_cache_time:
            cache_age_hours = (datetime.now() - self._profiles_cache_time).total_seconds() / 3600
            if cache_age_hours < self.config.data_cache_hours:
                # Check if we have all symbols
                missing = [s for s in symbols if s not in self._profiles_cache]
                if not missing:
                    logger.info(f"Using cached profiles ({cache_age_hours:.1f}h old)")
                    return {s: self._profiles_cache[s] for s in symbols}

                # Fetch only missing
                logger.info(f"Fetching {len(missing)} missing profiles")
                new_profiles = self._profile_fetcher.fetch_profiles(missing)
                self._profiles_cache.update(new_profiles)
                return {s: self._profiles_cache[s] for s in symbols if s in self._profiles_cache}

        # Fetch all profiles
        logger.info(f"Fetching profiles for {len(symbols)} symbols...")
        profiles = self._profile_fetcher.fetch_profiles(symbols)

        # Update cache
        self._profiles_cache = profiles
        self._profiles_cache_time = datetime.now()

        return profiles

    def _score_stocks(self, profiles: Dict[str, SecurityProfile]) -> Dict[str, float]:
        """Score stocks based on selected method."""
        method = self.config.selection_method

        if method == SelectionMethod.FEATURE_SCORING:
            return self._score_feature_based(profiles)
        elif method == SelectionMethod.MULTI_FACTOR:
            return self._score_multi_factor(profiles)
        elif method == SelectionMethod.MOMENTUM:
            return self._score_momentum(profiles)
        elif method == SelectionMethod.VALUE:
            return self._score_value(profiles)
        elif method == SelectionMethod.QUALITY:
            return self._score_quality(profiles)
        elif method == SelectionMethod.COMBINED:
            return self._score_combined(profiles)
        else:
            return self._score_multi_factor(profiles)

    def _score_feature_based(self, profiles: Dict[str, SecurityProfile]) -> Dict[str, float]:
        """
        Original feature-based scoring (momentum + quality).

        Score = 0.30 * momentum_60d + 0.20 * momentum_20d + 0.20 * stability + 0.30 * sharpe
        """
        scores = {}

        for symbol, profile in profiles.items():
            if not profile.technicals:
                continue

            tech = profile.technicals

            # Get component scores (handle None values)
            momentum_60d = tech.return_60d or 0
            momentum_20d = tech.return_20d or 0
            volatility = tech.volatility_60d or 0.3
            stability = 1 / (1 + volatility) if volatility > 0 else 0.5
            sharpe = tech.sharpe_60d or 0

            # Composite score
            score = (
                0.30 * momentum_60d +
                0.20 * momentum_20d +
                0.20 * stability +
                0.30 * sharpe
            )

            scores[symbol] = score

        return scores

    def _score_multi_factor(self, profiles: Dict[str, SecurityProfile]) -> Dict[str, float]:
        """
        Multi-factor scoring combining technical and fundamental factors.

        Factors:
        - Momentum (25%): Price momentum
        - Value (20%): P/E, P/B, EV/EBITDA
        - Quality (25%): ROE, margins, financial health
        - Risk (15%): Lower volatility, lower beta
        - Growth (15%): Revenue and earnings growth
        """
        scores = {}

        for symbol, profile in profiles.items():
            if not profile.is_complete():
                continue

            # Initialize factor scores
            momentum_score = 0.0
            value_score = 0.0
            quality_score = 0.0
            risk_score = 0.0
            growth_score = 0.0

            # Momentum Factor (from technicals)
            if profile.technicals:
                tech = profile.technicals
                # Combine short and medium-term momentum
                mom_20d = tech.return_20d or 0
                mom_60d = tech.return_60d or 0
                mom_252d = tech.return_252d or 0

                # Normalize to -1 to 1 range (roughly)
                momentum_score = (
                    0.40 * np.clip(mom_20d * 5, -1, 1) +
                    0.35 * np.clip(mom_60d * 2, -1, 1) +
                    0.25 * np.clip(mom_252d, -1, 1)
                )

            # Value Factor (from fundamentals)
            if profile.fundamentals:
                fund = profile.fundamentals
                val_score = fund.get_valuation_score()
                if val_score is not None:
                    # Invert (lower valuation = higher score)
                    value_score = 1 - np.clip(val_score, 0, 1)
                else:
                    value_score = 0.5  # Neutral if no data

            # Quality Factor (from fundamentals)
            if profile.fundamentals:
                fund = profile.fundamentals
                qual_score = fund.get_quality_score()
                if qual_score is not None:
                    quality_score = qual_score
                else:
                    quality_score = 0.5

            # Risk Factor (from risk metrics and technicals)
            if profile.technicals:
                vol = profile.technicals.volatility_60d or 0.3
                # Lower volatility = higher score
                risk_score = 1 - np.clip(vol / 0.6, 0, 1)  # 60% vol -> 0 score

            if profile.risk_metrics and profile.risk_metrics.beta:
                beta = profile.risk_metrics.beta
                # Beta around 1 is neutral, lower is better for risk
                beta_score = 1 - np.clip(abs(beta - 0.8) / 1.2, 0, 1)
                risk_score = 0.6 * risk_score + 0.4 * beta_score

            # Growth Factor (from fundamentals)
            if profile.fundamentals:
                fund = profile.fundamentals
                rev_growth = fund.revenue_growth or 0
                earn_growth = fund.earnings_growth or 0

                growth_score = (
                    0.5 * np.clip(rev_growth * 2, -1, 1) +
                    0.5 * np.clip(earn_growth * 2, -1, 1)
                )

            # Composite score
            composite = (
                0.25 * momentum_score +
                0.20 * value_score +
                0.25 * quality_score +
                0.15 * risk_score +
                0.15 * growth_score
            )

            scores[symbol] = composite

        return scores

    def _score_momentum(self, profiles: Dict[str, SecurityProfile]) -> Dict[str, float]:
        """Pure momentum scoring."""
        scores = {}

        for symbol, profile in profiles.items():
            if not profile.technicals:
                continue

            tech = profile.technicals
            mom_20d = tech.return_20d or 0
            mom_60d = tech.return_60d or 0
            mom_252d = tech.return_252d or 0

            # Momentum with recent bias
            score = 0.5 * mom_20d + 0.35 * mom_60d + 0.15 * mom_252d
            scores[symbol] = score

        return scores

    def _score_value(self, profiles: Dict[str, SecurityProfile]) -> Dict[str, float]:
        """Value investing scoring."""
        scores = {}

        for symbol, profile in profiles.items():
            if not profile.fundamentals:
                continue

            val_score = profile.fundamentals.get_valuation_score()
            if val_score is not None:
                # Lower valuation = higher score
                scores[symbol] = 1 - np.clip(val_score, 0, 2) / 2
            else:
                scores[symbol] = 0.5

        return scores

    def _score_quality(self, profiles: Dict[str, SecurityProfile]) -> Dict[str, float]:
        """Quality scoring based on profitability and financial health."""
        scores = {}

        for symbol, profile in profiles.items():
            if not profile.fundamentals:
                continue

            qual_score = profile.fundamentals.get_quality_score()
            scores[symbol] = qual_score if qual_score is not None else 0.5

        return scores

    def _score_combined(self, profiles: Dict[str, SecurityProfile]) -> Dict[str, float]:
        """Combined scoring using all methods."""
        # Get individual scores
        momentum = self._score_momentum(profiles)
        value = self._score_value(profiles)
        quality = self._score_quality(profiles)

        # Combine
        scores = {}
        for symbol in profiles:
            scores[symbol] = (
                0.40 * momentum.get(symbol, 0) +
                0.30 * quality.get(symbol, 0.5) +
                0.30 * value.get(symbol, 0.5)
            )

        return scores

    def _select_top_stocks(
        self,
        scores: Dict[str, float]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select top N stocks based on scores."""
        # Sort by score descending
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Filter by minimum score
        filtered = [(s, sc) for s, sc in sorted_stocks if sc >= self.config.min_selection_score]

        # Take top N
        selected = filtered[:self.config.num_stocks_to_select]

        symbols = [s for s, _ in selected]
        selection_scores = {s: sc for s, sc in selected}

        return symbols, selection_scores

    def _detect_regime(self, profiles: Dict[str, SecurityProfile]) -> Tuple[str, float]:
        """Detect market regime from profiles."""
        try:
            from ml_models.regime import RegimeDetector

            # Use SPY as market proxy
            if 'SPY' in profiles and profiles['SPY'].price_data:
                market_returns = profiles['SPY'].price_data.returns
            else:
                # Use average of all stocks
                returns_list = []
                for p in profiles.values():
                    if p.price_data:
                        returns_list.append(p.price_data.returns)
                if returns_list:
                    market_returns = pd.concat(returns_list, axis=1).mean(axis=1)
                else:
                    return 'neutral', 0.5

            detector = RegimeDetector()
            regime_info = detector.detect_regime(market_returns, method='combined')

            self._regime_info = regime_info
            return regime_info['regime'], regime_info['confidence']

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return 'neutral', 0.5

    def _optimize_portfolio(
        self,
        selected_symbols: List[str],
        profiles: Dict[str, SecurityProfile],
        regime: str
    ) -> Optional[Dict[str, float]]:
        """Run portfolio optimization on selected stocks."""
        try:
            from portfolio.cvar_allocator import CVaRAllocator
            from ml_models.regime import RegimeDetector

            # Get regime parameters
            detector = RegimeDetector()
            regime_params = detector.get_regime_parameters(regime)

            # Adjust RRR based on regime
            adjusted_rrr = self.config.rrr
            if regime in ['bear_market', 'crisis']:
                adjusted_rrr = min(1.0, self.config.rrr + 0.2)
            elif regime == 'bull_market':
                adjusted_rrr = max(0.0, self.config.rrr - 0.1)

            # Build returns DataFrame
            returns_dict = {}
            for symbol in selected_symbols:
                if symbol in profiles and profiles[symbol].price_data:
                    returns_dict[symbol] = profiles[symbol].price_data.returns

            if not returns_dict:
                logger.error("No returns data available")
                return None

            returns_df = pd.DataFrame(returns_dict).dropna()

            if len(returns_df) < 60:
                logger.error(f"Insufficient data: {len(returns_df)} days")
                return None

            # Initialize optimizer
            risk_aversion = 1.0 + (adjusted_rrr * 19.0)
            max_weight = min(self.config.max_position_pct, regime_params['max_weight'])

            allocator = CVaRAllocator(
                risk_aversion=risk_aversion,
                max_weight=max_weight,
            )

            # Run optimization
            result = allocator.optimize(returns=returns_df)
            weights = result['weights']

            # Filter small weights
            weight_dict = {}
            for symbol, weight in weights.items():
                if weight >= self.config.min_position_pct:
                    weight_dict[symbol] = float(weight)

            # Renormalize
            total = sum(weight_dict.values())
            if total > 0:
                weight_dict = {k: v / total for k, v in weight_dict.items()}

            logger.info(f"Optimized weights for {len(weight_dict)} stocks")
            return weight_dict

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _calculate_trades(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        portfolio_value: float,
        profiles: Dict[str, SecurityProfile],
        live_quotes: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[Dict]:
        """Calculate trades needed to reach target allocation."""
        trades = []

        # Get current prices
        current_prices = {}
        for symbol in set(target_weights.keys()) | set(current_weights.keys()):
            # Prefer live quotes
            if live_quotes and symbol in live_quotes:
                current_prices[symbol] = live_quotes[symbol]['mid']
            elif symbol in profiles and profiles[symbol].price_data:
                current_prices[symbol] = profiles[symbol].price_data.latest_price
            elif symbol in profiles and profiles[symbol].live_quote:
                current_prices[symbol] = profiles[symbol].live_quote.mid

        # Calculate trades
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
                logger.warning(f"No price for {symbol}, skipping")
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

        # Sort: sells first, then buys by value
        trades.sort(key=lambda x: (0 if x['side'] == 'sell' else 1, -x['value']))

        return trades

    def get_status(self) -> Dict:
        """Get current strategy status."""
        return {
            'config': {
                'universe_type': self.config.universe_type.value,
                'selection_method': self.config.selection_method.value,
                'num_stocks_to_select': self.config.num_stocks_to_select,
                'max_position_pct': self.config.max_position_pct,
                'min_market_cap': self.config.min_market_cap,
                'rrr': self.config.rrr,
                'enable_regime': self.config.enable_regime_detection,
                'enable_fundamentals': self.config.enable_fundamentals,
            },
            'state': {
                'last_rebalance': self._last_rebalance.isoformat() if self._last_rebalance else None,
                'current_weights': self._current_weights,
                'selected_symbols': self._selected_symbols,
                'num_cached_profiles': len(self._profiles_cache),
                'regime': self._regime_info.get('regime') if self._regime_info else None,
            }
        }


if __name__ == "__main__":
    # Test the strategy
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Trading Strategy V2")
    print("=" * 60)

    strategy = TradingStrategyV2()

    # Simulate current positions (empty portfolio)
    current_positions = {}
    portfolio_value = 100000
    cash = 100000

    print("\nGenerating signals...")
    signal = strategy.generate_signals(current_positions, portfolio_value, cash)

    if signal:
        print("\n=== Portfolio Signal ===")
        print(f"Regime: {signal.regime}")
        print(f"Selected {len(signal.selected_symbols)} stocks")
        print(f"\nTarget weights:")
        for sym, weight in sorted(signal.target_weights.items(), key=lambda x: -x[1]):
            print(f"  {sym}: {weight:.1%}")

        print(f"\nTrades needed: {len(signal.trades_needed)}")
        for trade in signal.trades_needed[:10]:
            print(f"  {trade['side'].upper()} {trade['qty']} {trade['symbol']} @ ${trade['price']:.2f}")
    else:
        print("No signal generated")

    print("\n=== Strategy Status ===")
    status = strategy.get_status()
    print(f"Config: {status['config']}")
    print(f"State: {status['state']}")
