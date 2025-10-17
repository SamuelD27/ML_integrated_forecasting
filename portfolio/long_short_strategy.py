"""
Long/Short Equity Strategy
===========================
Institutional-grade market-neutral long/short strategy with sector neutralization.

Strategy Components:
1. Multi-factor scoring (momentum, value, quality, volatility, alpha)
2. Sector-neutral ranking (z-scores within sectors)
3. Market-neutral construction (target β ≈ 0)
4. ATR-based position sizing (risk parity)
5. Dynamic rebalancing

Target Profile:
- Gross Exposure: 180-200% (90-100% long + 90-100% short)
- Net Exposure: ±10% (near market-neutral)
- Sector Exposure: ±10% per sector
- Position Size: 2-5% per position
- Number of Positions: 20 long + 20 short

Performance Targets:
- Sharpe Ratio: 0.5-1.0 (market-neutral strategies)
- Max Drawdown: <20%
- Win Rate: 52-55%
- Information Coefficient: 0.03-0.05
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio.sector_classifier import SectorClassifier
from portfolio.factor_models import FamaFrenchFactorModel

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Signal direction."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class FactorWeights:
    """Weights for multi-factor scoring."""
    momentum: float = 0.30
    alpha: float = 0.25
    quality: float = 0.20
    value: float = 0.15
    volatility: float = 0.10

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (self.momentum + self.alpha + self.quality +
                self.value + self.volatility)
        if not np.isclose(total, 1.0):
            raise ValueError(f"Factor weights must sum to 1.0, got {total}")


@dataclass
class PortfolioConstraints:
    """Portfolio construction constraints."""
    n_long: int = 20
    n_short: int = 20
    max_sector_exposure: float = 0.10  # ±10% per sector
    max_position_size: float = 0.05  # 5% per position
    min_position_size: float = 0.02  # 2% per position
    target_net_exposure: float = 0.0  # Market-neutral
    max_net_exposure: float = 0.10  # ±10%
    target_beta: float = 0.0  # Beta-neutral
    max_beta: float = 0.2  # ±0.2


class LongShortStrategy:
    """
    Long/Short equity strategy with sector neutralization.

    Combines multiple factors to generate composite scores, ranks stocks
    within sectors to ensure neutrality, and constructs market-neutral portfolios.

    Example:
        >>> strategy = LongShortStrategy(
        ...     universe=['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', ...],
        ...     factor_weights=FactorWeights(),
        ...     constraints=PortfolioConstraints()
        ... )
        >>> signals = strategy.generate_signals(features_df, date='2024-01-15')
        >>> long_positions = signals[signals == SignalDirection.LONG]
        >>> short_positions = signals[signals == SignalDirection.SHORT]
    """

    def __init__(
        self,
        universe: List[str],
        factor_weights: Optional[FactorWeights] = None,
        constraints: Optional[PortfolioConstraints] = None,
        sector_neutral: bool = True,
        use_factor_model: bool = True
    ):
        """
        Initialize long/short strategy.

        Args:
            universe: List of ticker symbols in universe
            factor_weights: Weights for multi-factor scoring
            constraints: Portfolio construction constraints
            sector_neutral: Whether to enforce sector neutrality
            use_factor_model: Whether to use Fama-French factor model for alpha
        """
        self.universe = universe
        self.factor_weights = factor_weights or FactorWeights()
        self.constraints = constraints or PortfolioConstraints()
        self.sector_neutral = sector_neutral
        self.use_factor_model = use_factor_model

        # Initialize components
        self.sector_classifier = SectorClassifier()
        if self.use_factor_model:
            self.factor_model = FamaFrenchFactorModel(model='5-factor')

        # Classify sectors
        logger.info(f"Initializing LongShortStrategy with {len(universe)} stocks")
        self.sectors = self.sector_classifier.classify(universe)
        logger.info(f"Classified into {len(set(self.sectors.values()))} sectors")

    def calculate_composite_score(
        self,
        features: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculate composite score from multiple factors.

        Args:
            features: DataFrame with features (index = ticker, columns = features)
            returns: Optional returns DataFrame for factor regression

        Returns:
            Series with composite scores (index = ticker)

        Score Components:
        - Momentum (30%): Recent price momentum
        - Alpha (25%): Fama-French alpha or residual
        - Quality (20%): Profitability, stability
        - Value (15%): Book-to-market, earnings yield
        - Volatility (10%): Inverse volatility (prefer low vol)
        """
        logger.info("Calculating composite scores...")

        scores = pd.DataFrame(index=features.index)

        # 1. Momentum Score (30%)
        if 'momentum_medium' in features.columns:
            momentum = features['momentum_medium']
        elif 'momentum_20' in features.columns:
            momentum = features['momentum_20']
        else:
            logger.warning("No momentum feature found, using zeros")
            momentum = pd.Series(0, index=features.index)

        scores['momentum'] = self._normalize_score(momentum)

        # 2. Alpha Score (25%)
        if self.use_factor_model and returns is not None:
            alpha_scores = self._calculate_alpha_scores(returns)
            scores['alpha'] = self._normalize_score(alpha_scores)
        elif 'factor_alpha' in features.columns:
            scores['alpha'] = self._normalize_score(features['factor_alpha'])
        else:
            logger.warning("No alpha available, using zeros")
            scores['alpha'] = 0

        # 3. Quality Score (20%)
        quality_features = []
        if 'quality_score' in features.columns:
            quality_features.append(features['quality_score'])
        if 'factor_beta_rmw' in features.columns:  # Profitability factor
            quality_features.append(features['factor_beta_rmw'])
        if 'sharpe_ratio' in features.columns:
            quality_features.append(features['sharpe_ratio'])

        if quality_features:
            quality = pd.concat(quality_features, axis=1).mean(axis=1)
            scores['quality'] = self._normalize_score(quality)
        else:
            logger.warning("No quality features found, using zeros")
            scores['quality'] = 0

        # 4. Value Score (15%)
        value_features = []
        if 'factor_beta_hml' in features.columns:  # Value factor
            value_features.append(features['factor_beta_hml'])
        if 'pe_ratio' in features.columns:
            value_features.append(-features['pe_ratio'])  # Lower P/E is better

        if value_features:
            value = pd.concat(value_features, axis=1).mean(axis=1)
            scores['value'] = self._normalize_score(value)
        else:
            logger.warning("No value features found, using zeros")
            scores['value'] = 0

        # 5. Volatility Score (10%) - inverse volatility (prefer low vol)
        if 'volatility_20' in features.columns:
            volatility = -features['volatility_20']  # Negative: prefer low vol
        elif 'realized_volatility' in features.columns:
            volatility = -features['realized_volatility']
        else:
            logger.warning("No volatility feature found, using zeros")
            volatility = pd.Series(0, index=features.index)

        scores['volatility'] = self._normalize_score(volatility)

        # Calculate weighted composite
        composite = (
            scores['momentum'] * self.factor_weights.momentum +
            scores['alpha'] * self.factor_weights.alpha +
            scores['quality'] * self.factor_weights.quality +
            scores['value'] * self.factor_weights.value +
            scores['volatility'] * self.factor_weights.volatility
        )

        logger.info(f"Composite scores calculated: mean={composite.mean():.3f}, std={composite.std():.3f}")

        return composite

    def _calculate_alpha_scores(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate alpha scores using Fama-French factor model."""
        logger.info("Calculating alpha scores using Fama-French model...")

        tickers = returns.columns.tolist()
        batch_results = self.factor_model.batch_regress(tickers, returns)

        # Extract alpha scores
        alpha_series = batch_results.set_index('ticker')['alpha']

        # Weight by significance (alpha * (1 - p_value))
        p_values = batch_results.set_index('ticker')['alpha_pvalue']
        significance_weight = 1 - p_values.clip(0, 1)

        weighted_alpha = alpha_series * significance_weight

        return weighted_alpha

    def _normalize_score(self, score: pd.Series) -> pd.Series:
        """
        Normalize score to [0, 1] using percentile ranks.

        Args:
            score: Raw scores

        Returns:
            Normalized scores [0, 1]
        """
        # Handle NaN
        score = score.fillna(score.median())

        # Percentile rank (0 to 1)
        ranks = score.rank(pct=True)

        return ranks

    def rank_sector_neutral(
        self,
        scores: pd.Series,
        sectors: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Calculate z-scores within sectors for sector-neutral ranking.

        Args:
            scores: Composite scores
            sectors: Sector mapping (uses self.sectors if None)

        Returns:
            Z-scores within sectors

        Example:
            If AAPL has score=0.8 and sector=Tech, and Tech mean=0.7, std=0.1:
            AAPL z-score = (0.8 - 0.7) / 0.1 = 1.0
            This ensures rankings are relative to sector peers, not absolute.
        """
        logger.info("Calculating sector-neutral rankings...")

        if sectors is None:
            sectors = self.sectors

        # Create DataFrame with scores and sectors
        df = pd.DataFrame({
            'score': scores,
            'sector': [sectors.get(t, 'Unknown') for t in scores.index]
        })

        # Calculate z-score within each sector
        def calc_z_score(group):
            mean = group['score'].mean()
            std = group['score'].std()
            if std == 0 or pd.isna(std):
                return group['score'] - mean  # Just center if no variance
            return (group['score'] - mean) / std

        df['z_score'] = df.groupby('sector').apply(calc_z_score).reset_index(level=0, drop=True)

        logger.info(f"Z-scores: mean={df['z_score'].mean():.3f}, std={df['z_score'].std():.3f}")

        return df['z_score']

    def generate_signals(
        self,
        features: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        date: Optional[str] = None
    ) -> Dict[str, SignalDirection]:
        """
        Generate long/short signals.

        Args:
            features: Features DataFrame (index = ticker)
            returns: Returns DataFrame for factor regression (optional)
            date: Date for signals (for logging)

        Returns:
            Dictionary mapping ticker -> signal direction

        Example:
            >>> signals = strategy.generate_signals(features)
            >>> long_tickers = [t for t, s in signals.items() if s == SignalDirection.LONG]
            >>> short_tickers = [t for t, s in signals.items() if s == SignalDirection.SHORT]
        """
        logger.info(f"Generating signals for {len(features)} stocks" +
                   (f" on {date}" if date else ""))

        # Calculate composite scores
        composite = self.calculate_composite_score(features, returns)

        # Apply sector-neutral ranking if enabled
        if self.sector_neutral:
            ranks = self.rank_sector_neutral(composite)
        else:
            ranks = composite

        # Sort by rank
        ranked = ranks.sort_values(ascending=False)

        # Select top N long and bottom N short
        n_long = self.constraints.n_long
        n_short = self.constraints.n_short

        long_tickers = ranked.head(n_long).index.tolist()
        short_tickers = ranked.tail(n_short).index.tolist()

        # Check sector neutrality
        if self.sector_neutral:
            is_neutral, net_exposures = self.sector_classifier.check_sector_neutrality(
                long_tickers,
                short_tickers,
                max_sector_exposure=self.constraints.max_sector_exposure
            )

            if not is_neutral:
                logger.warning(f"Portfolio is not sector-neutral! Net exposures: {net_exposures}")

        # Create signal dictionary
        signals = {}
        for ticker in long_tickers:
            signals[ticker] = SignalDirection.LONG
        for ticker in short_tickers:
            signals[ticker] = SignalDirection.SHORT
        for ticker in self.universe:
            if ticker not in signals:
                signals[ticker] = SignalDirection.NEUTRAL

        logger.info(f"Generated {len(long_tickers)} long, {len(short_tickers)} short signals")

        return signals

    def calculate_position_sizes(
        self,
        signals: Dict[str, SignalDirection],
        prices: pd.Series,
        volatilities: pd.Series,
        capital: float = 1_000_000,
        method: str = 'atr'
    ) -> Dict[str, float]:
        """
        Calculate position sizes using ATR-based risk parity.

        Args:
            signals: Signal dictionary from generate_signals()
            prices: Current prices (Series with ticker index)
            volatilities: ATR or realized volatility (Series with ticker index)
            capital: Total capital
            method: Position sizing method ('equal', 'atr', 'inverse_vol')

        Returns:
            Dictionary mapping ticker -> position size (in dollars)

        Methods:
        - 'equal': Equal dollar weighting
        - 'atr': ATR-based (position size ∝ 1/ATR)
        - 'inverse_vol': Inverse volatility weighting

        Example:
            >>> sizes = strategy.calculate_position_sizes(
            ...     signals, prices, volatilities, capital=1_000_000
            ... )
            >>> # Long positions sum to ~$900k, short to ~$900k
            >>> # Higher volatility stocks get smaller positions
        """
        logger.info(f"Calculating position sizes (method: {method}, capital: ${capital:,.0f})")

        long_tickers = [t for t, s in signals.items() if s == SignalDirection.LONG]
        short_tickers = [t for t, s in signals.items() if s == SignalDirection.SHORT]

        position_sizes = {}

        if method == 'equal':
            # Equal weight for each side
            long_size = capital * 0.9 / len(long_tickers) if long_tickers else 0
            short_size = capital * 0.9 / len(short_tickers) if short_tickers else 0

            for ticker in long_tickers:
                position_sizes[ticker] = long_size
            for ticker in short_tickers:
                position_sizes[ticker] = -short_size

        elif method in ['atr', 'inverse_vol']:
            # Calculate inverse volatility weights
            long_vols = volatilities[long_tickers]
            short_vols = volatilities[short_tickers]

            # Inverse volatility
            long_inv_vol = 1 / long_vols
            short_inv_vol = 1 / short_vols

            # Normalize to sum to 1.0
            long_weights = long_inv_vol / long_inv_vol.sum()
            short_weights = short_inv_vol / short_inv_vol.sum()

            # Scale to target capital (90% on each side)
            long_capital = capital * 0.9
            short_capital = capital * 0.9

            for ticker, weight in long_weights.items():
                position_sizes[ticker] = long_capital * weight

            for ticker, weight in short_weights.items():
                position_sizes[ticker] = -short_capital * weight

        # Apply position size constraints
        max_size = capital * self.constraints.max_position_size
        min_size = capital * self.constraints.min_position_size

        for ticker in position_sizes:
            size = position_sizes[ticker]
            abs_size = abs(size)

            # Clip to constraints
            if abs_size > max_size:
                position_sizes[ticker] = max_size * np.sign(size)
            elif abs_size < min_size:
                position_sizes[ticker] = min_size * np.sign(size)

        # Log summary
        long_exposure = sum(s for s in position_sizes.values() if s > 0)
        short_exposure = abs(sum(s for s in position_sizes.values() if s < 0))
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        logger.info(f"Position sizing complete:")
        logger.info(f"  Long exposure:  ${long_exposure:,.0f} ({long_exposure/capital:.1%})")
        logger.info(f"  Short exposure: ${short_exposure:,.0f} ({short_exposure/capital:.1%})")
        logger.info(f"  Net exposure:   ${net_exposure:,.0f} ({net_exposure/capital:.1%})")
        logger.info(f"  Gross exposure: ${gross_exposure:,.0f} ({gross_exposure/capital:.1%})")

        return position_sizes


def create_market_neutral_portfolio(
    universe: List[str],
    features: pd.DataFrame,
    prices: pd.Series,
    volatilities: pd.Series,
    returns: Optional[pd.DataFrame] = None,
    capital: float = 1_000_000,
    factor_weights: Optional[FactorWeights] = None,
    constraints: Optional[PortfolioConstraints] = None
) -> Dict[str, Union[Dict, pd.DataFrame]]:
    """
    Convenience function to create complete market-neutral portfolio.

    Args:
        universe: List of tickers
        features: Features DataFrame
        prices: Current prices
        volatilities: Volatilities (ATR or realized vol)
        returns: Historical returns for factor regression
        capital: Total capital
        factor_weights: Factor weights for scoring
        constraints: Portfolio constraints

    Returns:
        Dictionary with:
        - 'signals': Signal dictionary
        - 'positions': Position sizes
        - 'portfolio': Portfolio summary DataFrame
        - 'exposures': Sector exposures

    Example:
        >>> result = create_market_neutral_portfolio(
        ...     universe, features, prices, volatilities,
        ...     capital=1_000_000
        ... )
        >>> portfolio_df = result['portfolio']
        >>> print(portfolio_df[['ticker', 'signal', 'size', 'sector']])
    """
    strategy = LongShortStrategy(
        universe=universe,
        factor_weights=factor_weights,
        constraints=constraints,
        sector_neutral=True,
        use_factor_model=True
    )

    # Generate signals
    signals = strategy.generate_signals(features, returns)

    # Calculate position sizes
    positions = strategy.calculate_position_sizes(
        signals, prices, volatilities, capital, method='atr'
    )

    # Create portfolio DataFrame
    portfolio_data = []
    for ticker in universe:
        signal = signals.get(ticker, SignalDirection.NEUTRAL)
        size = positions.get(ticker, 0)
        sector = strategy.sectors.get(ticker, 'Unknown')

        if size != 0:
            portfolio_data.append({
                'ticker': ticker,
                'signal': signal.name,
                'size': size,
                'price': prices.get(ticker, np.nan),
                'shares': int(size / prices.get(ticker, 1)),
                'sector': sector,
                'volatility': volatilities.get(ticker, np.nan)
            })

    portfolio_df = pd.DataFrame(portfolio_data)

    # Calculate sector exposures
    long_tickers = portfolio_df[portfolio_df['size'] > 0]['ticker'].tolist()
    short_tickers = portfolio_df[portfolio_df['size'] < 0]['ticker'].tolist()
    long_weights = (portfolio_df[portfolio_df['size'] > 0]['size'] /
                   portfolio_df[portfolio_df['size'] > 0]['size'].sum()).tolist()
    short_weights = (abs(portfolio_df[portfolio_df['size'] < 0]['size']) /
                    abs(portfolio_df[portfolio_df['size'] < 0]['size']).sum()).tolist()

    _, net_exposures = strategy.sector_classifier.check_sector_neutrality(
        long_tickers, short_tickers, long_weights, short_weights
    )

    return {
        'signals': signals,
        'positions': positions,
        'portfolio': portfolio_df,
        'exposures': net_exposures,
        'strategy': strategy
    }
