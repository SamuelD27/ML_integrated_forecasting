"""
Trade Filter (Meta-Labeling Filter)
====================================
Filters trade signals based on meta-model probability threshold.

This module provides the final decision layer before execution:
- Takes a list of TradeSignals with meta_prob values
- Filters out signals below the threshold
- Applies additional regime-based and risk-based filters
"""

import logging
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Import pipeline types
try:
    from pipeline.core_types import TradeSignal
    HAS_PIPELINE_TYPES = True
except ImportError:
    HAS_PIPELINE_TYPES = False


def filter_by_meta_prob(
    signals: List['TradeSignal'],
    threshold: float = 0.65,
) -> List['TradeSignal']:
    """
    Filter signals by meta-model probability.

    Only signals with meta_prob >= threshold are returned.
    Signals without meta_prob are passed through (no filtering).

    Args:
        signals: List of TradeSignal objects
        threshold: Minimum meta_prob to pass filter

    Returns:
        Filtered list of TradeSignals
    """
    filtered = []

    for signal in signals:
        if signal.meta_prob is None:
            # No meta-prob computed, pass through
            filtered.append(signal)
            logger.debug(f"{signal.ticker}: No meta_prob, passing through")
        elif signal.meta_prob >= threshold:
            filtered.append(signal)
            logger.debug(f"{signal.ticker}: meta_prob={signal.meta_prob:.2f} >= {threshold}, kept")
        else:
            logger.debug(f"{signal.ticker}: meta_prob={signal.meta_prob:.2f} < {threshold}, filtered out")

    logger.info(f"Meta-prob filter: {len(filtered)}/{len(signals)} signals passed (threshold={threshold})")

    return filtered


def filter_by_regime(
    signals: List['TradeSignal'],
    allowed_regimes: Optional[List[int]] = None,
    blocked_regimes: Optional[List[int]] = None,
) -> List['TradeSignal']:
    """
    Filter signals by market regime.

    Args:
        signals: List of TradeSignal objects
        allowed_regimes: Only keep signals in these regimes (None = all allowed)
        blocked_regimes: Remove signals in these regimes

    Returns:
        Filtered list of TradeSignals
    """
    if allowed_regimes is None and blocked_regimes is None:
        return signals

    filtered = []

    for signal in signals:
        regime = signal.regime

        # Check blocked regimes first
        if blocked_regimes and regime in blocked_regimes:
            logger.debug(f"{signal.ticker}: regime {regime} is blocked, filtered out")
            continue

        # Check allowed regimes
        if allowed_regimes and regime not in allowed_regimes:
            logger.debug(f"{signal.ticker}: regime {regime} not in allowed list, filtered out")
            continue

        filtered.append(signal)

    logger.info(f"Regime filter: {len(filtered)}/{len(signals)} signals passed")

    return filtered


def filter_by_direction(
    signals: List['TradeSignal'],
    allowed_directions: Optional[List[str]] = None,
) -> List['TradeSignal']:
    """
    Filter signals by direction.

    Args:
        signals: List of TradeSignal objects
        allowed_directions: Only keep these directions (e.g., ['long', 'flat'])

    Returns:
        Filtered list of TradeSignals
    """
    if allowed_directions is None:
        return signals

    filtered = [s for s in signals if s.direction in allowed_directions]

    logger.info(f"Direction filter: {len(filtered)}/{len(signals)} signals passed")

    return filtered


def filter_by_expected_return(
    signals: List['TradeSignal'],
    min_return: float = 0.0,
    max_return: Optional[float] = None,
) -> List['TradeSignal']:
    """
    Filter signals by expected return.

    Args:
        signals: List of TradeSignal objects
        min_return: Minimum expected return (default 0)
        max_return: Maximum expected return (optional cap)

    Returns:
        Filtered list of TradeSignals
    """
    filtered = []

    for signal in signals:
        ret = signal.expected_return

        if ret < min_return:
            logger.debug(f"{signal.ticker}: expected_return {ret:.2%} < {min_return:.2%}, filtered out")
            continue

        if max_return is not None and ret > max_return:
            logger.debug(f"{signal.ticker}: expected_return {ret:.2%} > {max_return:.2%}, filtered out")
            continue

        filtered.append(signal)

    logger.info(f"Expected return filter: {len(filtered)}/{len(signals)} signals passed")

    return filtered


def filter_by_volatility(
    signals: List['TradeSignal'],
    max_vol: float = 0.50,
) -> List['TradeSignal']:
    """
    Filter signals by expected volatility.

    Args:
        signals: List of TradeSignal objects
        max_vol: Maximum annualized volatility

    Returns:
        Filtered list of TradeSignals
    """
    filtered = []

    for signal in signals:
        vol = signal.expected_vol

        if vol > max_vol:
            logger.debug(f"{signal.ticker}: expected_vol {vol:.2%} > {max_vol:.2%}, filtered out")
            continue

        filtered.append(signal)

    logger.info(f"Volatility filter: {len(filtered)}/{len(signals)} signals passed")

    return filtered


def filter_signals(
    signals: List['TradeSignal'],
    config: Optional[Dict[str, Any]] = None,
) -> List['TradeSignal']:
    """
    Apply all configured filters to signals.

    Reads filter configuration from config dict and applies
    filters in sequence.

    Args:
        signals: List of TradeSignal objects
        config: Configuration dict with filter settings

    Returns:
        Filtered list of TradeSignals

    Config keys:
        meta_labeling.meta_prob_threshold: float (default 0.65)
        filters.blocked_regimes: List[int] (default [3] = crisis)
        filters.long_only: bool (default True)
        filters.min_expected_return: float (default 0.01)
        filters.max_volatility: float (default 0.50)
    """
    if not signals:
        return []

    config = config or {}
    meta_config = config.get('meta_labeling', {})
    filter_config = config.get('filters', {})

    # 1. Meta-prob filter
    threshold = meta_config.get('meta_prob_threshold', 0.65)
    signals = filter_by_meta_prob(signals, threshold=threshold)

    # 2. Regime filter (default: block crisis)
    blocked_regimes = filter_config.get('blocked_regimes', [3])
    allowed_regimes = filter_config.get('allowed_regimes', None)
    signals = filter_by_regime(
        signals,
        allowed_regimes=allowed_regimes,
        blocked_regimes=blocked_regimes,
    )

    # 3. Direction filter (default: long only)
    if filter_config.get('long_only', True):
        signals = filter_by_direction(signals, allowed_directions=['long'])

    # 4. Expected return filter
    min_return = filter_config.get('min_expected_return', 0.01)
    signals = filter_by_expected_return(signals, min_return=min_return)

    # 5. Volatility filter
    max_vol = filter_config.get('max_volatility', 0.50)
    signals = filter_by_volatility(signals, max_vol=max_vol)

    logger.info(f"Total after all filters: {len(signals)} signals")

    return signals


def rank_signals(
    signals: List['TradeSignal'],
    ranking_metric: str = 'risk_reward',
    ascending: bool = False,
) -> List['TradeSignal']:
    """
    Rank signals by a metric.

    Args:
        signals: List of TradeSignal objects
        ranking_metric: Metric to rank by
            - 'meta_prob': Meta-model probability
            - 'expected_return': Expected return
            - 'risk_reward': Return / volatility ratio
            - 'confidence': Forecast confidence
        ascending: Sort in ascending order (default: descending)

    Returns:
        Ranked list of TradeSignals
    """
    if not signals:
        return []

    # Calculate ranking values
    def get_rank_value(signal):
        if ranking_metric == 'meta_prob':
            return signal.meta_prob or 0.0
        elif ranking_metric == 'expected_return':
            return signal.expected_return
        elif ranking_metric == 'risk_reward':
            if signal.expected_vol == 0:
                return 0.0
            return abs(signal.expected_return) / signal.expected_vol
        elif ranking_metric == 'confidence':
            if signal.forecast:
                return signal.forecast.confidence
            return 0.5
        else:
            return 0.0

    ranked = sorted(signals, key=get_rank_value, reverse=not ascending)

    return ranked


def select_top_signals(
    signals: List['TradeSignal'],
    max_signals: int = 10,
    ranking_metric: str = 'risk_reward',
) -> List['TradeSignal']:
    """
    Select top N signals after ranking.

    Args:
        signals: List of TradeSignal objects
        max_signals: Maximum number to select
        ranking_metric: Metric for ranking

    Returns:
        Top N signals
    """
    ranked = rank_signals(signals, ranking_metric=ranking_metric, ascending=False)
    selected = ranked[:max_signals]

    logger.info(f"Selected top {len(selected)} signals from {len(signals)}")

    return selected


def diversify_signals(
    signals: List['TradeSignal'],
    max_per_sector: int = 3,
) -> List['TradeSignal']:
    """
    Diversify signals by limiting per-sector exposure.

    Args:
        signals: List of TradeSignal objects
        max_per_sector: Maximum signals per sector

    Returns:
        Diversified list of TradeSignals
    """
    sector_counts: Dict[str, int] = {}
    diversified = []

    for signal in signals:
        sector = signal.snapshot.sector if signal.snapshot else 'Unknown'
        sector = sector or 'Unknown'

        current = sector_counts.get(sector, 0)
        if current < max_per_sector:
            diversified.append(signal)
            sector_counts[sector] = current + 1
        else:
            logger.debug(f"{signal.ticker}: Sector {sector} at max ({max_per_sector}), skipped")

    logger.info(f"Diversification filter: {len(diversified)}/{len(signals)} signals kept")

    return diversified
