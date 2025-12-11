"""
Allocation Utilities
====================
Pipeline interface for portfolio allocation.

Bridges between:
- TradeSignals from the pipeline
- CVaR allocator
- Scenario generator
- Instrument selections

Provides a unified interface for portfolio construction.
"""

import logging
from typing import Dict, Optional, List, Any, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Import pipeline types
try:
    from pipeline.core_types import TradeSignal, Forecast
    HAS_PIPELINE_TYPES = True
except ImportError:
    HAS_PIPELINE_TYPES = False

# Import allocator
try:
    from portfolio.cvar_allocator import CVaRAllocator, optimize_portfolio_cvar
    HAS_CVAR = True
except ImportError:
    HAS_CVAR = False
    logger.warning("CVaR allocator not available")

# Import scenario generator
try:
    from ml_models.scenario_generator import (
        generate_scenarios,
        ScenarioSet,
        calculate_scenario_cvar,
    )
    HAS_SCENARIOS = True
except ImportError:
    HAS_SCENARIOS = False


@dataclass
class AllocationResult:
    """Result of portfolio allocation."""
    weights: pd.Series
    tickers: List[str]
    expected_return: float
    expected_vol: float
    sharpe: float
    cvar: float
    method: str = 'cvar'

    # Additional metrics
    beta: Optional[float] = None
    regime: Optional[int] = None
    risk_aversion: float = 5.0

    # Position-level details
    notional_values: Optional[pd.Series] = None
    shares: Optional[pd.Series] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'weights': self.weights.to_dict(),
            'expected_return': self.expected_return,
            'expected_vol': self.expected_vol,
            'sharpe': self.sharpe,
            'cvar': self.cvar,
            'method': self.method,
            'beta': self.beta,
            'regime': self.regime,
            'risk_aversion': self.risk_aversion,
        }


def allocate_from_signals(
    signals: List['TradeSignal'],
    returns: pd.DataFrame,
    portfolio_value: float,
    regime: int = 2,
    config: Optional[Dict[str, Any]] = None,
) -> AllocationResult:
    """
    Allocate capital based on trade signals.

    Converts signals to expected returns and optimizes using CVaR.

    Args:
        signals: List of TradeSignal objects
        returns: Historical returns DataFrame
        portfolio_value: Total portfolio value
        regime: Current market regime
        config: Configuration dict

    Returns:
        AllocationResult with weights and metrics
    """
    if not HAS_CVAR:
        raise ImportError("CVaR allocator required for allocation")

    config = config or {}
    alloc_config = config.get('allocation', {})

    # Extract tickers from signals
    tickers = [s.ticker for s in signals]

    # Filter returns to signal tickers
    available_tickers = [t for t in tickers if t in returns.columns]
    if not available_tickers:
        raise ValueError("No signal tickers found in returns data")

    returns_filtered = returns[available_tickers]

    # Build ML scores from signals
    ml_scores = {}
    for signal in signals:
        if signal.ticker in available_tickers:
            # Combine expected return and meta_prob for score
            meta_prob = getattr(signal, 'meta_prob', 0.5) or 0.5
            score = signal.expected_return * meta_prob
            ml_scores[signal.ticker] = score

    ml_scores = pd.Series(ml_scores)

    # Get regime-adjusted parameters
    risk_aversion = _get_regime_risk_aversion(regime, config)
    max_weight = _get_regime_max_weight(regime, config)

    # Initialize allocator
    allocator = CVaRAllocator(
        risk_aversion=risk_aversion,
        cvar_alpha=alloc_config.get('cvar_alpha', 0.95),
        max_weight=max_weight,
        min_weight=alloc_config.get('min_weight', 0.0),
        turnover_penalty=alloc_config.get('turnover_penalty', 0.0),
    )

    # Optimize
    result = allocator.optimize(
        returns=returns_filtered,
        ml_scores=ml_scores,
        sector_map=_build_sector_map(signals),
    )

    weights = result['weights']

    # Calculate notional values and shares
    notional = weights * portfolio_value
    prices = _get_current_prices(available_tickers, returns)
    shares = (notional / prices).astype(int)

    return AllocationResult(
        weights=weights,
        tickers=available_tickers,
        expected_return=result['port_return'],
        expected_vol=result['port_vol'],
        sharpe=result['sharpe'],
        cvar=result['cvar'],
        method='cvar',
        beta=result.get('realized_beta'),
        regime=regime,
        risk_aversion=risk_aversion,
        notional_values=notional,
        shares=shares,
        metadata={
            'n_signals': len(signals),
            'n_allocated': len(weights[weights > 0.01]),
            'shrinkage_coeff': result.get('shrinkage_coeff'),
        },
    )


def allocate_with_scenarios(
    signals: List['TradeSignal'],
    returns: pd.DataFrame,
    portfolio_value: float,
    regime: int = 2,
    scenario_method: str = 'regime_conditional',
    n_scenarios: int = 1000,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[AllocationResult, ScenarioSet]:
    """
    Allocate capital using scenario analysis.

    Args:
        signals: List of TradeSignal objects
        returns: Historical returns DataFrame
        portfolio_value: Total portfolio value
        regime: Current market regime
        scenario_method: Method for scenario generation
        n_scenarios: Number of scenarios
        config: Configuration dict

    Returns:
        Tuple of (AllocationResult, ScenarioSet used)
    """
    if not HAS_SCENARIOS:
        logger.warning("Scenario generator not available, using standard allocation")
        result = allocate_from_signals(signals, returns, portfolio_value, regime, config)
        return result, None

    config = config or {}

    # Extract tickers
    tickers = [s.ticker for s in signals]
    available_tickers = [t for t in tickers if t in returns.columns]
    returns_filtered = returns[available_tickers]

    # Generate scenarios
    scenarios = generate_scenarios(
        returns=returns_filtered,
        method=scenario_method,
        n_scenarios=n_scenarios,
        horizon_days=config.get('horizon_days', 10),
        regime=regime,
        config=config.get('scenario_config', {}),
    )

    # Use scenario returns as expected returns
    scenario_returns = pd.DataFrame(
        scenarios.scenarios,
        columns=available_tickers,
    )

    # Standard allocation using scenario data
    result = allocate_from_signals(
        signals=signals,
        returns=scenario_returns,
        portfolio_value=portfolio_value,
        regime=regime,
        config=config,
    )

    # Calculate scenario-based CVaR
    var, cvar = calculate_scenario_cvar(scenarios, result.weights)
    result.metadata['scenario_var'] = var
    result.metadata['scenario_cvar'] = cvar
    result.metadata['scenario_method'] = scenario_method

    return result, scenarios


def calculate_target_weights(
    forecasts: Dict[str, 'Forecast'],
    returns: pd.DataFrame,
    regime: int = 2,
    config: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """
    Calculate target weights from forecasts.

    Simple interface for getting weights from forecast dict.

    Args:
        forecasts: Dict of ticker -> Forecast
        returns: Historical returns DataFrame
        regime: Current market regime
        config: Configuration

    Returns:
        Series of target weights
    """
    if not HAS_CVAR:
        # Equal weight fallback
        tickers = list(forecasts.keys())
        return pd.Series(1.0 / len(tickers), index=tickers)

    config = config or {}

    # Filter to available tickers
    available = [t for t in forecasts.keys() if t in returns.columns]
    returns_filtered = returns[available]

    # Build ML scores from forecasts
    ml_scores = {}
    for ticker, forecast in forecasts.items():
        if ticker in available:
            ml_scores[ticker] = forecast.expected_return * forecast.confidence

    ml_scores = pd.Series(ml_scores)

    # Get parameters
    risk_aversion = _get_regime_risk_aversion(regime, config)
    max_weight = _get_regime_max_weight(regime, config)

    # Allocate
    allocator = CVaRAllocator(
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )

    result = allocator.optimize(
        returns=returns_filtered,
        ml_scores=ml_scores,
    )

    return result['weights']


def rebalance_portfolio(
    current_weights: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
    prices: pd.Series,
    min_trade_value: float = 100.0,
    turnover_limit: float = 0.20,
) -> pd.DataFrame:
    """
    Calculate trades needed to rebalance portfolio.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        portfolio_value: Total portfolio value
        prices: Current prices
        min_trade_value: Minimum trade value to execute
        turnover_limit: Maximum turnover as fraction of portfolio

    Returns:
        DataFrame with trade details
    """
    # Align indices
    all_tickers = set(current_weights.index) | set(target_weights.index)
    current = current_weights.reindex(all_tickers).fillna(0)
    target = target_weights.reindex(all_tickers).fillna(0)
    prices = prices.reindex(all_tickers)

    # Calculate weight changes
    delta_weights = target - current

    # Check turnover
    total_turnover = delta_weights.abs().sum() / 2  # Divide by 2 (sells and buys)
    if total_turnover > turnover_limit:
        # Scale down changes
        scale = turnover_limit / total_turnover
        delta_weights = delta_weights * scale
        logger.info(f"Turnover limited from {total_turnover:.1%} to {turnover_limit:.1%}")

    # Calculate notional changes
    delta_notional = delta_weights * portfolio_value

    # Calculate shares
    shares_to_trade = (delta_notional / prices).fillna(0).astype(int)

    # Filter small trades
    small_mask = (shares_to_trade.abs() * prices).fillna(0) < min_trade_value
    shares_to_trade[small_mask] = 0

    # Build trade DataFrame
    trades = []
    for ticker in all_tickers:
        shares = shares_to_trade.get(ticker, 0)
        if shares != 0:
            price = prices.get(ticker, 0)
            trades.append({
                'ticker': ticker,
                'action': 'buy' if shares > 0 else 'sell',
                'shares': abs(int(shares)),
                'price': price,
                'notional': abs(shares * price),
                'weight_change': float(delta_weights.get(ticker, 0)),
            })

    return pd.DataFrame(trades)


def _get_regime_risk_aversion(regime: int, config: Dict[str, Any]) -> float:
    """Get risk aversion for regime."""
    regime_params = config.get('regime_params', {})
    defaults = {
        0: 3.0,   # Bull: lower risk aversion
        1: 8.0,   # Bear: higher risk aversion
        2: 5.0,   # Neutral: moderate
        3: 12.0,  # Crisis: very high
    }
    return regime_params.get(f'risk_aversion_{regime}', defaults.get(regime, 5.0))


def _get_regime_max_weight(regime: int, config: Dict[str, Any]) -> float:
    """Get max weight for regime."""
    regime_params = config.get('regime_params', {})
    defaults = {
        0: 0.25,  # Bull: normal concentration
        1: 0.15,  # Bear: more diversified
        2: 0.20,  # Neutral: moderate
        3: 0.10,  # Crisis: highly diversified
    }
    return regime_params.get(f'max_weight_{regime}', defaults.get(regime, 0.20))


def _build_sector_map(signals: List['TradeSignal']) -> Dict[str, str]:
    """Build sector map from signals."""
    sector_map = {}
    for signal in signals:
        if hasattr(signal, 'snapshot') and signal.snapshot:
            sector = getattr(signal.snapshot, 'sector', None)
            if sector:
                sector_map[signal.ticker] = sector
    return sector_map


def _get_current_prices(tickers: List[str], returns: pd.DataFrame) -> pd.Series:
    """Estimate current prices from returns (proxy)."""
    # Use last return as proxy (in real system, would fetch actual prices)
    # Assume base price of 100 for simplicity
    base_prices = pd.Series(100.0, index=tickers)
    if len(returns) > 0:
        # Compound returns over full history for rough price proxy
        compound = (1 + returns[tickers]).prod()
        return base_prices * compound
    return base_prices


def validate_allocation(
    weights: pd.Series,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate allocation meets constraints.

    Args:
        weights: Portfolio weights
        config: Configuration with constraints

    Returns:
        Dict with validation results
    """
    config = config or {}
    alloc_config = config.get('allocation', {})

    max_weight = alloc_config.get('max_weight', 0.25)
    min_weight = alloc_config.get('min_weight', 0.0)

    issues = []

    # Check sum
    weight_sum = weights.sum()
    if abs(weight_sum - 1.0) > 0.01:
        issues.append(f"Weights sum to {weight_sum:.4f}, not 1.0")

    # Check max weight
    max_actual = weights.max()
    if max_actual > max_weight + 0.01:
        issues.append(f"Max weight {max_actual:.2%} exceeds limit {max_weight:.2%}")

    # Check min weight
    active_weights = weights[weights > 0.001]
    if len(active_weights) > 0:
        min_actual = active_weights.min()
        if min_actual < min_weight - 0.001:
            issues.append(f"Min weight {min_actual:.2%} below limit {min_weight:.2%}")

    # Check negative weights
    if (weights < -0.001).any():
        neg_tickers = weights[weights < -0.001].index.tolist()
        issues.append(f"Negative weights for: {neg_tickers}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'weight_sum': weight_sum,
        'max_weight': max_actual,
        'min_weight': active_weights.min() if len(active_weights) > 0 else 0.0,
        'n_positions': len(active_weights),
    }
