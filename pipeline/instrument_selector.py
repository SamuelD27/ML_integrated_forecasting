"""
Instrument Selector
==================
Selects appropriate instruments (equity vs options) for each trade signal.

Given a filtered list of TradeSignals, this module:
1. Determines optimal instrument type (equity, call, put)
2. Selects specific option contracts if applicable
3. Applies regime-based instrument preferences
4. Returns enriched signals with instrument details
"""

import logging
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Import pipeline types
try:
    from pipeline.core_types import TradeSignal
    HAS_PIPELINE_TYPES = True
except ImportError:
    HAS_PIPELINE_TYPES = False

# Import options loader
try:
    from data.options_loader import (
        load_options_chain,
        get_hedge_candidates,
        get_income_candidates,
        get_cached_options_chain,
        OptionsChain,
    )
    HAS_OPTIONS_LOADER = True
except ImportError:
    HAS_OPTIONS_LOADER = False
    logger.warning("Options loader not available")


class InstrumentType(Enum):
    """Types of tradeable instruments."""
    EQUITY = 'equity'
    CALL = 'call'
    PUT = 'put'
    CALL_SPREAD = 'call_spread'
    PUT_SPREAD = 'put_spread'
    COLLAR = 'collar'


@dataclass
class InstrumentSelection:
    """
    Selected instrument for a trade signal.

    Contains all information needed to execute the trade:
    - Instrument type and ticker
    - Option details if applicable
    - Position sizing information
    """
    ticker: str
    instrument_type: InstrumentType
    direction: str  # 'long' or 'short'
    quantity: int = 0
    notional_value: float = 0.0

    # Option-specific fields
    option_contract: Optional[str] = None
    strike: Optional[float] = None
    expiration: Optional[pd.Timestamp] = None
    option_premium: float = 0.0
    option_delta: float = 0.0
    n_contracts: int = 0

    # Strategy fields
    strategy_name: str = 'equity_long'
    leg2_contract: Optional[str] = None  # For spreads
    leg2_strike: Optional[float] = None

    # Source signal reference
    signal_ticker: str = ''
    meta_prob: Optional[float] = None

    def __post_init__(self):
        if not self.signal_ticker:
            self.signal_ticker = self.ticker


@dataclass
class PortfolioHedge:
    """Portfolio-level hedge recommendation."""
    hedge_ticker: str  # Usually SPY or QQQ
    instrument_type: InstrumentType
    contracts: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    portfolio_delta_reduction: float = 0.0
    recommended_contracts: int = 0
    strategy_name: str = 'protective_put'


def select_instrument(
    signal: 'TradeSignal',
    portfolio_value: float,
    current_price: float,
    config: Optional[Dict[str, Any]] = None,
) -> InstrumentSelection:
    """
    Select optimal instrument for a trade signal.

    Decision tree:
    1. If regime = crisis and direction = long -> use call options (limited downside)
    2. If high confidence and direction = long -> equity
    3. If moderate confidence -> use options for leverage/protection
    4. If direction = short (if enabled) -> puts or short equity

    Args:
        signal: TradeSignal object
        portfolio_value: Total portfolio value
        current_price: Current price of underlying
        config: Configuration dict

    Returns:
        InstrumentSelection with instrument details
    """
    config = config or {}
    instrument_config = config.get('instruments', {})

    # Get signal attributes
    ticker = signal.ticker
    direction = signal.direction
    regime = signal.regime
    meta_prob = getattr(signal, 'meta_prob', None)
    expected_return = signal.expected_return

    # Default: equity position
    instrument_type = InstrumentType.EQUITY
    strategy_name = f'equity_{direction}'

    # Check if options are enabled
    use_options = instrument_config.get('use_options', False)
    min_prob_for_equity = instrument_config.get('min_prob_for_equity', 0.70)

    # Decision logic
    if use_options and HAS_OPTIONS_LOADER:
        # In crisis regime with bullish signal, prefer calls for limited downside
        if regime == 3 and direction == 'long':
            instrument_type = InstrumentType.CALL
            strategy_name = 'long_call'

        # High volatility regime, use options for defined risk
        elif regime == 1 and direction == 'long':  # Bear market
            instrument_type = InstrumentType.CALL
            strategy_name = 'long_call'

        # Low confidence signals: use options for leverage
        elif meta_prob is not None and meta_prob < min_prob_for_equity:
            instrument_type = InstrumentType.CALL if direction == 'long' else InstrumentType.PUT
            strategy_name = 'long_call' if direction == 'long' else 'long_put'

    # Calculate position size
    position_pct = instrument_config.get('max_position_pct', 0.05)
    max_notional = portfolio_value * position_pct

    # For equities, calculate shares
    if instrument_type == InstrumentType.EQUITY:
        if current_price > 0:
            quantity = int(max_notional / current_price)
        else:
            quantity = 0

        return InstrumentSelection(
            ticker=ticker,
            instrument_type=instrument_type,
            direction=direction,
            quantity=quantity,
            notional_value=quantity * current_price,
            strategy_name=strategy_name,
            signal_ticker=ticker,
            meta_prob=meta_prob,
        )

    # For options, select specific contract
    return _select_option_instrument(
        ticker=ticker,
        instrument_type=instrument_type,
        direction=direction,
        current_price=current_price,
        max_notional=max_notional,
        meta_prob=meta_prob,
        strategy_name=strategy_name,
        config=config,
    )


def _select_option_instrument(
    ticker: str,
    instrument_type: InstrumentType,
    direction: str,
    current_price: float,
    max_notional: float,
    meta_prob: Optional[float],
    strategy_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> InstrumentSelection:
    """
    Select specific option contract.

    Args:
        ticker: Underlying ticker
        instrument_type: CALL or PUT
        direction: 'long' or 'short'
        current_price: Current underlying price
        max_notional: Maximum notional exposure
        meta_prob: Meta-model probability
        strategy_name: Strategy identifier
        config: Configuration

    Returns:
        InstrumentSelection with option details
    """
    config = config or {}

    # Load options chain
    chain = get_cached_options_chain(ticker, config=config)

    if not chain or not chain.expirations:
        # Fallback to equity if no options
        logger.warning(f"No options available for {ticker}, falling back to equity")
        quantity = int(max_notional / current_price) if current_price > 0 else 0
        return InstrumentSelection(
            ticker=ticker,
            instrument_type=InstrumentType.EQUITY,
            direction=direction,
            quantity=quantity,
            notional_value=quantity * current_price,
            strategy_name=f'equity_{direction}',
            signal_ticker=ticker,
            meta_prob=meta_prob,
        )

    # Get target expiration (30-45 DTE)
    target_dte = config.get('instruments', {}).get('target_dte', 30)
    best_expiry = min(
        chain.expirations,
        key=lambda e: abs((e - pd.Timestamp.now()).days - target_dte)
    )

    # Select strike based on delta
    target_delta = config.get('instruments', {}).get('target_delta', 0.40)

    if instrument_type == InstrumentType.CALL:
        candidates = chain.get_otm_calls(best_expiry)
        if direction == 'long':
            # For long calls, prefer slightly OTM
            target_delta = 0.40
        else:
            # For covered calls, use lower delta
            target_delta = 0.25
    else:  # PUT
        candidates = chain.get_otm_puts(best_expiry)
        target_delta = -0.30

    if not candidates:
        # Fallback to ATM if no OTM options
        if instrument_type == InstrumentType.CALL:
            contract = chain.get_atm_call(best_expiry)
        else:
            contract = chain.get_atm_put(best_expiry)

        if not contract:
            # Ultimate fallback to equity
            quantity = int(max_notional / current_price) if current_price > 0 else 0
            return InstrumentSelection(
                ticker=ticker,
                instrument_type=InstrumentType.EQUITY,
                direction=direction,
                quantity=quantity,
                notional_value=quantity * current_price,
                strategy_name=f'equity_{direction}',
                signal_ticker=ticker,
                meta_prob=meta_prob,
            )
    else:
        # Select based on delta proximity
        # Estimate delta for OTM options
        def estimate_delta(c):
            moneyness = c.strike / current_price
            if instrument_type == InstrumentType.CALL:
                return max(0.1, 1 - moneyness) if moneyness > 1 else 0.5
            else:
                return min(-0.1, moneyness - 1) if moneyness < 1 else -0.5

        contract = min(candidates, key=lambda c: abs(estimate_delta(c) - abs(target_delta)))

    # Calculate number of contracts
    contract_cost = contract.mid_price * 100  # Each contract = 100 shares
    if contract_cost > 0:
        n_contracts = max(1, int(max_notional / contract_cost))
    else:
        n_contracts = 1

    return InstrumentSelection(
        ticker=ticker,
        instrument_type=instrument_type,
        direction=direction,
        quantity=n_contracts * 100,  # Share-equivalent exposure
        notional_value=n_contracts * contract_cost,
        option_contract=contract.contract_symbol,
        strike=contract.strike,
        expiration=contract.expiration,
        option_premium=contract.mid_price,
        option_delta=estimate_delta(contract) if 'estimate_delta' in dir() else 0.0,
        n_contracts=n_contracts,
        strategy_name=strategy_name,
        signal_ticker=ticker,
        meta_prob=meta_prob,
    )


def select_instruments_batch(
    signals: List['TradeSignal'],
    portfolio_value: float,
    prices: Dict[str, float],
    config: Optional[Dict[str, Any]] = None,
) -> List[InstrumentSelection]:
    """
    Select instruments for multiple signals.

    Args:
        signals: List of TradeSignal objects
        portfolio_value: Total portfolio value
        prices: Dict of ticker -> current price
        config: Configuration

    Returns:
        List of InstrumentSelection objects
    """
    selections = []

    for signal in signals:
        price = prices.get(signal.ticker, 0)
        if price <= 0:
            logger.warning(f"No price for {signal.ticker}, skipping")
            continue

        selection = select_instrument(
            signal=signal,
            portfolio_value=portfolio_value,
            current_price=price,
            config=config,
        )
        selections.append(selection)

    return selections


def build_portfolio_hedge(
    portfolio_value: float,
    portfolio_beta: float = 1.0,
    hedge_ratio: float = 0.10,
    hedge_ticker: str = 'SPY',
    config: Optional[Dict[str, Any]] = None,
) -> Optional[PortfolioHedge]:
    """
    Build portfolio-level hedge using index puts.

    Args:
        portfolio_value: Total portfolio value
        portfolio_beta: Portfolio beta to market
        hedge_ratio: Fraction of portfolio to protect
        hedge_ticker: Index ETF for hedging (SPY, QQQ)
        config: Configuration

    Returns:
        PortfolioHedge recommendation or None
    """
    if not HAS_OPTIONS_LOADER:
        logger.warning("Options loader not available for hedging")
        return None

    config = config or {}
    hedge_config = config.get('hedging', {})

    # Get current index price
    chain = get_cached_options_chain(hedge_ticker, config=config)
    if not chain:
        logger.warning(f"No options chain for {hedge_ticker}")
        return None

    underlying_price = chain.underlying_price

    # Get hedge candidates
    candidates = get_hedge_candidates(
        ticker=hedge_ticker,
        underlying_price=underlying_price,
        portfolio_value=portfolio_value,
        hedge_ratio=hedge_ratio,
        target_delta=hedge_config.get('target_delta', -0.25),
        config=config,
    )

    if not candidates:
        return None

    # Select best candidate (first one, sorted by delta proximity)
    best = candidates[0]

    # Adjust for portfolio beta
    adjusted_contracts = int(best['n_contracts'] * portfolio_beta)
    adjusted_contracts = max(1, adjusted_contracts)

    return PortfolioHedge(
        hedge_ticker=hedge_ticker,
        instrument_type=InstrumentType.PUT,
        contracts=candidates[:3],  # Top 3 alternatives
        total_cost=adjusted_contracts * best['mid_price'] * 100,
        portfolio_delta_reduction=adjusted_contracts * best['delta'] * 100,
        recommended_contracts=adjusted_contracts,
        strategy_name='protective_put',
    )


def build_income_overlay(
    positions: Dict[str, int],  # ticker -> shares held
    prices: Dict[str, float],
    config: Optional[Dict[str, Any]] = None,
) -> List[InstrumentSelection]:
    """
    Build covered call overlay for income generation.

    Args:
        positions: Dict of ticker -> shares held
        prices: Dict of ticker -> current price
        config: Configuration

    Returns:
        List of InstrumentSelection for covered calls
    """
    if not HAS_OPTIONS_LOADER:
        return []

    config = config or {}
    income_config = config.get('income_overlay', {})
    target_delta = income_config.get('target_delta', 0.25)
    min_shares = income_config.get('min_shares', 100)

    selections = []

    for ticker, shares in positions.items():
        if shares < min_shares:
            continue

        price = prices.get(ticker, 0)
        if price <= 0:
            continue

        candidates = get_income_candidates(
            ticker=ticker,
            shares_held=shares,
            underlying_price=price,
            target_delta=target_delta,
            config=config,
        )

        if not candidates:
            continue

        # Select best candidate
        best = candidates[0]

        selections.append(InstrumentSelection(
            ticker=ticker,
            instrument_type=InstrumentType.CALL,
            direction='short',  # Selling calls
            quantity=best['n_contracts'] * 100,
            notional_value=best['total_premium'],
            option_contract=best['contract_symbol'],
            strike=best['strike'],
            expiration=best['expiration'],
            option_premium=best['mid_price'],
            option_delta=best['delta'],
            n_contracts=best['n_contracts'],
            strategy_name='covered_call',
            signal_ticker=ticker,
        ))

    return selections


def apply_regime_adjustments(
    selections: List[InstrumentSelection],
    regime: int,
    config: Optional[Dict[str, Any]] = None,
) -> List[InstrumentSelection]:
    """
    Apply regime-based adjustments to instrument selections.

    Args:
        selections: List of InstrumentSelection
        regime: Current market regime (0-3)
        config: Configuration

    Returns:
        Adjusted list of InstrumentSelection
    """
    config = config or {}
    regime_config = config.get('regime_adjustments', {})

    # Get regime-specific multipliers
    regime_multipliers = {
        0: regime_config.get('bull_multiplier', 1.0),     # Bull
        1: regime_config.get('bear_multiplier', 0.5),     # Bear
        2: regime_config.get('neutral_multiplier', 0.8),  # Neutral
        3: regime_config.get('crisis_multiplier', 0.3),   # Crisis
    }

    multiplier = regime_multipliers.get(regime, 1.0)

    adjusted = []
    for sel in selections:
        # Create copy with adjusted quantity
        new_quantity = int(sel.quantity * multiplier)
        new_contracts = int(sel.n_contracts * multiplier) if sel.n_contracts > 0 else 0

        # Skip if quantity becomes zero
        if new_quantity < 1 and sel.instrument_type == InstrumentType.EQUITY:
            continue
        if new_contracts < 1 and sel.instrument_type in [InstrumentType.CALL, InstrumentType.PUT]:
            continue

        adjusted.append(InstrumentSelection(
            ticker=sel.ticker,
            instrument_type=sel.instrument_type,
            direction=sel.direction,
            quantity=new_quantity,
            notional_value=sel.notional_value * multiplier,
            option_contract=sel.option_contract,
            strike=sel.strike,
            expiration=sel.expiration,
            option_premium=sel.option_premium,
            option_delta=sel.option_delta,
            n_contracts=new_contracts,
            strategy_name=sel.strategy_name,
            leg2_contract=sel.leg2_contract,
            leg2_strike=sel.leg2_strike,
            signal_ticker=sel.signal_ticker,
            meta_prob=sel.meta_prob,
        ))

    return adjusted


def summarize_selections(
    selections: List[InstrumentSelection],
) -> Dict[str, Any]:
    """
    Summarize instrument selections.

    Args:
        selections: List of InstrumentSelection

    Returns:
        Summary dict with counts and totals
    """
    if not selections:
        return {
            'total_selections': 0,
            'by_type': {},
            'by_strategy': {},
            'total_notional': 0.0,
            'total_option_premium': 0.0,
        }

    by_type = {}
    by_strategy = {}
    total_notional = 0.0
    total_premium = 0.0

    for sel in selections:
        # Count by type
        type_name = sel.instrument_type.value
        by_type[type_name] = by_type.get(type_name, 0) + 1

        # Count by strategy
        by_strategy[sel.strategy_name] = by_strategy.get(sel.strategy_name, 0) + 1

        # Sum notional
        total_notional += sel.notional_value

        # Sum option premium
        if sel.option_premium > 0:
            total_premium += sel.n_contracts * sel.option_premium * 100

    return {
        'total_selections': len(selections),
        'by_type': by_type,
        'by_strategy': by_strategy,
        'total_notional': total_notional,
        'total_option_premium': total_premium,
        'tickers': [s.ticker for s in selections],
    }
