"""
Pipeline Core Types
====================
Shared data structures for the 6-phase trading pipeline.

This module defines the core dataclasses used throughout the pipeline:
- StockSnapshot: Point-in-time fundamental + factor data for a ticker
- TradeSignal: Trading signal with forecast and meta-labeling info
- PositionCandidate: Candidate position for CVaR allocation
- Forecast: Ensemble forecast output

These types ensure type safety and consistent data flow between pipeline phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any
import pandas as pd
import numpy as np


# =============================================================================
# Phase 1: Universe Building Types
# =============================================================================

@dataclass
class StockSnapshot:
    """
    Point-in-time snapshot of a stock's fundamentals and factor scores.

    Created by Phase 1 (Universe Builder) and used throughout the pipeline
    to track which stocks are in the investable universe and their characteristics.

    Attributes:
        ticker: Stock symbol (e.g., 'AAPL')
        date: Snapshot date (as_of_date)
        fundamentals: Dict of fundamental metrics (PE, EPS, ROE, etc.)
        factor_scores: Dict of factor scores (value, quality, momentum, etc.)
        current_price: Current stock price at snapshot time
        intrinsic_value: Estimated intrinsic value from DCF/multiples
        upside_pct: Percentage upside to intrinsic value
        sector: GICS sector classification
        market_cap: Market capitalization in USD
        metadata: Additional metadata (data source, confidence, etc.)

    Example:
        >>> snapshot = StockSnapshot(
        ...     ticker='AAPL',
        ...     date=pd.Timestamp('2024-01-15'),
        ...     fundamentals={'pe_ratio': 28.5, 'roe': 0.45, 'eps': 6.12},
        ...     factor_scores={'value': 0.3, 'quality': 0.8, 'momentum': 0.6},
        ...     current_price=185.50,
        ...     intrinsic_value=225.00,
        ...     upside_pct=0.213,
        ...     sector='Technology',
        ...     market_cap=2.9e12
        ... )
    """
    ticker: str
    date: pd.Timestamp
    fundamentals: Dict[str, Optional[float]] = field(default_factory=dict)
    factor_scores: Dict[str, float] = field(default_factory=dict)
    current_price: Optional[float] = None
    intrinsic_value: Optional[float] = None
    upside_pct: Optional[float] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and convert date to Timestamp."""
        if not isinstance(self.date, pd.Timestamp):
            self.date = pd.Timestamp(self.date)

    @property
    def is_value_stock(self) -> bool:
        """Check if stock qualifies as value based on factor score."""
        return self.factor_scores.get('value', 0) > 0

    @property
    def is_quality_stock(self) -> bool:
        """Check if stock qualifies as quality based on factor score."""
        return self.factor_scores.get('quality', 0) > 0

    @property
    def is_momentum_stock(self) -> bool:
        """Check if stock qualifies as momentum based on factor score."""
        return self.factor_scores.get('momentum', 0) > 0

    @property
    def has_upside(self) -> bool:
        """Check if stock has positive upside to intrinsic value."""
        return self.upside_pct is not None and self.upside_pct > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'date': self.date.isoformat(),
            'fundamentals': self.fundamentals,
            'factor_scores': self.factor_scores,
            'current_price': self.current_price,
            'intrinsic_value': self.intrinsic_value,
            'upside_pct': self.upside_pct,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockSnapshot':
        """Create from dictionary."""
        data = data.copy()
        data['date'] = pd.Timestamp(data['date'])
        return cls(**data)


# =============================================================================
# Phase 3: Forecast Types
# =============================================================================

@dataclass
class Forecast:
    """
    Ensemble forecast output from Phase 3.

    Contains expected return, volatility, and distribution metrics
    from the N-BEATS, LSTM, and TFT ensemble.

    Attributes:
        ticker: Stock symbol
        as_of_date: Date the forecast was generated
        horizon_days: Forecast horizon in trading days
        expected_return: Expected return over horizon (annualized or period)
        volatility: Forecasted volatility (annualized)
        p10: 10th percentile of return distribution
        p50: 50th percentile (median)
        p90: 90th percentile of return distribution
        model_disagreement: Standard deviation of individual model forecasts
        model_forecasts: Dict mapping model name to forecast value
        confidence: Overall confidence score (0-1)

    Example:
        >>> forecast = Forecast(
        ...     ticker='AAPL',
        ...     as_of_date=pd.Timestamp('2024-01-15'),
        ...     horizon_days=10,
        ...     expected_return=0.05,
        ...     volatility=0.25,
        ...     p10=-0.03,
        ...     p50=0.04,
        ...     p90=0.12,
        ...     model_disagreement=0.02
        ... )
    """
    ticker: str
    as_of_date: pd.Timestamp
    horizon_days: int
    expected_return: float
    expected_volatility: float = 0.0  # Also stored as 'volatility' for backwards compat
    volatility: float = 0.0
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    model_disagreement: float = 0.0
    model_forecasts: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    individual_forecasts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence: float = 0.5

    def __post_init__(self):
        """Validate forecast values."""
        if not isinstance(self.as_of_date, pd.Timestamp):
            self.as_of_date = pd.Timestamp(self.as_of_date)

        # Sync volatility and expected_volatility for backwards compatibility
        if self.expected_volatility != 0.0 and self.volatility == 0.0:
            self.volatility = self.expected_volatility
        elif self.volatility != 0.0 and self.expected_volatility == 0.0:
            self.expected_volatility = self.volatility

        # Validate percentile ordering
        if not (self.p10 <= self.p50 <= self.p90):
            # Allow some numerical tolerance
            if abs(self.p10 - self.p50) < 1e-6 and abs(self.p50 - self.p90) < 1e-6:
                pass  # All equal is fine
            else:
                # Log warning but don't fail - models may produce unusual forecasts
                pass

    @property
    def is_bullish(self) -> bool:
        """Check if forecast is bullish (positive expected return)."""
        return self.expected_return > 0

    @property
    def is_bearish(self) -> bool:
        """Check if forecast is bearish (negative expected return)."""
        return self.expected_return < 0

    @property
    def forecast_range(self) -> float:
        """Get forecast range (p90 - p10)."""
        return self.p90 - self.p10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'as_of_date': self.as_of_date.isoformat(),
            'horizon_days': self.horizon_days,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'volatility': self.volatility,
            'p10': self.p10,
            'p50': self.p50,
            'p90': self.p90,
            'model_disagreement': self.model_disagreement,
            'model_forecasts': self.model_forecasts,
            'model_weights': self.model_weights,
            'individual_forecasts': self.individual_forecasts,
            'confidence': self.confidence
        }


# =============================================================================
# Phase 4: Trade Signal Types
# =============================================================================

@dataclass
class TradeSignal:
    """
    Complete trade signal with forecast and meta-labeling information.

    Created by combining Phase 3 (forecast) and Phase 4 (meta-labeling) outputs.
    Used by Phase 5 (instrument selection) and Phase 6 (CVaR allocation).

    Attributes:
        ticker: Stock symbol
        timestamp: Signal generation time
        direction: Trade direction ('long', 'short', 'flat')
        expected_return: Expected return from forecast
        expected_vol: Expected volatility from forecast
        regime: Current market regime (0=Bull, 1=Bear, 2=Neutral, 3=Crisis)
        regime_label: Human-readable regime label
        meta_prob: Meta-model probability of success (None if not computed)
        horizon_days: Forecast horizon
        forecast: Full Forecast object (optional)
        snapshot: Stock snapshot from universe builder (optional)
        confidence_level: Signal confidence ('high', 'moderate', 'low')
        vol_level: Volatility level ('high', 'low')
        metadata: Additional signal metadata

    Example:
        >>> signal = TradeSignal(
        ...     ticker='AAPL',
        ...     timestamp=pd.Timestamp('2024-01-15 09:30:00'),
        ...     direction='long',
        ...     expected_return=0.08,
        ...     expected_vol=0.22,
        ...     regime=0,
        ...     regime_label='Bull',
        ...     meta_prob=0.72,
        ...     horizon_days=10
        ... )
    """
    ticker: str
    timestamp: pd.Timestamp
    direction: Literal['long', 'short', 'flat']
    expected_return: float
    expected_vol: float
    regime: int
    regime_label: str
    meta_prob: Optional[float] = None
    horizon_days: int = 10
    forecast: Optional[Forecast] = None
    snapshot: Optional[StockSnapshot] = None
    confidence_level: Optional[str] = None
    vol_level: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set defaults."""
        if not isinstance(self.timestamp, pd.Timestamp):
            self.timestamp = pd.Timestamp(self.timestamp)

        # Validate regime
        if self.regime not in {0, 1, 2, 3}:
            raise ValueError(f"Regime must be 0-3, got {self.regime}")

        # Validate direction
        if self.direction not in {'long', 'short', 'flat'}:
            raise ValueError(f"Direction must be 'long', 'short', or 'flat', got {self.direction}")

        # Validate meta_prob if provided
        if self.meta_prob is not None and not (0.0 <= self.meta_prob <= 1.0):
            raise ValueError(f"meta_prob must be 0-1, got {self.meta_prob}")

    @property
    def should_execute(self) -> bool:
        """
        Check if signal should be executed based on meta_prob.

        Default threshold is 0.5 - use threshold from config for real execution.
        """
        if self.meta_prob is None:
            return True  # No meta-labeling, execute by default
        return self.meta_prob >= 0.5

    @property
    def is_long(self) -> bool:
        """Check if signal is long."""
        return self.direction == 'long'

    @property
    def is_short(self) -> bool:
        """Check if signal is short."""
        return self.direction == 'short'

    @property
    def is_flat(self) -> bool:
        """Check if signal is flat (no trade)."""
        return self.direction == 'flat'

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate simple risk/reward ratio."""
        if self.expected_vol == 0:
            return 0.0
        return abs(self.expected_return) / self.expected_vol

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'expected_return': self.expected_return,
            'expected_vol': self.expected_vol,
            'regime': self.regime,
            'regime_label': self.regime_label,
            'meta_prob': self.meta_prob,
            'horizon_days': self.horizon_days,
            'confidence_level': self.confidence_level,
            'vol_level': self.vol_level,
            'metadata': self.metadata
        }
        if self.forecast:
            result['forecast'] = self.forecast.to_dict()
        if self.snapshot:
            result['snapshot'] = self.snapshot.to_dict()
        return result

    @classmethod
    def from_forecast(
        cls,
        forecast: Forecast,
        regime: int,
        regime_label: str,
        meta_prob: Optional[float] = None,
        snapshot: Optional[StockSnapshot] = None
    ) -> 'TradeSignal':
        """
        Create TradeSignal from a Forecast object.

        Automatically determines direction from expected return sign.
        """
        if forecast.expected_return > 0.01:  # 1% threshold
            direction = 'long'
        elif forecast.expected_return < -0.01:
            direction = 'short'
        else:
            direction = 'flat'

        return cls(
            ticker=forecast.ticker,
            timestamp=forecast.as_of_date,
            direction=direction,
            expected_return=forecast.expected_return,
            expected_vol=forecast.volatility,
            regime=regime,
            regime_label=regime_label,
            meta_prob=meta_prob,
            horizon_days=forecast.horizon_days,
            forecast=forecast,
            snapshot=snapshot
        )


# =============================================================================
# Phase 5: Instrument Selection Types
# =============================================================================

@dataclass
class InstrumentSelection:
    """
    Selected instrument/strategy from Phase 5.

    Describes the chosen options strategy or stock position
    based on the trade signal and market conditions.

    Attributes:
        ticker: Underlying stock symbol
        instrument_type: Type of instrument (stock, long_call, spread, etc.)
        strategy_name: Human-readable strategy name
        params: Strategy parameters (strikes, expiry, etc.)
        rationale: Why this strategy was chosen
        expected_cost: Expected cost to enter position
        max_loss: Maximum potential loss
        max_gain: Maximum potential gain (None if unlimited)
        breakeven: Breakeven price(s)
        payoff_profile: Dict with payoff at various prices
        liquidity_score: Liquidity score (0-1)
        greeks: Option Greeks if applicable

    Example:
        >>> selection = InstrumentSelection(
        ...     ticker='AAPL',
        ...     instrument_type='bull_call_spread',
        ...     strategy_name='Bull Call Spread',
        ...     params={'long_strike': 180, 'short_strike': 190, 'expiry': '2024-02-16'},
        ...     expected_cost=3.50,
        ...     max_loss=3.50,
        ...     max_gain=6.50,
        ...     breakeven=183.50
        ... )
    """
    ticker: str
    instrument_type: str
    strategy_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ''
    expected_cost: float = 0.0
    max_loss: float = 0.0
    max_gain: Optional[float] = None
    breakeven: Optional[float] = None
    payoff_profile: Dict[str, float] = field(default_factory=dict)
    liquidity_score: float = 1.0
    greeks: Dict[str, float] = field(default_factory=dict)

    @property
    def is_stock_only(self) -> bool:
        """Check if this is a stock-only position (no options)."""
        return self.instrument_type == 'stock'

    @property
    def is_options_strategy(self) -> bool:
        """Check if this involves options."""
        return self.instrument_type != 'stock'

    @property
    def risk_reward(self) -> Optional[float]:
        """Calculate risk/reward ratio."""
        if self.max_loss == 0 or self.max_gain is None:
            return None
        return self.max_gain / self.max_loss

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'instrument_type': self.instrument_type,
            'strategy_name': self.strategy_name,
            'params': self.params,
            'rationale': self.rationale,
            'expected_cost': self.expected_cost,
            'max_loss': self.max_loss,
            'max_gain': self.max_gain,
            'breakeven': self.breakeven,
            'liquidity_score': self.liquidity_score,
            'greeks': self.greeks
        }


# =============================================================================
# Phase 6: CVaR Allocation Types
# =============================================================================

@dataclass
class PositionCandidate:
    """
    Candidate position for CVaR portfolio optimization.

    Created by combining trade signal with instrument selection.
    Contains the P&L distribution needed for CVaR optimization.

    Attributes:
        ticker: Stock symbol
        instrument_type: Instrument type (stock, long_call, spread, etc.)
        params: Instrument parameters (strikes, expiry, etc.)
        forecast_distribution: Array of simulated P&L scenarios
        signal: Original trade signal
        selection: Instrument selection details
        expected_pnl: Expected P&L from distribution
        volatility_pnl: P&L volatility
        cvar_95: CVaR at 95% confidence
        position_size: Suggested position size (set after optimization)

    Example:
        >>> candidate = PositionCandidate(
        ...     ticker='AAPL',
        ...     instrument_type='long_call',
        ...     params={'strike': 185, 'expiry': '2024-02-16'},
        ...     forecast_distribution=np.random.randn(10000) * 100 + 50
        ... )
        >>> candidate.cvar_95
        -150.23  # Expected loss in worst 5% of scenarios
    """
    ticker: str
    instrument_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    forecast_distribution: Optional[np.ndarray] = None
    signal: Optional[TradeSignal] = None
    selection: Optional[InstrumentSelection] = None
    expected_pnl: float = 0.0
    volatility_pnl: float = 0.0
    cvar_95: float = 0.0
    position_size: float = 0.0

    def __post_init__(self):
        """Calculate P&L statistics from distribution."""
        if self.forecast_distribution is not None and len(self.forecast_distribution) > 0:
            self.expected_pnl = float(np.mean(self.forecast_distribution))
            self.volatility_pnl = float(np.std(self.forecast_distribution))
            self.cvar_95 = self._calculate_cvar(self.forecast_distribution, 0.95)

    @staticmethod
    def _calculate_cvar(pnl: np.ndarray, alpha: float) -> float:
        """Calculate CVaR (Expected Shortfall) at given confidence level."""
        if len(pnl) == 0:
            return 0.0
        var = np.percentile(pnl, (1 - alpha) * 100)
        cvar = np.mean(pnl[pnl <= var])
        return float(cvar) if not np.isnan(cvar) else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of P&L distribution."""
        if self.volatility_pnl == 0:
            return 0.0
        return self.expected_pnl / self.volatility_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            'ticker': self.ticker,
            'instrument_type': self.instrument_type,
            'params': self.params,
            'expected_pnl': self.expected_pnl,
            'volatility_pnl': self.volatility_pnl,
            'cvar_95': self.cvar_95,
            'position_size': self.position_size,
            'n_scenarios': len(self.forecast_distribution) if self.forecast_distribution is not None else 0
        }


# =============================================================================
# Pipeline Output Types
# =============================================================================

@dataclass
class PipelineOutput:
    """
    Complete output from daily pipeline run.

    Contains all decisions made by each phase for audit and execution.

    Attributes:
        as_of_date: Date the pipeline was run for
        universe: List of stocks in investable universe
        regime: Current market regime
        forecasts: Dict of forecasts by ticker
        signals: List of trade signals (filtered by meta-model)
        selections: Dict of instrument selections by ticker
        allocations: Dict of position allocations by ticker
        trades_to_execute: Final list of trades to execute
        metadata: Pipeline run metadata (timing, stats, etc.)
    """
    as_of_date: pd.Timestamp
    universe: List[StockSnapshot] = field(default_factory=list)
    regime: int = 0
    regime_label: str = 'Bull'
    forecasts: Dict[str, Forecast] = field(default_factory=dict)
    signals: List[TradeSignal] = field(default_factory=list)
    selections: Dict[str, InstrumentSelection] = field(default_factory=dict)
    allocations: Dict[str, float] = field(default_factory=dict)
    trades_to_execute: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.as_of_date, pd.Timestamp):
            self.as_of_date = pd.Timestamp(self.as_of_date)

    @property
    def n_universe(self) -> int:
        """Number of stocks in universe."""
        return len(self.universe)

    @property
    def n_signals(self) -> int:
        """Number of trade signals."""
        return len(self.signals)

    @property
    def n_trades(self) -> int:
        """Number of trades to execute."""
        return len(self.trades_to_execute)

    @property
    def avg_meta_prob(self) -> float:
        """Average meta probability across signals."""
        probs = [s.meta_prob for s in self.signals if s.meta_prob is not None]
        return float(np.mean(probs)) if probs else 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Pipeline Output: {self.as_of_date.strftime('%Y-%m-%d')}",
            f"  Regime: {self.regime_label} ({self.regime})",
            f"  Universe: {self.n_universe} stocks",
            f"  Signals: {self.n_signals} (avg meta_prob: {self.avg_meta_prob:.2%})",
            f"  Trades to execute: {self.n_trades}",
        ]
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'as_of_date': self.as_of_date.isoformat(),
            'regime': self.regime,
            'regime_label': self.regime_label,
            'universe': [s.to_dict() for s in self.universe],
            'forecasts': {k: v.to_dict() for k, v in self.forecasts.items()},
            'signals': [s.to_dict() for s in self.signals],
            'selections': {k: v.to_dict() for k, v in self.selections.items()},
            'allocations': self.allocations,
            'trades_to_execute': self.trades_to_execute,
            'metadata': self.metadata
        }
