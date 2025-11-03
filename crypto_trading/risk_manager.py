"""
Risk management with volatility-based position sizing.
"""
import pandas as pd
from ta.volatility import AverageTrueRange
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing information."""
    symbol: str
    weight: float  # 0.0 to 1.0
    amount_usd: float
    volatility: float


class VolatilityCalculator:
    """Calculate volatility metrics."""

    def __init__(self, period: int = 14):
        """
        Initialize volatility calculator.

        Args:
            period: ATR period
        """
        self.period = period

    def calculate_atr(self, ohlc_df: pd.DataFrame) -> float:
        """
        Calculate Average True Range.

        Args:
            ohlc_df: DataFrame with high, low, close columns

        Returns:
            ATR value
        """
        if len(ohlc_df) < self.period:
            return 0.0

        # Use ta library's AverageTrueRange
        atr_indicator = AverageTrueRange(
            high=ohlc_df['high'],
            low=ohlc_df['low'],
            close=ohlc_df['close'],
            window=self.period
        )
        atr_values = atr_indicator.average_true_range()

        # Return last ATR value
        return float(atr_values.iloc[-1]) if not pd.isna(atr_values.iloc[-1]) else 0.0


class PositionSizer:
    """Calculate position sizes based on volatility."""

    def __init__(
        self,
        atr_period: int = 14,
        min_position_pct: float = 0.02,
        max_position_pct: float = 0.20
    ):
        """
        Initialize position sizer.

        Args:
            atr_period: ATR calculation period
            min_position_pct: Minimum position size (2%)
            max_position_pct: Maximum position size (20%)
        """
        self.atr_period = atr_period
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.vol_calc = VolatilityCalculator(atr_period)

    def calculate_weights(
        self,
        symbols: List[str],
        ohlcv_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calculate allocation weights for symbols.

        Args:
            symbols: List of symbols to allocate
            ohlcv_data: Dict mapping symbol to OHLCV DataFrame

        Returns:
            Dict mapping symbol to weight (0.0 to 1.0)
        """
        if not symbols:
            return {}

        # Calculate ATR for each symbol
        volatilities = {}
        for symbol in symbols:
            if symbol not in ohlcv_data:
                logger.warning(f"No OHLCV data for {symbol}")
                continue

            atr = self.vol_calc.calculate_atr(ohlcv_data[symbol])
            if atr > 0:
                volatilities[symbol] = atr

        if not volatilities:
            logger.warning("No valid volatility data, using equal weights")
            equal_weight = 1.0 / len(symbols)
            return {s: equal_weight for s in symbols}

        # Calculate mean ATR
        mean_atr = sum(volatilities.values()) / len(volatilities)

        # Calculate weights (inverse volatility)
        weights = {}
        for symbol, atr in volatilities.items():
            # Base allocation
            base_allocation = 1.0 / len(volatilities)

            # Volatility factor (lower volatility = higher weight)
            vol_factor = mean_atr / atr

            # Combine
            weight = base_allocation * vol_factor

            # Apply bounds
            weight = max(self.min_position_pct, min(weight, self.max_position_pct))

            weights[symbol] = weight

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w / total_weight for s, w in weights.items()}

        logger.info(f"Calculated weights: {weights}")
        return weights

    def size_positions(
        self,
        symbols: List[str],
        balance: float,
        ohlcv_data: Dict[str, pd.DataFrame]
    ) -> List[PositionSize]:
        """
        Calculate position sizes for symbols.

        Args:
            symbols: Symbols to size
            balance: Available balance in USD
            ohlcv_data: OHLCV data for each symbol

        Returns:
            List of PositionSize objects
        """
        weights = self.calculate_weights(symbols, ohlcv_data)

        positions = []
        for symbol, weight in weights.items():
            amount_usd = balance * weight
            volatility = self.vol_calc.calculate_atr(ohlcv_data[symbol])

            positions.append(PositionSize(
                symbol=symbol,
                weight=weight,
                amount_usd=amount_usd,
                volatility=volatility
            ))

        return positions


class RiskManager:
    """Main risk management coordinator."""

    def __init__(
        self,
        max_portfolio_heat: float = 0.10,
        atr_period: int = 14,
        min_position_pct: float = 0.02,
        max_position_pct: float = 0.20
    ):
        """
        Initialize risk manager.

        Args:
            max_portfolio_heat: Maximum total portfolio risk (10%)
            atr_period: ATR calculation period
            min_position_pct: Minimum position size
            max_position_pct: Maximum position size
        """
        self.max_portfolio_heat = max_portfolio_heat
        self.position_sizer = PositionSizer(
            atr_period=atr_period,
            min_position_pct=min_position_pct,
            max_position_pct=max_position_pct
        )

    def validate_signal(
        self,
        symbol: str,
        balance: float,
        open_positions: int,
        ohlcv_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Validate if signal should be acted upon.

        Args:
            symbol: Symbol for signal
            balance: Current balance
            open_positions: Number of open positions
            ohlcv_data: OHLCV data

        Returns:
            True if signal is valid, False otherwise
        """
        # Check if we have data
        if symbol not in ohlcv_data:
            logger.warning(f"No OHLCV data for {symbol}, rejecting signal")
            return False

        # Simple validation for MVP
        # Future: add correlation checks, drawdown limits, etc.
        return True

    def size_position(
        self,
        symbols: List[str],
        balance: float,
        ohlcv_data: Dict[str, pd.DataFrame]
    ) -> List[PositionSize]:
        """
        Calculate position sizes for symbols.

        Args:
            symbols: Symbols to size
            balance: Available balance
            ohlcv_data: OHLCV data for symbols

        Returns:
            List of position sizes
        """
        return self.position_sizer.size_positions(symbols, balance, ohlcv_data)
