"""
Signal generation engine with EMA crossover detection.
"""
import pandas as pd
import talib
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trade signal with metadata."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    timestamp: datetime
    ema_fast: float
    ema_slow: float
    price: float
    confidence: float = 1.0


class EMACalculator:
    """Calculate Exponential Moving Average."""

    def __init__(self, period: int):
        """
        Initialize EMA calculator.

        Args:
            period: EMA period
        """
        self.period = period

    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate EMA using talib.

        Args:
            prices: Price series

        Returns:
            EMA series
        """
        if len(prices) < self.period:
            # Return NaN series if insufficient data
            return pd.Series([float('nan')] * len(prices))

        # Convert to float64 for talib
        prices_float = prices.astype('float64')
        ema = talib.EMA(prices_float.values, timeperiod=self.period)
        return pd.Series(ema, index=prices.index)


class CrossoverDetector:
    """Detect EMA crossovers."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize crossover detector.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_calc = EMACalculator(fast_period)
        self.slow_calc = EMACalculator(slow_period)

    def detect(
        self,
        prices: pd.Series
    ) -> Optional[Tuple[str, float, float]]:
        """
        Detect crossover from price series.

        Args:
            prices: Price series

        Returns:
            Tuple of (direction, ema_fast, ema_slow) if crossover detected, else None
        """
        if len(prices) < self.slow_period + 1:
            return None

        # Calculate EMAs
        ema_fast = self.fast_calc.calculate(prices)
        ema_slow = self.slow_calc.calculate(prices)

        # Check last two values for crossover
        if pd.isna(ema_fast.iloc[-2:]).any() or pd.isna(ema_slow.iloc[-2:]).any():
            return None

        prev_fast = ema_fast.iloc[-2]
        prev_slow = ema_slow.iloc[-2]
        curr_fast = ema_fast.iloc[-1]
        curr_slow = ema_slow.iloc[-1]

        # Bullish crossover: fast crosses above slow
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            logger.info(f"Bullish crossover detected: EMA{self.fast_period}={curr_fast:.2f} > EMA{self.slow_period}={curr_slow:.2f}")
            return ("BUY", curr_fast, curr_slow)

        # Bearish crossover: fast crosses below slow
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            logger.info(f"Bearish crossover detected: EMA{self.fast_period}={curr_fast:.2f} < EMA{self.slow_period}={curr_slow:.2f}")
            return ("SELL", curr_fast, curr_slow)

        return None


class SignalEngine:
    """Main signal generation engine."""

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        min_bars_required: int = 51
    ):
        """
        Initialize signal engine.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            min_bars_required: Minimum bars needed for signal generation
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_bars_required = min_bars_required
        self.detector = CrossoverDetector(fast_period, slow_period)

    async def process_tick(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame
    ) -> Optional[TradeSignal]:
        """
        Process tick and generate signal if crossover detected.

        Args:
            symbol: Trading pair symbol
            ohlcv_df: DataFrame with OHLCV data

        Returns:
            TradeSignal if crossover detected, else None
        """
        if len(ohlcv_df) < self.min_bars_required:
            logger.debug(
                f"Insufficient bars for {symbol}: "
                f"{len(ohlcv_df)}/{self.min_bars_required}"
            )
            return None

        # Use close prices for EMA calculation
        prices = ohlcv_df['close']

        # Detect crossover
        crossover = self.detector.detect(prices)

        if not crossover:
            return None

        direction, ema_fast, ema_slow = crossover

        # Create trade signal
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            timestamp=datetime.now(),
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            price=prices.iloc[-1],
            confidence=1.0
        )

        logger.info(
            f"Signal generated: {direction} {symbol} @ {signal.price:.2f} "
            f"(EMA{self.fast_period}={ema_fast:.2f}, EMA{self.slow_period}={ema_slow:.2f})"
        )

        return signal
