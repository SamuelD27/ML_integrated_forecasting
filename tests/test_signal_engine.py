import pytest
import pandas as pd
import numpy as np
from crypto_trading.signal_engine import EMACalculator, CrossoverDetector

def test_ema_calculation():
    """Test EMA calculation matches expected values"""
    # Create sample price data
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    calc = EMACalculator(period=5)
    ema = calc.calculate(prices)

    assert isinstance(ema, pd.Series)
    assert len(ema) == len(prices)
    # First 4 values should be NaN (need 5 for period=5)
    assert pd.isna(ema.iloc[0:4]).all()
    # After that should have values
    assert not pd.isna(ema.iloc[4:]).any()


def test_bullish_crossover():
    """Test bullish crossover detection"""
    # Create price series with bullish crossover
    # Start with declining prices to establish fast < slow

    # Start with declining prices to establish fast < slow
    decline = list(range(200, 100, -2))  # 50 bars decline to 100

    # Add sustained uptrend that brings fast close to slow
    sustained_rise = list(range(100, 200, 2))  # 50 bars rise back to 200

    # Add explosive move to push fast above slow
    explosion = [210, 220, 230, 240, 250, 260, 270, 280, 290, 300]

    prices = pd.Series(decline + sustained_rise + explosion, dtype=float)

    detector = CrossoverDetector(fast_period=20, slow_period=50)

    # Scan through series to find ANY bullish crossover
    crossover_found = False
    ema_fast = detector.fast_calc.calculate(prices)
    ema_slow = detector.slow_calc.calculate(prices)

    for i in range(51, len(prices)):
        prev_fast = ema_fast.iloc[i-1]
        prev_slow = ema_slow.iloc[i-1]
        curr_fast = ema_fast.iloc[i]
        curr_slow = ema_slow.iloc[i]

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            crossover_found = True
            # Test the detector at this position
            test_prices = prices.iloc[:i+1]
            result = detector.detect(test_prices)
            assert result is not None
            direction, ret_fast, ret_slow = result
            assert direction == "BUY"
            assert ret_fast > ret_slow
            break

    assert crossover_found, "No bullish crossover found in test data"


def test_bearish_crossover():
    """Test bearish crossover detection"""
    # Create price series with bearish crossover
    # Start with rising prices to establish fast > slow

    # Start with rising prices to establish fast > slow
    rise = list(range(100, 200, 2))  # 50 bars rise to 200

    # Add sustained downtrend that brings fast close to slow
    sustained_decline = list(range(200, 100, -2))  # 50 bars decline back to 100

    # Add explosive move to push fast below slow
    crash = [90, 80, 70, 60, 50, 40, 30, 20, 10, 5]

    prices = pd.Series(rise + sustained_decline + crash, dtype=float)

    detector = CrossoverDetector(fast_period=20, slow_period=50)

    # Scan through series to find ANY bearish crossover
    crossover_found = False
    ema_fast = detector.fast_calc.calculate(prices)
    ema_slow = detector.slow_calc.calculate(prices)

    for i in range(51, len(prices)):
        prev_fast = ema_fast.iloc[i-1]
        prev_slow = ema_slow.iloc[i-1]
        curr_fast = ema_fast.iloc[i]
        curr_slow = ema_slow.iloc[i]

        if prev_fast >= prev_slow and curr_fast < curr_slow:
            crossover_found = True
            # Test the detector at this position
            test_prices = prices.iloc[:i+1]
            result = detector.detect(test_prices)
            assert result is not None
            direction, ret_fast, ret_slow = result
            assert direction == "SELL"
            assert ret_fast < ret_slow
            break

    assert crossover_found, "No bearish crossover found in test data"


def test_no_crossover():
    """Test no crossover when prices are stable"""
    # Stable prices, no crossover
    prices = pd.Series([100.0] * 60)

    detector = CrossoverDetector(fast_period=20, slow_period=50)
    crossover = detector.detect(prices)

    assert crossover is None
