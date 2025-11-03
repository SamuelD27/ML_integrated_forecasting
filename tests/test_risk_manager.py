import pytest
import pandas as pd
from crypto_trading.risk_manager import VolatilityCalculator, PositionSizer

def test_atr_calculation():
    """Test ATR calculation"""
    # Create sample OHLC data
    data = {
        'high': [102, 105, 103, 107, 110],
        'low': [98, 101, 99, 103, 106],
        'close': [100, 103, 101, 105, 108]
    }
    df = pd.DataFrame(data)

    calc = VolatilityCalculator(period=3)
    atr = calc.calculate_atr(df)

    assert isinstance(atr, float)
    assert atr > 0


def test_position_sizing():
    """Test position sizing with volatility adjustment"""
    # Create sample data for two symbols with moderate volatility differences
    # High volatility symbol (ATR ~ 6-7)
    high_vol_data = pd.DataFrame({
        'high': [103, 104, 103, 104, 103, 104, 103, 104, 103, 104, 103, 104, 103, 104, 103],
        'low': [97, 96, 97, 96, 97, 96, 97, 96, 97, 96, 97, 96, 97, 96, 97],
        'close': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    })

    # Low volatility symbol (ATR ~ 3-4)
    low_vol_data = pd.DataFrame({
        'high': [102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102],
        'low': [98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98],
        'close': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    })

    ohlcv_data = {
        'HIGH/VOL': high_vol_data,
        'LOW/VOL': low_vol_data
    }

    sizer = PositionSizer(atr_period=14, min_position_pct=0.02, max_position_pct=0.80)
    positions = sizer.size_positions(['HIGH/VOL', 'LOW/VOL'], balance=1000.0, ohlcv_data=ohlcv_data)

    assert len(positions) == 2

    # Find positions by symbol
    high_vol_pos = next(p for p in positions if p.symbol == 'HIGH/VOL')
    low_vol_pos = next(p for p in positions if p.symbol == 'LOW/VOL')

    # Verify volatilities are calculated
    assert high_vol_pos.volatility > 0
    assert low_vol_pos.volatility > 0

    # High volatility symbol should have higher ATR
    assert high_vol_pos.volatility > low_vol_pos.volatility

    # Low volatility should get larger allocation (inverse volatility weighting)
    assert low_vol_pos.weight > high_vol_pos.weight

    # Weights should sum to 1.0
    total_weight = sum(p.weight for p in positions)
    assert abs(total_weight - 1.0) < 0.01

    # Amounts should sum to balance
    total_amount = sum(p.amount_usd for p in positions)
    assert abs(total_amount - 1000.0) < 1.0
