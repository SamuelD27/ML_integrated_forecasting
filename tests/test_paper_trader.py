import pytest
from crypto_trading.paper_trader import PaperTradingEngine
from crypto_trading.signal_engine import TradeSignal
from datetime import datetime

def test_paper_trading_buy():
    """Test paper trading buy execution"""
    engine = PaperTradingEngine(initial_balance=1000.0)

    signal = TradeSignal(
        symbol="XBT/USD",
        direction="BUY",
        timestamp=datetime.now(),
        ema_fast=67000.0,
        ema_slow=66500.0,
        price=67000.0
    )

    # Execute buy with $500
    execution = engine.execute(signal, amount_usd=500.0)

    assert execution is not None
    assert execution.symbol == "XBT/USD"
    assert execution.direction == "BUY"
    assert execution.amount_usd == 500.0
    assert engine.get_balance() < 1000.0  # Balance reduced
    assert len(engine.get_positions()) == 1  # Position opened


def test_paper_trading_sell():
    """Test paper trading sell execution"""
    engine = PaperTradingEngine(initial_balance=1000.0)

    # First buy
    buy_signal = TradeSignal(
        symbol="XBT/USD",
        direction="BUY",
        timestamp=datetime.now(),
        ema_fast=67000.0,
        ema_slow=66500.0,
        price=67000.0
    )
    engine.execute(buy_signal, amount_usd=500.0)

    # Then sell at higher price (profit)
    sell_signal = TradeSignal(
        symbol="XBT/USD",
        direction="SELL",
        timestamp=datetime.now(),
        ema_fast=67500.0,
        ema_slow=68000.0,
        price=68000.0
    )
    execution = engine.execute(sell_signal)

    assert execution is not None
    assert execution.direction == "SELL"
    assert len(engine.get_positions()) == 0  # Position closed
    assert engine.get_balance() > 1000.0  # Made profit
