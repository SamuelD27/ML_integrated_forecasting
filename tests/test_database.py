import pytest
import asyncio
from crypto_trading.database import DatabaseManager
from datetime import datetime
import pandas as pd

@pytest.mark.asyncio
async def test_database_connection():
    """Test database connects successfully"""
    db = DatabaseManager(
        host="localhost",
        port=5432,
        database="crypto_trading_test",
        user="postgres",
        password="test"
    )

    await db.connect()
    assert db.is_connected()
    await db.disconnect()

@pytest.mark.asyncio
async def test_write_tick():
    """Test writing tick data"""
    db = DatabaseManager(
        host="localhost",
        port=5432,
        database="crypto_trading_test",
        user="postgres",
        password="test"
    )

    await db.connect()

    await db.write_tick(
        exchange="kraken",
        symbol="XBT/USD",
        timestamp=datetime.now(),
        price=67000.0,
        volume=0.5,
        bid=66995.0,
        ask=67005.0
    )

    await db.disconnect()

@pytest.mark.asyncio
async def test_get_ohlcv():
    """Test fetching OHLCV data"""
    db = DatabaseManager(
        host="localhost",
        port=5432,
        database="crypto_trading_test",
        user="postgres",
        password="test"
    )

    await db.connect()

    df = await db.get_ohlcv(
        symbol="XBT/USD",
        interval="1m",
        limit=51
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 51
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    await db.disconnect()
