import pytest
import asyncio
from crypto_trading.websocket_manager import KrakenWebSocketClient

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connects to Kraken"""
    client = KrakenWebSocketClient(
        url="wss://ws.kraken.com/",
        pairs=["XBT/USD"]
    )

    # Connect but don't wait for data
    task = asyncio.create_task(client.connect())
    await asyncio.sleep(2)  # Give it time to connect

    assert client.is_connected()

    await client.disconnect()
    task.cancel()


@pytest.mark.asyncio
async def test_tick_event_callback():
    """Test tick events are emitted correctly"""
    received_ticks = []

    def on_tick(tick):
        received_ticks.append(tick)

    client = KrakenWebSocketClient(
        url="wss://ws.kraken.com/",
        pairs=["XBT/USD"],
        on_tick=on_tick
    )

    # Connect and wait for at least one tick
    task = asyncio.create_task(client.connect())

    # Wait for ticks (max 10 seconds)
    for _ in range(10):
        await asyncio.sleep(1)
        if received_ticks:
            break

    assert len(received_ticks) > 0, "Should receive at least one tick"

    tick = received_ticks[0]
    assert tick.exchange == "kraken"
    assert tick.symbol == "XBT/USD"
    assert tick.price > 0
    assert tick.bid > 0
    assert tick.ask > 0
    assert tick.ask >= tick.bid

    await client.disconnect()
    task.cancel()
