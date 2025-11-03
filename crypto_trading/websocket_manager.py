"""
Kraken WebSocket client with reconnection logic.
"""
import asyncio
import websockets
import json
import logging
from typing import List, Callable, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TickEvent:
    """Tick data event."""
    exchange: str
    symbol: str
    timestamp: datetime
    price: float
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None


class KrakenWebSocketClient:
    """WebSocket client for Kraken with automatic reconnection."""

    def __init__(
        self,
        url: str,
        pairs: List[str],
        on_tick: Optional[Callable[[TickEvent], None]] = None
    ):
        """
        Initialize Kraken WebSocket client.

        Args:
            url: Kraken WebSocket URL
            pairs: List of trading pairs to subscribe
            on_tick: Callback for tick events
        """
        self.url = url
        self.pairs = pairs
        self.on_tick = on_tick
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_delay = 5  # seconds
        self._max_reconnect_delay = 60

    async def connect(self) -> None:
        """Connect to Kraken WebSocket and start listening."""
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info("Connected to Kraken WebSocket")

                    # Subscribe to ticker for all pairs
                    await self._subscribe_ticker()

                    # Reset reconnect delay on successful connection
                    self._reconnect_delay = 5

                    # Listen for messages
                    await self._listen()

            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self._connected = False
                await self._reconnect_backoff()

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._connected = False
                await self._reconnect_backoff()

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self._ws:
            await self._ws.close()
            self._ws = None
            self._connected = False
            logger.info("Disconnected from Kraken WebSocket")

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    async def _subscribe_ticker(self) -> None:
        """Subscribe to ticker feed for configured pairs."""
        message = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": {"name": "ticker"}
        }
        await self._ws.send(json.dumps(message))
        logger.info(f"Subscribed to ticker for {len(self.pairs)} pairs")

    async def _listen(self) -> None:
        """Listen for WebSocket messages."""
        async for message in self._ws:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")

    async def _handle_message(self, data: Any) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            data: Parsed JSON message
        """
        # Kraken ticker format: [channel_id, data, "ticker", pair]
        if isinstance(data, list) and len(data) == 4 and data[2] == "ticker":
            ticker_data = data[1]
            pair = data[3]

            # Extract price data
            # ticker_data format: {"a": [ask_price, ...], "b": [bid_price, ...], "c": [last_price, ...]}
            try:
                last_price = float(ticker_data["c"][0])
                bid = float(ticker_data["b"][0])
                ask = float(ticker_data["a"][0])
                volume = float(ticker_data["v"][1])  # 24h volume

                tick = TickEvent(
                    exchange="kraken",
                    symbol=pair,
                    timestamp=datetime.now(),
                    price=last_price,
                    volume=volume,
                    bid=bid,
                    ask=ask
                )

                # Validate data
                if tick.price <= 0:
                    logger.warning(f"Invalid price for {pair}: {tick.price}")
                    return

                if tick.ask < tick.bid:
                    logger.warning(f"Invalid spread for {pair}: bid={tick.bid}, ask={tick.ask}")
                    return

                # Emit tick event
                if self.on_tick:
                    self.on_tick(tick)

            except (KeyError, ValueError, IndexError) as e:
                logger.error(f"Failed to parse ticker data: {e}")

        elif isinstance(data, dict) and data.get("event") == "subscriptionStatus":
            logger.info(f"Subscription status: {data.get('status')}")

    async def _reconnect_backoff(self) -> None:
        """Wait before reconnecting with exponential backoff."""
        logger.info(f"Reconnecting in {self._reconnect_delay}s...")
        await asyncio.sleep(self._reconnect_delay)

        # Exponential backoff
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self._max_reconnect_delay
        )
