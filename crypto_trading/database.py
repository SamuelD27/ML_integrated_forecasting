"""
Database layer for TimescaleDB tick storage and OHLCV queries.
"""
import asyncpg
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages TimescaleDB connection and tick storage."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_pool_size: int = 2,
        max_pool_size: int = 10
    ):
        """
        Initialize database manager.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish connection pool to database."""
        try:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size
            )
            logger.info(f"Connected to database {self.database}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection closed")

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._pool is not None and not self._pool._closed

    async def write_tick(
        self,
        exchange: str,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ) -> None:
        """
        Write tick data to database.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timestamp: Tick timestamp
            price: Last trade price
            volume: Trade volume
            bid: Bid price
            ask: Ask price
        """
        if not self.is_connected():
            raise RuntimeError("Database not connected")

        query = """
            INSERT INTO ticks (time, exchange, symbol, price, volume, bid, ask)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    query,
                    timestamp,
                    exchange,
                    symbol,
                    price,
                    volume,
                    bid,
                    ask
                )
            logger.debug(f"Wrote tick for {symbol} @ {price}")
        except Exception as e:
            logger.error(f"Failed to write tick: {e}")
            # Don't raise - we don't want to stop trading on DB errors

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from continuous aggregates.

        Args:
            symbol: Trading pair symbol
            interval: Time interval ('1m', '5m', '1h')
            limit: Maximum number of bars to return

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected():
            raise RuntimeError("Database not connected")

        # Map interval to view name
        view_map = {
            "1m": "ohlcv_1m",
            "5m": "ohlcv_5m",
            "1h": "ohlcv_1h"
        }

        view_name = view_map.get(interval)
        if not view_name:
            raise ValueError(f"Invalid interval: {interval}. Must be 1m, 5m, or 1h")

        query = f"""
            SELECT bucket, open, high, low, close, volume
            FROM {view_name}
            WHERE symbol = $1
            ORDER BY bucket DESC
            LIMIT $2
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, limit)

            if not rows:
                logger.warning(f"No OHLCV data found for {symbol}")
                return pd.DataFrame(columns=['bucket', 'open', 'high', 'low', 'close', 'volume'])

            df = pd.DataFrame(rows, columns=['bucket', 'open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('bucket').reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            raise
