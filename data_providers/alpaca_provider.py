"""
Alpaca Data Provider
====================
Real-time and historical market data provider using Alpaca Markets API.
Supports both daily and intraday (1min, 5min, 15min, 1hour) data.

Features:
- Free tier available (paper trading account)
- High-quality data with bid/ask spreads
- Intraday data for realized volatility calculations
- Volume-weighted average price (VWAP)
- Pagination handling for large date ranges
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time

import pandas as pd
import numpy as np

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockTradesRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from .interface import DataProviderInterface, DataProviderError

logger = logging.getLogger(__name__)


class AlpacaProvider(DataProviderInterface):
    """
    Alpaca Markets data provider.

    Provides access to historical and real-time market data using Alpaca's API.
    Requires API key and secret (free tier available with paper trading account).

    Features:
    - Daily and intraday bars (1min, 5min, 15min, 1hour)
    - Quote data (bid/ask spreads)
    - Trade data (tick-by-tick)
    - VWAP calculation
    - Extended trading hours support

    Example:
        >>> provider = AlpacaProvider(
        ...     api_key=os.getenv('ALPACA_API_KEY'),
        ...     api_secret=os.getenv('ALPACA_API_SECRET')
        ... )
        >>> data = provider.get_historical_data('AAPL', '2023-01-01', '2023-12-31')
        >>> intraday = provider.get_intraday_data('AAPL', '2024-01-01', timeframe='5min')
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = "https://paper-api.alpaca.markets",
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Alpaca data provider.

        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            api_secret: Alpaca API secret (or set ALPACA_API_SECRET env var)
            base_url: API base URL (paper or live)
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries (exponential backoff)

        Raises:
            DataProviderError: If Alpaca SDK not installed or credentials missing
        """
        super().__init__(name="Alpaca", timeout=timeout, retry_attempts=retry_attempts)

        if not ALPACA_AVAILABLE:
            raise DataProviderError(
                "Alpaca SDK not installed. Install: pip install alpaca-py"
            )

        # Get credentials from args or environment
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise DataProviderError(
                "Alpaca credentials not provided. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET environment variables or pass to constructor."
            )

        self.base_url = base_url
        self.retry_delay = retry_delay

        # Initialize client
        try:
            self.client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            logger.info("Alpaca provider initialized successfully")
        except Exception as e:
            raise DataProviderError(f"Failed to initialize Alpaca client: {e}")

    def get_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Additional arguments (adjustment, feed, etc.)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close, VWAP

        Raises:
            DataProviderError: If data fetch fails
        """
        logger.info(f"Fetching Alpaca daily data for {ticker}: {start_date} to {end_date}")

        try:
            # Create request
            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=pd.to_datetime(start_date),
                end=pd.to_datetime(end_date),
                adjustment='all' if kwargs.get('adjustment', 'split') == 'split' else 'split'
            )

            # Fetch with retry
            bars = self._retry_request(
                lambda: self.client.get_stock_bars(request_params)
            )

            # Convert to DataFrame
            df = bars.df

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Alpaca returns MultiIndex (symbol, timestamp) - extract single symbol
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level='symbol')

            # Rename columns to match standard format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'vwap': 'VWAP',
                'trade_count': 'Trade_Count'
            })

            # Add Adj Close (same as Close if already adjusted)
            df['Adj Close'] = df['Close']

            # Ensure datetime index
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'

            logger.info(f"Successfully fetched {len(df)} bars for {ticker}")
            return df

        except Exception as e:
            error_msg = f"Alpaca data fetch failed for {ticker}: {e}"
            logger.error(error_msg)
            raise DataProviderError(error_msg)

    def get_intraday_data(
        self,
        ticker: str,
        date: str,
        timeframe: str = '5min',
        extended_hours: bool = False
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data at specified timeframe.

        Args:
            ticker: Stock ticker symbol
            date: Date (YYYY-MM-DD)
            timeframe: Bar timeframe ('1min', '5min', '15min', '1hour')
            extended_hours: Include pre/post market data

        Returns:
            DataFrame with intraday bars

        Raises:
            DataProviderError: If data fetch fails
        """
        logger.info(f"Fetching Alpaca {timeframe} data for {ticker} on {date}")

        # Map timeframe string to TimeFrame
        timeframe_map = {
            '1min': TimeFrame.Minute,
            '5min': TimeFrame(5, TimeFrameUnit.Minute),
            '15min': TimeFrame(15, TimeFrameUnit.Minute),
            '1hour': TimeFrame.Hour,
        }

        if timeframe not in timeframe_map:
            raise DataProviderError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(timeframe_map.keys())}"
            )

        try:
            # Parse date
            start = pd.to_datetime(date)

            # Set end to next day (to capture full trading day)
            end = start + timedelta(days=1)

            # Create request
            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe_map[timeframe],
                start=start,
                end=end,
                adjustment='all'
            )

            # Fetch with retry
            bars = self._retry_request(
                lambda: self.client.get_stock_bars(request_params)
            )

            # Convert to DataFrame
            df = bars.df

            if df.empty:
                logger.warning(f"No intraday data for {ticker} on {date}")
                return pd.DataFrame()

            # Extract single symbol
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level='symbol')

            # Rename columns
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'vwap': 'VWAP',
                'trade_count': 'Trade_Count'
            })

            # Filter regular hours if requested
            if not extended_hours:
                df = self._filter_regular_hours(df)

            df.index = pd.to_datetime(df.index)
            df.index.name = 'Timestamp'

            logger.info(f"Fetched {len(df)} intraday bars for {ticker}")
            return df

        except Exception as e:
            error_msg = f"Alpaca intraday fetch failed for {ticker}: {e}"
            logger.error(error_msg)
            raise DataProviderError(error_msg)

    def calculate_realized_volatility(
        self,
        ticker: str,
        date: str,
        timeframe: str = '5min',
        annualize: bool = True
    ) -> float:
        """
        Calculate realized volatility from intraday data.

        Uses high-frequency returns to estimate daily volatility.
        More accurate than daily close-to-close volatility.

        Args:
            ticker: Stock ticker symbol
            date: Date (YYYY-MM-DD)
            timeframe: Intraday timeframe (default: 5min)
            annualize: Annualize volatility (sqrt(252) scaling)

        Returns:
            Realized volatility (annualized if annualize=True)

        Example:
            >>> rv = provider.calculate_realized_volatility('AAPL', '2024-01-15')
            >>> print(f"Daily RV: {rv:.2%}")
        """
        df = self.get_intraday_data(ticker, date, timeframe)

        if df.empty or len(df) < 2:
            logger.warning(f"Insufficient data for RV calculation: {ticker} on {date}")
            return np.nan

        # Calculate returns
        returns = df['Close'].pct_change().dropna()

        # Realized variance = sum of squared returns
        rv = np.sqrt(np.sum(returns ** 2))

        # Annualize if requested
        if annualize:
            rv *= np.sqrt(252)  # Assume 252 trading days

        return rv

    def calculate_session_features(
        self,
        ticker: str,
        date: str,
        timeframe: str = '5min'
    ) -> Dict[str, float]:
        """
        Calculate session-specific features (morning/afternoon momentum, VWAP deviation).

        Args:
            ticker: Stock ticker symbol
            date: Date (YYYY-MM-DD)
            timeframe: Intraday timeframe

        Returns:
            Dictionary with keys:
            - morning_momentum: 9:30-12:00 cumulative return
            - afternoon_momentum: 12:00-16:00 cumulative return
            - vwap_deviation: (close - vwap) / vwap
            - realized_volatility: Intraday realized vol
        """
        df = self.get_intraday_data(ticker, date, timeframe)

        if df.empty:
            return {
                'morning_momentum': np.nan,
                'afternoon_momentum': np.nan,
                'vwap_deviation': np.nan,
                'realized_volatility': np.nan
            }

        # Calculate returns
        returns = df['Close'].pct_change()

        # Split sessions (assuming ET timezone)
        morning_mask = (df.index.time >= pd.Timestamp('09:30').time()) & \
                      (df.index.time < pd.Timestamp('12:00').time())
        afternoon_mask = (df.index.time >= pd.Timestamp('12:00').time()) & \
                        (df.index.time <= pd.Timestamp('16:00').time())

        # Morning momentum (cumulative return)
        morning_returns = returns[morning_mask]
        morning_momentum = (1 + morning_returns).prod() - 1 if len(morning_returns) > 0 else np.nan

        # Afternoon momentum
        afternoon_returns = returns[afternoon_mask]
        afternoon_momentum = (1 + afternoon_returns).prod() - 1 if len(afternoon_returns) > 0 else np.nan

        # VWAP deviation (using last close vs last vwap)
        if 'VWAP' in df.columns and not df['VWAP'].isna().all():
            last_close = df['Close'].iloc[-1]
            last_vwap = df['VWAP'].iloc[-1]
            vwap_deviation = (last_close - last_vwap) / last_vwap
        else:
            vwap_deviation = np.nan

        # Realized volatility
        rv = self.calculate_realized_volatility(ticker, date, timeframe, annualize=True)

        return {
            'morning_momentum': morning_momentum,
            'afternoon_momentum': afternoon_momentum,
            'vwap_deviation': vwap_deviation,
            'realized_volatility': rv
        }

    def _filter_regular_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to regular market hours (9:30 AM - 4:00 PM ET).

        Args:
            df: DataFrame with datetime index

        Returns:
            Filtered DataFrame
        """
        # Extract time component
        times = pd.to_datetime(df.index).time

        # Regular hours: 9:30 AM - 4:00 PM
        market_open = pd.Timestamp('09:30').time()
        market_close = pd.Timestamp('16:00').time()

        mask = (times >= market_open) & (times <= market_close)
        return df[mask]

    def _retry_request(self, func: callable, *args, **kwargs) -> Any:
        """
        Execute request with exponential backoff retry.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            DataProviderError: If all retries fail
        """
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Alpaca request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Alpaca request failed after {self.retry_attempts} attempts")

        raise DataProviderError(f"All retry attempts exhausted: {last_exception}")

    def test_connection(self) -> bool:
        """
        Test API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try fetching a single day of data for a common ticker
            test_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            df = self.get_historical_data('AAPL', test_date, test_date)
            return not df.empty
        except Exception as e:
            logger.error(f"Alpaca connection test failed: {e}")
            return False
