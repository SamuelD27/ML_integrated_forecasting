"""
FRED Data Provider
==================
Federal Reserve Economic Data (FRED) provider for macroeconomic indicators.
Provides access to GDP, inflation, unemployment, yield curves, and market volatility.

Features:
- Free API from St. Louis Federal Reserve
- 800,000+ economic time series
- Daily, monthly, quarterly data
- Automatic frequency alignment
- Forward-fill for missing data

Key Indicators:
- GDP: Real Gross Domestic Product
- UNRATE: Unemployment Rate
- CPIAUCSL: Consumer Price Index (CPI)
- FEDFUNDS: Federal Funds Rate
- T10Y2Y: 10-Year minus 2-Year Treasury Yield Spread (recession indicator)
- DGS10: 10-Year Treasury Constant Maturity Rate
- VIXCLS: CBOE Volatility Index (VIX)
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

from .interface import DataProviderInterface, DataProviderError

logger = logging.getLogger(__name__)


class FREDProvider(DataProviderInterface):
    """
    FRED (Federal Reserve Economic Data) provider.

    Fetches macroeconomic indicators from the Federal Reserve Economic Data API.
    Free API key available at: https://fred.stlouisfed.org/docs/api/api_key.html

    Example:
        >>> provider = FREDProvider(api_key=os.getenv('FRED_API_KEY'))
        >>> macro_data = provider.get_all_indicators('2020-01-01', '2023-12-31')
        >>> gdp_growth = provider.get_gdp_growth('2023-01-01', '2023-12-31')
    """

    # Default indicators to fetch
    DEFAULT_INDICATORS = {
        'GDP': 'Real Gross Domestic Product',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'FEDFUNDS': 'Federal Funds Rate',
        'T10Y2Y': '10Y-2Y Treasury Spread',
        'DGS10': '10-Year Treasury Rate',
        'VIXCLS': 'VIX Volatility Index',
        'DCOILWTICO': 'WTI Crude Oil Price',
        'DEXUSEU': 'USD/EUR Exchange Rate',
        'M2SL': 'M2 Money Supply'
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize FRED data provider.

        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries

        Raises:
            DataProviderError: If fredapi not installed or API key missing
        """
        super().__init__(name="FRED", timeout=timeout, retry_attempts=retry_attempts)

        if not FRED_AVAILABLE:
            raise DataProviderError(
                "fredapi not installed. Install: pip install fredapi"
            )

        # Get API key from args or environment
        self.api_key = api_key or os.getenv('FRED_API_KEY')

        if not self.api_key:
            raise DataProviderError(
                "FRED API key not provided. Set FRED_API_KEY environment variable "
                "or pass to constructor. Get free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.retry_delay = retry_delay

        # Initialize client
        try:
            self.client = Fred(api_key=self.api_key)
            logger.info("FRED provider initialized successfully")
        except Exception as e:
            raise DataProviderError(f"Failed to initialize FRED client: {e}")

    def get_indicator(
        self,
        series_id: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch single economic indicator.

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            Series with indicator values

        Raises:
            DataProviderError: If fetch fails
        """
        logger.info(f"Fetching FRED series {series_id}: {start_date} to {end_date}")

        try:
            # Parse dates
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

            # Fetch with retry
            series = self._retry_request(
                lambda: self.client.get_series(series_id, observation_start=start, observation_end=end)
            )

            if series.empty:
                logger.warning(f"No data for FRED series {series_id}")
                return pd.Series(dtype=float)

            series.name = series_id
            logger.info(f"Fetched {len(series)} observations for {series_id}")
            return series

        except Exception as e:
            error_msg = f"FRED fetch failed for {series_id}: {e}"
            logger.error(error_msg)
            raise DataProviderError(error_msg)

    def get_all_indicators(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        indicators: Optional[Dict[str, str]] = None,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Fetch multiple economic indicators and align to common frequency.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            indicators: Dict of {series_id: description}, uses defaults if None
            frequency: Resample frequency ('daily', 'weekly', 'monthly')

        Returns:
            DataFrame with aligned indicator columns

        Example:
            >>> indicators = provider.get_all_indicators('2020-01-01', '2023-12-31')
            >>> print(indicators.columns)
            Index(['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', ...])
        """
        indicators = indicators or self.DEFAULT_INDICATORS
        logger.info(f"Fetching {len(indicators)} FRED indicators")

        series_dict = {}

        for series_id in indicators.keys():
            try:
                series = self.get_indicator(series_id, start_date, end_date)
                if not series.empty:
                    series_dict[series_id] = series
            except DataProviderError as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
                continue

        if not series_dict:
            logger.error("No indicators fetched successfully")
            return pd.DataFrame()

        # Combine into DataFrame
        df = pd.DataFrame(series_dict)

        # Resample to common frequency
        df = self._resample_to_frequency(df, frequency)

        # Forward-fill missing values (macro data is lower frequency)
        df = df.ffill()

        logger.info(f"Aligned {len(df.columns)} indicators to {frequency} frequency: {len(df)} rows")
        return df

    def get_gdp_growth(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate GDP year-over-year growth rate.

        Args:
            start_date: Start date
            end_date: End date
            annualize: Return annualized growth rate

        Returns:
            Series with GDP growth rates
        """
        gdp = self.get_indicator('GDP', start_date, end_date)

        if gdp.empty or len(gdp) < 2:
            return pd.Series(dtype=float)

        # Calculate YoY growth (GDP is quarterly)
        # Shift by 4 quarters for YoY comparison
        gdp_growth = gdp.pct_change(periods=4)

        if annualize:
            gdp_growth = (1 + gdp_growth) ** 1 - 1  # Already annualized for YoY

        gdp_growth.name = 'GDP_Growth'
        return gdp_growth

    def get_inflation_rate(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Calculate inflation rate (YoY CPI change).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Series with inflation rates
        """
        cpi = self.get_indicator('CPIAUCSL', start_date, end_date)

        if cpi.empty or len(cpi) < 12:
            return pd.Series(dtype=float)

        # Calculate YoY change (CPI is monthly)
        inflation = cpi.pct_change(periods=12)
        inflation.name = 'Inflation_Rate'
        return inflation

    def get_yield_curve_slope(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get yield curve slope (10Y-2Y spread).

        A negative slope (inversion) historically precedes recessions.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Series with yield curve slopes (in percentage points)
        """
        # FRED has pre-calculated 10Y-2Y spread
        slope = self.get_indicator('T10Y2Y', start_date, end_date)
        slope.name = 'Yield_Curve_Slope'
        return slope

    def get_recession_indicator(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get recession indicator based on yield curve inversion.

        Returns 1 if yield curve inverted (10Y-2Y < 0), else 0.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Series with recession signals (0 or 1)
        """
        slope = self.get_yield_curve_slope(start_date, end_date)

        if slope.empty:
            return pd.Series(dtype=int)

        recession_signal = (slope < 0).astype(int)
        recession_signal.name = 'Recession_Signal'
        return recession_signal

    def calculate_macro_features(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate derived macro features for ML models.

        Features:
        - GDP growth (YoY)
        - Inflation rate (YoY CPI)
        - Unemployment rate
        - Fed funds rate
        - Yield curve slope
        - VIX (market volatility)
        - Recession signal (yield curve inversion)

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with macro features

        Example:
            >>> features = provider.calculate_macro_features('2020-01-01')
            >>> # Join with stock data by date
            >>> stock_data = stock_data.join(features, how='left').ffill()
        """
        logger.info("Calculating macro features")

        features = {}

        # GDP growth
        try:
            features['macro_gdp_growth'] = self.get_gdp_growth(start_date, end_date)
        except Exception as e:
            logger.warning(f"GDP growth calculation failed: {e}")

        # Inflation
        try:
            features['macro_inflation'] = self.get_inflation_rate(start_date, end_date)
        except Exception as e:
            logger.warning(f"Inflation calculation failed: {e}")

        # Unemployment rate (already a rate, no calculation needed)
        try:
            features['macro_unemployment'] = self.get_indicator('UNRATE', start_date, end_date)
        except Exception as e:
            logger.warning(f"Unemployment fetch failed: {e}")

        # Fed funds rate
        try:
            features['macro_fed_funds'] = self.get_indicator('FEDFUNDS', start_date, end_date)
        except Exception as e:
            logger.warning(f"Fed funds fetch failed: {e}")

        # Yield curve slope
        try:
            features['macro_yield_curve'] = self.get_yield_curve_slope(start_date, end_date)
        except Exception as e:
            logger.warning(f"Yield curve fetch failed: {e}")

        # VIX
        try:
            features['macro_vix'] = self.get_indicator('VIXCLS', start_date, end_date)
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")

        # Recession signal
        try:
            features['macro_recession_signal'] = self.get_recession_indicator(start_date, end_date)
        except Exception as e:
            logger.warning(f"Recession signal failed: {e}")

        # Combine into DataFrame
        df = pd.DataFrame(features)

        # Resample to daily and forward-fill
        df = self._resample_to_frequency(df, 'daily')
        df = df.ffill()

        logger.info(f"Calculated {len(df.columns)} macro features")
        return df

    def _resample_to_frequency(
        self,
        df: pd.DataFrame,
        frequency: str
    ) -> pd.DataFrame:
        """
        Resample DataFrame to target frequency.

        Args:
            df: Input DataFrame with datetime index
            frequency: Target frequency ('daily', 'weekly', 'monthly')

        Returns:
            Resampled DataFrame
        """
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q'
        }

        if frequency not in freq_map:
            logger.warning(f"Unknown frequency {frequency}, using daily")
            frequency = 'daily'

        # Resample to target frequency (forward-fill for upsampling)
        resampled = df.resample(freq_map[frequency]).ffill()

        return resampled

    def _retry_request(self, func: callable, *args, **kwargs):
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
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"FRED request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"FRED request failed after {self.retry_attempts} attempts")

        raise DataProviderError(f"All retry attempts exhausted: {last_exception}")

    def test_connection(self) -> bool:
        """
        Test API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try fetching a simple series (VIX has daily data)
            series = self.get_indicator(
                'VIXCLS',
                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            return not series.empty
        except Exception as e:
            logger.error(f"FRED connection test failed: {e}")
            return False

    def get_historical_data(self, ticker: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Compatibility method for DataProviderInterface.

        FRED doesn't provide stock data, only macro indicators.
        Use get_all_indicators() or get_indicator() instead.

        Raises:
            NotImplementedError: This provider is for macro data only
        """
        raise NotImplementedError(
            "FRED provider is for macroeconomic data only. "
            "Use get_all_indicators() or get_indicator() instead."
        )
