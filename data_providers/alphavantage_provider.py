"""
Alpha Vantage Data Provider
============================

Concrete implementation of DerivativesProvider for Alpha Vantage API.

Features:
- Real-time and historical stock data
- Options data (premium tier required)
- Fundamental data
- Technical indicators
- Free tier: 25 requests/day, 5 requests/minute
"""

from typing import List, Optional, Tuple
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from .interface import (
    DerivativesProvider,
    ProviderType,
    OptionQuote,
    FutureQuote,
    InstrumentType
)


class AlphaVantageProvider(DerivativesProvider):
    """
    Alpha Vantage data provider implementation.

    API Key required: Get from https://www.alphavantage.co/support/#api-key
    Free tier: 25 requests/day, 5 requests/minute
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage provider.

        Parameters
        ----------
        api_key : str, optional
            Alpha Vantage API key. If not provided, reads from env var ALPHAVANTAGE_API_KEY
        """
        super().__init__(ProviderType.ALPHAVANTAGE)

        import os
        self.api_key = api_key or os.environ.get('ALPHAVANTAGE_API_KEY')

        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. Set ALPHAVANTAGE_API_KEY env var "
                "or pass api_key parameter. Get free key at: "
                "https://www.alphavantage.co/support/#api-key"
            )

        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Free tier: 5 requests/minute = 12 seconds between calls
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, params: dict) -> dict:
        """
        Make API request with rate limiting and error handling.

        Parameters
        ----------
        params : dict
            Query parameters

        Returns
        -------
        dict
            JSON response
        """
        self._rate_limit()

        params['apikey'] = self.api_key

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            if 'Note' in data:
                # Rate limit exceeded
                raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

            return data

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Alpha Vantage request failed: {e}")

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
    ) -> pd.DataFrame:
        """
        Get option chain for a symbol.

        Note: Alpha Vantage options data requires premium subscription.
        This is a placeholder implementation.

        Parameters
        ----------
        symbol : str
            Ticker symbol
        expiration : str, optional
            Expiration date (YYYY-MM-DD)
        instrument_type : InstrumentType
            Type of instrument

        Returns
        -------
        pd.DataFrame
            Empty DataFrame (options require premium tier)
        """
        print(f"  Warning: Alpha Vantage options data requires premium subscription")
        print(f"    Using free tier - options chain not available")

        return pd.DataFrame()

    def get_expirations(
        self,
        symbol: str,
        instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
    ) -> List[str]:
        """
        Get available expiration dates.

        Parameters
        ----------
        symbol : str
            Ticker symbol
        instrument_type : InstrumentType
            Type of instrument

        Returns
        -------
        List[str]
            Empty list (options require premium tier)
        """
        return []

    def get_futures_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get futures chain.

        Note: Alpha Vantage doesn't provide futures data.

        Returns
        -------
        pd.DataFrame
            Empty DataFrame
        """
        return pd.DataFrame()

    def get_option_greeks(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        option_type: str
    ) -> dict:
        """
        Get option Greeks.

        Note: Requires premium subscription.

        Returns
        -------
        dict
            Empty dict (premium tier required)
        """
        return {}

    def get_stock_quote(self, symbol: str) -> dict:
        """
        Get real-time stock quote.

        Parameters
        ----------
        symbol : str
            Ticker symbol

        Returns
        -------
        dict
            Quote data with price, volume, etc.
        """
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }

        data = self._make_request(params)

        quote = data.get('Global Quote', {})

        if not quote:
            return {}

        return {
            'symbol': symbol,
            'price': float(quote.get('05. price', 0)),
            'volume': int(quote.get('06. volume', 0)),
            'latest_trading_day': quote.get('07. latest trading day', ''),
            'change': float(quote.get('09. change', 0)),
            'change_percent': quote.get('10. change percent', '0%'),
        }

    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """
        Get historical price data.

        Parameters
        ----------
        symbol : str
            Ticker symbol
        start_date : str, optional
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date (YYYY-MM-DD)
        interval : str
            Time interval: 'daily', 'weekly', 'monthly'

        Returns
        -------
        pd.DataFrame
            Historical price data
        """
        # Map interval
        function_map = {
            'daily': 'TIME_SERIES_DAILY',
            'weekly': 'TIME_SERIES_WEEKLY',
            'monthly': 'TIME_SERIES_MONTHLY',
        }

        function = function_map.get(interval, 'TIME_SERIES_DAILY')

        params = {
            'function': function,
            'symbol': symbol,
            'outputsize': 'full',  # Get full history (compact = last 100 data points)
        }

        data = self._make_request(params)

        # Extract time series data
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break

        if not time_series_key:
            return pd.DataFrame()

        time_series = data[time_series_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns (remove '1. ', '2. ' prefixes)
        df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filter by date range if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def search_symbol(self, keywords: str) -> List[dict]:
        """
        Search for symbols matching keywords.

        Parameters
        ----------
        keywords : str
            Search keywords

        Returns
        -------
        List[dict]
            List of matching symbols with metadata
        """
        params = {
            'function': 'SYMBOL_SEARCH',
            'keywords': keywords
        }

        data = self._make_request(params)

        matches = data.get('bestMatches', [])

        results = []
        for match in matches:
            results.append({
                'symbol': match.get('1. symbol', ''),
                'name': match.get('2. name', ''),
                'type': match.get('3. type', ''),
                'region': match.get('4. region', ''),
                'currency': match.get('8. currency', ''),
            })

        return results


# Example usage
if __name__ == '__main__':
    # Requires API key
    import os

    api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
    if not api_key:
        print("Set ALPHAVANTAGE_API_KEY environment variable")
        print("Get free key at: https://www.alphavantage.co/support/#api-key")
    else:
        provider = AlphaVantageProvider(api_key)

        # Get quote
        quote = provider.get_stock_quote('AAPL')
        print(f"AAPL Quote: ${quote.get('price', 0):.2f}")

        # Search symbols
        results = provider.search_symbol('Apple')
        print(f"Search results: {len(results)} matches")
