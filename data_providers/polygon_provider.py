"""
Polygon.io Data Provider
=========================

Concrete implementation of DerivativesProvider for Polygon.io API.

Features:
- Real-time and historical stock data
- Options data
- Comprehensive market data
- Free tier: 5 API calls/minute
- Paid tiers: Unlimited calls, real-time data
"""

from typing import List, Optional
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


class PolygonProvider(DerivativesProvider):
    """
    Polygon.io data provider implementation.

    API Key required: Get from https://polygon.io
    Free tier: 5 calls/minute
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon provider.

        Parameters
        ----------
        api_key : str, optional
            Polygon API key. If not provided, reads from env var POLYGON_API_KEY
        """
        super().__init__(ProviderType.POLYGON)

        import os
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')

        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY env var "
                "or pass api_key parameter. Get free key at: https://polygon.io"
            )

        self.base_url = "https://api.polygon.io"
        self.rate_limit_delay = 12  # Free tier: 5 requests/minute
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make API request with rate limiting.

        Parameters
        ----------
        endpoint : str
            API endpoint (e.g., '/v2/aggs/ticker/AAPL/range/1/day/...')
        params : dict, optional
            Query parameters

        Returns
        -------
        dict
            JSON response
        """
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params['apiKey'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Polygon request failed: {e}")

    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
    ) -> pd.DataFrame:
        """
        Get option chain for a symbol.

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
            Option chain data
        """
        # Get options contracts
        endpoint = f"/v3/reference/options/contracts"
        params = {
            'underlying_ticker': symbol,
            'limit': 1000
        }

        if expiration:
            params['expiration_date'] = expiration

        data = self._make_request(endpoint, params)

        results = data.get('results', [])
        if not results:
            return pd.DataFrame()

        # Get quotes for each contract (requires paid tier for real-time)
        # For free tier, return contract details only
        contracts = []
        for contract in results[:50]:  # Limit to avoid rate limits
            ticker = contract.get('ticker', '')
            strike = contract.get('strike_price', 0)
            expiry = contract.get('expiration_date', '')
            contract_type = contract.get('contract_type', '')

            contracts.append({
                'symbol': ticker,
                'underlying': symbol,
                'strike': strike,
                'expiration': expiry,
                'option_type': contract_type.lower(),
                'contract_size': contract.get('shares_per_contract', 100),
                'bid': 0,  # Requires real-time data subscription
                'ask': 0,
                'last': 0,
                'volume': 0,
                'open_interest': 0,
            })

        return pd.DataFrame(contracts)

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
            List of expiration dates (YYYY-MM-DD)
        """
        endpoint = f"/v3/reference/options/contracts"
        params = {
            'underlying_ticker': symbol,
            'limit': 1000
        }

        data = self._make_request(endpoint, params)

        results = data.get('results', [])
        expirations = sorted(set(r.get('expiration_date', '') for r in results if r.get('expiration_date')))

        return expirations

    def get_futures_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get futures chain.

        Note: Polygon supports futures but requires different subscription.

        Returns
        -------
        pd.DataFrame
            Empty DataFrame (requires futures subscription)
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
        Get latest stock quote.

        Parameters
        ----------
        symbol : str
            Ticker symbol

        Returns
        -------
        dict
            Quote data
        """
        # Get previous close (free tier)
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"

        data = self._make_request(endpoint)

        results = data.get('results', [])
        if not results:
            return {}

        quote = results[0]

        return {
            'symbol': symbol,
            'price': quote.get('c', 0),  # close
            'open': quote.get('o', 0),
            'high': quote.get('h', 0),
            'low': quote.get('l', 0),
            'volume': quote.get('v', 0),
            'timestamp': quote.get('t', 0),
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
        # Default date range
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        # Map interval
        multiplier = 1
        timespan = 'day'
        if interval == 'weekly':
            timespan = 'week'
        elif interval == 'monthly':
            timespan = 'month'

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }

        data = self._make_request(endpoint, params)

        results = data.get('results', [])
        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Rename columns
        column_map = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        }
        df = df.rename(columns=column_map)

        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date')

        # Select OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    def search_ticker(self, query: str) -> List[dict]:
        """
        Search for tickers.

        Parameters
        ----------
        query : str
            Search query

        Returns
        -------
        List[dict]
            List of matching tickers
        """
        endpoint = "/v3/reference/tickers"
        params = {
            'search': query,
            'active': 'true',
            'limit': 100
        }

        data = self._make_request(endpoint, params)

        results = data.get('results', [])

        tickers = []
        for r in results:
            tickers.append({
                'symbol': r.get('ticker', ''),
                'name': r.get('name', ''),
                'market': r.get('market', ''),
                'primary_exchange': r.get('primary_exchange', ''),
                'type': r.get('type', ''),
                'currency': r.get('currency_name', ''),
            })

        return tickers


# Example usage
if __name__ == '__main__':
    import os

    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("Set POLYGON_API_KEY environment variable")
        print("Get free key at: https://polygon.io")
    else:
        provider = PolygonProvider(api_key)

        # Get quote
        quote = provider.get_stock_quote('AAPL')
        print(f"AAPL Quote: ${quote.get('price', 0):.2f}")

        # Get historical data
        hist = provider.get_historical_data('AAPL', start_date='2024-01-01')
        print(f"Historical data: {len(hist)} days")
