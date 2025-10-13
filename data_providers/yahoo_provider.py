import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
import time
import warnings

from .interface import DerivativesProvider, ProviderType, InstrumentType, OptionQuote

warnings.filterwarnings('ignore')


class YahooFinanceProvider(DerivativesProvider):
    """Yahoo Finance derivatives provider."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/cache"):
        super().__init__(api_key, cache_dir)
        self.provider_type = ProviderType.YAHOO_FINANCE
        self._rate_limit_delay = 0.1

    def get_option_chain(self, symbol: str, expiration: Optional[str] = None,
                         instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
                         ) -> pd.DataFrame:
        """Fetch options chain from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return pd.DataFrame()

            # If specific expiration requested
            if expiration:
                if expiration not in expirations:
                    # Find closest expiration
                    exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                    closest = min(expirations,
                                  key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d').date() - exp_date).days))
                    expiration = closest
                expirations = [expiration]

            # Fetch all requested expirations
            all_options = []
            for exp in expirations:
                try:
                    chain = ticker.option_chain(exp)

                    # Process calls
                    calls = chain.calls.copy()
                    calls['option_type'] = 'call'
                    calls['expiration'] = exp

                    # Process puts
                    puts = chain.puts.copy()
                    puts['option_type'] = 'put'
                    puts['expiration'] = exp

                    # Combine
                    options = pd.concat([calls, puts], ignore_index=True)
                    options['underlying'] = symbol
                    options['instrument_type'] = instrument_type.value

                    all_options.append(options)

                    time.sleep(self._rate_limit_delay)

                except Exception as e:
                    print(f"  Warning: Failed to fetch {symbol} chain for {exp}: {e}")
                    continue

            if not all_options:
                return pd.DataFrame()

            df = pd.concat(all_options, ignore_index=True)
            return self._normalize_option_chain(df)

        except Exception as e:
            print(f"  Error fetching {symbol} options from Yahoo: {e}")
            return pd.DataFrame()

    def get_expirations(self, symbol: str,
                       instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
                       ) -> List[str]:
        """Get available expiration dates."""
        try:
            ticker = yf.Ticker(symbol)
            return list(ticker.options)
        except Exception:
            return []

    def get_futures_chain(self, underlying: str) -> pd.DataFrame:
        """Yahoo Finance doesn't provide futures data."""
        return pd.DataFrame()

    def _normalize_option_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Yahoo Finance option data to standard schema."""
        if df.empty:
            return df

        # Map Yahoo columns to standard schema
        column_map = {
            'contractSymbol': 'symbol',
            'strike': 'strike',
            'lastPrice': 'last',
            'bid': 'bid',
            'ask': 'ask',
            'volume': 'volume',
            'openInterest': 'open_interest',
            'impliedVolatility': 'implied_volatility',
        }

        # Rename columns
        df = df.rename(columns=column_map)

        # Ensure required columns exist
        required_cols = ['symbol', 'underlying', 'strike', 'expiration', 'option_type',
                        'bid', 'ask', 'last', 'volume', 'open_interest']

        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan if col in ['bid', 'ask', 'last', 'strike'] else 0

        # Fill missing values
        df['volume'] = df['volume'].fillna(0).astype(int)
        df['open_interest'] = df['open_interest'].fillna(0).astype(int)

        # Calculate mid price
        df['mid_price'] = ((df['bid'] + df['ask']) / 2).where(
            (df['bid'] > 0) & (df['ask'] > 0),
            df['last']
        )

        # Add DTE (days to expiration)
        today = datetime.now().date()
        df['dte'] = df['expiration'].apply(
            lambda x: (datetime.strptime(x, '%Y-%m-%d').date() - today).days
        )

        # Add moneyness (will need current price - approximate from chain)
        # This is a simplified calculation
        if 'lastTradeDate' in df.columns:
            pass  # Could extract from last trade

        # Select and order columns
        standard_cols = [
            'symbol', 'underlying', 'strike', 'expiration', 'option_type',
            'bid', 'ask', 'last', 'mid_price', 'volume', 'open_interest',
            'implied_volatility', 'dte', 'instrument_type'
        ]

        # Add greeks if available (Yahoo typically doesn't provide them directly)
        greek_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for col in greek_cols:
            if col not in df.columns:
                df[col] = np.nan

        all_cols = standard_cols + greek_cols
        existing_cols = [c for c in all_cols if c in df.columns]

        return df[existing_cols].copy()

    def is_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        try:
            # Quick test fetch
            ticker = yf.Ticker("SPY")
            info = ticker.info
            return info is not None
        except Exception:
            return False
