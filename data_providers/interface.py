"""
Provider-Agnostic Derivatives Data Interface
============================================

Abstract base class for derivatives data providers.
Concrete implementations: YahooFinanceProvider, TradierProvider, FinnhubProvider
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import pandas as pd
from dataclasses import dataclass


class ProviderType(Enum):
    """Supported data provider types."""
    YAHOO_FINANCE = "yahoo"
    TRADIER = "tradier"
    FINNHUB = "finnhub"
    ALPHAVANTAGE = "alphavantage"
    POLYGON = "polygon"


class InstrumentType(Enum):
    """Derivative instrument types."""
    EQUITY_OPTION = "equity_option"
    INDEX_OPTION = "index_option"
    FUTURE = "future"
    OPTION_ON_FUTURE = "option_on_future"


@dataclass
class OptionQuote:
    """Standardized option quote structure."""
    symbol: str
    underlying: str
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION

    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last if self.last > 0 else 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'strike': self.strike,
            'expiration': self.expiration,
            'option_type': self.option_type,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'mid_price': self.mid_price,
            'instrument_type': self.instrument_type.value,
        }


@dataclass
class FutureQuote:
    """Standardized futures quote structure."""
    symbol: str
    underlying: str
    expiration: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    contract_size: float
    tick_size: float

    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last if self.last > 0 else 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'expiration': self.expiration,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'contract_size': self.contract_size,
            'tick_size': self.tick_size,
            'mid_price': self.mid_price,
        }


class DerivativesProvider(ABC):
    """Abstract base class for derivatives data providers."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/cache"):
        """
        Initialize provider.

        Parameters
        ----------
        api_key : str, optional
            API key for authenticated endpoints
        cache_dir : str
            Directory for local Parquet cache
        """
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.provider_type = None
        self._rate_limit_delay = 0.1  # seconds between requests

    @abstractmethod
    def get_option_chain(self, symbol: str, expiration: Optional[str] = None,
                         instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
                         ) -> pd.DataFrame:
        """
        Fetch options chain for a symbol.

        Parameters
        ----------
        symbol : str
            Underlying symbol
        expiration : str, optional
            Specific expiration date (YYYY-MM-DD). If None, fetch all available.
        instrument_type : InstrumentType
            Type of option (equity, index, etc.)

        Returns
        -------
        pd.DataFrame
            Standardized options chain
        """
        pass

    @abstractmethod
    def get_expirations(self, symbol: str,
                       instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
                       ) -> List[str]:
        """
        Get available expiration dates for a symbol.

        Parameters
        ----------
        symbol : str
            Underlying symbol
        instrument_type : InstrumentType
            Type of option

        Returns
        -------
        list
            List of expiration dates (YYYY-MM-DD format)
        """
        pass

    @abstractmethod
    def get_futures_chain(self, underlying: str) -> pd.DataFrame:
        """
        Fetch futures contracts for an underlying.

        Parameters
        ----------
        underlying : str
            Underlying symbol (e.g., 'ES', 'NQ')

        Returns
        -------
        pd.DataFrame
            Futures contracts with quotes
        """
        pass

    def get_option_greeks(self, symbol: str, expiration: str) -> pd.DataFrame:
        """
        Fetch option greeks for a specific expiration.

        Some providers include greeks in the chain, others provide separately.

        Parameters
        ----------
        symbol : str
            Underlying symbol
        expiration : str
            Expiration date

        Returns
        -------
        pd.DataFrame
            Options with greeks columns
        """
        # Default implementation: return chain with greeks if available
        return self.get_option_chain(symbol, expiration)

    def get_historical_options(self, symbol: str, start_date: str,
                               end_date: str) -> pd.DataFrame:
        """
        Fetch historical options data.

        Not all providers support this - default returns empty DataFrame.

        Parameters
        ----------
        symbol : str
            Underlying symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)

        Returns
        -------
        pd.DataFrame
            Historical options data
        """
        return pd.DataFrame()

    def is_available(self) -> bool:
        """
        Check if provider is available and properly configured.

        Returns
        -------
        bool
            True if provider can be used
        """
        return True

    def get_provider_name(self) -> str:
        """Get provider name."""
        return self.provider_type.value if self.provider_type else "unknown"

    def _normalize_option_chain(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize raw provider data to standard schema.

        Override in concrete implementations.
        """
        return raw_data

    def _cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Generate cache key for Parquet storage."""
        import hashlib
        key_parts = [symbol, data_type] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        key_str = "_".join(str(p) for p in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
