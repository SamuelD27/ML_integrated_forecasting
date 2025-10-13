"""
Data Providers Package
======================

Provider-agnostic interface for market data with multiple backends.
Supports equity options, index options, futures, and options on futures.

Available Providers:
- Yahoo Finance (free, no API key required)
- Alpha Vantage (free tier: 25 calls/day, requires API key)
- Polygon.io (free tier: 5 calls/minute, requires API key)
"""

from .interface import DerivativesProvider, ProviderType
from .manager import DataProviderManager
from .yahoo_provider import YahooFinanceProvider
from .alphavantage_provider import AlphaVantageProvider
from .polygon_provider import PolygonProvider

__all__ = [
    'DerivativesProvider',
    'ProviderType',
    'DataProviderManager',
    'YahooFinanceProvider',
    'AlphaVantageProvider',
    'PolygonProvider'
]
