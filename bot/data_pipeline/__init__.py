"""
Bot Data Pipeline
=================
Standardized data fetching and processing pipeline for the trading bot.

This module provides:
- Universe screening (S&P 500, Russell 1000, custom)
- Unified security data profiles (price, fundamentals, technicals)
- Parallel data fetching from multiple providers
- Standardized data structures for analysis

Architecture:
    UniverseScreener -> SecurityDataFetcher -> SecurityProfile

The pipeline is designed to be:
- Provider-agnostic (supports Yahoo, Alpaca, etc.)
- Analysis-method independent
- Cacheable and efficient
"""

from .universe_screener import UniverseScreener, UniverseConfig
from .security_profile import SecurityProfile, SecurityProfileFetcher
from .data_types import (
    PriceData,
    FundamentalData,
    TechnicalIndicators,
    RiskMetrics,
    SectorInfo,
)

__all__ = [
    'UniverseScreener',
    'UniverseConfig',
    'SecurityProfile',
    'SecurityProfileFetcher',
    'PriceData',
    'FundamentalData',
    'TechnicalIndicators',
    'RiskMetrics',
    'SectorInfo',
]
