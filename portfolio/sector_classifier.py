"""
Sector Classification System
=============================
Assigns stocks to sectors for sector-neutral portfolio construction.

Uses multiple methods:
1. yfinance sector data (primary)
2. Manual sector mapping (fallback)
3. Industry clustering (backup)

GICS (Global Industry Classification Standard) Sectors:
- Energy
- Materials
- Industrials
- Consumer Discretionary
- Consumer Staples
- Health Care
- Financials
- Information Technology
- Communication Services
- Utilities
- Real Estate
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class SectorClassifier:
    """
    Classify stocks into GICS sectors.

    Uses yfinance API as primary source, with manual mapping as fallback.

    Example:
        >>> classifier = SectorClassifier()
        >>> sectors = classifier.classify(['AAPL', 'MSFT', 'JPM', 'XOM'])
        >>> print(sectors)
        {'AAPL': 'Technology', 'MSFT': 'Technology',
         'JPM': 'Financials', 'XOM': 'Energy'}
    """

    # Manual sector mapping for common tickers (fallback)
    MANUAL_SECTOR_MAP = {
        # Technology
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'GOOG': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology',
        'ORCL': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology',
        'AVGO': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',

        # Financials
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
        'BLK': 'Financials', 'SCHW': 'Financials', 'AXP': 'Financials',
        'V': 'Financials', 'MA': 'Financials', 'PYPL': 'Financials',

        # Healthcare
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
        'ABBV': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
        'MRK': 'Healthcare', 'LLY': 'Healthcare', 'AMGN': 'Healthcare',
        'GILD': 'Healthcare', 'CVS': 'Healthcare',

        # Consumer Discretionary
        'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
        'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
        'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
        'LOW': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary',

        # Consumer Staples
        'WMT': 'Consumer Staples', 'PG': 'Consumer Staples',
        'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
        'COST': 'Consumer Staples', 'PM': 'Consumer Staples',

        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy',

        # Industrials
        'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials',
        'HON': 'Industrials', 'UNP': 'Industrials', 'UPS': 'Industrials',
        'LMT': 'Industrials', 'RTX': 'Industrials',

        # Materials
        'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
        'FCX': 'Materials', 'NEM': 'Materials',

        # Real Estate
        'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate',
        'EQIX': 'Real Estate', 'PSA': 'Real Estate',

        # Utilities
        'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
        'D': 'Utilities', 'AEP': 'Utilities',

        # Communication Services
        'DIS': 'Communication Services', 'NFLX': 'Communication Services',
        'CMCSA': 'Communication Services', 'T': 'Communication Services',
        'VZ': 'Communication Services', 'TMUS': 'Communication Services',
    }

    # Standard GICS sectors
    GICS_SECTORS = [
        'Energy',
        'Materials',
        'Industrials',
        'Consumer Discretionary',
        'Consumer Staples',
        'Healthcare',
        'Financials',
        'Technology',
        'Communication Services',
        'Utilities',
        'Real Estate'
    ]

    def __init__(self, use_cache: bool = True, cache_ttl_days: int = 30):
        """
        Initialize sector classifier.

        Args:
            use_cache: Whether to cache sector assignments
            cache_ttl_days: Cache time-to-live in days
        """
        self.use_cache = use_cache
        self.cache_ttl_days = cache_ttl_days
        self._cache: Dict[str, str] = {}

        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available, using manual mapping only")

    def classify(
        self,
        tickers: List[str],
        method: str = 'auto'
    ) -> Dict[str, str]:
        """
        Classify tickers into sectors.

        Args:
            tickers: List of ticker symbols
            method: Classification method ('auto', 'yfinance', 'manual')

        Returns:
            Dictionary mapping ticker -> sector

        Example:
            >>> sectors = classifier.classify(['AAPL', 'JPM', 'XOM'])
            >>> {'AAPL': 'Technology', 'JPM': 'Financials', 'XOM': 'Energy'}
        """
        logger.info(f"Classifying {len(tickers)} tickers into sectors (method: {method})")

        sectors = {}

        for ticker in tickers:
            # Check cache first
            if self.use_cache and ticker in self._cache:
                sectors[ticker] = self._cache[ticker]
                continue

            # Try classification methods
            sector = None

            if method in ['auto', 'yfinance'] and YFINANCE_AVAILABLE:
                sector = self._classify_yfinance(ticker)

            if sector is None and method in ['auto', 'manual']:
                sector = self._classify_manual(ticker)

            # Default to 'Unknown' if all methods fail
            if sector is None:
                sector = 'Unknown'
                logger.warning(f"Could not classify {ticker}, using 'Unknown'")

            sectors[ticker] = sector
            self._cache[ticker] = sector

        # Log sector distribution
        sector_counts = pd.Series(sectors).value_counts()
        logger.info(f"Sector distribution:\n{sector_counts}")

        return sectors

    def _classify_yfinance(self, ticker: str) -> Optional[str]:
        """
        Classify using yfinance API.

        Args:
            ticker: Ticker symbol

        Returns:
            Sector name or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Try 'sector' field first
            if 'sector' in info and info['sector']:
                sector = info['sector']
                # Normalize sector names
                sector = self._normalize_sector(sector)
                return sector

            # Try 'industry' field as fallback
            if 'industry' in info and info['industry']:
                industry = info['industry']
                sector = self._industry_to_sector(industry)
                if sector:
                    return sector

        except Exception as e:
            logger.debug(f"yfinance classification failed for {ticker}: {e}")

        return None

    def _classify_manual(self, ticker: str) -> Optional[str]:
        """
        Classify using manual mapping.

        Args:
            ticker: Ticker symbol

        Returns:
            Sector name or None if not in manual map
        """
        ticker_upper = ticker.upper()
        return self.MANUAL_SECTOR_MAP.get(ticker_upper)

    def _normalize_sector(self, sector: str) -> str:
        """
        Normalize sector name to GICS standard.

        Args:
            sector: Raw sector name

        Returns:
            Normalized GICS sector name
        """
        sector_lower = sector.lower()

        # Mapping from various names to GICS standard
        mappings = {
            'technology': 'Technology',
            'information technology': 'Technology',
            'tech': 'Technology',

            'financial': 'Financials',
            'financials': 'Financials',
            'financial services': 'Financials',

            'healthcare': 'Healthcare',
            'health care': 'Healthcare',
            'medical': 'Healthcare',

            'consumer cyclical': 'Consumer Discretionary',
            'consumer discretionary': 'Consumer Discretionary',

            'consumer defensive': 'Consumer Staples',
            'consumer staples': 'Consumer Staples',

            'energy': 'Energy',
            'oil & gas': 'Energy',

            'industrials': 'Industrials',
            'industrial': 'Industrials',

            'materials': 'Materials',
            'basic materials': 'Materials',

            'real estate': 'Real Estate',

            'utilities': 'Utilities',
            'utility': 'Utilities',

            'communication services': 'Communication Services',
            'telecommunications': 'Communication Services',
            'telecom': 'Communication Services',
        }

        for key, value in mappings.items():
            if key in sector_lower:
                return value

        # If no mapping found, return original (capitalized)
        return sector.title()

    def _industry_to_sector(self, industry: str) -> Optional[str]:
        """
        Map industry to sector.

        Args:
            industry: Industry name

        Returns:
            Sector name or None
        """
        industry_lower = industry.lower()

        # Simple keyword-based mapping
        if any(kw in industry_lower for kw in ['software', 'computer', 'semiconductor', 'electronic']):
            return 'Technology'
        elif any(kw in industry_lower for kw in ['bank', 'insurance', 'financial', 'credit']):
            return 'Financials'
        elif any(kw in industry_lower for kw in ['pharma', 'biotech', 'medical', 'health']):
            return 'Healthcare'
        elif any(kw in industry_lower for kw in ['oil', 'gas', 'energy', 'petroleum']):
            return 'Energy'
        elif any(kw in industry_lower for kw in ['retail', 'restaurant', 'hotel', 'automotive']):
            return 'Consumer Discretionary'
        elif any(kw in industry_lower for kw in ['food', 'beverage', 'tobacco', 'household']):
            return 'Consumer Staples'
        elif any(kw in industry_lower for kw in ['construction', 'aerospace', 'defense', 'machinery']):
            return 'Industrials'
        elif any(kw in industry_lower for kw in ['chemical', 'metal', 'mining', 'paper']):
            return 'Materials'
        elif any(kw in industry_lower for kw in ['utility', 'electric', 'water', 'gas distribution']):
            return 'Utilities'
        elif any(kw in industry_lower for kw in ['reit', 'real estate']):
            return 'Real Estate'
        elif any(kw in industry_lower for kw in ['media', 'entertainment', 'telecom', 'broadcasting']):
            return 'Communication Services'

        return None

    def group_by_sector(
        self,
        tickers: List[str],
        sectors: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[str]]:
        """
        Group tickers by sector.

        Args:
            tickers: List of tickers
            sectors: Pre-computed sector mapping (optional)

        Returns:
            Dictionary mapping sector -> list of tickers

        Example:
            >>> grouped = classifier.group_by_sector(['AAPL', 'MSFT', 'JPM'])
            >>> {'Technology': ['AAPL', 'MSFT'], 'Financials': ['JPM']}
        """
        if sectors is None:
            sectors = self.classify(tickers)

        grouped = defaultdict(list)
        for ticker, sector in sectors.items():
            grouped[sector].append(ticker)

        return dict(grouped)

    def get_sector_weights(
        self,
        tickers: List[str],
        weights: List[float],
        sectors: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Calculate portfolio weight by sector.

        Args:
            tickers: List of tickers
            weights: Position weights (must sum to 1.0)
            sectors: Pre-computed sector mapping (optional)

        Returns:
            Series with sector weights

        Example:
            >>> sector_weights = classifier.get_sector_weights(
            ...     ['AAPL', 'MSFT', 'JPM'],
            ...     [0.4, 0.4, 0.2]
            ... )
            >>> Technology    0.8
            >>> Financials    0.2
        """
        if sectors is None:
            sectors = self.classify(tickers)

        # Create DataFrame
        df = pd.DataFrame({
            'ticker': tickers,
            'weight': weights,
            'sector': [sectors[t] for t in tickers]
        })

        # Sum weights by sector
        sector_weights = df.groupby('sector')['weight'].sum()

        return sector_weights

    def check_sector_neutrality(
        self,
        long_tickers: List[str],
        short_tickers: List[str],
        long_weights: Optional[List[float]] = None,
        short_weights: Optional[List[float]] = None,
        max_sector_exposure: float = 0.10
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if long/short portfolio is sector-neutral.

        Args:
            long_tickers: Long position tickers
            short_tickers: Short position tickers
            long_weights: Long position weights (equal weight if None)
            short_weights: Short position weights (equal weight if None)
            max_sector_exposure: Maximum net sector exposure (e.g., 0.10 = 10%)

        Returns:
            Tuple of (is_neutral, net_exposures)
            net_exposures: Dictionary of sector -> net exposure

        Example:
            >>> is_neutral, exposures = classifier.check_sector_neutrality(
            ...     long_tickers=['AAPL', 'MSFT'],
            ...     short_tickers=['ORCL', 'CSCO'],
            ...     max_sector_exposure=0.10
            ... )
            >>> # Technology exposure should be near zero (long tech - short tech)
        """
        # Default to equal weights
        if long_weights is None:
            long_weights = [1.0 / len(long_tickers)] * len(long_tickers)
        if short_weights is None:
            short_weights = [1.0 / len(short_tickers)] * len(short_tickers)

        # Classify sectors
        all_tickers = long_tickers + short_tickers
        sectors = self.classify(all_tickers)

        # Calculate long sector exposures
        long_sector_weights = self.get_sector_weights(long_tickers, long_weights, sectors)

        # Calculate short sector exposures
        short_sector_weights = self.get_sector_weights(short_tickers, short_weights, sectors)

        # Calculate net exposures (long - short)
        all_sectors = set(long_sector_weights.index) | set(short_sector_weights.index)
        net_exposures = {}

        for sector in all_sectors:
            long_exp = long_sector_weights.get(sector, 0.0)
            short_exp = short_sector_weights.get(sector, 0.0)
            net_exposures[sector] = long_exp - short_exp

        # Check if all net exposures are within threshold
        max_abs_exposure = max(abs(exp) for exp in net_exposures.values())
        is_neutral = max_abs_exposure <= max_sector_exposure

        logger.info(f"Sector neutrality check: {is_neutral}")
        logger.info(f"Max absolute sector exposure: {max_abs_exposure:.2%}")
        logger.info(f"Net sector exposures:\n{pd.Series(net_exposures).sort_values()}")

        return is_neutral, net_exposures


def get_sector_etfs() -> Dict[str, str]:
    """
    Get sector ETF tickers for each GICS sector.

    Returns:
        Dictionary mapping sector -> ETF ticker

    Example:
        >>> sector_etfs = get_sector_etfs()
        >>> tech_etf = sector_etfs['Technology']  # 'XLK'
    """
    return {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }
