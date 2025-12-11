"""
Universe Screener
==================
Screens and filters stock universes for the trading bot.

Supports:
- S&P 500, S&P 100, Russell 1000/2000
- Custom symbol lists
- Liquidity and price filters
- Sector filters
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging
import json
import hashlib

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class UniverseType(Enum):
    """Predefined universe types."""
    SP500 = "sp500"
    SP100 = "sp100"
    NASDAQ100 = "nasdaq100"
    DOW30 = "dow30"
    RUSSELL1000 = "russell1000"
    RUSSELL2000 = "russell2000"
    CUSTOM = "custom"


@dataclass
class UniverseConfig:
    """
    Configuration for universe screening.

    Attributes:
        universe_type: Base universe to screen from
        custom_symbols: Additional symbols to include
        exclude_symbols: Symbols to exclude
        min_price: Minimum stock price
        max_price: Maximum stock price (None = no limit)
        min_avg_volume: Minimum average daily volume (shares)
        min_market_cap: Minimum market cap in USD
        max_market_cap: Maximum market cap (None = no limit)
        sectors: List of sectors to include (None = all)
        exclude_sectors: Sectors to exclude
        max_symbols: Maximum number of symbols to return
    """
    universe_type: UniverseType = UniverseType.SP500
    custom_symbols: List[str] = field(default_factory=list)
    exclude_symbols: List[str] = field(default_factory=list)
    min_price: float = 5.0
    max_price: Optional[float] = None
    min_avg_volume: int = 100_000  # 100k shares/day
    min_market_cap: float = 1_000_000_000  # $1B
    max_market_cap: Optional[float] = None
    sectors: Optional[List[str]] = None
    exclude_sectors: List[str] = field(default_factory=list)
    max_symbols: int = 500

    def to_dict(self) -> Dict:
        return {
            'universe_type': self.universe_type.value,
            'custom_symbols': self.custom_symbols,
            'exclude_symbols': self.exclude_symbols,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'min_avg_volume': self.min_avg_volume,
            'min_market_cap': self.min_market_cap,
            'max_market_cap': self.max_market_cap,
            'sectors': self.sectors,
            'exclude_sectors': self.exclude_sectors,
            'max_symbols': self.max_symbols,
        }


class UniverseScreener:
    """
    Screens and filters stock universes.

    Usage:
        screener = UniverseScreener()
        config = UniverseConfig(universe_type=UniverseType.SP500, min_market_cap=10e9)
        symbols = screener.get_universe(config)
    """

    # Cache settings
    CACHE_DIR = Path("data/cache/universe")
    CACHE_TTL_HOURS = 24

    # Sector ETF mappings for sector lookup
    SECTOR_ETFS = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC',
    }

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize universe screener.

        Args:
            cache_enabled: Whether to cache universe lists
        """
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_universe(self, config: UniverseConfig) -> List[str]:
        """
        Get filtered universe of symbols.

        Args:
            config: Universe screening configuration

        Returns:
            List of symbols meeting all criteria
        """
        logger.info(f"Getting universe: {config.universe_type.value}")

        # Get base universe
        base_symbols = self._get_base_universe(config.universe_type)

        # Add custom symbols
        all_symbols = list(set(base_symbols) | set(config.custom_symbols))

        # Remove excluded symbols
        all_symbols = [s for s in all_symbols if s not in config.exclude_symbols]

        logger.info(f"Base universe: {len(all_symbols)} symbols")

        # Apply filters (this is the expensive part - requires data fetching)
        filtered = self._apply_filters(all_symbols, config)

        logger.info(f"Filtered universe: {len(filtered)} symbols")

        # Limit to max_symbols
        if len(filtered) > config.max_symbols:
            filtered = filtered[:config.max_symbols]

        return filtered

    def _get_base_universe(self, universe_type: UniverseType) -> List[str]:
        """Get base symbol list for universe type."""
        # Check cache
        cache_key = f"base_{universe_type.value}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        symbols = []

        if universe_type == UniverseType.SP500:
            symbols = self._fetch_sp500()
        elif universe_type == UniverseType.SP100:
            symbols = self._fetch_sp100()
        elif universe_type == UniverseType.NASDAQ100:
            symbols = self._fetch_nasdaq100()
        elif universe_type == UniverseType.DOW30:
            symbols = self._fetch_dow30()
        elif universe_type == UniverseType.RUSSELL1000:
            symbols = self._fetch_russell1000()
        elif universe_type == UniverseType.RUSSELL2000:
            symbols = self._fetch_russell2000()
        elif universe_type == UniverseType.CUSTOM:
            symbols = []

        if symbols:
            self._save_cache(cache_key, symbols)

        return symbols

    def _fetch_sp500(self) -> List[str]:
        """Fetch S&P 500 constituents from Wikipedia."""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]

            # Column name varies: 'Symbol' or 'Ticker symbol'
            symbol_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker symbol'
            symbols = df[symbol_col].str.replace('.', '-', regex=False).tolist()

            logger.info(f"Fetched {len(symbols)} S&P 500 symbols")
            return symbols

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500: {e}")
            return self._get_fallback_sp500()

    def _fetch_sp100(self) -> List[str]:
        """Fetch S&P 100 constituents."""
        try:
            url = "https://en.wikipedia.org/wiki/S%26P_100"
            tables = pd.read_html(url)
            df = tables[2]  # The S&P 100 table

            symbol_col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
            symbols = df[symbol_col].str.replace('.', '-', regex=False).tolist()

            logger.info(f"Fetched {len(symbols)} S&P 100 symbols")
            return symbols

        except Exception as e:
            logger.error(f"Failed to fetch S&P 100: {e}")
            # Return known S&P 100 members
            return self._get_fallback_sp100()

    def _fetch_nasdaq100(self) -> List[str]:
        """Fetch NASDAQ 100 constituents."""
        try:
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url)

            # Find the table with ticker symbols
            for table in tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    symbols = table[col].str.replace('.', '-', regex=False).tolist()
                    logger.info(f"Fetched {len(symbols)} NASDAQ 100 symbols")
                    return symbols

            return self._get_fallback_nasdaq100()

        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ 100: {e}")
            return self._get_fallback_nasdaq100()

    def _fetch_dow30(self) -> List[str]:
        """Fetch Dow Jones 30 constituents."""
        try:
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            tables = pd.read_html(url)

            for table in tables:
                if 'Symbol' in table.columns:
                    symbols = table['Symbol'].str.replace('.', '-', regex=False).tolist()
                    if len(symbols) >= 30:
                        logger.info(f"Fetched {len(symbols)} Dow 30 symbols")
                        return symbols[:30]

            return self._get_fallback_dow30()

        except Exception as e:
            logger.error(f"Failed to fetch Dow 30: {e}")
            return self._get_fallback_dow30()

    def _fetch_russell1000(self) -> List[str]:
        """
        Fetch Russell 1000 constituents.
        Note: Full list not freely available, using approximation.
        """
        # Russell 1000 is roughly S&P 500 + next 500 large caps
        # For now, use S&P 500 as base
        sp500 = self._fetch_sp500()

        # Could add more symbols from other sources here
        logger.info(f"Russell 1000 approximation: {len(sp500)} symbols (using S&P 500)")
        return sp500

    def _fetch_russell2000(self) -> List[str]:
        """
        Fetch Russell 2000 constituents.
        Note: Full list not freely available.
        """
        # Russell 2000 is small caps - hard to get full list freely
        # Return empty for now - user should provide custom list or use ETF
        logger.warning("Russell 2000 full list not available. Consider using IWM holdings.")
        return []

    def _apply_filters(self, symbols: List[str], config: UniverseConfig) -> List[str]:
        """
        Apply filters to symbol list.

        This fetches basic data from yfinance to filter symbols.
        """
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(f"Applying filters to {len(symbols)} symbols...")

        # Batch fetch info using yfinance
        # This is more efficient than individual calls
        passed_symbols = []
        failed_symbols = []

        def check_symbol(symbol: str) -> Optional[Tuple[str, Dict]]:
            """Check if symbol passes filters."""
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                if not info or info.get('regularMarketPrice') is None:
                    return None

                # Extract data
                price = info.get('regularMarketPrice') or info.get('currentPrice', 0)
                volume = info.get('averageVolume', 0)
                market_cap = info.get('marketCap', 0)
                sector = info.get('sector', 'Unknown')

                # Apply filters
                if config.min_price and price < config.min_price:
                    return None
                if config.max_price and price > config.max_price:
                    return None
                if config.min_avg_volume and volume < config.min_avg_volume:
                    return None
                if config.min_market_cap and market_cap < config.min_market_cap:
                    return None
                if config.max_market_cap and market_cap > config.max_market_cap:
                    return None
                if config.sectors and sector not in config.sectors:
                    return None
                if sector in config.exclude_sectors:
                    return None

                return (symbol, {
                    'price': price,
                    'volume': volume,
                    'market_cap': market_cap,
                    'sector': sector,
                })

            except Exception as e:
                logger.debug(f"Failed to check {symbol}: {e}")
                return None

        # Use thread pool for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(check_symbol, sym): sym for sym in symbols}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    passed_symbols.append(result)

        # Sort by market cap (descending) for consistent ordering
        passed_symbols.sort(key=lambda x: x[1].get('market_cap', 0), reverse=True)

        return [sym for sym, _ in passed_symbols]

    def get_sector_symbols(self, sector: str) -> List[str]:
        """
        Get symbols for a specific sector from S&P 500.

        Args:
            sector: Sector name (e.g., 'Technology', 'Healthcare')

        Returns:
            List of symbols in that sector
        """
        config = UniverseConfig(
            universe_type=UniverseType.SP500,
            sectors=[sector],
        )
        return self.get_universe(config)

    def get_symbol_metadata(self, symbol: str) -> Optional[Dict]:
        """
        Get metadata for a single symbol.

        Returns:
            Dict with price, volume, market_cap, sector, etc.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            return {
                'symbol': symbol,
                'name': info.get('shortName') or info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'price': info.get('regularMarketPrice') or info.get('currentPrice'),
                'market_cap': info.get('marketCap'),
                'avg_volume': info.get('averageVolume'),
                'exchange': info.get('exchange'),
                'country': info.get('country'),
            }

        except Exception as e:
            logger.error(f"Failed to get metadata for {symbol}: {e}")
            return None

    def _get_cached(self, key: str) -> Optional[List[str]]:
        """Get cached universe list."""
        if not self.cache_enabled:
            return None

        cache_file = self.CACHE_DIR / f"{key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            # Check TTL
            cached_at = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
            if datetime.now() - cached_at > timedelta(hours=self.CACHE_TTL_HOURS):
                logger.info(f"Cache expired for {key}")
                return None

            return data.get('symbols', [])

        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {e}")
            return None

    def _save_cache(self, key: str, symbols: List[str]) -> None:
        """Save universe list to cache."""
        if not self.cache_enabled:
            return

        cache_file = self.CACHE_DIR / f"{key}.json"

        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            logger.warning(f"Failed to save cache for {key}: {e}")

    # Fallback lists (in case web scraping fails)
    def _get_fallback_sp500(self) -> List[str]:
        """Fallback S&P 500 - top 100 by market cap."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'XOM', 'V', 'JPM', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
            'DHR', 'NEE', 'WFC', 'LIN', 'ADBE', 'TXN', 'BMY', 'CRM', 'PM', 'UPS',
            'CMCSA', 'NKE', 'RTX', 'ORCL', 'HON', 'NFLX', 'QCOM', 'COP', 'LOW', 'T',
            'MS', 'UNP', 'BA', 'IBM', 'ELV', 'INTC', 'SPGI', 'CAT', 'INTU', 'GE',
            'DE', 'AMGN', 'AXP', 'AMD', 'SBUX', 'BLK', 'GILD', 'PLD', 'MDLZ', 'GS',
            'ISRG', 'ADI', 'SYK', 'BKNG', 'TJX', 'ADP', 'MMC', 'VRTX', 'REGN', 'CVS',
            'LRCX', 'C', 'SCHW', 'MO', 'ETN', 'ZTS', 'CB', 'BDX', 'PGR', 'NOW',
            'SO', 'CI', 'TMUS', 'DUK', 'AON', 'FI', 'BSX', 'EQIX', 'ITW', 'CME',
        ]

    def _get_fallback_sp100(self) -> List[str]:
        """Fallback S&P 100 list."""
        return [
            'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMGN', 'AMT', 'AMZN', 'AVGO',
            'AXP', 'BA', 'BAC', 'BIIB', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B', 'C',
            'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS',
            'CVX', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX', 'GD',
            'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC',
            'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD',
            'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE',
            'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM',
            'RTX', 'SBUX', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TSLA', 'TXN', 'UNH',
            'UNP', 'UPS', 'USB', 'V', 'VZ', 'WBA', 'WFC', 'WMT', 'XOM',
        ]

    def _get_fallback_nasdaq100(self) -> List[str]:
        """Fallback NASDAQ 100 list."""
        return [
            'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
            'AMZN', 'ANSS', 'ARM', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CDNS',
            'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX',
            'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG', 'FAST',
            'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC',
            'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR', 'MCHP',
            'MDB', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU', 'NFLX',
            'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP',
            'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'TEAM', 'TMUS',
            'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZS',
        ]

    def _get_fallback_dow30(self) -> List[str]:
        """Fallback Dow 30 list."""
        return [
            'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
            'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD',
            'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT',
        ]


if __name__ == "__main__":
    # Test the screener
    logging.basicConfig(level=logging.INFO)

    screener = UniverseScreener()

    # Test S&P 500
    config = UniverseConfig(
        universe_type=UniverseType.SP500,
        min_market_cap=50e9,  # $50B minimum
        min_avg_volume=1_000_000,  # 1M shares/day
        max_symbols=50,
    )

    symbols = screener.get_universe(config)
    print(f"\nTop 50 S&P 500 stocks (>$50B market cap):")
    print(symbols)
