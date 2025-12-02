import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import time
import json
import hashlib
import warnings
import logging
import os

from .interface import DerivativesProvider, ProviderType, InstrumentType
from .yahoo_provider import YahooFinanceProvider

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataProviderManager:
    """
    Manages multiple derivatives data providers with intelligent fallback.
    """

    def __init__(self, cache_dir: str = "data/cache", cache_ttl_hours: int = 24):
        """
        Initialize provider manager.

        Parameters
        ----------
        cache_dir : str
            Directory for Parquet cache
        cache_ttl_hours : int
            Cache time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.providers: List[DerivativesProvider] = []
        self.provider_stats: Dict[str, Dict] = {}
        self.provider_health: Dict[str, Dict] = {}

        # Health check settings
        self.max_consecutive_failures = 3  # Mark provider unhealthy after N failures
        self.health_check_interval = timedelta(minutes=5)

        # Initialize default providers
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers in priority order."""
        # Priority order: Yahoo (free) -> Polygon -> Alpha Vantage
        # Yahoo is primary since it's free and reliable for most use cases
        # Fallback providers require API keys

        # 1. Yahoo Finance (always available, free)
        yahoo = YahooFinanceProvider(cache_dir=str(self.cache_dir))
        if yahoo.is_available():
            self._add_provider(yahoo)
            logger.info(f"Initialized provider: {yahoo.get_provider_name()}")
            print(f"  ✓ Initialized provider: {yahoo.get_provider_name()}")

        # 2. Polygon (requires API key - first fallback)
        polygon_api_key = os.environ.get('POLYGON_API_KEY')
        if polygon_api_key:
            try:
                from .polygon_provider import PolygonProvider
                polygon = PolygonProvider(api_key=polygon_api_key, cache_dir=str(self.cache_dir))
                self._add_provider(polygon)
                logger.info(f"Initialized provider: Polygon")
                print(f"  ✓ Initialized provider: Polygon (fallback)")
            except Exception as e:
                logger.warning(f"Failed to initialize Polygon provider: {e}")
                print(f"  ✗ Polygon provider unavailable: {e}")

        # 3. Alpha Vantage (requires API key - second fallback)
        av_api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
        if av_api_key:
            try:
                from .alphavantage_provider import AlphaVantageProvider
                alpha_vantage = AlphaVantageProvider(api_key=av_api_key, cache_dir=str(self.cache_dir))
                self._add_provider(alpha_vantage)
                logger.info(f"Initialized provider: AlphaVantage")
                print(f"  ✓ Initialized provider: Alpha Vantage (fallback)")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpha Vantage provider: {e}")
                print(f"  ✗ Alpha Vantage provider unavailable: {e}")

        if not self.providers:
            raise RuntimeError("No data providers available. Ensure at least Yahoo Finance is working.")

        print(f"  Total providers configured: {len(self.providers)}")

    def _add_provider(self, provider: DerivativesProvider):
        """Add a provider and initialize its stats."""
        self.providers.append(provider)
        provider_name = provider.get_provider_name()
        self.provider_stats[provider_name] = {
            'requests': 0,
            'success': 0,
            'failures': 0,
            'cache_hits': 0,
            'last_success': None,
            'last_failure': None,
            'consecutive_failures': 0,
        }
        self.provider_health[provider_name] = {
            'healthy': True,
            'last_check': datetime.now(),
            'error_count': 0,
        }

    def get_option_chain(self, symbol: str, expiration: Optional[str] = None,
                         instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION,
                         use_cache: bool = True,
                         max_retries: int = 3) -> Tuple[pd.DataFrame, str]:
        """
        Fetch options chain with fallback and caching.

        Parameters
        ----------
        symbol : str
            Underlying symbol
        expiration : str, optional
            Specific expiration date
        instrument_type : InstrumentType
            Type of option
        use_cache : bool
            Whether to use cached data
        max_retries : int
            Max retry attempts per provider

        Returns
        -------
        tuple
            (DataFrame, provider_name)
        """
        # Try cache first
        if use_cache:
            cached = self._get_from_cache(symbol, 'option_chain', expiration=expiration,
                                          instrument_type=instrument_type)
            if cached is not None:
                # Update stats
                for provider_name in self.provider_stats:
                    self.provider_stats[provider_name]['cache_hits'] += 1
                return cached, 'cache'

        # Try providers in priority order
        for provider in self.providers:
            provider_name = provider.get_provider_name()

            for attempt in range(max_retries):
                try:
                    self.provider_stats[provider_name]['requests'] += 1

                    df = provider.get_option_chain(symbol, expiration, instrument_type)

                    if not df.empty:
                        self.provider_stats[provider_name]['success'] += 1

                        # Cache result
                        if use_cache:
                            self._save_to_cache(df, symbol, 'option_chain',
                                              expiration=expiration,
                                              instrument_type=instrument_type)

                        return df, provider_name

                    # Empty result - try next provider
                    break

                except Exception as e:
                    self.provider_stats[provider_name]['failures'] += 1

                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        print(f"  Retry {attempt+1}/{max_retries} for {symbol} on {provider_name} after {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        print(f"  Failed to fetch {symbol} from {provider_name}: {e}")
                        break

        # All providers failed
        return pd.DataFrame(), 'none'

    def get_option_chain_merged(self, symbol: str,
                                expirations: Optional[List[str]] = None,
                                instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
                                ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Fetch option chains from multiple providers and merge.

        Parameters
        ----------
        symbol : str
            Underlying symbol
        expirations : list, optional
            List of expiration dates
        instrument_type : InstrumentType
            Type of option

        Returns
        -------
        tuple
            (merged_df, expiration_to_provider_map)
        """
        all_chains = []
        provider_map = {}

        if expirations is None:
            # Get available expirations from first provider
            expirations = self.get_expirations(symbol, instrument_type)

        for expiration in expirations:
            df, provider = self.get_option_chain(symbol, expiration, instrument_type)

            if not df.empty:
                all_chains.append(df)
                provider_map[expiration] = provider

        if all_chains:
            merged = pd.concat(all_chains, ignore_index=True)
            # Remove duplicates, keeping best quote
            merged = self._deduplicate_quotes(merged)
            return merged, provider_map

        return pd.DataFrame(), {}

    def get_expirations(self, symbol: str,
                       instrument_type: InstrumentType = InstrumentType.EQUITY_OPTION
                       ) -> List[str]:
        """Get available expirations from first available provider."""
        for provider in self.providers:
            try:
                expirations = provider.get_expirations(symbol, instrument_type)
                if expirations:
                    return expirations
            except Exception:
                continue

        return []

    def get_futures_chain(self, underlying: str, use_cache: bool = True
                         ) -> Tuple[pd.DataFrame, str]:
        """Fetch futures chain with fallback and caching."""
        # Try cache first
        if use_cache:
            cached = self._get_from_cache(underlying, 'futures_chain')
            if cached is not None:
                return cached, 'cache'

        # Try providers
        for provider in self.providers:
            try:
                df = provider.get_futures_chain(underlying)

                if not df.empty:
                    if use_cache:
                        self._save_to_cache(df, underlying, 'futures_chain')

                    return df, provider.get_provider_name()

            except Exception:
                continue

        return pd.DataFrame(), 'none'

    def get_provider_diagnostics(self) -> pd.DataFrame:
        """Get diagnostics for all providers."""
        diagnostics = []

        for provider_name, stats in self.provider_stats.items():
            success_rate = (stats['success'] / stats['requests'] * 100
                           if stats['requests'] > 0 else 0)

            diagnostics.append({
                'provider': provider_name,
                'requests': stats['requests'],
                'success': stats['success'],
                'failures': stats['failures'],
                'cache_hits': stats['cache_hits'],
                'success_rate': f"{success_rate:.1f}%",
            })

        return pd.DataFrame(diagnostics)

    def _get_from_cache(self, symbol: str, data_type: str, **kwargs) -> Optional[pd.DataFrame]:
        """Retrieve data from Parquet cache if valid."""
        try:
            cache_key = self._cache_key(symbol, data_type, **kwargs)
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            meta_file = self.cache_dir / f"{cache_key}.meta.json"

            if not cache_file.exists() or not meta_file.exists():
                return None

            # Check TTL
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            cache_time = datetime.fromisoformat(meta['timestamp'])
            if datetime.now() - cache_time > self.cache_ttl:
                # Cache expired
                cache_file.unlink()
                meta_file.unlink()
                return None

            # Load from cache
            df = pd.read_parquet(cache_file)
            return df

        except Exception:
            return None

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, data_type: str, **kwargs):
        """Save data to Parquet cache."""
        try:
            cache_key = self._cache_key(symbol, data_type, **kwargs)
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            meta_file = self.cache_dir / f"{cache_key}.meta.json"

            # Save data
            df.to_parquet(cache_file)

            # Save metadata
            meta = {
                'symbol': symbol,
                'data_type': data_type,
                'timestamp': datetime.now().isoformat(),
                'kwargs': {k: str(v) for k, v in kwargs.items()},
            }

            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)

        except Exception as e:
            print(f"  Warning: Failed to cache {symbol} {data_type}: {e}")

    def _cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Generate cache key."""
        key_parts = [symbol, data_type] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        key_str = "_".join(str(p) for p in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _deduplicate_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate option quotes, keeping best (highest volume/OI).
        """
        if df.empty:
            return df

        # Group by option identifier
        key_cols = ['underlying', 'strike', 'expiration', 'option_type']

        # Sort by volume desc, OI desc
        df = df.sort_values(['volume', 'open_interest'], ascending=False)

        # Keep first (best) of each group
        df = df.drop_duplicates(subset=key_cols, keep='first')

        return df

    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cache files."""
        cleared = 0

        for file in self.cache_dir.glob("*.parquet"):
            should_delete = False

            if older_than_hours is None:
                should_delete = True
            else:
                meta_file = file.with_suffix('.meta.json')
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            meta = json.load(f)
                        cache_time = datetime.fromisoformat(meta['timestamp'])
                        age = datetime.now() - cache_time
                        if age > timedelta(hours=older_than_hours):
                            should_delete = True
                    except Exception:
                        should_delete = True

            if should_delete:
                file.unlink()
                meta_file = file.with_suffix('.meta.json')
                if meta_file.exists():
                    meta_file.unlink()
                cleared += 1

        print(f"Cleared {cleared} cache file(s)")
        return cleared

    def get_healthy_providers(self) -> List[DerivativesProvider]:
        """
        Get list of healthy providers for failover.

        Returns
        -------
        List[DerivativesProvider]
            Providers that are considered healthy
        """
        healthy = []
        for provider in self.providers:
            provider_name = provider.get_provider_name()
            health = self.provider_health.get(provider_name, {})

            # Skip if marked unhealthy
            if not health.get('healthy', True):
                # Check if enough time has passed to retry
                last_check = health.get('last_check', datetime.min)
                if datetime.now() - last_check < self.health_check_interval:
                    continue
                # Allow retry
                logger.info(f"Retrying unhealthy provider: {provider_name}")

            healthy.append(provider)

        return healthy

    def _mark_provider_failure(self, provider_name: str, error: str):
        """Mark a provider failure and update health status."""
        stats = self.provider_stats.get(provider_name, {})
        stats['failures'] = stats.get('failures', 0) + 1
        stats['last_failure'] = datetime.now()
        stats['consecutive_failures'] = stats.get('consecutive_failures', 0) + 1

        # Mark unhealthy if too many consecutive failures
        if stats['consecutive_failures'] >= self.max_consecutive_failures:
            self.provider_health[provider_name] = {
                'healthy': False,
                'last_check': datetime.now(),
                'error_count': stats.get('consecutive_failures', 0),
                'last_error': error,
            }
            logger.warning(f"Provider {provider_name} marked unhealthy after {stats['consecutive_failures']} consecutive failures")

    def _mark_provider_success(self, provider_name: str):
        """Mark a provider success and reset health status."""
        stats = self.provider_stats.get(provider_name, {})
        stats['success'] = stats.get('success', 0) + 1
        stats['last_success'] = datetime.now()
        stats['consecutive_failures'] = 0

        # Restore health
        self.provider_health[provider_name] = {
            'healthy': True,
            'last_check': datetime.now(),
            'error_count': 0,
        }

    def get_provider_health_dashboard(self) -> Dict:
        """
        Get comprehensive provider health dashboard.

        Returns
        -------
        Dict
            Health status for all providers with recommendations
        """
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'total_providers': len(self.providers),
            'healthy_providers': sum(1 for h in self.provider_health.values() if h.get('healthy', True)),
            'providers': {},
        }

        for provider in self.providers:
            provider_name = provider.get_provider_name()
            stats = self.provider_stats.get(provider_name, {})
            health = self.provider_health.get(provider_name, {})

            total_requests = stats.get('requests', 0)
            success_rate = (stats.get('success', 0) / total_requests * 100) if total_requests > 0 else 100

            dashboard['providers'][provider_name] = {
                'healthy': health.get('healthy', True),
                'requests': total_requests,
                'success': stats.get('success', 0),
                'failures': stats.get('failures', 0),
                'cache_hits': stats.get('cache_hits', 0),
                'success_rate': f"{success_rate:.1f}%",
                'consecutive_failures': stats.get('consecutive_failures', 0),
                'last_success': stats.get('last_success').isoformat() if stats.get('last_success') else None,
                'last_failure': stats.get('last_failure').isoformat() if stats.get('last_failure') else None,
                'last_error': health.get('last_error'),
            }

        # Add recommendations
        recommendations = []
        if dashboard['healthy_providers'] == 0:
            recommendations.append("CRITICAL: No healthy providers. Check API keys and network connectivity.")
        elif dashboard['healthy_providers'] < dashboard['total_providers']:
            unhealthy = [name for name, h in self.provider_health.items() if not h.get('healthy', True)]
            recommendations.append(f"WARNING: {len(unhealthy)} provider(s) unhealthy: {', '.join(unhealthy)}")

        if not os.environ.get('POLYGON_API_KEY'):
            recommendations.append("TIP: Set POLYGON_API_KEY for additional fallback coverage")
        if not os.environ.get('ALPHAVANTAGE_API_KEY'):
            recommendations.append("TIP: Set ALPHAVANTAGE_API_KEY for additional fallback coverage")

        dashboard['recommendations'] = recommendations

        return dashboard

    def fetch_with_fallback(self, ticker: str, start_date: str, end_date: str,
                           interval: str = 'daily') -> Tuple[pd.DataFrame, str]:
        """
        Fetch historical data with automatic provider fallback.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        interval : str
            Data interval: 'daily', 'weekly', 'monthly'

        Returns
        -------
        Tuple[pd.DataFrame, str]
            (data, provider_name) - data and which provider succeeded
        """
        healthy_providers = self.get_healthy_providers()

        if not healthy_providers:
            logger.error("No healthy providers available")
            return pd.DataFrame(), 'none'

        for provider in healthy_providers:
            provider_name = provider.get_provider_name()
            self.provider_stats[provider_name]['requests'] = \
                self.provider_stats[provider_name].get('requests', 0) + 1

            try:
                # Use provider's historical data method
                if hasattr(provider, 'get_historical_data'):
                    df = provider.get_historical_data(ticker, start_date, end_date, interval)
                else:
                    # Fallback for providers without get_historical_data
                    logger.warning(f"{provider_name} doesn't have get_historical_data method")
                    continue

                if not df.empty:
                    self._mark_provider_success(provider_name)
                    logger.info(f"Successfully fetched {ticker} from {provider_name}")
                    return df, provider_name

            except Exception as e:
                error_msg = str(e)
                self._mark_provider_failure(provider_name, error_msg)
                logger.warning(f"Failed to fetch {ticker} from {provider_name}: {error_msg}")
                continue

        return pd.DataFrame(), 'none'
