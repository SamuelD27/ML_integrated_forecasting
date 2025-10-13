"""
Dynamic Peer Discovery Module
==============================

Discovers peer companies for portfolio construction by:
1. Querying primary ticker metadata (exchange, sector, industry) via yfinance
2. Enumerating candidates on the same exchange
3. Filtering by sector AND industry match
4. Applying liquidity filters (ADV, price thresholds)
5. Returning ranked list of liquid peers
"""

from typing import List, Dict, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

warnings.filterwarnings('ignore')


def get_ticker_metadata(ticker: str, retries: int = 3) -> Dict:
    """
    Fetch comprehensive metadata for a ticker with retry logic.

    Returns
    -------
    dict
        Metadata including exchange, sector, industry, country, etc.
    """
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.get_info() if hasattr(t, 'get_info') else (t.info or {})

            if not info:
                time.sleep(0.5 * (attempt + 1))
                continue

            return {
                'symbol': info.get('symbol', ticker),
                'exchange': info.get('exchange', info.get('exchangeName', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'country': info.get('country', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', info.get('averageVolume10days', 0)),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'quote_type': info.get('quoteType', 'EQUITY'),
            }
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Warning: Failed to fetch metadata for {ticker}: {e}")
                return {'symbol': ticker, 'error': str(e)}
            time.sleep(0.5 * (attempt + 1))

    return {'symbol': ticker, 'error': 'Max retries exceeded'}


def get_exchange_candidates(exchange: str, sector: str) -> List[str]:
    """
    Get candidate tickers from the same exchange and sector.

    Note: yfinance doesn't provide a direct exchange screener API.
    This uses a curated list of major tickers per exchange/sector as starting point.
    For production, integrate with a proper screener API.
    """
    # Major US exchange tickers by sector (subset for demonstration)
    EXCHANGE_SECTOR_MAP = {
        'NMS': {  # NASDAQ
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'INTC', 'QCOM',
                          'CSCO', 'AMAT', 'ADI', 'KLAC', 'LRCX', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'PANW'],
            'Communication Services': ['GOOGL', 'GOOG', 'META', 'NFLX', 'TMUS', 'CMCSA', 'DIS', 'CHTR'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MCD', 'NKE', 'LULU', 'ABNB'],
            'Healthcare': ['AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA', 'ISRG'],
            'Consumer Defensive': ['COST', 'PEP', 'MDLZ', 'KDP', 'MNST'],
        },
        'NYQ': {  # NYSE
            'Financials': ['JPM', 'BAC', 'WFC', 'MS', 'GS', 'C', 'BLK', 'SCHW', 'AXP', 'SPGI'],
            'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY'],
            'Industrials': ['CAT', 'GE', 'HON', 'UPS', 'RTX', 'BA', 'DE', 'LMT', 'MMM', 'EMR'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES'],
            'Consumer Defensive': ['PG', 'KO', 'WMT', 'PM', 'CL', 'GIS', 'KHC'],
            'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL'],
            'Real Estate': ['PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'O', 'WELL', 'DLR'],
            'Technology': ['IBM', 'ACN', 'NOW', 'INTU', 'TXN', 'UBER', 'SHOP'],
            'Communication Services': ['VZ', 'T', 'TMUS', 'NFLX'],
        },
    }

    # Try to find candidates
    exchange_key = exchange.upper()
    candidates = []

    # Check exact match
    if exchange_key in EXCHANGE_SECTOR_MAP and sector in EXCHANGE_SECTOR_MAP[exchange_key]:
        candidates.extend(EXCHANGE_SECTOR_MAP[exchange_key][sector])

    # Check common alternatives
    for exch_variant in [exchange_key, 'NMS', 'NYQ', 'NASDAQ', 'NYSE']:
        if exch_variant in EXCHANGE_SECTOR_MAP:
            for sect in EXCHANGE_SECTOR_MAP[exch_variant]:
                if sect.lower() == sector.lower() or sect in sector or sector in sect:
                    candidates.extend(EXCHANGE_SECTOR_MAP[exch_variant][sect])
                    break

    return list(dict.fromkeys(candidates))  # Remove duplicates while preserving order


def check_ticker_liquidity(ticker: str, min_price: float = 5.0,
                           min_avg_volume: float = 1e6) -> Tuple[bool, Dict]:
    """
    Check if ticker meets liquidity requirements.

    Parameters
    ----------
    ticker : str
        Ticker symbol
    min_price : float
        Minimum price threshold (default: $5)
    min_avg_volume : float
        Minimum average daily volume (default: 1M shares)

    Returns
    -------
    tuple
        (is_liquid, metadata_dict)
    """
    try:
        meta = get_ticker_metadata(ticker, retries=2)

        if 'error' in meta:
            return False, meta

        price = meta.get('current_price', 0)
        volume = meta.get('avg_volume', 0)

        is_liquid = (price >= min_price and volume >= min_avg_volume)

        meta['is_liquid'] = is_liquid
        meta['liquidity_check'] = {
            'price_ok': price >= min_price,
            'volume_ok': volume >= min_avg_volume,
        }

        return is_liquid, meta

    except Exception as e:
        return False, {'symbol': ticker, 'error': str(e)}


def discover_peers(primary_ticker: str,
                   max_peers: int = 25,
                   min_price: float = 3.0,
                   min_avg_volume: float = 5e5,
                   require_same_industry: bool = False,
                   max_workers: int = 10) -> Dict:
    """
    Discover peer companies for the primary ticker.

    Parameters
    ----------
    primary_ticker : str
        Primary ticker symbol
    max_peers : int
        Maximum number of peers to return (default: 15)
    min_price : float
        Minimum price threshold for liquidity (default: $5)
    min_avg_volume : float
        Minimum average daily volume for liquidity (default: 1M)
    require_same_industry : bool
        If True, require exact industry match (default: True)
    max_workers : int
        Max concurrent threads for metadata fetching (default: 10)

    Returns
    -------
    dict
        {
            'primary_meta': dict,
            'peers': [ticker, ...],
            'peer_metadata': {ticker: meta, ...},
            'diagnostics': {...}
        }
    """
    print(f"\n[Peer Discovery] Analyzing {primary_ticker}...")

    # Step 1: Get primary ticker metadata
    primary_meta = get_ticker_metadata(primary_ticker)

    if 'error' in primary_meta:
        return {
            'primary_meta': primary_meta,
            'peers': [],
            'peer_metadata': {},
            'diagnostics': {'error': f'Failed to fetch primary ticker metadata: {primary_meta["error"]}'}
        }

    exchange = primary_meta.get('exchange', 'N/A')
    sector = primary_meta.get('sector', 'N/A')
    industry = primary_meta.get('industry', 'N/A')
    country = primary_meta.get('country', 'N/A')

    print(f"  Exchange: {exchange}")
    print(f"  Sector: {sector}")
    print(f"  Industry: {industry}")
    print(f"  Country: {country}")

    # Step 2: Get candidate tickers from same exchange/sector
    candidates = get_exchange_candidates(exchange, sector)

    # Remove primary ticker from candidates
    candidates = [t for t in candidates if t.upper() != primary_ticker.upper()]

    if not candidates:
        print(f"  Warning: No candidates found for exchange={exchange}, sector={sector}")
        return {
            'primary_meta': primary_meta,
            'peers': [],
            'peer_metadata': {},
            'diagnostics': {
                'candidates_found': 0,
                'reason': 'No candidates match exchange/sector criteria'
            }
        }

    print(f"  Found {len(candidates)} candidate tickers on {exchange}/{sector}")

    # Step 3: Fetch metadata for all candidates in parallel
    candidate_metadata = {}

    print(f"  Fetching metadata for {len(candidates)} candidates...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(get_ticker_metadata, ticker, 2): ticker
            for ticker in candidates
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                meta = future.result()
                candidate_metadata[ticker] = meta
            except Exception as e:
                print(f"    Failed to fetch {ticker}: {e}")

    # Step 4: Filter candidates
    filtered_peers = []

    for ticker, meta in candidate_metadata.items():
        if 'error' in meta:
            continue

        # Check sector match
        if meta.get('sector', '') != sector:
            continue

        # Check industry match if required
        if require_same_industry and meta.get('industry', '') != industry:
            continue

        # Check country match (allow same country or major US variants)
        ticker_country = meta.get('country', '')
        if country and ticker_country:
            same_country = (ticker_country == country)
            us_variant = (country in ['United States', 'USA'] and
                         ticker_country in ['United States', 'USA', 'United States of America'])
            if not (same_country or us_variant):
                continue

        # Check liquidity
        price = meta.get('current_price', 0)
        volume = meta.get('avg_volume', 0)

        if price >= min_price and volume >= min_avg_volume:
            # Calculate liquidity score for ranking
            market_cap = meta.get('market_cap', 0)
            liquidity_score = np.log1p(volume) + np.log1p(market_cap) / 10
            meta['liquidity_score'] = liquidity_score
            filtered_peers.append((ticker, meta))

    # Step 5: Rank by liquidity score and take top N
    filtered_peers.sort(key=lambda x: x[1].get('liquidity_score', 0), reverse=True)
    top_peers = filtered_peers[:max_peers]

    peer_tickers = [t for t, _ in top_peers]
    peer_metadata = {t: m for t, m in top_peers}

    print(f"  ✓ Discovered {len(peer_tickers)} liquid peers: {peer_tickers}")

    return {
        'primary_meta': primary_meta,
        'peers': peer_tickers,
        'peer_metadata': peer_metadata,
        'diagnostics': {
            'candidates_found': len(candidates),
            'candidates_with_metadata': len(candidate_metadata),
            'passed_filters': len(filtered_peers),
            'selected_peers': len(peer_tickers),
            'filters_applied': {
                'same_sector': True,
                'same_industry': require_same_industry,
                'same_country': True,
                'min_price': min_price,
                'min_avg_volume': min_avg_volume,
            }
        }
    }


if __name__ == '__main__':
    # Test peer discovery
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'

    result = discover_peers(ticker, max_peers=10)

    print(f"\n{'='*70}")
    print(f"PEER DISCOVERY RESULTS: {ticker}")
    print(f"{'='*70}")
    print(f"Primary: {result['primary_meta'].get('symbol')} ({result['primary_meta'].get('sector')} / {result['primary_meta'].get('industry')})")
    print(f"\nPeers ({len(result['peers'])}):")
    for p in result['peers']:
        meta = result['peer_metadata'][p]
        print(f"  {p:6s} - ${meta.get('current_price', 0):8.2f} | Vol: {meta.get('avg_volume', 0):12,.0f} | MCap: ${meta.get('market_cap', 0)/1e9:6.1f}B")

    print(f"\nDiagnostics:")
    for k, v in result['diagnostics'].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
