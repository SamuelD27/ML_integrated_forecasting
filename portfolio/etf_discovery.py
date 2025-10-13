"""
ETF Discovery Module
====================

Discovers relevant ETFs for portfolio construction:
1. Sector-specific ETF (based on primary ticker's sector)
2. Low-volatility ETF (defensive hedge)
3. Broad market ETF (benchmark/beta exposure)

All discovery is done via yfinance metadata validation.
"""

from typing import List, Dict, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# Comprehensive sector to ETF mapping (expanded)
SECTOR_ETF_MAP = {
    'Technology': ['XLK', 'VGT', 'IYW', 'QTEC', 'IGM', 'FTEC'],
    'Information Technology': ['XLK', 'VGT', 'IYW', 'QTEC', 'IGM', 'FTEC'],
    'Communication Services': ['XLC', 'VOX', 'IYZ', 'FCOM'],
    'Communications': ['XLC', 'VOX', 'IYZ', 'FCOM'],
    'Consumer Cyclical': ['XLY', 'VCR', 'IYC', 'FDIS'],
    'Consumer Discretionary': ['XLY', 'VCR', 'IYC', 'FDIS'],
    'Consumer Defensive': ['XLP', 'VDC', 'IYK', 'FSTA'],
    'Consumer Staples': ['XLP', 'VDC', 'IYK', 'FSTA'],
    'Healthcare': ['XLV', 'VHT', 'IYH', 'FHLC', 'IBB', 'XBI'],
    'Health Care': ['XLV', 'VHT', 'IYH', 'FHLC', 'IBB', 'XBI'],
    'Financial Services': ['XLF', 'VFH', 'IYF', 'FNCL', 'KBE', 'KRE'],
    'Financials': ['XLF', 'VFH', 'IYF', 'FNCL', 'KBE', 'KRE'],
    'Energy': ['XLE', 'VDE', 'IYE', 'FENY', 'XOP', 'OIH'],
    'Industrials': ['XLI', 'VIS', 'IYJ', 'FIDU', 'XTN'],
    'Basic Materials': ['XLB', 'VAW', 'IYM', 'MXI'],
    'Materials': ['XLB', 'VAW', 'IYM', 'MXI'],
    'Utilities': ['XLU', 'VPU', 'IDU', 'FUTY'],
    'Real Estate': ['XLRE', 'VNQ', 'IYR', 'USRT', 'REET'],
}

# Low volatility / defensive ETFs (expanded)
LOW_VOL_ETFS = ['USMV', 'SPLV', 'XMLV', 'EEMV', 'EFAV', 'ACWV', 'SPHD', 'FVD']

# Broad market ETFs by exchange/market (expanded)
MARKET_ETFS = {
    'US': ['SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'VTI', 'IWM', 'MDY', 'ITOT', 'SCHX'],
    'NASDAQ': ['QQQ', 'ONEQ', 'QQQM', 'QQQJ'],
    'NYSE': ['SPY', 'IVV', 'VOO', 'DIA', 'VTI'],
    'NMS': ['QQQ', 'ONEQ', 'QQQM', 'QQQJ'],  # NASDAQ alias
    'NYQ': ['SPY', 'IVV', 'VOO', 'DIA', 'VTI'],  # NYSE alias
    'International': ['VEA', 'EFA', 'VXUS', 'ACWI', 'IXUS', 'IEFA'],
}


def validate_etf(ticker: str, required_history_days: int = 252) -> Tuple[bool, Dict]:
    """
    Validate that an ETF exists and has sufficient trading history.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol
    required_history_days : int
        Minimum number of trading days required (default: 252 = 1 year)

    Returns
    -------
    tuple
        (is_valid, metadata_dict)
    """
    try:
        t = yf.Ticker(ticker)
        info = t.get_info() if hasattr(t, 'get_info') else (t.info or {})

        if not info or info.get('quoteType') not in ['ETF', 'MUTUALFUND']:
            return False, {
                'symbol': ticker,
                'valid': False,
                'reason': 'Not an ETF or insufficient metadata'
            }

        # Check historical data availability
        hist = t.history(period='1y')

        if hist.empty or len(hist) < required_history_days * 0.8:  # Allow 20% tolerance
            return False, {
                'symbol': ticker,
                'valid': False,
                'reason': f'Insufficient history: {len(hist)} days < {required_history_days}'
            }

        # Get current price for validation
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]

        meta = {
            'symbol': ticker,
            'valid': True,
            'name': info.get('longName', info.get('shortName', ticker)),
            'current_price': float(current_price),
            'avg_volume': info.get('averageVolume', info.get('averageVolume10days', 0)),
            'total_assets': info.get('totalAssets', 0),
            'expense_ratio': info.get('annualReportExpenseRatio', 0),
            'category': info.get('category', 'N/A'),
            'history_days': len(hist),
        }

        return True, meta

    except Exception as e:
        return False, {
            'symbol': ticker,
            'valid': False,
            'reason': f'Validation error: {str(e)}'
        }


def find_sector_etf(sector: str, max_etfs: int = 3) -> List[Dict]:
    """
    Find available sector ETFs for the given sector.

    Parameters
    ----------
    sector : str
        Sector name
    max_etfs : int
        Maximum number of ETFs to return (default: 3)

    Returns
    -------
    List[dict]
        List of ETF metadata
    """
    if not sector or sector == 'N/A':
        return []

    # Get candidate ETFs for this sector
    candidates = SECTOR_ETF_MAP.get(sector, [])

    if not candidates:
        # Try fuzzy matching
        for key, etfs in SECTOR_ETF_MAP.items():
            if sector.lower() in key.lower() or key.lower() in sector.lower():
                candidates = etfs
                break

    if not candidates:
        return []

    # Validate candidates and collect up to max_etfs
    valid_etfs = []
    for etf in candidates:
        if len(valid_etfs) >= max_etfs:
            break
        is_valid, meta = validate_etf(etf)
        if is_valid:
            meta['etf_type'] = 'sector'
            meta['mapped_sector'] = sector
            valid_etfs.append(meta)

    return valid_etfs


def find_low_vol_etf(max_etfs: int = 2) -> List[Dict]:
    """
    Find available low-volatility ETFs.

    Parameters
    ----------
    max_etfs : int
        Maximum number of ETFs to return (default: 2)

    Returns
    -------
    List[dict]
        List of ETF metadata
    """
    valid_etfs = []
    for etf in LOW_VOL_ETFS:
        if len(valid_etfs) >= max_etfs:
            break
        is_valid, meta = validate_etf(etf)
        if is_valid:
            meta['etf_type'] = 'low_volatility'
            valid_etfs.append(meta)

    return valid_etfs


def find_market_etf(exchange: str = 'US', country: str = 'United States') -> Optional[Dict]:
    """
    Find the best available market ETF based on exchange/country.

    Parameters
    ----------
    exchange : str
        Exchange code (NMS, NYQ, etc.)
    country : str
        Country name

    Returns
    -------
    dict or None
        ETF metadata if found, None otherwise
    """
    # Determine market category
    if country and country not in ['United States', 'USA']:
        candidates = MARKET_ETFS.get('International', [])
    elif 'NASDAQ' in exchange.upper() or exchange.upper() == 'NMS':
        candidates = MARKET_ETFS.get('NASDAQ', [])
    elif 'NYSE' in exchange.upper() or exchange.upper() == 'NYQ':
        candidates = MARKET_ETFS.get('NYSE', [])
    else:
        candidates = MARKET_ETFS.get('US', [])

    # Validate candidates
    for etf in candidates:
        is_valid, meta = validate_etf(etf)
        if is_valid:
            meta['etf_type'] = 'market'
            meta['mapped_exchange'] = exchange
            return meta

    # Fallback to any US market ETF
    for etf in MARKET_ETFS['US']:
        is_valid, meta = validate_etf(etf)
        if is_valid:
            meta['etf_type'] = 'market'
            meta['mapped_exchange'] = 'US_FALLBACK'
            return meta

    return None


def find_multiple_market_etfs(exchange: str = 'US', country: str = 'United States', max_etfs: int = 3) -> List[Dict]:
    """
    Find multiple market ETFs based on exchange/country.

    Parameters
    ----------
    exchange : str
        Exchange code (NMS, NYQ, etc.)
    country : str
        Country name
    max_etfs : int
        Maximum number of ETFs to return (default: 3)

    Returns
    -------
    List[dict]
        List of ETF metadata
    """
    # Determine market category
    if country and country not in ['United States', 'USA']:
        candidates = MARKET_ETFS.get('International', [])
    elif 'NASDAQ' in exchange.upper() or exchange.upper() == 'NMS':
        candidates = MARKET_ETFS.get('NASDAQ', [])
    elif 'NYSE' in exchange.upper() or exchange.upper() == 'NYQ':
        candidates = MARKET_ETFS.get('NYSE', [])
    else:
        candidates = MARKET_ETFS.get('US', [])

    # Validate candidates
    valid_etfs = []
    for etf in candidates:
        if len(valid_etfs) >= max_etfs:
            break
        is_valid, meta = validate_etf(etf)
        if is_valid:
            meta['etf_type'] = 'market'
            meta['mapped_exchange'] = exchange
            valid_etfs.append(meta)

    # If not enough, add from broad US market
    if len(valid_etfs) < max_etfs:
        for etf in MARKET_ETFS.get('US', []):
            if len(valid_etfs) >= max_etfs:
                break
            if etf not in [e['symbol'] for e in valid_etfs]:  # Avoid duplicates
                is_valid, meta = validate_etf(etf)
                if is_valid:
                    meta['etf_type'] = 'market'
                    meta['mapped_exchange'] = 'US_FALLBACK'
                    valid_etfs.append(meta)

    return valid_etfs


def discover_etfs(primary_ticker_meta: Dict) -> Dict:
    """
    Discover relevant ETFs for portfolio construction.

    Parameters
    ----------
    primary_ticker_meta : dict
        Metadata for the primary ticker (from peer_discovery)

    Returns
    -------
    dict
        {
            'sector_etf': dict or None,
            'low_vol_etf': dict or None,
            'market_etf': dict or None,
            'all_etfs': [ticker, ...],
            'etf_metadata': {ticker: meta, ...},
            'diagnostics': {...}
        }
    """
    print(f"\n[ETF Discovery] Finding ETFs for portfolio...")

    sector = primary_ticker_meta.get('sector', 'N/A')
    exchange = primary_ticker_meta.get('exchange', 'US')
    country = primary_ticker_meta.get('country', 'United States')

    # Find sector ETFs (multiple)
    print(f"  Searching for {sector} sector ETFs...")
    sector_etfs = find_sector_etf(sector, max_etfs=3)
    if sector_etfs:
        for etf in sector_etfs:
            print(f"    ✓ Found: {etf['symbol']} - {etf['name']}")
    else:
        print(f"    ✗ No sector ETFs found for {sector}")

    # Find low-vol ETFs (multiple)
    print(f"  Searching for low-volatility ETFs...")
    low_vol_etfs = find_low_vol_etf(max_etfs=2)
    if low_vol_etfs:
        for etf in low_vol_etfs:
            print(f"    ✓ Found: {etf['symbol']} - {etf['name']}")
    else:
        print(f"    ✗ No low-volatility ETFs found")

    # Find market ETFs (multiple)
    print(f"  Searching for market ETFs (exchange: {exchange})...")
    market_etfs = find_multiple_market_etfs(exchange, country, max_etfs=3)
    if market_etfs:
        for etf in market_etfs:
            print(f"    ✓ Found: {etf['symbol']} - {etf['name']}")
    else:
        print(f"    ✗ No market ETFs found")

    # Compile results
    all_etfs = []
    etf_metadata = {}

    # Add all discovered ETFs
    for etf_list in [sector_etfs, low_vol_etfs, market_etfs]:
        for etf_obj in etf_list:
            if etf_obj:
                ticker = etf_obj['symbol']
                if ticker not in all_etfs:  # Avoid duplicates
                    all_etfs.append(ticker)
                    etf_metadata[ticker] = etf_obj

    # Keep first ETF of each type for backward compatibility
    sector_etf = sector_etfs[0] if sector_etfs else None
    low_vol_etf = low_vol_etfs[0] if low_vol_etfs else None
    market_etf = market_etfs[0] if market_etfs else None

    print(f"  ✓ Discovered {len(all_etfs)} ETFs: {all_etfs}")

    return {
        'sector_etf': sector_etf,
        'low_vol_etf': low_vol_etf,
        'market_etf': market_etf,
        'all_etfs': all_etfs,
        'etf_metadata': etf_metadata,
        'diagnostics': {
            'sector': sector,
            'exchange': exchange,
            'country': country,
            'sector_etf_found': sector_etf is not None,
            'low_vol_etf_found': low_vol_etf is not None,
            'market_etf_found': market_etf is not None,
            'total_etfs': len(all_etfs),
        }
    }


if __name__ == '__main__':
    # Test ETF discovery
    import sys

    # Simulate primary ticker metadata
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'

    # Get real metadata
    from portfolio.peer_discovery import get_ticker_metadata

    primary_meta = get_ticker_metadata(test_ticker)

    result = discover_etfs(primary_meta)

    print(f"\n{'='*70}")
    print(f"ETF DISCOVERY RESULTS: {test_ticker}")
    print(f"{'='*70}")
    print(f"Primary Sector: {primary_meta.get('sector')}")
    print(f"\nDiscovered ETFs:")

    if result['sector_etf']:
        etf = result['sector_etf']
        print(f"\n  Sector ETF: {etf['symbol']}")
        print(f"    Name: {etf['name']}")
        print(f"    Price: ${etf['current_price']:.2f}")
        print(f"    Assets: ${etf.get('total_assets', 0)/1e9:.2f}B")

    if result['low_vol_etf']:
        etf = result['low_vol_etf']
        print(f"\n  Low-Vol ETF: {etf['symbol']}")
        print(f"    Name: {etf['name']}")
        print(f"    Price: ${etf['current_price']:.2f}")

    if result['market_etf']:
        etf = result['market_etf']
        print(f"\n  Market ETF: {etf['symbol']}")
        print(f"    Name: {etf['name']}")
        print(f"    Price: ${etf['current_price']:.2f}")

    print(f"\nDiagnostics:")
    for k, v in result['diagnostics'].items():
        print(f"  {k}: {v}")
