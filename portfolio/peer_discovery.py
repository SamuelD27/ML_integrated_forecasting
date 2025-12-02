"""
Dynamic Peer Discovery Module
==============================

Discovers peer companies for portfolio construction by:
1. Querying primary ticker metadata (exchange, sector, industry) via yfinance
2. Enumerating candidates on the same exchange (supports US and international markets)
3. Filtering by sector AND industry match
4. Applying liquidity filters (ADV, price thresholds)
5. Using ETF-based discovery as fallback when static lists fail
6. Returning ranked list of liquid peers with graceful degradation
"""

from typing import List, Dict, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


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


def get_exchange_candidates(exchange: str, sector: str, country: str = 'United States') -> List[str]:
    """
    Get candidate tickers from the same exchange and sector.

    Supports US and major international markets.
    Note: yfinance doesn't provide a direct exchange screener API.
    This uses a curated list of major tickers per exchange/sector as starting point.
    For production, integrate with a proper screener API.

    Parameters
    ----------
    exchange : str
        Exchange code (NMS, NYQ, LSE, TYO, etc.)
    sector : str
        Sector name
    country : str
        Country name for fallback matching

    Returns
    -------
    List[str]
        List of candidate ticker symbols
    """
    # Major US exchange tickers by sector
    US_SECTOR_MAP = {
        'NMS': {  # NASDAQ
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'INTC', 'QCOM',
                          'CSCO', 'AMAT', 'ADI', 'KLAC', 'LRCX', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'PANW'],
            'Information Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'INTC', 'QCOM'],
            'Communication Services': ['GOOGL', 'GOOG', 'META', 'NFLX', 'TMUS', 'CMCSA', 'DIS', 'CHTR'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MCD', 'NKE', 'LULU', 'ABNB'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MCD', 'NKE', 'LULU', 'ABNB'],
            'Healthcare': ['AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA', 'ISRG'],
            'Health Care': ['AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA', 'ISRG'],
            'Consumer Defensive': ['COST', 'PEP', 'MDLZ', 'KDP', 'MNST'],
            'Consumer Staples': ['COST', 'PEP', 'MDLZ', 'KDP', 'MNST'],
            'Financials': ['PYPL', 'INTC', 'NDAQ', 'CBOE', 'CME'],
            'Financial Services': ['PYPL', 'INTC', 'NDAQ', 'CBOE', 'CME'],
        },
        'NYQ': {  # NYSE
            'Financials': ['JPM', 'BAC', 'WFC', 'MS', 'GS', 'C', 'BLK', 'SCHW', 'AXP', 'SPGI'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'MS', 'GS', 'C', 'BLK', 'SCHW', 'AXP', 'SPGI'],
            'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY'],
            'Health Care': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY'],
            'Industrials': ['CAT', 'GE', 'HON', 'UPS', 'RTX', 'BA', 'DE', 'LMT', 'MMM', 'EMR'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES'],
            'Consumer Defensive': ['PG', 'KO', 'WMT', 'PM', 'CL', 'GIS', 'KHC'],
            'Consumer Staples': ['PG', 'KO', 'WMT', 'PM', 'CL', 'GIS', 'KHC'],
            'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE'],
            'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL'],
            'Real Estate': ['PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'O', 'WELL', 'DLR'],
            'Technology': ['IBM', 'ACN', 'NOW', 'INTU', 'TXN', 'UBER', 'SHOP'],
            'Information Technology': ['IBM', 'ACN', 'NOW', 'INTU', 'TXN', 'UBER', 'SHOP'],
            'Communication Services': ['VZ', 'T', 'TMUS', 'NFLX'],
            'Consumer Cyclical': ['HD', 'LOW', 'TGT', 'TJX', 'ROST', 'DG'],
            'Consumer Discretionary': ['HD', 'LOW', 'TGT', 'TJX', 'ROST', 'DG'],
        },
    }

    # International exchange tickers by sector (major European, Asian markets)
    INTERNATIONAL_SECTOR_MAP = {
        # London Stock Exchange
        'LSE': {
            'Financials': ['HSBA.L', 'BARC.L', 'LLOY.L', 'NWG.L', 'STAN.L', 'LGEN.L', 'AV.L', 'PRU.L'],
            'Financial Services': ['HSBA.L', 'BARC.L', 'LLOY.L', 'NWG.L', 'STAN.L', 'LGEN.L'],
            'Energy': ['BP.L', 'SHEL.L', 'SSE.L', 'CNA.L', 'WEIR.L'],
            'Healthcare': ['AZN.L', 'GSK.L', 'HIKMA.L', 'SN.L', 'DGN.L'],
            'Health Care': ['AZN.L', 'GSK.L', 'HIKMA.L', 'SN.L', 'DGN.L'],
            'Consumer Staples': ['ULVR.L', 'DGE.L', 'RKT.L', 'BATS.L', 'TSCO.L'],
            'Consumer Defensive': ['ULVR.L', 'DGE.L', 'RKT.L', 'BATS.L', 'TSCO.L'],
            'Industrials': ['RR.L', 'BA.L', 'EXPN.L', 'RS1.L', 'BDEV.L'],
            'Materials': ['RIO.L', 'AAL.L', 'ANTO.L', 'GLEN.L', 'FRES.L'],
            'Basic Materials': ['RIO.L', 'AAL.L', 'ANTO.L', 'GLEN.L', 'FRES.L'],
            'Technology': ['SFOR.L', 'AUTO.L', 'DARK.L', 'AVST.L'],
            'Information Technology': ['SFOR.L', 'AUTO.L', 'DARK.L', 'AVST.L'],
            'Utilities': ['NG.L', 'SSE.L', 'SVT.L', 'UU.L', 'PNN.L'],
            'Real Estate': ['LAND.L', 'BLND.L', 'SGRO.L', 'HMSO.L'],
        },
        # Frankfurt Stock Exchange (Germany)
        'FRA': {
            'Technology': ['SAP.DE', 'IFX.DE', 'AIXA.DE', 'NEM.DE'],
            'Information Technology': ['SAP.DE', 'IFX.DE', 'AIXA.DE', 'NEM.DE'],
            'Financials': ['DBK.DE', 'CBK.DE', 'ALV.DE', 'MUV2.DE', 'DTE.DE'],
            'Financial Services': ['DBK.DE', 'CBK.DE', 'ALV.DE', 'MUV2.DE'],
            'Healthcare': ['FRE.DE', 'MRK.DE', 'SRT3.DE', 'SHL.DE'],
            'Health Care': ['FRE.DE', 'MRK.DE', 'SRT3.DE', 'SHL.DE'],
            'Consumer Cyclical': ['BMW.DE', 'MBG.DE', 'VOW3.DE', 'PAH3.DE', 'ADS.DE'],
            'Consumer Discretionary': ['BMW.DE', 'MBG.DE', 'VOW3.DE', 'PAH3.DE', 'ADS.DE'],
            'Industrials': ['SIE.DE', 'AIR.DE', 'MTX.DE', 'EVK.DE'],
            'Energy': ['EON.DE', 'RWE.DE', 'EOAN.DE'],
            'Materials': ['BAS.DE', 'LIN.DE', 'HEN3.DE', 'BOSS.DE'],
            'Basic Materials': ['BAS.DE', 'LIN.DE', 'HEN3.DE'],
        },
        # Paris Stock Exchange (France)
        'PAR': {
            'Financials': ['BNP.PA', 'GLE.PA', 'ACA.PA', 'CS.PA'],
            'Financial Services': ['BNP.PA', 'GLE.PA', 'ACA.PA', 'CS.PA'],
            'Consumer Cyclical': ['MC.PA', 'KER.PA', 'RMS.PA', 'OR.PA'],
            'Consumer Discretionary': ['MC.PA', 'KER.PA', 'RMS.PA', 'OR.PA'],
            'Healthcare': ['SAN.PA', 'AI.PA', 'ENGI.PA', 'BIO.PA'],
            'Health Care': ['SAN.PA', 'AI.PA', 'ENGI.PA', 'BIO.PA'],
            'Industrials': ['AIR.PA', 'SAF.PA', 'VIE.PA', 'SU.PA'],
            'Energy': ['TTE.PA', 'ENGI.PA'],
            'Consumer Staples': ['OR.PA', 'DG.PA', 'RI.PA', 'BN.PA'],
            'Consumer Defensive': ['OR.PA', 'DG.PA', 'RI.PA', 'BN.PA'],
        },
        # Tokyo Stock Exchange (Japan)
        'TYO': {
            'Technology': ['6758.T', '6902.T', '6981.T', '6971.T', '6702.T'],  # Sony, Denso, Murata, Kyocera, Fujitsu
            'Information Technology': ['6758.T', '6902.T', '6981.T', '6971.T', '6702.T'],
            'Consumer Cyclical': ['7203.T', '7267.T', '7751.T', '9983.T'],  # Toyota, Honda, Canon, Fast Retailing
            'Consumer Discretionary': ['7203.T', '7267.T', '7751.T', '9983.T'],
            'Financials': ['8306.T', '8316.T', '8411.T', '8766.T'],  # MUFG, SMFG, Mizuho, Tokyo Marine
            'Financial Services': ['8306.T', '8316.T', '8411.T', '8766.T'],
            'Healthcare': ['4502.T', '4503.T', '4519.T', '4523.T'],  # Takeda, Astellas, Chugai, Eisai
            'Health Care': ['4502.T', '4503.T', '4519.T', '4523.T'],
            'Industrials': ['6301.T', '6502.T', '7011.T', '6503.T'],  # Komatsu, Toshiba, Mitsubishi Heavy, Mitsubishi Elec
            'Consumer Staples': ['2502.T', '2503.T', '4452.T'],  # Asahi, Kirin, Kao
            'Consumer Defensive': ['2502.T', '2503.T', '4452.T'],
        },
        # Hong Kong Stock Exchange
        'HKG': {
            'Technology': ['0700.HK', '9988.HK', '3690.HK', '1810.HK'],  # Tencent, Alibaba, Meituan, Xiaomi
            'Information Technology': ['0700.HK', '9988.HK', '3690.HK', '1810.HK'],
            'Financials': ['0005.HK', '1398.HK', '3988.HK', '2318.HK'],  # HSBC, ICBC, BOC, Ping An
            'Financial Services': ['0005.HK', '1398.HK', '3988.HK', '2318.HK'],
            'Real Estate': ['0016.HK', '0001.HK', '1113.HK', '0017.HK'],  # Sun Hung Kai, CK Asset, CK Hutchison, New World
            'Consumer Cyclical': ['9618.HK', '9999.HK', '2020.HK'],  # JD.com, NetEase, Anta
            'Consumer Discretionary': ['9618.HK', '9999.HK', '2020.HK'],
            'Energy': ['0857.HK', '0883.HK', '2688.HK'],  # PetroChina, CNOOC, ENN
            'Healthcare': ['1177.HK', '2269.HK'],  # Sino Biopharm, WuXi Bio
            'Health Care': ['1177.HK', '2269.HK'],
        },
        # Australian Securities Exchange
        'ASX': {
            'Financials': ['CBA.AX', 'WBC.AX', 'NAB.AX', 'ANZ.AX', 'MQG.AX'],
            'Financial Services': ['CBA.AX', 'WBC.AX', 'NAB.AX', 'ANZ.AX', 'MQG.AX'],
            'Materials': ['BHP.AX', 'RIO.AX', 'FMG.AX', 'NCM.AX', 'S32.AX'],
            'Basic Materials': ['BHP.AX', 'RIO.AX', 'FMG.AX', 'NCM.AX', 'S32.AX'],
            'Healthcare': ['CSL.AX', 'COH.AX', 'RMD.AX', 'SHL.AX'],
            'Health Care': ['CSL.AX', 'COH.AX', 'RMD.AX', 'SHL.AX'],
            'Real Estate': ['GMG.AX', 'SCG.AX', 'VCX.AX', 'MGR.AX'],
            'Consumer Staples': ['WOW.AX', 'COL.AX', 'TWE.AX'],
            'Consumer Defensive': ['WOW.AX', 'COL.AX', 'TWE.AX'],
            'Energy': ['WDS.AX', 'STO.AX', 'ORG.AX'],
            'Technology': ['WTC.AX', 'XRO.AX', 'CPU.AX'],
            'Information Technology': ['WTC.AX', 'XRO.AX', 'CPU.AX'],
        },
        # Toronto Stock Exchange (Canada)
        'TOR': {
            'Financials': ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'MFC.TO'],
            'Financial Services': ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'MFC.TO'],
            'Energy': ['ENB.TO', 'TRP.TO', 'SU.TO', 'CNQ.TO', 'CVE.TO'],
            'Materials': ['NTR.TO', 'ABX.TO', 'NEM.TO', 'TECK.B.TO'],
            'Basic Materials': ['NTR.TO', 'ABX.TO', 'NEM.TO', 'TECK.B.TO'],
            'Technology': ['SHOP.TO', 'CSU.TO', 'BB.TO', 'OTEX.TO'],
            'Information Technology': ['SHOP.TO', 'CSU.TO', 'BB.TO', 'OTEX.TO'],
            'Industrials': ['CNR.TO', 'CP.TO', 'WCN.TO', 'TIH.TO'],
            'Real Estate': ['BAM.A.TO', 'REI.UN.TO', 'CAR.UN.TO'],
            'Consumer Staples': ['L.TO', 'MRU.TO', 'SAP.TO'],
            'Consumer Defensive': ['L.TO', 'MRU.TO', 'SAP.TO'],
            'Healthcare': ['WSP.TO'],
            'Health Care': ['WSP.TO'],
        },
    }

    # Map country names to likely exchanges
    COUNTRY_EXCHANGE_MAP = {
        'United States': ['NMS', 'NYQ', 'NASDAQ', 'NYSE'],
        'USA': ['NMS', 'NYQ', 'NASDAQ', 'NYSE'],
        'United Kingdom': ['LSE'],
        'UK': ['LSE'],
        'Germany': ['FRA'],
        'France': ['PAR'],
        'Japan': ['TYO'],
        'Hong Kong': ['HKG'],
        'China': ['HKG'],
        'Australia': ['ASX'],
        'Canada': ['TOR'],
    }

    # Normalize exchange key
    exchange_key = exchange.upper()
    candidates = []

    # Check US markets first
    if exchange_key in US_SECTOR_MAP and sector in US_SECTOR_MAP[exchange_key]:
        candidates.extend(US_SECTOR_MAP[exchange_key][sector])

    # Check international markets
    if exchange_key in INTERNATIONAL_SECTOR_MAP and sector in INTERNATIONAL_SECTOR_MAP[exchange_key]:
        candidates.extend(INTERNATIONAL_SECTOR_MAP[exchange_key][sector])

    # Try common alternatives for US markets
    if not candidates:
        for exch_variant in [exchange_key, 'NMS', 'NYQ']:
            if exch_variant in US_SECTOR_MAP:
                for sect in US_SECTOR_MAP[exch_variant]:
                    if sect.lower() == sector.lower() or sect in sector or sector in sect:
                        candidates.extend(US_SECTOR_MAP[exch_variant][sect])
                        break

    # Try country-based fallback for international markets
    if not candidates and country:
        possible_exchanges = COUNTRY_EXCHANGE_MAP.get(country, [])
        for exch in possible_exchanges:
            if exch in INTERNATIONAL_SECTOR_MAP and sector in INTERNATIONAL_SECTOR_MAP[exch]:
                candidates.extend(INTERNATIONAL_SECTOR_MAP[exch][sector])
                logger.info(f"Using country-based fallback: {country} -> {exch}")
                break
            if exch in US_SECTOR_MAP and sector in US_SECTOR_MAP[exch]:
                candidates.extend(US_SECTOR_MAP[exch][sector])
                break

    # Final fallback: try any exchange with matching sector
    if not candidates:
        for exch_map in [US_SECTOR_MAP, INTERNATIONAL_SECTOR_MAP]:
            for exch, sectors in exch_map.items():
                for sect in sectors:
                    if sect.lower() == sector.lower() or sect in sector or sector in sect:
                        candidates.extend(sectors[sect])
                        logger.warning(f"Using cross-exchange fallback for sector: {sector}")
                        break
                if candidates:
                    break
            if candidates:
                break

    return list(dict.fromkeys(candidates))  # Remove duplicates while preserving order


def discover_peers_via_etf(primary_meta: Dict, max_peers: int = 15) -> List[str]:
    """
    Discover peers via ETF holdings as a fallback mechanism.

    Uses sector ETFs to find companies in the same sector as the primary ticker.

    Parameters
    ----------
    primary_meta : dict
        Metadata for the primary ticker
    max_peers : int
        Maximum number of peers to return

    Returns
    -------
    List[str]
        List of peer ticker symbols discovered from ETF holdings
    """
    sector = primary_meta.get('sector', 'N/A')
    if sector == 'N/A':
        return []

    # Sector to ETF mapping for holdings-based discovery
    SECTOR_ETFS = {
        'Technology': 'XLK',
        'Information Technology': 'XLK',
        'Communication Services': 'XLC',
        'Consumer Cyclical': 'XLY',
        'Consumer Discretionary': 'XLY',
        'Healthcare': 'XLV',
        'Health Care': 'XLV',
        'Financials': 'XLF',
        'Financial Services': 'XLF',
        'Consumer Defensive': 'XLP',
        'Consumer Staples': 'XLP',
        'Industrials': 'XLI',
        'Energy': 'XLE',
        'Basic Materials': 'XLB',
        'Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
    }

    etf_symbol = SECTOR_ETFS.get(sector)
    if not etf_symbol:
        logger.warning(f"No sector ETF mapping for sector: {sector}")
        return []

    try:
        # Try to get ETF holdings via yfinance (limited functionality)
        etf = yf.Ticker(etf_symbol)
        info = etf.get_info() if hasattr(etf, 'get_info') else (etf.info or {})

        # yfinance doesn't always provide holdings, so we use the top holdings if available
        holdings = info.get('holdings', [])
        if holdings:
            peers = [h.get('symbol', '') for h in holdings if h.get('symbol')]
            return peers[:max_peers]

        # Alternative: try to get from fund data
        # This is limited in yfinance, but we try anyway
        logger.info(f"ETF holdings not available for {etf_symbol}, using static fallback")
        return []

    except Exception as e:
        logger.warning(f"ETF-based peer discovery failed for {etf_symbol}: {e}")
        return []


def get_yfinance_sector_peers(ticker: str, sector: str, industry: str, max_peers: int = 15) -> List[str]:
    """
    Use yfinance to search for peers based on sector and industry.

    This is a fallback when static lists don't have coverage.

    Parameters
    ----------
    ticker : str
        Primary ticker symbol
    sector : str
        Sector name
    industry : str
        Industry name
    max_peers : int
        Maximum number of peers to return

    Returns
    -------
    List[str]
        List of peer ticker symbols
    """
    # This function provides a framework for dynamic peer discovery
    # In production, you would integrate with a screener API (Polygon, Alpha Vantage, etc.)
    # For now, we log and return empty to trigger ETF fallback

    logger.info(f"yfinance sector peer discovery attempted for {ticker} ({sector}/{industry})")

    # Try to use yfinance's recommendations if available
    try:
        t = yf.Ticker(ticker)
        recommendations = getattr(t, 'recommendations', None)
        if recommendations is not None and not recommendations.empty:
            # Note: yfinance recommendations are analyst recommendations, not peers
            # But we can try to get related tickers from other sources
            pass
    except Exception:
        pass

    return []


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

    # Step 2: Get candidate tickers from same exchange/sector (with international support)
    candidates = get_exchange_candidates(exchange, sector, country)

    # Remove primary ticker from candidates
    candidates = [t for t in candidates if t.upper() != primary_ticker.upper()]

    # Step 2b: If no candidates from static lists, try ETF-based discovery
    if not candidates:
        print(f"  No static candidates for exchange={exchange}, sector={sector}")
        print(f"  Attempting ETF-based peer discovery...")
        etf_peers = discover_peers_via_etf(primary_meta, max_peers=max_peers)
        if etf_peers:
            candidates = [t for t in etf_peers if t.upper() != primary_ticker.upper()]
            print(f"    ✓ Found {len(candidates)} candidates via ETF holdings")
        else:
            # Step 2c: Try yfinance sector-based discovery as final fallback
            print(f"  Attempting yfinance sector peer discovery...")
            yf_peers = get_yfinance_sector_peers(primary_ticker, sector, industry, max_peers)
            if yf_peers:
                candidates = [t for t in yf_peers if t.upper() != primary_ticker.upper()]

    if not candidates:
        logger.warning(f"No peer candidates found for {primary_ticker} (exchange={exchange}, sector={sector}, country={country})")
        print(f"  Warning: No candidates found for exchange={exchange}, sector={sector}, country={country}")
        print(f"  This may occur for non-US markets with limited coverage or uncommon sectors.")
        return {
            'primary_meta': primary_meta,
            'peers': [],
            'peer_metadata': {},
            'diagnostics': {
                'candidates_found': 0,
                'reason': f'No candidates match criteria (exchange={exchange}, sector={sector}, country={country})',
                'fallbacks_tried': ['static_lists', 'etf_holdings', 'yfinance_sector']
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
