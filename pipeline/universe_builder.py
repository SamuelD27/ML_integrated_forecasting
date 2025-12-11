"""
Universe Builder - Phase 1
===========================
Build the investable universe by computing fundamentals and factor scores
for each ticker, then filtering by upside and factor thresholds.

This is Phase 1 of the 6-phase trading pipeline:
1. Universe Builder (Fundamentals + Factors) <-- YOU ARE HERE
2. Regime Detection
3. Ensemble Forecaster
4. Meta-Labeling
5. Instrument Selection
6. CVaR Allocation

Usage:
    >>> from pipeline.universe_builder import build_universe
    >>> from utils.config import get_config
    >>>
    >>> config = get_config()
    >>> universe = build_universe(pd.Timestamp.now(), config.trading)
    >>> print(f"Universe contains {len(universe)} stocks")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.core_types import StockSnapshot
from utils.logger import get_logger, PipelineLogger

# Lazy imports to avoid circular dependencies
logger = PipelineLogger('pipeline.universe_builder')


# =============================================================================
# Universe Sources
# =============================================================================

def get_sp500_tickers() -> List[str]:
    """Get S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = tables[0]
        return sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 tickers: {e}")
        # Fallback to a small default list
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'JPM', 'V', 'JNJ']


def get_nasdaq100_tickers() -> List[str]:
    """Get NASDAQ 100 tickers."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        for table in tables:
            if 'Ticker' in table.columns:
                return table['Ticker'].tolist()
            elif 'Symbol' in table.columns:
                return table['Symbol'].tolist()
    except Exception as e:
        logger.warning(f"Failed to fetch NASDAQ 100 tickers: {e}")

    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX']


def get_dow30_tickers() -> List[str]:
    """Get Dow 30 tickers."""
    return [
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
    ]


def get_base_universe(source: str, custom_tickers: Optional[List[str]] = None) -> List[str]:
    """
    Get base universe tickers from specified source.

    Args:
        source: Universe source ('sp500', 'nasdaq100', 'dow30', 'custom')
        custom_tickers: List of tickers when source is 'custom'

    Returns:
        List of ticker symbols
    """
    source = source.lower()

    if source == 'sp500':
        return get_sp500_tickers()
    elif source == 'nasdaq100':
        return get_nasdaq100_tickers()
    elif source == 'dow30':
        return get_dow30_tickers()
    elif source == 'custom':
        return custom_tickers or []
    else:
        logger.warning(f"Unknown universe source: {source}, using S&P 500")
        return get_sp500_tickers()


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_ticker_data(
    ticker: str,
    as_of_date: pd.Timestamp,
    lookback_days: int = 252
) -> Dict[str, Any]:
    """
    Fetch all data needed for a ticker.

    Returns:
        Dict with keys: 'info', 'history', 'current_price'
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)

        # Get info (fundamentals)
        info = stock.info or {}

        # Get price history
        history = stock.history(period='1y')

        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if current_price is None and len(history) > 0:
            current_price = float(history['Close'].iloc[-1])

        return {
            'ticker': ticker,
            'info': info,
            'history': history,
            'current_price': current_price,
            'error': None
        }

    except Exception as e:
        return {
            'ticker': ticker,
            'info': {},
            'history': pd.DataFrame(),
            'current_price': None,
            'error': str(e)
        }


# =============================================================================
# Universe Building
# =============================================================================

def process_single_ticker(
    ticker: str,
    as_of_date: pd.Timestamp,
    config: Dict[str, Any],
    ticker_data: Optional[Dict] = None
) -> Optional[StockSnapshot]:
    """
    Process a single ticker and create a StockSnapshot if it passes filters.

    Args:
        ticker: Stock ticker symbol
        as_of_date: Date for analysis
        config: Trading configuration
        ticker_data: Pre-fetched ticker data (optional)

    Returns:
        StockSnapshot if ticker passes filters, None otherwise
    """
    from single_stock.valuation import compute_intrinsic_value
    from portfolio.factor_models import compute_factor_scores

    # Fetch data if not provided
    if ticker_data is None:
        ticker_data = fetch_ticker_data(ticker, as_of_date)

    if ticker_data.get('error'):
        logger.debug(f"Skipping {ticker}: {ticker_data['error']}")
        return None

    if ticker_data['current_price'] is None:
        logger.debug(f"Skipping {ticker}: No price data")
        return None

    info = ticker_data['info']
    history = ticker_data['history']
    current_price = ticker_data['current_price']

    # Get filter thresholds from config
    filters = config.get('universe', {}).get('filters', {})
    min_upside = filters.get('min_upside_pct', 0.20)
    min_quality = filters.get('min_quality_score', 0.0)
    min_momentum = filters.get('min_momentum_score', 0.0)
    min_market_cap = filters.get('min_market_cap', 1e9)
    min_volume = filters.get('min_avg_volume', 100000)

    # Check market cap
    market_cap = info.get('marketCap')
    if market_cap is not None and market_cap < min_market_cap:
        logger.debug(f"Skipping {ticker}: Market cap ${market_cap/1e9:.1f}B < ${min_market_cap/1e9:.1f}B")
        return None

    # Check average volume
    avg_volume = info.get('averageVolume')
    if avg_volume is not None and avg_volume < min_volume:
        logger.debug(f"Skipping {ticker}: Avg volume {avg_volume:,.0f} < {min_volume:,.0f}")
        return None

    # Compute intrinsic value
    valuation = compute_intrinsic_value(
        ticker=ticker,
        as_of_date=as_of_date,
        config=config,
        ticker_info=info,
        current_price=current_price
    )

    # Check upside threshold
    upside_pct = valuation.get('upside_pct')
    if upside_pct is not None and upside_pct < min_upside:
        logger.debug(f"Skipping {ticker}: Upside {upside_pct:.1%} < {min_upside:.1%}")
        return None

    # Compute factor scores
    factor_scores = compute_factor_scores(
        ticker=ticker,
        as_of_date=as_of_date,
        config=config,
        ticker_info=info,
        price_history=history
    )

    # Check quality threshold
    if factor_scores['quality'] < min_quality:
        logger.debug(f"Skipping {ticker}: Quality {factor_scores['quality']:.2f} < {min_quality:.2f}")
        return None

    # Check momentum threshold
    if factor_scores['momentum'] < min_momentum:
        logger.debug(f"Skipping {ticker}: Momentum {factor_scores['momentum']:.2f} < {min_momentum:.2f}")
        return None

    # Extract fundamentals for snapshot
    fundamentals = {
        'pe_ratio': valuation.get('pe_ratio'),
        'pb_ratio': valuation.get('pb_ratio'),
        'ev_ebitda': valuation.get('ev_ebitda'),
        'roe': info.get('returnOnEquity'),
        'roa': info.get('returnOnAssets'),
        'profit_margin': info.get('profitMargins'),
        'revenue_growth': info.get('revenueGrowth'),
        'earnings_growth': info.get('earningsGrowth'),
        'eps': info.get('trailingEps'),
        'dividend_yield': info.get('dividendYield'),
    }

    # Create snapshot
    snapshot = StockSnapshot(
        ticker=ticker,
        date=as_of_date,
        fundamentals=fundamentals,
        factor_scores=factor_scores,
        current_price=current_price,
        intrinsic_value=valuation.get('intrinsic_value'),
        upside_pct=upside_pct,
        sector=info.get('sector'),
        market_cap=market_cap,
        metadata={
            'valuation_method': valuation.get('valuation_method'),
            'valuation_confidence': valuation.get('confidence'),
        }
    )

    return snapshot


def build_universe(
    as_of_date: pd.Timestamp,
    config: Dict[str, Any],
    tickers: Optional[List[str]] = None,
    max_workers: int = 10,
    progress_callback: Optional[callable] = None
) -> List[StockSnapshot]:
    """
    Build the investable universe for a given date.

    This is the main entry point for Phase 1 of the pipeline.

    Args:
        as_of_date: Date for universe construction
        config: Trading configuration (from config.trading)
        tickers: Override list of tickers (default: from config source)
        max_workers: Number of parallel workers for data fetching
        progress_callback: Optional callback for progress updates

    Returns:
        List of StockSnapshot objects for stocks that pass all filters

    Example:
        >>> from utils.config import get_config
        >>> config = get_config()
        >>> universe = build_universe(pd.Timestamp.now(), config.trading)
        >>> print(f"Universe: {len(universe)} stocks")
        >>> for s in universe[:5]:
        ...     print(f"  {s.ticker}: upside={s.upside_pct:.1%}, quality={s.factor_scores['quality']:.2f}")
    """
    logger.phase_start("Universe Building", as_of_date=as_of_date.strftime('%Y-%m-%d'))

    # Get universe config
    universe_cfg = config.get('universe', {})
    source = universe_cfg.get('source', 'sp500')
    custom_tickers = universe_cfg.get('custom_tickers', [])
    max_size = universe_cfg.get('filters', {}).get('max_universe_size', 100)

    # Get base tickers
    if tickers is None:
        tickers = get_base_universe(source, custom_tickers)

    logger.info(f"Base universe: {len(tickers)} tickers from {source}")

    # Fetch data in parallel
    logger.info("Fetching ticker data...")
    ticker_data_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_ticker_data, ticker, as_of_date): ticker
            for ticker in tickers
        }

        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            try:
                data = future.result()
                ticker_data_map[ticker] = data
            except Exception as e:
                logger.debug(f"Error fetching {ticker}: {e}")

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, len(tickers))

    logger.info(f"Fetched data for {len(ticker_data_map)} tickers")

    # Process each ticker
    logger.info("Processing tickers...")
    universe = []
    processed = 0
    errors = 0

    for ticker in tickers:
        try:
            ticker_data = ticker_data_map.get(ticker)
            snapshot = process_single_ticker(ticker, as_of_date, config, ticker_data)

            if snapshot is not None:
                universe.append(snapshot)

            processed += 1

        except Exception as e:
            logger.debug(f"Error processing {ticker}: {e}")
            errors += 1

    # Sort by composite factor score (descending)
    universe.sort(key=lambda s: s.factor_scores.get('composite', 0), reverse=True)

    # Limit to max size
    if len(universe) > max_size:
        universe = universe[:max_size]
        logger.info(f"Limited universe to {max_size} top-ranked stocks")

    # Log summary
    logger.items_processed(len(tickers), len(universe), "stocks")

    if universe:
        avg_upside = np.mean([s.upside_pct for s in universe if s.upside_pct is not None])
        avg_quality = np.mean([s.factor_scores.get('quality', 0) for s in universe])
        avg_momentum = np.mean([s.factor_scores.get('momentum', 0) for s in universe])

        logger.stats({
            'n_stocks': len(universe),
            'avg_upside': avg_upside,
            'avg_quality': avg_quality,
            'avg_momentum': avg_momentum,
            'errors': errors
        })

    sectors = {}
    for s in universe:
        sector = s.sector or 'Unknown'
        sectors[sector] = sectors.get(sector, 0) + 1

    logger.info(f"Sector distribution: {sectors}")

    logger.phase_end(n_stocks=len(universe), n_processed=processed, n_errors=errors)

    return universe


def universe_to_dataframe(universe: List[StockSnapshot]) -> pd.DataFrame:
    """
    Convert universe list to DataFrame for analysis.

    Args:
        universe: List of StockSnapshot objects

    Returns:
        DataFrame with one row per stock
    """
    rows = []
    for s in universe:
        row = {
            'ticker': s.ticker,
            'date': s.date,
            'current_price': s.current_price,
            'intrinsic_value': s.intrinsic_value,
            'upside_pct': s.upside_pct,
            'sector': s.sector,
            'market_cap': s.market_cap,
        }
        row.update({f'factor_{k}': v for k, v in s.factor_scores.items()})
        row.update({f'fund_{k}': v for k, v in s.fundamentals.items()})
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for universe builder."""
    parser = argparse.ArgumentParser(
        description="Build investable universe (Phase 1 of trading pipeline)"
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='As-of date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='sp500',
        choices=['sp500', 'nasdaq100', 'dow30', 'custom'],
        help='Universe source'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=None,
        help='Custom tickers (overrides --source)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/universe.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--min-upside',
        type=float,
        default=0.20,
        help='Minimum upside percentage (default: 0.20 = 20%%)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Set up logging
    from utils.logger import setup_pipeline_logging
    setup_pipeline_logging(level='DEBUG' if args.verbose else 'INFO')

    # Parse date
    if args.date:
        as_of_date = pd.Timestamp(args.date)
    else:
        as_of_date = pd.Timestamp.now().normalize()

    # Build config
    config = {
        'universe': {
            'source': args.source,
            'custom_tickers': args.tickers or [],
            'filters': {
                'min_upside_pct': args.min_upside,
                'min_quality_score': 0.0,
                'min_momentum_score': 0.0,
                'min_market_cap': 1e9,
                'min_avg_volume': 100000,
                'max_universe_size': 100
            }
        }
    }

    print(f"\n{'='*60}")
    print("UNIVERSE BUILDER - Phase 1")
    print(f"{'='*60}")
    print(f"  As-of date: {as_of_date.strftime('%Y-%m-%d')}")
    print(f"  Source: {args.source}")
    print(f"  Min upside: {args.min_upside:.0%}")
    print(f"{'='*60}\n")

    # Build universe
    universe = build_universe(
        as_of_date=as_of_date,
        config=config,
        tickers=args.tickers,
        max_workers=args.max_workers
    )

    # Convert to DataFrame and save
    df = universe_to_dataframe(universe)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Total stocks in universe: {len(universe)}")
    print(f"  Output saved to: {output_path}")

    if universe:
        print(f"\n  Top 10 by composite score:")
        for s in universe[:10]:
            print(f"    {s.ticker:6} | upside={s.upside_pct:+.1%} | "
                  f"quality={s.factor_scores.get('quality', 0):+.2f} | "
                  f"momentum={s.factor_scores.get('momentum', 0):+.2f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
