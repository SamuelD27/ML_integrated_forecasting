import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_fetching import fetch_full_bundle, save_bundle, set_last_fetch_globals
from portfolio.peer_discovery import discover_peers, get_ticker_metadata
from portfolio.etf_discovery import discover_etfs
from portfolio.options_overlay import select_hedge_options, calculate_hedge_budget
from portfolio.advanced_optimizer import (
    compute_returns_matrix,
    optimize_mean_variance,
    calculate_portfolio_shares
)


def extract_prices_multiindex(prices_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Extract adjusted close prices from yfinance MultiIndex DataFrame.

    Handles both single-ticker and multi-ticker responses from yfinance.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Raw yfinance data (may have MultiIndex columns)
    tickers : list
        List of tickers to extract

    Returns
    -------
    pd.DataFrame
        Clean prices DataFrame with tickers as columns
    """
    prices_clean = {}

    try:
        if isinstance(prices_df.columns, pd.MultiIndex):
            # MultiIndex: (field, ticker) or (ticker, field)
            lvl0 = prices_df.columns.get_level_values(0)
            lvl1 = prices_df.columns.get_level_values(1)

            # Determine which level contains 'Adj Close'
            if 'Adj Close' in set(lvl0):
                # Level 0 is field, level 1 is ticker
                for ticker in tickers:
                    if ticker in set(lvl1):
                        try:
                            prices_clean[ticker] = prices_df[('Adj Close', ticker)].astype(float)
                        except Exception:
                            pass
            elif 'Adj Close' in set(lvl1):
                # Level 1 is field, level 0 is ticker
                for ticker in tickers:
                    if ticker in set(lvl0):
                        try:
                            prices_clean[ticker] = prices_df[(ticker, 'Adj Close')].astype(float)
                        except Exception:
                            pass
        else:
            # Single ticker or flat structure
            for ticker in tickers:
                if ticker in prices_df.columns:
                    prices_clean[ticker] = prices_df[ticker].astype(float)
                elif 'Adj Close' in prices_df.columns:
                    # Single ticker case
                    prices_clean[ticker] = prices_df['Adj Close'].astype(float)

    except Exception as e:
        print(f"  Warning: Error extracting prices: {e}")

    if not prices_clean:
        raise ValueError("Failed to extract any price data from DataFrame")

    prices_df_clean = pd.DataFrame(prices_clean).ffill().dropna()
    return prices_df_clean


def run_portfolio_construction(ticker: str, capital: float, rrr: float,
                                start_date: Optional[str] = None,
                                max_peers: int = 10,
                                target_dte: int = 90) -> Dict:
    """
    Complete portfolio construction workflow.

    Parameters
    ----------
    ticker : str
        Primary ticker symbol
    capital : float
        Investment capital (USD)
    rrr : float
        Reward-to-risk ratio (0 to 1)
        Higher RRR → more defensive, lower beta, larger hedge
    start_date : str, optional
        Historical data start date (default: 1 year ago)
    max_peers : int
        Maximum number of peer stocks (default: 10)
    target_dte : int
        Target days to expiration for options (default: 90)

    Returns
    -------
    dict
        Complete analysis results
    """
    print(f"\n{'='*80}")
    print(f"MULTI-ASSET PORTFOLIO CONSTRUCTION: {ticker}")
    print(f"{'='*80}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Reward-to-Risk Ratio: {rrr:.2f}")
    print(f"{'='*80}\n")

    workflow_start = time.time()

    # Step 1: Discover peers
    print("Step 1/6: Discovering peer companies...")
    peer_result = discover_peers(ticker, max_peers=max_peers)

    primary_meta = peer_result['primary_meta']
    peers = peer_result['peers']

    if not peers:
        print(f"  Warning: No peers found. Portfolio will include only {ticker} + ETFs.")

    # Step 2: Discover ETFs
    print("\nStep 2/6: Discovering relevant ETFs...")
    etf_result = discover_etfs(primary_meta)

    etf_tickers = etf_result['all_etfs']

    # Step 3: Build final universe
    universe = [ticker] + peers + etf_tickers
    universe = list(dict.fromkeys(universe))  # Remove duplicates

    print(f"\nStep 3/6: Building portfolio universe...")
    print(f"  Primary: {ticker}")
    print(f"  Peers: {peers} ({len(peers)})")
    print(f"  ETFs: {etf_tickers} ({len(etf_tickers)})")
    print(f"  Total universe: {len(universe)} assets")

    # Step 4: Fetch historical data
    print(f"\nStep 4/6: Fetching historical data...")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    bundle = fetch_full_bundle(tickers=universe, start=start_date)
    save_bundle(bundle, base_name="last_fetch")
    set_last_fetch_globals(bundle)

    prices_df_raw = bundle['prices']

    # Extract clean prices
    try:
        prices_df = extract_prices_multiindex(prices_df_raw, universe)
        print(f"  ✓ Extracted prices for {len(prices_df.columns)} assets")
    except Exception as e:
        print(f"  ✗ Error extracting prices: {e}")
        # Fallback: primary ticker only
        print(f"  Falling back to primary ticker only: {ticker}")
        prices_df = extract_prices_multiindex(prices_df_raw, [ticker])
        universe = [ticker]

    # Step 5: Portfolio optimization
    print(f"\nStep 5/6: Optimizing portfolio allocation...")

    # Identify market ETF for beta calculation
    market_etf = etf_result.get('market_etf', {}).get('symbol') if etf_result.get('market_etf') else None
    if market_etf not in prices_df.columns:
        market_etf = None

    returns = compute_returns_matrix(prices_df)

    opt_result = optimize_mean_variance(
        returns=returns,
        rrr=rrr,
        market_ticker=market_etf,
        max_weight=0.40,
        min_weight=0.0
    )

    weights = opt_result['weights']
    current_prices = prices_df.iloc[-1]

    holdings_df = calculate_portfolio_shares(weights, capital, current_prices)

    print(f"\n  Portfolio allocation:")
    for _, row in holdings_df.iterrows():
        print(f"    {row['ticker']:6s}: {row['weight']*100:6.2f}% | ${row['dollars']:12,.2f} | {row['shares']:6.0f} shares @ ${row['price']:.2f}")

    # Step 6: Options overlay
    print(f"\nStep 6/6: Constructing options hedge overlay...")

    # Calculate hedge budget from RRR
    hedge_budget = calculate_hedge_budget(rrr, capital)
    print(f"  Hedge budget (based on RRR={rrr:.2f}): ${hedge_budget:,.2f} ({hedge_budget/capital*100:.1f}%)")

    # Get current price for primary ticker
    primary_price = current_prices.get(ticker, 0)

    # Calculate portfolio exposure to primary ticker
    primary_weight = weights.get(ticker, 0)
    primary_portfolio_value = capital * primary_weight

    options_overlay = None
    if primary_price > 0:
        options_overlay = select_hedge_options(
            ticker=ticker,
            current_price=primary_price,
            portfolio_value=primary_portfolio_value,
            hedge_budget=hedge_budget,
            strategy='put',
            target_dte=target_dte
        )

    if not options_overlay:
        print(f"  ✗ No options overlay available for {ticker}")

    # Compile results
    elapsed = time.time() - workflow_start

    print(f"\n{'='*80}")
    print(f"✓ Portfolio construction completed in {elapsed:.1f}s")
    print(f"{'='*80}\n")

    return {
        'ticker': ticker,
        'capital': capital,
        'rrr': rrr,
        'universe': universe,
        'peer_result': peer_result,
        'etf_result': etf_result,
        'prices_df': prices_df,
        'returns': returns,
        'opt_result': opt_result,
        'holdings_df': holdings_df,
        'options_overlay': options_overlay,
        'hedge_budget': hedge_budget,
        'elapsed_time': elapsed,
    }


def export_portfolio_report(analysis: Dict, output_path: str):
    """
    Generate comprehensive Excel report.

    Parameters
    ----------
    analysis : dict
        Portfolio analysis results
    output_path : str
        Output Excel file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ticker = analysis['ticker']
    capital = analysis['capital']
    rrr = analysis['rrr']

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formats
        title_fmt = workbook.add_format({
            'bold': True, 'font_size': 16, 'font_color': '#1F4788', 'align': 'center'
        })
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1, 'align': 'center'
        })
        section_fmt = workbook.add_format({
            'bold': True, 'font_size': 12, 'bg_color': '#D9E1F2', 'border': 1
        })
        number_fmt = workbook.add_format({'num_format': '#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})

        # ===== SHEET 1: PORTFOLIO SUMMARY =====
        ws = workbook.add_worksheet('Portfolio Summary')
        ws.set_column('A:A', 30)
        ws.set_column('B:E', 18)

        row = 0

        # Title
        ws.merge_range(row, 0, row, 4, f'MULTI-ASSET PORTFOLIO: {ticker}', title_fmt)
        row += 2

        # Parameters
        ws.merge_range(row, 0, row, 4, 'PORTFOLIO PARAMETERS', section_fmt)
        row += 1

        params = [
            ['Primary Ticker', ticker, '', '', ''],
            ['Total Capital', f'${capital:,.2f}', '', '', ''],
            ['Reward-to-Risk Ratio', f'{rrr:.2f}', '', '', ''],
            ['Universe Size', len(analysis['universe']), '', '', ''],
        ]

        for data in params:
            for col, val in enumerate(data):
                ws.write(row, col, val)
            row += 1

        row += 1

        # Holdings
        ws.merge_range(row, 0, row, 4, 'PORTFOLIO HOLDINGS', section_fmt)
        row += 1

        holdings_headers = ['Ticker', 'Weight', 'Dollars', 'Shares', 'Price']
        for col, header in enumerate(holdings_headers):
            ws.write(row, col, header, header_fmt)
        row += 1

        holdings_df = analysis['holdings_df'].sort_values('weight', ascending=False)
        for _, holding in holdings_df.iterrows():
            ws.write(row, 0, holding['ticker'])
            ws.write(row, 1, f"{holding['weight']*100:.2f}%")
            ws.write(row, 2, f"${holding['dollars']:,.2f}")
            ws.write(row, 3, int(holding['shares']))
            ws.write(row, 4, f"${holding['price']:.2f}")
            row += 1

        row += 1

        # Portfolio metrics
        ws.merge_range(row, 0, row, 4, 'PORTFOLIO METRICS', section_fmt)
        row += 1

        opt = analysis['opt_result']
        metrics = [
            ['Metric', 'Value', '', '', ''],
            ['Expected Return (annual)', f"{opt['port_return']*100:.2f}%", '', '', ''],
            ['Volatility (annual)', f"{opt['port_vol']*100:.2f}%", '', '', ''],
            ['Sharpe Ratio', f"{opt['sharpe']:.3f}", '', '', ''],
            ['Risk Aversion (lambda)', f"{opt['risk_aversion']:.2f}", '', '', ''],
        ]

        if opt['realized_beta'] is not None:
            metrics.append(['Realized Beta', f"{opt['realized_beta']:.3f}", '', '', ''])
            if opt['target_beta'] is not None:
                metrics.append(['Target Beta', f"{opt['target_beta']:.3f}", '', '', ''])

        for data in metrics:
            for col, val in enumerate(data):
                if col == 0:
                    ws.write(row, col, val, header_fmt)
                else:
                    ws.write(row, col, val)
            row += 1

        # ===== SHEET 2: OPTIONS OVERLAY =====
        ws2 = workbook.add_worksheet('Options Overlay')
        ws2.set_column('A:A', 30)
        ws2.set_column('B:G', 18)

        row = 0
        ws2.merge_range(row, 0, row, 6, 'OPTIONS HEDGE OVERLAY', title_fmt)
        row += 2

        overlay = analysis['options_overlay']

        if overlay:
            ws2.merge_range(row, 0, row, 6, 'HEDGE DETAILS', section_fmt)
            row += 1

            hedge_info = [
                ['Hedge Budget', f"${analysis['hedge_budget']:,.2f}"],
                ['Strategy', overlay['strategy'].upper()],
                ['Underlying', overlay['ticker']],
                ['Strike Price', f"${overlay['strike']:.2f}"],
                ['Moneyness', f"{overlay['moneyness']*100:.1f}%"],
                ['Expiration', overlay['expiration']],
                ['Days to Expiration', overlay['dte']],
                ['Contracts', overlay['contracts']],
                ['Shares Hedged', overlay['shares_hedged']],
                ['Option Price', f"${overlay['option_price']:.2f}"],
                ['Total Premium', f"${overlay['total_premium']:,.2f}"],
                ['Portfolio Coverage', f"{overlay['hedge_coverage']*100:.1f}%"],
            ]

            for label, value in hedge_info:
                ws2.write(row, 0, label)
                ws2.write(row, 1, value)
                row += 1

            row += 1
            ws2.merge_range(row, 0, row, 6, 'OPTION GREEKS & IV', section_fmt)
            row += 1

            greeks = overlay['greeks']
            greeks_info = [
                ['Delta', f"{greeks['delta']:.4f}"],
                ['Gamma', f"{greeks['gamma']:.4f}"],
                ['Vega', f"{greeks['vega']:.4f}"],
                ['Theta (per day)', f"{greeks['theta']:.4f}"],
            ]

            if overlay['implied_vol']:
                greeks_info.append(['Implied Volatility', f"{overlay['implied_vol']*100:.1f}%"])

            for label, value in greeks_info:
                ws2.write(row, 0, label)
                ws2.write(row, 1, value)
                row += 1

        else:
            ws2.write(row, 0, 'No options overlay available')

        # ===== SHEET 3: DIAGNOSTICS =====
        ws3 = workbook.add_worksheet('Diagnostics')
        ws3.set_column('A:A', 30)
        ws3.set_column('B:C', 25)

        row = 0
        ws3.merge_range(row, 0, row, 2, 'DIAGNOSTICS & DISCOVERY', title_fmt)
        row += 2

        # Peer discovery
        ws3.merge_range(row, 0, row, 2, 'PEER DISCOVERY', section_fmt)
        row += 1

        peer_diag = analysis['peer_result']['diagnostics']
        for key, val in peer_diag.items():
            if isinstance(val, dict):
                ws3.write(row, 0, key)
                row += 1
                for k2, v2 in val.items():
                    ws3.write(row, 0, f'  {k2}')
                    ws3.write(row, 1, str(v2))
                    row += 1
            else:
                ws3.write(row, 0, key)
                ws3.write(row, 1, str(val))
                row += 1

        row += 1

        # ETF discovery
        ws3.merge_range(row, 0, row, 2, 'ETF DISCOVERY', section_fmt)
        row += 1

        etf_diag = analysis['etf_result']['diagnostics']
        for key, val in etf_diag.items():
            ws3.write(row, 0, key)
            ws3.write(row, 1, str(val))
            row += 1

        row += 1

        # Optimization diagnostics
        ws3.merge_range(row, 0, row, 2, 'OPTIMIZATION', section_fmt)
        row += 1

        opt_info = [
            ['Shrinkage Coefficient', f"{opt['shrinkage_coeff']:.3f}"],
            ['Risk Aversion', f"{opt['risk_aversion']:.2f}"],
            ['Assets in Portfolio', len(opt['weights'])],
        ]

        for label, value in opt_info:
            ws3.write(row, 0, label)
            ws3.write(row, 1, value)
            row += 1

    print(f"\n✓ Portfolio report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Asset Portfolio Constructor with Options Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python portfolio_creation.py AAPL --capital 100000 --rrr 0.6
  python portfolio_creation.py TSLA --capital 50000 --rrr 0.8 --start 2023-01-01
  python portfolio_creation.py MSFT --capital 200000 --rrr 0.3 --max-peers 15

RRR (Reward-to-Risk Ratio) Guide:
  0.0 - 0.3: Aggressive (high growth, minimal hedge, higher beta)
  0.4 - 0.6: Moderate (balanced growth/defense)
  0.7 - 1.0: Defensive (low beta, large hedge, capital preservation)
        """
    )

    parser.add_argument('ticker', type=str, help='Primary stock ticker')
    parser.add_argument('--capital', type=float, required=True,
                       help='Investment capital in USD (required)')
    parser.add_argument('--rrr', type=float, required=True,
                       help='Reward-to-risk ratio, 0.0 to 1.0 (required)')
    parser.add_argument('--start', type=str, default=None,
                       help='Historical data start date YYYY-MM-DD (default: 1 year ago)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: reports/{ticker}_portfolio.xlsx)')
    parser.add_argument('--max-peers', type=int, default=10,
                       help='Maximum number of peer stocks (default: 10)')
    parser.add_argument('--target-dte', type=int, default=90,
                       help='Target days to expiration for options (default: 90)')

    args = parser.parse_args()

    # Validate inputs
    if args.capital <= 0:
        print("Error: Capital must be positive")
        sys.exit(1)

    if not (0.0 <= args.rrr <= 1.0):
        print("Error: RRR must be between 0.0 and 1.0")
        sys.exit(1)

    if args.output is None:
        args.output = f"reports/{args.ticker}_portfolio.xlsx"

    try:
        # Run portfolio construction
        analysis = run_portfolio_construction(
            ticker=args.ticker,
            capital=args.capital,
            rrr=args.rrr,
            start_date=args.start,
            max_peers=args.max_peers,
            target_dte=args.target_dte
        )

        # Export report
        print("\nGenerating Excel report...")
        export_portfolio_report(analysis, args.output)

        # Console summary
        print(f"\n{'='*80}")
        print("PORTFOLIO CONSTRUCTION SUMMARY")
        print(f"{'='*80}")
        print(f"Primary Ticker: {args.ticker}")
        print(f"Total Capital: ${args.capital:,.2f}")
        print(f"RRR: {args.rrr:.2f}")
        print(f"\nUniverse: {len(analysis['universe'])} assets")
        print(f"  Primary: {args.ticker}")
        print(f"  Peers: {len(analysis['peer_result']['peers'])}")
        print(f"  ETFs: {len(analysis['etf_result']['all_etfs'])}")

        print(f"\nPortfolio Metrics:")
        print(f"  Expected Return: {analysis['opt_result']['port_return']*100:.2f}%")
        print(f"  Volatility: {analysis['opt_result']['port_vol']*100:.2f}%")
        print(f"  Sharpe Ratio: {analysis['opt_result']['sharpe']:.3f}")

        if analysis['opt_result']['realized_beta'] is not None:
            print(f"  Realized Beta: {analysis['opt_result']['realized_beta']:.3f}")

        if analysis['options_overlay']:
            overlay = analysis['options_overlay']
            print(f"\nOptions Hedge:")
            print(f"  Strategy: {overlay['strategy'].upper()}")
            print(f"  Strike: ${overlay['strike']:.2f}")
            print(f"  Contracts: {overlay['contracts']}")
            print(f"  Premium: ${overlay['total_premium']:,.2f}")
            print(f"  Coverage: {overlay['hedge_coverage']*100:.1f}%")
        else:
            print(f"\nOptions Hedge: Not available")

        print(f"\nReport: {args.output}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
