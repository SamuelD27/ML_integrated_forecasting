#!/usr/bin/env python3
"""
Simplified Stock Analysis Runner
=================================
Run deep quantitative analysis on a single stock with just the ticker.

Usage:
    python run_portfolio.py aapl
    python run_portfolio.py MSFT
    python run_portfolio.py 9992.HK
    python run_portfolio.py tsla --years 5
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Standardized defaults
DEFAULT_YEARS = 2  # Historical data lookback in years
DEFAULT_BENCHMARK = "^GSPC"  # S&P 500 as default benchmark

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def run_stock_analysis(ticker: str, years: int = None, benchmark: str = None,
                      verbose: bool = True):
    """
    Run deep quantitative analysis on a single stock.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (will be converted to uppercase)
    years : int, optional
        Years of historical data (defaults to 2)
    benchmark : str, optional
        Benchmark ticker (defaults to ^GSPC - S&P 500)
    verbose : bool
        Whether to print execution details
    """
    # Normalize ticker to uppercase
    ticker = ticker.upper()

    # Use defaults if not specified
    years = years or DEFAULT_YEARS
    benchmark = benchmark or DEFAULT_BENCHMARK
    start_date = (datetime.now() - timedelta(days=365*years)).strftime("%Y-%m-%d")

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = REPORTS_DIR / f"{ticker}_analysis_{timestamp}.xlsx"

    # Use quick_stock_report.py for comprehensive single stock analysis
    script_path = SCRIPT_DIR / "quick_stock_report.py"
    script_name = "Deep Quantitative Analysis"

    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        ticker,
        "--start", start_date,
        "--benchmark", benchmark,
        "--output", str(output_file)
    ]

    # Print execution summary
    if verbose:
        print("=" * 70)
        print(f"  {script_name}")
        print("=" * 70)
        print(f"Ticker:           {ticker}")
        print(f"Historical Data:  {years} years ({start_date} to present)")
        print(f"Benchmark:        {benchmark}")
        print(f"Output:           {output_file.name}")
        print("=" * 70)
        print()

    # Run the command
    try:
        result = subprocess.run(cmd, check=True)

        if verbose:
            print()
            print("=" * 70)
            print("✓ Stock analysis completed successfully!")
            print(f"✓ Report saved to: {output_file}")
            print("=" * 70)

        return result.returncode

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running stock analysis: {e}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run deep quantitative analysis on a single stock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s aapl                    # Analyze Apple with 2 years of data
  %(prog)s MSFT --years 5          # Analyze Microsoft with 5 years of data
  %(prog)s 9992.HK                 # Analyze Hong Kong stock
  %(prog)s tsla --benchmark QQQ    # Use NASDAQ as benchmark

Analysis Includes:
  - Descriptive statistics & price history
  - Risk metrics (VaR, Sharpe, Sortino, Beta, etc.)
  - Technical indicators (Moving averages, RSI, MACD)
  - Volatility analysis (rolling vol, GARCH if available)
  - Returns distribution & correlation analysis
  - ARIMA time series forecasting
  - Fundamental ratios (if available)
  - Comprehensive Excel report with charts

Default Settings:
  Historical Data:  2 years
  Benchmark:        ^GSPC (S&P 500)
        """
    )

    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (case insensitive)"
    )

    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help=f"Years of historical data (default: {DEFAULT_YEARS})"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help=f"Benchmark ticker for comparison (default: {DEFAULT_BENCHMARK})"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress execution details"
    )

    args = parser.parse_args()

    # Validate years if provided
    if args.years is not None and args.years <= 0:
        parser.error("Years must be positive")

    # Run analysis
    return run_stock_analysis(
        ticker=args.ticker,
        years=args.years,
        benchmark=args.benchmark,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    sys.exit(main())
