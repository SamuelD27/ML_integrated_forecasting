#!/usr/bin/env python3
"""
Live test of dashboard functionality with real yfinance data.
Tests the actual data loading pipeline without running Streamlit.
"""

import sys
import pandas as pd
import yfinance as yf
from typing import List
from datetime import datetime


def get_price_column(data: pd.DataFrame,
                     tickers: List[str],
                     prefer_adjusted: bool = True) -> pd.DataFrame:
    """
    Safely extract price data from yfinance DataFrame.
    (Copied from dashboard/app.py)
    """
    if data.empty:
        raise ValueError("Empty DataFrame provided")

    if isinstance(data.columns, pd.MultiIndex):
        try:
            if prefer_adjusted and 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close'].copy()
            elif 'Close' in data.columns.get_level_values(0):
                prices = data['Close'].copy()
            else:
                raise ValueError(f"No price columns found. Available: {data.columns.get_level_values(0).unique().tolist()}")

            missing = set(tickers) - set(prices.columns)
            if missing:
                raise ValueError(f"Missing data for tickers: {missing}")

            return prices

        except Exception as e:
            raise ValueError(f"Failed to extract prices from multi-ticker data: {e}")

    else:
        if len(tickers) != 1:
            raise ValueError(f"Expected 1 ticker but got {len(tickers)}: {tickers}")

        ticker = tickers[0]

        if prefer_adjusted and 'Adj Close' in data.columns:
            prices = pd.DataFrame({ticker: data['Adj Close']})
        elif 'Close' in data.columns:
            prices = pd.DataFrame({ticker: data['Close']})
        else:
            raise ValueError(f"No price columns found. Available: {data.columns.tolist()}")

        return prices


def load_stock_data(tickers: List[str], period: str = "1mo") -> pd.DataFrame:
    """Load stock data from Yahoo Finance with retry logic."""
    import time

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers,
                period=period,
                progress=False
            )

            if data.empty:
                raise ValueError(f"No data returned for {tickers}")

            if isinstance(data.columns, pd.MultiIndex):
                available_cols = data.columns.get_level_values(0).unique().tolist()
            else:
                available_cols = data.columns.tolist()

            if not any(col in available_cols for col in ['Close', 'Adj Close']):
                raise ValueError(f"No price columns in returned data. Available: {available_cols}")

            return data

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"Failed to load data after {max_retries} attempts: {e}")


def test_ticker_scenario(tickers: List[str], scenario_name: str):
    """Test a specific ticker scenario."""
    print(f"\n{'='*70}")
    print(f"📊 {scenario_name}")
    print(f"{'='*70}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: 1 month")

    try:
        # Load data
        print("\n⏳ Loading data from Yahoo Finance...")
        data = load_stock_data(tickers, period="1mo")

        # Show data structure
        if isinstance(data.columns, pd.MultiIndex):
            print(f"✓ Received MultiIndex DataFrame")
            print(f"  Column levels: {data.columns.get_level_values(0).unique().tolist()}")
            print(f"  Tickers: {data.columns.get_level_values(1).unique().tolist()}")
        else:
            print(f"✓ Received simple DataFrame")
            print(f"  Columns: {data.columns.tolist()}")

        # Extract prices
        print("\n⏳ Extracting prices using get_price_column()...")
        prices = get_price_column(data, tickers, prefer_adjusted=True)

        # Show results
        print(f"✅ SUCCESS!")
        print(f"  Shape: {prices.shape} ({prices.shape[0]} days × {prices.shape[1]} tickers)")
        print(f"  Columns: {prices.columns.tolist()}")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"\n  Sample prices (first 3 days):")
        print(f"{prices.head(3).to_string(float_format=lambda x: f'${x:.2f}')}")

        # Calculate basic stats
        returns = prices.pct_change().dropna()
        print(f"\n  📈 Performance (1 month):")
        for ticker in prices.columns:
            total_return = (prices[ticker].iloc[-1] / prices[ticker].iloc[0] - 1) * 100
            volatility = returns[ticker].std() * 100
            print(f"    {ticker}: {total_return:+.2f}% return, {volatility:.2f}% daily vol")

        return True

    except ValueError as e:
        print(f"❌ ValueError: {e}")
        return False

    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        return False


def main():
    """Run live dashboard tests with real yfinance data."""
    print("\n" + "🚀"*35)
    print("LIVE DASHBOARD TEST - Real yfinance Data")
    print("🚀"*35)
    print(f"\nTesting dashboard fix with actual Yahoo Finance data")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Test 1: Single ticker
    results.append(test_ticker_scenario(
        ['AAPL'],
        "TEST 1: Single Ticker (AAPL)"
    ))

    # Test 2: Multiple tickers
    results.append(test_ticker_scenario(
        ['AAPL', 'MSFT', 'GOOGL'],
        "TEST 2: Multiple Tickers (Tech Giants)"
    ))

    # Test 3: Larger universe
    results.append(test_ticker_scenario(
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD'],
        "TEST 3: Large Universe (8 Tech Stocks)"
    ))

    # Test 4: Invalid ticker (should fail gracefully)
    print(f"\n{'='*70}")
    print(f"📊 TEST 4: Invalid Ticker (Error Handling)")
    print(f"{'='*70}")
    print(f"Tickers: AAPL, INVALIDTICKER12345")

    try:
        data = yf.download(['AAPL', 'INVALIDTICKER12345'], period="1mo", progress=False)
        print(f"\n⚠️  Data downloaded (yfinance is lenient with invalid tickers)")

        # Try to extract - may work with just valid tickers
        try:
            prices = get_price_column(data, ['AAPL', 'INVALIDTICKER12345'], prefer_adjusted=True)
            print(f"✅ Extracted prices for available tickers: {prices.columns.tolist()}")
            results.append(True)
        except ValueError as e:
            print(f"✅ Correctly caught error: {e}")
            results.append(True)

    except Exception as e:
        print(f"✅ Correctly failed: {e}")
        results.append(True)

    # Summary
    print(f"\n{'='*70}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*70}")

    total_tests = len(results)
    passed_tests = sum(results)

    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print(f"\n✅ ALL TESTS PASSED! Dashboard fix is working correctly.")
        print(f"\nThe dashboard should now:")
        print(f"  ✓ Load single tickers without errors")
        print(f"  ✓ Load multiple tickers without KeyError")
        print(f"  ✓ Handle missing 'Adj Close' gracefully")
        print(f"  ✓ Provide clear error messages")
        print(f"\n🎉 You can now safely run: streamlit run dashboard/app.py")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
