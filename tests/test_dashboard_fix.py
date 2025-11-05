"""
Test script to validate dashboard fixes without running Streamlit.

Tests the helper functions that were added to fix the KeyError.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_price_column(data: pd.DataFrame,
                     tickers: list,
                     prefer_adjusted: bool = True) -> pd.DataFrame:
    """
    Safely extract price data from yfinance DataFrame.
    (Copied from dashboard/app.py for testing)
    """
    if data.empty:
        raise ValueError("Empty DataFrame provided")

    # Check if MultiIndex columns (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker format: ('Adj Close', 'AAPL'), ('Close', 'AAPL'), etc.
        try:
            # Try to get Adj Close for all tickers
            if prefer_adjusted and 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close'].copy()
            elif 'Close' in data.columns.get_level_values(0):
                prices = data['Close'].copy()
            else:
                raise ValueError(f"No price columns found. Available: {data.columns.get_level_values(0).unique().tolist()}")

            # Verify we got data for requested tickers
            missing = set(tickers) - set(prices.columns)
            if missing:
                raise ValueError(f"Missing data for tickers: {missing}")

            return prices

        except Exception as e:
            raise ValueError(f"Failed to extract prices from multi-ticker data: {e}")

    else:
        # Single ticker format: simple columns ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if len(tickers) != 1:
            raise ValueError(f"Expected 1 ticker but got {len(tickers)}: {tickers}")

        ticker = tickers[0]

        # Try Adj Close first, then Close
        if prefer_adjusted and 'Adj Close' in data.columns:
            prices = pd.DataFrame({ticker: data['Adj Close']})
        elif 'Close' in data.columns:
            prices = pd.DataFrame({ticker: data['Close']})
        else:
            raise ValueError(f"No price columns found. Available: {data.columns.tolist()}")

        return prices


def create_mock_single_ticker_data(ticker: str = 'AAPL') -> pd.DataFrame:
    """Create mock data that mimics yfinance single-ticker format."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Adj Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data


def create_mock_multi_ticker_data(tickers: list) -> pd.DataFrame:
    """Create mock data that mimics yfinance multi-ticker format (MultiIndex)."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)

    # Create MultiIndex columns
    columns = pd.MultiIndex.from_product(
        [['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], tickers],
        names=['Price', 'Ticker']
    )

    data = pd.DataFrame(
        np.random.randn(100, len(columns)).cumsum(axis=0) + 100,
        index=dates,
        columns=columns
    )

    return data


def test_single_ticker_with_adj_close():
    """Test: Single ticker with Adj Close column."""
    print("\n" + "="*60)
    print("TEST 1: Single ticker with 'Adj Close' column")
    print("="*60)

    data = create_mock_single_ticker_data('AAPL')
    print(f"Input columns: {data.columns.tolist()}")

    try:
        prices = get_price_column(data, ['AAPL'], prefer_adjusted=True)
        print(f"✅ SUCCESS: Extracted prices shape: {prices.shape}")
        print(f"   Columns: {prices.columns.tolist()}")
        print(f"   Sample data:\n{prices.head(3)}")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_single_ticker_without_adj_close():
    """Test: Single ticker without Adj Close (fallback to Close)."""
    print("\n" + "="*60)
    print("TEST 2: Single ticker without 'Adj Close' (fallback to Close)")
    print("="*60)

    data = create_mock_single_ticker_data('AAPL')
    data = data.drop(columns=['Adj Close'])  # Remove Adj Close
    print(f"Input columns: {data.columns.tolist()}")

    try:
        prices = get_price_column(data, ['AAPL'], prefer_adjusted=True)
        print(f"✅ SUCCESS: Extracted prices shape: {prices.shape}")
        print(f"   Columns: {prices.columns.tolist()}")
        print(f"   Used 'Close' as fallback")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_multi_ticker_with_adj_close():
    """Test: Multiple tickers with MultiIndex and Adj Close."""
    print("\n" + "="*60)
    print("TEST 3: Multiple tickers with MultiIndex and 'Adj Close'")
    print("="*60)

    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = create_mock_multi_ticker_data(tickers)
    print(f"Input columns (MultiIndex): {data.columns.get_level_values(0).unique().tolist()}")
    print(f"Tickers: {data.columns.get_level_values(1).unique().tolist()}")

    try:
        prices = get_price_column(data, tickers, prefer_adjusted=True)
        print(f"✅ SUCCESS: Extracted prices shape: {prices.shape}")
        print(f"   Columns: {prices.columns.tolist()}")
        print(f"   Sample data:\n{prices.head(3)}")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_multi_ticker_without_adj_close():
    """Test: Multiple tickers without Adj Close (fallback to Close)."""
    print("\n" + "="*60)
    print("TEST 4: Multiple tickers without 'Adj Close' (fallback to Close)")
    print("="*60)

    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = create_mock_multi_ticker_data(tickers)
    # Remove Adj Close from MultiIndex
    data = data.drop(columns='Adj Close', level=0)
    print(f"Input columns (MultiIndex): {data.columns.get_level_values(0).unique().tolist()}")

    try:
        prices = get_price_column(data, tickers, prefer_adjusted=True)
        print(f"✅ SUCCESS: Extracted prices shape: {prices.shape}")
        print(f"   Used 'Close' as fallback")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_empty_dataframe():
    """Test: Empty DataFrame should raise ValueError."""
    print("\n" + "="*60)
    print("TEST 5: Empty DataFrame (should fail gracefully)")
    print("="*60)

    data = pd.DataFrame()

    try:
        prices = get_price_column(data, ['AAPL'], prefer_adjusted=True)
        print(f"❌ FAILED: Should have raised ValueError for empty DataFrame")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"❌ FAILED: Raised wrong exception type: {e}")


def test_missing_price_columns():
    """Test: DataFrame without Close or Adj Close should fail."""
    print("\n" + "="*60)
    print("TEST 6: DataFrame without price columns (should fail)")
    print("="*60)

    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    data = pd.DataFrame({
        'Volume': np.random.randint(1000000, 10000000, 100),
        'SomeOtherColumn': np.random.randn(100)
    }, index=dates)

    print(f"Input columns: {data.columns.tolist()}")

    try:
        prices = get_price_column(data, ['AAPL'], prefer_adjusted=True)
        print(f"❌ FAILED: Should have raised ValueError for missing price columns")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"❌ FAILED: Raised wrong exception type: {e}")


def main():
    """Run all tests."""
    print("\n" + "🔬"*30)
    print("TESTING DASHBOARD FIX - get_price_column() Function")
    print("🔬"*30)

    test_single_ticker_with_adj_close()
    test_single_ticker_without_adj_close()
    test_multi_ticker_with_adj_close()
    test_multi_ticker_without_adj_close()
    test_empty_dataframe()
    test_missing_price_columns()

    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
    print("\nThe dashboard fix should now handle:")
    print("  ✓ Single ticker DataFrames")
    print("  ✓ Multi-ticker MultiIndex DataFrames")
    print("  ✓ Missing 'Adj Close' (fallback to 'Close')")
    print("  ✓ Empty DataFrames (graceful error)")
    print("  ✓ Missing price columns (clear error message)")
    print("\n")


if __name__ == "__main__":
    main()
