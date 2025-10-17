"""
Large-Scale Data Fetching for RunPod Training
==============================================
Fetch comprehensive historical data for all major stocks.
Optimized for high-performance GPU training.

Run: python runpod_setup/fetch_training_data_large.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TICKER UNIVERSE - COMPREHENSIVE SET
# ============================================================================

# S&P 500 Tech Giants
MEGA_CAP_TECH = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
    'ORCL', 'ADBE', 'CRM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN',
    'AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ADI'
]

# Large Cap Diversified
LARGE_CAP = [
    'JPM', 'V', 'MA', 'WMT', 'JNJ', 'UNH', 'HD', 'PG', 'BAC', 'XOM',
    'CVX', 'ABBV', 'KO', 'PEP', 'COST', 'MRK', 'LLY', 'TMO', 'ABT',
    'NKE', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'ADBE', 'PFE', 'CSCO'
]

# Mid Cap Growth
MID_CAP = [
    'SQ', 'ROKU', 'DKNG', 'PLTR', 'SNOW', 'NET', 'DDOG', 'CRWD',
    'ZM', 'DOCU', 'TWLO', 'OKTA', 'MDB', 'ESTC', 'ZS', 'FTNT'
]

# ETFs for Market Factors
ETFS = [
    'SPY',   # S&P 500
    'QQQ',   # NASDAQ 100
    'IWM',   # Russell 2000
    'DIA',   # Dow Jones
    'EFA',   # International Developed
    'EEM',   # Emerging Markets
    'TLT',   # 20+ Year Treasury
    'GLD',   # Gold
    'USO',   # Oil
    'VIX',   # Volatility (if available)
]

# Combine all
ALL_TICKERS = list(set(MEGA_CAP_TECH + LARGE_CAP + MID_CAP + ETFS))

logger.info(f"Total tickers to fetch: {len(ALL_TICKERS)}")


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_ticker_data(ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch data for a single ticker with error handling.

    Args:
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1d, 1h, etc.)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True  # Adjust for splits/dividends
        )

        if data.empty:
            logger.warning(f"No data for {ticker}")
            return None

        # Add ticker column
        data['ticker'] = ticker

        # Reset index to have date as column
        data = data.reset_index()

        return data

    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return None


def fetch_all_tickers(
    tickers: list,
    start_date: str,
    end_date: str,
    interval: str = '1d',
    max_workers: int = 10
) -> pd.DataFrame:
    """
    Fetch data for all tickers in parallel.

    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        interval: Data interval
        max_workers: Number of parallel workers

    Returns:
        Combined DataFrame with all ticker data
    """
    all_data = []

    logger.info(f"Fetching {len(tickers)} tickers with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(fetch_ticker_data, ticker, start_date, end_date, interval): ticker
            for ticker in tickers
        }

        # Process results with progress bar
        with tqdm(total=len(tickers), desc="Fetching data") as pbar:
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        all_data.append(data)
                        pbar.set_postfix({'ticker': ticker, 'rows': len(data)})
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                finally:
                    pbar.update(1)

    if not all_data:
        raise ValueError("No data fetched for any ticker!")

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    logger.info(f"Total rows fetched: {len(combined):,}")
    logger.info(f"Tickers with data: {combined['ticker'].nunique()}")
    logger.info(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")

    return combined


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators for better ML features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added features
    """
    logger.info("Adding technical features...")

    # Sort by ticker and date
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    # Group by ticker for calculations
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_indices = df[mask].index
        ticker_data = df.loc[ticker_indices, 'Close'].copy()

        # Returns
        df.loc[ticker_indices, 'returns'] = ticker_data.pct_change().values
        df.loc[ticker_indices, 'log_returns'] = np.log(ticker_data / ticker_data.shift(1)).values

        # Moving averages
        df.loc[ticker_indices, 'sma_5'] = ticker_data.rolling(5).mean().values
        df.loc[ticker_indices, 'sma_10'] = ticker_data.rolling(10).mean().values
        df.loc[ticker_indices, 'sma_20'] = ticker_data.rolling(20).mean().values
        df.loc[ticker_indices, 'sma_50'] = ticker_data.rolling(50).mean().values

        # Volatility
        returns = ticker_data.pct_change()
        df.loc[ticker_indices, 'volatility_5'] = returns.rolling(5).std().values
        df.loc[ticker_indices, 'volatility_20'] = returns.rolling(20).std().values

        # Volume features
        volume = df.loc[ticker_indices, 'Volume'].copy()
        volume_sma = volume.rolling(20).mean()
        df.loc[ticker_indices, 'volume_sma_20'] = volume_sma.values
        df.loc[ticker_indices, 'volume_ratio'] = (volume / volume_sma).values

        # RSI
        delta = ticker_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df.loc[ticker_indices, 'rsi'] = rsi.values

    logger.info(f"Added {df.shape[1] - 7} technical features")  # Subtract original columns

    return df


def save_data(df: pd.DataFrame, output_dir: Path):
    """
    Save data in multiple formats for flexibility.

    Args:
        df: DataFrame to save
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as parquet (efficient)
    parquet_path = output_dir / 'training_data.parquet'
    df.to_parquet(parquet_path, compression='snappy', index=False)
    logger.info(f"Saved parquet: {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    # Save metadata
    metadata = {
        'n_rows': len(df),
        'n_tickers': df['ticker'].nunique(),
        'tickers': sorted(df['ticker'].unique().tolist()),
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d')
        },
        'columns': df.columns.tolist(),
        'missing_data': df.isnull().sum().to_dict(),
        'generated_at': datetime.now().isoformat()
    }

    import json
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved metadata: {metadata_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total rows: {len(df):,}")
    print(f"Total tickers: {df['ticker'].nunique()}")
    print(f"Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nMissing data by column:")
    for col, missing in metadata['missing_data'].items():
        if missing > 0:
            print(f"  {col}: {missing:,} ({missing/len(df)*100:.1f}%)")
    print("=" * 80)


def main():
    """Main data fetching pipeline."""
    print("\n" + "=" * 80)
    print("LARGE-SCALE DATA FETCHING FOR ML TRAINING")
    print("=" * 80)

    # Configuration
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')  # 10 years
    INTERVAL = '1d'  # Daily data
    OUTPUT_DIR = Path('data/training')

    print(f"\nConfiguration:")
    print(f"  Start date: {START_DATE}")
    print(f"  End date: {END_DATE}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Tickers: {len(ALL_TICKERS)}")
    print(f"  Output: {OUTPUT_DIR}")

    # Fetch data
    print("\n" + "-" * 80)
    print("STEP 1: FETCHING RAW DATA")
    print("-" * 80)

    df = fetch_all_tickers(
        ALL_TICKERS,
        START_DATE,
        END_DATE,
        INTERVAL,
        max_workers=20  # Parallel fetching
    )

    # Add features
    print("\n" + "-" * 80)
    print("STEP 2: ADDING TECHNICAL FEATURES")
    print("-" * 80)

    df = add_technical_features(df)

    # Drop NaN rows (from feature calculations)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    logger.info(f"Dropped {dropped_rows:,} rows with NaN values ({dropped_rows/initial_rows*100:.1f}%)")

    # Save
    print("\n" + "-" * 80)
    print("STEP 3: SAVING DATA")
    print("-" * 80)

    save_data(df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("✅ DATA FETCHING COMPLETE!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Upload to RunPod: scp -r {OUTPUT_DIR} runpod:/workspace/data/")
    print(f"  2. Start training: python runpod_setup/train_runpod.py")


if __name__ == '__main__':
    main()
