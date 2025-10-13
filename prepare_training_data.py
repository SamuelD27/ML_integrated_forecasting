#!/usr/bin/env python3
"""
Prepare Training Data for Hybrid Model
=======================================

This script prepares the collected data specifically for training the hybrid deep learning model.
It creates train/validation/test splits with proper temporal ordering and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from utils.advanced_feature_engineering import AdvancedFeatureEngineer, FeatureConfig
from data_fetching import download_data, load_current_session_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataPreparer:
    """Prepares data for model training with all necessary features."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)

        # Initialize feature engineer
        self.feature_engineer = AdvancedFeatureEngineer()

    def prepare_from_existing_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare training data from existing collected data.

        Returns:
            Dictionary with prepared datasets
        """
        logger.info("Loading existing data...")

        # Try to load from multiple sources
        data_sources = []

        # 1. Load from last_fetch
        try:
            prices_df, bundle = load_current_session_data(base_name="last_fetch")
            if prices_df is not None and not prices_df.empty:
                data_sources.append(('last_fetch', prices_df, bundle))
                logger.info(f"Loaded last_fetch data: {prices_df.shape}")
        except Exception as e:
            logger.warning(f"Could not load last_fetch data: {e}")

        # 2. Load from training directory
        training_file = self.training_dir / "training_data.parquet"
        if training_file.exists():
            try:
                training_df = pd.read_parquet(training_file)
                data_sources.append(('training', training_df, {}))
                logger.info(f"Loaded training data: {training_df.shape}")
            except Exception as e:
                logger.warning(f"Could not load training data: {e}")

        if not data_sources:
            logger.warning("No existing data found. Fetching new data...")
            return self.prepare_from_yfinance()

        # Process the best available data source
        source_name, df, metadata = data_sources[0]
        logger.info(f"Using {source_name} as primary data source")

        return self._process_dataframe(df, metadata)

    def prepare_from_yfinance(self, tickers: Optional[List[str]] = None,
                            start_date: str = "2020-01-01") -> Dict[str, pd.DataFrame]:
        """
        Fetch and prepare data directly from yfinance.

        Args:
            tickers: List of tickers to fetch
            start_date: Start date for data

        Returns:
            Dictionary with prepared datasets
        """
        if tickers is None:
            # Default tickers for training
            tickers = [
                "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                "META", "TSLA", "AMD", "INTC", "CRM",
                "ORCL", "QCOM", "TXN", "AVGO", "PANW"
            ]

        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date}")

        # Download data
        prices_df, bundle = download_data(
            tickers=tickers,
            start=start_date,
            save_to_module=True,
            return_bundle=True
        )

        if prices_df is None or prices_df.empty:
            raise ValueError("Failed to fetch data from yfinance")

        return self._process_dataframe(prices_df, bundle.get('meta', {}))

    def _process_dataframe(self, df: pd.DataFrame, metadata: dict) -> Dict[str, pd.DataFrame]:
        """
        Process raw dataframe into training-ready datasets.

        Args:
            df: Raw price dataframe
            metadata: Metadata about the data

        Returns:
            Dictionary with processed datasets
        """
        results = {}

        # Determine if we have multi-ticker data
        if isinstance(df.columns, pd.MultiIndex):
            # Multi-ticker format
            tickers = metadata.get('tickers', [])
            if not tickers:
                # Extract tickers from column names
                tickers = list(set([col[0] for col in df.columns if isinstance(col, tuple)]))

            logger.info(f"Processing {len(tickers)} tickers")

            for ticker in tickers[:5]:  # Process first 5 tickers for speed
                ticker_data = self._extract_ticker_data(df, ticker)
                if ticker_data is not None and not ticker_data.empty:
                    processed = self._process_single_ticker(ticker, ticker_data)
                    if processed is not None:
                        results[ticker] = processed
        else:
            # Single ticker format
            ticker = metadata.get('tickers', ['UNKNOWN'])[0]
            processed = self._process_single_ticker(ticker, df)
            if processed is not None:
                results[ticker] = processed

        # Combine all tickers into training sets
        if results:
            combined = self._create_combined_dataset(results)
            results['combined'] = combined

            # Save processed data
            self._save_processed_data(results)

        return results

    def _extract_ticker_data(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Extract data for a single ticker from multi-ticker dataframe.

        Args:
            df: Multi-ticker dataframe
            ticker: Ticker to extract

        Returns:
            Single ticker dataframe
        """
        try:
            # Extract columns for this ticker
            ticker_cols = [col for col in df.columns if col[0] == ticker]
            if not ticker_cols:
                return None

            ticker_df = df[ticker_cols].copy()

            # Flatten column names
            ticker_df.columns = [col[1] for col in ticker_cols]

            # Standard column names
            column_mapping = {
                'Adj Close': 'close',
                'Close': 'close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume'
            }

            ticker_df = ticker_df.rename(columns=column_mapping)

            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in ticker_df.columns:
                    if col == 'close' and 'Adj Close' in ticker_df.columns:
                        ticker_df['close'] = ticker_df['Adj Close']
                    elif col == 'volume':
                        ticker_df['volume'] = 1000000  # Default volume
                    else:
                        return None

            return ticker_df[required]

        except Exception as e:
            logger.error(f"Error extracting {ticker}: {e}")
            return None

    def _process_single_ticker(self, ticker: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process data for a single ticker with feature engineering.

        Args:
            ticker: Ticker symbol
            df: OHLCV dataframe

        Returns:
            Processed dataframe with features
        """
        try:
            # Clean and prepare data
            df = df.dropna()
            if len(df) < 100:
                logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
                return None

            # Ensure proper column names
            df.columns = df.columns.str.lower()

            # Generate features
            logger.info(f"Generating features for {ticker}")
            features_df = self.feature_engineer.generate_features(
                ticker=ticker,
                df=df,
                save_features=False
            )

            if features_df is None or features_df.empty:
                return None

            # Add labels for supervised learning
            features_df = self._add_labels(features_df, df)

            # Remove any remaining NaN values
            features_df = features_df.dropna()

            logger.info(f"Processed {ticker}: {features_df.shape}")
            return features_df

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return None

    def _add_labels(self, features_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target labels for training.

        Args:
            features_df: Features dataframe
            price_df: Price dataframe

        Returns:
            Features with labels added
        """
        df = features_df.copy()

        # Ensure we have close prices
        if 'close' in price_df.columns:
            close_prices = price_df['close']
        elif 'Close' in price_df.columns:
            close_prices = price_df['Close']
        else:
            close_prices = price_df.iloc[:, -1]  # Use last column

        # Align indices
        close_prices = close_prices.reindex(df.index)

        # Forward returns (what we want to predict)
        df['target_return_1d'] = close_prices.shift(-1) / close_prices - 1
        df['target_return_5d'] = close_prices.shift(-5) / close_prices - 1
        df['target_return_20d'] = close_prices.shift(-20) / close_prices - 1

        # Direction labels (classification targets)
        df['target_direction_1d'] = np.where(df['target_return_1d'] > 0.001, 1,
                                            np.where(df['target_return_1d'] < -0.001, -1, 0))

        # Volatility target (for risk management)
        df['target_volatility_5d'] = close_prices.pct_change().rolling(5).std().shift(-5)

        # Price levels (for regression)
        df['target_price_1d'] = close_prices.shift(-1)
        df['target_price_5d'] = close_prices.shift(-5)

        return df

    def _create_combined_dataset(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Combine multiple tickers into train/val/test sets.

        Args:
            ticker_data: Dictionary of processed dataframes by ticker

        Returns:
            Dictionary with train, validation, and test sets
        """
        all_data = []

        for ticker, df in ticker_data.items():
            if ticker != 'combined':  # Skip if already combined
                df_copy = df.copy()
                df_copy['ticker'] = ticker
                all_data.append(df_copy)

        if not all_data:
            return {}

        # Combine all tickers
        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_index()

        # Create temporal splits (no shuffling to avoid lookahead bias)
        total_samples = len(combined_df)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)

        # Split by index (time-based)
        unique_dates = combined_df.index.unique().sort_values()
        train_dates = unique_dates[:int(len(unique_dates) * 0.7)]
        val_dates = unique_dates[int(len(unique_dates) * 0.7):int(len(unique_dates) * 0.85)]
        test_dates = unique_dates[int(len(unique_dates) * 0.85):]

        train_df = combined_df.loc[combined_df.index.isin(train_dates)]
        val_df = combined_df.loc[combined_df.index.isin(val_dates)]
        test_df = combined_df.loc[combined_df.index.isin(test_dates)]

        logger.info(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }

    def _save_processed_data(self, results: Dict[str, any]):
        """
        Save processed data to disk.

        Args:
            results: Dictionary of processed dataframes
        """
        # Save individual ticker data
        for ticker, data in results.items():
            if ticker == 'combined':
                # Save train/val/test splits
                for split_name, split_df in data.items():
                    file_path = self.training_dir / f"{split_name}_data.parquet"
                    split_df.to_parquet(file_path)
                    logger.info(f"Saved {split_name} data: {file_path}")

                    # Also save as CSV for inspection
                    csv_path = self.training_dir / f"{split_name}_data.csv"
                    split_df.head(1000).to_csv(csv_path)  # Save first 1000 rows as CSV
            else:
                # Save individual ticker data
                file_path = self.training_dir / f"{ticker}_features.parquet"
                data.to_parquet(file_path)

        # Save metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'tickers': list(results.keys()),
            'features': list(results[list(results.keys())[0]].columns) if results else [],
            'splits': ['train', 'validation', 'test'] if 'combined' in results else [],
            'feature_count': len(results[list(results.keys())[0]].columns) if results else 0
        }

        metadata_path = self.training_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata: {metadata_path}")


def main():
    """Main function to prepare training data."""
    print("\n" + "="*60)
    print("PREPARING TRAINING DATA FOR HYBRID MODEL")
    print("="*60)

    preparer = TrainingDataPreparer()

    # Try to use existing data first
    results = preparer.prepare_from_existing_data()

    if not results:
        print("\n⚠️  No existing data found. Fetching from Yahoo Finance...")
        results = preparer.prepare_from_yfinance()

    if results and 'combined' in results:
        print("\n✅ Training data prepared successfully!")
        print("\n📊 Data Statistics:")
        print("-" * 40)

        for split_name, split_df in results['combined'].items():
            print(f"{split_name.capitalize():12} {len(split_df):7,} samples")

        print("\n📁 Data saved to: data/training/")
        print("\n🚀 Ready to train the model!")
        print("   Run: python training/train_hybrid.py")
    else:
        print("\n❌ Failed to prepare training data")
        print("   Please check the logs for errors")


if __name__ == "__main__":
    main()