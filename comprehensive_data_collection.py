#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

class DataConfig:
    """Configuration for data collection."""

    # Tickers to collect (mix of stocks, ETFs, and indices)
    STOCKS = [
        # Tech giants
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
        # Semiconductors
        "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "MCHP", "ADI",
        # Software
        "CRM", "ORCL", "ADBE", "NOW", "PANW", "FTNT",
        # Hardware/Equipment
        "AMAT", "LRCX", "KLAC", "ASML",
        # Financial
        "JPM", "BAC", "GS", "MS", "V", "MA", "PYPL",
        # Energy
        "XOM", "CVX", "COP",
        # Healthcare
        "UNH", "JNJ", "PFE", "ABBV", "LLY"
    ]

    # Market indices and ETFs
    INDICES = [
        "^GSPC",  # S&P 500
        "^DJI",   # Dow Jones
        "^IXIC",  # NASDAQ
        "^VIX",   # Volatility Index
        "^TNX",   # 10-Year Treasury
        "^TYX",   # 30-Year Treasury
    ]

    ETFS = [
        "SPY", "QQQ", "IWM",  # Market
        "XLK", "XLF", "XLE",  # Sectors
        "VGT", "SOXX", "SMH",  # Tech focused
        "TLT", "IEF", "SHY",  # Bonds
        "GLD", "SLV",  # Commodities
        "UUP",  # Dollar
    ]

    # Date ranges for different purposes
    TRAINING_START = "2015-01-01"  # Long history for training
    BACKTEST_START = "2020-01-01"  # Recent history for backtesting
    VALIDATION_START = "2023-01-01"  # Most recent for validation

    # Data storage paths
    DATA_DIR = Path("data")
    TRAINING_DIR = DATA_DIR / "training"
    OPTIONS_DIR = DATA_DIR / "options"
    FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
    MARKET_DIR = DATA_DIR / "market"

    # Options configuration
    OPTIONS_EXPIRIES = 4  # Number of expiry dates to fetch
    OPTIONS_STRIKES = 10  # Number of strikes around ATM

    # API rate limiting
    RATE_LIMIT_DELAY = 0.5  # Seconds between API calls
    MAX_RETRIES = 3
    CHUNK_SIZE = 10  # Process tickers in chunks


# ==================== DATA FETCHERS ====================

class StockDataFetcher:
    """Fetches historical stock data."""

    @staticmethod
    def fetch_ohlcv(ticker: str, start: str, end: str = None,
                    interval: str = "1d") -> pd.DataFrame:
        """
        Fetch OHLCV data for a ticker.

        Args:
            ticker: Stock symbol
            start: Start date
            end: End date (default: today)
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start,
                end=end or datetime.now().strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,  # Adjust for splits and dividends
                actions=True  # Include dividends and splits
            )

            if not data.empty:
                data['ticker'] = ticker
                logger.info(f"Fetched {len(data)} rows for {ticker}")
            else:
                logger.warning(f"No data found for {ticker}")

            return data

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_multi_timeframe(ticker: str, start: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data at multiple timeframes.

        Args:
            ticker: Stock symbol
            start: Start date

        Returns:
            Dictionary of DataFrames by timeframe
        """
        timeframes = {
            "1d": "daily",
            "1wk": "weekly",
            "1mo": "monthly"
        }

        data = {}
        for interval, name in timeframes.items():
            df = StockDataFetcher.fetch_ohlcv(ticker, start, interval=interval)
            if not df.empty:
                data[name] = df
            time.sleep(DataConfig.RATE_LIMIT_DELAY)

        # For intraday, fetch last 60 days only (yfinance limitation)
        intraday_start = (datetime.now() - timedelta(days=59)).strftime("%Y-%m-%d")
        for interval in ["1h", "30m", "15m"]:
            df = StockDataFetcher.fetch_ohlcv(ticker, intraday_start, interval=interval)
            if not df.empty:
                data[interval] = df
            time.sleep(DataConfig.RATE_LIMIT_DELAY)

        return data


class OptionsDataFetcher:
    """Fetches options data including chains and Greeks."""

    @staticmethod
    def fetch_options_chain(ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch current options chain.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with calls, puts, and metadata
        """
        try:
            stock = yf.Ticker(ticker)

            # Get available expiration dates
            expirations = stock.options[:DataConfig.OPTIONS_EXPIRIES] if stock.options else []

            if not expirations:
                logger.warning(f"No options data available for {ticker}")
                return {}

            all_calls = []
            all_puts = []

            for exp_date in expirations:
                opt_chain = stock.option_chain(exp_date)

                # Add expiration date to the data
                calls = opt_chain.calls.copy()
                puts = opt_chain.puts.copy()

                calls['expiration'] = exp_date
                puts['expiration'] = exp_date

                # Calculate moneyness
                current_price = stock.info.get('currentPrice', stock.info.get('regularMarketPrice', 0))
                if current_price:
                    calls['moneyness'] = calls['strike'] / current_price
                    puts['moneyness'] = puts['strike'] / current_price

                all_calls.append(calls)
                all_puts.append(puts)

            result = {
                'calls': pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame(),
                'puts': pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame(),
                'metadata': {
                    'ticker': ticker,
                    'current_price': current_price,
                    'fetch_time': datetime.now().isoformat(),
                    'expirations': expirations
                }
            }

            logger.info(f"Fetched options data for {ticker}: {len(all_calls)} expirations")
            return result

        except Exception as e:
            logger.error(f"Error fetching options for {ticker}: {e}")
            return {}

    @staticmethod
    def calculate_greeks(options_df: pd.DataFrame, risk_free_rate: float = 0.05) -> pd.DataFrame:
        """
        Calculate Greeks for options (simplified Black-Scholes).

        Args:
            options_df: DataFrame with options data
            risk_free_rate: Risk-free interest rate

        Returns:
            DataFrame with Greeks added
        """
        from scipy.stats import norm

        if options_df.empty or 'strike' not in options_df.columns:
            return options_df

        # This is a simplified calculation - in production use proper options library
        df = options_df.copy()

        # Calculate time to expiration
        if 'expiration' in df.columns:
            df['expiration_date'] = pd.to_datetime(df['expiration'])
            df['days_to_expiry'] = (df['expiration_date'] - pd.Timestamp.now()).dt.days
            df['T'] = df['days_to_expiry'] / 365.0

        # Use implied volatility if available, else use a default
        if 'impliedVolatility' not in df.columns:
            df['impliedVolatility'] = 0.25  # Default 25% volatility

        return df


class FundamentalDataFetcher:
    """Fetches fundamental data and financial statements."""

    @staticmethod
    def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with fundamental data
        """
        try:
            stock = yf.Ticker(ticker)

            fundamentals = {
                'info': stock.info,
                'financials': stock.financials.to_dict() if hasattr(stock, 'financials') and stock.financials is not None else {},
                'balance_sheet': stock.balance_sheet.to_dict() if hasattr(stock, 'balance_sheet') and stock.balance_sheet is not None else {},
                'cashflow': stock.cashflow.to_dict() if hasattr(stock, 'cashflow') and stock.cashflow is not None else {},
                'earnings': stock.earnings.to_dict() if hasattr(stock, 'earnings') and stock.earnings is not None else {},
                'recommendations': stock.recommendations.to_dict() if hasattr(stock, 'recommendations') and stock.recommendations is not None else {},
            }

            # Extract key metrics
            info = stock.info
            key_metrics = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE', info.get('trailingPE')),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'profit_margin': info.get('profitMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
            }

            fundamentals['key_metrics'] = key_metrics

            logger.info(f"Fetched fundamentals for {ticker}")
            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {}


class MarketDataFetcher:
    """Fetches market-wide data including indices, bonds, forex."""

    @staticmethod
    def fetch_market_indicators() -> pd.DataFrame:
        """
        Fetch key market indicators.

        Returns:
            DataFrame with market indicators
        """
        indicators = {}

        # Fetch major indices
        for symbol in DataConfig.INDICES:
            try:
                data = yf.Ticker(symbol).history(period="1mo")
                if not data.empty:
                    indicators[symbol] = {
                        'last_price': data['Close'].iloc[-1],
                        '1d_change': data['Close'].pct_change().iloc[-1],
                        '5d_change': data['Close'].pct_change(5).iloc[-1],
                        '20d_change': data['Close'].pct_change(20).iloc[-1],
                        'volume': data['Volume'].iloc[-1] if 'Volume' in data else None
                    }
                time.sleep(DataConfig.RATE_LIMIT_DELAY)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")

        return pd.DataFrame(indicators).T

    @staticmethod
    def fetch_sector_performance() -> pd.DataFrame:
        """
        Fetch sector ETF performance.

        Returns:
            DataFrame with sector performance
        """
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLC': 'Communication'
        }

        performance = {}
        for etf, sector in sector_etfs.items():
            try:
                data = yf.Ticker(etf).history(period="1mo")
                if not data.empty:
                    performance[sector] = {
                        'etf': etf,
                        '1d_return': data['Close'].pct_change().iloc[-1],
                        '5d_return': data['Close'].pct_change(5).iloc[-1],
                        '20d_return': data['Close'].pct_change(20).iloc[-1],
                        'volatility': data['Close'].pct_change().std() * np.sqrt(252)
                    }
                time.sleep(DataConfig.RATE_LIMIT_DELAY)
            except Exception as e:
                logger.error(f"Error fetching {etf}: {e}")

        return pd.DataFrame(performance).T


# ==================== DATA PROCESSOR ====================

class DataProcessor:
    """Processes and prepares data for ML training."""

    @staticmethod
    def create_training_dataset(tickers: List[str], start_date: str) -> pd.DataFrame:
        """
        Create comprehensive training dataset.

        Args:
            tickers: List of tickers
            start_date: Start date for data

        Returns:
            Combined DataFrame ready for training
        """
        all_data = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            for ticker in tickers:
                future = executor.submit(
                    DataProcessor._process_single_ticker,
                    ticker,
                    start_date
                )
                futures.append((ticker, future))

            for ticker, future in tqdm(futures, desc="Processing tickers"):
                try:
                    result = future.result(timeout=60)
                    if result is not None:
                        all_data.append(result)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Created dataset with {len(combined_df)} rows from {len(all_data)} tickers")
            return combined_df

        return pd.DataFrame()

    @staticmethod
    def _process_single_ticker(ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        """
        Process data for a single ticker.

        Args:
            ticker: Stock symbol
            start_date: Start date

        Returns:
            Processed DataFrame or None
        """
        try:
            # Fetch daily data
            daily_data = StockDataFetcher.fetch_ohlcv(ticker, start_date)

            if daily_data.empty:
                return None

            # Add technical indicators
            df = DataProcessor._add_technical_indicators(daily_data)

            # Add ticker column
            df['ticker'] = ticker

            # Add date features
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter

            return df

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return None

    @staticmethod
    def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicators added
        """
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # Volume indicators
        df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']

        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)

        # RSI
        df['rsi'] = DataProcessor._calculate_rsi(df['Close'])

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# ==================== MAIN COLLECTION SCRIPT ====================

def collect_all_data():
    """
    Main function to collect all data needed for training.
    """
    # Create directories
    for dir_path in [DataConfig.TRAINING_DIR, DataConfig.OPTIONS_DIR,
                     DataConfig.FUNDAMENTALS_DIR, DataConfig.MARKET_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*50)
    logger.info("Starting comprehensive data collection")
    logger.info("="*50)

    # 1. Collect historical price data
    logger.info("\n1. Collecting historical price data...")
    all_tickers = DataConfig.STOCKS + DataConfig.ETFS

    training_data = DataProcessor.create_training_dataset(
        all_tickers[:10],  # Start with first 10 for testing
        DataConfig.TRAINING_START
    )

    if not training_data.empty:
        # Save training data
        training_file = DataConfig.TRAINING_DIR / "training_data.parquet"
        training_data.to_parquet(training_file)
        logger.info(f"Saved training data to {training_file}")

        # Save as CSV for easy inspection
        csv_file = DataConfig.TRAINING_DIR / "training_data.csv"
        training_data.to_csv(csv_file)
        logger.info(f"Also saved as CSV to {csv_file}")

    # 2. Collect options data
    logger.info("\n2. Collecting options data...")
    options_data = {}

    for ticker in DataConfig.STOCKS[:5]:  # Top 5 stocks for options
        options = OptionsDataFetcher.fetch_options_chain(ticker)
        if options:
            options_data[ticker] = options

            # Save options data
            if 'calls' in options and not options['calls'].empty:
                calls_file = DataConfig.OPTIONS_DIR / f"{ticker}_calls.parquet"
                options['calls'].to_parquet(calls_file)

            if 'puts' in options and not options['puts'].empty:
                puts_file = DataConfig.OPTIONS_DIR / f"{ticker}_puts.parquet"
                options['puts'].to_parquet(puts_file)

        time.sleep(DataConfig.RATE_LIMIT_DELAY)

    # 3. Collect fundamental data
    logger.info("\n3. Collecting fundamental data...")
    fundamentals = {}

    for ticker in DataConfig.STOCKS[:10]:
        fund_data = FundamentalDataFetcher.fetch_fundamentals(ticker)
        if fund_data:
            fundamentals[ticker] = fund_data['key_metrics']
        time.sleep(DataConfig.RATE_LIMIT_DELAY)

    if fundamentals:
        fund_df = pd.DataFrame(fundamentals).T
        fund_file = DataConfig.FUNDAMENTALS_DIR / "fundamentals.parquet"
        fund_df.to_parquet(fund_file)
        logger.info(f"Saved fundamentals to {fund_file}")

    # 4. Collect market indicators
    logger.info("\n4. Collecting market indicators...")
    market_indicators = MarketDataFetcher.fetch_market_indicators()

    if not market_indicators.empty:
        market_file = DataConfig.MARKET_DIR / "market_indicators.parquet"
        market_indicators.to_parquet(market_file)
        logger.info(f"Saved market indicators to {market_file}")

    # 5. Collect sector performance
    logger.info("\n5. Collecting sector performance...")
    sector_performance = MarketDataFetcher.fetch_sector_performance()

    if not sector_performance.empty:
        sector_file = DataConfig.MARKET_DIR / "sector_performance.parquet"
        sector_performance.to_parquet(sector_file)
        logger.info(f"Saved sector performance to {sector_file}")

    # Create metadata file
    metadata = {
        'collection_date': datetime.now().isoformat(),
        'tickers': all_tickers,
        'start_date': DataConfig.TRAINING_START,
        'data_types': ['prices', 'options', 'fundamentals', 'market_indicators', 'sectors'],
        'files_created': {
            'training_data': str(DataConfig.TRAINING_DIR / "training_data.parquet"),
            'options': str(DataConfig.OPTIONS_DIR),
            'fundamentals': str(DataConfig.FUNDAMENTALS_DIR / "fundamentals.parquet"),
            'market': str(DataConfig.MARKET_DIR)
        }
    }

    metadata_file = DataConfig.DATA_DIR / "collection_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "="*50)
    logger.info("Data collection completed successfully!")
    logger.info(f"Metadata saved to {metadata_file}")
    logger.info("="*50)

    # Print summary
    print("\n📊 DATA COLLECTION SUMMARY")
    print("="*40)
    print(f"✅ Historical data: {len(training_data)} rows")
    print(f"✅ Options data: {len(options_data)} tickers")
    print(f"✅ Fundamental data: {len(fundamentals)} tickers")
    print(f"✅ Market indicators: {len(market_indicators)} indices")
    print(f"✅ Sector performance: {len(sector_performance)} sectors")
    print("\n📁 Data saved to:")
    print(f"  - {DataConfig.DATA_DIR}")
    print("\n🚀 Ready for model training!")


if __name__ == "__main__":
    collect_all_data()