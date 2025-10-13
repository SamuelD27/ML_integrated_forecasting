from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import json
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import RobustScaler
import warnings
import pickle

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_LOOKBACK_WINDOWS = [1, 5, 10, 20, 60, 120]
DEFAULT_MA_PERIODS = [5, 10, 20, 50, 200]
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_PARAMS = (12, 26, 9)
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    lookback_windows: List[int] = None
    ma_periods: List[int] = None
    rsi_period: int = DEFAULT_RSI_PERIOD
    macd_params: Tuple[int, int, int] = DEFAULT_MACD_PARAMS
    bb_period: int = DEFAULT_BB_PERIOD
    bb_std: float = DEFAULT_BB_STD
    compute_fft: bool = True
    fft_components: int = 10
    compute_microstructure: bool = True
    compute_volatility: bool = True
    compute_cross_sectional: bool = True

    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = DEFAULT_LOOKBACK_WINDOWS
        if self.ma_periods is None:
            self.ma_periods = DEFAULT_MA_PERIODS


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for financial time series.

    This class creates comprehensive feature sets from OHLCV data,
    designed specifically for hybrid deep learning models.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature engineer.

        Args:
            config: Feature configuration. If None, uses defaults.
        """
        self.config = config or FeatureConfig()
        self.scalers: Dict[str, RobustScaler] = {}
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}

    def compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multi-timeframe price-based features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=df.index)

        # Use adjusted close if available, otherwise close
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        price = df[price_col]

        # Multi-timeframe returns
        for window in self.config.lookback_windows:
            # Simple returns
            features[f'return_{window}d'] = price.pct_change(window)

            # Log returns (more stable for modeling)
            features[f'log_return_{window}d'] = np.log(price / price.shift(window))

            # Normalized returns (return / volatility)
            rolling_vol = price.pct_change().rolling(window).std()
            features[f'norm_return_{window}d'] = features[f'return_{window}d'] / (rolling_vol + 1e-8)

        # Price relative to moving averages
        for period in self.config.ma_periods:
            ma = price.rolling(period).mean()
            features[f'price_to_ma_{period}'] = (price - ma) / ma

        # High-Low spread (intraday volatility proxy)
        if 'High' in df.columns and 'Low' in df.columns:
            features['hl_spread'] = (df['High'] - df['Low']) / df['Low']
            features['hl_spread_ma20'] = features['hl_spread'].rolling(20).mean()

        # Close position in daily range
        if 'High' in df.columns and 'Low' in df.columns:
            daily_range = df['High'] - df['Low']
            features['close_position'] = (df[price_col] - df['Low']) / (daily_range + 1e-8)

        # Gap features (overnight returns)
        if 'Open' in df.columns:
            features['overnight_return'] = (df['Open'] - df[price_col].shift(1)) / df[price_col].shift(1)
            features['intraday_return'] = (df[price_col] - df['Open']) / df['Open']

        return features

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators commonly used in trading.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with technical indicators
        """
        features = pd.DataFrame(index=df.index)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        price = df[price_col]

        # RSI (Relative Strength Index)
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.rsi_period).mean()
        rs = gain / (loss + 1e-8)
        features[f'rsi_{self.config.rsi_period}'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = price.ewm(span=self.config.macd_params[0], adjust=False).mean()
        exp2 = price.ewm(span=self.config.macd_params[1], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.config.macd_params[2], adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal

        # Bollinger Bands
        ma = price.rolling(self.config.bb_period).mean()
        std = price.rolling(self.config.bb_period).std()
        features['bb_upper'] = ma + (std * self.config.bb_std)
        features['bb_lower'] = ma - (std * self.config.bb_std)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (price - features['bb_lower']) / (features['bb_width'] + 1e-8)

        # ATR (Average True Range)
        if 'High' in df.columns and 'Low' in df.columns:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - price.shift())
            low_close = np.abs(df['Low'] - price.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            features['atr_14'] = true_range.rolling(14).mean()
            features['atr_normalized'] = features['atr_14'] / price

        # Volume indicators
        if 'Volume' in df.columns:
            # On-Balance Volume (OBV)
            obv = (np.sign(price.diff()) * df['Volume']).cumsum()
            features['obv'] = obv
            features['obv_ma20'] = obv.rolling(20).mean()

            # Money Flow Index (MFI)
            typical_price = (df['High'] + df['Low'] + price) / 3 if 'High' in df.columns else price
            raw_money_flow = typical_price * df['Volume']

            money_flow_pos = raw_money_flow.where(typical_price > typical_price.shift(), 0)
            money_flow_neg = raw_money_flow.where(typical_price < typical_price.shift(), 0)

            mf_pos_14 = money_flow_pos.rolling(14).sum()
            mf_neg_14 = money_flow_neg.rolling(14).sum()

            mf_ratio = mf_pos_14 / (mf_neg_14 + 1e-8)
            features['mfi'] = 100 - (100 / (1 + mf_ratio))

            # Volume Rate of Change
            features['volume_roc'] = df['Volume'].pct_change(10)

            # Price-Volume Trend (PVT)
            features['pvt'] = ((price - price.shift()) / price.shift() * df['Volume']).cumsum()

        # Stochastic Oscillator
        if 'High' in df.columns and 'Low' in df.columns:
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            features['stochastic_k'] = 100 * ((price - low_14) / (high_14 - low_14 + 1e-8))
            features['stochastic_d'] = features['stochastic_k'].rolling(3).mean()

        return features

    def compute_frequency_domain_features(self, df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        Compute frequency domain features using FFT for cyclical pattern detection.

        Args:
            df: DataFrame with price data
            n_components: Number of FFT components to extract

        Returns:
            DataFrame with frequency domain features
        """
        features = pd.DataFrame(index=df.index)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        price = df[price_col].fillna(method='ffill')

        # Detrend the price series
        detrended = signal.detrend(price.dropna())
        detrended_series = pd.Series(detrended, index=price.dropna().index)

        # Apply FFT on rolling windows
        window_size = 60  # 60-day window for FFT

        for i in range(n_components):
            fft_component = pd.Series(index=df.index, dtype=float)

            for end_idx in range(window_size, len(price)):
                window_data = price.iloc[end_idx-window_size:end_idx].values

                # Remove trend from window
                window_detrended = signal.detrend(window_data)

                # Apply FFT
                fft_vals = np.fft.fft(window_detrended)
                fft_freq = np.fft.fftfreq(window_size)

                # Get magnitude of the i-th frequency component
                if i < len(fft_vals) // 2:  # Only use positive frequencies
                    fft_component.iloc[end_idx] = np.abs(fft_vals[i])

            features[f'fft_component_{i}'] = fft_component

        # Dominant frequency and its power
        dominant_freq = pd.Series(index=df.index, dtype=float)
        dominant_power = pd.Series(index=df.index, dtype=float)

        for end_idx in range(window_size, len(price)):
            window_data = price.iloc[end_idx-window_size:end_idx].values
            window_detrended = signal.detrend(window_data)

            fft_vals = np.fft.fft(window_detrended)
            fft_freq = np.fft.fftfreq(window_size)

            # Get positive frequencies only
            pos_freq_idx = fft_freq > 0
            pos_freqs = fft_freq[pos_freq_idx]
            pos_powers = np.abs(fft_vals[pos_freq_idx])

            if len(pos_powers) > 0:
                max_idx = np.argmax(pos_powers)
                dominant_freq.iloc[end_idx] = pos_freqs[max_idx]
                dominant_power.iloc[end_idx] = pos_powers[max_idx]

        features['fft_dominant_freq'] = dominant_freq
        features['fft_dominant_power'] = dominant_power

        # Spectral entropy (measure of signal complexity)
        spectral_entropy = pd.Series(index=df.index, dtype=float)

        for end_idx in range(window_size, len(price)):
            window_data = price.iloc[end_idx-window_size:end_idx].values
            window_detrended = signal.detrend(window_data)

            fft_vals = np.fft.fft(window_detrended)
            power_spectrum = np.abs(fft_vals) ** 2

            # Normalize to get probability distribution
            power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-8)

            # Calculate entropy
            entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-8))
            spectral_entropy.iloc[end_idx] = entropy

        features['spectral_entropy'] = spectral_entropy

        return features

    def compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market microstructure features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=df.index)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        # Bid-Ask Spread Proxy (Roll's measure)
        if 'Close' in df.columns:
            returns = df[price_col].pct_change()
            autocorr = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))

            # Roll's spread estimate
            spread_estimate = 2 * np.sqrt(np.maximum(-autocorr, 0))
            features['roll_spread'] = spread_estimate

        # Kyle's Lambda (price impact measure)
        if 'Volume' in df.columns:
            # Signed volume (volume * sign of return)
            signed_volume = df['Volume'] * np.sign(df[price_col].pct_change())

            # Rolling regression of price changes on signed volume
            window = 20
            kyle_lambda = pd.Series(index=df.index, dtype=float)

            for i in range(window, len(df)):
                y = df[price_col].iloc[i-window:i].pct_change().dropna()
                x = signed_volume.iloc[i-window:i][y.index]

                if len(y) > 10 and x.std() > 0:
                    slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-8)
                    kyle_lambda.iloc[i] = slope

            features['kyle_lambda'] = kyle_lambda

        # Amihud Illiquidity Measure
        if 'Volume' in df.columns:
            returns_abs = np.abs(df[price_col].pct_change())
            dollar_volume = df['Volume'] * df[price_col]

            amihud = returns_abs / (dollar_volume + 1e-8)
            features['amihud_illiquidity'] = amihud.rolling(20).mean()

        # Volume-Price Divergence
        if 'Volume' in df.columns:
            # Correlation between volume and absolute returns
            vol_price_corr = df['Volume'].rolling(20).corr(np.abs(df[price_col].pct_change()))
            features['vol_price_correlation'] = vol_price_corr

            # Volume surprise (current volume vs average)
            vol_ma = df['Volume'].rolling(20).mean()
            vol_std = df['Volume'].rolling(20).std()
            features['volume_zscore'] = (df['Volume'] - vol_ma) / (vol_std + 1e-8)

        # Order Flow Imbalance (using volume as proxy)
        if 'Volume' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            # Estimate buy volume (close near high) and sell volume (close near low)
            close_position = (df[price_col] - df['Low']) / (df['High'] - df['Low'] + 1e-8)

            buy_volume = df['Volume'] * close_position
            sell_volume = df['Volume'] * (1 - close_position)

            features['order_flow_imbalance'] = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-8)
            features['order_flow_imbalance_ma5'] = features['order_flow_imbalance'].rolling(5).mean()

        return features

    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute various volatility estimators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=df.index)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        # Simple historical volatility
        returns = df[price_col].pct_change()
        for window in [5, 10, 20, 60]:
            features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

        if 'High' in df.columns and 'Low' in df.columns:
            # Parkinson volatility estimator
            hl_ratio = np.log(df['High'] / df['Low'])
            for window in [5, 10, 20]:
                parkinson = np.sqrt(1 / (4 * np.log(2)) * (hl_ratio ** 2).rolling(window).mean()) * np.sqrt(252)
                features[f'parkinson_vol_{window}d'] = parkinson

            # Garman-Klass volatility estimator
            if 'Open' in df.columns:
                a = 0.5 * np.log(df['High'] / df['Low']) ** 2
                b = (2 * np.log(2) - 1) * np.log(df[price_col] / df['Open']) ** 2

                for window in [5, 10, 20]:
                    gk = np.sqrt((a - b).rolling(window).mean()) * np.sqrt(252)
                    features[f'garman_klass_vol_{window}d'] = gk

                # Rogers-Satchell volatility estimator
                rs_component = (np.log(df['High'] / df[price_col]) *
                               np.log(df['High'] / df['Open']) +
                               np.log(df['Low'] / df[price_col]) *
                               np.log(df['Low'] / df['Open']))

                for window in [5, 10, 20]:
                    rs = np.sqrt(rs_component.rolling(window).mean()) * np.sqrt(252)
                    features[f'rogers_satchell_vol_{window}d'] = rs

        # Volatility of volatility
        vol_20d = returns.rolling(20).std()
        features['vol_of_vol'] = vol_20d.rolling(20).std()

        # Volatility regime (high/low vol environment)
        vol_median = vol_20d.rolling(60).median()
        features['vol_regime'] = (vol_20d > vol_median).astype(int)

        # Volatility term structure
        features['vol_term_structure'] = features['volatility_60d'] - features['volatility_5d']

        return features

    def compute_sentiment_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sentiment proxy features from market data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with sentiment features
        """
        features = pd.DataFrame(index=df.index)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        # Volume spikes
        if 'Volume' in df.columns:
            vol_ma = df['Volume'].rolling(20).mean()
            vol_std = df['Volume'].rolling(20).std()

            features['volume_spike'] = ((df['Volume'] - vol_ma) / (vol_std + 1e-8)).clip(-3, 3)
            features['unusual_volume'] = (df['Volume'] > vol_ma + 2 * vol_std).astype(int)

        # Price momentum indicators
        for window in [5, 10, 20]:
            features[f'momentum_{window}d'] = df[price_col] / df[price_col].shift(window) - 1

        # Consecutive up/down days
        returns = df[price_col].pct_change()

        # Count consecutive positive returns
        positive_streak = pd.Series(0, index=df.index)
        streak_count = 0
        for i in range(1, len(returns)):
            if returns.iloc[i] > 0:
                streak_count += 1
            else:
                streak_count = 0
            positive_streak.iloc[i] = streak_count

        features['positive_streak'] = positive_streak

        # Count consecutive negative returns
        negative_streak = pd.Series(0, index=df.index)
        streak_count = 0
        for i in range(1, len(returns)):
            if returns.iloc[i] < 0:
                streak_count += 1
            else:
                streak_count = 0
            negative_streak.iloc[i] = streak_count

        features['negative_streak'] = negative_streak

        # Put-Call Ratio proxy (using volatility skew as proxy)
        if 'volatility_20d' in features.columns:
            # Volatility asymmetry as sentiment proxy
            returns_up = returns.where(returns > 0, 0)
            returns_down = returns.where(returns < 0, 0)

            vol_up = returns_up.rolling(20).std()
            vol_down = np.abs(returns_down).rolling(20).std()

            features['volatility_skew'] = (vol_down - vol_up) / (vol_down + vol_up + 1e-8)

        return features

    def compute_cross_sectional_features(self, df: pd.DataFrame,
                                        market_df: Optional[pd.DataFrame] = None,
                                        sector_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute cross-sectional features relative to market/sector.

        Args:
            df: DataFrame with stock OHLCV data
            market_df: DataFrame with market index data (e.g., SPY)
            sector_df: DataFrame with sector ETF data

        Returns:
            DataFrame with cross-sectional features
        """
        features = pd.DataFrame(index=df.index)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        stock_returns = df[price_col].pct_change()

        if market_df is not None:
            market_price_col = 'Adj Close' if 'Adj Close' in market_df.columns else 'Close'
            market_returns = market_df[market_price_col].pct_change()

            # Ensure index alignment
            market_returns = market_returns.reindex(df.index, method='ffill')

            # Beta (market sensitivity)
            for window in [20, 60]:
                rolling_cov = stock_returns.rolling(window).cov(market_returns)
                rolling_var = market_returns.rolling(window).var()
                features[f'beta_{window}d'] = rolling_cov / (rolling_var + 1e-8)

            # Relative strength
            for window in [5, 10, 20]:
                stock_perf = df[price_col] / df[price_col].shift(window) - 1
                market_perf = market_df[market_price_col] / market_df[market_price_col].shift(window) - 1
                market_perf = market_perf.reindex(df.index, method='ffill')

                features[f'relative_strength_{window}d'] = stock_perf - market_perf

            # Correlation with market
            features['market_correlation_20d'] = stock_returns.rolling(20).corr(market_returns)
            features['market_correlation_60d'] = stock_returns.rolling(60).corr(market_returns)

        if sector_df is not None:
            sector_price_col = 'Adj Close' if 'Adj Close' in sector_df.columns else 'Close'
            sector_returns = sector_df[sector_price_col].pct_change()

            # Ensure index alignment
            sector_returns = sector_returns.reindex(df.index, method='ffill')

            # Sector relative performance
            for window in [5, 10, 20]:
                stock_perf = df[price_col] / df[price_col].shift(window) - 1
                sector_perf = sector_df[sector_price_col] / sector_df[sector_price_col].shift(window) - 1
                sector_perf = sector_perf.reindex(df.index, method='ffill')

                features[f'sector_relative_perf_{window}d'] = stock_perf - sector_perf

            # Sector correlation
            features['sector_correlation_20d'] = stock_returns.rolling(20).corr(sector_returns)

        # Market regime indicators
        if market_df is not None:
            # Bull/Bear market indicator (market above/below 200-day MA)
            market_ma200 = market_df[market_price_col].rolling(200).mean()
            market_ma200 = market_ma200.reindex(df.index, method='ffill')
            features['bull_market'] = (market_df[market_price_col].reindex(df.index, method='ffill') > market_ma200).astype(int)

            # Market volatility regime
            market_vol = market_returns.rolling(20).std() * np.sqrt(252)
            market_vol_median = market_vol.rolling(60).median()
            features['high_market_vol'] = (market_vol > market_vol_median).astype(int)

        return features

    def generate_features(self,
                         ticker: str,
                         df: pd.DataFrame,
                         market_df: Optional[pd.DataFrame] = None,
                         sector_df: Optional[pd.DataFrame] = None,
                         save_features: bool = True) -> pd.DataFrame:
        """
        Generate all features for a given stock.

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data
            market_df: Market index data for cross-sectional features
            sector_df: Sector ETF data for relative features
            save_features: Whether to save features to disk

        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Generating features for {ticker}")

        # Initialize features DataFrame
        all_features = pd.DataFrame(index=df.index)

        # 1. Price features
        logger.info("Computing price features...")
        price_features = self.compute_price_features(df)
        all_features = pd.concat([all_features, price_features], axis=1)

        # 2. Technical indicators
        logger.info("Computing technical indicators...")
        technical_features = self.compute_technical_indicators(df)
        all_features = pd.concat([all_features, technical_features], axis=1)

        # 3. Frequency domain features
        if self.config.compute_fft:
            logger.info("Computing frequency domain features...")
            fft_features = self.compute_frequency_domain_features(df, self.config.fft_components)
            all_features = pd.concat([all_features, fft_features], axis=1)

        # 4. Market microstructure features
        if self.config.compute_microstructure:
            logger.info("Computing market microstructure features...")
            micro_features = self.compute_microstructure_features(df)
            all_features = pd.concat([all_features, micro_features], axis=1)

        # 5. Volatility features
        if self.config.compute_volatility:
            logger.info("Computing volatility features...")
            vol_features = self.compute_volatility_features(df)
            all_features = pd.concat([all_features, vol_features], axis=1)

        # 6. Sentiment proxies
        logger.info("Computing sentiment proxies...")
        sentiment_features = self.compute_sentiment_proxies(df)
        all_features = pd.concat([all_features, sentiment_features], axis=1)

        # 7. Cross-sectional features
        if self.config.compute_cross_sectional and (market_df is not None or sector_df is not None):
            logger.info("Computing cross-sectional features...")
            cross_features = self.compute_cross_sectional_features(df, market_df, sector_df)
            all_features = pd.concat([all_features, cross_features], axis=1)

        # Store feature names
        self.feature_names = all_features.columns.tolist()

        # Handle missing values
        all_features = all_features.fillna(method='ffill').fillna(0)

        # Add metadata columns
        all_features['ticker'] = ticker
        all_features['date'] = all_features.index

        # Save features if requested
        if save_features:
            self._save_features(ticker, all_features)

        logger.info(f"Generated {len(all_features.columns)} features for {ticker}")

        return all_features

    def _save_features(self, ticker: str, features: pd.DataFrame):
        """Save features to disk in Parquet format."""
        # Create features directory
        features_dir = Path(__file__).resolve().parents[1] / "data" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        # Save as Parquet (efficient for time series)
        output_path = features_dir / f"{ticker}_features.parquet"
        features.to_parquet(output_path, engine='pyarrow', compression='snappy')

        logger.info(f"Saved features to {output_path}")

        # Also save feature metadata
        metadata = {
            'ticker': ticker,
            'n_features': len(features.columns),
            'feature_names': self.feature_names,
            'date_range': {
                'start': str(features.index.min()),
                'end': str(features.index.max())
            },
            'config': {
                'lookback_windows': self.config.lookback_windows,
                'ma_periods': self.config.ma_periods,
                'rsi_period': self.config.rsi_period,
                'macd_params': self.config.macd_params
            }
        }

        metadata_path = features_dir / f"{ticker}_features_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def fit_scalers(self, features: pd.DataFrame,
                    columns_to_scale: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit RobustScaler on training data.

        Args:
            features: Training features DataFrame
            columns_to_scale: Specific columns to scale. If None, scales all numeric columns.

        Returns:
            Scaled features DataFrame
        """
        if columns_to_scale is None:
            # Scale all numeric columns except metadata
            columns_to_scale = [col for col in features.columns
                              if features[col].dtype in [np.float64, np.float32, np.int64, np.int32]
                              and col not in ['ticker', 'date']]

        scaled_features = features.copy()

        for col in columns_to_scale:
            scaler = RobustScaler()
            scaled_features[col] = scaler.fit_transform(features[[col]])
            self.scalers[col] = scaler

        return scaled_features

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers.

        Args:
            features: Features DataFrame to transform

        Returns:
            Scaled features DataFrame
        """
        scaled_features = features.copy()

        for col, scaler in self.scalers.items():
            if col in features.columns:
                scaled_features[col] = scaler.transform(features[[col]])

        return scaled_features

    def save_scalers(self, path: str):
        """Save fitted scalers to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved scalers to {path}")

    def load_scalers(self, path: str):
        """Load fitted scalers from disk."""
        with open(path, 'rb') as f:
            self.scalers = pickle.load(f)
        logger.info(f"Loaded scalers from {path}")


def main():
    """
    Main function to demonstrate feature engineering pipeline.
    """
    import argparse
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from data_fetching import load_data

    parser = argparse.ArgumentParser(description='Advanced feature engineering for stock data')
    parser.add_argument('--base', type=str, default='last_fetch',
                       help='Base name for input data files')
    parser.add_argument('--tickers', type=str, nargs='+',
                       help='Specific tickers to process')
    parser.add_argument('--market-ticker', type=str, default='SPY',
                       help='Market index ticker for cross-sectional features')

    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")
    data = load_data(base_name=args.base)

    if data['prices'] is None:
        logger.error("No price data found. Please run data_fetching.py first.")
        return

    # Determine tickers to process
    tickers = args.tickers if args.tickers else data.get('tickers', [])

    if not tickers:
        logger.error("No tickers to process.")
        return

    # Fetch market data for cross-sectional features
    market_df = None
    if args.market_ticker:
        try:
            import yfinance as yf
            market = yf.Ticker(args.market_ticker)
            market_df = market.history(start=data['prices'].index.min(),
                                      end=data['prices'].index.max())
            logger.info(f"Loaded market data for {args.market_ticker}")
        except Exception as e:
            logger.warning(f"Could not load market data: {e}")

    # Initialize feature engineer
    config = FeatureConfig()
    engineer = AdvancedFeatureEngineer(config)

    # Process each ticker
    for ticker in tickers:
        logger.info(f"\nProcessing {ticker}...")

        # Prepare OHLCV DataFrame for the ticker
        ticker_df = pd.DataFrame(index=data['prices'].index)

        # Get all columns for this ticker
        ticker_cols = [col for col in data['prices'].columns if ticker in col]

        for col in ticker_cols:
            # Extract column name without ticker
            if f'({ticker})' in col:
                clean_col = col.replace(f'({ticker})', '').strip()
                ticker_df[clean_col] = data['prices'][col]

        # Generate features
        features = engineer.generate_features(
            ticker=ticker,
            df=ticker_df,
            market_df=market_df,
            sector_df=None,  # Could add sector ETF data here
            save_features=True
        )

        # Display feature summary
        logger.info(f"\nFeature summary for {ticker}:")
        logger.info(f"Total features: {len(features.columns)}")
        logger.info(f"Date range: {features.index.min()} to {features.index.max()}")
        logger.info(f"Missing values: {features.isnull().sum().sum()}")

        # Save scaled version
        train_end = int(len(features) * 0.8)
        train_features = features.iloc[:train_end]

        # Fit scalers on training data
        scaled_train = engineer.fit_scalers(train_features)

        # Transform all features
        scaled_features = engineer.transform_features(features)

        # Save scaled features
        features_dir = Path(__file__).resolve().parents[1] / "data" / "features"
        scaled_path = features_dir / f"{ticker}_features_scaled.parquet"
        scaled_features.to_parquet(scaled_path, engine='pyarrow', compression='snappy')

        # Save scalers
        scaler_path = features_dir / f"{ticker}_scalers.pkl"
        engineer.save_scalers(scaler_path)

        logger.info(f"Saved scaled features to {scaled_path}")

    logger.info("\nFeature engineering complete!")


if __name__ == "__main__":
    main()