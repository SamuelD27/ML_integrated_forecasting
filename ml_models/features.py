import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering for ML stock selection."""

    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Initialize feature engineer.

        Parameters
        ----------
        lookback_periods : dict
            Custom lookback periods for features
            Default: {'short': 20, 'medium': 60, 'long': 252}
        """
        self.lookback_periods = lookback_periods or {
            'short': 20,   # ~1 month
            'medium': 60,  # ~3 months
            'long': 252,   # ~1 year
        }

    def engineer_features(self, prices_df: pd.DataFrame,
                         volumes_df: Optional[pd.DataFrame] = None,
                         market_ticker: str = 'SPY') -> pd.DataFrame:
        """
        Engineer comprehensive feature set for all assets.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Price data with tickers as columns
        volumes_df : pd.DataFrame, optional
            Volume data with tickers as columns
        market_ticker : str
            Market benchmark ticker for relative features

        Returns
        -------
        pd.DataFrame
            Features for each ticker at each date
        """
        features_list = []

        for ticker in prices_df.columns:
            print(f"    Engineering features for {ticker}...")

            price_series = prices_df[ticker].dropna()
            volume_series = volumes_df[ticker].dropna() if volumes_df is not None else None

            # Calculate all features
            ticker_features = self._calculate_ticker_features(
                price_series, volume_series, ticker
            )

            # Add market-relative features
            if market_ticker in prices_df.columns and ticker != market_ticker:
                market_features = self._calculate_market_relative_features(
                    price_series, prices_df[market_ticker]
                )
                ticker_features.update(market_features)

            # Convert to DataFrame (single row)
            df = pd.DataFrame([ticker_features])  # Wrap in list for single row
            df['ticker'] = ticker
            # Add the latest date from price series
            df['date'] = price_series.index[-1] if len(price_series) > 0 else pd.Timestamp.now()
            features_list.append(df)

        # Combine all tickers
        features_df = pd.concat(features_list, ignore_index=True)

        # Add cross-sectional features (rank-based)
        features_df = self._add_cross_sectional_features(features_df)

        return features_df

    def _calculate_ticker_features(self, prices: pd.Series,
                                   volumes: Optional[pd.Series],
                                   ticker: str) -> Dict:
        """Calculate features for a single ticker."""
        features = {}

        # Returns
        returns = prices.pct_change()

        # 1. MOMENTUM FEATURES
        for period_name, period in self.lookback_periods.items():
            if len(prices) >= period:
                # Simple momentum (total return)
                features[f'momentum_{period_name}'] = (
                    prices.iloc[-1] / prices.iloc[-period] - 1
                )

                # Annualized momentum
                features[f'momentum_{period_name}_ann'] = (
                    ((1 + features[f'momentum_{period_name}']) ** (252/period) - 1)
                )

                # Risk-adjusted momentum (Sharpe-like)
                period_returns = returns.iloc[-period:]
                if period_returns.std() > 0:
                    features[f'momentum_{period_name}_sharpe'] = (
                        period_returns.mean() / period_returns.std() * np.sqrt(252)
                    )
                else:
                    features[f'momentum_{period_name}_sharpe'] = 0

        # 2. VOLATILITY FEATURES
        for period_name, period in self.lookback_periods.items():
            if len(returns) >= period:
                period_returns = returns.iloc[-period:]

                # Realized volatility
                features[f'volatility_{period_name}'] = (
                    period_returns.std() * np.sqrt(252)
                )

                # Downside deviation (semi-deviation)
                downside_returns = period_returns[period_returns < 0]
                if len(downside_returns) > 0:
                    features[f'downside_vol_{period_name}'] = (
                        downside_returns.std() * np.sqrt(252)
                    )
                else:
                    features[f'downside_vol_{period_name}'] = 0

                # Skewness (tail risk)
                if len(period_returns) >= 30:
                    features[f'skewness_{period_name}'] = stats.skew(period_returns.dropna())
                else:
                    features[f'skewness_{period_name}'] = 0

                # Kurtosis (tail thickness)
                if len(period_returns) >= 30:
                    features[f'kurtosis_{period_name}'] = stats.kurtosis(period_returns.dropna())
                else:
                    features[f'kurtosis_{period_name}'] = 0

        # 3. MEAN REVERSION FEATURES
        for period_name, period in [('short', self.lookback_periods['short'])]:
            if len(prices) >= period:
                # Distance from moving average
                ma = prices.iloc[-period:].mean()
                features[f'price_to_ma_{period_name}'] = prices.iloc[-1] / ma - 1

                # Z-score (standardized distance)
                std = prices.iloc[-period:].std()
                if std > 0:
                    features[f'zscore_{period_name}'] = (prices.iloc[-1] - ma) / std
                else:
                    features[f'zscore_{period_name}'] = 0

        # 4. TECHNICAL INDICATORS
        if len(prices) >= 30:
            # RSI (Relative Strength Index)
            features['rsi_14'] = self._calculate_rsi(prices, period=14)

            # MACD
            macd, signal = self._calculate_macd(prices)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = macd - signal

        # Bollinger Bands
        if len(prices) >= 20:
            bb_upper, bb_lower, bb_mid = self._calculate_bollinger_bands(prices, period=20)
            features['bb_position'] = (prices.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            features['bb_width'] = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0

        # 5. VOLUME FEATURES (if available)
        if volumes is not None and len(volumes) >= 20:
            # Volume trend
            recent_vol = volumes.iloc[-self.lookback_periods['short']:].mean()
            longer_vol = volumes.iloc[-self.lookback_periods['medium']:].mean()
            features['volume_trend'] = (recent_vol / longer_vol - 1) if longer_vol > 0 else 0

            # Volume volatility (liquidity proxy)
            features['volume_volatility'] = volumes.iloc[-self.lookback_periods['short']:].std() / longer_vol if longer_vol > 0 else 0

        # 6. QUALITY METRICS
        # Price stability (inverse of volatility range)
        if len(prices) >= self.lookback_periods['medium']:
            price_range = prices.iloc[-self.lookback_periods['medium']:].max() - prices.iloc[-self.lookback_periods['medium']:].min()
            features['price_stability'] = 1 - (price_range / prices.iloc[-self.lookback_periods['medium']:].mean())

        # Drawdown from peak
        if len(prices) >= self.lookback_periods['long']:
            peak = prices.iloc[-self.lookback_periods['long']:].max()
            features['drawdown_from_peak'] = (prices.iloc[-1] / peak - 1)

        # 7. RECENT PERFORMANCE
        # Last week return
        if len(prices) >= 5:
            features['return_1w'] = prices.iloc[-1] / prices.iloc[-5] - 1

        # Last month return
        if len(prices) >= 20:
            features['return_1m'] = prices.iloc[-1] / prices.iloc[-20] - 1

        return features

    def _calculate_market_relative_features(self, prices: pd.Series,
                                           market_prices: pd.Series) -> Dict:
        """Calculate market-relative features (beta, correlation, etc.)."""
        features = {}

        # Align series
        aligned = pd.DataFrame({
            'asset': prices,
            'market': market_prices
        }).dropna()

        if len(aligned) < 30:
            return features

        asset_returns = aligned['asset'].pct_change().dropna()
        market_returns = aligned['market'].pct_change().dropna()

        # Beta (rolling periods)
        for period_name, period in self.lookback_periods.items():
            if len(aligned) >= period:
                recent_asset = asset_returns.iloc[-period:]
                recent_market = market_returns.iloc[-period:]

                # CAPM beta
                covar = recent_asset.cov(recent_market)
                market_var = recent_market.var()
                features[f'beta_{period_name}'] = covar / market_var if market_var > 0 else 1.0

                # Correlation
                features[f'correlation_{period_name}'] = recent_asset.corr(recent_market)

                # Alpha (excess return over beta-adjusted market)
                beta = features[f'beta_{period_name}']
                asset_mean = recent_asset.mean() * 252
                market_mean = recent_market.mean() * 252
                features[f'alpha_{period_name}'] = asset_mean - beta * market_mean

        # Relative strength
        if len(aligned) >= self.lookback_periods['medium']:
            asset_mom = prices.iloc[-1] / prices.iloc[-self.lookback_periods['medium']] - 1
            market_mom = market_prices.iloc[-1] / market_prices.iloc[-self.lookback_periods['medium']] - 1
            features['relative_strength'] = asset_mom - market_mom

        return features

    def _add_cross_sectional_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional (rank-based) features."""
        # Get most recent date for each ticker
        latest_features = features_df.groupby('ticker').tail(1).copy()

        # Rank features across tickers (percentile rank)
        rank_features = [
            'momentum_medium', 'momentum_long',
            'volatility_medium', 'sharpe_ratio',
            'rsi_14', 'beta_medium'
        ]

        for feature in rank_features:
            if feature in latest_features.columns:
                rank_col = f'{feature}_rank'
                latest_features[rank_col] = latest_features[feature].rank(pct=True)

                # Merge back to main df
                ticker_ranks = latest_features.set_index('ticker')[rank_col].to_dict()
                features_df[rank_col] = features_df['ticker'].map(ticker_ranks)

        return features_df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0

        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).iloc[-period:].mean()
        loss = -deltas.where(deltas < 0, 0).iloc[-period:].mean()

        if loss == 0:
            return 100.0

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series,
                       fast: int = 12, slow: int = 26, signal: int = 9
                       ) -> Tuple[float, float]:
        """Calculate MACD and signal line."""
        if len(prices) < slow + signal:
            return 0.0, 0.0

        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()

        return macd_line.iloc[-1], signal_line.iloc[-1]

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20,
                                   num_std: float = 2.0
                                   ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return prices.iloc[-1], prices.iloc[-1], prices.iloc[-1]

        ma = prices.iloc[-period:].mean()
        std = prices.iloc[-period:].std()

        upper = ma + num_std * std
        lower = ma - num_std * std

        return upper, lower, ma

    def get_feature_importance_names(self) -> List[str]:
        """Get list of all feature names for ML model."""
        # This should match all features created above
        base_features = []

        # Momentum
        for period in ['short', 'medium', 'long']:
            base_features.extend([
                f'momentum_{period}',
                f'momentum_{period}_ann',
                f'momentum_{period}_sharpe',
            ])

        # Volatility
        for period in ['short', 'medium', 'long']:
            base_features.extend([
                f'volatility_{period}',
                f'downside_vol_{period}',
                f'skewness_{period}',
                f'kurtosis_{period}',
            ])

        # Mean reversion
        base_features.extend([
            'price_to_ma_short',
            'zscore_short',
        ])

        # Technical
        base_features.extend([
            'rsi_14',
            'macd',
            'macd_signal',
            'macd_histogram',
            'bb_position',
            'bb_width',
        ])

        # Volume
        base_features.extend([
            'volume_trend',
            'volume_volatility',
        ])

        # Quality
        base_features.extend([
            'price_stability',
            'drawdown_from_peak',
            'return_1w',
            'return_1m',
        ])

        # Market relative
        for period in ['short', 'medium', 'long']:
            base_features.extend([
                f'beta_{period}',
                f'correlation_{period}',
                f'alpha_{period}',
            ])

        base_features.append('relative_strength')

        # Cross-sectional ranks
        rank_features = [
            'momentum_medium_rank',
            'momentum_long_rank',
            'volatility_medium_rank',
            'rsi_14_rank',
            'beta_medium_rank',
        ]

        base_features.extend(rank_features)

        return base_features


def create_ml_features(prices_df: pd.DataFrame,
                       volumes_df: Optional[pd.DataFrame] = None,
                       market_ticker: str = 'SPY') -> pd.DataFrame:
    """
    Convenience function to create ML features.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Price data
    volumes_df : pd.DataFrame, optional
        Volume data
    market_ticker : str
        Market benchmark

    Returns
    -------
    pd.DataFrame
        Engineered features
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(prices_df, volumes_df, market_ticker)
