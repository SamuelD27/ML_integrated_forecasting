"""
Factor-Based Feature Engineering
=================================
Integrates Fama-French factor analysis into ML feature pipeline.

Creates features based on factor exposures, factor momentum, and factor spreads.
These features capture systematic risk exposures and can improve ML model performance.

Key Features:
- Factor betas (exposures to MKT, SMB, HML, RMW, CMA)
- Alpha (skill-based excess returns)
- Factor momentum (which factors are currently performing)
- Factor spreads (value premium, size premium, etc.)
- Factor correlation regime (are factors clustered or dispersed?)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

# Import our factor model
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.factor_models import FamaFrenchFactorModel

logger = logging.getLogger(__name__)


class FactorFeatureEngineer:
    """
    Engineer factor-based features for ML models.

    Combines traditional Fama-French factor analysis with ML-ready features:
    - Static factor exposures (betas) from regression
    - Dynamic factor momentum (rolling factor returns)
    - Cross-sectional factor rankings
    - Factor correlation regime

    Example:
        >>> engineer = FactorFeatureEngineer()
        >>> features = engineer.create_features(returns_df, ['AAPL', 'MSFT', 'GOOGL'])
        >>> # Combine with other features
        >>> all_features = pd.concat([technical_features, factor_features], axis=1)
    """

    def __init__(
        self,
        model: str = '5-factor',
        rolling_windows: Dict[str, int] = None,
        cache_dir: str = 'data/factors'
    ):
        """
        Initialize factor feature engineer.

        Args:
            model: '3-factor' or '5-factor'
            rolling_windows: Dict of window names to days (e.g., {'short': 60, 'long': 252})
            cache_dir: Directory for factor data cache
        """
        self.model = model
        self.rolling_windows = rolling_windows or {
            'short': 60,    # ~3 months
            'medium': 126,  # ~6 months
            'long': 252     # ~1 year
        }
        self.factor_model = FamaFrenchFactorModel(model=model, cache_dir=cache_dir)

        logger.info(f"Initialized FactorFeatureEngineer with {model}")

    def create_features(
        self,
        returns_df: pd.DataFrame,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_factor_momentum: bool = True,
        include_factor_spreads: bool = True,
        include_cross_sectional: bool = True
    ) -> pd.DataFrame:
        """
        Create comprehensive factor-based features.

        Args:
            returns_df: DataFrame with tickers as columns, daily returns as values
            tickers: List of tickers to process
            start_date: Start date for factor data (optional, uses returns range)
            end_date: End date (optional)
            include_factor_momentum: Include factor momentum features
            include_factor_spreads: Include factor spread features
            include_cross_sectional: Include cross-sectional rankings

        Returns:
            DataFrame with factor features, index = tickers

        Features Created:
        - factor_alpha: Annualized alpha from factor regression
        - factor_beta_mkt: Market beta
        - factor_beta_smb: Size beta
        - factor_beta_hml: Value beta
        - factor_beta_rmw: Profitability beta (5-factor only)
        - factor_beta_cma: Investment beta (5-factor only)
        - factor_r_squared: Variance explained by factors
        - factor_alpha_significant: 1 if alpha is significant, 0 otherwise
        - factor_momentum_<window>: Recent factor performance
        - factor_spread_<name>: Factor spread (e.g., SMB premium)
        - factor_rank_alpha: Cross-sectional alpha ranking
        """
        logger.info(f"Creating factor features for {len(tickers)} tickers")

        # Determine date range
        if start_date is None:
            start_date = returns_df.index.min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = returns_df.index.max().strftime('%Y-%m-%d')

        # 1. Calculate factor exposures (betas and alpha)
        logger.info("Calculating factor exposures...")
        regression_results = self.factor_model.batch_regress(
            tickers, returns_df, start_date, end_date, frequency='daily'
        )

        # Convert to features DataFrame (index = ticker)
        features = regression_results.set_index('ticker')[[
            'alpha', 'beta_MKT', 'beta_SMB', 'beta_HML', 'r_squared', 'significant_alpha'
        ]].copy()

        if self.model == '5-factor':
            features['beta_RMW'] = regression_results.set_index('ticker')['beta_RMW']
            features['beta_CMA'] = regression_results.set_index('ticker')['beta_CMA']

        # Rename columns to match feature naming convention
        features = features.rename(columns={
            'alpha': 'factor_alpha',
            'beta_MKT': 'factor_beta_mkt',
            'beta_SMB': 'factor_beta_smb',
            'beta_HML': 'factor_beta_hml',
            'beta_RMW': 'factor_beta_rmw',
            'beta_CMA': 'factor_beta_cma',
            'r_squared': 'factor_r_squared',
            'significant_alpha': 'factor_alpha_significant'
        })

        # 2. Factor momentum features
        if include_factor_momentum:
            logger.info("Calculating factor momentum...")
            momentum_features = self._calculate_factor_momentum(start_date, end_date)

            # Broadcast to all tickers (factor momentum is market-wide)
            for col in momentum_features.columns:
                features[col] = momentum_features[col].iloc[-1]  # Use most recent value

        # 3. Factor spreads (premiums)
        if include_factor_spreads:
            logger.info("Calculating factor spreads...")
            spread_features = self._calculate_factor_spreads(start_date, end_date)

            # Broadcast to all tickers
            for col in spread_features.columns:
                features[col] = spread_features[col].iloc[-1]

        # 4. Cross-sectional rankings
        if include_cross_sectional:
            logger.info("Calculating cross-sectional rankings...")
            features = self._add_cross_sectional_rankings(features)

        logger.info(f"Created {len(features.columns)} factor features")
        return features

    def _calculate_factor_momentum(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Calculate factor momentum (recent factor performance).

        Factor momentum indicates which factors are currently "working".
        For example, if SMB has positive momentum, small caps are outperforming.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with factor momentum features
        """
        # Fetch factor returns
        factors = self.factor_model.fetch_factors(start_date, end_date, frequency='daily')

        # Convert from percentage to decimal
        factors = factors / 100.0

        momentum = pd.DataFrame(index=factors.index)

        factor_cols = ['Mkt-RF', 'SMB', 'HML']
        if self.model == '5-factor':
            factor_cols.extend(['RMW', 'CMA'])

        # Calculate cumulative returns over rolling windows
        for window_name, window_days in self.rolling_windows.items():
            for factor in factor_cols:
                col_name = f'factor_momentum_{factor.lower().replace("-", "_")}_{window_name}'

                # Cumulative return over window
                cumulative_return = (1 + factors[factor]).rolling(window_days).apply(
                    lambda x: x.prod() - 1, raw=True
                )

                momentum[col_name] = cumulative_return

        return momentum.dropna()

    def _calculate_factor_spreads(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Calculate factor spreads (premiums).

        Factor spreads represent the excess return from factor exposures.
        For example, SMB spread = return of small caps - return of large caps.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with factor spread features
        """
        # Fetch factor returns
        factors = self.factor_model.fetch_factors(start_date, end_date, frequency='daily')

        # Convert from percentage to decimal
        factors = factors / 100.0

        spreads = pd.DataFrame(index=factors.index)

        # Calculate rolling average spreads
        for window_name, window_days in self.rolling_windows.items():
            # Size spread (SMB)
            spreads[f'factor_spread_size_{window_name}'] = \
                factors['SMB'].rolling(window_days).mean()

            # Value spread (HML)
            spreads[f'factor_spread_value_{window_name}'] = \
                factors['HML'].rolling(window_days).mean()

            if self.model == '5-factor':
                # Profitability spread (RMW)
                spreads[f'factor_spread_profitability_{window_name}'] = \
                    factors['RMW'].rolling(window_days).mean()

                # Investment spread (CMA)
                spreads[f'factor_spread_investment_{window_name}'] = \
                    factors['CMA'].rolling(window_days).mean()

        return spreads.dropna()

    def _add_cross_sectional_rankings(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add cross-sectional rankings (percentile ranks across universe).

        Converts absolute values to relative rankings within the universe.

        Args:
            features: Features DataFrame (index = ticker)

        Returns:
            Features with added ranking columns
        """
        # Rank alpha (higher is better)
        if 'factor_alpha' in features.columns:
            features['factor_rank_alpha'] = features['factor_alpha'].rank(pct=True)

        # Rank R-squared (higher means more explained by factors = less idiosyncratic risk)
        if 'factor_r_squared' in features.columns:
            features['factor_rank_r_squared'] = features['factor_r_squared'].rank(pct=True)

        # Rank market beta (useful for low-beta strategies)
        if 'factor_beta_mkt' in features.columns:
            features['factor_rank_beta_mkt'] = features['factor_beta_mkt'].rank(pct=True)

        return features

    def calculate_factor_correlation_regime(
        self,
        start_date: str,
        end_date: str,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate factor correlation regime.

        High correlation = factors move together (risk-on/risk-off)
        Low correlation = factors dispersed (stock-picking environment)

        Args:
            start_date: Start date
            end_date: End date
            window: Rolling window for correlation calculation

        Returns:
            DataFrame with correlation regime features
        """
        # Fetch factors
        factors = self.factor_model.fetch_factors(start_date, end_date, frequency='daily')
        factors = factors / 100.0  # Convert to decimal

        factor_cols = ['Mkt-RF', 'SMB', 'HML']
        if self.model == '5-factor':
            factor_cols.extend(['RMW', 'CMA'])

        # Calculate rolling correlation matrix
        correlations = []

        for i in range(window, len(factors)):
            window_data = factors.iloc[i-window:i][factor_cols]
            corr_matrix = window_data.corr()

            # Extract upper triangle (excluding diagonal)
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Average absolute correlation
            avg_corr = upper_tri.abs().stack().mean()

            correlations.append({
                'date': factors.index[i],
                'factor_correlation_avg': avg_corr,
                'factor_correlation_max': upper_tri.abs().stack().max(),
                'factor_correlation_min': upper_tri.abs().stack().min()
            })

        corr_df = pd.DataFrame(correlations).set_index('date')

        # Add regime classification
        # High correlation > 0.6, Low < 0.3
        corr_df['factor_correlation_regime'] = pd.cut(
            corr_df['factor_correlation_avg'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )

        return corr_df


def integrate_factor_features(
    stock_features: pd.DataFrame,
    returns_df: pd.DataFrame,
    tickers: List[str]
) -> pd.DataFrame:
    """
    Convenience function to integrate factor features into existing feature set.

    Args:
        stock_features: Existing features (technical, volume, etc.)
        returns_df: Returns DataFrame for factor regression
        tickers: List of tickers

    Returns:
        Combined features DataFrame

    Example:
        >>> # After calculating technical features
        >>> all_features = integrate_factor_features(
        ...     technical_features, returns_df, tickers
        ... )
        >>> # Now all_features includes both technical and factor features
    """
    logger.info("Integrating factor features...")

    engineer = FactorFeatureEngineer(model='5-factor')

    factor_features = engineer.create_features(
        returns_df,
        tickers,
        include_factor_momentum=True,
        include_factor_spreads=True,
        include_cross_sectional=True
    )

    # Merge with existing features
    combined = stock_features.join(factor_features, how='left')

    logger.info(f"Combined features shape: {combined.shape}")
    return combined
