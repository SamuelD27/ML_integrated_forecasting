"""
Fama-French Factor Models
==========================
Implementation of Fama-French factor models for return decomposition and alpha generation.

The Fama-French model decomposes stock returns into systematic factors:
- Market (MKT-RF): Market return minus risk-free rate
- Size (SMB): Small Minus Big - small cap premium
- Value (HML): High Minus Low - value premium
- Profitability (RMW): Robust Minus Weak - profitability premium
- Investment (CMA): Conservative Minus Aggressive - investment premium

Alpha is the intercept - returns NOT explained by factor exposures.
Significant positive alpha indicates genuine skill-based returns.

Data Source: Kenneth French Data Library (via pandas-datareader)
Updated monthly: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

try:
    from pandas_datareader import data as web
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False

logger = logging.getLogger(__name__)


class FamaFrenchFactorModel:
    """
    Fama-French 3-Factor and 5-Factor model implementation.

    Performs factor regression to decompose returns into:
    - Systematic factor exposures (betas)
    - Alpha (skill-based excess returns)
    - Factor attribution (contribution from each factor)

    Example:
        >>> ff = FamaFrenchFactorModel()
        >>> results = ff.regress_returns('AAPL', stock_returns, '2020-01-01', '2023-12-31')
        >>> print(f"Alpha: {results['alpha']:.4f}, Market Beta: {results['beta_MKT']:.4f}")
    """

    FACTOR_NAMES_3 = ['Mkt-RF', 'SMB', 'HML', 'RF']
    FACTOR_NAMES_5 = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

    def __init__(
        self,
        model: str = '5-factor',
        cache_dir: str = 'data/factors',
        cache_ttl_days: int = 30
    ):
        """
        Initialize Fama-French factor model.

        Args:
            model: '3-factor' or '5-factor'
            cache_dir: Directory to cache factor data
            cache_ttl_days: Cache time-to-live in days

        Raises:
            ValueError: If model not supported
            ImportError: If pandas-datareader not installed
        """
        if not DATAREADER_AVAILABLE:
            raise ImportError(
                "pandas-datareader not installed. Install: pip install pandas-datareader"
            )

        if model not in ['3-factor', '5-factor']:
            raise ValueError(f"Model must be '3-factor' or '5-factor', got {model}")

        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(days=cache_ttl_days)

        self._factors_cache: Optional[pd.DataFrame] = None

        logger.info(f"Initialized Fama-French {model} model")

    def fetch_factors(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        frequency: str = 'daily',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch Fama-French factor data from Kenneth French Data Library.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            frequency: 'daily' or 'monthly'
            use_cache: Whether to use cached data

        Returns:
            DataFrame with factor returns (in percentage)
            Columns: Mkt-RF, SMB, HML, RMW, CMA, RF

        Example:
            >>> ff = FamaFrenchFactorModel()
            >>> factors = ff.fetch_factors('2020-01-01', '2023-12-31')
            >>> print(factors.head())
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(frequency)
            if cached is not None:
                # Filter to date range
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
                cached = cached.loc[start:end]

                if not cached.empty:
                    logger.info(f"Loaded {len(cached)} factor observations from cache")
                    return cached

        logger.info(f"Fetching Fama-French {self.model} data from web...")

        try:
            # Determine dataset name
            if self.model == '5-factor':
                if frequency == 'daily':
                    dataset = 'F-F_Research_Data_5_Factors_2x3_daily'
                else:
                    dataset = 'F-F_Research_Data_5_Factors_2x3'
            else:  # 3-factor
                if frequency == 'daily':
                    dataset = 'F-F_Research_Data_Factors_daily'
                else:
                    dataset = 'F-F_Research_Data_Factors'

            # Fetch from Kenneth French Data Library
            ff_data = web.DataReader(dataset, 'famafrench', start_date, end_date)

            # Extract the first DataFrame (contains the factors)
            factors_df = ff_data[0]

            # Convert to datetime index
            factors_df.index = pd.to_datetime(factors_df.index, format='%Y%m%d')

            # Ensure we have all required columns
            expected_cols = self.FACTOR_NAMES_5 if self.model == '5-factor' else self.FACTOR_NAMES_3
            missing_cols = set(expected_cols) - set(factors_df.columns)

            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")

            # Save to cache
            self._save_to_cache(factors_df, frequency)

            logger.info(f"Fetched {len(factors_df)} factor observations")
            return factors_df

        except Exception as e:
            logger.error(f"Failed to fetch Fama-French factors: {e}")
            raise

    def regress_returns(
        self,
        ticker: str,
        returns: pd.Series,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = 'daily'
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Perform factor regression on stock returns.

        Regresses excess returns (stock - RF) against factor returns:
        R_i - R_f = alpha + beta_MKT*(R_M - R_f) + beta_SMB*SMB + beta_HML*HML +
                    beta_RMW*RMW + beta_CMA*CMA + epsilon

        Args:
            ticker: Stock ticker symbol
            returns: Stock returns (daily or monthly)
            start_date: Start date for regression (optional, uses full returns range)
            end_date: End date for regression (optional)
            frequency: 'daily' or 'monthly'

        Returns:
            Dictionary with:
            - alpha: Intercept (annualized)
            - alpha_tstat: t-statistic for alpha
            - alpha_pvalue: p-value for alpha significance
            - beta_MKT: Market beta
            - beta_SMB: Size beta
            - beta_HML: Value beta
            - beta_RMW: Profitability beta (5-factor only)
            - beta_CMA: Investment beta (5-factor only)
            - r_squared: R-squared of regression
            - residuals: Regression residuals

        Example:
            >>> results = ff.regress_returns('AAPL', aapl_returns)
            >>> if results['alpha_pvalue'] < 0.05:
            >>>     print(f"Significant alpha: {results['alpha']:.2%}")
        """
        logger.info(f"Performing factor regression for {ticker}")

        # Ensure returns is a Series with datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        # Determine date range
        if start_date is None:
            start_date = returns.index.min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = returns.index.max().strftime('%Y-%m-%d')

        # Fetch factors
        factors = self.fetch_factors(start_date, end_date, frequency)

        # Align returns and factors (inner join on date)
        merged = pd.concat([returns.rename('Stock_Return'), factors], axis=1, join='inner')
        merged = merged.dropna()

        if len(merged) < 30:
            logger.warning(f"Insufficient data for regression: {len(merged)} observations")
            return self._empty_results()

        # Calculate excess returns (stock return - risk-free rate)
        # Factors are in percentage, convert to decimal
        merged['Excess_Return'] = merged['Stock_Return'] - (merged['RF'] / 100.0)

        # Prepare independent variables (factors in decimal)
        X_cols = ['Mkt-RF', 'SMB', 'HML']
        if self.model == '5-factor':
            X_cols.extend(['RMW', 'CMA'])

        X = merged[X_cols] / 100.0  # Convert from percentage to decimal
        y = merged['Excess_Return']

        # Add constant for alpha
        X = sm.add_constant(X)

        # OLS regression
        model = sm.OLS(y, X).fit()

        # Extract results
        results = {
            'ticker': ticker,
            'n_observations': len(merged),
            'start_date': merged.index.min().strftime('%Y-%m-%d'),
            'end_date': merged.index.max().strftime('%Y-%m-%d'),
        }

        # Alpha (annualize based on frequency)
        alpha_daily = model.params['const']
        if frequency == 'daily':
            results['alpha'] = alpha_daily * 252  # Annualize
        else:
            results['alpha'] = alpha_daily * 12  # Monthly to annual

        results['alpha_tstat'] = model.tvalues['const']
        results['alpha_pvalue'] = model.pvalues['const']

        # Betas
        results['beta_MKT'] = model.params['Mkt-RF']
        results['beta_SMB'] = model.params['SMB']
        results['beta_HML'] = model.params['HML']

        if self.model == '5-factor':
            results['beta_RMW'] = model.params['RMW']
            results['beta_CMA'] = model.params['CMA']

        # Model statistics
        results['r_squared'] = model.rsquared
        results['adj_r_squared'] = model.rsquared_adj
        results['residuals'] = model.resid

        # Interpretation flags
        results['significant_alpha'] = results['alpha_pvalue'] < 0.05
        results['alpha_direction'] = 'positive' if results['alpha'] > 0 else 'negative'

        logger.info(
            f"{ticker}: Alpha={results['alpha']:.4f} "
            f"(t={results['alpha_tstat']:.2f}, p={results['alpha_pvalue']:.4f}), "
            f"Beta_MKT={results['beta_MKT']:.2f}, R²={results['r_squared']:.3f}"
        )

        return results

    def calculate_factor_attribution(
        self,
        regression_results: Dict,
        factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate factor contribution to total returns.

        Attribution = beta * factor_return for each factor

        Args:
            regression_results: Output from regress_returns()
            factor_returns: Factor returns DataFrame

        Returns:
            DataFrame with columns for each factor's contribution

        Example:
            >>> results = ff.regress_returns('AAPL', returns)
            >>> factors = ff.fetch_factors('2023-01-01', '2023-12-31')
            >>> attribution = ff.calculate_factor_attribution(results, factors)
            >>> print(attribution.sum())  # Total contribution from each factor
        """
        # Convert factor returns from percentage to decimal
        factors = factor_returns / 100.0

        attribution = pd.DataFrame(index=factors.index)

        # Alpha contribution (constant daily)
        if 'alpha' in regression_results:
            # De-annualize alpha
            alpha_daily = regression_results['alpha'] / 252
            attribution['Alpha'] = alpha_daily

        # Factor contributions
        attribution['Market'] = regression_results['beta_MKT'] * factors['Mkt-RF']
        attribution['Size'] = regression_results['beta_SMB'] * factors['SMB']
        attribution['Value'] = regression_results['beta_HML'] * factors['HML']

        if self.model == '5-factor':
            attribution['Profitability'] = regression_results['beta_RMW'] * factors['RMW']
            attribution['Investment'] = regression_results['beta_CMA'] * factors['CMA']

        # Residual (unexplained)
        if 'residuals' in regression_results:
            residuals = regression_results['residuals']
            # Align residuals with attribution index
            attribution['Residual'] = residuals.reindex(attribution.index, fill_value=0)

        return attribution

    def batch_regress(
        self,
        tickers: List[str],
        returns_df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Perform factor regression for multiple tickers.

        Args:
            tickers: List of ticker symbols
            returns_df: DataFrame with tickers as columns, returns as values
            start_date: Start date (optional)
            end_date: End date (optional)
            frequency: 'daily' or 'monthly'

        Returns:
            DataFrame with regression results for each ticker

        Example:
            >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
            >>> results_df = ff.batch_regress(tickers, returns_df)
            >>> print(results_df[['ticker', 'alpha', 'beta_MKT', 'r_squared']])
        """
        logger.info(f"Batch regression for {len(tickers)} tickers")

        results_list = []

        for ticker in tickers:
            if ticker not in returns_df.columns:
                logger.warning(f"Ticker {ticker} not in returns DataFrame")
                continue

            try:
                returns = returns_df[ticker].dropna()
                results = self.regress_returns(ticker, returns, start_date, end_date, frequency)

                # Extract scalar values only (exclude residuals)
                scalar_results = {k: v for k, v in results.items()
                                if not isinstance(v, (pd.Series, pd.DataFrame))}
                results_list.append(scalar_results)

            except Exception as e:
                logger.error(f"Regression failed for {ticker}: {e}")
                continue

        results_df = pd.DataFrame(results_list)
        logger.info(f"Successfully regressed {len(results_df)} tickers")

        return results_df

    def rank_by_alpha(
        self,
        regression_results: pd.DataFrame,
        min_observations: int = 60,
        significance_level: float = 0.10
    ) -> pd.DataFrame:
        """
        Rank stocks by alpha, filtering by significance and sample size.

        Args:
            regression_results: Output from batch_regress()
            min_observations: Minimum observations required
            significance_level: Maximum p-value for alpha significance

        Returns:
            DataFrame sorted by alpha (descending)

        Example:
            >>> ranked = ff.rank_by_alpha(batch_results, significance_level=0.05)
            >>> top_alphas = ranked.head(20)  # Top 20 alpha generators
        """
        filtered = regression_results[
            (regression_results['n_observations'] >= min_observations) &
            (regression_results['alpha_pvalue'] <= significance_level)
        ].copy()

        filtered = filtered.sort_values('alpha', ascending=False)

        logger.info(
            f"Ranked {len(filtered)} stocks with significant alpha "
            f"(p<{significance_level}) from {len(regression_results)} total"
        )

        return filtered

    def _empty_results(self) -> Dict:
        """Return empty results dict for failed regressions."""
        return {
            'alpha': np.nan,
            'alpha_tstat': np.nan,
            'alpha_pvalue': np.nan,
            'beta_MKT': np.nan,
            'beta_SMB': np.nan,
            'beta_HML': np.nan,
            'beta_RMW': np.nan if self.model == '5-factor' else None,
            'beta_CMA': np.nan if self.model == '5-factor' else None,
            'r_squared': np.nan,
            'n_observations': 0
        }

    def _load_from_cache(self, frequency: str) -> Optional[pd.DataFrame]:
        """Load factors from cache if available and fresh."""
        cache_file = self.cache_dir / f"ff_{self.model}_{frequency}.parquet"

        if not cache_file.exists():
            return None

        # Check cache age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > self.cache_ttl:
            logger.info(f"Cache expired (age: {file_age.days} days)")
            return None

        try:
            df = pd.read_parquet(cache_file)
            logger.info(f"Loaded factors from cache ({file_age.days} days old)")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, factors: pd.DataFrame, frequency: str):
        """Save factors to cache."""
        cache_file = self.cache_dir / f"ff_{self.model}_{frequency}.parquet"

        try:
            factors.to_parquet(cache_file)
            logger.info(f"Saved factors to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


def create_factor_features(
    returns_df: pd.DataFrame,
    tickers: List[str],
    lookback_window: int = 252,
    model: str = '5-factor'
) -> pd.DataFrame:
    """
    Create factor-based features for ML models.

    For each ticker, calculates:
    - Factor betas (MKT, SMB, HML, RMW, CMA)
    - Alpha (annualized)
    - R-squared (how much variance explained by factors)
    - Factor momentum (which factors are working recently)

    Args:
        returns_df: DataFrame with tickers as columns
        tickers: List of tickers to process
        lookback_window: Rolling window for factor regression (default: 252 days = 1 year)
        model: '3-factor' or '5-factor'

    Returns:
        DataFrame with factor features for each ticker

    Example:
        >>> features = create_factor_features(returns_df, ['AAPL', 'MSFT', 'GOOGL'])
        >>> # Use in ML model alongside other features
        >>> X = pd.concat([technical_features, features], axis=1)
    """
    logger.info(f"Creating factor features for {len(tickers)} tickers")

    ff = FamaFrenchFactorModel(model=model)

    # Get date range from returns
    start_date = returns_df.index.min().strftime('%Y-%m-%d')
    end_date = returns_df.index.max().strftime('%Y-%m-%d')

    # Perform batch regression
    results = ff.batch_regress(tickers, returns_df, start_date, end_date)

    # Create feature DataFrame
    features = pd.DataFrame()

    for _, row in results.iterrows():
        ticker = row['ticker']

        features.loc[ticker, 'factor_alpha'] = row['alpha']
        features.loc[ticker, 'factor_beta_mkt'] = row['beta_MKT']
        features.loc[ticker, 'factor_beta_smb'] = row['beta_SMB']
        features.loc[ticker, 'factor_beta_hml'] = row['beta_HML']

        if model == '5-factor':
            features.loc[ticker, 'factor_beta_rmw'] = row['beta_RMW']
            features.loc[ticker, 'factor_beta_cma'] = row['beta_CMA']

        features.loc[ticker, 'factor_r_squared'] = row['r_squared']
        features.loc[ticker, 'factor_alpha_significant'] = float(row['significant_alpha'])

    logger.info(f"Created factor features: {features.shape}")
    return features
