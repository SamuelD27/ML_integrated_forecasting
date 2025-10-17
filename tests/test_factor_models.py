"""
Test Fama-French Factor Models
===============================
Unit tests for factor model integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio.factor_models import FamaFrenchFactorModel, create_factor_features
from ml_models.factor_features import FactorFeatureEngineer, integrate_factor_features


class TestFamaFrenchFactorModel:
    """Test Fama-French factor model."""

    @pytest.fixture
    def ff_model(self):
        """Create factor model instance."""
        return FamaFrenchFactorModel(model='5-factor')

    @pytest.fixture
    def sample_returns(self):
        """Create sample stock returns."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)

        # Generate correlated returns (simulate market relationship)
        market_returns = np.random.normal(0.0005, 0.01, len(dates))
        stock_returns = 0.8 * market_returns + np.random.normal(0.0002, 0.005, len(dates))

        return pd.Series(stock_returns, index=dates, name='AAPL')

    def test_fetch_factors(self, ff_model):
        """Test fetching Fama-French factors."""
        factors = ff_model.fetch_factors('2023-01-01', '2023-12-31', frequency='daily')

        assert not factors.empty
        assert 'Mkt-RF' in factors.columns
        assert 'SMB' in factors.columns
        assert 'HML' in factors.columns
        assert 'RMW' in factors.columns
        assert 'CMA' in factors.columns
        assert 'RF' in factors.columns

        print(f"\n✓ Fetched {len(factors)} daily factor observations")
        print(f"✓ Columns: {list(factors.columns)}")

    def test_regress_returns(self, ff_model, sample_returns):
        """Test factor regression."""
        results = ff_model.regress_returns(
            'AAPL',
            sample_returns,
            frequency='daily'
        )

        assert 'alpha' in results
        assert 'beta_MKT' in results
        assert 'beta_SMB' in results
        assert 'beta_HML' in results
        assert 'beta_RMW' in results
        assert 'beta_CMA' in results
        assert 'r_squared' in results

        assert results['n_observations'] > 0
        assert 0 <= results['r_squared'] <= 1

        print(f"\n✓ Regression results:")
        print(f"  Alpha: {results['alpha']:.4f} (annualized)")
        print(f"  Beta (Market): {results['beta_MKT']:.2f}")
        print(f"  R²: {results['r_squared']:.3f}")
        print(f"  Observations: {results['n_observations']}")

    def test_batch_regress(self, ff_model):
        """Test batch regression for multiple tickers."""
        # Generate sample data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)

        returns_df = pd.DataFrame({
            'AAPL': np.random.normal(0.0005, 0.01, len(dates)),
            'MSFT': np.random.normal(0.0004, 0.009, len(dates)),
            'GOOGL': np.random.normal(0.0006, 0.011, len(dates))
        }, index=dates)

        results = ff_model.batch_regress(
            ['AAPL', 'MSFT', 'GOOGL'],
            returns_df,
            frequency='daily'
        )

        assert len(results) == 3
        assert 'ticker' in results.columns
        assert 'alpha' in results.columns
        assert 'beta_MKT' in results.columns

        print(f"\n✓ Batch regression for {len(results)} tickers")
        print(results[['ticker', 'alpha', 'beta_MKT', 'r_squared']])

    def test_rank_by_alpha(self, ff_model):
        """Test alpha ranking."""
        # Create sample regression results
        results = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'alpha': [0.05, -0.02, 0.08, 0.03],
            'alpha_pvalue': [0.03, 0.15, 0.01, 0.08],
            'n_observations': [250, 250, 250, 250],
            'beta_MKT': [1.2, 0.9, 1.1, 1.5],
            'r_squared': [0.65, 0.72, 0.58, 0.45]
        })

        ranked = ff_model.rank_by_alpha(
            results,
            min_observations=200,
            significance_level=0.10
        )

        # Should filter out MSFT (p-value too high)
        assert len(ranked) == 3
        assert ranked.iloc[0]['ticker'] == 'GOOGL'  # Highest alpha
        assert ranked.iloc[-1]['ticker'] == 'TSLA'  # Lowest alpha

        print(f"\n✓ Ranked {len(ranked)} stocks by alpha")
        print(ranked[['ticker', 'alpha', 'alpha_pvalue']])


class TestFactorFeatureEngineer:
    """Test factor feature engineering."""

    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance."""
        return FactorFeatureEngineer(model='5-factor')

    @pytest.fixture
    def sample_returns_df(self):
        """Create sample returns DataFrame."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)

        return pd.DataFrame({
            'AAPL': np.random.normal(0.0005, 0.01, len(dates)),
            'MSFT': np.random.normal(0.0004, 0.009, len(dates)),
            'GOOGL': np.random.normal(0.0006, 0.011, len(dates))
        }, index=dates)

    def test_create_features(self, engineer, sample_returns_df):
        """Test factor feature creation."""
        features = engineer.create_features(
            sample_returns_df,
            ['AAPL', 'MSFT', 'GOOGL'],
            include_factor_momentum=True,
            include_factor_spreads=True,
            include_cross_sectional=True
        )

        # Check basic factor features
        assert 'factor_alpha' in features.columns
        assert 'factor_beta_mkt' in features.columns
        assert 'factor_beta_smb' in features.columns
        assert 'factor_r_squared' in features.columns

        # Check cross-sectional features
        assert 'factor_rank_alpha' in features.columns

        assert len(features) == 3  # One row per ticker

        print(f"\n✓ Created {len(features.columns)} factor features")
        print(f"✓ Tickers: {list(features.index)}")
        print("\nFeature columns:")
        for col in sorted(features.columns):
            print(f"  - {col}")

    def test_factor_momentum(self, engineer):
        """Test factor momentum calculation."""
        momentum = engineer._calculate_factor_momentum('2023-01-01', '2023-12-31')

        assert not momentum.empty
        assert 'factor_momentum_smb_short' in momentum.columns
        assert 'factor_momentum_hml_long' in momentum.columns

        print(f"\n✓ Calculated factor momentum: {momentum.shape}")
        print(f"✓ Recent momentum (last 5 days):")
        print(momentum.tail())

    def test_factor_spreads(self, engineer):
        """Test factor spread calculation."""
        spreads = engineer._calculate_factor_spreads('2023-01-01', '2023-12-31')

        assert not spreads.empty
        assert 'factor_spread_size_short' in spreads.columns
        assert 'factor_spread_value_long' in spreads.columns

        print(f"\n✓ Calculated factor spreads: {spreads.shape}")
        print(f"✓ Recent spreads (last 5 days):")
        print(spreads.tail())


class TestFactorIntegration:
    """Test integration with existing features."""

    def test_integrate_factor_features(self):
        """Test integrating factor features with existing features."""
        # Create dummy existing features
        existing_features = pd.DataFrame({
            'momentum_20': [0.05, 0.03, 0.07],
            'volatility_20': [0.15, 0.12, 0.18],
            'rsi_14': [65, 55, 70]
        }, index=['AAPL', 'MSFT', 'GOOGL'])

        # Create dummy returns
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)

        returns_df = pd.DataFrame({
            'AAPL': np.random.normal(0.0005, 0.01, len(dates)),
            'MSFT': np.random.normal(0.0004, 0.009, len(dates)),
            'GOOGL': np.random.normal(0.0006, 0.011, len(dates))
        }, index=dates)

        # Integrate
        combined = integrate_factor_features(
            existing_features,
            returns_df,
            ['AAPL', 'MSFT', 'GOOGL']
        )

        # Check that both sets of features are present
        assert 'momentum_20' in combined.columns  # Existing feature
        assert 'factor_alpha' in combined.columns  # New factor feature

        assert len(combined) == 3

        print(f"\n✓ Integrated features: {combined.shape}")
        print(f"✓ Total features: {len(combined.columns)}")
        print("\nSample combined features:")
        print(combined.head())


def test_create_factor_features_convenience():
    """Test convenience function for creating factor features."""
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)

    returns_df = pd.DataFrame({
        'AAPL': np.random.normal(0.0005, 0.01, len(dates)),
        'MSFT': np.random.normal(0.0004, 0.009, len(dates))
    }, index=dates)

    # Use convenience function
    features = create_factor_features(
        returns_df,
        ['AAPL', 'MSFT'],
        lookback_window=252,
        model='5-factor'
    )

    assert not features.empty
    assert len(features) == 2

    print(f"\n✓ Created factor features using convenience function")
    print(features)


if __name__ == '__main__':
    # Run tests with output
    pytest.main([__file__, '-v', '-s'])
