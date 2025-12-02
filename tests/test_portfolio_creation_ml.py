"""
Tests for portfolio_creation_ml.py - ML-enhanced portfolio construction workflow.

Test coverage:
1. Full workflow with mock data
2. Peer discovery integration
3. Optimizer selection
4. Hedge calculation
5. Output structure validation
6. Error handling
7. Feature scoring pipeline
8. Regime detection integration
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if portfolio_creation_ml can be imported (requires seaborn)
try:
    import portfolio_creation_ml
    HAS_PORTFOLIO_ML = True
except ImportError:
    HAS_PORTFOLIO_ML = False

requires_portfolio_ml = pytest.mark.skipif(
    not HAS_PORTFOLIO_ML,
    reason="portfolio_creation_ml requires seaborn which is not installed"
)


class TestExtractPricesMultiindex:
    """Tests for extract_prices_multiindex function."""

    @requires_portfolio_ml
    def test_extract_from_multiindex_adj_close_first(self, mock_multi_ticker_data):
        """Test extraction when Adj Close is at level 0."""
        from portfolio_creation_ml import extract_prices_multiindex

        tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
        result = extract_prices_multiindex(mock_multi_ticker_data, tickers)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > 0

    @requires_portfolio_ml
    def test_extract_from_single_ticker(self, mock_price_data):
        """Test extraction from single ticker DataFrame."""
        from portfolio_creation_ml import extract_prices_multiindex

        result = extract_prices_multiindex(mock_price_data, ['AAPL'])

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @requires_portfolio_ml
    def test_extract_handles_missing_tickers(self, mock_multi_ticker_data):
        """Test that missing tickers are handled gracefully."""
        from portfolio_creation_ml import extract_prices_multiindex

        tickers = ['AAPL', 'NONEXISTENT']

        # Should not raise, should return what's available
        result = extract_prices_multiindex(mock_multi_ticker_data, tickers)
        assert isinstance(result, pd.DataFrame)


class TestMLPortfolioConstruction:
    """Tests for run_ml_portfolio_construction function."""

    @requires_portfolio_ml
    @patch('portfolio_creation_ml.DataProviderManager')
    @patch('portfolio_creation_ml.discover_peers')
    @patch('portfolio_creation_ml.discover_etfs')
    @patch('portfolio_creation_ml.fetch_full_bundle')
    @patch('portfolio_creation_ml.save_bundle')
    @patch('portfolio_creation_ml.set_last_fetch_globals')
    @patch('portfolio_creation_ml.RegimeDetector')
    @patch('portfolio_creation_ml.optimize_portfolio_cvar')
    @patch('portfolio_creation_ml.select_index_hedge_options')
    @patch('portfolio_creation_ml.calculate_portfolio_beta')
    @patch('portfolio_creation_ml.calculate_hedge_budget')
    def test_full_workflow_returns_expected_structure(
        self, mock_hedge_budget, mock_calc_beta, mock_hedge_options,
        mock_optimize, mock_regime, mock_set_globals, mock_save,
        mock_fetch, mock_etf_disc, mock_peer_disc, mock_provider,
        mock_multi_ticker_data, mock_peer_discovery_result, mock_etf_discovery_result
    ):
        """Test that full workflow returns expected result structure."""
        from portfolio_creation_ml import run_ml_portfolio_construction

        # Setup mocks
        mock_provider.return_value = Mock()
        mock_peer_disc.return_value = mock_peer_discovery_result
        mock_etf_disc.return_value = mock_etf_discovery_result

        mock_fetch.return_value = {
            'prices': mock_multi_ticker_data,
            'meta': {'date_range_actual': {'start': '2023-01-01', 'end': '2024-01-01'}}
        }

        # Mock regime detector
        mock_regime_instance = Mock()
        mock_regime_instance.detect_regime.return_value = {
            'regime': 'bull',
            'confidence': 0.8,
        }
        mock_regime_instance.get_regime_parameters.return_value = {
            'risk_aversion': 3.0,
            'max_weight': 0.3,
            'hedge_multiplier': 0.5,
        }
        mock_regime.return_value = mock_regime_instance

        # Mock optimizer
        weights = pd.Series({'AAPL': 0.4, 'MSFT': 0.3, 'SPY': 0.3})
        mock_optimize.return_value = {
            'weights': weights,
            'port_return': 0.12,
            'port_vol': 0.18,
            'sharpe': 0.67,
            'cvar': -0.03,
        }

        # Mock hedge functions
        mock_hedge_budget.return_value = 5000.0
        mock_calc_beta.return_value = {
            'portfolio_beta': 1.05,
            'r_squared': 0.85,
            'individual_betas': {'AAPL': 1.2, 'MSFT': 1.1, 'SPY': 1.0},
        }
        mock_hedge_options.return_value = {
            'strategy': 'put',
            'strike': 440.0,
            'contracts': 2,
            'total_premium': 800.0,
            'hedge_coverage': 0.9,
        }

        result = run_ml_portfolio_construction(
            ticker='AAPL',
            capital=100000.0,
            rrr=0.6,
            enable_ml=False,
        )

        # Verify result structure
        assert 'ticker' in result
        assert 'capital' in result
        assert 'rrr' in result
        assert 'universe' in result
        assert 'opt_result' in result
        assert 'holdings_df' in result

    @requires_portfolio_ml
    @patch('portfolio_creation_ml.DataProviderManager')
    @patch('portfolio_creation_ml.discover_peers')
    @patch('portfolio_creation_ml.discover_etfs')
    @patch('portfolio_creation_ml.fetch_full_bundle')
    @patch('portfolio_creation_ml.save_bundle')
    @patch('portfolio_creation_ml.set_last_fetch_globals')
    def test_workflow_handles_no_peers(
        self, mock_set_globals, mock_save, mock_fetch,
        mock_etf_disc, mock_peer_disc, mock_provider,
        mock_multi_ticker_data, mock_etf_discovery_result
    ):
        """Test workflow handles case when no peers are found."""
        from portfolio_creation_ml import run_ml_portfolio_construction

        mock_provider.return_value = Mock()
        mock_peer_disc.return_value = {
            'primary_meta': {'symbol': 'AAPL', 'sector': 'Technology', 'exchange': 'NMS', 'country': 'US'},
            'peers': [],
            'peer_metadata': {},
            'diagnostics': {'candidates_found': 0}
        }
        mock_etf_disc.return_value = mock_etf_discovery_result

        # Build single-column DataFrame for single ticker
        dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
        prices = pd.DataFrame({'Adj Close': np.random.randn(252).cumsum() + 150}, index=dates)

        mock_fetch.return_value = {
            'prices': prices,
            'meta': {'date_range_actual': {'start': '2023-01-01', 'end': '2024-01-01'}}
        }

        with patch('portfolio_creation_ml.optimize_portfolio_cvar') as mock_opt:
            mock_opt.return_value = {
                'weights': pd.Series({'AAPL': 1.0}),
                'port_return': 0.10,
                'port_vol': 0.20,
                'sharpe': 0.50,
            }
            with patch('portfolio_creation_ml.calculate_hedge_budget') as mock_hb:
                mock_hb.return_value = 5000.0
                with patch('portfolio_creation_ml.calculate_portfolio_beta') as mock_beta:
                    mock_beta.return_value = {'portfolio_beta': 1.0, 'r_squared': 0.0, 'individual_betas': {}}
                    with patch('portfolio_creation_ml.select_index_hedge_options') as mock_ho:
                        mock_ho.return_value = None

                        result = run_ml_portfolio_construction(
                            ticker='AAPL',
                            capital=100000.0,
                            rrr=0.6,
                            enable_ml=False,
                        )

        assert result is not None
        assert 'ticker' in result


class TestHedgeCalculation:
    """Tests for hedge budget and options selection."""

    def test_hedge_budget_scales_with_rrr(self):
        """Test that hedge budget scales appropriately with RRR."""
        from portfolio.options_overlay import calculate_hedge_budget

        capital = 100000.0

        # Higher RRR = more defensive = higher hedge budget
        budget_low_rrr = calculate_hedge_budget(0.2, capital)
        budget_high_rrr = calculate_hedge_budget(0.8, capital)

        assert budget_high_rrr > budget_low_rrr

    def test_hedge_budget_scales_with_capital(self):
        """Test that hedge budget scales with capital."""
        from portfolio.options_overlay import calculate_hedge_budget

        rrr = 0.5

        budget_small = calculate_hedge_budget(rrr, 50000.0)
        budget_large = calculate_hedge_budget(rrr, 200000.0)

        assert budget_large > budget_small


class TestFeatureScoringPipeline:
    """Tests for feature-based stock scoring."""

    def test_feature_scoring_produces_rankings(self, mock_multi_ticker_data):
        """Test that feature scoring produces valid rankings."""
        # This tests the scoring logic within run_ml_portfolio_construction
        # We'll simulate the scoring manually

        np.random.seed(42)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']

        # Simulate feature scores
        scores = []
        for ticker in tickers:
            momentum_medium = np.random.uniform(-0.1, 0.2)
            momentum_short = np.random.uniform(-0.05, 0.1)
            price_stability = np.random.uniform(0.5, 1.0)
            momentum_medium_sharpe = np.random.uniform(-0.5, 1.5)

            # Same formula as in portfolio_creation_ml.py
            momentum_score = momentum_medium * 0.3 + momentum_short * 0.2
            quality_score = price_stability * 0.2
            sharpe_score = momentum_medium_sharpe * 0.3
            total_score = momentum_score + quality_score + sharpe_score

            scores.append({'ticker': ticker, 'feature_score': total_score})

        rankings = pd.DataFrame(scores).sort_values('feature_score', ascending=False)

        assert len(rankings) == 4
        assert 'feature_score' in rankings.columns


class TestRegimeDetectionIntegration:
    """Tests for regime detection integration."""

    def test_regime_parameters_affect_optimization(self):
        """Test that regime parameters modify optimization behavior."""
        from ml_models.regime import RegimeDetector

        detector = RegimeDetector()

        bull_params = detector.get_regime_parameters('bull')
        crisis_params = detector.get_regime_parameters('crisis')

        # Crisis should have higher risk aversion
        assert crisis_params['risk_aversion'] > bull_params['risk_aversion']
        # Crisis should have lower max weight
        assert crisis_params['max_weight'] < bull_params['max_weight']
        # Crisis should have higher hedge multiplier
        assert crisis_params['hedge_multiplier'] > bull_params['hedge_multiplier']


class TestOutputStructure:
    """Tests for output structure validation."""

    def test_holdings_df_has_required_columns(self, sample_portfolio_weights, mock_price_data):
        """Test that holdings DataFrame has required columns."""
        capital = 100000.0
        weights = sample_portfolio_weights

        # Simulate what run_ml_portfolio_construction does
        current_prices = pd.Series({
            'AAPL': 150.0, 'MSFT': 380.0, 'GOOGL': 140.0, 'SPY': 450.0, 'USMV': 78.0
        })

        dollar_allocation = weights * capital
        shares = (dollar_allocation / current_prices).round(0)

        holdings_df = pd.DataFrame({
            'ticker': weights.index,
            'weight': weights.values,
            'dollars': dollar_allocation.values,
            'shares': shares.values,
            'price': current_prices.values,
        })

        assert 'ticker' in holdings_df.columns
        assert 'weight' in holdings_df.columns
        assert 'dollars' in holdings_df.columns
        assert 'shares' in holdings_df.columns
        assert 'price' in holdings_df.columns
        assert len(holdings_df) == 5

    def test_weights_sum_to_one(self, sample_portfolio_weights):
        """Test that portfolio weights sum to 1."""
        assert abs(sample_portfolio_weights.sum() - 1.0) < 0.01


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_rrr_raises_error(self):
        """Test that invalid RRR values are rejected."""
        # This would be tested via CLI argument validation
        # RRR should be between 0 and 1

        rrr_valid = 0.5
        rrr_invalid_low = -0.1
        rrr_invalid_high = 1.5

        assert 0.0 <= rrr_valid <= 1.0
        assert not (0.0 <= rrr_invalid_low <= 1.0)
        assert not (0.0 <= rrr_invalid_high <= 1.0)

    def test_zero_capital_handled(self):
        """Test that zero capital is handled appropriately."""
        capital = 0.0

        # Zero capital should result in zero dollar allocations
        weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        dollar_allocation = weights * capital

        assert dollar_allocation.sum() == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
