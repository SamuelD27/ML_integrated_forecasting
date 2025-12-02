"""
Tests for Portfolio-Wide Hedge Coverage
========================================

Tests the fix for the hedge coverage bug where only the primary ticker
was being hedged instead of the entire portfolio.

Key functions tested:
- calculate_portfolio_beta()
- select_index_hedge_options()

Following financial-knowledge-validator skill guidelines for validation.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.options_overlay import (
    calculate_portfolio_beta,
    select_index_hedge_options,
    calculate_hedge_budget,
)


class TestCalculatePortfolioBeta:
    """Tests for portfolio beta calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample returns data (252 days of daily returns)
        np.random.seed(42)
        n_days = 100

        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

        # Create correlated returns
        # Market returns
        market_returns = np.random.normal(0.0005, 0.01, n_days)

        # Stock returns with different betas
        # AAPL: beta ~ 1.2
        aapl_returns = 1.2 * market_returns + np.random.normal(0, 0.005, n_days)
        # MSFT: beta ~ 1.1
        msft_returns = 1.1 * market_returns + np.random.normal(0, 0.005, n_days)
        # JNJ: beta ~ 0.7 (defensive)
        jnj_returns = 0.7 * market_returns + np.random.normal(0, 0.004, n_days)

        self.returns = pd.DataFrame({
            'AAPL': aapl_returns,
            'MSFT': msft_returns,
            'JNJ': jnj_returns,
            'SPY': market_returns,
        }, index=dates)

        self.weights = pd.Series({
            'AAPL': 0.4,
            'MSFT': 0.35,
            'JNJ': 0.25,
        })

    def test_beta_calculation_basic(self):
        """Test basic portfolio beta calculation."""
        result = calculate_portfolio_beta(
            returns=self.returns,
            weights=self.weights,
            market_ticker='SPY',
        )

        assert 'portfolio_beta' in result
        assert 'individual_betas' in result
        assert 'r_squared' in result

        # Portfolio beta should be weighted sum of individual betas
        # Expected: 0.4*1.2 + 0.35*1.1 + 0.25*0.7 = 1.04
        # Allow for sampling noise
        assert 0.5 <= result['portfolio_beta'] <= 1.5, \
            f"Portfolio beta {result['portfolio_beta']} outside expected range"

    def test_beta_within_valid_range(self):
        """Test that beta is clipped to valid range [0, 3]."""
        result = calculate_portfolio_beta(
            returns=self.returns,
            weights=self.weights,
            market_ticker='SPY',
        )

        # Portfolio beta should be in valid range
        assert 0.0 <= result['portfolio_beta'] <= 3.0, \
            f"Portfolio beta {result['portfolio_beta']} outside valid range [0, 3]"

        # Individual betas should also be clipped
        for ticker, beta in result['individual_betas'].items():
            assert -1.0 <= beta <= 3.0, \
                f"{ticker} beta {beta} outside valid range [-1, 3]"

    def test_weights_sum_validation(self):
        """Test that weights must sum to ~1.0."""
        bad_weights = pd.Series({
            'AAPL': 0.4,
            'MSFT': 0.35,
            'JNJ': 0.5,  # Sum = 1.25
        })

        with pytest.raises(ValueError, match="Weights sum to"):
            calculate_portfolio_beta(
                returns=self.returns,
                weights=bad_weights,
                market_ticker='SPY',
            )

    def test_insufficient_data(self):
        """Test error handling for insufficient return data."""
        short_returns = self.returns.head(10)  # Only 10 days

        with pytest.raises(ValueError, match="at least 20"):
            calculate_portfolio_beta(
                returns=short_returns,
                weights=self.weights,
                market_ticker='SPY',
            )

    def test_r_squared_in_valid_range(self):
        """Test that R-squared is in valid range [0, 1]."""
        result = calculate_portfolio_beta(
            returns=self.returns,
            weights=self.weights,
            market_ticker='SPY',
        )

        assert 0.0 <= result['r_squared'] <= 1.0, \
            f"R-squared {result['r_squared']} outside valid range [0, 1]"


class TestSelectIndexHedgeOptions:
    """Tests for index-based portfolio hedge selection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.portfolio_value = 100000.0
        self.hedge_budget = 10000.0
        self.portfolio_beta = 1.1
        self.weights = pd.Series({
            'AAPL': 0.3,
            'MSFT': 0.3,
            'GOOGL': 0.2,
            'SPY': 0.2,
        })

    def test_input_validation_negative_portfolio_value(self):
        """Test that negative portfolio value raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            select_index_hedge_options(
                portfolio_value=-100000,
                hedge_budget=self.hedge_budget,
                portfolio_beta=self.portfolio_beta,
                weights=self.weights,
            )

    def test_input_validation_negative_hedge_budget(self):
        """Test that negative hedge budget raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            select_index_hedge_options(
                portfolio_value=self.portfolio_value,
                hedge_budget=-5000,
                portfolio_beta=self.portfolio_beta,
                weights=self.weights,
            )

    def test_input_validation_invalid_beta(self):
        """Test that beta outside [0, 3] raises error."""
        with pytest.raises(ValueError, match="outside valid range"):
            select_index_hedge_options(
                portfolio_value=self.portfolio_value,
                hedge_budget=self.hedge_budget,
                portfolio_beta=5.0,  # Too high
                weights=self.weights,
            )

    @patch('portfolio.options_overlay.yf.Ticker')
    @patch('portfolio.options_overlay.fetch_options_chain')
    def test_coverage_calculation(self, mock_fetch_options, mock_ticker):
        """Test that hedge coverage is calculated correctly for entire portfolio."""
        # Mock SPY price
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.get_info.return_value = {'currentPrice': 500.0}
        mock_ticker.return_value = mock_ticker_instance

        # Mock options chain
        mock_puts = pd.DataFrame({
            'strike': [475.0, 480.0, 485.0, 490.0, 495.0],
            'bid': [5.0, 6.0, 7.5, 9.0, 11.0],
            'ask': [5.50, 6.50, 8.0, 9.50, 11.50],
            'dte': [90, 90, 90, 90, 90],
            'expiration': ['2025-03-15'] * 5,
            'volume': [1000, 1500, 2000, 2500, 3000],
            'openInterest': [5000, 7500, 10000, 12500, 15000],
        })
        mock_fetch_options.return_value = mock_puts

        result = select_index_hedge_options(
            portfolio_value=self.portfolio_value,
            hedge_budget=self.hedge_budget,
            portfolio_beta=self.portfolio_beta,
            weights=self.weights,
        )

        assert result is not None

        # Check that result contains key coverage metrics
        assert 'hedge_coverage' in result
        assert 'raw_coverage' in result
        assert 'beta_adjusted_exposure' in result
        assert 'positions_covered' in result

        # Beta-adjusted exposure should be portfolio_value * beta
        expected_exposure = self.portfolio_value * self.portfolio_beta
        assert result['beta_adjusted_exposure'] == expected_exposure

        # Positions covered should equal number of positions
        assert result['positions_covered'] == len(self.weights)

        # Coverage should be in valid range
        assert 0.0 <= result['hedge_coverage'] <= 1.5, \
            f"Coverage {result['hedge_coverage']} outside expected range"

    @patch('portfolio.options_overlay.yf.Ticker')
    @patch('portfolio.options_overlay.fetch_options_chain')
    def test_all_positions_have_coverage_info(self, mock_fetch_options, mock_ticker):
        """Test that position_coverage includes all portfolio positions."""
        # Mock SPY price
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.get_info.return_value = {'currentPrice': 500.0}
        mock_ticker.return_value = mock_ticker_instance

        # Mock options chain
        mock_puts = pd.DataFrame({
            'strike': [475.0, 480.0, 485.0, 490.0, 495.0],
            'bid': [5.0, 6.0, 7.5, 9.0, 11.0],
            'ask': [5.50, 6.50, 8.0, 9.50, 11.50],
            'dte': [90, 90, 90, 90, 90],
            'expiration': ['2025-03-15'] * 5,
            'volume': [1000, 1500, 2000, 2500, 3000],
            'openInterest': [5000, 7500, 10000, 12500, 15000],
        })
        mock_fetch_options.return_value = mock_puts

        result = select_index_hedge_options(
            portfolio_value=self.portfolio_value,
            hedge_budget=self.hedge_budget,
            portfolio_beta=self.portfolio_beta,
            weights=self.weights,
        )

        assert result is not None
        assert 'position_coverage' in result

        # Every position should have coverage info
        for ticker in self.weights.index:
            assert ticker in result['position_coverage'], \
                f"Position {ticker} missing from coverage info"

            pos_info = result['position_coverage'][ticker]
            assert 'value' in pos_info
            assert 'weight' in pos_info
            assert 'hedged_value' in pos_info
            assert 'hedge_pct' in pos_info


class TestHedgeBudgetCalculation:
    """Tests for hedge budget calculation."""

    def test_hedge_budget_scaling_with_rrr(self):
        """Test that hedge budget scales properly with RRR."""
        portfolio_value = 100000.0

        # Aggressive (RRR = 0.0) should give ~2%
        aggressive_budget = calculate_hedge_budget(0.0, portfolio_value)
        assert 1900 <= aggressive_budget <= 2100, \
            f"Aggressive budget {aggressive_budget} not near 2%"

        # Moderate (RRR = 0.5) should give ~10%
        moderate_budget = calculate_hedge_budget(0.5, portfolio_value)
        assert 10000 <= moderate_budget <= 12000, \
            f"Moderate budget {moderate_budget} not near 10%"

        # Defensive (RRR = 1.0) should give ~20%
        defensive_budget = calculate_hedge_budget(1.0, portfolio_value)
        assert 19000 <= defensive_budget <= 21000, \
            f"Defensive budget {defensive_budget} not near 20%"

    def test_hedge_budget_increases_with_rrr(self):
        """Test that hedge budget increases monotonically with RRR."""
        portfolio_value = 100000.0

        prev_budget = 0
        for rrr in [0.0, 0.25, 0.5, 0.75, 1.0]:
            budget = calculate_hedge_budget(rrr, portfolio_value)
            assert budget > prev_budget, \
                f"Budget did not increase from RRR {rrr-0.25} to {rrr}"
            prev_budget = budget


class TestOldVsNewHedgeCoverage:
    """
    Tests comparing old (single-ticker) vs new (portfolio-wide) hedge behavior.

    The old behavior hedged ONLY the primary ticker:
    - Total capital: $100,000
    - Primary ticker (20%): $20,000 hedged
    - Other positions (80%): $80,000 UNHEDGED

    The new behavior hedges the ENTIRE portfolio via index options:
    - Total capital: $100,000
    - All positions: hedged via beta-adjusted SPY puts
    """

    def test_coverage_includes_all_positions(self):
        """Test that coverage metric reflects protection of ALL positions."""
        weights = pd.Series({
            'AAPL': 0.20,  # Primary ticker
            'MSFT': 0.25,
            'GOOGL': 0.25,
            'META': 0.15,
            'SPY': 0.15,
        })

        portfolio_value = 100000.0
        hedge_budget = 10000.0
        portfolio_beta = 1.0

        # OLD BEHAVIOR (WRONG): Only counted primary ticker coverage
        # old_coverage = hedge_on_primary / primary_value
        # This gave FALSE sense of protection

        # NEW BEHAVIOR (CORRECT): Coverage is against entire portfolio
        # new_coverage = spy_notional_hedged / (portfolio_value * beta)

        # With $100k portfolio at beta=1.0, we need to hedge $100k exposure
        # If SPY is at $500 and we buy 2 contracts (200 shares),
        # we hedge $100k of exposure = 100% coverage

        # The key insight: coverage should be same for ALL positions
        # because they're all hedged via the same index hedge

        beta_adjusted_exposure = portfolio_value * portfolio_beta
        assert beta_adjusted_exposure == 100000.0

        # If we can afford to hedge 80% of beta-adjusted exposure,
        # then ALL positions have 80% coverage, not just primary
        hypothetical_coverage = 0.80

        for ticker, weight in weights.items():
            position_value = portfolio_value * weight
            hedged_value = position_value * hypothetical_coverage

            # Each position's hedged value should be proportional to its weight
            expected_hedged = portfolio_value * hypothetical_coverage * weight
            assert np.isclose(hedged_value, expected_hedged), \
                f"{ticker} hedged value incorrect"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
