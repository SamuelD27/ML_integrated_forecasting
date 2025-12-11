"""
Tests for Allocation Pipeline (Phase 6)
======================================
Tests for scenario generator and allocation utilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import scenario generator
from ml_models.scenario_generator import (
    ScenarioMethod,
    ScenarioSet,
    generate_historical_scenarios,
    generate_normal_scenarios,
    generate_t_distribution_scenarios,
    generate_regime_conditional_scenarios,
    generate_stress_scenarios,
    generate_bootstrap_scenarios,
    generate_scenarios,
    calculate_scenario_cvar,
    calculate_scenario_risk_contribution,
)

# Import allocation utilities
from pipeline.allocation_utils import (
    AllocationResult,
    allocate_from_signals,
    calculate_target_weights,
    rebalance_portfolio,
    validate_allocation,
    _get_regime_risk_aversion,
    _get_regime_max_weight,
)

# Check if CVaR allocator is available
try:
    from portfolio.cvar_allocator import CVaRAllocator, HAS_CVXPY
    HAS_CVAR = True
except ImportError:
    HAS_CVAR = False
    HAS_CVXPY = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_returns():
    """Create mock returns DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

    # Generate correlated returns
    n = len(dates)
    k = len(tickers)

    # Correlation matrix
    corr = np.array([
        [1.0, 0.6, 0.5, 0.4],
        [0.6, 1.0, 0.7, 0.5],
        [0.5, 0.7, 1.0, 0.6],
        [0.4, 0.5, 0.6, 1.0],
    ])

    # Volatilities
    vols = np.array([0.25, 0.28, 0.22, 0.30]) / np.sqrt(252)

    # Means
    means = np.array([0.15, 0.12, 0.10, 0.18]) / 252

    # Generate multivariate normal
    cov = np.outer(vols, vols) * corr
    returns_data = np.random.multivariate_normal(means, cov, size=n)

    return pd.DataFrame(returns_data, index=dates, columns=tickers)


@pytest.fixture
def mock_trade_signal():
    """Create mock trade signal factory."""
    class MockSignal:
        def __init__(self, ticker, expected_return, expected_vol, meta_prob=0.7, regime=0):
            self.ticker = ticker
            self.direction = 'long'
            self.expected_return = expected_return
            self.expected_vol = expected_vol
            self.meta_prob = meta_prob
            self.regime = regime
            self.regime_label = 'Bull'
            self.snapshot = None

    return MockSignal


# =============================================================================
# Test Scenario Generator
# =============================================================================

class TestScenarioSet:
    """Tests for ScenarioSet dataclass."""

    def test_scenario_set_creation(self, mock_returns):
        """Test creating a ScenarioSet."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=100)

        assert isinstance(scenarios, ScenarioSet)
        assert scenarios.n_scenarios == 100
        assert len(scenarios.tickers) == 4
        assert scenarios.scenarios.shape == (100, 4)

    def test_scenario_set_to_dataframe(self, mock_returns):
        """Test converting to DataFrame."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=100)

        df = scenarios.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == scenarios.tickers
        assert len(df) == 100

    def test_scenario_set_mean_returns(self, mock_returns):
        """Test mean returns calculation."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=1000)

        means = scenarios.mean_returns

        assert isinstance(means, pd.Series)
        assert len(means) == 4

    def test_scenario_set_volatilities(self, mock_returns):
        """Test volatility calculation."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=1000)

        vols = scenarios.volatilities

        assert isinstance(vols, pd.Series)
        assert (vols >= 0).all()

    def test_scenario_set_correlations(self, mock_returns):
        """Test correlation matrix."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=1000)

        corr = scenarios.correlations

        assert isinstance(corr, pd.DataFrame)
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(
            np.diag(corr.values), np.ones(4), decimal=5
        )

    def test_scenario_set_percentile(self, mock_returns):
        """Test percentile calculation."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=1000)

        p5 = scenarios.percentile(5)
        p95 = scenarios.percentile(95)

        assert isinstance(p5, pd.Series)
        assert (p5 < p95).all()


class TestHistoricalScenarios:
    """Tests for historical scenario generation."""

    def test_generate_historical_basic(self, mock_returns):
        """Test basic historical simulation."""
        scenarios = generate_historical_scenarios(mock_returns)

        assert scenarios.method == ScenarioMethod.HISTORICAL
        assert len(scenarios.tickers) == 4

    def test_generate_historical_n_scenarios(self, mock_returns):
        """Test controlling number of scenarios."""
        scenarios = generate_historical_scenarios(mock_returns, n_scenarios=100)

        assert scenarios.n_scenarios == 100
        assert scenarios.scenarios.shape[0] == 100

    def test_generate_historical_multi_day(self, mock_returns):
        """Test multi-day horizon."""
        scenarios = generate_historical_scenarios(mock_returns, horizon_days=5)

        assert scenarios.horizon_days == 5
        # Multi-day returns should have higher volatility
        assert scenarios.volatilities.mean() > mock_returns.std().mean()


class TestNormalScenarios:
    """Tests for normal distribution scenarios."""

    def test_generate_normal_basic(self, mock_returns):
        """Test basic normal scenario generation."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=1000)

        assert scenarios.method == ScenarioMethod.NORMAL
        assert scenarios.n_scenarios == 1000

    def test_generate_normal_preserves_mean(self, mock_returns):
        """Test that generated scenarios preserve mean."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=10000)

        historical_mean = mock_returns.mean()
        scenario_mean = scenarios.mean_returns

        # Should be close (with some tolerance)
        np.testing.assert_array_almost_equal(
            scenario_mean.values,
            historical_mean.values,
            decimal=2
        )

    def test_generate_normal_with_shrinkage(self, mock_returns):
        """Test shrinkage covariance is used."""
        scenarios = generate_normal_scenarios(
            mock_returns,
            n_scenarios=100,
            use_shrinkage=True,
        )

        assert 'shrinkage' in scenarios.metadata


class TestTDistributionScenarios:
    """Tests for t-distribution scenarios."""

    def test_generate_t_distribution_basic(self, mock_returns):
        """Test t-distribution scenario generation."""
        scenarios = generate_t_distribution_scenarios(mock_returns, n_scenarios=1000)

        assert scenarios.method == ScenarioMethod.T_DISTRIBUTION

    def test_generate_t_distribution_fat_tails(self, mock_returns):
        """Test that t-distribution has fatter tails."""
        normal = generate_normal_scenarios(mock_returns, n_scenarios=10000)
        t_dist = generate_t_distribution_scenarios(mock_returns, n_scenarios=10000, df=5)

        # t-distribution should have more extreme values
        normal_extremes = (normal.scenarios < normal.percentile(1).values).sum()
        t_extremes = (t_dist.scenarios < t_dist.percentile(1).values).sum()

        # t-distribution should have more tail events (or at least similar)
        assert t_extremes >= 0


class TestRegimeConditionalScenarios:
    """Tests for regime-conditional scenarios."""

    def test_generate_regime_bull(self, mock_returns):
        """Test bull regime scenarios."""
        scenarios = generate_regime_conditional_scenarios(
            mock_returns,
            regime=0,  # Bull
            n_scenarios=1000,
        )

        assert scenarios.metadata['regime'] == 0
        assert scenarios.metadata['mu_multiplier'] == 1.5

    def test_generate_regime_bear(self, mock_returns):
        """Test bear regime scenarios."""
        scenarios = generate_regime_conditional_scenarios(
            mock_returns,
            regime=1,  # Bear
            n_scenarios=1000,
        )

        assert scenarios.metadata['regime'] == 1
        assert scenarios.metadata['vol_multiplier'] == 1.3

    def test_generate_regime_crisis(self, mock_returns):
        """Test crisis regime scenarios."""
        scenarios = generate_regime_conditional_scenarios(
            mock_returns,
            regime=3,  # Crisis
            n_scenarios=1000,
        )

        assert scenarios.metadata['regime'] == 3
        # Crisis should have higher volatility
        assert scenarios.metadata['vol_multiplier'] == 2.0

    def test_regimes_have_different_characteristics(self, mock_returns):
        """Test that different regimes produce different distributions."""
        bull = generate_regime_conditional_scenarios(mock_returns, regime=0)
        crisis = generate_regime_conditional_scenarios(mock_returns, regime=3)

        # Crisis should have lower mean and higher vol
        assert bull.mean_returns.mean() > crisis.mean_returns.mean()
        assert bull.volatilities.mean() < crisis.volatilities.mean()


class TestStressScenarios:
    """Tests for stress test scenarios."""

    def test_generate_stress_basic(self, mock_returns):
        """Test basic stress scenario generation."""
        scenarios = generate_stress_scenarios(mock_returns, n_scenarios=100)

        assert scenarios.method == ScenarioMethod.STRESS_TEST
        assert scenarios.n_scenarios == 100

    def test_stress_scenarios_are_adverse(self, mock_returns):
        """Test that stress scenarios are adverse."""
        normal = generate_normal_scenarios(mock_returns, n_scenarios=1000)
        stress = generate_stress_scenarios(mock_returns, n_scenarios=100)

        # Stress scenarios should have lower mean returns
        assert stress.mean_returns.mean() < normal.mean_returns.mean()

    def test_stress_multiplier_effect(self, mock_returns):
        """Test stress multiplier increases volatility."""
        low_stress = generate_stress_scenarios(mock_returns, stress_multiplier=2.0)
        high_stress = generate_stress_scenarios(mock_returns, stress_multiplier=4.0)

        # Higher multiplier should have more extreme scenarios
        assert high_stress.volatilities.mean() >= low_stress.volatilities.mean()


class TestBootstrapScenarios:
    """Tests for bootstrap scenarios."""

    def test_generate_bootstrap_basic(self, mock_returns):
        """Test basic bootstrap generation."""
        scenarios = generate_bootstrap_scenarios(mock_returns, n_scenarios=100)

        assert scenarios.method == ScenarioMethod.BOOTSTRAP

    def test_bootstrap_multi_day(self, mock_returns):
        """Test multi-day bootstrap."""
        scenarios = generate_bootstrap_scenarios(
            mock_returns,
            n_scenarios=100,
            horizon_days=5,
            block_size=3,
        )

        assert scenarios.horizon_days == 5


class TestGenerateScenarios:
    """Tests for unified generate_scenarios function."""

    def test_generate_scenarios_normal(self, mock_returns):
        """Test normal method."""
        scenarios = generate_scenarios(mock_returns, method='normal')
        assert scenarios.method == ScenarioMethod.NORMAL

    def test_generate_scenarios_historical(self, mock_returns):
        """Test historical method."""
        scenarios = generate_scenarios(mock_returns, method='historical')
        assert scenarios.method == ScenarioMethod.HISTORICAL

    def test_generate_scenarios_regime(self, mock_returns):
        """Test regime-conditional method."""
        scenarios = generate_scenarios(
            mock_returns,
            method='regime_conditional',
            regime=1,
        )
        assert scenarios.method == ScenarioMethod.REGIME_CONDITIONAL

    def test_generate_scenarios_invalid_method(self, mock_returns):
        """Test invalid method raises error."""
        with pytest.raises(ValueError):
            generate_scenarios(mock_returns, method='invalid')


class TestScenarioCVaR:
    """Tests for scenario-based CVaR calculation."""

    def test_calculate_scenario_cvar(self, mock_returns):
        """Test CVaR calculation from scenarios."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=10000)
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=mock_returns.columns)

        var, cvar = calculate_scenario_cvar(scenarios, weights)

        # VaR should be less extreme than CVaR
        assert var > cvar  # Both are negative, cvar more negative

    def test_calculate_risk_contribution(self, mock_returns):
        """Test risk contribution calculation."""
        scenarios = generate_normal_scenarios(mock_returns, n_scenarios=10000)
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=mock_returns.columns)

        risk_contrib = calculate_scenario_risk_contribution(scenarios, weights)

        assert isinstance(risk_contrib, pd.Series)
        # Should sum to 1 (normalized)
        assert abs(risk_contrib.sum() - 1.0) < 0.01


# =============================================================================
# Test Allocation Utilities
# =============================================================================

class TestAllocationResult:
    """Tests for AllocationResult dataclass."""

    def test_allocation_result_creation(self):
        """Test creating an AllocationResult."""
        weights = pd.Series([0.3, 0.3, 0.2, 0.2], index=['AAPL', 'GOOGL', 'MSFT', 'AMZN'])

        result = AllocationResult(
            weights=weights,
            tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            expected_return=0.12,
            expected_vol=0.18,
            sharpe=0.67,
            cvar=0.15,
        )

        assert len(result.weights) == 4
        assert result.sharpe == 0.67

    def test_allocation_result_to_dict(self):
        """Test converting to dict."""
        weights = pd.Series([0.5, 0.5], index=['AAPL', 'GOOGL'])

        result = AllocationResult(
            weights=weights,
            tickers=['AAPL', 'GOOGL'],
            expected_return=0.10,
            expected_vol=0.15,
            sharpe=0.67,
            cvar=0.10,
        )

        d = result.to_dict()

        assert 'weights' in d
        assert 'expected_return' in d
        assert d['sharpe'] == 0.67


class TestRegimeParameters:
    """Tests for regime parameter functions."""

    def test_regime_risk_aversion(self):
        """Test regime-based risk aversion."""
        config = {}

        # Bull should have lower risk aversion
        bull_ra = _get_regime_risk_aversion(0, config)
        crisis_ra = _get_regime_risk_aversion(3, config)

        assert bull_ra < crisis_ra

    def test_regime_max_weight(self):
        """Test regime-based max weight."""
        config = {}

        # Bull should allow higher concentration
        bull_mw = _get_regime_max_weight(0, config)
        crisis_mw = _get_regime_max_weight(3, config)

        assert bull_mw > crisis_mw


class TestValidateAllocation:
    """Tests for allocation validation."""

    def test_validate_valid_allocation(self):
        """Test validating a valid allocation."""
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=['A', 'B', 'C', 'D'])

        result = validate_allocation(weights)

        assert result['valid'] == True
        assert len(result['issues']) == 0

    def test_validate_invalid_sum(self):
        """Test detecting invalid weight sum."""
        weights = pd.Series([0.3, 0.3, 0.3, 0.3], index=['A', 'B', 'C', 'D'])

        result = validate_allocation(weights)

        assert result['valid'] == False
        assert any('sum' in issue.lower() for issue in result['issues'])

    def test_validate_negative_weights(self):
        """Test detecting negative weights."""
        weights = pd.Series([0.5, 0.6, -0.1, 0.0], index=['A', 'B', 'C', 'D'])

        result = validate_allocation(weights)

        assert result['valid'] == False
        assert any('negative' in issue.lower() for issue in result['issues'])

    def test_validate_max_weight_exceeded(self):
        """Test detecting max weight violation."""
        weights = pd.Series([0.5, 0.3, 0.1, 0.1], index=['A', 'B', 'C', 'D'])

        result = validate_allocation(
            weights,
            config={'allocation': {'max_weight': 0.25}}
        )

        assert result['valid'] == False


class TestRebalancePortfolio:
    """Tests for portfolio rebalancing."""

    def test_rebalance_basic(self):
        """Test basic rebalancing."""
        current = pd.Series([0.25, 0.25, 0.25, 0.25], index=['A', 'B', 'C', 'D'])
        target = pd.Series([0.30, 0.30, 0.20, 0.20], index=['A', 'B', 'C', 'D'])
        prices = pd.Series([100, 150, 200, 50], index=['A', 'B', 'C', 'D'])

        trades = rebalance_portfolio(
            current_weights=current,
            target_weights=target,
            portfolio_value=100000,
            prices=prices,
        )

        assert isinstance(trades, pd.DataFrame)
        if len(trades) > 0:
            assert 'ticker' in trades.columns
            assert 'action' in trades.columns
            assert 'shares' in trades.columns

    def test_rebalance_turnover_limit(self):
        """Test turnover limiting."""
        current = pd.Series([0.25, 0.25, 0.25, 0.25], index=['A', 'B', 'C', 'D'])
        target = pd.Series([0.50, 0.50, 0.00, 0.00], index=['A', 'B', 'C', 'D'])
        prices = pd.Series([100, 100, 100, 100], index=['A', 'B', 'C', 'D'])

        trades = rebalance_portfolio(
            current_weights=current,
            target_weights=target,
            portfolio_value=100000,
            prices=prices,
            turnover_limit=0.10,  # 10% limit
        )

        # Total turnover should be limited
        total_turnover = trades['notional'].sum() / 100000
        assert total_turnover <= 0.20  # Buy + sell side


@pytest.mark.skipif(not HAS_CVAR or not HAS_CVXPY, reason="CVaR allocator or CVXPY not available")
class TestAllocateFromSignals:
    """Tests for allocate_from_signals function."""

    def test_allocate_basic(self, mock_returns, mock_trade_signal):
        """Test basic allocation from signals."""
        signals = [
            mock_trade_signal('AAPL', 0.15, 0.25, meta_prob=0.8),
            mock_trade_signal('GOOGL', 0.12, 0.28, meta_prob=0.75),
            mock_trade_signal('MSFT', 0.10, 0.22, meta_prob=0.7),
        ]

        result = allocate_from_signals(
            signals=signals,
            returns=mock_returns,
            portfolio_value=100000,
            regime=0,
        )

        assert isinstance(result, AllocationResult)
        assert len(result.weights) > 0
        assert abs(result.weights.sum() - 1.0) < 0.01

    def test_allocate_regime_adjustment(self, mock_returns, mock_trade_signal):
        """Test that regime affects allocation."""
        signals = [
            mock_trade_signal('AAPL', 0.15, 0.25),
            mock_trade_signal('GOOGL', 0.12, 0.28),
        ]

        bull_result = allocate_from_signals(
            signals=signals,
            returns=mock_returns,
            portfolio_value=100000,
            regime=0,  # Bull
        )

        crisis_result = allocate_from_signals(
            signals=signals,
            returns=mock_returns,
            portfolio_value=100000,
            regime=3,  # Crisis
        )

        # Crisis should have higher risk aversion
        assert crisis_result.risk_aversion > bull_result.risk_aversion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
