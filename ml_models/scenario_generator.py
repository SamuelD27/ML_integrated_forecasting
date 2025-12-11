"""
Scenario Generator
==================
Generates return scenarios for portfolio optimization and stress testing.

Supports multiple scenario generation methods:
1. Historical simulation
2. Parametric Monte Carlo (normal/t-distribution)
3. Regime-conditional scenarios
4. Stress test scenarios (tail events)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Try importing scipy for distribution fitting
try:
    from scipy import stats
    from scipy.stats import t as t_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Try importing sklearn for covariance
try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ScenarioMethod(Enum):
    """Scenario generation methods."""
    HISTORICAL = 'historical'
    NORMAL = 'normal'
    T_DISTRIBUTION = 't_distribution'
    REGIME_CONDITIONAL = 'regime_conditional'
    STRESS_TEST = 'stress_test'
    BOOTSTRAP = 'bootstrap'


@dataclass
class ScenarioSet:
    """Collection of generated scenarios."""
    scenarios: np.ndarray  # (n_scenarios, n_assets)
    tickers: List[str]
    method: ScenarioMethod
    n_scenarios: int
    horizon_days: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame(self.scenarios, columns=self.tickers)

    @property
    def mean_returns(self) -> pd.Series:
        """Mean returns across scenarios."""
        return pd.Series(np.mean(self.scenarios, axis=0), index=self.tickers)

    @property
    def volatilities(self) -> pd.Series:
        """Volatility (std) across scenarios."""
        return pd.Series(np.std(self.scenarios, axis=0), index=self.tickers)

    @property
    def correlations(self) -> pd.DataFrame:
        """Correlation matrix across scenarios."""
        return pd.DataFrame(
            np.corrcoef(self.scenarios.T),
            index=self.tickers,
            columns=self.tickers,
        )

    def percentile(self, pct: float) -> pd.Series:
        """Get percentile returns."""
        return pd.Series(
            np.percentile(self.scenarios, pct, axis=0),
            index=self.tickers,
        )


def generate_historical_scenarios(
    returns: pd.DataFrame,
    n_scenarios: Optional[int] = None,
    horizon_days: int = 1,
    with_replacement: bool = True,
) -> ScenarioSet:
    """
    Generate scenarios using historical returns.

    Args:
        returns: Historical returns DataFrame
        n_scenarios: Number of scenarios (default: use all historical)
        horizon_days: Holding period in days
        with_replacement: Sample with replacement (bootstrap)

    Returns:
        ScenarioSet with historical scenarios
    """
    tickers = returns.columns.tolist()
    n_obs = len(returns)

    if n_scenarios is None:
        n_scenarios = n_obs

    # For multi-day horizon, compound returns
    if horizon_days > 1:
        # Rolling compound returns
        compound_returns = returns.rolling(window=horizon_days).apply(
            lambda x: (1 + x).prod() - 1,
            raw=True
        ).dropna()

        scenarios = compound_returns.values
    else:
        scenarios = returns.values

    # Sample scenarios
    if with_replacement or n_scenarios > len(scenarios):
        indices = np.random.choice(len(scenarios), size=n_scenarios, replace=True)
    else:
        indices = np.random.choice(len(scenarios), size=n_scenarios, replace=False)

    sampled = scenarios[indices]

    return ScenarioSet(
        scenarios=sampled,
        tickers=tickers,
        method=ScenarioMethod.HISTORICAL,
        n_scenarios=n_scenarios,
        horizon_days=horizon_days,
        metadata={
            'source_obs': n_obs,
            'with_replacement': with_replacement,
        },
    )


def generate_normal_scenarios(
    returns: pd.DataFrame,
    n_scenarios: int = 1000,
    horizon_days: int = 1,
    use_shrinkage: bool = True,
) -> ScenarioSet:
    """
    Generate scenarios from multivariate normal distribution.

    Args:
        returns: Historical returns for fitting
        n_scenarios: Number of scenarios
        horizon_days: Holding period in days
        use_shrinkage: Use Ledoit-Wolf shrinkage for covariance

    Returns:
        ScenarioSet with normal scenarios
    """
    tickers = returns.columns.tolist()
    n_assets = len(tickers)

    # Calculate mean
    mu_daily = returns.mean().values

    # Calculate covariance
    if use_shrinkage and HAS_SKLEARN:
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_daily = lw.covariance_
        shrinkage = lw.shrinkage_
    else:
        cov_daily = returns.cov().values
        shrinkage = 0.0

    # Scale to horizon
    mu_horizon = mu_daily * horizon_days
    cov_horizon = cov_daily * horizon_days

    # Generate scenarios
    scenarios = np.random.multivariate_normal(
        mean=mu_horizon,
        cov=cov_horizon,
        size=n_scenarios,
    )

    return ScenarioSet(
        scenarios=scenarios,
        tickers=tickers,
        method=ScenarioMethod.NORMAL,
        n_scenarios=n_scenarios,
        horizon_days=horizon_days,
        metadata={
            'mu_daily': mu_daily.tolist(),
            'shrinkage': shrinkage,
        },
    )


def generate_t_distribution_scenarios(
    returns: pd.DataFrame,
    n_scenarios: int = 1000,
    horizon_days: int = 1,
    df: Optional[float] = None,
) -> ScenarioSet:
    """
    Generate scenarios from multivariate t-distribution.

    Better for capturing fat tails in financial returns.

    Args:
        returns: Historical returns for fitting
        n_scenarios: Number of scenarios
        horizon_days: Holding period in days
        df: Degrees of freedom (None = estimate from data)

    Returns:
        ScenarioSet with t-distribution scenarios
    """
    if not HAS_SCIPY:
        logger.warning("scipy not available, falling back to normal")
        return generate_normal_scenarios(returns, n_scenarios, horizon_days)

    tickers = returns.columns.tolist()
    n_assets = len(tickers)

    # Calculate parameters
    mu_daily = returns.mean().values

    # Covariance with shrinkage
    if HAS_SKLEARN:
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_daily = lw.covariance_
    else:
        cov_daily = returns.cov().values

    # Estimate degrees of freedom from marginal kurtosis
    if df is None:
        kurtosis = returns.kurtosis().mean()
        if kurtosis > 0:
            df = max(4.0, 6 / kurtosis + 4)  # Approximation
        else:
            df = 30.0  # High df = close to normal

    # Scale to horizon
    mu_horizon = mu_daily * horizon_days
    cov_horizon = cov_daily * horizon_days

    # Generate multivariate t scenarios
    # Using the relationship with chi-squared
    # t = normal / sqrt(chi2/df)
    normal_samples = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=cov_horizon,
        size=n_scenarios,
    )

    chi2_samples = np.random.chisquare(df, size=n_scenarios)
    t_samples = normal_samples / np.sqrt(chi2_samples[:, np.newaxis] / df)

    # Add mean
    scenarios = t_samples + mu_horizon

    return ScenarioSet(
        scenarios=scenarios,
        tickers=tickers,
        method=ScenarioMethod.T_DISTRIBUTION,
        n_scenarios=n_scenarios,
        horizon_days=horizon_days,
        metadata={
            'degrees_of_freedom': df,
            'estimated_kurtosis': float(returns.kurtosis().mean()),
        },
    )


def generate_regime_conditional_scenarios(
    returns: pd.DataFrame,
    regime: int,
    regime_history: Optional[pd.Series] = None,
    n_scenarios: int = 1000,
    horizon_days: int = 1,
    regime_params: Optional[Dict[int, Dict[str, float]]] = None,
) -> ScenarioSet:
    """
    Generate scenarios conditional on market regime.

    Different regimes have different return/volatility characteristics.

    Args:
        returns: Historical returns
        regime: Current regime (0=Bull, 1=Bear, 2=Neutral, 3=Crisis)
        regime_history: Historical regime labels
        n_scenarios: Number of scenarios
        horizon_days: Holding period
        regime_params: Optional override for regime parameters

    Returns:
        ScenarioSet with regime-conditional scenarios
    """
    tickers = returns.columns.tolist()

    # Default regime parameters
    default_params = {
        0: {'mu_mult': 1.5, 'vol_mult': 0.8},   # Bull: higher returns, lower vol
        1: {'mu_mult': 0.5, 'vol_mult': 1.3},   # Bear: lower returns, higher vol
        2: {'mu_mult': 1.0, 'vol_mult': 1.0},   # Neutral: baseline
        3: {'mu_mult': -0.5, 'vol_mult': 2.0},  # Crisis: negative returns, high vol
    }

    params = regime_params or default_params
    regime_config = params.get(regime, params[2])

    mu_mult = regime_config['mu_mult']
    vol_mult = regime_config['vol_mult']

    # Base parameters from historical
    mu_base = returns.mean().values * horizon_days

    if HAS_SKLEARN:
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_base = lw.covariance_ * horizon_days
    else:
        cov_base = returns.cov().values * horizon_days

    # Adjust for regime
    mu_regime = mu_base * mu_mult
    cov_regime = cov_base * (vol_mult ** 2)

    # If we have historical regime labels, filter to matching regime
    if regime_history is not None and len(regime_history) == len(returns):
        regime_mask = regime_history == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) >= 30:
            # Use actual regime data
            mu_regime = regime_returns.mean().values * horizon_days
            if HAS_SKLEARN:
                lw = LedoitWolf()
                lw.fit(regime_returns.values)
                cov_regime = lw.covariance_ * horizon_days
            else:
                cov_regime = regime_returns.cov().values * horizon_days

    # Generate scenarios
    scenarios = np.random.multivariate_normal(
        mean=mu_regime,
        cov=cov_regime,
        size=n_scenarios,
    )

    return ScenarioSet(
        scenarios=scenarios,
        tickers=tickers,
        method=ScenarioMethod.REGIME_CONDITIONAL,
        n_scenarios=n_scenarios,
        horizon_days=horizon_days,
        metadata={
            'regime': regime,
            'regime_label': {0: 'Bull', 1: 'Bear', 2: 'Neutral', 3: 'Crisis'}.get(regime, 'Unknown'),
            'mu_multiplier': mu_mult,
            'vol_multiplier': vol_mult,
        },
    )


def generate_stress_scenarios(
    returns: pd.DataFrame,
    n_scenarios: int = 100,
    horizon_days: int = 1,
    stress_multiplier: float = 3.0,
    tail_percentile: float = 5.0,
) -> ScenarioSet:
    """
    Generate stress test scenarios (tail events).

    Focuses on adverse scenarios for risk assessment.

    Args:
        returns: Historical returns
        n_scenarios: Number of scenarios
        horizon_days: Holding period
        stress_multiplier: Volatility multiplier for stress
        tail_percentile: Percentile for tail selection

    Returns:
        ScenarioSet with stress scenarios
    """
    tickers = returns.columns.tolist()

    # Calculate parameters
    mu_daily = returns.mean().values

    if HAS_SKLEARN:
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_daily = lw.covariance_
    else:
        cov_daily = returns.cov().values

    # Scale for horizon
    mu_horizon = mu_daily * horizon_days
    cov_horizon = cov_daily * horizon_days

    # Inflate volatility for stress
    cov_stress = cov_horizon * (stress_multiplier ** 2)

    # Generate many scenarios and keep worst
    oversample = n_scenarios * 10
    all_scenarios = np.random.multivariate_normal(
        mean=mu_horizon,
        cov=cov_stress,
        size=oversample,
    )

    # Select worst scenarios based on mean return
    scenario_means = all_scenarios.mean(axis=1)
    worst_indices = np.argsort(scenario_means)[:n_scenarios]
    stress_scenarios = all_scenarios[worst_indices]

    # Also add historical tail events
    if horizon_days == 1:
        portfolio_rets = returns.mean(axis=1)
        threshold = portfolio_rets.quantile(tail_percentile / 100)
        tail_dates = portfolio_rets[portfolio_rets <= threshold].index
        tail_scenarios = returns.loc[tail_dates].values

        if len(tail_scenarios) > 0:
            # Replace some generated scenarios with historical tails
            n_historical = min(len(tail_scenarios), n_scenarios // 4)
            historical_indices = np.random.choice(len(tail_scenarios), n_historical, replace=False)
            stress_scenarios[:n_historical] = tail_scenarios[historical_indices]

    return ScenarioSet(
        scenarios=stress_scenarios,
        tickers=tickers,
        method=ScenarioMethod.STRESS_TEST,
        n_scenarios=n_scenarios,
        horizon_days=horizon_days,
        metadata={
            'stress_multiplier': stress_multiplier,
            'tail_percentile': tail_percentile,
            'worst_mean_return': float(np.mean(stress_scenarios)),
        },
    )


def generate_bootstrap_scenarios(
    returns: pd.DataFrame,
    n_scenarios: int = 1000,
    horizon_days: int = 1,
    block_size: int = 5,
) -> ScenarioSet:
    """
    Generate scenarios using block bootstrap.

    Preserves autocorrelation structure in returns.

    Args:
        returns: Historical returns
        n_scenarios: Number of scenarios
        horizon_days: Holding period
        block_size: Size of blocks for bootstrap

    Returns:
        ScenarioSet with bootstrapped scenarios
    """
    tickers = returns.columns.tolist()
    n_obs = len(returns)
    values = returns.values

    scenarios = np.zeros((n_scenarios, len(tickers)))

    # Number of blocks needed
    n_blocks = (horizon_days + block_size - 1) // block_size

    for i in range(n_scenarios):
        # Sample random block starting points
        starts = np.random.randint(0, n_obs - block_size + 1, size=n_blocks)

        # Collect returns from blocks
        block_returns = []
        for start in starts:
            block = values[start:start + block_size]
            block_returns.append(block)

        # Concatenate and trim to horizon
        all_returns = np.vstack(block_returns)[:horizon_days]

        # Compound returns over horizon
        compound = (1 + all_returns).prod(axis=0) - 1
        scenarios[i] = compound

    return ScenarioSet(
        scenarios=scenarios,
        tickers=tickers,
        method=ScenarioMethod.BOOTSTRAP,
        n_scenarios=n_scenarios,
        horizon_days=horizon_days,
        metadata={
            'block_size': block_size,
        },
    )


def generate_scenarios(
    returns: pd.DataFrame,
    method: Union[str, ScenarioMethod] = 'normal',
    n_scenarios: int = 1000,
    horizon_days: int = 1,
    regime: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ScenarioSet:
    """
    Generate scenarios using specified method.

    Args:
        returns: Historical returns DataFrame
        method: Scenario generation method
        n_scenarios: Number of scenarios
        horizon_days: Holding period
        regime: Current market regime (for regime-conditional)
        config: Additional configuration

    Returns:
        ScenarioSet with generated scenarios
    """
    config = config or {}

    # Convert string to enum
    if isinstance(method, str):
        method = ScenarioMethod(method.lower())

    if method == ScenarioMethod.HISTORICAL:
        return generate_historical_scenarios(
            returns=returns,
            n_scenarios=n_scenarios,
            horizon_days=horizon_days,
            with_replacement=config.get('with_replacement', True),
        )

    elif method == ScenarioMethod.NORMAL:
        return generate_normal_scenarios(
            returns=returns,
            n_scenarios=n_scenarios,
            horizon_days=horizon_days,
            use_shrinkage=config.get('use_shrinkage', True),
        )

    elif method == ScenarioMethod.T_DISTRIBUTION:
        return generate_t_distribution_scenarios(
            returns=returns,
            n_scenarios=n_scenarios,
            horizon_days=horizon_days,
            df=config.get('degrees_of_freedom'),
        )

    elif method == ScenarioMethod.REGIME_CONDITIONAL:
        if regime is None:
            regime = 2  # Default to neutral
        return generate_regime_conditional_scenarios(
            returns=returns,
            regime=regime,
            n_scenarios=n_scenarios,
            horizon_days=horizon_days,
            regime_params=config.get('regime_params'),
        )

    elif method == ScenarioMethod.STRESS_TEST:
        return generate_stress_scenarios(
            returns=returns,
            n_scenarios=n_scenarios,
            horizon_days=horizon_days,
            stress_multiplier=config.get('stress_multiplier', 3.0),
            tail_percentile=config.get('tail_percentile', 5.0),
        )

    elif method == ScenarioMethod.BOOTSTRAP:
        return generate_bootstrap_scenarios(
            returns=returns,
            n_scenarios=n_scenarios,
            horizon_days=horizon_days,
            block_size=config.get('block_size', 5),
        )

    else:
        raise ValueError(f"Unknown scenario method: {method}")


def calculate_scenario_cvar(
    scenarios: ScenarioSet,
    weights: Union[np.ndarray, pd.Series],
    alpha: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate VaR and CVaR from scenarios.

    Args:
        scenarios: ScenarioSet object
        weights: Portfolio weights
        alpha: Confidence level

    Returns:
        Tuple of (VaR, CVaR)
    """
    if isinstance(weights, pd.Series):
        weights = weights.values

    # Calculate portfolio returns for each scenario
    portfolio_returns = scenarios.scenarios @ weights

    # VaR
    var = np.percentile(portfolio_returns, (1 - alpha) * 100)

    # CVaR (expected shortfall)
    cvar = portfolio_returns[portfolio_returns <= var].mean()

    return float(var), float(cvar)


def calculate_scenario_risk_contribution(
    scenarios: ScenarioSet,
    weights: Union[np.ndarray, pd.Series],
    alpha: float = 0.95,
) -> pd.Series:
    """
    Calculate risk contribution of each asset to CVaR.

    Args:
        scenarios: ScenarioSet object
        weights: Portfolio weights
        alpha: Confidence level

    Returns:
        Series of risk contributions per asset
    """
    if isinstance(weights, pd.Series):
        weights = weights.values

    portfolio_returns = scenarios.scenarios @ weights
    var = np.percentile(portfolio_returns, (1 - alpha) * 100)

    # Identify tail scenarios
    tail_mask = portfolio_returns <= var
    tail_scenarios = scenarios.scenarios[tail_mask]

    # Average return in tail for each asset
    tail_asset_returns = tail_scenarios.mean(axis=0)

    # Weight by portfolio weight
    risk_contrib = tail_asset_returns * weights

    # Normalize to sum to 1
    risk_contrib = risk_contrib / risk_contrib.sum()

    return pd.Series(risk_contrib, index=scenarios.tickers)
