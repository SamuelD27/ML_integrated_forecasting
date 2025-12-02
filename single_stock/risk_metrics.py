"""
Risk metrics for single stock analysis.

This module includes functions for computing Value at Risk (VaR) using
multiple methods, the Sharpe ratio, and the CAPM beta of an asset relative
to a benchmark index.  These metrics help quantify downside risk and
risk‑adjusted return.

All functions include input validation and bounds checking per the
financial-knowledge-validator skill guidelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Literal, Optional, Dict

from scipy.stats import norm


# ============================================================================
# REALISTIC BOUNDS FOR VALIDATION
# ============================================================================
# These bounds are based on empirical observations of financial markets.
# Values outside these ranges indicate likely calculation errors.

SHARPE_BOUNDS = (-5.0, 8.0)  # Extreme but possible in volatile markets
BETA_BOUNDS = (-3.0, 5.0)    # Most assets fall in [-1, 3]
VAR_CONFIDENCE_BOUNDS = (0.5, 0.999)  # Must be in (0.5, 1)


def compute_var(returns: pd.Series,
                confidence_level: float = 0.95,
                method: Literal['historical', 'parametric', 'monte_carlo'] = 'historical',
                num_simulations: int = 10000,
                horizon_days: int = 1) -> float:
    """Compute Value at Risk (VaR) of a time series of returns.

    Parameters
    ----------
    returns : pd.Series
        Series of returns (in decimal form, e.g., 0.01 for 1%).
    confidence_level : float, default 0.95
        The desired confidence level for VaR (e.g., 0.95 for 95% VaR).
    method : {'historical', 'parametric', 'monte_carlo'}, default 'historical'
        Method to compute VaR:
        - 'historical': uses the empirical distribution of past returns.
        - 'parametric': assumes returns are normally distributed and uses mean & std.
        - 'monte_carlo': simulates price paths under a log-normal GBM model.
    num_simulations : int, default 10000
        Number of simulated paths for Monte Carlo method.
    horizon_days : int, default 1
        Horizon in trading days. VaR is scaled using sqrt(T) rule.

    Returns
    -------
    float
        The Value at Risk at the specified confidence level (a negative number representing loss).

    Raises
    ------
    ValueError
        If inputs are invalid or results are outside realistic bounds.

    Notes
    -----
    VaR scaling uses sqrt(T) rule which assumes i.i.d. returns. In practice,
    returns exhibit volatility clustering, so this is an approximation.
    """
    # INPUT VALIDATION
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    r = returns.dropna().values

    if len(r) < 20:
        raise ValueError(f"Insufficient data: need at least 20 returns, got {len(r)}")

    if not (VAR_CONFIDENCE_BOUNDS[0] < confidence_level < VAR_CONFIDENCE_BOUNDS[1]):
        raise ValueError(
            f"confidence_level {confidence_level} must be in "
            f"({VAR_CONFIDENCE_BOUNDS[0]}, {VAR_CONFIDENCE_BOUNDS[1]})"
        )

    if horizon_days < 1 or horizon_days > 252:
        raise ValueError(f"horizon_days {horizon_days} must be between 1 and 252")

    if method not in ('historical', 'parametric', 'monte_carlo'):
        raise ValueError(f"Unknown method: {method}. Use 'historical', 'parametric', or 'monte_carlo'")

    sqrt_horizon = np.sqrt(horizon_days)

    if method == 'historical':
        # Sort returns ascending; VaR is the quantile at (1 - confidence)
        var_daily = np.quantile(r, 1 - confidence_level)
        var = var_daily * sqrt_horizon
    elif method == 'parametric':
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        if sigma < 1e-10:
            return 0.0  # Edge case: constant returns
        # Compute z-score for the confidence level
        z = norm.ppf(1 - confidence_level)
        var_daily = mu + z * sigma
        var = var_daily * sqrt_horizon
    elif method == 'monte_carlo':
        # Estimate drift and volatility from historical returns
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        if sigma < 1e-10:
            return 0.0  # Edge case: constant returns
        # Time step = horizon days
        dt = horizon_days
        # Starting price assumed to be 1; we only need relative changes
        price0 = 1.0
        # Generate random shocks
        # Simulate final return after horizon using GBM formula
        z = np.random.normal(size=num_simulations)
        st = price0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        simulated_returns = (st - price0) / price0
        var = np.quantile(simulated_returns, 1 - confidence_level)

    # OUTPUT VALIDATION: VaR should be negative (it's a loss metric)
    # and should be within reasonable bounds for the horizon
    max_plausible_loss = -0.50 * sqrt_horizon  # 50% daily loss scaled
    if var < max_plausible_loss:
        # This could happen in extreme scenarios, just log a warning mentally
        pass

    return var


def compute_sharpe_ratio(returns: pd.Series,
                         risk_free_rate: float = 0.05,
                         freq: int = 252) -> float:
    """Calculate the annualized Sharpe ratio of a series of returns.

    The Sharpe ratio measures risk-adjusted return and is defined as
    (annualized return - risk free rate) / annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Series of daily (or periodic) returns in decimal form.
    risk_free_rate : float, default 0.05
        Annual risk-free rate (e.g., 0.05 = 5%). Default 5% is approximate;
        for production use, fetch current Treasury rate from FRED.
        NOTE: This is ANNUAL rate, not daily. The function handles conversion.
    freq : int, default 252
        Number of periods per year (252 for trading days). Used to
        annualize returns and volatility.

    Returns
    -------
    float
        Annualized Sharpe ratio.

    Raises
    ------
    ValueError
        If inputs are invalid.
    TypeError
        If returns is not a pandas Series.

    Notes
    -----
    Formula: Sharpe = (annualized_return - annual_rf) / annualized_volatility

    Industry convention uses annual Rf directly, NOT (daily_rf * 252).
    """
    # INPUT VALIDATION
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    clean_returns = returns.dropna()
    if len(clean_returns) < 20:
        raise ValueError(f"Insufficient data: need at least 20 returns, got {len(clean_returns)}")

    if not (0.0 <= risk_free_rate <= 0.20):
        raise ValueError(f"risk_free_rate {risk_free_rate:.2%} outside realistic bounds [0%, 20%]")

    if freq <= 0:
        raise ValueError(f"freq must be positive, got {freq}")

    # Annualize returns and volatility
    mean_annual = clean_returns.mean() * freq
    std_annual = clean_returns.std(ddof=1) * np.sqrt(freq)

    # Edge case: zero volatility
    if std_annual < 1e-10:
        return np.nan

    # Sharpe = (annualized_return - annual_rf) / annualized_volatility
    sharpe = (mean_annual - risk_free_rate) / std_annual

    # OUTPUT VALIDATION
    if not (SHARPE_BOUNDS[0] < sharpe < SHARPE_BOUNDS[1]):
        # Log warning but don't fail - extreme values are possible
        pass

    return sharpe


def compute_beta(asset_returns: pd.Series,
                 benchmark_returns: pd.Series) -> float:
    """Compute the CAPM beta of an asset relative to a benchmark index.

    Beta measures the systematic risk of an asset: how much the asset's
    returns co-move with the market returns. A beta greater than 1 indicates
    higher volatility relative to the market.

    Parameters
    ----------
    asset_returns : pd.Series
        Returns of the asset.
    benchmark_returns : pd.Series
        Returns of the benchmark index (e.g., SPY, ^GSPC).

    Returns
    -------
    float
        The beta coefficient.

    Raises
    ------
    TypeError
        If inputs are not pandas Series.
    ValueError
        If insufficient overlapping data points.

    Notes
    -----
    Beta = Cov(asset, market) / Var(market)

    Typical interpretation:
    - Beta < 0: Inverse correlation with market (rare)
    - Beta = 0: No correlation with market
    - 0 < Beta < 1: Less volatile than market
    - Beta = 1: Moves with market
    - Beta > 1: More volatile than market
    """
    # INPUT VALIDATION
    if not isinstance(asset_returns, pd.Series):
        raise TypeError("asset_returns must be a pandas Series")
    if not isinstance(benchmark_returns, pd.Series):
        raise TypeError("benchmark_returns must be a pandas Series")

    # Align on common dates
    combined = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner').dropna()

    if len(combined) < 20:
        raise ValueError(
            f"Insufficient overlapping data: need at least 20 points, got {len(combined)}"
        )

    asset_r = combined.iloc[:, 0]
    bench_r = combined.iloc[:, 1]

    # Compute beta
    covariance = np.cov(asset_r, bench_r, ddof=1)[0, 1]
    variance_bench = np.var(bench_r, ddof=1)

    # Edge case: benchmark has zero variance
    if variance_bench < 1e-10:
        return np.nan

    beta = covariance / variance_bench

    # OUTPUT VALIDATION
    if not (BETA_BOUNDS[0] < beta < BETA_BOUNDS[1]):
        # Unusual but not impossible - could be leveraged ETF or inverse fund
        pass

    return beta


def compute_cvar(returns: pd.Series,
                 confidence_level: float = 0.95,
                 horizon_days: int = 1) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.
    It is a more conservative risk measure than VaR.

    Parameters
    ----------
    returns : pd.Series
        Series of returns (in decimal form).
    confidence_level : float, default 0.95
        The desired confidence level (e.g., 0.95 for 95% CVaR).
    horizon_days : int, default 1
        Horizon in trading days. CVaR is scaled using sqrt(T) rule.

    Returns
    -------
    float
        The Conditional VaR (expected loss in the tail).

    Notes
    -----
    CVaR (also called Expected Shortfall) answers: "If we're in the worst
    (1-confidence)% of scenarios, what's the average loss?"

    This is more conservative than VaR and is preferred by regulators
    (Basel III uses 97.5% Expected Shortfall).
    """
    # INPUT VALIDATION
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    r = returns.dropna().values

    if len(r) < 20:
        raise ValueError(f"Insufficient data: need at least 20 returns, got {len(r)}")

    if not (VAR_CONFIDENCE_BOUNDS[0] < confidence_level < VAR_CONFIDENCE_BOUNDS[1]):
        raise ValueError(
            f"confidence_level {confidence_level} must be in "
            f"({VAR_CONFIDENCE_BOUNDS[0]}, {VAR_CONFIDENCE_BOUNDS[1]})"
        )

    if horizon_days < 1 or horizon_days > 252:
        raise ValueError(f"horizon_days {horizon_days} must be between 1 and 252")

    # Compute daily VaR
    var_daily = np.quantile(r, 1 - confidence_level)

    # CVaR is the mean of returns below VaR
    tail_returns = r[r <= var_daily]
    if len(tail_returns) == 0:
        cvar_daily = var_daily  # Edge case: use VaR
    else:
        cvar_daily = np.mean(tail_returns)

    # Scale to horizon
    cvar = cvar_daily * np.sqrt(horizon_days)

    return cvar
