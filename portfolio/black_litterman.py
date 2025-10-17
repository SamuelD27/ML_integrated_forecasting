"""
Black-Litterman Portfolio Optimization
=======================================
Combines market equilibrium with investor views (e.g., ML forecasts).

The Black-Litterman model:
1. Starts with market equilibrium (CAPM implied returns)
2. Incorporates investor views with confidence levels
3. Produces posterior expected returns
4. Uses these for portfolio optimization

Perfect for ML integration:
- ML forecasts = "views"
- ML confidence = view uncertainty
- Historical returns = equilibrium

Reference:
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization"
- He, G., & Litterman, R. (1999). "The Intuition Behind Black-Litterman"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization.

    Combines:
    - Market equilibrium (from historical returns/CAPM)
    - Investor views (e.g., from ML models)
    - View confidence levels

    Example:
        >>> bl = BlackLittermanOptimizer(risk_aversion=2.5)
        >>> # Set equilibrium (historical)
        >>> bl.set_equilibrium(returns_df)
        >>> # Add ML view
        >>> bl.add_view('AAPL', expected_return=0.15, confidence=0.8)
        >>> # Optimize
        >>> weights = bl.optimize()
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.025,
        market_cap_weights: Optional[pd.Series] = None
    ):
        """
        Initialize Black-Litterman optimizer.

        Args:
            risk_aversion: Risk aversion parameter (λ), typically 2-4
            tau: Scaling factor for uncertainty, typically 0.01-0.05
            market_cap_weights: Market cap weights (if None, uses equal weight)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.market_cap_weights = market_cap_weights

        # Internal state
        self.equilibrium_returns = None
        self.covariance_matrix = None
        self.assets = None

        # Views
        self.P = None  # Picking matrix
        self.Q = None  # View returns
        self.Omega = None  # View uncertainty

    def set_equilibrium(
        self,
        returns: pd.DataFrame,
        use_capm: bool = True,
        market_return: Optional[float] = None
    ):
        """
        Set market equilibrium from historical returns.

        Args:
            returns: Historical returns DataFrame (assets as columns)
            use_capm: Whether to use CAPM implied returns
            market_return: Market return for CAPM (if None, uses mean of returns)
        """
        self.assets = returns.columns.tolist()
        n_assets = len(self.assets)

        # Calculate covariance
        self.covariance_matrix = returns.cov()

        if use_capm:
            # CAPM implied equilibrium returns
            # π = λ * Σ * w_mkt
            if self.market_cap_weights is None:
                # Equal weight if no market caps provided
                w_mkt = pd.Series(1.0 / n_assets, index=self.assets)
            else:
                w_mkt = self.market_cap_weights / self.market_cap_weights.sum()

            self.equilibrium_returns = (
                self.risk_aversion *
                self.covariance_matrix.dot(w_mkt)
            )
        else:
            # Use historical mean returns
            self.equilibrium_returns = returns.mean() * 252  # Annualized

        logger.info(f"Set equilibrium for {n_assets} assets")

    def add_view(
        self,
        asset: str,
        expected_return: float,
        confidence: float = 0.5,
        view_type: str = 'absolute'
    ):
        """
        Add a view on expected return.

        Args:
            asset: Asset ticker
            expected_return: Expected return (annualized)
            confidence: Confidence in view (0-1), higher = more confident
            view_type: 'absolute' or 'relative' (relative not yet implemented)
        """
        if self.assets is None:
            raise ValueError("Must call set_equilibrium() first")

        if asset not in self.assets:
            raise ValueError(f"Asset {asset} not in portfolio")

        # Initialize P, Q, Omega if first view
        if self.P is None:
            self.P = []
            self.Q = []
            self.Omega = []

        # Create picking vector (1 for this asset, 0 for others)
        p = pd.Series(0.0, index=self.assets)
        p[asset] = 1.0

        # Convert confidence to uncertainty
        # Higher confidence → lower uncertainty
        # Omega_i = (1 - confidence) * var(asset)
        asset_var = self.covariance_matrix.loc[asset, asset]
        uncertainty = (1.0 - confidence) * asset_var * self.tau

        self.P.append(p.values)
        self.Q.append(expected_return)
        self.Omega.append(uncertainty)

        logger.info(
            f"Added view: {asset} expected return {expected_return:.2%} "
            f"with confidence {confidence:.1%}"
        )

    def add_relative_view(
        self,
        asset1: str,
        asset2: str,
        expected_outperformance: float,
        confidence: float = 0.5
    ):
        """
        Add relative view: asset1 will outperform asset2.

        Args:
            asset1: First asset
            asset2: Second asset
            expected_outperformance: Expected return difference
            confidence: Confidence in view
        """
        if self.assets is None:
            raise ValueError("Must call set_equilibrium() first")

        if asset1 not in self.assets or asset2 not in self.assets:
            raise ValueError("Both assets must be in portfolio")

        # Initialize if needed
        if self.P is None:
            self.P = []
            self.Q = []
            self.Omega = []

        # Picking vector: +1 for asset1, -1 for asset2
        p = pd.Series(0.0, index=self.assets)
        p[asset1] = 1.0
        p[asset2] = -1.0

        # Uncertainty for relative view
        var_diff = (
            self.covariance_matrix.loc[asset1, asset1] +
            self.covariance_matrix.loc[asset2, asset2] -
            2 * self.covariance_matrix.loc[asset1, asset2]
        )
        uncertainty = (1.0 - confidence) * var_diff * self.tau

        self.P.append(p.values)
        self.Q.append(expected_outperformance)
        self.Omega.append(uncertainty)

        logger.info(
            f"Added relative view: {asset1} vs {asset2}, "
            f"expected outperformance {expected_outperformance:.2%}"
        )

    def compute_posterior_returns(self) -> pd.Series:
        """
        Compute posterior expected returns using Black-Litterman formula.

        Returns:
            Posterior expected returns
        """
        if self.equilibrium_returns is None:
            raise ValueError("Must set equilibrium first")

        # If no views, return equilibrium
        if self.P is None or len(self.P) == 0:
            logger.info("No views specified, using equilibrium returns")
            return self.equilibrium_returns

        # Convert lists to arrays
        P = np.array(self.P)  # Views matrix (k x n)
        Q = np.array(self.Q)  # View returns (k x 1)
        Omega = np.diag(self.Omega)  # View uncertainty (k x k)

        # Covariance and equilibrium
        Sigma = self.covariance_matrix.values
        pi = self.equilibrium_returns.values

        # Black-Litterman formula
        # E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]

        tau_Sigma_inv = np.linalg.inv(self.tau * Sigma)
        Omega_inv = np.linalg.inv(Omega)

        # Posterior precision
        M = tau_Sigma_inv + P.T @ Omega_inv @ P

        # Posterior mean
        posterior_mean = np.linalg.inv(M) @ (
            tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q
        )

        posterior_returns = pd.Series(posterior_mean, index=self.assets)

        logger.info("Computed posterior returns")
        logger.debug(f"Equilibrium: {self.equilibrium_returns.to_dict()}")
        logger.debug(f"Posterior: {posterior_returns.to_dict()}")

        return posterior_returns

    def optimize(
        self,
        target_return: Optional[float] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> pd.Series:
        """
        Optimize portfolio weights using posterior returns.

        Args:
            target_return: Target return (if None, maximizes Sharpe)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            Optimal weights
        """
        # Get posterior returns
        expected_returns = self.compute_posterior_returns()
        Sigma = self.covariance_matrix.values
        mu = expected_returns.values
        n_assets = len(self.assets)

        # Objective: Minimize variance - λ * return
        # Equivalent to maximizing Sharpe ratio
        def objective(w):
            portfolio_var = w @ Sigma @ w
            portfolio_return = w @ mu
            # Minimize variance for given return level
            return portfolio_var - self.risk_aversion * portfolio_return

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ mu - target_return
            })

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Initial guess: equal weight
        w0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        weights = pd.Series(result.x, index=self.assets)

        logger.info("Optimization complete")
        logger.info(f"Weights: {weights.to_dict()}")

        return weights

    def get_portfolio_stats(self, weights: pd.Series) -> Dict:
        """
        Calculate portfolio statistics.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary with portfolio stats
        """
        expected_returns = self.compute_posterior_returns()

        portfolio_return = (weights * expected_returns).sum()
        portfolio_var = weights.values @ self.covariance_matrix.values @ weights.values
        portfolio_vol = np.sqrt(portfolio_var)

        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'weights': weights.to_dict()
        }


def integrate_ml_forecasts(
    returns: pd.DataFrame,
    ml_forecasts: Dict[str, float],
    ml_confidence: Dict[str, float],
    risk_aversion: float = 2.5
) -> Tuple[pd.Series, Dict]:
    """
    Convenience function to integrate ML forecasts into portfolio optimization.

    Args:
        returns: Historical returns DataFrame
        ml_forecasts: Dictionary {ticker: expected_return}
        ml_confidence: Dictionary {ticker: confidence} (0-1)
        risk_aversion: Risk aversion parameter

    Returns:
        weights: Optimal portfolio weights
        stats: Portfolio statistics

    Example:
        >>> ml_forecasts = {'AAPL': 0.15, 'MSFT': 0.12, 'GOOGL': 0.10}
        >>> ml_confidence = {'AAPL': 0.8, 'MSFT': 0.7, 'GOOGL': 0.6}
        >>> weights, stats = integrate_ml_forecasts(returns, ml_forecasts, ml_confidence)
    """
    bl = BlackLittermanOptimizer(risk_aversion=risk_aversion)

    # Set equilibrium from historical data
    bl.set_equilibrium(returns, use_capm=True)

    # Add ML views
    for ticker, forecast in ml_forecasts.items():
        if ticker in returns.columns:
            confidence = ml_confidence.get(ticker, 0.5)
            bl.add_view(ticker, forecast, confidence)

    # Optimize
    weights = bl.optimize()

    # Get stats
    stats = bl.get_portfolio_stats(weights)

    return weights, stats
