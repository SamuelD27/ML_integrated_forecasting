"""
CVaR-Based Portfolio Allocator
================================

Enhanced multi-asset allocator with:
- CVaR (Conditional Value at Risk) optimization
- Sector/country constraints
- Turnover penalties
- Beta alignment with futures/ETFs
- ML score integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: cvxpy not available. Limited optimization capabilities.")

try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class CVaRAllocator:
    """CVaR-based portfolio allocator with constraints."""

    def __init__(self, risk_aversion: float = 5.0,
                 cvar_alpha: float = 0.95,
                 max_weight: float = 0.25,
                 min_weight: float = 0.0,
                 turnover_penalty: float = 0.0):
        """
        Initialize CVaR allocator.

        Parameters
        ----------
        risk_aversion : float
            Risk aversion parameter (higher = more conservative)
        cvar_alpha : float
            CVaR confidence level (e.g., 0.95 = 95% CVaR)
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
        turnover_penalty : float
            Penalty for turnover from previous weights
        """
        self.risk_aversion = risk_aversion
        self.cvar_alpha = cvar_alpha
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.turnover_penalty = turnover_penalty

    def optimize(self, returns: pd.DataFrame,
                ml_scores: Optional[pd.Series] = None,
                sector_map: Optional[Dict[str, str]] = None,
                market_ticker: Optional[str] = None,
                target_beta: Optional[float] = None,
                previous_weights: Optional[pd.Series] = None) -> Dict:
        """
        Optimize portfolio using CVaR objective.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns (dates x tickers)
        ml_scores : pd.Series, optional
            ML ranking scores for each ticker (higher = better)
        sector_map : dict, optional
            Mapping of ticker -> sector for sector constraints
        market_ticker : str, optional
            Market benchmark ticker for beta calculation
        target_beta : float, optional
            Target portfolio beta
        previous_weights : pd.Series, optional
            Previous portfolio weights for turnover penalty

        Returns
        -------
        dict
            Optimization results
        """
        print(f"\n[CVaR Allocator] Optimizing portfolio...")
        print(f"  Risk aversion: {self.risk_aversion:.2f}")
        print(f"  CVaR alpha: {self.cvar_alpha:.2%}")

        tickers = returns.columns.tolist()
        n = len(tickers)

        # Compute expected returns
        if ml_scores is not None:
            # Use ML scores as signal for expected returns
            mu_annual = self._ml_scores_to_returns(ml_scores, returns)
            print(f"  Using ML scores for expected returns")
        else:
            # Historical mean
            mu_annual = returns.mean() * 252

        # Compute covariance
        cov_daily, shrinkage = self._compute_covariance(returns)
        cov_annual = cov_daily * 252

        # CVaR optimization
        if HAS_CVXPY and len(returns) >= n:
            weights = self._optimize_cvar_cvxpy(
                returns.values, mu_annual.values, cov_annual.values,
                tickers, ml_scores, sector_map, market_ticker,
                target_beta, previous_weights, returns
            )
        else:
            # Fallback: mean-variance
            print(f"  Falling back to mean-variance optimization")
            weights = self._optimize_mean_variance_fallback(
                mu_annual, cov_annual, ml_scores
            )

        weights = pd.Series(weights, index=tickers)

        # Calculate portfolio metrics
        port_return = float(mu_annual @ weights)
        port_vol = float(np.sqrt(weights @ cov_annual @ weights))
        port_sharpe = port_return / port_vol if port_vol > 0 else 0

        # Calculate CVaR
        port_cvar = self._calculate_cvar(returns, weights, self.cvar_alpha)

        # Calculate beta if market provided
        realized_beta = None
        if market_ticker and market_ticker in returns.columns:
            realized_beta = self._calculate_beta(returns, weights, market_ticker)

        print(f"  ✓ Optimization complete")
        print(f"    Expected return: {port_return*100:.2f}%")
        print(f"    Volatility: {port_vol*100:.2f}%")
        print(f"    Sharpe ratio: {port_sharpe:.3f}")
        print(f"    CVaR (95%): {port_cvar*100:.2f}%")
        if realized_beta is not None:
            print(f"    Beta: {realized_beta:.3f}")

        return {
            'weights': weights,
            'mu_annual': mu_annual,
            'cov_annual': cov_annual,
            'port_return': port_return,
            'port_vol': port_vol,
            'sharpe': port_sharpe,
            'cvar': port_cvar,
            'realized_beta': realized_beta,
            'target_beta': target_beta,
            'shrinkage_coeff': shrinkage,
            'risk_aversion': self.risk_aversion,
        }

    def _optimize_cvar_cvxpy(self, returns_matrix: np.ndarray,
                            mu: np.ndarray, cov: np.ndarray,
                            tickers: List[str],
                            ml_scores: Optional[pd.Series],
                            sector_map: Optional[Dict],
                            market_ticker: Optional[str],
                            target_beta: Optional[float],
                            previous_weights: Optional[pd.Series],
                            returns_df: pd.DataFrame) -> np.ndarray:
        """Optimize using CVaR objective with CVXPY."""
        n_assets = len(mu)
        n_scenarios = len(returns_matrix)

        # Variables
        w = cp.Variable(n_assets)
        z = cp.Variable()  # VaR
        u = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR

        # Scenario returns
        scenario_returns = returns_matrix @ w

        # CVaR calculation
        cvar = z + (1 / ((1 - self.cvar_alpha) * n_scenarios)) * cp.sum(u)

        # Expected return
        expected_return = mu @ w

        # Objective: maximize return - risk_aversion * CVaR
        objective = expected_return - self.risk_aversion * cvar

        # Add ML score tilt if provided
        if ml_scores is not None:
            ml_scores_aligned = np.array([ml_scores.get(t, 0) for t in tickers])
            ml_scores_norm = ml_scores_aligned / np.sum(np.abs(ml_scores_aligned))
            objective += 0.5 * (ml_scores_norm @ w)  # Small ML tilt

        # Add turnover penalty if provided
        if previous_weights is not None and self.turnover_penalty > 0:
            prev_w = np.array([previous_weights.get(t, 0) for t in tickers])
            turnover = cp.norm(w - prev_w, 1)
            objective -= self.turnover_penalty * turnover

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= self.min_weight,  # Min weight
            w <= self.max_weight,  # Max weight
            u >= 0,  # CVaR auxiliary
            u >= -scenario_returns - z,  # CVaR constraint
        ]

        # Sector constraints (max 40% per sector)
        if sector_map:
            sectors = {}
            for ticker in tickers:
                sector = sector_map.get(ticker, 'Other')
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(tickers.index(ticker))

            for sector, indices in sectors.items():
                if len(indices) > 1:  # Only constrain if multiple assets in sector
                    sector_weight = cp.sum([w[i] for i in indices])
                    constraints.append(sector_weight <= 0.40)

        # Beta constraint
        if target_beta is not None and market_ticker and market_ticker in returns_df.columns:
            # Calculate asset betas
            try:
                market_returns = returns_df[market_ticker].values
                betas = []

                for ticker in tickers:
                    if ticker == market_ticker:
                        betas.append(1.0)
                    elif ticker in returns_df.columns:
                        asset_returns = returns_df[ticker].values
                        covar = np.cov(asset_returns, market_returns)[0, 1]
                        market_var = np.var(market_returns)
                        beta = covar / market_var if market_var > 0 else 1.0
                        betas.append(beta)
                    else:
                        betas.append(1.0)

                betas = np.array(betas)
                portfolio_beta = betas @ w
                constraints.append(portfolio_beta <= target_beta + 0.15)
                constraints.append(portfolio_beta >= target_beta - 0.15)
            except Exception as e:
                print(f"    Warning: Could not apply beta constraint: {e}")
                pass

        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5)

            if w.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
                raise ValueError(f"Optimization failed with status: {problem.status}")

            weights = w.value
            weights = np.clip(weights, 0, 1)  # Ensure valid weights
            weights = weights / np.sum(weights)  # Renormalize

            return weights

        except Exception as e:
            print(f"  Warning: CVaR optimization failed: {e}")
            print(f"  Falling back to mean-variance")
            return self._optimize_mean_variance_fallback(
                pd.Series(mu, index=tickers),
                pd.DataFrame(cov, index=tickers, columns=tickers),
                ml_scores
            ).values

    def _optimize_mean_variance_fallback(self, mu: pd.Series,
                                         cov: pd.DataFrame,
                                         ml_scores: Optional[pd.Series]) -> pd.Series:
        """Fallback mean-variance optimization."""
        n = len(mu)
        tickers = mu.index

        if not HAS_CVXPY:
            # Equal weight
            return pd.Series(1.0 / n, index=tickers)

        w = cp.Variable(n)

        # Objective
        expected_return = mu.values @ w
        variance = cp.quad_form(w, cov.values)
        objective = expected_return - self.risk_aversion * variance

        # ML tilt
        if ml_scores is not None:
            ml_scores_aligned = np.array([ml_scores.get(t, 0) for t in tickers])
            ml_scores_norm = ml_scores_aligned / np.sum(np.abs(ml_scores_aligned))
            objective += 0.5 * (ml_scores_norm @ w)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight,
        ]

        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        if w.value is None:
            return pd.Series(1.0 / n, index=tickers)

        weights = pd.Series(w.value, index=tickers)
        weights = weights.clip(lower=0)
        weights = weights / weights.sum()

        return weights

    def _compute_covariance(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Compute shrinkage covariance matrix."""
        if HAS_SKLEARN and len(returns) > len(returns.columns):
            lw = LedoitWolf()
            lw.fit(returns.values)
            cov_matrix = pd.DataFrame(
                lw.covariance_,
                index=returns.columns,
                columns=returns.columns
            )
            return cov_matrix, lw.shrinkage_
        else:
            return returns.cov(), 0.0

    def _ml_scores_to_returns(self, ml_scores: pd.Series,
                              returns: pd.DataFrame) -> pd.Series:
        """Convert ML scores to expected returns."""
        # Normalize scores to returns-like scale
        hist_returns = returns.mean() * 252
        mean_ret = hist_returns.mean()
        std_ret = hist_returns.std()

        # Scale ML scores
        ml_aligned = pd.Series([ml_scores.get(t, 0) for t in returns.columns],
                              index=returns.columns)

        # Normalize to zero mean, unit variance
        ml_norm = (ml_aligned - ml_aligned.mean()) / ml_aligned.std() if ml_aligned.std() > 0 else ml_aligned

        # Scale to returns distribution
        expected_returns = mean_ret + ml_norm * std_ret * 0.5  # Dampen signal

        return expected_returns

    def _calculate_cvar(self, returns: pd.DataFrame, weights: pd.Series,
                       alpha: float) -> float:
        """Calculate portfolio CVaR."""
        portfolio_returns = (returns * weights).sum(axis=1)
        var = portfolio_returns.quantile(1 - alpha)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return abs(cvar) * np.sqrt(252)  # Annualize

    def _calculate_beta(self, returns: pd.DataFrame, weights: pd.Series,
                       market_ticker: str) -> float:
        """Calculate portfolio beta."""
        if market_ticker not in returns.columns:
            return 1.0

        portfolio_returns = (returns.drop(columns=[market_ticker], errors='ignore') * weights).sum(axis=1)
        market_returns = returns[market_ticker]

        covar = portfolio_returns.cov(market_returns)
        market_var = market_returns.var()

        return covar / market_var if market_var > 0 else 1.0


def optimize_portfolio_cvar(returns: pd.DataFrame,
                           rrr: float,
                           ml_scores: Optional[pd.Series] = None,
                           sector_map: Optional[Dict] = None,
                           market_ticker: Optional[str] = None) -> Dict:
    """
    Convenience function for CVaR portfolio optimization.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns
    rrr : float
        Reward-to-risk ratio (0 to 1)
    ml_scores : pd.Series, optional
        ML scores
    sector_map : dict, optional
        Sector mapping
    market_ticker : str, optional
        Market benchmark

    Returns
    -------
    dict
        Optimization results
    """
    # Map RRR to risk aversion
    risk_aversion = 1.0 + (rrr * 19.0)  # 1 to 20

    # Target beta from RRR
    target_beta = 1.1 - (rrr * 0.6) if rrr > 0.5 else None  # Defensive for high RRR

    # Max weight from RRR
    max_weight = 0.30 - (rrr * 0.15)  # 0.30 to 0.15

    allocator = CVaRAllocator(
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )

    return allocator.optimize(
        returns=returns,
        ml_scores=ml_scores,
        sector_map=sector_map,
        market_ticker=market_ticker,
        target_beta=target_beta,
    )
