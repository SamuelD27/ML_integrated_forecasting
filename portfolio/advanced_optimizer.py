from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def compute_returns_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns matrix from prices.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Prices with tickers as columns

    Returns
    -------
    pd.DataFrame
        Returns matrix
    """
    returns = prices_df.pct_change().dropna()
    return returns


def compute_shrinkage_covariance(returns: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Compute shrinkage covariance matrix using Ledoit-Wolf estimator.

    Returns
    -------
    tuple
        (covariance_matrix, shrinkage_coefficient)
    """
    if HAS_SKLEARN and len(returns) > len(returns.columns):
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns
        )
        shrinkage = lw.shrinkage_
    else:
        # Fallback to sample covariance
        cov_matrix = returns.cov()
        shrinkage = 0.0

    return cov_matrix, shrinkage


def compute_capm_betas(returns: pd.DataFrame, market_ticker: str) -> pd.Series:
    """
    Compute CAPM betas for all assets vs market benchmark.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix with market ticker included
    market_ticker : str
        Market benchmark ticker (e.g., 'SPY')

    Returns
    -------
    pd.Series
        Beta values for each asset
    """
    if market_ticker not in returns.columns:
        print(f"  Warning: Market ticker {market_ticker} not in returns. Using equal betas.")
        return pd.Series(1.0, index=returns.columns)

    market_returns = returns[market_ticker]
    market_var = market_returns.var()

    betas = {}
    for ticker in returns.columns:
        if ticker == market_ticker:
            betas[ticker] = 1.0
        else:
            covar = returns[ticker].cov(market_returns)
            betas[ticker] = covar / market_var if market_var > 0 else 1.0

    return pd.Series(betas)


def compute_partial_correlations(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute partial correlation matrix.

    Returns
    -------
    pd.DataFrame
        Partial correlation matrix
    """
    corr_matrix = returns.corr()

    try:
        # Partial correlation via precision matrix
        cov_matrix = returns.cov()
        precision = np.linalg.inv(cov_matrix.values)

        # Partial correlation formula: -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
        diag = np.sqrt(np.diag(precision))
        partial_corr = -precision / np.outer(diag, diag)
        np.fill_diagonal(partial_corr, 1.0)

        partial_corr_df = pd.DataFrame(
            partial_corr,
            index=returns.columns,
            columns=returns.columns
        )
    except Exception:
        # Fallback to regular correlation
        partial_corr_df = corr_matrix

    return partial_corr_df


def compute_tail_comovement(returns: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    """
    Compute tail co-movement (correlation in extreme events).

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix
    quantile : float
        Quantile threshold for tail events (default: 0.95)

    Returns
    -------
    pd.DataFrame
        Tail correlation matrix
    """
    tail_corr = pd.DataFrame(1.0, index=returns.columns, columns=returns.columns)

    for col in returns.columns:
        threshold = returns[col].quantile(1 - quantile)  # Negative tail
        tail_events = returns[returns[col] <= threshold]

        if len(tail_events) > 5:  # Need sufficient tail observations
            for col2 in returns.columns:
                if col != col2:
                    tail_corr.loc[col, col2] = tail_events[col].corr(tail_events[col2])

    return tail_corr


def map_rrr_to_risk_aversion(rrr: float) -> float:
    """
    Map reward-to-risk ratio to risk aversion parameter.

    Higher RRR → lower risk tolerance → higher risk aversion
    Lower RRR → higher risk tolerance → lower risk aversion

    Parameters
    ----------
    rrr : float
        Reward-to-risk ratio (0 to 1)

    Returns
    -------
    float
        Risk aversion parameter (lambda)
    """
    # RRR 0.0 (aggressive) → lambda = 1 (low risk aversion)
    # RRR 0.5 (moderate) → lambda = 5 (medium risk aversion)
    # RRR 1.0 (defensive) → lambda = 15 (high risk aversion)

    risk_aversion = 1.0 + (rrr * 14.0)
    return risk_aversion


def optimize_mean_variance(returns: pd.DataFrame, rrr: float,
                           market_ticker: Optional[str] = None,
                           max_weight: float = 0.40,
                           min_weight: float = 0.0) -> Dict:
    """
    Perform mean-variance optimization with RRR-based risk aversion.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix
    rrr : float
        Reward-to-risk ratio (0 to 1)
    market_ticker : str, optional
        Market benchmark ticker for beta calculation
    max_weight : float
        Maximum weight per asset (default: 0.40 = 40%)
    min_weight : float
        Minimum weight per asset (default: 0.0)

    Returns
    -------
    dict
        Optimization results including weights, metrics, and diagnostics
    """
    print(f"\n[Portfolio Optimization] RRR={rrr:.2f}")

    # Compute statistics
    mu_daily = returns.mean()
    mu_annual = mu_daily * 252

    cov_daily, shrinkage = compute_shrinkage_covariance(returns)
    cov_annual = cov_daily * 252

    print(f"  Returns matrix: {returns.shape}")
    print(f"  Shrinkage covariance: alpha={shrinkage:.3f}" if shrinkage > 0 else "  Sample covariance")

    # Compute betas
    if market_ticker and market_ticker in returns.columns:
        betas = compute_capm_betas(returns, market_ticker)
        print(f"  CAPM betas computed vs {market_ticker}")
    else:
        betas = pd.Series(1.0, index=returns.columns)
        print(f"  Market ticker not available, assuming unit betas")

    # Compute partial correlations and tail co-movement
    partial_corr = compute_partial_correlations(returns)
    tail_corr = compute_tail_comovement(returns, quantile=0.95)

    # Map RRR to risk aversion
    risk_aversion = map_rrr_to_risk_aversion(rrr)
    print(f"  Risk aversion (lambda): {risk_aversion:.2f}")

    # Optimize portfolio
    n = len(returns.columns)
    tickers = returns.columns.tolist()

    if HAS_CVXPY and n > 1:
        try:
            # CVXPY optimization
            w = cp.Variable(n)

            # Objective: maximize utility = expected return - lambda * variance
            portfolio_return = mu_annual.values @ w
            portfolio_variance = cp.quad_form(w, cov_annual.values)
            utility = portfolio_return - risk_aversion * portfolio_variance

            # Constraints
            constraints = [
                cp.sum(w) == 1,
                w >= min_weight,
                w <= max_weight,
            ]

            # For high RRR, add beta constraint (target lower beta)
            if rrr > 0.6 and market_ticker:
                target_beta = 1.0 - (rrr - 0.6) * 0.5  # RRR 0.6→beta 1.0, RRR 1.0→beta 0.8
                portfolio_beta = betas.values @ w
                constraints.append(portfolio_beta <= target_beta + 0.1)
                print(f"  Target beta constraint: <= {target_beta:.2f}")

            problem = cp.Problem(cp.Maximize(utility), constraints)
            problem.solve(solver=cp.OSQP, verbose=False)

            if w.value is None:
                raise ValueError("Optimization failed")

            weights = pd.Series(w.value, index=tickers)
            weights = weights.clip(lower=0)  # Ensure non-negative
            weights = weights / weights.sum()  # Renormalize

            print(f"  ✓ Mean-variance optimization succeeded")

        except Exception as e:
            print(f"  Warning: CVXPY optimization failed ({e}), using minimum variance")
            weights = optimize_minimum_variance(cov_annual, max_weight, min_weight)

    else:
        # Fallback: minimum variance or equal weight
        if n > 1:
            print(f"  CVXPY not available, using minimum variance")
            weights = optimize_minimum_variance(cov_annual, max_weight, min_weight)
        else:
            print(f"  Single asset, using weight=1.0")
            weights = pd.Series(1.0, index=tickers)

    # Calculate portfolio metrics
    port_return = float(mu_annual @ weights)
    port_vol = float(np.sqrt(weights @ cov_annual @ weights))
    port_sharpe = port_return / port_vol if port_vol > 0 else 0

    # Calculate realized beta
    if market_ticker and market_ticker in betas.index:
        realized_beta = float(betas @ weights)
        target_beta = 1.0 - (rrr - 0.6) * 0.5 if rrr > 0.6 else None
    else:
        realized_beta = None
        target_beta = None

    print(f"  Portfolio return: {port_return*100:.2f}%")
    print(f"  Portfolio volatility: {port_vol*100:.2f}%")
    print(f"  Sharpe ratio: {port_sharpe:.3f}")
    if realized_beta is not None:
        print(f"  Realized beta: {realized_beta:.3f}" + (f" (target: {target_beta:.3f})" if target_beta else ""))

    return {
        'weights': weights,
        'mu_annual': mu_annual,
        'cov_annual': cov_annual,
        'port_return': port_return,
        'port_vol': port_vol,
        'sharpe': port_sharpe,
        'betas': betas,
        'realized_beta': realized_beta,
        'target_beta': target_beta,
        'partial_correlations': partial_corr,
        'tail_correlations': tail_corr,
        'shrinkage_coeff': shrinkage,
        'risk_aversion': risk_aversion,
    }


def optimize_minimum_variance(cov_matrix: pd.DataFrame,
                              max_weight: float = 0.40,
                              min_weight: float = 0.0) -> pd.Series:
    """
    Minimum variance portfolio optimization.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix
    max_weight : float
        Maximum weight per asset
    min_weight : float
        Minimum weight per asset

    Returns
    -------
    pd.Series
        Optimal weights
    """
    n = len(cov_matrix)
    tickers = cov_matrix.index.tolist()

    if HAS_CVXPY and n > 1:
        try:
            w = cp.Variable(n)
            portfolio_variance = cp.quad_form(w, cov_matrix.values)

            constraints = [
                cp.sum(w) == 1,
                w >= min_weight,
                w <= max_weight,
            ]

            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
            problem.solve(solver=cp.OSQP, verbose=False)

            if w.value is not None:
                weights = pd.Series(w.value, index=tickers)
                weights = weights.clip(lower=0)
                weights = weights / weights.sum()
                return weights

        except Exception:
            pass

    # Fallback: equal weight
    return pd.Series(1.0 / n, index=tickers)


def calculate_portfolio_shares(weights: pd.Series, capital: float,
                               current_prices: pd.Series) -> pd.DataFrame:
    """
    Convert portfolio weights to dollar allocations and share quantities.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (sum to 1)
    capital : float
        Total capital to invest
    current_prices : pd.Series
        Current prices for each asset

    Returns
    -------
    pd.DataFrame
        Holdings with columns: ticker, weight, dollars, shares, price
    """
    holdings = []

    for ticker in weights.index:
        weight = weights[ticker]
        dollars = capital * weight
        price = current_prices.get(ticker, np.nan)

        if pd.notna(price) and price > 0:
            shares = int(dollars / price)
        else:
            shares = 0

        holdings.append({
            'ticker': ticker,
            'weight': weight,
            'dollars': dollars,
            'shares': shares,
            'price': price,
        })

    holdings_df = pd.DataFrame(holdings)
    return holdings_df


if __name__ == '__main__':
    # Test optimization
    import yfinance as yf
    from datetime import datetime, timedelta

    # Test tickers
    tickers = ['AAPL', 'MSFT', 'NVDA', 'SPY']

    # Download prices
    end = datetime.now()
    start = end - timedelta(days=365)

    print("Downloading test data...")
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']

    # Compute returns
    returns = compute_returns_matrix(data)

    # Test optimization with different RRR values
    for rrr in [0.2, 0.5, 0.8]:
        result = optimize_mean_variance(returns, rrr=rrr, market_ticker='SPY')

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION RESULTS (RRR={rrr})")
        print(f"{'='*70}")
        print("\nWeights:")
        for ticker, weight in result['weights'].items():
            print(f"  {ticker}: {weight*100:6.2f}%")

        print(f"\nPortfolio Metrics:")
        print(f"  Return: {result['port_return']*100:.2f}%")
        print(f"  Volatility: {result['port_vol']*100:.2f}%")
        print(f"  Sharpe: {result['sharpe']:.3f}")
        if result['realized_beta']:
            print(f"  Beta: {result['realized_beta']:.3f}")

        # Calculate shares
        capital = 100000
        current_prices = data.iloc[-1]
        holdings = calculate_portfolio_shares(result['weights'], capital, current_prices)

        print(f"\nHoldings (Capital: ${capital:,.0f}):")
        print(holdings.to_string(index=False))
