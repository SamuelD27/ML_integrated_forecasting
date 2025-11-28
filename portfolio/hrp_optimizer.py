"""
Hierarchical Risk Parity (HRP) Portfolio Optimization
======================================================
Robust portfolio construction using hierarchical clustering.

Key Advantages:
1. No optimization required (no matrix inversion)
2. Robust to estimation errors in covariance
3. Outperforms mean-variance out-of-sample
4. Incorporates diversification at all hierarchical levels

Reference:
- Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample"
- Journal of Portfolio Management, 42(4), 59-69
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# Import Ledoit-Wolf for shrinkage covariance (RECOMMENDED)
try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class HRPOptimizer:
    """
    Hierarchical Risk Parity portfolio optimization.

    Algorithm:
    1. Tree Clustering: Cluster assets using hierarchical clustering
    2. Quasi-Diagonalization: Reorder covariance matrix to cluster similar assets
    3. Recursive Bisection: Allocate weights recursively based on cluster variance

    Example:
        >>> hrp = HRPOptimizer()
        >>> returns = pd.DataFrame(...)  # Historical returns
        >>> weights = hrp.allocate(returns)
        >>> print(weights)
    """

    def __init__(
        self,
        linkage_method: str = 'single',
        distance_metric: str = 'correlation',
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        use_shrinkage: bool = True
    ):
        """
        Initialize HRP optimizer.

        Args:
            linkage_method: Clustering linkage method
                - 'single': Minimum distance (Lopez de Prado default)
                - 'complete': Maximum distance
                - 'average': Average distance
                - 'ward': Minimizes variance
            distance_metric: Distance metric for clustering
                - 'correlation': 1 - correlation (default)
                - 'euclidean': Euclidean distance
                - 'manhattan': Manhattan distance
            min_weight: Minimum asset weight (default: 0%)
            max_weight: Maximum asset weight (default: 100%)
            use_shrinkage: Whether to use Ledoit-Wolf shrinkage for covariance
                (RECOMMENDED for stability, default: True)
        """
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_shrinkage = use_shrinkage
        self.shrinkage_coefficient = None

    def _compute_covariance_and_correlation(
        self,
        returns: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute covariance and correlation matrices.

        If use_shrinkage is True (default), uses Ledoit-Wolf shrinkage for
        more stable covariance estimates. This is RECOMMENDED for HRP
        despite not inverting the covariance matrix.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Tuple of (covariance_matrix, correlation_matrix)
        """
        if self.use_shrinkage and HAS_SKLEARN:
            n_obs, n_assets = returns.shape

            if n_obs > n_assets:  # Shrinkage requires more obs than assets
                lw = LedoitWolf()
                lw.fit(returns.values)

                self.shrinkage_coefficient = lw.shrinkage_

                cov = pd.DataFrame(
                    lw.covariance_,
                    index=returns.columns,
                    columns=returns.columns
                )

                # Derive correlation from shrinkage covariance
                std_devs = np.sqrt(np.diag(cov.values))
                std_outer = np.outer(std_devs, std_devs)
                corr_values = cov.values / (std_outer + 1e-10)
                # Ensure diagonal is exactly 1
                np.fill_diagonal(corr_values, 1.0)
                corr = pd.DataFrame(
                    corr_values,
                    index=returns.columns,
                    columns=returns.columns
                )

                logger.info(f"✓ Ledoit-Wolf shrinkage: {lw.shrinkage_:.3f}")
                return cov, corr

            else:
                logger.warning(
                    f"Insufficient data for shrinkage: {n_obs} obs <= {n_assets} assets. "
                    f"Using sample covariance."
                )

        # Fallback to sample covariance
        self.shrinkage_coefficient = 0.0
        cov = returns.cov()
        corr = returns.corr()

        return cov, corr

    def allocate(
        self,
        returns: pd.DataFrame,
        min_periods: int = 60
    ) -> pd.Series:
        """
        Calculate HRP portfolio weights.

        Args:
            returns: DataFrame of asset returns (columns = assets, rows = time)
            min_periods: Minimum periods required for covariance calculation

        Returns:
            Series of portfolio weights (index = assets, values = weights)

        Raises:
            ValueError: If insufficient data or singular covariance matrix
        """
        # Validate inputs
        if returns.shape[0] < min_periods:
            raise ValueError(
                f"Insufficient data: {returns.shape[0]} rows < {min_periods} min_periods"
            )

        if returns.shape[1] < 2:
            raise ValueError(
                f"At least 2 assets required for HRP, got {returns.shape[1]}"
            )

        # Clean data
        returns = returns.dropna(axis=1, how='all')  # Drop all-NaN columns
        returns = returns.fillna(0)  # Fill remaining NaNs with 0

        logger.info(f"HRP optimization for {returns.shape[1]} assets, "
                   f"{returns.shape[0]} observations")

        # Calculate covariance matrix (with optional Ledoit-Wolf shrinkage)
        cov, corr = self._compute_covariance_and_correlation(returns)

        # Validate covariance matrix
        if not self._validate_covariance_matrix(cov):
            logger.warning("Covariance matrix validation failed, results may be unreliable")

        # Step 1: Tree clustering
        distance_matrix = self._calculate_distance_matrix(corr)
        link = linkage(squareform(distance_matrix), method=self.linkage_method)

        # Step 2: Quasi-diagonalization
        sort_idx = self._quasi_diag(link)
        sorted_tickers = returns.columns[sort_idx].tolist()

        # Step 3: Recursive bisection
        weights = self._recursive_bisection(cov, sorted_tickers)

        # Apply weight constraints
        weights = self._apply_constraints(weights)

        # Normalize to sum to 1
        weights = weights / weights.sum()

        # POST-ALLOCATION VALIDATION (MANDATORY)
        self._validate_hrp_weights(weights)

        logger.info(f"HRP weights calculated: min={weights.min():.3f}, "
                   f"max={weights.max():.3f}, std={weights.std():.3f}")

        return weights

    def _validate_hrp_weights(
        self,
        weights: pd.Series,
        tolerance: float = 1e-6
    ) -> None:
        """
        Validate HRP weights with assertions.

        Args:
            weights: Portfolio weights
            tolerance: Numerical tolerance

        Raises:
            AssertionError: If weights invalid
        """
        # ENFORCE: No NaN or infinite
        assert not weights.isna().any(), "Weights contain NaN"
        assert not np.isinf(weights).any(), "Weights contain infinite values"

        # ENFORCE: Non-negative (HRP is long-only by design)
        assert (weights >= -tolerance).all(), \
            f"Negative weights found: {weights[weights < -tolerance].to_dict()}"

        # ENFORCE: Sum to 1
        assert abs(weights.sum() - 1.0) < tolerance, \
            f"Weights sum to {weights.sum():.8f}, not 1.0"

        # ENFORCE: Max weight constraint
        assert (weights <= self.max_weight + tolerance).all(), \
            f"Max weight violated: {weights.max():.4f} > {self.max_weight}"

        logger.info(f"✓ HRP weights validated: sum={weights.sum():.6f}, "
                   f"min={weights.min():.4f}, max={weights.max():.4f}")

    def allocate_with_metadata(
        self,
        returns: pd.DataFrame,
        min_periods: int = 60
    ) -> Dict:
        """
        Calculate HRP weights with additional metadata.

        Returns:
            Dictionary with:
                - weights: Portfolio weights (pd.Series)
                - sorted_tickers: Asset ordering from clustering
                - linkage_matrix: Hierarchical clustering linkage
                - distance_matrix: Pairwise distance matrix
                - cluster_variance: Variance contribution by cluster
        """
        # Calculate distance and linkage
        returns_clean = returns.dropna(axis=1, how='all').fillna(0)
        cov, corr = self._compute_covariance_and_correlation(returns_clean)

        distance_matrix = self._calculate_distance_matrix(corr)
        link = linkage(squareform(distance_matrix), method=self.linkage_method)

        # Quasi-diagonalization
        sort_idx = self._quasi_diag(link)
        sorted_tickers = returns_clean.columns[sort_idx].tolist()

        # Recursive bisection
        weights = self._recursive_bisection(cov, sorted_tickers)
        weights = self._apply_constraints(weights)
        weights = weights / weights.sum()

        # Calculate cluster variance contributions
        cluster_var = self._calculate_cluster_variance(weights, cov)

        return {
            'weights': weights,
            'sorted_tickers': sorted_tickers,
            'linkage_matrix': link,
            'distance_matrix': distance_matrix,
            'cluster_variance': cluster_var
        }

    def _calculate_distance_matrix(self, corr: pd.DataFrame) -> np.ndarray:
        """
        Calculate distance matrix from correlation matrix with numerical stability.

        Args:
            corr: Correlation matrix

        Returns:
            Distance matrix (condensed form for scipy linkage)
        """
        # Validate correlation matrix
        if not self._validate_correlation_matrix(corr):
            logger.warning("Correlation matrix validation failed, proceeding with caution")

        # Clip correlations to valid range [-1, 1] (handles numerical errors)
        corr_clipped = np.clip(corr.values, -1.0, 1.0)
        corr = pd.DataFrame(corr_clipped, index=corr.index, columns=corr.columns)

        if self.distance_metric == 'correlation':
            # Lopez de Prado: distance = sqrt(0.5 * (1 - correlation))
            # Ensure (1 - corr) / 2 is non-negative before sqrt
            dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
        elif self.distance_metric == 'euclidean':
            # Euclidean distance between correlation vectors
            # Ensure 2 * (1 - corr) is non-negative before sqrt
            dist = np.sqrt(np.maximum(2 * (1 - corr), 0))
        else:
            # Manhattan distance
            dist = 1 - np.abs(corr)

        # Ensure valid distance matrix
        dist = dist.fillna(1)  # NaN correlations = max distance
        dist = dist.clip(lower=0)  # No negative distances

        # Check for NaN or Inf
        if np.any(~np.isfinite(dist)):
            logger.error("Distance matrix contains NaN or Inf values!")
            raise ValueError("Invalid distance matrix")

        return dist.values

    def _validate_covariance_matrix(self, cov: pd.DataFrame) -> bool:
        """
        Validate covariance matrix properties.

        Args:
            cov: Covariance matrix

        Returns:
            True if validation passes
        """
        # Check symmetry
        if not np.allclose(cov, cov.T, rtol=1e-5):
            logger.error("Covariance matrix is not symmetric!")
            return False

        # Check positive semi-definite (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvals(cov.values)
        if np.any(eigenvalues < -1e-8):  # Allow small numerical errors
            logger.error(f"Covariance matrix is not PSD! Min eigenvalue: {eigenvalues.min()}")
            return False

        # Check for reasonable values
        variances = np.diag(cov)
        if np.any(variances < 0):
            logger.error("Negative variances detected!")
            return False

        if np.any(variances == 0):
            zero_var_assets = cov.index[variances == 0].tolist()
            logger.error(f"Zero variance assets detected: {zero_var_assets}")
            return False

        # Check for very small variances (numerical instability)
        min_variance = 1e-8
        if np.any(variances < min_variance):
            small_var_assets = cov.index[variances < min_variance].tolist()
            logger.warning(f"Assets with very small variance: {small_var_assets}")

        # Check correlations are in valid range
        std_devs = np.sqrt(variances)
        std_outer = np.outer(std_devs, std_devs)
        epsilon = 1e-10
        corr = cov / (std_outer + epsilon)

        if np.any(np.abs(corr) > 1.001):  # Allow tiny numerical error
            logger.error("Invalid correlations detected in covariance matrix (abs > 1)")
            return False

        logger.info("✓ Covariance matrix validation passed")
        return True

    def _validate_correlation_matrix(self, corr: pd.DataFrame) -> bool:
        """
        Validate correlation matrix properties.

        Args:
            corr: Correlation matrix

        Returns:
            True if validation passes
        """
        # Check symmetry
        if not np.allclose(corr, corr.T, rtol=1e-5):
            logger.error("Correlation matrix is not symmetric!")
            return False

        # Check correlations are in valid range [-1, 1]
        if np.any(np.abs(corr.values) > 1.001):  # Allow tiny numerical error
            max_corr = np.abs(corr.values).max()
            logger.error(f"Invalid correlations detected (abs > 1): max={max_corr}")
            return False

        # Check diagonal is 1
        diagonal = np.diag(corr.values)
        if not np.allclose(diagonal, 1.0, rtol=1e-5):
            logger.warning(f"Correlation diagonal not 1.0: {diagonal}")

        logger.info("✓ Correlation matrix validation passed")
        return True

    def _quasi_diag(self, link: np.ndarray) -> List[int]:
        """
        Quasi-diagonalization: reorder assets to cluster similar assets.

        Args:
            link: Linkage matrix from hierarchical clustering

        Returns:
            List of asset indices in clustered order
        """
        # Number of original observations (leaves)
        n = link.shape[0] + 1

        # Recursive function to get sorted indices
        def _get_quasi_diag(link, sort_idx):
            if len(sort_idx) == 0:
                return []
            if len(sort_idx) == 1:
                return sort_idx

            # Get clusters from linkage
            clusters = [sort_idx]

            while len(clusters) < len(sort_idx):
                new_clusters = []
                for cluster in clusters:
                    if len(cluster) > 1:
                        # Find linkage row that created this cluster
                        idx = None
                        for i, row in enumerate(link):
                            if set([int(row[0]), int(row[1])]).issubset(set(cluster)):
                                idx = i
                                break

                        if idx is not None:
                            # Split cluster
                            left = int(link[idx, 0])
                            right = int(link[idx, 1])

                            # Convert cluster indices
                            left_cluster = [c for c in cluster if c == left or
                                          (left >= n and c in self._get_cluster_members(link, left, n))]
                            right_cluster = [c for c in cluster if c == right or
                                           (right >= n and c in self._get_cluster_members(link, right, n))]

                            if len(left_cluster) > 0:
                                new_clusters.append(left_cluster)
                            if len(right_cluster) > 0:
                                new_clusters.append(right_cluster)
                        else:
                            new_clusters.append(cluster)
                    else:
                        new_clusters.append(cluster)

                clusters = new_clusters

            # Flatten and return
            result = []
            for cluster in clusters:
                result.extend(cluster)
            return result

        # Simplified approach: use dendrogram ordering
        from scipy.cluster.hierarchy import dendrogram
        dend = dendrogram(link, no_plot=True)
        return dend['leaves']

    def _get_cluster_members(self, link: np.ndarray, cluster_idx: int, n: int) -> List[int]:
        """Get all leaf members of a cluster."""
        if cluster_idx < n:
            return [cluster_idx]

        # Cluster is internal node
        node_idx = int(cluster_idx - n)
        left = int(link[node_idx, 0])
        right = int(link[node_idx, 1])

        members = []
        members.extend(self._get_cluster_members(link, left, n))
        members.extend(self._get_cluster_members(link, right, n))

        return members

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sorted_tickers: List[str]
    ) -> pd.Series:
        """
        Recursive bisection to allocate weights.

        Key Insight: At each split, allocate to clusters inversely proportional
        to their variance. Recurse until individual assets.

        Args:
            cov: Covariance matrix
            sorted_tickers: Asset tickers in clustered order

        Returns:
            Portfolio weights (pd.Series)
        """
        weights = pd.Series(1.0, index=sorted_tickers)
        clusters = [sorted_tickers]

        while len(clusters) < len(sorted_tickers):
            # Split each cluster with > 1 asset
            new_clusters = []

            for cluster in clusters:
                if len(cluster) > 1:
                    # Split cluster in half
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]

                    # Calculate cluster variances
                    left_var = self._cluster_variance(cov, left)
                    right_var = self._cluster_variance(cov, right)

                    # Allocate inversely proportional to variance
                    # (lower variance → higher weight)
                    total_inv_var = 1.0 / left_var + 1.0 / right_var
                    left_alloc = (1.0 / left_var) / total_inv_var
                    right_alloc = (1.0 / right_var) / total_inv_var

                    # Update weights
                    current_weight = weights[cluster].iloc[0]
                    weights[left] *= left_alloc
                    weights[right] *= right_alloc

                    new_clusters.extend([left, right])
                else:
                    new_clusters.append(cluster)

            clusters = new_clusters

        return weights

    def _cluster_variance(self, cov: pd.DataFrame, tickers: List[str]) -> float:
        """
        Calculate variance of equally-weighted cluster.

        Args:
            cov: Covariance matrix
            tickers: Tickers in cluster

        Returns:
            Cluster variance
        """
        if len(tickers) == 1:
            return cov.loc[tickers[0], tickers[0]]

        # Equal weight within cluster
        w = np.ones(len(tickers)) / len(tickers)
        cov_cluster = cov.loc[tickers, tickers].values

        # Portfolio variance: w^T @ Cov @ w
        variance = w @ cov_cluster @ w

        return variance

    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """
        Apply min/max weight constraints.

        Args:
            weights: Unconstrained weights

        Returns:
            Constrained weights (may not sum to 1 - requires renormalization)
        """
        # Clip to bounds
        weights = weights.clip(lower=self.min_weight, upper=self.max_weight)

        return weights

    def _calculate_cluster_variance(
        self,
        weights: pd.Series,
        cov: pd.DataFrame
    ) -> float:
        """Calculate variance contribution of each cluster."""
        w = weights.values
        cov_matrix = cov.loc[weights.index, weights.index].values

        # Portfolio variance
        portfolio_var = w @ cov_matrix @ w

        return portfolio_var


def compare_hrp_vs_meanvar(
    returns: pd.DataFrame,
    risk_aversion: float = 5.0
) -> pd.DataFrame:
    """
    Compare HRP vs Mean-Variance optimization.

    Args:
        returns: Historical returns
        risk_aversion: Risk aversion parameter for mean-variance

    Returns:
        DataFrame comparing allocations and metrics
    """
    # HRP allocation
    hrp = HRPOptimizer()
    hrp_weights = hrp.allocate(returns)

    # Mean-variance allocation (using CVaR allocator as fallback)
    try:
        from portfolio.cvar_allocator import CVaRAllocator

        mv_allocator = CVaRAllocator(
            capital=100000,
            risk_aversion=risk_aversion,
            max_position_pct=1.0  # No constraints for fair comparison
        )

        # Prepare data
        prices = (1 + returns).cumprod()
        current_prices = prices.iloc[-1]

        allocation_result = mv_allocator.optimize(
            universe=returns.columns.tolist(),
            prices=current_prices.to_dict(),
            historical_returns=returns
        )

        mv_weights = pd.Series(allocation_result['weights'])
        mv_weights = mv_weights / mv_weights.sum()  # Normalize

    except Exception as e:
        logger.warning(f"Mean-variance optimization failed: {e}, using equal weight")
        mv_weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)

    # Compare
    comparison = pd.DataFrame({
        'HRP': hrp_weights,
        'MeanVar': mv_weights,
        'Difference': hrp_weights - mv_weights
    })

    # Calculate metrics
    from utils.financial_metrics import FinancialMetrics

    hrp_returns = (returns * hrp_weights).sum(axis=1)
    mv_returns = (returns * mv_weights).sum(axis=1)

    hrp_metrics = FinancialMetrics(hrp_returns).summary()
    mv_metrics = FinancialMetrics(mv_returns).summary()

    print("\n" + "="*60)
    print("HRP vs Mean-Variance Comparison")
    print("="*60)
    print("\nWeight Allocation:")
    print(comparison.round(4))

    print("\n" + "-"*60)
    print("Performance Metrics:")
    print("-"*60)

    metrics_comparison = pd.DataFrame({
        'HRP': hrp_metrics,
        'MeanVar': mv_metrics
    }).T

    print(metrics_comparison[['annualized_return', 'volatility', 'sharpe_ratio',
                              'max_drawdown', 'calmar_ratio']].round(4))

    return comparison


def allocate_hrp(
    returns: pd.DataFrame,
    **kwargs
) -> pd.Series:
    """
    Convenience function for HRP allocation.

    Args:
        returns: Historical returns DataFrame
        **kwargs: Additional arguments for HRPOptimizer

    Returns:
        Portfolio weights

    Example:
        >>> returns = pd.DataFrame(...)
        >>> weights = allocate_hrp(returns, linkage_method='ward')
        >>> print(weights)
    """
    hrp = HRPOptimizer(**kwargs)
    return hrp.allocate(returns)


if __name__ == '__main__':
    # Example usage
    import yfinance as yf
    from datetime import datetime, timedelta

    # Fetch data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)

    print("Fetching data...")
    prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = prices.pct_change().dropna()

    print(f"\nData: {len(tickers)} assets, {len(returns)} days")

    # HRP allocation
    print("\n" + "="*60)
    print("Hierarchical Risk Parity Allocation")
    print("="*60)

    hrp = HRPOptimizer(linkage_method='single')
    result = hrp.allocate_with_metadata(returns)

    weights = result['weights']
    sorted_tickers = result['sorted_tickers']

    print("\nCluster Ordering (similar assets grouped):")
    print(sorted_tickers)

    print("\nHRP Weights:")
    print(weights.sort_values(ascending=False).round(4))

    print(f"\nWeight Statistics:")
    print(f"  Min: {weights.min():.2%}")
    print(f"  Max: {weights.max():.2%}")
    print(f"  Std: {weights.std():.2%}")
    print(f"  Concentration (top 3): {weights.nlargest(3).sum():.2%}")

    # Compare with mean-variance
    print("\n" + "="*60)
    comparison = compare_hrp_vs_meanvar(returns, risk_aversion=5.0)
