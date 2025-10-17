"""
Advanced Monte Carlo Simulations
================================
Jump diffusion, fat tails, variance reduction techniques, and regime switching.

Implements:
- Merton jump diffusion model
- Student's t-distribution (fat tails)
- Variance reduction (antithetic variates, control variates)
- Regime switching models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class JumpDiffusionParams:
    """Parameters for Merton jump diffusion model."""
    mu: float  # Drift
    sigma: float  # Diffusion volatility
    jump_lambda: float  # Jump intensity (jumps per year)
    jump_mean: float  # Mean jump size (log scale)
    jump_std: float  # Jump size volatility


class JumpDiffusionSimulator:
    """
    Merton jump diffusion model: dS = μS dt + σS dW + S(e^J - 1)dN

    Models both continuous diffusion AND discrete jumps (crashes/rallies).
    More realistic than pure GBM for financial assets.

    Example:
        >>> params = JumpDiffusionParams(
        ...     mu=0.10, sigma=0.20,
        ...     jump_lambda=2, jump_mean=-0.05, jump_std=0.10
        ... )
        >>> sim = JumpDiffusionSimulator(params)
        >>> paths = sim.simulate(S0=100, T=1.0, steps=252, n_paths=10000)
    """

    def __init__(self, params: JumpDiffusionParams):
        """
        Initialize jump diffusion simulator.

        Args:
            params: Jump diffusion parameters
        """
        self.params = params

    def simulate(self, S0: float, T: float, steps: int,
                 n_paths: int = 10000) -> np.ndarray:
        """
        Simulate jump diffusion paths.

        Args:
            S0: Initial price
            T: Time horizon (years)
            steps: Number of time steps
            n_paths: Number of simulation paths

        Returns:
            Array of shape (n_paths, steps+1) with simulated prices
        """
        dt = T / steps

        # Initialize price paths
        S = np.zeros((n_paths, steps + 1))
        S[:, 0] = S0

        # Simulate for each time step
        for t in range(1, steps + 1):
            # 1. Diffusion component (continuous)
            Z = np.random.standard_normal(n_paths)
            diffusion = (self.params.mu - 0.5 * self.params.sigma**2) * dt + \
                       self.params.sigma * np.sqrt(dt) * Z

            # 2. Jump component (discrete)
            # Number of jumps: Poisson(lambda * dt)
            n_jumps = np.random.poisson(self.params.jump_lambda * dt, n_paths)

            # Jump sizes: log-normal distribution
            jump_component = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jumps = np.random.normal(
                        self.params.jump_mean,
                        self.params.jump_std,
                        n_jumps[i]
                    )
                    jump_component[i] = np.sum(jumps)

            # 3. Combine diffusion and jumps
            S[:, t] = S[:, t-1] * np.exp(diffusion + jump_component)

        return S

    def estimate_parameters_from_data(self, returns: np.ndarray,
                                     dt: float = 1/252) -> JumpDiffusionParams:
        """
        Estimate jump diffusion parameters from historical returns.

        Uses method of moments estimation.

        Args:
            returns: Array of log returns
            dt: Time increment between returns (1/252 for daily)

        Returns:
            Estimated parameters
        """
        # Remove extreme outliers to separate jumps from diffusion
        threshold = 3 * np.std(returns)
        normal_returns = returns[np.abs(returns) < threshold]
        jump_returns = returns[np.abs(returns) >= threshold]

        # Estimate diffusion parameters from normal returns
        sigma = np.std(normal_returns) / np.sqrt(dt)
        mu = np.mean(normal_returns) / dt + 0.5 * sigma**2

        # Estimate jump parameters
        jump_lambda = len(jump_returns) / (len(returns) * dt)

        if len(jump_returns) > 0:
            jump_mean = np.mean(jump_returns)
            jump_std = np.std(jump_returns)
        else:
            jump_mean = -0.05  # Default: negative jumps (crashes)
            jump_std = 0.10

        return JumpDiffusionParams(
            mu=mu,
            sigma=sigma,
            jump_lambda=jump_lambda,
            jump_mean=jump_mean,
            jump_std=jump_std
        )


class FatTailSimulator:
    """
    Simulate returns with fat tails using Student's t-distribution.
    Captures extreme events better than normal distribution.
    """

    def __init__(self, mu: float, sigma: float, df: float = 5):
        """
        Initialize fat tail simulator.

        Args:
            mu: Expected return
            sigma: Volatility
            df: Degrees of freedom (lower = fatter tails, typically 3-7)
        """
        self.mu = mu
        self.sigma = sigma
        self.df = df

    def simulate(self, S0: float, T: float, steps: int,
                 n_paths: int = 10000) -> np.ndarray:
        """
        Simulate price paths using Student's t-distribution.

        Returns:
            Array of shape (n_paths, steps+1) with simulated prices
        """
        dt = T / steps

        S = np.zeros((n_paths, steps + 1))
        S[:, 0] = S0

        # Student's t random variables
        # Scale to match desired volatility
        scale_factor = np.sqrt((self.df - 2) / self.df) if self.df > 2 else 1.0

        for t in range(1, steps + 1):
            # Generate t-distributed returns
            Z = np.random.standard_t(self.df, n_paths) * scale_factor

            returns = (self.mu - 0.5 * self.sigma**2) * dt + \
                     self.sigma * np.sqrt(dt) * Z

            S[:, t] = S[:, t-1] * np.exp(returns)

        return S


class VarianceReductionMC:
    """
    Monte Carlo with variance reduction techniques.
    Achieves same accuracy with fewer simulations.
    """

    @staticmethod
    def antithetic_variates(S0: float, mu: float, sigma: float,
                           T: float, steps: int, n_paths: int = 5000) -> np.ndarray:
        """
        Antithetic variates: For each random path, also simulate with -Z.
        Reduces variance by exploiting symmetry.

        Achieves same accuracy as 2*n_paths standard MC paths.

        Args:
            S0: Initial price
            mu: Drift
            sigma: Volatility
            T: Time horizon
            steps: Number of time steps
            n_paths: Number of path pairs (total paths = n_paths)

        Returns:
            Simulated price paths
        """
        dt = T / steps
        half_paths = n_paths // 2

        S = np.zeros((n_paths, steps + 1))
        S[:, 0] = S0

        for t in range(1, steps + 1):
            # Generate random numbers
            Z = np.random.standard_normal(half_paths)

            # First half: standard path
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion1 = drift + sigma * np.sqrt(dt) * Z
            S[:half_paths, t] = S[:half_paths, t-1] * np.exp(diffusion1)

            # Second half: antithetic path (use -Z)
            diffusion2 = drift + sigma * np.sqrt(dt) * (-Z)
            S[half_paths:, t] = S[half_paths:, t-1] * np.exp(diffusion2)

        return S

    @staticmethod
    def control_variates(paths: np.ndarray, control_mean: float) -> np.ndarray:
        """
        Control variates: Use known analytic result to reduce variance.

        For GBM, we know E[S_T] = S_0 * exp(μT).
        Adjust MC estimates using deviation from this known mean.

        Args:
            paths: Simulated final prices (n_paths,)
            control_mean: Known theoretical mean E[S_T]

        Returns:
            Variance-reduced estimates
        """
        # Sample mean
        sample_mean = np.mean(paths)

        # Optimal adjustment coefficient
        # c* = Cov(X, C) / Var(C) where C is control variate
        if np.var(paths) > 0:
            c = np.cov(paths, paths)[0, 1] / np.var(paths)
        else:
            c = 0

        # Adjusted estimator
        adjusted = paths - c * (sample_mean - control_mean)

        return adjusted


class RegimeSwitchingMC:
    """
    Simulate prices with regime switching (bull/bear markets).
    Uses hidden Markov model approach.
    """

    def __init__(self, regime_params: List[Tuple[float, float]],
                 transition_probs: np.ndarray):
        """
        Initialize regime switching simulator.

        Args:
            regime_params: List of (mu, sigma) for each regime
                          e.g., [(0.15, 0.15), (-0.10, 0.30)] for bull/bear
            transition_probs: Matrix of transition probabilities
                             P[i,j] = prob of moving from regime i to j
        """
        self.regime_params = regime_params
        self.transition_probs = transition_probs
        self.n_regimes = len(regime_params)

    def simulate(self, S0: float, T: float, steps: int,
                 initial_regime: int = 0, n_paths: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate with regime switching.

        Args:
            S0: Initial price
            T: Time horizon
            steps: Number of time steps
            initial_regime: Starting regime (0=bull, 1=bear)
            n_paths: Number of simulation paths

        Returns:
            Tuple of (price_paths, regime_paths)
        """
        dt = T / steps

        S = np.zeros((n_paths, steps + 1))
        regimes = np.zeros((n_paths, steps + 1), dtype=int)

        S[:, 0] = S0
        regimes[:, 0] = initial_regime

        for t in range(1, steps + 1):
            for path in range(n_paths):
                current_regime = regimes[path, t-1]

                # Determine next regime
                next_regime = np.random.choice(
                    self.n_regimes,
                    p=self.transition_probs[current_regime, :]
                )
                regimes[path, t] = next_regime

                # Simulate return with regime parameters
                mu, sigma = self.regime_params[next_regime]
                Z = np.random.standard_normal()

                ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
                S[path, t] = S[path, t-1] * np.exp(ret)

        return S, regimes


def calculate_path_statistics(paths: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics from MC paths.

    Args:
        paths: Array of shape (n_paths, steps+1)

    Returns:
        Dictionary with statistics
    """
    final_prices = paths[:, -1]

    stats = {
        'mean': float(np.mean(final_prices)),
        'median': float(np.median(final_prices)),
        'std': float(np.std(final_prices)),
        'min': float(np.min(final_prices)),
        'max': float(np.max(final_prices)),
        'percentile_5': float(np.percentile(final_prices, 5)),
        'percentile_25': float(np.percentile(final_prices, 25)),
        'percentile_75': float(np.percentile(final_prices, 75)),
        'percentile_95': float(np.percentile(final_prices, 95)),
    }

    return stats


def calculate_risk_metrics(paths: np.ndarray, S0: float, confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate risk metrics from MC paths.

    Args:
        paths: Simulated price paths
        S0: Initial price
        confidence_level: Confidence level for VaR/CVaR

    Returns:
        Risk metrics including VaR, CVaR, probability of loss
    """
    final_prices = paths[:, -1]
    returns = (final_prices - S0) / S0

    # Value at Risk
    var_quantile = 1 - confidence_level
    var = -np.percentile(returns, var_quantile * 100)

    # Conditional Value at Risk (Expected Shortfall)
    var_threshold = np.percentile(returns, var_quantile * 100)
    cvar = -np.mean(returns[returns <= var_threshold])

    # Probability of loss
    prob_loss = (returns < 0).mean()

    # Maximum drawdown
    max_prices = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - max_prices) / max_prices
    max_drawdown = float(np.min(drawdowns))

    return {
        'var': float(var),
        'cvar': float(cvar),
        'prob_loss': float(prob_loss),
        'max_drawdown': float(max_drawdown),
        'expected_return': float(np.mean(returns)),
        'volatility': float(np.std(returns))
    }
