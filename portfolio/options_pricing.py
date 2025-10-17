"""
Options Pricing and Greeks Calculation
======================================
Professional-grade options pricing with full Greeks support.

Implements:
- Black-Scholes-Merton model
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility calculation
- ML-integrated options strategies
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Tuple, Optional, List
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptionParameters:
    """Parameters for option pricing."""
    S: float  # Current stock price
    K: float  # Strike price
    T: float  # Time to expiration (years)
    r: float  # Risk-free rate
    sigma: float  # Volatility (annualized)
    q: float = 0.0  # Dividend yield


class BlackScholesModel:
    """
    Black-Scholes-Merton option pricing with Greeks.

    Example:
        >>> params = OptionParameters(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
        >>> bs = BlackScholesModel(params)
        >>> call_price = bs.call_price()
        >>> greeks = bs.greeks('call')
    """

    def __init__(self, params: OptionParameters):
        """
        Initialize Black-Scholes model.

        Args:
            params: Option parameters
        """
        self.params = params
        self._validate_parameters()
        self._d1 = None
        self._d2 = None

    def _validate_parameters(self):
        """Validate input parameters."""
        if self.params.S <= 0:
            raise ValueError(f"Stock price must be positive: {self.params.S}")
        if self.params.K <= 0:
            raise ValueError(f"Strike price must be positive: {self.params.K}")
        if self.params.T <= 0:
            raise ValueError(f"Time to expiration must be positive: {self.params.T}")
        if self.params.sigma <= 0:
            raise ValueError(f"Volatility must be positive: {self.params.sigma}")

    def _calculate_d1_d2(self) -> Tuple[float, float]:
        """
        Calculate d1 and d2 terms.

        Returns:
            Tuple of (d1, d2)
        """
        if self._d1 is not None and self._d2 is not None:
            return self._d1, self._d2

        S, K, T, r, sigma, q = (
            self.params.S, self.params.K, self.params.T,
            self.params.r, self.params.sigma, self.params.q
        )

        # d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        self._d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        # d2 = d1 - σ√T
        self._d2 = self._d1 - sigma * np.sqrt(T)

        return self._d1, self._d2

    def call_price(self) -> float:
        """
        Calculate European call option price.

        Formula: C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)

        Returns:
            Call option price
        """
        d1, d2 = self._calculate_d1_d2()
        S, K, T, r, q = self.params.S, self.params.K, self.params.T, self.params.r, self.params.q

        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return float(call)

    def put_price(self) -> float:
        """
        Calculate European put option price.

        Formula: P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

        Returns:
            Put option price
        """
        d1, d2 = self._calculate_d1_d2()
        S, K, T, r, q = self.params.S, self.params.K, self.params.T, self.params.r, self.params.q

        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return float(put)

    def greeks(self, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for the option.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Dictionary with Delta, Gamma, Vega, Theta, Rho
        """
        d1, d2 = self._calculate_d1_d2()
        S, K, T, r, sigma, q = (
            self.params.S, self.params.K, self.params.T,
            self.params.r, self.params.sigma, self.params.q
        )

        # Standard normal PDF at d1
        nd1 = norm.pdf(d1)

        # Delta
        if option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:  # put
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

        # Gamma (same for call and put)
        gamma = np.exp(-q * T) * nd1 / (S * sigma * np.sqrt(T))

        # Vega (same for call and put) - per 1% change in volatility
        vega = S * np.exp(-q * T) * nd1 * np.sqrt(T) / 100

        # Theta (different for call and put) - per day
        theta_term1 = -(S * nd1 * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))

        if option_type == 'call':
            theta_term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta_term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
            theta = (theta_term1 + theta_term2 + theta_term3) / 365
        else:  # put
            theta_term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta_term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            theta = (theta_term1 + theta_term2 + theta_term3) / 365

        # Rho (different for call and put) - per 1% change in interest rate
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta),
            'rho': float(rho)
        }


class ImpliedVolatility:
    """Calculate implied volatility from market option prices."""

    @staticmethod
    def calculate(market_price: float, params: OptionParameters,
                  option_type: str = 'call', max_iterations: int = 100,
                  tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            market_price: Observed market price of option
            params: Option parameters (with initial sigma guess)
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance

        Returns:
            Implied volatility (annualized) or None if no convergence
        """
        # Initial guess (use provided sigma or 0.20)
        sigma = params.sigma if params.sigma > 0 else 0.20

        for i in range(max_iterations):
            # Update params with current sigma guess
            test_params = OptionParameters(
                S=params.S, K=params.K, T=params.T,
                r=params.r, sigma=sigma, q=params.q
            )

            bs = BlackScholesModel(test_params)

            # Calculate option price and vega
            if option_type == 'call':
                model_price = bs.call_price()
            else:
                model_price = bs.put_price()

            greeks = bs.greeks(option_type)
            vega = greeks['vega'] * 100  # Convert back to per unit change

            # Price difference
            diff = model_price - market_price

            # Check convergence
            if abs(diff) < tolerance:
                return sigma

            # Newton-Raphson update: σ_new = σ_old - f(σ)/f'(σ)
            if abs(vega) < 1e-10:
                return None  # Vega too small, cannot converge

            sigma = sigma - diff / vega

            # Keep sigma in reasonable range
            sigma = max(0.001, min(sigma, 5.0))

        return None  # Did not converge


class OptionStrategy:
    """Pre-built option strategies for portfolio hedging."""

    @staticmethod
    def protective_put(stock_price: float, stock_shares: int,
                       put_strike: float, params: OptionParameters) -> Dict:
        """
        Protective Put: Long stock + Long put.
        Limits downside risk while maintaining upside potential.

        Args:
            stock_price: Current stock price
            stock_shares: Number of shares owned
            put_strike: Strike price for protective put
            params: Option parameters

        Returns:
            Strategy details including cost, max loss, breakeven
        """
        # Put option cost
        put_params = OptionParameters(
            S=stock_price, K=put_strike, T=params.T,
            r=params.r, sigma=params.sigma, q=params.q
        )
        bs = BlackScholesModel(put_params)
        put_price = bs.put_price()

        total_cost = put_price * stock_shares * 100  # Contract multiplier

        # Max loss = current stock value - strike value + put cost
        max_loss = (stock_price - put_strike) * stock_shares + total_cost

        # Breakeven = current price + put cost per share
        breakeven = stock_price + put_price

        return {
            'strategy': 'Protective Put',
            'put_strike': put_strike,
            'put_price': put_price,
            'total_cost': total_cost,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'max_profit': 'unlimited',
            'greeks': bs.greeks('put')
        }

    @staticmethod
    def covered_call(stock_price: float, stock_shares: int,
                     call_strike: float, params: OptionParameters) -> Dict:
        """
        Covered Call: Long stock + Short call.
        Generates income but caps upside potential.

        Args:
            stock_price: Current stock price
            stock_shares: Number of shares owned
            call_strike: Strike price for covered call
            params: Option parameters

        Returns:
            Strategy details
        """
        call_params = OptionParameters(
            S=stock_price, K=call_strike, T=params.T,
            r=params.r, sigma=params.sigma, q=params.q
        )
        bs = BlackScholesModel(call_params)
        call_price = bs.call_price()

        income = call_price * stock_shares * 100

        # Max profit = (strike - current price) * shares + premium
        max_profit = (call_strike - stock_price) * stock_shares + income

        # Breakeven = current price - premium per share
        breakeven = stock_price - call_price

        return {
            'strategy': 'Covered Call',
            'call_strike': call_strike,
            'call_price': call_price,
            'income_received': income,
            'max_profit': max_profit,
            'breakeven': breakeven,
            'max_loss': (stock_price * stock_shares) - income,
            'greeks': bs.greeks('call')
        }

    @staticmethod
    def collar(stock_price: float, stock_shares: int,
               put_strike: float, call_strike: float,
               params: OptionParameters) -> Dict:
        """
        Collar: Long stock + Long put + Short call.
        Zero-cost or low-cost hedge with defined risk/reward.

        Args:
            stock_price: Current stock price
            stock_shares: Number of shares owned
            put_strike: Protective put strike
            call_strike: Covered call strike
            params: Option parameters

        Returns:
            Strategy details
        """
        # Buy put
        protective_put = OptionStrategy.protective_put(
            stock_price, stock_shares, put_strike, params
        )

        # Sell call
        covered_call = OptionStrategy.covered_call(
            stock_price, stock_shares, call_strike, params
        )

        # Net cost = put cost - call premium
        net_cost = protective_put['total_cost'] - covered_call['income_received']

        # Max loss = (current - put strike) * shares + net cost
        max_loss = (stock_price - put_strike) * stock_shares + net_cost

        # Max profit = (call strike - current) * shares - net cost
        max_profit = (call_strike - stock_price) * stock_shares - net_cost

        return {
            'strategy': 'Collar',
            'put_strike': put_strike,
            'call_strike': call_strike,
            'net_cost': net_cost,
            'is_zero_cost': abs(net_cost) < 0.01,
            'max_loss': max_loss,
            'max_profit': max_profit,
            'breakeven': stock_price + (net_cost / stock_shares)
        }


class MLOptionsSelector:
    """
    Use ML forecasts to select optimal option strikes and strategies.
    """

    def __init__(self, ml_forecast_model=None):
        """
        Initialize ML options selector.

        Args:
            ml_forecast_model: Your TFT, LSTM, or Hybrid model (optional)
        """
        self.model = ml_forecast_model

    def select_protective_put_strike(self, current_price: float,
                                     forecast_quantiles: Dict[float, float],
                                     max_loss_tolerance: float = 0.10) -> float:
        """
        Select protective put strike based on ML forecast distribution.

        Args:
            current_price: Current stock price
            forecast_quantiles: Dict of {quantile: price} from ML model
                               e.g., {0.05: 95, 0.50: 105, 0.95: 115}
            max_loss_tolerance: Maximum acceptable loss (e.g., 0.10 = 10%)

        Returns:
            Recommended put strike price
        """
        # Use 5th percentile (95% confidence floor)
        downside_forecast = forecast_quantiles.get(0.05, current_price * 0.90)

        # Strike should be at or above worst-case scenario
        # but not so high that premium is excessive
        max_loss_price = current_price * (1 - max_loss_tolerance)

        # Choose the higher of: ML 5th percentile or max loss tolerance
        recommended_strike = max(downside_forecast, max_loss_price)

        # Round to nearest standard strike
        strike_increment = 2.50 if current_price < 100 else 5.00
        recommended_strike = round(recommended_strike / strike_increment) * strike_increment

        return recommended_strike

    def select_covered_call_strike(self, current_price: float,
                                   forecast_quantiles: Dict[float, float],
                                   target_income: float = 0.02) -> float:
        """
        Select covered call strike based on ML upside forecast.

        Args:
            current_price: Current stock price
            forecast_quantiles: ML forecast distribution
            target_income: Target premium income (e.g., 0.02 = 2%)

        Returns:
            Recommended call strike price
        """
        # Use 95th percentile (upside potential)
        upside_forecast = forecast_quantiles.get(0.95, current_price * 1.10)

        # Median forecast (50th percentile)
        median_forecast = forecast_quantiles.get(0.50, current_price * 1.02)

        # Strike should be above median but below 95th percentile
        # This captures income while allowing reasonable upside
        recommended_strike = (median_forecast + upside_forecast) / 2

        # Round to standard strike
        strike_increment = 2.50 if current_price < 100 else 5.00
        recommended_strike = round(recommended_strike / strike_increment) * strike_increment

        return recommended_strike

    def strategy_recommendation(self, current_price: float,
                               forecast_quantiles: Dict[float, float],
                               volatility: float,
                               market_sentiment: str = 'neutral') -> Dict:
        """
        Recommend option strategy based on ML forecasts and market conditions.

        Args:
            current_price: Current stock price
            forecast_quantiles: ML forecast distribution
            volatility: Current/forecasted volatility
            market_sentiment: 'bullish', 'bearish', or 'neutral'

        Returns:
            Recommended strategy details
        """
        median = forecast_quantiles.get(0.50, current_price)
        p05 = forecast_quantiles.get(0.05, current_price * 0.90)
        p95 = forecast_quantiles.get(0.95, current_price * 1.10)

        # Calculate expected move
        expected_upside = (p95 - current_price) / current_price
        expected_downside = (current_price - p05) / current_price

        # Decision logic
        if market_sentiment == 'bullish' and expected_upside > 0.05:
            strategy = 'long_call'
            details = {
                'strategy': 'Long Call',
                'rationale': f'ML forecasts {expected_upside:.1%} upside potential',
                'recommended_strike': median,
                'confidence': 'high' if expected_upside > 0.10 else 'medium'
            }

        elif market_sentiment == 'bearish' and expected_downside > 0.05:
            strategy = 'protective_put'
            details = {
                'strategy': 'Protective Put',
                'rationale': f'ML forecasts {expected_downside:.1%} downside risk',
                'recommended_strike': p05,
                'confidence': 'high' if expected_downside > 0.10 else 'medium'
            }

        elif volatility > 0.30:  # High volatility
            strategy = 'collar'
            details = {
                'strategy': 'Collar',
                'rationale': f'High volatility ({volatility:.1%}) suggests hedging',
                'put_strike': p05,
                'call_strike': p95,
                'confidence': 'medium'
            }

        else:  # Neutral/moderate conditions
            strategy = 'covered_call'
            details = {
                'strategy': 'Covered Call',
                'rationale': 'Generate income in neutral market',
                'recommended_strike': median * 1.05,
                'confidence': 'medium'
            }

        return details
