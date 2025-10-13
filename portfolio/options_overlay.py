"""
Options Overlay Module
======================

Implements options-based hedging strategies:
1. Fetch options chains via yfinance
2. Select near-ATM puts or collar structures
3. Calculate implied volatility from bid-ask midpoints
4. Compute greeks (delta, gamma, vega, theta)
5. Size overlay based on hedge budget and portfolio exposure
"""

from typing import List, Dict, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Try to import py_vollib for greeks/IV, fallback to scipy approximations
try:
    from py_vollib.black_scholes import black_scholes as bs_price
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta
    from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv
    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False
    print("Warning: py_vollib not available. Using scipy approximations for greeks/IV.")

from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'put') -> float:
    """
    Calculate Black-Scholes option price.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (annualized)
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        return max(K - S, 0) if option_type == 'put' else max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility_from_price(option_price: float, S: float, K: float, T: float,
                                   r: float, option_type: str = 'put') -> Optional[float]:
    """
    Calculate implied volatility from option price using Brent's method.

    Returns
    -------
    float or None
        Implied volatility or None if calculation fails
    """
    if T <= 0 or option_price <= 0:
        return None

    try:
        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type) - option_price

        iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
        return iv if 0.01 < iv < 5.0 else None
    except Exception:
        return None


def calculate_greeks_scipy(S: float, K: float, T: float, r: float, sigma: float,
                           option_type: str = 'put') -> Dict:
    """
    Calculate option greeks using scipy (fallback when py_vollib unavailable).

    Returns
    -------
    dict
        {'delta', 'gamma', 'vega', 'theta'}
    """
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'put':
        delta_val = -norm.cdf(-d1)
        theta_val = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        delta_val = norm.cdf(d1)
        theta_val = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2))

    gamma_val = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega_val = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in vol

    return {
        'delta': float(delta_val),
        'gamma': float(gamma_val),
        'vega': float(vega_val),
        'theta': float(theta_val) / 365  # Per day
    }


def fetch_options_chain(ticker: str, target_dte: int = 90,
                        dte_range: Tuple[int, int] = (60, 120)) -> Optional[pd.DataFrame]:
    """
    Fetch options chain for a ticker targeting specific days to expiration.

    Parameters
    ----------
    ticker : str
        Ticker symbol
    target_dte : int
        Target days to expiration (default: 90)
    dte_range : tuple
        Acceptable DTE range (default: 60-120 days)

    Returns
    -------
    pd.DataFrame or None
        Options chain with puts, or None if unavailable
    """
    try:
        t = yf.Ticker(ticker)
        expiration_dates = t.options

        if not expiration_dates:
            print(f"  Warning: No options chain available for {ticker}")
            return None

        # Find expiration closest to target DTE
        today = datetime.now().date()
        best_expiry = None
        min_diff = float('inf')

        for exp_str in expiration_dates:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            dte = (exp_date - today).days

            if dte_range[0] <= dte <= dte_range[1]:
                diff = abs(dte - target_dte)
                if diff < min_diff:
                    min_diff = diff
                    best_expiry = exp_str

        if not best_expiry:
            print(f"  Warning: No options expiration in range {dte_range} for {ticker}")
            return None

        # Fetch options chain for selected expiration
        opt_chain = t.option_chain(best_expiry)
        puts = opt_chain.puts.copy()

        # Add expiration info
        exp_date = datetime.strptime(best_expiry, '%Y-%m-%d').date()
        puts['expiration'] = best_expiry
        puts['dte'] = (exp_date - today).days

        return puts

    except Exception as e:
        print(f"  Warning: Failed to fetch options for {ticker}: {e}")
        return None


def select_hedge_options(ticker: str, current_price: float, portfolio_value: float,
                        hedge_budget: float, strategy: str = 'put',
                        target_dte: int = 90) -> Optional[Dict]:
    """
    Select appropriate options for hedging based on strategy and budget.

    Parameters
    ----------
    ticker : str
        Ticker symbol
    current_price : float
        Current stock price
    portfolio_value : float
        Total portfolio value to hedge
    hedge_budget : float
        Dollar amount allocated to hedge
    strategy : str
        'put' or 'collar' (default: 'put')
    target_dte : int
        Target days to expiration (default: 90)

    Returns
    -------
    dict or None
        Hedge overlay details or None if unavailable
    """
    print(f"\n[Options Overlay] Selecting {strategy} hedge for {ticker}...")
    print(f"  Current price: ${current_price:.2f}")
    print(f"  Portfolio value: ${portfolio_value:,.2f}")
    print(f"  Hedge budget: ${hedge_budget:,.2f}")

    # Fetch options chain
    puts = fetch_options_chain(ticker, target_dte=target_dte)

    if puts is None or puts.empty:
        return None

    # Filter for near-ATM puts (90-100% moneyness)
    puts['moneyness'] = puts['strike'] / current_price
    atm_puts = puts[(puts['moneyness'] >= 0.90) & (puts['moneyness'] <= 1.00)].copy()

    if atm_puts.empty:
        print(f"  Warning: No near-ATM puts found for {ticker}")
        return None

    # Calculate mid price and IV for each option
    atm_puts['mid_price'] = (atm_puts['bid'] + atm_puts['ask']) / 2
    atm_puts = atm_puts[atm_puts['mid_price'] > 0]  # Filter invalid quotes

    if atm_puts.empty:
        return None

    # Calculate implied volatility
    r = 0.04  # Risk-free rate assumption
    T_years = atm_puts['dte'].iloc[0] / 365.0

    iv_list = []
    for idx, row in atm_puts.iterrows():
        if HAS_VOLLIB:
            try:
                iv = bs_iv(row['mid_price'], current_price, row['strike'],
                          T_years, r, 'p')
                iv_list.append(iv)
            except Exception:
                iv = implied_volatility_from_price(row['mid_price'], current_price,
                                                   row['strike'], T_years, r, 'put')
                iv_list.append(iv)
        else:
            iv = implied_volatility_from_price(row['mid_price'], current_price,
                                               row['strike'], T_years, r, 'put')
            iv_list.append(iv)

    atm_puts['implied_vol'] = iv_list

    # Select option closest to 95% moneyness (5% OTM protective put)
    target_moneyness = 0.95
    atm_puts['moneyness_diff'] = abs(atm_puts['moneyness'] - target_moneyness)
    selected_put = atm_puts.loc[atm_puts['moneyness_diff'].idxmin()]

    # Calculate greeks
    strike = selected_put['strike']
    iv = selected_put['implied_vol'] if pd.notna(selected_put['implied_vol']) else 0.30

    if HAS_VOLLIB and iv and iv > 0:
        try:
            greeks = {
                'delta': delta('p', current_price, strike, T_years, r, iv),
                'gamma': gamma('p', current_price, strike, T_years, r, iv),
                'vega': vega('p', current_price, strike, T_years, r, iv),
                'theta': theta('p', current_price, strike, T_years, r, iv),
            }
        except Exception:
            greeks = calculate_greeks_scipy(current_price, strike, T_years, r, iv, 'put')
    else:
        greeks = calculate_greeks_scipy(current_price, strike, T_years, r,
                                        iv if iv and iv > 0 else 0.30, 'put')

    # Calculate number of contracts based on budget
    option_price = float(selected_put['mid_price'])
    shares_per_contract = 100
    max_contracts = int(hedge_budget / (option_price * shares_per_contract))

    # Also calculate contracts needed for full portfolio hedge
    shares_to_hedge = portfolio_value / current_price
    contracts_for_full_hedge = int(np.ceil(shares_to_hedge / shares_per_contract))

    # Use the lesser of budget-constrained or full-hedge
    num_contracts = min(max_contracts, contracts_for_full_hedge)

    if num_contracts <= 0:
        print(f"  Warning: Insufficient budget for even 1 contract")
        return None

    total_premium = num_contracts * option_price * shares_per_contract
    shares_hedged = num_contracts * shares_per_contract
    hedge_coverage = (shares_hedged * current_price) / portfolio_value

    print(f"  ✓ Selected {num_contracts} {strategy.upper()} contract(s)")
    print(f"    Strike: ${strike:.2f} ({selected_put['moneyness']*100:.1f}% moneyness)")
    print(f"    DTE: {int(selected_put['dte'])} days")
    print(f"    Premium: ${option_price:.2f} per share (${total_premium:,.2f} total)")
    print(f"    Coverage: {hedge_coverage*100:.1f}% of portfolio")
    print(f"    IV: {iv*100:.1f}%" if iv else "    IV: N/A")
    print(f"    Delta: {greeks['delta']:.3f}")

    return {
        'ticker': ticker,
        'strategy': strategy,
        'option_type': 'put',
        'strike': float(strike),
        'expiration': selected_put['expiration'],
        'dte': int(selected_put['dte']),
        'contracts': num_contracts,
        'shares_hedged': shares_hedged,
        'option_price': option_price,
        'total_premium': total_premium,
        'hedge_coverage': hedge_coverage,
        'implied_vol': float(iv) if iv else None,
        'greeks': greeks,
        'moneyness': float(selected_put['moneyness']),
        'bid': float(selected_put['bid']),
        'ask': float(selected_put['ask']),
        'volume': int(selected_put['volume']) if pd.notna(selected_put['volume']) else 0,
        'open_interest': int(selected_put['openInterest']) if pd.notna(selected_put['openInterest']) else 0,
    }


def calculate_hedge_budget(rrr: float, portfolio_value: float) -> float:
    """
    Calculate hedge budget based on reward-to-risk ratio.

    Higher RRR → more risk-averse → larger hedge budget
    Lower RRR → more aggressive → smaller hedge budget

    Parameters
    ----------
    rrr : float
        Reward-to-risk ratio (0 to 1)
    portfolio_value : float
        Total portfolio value

    Returns
    -------
    float
        Hedge budget in dollars
    """
    # Map RRR to hedge budget percentage
    # RRR 0.0 (aggressive) → 2% hedge
    # RRR 0.5 (moderate) → 10% hedge
    # RRR 1.0 (defensive) → 20% hedge

    hedge_pct = 0.02 + (rrr * 0.18)  # Linear interpolation 2% to 20%

    hedge_budget = portfolio_value * hedge_pct

    return hedge_budget


if __name__ == '__main__':
    # Test options overlay
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'

    # Fetch current price
    t = yf.Ticker(ticker)
    info = t.get_info() if hasattr(t, 'get_info') else t.info
    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 100

    # Test parameters
    portfolio_value = 100000
    rrr = 0.6
    hedge_budget = calculate_hedge_budget(rrr, portfolio_value)

    result = select_hedge_options(ticker, current_price, portfolio_value,
                                  hedge_budget, strategy='put')

    if result:
        print(f"\n{'='*70}")
        print(f"OPTIONS OVERLAY RESULTS: {ticker}")
        print(f"{'='*70}")
        print(f"Strategy: {result['strategy'].upper()}")
        print(f"Strike: ${result['strike']:.2f} ({result['moneyness']*100:.1f}% moneyness)")
        print(f"Expiration: {result['expiration']} ({result['dte']} DTE)")
        print(f"Contracts: {result['contracts']}")
        print(f"Total Premium: ${result['total_premium']:,.2f}")
        print(f"Coverage: {result['hedge_coverage']*100:.1f}% of portfolio")
        print(f"\nGreeks:")
        for k, v in result['greeks'].items():
            print(f"  {k.capitalize()}: {v:.4f}")
        print(f"\nImplied Volatility: {result['implied_vol']*100:.1f}%" if result['implied_vol'] else "\nImplied Volatility: N/A")
    else:
        print(f"\n✗ No hedge overlay available for {ticker}")
