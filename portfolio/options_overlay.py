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


def calculate_portfolio_beta(returns: pd.DataFrame, weights: pd.Series,
                              market_ticker: str = 'SPY') -> Dict:
    """
    Calculate portfolio beta relative to market index.

    Beta measures systematic risk - how much the portfolio moves with the market.
    Used to size index hedges (SPY puts) to cover portfolio-wide exposure.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for portfolio constituents (columns = tickers)
    weights : pd.Series
        Portfolio weights (must sum to 1.0)
    market_ticker : str
        Market index ticker (default: SPY)

    Returns
    -------
    dict
        {'portfolio_beta': float, 'individual_betas': dict, 'r_squared': float}
    """
    # INPUT VALIDATION (following financial-knowledge-validator skill)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame")
    if not isinstance(weights, pd.Series):
        raise TypeError("Weights must be a pandas Series")
    if len(returns) < 20:
        raise ValueError(f"Need at least 20 return observations, got {len(returns)}")

    # Validate weights sum to ~1.0
    weights_sum = weights.sum()
    if not np.isclose(weights_sum, 1.0, atol=0.01):
        raise ValueError(f"Weights sum to {weights_sum:.4f}, must be ~1.0")

    # Fetch market returns if not in DataFrame
    if market_ticker not in returns.columns:
        try:
            market = yf.Ticker(market_ticker)
            start_date = returns.index[0].strftime('%Y-%m-%d')
            end_date = returns.index[-1].strftime('%Y-%m-%d')
            market_data = market.history(start=start_date, end=end_date)['Close']
            market_returns = market_data.pct_change().dropna()

            # Align dates
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) < 20:
                raise ValueError(f"Insufficient overlapping dates: {len(common_dates)}")

            returns = returns.loc[common_dates]
            market_returns = market_returns.loc[common_dates]
        except Exception as e:
            print(f"  Warning: Could not fetch {market_ticker} data: {e}")
            # Fallback to beta = 1.0
            return {
                'portfolio_beta': 1.0,
                'individual_betas': {tk: 1.0 for tk in weights.index},
                'r_squared': 0.0,
                'fallback': True,
            }
    else:
        market_returns = returns[market_ticker]
        returns = returns.drop(columns=[market_ticker], errors='ignore')

    # Calculate individual betas via regression
    individual_betas = {}
    for ticker in weights.index:
        if ticker in returns.columns and ticker != market_ticker:
            stock_returns = returns[ticker].dropna()
            common_idx = stock_returns.index.intersection(market_returns.index)

            if len(common_idx) >= 20:
                x = market_returns.loc[common_idx].values
                y = stock_returns.loc[common_idx].values

                # Beta = Cov(stock, market) / Var(market)
                covariance = np.cov(y, x)[0, 1]
                market_variance = np.var(x, ddof=1)

                if market_variance > 1e-10:
                    beta = covariance / market_variance
                    # VALIDATION: Beta typically in [-1, 3] for equities
                    beta = np.clip(beta, -1.0, 3.0)
                    individual_betas[ticker] = float(beta)
                else:
                    individual_betas[ticker] = 1.0
            else:
                individual_betas[ticker] = 1.0
        elif ticker == market_ticker:
            individual_betas[ticker] = 1.0
        else:
            individual_betas[ticker] = 1.0

    # Portfolio beta = weighted sum of individual betas
    portfolio_beta = 0.0
    for ticker, weight in weights.items():
        beta = individual_betas.get(ticker, 1.0)
        portfolio_beta += weight * beta

    # Calculate R-squared for portfolio vs market
    # Create portfolio returns
    aligned_weights = weights.reindex(returns.columns, fill_value=0)
    portfolio_returns = (returns * aligned_weights).sum(axis=1)

    common_idx = portfolio_returns.index.intersection(market_returns.index)
    if len(common_idx) >= 20:
        corr = np.corrcoef(
            portfolio_returns.loc[common_idx].values,
            market_returns.loc[common_idx].values
        )[0, 1]
        r_squared = corr ** 2 if not np.isnan(corr) else 0.0
    else:
        r_squared = 0.0

    # OUTPUT VALIDATION
    if not (0.0 <= portfolio_beta <= 3.0):
        print(f"  Warning: Portfolio beta {portfolio_beta:.2f} outside typical range [0, 3]")
        portfolio_beta = np.clip(portfolio_beta, 0.0, 3.0)

    return {
        'portfolio_beta': float(portfolio_beta),
        'individual_betas': individual_betas,
        'r_squared': float(r_squared),
        'fallback': False,
    }


def select_index_hedge_options(portfolio_value: float, hedge_budget: float,
                                portfolio_beta: float, weights: pd.Series,
                                index_ticker: str = 'SPY',
                                strategy: str = 'put',
                                target_dte: int = 90) -> Optional[Dict]:
    """
    Select index options (SPY puts) to hedge ENTIRE portfolio's systematic risk.

    This fixes the critical bug where only the primary ticker was hedged.
    Uses portfolio beta to size the hedge appropriately for the whole portfolio.

    Parameters
    ----------
    portfolio_value : float
        Total portfolio value to hedge
    hedge_budget : float
        Dollar amount allocated to hedge
    portfolio_beta : float
        Portfolio's beta to the index (from calculate_portfolio_beta)
    weights : pd.Series
        Portfolio weights by ticker
    index_ticker : str
        Index to hedge with (default: SPY)
    strategy : str
        'put' or 'collar' (default: 'put')
    target_dte : int
        Target days to expiration (default: 90)

    Returns
    -------
    dict or None
        Index hedge overlay details or None if unavailable
    """
    # INPUT VALIDATION
    if portfolio_value <= 0:
        raise ValueError(f"Portfolio value must be positive, got {portfolio_value}")
    if hedge_budget <= 0:
        raise ValueError(f"Hedge budget must be positive, got {hedge_budget}")
    if not (0.0 <= portfolio_beta <= 3.0):
        raise ValueError(f"Portfolio beta {portfolio_beta} outside valid range [0, 3]")

    print(f"\n[Index Hedge] Selecting {strategy.upper()} hedge for entire portfolio via {index_ticker}...")
    print(f"  Portfolio value: ${portfolio_value:,.2f}")
    print(f"  Portfolio beta: {portfolio_beta:.2f}")
    print(f"  Hedge budget: ${hedge_budget:,.2f}")
    print(f"  Positions hedged: {len(weights)} tickers")

    # Beta-adjusted exposure = how much SPY we need to hedge
    beta_adjusted_exposure = portfolio_value * portfolio_beta
    print(f"  Beta-adjusted exposure: ${beta_adjusted_exposure:,.2f}")

    # Fetch current SPY price
    try:
        spy = yf.Ticker(index_ticker)
        info = spy.get_info() if hasattr(spy, 'get_info') else spy.info
        index_price = info.get('currentPrice') or info.get('regularMarketPrice')

        if not index_price or index_price <= 0:
            hist = spy.history(period='5d')
            if not hist.empty:
                index_price = float(hist['Close'].iloc[-1])
            else:
                print(f"  Warning: Could not fetch {index_ticker} price")
                return None
    except Exception as e:
        print(f"  Warning: Failed to get {index_ticker} price: {e}")
        return None

    print(f"  {index_ticker} price: ${index_price:.2f}")

    # Fetch SPY options chain
    puts = fetch_options_chain(index_ticker, target_dte=target_dte)

    if puts is None or puts.empty:
        print(f"  Warning: No options chain available for {index_ticker}")
        return None

    # Filter for near-ATM puts (92-100% moneyness for protective puts)
    puts['moneyness'] = puts['strike'] / index_price
    atm_puts = puts[(puts['moneyness'] >= 0.92) & (puts['moneyness'] <= 1.00)].copy()

    if atm_puts.empty:
        print(f"  Warning: No near-ATM puts found for {index_ticker}")
        return None

    # Calculate mid price
    atm_puts['mid_price'] = (atm_puts['bid'] + atm_puts['ask']) / 2
    atm_puts = atm_puts[atm_puts['mid_price'] > 0]

    if atm_puts.empty:
        return None

    # Calculate IV for selected options
    r = 0.04  # Risk-free rate
    T_years = atm_puts['dte'].iloc[0] / 365.0

    iv_list = []
    for idx, row in atm_puts.iterrows():
        iv = implied_volatility_from_price(row['mid_price'], index_price,
                                            row['strike'], T_years, r, 'put')
        iv_list.append(iv)
    atm_puts['implied_vol'] = iv_list

    # Select put at ~95% moneyness (5% OTM)
    target_moneyness = 0.95
    atm_puts['moneyness_diff'] = abs(atm_puts['moneyness'] - target_moneyness)
    selected_put = atm_puts.loc[atm_puts['moneyness_diff'].idxmin()]

    # Calculate greeks
    strike = selected_put['strike']
    iv = selected_put['implied_vol'] if pd.notna(selected_put['implied_vol']) else 0.18  # SPY IV typically lower

    greeks = calculate_greeks_scipy(index_price, strike, T_years, r,
                                     iv if iv and iv > 0 else 0.18, 'put')

    # SIZE HEDGE: Need to hedge beta-adjusted exposure
    option_price = float(selected_put['mid_price'])
    shares_per_contract = 100

    # Shares of SPY needed to hedge beta-adjusted exposure
    spy_shares_to_hedge = beta_adjusted_exposure / index_price
    contracts_for_full_hedge = int(np.ceil(spy_shares_to_hedge / shares_per_contract))

    # Budget-constrained contracts
    max_contracts_by_budget = int(hedge_budget / (option_price * shares_per_contract))

    # Use lesser of budget-constrained or full hedge
    num_contracts = min(max_contracts_by_budget, contracts_for_full_hedge)

    if num_contracts <= 0:
        print(f"  Warning: Insufficient budget for even 1 contract")
        return None

    total_premium = num_contracts * option_price * shares_per_contract
    spy_shares_hedged = num_contracts * shares_per_contract
    spy_notional_hedged = spy_shares_hedged * index_price

    # TRUE PORTFOLIO HEDGE COVERAGE
    # Coverage = (SPY notional hedged) / (Beta-adjusted exposure)
    # This represents what % of systematic risk is covered
    hedge_coverage = spy_notional_hedged / beta_adjusted_exposure if beta_adjusted_exposure > 0 else 0.0

    # Also calculate raw portfolio coverage (vs total portfolio value)
    raw_coverage = spy_notional_hedged / portfolio_value if portfolio_value > 0 else 0.0

    # VALIDATION: Coverage should be in [0, 1] for partial hedge, can exceed 1 if over-hedged
    if hedge_coverage > 1.5:
        print(f"  Warning: Over-hedged at {hedge_coverage*100:.1f}% coverage")

    print(f"  ✓ Selected {num_contracts} {index_ticker} PUT contract(s)")
    print(f"    Strike: ${strike:.2f} ({selected_put['moneyness']*100:.1f}% moneyness)")
    print(f"    DTE: {int(selected_put['dte'])} days")
    print(f"    Premium: ${option_price:.2f}/share (${total_premium:,.2f} total)")
    print(f"    Beta-adjusted coverage: {hedge_coverage*100:.1f}%")
    print(f"    Raw portfolio coverage: {raw_coverage*100:.1f}%")
    print(f"    Delta: {greeks['delta']:.3f}")
    if iv:
        print(f"    IV: {iv*100:.1f}%")

    # Create position breakdown for reporting
    position_coverage = {}
    for ticker, weight in weights.items():
        position_value = portfolio_value * weight
        # Each position's share of hedge coverage
        position_coverage[ticker] = {
            'value': position_value,
            'weight': weight,
            'hedged_value': position_value * hedge_coverage,
            'hedge_pct': hedge_coverage * 100,
        }

    return {
        'strategy': f'Portfolio Index Hedge ({index_ticker})',
        'hedge_type': 'index',
        'index_ticker': index_ticker,
        'option_type': 'put',
        'strike': float(strike),
        'expiration': selected_put['expiration'],
        'dte': int(selected_put['dte']),
        'contracts': num_contracts,
        'option_price': option_price,
        'total_premium': total_premium,
        # KEY FIX: Report TRUE portfolio-wide coverage
        'hedge_coverage': hedge_coverage,  # Beta-adjusted coverage
        'raw_coverage': raw_coverage,  # Simple notional coverage
        'beta_adjusted_exposure': beta_adjusted_exposure,
        'spy_notional_hedged': spy_notional_hedged,
        'portfolio_beta': portfolio_beta,
        'positions_covered': len(weights),
        'position_coverage': position_coverage,
        'implied_vol': float(iv) if iv else None,
        'greeks': greeks,
        'moneyness': float(selected_put['moneyness']),
        'bid': float(selected_put['bid']),
        'ask': float(selected_put['ask']),
        'volume': int(selected_put['volume']) if pd.notna(selected_put['volume']) else 0,
        'open_interest': int(selected_put['openInterest']) if pd.notna(selected_put['openInterest']) else 0,
        # Warning flags
        'is_budget_constrained': max_contracts_by_budget < contracts_for_full_hedge,
        'full_hedge_contracts': contracts_for_full_hedge,
    }


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
