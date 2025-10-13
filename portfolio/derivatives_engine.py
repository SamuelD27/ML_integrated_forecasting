import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm


class DerivativesEngine:
    """Advanced derivatives analysis and hedging engine."""

    def __init__(self, ticker: str, current_price: float,
                 portfolio_value: float, risk_free_rate: float = 0.05):
        """
        Initialize derivatives engine.

        Parameters
        ----------
        ticker : str
            Underlying ticker symbol
        current_price : float
            Current price of underlying
        portfolio_value : float
            Total portfolio value to hedge
        risk_free_rate : float
            Risk-free rate for pricing (default: 5%)
        """
        self.ticker = ticker
        self.current_price = current_price
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate

    def analyze_protective_put_spread(self, hedge_budget: float,
                                      dte: int = 90) -> Optional[Dict]:
        """
        Analyze protective put spread strategy (buy put, sell lower put).

        Parameters
        ----------
        hedge_budget : float
            Available budget for hedge
        dte : int
            Days to expiration

        Returns
        -------
        dict or None
            Put spread analysis
        """
        try:
            stock = yf.Ticker(self.ticker)
            expiry_date = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')

            # Get options chain
            opts = stock.option_chain(stock.options[0])  # Nearest expiry
            puts = opts.puts

            if puts.empty:
                return None

            # Find ATM strike
            atm_strike = puts.iloc[(puts['strike'] - self.current_price).abs().argsort()[0]]['strike']

            # Find 5-10% OTM put to buy (protection)
            buy_strike_target = self.current_price * 0.93  # 7% OTM
            buy_put = puts.iloc[(puts['strike'] - buy_strike_target).abs().argsort()[0]]

            # Find 15-20% OTM put to sell (reduce cost)
            sell_strike_target = self.current_price * 0.85  # 15% OTM
            sell_put = puts.iloc[(puts['strike'] - sell_strike_target).abs().argsort()[0]]

            # Calculate spread cost and contracts
            spread_cost = buy_put['lastPrice'] - sell_put['lastPrice']
            max_contracts = int(hedge_budget / (spread_cost * 100))

            if max_contracts <= 0:
                return None

            actual_cost = max_contracts * spread_cost * 100
            max_protection = max_contracts * (buy_put['strike'] - sell_put['strike']) * 100

            return {
                'strategy': 'Protective Put Spread',
                'buy_put_strike': buy_put['strike'],
                'buy_put_price': buy_put['lastPrice'],
                'sell_put_strike': sell_put['strike'],
                'sell_put_price': sell_put['lastPrice'],
                'spread_cost': spread_cost,
                'contracts': max_contracts,
                'total_cost': actual_cost,
                'max_protection': max_protection,
                'protection_level': buy_put['strike'],
                'floor_level': sell_put['strike'],
                'expiration': buy_put.get('expiration', 'N/A'),
                'efficiency': max_protection / actual_cost if actual_cost > 0 else 0,
            }

        except Exception as e:
            print(f"  Warning: Put spread analysis failed: {e}")
            return None

    def analyze_collar_strategy(self, hedge_budget: float,
                                dte: int = 90) -> Optional[Dict]:
        """
        Analyze collar strategy (buy put + sell call).

        Parameters
        ----------
        hedge_budget : float
            Available budget for hedge
        dte : int
            Days to expiration

        Returns
        -------
        dict or None
            Collar strategy analysis
        """
        try:
            stock = yf.Ticker(self.ticker)
            opts = stock.option_chain(stock.options[0])
            puts = opts.puts
            calls = opts.calls

            if puts.empty or calls.empty:
                return None

            # Buy 5% OTM put for protection
            put_strike_target = self.current_price * 0.95
            buy_put = puts.iloc[(puts['strike'] - put_strike_target).abs().argsort()[0]]

            # Sell 10% OTM call to finance
            call_strike_target = self.current_price * 1.10
            sell_call = calls.iloc[(calls['strike'] - call_strike_target).abs().argsort()[0]]

            # Net cost (can be negative = net credit)
            net_cost = buy_put['lastPrice'] - sell_call['lastPrice']
            shares_to_hedge = int(self.portfolio_value / self.current_price)
            contracts = min(shares_to_hedge // 100, int(hedge_budget / (net_cost * 100)) if net_cost > 0 else shares_to_hedge // 100)

            if contracts <= 0:
                return None

            total_cost = contracts * net_cost * 100

            return {
                'strategy': 'Collar (Put + Short Call)',
                'buy_put_strike': buy_put['strike'],
                'buy_put_price': buy_put['lastPrice'],
                'sell_call_strike': sell_call['strike'],
                'sell_call_price': sell_call['lastPrice'],
                'net_cost': net_cost,
                'contracts': contracts,
                'total_cost': total_cost,
                'is_zero_cost': abs(net_cost) < 0.50,
                'downside_protection': buy_put['strike'],
                'upside_cap': sell_call['strike'],
                'max_gain': (sell_call['strike'] - self.current_price) * contracts * 100,
                'max_loss': (self.current_price - buy_put['strike']) * contracts * 100,
            }

        except Exception as e:
            print(f"  Warning: Collar analysis failed: {e}")
            return None

    def synthetic_futures_position(self) -> Dict:
        """
        Create synthetic long futures position using options (buy call + sell put).

        Returns
        -------
        dict
            Synthetic futures analysis
        """
        # Synthetic futures: Long Call + Short Put at same strike = Long Futures
        # This is capital efficient vs buying stock
        try:
            stock = yf.Ticker(self.ticker)
            opts = stock.option_chain(stock.options[0])

            # ATM strike
            atm_strike = self.current_price

            calls = opts.calls
            puts = opts.puts

            # Find ATM options
            call = calls.iloc[(calls['strike'] - atm_strike).abs().argsort()[0]]
            put = puts.iloc[(puts['strike'] - atm_strike).abs().argsort()[0]]

            # Synthetic futures cost = call premium - put premium (often near zero)
            synthetic_cost = call['lastPrice'] - put['lastPrice']

            # Margin requirement (typically 20% of notional)
            notional_per_contract = atm_strike * 100
            margin_per_contract = notional_per_contract * 0.20

            contracts_possible = int(self.portfolio_value / margin_per_contract)

            return {
                'strategy': 'Synthetic Long Futures',
                'strike': atm_strike,
                'call_price': call['lastPrice'],
                'put_price': put['lastPrice'],
                'net_cost': synthetic_cost,
                'contracts': contracts_possible,
                'total_exposure': contracts_possible * notional_per_contract,
                'margin_required': contracts_possible * margin_per_contract,
                'leverage': (contracts_possible * notional_per_contract) / self.portfolio_value,
                'description': 'Synthetic futures using options - capital efficient long exposure',
            }

        except Exception as e:
            print(f"  Warning: Synthetic futures analysis failed: {e}")
            return {'strategy': 'Synthetic Long Futures', 'error': str(e)}

    def variance_swap_analysis(self, historical_vol: float,
                               implied_vol: Optional[float] = None) -> Dict:
        """
        Analyze variance swap for volatility trading.

        Parameters
        ----------
        historical_vol : float
            Historical volatility (annualized)
        implied_vol : float, optional
            Implied volatility from options

        Returns
        -------
        dict
            Variance swap analysis
        """
        # Variance swap: bet on realized variance vs implied variance
        # Payoff = Notional * (Realized Variance - Strike Variance)

        strike_var = (historical_vol ** 2) * 100  # Strike in variance points
        notional = 1000000  # $1M notional (standard)

        # Estimate P&L under different scenarios
        scenarios = {
            'Low Vol (50% of current)': historical_vol * 0.5,
            'Current Vol': historical_vol,
            'High Vol (150% of current)': historical_vol * 1.5,
            'Extreme Vol (200% of current)': historical_vol * 2.0,
        }

        scenario_payoffs = {}
        for scenario, realized_vol in scenarios.items():
            realized_var = (realized_vol ** 2) * 100
            payoff = notional * (realized_var - strike_var) / 10000  # Convert variance points
            scenario_payoffs[scenario] = payoff

        return {
            'strategy': 'Variance Swap',
            'notional': notional,
            'strike_variance': strike_var,
            'strike_volatility': historical_vol * 100,
            'implied_volatility': implied_vol * 100 if implied_vol else None,
            'vol_risk_premium': ((implied_vol - historical_vol) * 100) if implied_vol else None,
            'scenario_payoffs': scenario_payoffs,
            'description': 'Bet on realized volatility vs implied volatility',
            'risk': 'Unlimited loss if volatility spikes',
        }

    def dividend_swap_strategy(self, expected_dividend_yield: float) -> Dict:
        """
        Analyze dividend swap for dividend capture.

        Parameters
        ----------
        expected_dividend_yield : float
            Expected annual dividend yield

        Returns
        -------
        dict
            Dividend swap analysis
        """
        # Dividend swap: exchange fixed rate for actual dividends
        # Useful for dividend capture without owning stock

        notional = self.portfolio_value
        fixed_rate = expected_dividend_yield * 0.95  # Take 95% of expected

        # Estimate payoff
        actual_div_scenarios = {
            'Dividend Cut (50%)': expected_dividend_yield * 0.5,
            'Expected': expected_dividend_yield,
            'Dividend Growth (125%)': expected_dividend_yield * 1.25,
        }

        payoffs = {}
        for scenario, actual_yield in actual_div_scenarios.items():
            payoff = notional * (actual_yield - fixed_rate)
            payoffs[scenario] = payoff

        return {
            'strategy': 'Dividend Swap',
            'notional': notional,
            'fixed_rate': fixed_rate * 100,
            'expected_dividend_yield': expected_dividend_yield * 100,
            'scenario_payoffs': payoffs,
            'description': 'Receive actual dividends, pay fixed rate',
            'use_case': 'Dividend capture without stock ownership',
        }

    def total_return_swap(self, financing_rate: Optional[float] = None) -> Dict:
        """
        Analyze total return swap for leveraged exposure.

        Parameters
        ----------
        financing_rate : float, optional
            Financing rate (default: risk_free_rate + 2%)

        Returns
        -------
        dict
            Total return swap analysis
        """
        if financing_rate is None:
            financing_rate = self.risk_free_rate + 0.02  # RF + 200bps

        # Total return swap: receive total return, pay financing
        # Capital efficient way to gain exposure

        notional = self.portfolio_value * 3  # 3x leverage possible
        financing_cost = notional * financing_rate

        # Estimate payoff scenarios
        return_scenarios = {
            'Bear (-20%)': -0.20,
            'Flat': 0.00,
            'Modest (+10%)': 0.10,
            'Strong (+25%)': 0.25,
        }

        payoffs = {}
        for scenario, stock_return in return_scenarios.items():
            stock_pnl = notional * stock_return
            financing_pnl = -financing_cost
            total_pnl = stock_pnl + financing_pnl
            payoffs[scenario] = total_pnl

        return {
            'strategy': 'Total Return Swap',
            'notional': notional,
            'leverage': notional / self.portfolio_value,
            'financing_rate': financing_rate * 100,
            'annual_financing_cost': financing_cost,
            'scenario_payoffs': payoffs,
            'description': 'Leveraged exposure via swap - pay financing, receive total return',
            'capital_required': 0,  # Unfunded
            'risk': 'Margin calls if position moves against you',
        }

    def analyze_all_derivatives(self, hedge_budget: float,
                                historical_vol: float,
                                implied_vol: Optional[float] = None,
                                expected_div_yield: float = 0.02) -> Dict:
        """
        Comprehensive analysis of all derivative strategies.

        Parameters
        ----------
        hedge_budget : float
            Budget for hedging
        historical_vol : float
            Historical volatility
        implied_vol : float, optional
            Implied volatility from options
        expected_div_yield : float
            Expected dividend yield (default: 2%)

        Returns
        -------
        dict
            All derivative strategies
        """
        print(f"\n  Analyzing comprehensive derivative strategies...")

        strategies = {}

        # Options strategies
        strategies['put_spread'] = self.analyze_protective_put_spread(hedge_budget)
        strategies['collar'] = self.analyze_collar_strategy(hedge_budget)
        strategies['synthetic_futures'] = self.synthetic_futures_position()

        # Swaps and structured products
        strategies['variance_swap'] = self.variance_swap_analysis(historical_vol, implied_vol)
        strategies['dividend_swap'] = self.dividend_swap_strategy(expected_div_yield)
        strategies['total_return_swap'] = self.total_return_swap()

        return strategies

    def compare_hedging_strategies(self, strategies: Dict) -> pd.DataFrame:
        """
        Compare all hedging strategies side-by-side.

        Parameters
        ----------
        strategies : dict
            All strategies from analyze_all_derivatives()

        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        comparison = []

        for name, strategy in strategies.items():
            if strategy is None or 'error' in strategy:
                continue

            row = {
                'Strategy': strategy.get('strategy', name),
                'Cost/Margin': strategy.get('total_cost', strategy.get('margin_required', strategy.get('annual_financing_cost', 0))),
                'Max Protection': strategy.get('max_protection', strategy.get('downside_protection', 'N/A')),
                'Efficiency': strategy.get('efficiency', 'N/A'),
                'Description': strategy.get('description', '')[:50],
            }

            comparison.append(row)

        return pd.DataFrame(comparison)
