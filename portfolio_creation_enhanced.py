"""
Enhanced Multi-Asset Portfolio Constructor with Advanced Analytics
===================================================================

Complete portfolio construction workflow with:
1. Dynamic peer discovery (same exchange + sector/industry)
2. ETF selection (sector ETF, low-vol ETF, market ETF)
3. Multiple derivative instruments (options, futures, swaps, variance swaps, etc.)
4. Advanced portfolio optimization (shrinkage covariance, CAPM beta, RRR-based)
5. Monte Carlo simulations (10,000+ paths)
6. Comprehensive visualizations and insights
7. Excel reporting with charts and detailed analysis

Usage:
    python portfolio_creation_enhanced.py TICKER --capital 100000 --rrr 0.6
    python portfolio_creation_enhanced.py AAPL --capital 50000 --rrr 0.8 --start 2023-01-01
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO

warnings.filterwarnings('ignore')

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_fetching import fetch_full_bundle, save_bundle, set_last_fetch_globals
from portfolio.peer_discovery import discover_peers, get_ticker_metadata
from portfolio.etf_discovery import discover_etfs
from portfolio.options_overlay import select_hedge_options, calculate_hedge_budget
from portfolio.advanced_optimizer import (
    compute_returns_matrix,
    optimize_mean_variance,
    calculate_portfolio_shares
)
from portfolio.chart_generator import ChartGenerator
from portfolio.monte_carlo import MonteCarloSimulator
from portfolio.derivatives_engine import DerivativesEngine


def extract_prices_multiindex(prices_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Extract adjusted close prices from yfinance MultiIndex DataFrame."""
    prices_clean = {}

    try:
        if isinstance(prices_df.columns, pd.MultiIndex):
            lvl0 = prices_df.columns.get_level_values(0)
            lvl1 = prices_df.columns.get_level_values(1)

            if 'Adj Close' in set(lvl0):
                for ticker in tickers:
                    if ticker in set(lvl1):
                        try:
                            prices_clean[ticker] = prices_df[('Adj Close', ticker)].astype(float)
                        except Exception:
                            pass
            elif 'Adj Close' in set(lvl1):
                for ticker in tickers:
                    if ticker in set(lvl0):
                        try:
                            prices_clean[ticker] = prices_df[(ticker, 'Adj Close')].astype(float)
                        except Exception:
                            pass
        else:
            for ticker in tickers:
                if ticker in prices_df.columns:
                    prices_clean[ticker] = prices_df[ticker].astype(float)
                elif 'Adj Close' in prices_df.columns:
                    prices_clean[ticker] = prices_df['Adj Close'].astype(float)

    except Exception as e:
        print(f"  Warning: Error extracting prices: {e}")

    if not prices_clean:
        raise ValueError("Failed to extract any price data from DataFrame")

    prices_df_clean = pd.DataFrame(prices_clean).ffill().dropna()
    return prices_df_clean


def run_enhanced_portfolio_construction(ticker: str, capital: float, rrr: float,
                                        start_date: Optional[str] = None,
                                        max_peers: int = 10,
                                        target_dte: int = 90,
                                        num_simulations: int = 10000) -> Dict:
    """
    Complete enhanced portfolio construction workflow.

    Parameters
    ----------
    ticker : str
        Primary ticker symbol
    capital : float
        Investment capital (USD)
    rrr : float
        Reward-to-risk ratio (0 to 1)
    start_date : str, optional
        Historical data start date (default: 1 year ago)
    max_peers : int
        Maximum number of peer stocks (default: 10)
    target_dte : int
        Target days to expiration for options (default: 90)
    num_simulations : int
        Number of Monte Carlo simulations (default: 10000)

    Returns
    -------
    dict
        Complete analysis results with charts and insights
    """
    print(f"\n{'='*80}")
    print(f"ENHANCED MULTI-ASSET PORTFOLIO CONSTRUCTION: {ticker}")
    print(f"{'='*80}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Reward-to-Risk Ratio: {rrr:.2f}")
    print(f"Monte Carlo Simulations: {num_simulations:,}")
    print(f"{'='*80}\n")

    workflow_start = time.time()

    # Step 1: Discover peers
    print("Step 1/8: Discovering peer companies...")
    peer_result = discover_peers(ticker, max_peers=max_peers)
    primary_meta = peer_result['primary_meta']
    peers = peer_result['peers']

    if not peers:
        print(f"  Warning: No peers found. Portfolio will include only {ticker} + ETFs.")

    # Step 2: Discover ETFs
    print("\nStep 2/8: Discovering relevant ETFs...")
    etf_result = discover_etfs(primary_meta)
    etf_tickers = etf_result['all_etfs']

    # Step 3: Build universe
    universe = [ticker] + peers + etf_tickers
    universe = list(dict.fromkeys(universe))

    print(f"\nStep 3/8: Building portfolio universe...")
    print(f"  Primary: {ticker}")
    print(f"  Peers: {peers} ({len(peers)})")
    print(f"  ETFs: {etf_tickers} ({len(etf_tickers)})")
    print(f"  Total universe: {len(universe)} assets")

    # Step 4: Fetch historical data
    print(f"\nStep 4/8: Fetching historical data...")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    bundle = fetch_full_bundle(tickers=universe, start=start_date)
    save_bundle(bundle, base_name="last_fetch")
    set_last_fetch_globals(bundle)

    prices_df_raw = bundle['prices']

    try:
        prices_df = extract_prices_multiindex(prices_df_raw, universe)
        print(f"  ✓ Extracted prices for {len(prices_df.columns)} assets")
    except Exception as e:
        print(f"  ✗ Error extracting prices: {e}")
        prices_df = extract_prices_multiindex(prices_df_raw, [ticker])
        universe = [ticker]

    # Step 5: Portfolio optimization
    print(f"\nStep 5/8: Optimizing portfolio allocation...")

    market_etf = etf_result.get('market_etf', {}).get('symbol') if etf_result.get('market_etf') else None
    if market_etf not in prices_df.columns:
        market_etf = None

    returns = compute_returns_matrix(prices_df)

    opt_result = optimize_mean_variance(
        returns=returns,
        rrr=rrr,
        market_ticker=market_etf,
        max_weight=0.40,
        min_weight=0.0
    )

    weights = opt_result['weights']
    current_prices = prices_df.iloc[-1]
    holdings_df = calculate_portfolio_shares(weights, capital, current_prices)

    print(f"\n  Portfolio allocation:")
    for _, row in holdings_df.iterrows():
        print(f"    {row['ticker']:6s}: {row['weight']*100:6.2f}% | ${row['dollars']:12,.2f} | {row['shares']:6.0f} shares @ ${row['price']:.2f}")

    # Step 6: Comprehensive derivative analysis
    print(f"\nStep 6/8: Analyzing comprehensive derivative strategies...")

    hedge_budget = calculate_hedge_budget(rrr, capital)
    print(f"  Hedge budget: ${hedge_budget:,.2f} ({hedge_budget/capital*100:.1f}%)")

    primary_price = current_prices.get(ticker, 0)
    primary_weight = weights.get(ticker, 0)
    primary_portfolio_value = capital * primary_weight

    # Calculate volatility for derivative pricing
    historical_vol = returns[ticker].std() * np.sqrt(252)

    # Get implied volatility from options
    implied_vol = None
    try:
        stock = yf.Ticker(ticker)
        opts = stock.option_chain(stock.options[0])
        if not opts.calls.empty:
            # Average IV from ATM options
            atm_calls = opts.calls[abs(opts.calls['strike'] - primary_price) < primary_price * 0.05]
            if not atm_calls.empty and 'impliedVolatility' in atm_calls.columns:
                implied_vol = atm_calls['impliedVolatility'].mean()
    except Exception:
        pass

    # Estimate dividend yield
    try:
        stock_info = yf.Ticker(ticker).info
        div_yield = stock_info.get('dividendYield', 0.02) or 0.02
    except Exception:
        div_yield = 0.02

    derivatives_engine = DerivativesEngine(
        ticker=ticker,
        current_price=primary_price,
        portfolio_value=primary_portfolio_value,
        risk_free_rate=0.05
    )

    derivatives_strategies = derivatives_engine.analyze_all_derivatives(
        hedge_budget=hedge_budget,
        historical_vol=historical_vol,
        implied_vol=implied_vol,
        expected_div_yield=div_yield
    )

    # Also get basic protective put for comparison
    options_overlay = None
    if primary_price > 0:
        options_overlay = select_hedge_options(
            ticker=ticker,
            current_price=primary_price,
            portfolio_value=primary_portfolio_value,
            hedge_budget=hedge_budget,
            strategy='put',
            target_dte=target_dte
        )

    # Step 7: Generate visualizations
    print(f"\nStep 7/8: Generating comprehensive visualizations and insights...")

    chart_gen = ChartGenerator(prices_df, returns, holdings_df, ticker)

    # Generate all charts
    price_chart, price_insights = chart_gen.generate_price_performance_chart()
    corr_chart, corr_insights = chart_gen.generate_correlation_heatmap()
    risk_return_chart, risk_insights = chart_gen.generate_risk_return_scatter(opt_result)
    allocation_chart, allocation_insights = chart_gen.generate_allocation_charts()

    chart_gen.insights['price_performance'] = price_insights
    chart_gen.insights['correlation'] = corr_insights
    chart_gen.insights['risk_return'] = risk_insights
    chart_gen.insights['allocation'] = allocation_insights

    charts = {
        'price_performance': price_chart,
        'correlation': corr_chart,
        'risk_return': risk_return_chart,
        'allocation': allocation_chart,
    }

    # Step 8: Monte Carlo simulation
    print(f"\nStep 8/8: Running Monte Carlo simulations...")

    mc_simulator = MonteCarloSimulator(returns, weights, capital)
    mc_results = mc_simulator.run_simulation(num_simulations=num_simulations, time_horizon=252)
    mc_chart, mc_insights = mc_simulator.generate_simulation_chart(mc_results)

    charts['monte_carlo'] = mc_chart

    # Scenario analysis
    scenarios = {
        'Bull Market': 0.15,
        'Base Case': 0.00,
        'Bear Market': -0.20,
        'Market Crash': -0.40,
        'Recovery': 0.30,
    }
    scenario_df = mc_simulator.scenario_analysis(scenarios)

    # Compile all insights
    insights_df = chart_gen.generate_all_insights_report()

    elapsed = time.time() - workflow_start

    print(f"\n{'='*80}")
    print(f"✓ Enhanced portfolio construction completed in {elapsed:.1f}s")
    print(f"{'='*80}\n")

    return {
        'ticker': ticker,
        'capital': capital,
        'rrr': rrr,
        'universe': universe,
        'peer_result': peer_result,
        'etf_result': etf_result,
        'prices_df': prices_df,
        'returns': returns,
        'opt_result': opt_result,
        'holdings_df': holdings_df,
        'options_overlay': options_overlay,
        'derivatives_strategies': derivatives_strategies,
        'hedge_budget': hedge_budget,
        'charts': charts,
        'insights_df': insights_df,
        'mc_results': mc_results,
        'mc_insights': mc_insights,
        'scenario_df': scenario_df,
        'elapsed_time': elapsed,
    }


def export_enhanced_report(analysis: Dict, output_path: str):
    """
    Generate comprehensive Excel report with embedded charts.

    Parameters
    ----------
    analysis : dict
        Portfolio analysis results
    output_path : str
        Output Excel file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ticker = analysis['ticker']
    capital = analysis['capital']
    rrr = analysis['rrr']

    print(f"\nGenerating enhanced Excel report with visualizations...")

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formats
        title_fmt = workbook.add_format({
            'bold': True, 'font_size': 16, 'font_color': '#1F4788', 'align': 'center'
        })
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1, 'align': 'center'
        })
        section_fmt = workbook.add_format({
            'bold': True, 'font_size': 12, 'bg_color': '#D9E1F2', 'border': 1
        })

        # ===== SHEET 1: EXECUTIVE SUMMARY =====
        ws = workbook.add_worksheet('Executive Summary')
        ws.set_column('A:A', 35)
        ws.set_column('B:E', 20)

        row = 0
        ws.merge_range(row, 0, row, 4, f'PORTFOLIO CONSTRUCTION: {ticker}', title_fmt)
        row += 2

        # Key metrics
        ws.merge_range(row, 0, row, 4, 'KEY PORTFOLIO METRICS', section_fmt)
        row += 1

        mc_insights = analysis['mc_insights']
        opt = analysis['opt_result']

        summary_data = [
            ['Total Capital', f'${capital:,.2f}'],
            ['Expected Annual Return', f"{opt['port_return']*100:.2f}%"],
            ['Portfolio Volatility', f"{opt['port_vol']*100:.2f}%"],
            ['Sharpe Ratio', f"{opt['sharpe']:.3f}"],
            ['Monte Carlo Expected Return', f"{mc_insights['expected_return_pct']:.2f}%"],
            ['Probability of Profit (1Y)', f"{mc_insights['probability_profit_pct']:.1f}%"],
            ['Value at Risk (95%)', f"${mc_insights['var_95_dollars']:,.0f}"],
            ['Best Case (95th %ile)', f"${mc_insights['best_case']:,.0f}"],
            ['Worst Case (5th %ile)', f"${mc_insights['worst_case']:,.0f}"],
            ['Number of Assets', len(analysis['universe'])],
            ['Hedge Budget', f"${analysis['hedge_budget']:,.2f}"],
        ]

        for label, value in summary_data:
            ws.write(row, 0, label)
            ws.write(row, 1, value)
            row += 1

        # ===== SHEET 2: PORTFOLIO HOLDINGS =====
        holdings_df = analysis['holdings_df'].sort_values('weight', ascending=False)
        holdings_df.to_excel(writer, sheet_name='Holdings', index=False)

        ws2 = writer.sheets['Holdings']
        ws2.set_column('A:A', 15)
        ws2.set_column('B:E', 18)

        # ===== SHEET 3: MONTE CARLO RESULTS =====
        ws3 = workbook.add_worksheet('Monte Carlo Analysis')
        ws3.set_column('A:A', 35)
        ws3.set_column('B:B', 20)

        row = 0
        ws3.merge_range(row, 0, row, 1, 'MONTE CARLO SIMULATION RESULTS', title_fmt)
        row += 2

        mc_data = [
            ['Number of Simulations', f"{len(analysis['mc_results']['final_values']):,}"],
            ['Time Horizon', '1 Year (252 trading days)'],
            ['Expected Final Value', f"${mc_insights['median_outcome']:,.0f}"],
            ['Expected Return', f"{mc_insights['expected_return_pct']:.2f}%"],
            ['Probability of Profit', f"{mc_insights['probability_profit_pct']:.1f}%"],
            ['Probability of >10% Loss', f"{mc_insights['probability_loss_10_pct']:.1f}%"],
            ['Upside Potential (95th %ile)', f"{mc_insights['upside_potential']:.2f}%"],
            ['Downside Risk (5th %ile)', f"{mc_insights['downside_risk']:.2f}%"],
            ['Value at Risk (95%)', f"${mc_insights['var_95_dollars']:,.0f}"],
            ['Conditional VaR (95%)', f"${mc_insights['cvar_95_dollars']:,.0f}"],
        ]

        for label, value in mc_data:
            ws3.write(row, 0, label)
            ws3.write(row, 1, value)
            row += 1

        row += 2
        ws3.merge_range(row, 0, row, 1, 'SCENARIO ANALYSIS', section_fmt)
        row += 1

        scenario_df = analysis['scenario_df']
        for col_idx, col_name in enumerate(scenario_df.columns):
            ws3.write(row, col_idx, col_name, header_fmt)
        row += 1

        for _, scenario_row in scenario_df.iterrows():
            for col_idx, value in enumerate(scenario_row):
                if isinstance(value, (int, float)):
                    ws3.write(row, col_idx, f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4f}")
                else:
                    ws3.write(row, col_idx, value)
            row += 1

        # ===== SHEET 4: DERIVATIVE STRATEGIES =====
        ws4 = workbook.add_worksheet('Derivative Strategies')
        ws4.set_column('A:A', 30)
        ws4.set_column('B:D', 20)

        row = 0
        ws4.merge_range(row, 0, row, 3, 'COMPREHENSIVE DERIVATIVE ANALYSIS', title_fmt)
        row += 2

        derivatives = analysis['derivatives_strategies']

        for strategy_name, strategy_data in derivatives.items():
            if strategy_data is None or 'error' in strategy_data:
                continue

            ws4.merge_range(row, 0, row, 3, strategy_data.get('strategy', strategy_name), section_fmt)
            row += 1

            # Write strategy details
            for key, value in strategy_data.items():
                if key in ['strategy', 'scenario_payoffs']:
                    continue

                ws4.write(row, 0, key.replace('_', ' ').title())

                if isinstance(value, (int, float)):
                    ws4.write(row, 1, f"{value:,.2f}")
                else:
                    ws4.write(row, 1, str(value))
                row += 1

            # Scenario payoffs if available
            if 'scenario_payoffs' in strategy_data:
                ws4.write(row, 0, 'Scenario Analysis', header_fmt)
                ws4.write(row, 1, 'Payoff', header_fmt)
                row += 1

                for scenario, payoff in strategy_data['scenario_payoffs'].items():
                    ws4.write(row, 0, scenario)
                    ws4.write(row, 1, f"${payoff:,.0f}" if isinstance(payoff, (int, float)) else str(payoff))
                    row += 1

            row += 1

        # ===== SHEET 5: INSIGHTS SUMMARY =====
        insights_df = analysis['insights_df']
        insights_df.to_excel(writer, sheet_name='All Insights')

        # ===== SHEET 6+: CHARTS =====
        # Embed charts as images
        chart_names = ['price_performance', 'correlation', 'risk_return', 'allocation', 'monte_carlo']

        for chart_name in chart_names:
            if chart_name not in analysis['charts']:
                continue

            ws_chart = workbook.add_worksheet(chart_name.replace('_', ' ').title())

            # Save chart to bytes
            img_data = BytesIO()
            analysis['charts'][chart_name].savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)

            # Insert image
            ws_chart.insert_image('A1', chart_name + '.png', {'image_data': img_data})

        # ===== SHEET: DIAGNOSTICS =====
        ws_diag = workbook.add_worksheet('Diagnostics')
        ws_diag.set_column('A:A', 30)
        ws_diag.set_column('B:C', 25)

        row = 0
        ws_diag.merge_range(row, 0, row, 2, 'DIAGNOSTICS & DISCOVERY', title_fmt)
        row += 2

        # Peer discovery
        ws_diag.merge_range(row, 0, row, 2, 'PEER DISCOVERY', section_fmt)
        row += 1

        peer_diag = analysis['peer_result']['diagnostics']
        for key, val in peer_diag.items():
            if isinstance(val, dict):
                ws_diag.write(row, 0, key)
                row += 1
                for k2, v2 in val.items():
                    ws_diag.write(row, 0, f'  {k2}')
                    ws_diag.write(row, 1, str(v2))
                    row += 1
            else:
                ws_diag.write(row, 0, key)
                ws_diag.write(row, 1, str(val))
                row += 1

    print(f"✓ Enhanced portfolio report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Asset Portfolio Constructor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('ticker', type=str, help='Primary stock ticker')
    parser.add_argument('--capital', type=float, required=True, help='Investment capital in USD')
    parser.add_argument('--rrr', type=float, required=True, help='Reward-to-risk ratio, 0.0 to 1.0')
    parser.add_argument('--start', type=str, default=None, help='Historical data start date YYYY-MM-DD')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--max-peers', type=int, default=10, help='Maximum number of peer stocks')
    parser.add_argument('--target-dte', type=int, default=90, help='Target days to expiration for options')
    parser.add_argument('--simulations', type=int, default=10000, help='Number of Monte Carlo simulations')

    args = parser.parse_args()

    if args.capital <= 0:
        print("Error: Capital must be positive")
        sys.exit(1)

    if not (0.0 <= args.rrr <= 1.0):
        print("Error: RRR must be between 0.0 and 1.0")
        sys.exit(1)

    if args.output is None:
        args.output = f"reports/{args.ticker}_comprehensive_portfolio.xlsx"

    try:
        # Run enhanced portfolio construction
        analysis = run_enhanced_portfolio_construction(
            ticker=args.ticker,
            capital=args.capital,
            rrr=args.rrr,
            start_date=args.start,
            max_peers=args.max_peers,
            target_dte=args.target_dte,
            num_simulations=args.simulations
        )

        # Export report
        export_enhanced_report(analysis, args.output)

        # Console summary
        print(f"\n{'='*80}")
        print("ENHANCED PORTFOLIO SUMMARY")
        print(f"{'='*80}")
        print(f"Primary Ticker: {args.ticker}")
        print(f"Total Capital: ${args.capital:,.2f}")
        print(f"RRR: {args.rrr:.2f}")
        print(f"\nExpected Return: {analysis['opt_result']['port_return']*100:.2f}%")
        print(f"Volatility: {analysis['opt_result']['port_vol']*100:.2f}%")
        print(f"Sharpe Ratio: {analysis['opt_result']['sharpe']:.3f}")
        print(f"\nMonte Carlo Results:")
        print(f"  Probability of Profit: {analysis['mc_insights']['probability_profit_pct']:.1f}%")
        print(f"  Expected Return: {analysis['mc_insights']['expected_return_pct']:.2f}%")
        print(f"  Value at Risk (95%): ${analysis['mc_insights']['var_95_dollars']:,.0f}")
        print(f"\nReport: {args.output}")
        print(f"{'='*80}\n")

        # Close all figures
        plt.close('all')

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
