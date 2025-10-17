"""
Streamlit Interactive Dashboard
================================
Real-time portfolio construction, backtesting, and analysis dashboard.

Features:
1. Portfolio Builder - Interactive long/short portfolio construction
2. Factor Analysis - Fama-French regression and alpha discovery
3. Backtest Runner - Walk-forward backtesting with visualizations
4. Performance Monitor - Real-time metrics and comparisons
5. Model Predictions - TFT forecasts with uncertainty
6. Risk Analytics - HRP optimization and risk decomposition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Optional

# Import our modules
import sys
sys.path.append('..')

from portfolio.factor_models import FamaFrenchFactorModel
from portfolio.long_short_strategy import create_market_neutral_portfolio, LongShortStrategy
from portfolio.hrp_optimizer import HRPOptimizer, compare_hrp_vs_meanvar
from backtesting.vectorbt_engine import VectorBTBacktest, BacktestConfig, TransactionCosts
from utils.financial_metrics import FinancialMetrics
from ml_models.features import FeatureEngineer

# Page config
st.set_page_config(
    page_title="Quantitative Finance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .reportview-container .main .block-container {
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)


# Cache data functions
@st.cache_data(ttl=3600)
def load_stock_data(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """Load stock data from Yahoo Finance."""
    try:
        if len(tickers) == 1:
            # Single ticker - yfinance returns simple DataFrame
            data = yf.download(tickers[0], period=period, progress=False)
            # Add ticker name to columns if needed
            if 'Adj Close' in data.columns:
                prices = pd.DataFrame(data['Adj Close'])
                prices.columns = tickers
                return prices
            return data
        else:
            # Multiple tickers - returns MultiIndex columns
            data = yf.download(tickers, period=period, progress=False)
            return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def calculate_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical features."""
    features = pd.DataFrame(index=prices.columns)

    # Returns-based features
    returns = prices.pct_change()

    features['momentum_20'] = (prices.iloc[-1] / prices.iloc[-20] - 1)
    features['momentum_60'] = (prices.iloc[-1] / prices.iloc[-60] - 1)
    features['volatility_20'] = returns.tail(20).std()
    features['volatility_60'] = returns.tail(60).std()

    # RSI
    for ticker in prices.columns:
        delta = returns[ticker]
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.loc[ticker, 'rsi_14'] = rsi.iloc[-1]

    return features


@st.cache_data(ttl=3600)
def run_factor_analysis(tickers: List[str], returns: pd.DataFrame) -> pd.DataFrame:
    """Run Fama-French factor regression."""
    ff = FamaFrenchFactorModel(model='5-factor')
    results = ff.batch_regress(tickers, returns, frequency='daily')
    return results


# ============================================================================
# SIDEBAR - Navigation and Global Settings
# ============================================================================

st.sidebar.title("📊 Quant Finance Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Portfolio Builder", "Factor Analysis", "Backtest Runner",
     "Performance Monitor", "Risk Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Settings")

# Universe selection
universe_preset = st.sidebar.selectbox(
    "Universe Preset",
    ["Tech Giants", "Dow 30", "Custom"]
)

if universe_preset == "Tech Giants":
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD']
elif universe_preset == "Dow 30":
    default_tickers = ['AAPL', 'MSFT', 'JPM', 'V', 'UNH', 'HD', 'PG', 'MA']
else:
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

universe_input = st.sidebar.text_area(
    "Tickers (comma-separated)",
    value=", ".join(default_tickers)
)
universe = [t.strip().upper() for t in universe_input.split(",")]

# Time period
period = st.sidebar.selectbox(
    "Data Period",
    ["1y", "2y", "3y", "5y"],
    index=1
)

# Load data
with st.spinner("Loading data..."):
    data = load_stock_data(universe, period=period)

if data.empty:
    st.error("Failed to load data. Please check tickers and try again.")
    st.stop()

# Handle both single and multi-ticker data formats
if isinstance(data.columns, pd.MultiIndex):
    # Multi-ticker format: extract 'Adj Close' column
    prices = data['Adj Close']
elif len(universe) == 1:
    # Single ticker already processed in load function
    prices = data
else:
    # Fallback: assume data is already prices
    prices = data

returns = prices.pct_change().dropna()

st.sidebar.success(f"✓ Loaded {len(universe)} stocks")
st.sidebar.caption(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}")


# ============================================================================
# PAGE 1: Portfolio Builder
# ============================================================================

if page == "Portfolio Builder":
    st.title("🎯 Long/Short Portfolio Builder")
    st.markdown("Build market-neutral portfolios with multi-factor scoring and sector constraints")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Portfolio Parameters")

        pcol1, pcol2, pcol3 = st.columns(3)

        with pcol1:
            capital = st.number_input(
                "Capital ($)",
                min_value=10000,
                max_value=10000000,
                value=100000,
                step=10000
            )

        with pcol2:
            target_net = st.slider(
                "Target Net Exposure",
                min_value=-0.2,
                max_value=0.2,
                value=0.0,
                step=0.05,
                format="%.2f"
            )

        with pcol3:
            max_sector = st.slider(
                "Max Sector Exposure",
                min_value=0.05,
                max_value=0.30,
                value=0.10,
                step=0.05,
                format="%.2f"
            )

    with col2:
        st.subheader("Factor Weights")

        momentum_weight = st.slider("Momentum", 0.0, 1.0, 0.30, 0.05)
        alpha_weight = st.slider("Alpha", 0.0, 1.0, 0.25, 0.05)
        quality_weight = st.slider("Quality", 0.0, 1.0, 0.20, 0.05)
        value_weight = st.slider("Value", 0.0, 1.0, 0.15, 0.05)
        volatility_weight = st.slider("Low Vol", 0.0, 1.0, 0.10, 0.05)

    # Build portfolio button
    if st.button("🚀 Build Portfolio", type="primary"):
        with st.spinner("Building portfolio..."):
            # Calculate features
            features = calculate_features(prices)

            # Get current prices and volatilities
            current_prices = prices.iloc[-1].to_dict()
            volatilities = returns.tail(20).std().to_dict()

            try:
                # Create portfolio
                portfolio = create_market_neutral_portfolio(
                    universe=universe,
                    features=features,
                    prices=current_prices,
                    volatilities=volatilities,
                    capital=capital,
                    target_net_exposure=target_net,
                    max_sector_exposure=max_sector,
                    method='atr'
                )

                # Display results
                st.success("✓ Portfolio constructed successfully!")

                # Metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                with mcol1:
                    st.metric("Net Exposure", f"{portfolio['net_exposure']:.2%}")

                with mcol2:
                    st.metric("Gross Exposure", f"{portfolio['gross_exposure']:.2%}")

                with mcol3:
                    n_long = len(portfolio['long_positions'])
                    n_short = len(portfolio['short_positions'])
                    st.metric("Long/Short", f"{n_long}/{n_short}")

                with mcol4:
                    total_value = sum(p['value'] for p in portfolio['long_positions'].values())
                    total_value += abs(sum(p['value'] for p in portfolio['short_positions'].values()))
                    st.metric("Total Position Value", f"${total_value:,.0f}")

                # Long positions
                st.subheader("📈 Long Positions")

                long_df = pd.DataFrame.from_dict(portfolio['long_positions'], orient='index')
                long_df = long_df[['shares', 'value', 'weight', 'sector']].sort_values('weight', ascending=False)
                long_df['weight'] = long_df['weight'].apply(lambda x: f"{x:.2%}")
                long_df['value'] = long_df['value'].apply(lambda x: f"${x:,.0f}")

                st.dataframe(long_df, use_container_width=True)

                # Short positions
                st.subheader("📉 Short Positions")

                short_df = pd.DataFrame.from_dict(portfolio['short_positions'], orient='index')
                short_df = short_df[['shares', 'value', 'weight', 'sector']].sort_values('weight')
                short_df['weight'] = short_df['weight'].apply(lambda x: f"{x:.2%}")
                short_df['value'] = short_df['value'].apply(lambda x: f"${x:,.0f}")

                st.dataframe(short_df, use_container_width=True)

                # Sector exposures
                st.subheader("🏢 Sector Exposures")

                sector_df = pd.DataFrame.from_dict(
                    portfolio['sector_exposures'],
                    orient='index',
                    columns=['Exposure']
                )
                sector_df = sector_df.sort_values('Exposure')

                fig = go.Figure()

                colors = ['red' if x < 0 else 'green' for x in sector_df['Exposure']]

                fig.add_trace(go.Bar(
                    x=sector_df['Exposure'],
                    y=sector_df.index,
                    orientation='h',
                    marker_color=colors
                ))

                fig.update_layout(
                    title="Sector Net Exposures",
                    xaxis_title="Net Exposure",
                    yaxis_title="Sector",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error building portfolio: {e}")
                st.exception(e)


# ============================================================================
# PAGE 2: Factor Analysis
# ============================================================================

elif page == "Factor Analysis":
    st.title("📊 Fama-French Factor Analysis")
    st.markdown("Alpha vs beta decomposition using 5-factor model")

    if st.button("🔍 Run Factor Analysis", type="primary"):
        with st.spinner("Running factor regression..."):
            try:
                results = run_factor_analysis(universe, returns)

                st.success("✓ Factor analysis complete!")

                # Summary metrics
                st.subheader("Summary Statistics")

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                with mcol1:
                    significant_alpha = (results['alpha_pvalue'] < 0.05).sum()
                    st.metric("Significant Alphas", f"{significant_alpha}/{len(results)}")

                with mcol2:
                    avg_r2 = results['r_squared'].mean()
                    st.metric("Avg R²", f"{avg_r2:.2%}")

                with mcol3:
                    avg_beta = results['beta_MKT'].mean()
                    st.metric("Avg Market Beta", f"{avg_beta:.2f}")

                with mcol4:
                    avg_obs = results['n_observations'].mean()
                    st.metric("Avg Observations", f"{avg_obs:.0f}")

                # Results table
                st.subheader("Factor Regression Results")

                display_df = results[['ticker', 'alpha', 'alpha_pvalue', 'beta_MKT',
                                     'beta_SMB', 'beta_HML', 'r_squared']].copy()

                # Format
                display_df['alpha'] = display_df['alpha'].apply(lambda x: f"{x:.4f}")
                display_df['alpha_pvalue'] = display_df['alpha_pvalue'].apply(lambda x: f"{x:.3f}")
                display_df['beta_MKT'] = display_df['beta_MKT'].apply(lambda x: f"{x:.2f}")
                display_df['beta_SMB'] = display_df['beta_SMB'].apply(lambda x: f"{x:.2f}")
                display_df['beta_HML'] = display_df['beta_HML'].apply(lambda x: f"{x:.2f}")
                display_df['r_squared'] = display_df['r_squared'].apply(lambda x: f"{x:.2%}")

                st.dataframe(display_df, use_container_width=True)

                # Alpha ranking chart
                st.subheader("Alpha Ranking")

                fig = go.Figure()

                sorted_results = results.sort_values('alpha', ascending=True)
                colors = ['green' if x < 0.05 else 'gray'
                         for x in sorted_results['alpha_pvalue']]

                fig.add_trace(go.Bar(
                    x=sorted_results['alpha'],
                    y=sorted_results['ticker'],
                    orientation='h',
                    marker_color=colors,
                    text=sorted_results['alpha'].apply(lambda x: f"{x:.3f}"),
                    textposition='auto'
                ))

                fig.update_layout(
                    title="Alpha by Ticker (green = significant p<0.05)",
                    xaxis_title="Alpha (annualized)",
                    yaxis_title="Ticker",
                    height=max(400, len(universe) * 30)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Factor exposures heatmap
                st.subheader("Factor Exposures")

                factor_cols = ['beta_MKT', 'beta_SMB', 'beta_HML', 'beta_RMW', 'beta_CMA']
                exposure_matrix = results.set_index('ticker')[factor_cols]

                fig = px.imshow(
                    exposure_matrix.T,
                    labels=dict(x="Ticker", y="Factor", color="Beta"),
                    x=exposure_matrix.index,
                    y=exposure_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )

                fig.update_layout(
                    title="Factor Beta Exposures",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error in factor analysis: {e}")
                st.exception(e)


# ============================================================================
# PAGE 3: Backtest Runner
# ============================================================================

elif page == "Backtest Runner":
    st.title("⏮️ Strategy Backtester")
    st.markdown("Walk-forward backtesting with realistic transaction costs")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Backtest Parameters")

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )

        commission = st.slider(
            "Commission (bps)",
            min_value=0,
            max_value=50,
            value=10,
            step=1
        ) / 10000

        slippage = st.slider(
            "Slippage (bps)",
            min_value=0,
            max_value=50,
            value=5,
            step=1
        ) / 10000

    with col2:
        st.subheader("Strategy Parameters")

        strategy_type = st.selectbox(
            "Strategy",
            ["Simple Momentum", "Mean Reversion"]
        )

        if strategy_type == "Simple Momentum":
            sma_short = st.slider("Short SMA", 5, 50, 20, 5)
            sma_long = st.slider("Long SMA", 20, 200, 50, 10)
        else:
            lookback = st.slider("Lookback Period", 10, 60, 20, 5)
            entry_threshold = st.slider("Entry Threshold (std)", 1.0, 3.0, 2.0, 0.5)

    ticker_select = st.selectbox("Select Ticker for Backtest", universe)

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                # Get data for selected ticker
                ticker_prices = prices[ticker_select]
                benchmark_prices = prices[universe[0]] if len(universe) > 1 else ticker_prices

                # Generate signals
                signals = pd.Series(0, index=ticker_prices.index)

                if strategy_type == "Simple Momentum":
                    sma_s = ticker_prices.rolling(sma_short).mean()
                    sma_l = ticker_prices.rolling(sma_long).mean()
                    signals[sma_s > sma_l] = 1
                    signals[sma_s <= sma_l] = 0
                else:
                    # Mean reversion
                    rolling_mean = ticker_prices.rolling(lookback).mean()
                    rolling_std = ticker_prices.rolling(lookback).std()
                    z_score = (ticker_prices - rolling_mean) / rolling_std
                    signals[z_score < -entry_threshold] = 1  # Buy oversold
                    signals[z_score > entry_threshold] = -1  # Short overbought
                    signals[abs(z_score) < 0.5] = 0  # Exit near mean

                # Run backtest
                config = BacktestConfig(
                    initial_capital=initial_capital,
                    costs=TransactionCosts(
                        commission_pct=commission,
                        slippage_pct=slippage,
                        short_borrow_rate=0.0
                    )
                )

                backtest = VectorBTBacktest(
                    ticker_prices,
                    signals,
                    config,
                    benchmark=benchmark_prices
                )

                results = backtest.run()

                st.success("✓ Backtest complete!")

                # Performance metrics
                st.subheader("Performance Summary")

                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

                with mcol1:
                    st.metric("Total Return", f"{results.total_return:.2%}")

                with mcol2:
                    st.metric("Sharpe Ratio", f"{results.sharpe_ratio:.2f}")

                with mcol3:
                    st.metric("Max Drawdown", f"{results.max_drawdown:.2%}")

                with mcol4:
                    st.metric("Win Rate", f"{results.win_rate:.2%}")

                with mcol5:
                    st.metric("Total Trades", f"{results.total_trades}")

                # Equity curve
                st.subheader("Equity Curve")

                fig = go.Figure()

                # Strategy equity
                equity = results.portfolio_value
                fig.add_trace(go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    mode='lines',
                    name='Strategy',
                    line=dict(color='blue', width=2)
                ))

                # Benchmark (buy & hold)
                benchmark_equity = initial_capital * (1 + (ticker_prices / ticker_prices.iloc[0] - 1))
                fig.add_trace(go.Scatter(
                    x=benchmark_equity.index,
                    y=benchmark_equity.values,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', width=1, dash='dash')
                ))

                fig.update_layout(
                    title=f"{ticker_select} - Strategy vs Buy & Hold",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Returns distribution
                st.subheader("Returns Distribution")

                col1, col2 = st.columns(2)

                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=results.returns,
                        nbinsx=50,
                        name='Returns'
                    ))
                    fig.update_layout(
                        title="Daily Returns Distribution",
                        xaxis_title="Return",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Cumulative returns
                    cum_returns = (1 + results.returns).cumprod()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns.values,
                        mode='lines',
                        fill='tozeroy'
                    ))
                    fig.update_layout(
                        title="Cumulative Returns",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error running backtest: {e}")
                st.exception(e)


# ============================================================================
# PAGE 4: Performance Monitor
# ============================================================================

elif page == "Performance Monitor":
    st.title("📈 Performance Monitor")
    st.markdown("Real-time performance metrics and attribution")

    # Calculate metrics for each stock
    st.subheader("Individual Stock Performance")

    metrics_data = []

    for ticker in universe:
        ticker_returns = returns[ticker]

        # Calculate metrics
        total_return = (prices[ticker].iloc[-1] / prices[ticker].iloc[0] - 1)
        annual_return = ((1 + total_return) ** (252 / len(ticker_returns)) - 1)
        volatility = ticker_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0

        # Max drawdown
        cum_returns = (1 + ticker_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()

        metrics_data.append({
            'Ticker': ticker,
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Display table
    display_metrics = metrics_df.copy()
    display_metrics['Total Return'] = display_metrics['Total Return'].apply(lambda x: f"{x:.2%}")
    display_metrics['Annual Return'] = display_metrics['Annual Return'].apply(lambda x: f"{x:.2%}")
    display_metrics['Volatility'] = display_metrics['Volatility'].apply(lambda x: f"{x:.2%}")
    display_metrics['Sharpe Ratio'] = display_metrics['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
    display_metrics['Max Drawdown'] = display_metrics['Max Drawdown'].apply(lambda x: f"{x:.2%}")

    st.dataframe(display_metrics, use_container_width=True)

    # Performance comparison chart
    st.subheader("Cumulative Returns Comparison")

    fig = go.Figure()

    for ticker in universe:
        cum_returns = (1 + returns[ticker]).cumprod()
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode='lines',
            name=ticker
        ))

    fig.update_layout(
        title="Cumulative Returns - All Stocks",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Risk-return scatter
    st.subheader("Risk-Return Profile")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics_df['Volatility'],
        y=metrics_df['Annual Return'],
        mode='markers+text',
        text=metrics_df['Ticker'],
        textposition='top center',
        marker=dict(
            size=metrics_df['Sharpe Ratio'].abs() * 20,
            color=metrics_df['Sharpe Ratio'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        )
    ))

    fig.update_layout(
        title="Risk-Return Scatter (bubble size = Sharpe Ratio)",
        xaxis_title="Volatility (annualized)",
        yaxis_title="Return (annualized)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 5: Risk Analytics
# ============================================================================

elif page == "Risk Analytics":
    st.title("⚠️ Risk Analytics")
    st.markdown("Hierarchical Risk Parity and portfolio risk decomposition")

    if st.button("🔍 Run HRP Analysis", type="primary"):
        with st.spinner("Running HRP optimization..."):
            try:
                # HRP optimization
                hrp = HRPOptimizer(linkage_method='single', distance_metric='correlation')
                result = hrp.allocate_with_metadata(returns)

                weights = result['weights']
                sorted_tickers = result['sorted_tickers']

                st.success("✓ HRP optimization complete!")

                # Weight statistics
                st.subheader("HRP Portfolio Weights")

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                with mcol1:
                    st.metric("Min Weight", f"{weights.min():.2%}")

                with mcol2:
                    st.metric("Max Weight", f"{weights.max():.2%}")

                with mcol3:
                    st.metric("Std Weight", f"{weights.std():.2%}")

                with mcol4:
                    top3 = weights.nlargest(3).sum()
                    st.metric("Top 3 Concentration", f"{top3:.2%}")

                # Weight allocation chart
                col1, col2 = st.columns(2)

                with col1:
                    fig = go.Figure()

                    sorted_weights = weights.sort_values(ascending=True)

                    fig.add_trace(go.Bar(
                        x=sorted_weights.values,
                        y=sorted_weights.index,
                        orientation='h',
                        marker_color='steelblue'
                    ))

                    fig.update_layout(
                        title="HRP Portfolio Weights",
                        xaxis_title="Weight",
                        yaxis_title="Ticker",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = go.Figure()

                    fig.add_trace(go.Pie(
                        labels=weights.index,
                        values=weights.values,
                        hole=0.3
                    ))

                    fig.update_layout(
                        title="HRP Weight Distribution",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Cluster ordering
                st.subheader("Hierarchical Clustering")

                st.write("**Cluster Ordering** (similar assets grouped):")
                st.write(" → ".join(sorted_tickers))

                # Correlation heatmap
                st.subheader("Correlation Matrix")

                corr = returns[sorted_tickers].corr()

                fig = px.imshow(
                    corr,
                    labels=dict(x="Ticker", y="Ticker", color="Correlation"),
                    x=corr.columns,
                    y=corr.index,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    aspect="auto"
                )

                fig.update_layout(
                    title="Correlation Matrix (HRP ordering)",
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                # Compare with mean-variance
                st.subheader("HRP vs Mean-Variance Comparison")

                with st.spinner("Comparing optimization methods..."):
                    comparison = compare_hrp_vs_meanvar(returns, risk_aversion=5.0)

                    # This will print to console, capture output
                    st.info("Comparison complete! Check console output for detailed metrics.")

            except Exception as e:
                st.error(f"Error in HRP analysis: {e}")
                st.exception(e)


# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.caption("💻 Quantitative Finance System v1.0")
st.sidebar.caption("Built with Streamlit + Python")
