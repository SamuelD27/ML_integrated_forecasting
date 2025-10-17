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
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try importing required modules, provide helpful errors if missing
try:
    from portfolio.factor_models import FamaFrenchFactorModel
    from portfolio.long_short_strategy import (
        create_market_neutral_portfolio,
        LongShortStrategy,
        PortfolioConstraints,
        FactorWeights
    )
    from portfolio.hrp_optimizer import HRPOptimizer, compare_hrp_vs_meanvar
    from backtesting.vectorbt_engine import VectorBTBacktest, BacktestConfig, TransactionCosts
    from utils.financial_metrics import FinancialMetrics
    from ml_models.features import FeatureEngineer
except ImportError as e:
    # If imports fail, we'll handle it gracefully in the UI
    import_error = str(e)
    FamaFrenchFactorModel = None
    create_market_neutral_portfolio = None
    LongShortStrategy = None
    PortfolioConstraints = None
    FactorWeights = None
    HRPOptimizer = None
    compare_hrp_vs_meanvar = None
    VectorBTBacktest = None
    BacktestConfig = None
    TransactionCosts = None
    FinancialMetrics = None
    FeatureEngineer = None
else:
    import_error = None

# Page config
st.set_page_config(
    page_title="Quantitative Finance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fix metric readability
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }

    /* Fix metric contrast - make text highly readable */
    .stMetric {
        background-color: #ffffff !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* Metric label - dark text */
    .stMetric label {
        color: #1f1f1f !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* Metric value - very dark text, large */
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Metric delta - keep default colors but ensure visibility */
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    .reportview-container .main .block-container {
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA EXTRACTION HELPER FUNCTIONS
# ============================================================================

def get_price_column(data: pd.DataFrame,
                     tickers: List[str],
                     prefer_adjusted: bool = True) -> pd.DataFrame:
    """
    Safely extract price data from yfinance DataFrame.

    Handles both single-ticker and multi-ticker formats.
    Falls back to 'Close' if 'Adj Close' not available.

    Args:
        data: Raw DataFrame from yfinance
        tickers: List of ticker symbols
        prefer_adjusted: If True, prefer 'Adj Close' over 'Close'

    Returns:
        DataFrame with tickers as columns, dates as index

    Raises:
        ValueError: If no price columns found

    Example:
        >>> data = yf.download(['AAPL', 'MSFT'], period='1y')
        >>> prices = get_price_column(data, ['AAPL', 'MSFT'])
    """
    if data.empty:
        raise ValueError("Empty DataFrame provided")

    # Check if MultiIndex columns (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker format: ('Adj Close', 'AAPL'), ('Close', 'AAPL'), etc.
        try:
            # Try to get Adj Close for all tickers
            if prefer_adjusted and 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close'].copy()
            elif 'Close' in data.columns.get_level_values(0):
                prices = data['Close'].copy()
            else:
                raise ValueError(f"No price columns found. Available: {data.columns.get_level_values(0).unique().tolist()}")

            # Verify we got data for requested tickers
            missing = set(tickers) - set(prices.columns)
            if missing:
                raise ValueError(f"Missing data for tickers: {missing}")

            return prices

        except Exception as e:
            raise ValueError(f"Failed to extract prices from multi-ticker data: {e}")

    else:
        # Single ticker format: simple columns ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if len(tickers) != 1:
            raise ValueError(f"Expected 1 ticker but got {len(tickers)}: {tickers}")

        ticker = tickers[0]

        # Try Adj Close first, then Close
        if prefer_adjusted and 'Adj Close' in data.columns:
            prices = pd.DataFrame({ticker: data['Adj Close']})
        elif 'Close' in data.columns:
            prices = pd.DataFrame({ticker: data['Close']})
        else:
            raise ValueError(f"No price columns found. Available: {data.columns.tolist()}")

        return prices


# Cache data functions
@st.cache_data(ttl=3600)
def load_stock_data(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """
    Load stock data from Yahoo Finance with robust error handling.

    Args:
        tickers: List of ticker symbols
        period: Time period ('1y', '2y', '3y', '5y', etc.)

    Returns:
        Raw DataFrame from yfinance (will be processed by get_price_column)

    Raises:
        Exception: If data fetch fails after retries
    """
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            # Download data
            data = yf.download(
                tickers,
                period=period,
                progress=False
            )

            if data.empty:
                raise ValueError(f"No data returned for {tickers}")

            # Validate that we have at least some price columns
            if isinstance(data.columns, pd.MultiIndex):
                available_cols = data.columns.get_level_values(0).unique().tolist()
            else:
                available_cols = data.columns.tolist()

            if not any(col in available_cols for col in ['Close', 'Adj Close']):
                raise ValueError(f"No price columns in returned data. Available: {available_cols}")

            return data

        except Exception as e:
            if attempt < max_retries - 1:
                # Retry with exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                # Final attempt failed
                raise Exception(f"Failed to load data after {max_retries} attempts: {e}")


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

# Show import error warning if modules failed to load
if import_error:
    st.sidebar.error("⚠️ Some modules failed to import")
    st.error(f"**Import Error**: {import_error}")
    st.info("Some dashboard features may not be available. Please ensure all dependencies are installed.")
    st.code("pip install -r requirements_training.txt", language="bash")

page = st.sidebar.radio(
    "Navigation",
    ["Single Stock Analysis", "Portfolio Builder", "Factor Analysis", "Backtest Runner",
     "Performance Monitor", "Risk Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Settings")

# Data refresh button
if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload fresh data"):
    st.cache_data.clear()
    st.rerun()

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
    try:
        data = load_stock_data(universe, period=period)
        prices = get_price_column(data, universe, prefer_adjusted=True)
        returns = prices.pct_change().dropna()

        st.sidebar.success(f"✓ Loaded {len(universe)} stocks")
        st.sidebar.caption(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}")

    except ValueError as e:
        st.sidebar.error(f"❌ Data extraction error")
        st.error(f"**Could not extract price data:** {e}")
        st.info("💡 **Suggestions:**\n"
                "- Check that ticker symbols are valid\n"
                "- Try reducing the number of tickers\n"
                "- Use the refresh button to retry")
        st.stop()

    except Exception as e:
        st.sidebar.error(f"❌ Data loading failed")
        st.error(f"**Failed to load data:** {e}")
        st.info("💡 **Suggestions:**\n"
                "- Check your internet connection\n"
                "- Verify ticker symbols are correct\n"
                "- Try a different time period\n"
                "- Use the refresh button to retry")
        st.stop()


# ============================================================================
# PAGE 0: Single Stock Analysis
# ============================================================================

if page == "Single Stock Analysis":
    st.title("🔍 Single Stock Analysis")
    st.markdown("Comprehensive analysis of individual stocks with buy/sell recommendation")

    # Ticker selection
    ticker_input = st.text_input(
        "Enter Stock Ticker",
        value="AAPL",
        help="Enter a single stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
    ).upper()

    analyze_button = st.button("📊 Analyze Stock", type="primary")

    if analyze_button and ticker_input:
        with st.spinner(f"Analyzing {ticker_input}..."):
            try:
                # Load data for single ticker
                single_data = load_stock_data([ticker_input], period=period)
                single_prices = get_price_column(single_data, [ticker_input], prefer_adjusted=True)
                single_returns = single_prices.pct_change().dropna()

                # Calculate comprehensive metrics
                st.success(f"✓ Loaded {len(single_prices)} days of data for {ticker_input}")

                # === Technical Analysis ===
                st.subheader("📈 Technical Analysis")

                tcol1, tcol2, tcol3, tcol4 = st.columns(4)

                # Current price & change
                current_price = single_prices[ticker_input].iloc[-1]
                price_1d_ago = single_prices[ticker_input].iloc[-2] if len(single_prices) > 1 else current_price
                price_change_1d = (current_price - price_1d_ago) / price_1d_ago

                with tcol1:
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        delta=f"{price_change_1d:.2%}",
                        help="Current price vs 1 day ago"
                    )

                # Volatility
                volatility_20d = single_returns[ticker_input].tail(20).std() * np.sqrt(252)
                with tcol2:
                    st.metric(
                        "Volatility (20d)",
                        f"{volatility_20d:.1%}",
                        help="Annualized volatility based on 20-day returns"
                    )

                # Momentum
                momentum_20 = (single_prices[ticker_input].iloc[-1] / single_prices[ticker_input].iloc[-20] - 1) if len(single_prices) >= 20 else 0
                with tcol3:
                    st.metric(
                        "20-Day Momentum",
                        f"{momentum_20:.2%}",
                        help="Price change over last 20 days"
                    )

                # RSI
                delta = single_returns[ticker_input]
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50

                with tcol4:
                    rsi_label = "Oversold" if current_rsi < 30 else ("Overbought" if current_rsi > 70 else "Neutral")
                    st.metric(
                        "RSI (14)",
                        f"{current_rsi:.1f}",
                        delta=rsi_label,
                        help="Relative Strength Index: <30 oversold, >70 overbought"
                    )

                # === Factor Analysis ===
                st.subheader("🎯 Factor Analysis (Fama-French)")

                try:
                    # Run factor regression
                    ff_results = run_factor_analysis([ticker_input], single_returns)

                    fcol1, fcol2, fcol3, fcol4 = st.columns(4)

                    with fcol1:
                        alpha = ff_results.iloc[0]['alpha'] * 252  # Annualize
                        alpha_pval = ff_results.iloc[0]['alpha_pvalue']
                        is_significant = "✓ Significant" if alpha_pval < 0.05 else "Not significant"
                        st.metric(
                            "Alpha (Annual)",
                            f"{alpha:.2%}",
                            delta=is_significant,
                            help=f"Risk-adjusted excess return (p={alpha_pval:.3f})"
                        )

                    with fcol2:
                        beta = ff_results.iloc[0]['beta_MKT']
                        beta_label = "High Beta" if beta > 1.2 else ("Low Beta" if beta < 0.8 else "Market Beta")
                        st.metric(
                            "Market Beta",
                            f"{beta:.2f}",
                            delta=beta_label,
                            help="Sensitivity to market movements (1.0 = market)"
                        )

                    with fcol3:
                        r2 = ff_results.iloc[0]['r_squared']
                        st.metric(
                            "R² (Fit)",
                            f"{r2:.1%}",
                            help="How much variance explained by factors"
                        )

                    with fcol4:
                        sharpe = (single_returns[ticker_input].mean() * 252) / (single_returns[ticker_input].std() * np.sqrt(252))
                        st.metric(
                            "Sharpe Ratio",
                            f"{sharpe:.2f}",
                            help="Risk-adjusted return (>1 good, >2 excellent)"
                        )

                except Exception as e:
                    st.warning(f"Factor analysis unavailable: {e}")

                # === Recommendation ===
                st.subheader("💡 Trading Recommendation")

                # Simple scoring system
                score = 0
                reasons = []

                # Technical signals
                if current_rsi < 30:
                    score += 2
                    reasons.append("✓ RSI oversold (bullish)")
                elif current_rsi > 70:
                    score -= 2
                    reasons.append("✗ RSI overbought (bearish)")

                if momentum_20 > 0.05:
                    score += 1
                    reasons.append("✓ Strong positive momentum")
                elif momentum_20 < -0.05:
                    score -= 1
                    reasons.append("✗ Negative momentum")

                # Factor signals
                try:
                    if alpha > 0.02 and alpha_pval < 0.10:
                        score += 2
                        reasons.append("✓ Positive alpha (outperforming)")
                    elif alpha < -0.02:
                        score -= 1
                        reasons.append("✗ Negative alpha")
                except:
                    pass

                # Generate recommendation
                if score >= 3:
                    recommendation = "🟢 STRONG BUY"
                    explanation = "Multiple bullish signals indicate good long opportunity"
                    position = "LONG"
                elif score >= 1:
                    recommendation = "🟡 BUY"
                    explanation = "Moderately bullish signals, consider long position"
                    position = "LONG"
                elif score <= -3:
                    recommendation = "🔴 STRONG SELL / SHORT"
                    explanation = "Multiple bearish signals indicate good short opportunity"
                    position = "SHORT"
                elif score <= -1:
                    recommendation = "🟠 SELL / CONSIDER SHORT"
                    explanation = "Moderately bearish signals, consider short position"
                    position = "SHORT"
                else:
                    recommendation = "⚪ NEUTRAL / HOLD"
                    explanation = "Mixed signals, no clear directional bias"
                    position = "NEUTRAL"

                # Display recommendation prominently
                rec_col1, rec_col2 = st.columns([1, 2])

                with rec_col1:
                    st.markdown(f"### {recommendation}")
                    st.markdown(f"**Position:** {position}")
                    st.markdown(f"**Score:** {score}/5")

                with rec_col2:
                    st.info(f"**Analysis:** {explanation}")

                    st.markdown("**Supporting Factors:**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")

                # === Price Chart ===
                st.subheader("📉 Price History")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=single_prices.index,
                    y=single_prices[ticker_input],
                    mode='lines',
                    name=ticker_input,
                    line=dict(color='blue', width=2)
                ))

                # Add SMA
                sma_20 = single_prices[ticker_input].rolling(20).mean()
                sma_50 = single_prices[ticker_input].rolling(50).mean()

                fig.add_trace(go.Scatter(
                    x=single_prices.index,
                    y=sma_20,
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=single_prices.index,
                    y=sma_50,
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1, dash='dash')
                ))

                fig.update_layout(
                    title=f"{ticker_input} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing {ticker_input}: {e}")
                st.exception(e)


# ============================================================================
# PAGE 1: Portfolio Builder
# ============================================================================

elif page == "Portfolio Builder":
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
            current_prices = prices.iloc[-1]  # Keep as Series
            volatilities = returns.tail(20).std()  # Keep as Series

            try:
                # Create portfolio constraints
                constraints = PortfolioConstraints(
                    target_net_exposure=target_net,
                    max_sector_exposure=max_sector,
                    max_position_size=0.30,  # 30% max per position
                    min_position_size=0.01   # 1% min per position
                )

                # Create factor weights
                factor_weights = FactorWeights(
                    momentum=momentum_weight,
                    alpha=alpha_weight,
                    quality=quality_weight,
                    value=value_weight,
                    volatility=volatility_weight
                )

                # Create portfolio
                portfolio = create_market_neutral_portfolio(
                    universe=universe,
                    features=features,
                    prices=current_prices,
                    volatilities=volatilities,
                    capital=capital,
                    factor_weights=factor_weights,
                    constraints=constraints
                )

                # Display results
                st.success("✓ Portfolio constructed successfully!")

                # Extract data from portfolio structure
                portfolio_df = portfolio['portfolio']
                positions = portfolio['positions']
                exposures = portfolio['exposures']

                # Calculate summary metrics
                long_df = portfolio_df[portfolio_df['size'] > 0].copy()
                short_df = portfolio_df[portfolio_df['size'] < 0].copy()

                total_long = long_df['size'].sum() if not long_df.empty else 0
                total_short = abs(short_df['size'].sum()) if not short_df.empty else 0
                net_exposure = (total_long - total_short) / capital if capital > 0 else 0
                gross_exposure = (total_long + total_short) / capital if capital > 0 else 0

                # Metrics with better styling and clarity
                st.markdown("### 📊 Portfolio Summary")

                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

                with mcol1:
                    st.metric(
                        "Net Exposure",
                        f"{net_exposure:.1%}",
                        delta=None,
                        help="(Long $ - Short $) / Capital. Target: 0% for market neutral"
                    )

                with mcol2:
                    st.metric(
                        "Gross Exposure",
                        f"{gross_exposure:.1%}",
                        help="(Long $ + Short $) / Capital. Higher = more leverage"
                    )

                with mcol3:
                    n_long = len(long_df)
                    n_short = len(short_df)
                    st.metric(
                        "Positions",
                        f"{n_long}L / {n_short}S",
                        help=f"{n_long} long positions, {n_short} short positions"
                    )

                with mcol4:
                    st.metric(
                        "Total Deployed",
                        f"${total_value:,.0f}",
                        help=f"Capital used: ${total_value:,.0f} of ${capital:,.0f}"
                    )

                with mcol5:
                    cash_remaining = capital - total_value
                    st.metric(
                        "Cash Remaining",
                        f"${cash_remaining:,.0f}",
                        help="Unallocated capital"
                    )

                # Add explanation row
                st.markdown("---")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.info(f"💰 **Long Positions:** ${total_long:,.0f} ({total_long/capital*100:.1f}% of capital)")
                with col_info2:
                    st.info(f"📉 **Short Positions:** ${total_short:,.0f} ({total_short/capital*100:.1f}% of capital)")

                # Long positions
                st.subheader("📈 Long Positions")
                st.caption("💡 Weight = Position size as % of total capital")

                if not long_df.empty:
                    display_long = long_df.copy()
                    # Weight is % of total capital
                    display_long['% of Capital'] = (display_long['size'] / capital).apply(lambda x: f"{x:.2%}")
                    # Also show % of long book
                    display_long['% of Long Book'] = (display_long['size'] / total_long).apply(lambda x: f"{x:.1%}") if total_long > 0 else "0%"
                    display_long['Position Size'] = display_long['size'].apply(lambda x: f"${x:,.0f}")
                    display_long = display_long[['ticker', 'Position Size', '% of Capital', '% of Long Book', 'sector']].sort_values('% of Capital', ascending=False)
                    st.dataframe(display_long, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ No long positions created - portfolio may be unbalanced")

                # Short positions
                st.subheader("📉 Short Positions")
                st.caption("💡 Weight = Position size as % of total capital")

                if not short_df.empty:
                    display_short = short_df.copy()
                    # Weight is % of total capital
                    display_short['% of Capital'] = (abs(display_short['size']) / capital).apply(lambda x: f"{x:.2%}")
                    # Also show % of short book
                    display_short['% of Short Book'] = (abs(display_short['size']) / total_short).apply(lambda x: f"{x:.1%}") if total_short > 0 else "0%"
                    display_short['Position Size'] = display_short['size'].apply(lambda x: f"${x:,.0f}")
                    display_short = display_short[['ticker', 'Position Size', '% of Capital', '% of Short Book', 'sector']].sort_values('% of Capital', ascending=False)
                    st.dataframe(display_short, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ No short positions created - portfolio may be unbalanced")

                # Sector exposures
                st.subheader("🏢 Sector Exposures")

                if exposures:
                    sector_df = pd.DataFrame.from_dict(
                        exposures,
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
                else:
                    st.info("No sector exposure data available")

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
