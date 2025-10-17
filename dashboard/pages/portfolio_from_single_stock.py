"""
Portfolio from Single Stock Analysis
====================================
Build a portfolio starting from a single stock and finding optimal peers.

Features:
- Automatic peer discovery
- Multiple optimization methods
- Risk/return analysis
- Export to CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

# Import backend modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from portfolio.peer_discovery import discover_peers
from portfolio.hrp_optimizer import HRPOptimizer
from portfolio.advanced_optimizer import optimize_mean_variance
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


def calculate_risk_metrics(returns: pd.Series) -> Dict:
    """Calculate comprehensive risk metrics from returns series."""
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annual_return / downside_std if downside_std > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown
    }


def show():
    """Main function for portfolio from single stock page."""
    st.title("📊 Portfolio from Single Stock")
    st.markdown("""
    Build a diversified portfolio starting from a single stock.
    We'll find optimal peers and construct an efficient portfolio.
    """)

    # Sidebar inputs
    st.sidebar.header("Configuration")

    ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
    n_peers = st.sidebar.slider("Number of Peers", 3, 15, 5)
    lookback_period = st.sidebar.selectbox(
        "Lookback Period",
        ["1y", "2y", "3y", "5y"],
        index=1
    )

    optimization_method = st.sidebar.selectbox(
        "Optimization Method",
        ["HRP (Hierarchical Risk Parity)", "Mean-Variance", "Equal Weight"]
    )

    if st.sidebar.button("Build Portfolio", type="primary"):
        with st.spinner("Building portfolio..."):
            try:
                # 1. Fetch base stock data
                st.subheader(f"1️⃣ Analyzing Base Stock: {ticker}")
                base_stock = yf.Ticker(ticker)
                base_info = base_stock.info

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Company", base_info.get('longName', ticker))
                with col2:
                    st.metric("Sector", base_info.get('sector', 'N/A'))
                with col3:
                    st.metric("Industry", base_info.get('industry', 'N/A'))

                # 2. Discover peers
                st.subheader("2️⃣ Discovering Similar Stocks")

                # Get peers based on sector and characteristics
                sector = base_info.get('sector', '')
                industry = base_info.get('industry', '')
                market_cap = base_info.get('marketCap', 0)

                # Use peer discovery function
                all_peers = discover_peers(
                    primary_ticker=ticker,
                    max_peers=n_peers * 3,  # Get more candidates
                    min_price=5.0,
                    min_avg_volume=1_000_000,
                    match_industry=True
                )

                # Filter out the primary ticker itself
                all_peers = [p for p in all_peers if p != ticker]

                if len(all_peers) < n_peers:
                    st.warning(f"Found only {len(all_peers)} peers (requested {n_peers})")
                    selected_peers = all_peers
                else:
                    # Select top N
                    selected_peers = all_peers[:n_peers]

                st.info(f"✅ Found {len(selected_peers)} peer stocks")

                # Display peers
                peer_df = pd.DataFrame([
                    {
                        'Ticker': p,
                        'Name': yf.Ticker(p).info.get('longName', p),
                        'Sector': yf.Ticker(p).info.get('sector', 'N/A')
                    }
                    for p in selected_peers
                ])
                st.dataframe(peer_df, use_container_width=True)

                # 3. Fetch historical data for all stocks
                st.subheader("3️⃣ Fetching Historical Data")
                all_tickers = [ticker] + selected_peers

                # Download data
                data = yf.download(
                    all_tickers,
                    period=lookback_period,
                    progress=False
                )['Adj Close']

                if isinstance(data, pd.Series):
                    data = data.to_frame(ticker)

                # Calculate returns
                returns = data.pct_change().dropna()

                st.success(f"✅ Downloaded {len(returns)} days of data")

                # 4. Optimize portfolio
                st.subheader("4️⃣ Portfolio Optimization")

                if optimization_method == "HRP (Hierarchical Risk Parity)":
                    optimizer = HRPOptimizer()
                    weights = optimizer.allocate(returns)
                    method_name = "HRP"

                elif optimization_method == "Mean-Variance":
                    # Use optimize_mean_variance with risk aversion = 5.0 (balanced)
                    weights = optimize_mean_variance(returns, rrr=0.5, allow_short=False)
                    method_name = "Mean-Variance"

                else:  # Equal Weight
                    weights = pd.Series(
                        1.0 / len(all_tickers),
                        index=all_tickers
                    )
                    method_name = "Equal Weight"

                # 5. Display results
                st.subheader(f"5️⃣ Optimal Weights ({method_name})")

                # Prepare weights dataframe
                weights_df = pd.DataFrame({
                    'Ticker': weights.index,
                    'Weight': weights.values,
                    'Weight %': weights.values * 100
                }).sort_values('Weight', ascending=False)

                col1, col2 = st.columns(2)

                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        weights_df,
                        values='Weight',
                        names='Ticker',
                        title='Portfolio Allocation'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        weights_df,
                        x='Ticker',
                        y='Weight %',
                        title='Portfolio Weights (%)',
                        color='Weight %',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Weights table
                st.dataframe(
                    weights_df.style.format({'Weight': '{:.4f}', 'Weight %': '{:.2f}%'}),
                    use_container_width=True
                )

                # 6. Portfolio metrics
                st.subheader("6️⃣ Portfolio Performance Metrics")

                # Calculate portfolio returns
                portfolio_returns = (returns * weights).sum(axis=1)

                # Risk metrics
                risk_metrics = calculate_risk_metrics(portfolio_returns)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Annual Return",
                        f"{risk_metrics.get('annual_return', 0):.2%}"
                    )
                with col2:
                    st.metric(
                        "Annual Volatility",
                        f"{risk_metrics.get('annual_volatility', 0):.2%}"
                    )
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{risk_metrics.get('sharpe_ratio', 0):.2f}"
                    )
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{risk_metrics.get('max_drawdown', 0):.2%}"
                    )

                # 7. Cumulative returns chart
                st.subheader("7️⃣ Cumulative Returns")

                # Calculate cumulative returns
                cumulative_returns = (1 + portfolio_returns).cumprod()
                base_cumulative = (1 + returns[ticker]).cumprod()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    name='Portfolio',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=base_cumulative.index,
                    y=base_cumulative.values,
                    name=f'{ticker} (Base Stock)',
                    line=dict(color='gray', width=1, dash='dash')
                ))

                fig.update_layout(
                    title='Portfolio vs Base Stock Performance',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # 8. Correlation heatmap
                st.subheader("8️⃣ Correlation Matrix")

                corr_matrix = returns.corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10}
                ))

                fig_corr.update_layout(
                    title='Asset Correlation Heatmap',
                    height=600
                )

                st.plotly_chart(fig_corr, use_container_width=True)

                # 9. Export functionality
                st.subheader("9️⃣ Export Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Export weights to CSV
                    csv_weights = weights_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Weights (CSV)",
                        data=csv_weights,
                        file_name=f"portfolio_weights_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    # Export full report
                    report_data = {
                        'Base Stock': [ticker],
                        'Peers': [', '.join(selected_peers)],
                        'Optimization Method': [method_name],
                        'Annual Return': [f"{risk_metrics.get('annual_return', 0):.2%}"],
                        'Annual Volatility': [f"{risk_metrics.get('annual_volatility', 0):.2%}"],
                        'Sharpe Ratio': [f"{risk_metrics.get('sharpe_ratio', 0):.2f}"],
                        'Max Drawdown': [f"{risk_metrics.get('max_drawdown', 0):.2%}"]
                    }
                    report_df = pd.DataFrame(report_data)
                    csv_report = report_df.to_csv(index=False)

                    st.download_button(
                        label="📥 Download Report (CSV)",
                        data=csv_report,
                        file_name=f"portfolio_report_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ Error building portfolio: {str(e)}")
                logger.error(f"Portfolio building error: {e}", exc_info=True)


if __name__ == "__main__":
    show()
