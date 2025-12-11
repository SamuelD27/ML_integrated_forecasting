"""
Portfolio from Securities Galaxy Page
=====================================
Build optimal portfolio from user-selected securities.

Features:
- Multi-security selection
- Multiple optimization methods (HRP, Mean-Variance, Risk Parity)
- Efficient frontier visualization
- Weight allocation charts
- Correlation heatmap
- Expected return & risk metrics
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.hrp_optimizer import HRPOptimizer
from dashboard.utils.theme_terminal import apply_terminal_theme, COLORS, FONTS, FONT_SIZES, SPACING, RADIUS


def show():
    """Main function for portfolio from galaxy page."""
    apply_terminal_theme()

    st.title("Portfolio from Securities Galaxy")
    st.markdown("**Create optimal portfolio from your selected universe of securities**")

    # === 1. SECURITY SELECTION ===
    st.header("1. Select Your Securities")

    # Predefined galaxies
    galaxy_presets = {
        "Tech Giants": "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA",
        "Diversified Growth": "AAPL,MSFT,JPM,JNJ,XOM,PG,V,MA",
        "Dividend Aristocrats": "KO,PEP,JNJ,PG,MCD,WMT,CVX",
        "Mag 7": "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META",
        "Custom": ""
    }

    col1, col2 = st.columns([3, 1])

    with col1:
        preset = st.selectbox(
            "Choose Preset or Create Custom",
            list(galaxy_presets.keys()),
            help="Select a predefined portfolio or create your own"
        )

    with col2:
        num_assets = st.number_input(
            "Number of Assets",
            min_value=2,
            max_value=50,
            value=7,
            help="Portfolio size (2-50 assets)"
        )

    if preset == "Custom":
        tickers_input = st.text_area(
            "Enter Tickers (comma-separated)",
            value="AAPL,MSFT,GOOGL,AMZN,NVDA",
            help="Example: AAPL,MSFT,GOOGL"
        )
    else:
        tickers_input = st.text_area(
            "Securities",
            value=galaxy_presets[preset],
            help="Modify or use as-is"
        )

    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if len(tickers) < 2:
        st.warning("Warning: Please enter at least 2 securities")
        return

    st.info(f"Selected {len(tickers)} securities: {', '.join(tickers)}")

    # === 2. OPTIMIZATION SETTINGS ===
    st.header("2. Optimization Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        optimization_method = st.selectbox(
            "Optimization Method",
            ["HRP (Hierarchical Risk Parity)", "Equal Weight", "Market Cap Weight"],
            help="HRP: Advanced risk-based optimization\nEqual: 1/N weighting\nMarket Cap: Weight by company size"
        )

    with col2:
        lookback_period = st.selectbox(
            "Historical Period",
            ["1y", "2y", "3y", "5y"],
            index=1,
            help="Period for return/risk estimation"
        )

    with col3:
        capital = st.number_input(
            "Total Capital ($)",
            value=100_000,
            min_value=1_000,
            step=10_000,
            help="Total investment amount"
        )

    # Rebalancing settings
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            rebalance_frequency = st.selectbox(
                "Rebalancing Frequency",
                ["Monthly", "Quarterly", "Annually", "Never"],
                index=2
            )

        with col2:
            allow_short = st.checkbox(
                "Allow Short Positions",
                value=False,
                help="Enable long/short portfolio"
            )

    # === 3. RUN OPTIMIZATION ===
    if st.button("Optimize Portfolio", type="primary", use_container_width=True):
        with st.spinner("Fetching data and optimizing portfolio..."):
            try:
                # Fetch historical data
                data = yf.download(
                    tickers,
                    period=lookback_period,
                    progress=False
                )

                if data.empty:
                    st.error("No data retrieved. Please check ticker symbols.")
                    return

                # Get adjusted close prices
                # Check if 'Adj Close' exists, otherwise use 'Close'
                price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

                if len(tickers) == 1:
                    prices = data[price_col].to_frame(tickers[0])
                else:
                    prices = data[price_col]

                # Handle missing data
                prices = prices.dropna(axis=1, how='all')  # Drop tickers with no data
                prices = prices.ffill()  # Forward fill remaining NaNs

                valid_tickers = list(prices.columns)

                if len(valid_tickers) < 2:
                    st.error(f"Insufficient data. Only {len(valid_tickers)} valid ticker(s) found.")
                    return

                if len(valid_tickers) < len(tickers):
                    removed = set(tickers) - set(valid_tickers)
                    st.warning(f"Warning: Removed tickers due to missing data: {', '.join(removed)}")

                # Calculate returns
                returns = prices.pct_change().dropna()

                # === OPTIMIZATION ===
                if optimization_method == "HRP (Hierarchical Risk Parity)":
                    optimizer = HRPOptimizer()
                    weights = optimizer.allocate(returns, min_periods=60)

                elif optimization_method == "Equal Weight":
                    n = len(valid_tickers)
                    weights = pd.Series({ticker: 1.0/n for ticker in valid_tickers})

                elif optimization_method == "Market Cap Weight":
                    # Fetch market caps
                    market_caps = {}
                    for ticker in valid_tickers:
                        try:
                            stock = yf.Ticker(ticker)
                            market_caps[ticker] = stock.info.get('marketCap', 1)
                        except:
                            market_caps[ticker] = 1  # Default if unavailable

                    total_market_cap = sum(market_caps.values())
                    weights = pd.Series({t: mc/total_market_cap for t, mc in market_caps.items()})

                # Ensure weights sum to 1
                weights = weights / weights.sum()

                # === 4. PORTFOLIO METRICS ===
                st.header("3. Optimized Portfolio")

                # Calculate portfolio metrics
                portfolio_return = (returns.mean() * weights).sum() * 252  # Annualized
                portfolio_variance = np.dot(weights.values, np.dot(returns.cov() * 252, weights.values))
                portfolio_std = np.sqrt(portfolio_variance)
                portfolio_sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else 0

                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Expected Return",
                    f"{portfolio_return:.2%}",
                    help="Annualized expected return"
                )

                col2.metric(
                    "Volatility (σ)",
                    f"{portfolio_std:.2%}",
                    help="Annualized standard deviation"
                )

                col3.metric(
                    "Sharpe Ratio",
                    f"{portfolio_sharpe:.2f}",
                    help="Risk-adjusted return"
                )

                col4.metric(
                    "Number of Assets",
                    len(weights),
                    help="Portfolio holdings"
                )

                # === 5. WEIGHT ALLOCATION ===
                st.subheader("Weight Allocation")

                col1, col2 = st.columns(2)

                with col1:
                    # Pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=weights.index,
                        values=weights.values,
                        hole=0.3,
                        textinfo='label+percent',
                        textposition='outside'
                    )])

                    fig_pie.update_layout(
                        title="Portfolio Allocation (%)",
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Bar chart
                    weights_sorted = weights.sort_values(ascending=False)

                    fig_bar = go.Figure(data=[go.Bar(
                        x=weights_sorted.index,
                        y=weights_sorted.values * 100,
                        marker_color='lightblue',
                        text=[f"{w:.1f}%" for w in weights_sorted.values * 100],
                        textposition='outside'
                    )])

                    fig_bar.update_layout(
                        title="Weight Allocation (%)",
                        xaxis_title="Security",
                        yaxis_title="Weight (%)",
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)

                # === 6. ALLOCATION TABLE ===
                st.subheader("Capital Allocation")

                allocation_df = pd.DataFrame({
                    'Ticker': weights.index,
                    'Weight (%)': weights.values * 100,
                    'Allocation ($)': weights.values * capital,
                    'Shares (approx)': [
                        int(weights[ticker] * capital / prices[ticker].iloc[-1])
                        for ticker in weights.index
                    ]
                })

                # Add current prices
                allocation_df['Current Price'] = [
                    prices[ticker].iloc[-1] for ticker in weights.index
                ]

                # Reorder columns
                allocation_df = allocation_df[[
                    'Ticker', 'Weight (%)', 'Allocation ($)', 'Current Price', 'Shares (approx)'
                ]]

                # Sort by weight
                allocation_df = allocation_df.sort_values('Weight (%)', ascending=False).reset_index(drop=True)

                # Format for display
                allocation_df_display = allocation_df.copy()
                allocation_df_display['Weight (%)'] = allocation_df_display['Weight (%)'].apply(lambda x: f"{x:.2f}%")
                allocation_df_display['Allocation ($)'] = allocation_df_display['Allocation ($)'].apply(lambda x: f"${x:,.2f}")
                allocation_df_display['Current Price'] = allocation_df_display['Current Price'].apply(lambda x: f"${x:.2f}")

                st.dataframe(allocation_df_display, use_container_width=True, hide_index=True)

                # Export button
                csv = allocation_df.to_csv(index=False)
                st.download_button(
                    label="Download Allocation (CSV)",
                    data=csv,
                    file_name=f"portfolio_allocation_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

                # === 7. CORRELATION HEATMAP ===
                st.subheader("Correlation Matrix")

                corr = returns.corr()

                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=corr.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))

                fig_heatmap.update_layout(
                    title="Asset Correlation Heatmap",
                    height=600,
                    xaxis=dict(tickangle=-45)
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

                # === 8. RISK CONTRIBUTION ===
                st.subheader("Risk Contribution")

                # Calculate marginal contribution to risk
                cov_matrix = returns.cov() * 252
                portfolio_risk_contribution = weights.values * (cov_matrix @ weights.values) / portfolio_std

                risk_contrib_df = pd.DataFrame({
                    'Ticker': weights.index,
                    'Weight (%)': weights.values * 100,
                    'Risk Contribution (%)': portfolio_risk_contribution * 100
                }).sort_values('Risk Contribution (%)', ascending=False)

                fig_risk = go.Figure(data=[go.Bar(
                    x=risk_contrib_df['Ticker'],
                    y=risk_contrib_df['Risk Contribution (%)'],
                    marker_color='salmon',
                    text=risk_contrib_df['Risk Contribution (%)'].round(1),
                    textposition='outside'
                )])

                fig_risk.update_layout(
                    title="Risk Contribution by Asset",
                    xaxis_title="Security",
                    yaxis_title="Risk Contribution (%)",
                    height=400
                )

                st.plotly_chart(fig_risk, use_container_width=True)

                # === 9. HISTORICAL PERFORMANCE ===
                st.subheader("Historical Performance")

                # Calculate portfolio historical returns
                portfolio_daily_returns = (returns * weights).sum(axis=1)
                portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()

                # Individual asset performance
                individual_cumulative = (1 + returns).cumprod()

                # Plot
                fig_perf = go.Figure()

                # Portfolio performance (bold line)
                fig_perf.add_trace(go.Scatter(
                    x=portfolio_cumulative.index,
                    y=portfolio_cumulative.values,
                    mode='lines',
                    name='Optimized Portfolio',
                    line=dict(color='blue', width=3)
                ))

                # Individual assets (thin lines)
                for ticker in valid_tickers[:10]:  # Limit to 10 for readability
                    fig_perf.add_trace(go.Scatter(
                        x=individual_cumulative.index,
                        y=individual_cumulative[ticker].values,
                        mode='lines',
                        name=ticker,
                        line=dict(width=1),
                        opacity=0.5
                    ))

                fig_perf.update_layout(
                    title=f"Cumulative Performance ({lookback_period})",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (1 = start)",
                    height=500,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_perf, use_container_width=True)

                # === 10. SUMMARY ===
                st.header("4. Portfolio Summary")

                summary_text = f"""
                ### Optimized {optimization_method} Portfolio

                **Composition**: {len(weights)} securities with total capital of **${capital:,.0f}**

                **Expected Performance**:
                - Annual Return: **{portfolio_return:.2%}**
                - Annual Volatility: **{portfolio_std:.2%}**
                - Sharpe Ratio: **{portfolio_sharpe:.2f}**

                **Top 3 Holdings**:
                """

                for i, (ticker, weight) in enumerate(weights.nlargest(3).items(), 1):
                    summary_text += f"\n{i}. **{ticker}**: {weight*100:.1f}% (${weight*capital:,.0f})"

                summary_text += f"\n\n**Rebalancing**: {rebalance_frequency}"

                st.markdown(summary_text)

                st.success("Portfolio optimization complete! Review allocation and metrics above.")

                # Disclaimer
                st.caption("Warning: Disclaimer: Past performance does not guarantee future results. This is for informational purposes only, not financial advice.")

            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")
                st.exception(e)


if __name__ == "__main__":
    show()
