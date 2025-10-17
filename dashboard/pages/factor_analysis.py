"""
Factor Analysis
===============
Fama-French factor analysis for stocks and portfolios.

Features:
- 3-factor and 5-factor models
- Factor exposure analysis
- Alpha and attribution
- Rolling factor betas
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List
import logging

# Import backend modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from portfolio.factor_models import FamaFrenchFactorModel
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


def show():
    """Main function for factor analysis page."""
    st.title("📈 Factor Analysis (Fama-French)")
    st.markdown("""
    Analyze factor exposures using Fama-French models.
    Understand what drives your returns: market, size, value, profitability, or investment.
    """)

    # Sidebar inputs
    st.sidebar.header("Configuration")

    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Single Stock", "Multiple Stocks", "Portfolio"]
    )

    model_type = st.sidebar.selectbox(
        "Factor Model",
        ["3-Factor (Market, Size, Value)", "5-Factor (+ Profitability, Investment)"]
    )

    lookback_period = st.sidebar.selectbox(
        "Lookback Period",
        ["1y", "2y", "3y", "5y"],
        index=2
    )

    # Input based on analysis type
    tickers = []
    weights = None

    if analysis_type == "Single Stock":
        ticker_input = st.sidebar.text_input("Enter Ticker", "AAPL")
        if ticker_input:
            tickers = [ticker_input.upper()]

    elif analysis_type == "Multiple Stocks":
        ticker_input = st.sidebar.text_area(
            "Enter Tickers (one per line)",
            "AAPL\nMSFT\nGOOGL"
        )
        if ticker_input:
            tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

    else:  # Portfolio
        st.sidebar.subheader("Portfolio Holdings")
        n_holdings = st.sidebar.number_input("Number of Holdings", 2, 10, 3)

        ticker_list = []
        weight_list = []

        for i in range(n_holdings):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                ticker = st.text_input(f"Ticker {i+1}", "", key=f"ticker_{i}")
            with col2:
                weight = st.number_input(f"Weight %", 0.0, 100.0, 0.0, key=f"weight_{i}")

            if ticker:
                ticker_list.append(ticker.upper())
                weight_list.append(weight / 100.0)

        if ticker_list and abs(sum(weight_list) - 1.0) < 0.01:
            tickers = ticker_list
            weights = np.array(weight_list)

    if st.sidebar.button("Analyze Factors", type="primary") and tickers:
        with st.spinner("Analyzing factor exposures..."):
            try:
                # Initialize Fama-French analyzer
                model_name = '5-factor' if "5-Factor" in model_type else '3-factor'
                ff_analyzer = FamaFrenchFactorModel(model=model_name)

                # 1. Fetch stock data
                st.subheader("1️⃣ Downloading Data")

                raw_data = yf.download(tickers, period=lookback_period, progress=False)

                # Handle Adj Close vs Close
                if 'Adj Close' in raw_data.columns:
                    data = raw_data['Adj Close']
                elif 'Close' in raw_data.columns:
                    data = raw_data['Close']
                else:
                    # Single ticker case - raw_data might already be the price series
                    data = raw_data

                if isinstance(data, pd.Series):
                    data = data.to_frame(tickers[0])

                returns = data.pct_change().dropna()

                st.success(f"✅ Downloaded {len(returns)} days of data for {len(tickers)} asset(s)")

                # 2. Calculate portfolio returns if applicable
                if analysis_type == "Portfolio" and weights is not None:
                    portfolio_returns = (returns * weights).sum(axis=1)
                    analysis_returns = pd.DataFrame({'Portfolio': portfolio_returns})
                    st.info(f"Analyzing portfolio with {len(tickers)} holdings")
                else:
                    analysis_returns = returns

                # 3. Run factor regression for each asset
                st.subheader("2️⃣ Factor Regression Results")

                n_factors = 5 if "5-Factor" in model_type else 3

                all_results = {}

                for column in analysis_returns.columns:
                    asset_returns = analysis_returns[column]

                    # Run regression
                    result = ff_analyzer.regress_returns(
                        ticker=column,
                        returns=asset_returns
                    )

                    all_results[column] = result

                # 4. Display results for each asset
                for asset_name, result in all_results.items():
                    with st.expander(f"📊 {asset_name} - Factor Analysis", expanded=len(all_results)==1):

                        # Alpha and R-squared
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            alpha = result['alpha']
                            alpha_annual = alpha * 252  # Annualize
                            st.metric(
                                "Alpha (Annual)",
                                f"{alpha_annual:.2%}",
                                delta="Outperformance" if alpha_annual > 0 else "Underperformance"
                            )

                        with col2:
                            r_squared = result['r_squared']
                            st.metric("R-Squared", f"{r_squared:.2%}")

                        with col3:
                            p_value = result.get('alpha_pvalue', 1.0)
                            is_significant = p_value < 0.05
                            st.metric(
                                "Alpha Significance",
                                "Yes ✅" if is_significant else "No ❌",
                                delta=f"p-value: {p_value:.4f}"
                            )

                        st.markdown(f"""
                        **Interpretation**:
                        - **Alpha**: {alpha_annual:.2%} annual excess return (risk-adjusted)
                        - **R-Squared**: {r_squared:.1%} of returns explained by factors
                        - **Significance**: Alpha is {"statistically significant" if is_significant else "not statistically significant"}
                        """)

                        # Factor loadings (betas)
                        st.markdown("**Factor Exposures (Betas)**")

                        factor_names = {
                            'mkt_rf': 'Market',
                            'smb': 'Size (Small - Big)',
                            'hml': 'Value (High - Low)',
                            'rmw': 'Profitability (Robust - Weak)',
                            'cma': 'Investment (Conservative - Aggressive)'
                        }

                        betas_data = []
                        for factor, display_name in factor_names.items():
                            if factor in result:
                                beta = result[factor]
                                pvalue = result.get(f'{factor}_pvalue', 1.0)
                                significant = "✅" if pvalue < 0.05 else "❌"

                                betas_data.append({
                                    'Factor': display_name,
                                    'Beta': beta,
                                    'Significant': significant,
                                    'P-Value': pvalue
                                })

                        betas_df = pd.DataFrame(betas_data)

                        if not betas_df.empty:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.dataframe(
                                    betas_df.style.format({
                                        'Beta': '{:.3f}',
                                        'P-Value': '{:.4f}'
                                    }),
                                    use_container_width=True
                                )

                            with col2:
                                # Beta chart
                                fig_beta = px.bar(
                                    betas_df,
                                    x='Factor',
                                    y='Beta',
                                    title=f'{asset_name} Factor Betas',
                                    color='Beta',
                                    color_continuous_scale='RdBu',
                                    color_continuous_midpoint=0
                                )
                                st.plotly_chart(fig_beta, use_container_width=True)
                        else:
                            st.warning(f"No factor data available for {asset_name}. The regression may have failed.")
                            st.info(f"Available result keys: {list(result.keys())}")

                        # Interpretation guide
                        st.markdown("""
                        **Beta Interpretation**:
                        - **Market > 1**: More volatile than market
                        - **Size > 0**: Behaves like small-cap stocks
                        - **Value > 0**: Behaves like value stocks (high book-to-market)
                        - **Profitability > 0**: Exposed to profitable companies
                        - **Investment < 0**: Exposed to conservative investment (low growth)
                        """)

                # 5. Comparison across assets (if multiple)
                if len(all_results) > 1:
                    st.subheader("3️⃣ Cross-Asset Comparison")

                    # Alpha comparison
                    alpha_data = []
                    for asset_name, result in all_results.items():
                        alpha_data.append({
                            'Asset': asset_name,
                            'Alpha (Annual)': result['alpha'] * 252,
                            'R-Squared': result['r_squared']
                        })

                    alpha_df = pd.DataFrame(alpha_data).sort_values('Alpha (Annual)', ascending=False)

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_alpha = px.bar(
                            alpha_df,
                            x='Asset',
                            y='Alpha (Annual)',
                            title='Alpha Comparison',
                            color='Alpha (Annual)',
                            color_continuous_scale='RdYlGn',
                            color_continuous_midpoint=0
                        )
                        st.plotly_chart(fig_alpha, use_container_width=True)

                    with col2:
                        fig_r2 = px.bar(
                            alpha_df,
                            x='Asset',
                            y='R-Squared',
                            title='Model Fit (R-Squared)',
                            color='R-Squared',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_r2, use_container_width=True)

                    # Beta heatmap
                    st.markdown("**Factor Beta Heatmap**")

                    beta_matrix_data = []
                    for asset_name, result in all_results.items():
                        row = {'Asset': asset_name}
                        for factor in ['mkt_rf', 'smb', 'hml', 'rmw', 'cma']:
                            if factor in result:
                                row[factor_names[factor]] = result[factor]
                        beta_matrix_data.append(row)

                    beta_matrix_df = pd.DataFrame(beta_matrix_data).set_index('Asset')

                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=beta_matrix_df.values,
                        x=beta_matrix_df.columns,
                        y=beta_matrix_df.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=beta_matrix_df.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 10}
                    ))

                    fig_heatmap.update_layout(
                        title='Factor Beta Heatmap',
                        height=400
                    )

                    st.plotly_chart(fig_heatmap, use_container_width=True)

                # 6. Export results
                st.subheader("4️⃣ Export Results")

                # Prepare export data
                export_data = []
                for asset_name, result in all_results.items():
                    row = {
                        'Asset': asset_name,
                        'Alpha (Daily)': result['alpha'],
                        'Alpha (Annual)': result['alpha'] * 252,
                        'R-Squared': result['r_squared'],
                        'Alpha P-Value': result.get('alpha_pvalue', None)
                    }

                    for factor in ['mkt_rf', 'smb', 'hml', 'rmw', 'cma']:
                        if factor in result:
                            row[f'{factor}_beta'] = result[factor]
                            row[f'{factor}_pvalue'] = result.get(f'{factor}_pvalue', None)

                    export_data.append(row)

                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download Factor Analysis (CSV)",
                    data=csv,
                    file_name=f"factor_analysis_{analysis_type}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error analyzing factors: {str(e)}")
                logger.error(f"Factor analysis error: {e}", exc_info=True)

    elif not tickers:
        st.info("👆 Please enter ticker symbol(s) in the sidebar to begin analysis.")


if __name__ == "__main__":
    show()
