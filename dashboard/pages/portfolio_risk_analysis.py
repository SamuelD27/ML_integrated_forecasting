"""
Portfolio Risk Analysis
=======================
Comprehensive risk analysis for existing portfolios.

Features:
- VaR and CVaR analysis
- Risk decomposition
- Stress testing
- Drawdown analysis
- Risk contribution by asset
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List
import logging
from datetime import datetime

# Import backend modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

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


def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk."""
    return -np.percentile(returns, (1 - confidence_level) * 100)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    var = calculate_var(returns, confidence_level)
    return -returns[returns <= -var].mean()


def show():
    """Main function for portfolio risk analysis page."""
    st.title("⚠️ Portfolio Risk Analysis")
    st.markdown("""
    Comprehensive risk assessment of your portfolio.
    Analyze VaR, CVaR, risk contribution, and stress scenarios.
    """)

    # Sidebar inputs
    st.sidebar.header("Portfolio Configuration")

    # Portfolio input method
    input_method = st.sidebar.radio(
        "Input Method",
        ["Manual Entry", "Upload CSV"]
    )

    portfolio_data = None

    if input_method == "Manual Entry":
        st.sidebar.subheader("Enter Holdings")
        n_holdings = st.sidebar.number_input("Number of Holdings", 2, 20, 5)

        tickers = []
        weights = []

        for i in range(n_holdings):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                ticker = st.text_input(f"Ticker {i+1}", f"", key=f"ticker_{i}")
            with col2:
                weight = st.number_input(f"Weight {i+1} (%)", 0.0, 100.0, 0.0, key=f"weight_{i}")

            if ticker:
                tickers.append(ticker.upper())
                weights.append(weight / 100.0)

        if tickers and abs(sum(weights) - 1.0) < 0.01:
            portfolio_data = pd.DataFrame({
                'Ticker': tickers,
                'Weight': weights
            })

    else:  # Upload CSV
        uploaded_file = st.sidebar.file_uploader(
            "Upload Portfolio CSV",
            type=['csv'],
            help="CSV with columns: Ticker, Weight (as decimal, e.g., 0.25 for 25%)"
        )

        if uploaded_file is not None:
            portfolio_data = pd.read_csv(uploaded_file)

    # Analysis parameters
    st.sidebar.header("Analysis Parameters")
    lookback_period = st.sidebar.selectbox("Lookback Period", ["1y", "2y", "3y", "5y"], index=1)
    confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95) / 100.0

    if st.sidebar.button("Analyze Risk", type="primary") and portfolio_data is not None:
        with st.spinner("Analyzing portfolio risk..."):
            try:
                # Validate weights
                total_weight = portfolio_data['Weight'].sum()
                if abs(total_weight - 1.0) > 0.01:
                    st.error(f"❌ Weights must sum to 100% (current: {total_weight*100:.1f}%)")
                    return

                # 1. Fetch historical data
                st.subheader("1️⃣ Portfolio Holdings")

                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(
                        portfolio_data.style.format({'Weight': '{:.2%}'}),
                        use_container_width=True
                    )

                with col2:
                    fig_pie = px.pie(
                        portfolio_data,
                        values='Weight',
                        names='Ticker',
                        title='Portfolio Allocation'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Download price data
                st.subheader("2️⃣ Downloading Historical Data")
                tickers = portfolio_data['Ticker'].tolist()
                weights = portfolio_data['Weight'].values

                raw_data = yf.download(tickers, period=lookback_period, progress=False)

                # Handle Adj Close vs Close
                if 'Adj Close' in raw_data.columns:
                    data = raw_data['Adj Close']
                elif 'Close' in raw_data.columns:
                    data = raw_data['Close']
                else:
                    data = raw_data

                if isinstance(data, pd.Series):
                    data = data.to_frame(tickers[0])

                returns = data.pct_change().dropna()
                st.success(f"✅ Downloaded {len(returns)} days of data")

                # Calculate portfolio returns
                portfolio_returns = (returns * weights).sum(axis=1)

                # 2. Basic risk metrics
                st.subheader("3️⃣ Risk Metrics")

                risk_metrics = calculate_risk_metrics(portfolio_returns)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Annual Volatility", f"{risk_metrics.get('annual_volatility', 0):.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
                with col3:
                    st.metric("Sortino Ratio", f"{risk_metrics.get('sortino_ratio', 0):.2f}")
                with col4:
                    st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}")

                # 3. VaR and CVaR
                st.subheader("4️⃣ Value at Risk (VaR) Analysis")

                var_1d = calculate_var(portfolio_returns, confidence_level=confidence_level)
                cvar_1d = calculate_cvar(portfolio_returns, confidence_level=confidence_level)

                # Scale to different time horizons
                var_1w = var_1d * np.sqrt(5)
                var_1m = var_1d * np.sqrt(21)

                cvar_1w = cvar_1d * np.sqrt(5)
                cvar_1m = cvar_1d * np.sqrt(21)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        f"1-Day VaR ({confidence_level:.0%})",
                        f"{var_1d:.2%}",
                        delta=f"CVaR: {cvar_1d:.2%}",
                        delta_color="inverse"
                    )
                with col2:
                    st.metric(
                        f"1-Week VaR ({confidence_level:.0%})",
                        f"{var_1w:.2%}",
                        delta=f"CVaR: {cvar_1w:.2%}",
                        delta_color="inverse"
                    )
                with col3:
                    st.metric(
                        f"1-Month VaR ({confidence_level:.0%})",
                        f"{var_1m:.2%}",
                        delta=f"CVaR: {cvar_1m:.2%}",
                        delta_color="inverse"
                    )

                st.info(f"""
                **Interpretation**: With {confidence_level:.0%} confidence, you will not lose more than:
                - **{var_1d:.2%}** in one day
                - **{var_1w:.2%}** in one week
                - **{var_1m:.2%}** in one month

                If losses exceed VaR, the expected loss is **CVaR** (Conditional VaR).
                """)

                # 4. Risk contribution by asset
                st.subheader("5️⃣ Risk Contribution by Asset")

                # Calculate covariance matrix
                cov_matrix = returns.cov() * 252  # Annualized

                # Portfolio variance
                portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_vol = np.sqrt(portfolio_var)

                # Marginal contribution to risk
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol

                # Component contribution to risk
                component_contrib = weights * marginal_contrib

                # Percentage contribution
                pct_contrib = component_contrib / portfolio_vol

                risk_contrib_df = pd.DataFrame({
                    'Ticker': tickers,
                    'Weight': weights,
                    'Risk Contribution': pct_contrib,
                    'Risk Contribution %': pct_contrib * 100
                }).sort_values('Risk Contribution', ascending=False)

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(
                        risk_contrib_df.style.format({
                            'Weight': '{:.2%}',
                            'Risk Contribution': '{:.2%}',
                            'Risk Contribution %': '{:.2f}%'
                        }),
                        use_container_width=True
                    )

                with col2:
                    fig_risk = px.bar(
                        risk_contrib_df,
                        x='Ticker',
                        y='Risk Contribution %',
                        title='Risk Contribution by Asset',
                        color='Risk Contribution %',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                # 5. Drawdown analysis
                st.subheader("6️⃣ Drawdown Analysis")

                # Calculate cumulative returns
                cumulative_returns = (1 + portfolio_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max

                fig_dd = go.Figure()

                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values * 100,
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='red')
                ))

                fig_dd.update_layout(
                    title='Portfolio Drawdown Over Time',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    hovermode='x unified'
                )

                st.plotly_chart(fig_dd, use_container_width=True)

                # Drawdown statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Max Drawdown", f"{drawdown.min():.2%}")
                with col2:
                    max_dd_date = drawdown.idxmin()
                    st.metric("Max DD Date", max_dd_date.strftime('%Y-%m-%d'))
                with col3:
                    # Average drawdown (when in drawdown)
                    avg_dd = drawdown[drawdown < 0].mean()
                    st.metric("Avg Drawdown", f"{avg_dd:.2%}")

                # 6. Stress testing
                st.subheader("7️⃣ Stress Testing")

                # Historical stress scenarios
                st.markdown("**Historical Stress Scenarios**")

                # Define stress periods (example dates)
                stress_scenarios = {
                    'COVID-19 Crash': ('2020-02-19', '2020-03-23'),
                    '2022 Bear Market': ('2022-01-01', '2022-10-31'),
                }

                stress_results = []

                for scenario_name, (start, end) in stress_scenarios.items():
                    try:
                        scenario_returns = portfolio_returns.loc[start:end]
                        if len(scenario_returns) > 0:
                            cumulative_return = (1 + scenario_returns).prod() - 1
                            max_dd = ((1 + scenario_returns).cumprod() /
                                     (1 + scenario_returns).cumprod().expanding().max() - 1).min()

                            stress_results.append({
                                'Scenario': scenario_name,
                                'Return': cumulative_return,
                                'Max Drawdown': max_dd,
                                'Days': len(scenario_returns)
                            })
                    except:
                        pass

                if stress_results:
                    stress_df = pd.DataFrame(stress_results)
                    st.dataframe(
                        stress_df.style.format({
                            'Return': '{:.2%}',
                            'Max Drawdown': '{:.2%}'
                        }),
                        use_container_width=True
                    )

                # 7. Return distribution
                st.subheader("8️⃣ Return Distribution")

                fig_dist = go.Figure()

                # Histogram
                fig_dist.add_trace(go.Histogram(
                    x=portfolio_returns * 100,
                    name='Returns',
                    nbinsx=50,
                    opacity=0.7
                ))

                # Add VaR line
                fig_dist.add_vline(
                    x=-var_1d * 100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"VaR ({confidence_level:.0%})"
                )

                fig_dist.update_layout(
                    title='Daily Return Distribution',
                    xaxis_title='Return (%)',
                    yaxis_title='Frequency',
                    showlegend=True
                )

                st.plotly_chart(fig_dist, use_container_width=True)

                # Distribution statistics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Skewness", f"{stats.skew(portfolio_returns):.2f}")
                with col2:
                    st.metric("Kurtosis", f"{stats.kurtosis(portfolio_returns):.2f}")
                with col3:
                    positive_days = (portfolio_returns > 0).sum() / len(portfolio_returns)
                    st.metric("Positive Days", f"{positive_days:.1%}")
                with col4:
                    st.metric("Std Dev", f"{portfolio_returns.std():.2%}")

            except Exception as e:
                st.error(f"❌ Error analyzing portfolio: {str(e)}")
                logger.error(f"Portfolio risk analysis error: {e}", exc_info=True)

    elif portfolio_data is None:
        st.info("👆 Please enter your portfolio holdings in the sidebar to begin analysis.")


if __name__ == "__main__":
    show()
