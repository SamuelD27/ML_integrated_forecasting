"""
Single Stock Analysis Page
==========================
Complete quantitative analysis with long/short recommendation.

Features:
- DCF Valuation
- Technical indicators
- Risk metrics (Sharpe, Sortino, VaR, CVaR)
- Fama-French factor exposure
- Long/Short trading decision
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.security_valuation import DCFInputs, DCFValuation, SecurityType
from portfolio.factor_models import FamaFrenchFactorModel


def show():
    """Main function for single stock analysis page."""
    st.title("🔍 Single Stock Analysis")
    st.markdown("**Complete quantitative analysis with institutional-grade metrics**")

    # Input section
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input("Ticker Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, MSFT, TSLA)")

        with col2:
            period = st.selectbox(
                "Analysis Period",
                ["1y", "2y", "3y", "5y"],
                index=1,
                help="Historical period for analysis"
            )

        with col3:
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                value=5.0,
                min_value=0.0,
                max_value=20.0,
                help="Used for Sharpe ratio and CAPM"
            ) / 100

    # Run analysis button
    if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Fetch data
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                info = stock.info

                if hist.empty:
                    st.error(f"No data found for ticker {ticker}. Please check the symbol.")
                    return

                current_price = hist['Close'].iloc[-1]
                returns = hist['Close'].pct_change().dropna()

                # === 1. VALUATION ===
                st.header("1️⃣ Valuation Analysis")

                fcf = info.get('freeCashflow', 0)

                if fcf > 0:
                    # DCF Valuation
                    try:
                        dcf_inputs = DCFInputs(
                            fcf_current=fcf,
                            growth_rate_stage1=0.15,  # 15% high growth
                            growth_rate_stage2=0.03,  # 3% stable growth
                            wacc=0.10,  # 10% WACC
                            shares_outstanding=info.get('sharesOutstanding', 1)
                        )
                        dcf = DCFValuation(dcf_inputs)
                        valuation = dcf.calculate_intrinsic_value()

                        intrinsic = valuation['intrinsic_value']
                        upside = (intrinsic - current_price) / current_price

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Current Price",
                            f"${current_price:.2f}",
                            help="Latest closing price"
                        )

                        col2.metric(
                            "Intrinsic Value (DCF)",
                            f"${intrinsic:.2f}",
                            help="Two-stage DCF valuation"
                        )

                        col3.metric(
                            "Upside/Downside",
                            f"{upside:+.1%}",
                            delta=f"{upside:.1%}",
                            help="Potential gain/loss to intrinsic value"
                        )

                        # DCF breakdown
                        with st.expander("📊 View DCF Breakdown"):
                            breakdown_df = pd.DataFrame({
                                'Component': ['Stage 1 PV (High Growth)', 'Terminal PV (Stable Growth)', 'Enterprise Value', 'Per Share Value'],
                                'Value ($M)': [
                                    valuation['stage1_pv'] / 1_000_000,
                                    valuation['terminal_pv'] / 1_000_000,
                                    valuation['enterprise_value'] / 1_000_000,
                                    intrinsic
                                ]
                            })
                            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

                    except Exception as e:
                        st.warning(f"DCF calculation error: {e}")
                        intrinsic = current_price
                        upside = 0
                else:
                    st.info("ℹ️ DCF not available (no Free Cash Flow data). Using market price as reference.")
                    intrinsic = current_price
                    upside = 0

                # === 2. LONG/SHORT DECISION ===
                st.header("2️⃣ Trading Recommendation")

                # Decision logic
                if upside > 0.20:
                    decision = "STRONG LONG 🚀"
                    decision_color = "green"
                    reasoning = "Significantly undervalued with >20% upside potential"
                    confidence = "High"
                elif upside > 0.10:
                    decision = "LONG 📈"
                    decision_color = "green"
                    reasoning = "Undervalued with moderate upside potential"
                    confidence = "Medium"
                elif upside < -0.20:
                    decision = "STRONG SHORT 📉"
                    decision_color = "red"
                    reasoning = "Significantly overvalued with >20% downside risk"
                    confidence = "High"
                elif upside < -0.10:
                    decision = "SHORT 🔻"
                    decision_color = "red"
                    reasoning = "Overvalued with moderate downside risk"
                    confidence = "Medium"
                else:
                    decision = "NEUTRAL ⚪"
                    decision_color = "gray"
                    reasoning = "Fairly valued, within ±10% of intrinsic value"
                    confidence = "Low"

                # Display decision
                if decision_color == "green":
                    st.success(f"**{decision}**")
                elif decision_color == "red":
                    st.error(f"**{decision}**")
                else:
                    st.warning(f"**{decision}**")

                st.info(f"**Reasoning**: {reasoning}")

                # Decision summary table
                decision_df = pd.DataFrame({
                    'Metric': ['Decision', 'Confidence', 'Target Price', 'Current Price', 'Potential Move', 'Recommendation'],
                    'Value': [
                        decision,
                        confidence,
                        f"${intrinsic:.2f}",
                        f"${current_price:.2f}",
                        f"{upside:+.1%}",
                        "Enter Position" if confidence == "High" else "Monitor"
                    ]
                })
                st.dataframe(decision_df, use_container_width=True, hide_index=True)

                # === 3. RISK METRICS ===
                st.header("3️⃣ Risk Analysis")

                # Calculate risk metrics
                sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

                downside_returns = returns[returns < 0]
                sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                max_dd = drawdown.min()

                # Volatility
                volatility = returns.std() * np.sqrt(252)

                # VaR and CVaR
                var_95 = -np.percentile(returns, 5)
                var_threshold = np.percentile(returns, 5)
                cvar_95 = -returns[returns <= var_threshold].mean()

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}",
                    help="Risk-adjusted return (>1 is good, >2 is excellent)"
                )

                col2.metric(
                    "Sortino Ratio",
                    f"{sortino:.2f}",
                    help="Downside risk-adjusted return"
                )

                col3.metric(
                    "Max Drawdown",
                    f"{max_dd:.1%}",
                    help="Largest peak-to-trough decline",
                    delta=None,
                    delta_color="inverse"
                )

                col4.metric(
                    "Volatility (σ)",
                    f"{volatility:.1%}",
                    help="Annualized standard deviation"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "VaR (95%)",
                    f"{var_95:.2%}",
                    help="Maximum expected loss (95% confidence)"
                )

                col2.metric(
                    "CVaR (95%)",
                    f"{cvar_95:.2%}",
                    help="Expected loss when VaR is exceeded"
                )

                col3.metric(
                    "Beta vs SPY",
                    f"{info.get('beta', 1.0):.2f}",
                    help="Sensitivity to market movements"
                )

                col4.metric(
                    "52-Week High/Low",
                    f"${info.get('fiftyTwoWeekLow', 0):.0f} - ${info.get('fiftyTwoWeekHigh', 0):.0f}"
                )

                # === 4. FACTOR ANALYSIS ===
                st.header("4️⃣ Fama-French Factor Exposure")

                try:
                    ff = FamaFrenchFactorModel(model='5-factor')
                    ff_results = ff.regress_returns(ticker, returns, frequency='daily')

                    # Display alpha and model fit
                    col1, col2, col3 = st.columns(3)

                    alpha_color = "normal" if ff_results['alpha_pvalue'] > 0.05 else "inverse"

                    col1.metric(
                        "Alpha (annualized)",
                        f"{ff_results['alpha']:.2%}",
                        help=f"Skill-based return (p-value: {ff_results['alpha_pvalue']:.4f})"
                    )

                    col2.metric(
                        "Market Beta",
                        f"{ff_results['beta_MKT']:.2f}",
                        help="Sensitivity to market factor"
                    )

                    col3.metric(
                        "R-squared",
                        f"{ff_results['r_squared']:.1%}",
                        help="Variance explained by factors"
                    )

                    # Factor betas chart
                    betas_df = pd.DataFrame({
                        'Factor': ['Market\n(Mkt-RF)', 'Size\n(SMB)', 'Value\n(HML)', 'Profitability\n(RMW)', 'Investment\n(CMA)'],
                        'Beta': [
                            ff_results['beta_MKT'],
                            ff_results['beta_SMB'],
                            ff_results['beta_HML'],
                            ff_results['beta_RMW'],
                            ff_results['beta_CMA']
                        ]
                    })

                    # Color bars based on magnitude
                    colors = ['green' if b > 0 else 'red' for b in betas_df['Beta']]

                    fig = go.Figure(data=go.Bar(
                        x=betas_df['Factor'],
                        y=betas_df['Beta'],
                        marker_color=colors,
                        text=betas_df['Beta'].round(2),
                        textposition='outside'
                    ))

                    fig.update_layout(
                        title="Factor Exposures (5-Factor Model)",
                        xaxis_title="Factor",
                        yaxis_title="Beta",
                        height=400,
                        showlegend=False
                    )

                    fig.add_hline(y=0, line_dash="dash", line_color="gray")

                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    with st.expander("📖 Factor Interpretation"):
                        st.markdown(f"""
                        **Alpha**: {ff_results['alpha']:.2%} - {'Significant' if ff_results['significant_alpha'] else 'Not significant'} skill-based return

                        **Market Beta**: {ff_results['beta_MKT']:.2f} - {'More' if ff_results['beta_MKT'] > 1 else 'Less'} volatile than market

                        **Size (SMB)**: {ff_results['beta_SMB']:.2f} - {'Small' if ff_results['beta_SMB'] > 0 else 'Large'} cap tilt

                        **Value (HML)**: {ff_results['beta_HML']:.2f} - {'Value' if ff_results['beta_HML'] > 0 else 'Growth'} oriented

                        **Profitability (RMW)**: {ff_results['beta_RMW']:.2f} - {'Robust' if ff_results['beta_RMW'] > 0 else 'Weak'} profitability

                        **Investment (CMA)**: {ff_results['beta_CMA']:.2f} - {'Conservative' if ff_results['beta_CMA'] > 0 else 'Aggressive'} investment
                        """)

                except Exception as e:
                    st.warning(f"⚠️ Factor analysis unavailable: {e}")
                    st.info("This may occur if pandas-datareader is not installed or Fama-French data is unavailable.")

                # === 5. PRICE CHART ===
                st.header("5️⃣ Price History & Technical Indicators")

                # Create subplot figure
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"{ticker} Price Chart", "Volume")
                )

                # Price chart
                fig.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )

                # Add intrinsic value line if available
                if intrinsic > 0 and abs(upside) > 0.01:
                    fig.add_hline(
                        y=intrinsic,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Intrinsic: ${intrinsic:.2f}",
                        row=1, col=1
                    )

                # Add 50-day and 200-day moving averages
                if len(hist) >= 200:
                    ma_50 = hist['Close'].rolling(50).mean()
                    ma_200 = hist['Close'].rolling(200).mean()

                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=ma_50,
                            mode='lines',
                            name='MA 50',
                            line=dict(color='orange', width=1, dash='dot')
                        ),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=ma_200,
                            mode='lines',
                            name='MA 200',
                            line=dict(color='red', width=1, dash='dot')
                        ),
                        row=1, col=1
                    )

                # Volume chart
                fig.add_trace(
                    go.Bar(
                        x=hist.index,
                        y=hist['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ),
                    row=2, col=1
                )

                fig.update_layout(
                    title=f"{ticker} Technical Analysis ({period})",
                    xaxis2_title="Date",
                    yaxis_title="Price ($)",
                    yaxis2_title="Volume",
                    height=700,
                    hovermode='x unified',
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # === 6. SUMMARY ===
                st.header("6️⃣ Executive Summary")

                summary_text = f"""
                ### {ticker} - {decision}

                **Valuation**: Current price of **${current_price:.2f}** vs intrinsic value of **${intrinsic:.2f}** suggests **{upside:+.1%}** potential move.

                **Risk Profile**: Sharpe ratio of **{sharpe:.2f}** with **{volatility:.1%}** annualized volatility. Maximum drawdown observed: **{max_dd:.1%}**.

                **Factor Exposure**: {'Significant' if ff_results.get('significant_alpha', False) else 'No significant'} alpha detected. Market beta of **{ff_results.get('beta_MKT', info.get('beta', 1.0)):.2f}**.

                **Recommendation**: {reasoning}
                """

                st.markdown(summary_text)

                st.success("✅ Analysis complete! Review the recommendation and metrics above before making investment decisions.")

                # Disclaimer
                st.caption("⚠️ Disclaimer: This analysis is for informational purposes only. Not financial advice. Always do your own research and consult with a financial advisor.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e)


if __name__ == "__main__":
    show()
