"""
Single Stock Analysis Page - FIXED VERSION
==========================================
Complete quantitative analysis with multi-signal long/short recommendation.

Improvements:
- Robust Fama-French factor analysis with error handling
- Multi-signal trading decision (valuation + momentum + mean reversion)
- No neutral positions - always Long or Short
- Better data alignment for factor regression
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

from portfolio.security_valuation import DCFInputs, DCFValuation
from portfolio.factor_models import FamaFrenchFactorModel


def calculate_risk_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Calculate comprehensive risk metrics."""
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

    # Sortino
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # VaR and CVaR (95%)
    var_95 = -np.percentile(returns, 5)
    cvar_95 = -returns[returns <= -var_95].mean() if len(returns[returns <= -var_95]) > 0 else var_95

    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95
    }


def show():
    """Main function for single stock analysis page."""
    st.title("🔍 Single Stock Analysis")
    st.markdown("**Complete quantitative analysis with institutional-grade metrics**")

    # Input section
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input("Ticker Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, MSFT, TSLA)").upper()

        with col2:
            period = st.selectbox(
                "Analysis Period",
                ["1y", "2y", "3y", "5y"],
                index=2,  # Default to 3y for better factor regression
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
                valuation_signal = 0  # -1 to 1 scale

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

                        # Valuation signal: -1 (overvalued) to +1 (undervalued)
                        valuation_signal = np.clip(upside / 0.3, -1, 1)  # ±30% = max signal

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Current Price",
                            f"${current_price:.2f}"
                        )

                        col2.metric(
                            "Intrinsic Value (DCF)",
                            f"${intrinsic:.2f}",
                            delta=f"{upside:+.1%}"
                        )

                        col3.metric(
                            "Valuation Signal",
                            f"{valuation_signal:+.2f}",
                            help="±1.0 scale: +1 = very undervalued, -1 = very overvalued"
                        )

                    except Exception as e:
                        st.warning(f"DCF valuation unavailable: {e}")
                        intrinsic = current_price
                        upside = 0
                else:
                    st.info("💡 DCF valuation not available (negative or missing free cash flow)")
                    intrinsic = current_price
                    upside = 0

                # === 2. TECHNICAL SIGNALS ===
                st.header("2️⃣ Technical Analysis")

                # Calculate technical indicators
                sma_20 = hist['Close'].rolling(20).mean()
                sma_50 = hist['Close'].rolling(50).mean()

                # Momentum signal
                momentum_20 = (current_price / sma_20.iloc[-1] - 1) if not sma_20.empty else 0
                momentum_50 = (current_price / sma_50.iloc[-1] - 1) if not sma_50.empty else 0
                momentum_signal = np.clip((momentum_20 + momentum_50) / 0.2, -1, 1)  # ±10% avg = max

                # Mean reversion signal (inverse)
                vol_20 = returns.tail(20).std()
                z_score = (current_price - sma_20.iloc[-1]) / (vol_20 * sma_20.iloc[-1]) if vol_20 > 0 and not sma_20.empty else 0
                mean_reversion_signal = -np.clip(z_score / 2, -1, 1)  # Inverse: oversold = buy

                # RSI
                delta = returns
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50

                # RSI signal: 30 = +1 (oversold/buy), 70 = -1 (overbought/sell)
                rsi_signal = np.clip((50 - current_rsi) / 20, -1, 1)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "20-Day Momentum",
                    f"{momentum_20:+.2%}",
                    help="Price vs 20-day SMA"
                )

                col2.metric(
                    "50-Day Momentum",
                    f"{momentum_50:+.2%}",
                    help="Price vs 50-day SMA"
                )

                col3.metric(
                    "RSI (14)",
                    f"{current_rsi:.1f}",
                    help="<30 oversold, >70 overbought"
                )

                col4.metric(
                    "Technical Signal",
                    f"{momentum_signal:+.2f}",
                    help="Combined momentum + RSI signal"
                )

                # === 3. RISK METRICS ===
                st.header("3️⃣ Risk Analysis")

                risk = calculate_risk_metrics(returns, risk_free_rate)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Sharpe Ratio",
                    f"{risk['sharpe_ratio']:.2f}",
                    help="Risk-adjusted return (>1 good, >2 excellent)"
                )

                col2.metric(
                    "Sortino Ratio",
                    f"{risk['sortino_ratio']:.2f}",
                    help="Downside risk-adjusted return"
                )

                col3.metric(
                    "Max Drawdown",
                    f"{risk['max_drawdown']:.2%}",
                    help="Largest peak-to-trough decline"
                )

                col4.metric(
                    "95% VaR",
                    f"{risk['var_95']:.2%}",
                    help="Maximum 1-day loss (95% confidence)"
                )

                # === 4. FACTOR ANALYSIS (IMPROVED) ===
                st.header("4️⃣ Fama-French Factor Exposure")

                alpha_signal = 0
                ff_success = False

                try:
                    # Ensure we have enough data
                    if len(returns) < 252:
                        st.warning(f"⚠️ Only {len(returns)} days of data. Factor analysis works best with 1+ years.")

                    ff = FamaFrenchFactorModel(model='5-factor')
                    ff_results = ff.regress_returns(ticker, returns, frequency='daily')

                    # Check if we got valid results
                    if ff_results and not np.isnan(ff_results.get('alpha', np.nan)):
                        ff_success = True

                        # Alpha signal: positive alpha = buy signal
                        alpha_annual = ff_results['alpha'] * 252
                        alpha_signal = np.clip(alpha_annual / 0.1, -1, 1)  # ±10% annual = max

                        # Display alpha and model fit
                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Alpha (annualized)",
                            f"{alpha_annual:.2%}",
                            delta="Significant ✓" if ff_results.get('alpha_pvalue', 1) < 0.05 else "Not sig.",
                            help=f"Skill-based return (p-value: {ff_results.get('alpha_pvalue', 1):.4f})"
                        )

                        col2.metric(
                            "Market Beta",
                            f"{ff_results.get('beta_MKT', np.nan):.2f}",
                            help="Sensitivity to market factor"
                        )

                        col3.metric(
                            "R-squared",
                            f"{ff_results.get('r_squared', 0):.1%}",
                            help="Variance explained by factors"
                        )

                        # Factor betas chart
                        betas_df = pd.DataFrame({
                            'Factor': ['Market\n(Mkt-RF)', 'Size\n(SMB)', 'Value\n(HML)', 'Profitability\n(RMW)', 'Investment\n(CMA)'],
                            'Beta': [
                                ff_results.get('beta_MKT', 0),
                                ff_results.get('beta_SMB', 0),
                                ff_results.get('beta_HML', 0),
                                ff_results.get('beta_RMW', 0),
                                ff_results.get('beta_CMA', 0)
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

                except Exception as e:
                    st.warning(f"⚠️ Factor analysis unavailable: {str(e)[:200]}")
                    st.info("💡 Factor data requires pandas-datareader and may fail for recent IPOs or non-US stocks.")

                # === 5. TRADING DECISION (MULTI-SIGNAL) ===
                st.header("5️⃣ Trading Recommendation")

                # Combine signals with weights
                signals = {
                    'Valuation': (valuation_signal, 0.30),
                    'Momentum': (momentum_signal, 0.25),
                    'Mean Reversion': (mean_reversion_signal, 0.15),
                    'RSI': (rsi_signal, 0.15),
                    'Alpha': (alpha_signal, 0.15) if ff_success else (0, 0)
                }

                # Calculate weighted average
                total_weight = sum(w for _, w in signals.values() if w > 0)
                if total_weight > 0:
                    combined_signal = sum(s * w for s, w in signals.values()) / total_weight
                else:
                    combined_signal = 0

                # Decision logic: NO NEUTRAL - always Long or Short
                if combined_signal > 0.4:
                    decision = "STRONG LONG 🚀"
                    decision_color = "green"
                    reasoning = "Multiple strong bullish signals detected"
                    confidence = "High"
                elif combined_signal > 0:
                    decision = "LONG 📈"
                    decision_color = "green"
                    reasoning = "Net bullish signals, moderate conviction"
                    confidence = "Medium"
                elif combined_signal < -0.4:
                    decision = "STRONG SHORT 📉"
                    decision_color = "red"
                    reasoning = "Multiple strong bearish signals detected"
                    confidence = "High"
                else:  # combined_signal <= 0
                    decision = "SHORT 🔻"
                    decision_color = "red"
                    reasoning = "Net bearish signals, moderate conviction"
                    confidence = "Medium"

                # Display decision prominently
                if decision_color == "green":
                    st.success(f"**{decision}**")
                elif decision_color == "red":
                    st.error(f"**{decision}**")

                st.info(f"**Reasoning**: {reasoning}")
                st.caption(f"**Combined Signal**: {combined_signal:+.2f} (±1.0 scale)")

                # Signal breakdown
                with st.expander("📊 Signal Breakdown"):
                    signal_df = pd.DataFrame([
                        {'Signal': name, 'Value': f"{sig:+.2f}", 'Weight': f"{weight:.0%}"}
                        for name, (sig, weight) in signals.items()
                        if weight > 0
                    ])
                    st.dataframe(signal_df, use_container_width=True, hide_index=True)

                # === 6. PRICE CHART ===
                st.header("6️⃣ Price History & Technical Indicators")

                # Create subplot figure
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price & Moving Averages', 'RSI'),
                    row_heights=[0.7, 0.3]
                )

                # Price and SMAs
                fig.add_trace(
                    go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=hist.index, y=sma_20, name='SMA 20', line=dict(color='orange', dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=hist.index, y=sma_50, name='SMA 50', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )

                # RSI
                fig.add_trace(
                    go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)

                fig.update_layout(height=700, showlegend=True)

                st.plotly_chart(fig, use_container_width=True)

                st.success("✅ Analysis complete! Review the recommendation and metrics above before making investment decisions.")
                st.caption("⚠️ This is for educational purposes only. Not financial advice.")

            except Exception as e:
                st.error(f"❌ Error analyzing {ticker}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    show()
