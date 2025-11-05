"""
ML-Optimized Portfolio
=======================
Build optimized portfolios using ML forecasts and advanced portfolio optimization.

Features:
- ML-driven expected returns (Ensemble, TFT, Hybrid models)
- Multiple optimization methods (Black-Litterman, HRP, Mean-Variance, CVaR)
- Trading signal generation
- Risk/return analysis
- Interactive parameter tuning
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from integration.dashboard_connector import create_dashboard_connector
from dashboard.utils.theme import apply_vscode_theme

# Page config
st.set_page_config(
    page_title="ML-Optimized Portfolio",
    page_icon="🤖",
    layout="wide"
)

# Apply theme
apply_vscode_theme()

# Page title
st.title("🤖 ML-Optimized Portfolio")
st.markdown("""
Build optimized portfolios using **machine learning forecasts** combined with
institutional-grade portfolio optimization algorithms.
""")

st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Ticker selection
st.sidebar.subheader("1. Select Universe")

universe_preset = st.sidebar.selectbox(
    "Preset Universe",
    [
        "Custom",
        "Tech Giants (FAANG+)",
        "Magnificent 7",
        "Dow 30",
        "S&P 500 Top 10"
    ]
)

if universe_preset == "Tech Giants (FAANG+)":
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
elif universe_preset == "Magnificent 7":
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
elif universe_preset == "Dow 30":
    default_tickers = ["AAPL", "MSFT", "JPM", "JNJ", "V", "PG", "UNH", "HD", "DIS", "BA"]
elif universe_preset == "S&P 500 Top 10":
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "XOM"]
else:
    default_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=", ".join(default_tickers),
    help="Enter stock tickers separated by commas"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(tickers) < 2:
    st.error("⚠️ Please enter at least 2 tickers")
    st.stop()

if len(tickers) > 20:
    st.warning("⚠️ Large portfolios (>20 stocks) may take longer to process")

st.sidebar.markdown("---")

# Model configuration
st.sidebar.subheader("2. ML Model")

model_type = st.sidebar.selectbox(
    "Forecasting Model",
    ["ensemble", "tft", "hybrid"],
    index=0,
    help=(
        "• Ensemble: LightGBM + Ridge + Momentum (fast, no GPU needed)\n"
        "• TFT: Temporal Fusion Transformer (deep learning, requires training)\n"
        "• Hybrid: CNN-LSTM-Transformer (deep learning, requires training)"
    )
)

if model_type in ["tft", "hybrid"]:
    st.sidebar.warning("⚠️ Deep learning models require pre-trained checkpoints")

horizon = st.sidebar.select_slider(
    "Forecast Horizon",
    options=["1w", "1m", "3m"],
    value="1m",
    help="Time horizon for forecasts"
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Minimum confidence threshold for views"
)

st.sidebar.markdown("---")

# Optimizer configuration
st.sidebar.subheader("3. Portfolio Optimizer")

optimizer_type = st.sidebar.selectbox(
    "Optimization Method",
    ["black_litterman", "mean_variance", "cvar", "hrp"],
    index=0,
    help=(
        "• Black-Litterman: Bayesian approach with ML views (recommended)\n"
        "• Mean-Variance: Classic Markowitz optimization\n"
        "• CVaR: Conditional Value at Risk (tail risk focus)\n"
        "• HRP: Hierarchical Risk Parity (ignores ML views)"
    )
)

risk_aversion = st.sidebar.slider(
    "Risk Aversion",
    min_value=0.5,
    max_value=5.0,
    value=2.5,
    step=0.5,
    help="Higher = more conservative (1-2 aggressive, 2-4 moderate, 4-5 conservative)"
)

st.sidebar.markdown("---")

# Signal configuration
st.sidebar.subheader("4. Trading Signals")

signal_strategy = st.sidebar.selectbox(
    "Signal Strategy",
    ["conservative", "moderate", "aggressive"],
    index=1,
    help=(
        "• Conservative: High confidence, 8%+ return, 2.0+ R/R\n"
        "• Moderate: Medium confidence, 5%+ return, 1.5+ R/R\n"
        "• Aggressive: Lower thresholds, more signals"
    )
)

st.sidebar.markdown("---")

# Action buttons
run_optimization = st.sidebar.button(
    "🚀 Run Optimization",
    type="primary",
    use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.info("""
**How it works:**
1. ML models forecast future returns
2. Forecasts → portfolio views (return + confidence)
3. Optimizer combines views with market data
4. Output: optimized weights + signals
""")

# Main content
if run_optimization:
    try:
        # Create connector
        connector = create_dashboard_connector()

        # Fetch data
        with st.spinner("📥 Fetching historical data..."):
            historical_data = connector.fetch_historical_data(tickers, years=2)

        if len(historical_data) == 0:
            st.error("❌ No data fetched. Please check tickers and try again.")
            st.stop()

        if len(historical_data) < len(tickers):
            missing = set(tickers) - set(historical_data.keys())
            st.warning(f"⚠️ Could not fetch data for: {', '.join(missing)}")

        # Generate allocation
        with st.spinner("🤖 Generating ML forecasts and optimizing portfolio..."):
            allocation = connector.get_ml_portfolio(
                tickers=list(historical_data.keys()),
                model_type=model_type,
                optimizer_type=optimizer_type,
                risk_aversion=risk_aversion,
                horizon=horizon,
                min_confidence=min_confidence,
                _historical_data=historical_data
            )

        # Display allocation
        st.success("✅ Optimization complete!")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Portfolio Allocation",
            "🔮 ML Forecast Views",
            "📈 Trading Signals",
            "ℹ️ Methodology"
        ])

        with tab1:
            st.markdown("### Portfolio Allocation")
            connector.display_allocation(allocation)

            # Additional metrics
            st.markdown("---")
            st.subheader("📋 Portfolio Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Configuration:**")
                st.write(f"• Model: {model_type.upper()}")
                st.write(f"• Optimizer: {optimizer_type.replace('_', ' ').title()}")
                st.write(f"• Horizon: {horizon}")
                st.write(f"• Risk Aversion: {risk_aversion}")

            with col2:
                st.markdown("**Statistics:**")
                st.write(f"• Universe Size: {len(tickers)} stocks")
                st.write(f"• Positions: {len([w for w in allocation.weights.values() if w > 0])}")
                st.write(f"• ML Views: {len(allocation.views)}")
                st.write(f"• Min Confidence: {min_confidence:.0%}")

        with tab2:
            st.markdown("### ML Forecast Views")
            st.markdown("""
            These are the **ML model's predictions** converted to portfolio views.
            Each view includes expected return and confidence level.
            """)
            connector.display_forecast_views(allocation)

        with tab3:
            st.markdown("### Trading Signals")
            st.markdown(f"""
            Trading signals generated using **{signal_strategy}** strategy.
            Signals indicate BUY/SELL/HOLD recommendations with strength and confidence.
            """)

            with st.spinner("📊 Generating trading signals..."):
                signals_df = connector.get_trading_signals(
                    tickers=list(historical_data.keys()),
                    strategy=signal_strategy,
                    _historical_data=historical_data
                )

            connector.display_signals(signals_df)

        with tab4:
            st.markdown("### Methodology")

            st.markdown("""
            #### Workflow

            1. **ML Forecasting**
               - Models: Ensemble (LightGBM+Ridge+Momentum), TFT, or Hybrid
               - Output: Expected returns with confidence levels
               - Time horizons: 1 week, 1 month, or 3 months

            2. **View Generation**
               - ML forecasts → Black-Litterman views
               - Format: (ticker, expected return, confidence)
               - Filtering by minimum confidence threshold

            3. **Portfolio Optimization**
               - **Black-Litterman**: Bayesian combination of market equilibrium + ML views
               - **Mean-Variance**: Markowitz optimization with ML-predicted returns
               - **CVaR**: Tail risk optimization (95% confidence)
               - **HRP**: Hierarchical risk parity (correlation-based, ignores views)

            4. **Trading Signals**
               - Convert forecasts to actionable BUY/SELL/HOLD signals
               - Confidence and risk/reward filtering
               - Strength scoring (0-100)

            #### Parameters

            - **Risk Aversion (λ)**: Controls risk/return tradeoff
              - 1-2: Aggressive (prioritize return)
              - 2-4: Moderate (balanced)
              - 4-5: Conservative (prioritize risk minimization)

            - **Minimum Confidence**: Filter out low-confidence forecasts
              - Higher = fewer but more confident views
              - Lower = more views but less selective

            - **Forecast Horizon**: Time period for predictions
              - 1w: Short-term tactical
              - 1m: Medium-term strategic
              - 3m: Long-term positioning

            #### Risk Management

            - **Long-only constraint**: No short selling
            - **Position limits**: 5% min, 40% max per stock
            - **Confidence weighting**: Higher confidence = stronger views
            - **Covariance estimation**: Based on historical returns

            #### Performance Metrics

            - **Expected Return**: Portfolio-weighted ML forecasts (annualized)
            - **Expected Volatility**: Portfolio risk based on historical covariance
            - **Sharpe Ratio**: (Return - Risk-free rate) / Volatility
            - **Risk/Reward Ratio**: Upside potential / Downside risk
            """)

            st.markdown("---")

            st.markdown("#### Model Comparison")

            comparison_data = {
                "Model": ["Ensemble", "TFT", "Hybrid"],
                "Speed": ["Fast (2-3s)", "Slow (5-10s)", "Slow (5-10s)"],
                "GPU Required": ["No", "Yes", "Yes"],
                "Training": ["Auto", "Pre-trained needed", "Pre-trained needed"],
                "Best For": [
                    "Quick analysis, production",
                    "Sophisticated forecasting",
                    "Complex patterns"
                ]
            }

            import pandas as pd
            st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error during optimization: {str(e)}")
        st.exception(e)
        st.info("💡 Tip: Try using 'ensemble' model for faster, more reliable results")

else:
    # Initial state - show instructions
    st.info("""
    ### 🚀 Quick Start

    1. **Select your stock universe** in the sidebar (or use a preset)
    2. **Choose ML model and optimizer** settings
    3. **Click "Run Optimization"** to generate your portfolio

    ### 💡 Recommended Settings for Beginners

    - **Model**: Ensemble (fast, no GPU needed)
    - **Optimizer**: Black-Litterman (best for ML integration)
    - **Horizon**: 1m (medium-term)
    - **Risk Aversion**: 2.5 (moderate)
    - **Signal Strategy**: Moderate

    ### 📊 What You'll Get

    - **Optimized portfolio weights** based on ML forecasts
    - **Expected return and risk metrics** (Sharpe ratio, volatility)
    - **ML forecast views** with confidence levels
    - **Trading signals** (BUY/SELL/HOLD) for each stock
    """)

    # Example results preview
    with st.expander("📸 Preview: What the results look like"):
        st.markdown("""
        **Portfolio Allocation Tab:**
        - Bar chart of portfolio weights
        - Key metrics: Expected Return, Volatility, Sharpe Ratio
        - Number of positions

        **ML Forecast Views Tab:**
        - Expected return for each stock
        - Confidence levels
        - Current vs forecast price
        - Risk/reward ratios

        **Trading Signals Tab:**
        - BUY/SELL/HOLD recommendations
        - Signal strength (0-100)
        - Confidence levels
        - Count of each signal type
        """)
