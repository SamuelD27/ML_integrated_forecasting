"""
Streamlit Interactive Dashboard
================================
Quantitative Finance Dashboard with 7 core features.

Features:
1. Single Stock Analysis - Long/short decision with full quant analysis
2. Portfolio from Galaxy - Build portfolio from multiple securities
3. Portfolio from Single Stock - Build portfolio starting from one stock
4. Portfolio Risk Analysis - Comprehensive risk assessment
5. Factor Analysis - Fama-French factor exposure
6. Advanced Monte Carlo - Price path simulations
7. Security Pricing - Valuation models for different security types
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dashboard pages
from dashboard.pages import (
    single_stock_analysis,
    portfolio_from_galaxy,
    portfolio_from_single_stock,
    portfolio_risk_analysis,
    factor_analysis,
    advanced_monte_carlo,
    security_pricing
)

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

    /* Fix metric contrast */
    .stMetric {
        background-color: #ffffff !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    .stMetric label {
        color: #1f1f1f !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    .reportview-container .main .block-container {
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Quant Finance Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Single Stock Analysis",
        "🌌 Portfolio from Galaxy",
        "🔄 Portfolio from Single Stock",
        "⚠️ Portfolio Risk Analysis",
        "📊 Factor Analysis",
        "🎲 Advanced Monte Carlo",
        "💰 Security Pricing"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
**Quantitative Finance Dashboard**

Comprehensive suite of quantitative analysis tools for:
- Stock valuation & recommendation
- Portfolio construction & optimization
- Risk analysis & management
- Factor models & attribution
- Monte Carlo simulation
- Options & bond pricing

Built with Streamlit + Python
""")

# Data refresh button
if st.sidebar.button("🔄 Refresh Data Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")
    st.rerun()

# Route to pages
if page == "🏠 Home":
    st.title("Welcome to Quantitative Finance Dashboard")
    st.markdown("""
    ## 🎯 Available Tools

    Select a tool from the sidebar to get started:

    ### 📈 Single Stock Analysis
    - DCF valuation
    - Long/short recommendation
    - Risk metrics (VaR, Sharpe, Sortino)
    - Factor exposure analysis
    - Technical indicators

    ### 🌌 Portfolio from Galaxy
    - Multi-ticker input
    - HRP optimization
    - Weight allocation
    - Correlation analysis
    - Historical performance

    ### 🔄 Portfolio from Single Stock
    - Automatic peer discovery
    - Portfolio construction
    - Optimization methods
    - Risk/return analysis

    ### ⚠️ Portfolio Risk Analysis
    - VaR and CVaR calculation
    - Risk decomposition
    - Drawdown analysis
    - Stress testing
    - Return distribution

    ### 📊 Factor Analysis
    - Fama-French 3-factor & 5-factor models
    - Alpha and beta estimation
    - Statistical significance testing
    - Factor exposure heatmaps

    ### 🎲 Advanced Monte Carlo
    - Jump diffusion (Merton model)
    - Fat tail distributions
    - Regime switching
    - Variance reduction techniques
    - Risk metrics from simulations

    ### 💰 Security Pricing
    - **Equities**: DCF, DDM, Relative valuation
    - **Bonds**: Pricing and duration
    - **Options**: Black-Scholes with Greeks
    - Automatic model selection

    ---

    ## 📚 Quick Start

    1. Select a tool from the sidebar
    2. Enter your parameters
    3. Click the analyze/run button
    4. Review results and export if needed

    ## 💡 Tips

    - Use the **Refresh Data Cache** button to force fresh data fetch
    - Most pages support CSV export
    - Hover over metrics for explanations
    - Check parameter tooltips for guidance
    """)

    # Display some quick stats
    st.subheader("📊 Dashboard Features")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tools", "7")
    with col2:
        st.metric("Valuation Models", "6+")
    with col3:
        st.metric("Risk Metrics", "15+")
    with col4:
        st.metric("Chart Types", "10+")

elif page == "📈 Single Stock Analysis":
    single_stock_analysis.show()

elif page == "🌌 Portfolio from Galaxy":
    portfolio_from_galaxy.show()

elif page == "🔄 Portfolio from Single Stock":
    portfolio_from_single_stock.show()

elif page == "⚠️ Portfolio Risk Analysis":
    portfolio_risk_analysis.show()

elif page == "📊 Factor Analysis":
    factor_analysis.show()

elif page == "🎲 Advanced Monte Carlo":
    advanced_monte_carlo.show()

elif page == "💰 Security Pricing":
    security_pricing.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("💻 Quantitative Finance System v2.0")
st.sidebar.caption("Built with Streamlit + Python")
