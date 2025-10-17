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

# Professional Institutional CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;600&display=swap');

    /* Dark Professional Theme */
    .stApp {
        background: linear-gradient(135deg, #0a1929 0%, #1a2332 100%);
        font-family: 'Inter', sans-serif;
    }

    .main {
        padding: 1rem 2rem;
        max-width: 1600px;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        color: #f8f9fa !important;
        font-weight: 600 !important;
    }

    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h2 {
        border-bottom: 2px solid rgba(59,130,246,0.3);
        padding-bottom: 0.5rem;
    }

    /* Metrics Cards - Professional Design */
    .stMetric {
        background: linear-gradient(135deg, rgba(15,23,42,0.8) 0%, rgba(30,41,59,0.8) 100%) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(59,130,246,0.2) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3), 0 0 20px rgba(59,130,246,0.1) !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.4), 0 0 30px rgba(59,130,246,0.2) !important;
        border-color: rgba(59,130,246,0.4) !important;
    }

    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #f8f9fa !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'Roboto Mono', monospace !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }

    /* Tables */
    .dataframe {
        font-family: 'Roboto Mono', monospace !important;
        background: rgba(15,23,42,0.6);
        border-radius: 8px;
    }

    .dataframe thead tr th {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        color: #f8f9fa !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.75rem;
        border-bottom: 2px solid rgba(59,130,246,0.3) !important;
    }

    .dataframe tbody tr:hover {
        background: rgba(59,130,246,0.05) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(59,130,246,0.2);
    }

    /* Charts */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .reportview-container .main .block-container {
        max-width: 1600px;
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
