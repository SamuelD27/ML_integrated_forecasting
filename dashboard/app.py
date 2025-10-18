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

from dashboard.utils.theme import apply_vscode_theme

# Page config
st.set_page_config(
    page_title="Stock Analysis | Quantitative Trading Platform",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Professional quantitative finance platform for institutional-grade analysis"
    }
)

# Apply VS Code theme
apply_vscode_theme()

# Sidebar
st.sidebar.title("QUANTITATIVE FINANCE")
st.sidebar.markdown("---")
st.sidebar.info("""
Use the navigation menu above to access different analysis tools.

All pages now available as separate tabs in the sidebar.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("SYSTEM")
if st.sidebar.button("Clear Data Cache"):
    st.cache_data.clear()
    st.success("Cache cleared successfully")
    st.rerun()

# Home page content
st.title("Quantitative Finance Platform")

st.markdown("""
## Overview

Professional-grade quantitative finance platform for institutional analysis and trading strategies.

### Available Tools

Navigate using the sidebar menu to access:

1. **Single Stock Analysis** - Comprehensive valuation and trading recommendations
2. **Portfolio from Galaxy** - Multi-asset portfolio optimization
3. **Portfolio from Single Stock** - Automated peer discovery and portfolio construction
4. **Portfolio Risk Analysis** - Advanced risk metrics and decomposition
5. **Factor Analysis** - Fama-French factor exposure and attribution
6. **Advanced Monte Carlo** - Price path simulation and forecasting
7. **Security Pricing** - Valuation models for equities, bonds, and options

---

## Features

### Quantitative Analysis
- DCF and DDM valuation models
- Fama-French 3-factor and 5-factor models
- Machine learning ensemble forecasting
- Technical indicators and momentum signals

### Portfolio Optimization
- Hierarchical Risk Parity (HRP)
- Mean-variance optimization
- CVaR optimization with shrinkage estimators
- Automatic peer discovery

### Risk Management
- Value at Risk (VaR) and Conditional VaR
- Maximum drawdown analysis
- Sharpe and Sortino ratios
- Stress testing and scenario analysis

### Advanced Modeling
- Monte Carlo simulation (GBM, Jump Diffusion, Regime Switching)
- Black-Scholes options pricing with Greeks
- Bond pricing and duration analysis
- Market regime detection

---

## Quick Start

1. Select a tool from the sidebar navigation
2. Enter your analysis parameters
3. Click the run/analyze button
4. Review results and export if needed

## System

- **Clear Data Cache**: Force refresh of cached market data
- **Theme**: Professional VS Code dark theme
- **Stock Search**: Autocomplete search across 200+ stocks from 7 major exchanges

---

""")

# Display system metrics
st.subheader("Platform Capabilities")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ANALYSIS TOOLS", "7")
with col2:
    st.metric("VALUATION MODELS", "6+")
with col3:
    st.metric("RISK METRICS", "15+")
with col4:
    st.metric("STOCK DATABASE", "200+")

st.markdown("---")
st.caption("Professional quantitative finance platform | Built with Python + Streamlit")
