"""
Analysis Terminal - Professional Trading Interface
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import pytz

st.set_page_config(page_title="Analysis", page_icon="", layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
# PROFESSIONAL DARK THEME
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg-0: #000000;
    --bg-1: #0a0a0a;
    --bg-2: #111111;
    --bg-3: #1a1a1a;
    --border: #222222;
    --border-light: #333333;
    --text-0: #ffffff;
    --text-1: #cccccc;
    --text-2: #888888;
    --text-3: #555555;
    --accent: #00ff88;
    --accent-dim: #00cc6a;
    --red: #ff4444;
    --yellow: #ffaa00;
    --blue: #4488ff;
}

.stApp { background: var(--bg-0) !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

* { font-family: 'IBM Plex Sans', -apple-system, sans-serif !important; }
code, .mono { font-family: 'IBM Plex Mono', monospace !important; }

.stButton > button {
    background: var(--bg-2) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
    border-radius: 2px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover { background: var(--bg-3) !important; }
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: var(--bg-0) !important;
    border: none !important;
    font-weight: 600 !important;
}

.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-1) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-0) !important;
    font-size: 12px !important;
    border-radius: 2px !important;
}

.stSlider > div > div { background: var(--border) !important; }
.stSlider [data-baseweb="slider"] div { background: var(--accent) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-1) !important;
    gap: 0 !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-2) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 10px 16px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

label { color: var(--text-2) !important; font-size: 11px !important; text-transform: uppercase !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

now = datetime.now()
ny_time = datetime.now(pytz.timezone('America/New_York'))
is_market_open = ny_time.weekday() < 5 and 9 <= ny_time.hour < 16
market_color = "#00ff88" if is_market_open else "#ff4444"
market_status = "OPEN" if is_market_open else "CLOSED"

st.markdown(f'''
<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 16px;background:#0a0a0a;border-bottom:1px solid #222;">
<div style="display:flex;align-items:center;gap:24px;">
<span style="font-family:IBM Plex Mono,monospace;font-size:14px;font-weight:600;color:#00ff88;letter-spacing:1px;">TERMINAL</span>
<span style="font-size:11px;color:#888;">Analysis</span>
</div>
<div style="display:flex;align-items:center;gap:16px;">
<div style="display:flex;align-items:center;gap:6px;">
<span style="width:6px;height:6px;border-radius:50%;background:{market_color};"></span>
<span style="font-size:10px;color:{market_color};font-weight:500;">NYSE {market_status}</span>
</div>
<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#555;">{ny_time.strftime("%H:%M:%S")} ET</span>
</div>
</div>
''', unsafe_allow_html=True)

# Navigation
col1, col2, col3 = st.columns([1, 1, 8])
with col1:
    if st.button("DASHBOARD", key="nav_home", use_container_width=True):
        st.switch_page("app.py")
with col2:
    if st.button("BOT CONTROL", key="nav_bot", use_container_width=True):
        st.switch_page("pages/bot_control.py")

# =============================================================================
# ANALYSIS TABS
# =============================================================================

tabs = st.tabs(["STOCK", "PORTFOLIO", "RISK", "FACTORS", "MONTE CARLO", "ML", "OPTIONS"])

# TAB 1: Single Stock
with tabs[0]:
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin:1px 0;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Single Stock Analysis</span>
</div>
</div>
''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)
        ticker = st.text_input("Ticker", value="AAPL", key="stock_ticker")
        years = st.slider("Years", 1, 5, 3, key="stock_years")
        if st.button("ANALYZE", key="stock_run", use_container_width=True, type="primary"):
            st.info(f"Analyzing {ticker}...")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#050505;border:1px solid #222;margin:16px;padding:40px;text-align:center;color:#555;font-size:11px;">Enter ticker and click ANALYZE</div>', unsafe_allow_html=True)

# TAB 2: Portfolio
with tabs[1]:
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin:1px 0;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Portfolio Optimization</span>
</div>
</div>
''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)
        tickers = st.text_area("Tickers", value="AAPL\nMSFT\nGOOGL\nAMZN", key="port_tickers", height=100)
        capital = st.number_input("Capital", value=100000, key="port_capital")
        method = st.selectbox("Method", ["Black-Litterman", "HRP", "CVaR"], key="port_method")
        if st.button("OPTIMIZE", key="port_run", use_container_width=True, type="primary"):
            st.info("Optimizing...")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#050505;border:1px solid #222;margin:16px;padding:40px;text-align:center;color:#555;font-size:11px;">Configure and click OPTIMIZE</div>', unsafe_allow_html=True)

# TAB 3: Risk
with tabs[2]:
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin:1px 0;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Risk Analysis</span>
</div>
</div>
''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)
        conf = st.slider("Confidence", 0.90, 0.99, 0.95, key="risk_conf")
        horizon = st.selectbox("Horizon", ["1 Day", "1 Week", "1 Month"], key="risk_horizon")
        if st.button("CALCULATE", key="risk_run", use_container_width=True, type="primary"):
            st.info("Calculating VaR/CVaR...")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#050505;border:1px solid #222;margin:16px;padding:40px;text-align:center;color:#555;font-size:11px;">Set parameters and click CALCULATE</div>', unsafe_allow_html=True)

# TAB 4: Factors
with tabs[3]:
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin:1px 0;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Factor Analysis</span>
</div>
</div>
''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)
        model = st.selectbox("Model", ["Fama-French 3", "Fama-French 5", "Carhart 4"], key="factor_model")
        if st.button("ANALYZE", key="factor_run", use_container_width=True, type="primary"):
            st.info("Running factor analysis...")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#050505;border:1px solid #222;margin:16px;padding:40px;text-align:center;color:#555;font-size:11px;">Select model and click ANALYZE</div>', unsafe_allow_html=True)

# TAB 5: Monte Carlo
with tabs[4]:
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin:1px 0;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Monte Carlo Simulation</span>
</div>
</div>
''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)
        sims = st.number_input("Simulations", value=10000, key="mc_sims")
        days = st.number_input("Days", value=252, key="mc_days")
        if st.button("SIMULATE", key="mc_run", use_container_width=True, type="primary"):
            st.info("Running simulation...")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#050505;border:1px solid #222;margin:16px;padding:40px;text-align:center;color:#555;font-size:11px;">Configure and click SIMULATE</div>', unsafe_allow_html=True)

# TAB 6: ML
with tabs[5]:
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin:1px 0;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">ML Forecast</span>
</div>
</div>
''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)
        ml_ticker = st.text_input("Ticker", value="AAPL", key="ml_ticker")
        ml_horizon = st.selectbox("Horizon", ["5 Days", "10 Days", "20 Days"], key="ml_horizon")
        if st.button("FORECAST", key="ml_run", use_container_width=True, type="primary"):
            st.info("Generating forecast...")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#050505;border:1px solid #222;margin:16px;padding:40px;text-align:center;color:#555;font-size:11px;">Enter ticker and click FORECAST</div>', unsafe_allow_html=True)

# TAB 7: Options
with tabs[6]:
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin:1px 0;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Options Pricing</span>
</div>
</div>
''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)
        opt_ticker = st.text_input("Underlying", value="SPY", key="opt_ticker")
        opt_type = st.selectbox("Type", ["Call", "Put"], key="opt_type")
        strike = st.number_input("Strike", value=450.0, key="opt_strike")
        if st.button("PRICE", key="opt_run", use_container_width=True, type="primary"):
            st.info("Pricing option...")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#050505;border:1px solid #222;margin:16px;padding:40px;text-align:center;color:#555;font-size:11px;">Configure option and click PRICE</div>', unsafe_allow_html=True)

# Footer
st.markdown(f'''
<div style="position:fixed;bottom:0;left:0;right:0;padding:6px 16px;background:#0a0a0a;border-top:1px solid #222;display:flex;justify-content:space-between;font-size:10px;color:#555;">
<span>Analysis v3.0 | 7 Tools</span>
<span>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
</div>
''', unsafe_allow_html=True)
