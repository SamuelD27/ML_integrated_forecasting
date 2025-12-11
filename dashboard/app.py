"""
Trading Terminal - Professional Interface
=========================================
Bloomberg/Binance-inspired trading terminal.
"""

import streamlit as st
from datetime import datetime
import pytz

st.set_page_config(page_title="Terminal", page_icon="", layout="wide", initial_sidebar_state="collapsed")

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

* { font-family: 'IBM Plex Sans', -apple-system, sans-serif !important; }
code, .mono { font-family: 'IBM Plex Mono', monospace !important; }

/* Hide sidebar completely */
[data-testid="stSidebar"] { display: none !important; }

/* Streamlit element overrides */
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
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: var(--bg-3) !important;
    border-color: var(--border-light) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: var(--bg-0) !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-dim) !important;
}

.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-1) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-0) !important;
    font-size: 12px !important;
    border-radius: 2px !important;
}

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

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER BAR
# =============================================================================

now = datetime.now()
ny_time = datetime.now(pytz.timezone('America/New_York'))
is_market_open = ny_time.weekday() < 5 and 9 <= ny_time.hour < 16

market_status = "OPEN" if is_market_open else "CLOSED"
market_color = "#00ff88" if is_market_open else "#ff4444"

st.markdown(f'''
<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 16px;background:#0a0a0a;border-bottom:1px solid #222;">
<div style="display:flex;align-items:center;gap:24px;">
<span style="font-family:IBM Plex Mono,monospace;font-size:14px;font-weight:600;color:#00ff88;letter-spacing:1px;">TERMINAL</span>
<div style="display:flex;gap:12px;">
<span style="font-size:11px;color:#888;padding:4px 8px;background:#111;border-radius:2px;cursor:pointer;" onclick="window.location.reload()">BOT</span>
<span style="font-size:11px;color:#888;padding:4px 8px;background:#111;border-radius:2px;cursor:pointer;">ANALYSIS</span>
</div>
</div>
<div style="display:flex;align-items:center;gap:16px;">
<div style="display:flex;align-items:center;gap:6px;">
<span style="width:6px;height:6px;border-radius:50%;background:{market_color};"></span>
<span style="font-size:10px;color:{market_color};font-weight:500;">NYSE {market_status}</span>
</div>
<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#555;">{ny_time.strftime("%H:%M:%S")} ET</span>
<span style="font-size:10px;color:#ffaa00;padding:2px 6px;border:1px solid #ffaa00;border-radius:2px;">PAPER</span>
</div>
</div>
''', unsafe_allow_html=True)

# =============================================================================
# NAVIGATION TABS
# =============================================================================

col_nav1, col_nav2, col_spacer = st.columns([1, 1, 8])
with col_nav1:
    if st.button("BOT CONTROL", key="nav_bot", use_container_width=True, type="primary"):
        st.switch_page("pages/bot_control.py")
with col_nav2:
    if st.button("ANALYSIS", key="nav_analysis", use_container_width=True):
        st.switch_page("pages/deep_analysis.py")

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

# Get account data
try:
    from bot.trader import EnhancedTrader
    from bot.config import load_config
    config = load_config()
    trader = EnhancedTrader(config=config)
    account = trader.get_account()
    positions = trader.get_positions()
    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    daily_pnl = float(getattr(account, 'equity', 0)) - 100000  # Assuming 100k start
    has_account = True
except:
    equity = 100000.0
    cash = 100000.0
    buying_power = 200000.0
    daily_pnl = 0.0
    positions = []
    has_account = False

# Check bot status
import subprocess
try:
    result = subprocess.run(["pgrep", "-f", "python.*bot.main"], capture_output=True, text=True)
    bot_running = bool(result.stdout.strip())
except:
    bot_running = False

# KPI Bar
pnl_color = "#00ff88" if daily_pnl >= 0 else "#ff4444"
pnl_sign = "+" if daily_pnl >= 0 else ""
bot_status_color = "#00ff88" if bot_running else "#ff4444"
bot_status_text = "RUNNING" if bot_running else "STOPPED"

st.markdown(f'''
<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:#222;margin:1px 0;">
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Equity</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#fff;font-weight:500;">${equity:,.2f}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Cash</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#ccc;">${cash:,.2f}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Buying Power</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#ccc;">${buying_power:,.2f}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">P&L</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:{pnl_color};font-weight:500;">{pnl_sign}${abs(daily_pnl):,.2f}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Positions</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#ccc;">{len(positions)}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Bot</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:14px;color:{bot_status_color};font-weight:600;">{bot_status_text}</div>
</div>
</div>
''', unsafe_allow_html=True)

# Main Content Grid
col_left, col_right = st.columns([2, 1])

with col_left:
    # Positions Table
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;display:flex;justify-content:space-between;align-items:center;">
<span style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Open Positions</span>
<span style="font-size:10px;color:#555;">Updated just now</span>
</div>
''', unsafe_allow_html=True)

    if positions:
        # Table header
        st.markdown('''
<div style="display:grid;grid-template-columns:80px 1fr 80px 100px 100px 100px;padding:8px 16px;border-bottom:1px solid #222;font-size:10px;color:#555;text-transform:uppercase;">
<span>Symbol</span><span>Side</span><span>Qty</span><span>Entry</span><span>Current</span><span>P&L</span>
</div>
''', unsafe_allow_html=True)
        for pos in positions:
            pnl = float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else 0
            pnl_pct = float(pos.unrealized_plpc) * 100 if hasattr(pos, 'unrealized_plpc') else 0
            current = float(pos.current_price) if hasattr(pos, 'current_price') else float(pos.avg_entry_price)
            entry = float(pos.avg_entry_price)
            qty = float(pos.qty)
            row_pnl_color = "#00ff88" if pnl >= 0 else "#ff4444"
            st.markdown(f'''
<div style="display:grid;grid-template-columns:80px 1fr 80px 100px 100px 100px;padding:10px 16px;border-bottom:1px solid #1a1a1a;font-size:12px;">
<span style="font-family:IBM Plex Mono,monospace;color:#fff;font-weight:500;">{pos.symbol}</span>
<span style="color:#00ff88;">LONG</span>
<span style="font-family:IBM Plex Mono,monospace;color:#ccc;">{qty:.0f}</span>
<span style="font-family:IBM Plex Mono,monospace;color:#888;">${entry:.2f}</span>
<span style="font-family:IBM Plex Mono,monospace;color:#ccc;">${current:.2f}</span>
<span style="font-family:IBM Plex Mono,monospace;color:{row_pnl_color};">{("+" if pnl >= 0 else "")}${pnl:.2f}</span>
</div>
''', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:40px 16px;text-align:center;color:#555;font-size:12px;">No open positions</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Activity Log
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Activity Log</span>
</div>
<div style="padding:8px 16px;font-family:IBM Plex Mono,monospace;font-size:11px;line-height:1.8;max-height:200px;overflow-y:auto;">
''', unsafe_allow_html=True)

    # Read actual logs if available
    from pathlib import Path
    log_file = Path(__file__).parent.parent / "logs" / "bot.log"
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-20:]
            for line in lines:
                line = line.strip()[:100]
                color = "#555"
                if "ERROR" in line.upper():
                    color = "#ff4444"
                elif "SUCCESS" in line.upper() or "CONNECTED" in line.upper():
                    color = "#00ff88"
                elif "WARNING" in line.upper():
                    color = "#ffaa00"
                st.markdown(f'<div style="color:{color};padding:2px 0;">{line}</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div style="color:#555;">No recent activity</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#555;">No log file</div>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

with col_right:
    # Quick Actions Panel
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Controls</span>
</div>
<div style="padding:16px;">
''', unsafe_allow_html=True)

    if bot_running:
        if st.button("STOP BOT", key="stop_bot", use_container_width=True):
            import os, signal
            result = subprocess.run(["pgrep", "-f", "python.*bot.main"], capture_output=True, text=True)
            for pid in result.stdout.strip().split('\n'):
                if pid.strip():
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except:
                        pass
            st.rerun()
    else:
        if st.button("START BOT", key="start_bot", use_container_width=True, type="primary"):
            project_root = Path(__file__).parent.parent
            cmd = f"cd {project_root} && source venv_ml/bin/activate && nohup python -m bot.main > logs/bot.log 2>&1 &"
            subprocess.Popen(cmd, shell=True, executable='/bin/bash')
            import time
            time.sleep(2)
            st.rerun()

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    if st.button("REFRESH", key="refresh", use_container_width=True):
        st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Market Hours
    markets = [
        ('NYSE', 'America/New_York', 9, 30, 16, 0),
        ('LSE', 'Europe/London', 8, 0, 16, 30),
        ('TSE', 'Asia/Tokyo', 9, 0, 15, 0),
    ]

    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Markets</span>
</div>
<div style="padding:12px 16px;">
''', unsafe_allow_html=True)

    for label, tz_name, oh, om, ch, cm in markets:
        tz = pytz.timezone(tz_name)
        t = datetime.now(tz)
        is_open = t.weekday() < 5 and t.replace(hour=oh, minute=om) <= t <= t.replace(hour=ch, minute=cm)
        status_color = "#00ff88" if is_open else "#ff4444"
        status_text = "OPEN" if is_open else "CLOSED"
        st.markdown(f'''
<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #1a1a1a;">
<div>
<span style="font-size:12px;color:#ccc;">{label}</span>
<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#555;margin-left:8px;">{t.strftime("%H:%M")}</span>
</div>
<span style="font-size:10px;color:{status_color};">{status_text}</span>
</div>
''', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # System Status
    st.markdown(f'''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">System</span>
</div>
<div style="padding:12px 16px;font-size:11px;">
<div style="display:flex;justify-content:space-between;padding:4px 0;"><span style="color:#888;">Alpaca</span><span style="color:{"#00ff88" if has_account else "#ff4444"};">{"OK" if has_account else "ERR"}</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0;"><span style="color:#888;">Bot Process</span><span style="color:{bot_status_color};">{bot_status_text}</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0;"><span style="color:#888;">Mode</span><span style="color:#ffaa00;">PAPER</span></div>
</div>
</div>
''', unsafe_allow_html=True)

# Footer
st.markdown(f'''
<div style="position:fixed;bottom:0;left:0;right:0;padding:6px 16px;background:#0a0a0a;border-top:1px solid #222;display:flex;justify-content:space-between;font-size:10px;color:#555;">
<span>Terminal v3.0</span>
<span>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
</div>
''', unsafe_allow_html=True)
