"""
Bot Control - Professional Trading Terminal
"""

import streamlit as st
import subprocess
import os
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import pytz

project_root = Path(__file__).parent.parent.parent

st.set_page_config(page_title="Bot Control", page_icon="", layout="wide", initial_sidebar_state="collapsed")

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

.stCheckbox label { color: var(--text-2) !important; font-size: 11px !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_bot_pid() -> Optional[int]:
    try:
        result = subprocess.run(["pgrep", "-f", "python.*bot.main"], capture_output=True, text=True)
        if result.stdout.strip():
            pids = [int(p) for p in result.stdout.strip().split('\n') if p.strip()]
            return pids[0] if pids else None
    except:
        pass
    return None

def start_bot() -> Tuple[bool, str]:
    try:
        if get_bot_pid():
            return False, "Already running"
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        cmd = f"cd {project_root} && source venv_ml/bin/activate && nohup python -m bot.main > logs/bot.log 2>&1 &"
        subprocess.Popen(cmd, shell=True, executable='/bin/bash')
        time.sleep(2)
        return bool(get_bot_pid()), "Started" if get_bot_pid() else "Failed"
    except Exception as e:
        return False, str(e)

def stop_bot() -> Tuple[bool, str]:
    try:
        result = subprocess.run(["pgrep", "-f", "python.*bot.main"], capture_output=True, text=True)
        if not result.stdout.strip():
            return False, "Not running"
        for pid in result.stdout.strip().split('\n'):
            if pid.strip():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except:
                    pass
        time.sleep(1)
        return not get_bot_pid(), "Stopped"
    except Exception as e:
        return False, str(e)

def get_logs(n: int = 50) -> List[str]:
    log_file = project_root / "logs" / "bot.log"
    if not log_file.exists():
        return []
    try:
        with open(log_file, 'r') as f:
            return f.readlines()[-n:]
    except:
        return []

def get_account() -> Optional[Dict]:
    try:
        from bot.trader import EnhancedTrader
        from bot.config import load_config
        config = load_config()
        trader = EnhancedTrader(config=config)
        account = trader.get_account()
        positions = trader.get_positions()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'positions': positions,
        }
    except:
        return None

# =============================================================================
# HEADER
# =============================================================================

now = datetime.now()
ny_time = datetime.now(pytz.timezone('America/New_York'))
pid = get_bot_pid()
bot_color = "#00ff88" if pid else "#ff4444"
bot_status = "RUNNING" if pid else "STOPPED"

st.markdown(f'''
<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 16px;background:#0a0a0a;border-bottom:1px solid #222;">
<div style="display:flex;align-items:center;gap:24px;">
<span style="font-family:IBM Plex Mono,monospace;font-size:14px;font-weight:600;color:#00ff88;letter-spacing:1px;">TERMINAL</span>
<span style="font-size:11px;color:#888;">Bot Control</span>
</div>
<div style="display:flex;align-items:center;gap:16px;">
<div style="display:flex;align-items:center;gap:6px;">
<span style="width:6px;height:6px;border-radius:50%;background:{bot_color};"></span>
<span style="font-size:10px;color:{bot_color};font-weight:500;">{bot_status}</span>
</div>
<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#555;">{ny_time.strftime("%H:%M:%S")} ET</span>
<span style="font-size:10px;color:#ffaa00;padding:2px 6px;border:1px solid #ffaa00;border-radius:2px;">PAPER</span>
</div>
</div>
''', unsafe_allow_html=True)

# Navigation
col1, col2, col3 = st.columns([1, 1, 8])
with col1:
    if st.button("DASHBOARD", key="nav_home", use_container_width=True):
        st.switch_page("app.py")
with col2:
    if st.button("ANALYSIS", key="nav_analysis", use_container_width=True):
        st.switch_page("pages/deep_analysis.py")

# =============================================================================
# MAIN CONTENT
# =============================================================================

account = get_account()

# KPI Bar
if account:
    equity = account['equity']
    cash = account['cash']
    bp = account['buying_power']
    pos_count = len(account['positions'])
else:
    equity = cash = bp = 0
    pos_count = 0

st.markdown(f'''
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:#222;margin:1px 0;">
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;margin-bottom:4px;">Equity</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#fff;">${equity:,.2f}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;margin-bottom:4px;">Cash</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#ccc;">${cash:,.2f}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;margin-bottom:4px;">Buying Power</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#ccc;">${bp:,.2f}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;margin-bottom:4px;">Positions</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#ccc;">{pos_count}</div>
</div>
<div style="background:#0a0a0a;padding:12px 16px;">
<div style="font-size:10px;color:#555;text-transform:uppercase;margin-bottom:4px;">Bot Status</div>
<div style="font-family:IBM Plex Mono,monospace;font-size:14px;color:{bot_color};font-weight:600;">{bot_status}</div>
</div>
</div>
''', unsafe_allow_html=True)

# Three column layout
col_ctrl, col_pos, col_log = st.columns([1, 1.5, 2.5])

with col_ctrl:
    # Bot Control Panel
    st.markdown(f'''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Bot Control</span>
</div>
<div style="padding:20px 16px;text-align:center;">
<div style="width:60px;height:60px;border-radius:50%;background:{"rgba(0,255,136,0.1)" if pid else "rgba(255,68,68,0.1)"};border:2px solid {bot_color};display:inline-flex;align-items:center;justify-content:center;margin-bottom:12px;">
<span style="font-family:IBM Plex Mono,monospace;font-size:12px;color:{bot_color};font-weight:600;">{"ON" if pid else "OFF"}</span>
</div>
<div style="font-size:12px;color:{bot_color};font-weight:500;margin-bottom:4px;">{bot_status}</div>
<div style="font-size:10px;color:#555;">{"PID: " + str(pid) if pid else "Not running"}</div>
</div>
</div>
''', unsafe_allow_html=True)

    if pid:
        if st.button("STOP BOT", key="stop", use_container_width=True):
            success, msg = stop_bot()
            time.sleep(0.5)
            st.rerun()
    else:
        if st.button("START BOT", key="start", use_container_width=True, type="primary"):
            success, msg = start_bot()
            st.rerun()

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    if st.button("REFRESH", key="refresh", use_container_width=True):
        st.rerun()

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    auto = st.checkbox("Auto-refresh (10s)", key="auto")
    if auto:
        time.sleep(10)
        st.rerun()

with col_pos:
    # Positions
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Positions</span>
</div>
''', unsafe_allow_html=True)

    if account and account['positions']:
        st.markdown('''
<div style="display:grid;grid-template-columns:60px 50px 70px 70px 80px;padding:8px 16px;border-bottom:1px solid #222;font-size:9px;color:#555;text-transform:uppercase;">
<span>Symbol</span><span>Qty</span><span>Entry</span><span>Current</span><span>P&L</span>
</div>
''', unsafe_allow_html=True)
        for pos in account['positions']:
            pnl = float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else 0
            current = float(pos.current_price) if hasattr(pos, 'current_price') else float(pos.avg_entry_price)
            entry = float(pos.avg_entry_price)
            qty = float(pos.qty)
            pnl_color = "#00ff88" if pnl >= 0 else "#ff4444"
            st.markdown(f'''
<div style="display:grid;grid-template-columns:60px 50px 70px 70px 80px;padding:8px 16px;border-bottom:1px solid #1a1a1a;font-size:11px;">
<span style="font-family:IBM Plex Mono,monospace;color:#fff;">{pos.symbol}</span>
<span style="font-family:IBM Plex Mono,monospace;color:#ccc;">{qty:.0f}</span>
<span style="font-family:IBM Plex Mono,monospace;color:#888;">${entry:.2f}</span>
<span style="font-family:IBM Plex Mono,monospace;color:#ccc;">${current:.2f}</span>
<span style="font-family:IBM Plex Mono,monospace;color:{pnl_color};">{("+" if pnl >= 0 else "")}${pnl:.2f}</span>
</div>
''', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:30px 16px;text-align:center;color:#555;font-size:11px;">No positions</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Orders panel
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Pending Orders</span>
</div>
<div style="padding:20px 16px;text-align:center;color:#555;font-size:11px;">No pending orders</div>
</div>
''', unsafe_allow_html=True)

with col_log:
    # Live Log
    st.markdown('''
<div style="background:#0a0a0a;border:1px solid #222;margin-top:1px;">
<div style="padding:10px 16px;border-bottom:1px solid #222;">
<span style="font-size:11px;color:#888;text-transform:uppercase;">Live Log</span>
</div>
<div style="padding:8px 16px;font-family:IBM Plex Mono,monospace;font-size:10px;line-height:1.6;height:350px;overflow-y:auto;background:#050505;">
''', unsafe_allow_html=True)

    logs = get_logs(40)
    for line in logs:
        line = line.strip()[:120]
        if not line:
            continue
        color = "#555"
        if "ERROR" in line.upper():
            color = "#ff4444"
        elif "SUCCESS" in line.upper() or "CONNECTED" in line.upper():
            color = "#00ff88"
        elif "WARNING" in line.upper():
            color = "#ffaa00"
        st.markdown(f'<div style="color:{color};padding:1px 0;">{line}</div>', unsafe_allow_html=True)

    if not logs:
        st.markdown('<div style="color:#555;">No logs available</div>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# Footer
st.markdown(f'''
<div style="position:fixed;bottom:0;left:0;right:0;padding:6px 16px;background:#0a0a0a;border-top:1px solid #222;display:flex;justify-content:space-between;font-size:10px;color:#555;">
<span>Bot Control v3.0</span>
<span>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
</div>
''', unsafe_allow_html=True)
