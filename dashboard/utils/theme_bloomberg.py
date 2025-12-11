"""
Bloomberg Terminal Theme
========================
Professional, minimalistic trading terminal interface.
Black background, orange/green accents, monospace fonts.
"""

import streamlit as st


def apply_bloomberg_theme():
    """
    Apply Bloomberg Terminal styling to Streamlit dashboard.

    Theme colors:
    - Background: #000000 (pure black)
    - Panel: #111111 (dark panel)
    - Orange: #ff6600 (Bloomberg orange - headers)
    - Green: #00ff00 (positive/buy)
    - Red: #ff3333 (negative/sell)
    - Yellow: #ffcc00 (warnings)
    - Blue: #0099ff (info)
    - White: #ffffff (primary text)
    - Gray: #666666 (secondary text)
    """

    st.markdown("""
    <style>
    /* ========================================================================
       BLOOMBERG TERMINAL THEME
       ======================================================================== */

    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

    :root {
        --bb-black: #000000;
        --bb-dark: #0a0a0a;
        --bb-panel: #111111;
        --bb-border: #1a1a1a;
        --bb-orange: #ff6600;
        --bb-green: #00ff00;
        --bb-red: #ff3333;
        --bb-yellow: #ffcc00;
        --bb-blue: #0099ff;
        --bb-white: #ffffff;
        --bb-gray: #666666;
        --bb-dim: #333333;
    }

    /* ========================================================================
       Base
       ======================================================================== */

    .main {
        background-color: var(--bb-black) !important;
    }

    .stApp {
        background-color: var(--bb-black) !important;
    }

    * {
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    }

    /* ========================================================================
       Typography
       ======================================================================== */

    h1, h2, h3, h4, h5, h6 {
        color: var(--bb-orange) !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        margin-bottom: 0.5rem !important;
    }

    h1 {
        font-size: 1.1rem !important;
        border-bottom: 1px solid var(--bb-orange) !important;
        padding-bottom: 0.3rem !important;
    }

    h2 {
        font-size: 0.95rem !important;
        color: var(--bb-yellow) !important;
        border-bottom: 1px solid var(--bb-border) !important;
        padding-bottom: 0.2rem !important;
    }

    h3 {
        font-size: 0.85rem !important;
        color: var(--bb-blue) !important;
    }

    p, span, div, label {
        color: var(--bb-white) !important;
        font-size: 0.8rem !important;
        line-height: 1.4 !important;
    }

    code, pre {
        background-color: var(--bb-panel) !important;
        color: var(--bb-green) !important;
        border: 1px solid var(--bb-border) !important;
        font-size: 0.75rem !important;
    }

    /* ========================================================================
       Sidebar
       ======================================================================== */

    [data-testid="stSidebar"] {
        background-color: var(--bb-dark) !important;
        border-right: 1px solid var(--bb-border) !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 0.8rem !important;
    }

    /* ========================================================================
       Buttons
       ======================================================================== */

    .stButton > button {
        background-color: var(--bb-panel) !important;
        color: var(--bb-orange) !important;
        border: 1px solid var(--bb-orange) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 0.4rem 1rem !important;
        border-radius: 0 !important;
        transition: all 0.1s ease !important;
    }

    .stButton > button:hover {
        background-color: var(--bb-orange) !important;
        color: var(--bb-black) !important;
    }

    .stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* ========================================================================
       Inputs
       ======================================================================== */

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: var(--bb-panel) !important;
        color: var(--bb-white) !important;
        border: 1px solid var(--bb-border) !important;
        border-radius: 0 !important;
        font-size: 0.8rem !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--bb-orange) !important;
        box-shadow: none !important;
    }

    .stSelectbox > div > div {
        background-color: var(--bb-panel) !important;
        border: 1px solid var(--bb-border) !important;
        border-radius: 0 !important;
    }

    /* ========================================================================
       Metrics
       ======================================================================== */

    [data-testid="stMetricValue"] {
        color: var(--bb-green) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--bb-gray) !important;
        font-size: 0.65rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.7rem !important;
    }

    div[data-testid="metric-container"] {
        background-color: var(--bb-panel) !important;
        border: 1px solid var(--bb-border) !important;
        border-left: 2px solid var(--bb-orange) !important;
        padding: 0.5rem !important;
        border-radius: 0 !important;
    }

    /* ========================================================================
       Tables/DataFrames
       ======================================================================== */

    .dataframe {
        background-color: var(--bb-dark) !important;
        border: 1px solid var(--bb-border) !important;
        font-size: 0.7rem !important;
        border-radius: 0 !important;
    }

    .dataframe th {
        background-color: var(--bb-panel) !important;
        color: var(--bb-orange) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.5rem !important;
        border-bottom: 1px solid var(--bb-orange) !important;
    }

    .dataframe td {
        background-color: var(--bb-dark) !important;
        color: var(--bb-white) !important;
        padding: 0.4rem !important;
        border-bottom: 1px solid var(--bb-border) !important;
    }

    .dataframe tr:hover td {
        background-color: var(--bb-panel) !important;
    }

    /* ========================================================================
       Alerts
       ======================================================================== */

    .stSuccess {
        background-color: rgba(0, 255, 0, 0.1) !important;
        border-left: 2px solid var(--bb-green) !important;
        color: var(--bb-green) !important;
        border-radius: 0 !important;
    }

    .stInfo {
        background-color: rgba(0, 153, 255, 0.1) !important;
        border-left: 2px solid var(--bb-blue) !important;
        color: var(--bb-blue) !important;
        border-radius: 0 !important;
    }

    .stWarning {
        background-color: rgba(255, 204, 0, 0.1) !important;
        border-left: 2px solid var(--bb-yellow) !important;
        color: var(--bb-yellow) !important;
        border-radius: 0 !important;
    }

    .stError {
        background-color: rgba(255, 51, 51, 0.1) !important;
        border-left: 2px solid var(--bb-red) !important;
        color: var(--bb-red) !important;
        border-radius: 0 !important;
    }

    /* ========================================================================
       Tabs
       ======================================================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background-color: var(--bb-dark) !important;
        border-bottom: 1px solid var(--bb-border) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--bb-gray) !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--bb-white) !important;
        background-color: var(--bb-panel) !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--bb-orange) !important;
        color: var(--bb-black) !important;
        font-weight: 600 !important;
    }

    /* ========================================================================
       Expanders
       ======================================================================== */

    .streamlit-expanderHeader {
        background-color: var(--bb-panel) !important;
        border: 1px solid var(--bb-border) !important;
        color: var(--bb-white) !important;
        font-size: 0.8rem !important;
        border-radius: 0 !important;
    }

    .streamlit-expanderContent {
        background-color: var(--bb-dark) !important;
        border: 1px solid var(--bb-border) !important;
        border-top: none !important;
        border-radius: 0 !important;
    }

    /* ========================================================================
       Charts
       ======================================================================== */

    .js-plotly-plot {
        background-color: var(--bb-dark) !important;
        border: 1px solid var(--bb-border) !important;
    }

    /* ========================================================================
       Scrollbar
       ======================================================================== */

    ::-webkit-scrollbar {
        width: 6px !important;
        height: 6px !important;
    }

    ::-webkit-scrollbar-track {
        background: var(--bb-dark) !important;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--bb-dim) !important;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--bb-gray) !important;
    }

    /* ========================================================================
       Dividers
       ======================================================================== */

    hr {
        border: none !important;
        border-top: 1px solid var(--bb-border) !important;
        margin: 1rem 0 !important;
    }

    /* ========================================================================
       Hide Streamlit elements
       ======================================================================== */

    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}

    /* ========================================================================
       Custom classes
       ======================================================================== */

    .bb-positive { color: var(--bb-green) !important; }
    .bb-negative { color: var(--bb-red) !important; }
    .bb-info { color: var(--bb-blue) !important; }
    .bb-warn { color: var(--bb-yellow) !important; }
    .bb-header { color: var(--bb-orange) !important; }
    .bb-muted { color: var(--bb-gray) !important; }

    .bb-panel {
        background-color: var(--bb-panel) !important;
        border: 1px solid var(--bb-border) !important;
        border-left: 2px solid var(--bb-orange) !important;
        padding: 0.75rem !important;
        margin: 0.5rem 0 !important;
    }

    .bb-log {
        background-color: var(--bb-dark) !important;
        border: 1px solid var(--bb-border) !important;
        padding: 0.5rem !important;
        font-size: 0.7rem !important;
        line-height: 1.5 !important;
        max-height: 400px !important;
        overflow-y: auto !important;
    }

    </style>
    """, unsafe_allow_html=True)


def get_plotly_theme() -> dict:
    """Get Plotly chart theme matching Bloomberg style."""
    return {
        'paper_bgcolor': '#0a0a0a',
        'plot_bgcolor': '#0a0a0a',
        'font': {
            'family': 'JetBrains Mono, monospace',
            'color': '#ffffff',
            'size': 10
        },
        'title': {
            'font': {
                'color': '#ff6600',
                'size': 12
            }
        },
        'xaxis': {
            'gridcolor': '#1a1a1a',
            'linecolor': '#1a1a1a',
            'tickfont': {'color': '#666666', 'size': 9}
        },
        'yaxis': {
            'gridcolor': '#1a1a1a',
            'linecolor': '#1a1a1a',
            'tickfont': {'color': '#666666', 'size': 9}
        },
        'legend': {
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#ffffff', 'size': 9}
        }
    }


# Color constants for use in code
BB_COLORS = {
    'black': '#000000',
    'dark': '#0a0a0a',
    'panel': '#111111',
    'border': '#1a1a1a',
    'orange': '#ff6600',
    'green': '#00ff00',
    'red': '#ff3333',
    'yellow': '#ffcc00',
    'blue': '#0099ff',
    'white': '#ffffff',
    'gray': '#666666',
    'dim': '#333333',
}
