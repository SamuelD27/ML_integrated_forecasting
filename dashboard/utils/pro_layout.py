"""
Professional Trading Terminal Layout
====================================
Binance/TradingView/Kraken Pro inspired layout system.

Design References:
- Binance: Flexible grid, KPI stripe, dense data display
- TradingView: Collapsible sidebar, resizable panels, clean charts
- Kraken Pro: Widget-based layout, custom workspaces

Components:
- init_pro_page: Enhanced page initialization
- render_topbar: Professional top navigation bar
- render_sidebar_pro: Collapsible sidebar with spaces
- render_kpi_stripe: Horizontal KPI metrics bar
- tv_panel: TradingView-style widget panel
- grid_layout: Flexible grid system
- command_modal: Ctrl+K command palette

Usage:
    from dashboard.utils.pro_layout import (
        init_pro_page,
        render_topbar,
        render_sidebar_pro,
        render_kpi_stripe,
    )

    init_pro_page("Bot Control")
    render_topbar(current_space="bot")
    render_sidebar_pro()
"""

import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict, Literal, Callable, Any, Union
import pytz
from pathlib import Path


# =============================================================================
# DESIGN TOKENS - Binance-Inspired
# =============================================================================

COLORS = {
    # Backgrounds
    'bg_base': '#0b0e11',
    'bg_primary': '#0f1214',
    'bg_secondary': '#161a1e',
    'bg_surface': '#1e2329',
    'bg_elevated': '#2b3139',
    'bg_hover': '#363d47',
    'bg_active': '#474f5c',

    # Borders
    'border_subtle': '#1e2329',
    'border_default': '#2b3139',
    'border_strong': '#363d47',

    # Primary - Binance Teal
    'primary': '#00d4aa',
    'primary_dim': '#00b897',
    'primary_bright': '#0ff0c3',
    'primary_glow': 'rgba(0, 212, 170, 0.15)',
    'primary_bg': 'rgba(0, 212, 170, 0.08)',

    # Secondary - Purple for AI/ML
    'secondary': '#7b61ff',
    'secondary_dim': '#6651e3',
    'secondary_glow': 'rgba(123, 97, 255, 0.15)',
    'secondary_bg': 'rgba(123, 97, 255, 0.08)',

    # Status
    'success': '#0ecb81',
    'success_dim': '#0aa66a',
    'success_glow': 'rgba(14, 203, 129, 0.15)',
    'success_bg': 'rgba(14, 203, 129, 0.08)',

    'danger': '#f6465d',
    'danger_dim': '#d43d52',
    'danger_glow': 'rgba(246, 70, 93, 0.15)',
    'danger_bg': 'rgba(246, 70, 93, 0.08)',

    'warning': '#f0b90b',
    'warning_dim': '#d4a30a',
    'warning_glow': 'rgba(240, 185, 11, 0.15)',
    'warning_bg': 'rgba(240, 185, 11, 0.08)',

    'info': '#1e90ff',
    'info_dim': '#1a7ee0',

    # Text
    'text_primary': '#eaecef',
    'text_secondary': '#b7bdc6',
    'text_tertiary': '#848e9c',
    'text_muted': '#5e6673',
    'text_disabled': '#474d57',
}

FONTS = {
    'display': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    'body': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    'mono': "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
}

SIZES = {
    '2xs': '0.625rem',    # 10px
    'xs': '0.6875rem',    # 11px
    'sm': '0.75rem',      # 12px
    'base': '0.8125rem',  # 13px
    'md': '0.875rem',     # 14px
    'lg': '1rem',         # 16px
    'xl': '1.125rem',     # 18px
    '2xl': '1.25rem',     # 20px
    '3xl': '1.5rem',      # 24px
    '4xl': '2rem',        # 32px
}

SPACING = {
    '0': '0',
    '1': '2px',
    '2': '4px',
    '3': '6px',
    '4': '8px',
    '5': '10px',
    '6': '12px',
    '8': '16px',
    '10': '20px',
    '12': '24px',
    '16': '32px',
}

RADIUS = {
    'none': '0',
    'sm': '2px',
    'md': '4px',
    'lg': '6px',
    'xl': '8px',
    '2xl': '12px',
    'full': '9999px',
}


# =============================================================================
# NAVIGATION CONFIGURATION - Two Spaces
# =============================================================================

SPACES = {
    'bot': {
        'label': 'Bot Trading',
        'icon': '',
        'pages': [
            {'label': 'Dashboard', 'page': 'app.py', 'icon': ''},
            {'label': 'Bot Control', 'page': 'pages/bot_control.py', 'icon': ''},
            {'label': 'Positions', 'page': 'pages/positions.py', 'icon': ''},
            {'label': 'Trade History', 'page': 'pages/trade_history.py', 'icon': ''},
        ]
    },
    'analysis': {
        'label': 'Analysis',
        'icon': '',
        'pages': [
            {'label': 'Deep Analysis', 'page': 'pages/deep_analysis.py', 'icon': ''},
            {'label': 'Single Stock', 'page': 'pages/single_stock_analysis.py', 'icon': ''},
            {'label': 'Portfolio Builder', 'page': 'pages/portfolio_from_single_stock.py', 'icon': ''},
            {'label': 'Risk Analysis', 'page': 'pages/portfolio_risk_analysis.py', 'icon': ''},
            {'label': 'ML Portfolio', 'page': 'pages/ml_optimized_portfolio.py', 'icon': ''},
            {'label': 'Factor Analysis', 'page': 'pages/factor_analysis.py', 'icon': ''},
            {'label': 'Monte Carlo', 'page': 'pages/advanced_monte_carlo.py', 'icon': ''},
            {'label': 'Security Pricing', 'page': 'pages/security_pricing.py', 'icon': ''},
        ]
    }
}


# =============================================================================
# PAGE INITIALIZATION
# =============================================================================

def init_pro_page(
    title: str = "Trading Terminal",
    icon: str = "",
    layout: Literal["wide", "centered"] = "wide",
    sidebar_state: Literal["expanded", "collapsed", "auto"] = "collapsed"
) -> None:
    """
    Initialize page with professional trading terminal theme.

    Args:
        title: Browser tab title
        icon: Page icon
        layout: Page layout
        sidebar_state: Initial sidebar state
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state=sidebar_state,
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )

    # Load external CSS
    css_path = Path(__file__).parent.parent / "theme.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Additional inline styles for Streamlit-specific overrides
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    /* App container */
    .main {{
        background-color: {COLORS['bg_base']} !important;
    }}

    .stApp {{
        background-color: {COLORS['bg_base']} !important;
    }}

    .block-container {{
        padding: {SPACING['4']} {SPACING['8']} !important;
        max-width: 100% !important;
    }}

    /* Hide defaults */
    #MainMenu, footer, header {{
        visibility: hidden !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_primary']} !important;
        border-right: 1px solid {COLORS['border_subtle']} !important;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0 !important;
    }}

    /* Typography */
    * {{
        font-family: {FONTS['body']} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: {FONTS['display']} !important;
        color: {COLORS['text_primary']} !important;
        font-weight: 600 !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['bg_surface']} !important;
        color: {COLORS['text_primary']} !important;
        border: 1px solid {COLORS['border_default']} !important;
        font-family: {FONTS['body']} !important;
        font-size: {SIZES['sm']} !important;
        font-weight: 600 !important;
        padding: {SPACING['3']} {SPACING['6']} !important;
        border-radius: {RADIUS['md']} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
        transition: all 0.15s ease !important;
    }}

    .stButton > button:hover {{
        background-color: {COLORS['bg_hover']} !important;
        border-color: {COLORS['border_strong']} !important;
    }}

    .stButton > button[kind="primary"] {{
        background-color: {COLORS['primary']} !important;
        color: {COLORS['bg_base']} !important;
        border: none !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        background-color: {COLORS['primary_bright']} !important;
        box-shadow: 0 0 20px {COLORS['primary_glow']} !important;
    }}

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background-color: {COLORS['bg_surface']} !important;
        color: {COLORS['text_primary']} !important;
        border: 1px solid {COLORS['border_default']} !important;
        border-radius: {RADIUS['md']} !important;
        font-family: {FONTS['mono']} !important;
        font-size: {SIZES['sm']} !important;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 2px {COLORS['primary_glow']} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {COLORS['bg_primary']} !important;
        border-bottom: 1px solid {COLORS['border_default']} !important;
        gap: 0 !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_tertiary']} !important;
        font-size: {SIZES['sm']} !important;
        font-weight: 500 !important;
        padding: {SPACING['4']} {SPACING['8']} !important;
        border-bottom: 2px solid transparent !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: {COLORS['primary']} !important;
        border-bottom-color: {COLORS['primary']} !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        font-family: {FONTS['mono']} !important;
        font-size: {SIZES['2xl']} !important;
        font-weight: 700 !important;
        color: {COLORS['text_primary']} !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: {SIZES['2xs']} !important;
        color: {COLORS['text_muted']} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px !important;
        height: 6px !important;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS['bg_secondary']} !important;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS['border_strong']} !important;
        border-radius: {RADIUS['full']} !important;
    }}

    /* Divider */
    hr {{
        border: none !important;
        border-top: 1px solid {COLORS['border_subtle']} !important;
        margin: {SPACING['8']} 0 !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# TOP BAR - Binance Inspired
# =============================================================================

def render_topbar(
    current_space: Literal["bot", "analysis"] = "bot",
    environment: Literal["paper", "live"] = "paper",
    connection_status: Literal["online", "offline", "pending"] = "online",
    show_clock: bool = True,
    show_search: bool = True,
) -> Optional[str]:
    """
    Render professional top navigation bar.

    Args:
        current_space: Current navigation space
        environment: Trading environment
        connection_status: API connection status
        show_clock: Show market clock
        show_search: Show command search

    Returns:
        Space key if space changed, None otherwise
    """
    # Top bar container
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 {SPACING['8']};
        height: 48px;
        background-color: {COLORS['bg_primary']};
        border-bottom: 1px solid {COLORS['border_subtle']};
        margin: -{SPACING['4']} -{SPACING['8']} {SPACING['8']} -{SPACING['8']};
    ">
        <!-- Left: Logo + Space Tabs -->
        <div style="display: flex; align-items: center; gap: {SPACING['12']};">
            <div style="
                font-family: {FONTS['mono']};
                font-size: {SIZES['lg']};
                font-weight: 700;
                color: {COLORS['primary']};
                letter-spacing: -0.02em;
            ">
                TERMINAL
            </div>
        </div>

        <!-- Right: Status indicators -->
        <div style="display: flex; align-items: center; gap: {SPACING['8']};">
            <!-- Time -->
            <div style="
                font-family: {FONTS['mono']};
                font-size: {SIZES['sm']};
                color: {COLORS['text_tertiary']};
            ">
                {datetime.now().strftime('%H:%M:%S')}
            </div>

            <!-- Divider -->
            <div style="width: 1px; height: 20px; background: {COLORS['border_default']};"></div>

            <!-- Environment Badge -->
            <div style="
                padding: 2px 10px;
                background: {COLORS['warning_bg'] if environment == 'paper' else COLORS['danger_bg']};
                color: {COLORS['warning'] if environment == 'paper' else COLORS['danger']};
                border: 1px solid {COLORS['warning'] if environment == 'paper' else COLORS['danger']};
                border-radius: {RADIUS['full']};
                font-size: {SIZES['2xs']};
                font-weight: 600;
                text-transform: uppercase;
            ">
                {environment.upper()}
            </div>

            <!-- Connection Status -->
            <div style="display: flex; align-items: center; gap: {SPACING['2']};">
                <span style="
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: {COLORS['success'] if connection_status == 'online' else COLORS['danger'] if connection_status == 'offline' else COLORS['warning']};
                    {'box-shadow: 0 0 8px ' + COLORS['success'] + ';' if connection_status == 'online' else ''}
                "></span>
                <span style="
                    font-size: {SIZES['2xs']};
                    color: {COLORS['success'] if connection_status == 'online' else COLORS['danger'] if connection_status == 'offline' else COLORS['warning']};
                    text-transform: uppercase;
                ">
                    {connection_status}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Space tabs using Streamlit columns
    cols = st.columns([1, 1, 6])

    with cols[0]:
        if st.button("BOT", key="space_bot", use_container_width=True,
                     type="primary" if current_space == "bot" else "secondary"):
            return "bot"

    with cols[1]:
        if st.button("ANALYSIS", key="space_analysis", use_container_width=True,
                     type="primary" if current_space == "analysis" else "secondary"):
            return "analysis"

    return None


# =============================================================================
# SIDEBAR - TradingView Inspired
# =============================================================================

def render_sidebar_pro(
    current_space: Literal["bot", "analysis"] = "bot",
    current_page: Optional[str] = None,
    show_account: bool = True,
) -> None:
    """
    Render professional collapsible sidebar.

    Args:
        current_space: Current navigation space
        current_page: Current page path
        show_account: Show account status
    """
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div style="
            padding: {SPACING['6']} {SPACING['6']};
            border-bottom: 1px solid {COLORS['border_subtle']};
        ">
            <div style="
                font-family: {FONTS['mono']};
                font-size: {SIZES['md']};
                font-weight: 700;
                color: {COLORS['primary']};
            ">
                TRADING TERMINAL
            </div>
            <div style="
                font-size: {SIZES['2xs']};
                color: {COLORS['text_muted']};
                margin-top: 2px;
            ">
                Algorithmic Trading System
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='height: {SPACING['4']};'></div>", unsafe_allow_html=True)

        # Navigation items
        space_config = SPACES.get(current_space, SPACES['bot'])

        st.markdown(f"""
        <div style="
            padding: 0 {SPACING['6']};
            font-size: {SIZES['2xs']};
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: {SPACING['3']};
        ">
            {space_config['label']}
        </div>
        """, unsafe_allow_html=True)

        for item in space_config['pages']:
            is_active = current_page and item['page'] in current_page

            if is_active:
                st.markdown(f"""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: {SPACING['4']};
                    padding: {SPACING['4']} {SPACING['6']};
                    background: {COLORS['primary_bg']};
                    color: {COLORS['primary']};
                    border-left: 2px solid {COLORS['primary']};
                    font-size: {SIZES['sm']};
                    font-weight: 500;
                ">
                    <span>{item.get('icon', '')}</span>
                    <span>{item['label']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button(
                    f"{item.get('icon', '')} {item['label']}",
                    key=f"nav_{item['label']}",
                    use_container_width=True
                ):
                    st.switch_page(item['page'])

        # Account status at bottom
        if show_account:
            st.markdown(f"""
            <div style="
                position: fixed;
                bottom: 0;
                left: 0;
                width: 240px;
                padding: {SPACING['6']};
                background: {COLORS['bg_primary']};
                border-top: 1px solid {COLORS['border_subtle']};
            ">
                <div style="
                    font-size: {SIZES['2xs']};
                    color: {COLORS['text_muted']};
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    margin-bottom: {SPACING['2']};
                ">
                    Account
                </div>
                <div style="display: flex; align-items: center; gap: {SPACING['2']};">
                    <span style="
                        width: 8px;
                        height: 8px;
                        border-radius: 50%;
                        background: {COLORS['success']};
                        box-shadow: 0 0 6px {COLORS['success']};
                    "></span>
                    <span style="
                        font-size: {SIZES['sm']};
                        color: {COLORS['text_secondary']};
                    ">
                        Alpaca Paper
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# KPI STRIPE - Binance Market Stats Style
# =============================================================================

def render_kpi_stripe(
    kpis: List[Dict[str, Any]],
    key: Optional[str] = None
) -> None:
    """
    Render horizontal KPI metrics bar (Binance-style).

    Args:
        kpis: List of KPI dicts with keys:
            - label: KPI label
            - value: Main value
            - delta: Change value (optional)
            - delta_positive: True/False/None
        key: Unique key
    """
    items_html = ""
    for i, kpi in enumerate(kpis):
        delta_html = ""
        if kpi.get('delta'):
            delta_color = COLORS['success'] if kpi.get('delta_positive') else COLORS['danger'] if kpi.get('delta_positive') is False else COLORS['text_muted']
            delta_html = f"""
            <div style="
                font-family: {FONTS['mono']};
                font-size: {SIZES['2xs']};
                color: {delta_color};
            ">
                {kpi['delta']}
            </div>
            """

        value_color = COLORS['text_primary']
        if kpi.get('delta_positive') is True:
            value_color = COLORS['success']
        elif kpi.get('delta_positive') is False:
            value_color = COLORS['danger']

        items_html += f"""
        <div style="
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: fit-content;
        ">
            <div style="
                font-size: {SIZES['2xs']};
                color: {COLORS['text_muted']};
                text-transform: uppercase;
                letter-spacing: 0.05em;
            ">{kpi['label']}</div>
            <div style="
                font-family: {FONTS['mono']};
                font-size: {SIZES['md']};
                font-weight: 600;
                color: {value_color};
            ">{kpi['value']}</div>
            {delta_html}
        </div>
        """

        if i < len(kpis) - 1:
            items_html += f"""
            <div style="
                width: 1px;
                height: 32px;
                background: {COLORS['border_default']};
            "></div>
            """

    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: {SPACING['10']};
        padding: {SPACING['4']} {SPACING['8']};
        background: {COLORS['bg_primary']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['lg']};
        overflow-x: auto;
        margin-bottom: {SPACING['8']};
    ">
        {items_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PANEL - TradingView Widget Style
# =============================================================================

def tv_panel(
    title: str,
    content: str = "",
    height: Optional[str] = None,
    show_actions: bool = False,
    actions: Optional[List[Dict]] = None,
    accent_color: str = "primary",
    key: Optional[str] = None
) -> None:
    """
    Render TradingView-style widget panel.

    Args:
        title: Panel title
        content: HTML content
        height: Fixed height
        show_actions: Show action buttons
        actions: List of action dicts [{'label': str, 'icon': str}]
        accent_color: Accent color key
        key: Unique key
    """
    accent = COLORS.get(accent_color, COLORS['primary'])
    height_style = f"height: {height}; overflow-y: auto;" if height else ""

    actions_html = ""
    if show_actions and actions:
        action_btns = "".join([f"""
        <button style="
            background: transparent;
            border: none;
            color: {COLORS['text_tertiary']};
            cursor: pointer;
            padding: {SPACING['2']};
            font-size: {SIZES['sm']};
        ">{a.get('icon', '')} {a.get('label', '')}</button>
        """ for a in actions])

        actions_html = f"""
        <div style="display: flex; gap: {SPACING['2']};">
            {action_btns}
        </div>
        """

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border_default']};
        border-radius: {RADIUS['lg']};
        overflow: hidden;
        {height_style}
    ">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: {SPACING['4']} {SPACING['6']};
            background: {COLORS['bg_surface']};
            border-bottom: 1px solid {COLORS['border_default']};
        ">
            <div style="
                font-size: {SIZES['sm']};
                font-weight: 600;
                color: {COLORS['text_secondary']};
                text-transform: uppercase;
                letter-spacing: 0.05em;
            ">
                {title}
            </div>
            {actions_html}
        </div>
        <div style="padding: {SPACING['6']};">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# STAT TILE V2 - Binance Metric Style
# =============================================================================

def stat_tile_v2(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: Optional[bool] = None,
    accent: Literal["primary", "success", "danger", "warning", "neutral"] = "primary",
    size: Literal["sm", "md", "lg"] = "md",
    mini_chart: Optional[str] = None,  # SVG or base64 image
    key: Optional[str] = None
) -> None:
    """
    Render Binance-style stat tile with optional mini chart.

    Args:
        label: Metric label
        value: Main value
        delta: Change value
        delta_positive: True/False/None
        accent: Accent color
        size: Tile size
        mini_chart: Optional mini chart SVG
        key: Unique key
    """
    accent_colors = {
        'primary': COLORS['primary'],
        'success': COLORS['success'],
        'danger': COLORS['danger'],
        'warning': COLORS['warning'],
        'neutral': COLORS['text_secondary'],
    }
    accent_color = accent_colors.get(accent, COLORS['primary'])

    sizes_config = {
        'sm': {'value': SIZES['lg'], 'label': SIZES['2xs'], 'padding': SPACING['4']},
        'md': {'value': SIZES['2xl'], 'label': SIZES['2xs'], 'padding': SPACING['6']},
        'lg': {'value': SIZES['3xl'], 'label': SIZES['xs'], 'padding': SPACING['8']},
    }
    config = sizes_config.get(size, sizes_config['md'])

    delta_html = ""
    if delta:
        delta_color = COLORS['success'] if delta_positive else COLORS['danger'] if delta_positive is False else COLORS['text_muted']
        delta_html = f"""
        <div style="
            font-family: {FONTS['mono']};
            font-size: {SIZES['sm']};
            color: {delta_color};
            margin-top: {SPACING['2']};
        ">{delta}</div>
        """

    mini_chart_html = ""
    if mini_chart:
        mini_chart_html = f"""
        <div style="
            position: absolute;
            right: {SPACING['4']};
            bottom: {SPACING['4']};
            opacity: 0.3;
        ">{mini_chart}</div>
        """

    st.markdown(f"""
    <div style="
        position: relative;
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_default']};
        border-left: 3px solid {accent_color};
        border-radius: {RADIUS['lg']};
        padding: {config['padding']};
        overflow: hidden;
    ">
        <div style="
            font-size: {config['label']};
            font-weight: 500;
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: {SPACING['2']};
        ">{label}</div>
        <div style="
            font-family: {FONTS['mono']};
            font-size: {config['value']};
            font-weight: 700;
            color: {accent_color};
            line-height: 1.2;
        ">{value}</div>
        {delta_html}
        {mini_chart_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MINI LOG TERMINAL
# =============================================================================

def mini_log_terminal(
    logs: List[str],
    height: str = "200px",
    key: Optional[str] = None
) -> None:
    """
    Render compact log terminal.

    Args:
        logs: Log lines
        height: Terminal height
        key: Unique key
    """
    def format_line(line: str) -> str:
        line = line.strip()
        if not line:
            return ""

        if "ERROR" in line or "CRITICAL" in line:
            color = COLORS['danger']
        elif "WARNING" in line:
            color = COLORS['warning']
        elif "SUCCESS" in line or "CONNECTED" in line or "BUY" in line:
            color = COLORS['success']
        elif "SELL" in line:
            color = COLORS['danger']
        else:
            color = COLORS['text_tertiary']

        return f'<div style="color: {color}; padding: 1px 0; font-size: {SIZES["2xs"]}; line-height: 1.5;">{line}</div>'

    formatted = [format_line(log) for log in logs if log.strip()]
    log_html = "\n".join(formatted[-50:])  # Last 50 lines

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_base']};
        border: 1px solid {COLORS['border_default']};
        border-radius: {RADIUS['md']};
        padding: {SPACING['4']};
        height: {height};
        overflow-y: auto;
        font-family: {FONTS['mono']};
    ">
        {log_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# QUICK ACTION BAR
# =============================================================================

def quick_action_bar(
    actions: List[Dict[str, Any]],
    key: Optional[str] = None
) -> Optional[str]:
    """
    Render quick action toolbar.

    Args:
        actions: List of action dicts [{'label': str, 'icon': str, 'key': str, 'type': str}]
        key: Unique key

    Returns:
        Key of clicked action or None
    """
    cols = st.columns(len(actions))
    clicked = None

    for i, action in enumerate(actions):
        with cols[i]:
            btn_type = "primary" if action.get('type') == 'primary' else "secondary"
            if st.button(
                f"{action.get('icon', '')} {action['label']}",
                key=action.get('key', f"action_{i}"),
                use_container_width=True,
                type=btn_type
            ):
                clicked = action.get('key', action['label'])

    return clicked


# =============================================================================
# MARKET CLOCKS - Compact Version
# =============================================================================

def render_market_clocks_compact(
    markets: Optional[List[str]] = None
) -> None:
    """
    Render compact market clocks strip.

    Args:
        markets: List of market codes (NY, LON, TKY, HK)
    """
    market_config = {
        'NY': {'tz': 'America/New_York', 'hours': (9, 30, 16, 0), 'label': 'NYSE'},
        'LON': {'tz': 'Europe/London', 'hours': (8, 0, 16, 30), 'label': 'LSE'},
        'TKY': {'tz': 'Asia/Tokyo', 'hours': (9, 0, 15, 0), 'label': 'TSE'},
        'HK': {'tz': 'Asia/Hong_Kong', 'hours': (9, 30, 16, 0), 'label': 'HKG'},
    }

    if markets is None:
        markets = list(market_config.keys())

    clocks_html = ""
    for code in markets:
        if code not in market_config:
            continue

        config = market_config[code]
        tz = pytz.timezone(config['tz'])
        local_time = datetime.now(tz)

        is_open = False
        if local_time.weekday() < 5:
            open_h, open_m, close_h, close_m = config['hours']
            market_open = local_time.replace(hour=open_h, minute=open_m, second=0)
            market_close = local_time.replace(hour=close_h, minute=close_m, second=0)
            is_open = market_open <= local_time <= market_close

        status_color = COLORS['success'] if is_open else COLORS['danger']

        clocks_html += f"""
        <div style="
            display: flex;
            align-items: center;
            gap: {SPACING['3']};
            padding: {SPACING['2']} {SPACING['4']};
            background: {COLORS['bg_surface']};
            border-radius: {RADIUS['md']};
        ">
            <span style="
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: {status_color};
            "></span>
            <span style="
                font-size: {SIZES['2xs']};
                color: {COLORS['text_muted']};
            ">{config['label']}</span>
            <span style="
                font-family: {FONTS['mono']};
                font-size: {SIZES['sm']};
                color: {COLORS['text_primary']};
            ">{local_time.strftime('%H:%M')}</span>
        </div>
        """

    st.markdown(f"""
    <div style="
        display: flex;
        gap: {SPACING['3']};
        margin-bottom: {SPACING['6']};
    ">
        {clocks_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# LOADING STATES
# =============================================================================

def loading_skeleton(
    height: str = "100px",
    width: str = "100%",
    key: Optional[str] = None
) -> None:
    """
    Render a loading skeleton placeholder.

    Args:
        height: Skeleton height
        width: Skeleton width
        key: Unique key
    """
    st.markdown(f"""
    <div style="
        width: {width};
        height: {height};
        background: linear-gradient(
            90deg,
            {COLORS['bg_surface']} 0%,
            {COLORS['bg_elevated']} 50%,
            {COLORS['bg_surface']} 100%
        );
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: {RADIUS['lg']};
    "></div>
    <style>
    @keyframes shimmer {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}
    </style>
    """, unsafe_allow_html=True)


def loading_spinner(
    message: str = "Loading...",
    size: Literal["sm", "md", "lg"] = "md",
    key: Optional[str] = None
) -> None:
    """
    Render a professional loading spinner.

    Args:
        message: Loading message
        size: Spinner size
        key: Unique key
    """
    sizes = {'sm': '24px', 'md': '40px', 'lg': '56px'}
    spinner_size = sizes.get(size, sizes['md'])

    st.markdown(f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: {SPACING['10']};
        gap: {SPACING['4']};
    ">
        <div style="
            width: {spinner_size};
            height: {spinner_size};
            border: 3px solid {COLORS['bg_elevated']};
            border-top-color: {COLORS['primary']};
            border-radius: 50%;
            animation: spin 1s linear infinite;
        "></div>
        <div style="
            font-size: {SIZES['sm']};
            color: {COLORS['text_muted']};
        ">{message}</div>
    </div>
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# COMMAND PALETTE
# =============================================================================

def render_command_palette(key: Optional[str] = None) -> Optional[str]:
    """
    Render command palette modal (Ctrl+K style).

    Returns:
        Selected command or None
    """
    # Initialize state
    if 'command_palette_open' not in st.session_state:
        st.session_state.command_palette_open = False

    # Command palette CSS and JS
    st.markdown(f"""
    <style>
    .command-palette-overlay {{
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        z-index: 1000;
        backdrop-filter: blur(4px);
    }}

    .command-palette-overlay.open {{
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding-top: 100px;
    }}

    .command-palette {{
        width: 560px;
        max-width: 90vw;
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border_default']};
        border-radius: {RADIUS['xl']};
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        overflow: hidden;
    }}

    .command-palette-input {{
        width: 100%;
        padding: {SPACING['6']};
        background: transparent;
        border: none;
        border-bottom: 1px solid {COLORS['border_default']};
        color: {COLORS['text_primary']};
        font-size: {SIZES['md']};
        font-family: {FONTS['body']};
        outline: none;
    }}

    .command-palette-input::placeholder {{
        color: {COLORS['text_muted']};
    }}

    .command-list {{
        max-height: 400px;
        overflow-y: auto;
    }}

    .command-item {{
        display: flex;
        align-items: center;
        gap: {SPACING['4']};
        padding: {SPACING['4']} {SPACING['6']};
        cursor: pointer;
        transition: background 0.1s;
    }}

    .command-item:hover {{
        background: {COLORS['bg_hover']};
    }}

    .command-item.selected {{
        background: {COLORS['primary_bg']};
    }}

    .command-icon {{
        width: 20px;
        text-align: center;
        color: {COLORS['text_muted']};
    }}

    .command-label {{
        flex: 1;
        font-size: {SIZES['sm']};
        color: {COLORS['text_primary']};
    }}

    .command-shortcut {{
        font-family: {FONTS['mono']};
        font-size: {SIZES['2xs']};
        color: {COLORS['text_muted']};
        background: {COLORS['bg_surface']};
        padding: 2px 6px;
        border-radius: {RADIUS['sm']};
    }}

    .command-category {{
        padding: {SPACING['3']} {SPACING['6']};
        font-size: {SIZES['2xs']};
        font-weight: 600;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        background: {COLORS['bg_surface']};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Keyboard shortcut hint in topbar
    return None


# =============================================================================
# COMPACT MODE
# =============================================================================

def get_compact_mode() -> bool:
    """Get current compact mode state."""
    if 'compact_mode' not in st.session_state:
        st.session_state.compact_mode = False
    return st.session_state.compact_mode


def set_compact_mode(enabled: bool) -> None:
    """Set compact mode state."""
    st.session_state.compact_mode = enabled


def compact_toggle(key: Optional[str] = None) -> bool:
    """
    Render compact mode toggle.

    Returns:
        Current compact mode state
    """
    compact = get_compact_mode()

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.checkbox("Compact", value=compact, key=key or "compact_toggle"):
            set_compact_mode(True)
        else:
            set_compact_mode(False)

    return get_compact_mode()


# =============================================================================
# TOOLTIP SYSTEM
# =============================================================================

def tooltip(
    content: str,
    text: str,
    position: Literal["top", "bottom", "left", "right"] = "top",
    key: Optional[str] = None
) -> None:
    """
    Render content with tooltip.

    Args:
        content: HTML content to wrap
        text: Tooltip text
        position: Tooltip position
        key: Unique key
    """
    positions = {
        'top': 'bottom: 100%; left: 50%; transform: translateX(-50%); margin-bottom: 8px;',
        'bottom': 'top: 100%; left: 50%; transform: translateX(-50%); margin-top: 8px;',
        'left': 'right: 100%; top: 50%; transform: translateY(-50%); margin-right: 8px;',
        'right': 'left: 100%; top: 50%; transform: translateY(-50%); margin-left: 8px;',
    }
    pos_style = positions.get(position, positions['top'])

    unique_id = key or f"tooltip_{hash(content)}"

    st.markdown(f"""
    <div class="tooltip-wrapper" style="position: relative; display: inline-block;">
        {content}
        <div class="tooltip-text" style="
            visibility: hidden;
            opacity: 0;
            position: absolute;
            {pos_style}
            background: {COLORS['bg_elevated']};
            color: {COLORS['text_primary']};
            padding: {SPACING['2']} {SPACING['4']};
            border-radius: {RADIUS['md']};
            font-size: {SIZES['2xs']};
            white-space: nowrap;
            z-index: 1000;
            transition: opacity 0.15s, visibility 0.15s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        ">{text}</div>
    </div>
    <style>
    .tooltip-wrapper:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# CONTEXTUAL PANEL (Right Side)
# =============================================================================

def contextual_panel(
    title: str,
    content: str = "",
    width: str = "320px",
    show: bool = True,
    key: Optional[str] = None
) -> None:
    """
    Render a contextual side panel (TradingView-style).

    Args:
        title: Panel title
        content: HTML content
        width: Panel width
        show: Whether to show the panel
        key: Unique key
    """
    if not show:
        return

    st.markdown(f"""
    <div style="
        position: fixed;
        top: 60px;
        right: 0;
        width: {width};
        height: calc(100vh - 60px);
        background: {COLORS['bg_secondary']};
        border-left: 1px solid {COLORS['border_default']};
        z-index: 100;
        overflow-y: auto;
    ">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: {SPACING['4']} {SPACING['6']};
            border-bottom: 1px solid {COLORS['border_default']};
            background: {COLORS['bg_surface']};
        ">
            <span style="
                font-size: {SIZES['sm']};
                font-weight: 600;
                color: {COLORS['text_secondary']};
                text-transform: uppercase;
                letter-spacing: 0.05em;
            ">{title}</span>
            <button style="
                background: transparent;
                border: none;
                color: {COLORS['text_muted']};
                cursor: pointer;
                font-size: {SIZES['md']};
            ">×</button>
        </div>
        <div style="padding: {SPACING['6']};">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SECTION CARD
# =============================================================================

def section_card(
    title: str,
    content: str = "",
    subtitle: Optional[str] = None,
    actions: Optional[List[Dict]] = None,
    collapsible: bool = False,
    collapsed: bool = False,
    accent_color: str = "primary",
    key: Optional[str] = None
) -> None:
    """
    Render a professional section card with header and body.

    Args:
        title: Section title
        content: HTML content
        subtitle: Optional subtitle
        actions: Header action buttons
        collapsible: Whether the card can be collapsed
        collapsed: Initial collapsed state
        accent_color: Accent color key
        key: Unique key
    """
    accent = COLORS.get(accent_color, COLORS['primary'])

    subtitle_html = ""
    if subtitle:
        subtitle_html = f"""
        <div style="
            font-size: {SIZES['2xs']};
            color: {COLORS['text_muted']};
            margin-top: 2px;
        ">{subtitle}</div>
        """

    actions_html = ""
    if actions:
        btns = "".join([f"""
        <button style="
            background: transparent;
            border: none;
            color: {COLORS['text_tertiary']};
            cursor: pointer;
            padding: {SPACING['2']};
            font-size: {SIZES['sm']};
            transition: color 0.15s;
        ">{a.get('icon', '')} {a.get('label', '')}</button>
        """ for a in actions])
        actions_html = f"<div style='display: flex; gap: {SPACING['2']};'>{btns}</div>"

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border_default']};
        border-radius: {RADIUS['lg']};
        overflow: hidden;
        margin-bottom: {SPACING['4']};
    ">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: {SPACING['4']} {SPACING['6']};
            background: {COLORS['bg_surface']};
            border-bottom: 1px solid {COLORS['border_default']};
            border-left: 3px solid {accent};
        ">
            <div>
                <div style="
                    font-size: {SIZES['sm']};
                    font-weight: 600;
                    color: {COLORS['text_secondary']};
                ">{title}</div>
                {subtitle_html}
            </div>
            {actions_html}
        </div>
        <div style="padding: {SPACING['6']};">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# METRIC ROW PRO (Dense Horizontal KPIs)
# =============================================================================

def metric_row_pro(
    metrics: List[Dict[str, Any]],
    key: Optional[str] = None
) -> None:
    """
    Render a dense horizontal metric row (Binance orderbook summary style).

    Args:
        metrics: List of metric dicts with keys:
            - label: Metric label
            - value: Main value
            - color: Optional color override
        key: Unique key
    """
    items_html = ""
    for i, m in enumerate(metrics):
        color = m.get('color', COLORS['text_primary'])
        if m.get('positive') is True:
            color = COLORS['success']
        elif m.get('positive') is False:
            color = COLORS['danger']

        items_html += f"""
        <div style="display: flex; align-items: center; gap: {SPACING['2']};">
            <span style="
                font-size: {SIZES['2xs']};
                color: {COLORS['text_muted']};
                text-transform: uppercase;
            ">{m['label']}</span>
            <span style="
                font-family: {FONTS['mono']};
                font-size: {SIZES['sm']};
                font-weight: 600;
                color: {color};
            ">{m['value']}</span>
        </div>
        """
        if i < len(metrics) - 1:
            items_html += f"<div style='width: 1px; height: 16px; background: {COLORS['border_default']};'></div>"

    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: {SPACING['6']};
        padding: {SPACING['3']} {SPACING['4']};
        background: {COLORS['bg_surface']};
        border-radius: {RADIUS['md']};
        overflow-x: auto;
    ">
        {items_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# EMPTY STATE
# =============================================================================

def empty_state(
    title: str,
    message: str,
    icon: str = "",
    action_label: Optional[str] = None,
    action_key: Optional[str] = None,
    key: Optional[str] = None
) -> bool:
    """
    Render a professional empty state placeholder.

    Args:
        title: Empty state title
        message: Description message
        icon: Icon emoji
        action_label: Optional action button label
        action_key: Action button key

    Returns:
        True if action button was clicked
    """
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: {SPACING['16']} {SPACING['8']};
        background: {COLORS['bg_secondary']};
        border: 1px dashed {COLORS['border_default']};
        border-radius: {RADIUS['lg']};
    ">
        <div style="font-size: 48px; margin-bottom: {SPACING['4']}; opacity: 0.3;">{icon}</div>
        <div style="
            font-size: {SIZES['md']};
            font-weight: 600;
            color: {COLORS['text_secondary']};
            margin-bottom: {SPACING['2']};
        ">{title}</div>
        <div style="
            font-size: {SIZES['sm']};
            color: {COLORS['text_muted']};
            max-width: 300px;
            margin: 0 auto;
        ">{message}</div>
    </div>
    """, unsafe_allow_html=True)

    if action_label and action_key:
        return st.button(action_label, key=action_key, use_container_width=True)
    return False


# =============================================================================
# ERROR STATE
# =============================================================================

def error_state(
    title: str,
    message: str,
    suggestion: Optional[str] = None,
    key: Optional[str] = None
) -> None:
    """
    Render a professional error state.

    Args:
        title: Error title
        message: Error message
        suggestion: Optional suggestion for fixing
        key: Unique key
    """
    suggestion_html = ""
    if suggestion:
        suggestion_html = f"""
        <div style="
            margin-top: {SPACING['4']};
            padding: {SPACING['3']} {SPACING['4']};
            background: {COLORS['bg_surface']};
            border-radius: {RADIUS['md']};
            font-size: {SIZES['sm']};
            color: {COLORS['text_secondary']};
        ">
            <span style="color: {COLORS['primary']};">Tip:</span> {suggestion}
        </div>
        """

    st.markdown(f"""
    <div style="
        background: {COLORS['danger_bg']};
        border: 1px solid {COLORS['danger']};
        border-left: 3px solid {COLORS['danger']};
        border-radius: {RADIUS['lg']};
        padding: {SPACING['6']};
    ">
        <div style="display: flex; align-items: flex-start; gap: {SPACING['4']};">
            <div style="
                font-size: 24px;
                color: {COLORS['danger']};
            "></div>
            <div style="flex: 1;">
                <div style="
                    font-size: {SIZES['md']};
                    font-weight: 600;
                    color: {COLORS['danger']};
                    margin-bottom: {SPACING['2']};
                ">{title}</div>
                <div style="
                    font-size: {SIZES['sm']};
                    color: {COLORS['text_secondary']};
                ">{message}</div>
                {suggestion_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Design Tokens
    'COLORS',
    'FONTS',
    'SIZES',
    'SPACING',
    'RADIUS',
    'SPACES',
    # Page Layout
    'init_pro_page',
    'render_topbar',
    'render_sidebar_pro',
    'render_kpi_stripe',
    # Panels & Cards
    'tv_panel',
    'section_card',
    'contextual_panel',
    # Metrics
    'stat_tile_v2',
    'metric_row_pro',
    # Utilities
    'mini_log_terminal',
    'quick_action_bar',
    'render_market_clocks_compact',
    # Loading States
    'loading_skeleton',
    'loading_spinner',
    # Command Palette
    'render_command_palette',
    # Compact Mode
    'get_compact_mode',
    'set_compact_mode',
    'compact_toggle',
    # Tooltips
    'tooltip',
    # States
    'empty_state',
    'error_state',
]
