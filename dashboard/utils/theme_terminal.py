"""
Modern Terminal Trading Theme
=============================
Professional dark terminal theme with neon accents.
Orion-style algorithmic trading dashboard aesthetic.

Design System:
- Background: Deep slate/near-black
- Surfaces: Glassy panels with subtle borders
- Primary: Cyan/teal for main actions
- Secondary: Purple for AI-related elements
- Success/Loss: Green/red for P&L
"""

import streamlit as st
from typing import Dict, Any


# =============================================================================
# DESIGN SYSTEM CONSTANTS
# =============================================================================

# Color Palette
COLORS = {
    # Backgrounds
    'bg_primary': '#020617',      # Near-black, deep slate
    'bg_secondary': '#0f172a',    # Slightly lighter for surfaces
    'bg_surface': '#1e293b',      # Card/panel backgrounds
    'bg_elevated': '#334155',     # Elevated surfaces (hovers, etc.)

    # Borders & Lines
    'border_subtle': '#1e293b',   # Subtle panel borders
    'border_default': '#334155',  # Default borders
    'border_strong': '#475569',   # Strong/focus borders

    # Primary Accent (Cyan/Teal)
    'primary': '#22d3ee',         # Cyan - main actions, key stats
    'primary_dim': '#0891b2',     # Dimmed primary
    'primary_glow': 'rgba(34, 211, 238, 0.15)',  # Glow effect

    # Secondary Accent (Purple)
    'secondary': '#a855f7',       # Purple - AI, ML, secondary states
    'secondary_dim': '#7c3aed',   # Dimmed secondary
    'secondary_glow': 'rgba(168, 85, 247, 0.15)',

    # Status Colors
    'success': '#22c55e',         # Green - positive P&L, buys
    'success_dim': '#16a34a',
    'success_glow': 'rgba(34, 197, 94, 0.15)',

    'danger': '#ef4444',          # Red - negative P&L, sells
    'danger_dim': '#dc2626',
    'danger_glow': 'rgba(239, 68, 68, 0.15)',

    'warning': '#f59e0b',         # Amber - warnings
    'warning_dim': '#d97706',

    'info': '#3b82f6',            # Blue - info
    'info_dim': '#2563eb',

    # Text
    'text_primary': '#f1f5f9',    # Off-white, primary text
    'text_secondary': '#94a3b8',  # Muted gray, secondary
    'text_muted': '#64748b',      # Very subtle, labels
    'text_disabled': '#475569',   # Disabled state
}

# Typography
FONTS = {
    'heading': "'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
    'body': "'Inter', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif",
    'mono': "'JetBrains Mono', 'SF Mono', 'Fira Code', 'Consolas', monospace",
}

FONT_SIZES = {
    'xs': '0.65rem',
    'sm': '0.75rem',
    'base': '0.85rem',
    'md': '0.95rem',
    'lg': '1.1rem',
    'xl': '1.25rem',
    '2xl': '1.5rem',
    '3xl': '2rem',
}

# Spacing
SPACING = {
    'xs': '0.25rem',
    'sm': '0.5rem',
    'md': '0.75rem',
    'lg': '1rem',
    'xl': '1.5rem',
    '2xl': '2rem',
    '3xl': '3rem',
}

# Border Radius
RADIUS = {
    'none': '0',
    'sm': '4px',
    'md': '6px',
    'lg': '8px',
    'xl': '12px',
    'full': '9999px',
}

# Shadows
SHADOWS = {
    'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
    'md': '0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3)',
    'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -4px rgba(0, 0, 0, 0.4)',
    'glow_primary': f'0 0 20px {COLORS["primary_glow"]}',
    'glow_success': f'0 0 15px {COLORS["success_glow"]}',
    'glow_danger': f'0 0 15px {COLORS["danger_glow"]}',
}


# =============================================================================
# THEME APPLICATION
# =============================================================================

def apply_terminal_theme():
    """
    Apply the modern terminal trading theme to Streamlit.
    Call this at the top of every page after st.set_page_config().
    """

    st.markdown(f"""
    <style>
    /* =======================================================================
       MODERN TERMINAL TRADING THEME
       ======================================================================= */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    :root {{
        /* Colors */
        --bg-primary: {COLORS['bg_primary']};
        --bg-secondary: {COLORS['bg_secondary']};
        --bg-surface: {COLORS['bg_surface']};
        --bg-elevated: {COLORS['bg_elevated']};

        --border-subtle: {COLORS['border_subtle']};
        --border-default: {COLORS['border_default']};
        --border-strong: {COLORS['border_strong']};

        --primary: {COLORS['primary']};
        --primary-dim: {COLORS['primary_dim']};
        --secondary: {COLORS['secondary']};
        --secondary-dim: {COLORS['secondary_dim']};

        --success: {COLORS['success']};
        --danger: {COLORS['danger']};
        --warning: {COLORS['warning']};
        --info: {COLORS['info']};

        --text-primary: {COLORS['text_primary']};
        --text-secondary: {COLORS['text_secondary']};
        --text-muted: {COLORS['text_muted']};

        /* Typography */
        --font-heading: {FONTS['heading']};
        --font-body: {FONTS['body']};
        --font-mono: {FONTS['mono']};
    }}

    /* =======================================================================
       BASE STYLES
       ======================================================================= */

    .main {{
        background-color: var(--bg-primary) !important;
    }}

    .stApp {{
        background-color: var(--bg-primary) !important;
    }}

    /* Hide default Streamlit elements */
    #MainMenu {{visibility: hidden !important;}}
    footer {{visibility: hidden !important;}}
    header {{visibility: hidden !important;}}

    /* =======================================================================
       TYPOGRAPHY
       ======================================================================= */

    * {{
        font-family: var(--font-body) !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--font-heading) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: {SPACING['md']} !important;
    }}

    h1 {{
        font-size: {FONT_SIZES['2xl']} !important;
        color: var(--primary) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        padding-bottom: {SPACING['sm']} !important;
    }}

    h2 {{
        font-size: {FONT_SIZES['xl']} !important;
        color: var(--text-primary) !important;
    }}

    h3 {{
        font-size: {FONT_SIZES['lg']} !important;
        color: var(--text-secondary) !important;
    }}

    p, span, div, label {{
        color: var(--text-primary) !important;
        font-size: {FONT_SIZES['base']} !important;
        line-height: 1.5 !important;
    }}

    code, pre, .stCodeBlock {{
        font-family: var(--font-mono) !important;
        background-color: var(--bg-surface) !important;
        color: var(--primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: {RADIUS['md']} !important;
        font-size: {FONT_SIZES['sm']} !important;
    }}

    /* =======================================================================
       SIDEBAR
       ======================================================================= */

    [data-testid="stSidebar"] {{
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }}

    [data-testid="stSidebar"] .stMarkdown {{
        padding: {SPACING['sm']} !important;
    }}

    /* Sidebar nav items */
    [data-testid="stSidebar"] .stButton > button {{
        background-color: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        text-align: left !important;
        padding: {SPACING['md']} {SPACING['lg']} !important;
        font-size: {FONT_SIZES['sm']} !important;
        font-weight: 500 !important;
        border-radius: {RADIUS['md']} !important;
        transition: all 0.15s ease !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: var(--bg-surface) !important;
        color: var(--primary) !important;
    }}

    /* =======================================================================
       BUTTONS
       ======================================================================= */

    .stButton > button {{
        background-color: var(--bg-surface) !important;
        color: var(--primary) !important;
        border: 1px solid var(--primary-dim) !important;
        font-family: var(--font-body) !important;
        font-size: {FONT_SIZES['sm']} !important;
        font-weight: 600 !important;
        padding: {SPACING['sm']} {SPACING['lg']} !important;
        border-radius: {RADIUS['md']} !important;
        transition: all 0.15s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    .stButton > button:hover {{
        background-color: var(--primary) !important;
        color: var(--bg-primary) !important;
        box-shadow: {SHADOWS['glow_primary']} !important;
    }}

    .stButton > button:active {{
        transform: scale(0.98) !important;
    }}

    /* Primary button variant */
    .stButton > button[kind="primary"] {{
        background-color: var(--primary) !important;
        color: var(--bg-primary) !important;
        border: none !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        background-color: var(--primary-dim) !important;
    }}

    /* =======================================================================
       INPUTS
       ======================================================================= */

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: {RADIUS['md']} !important;
        font-family: var(--font-mono) !important;
        font-size: {FONT_SIZES['sm']} !important;
        padding: {SPACING['sm']} {SPACING['md']} !important;
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px {COLORS['primary_glow']} !important;
    }}

    .stSelectbox > div > div {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: {RADIUS['md']} !important;
    }}

    .stSelectbox [data-baseweb="select"] {{
        background-color: var(--bg-surface) !important;
    }}

    .stSlider > div > div > div {{
        background-color: var(--border-default) !important;
    }}

    .stSlider [data-baseweb="slider"] div {{
        background-color: var(--primary) !important;
    }}

    /* =======================================================================
       METRICS
       ======================================================================= */

    [data-testid="stMetricValue"] {{
        font-family: var(--font-mono) !important;
        color: var(--primary) !important;
        font-size: {FONT_SIZES['2xl']} !important;
        font-weight: 700 !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--text-muted) !important;
        font-size: {FONT_SIZES['xs']} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-weight: 500 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-family: var(--font-mono) !important;
        font-size: {FONT_SIZES['sm']} !important;
    }}

    [data-testid="stMetricDeltaIcon-Up"] {{
        color: var(--success) !important;
    }}

    [data-testid="stMetricDeltaIcon-Down"] {{
        color: var(--danger) !important;
    }}

    div[data-testid="metric-container"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: {RADIUS['lg']} !important;
        padding: {SPACING['lg']} !important;
    }}

    /* =======================================================================
       TABLES / DATAFRAMES
       ======================================================================= */

    .dataframe {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: {RADIUS['lg']} !important;
        font-family: var(--font-mono) !important;
        font-size: {FONT_SIZES['xs']} !important;
    }}

    .dataframe th {{
        background-color: var(--bg-surface) !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: {FONT_SIZES['xs']} !important;
        letter-spacing: 0.05em !important;
        padding: {SPACING['md']} !important;
        border-bottom: 1px solid var(--border-default) !important;
    }}

    .dataframe td {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        padding: {SPACING['sm']} {SPACING['md']} !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }}

    .dataframe tr:hover td {{
        background-color: var(--bg-surface) !important;
    }}

    /* =======================================================================
       ALERTS
       ======================================================================= */

    .stSuccess {{
        background-color: {COLORS['success_glow']} !important;
        border: 1px solid var(--success) !important;
        border-left: 3px solid var(--success) !important;
        color: var(--success) !important;
        border-radius: {RADIUS['md']} !important;
    }}

    .stInfo {{
        background-color: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid var(--info) !important;
        border-left: 3px solid var(--info) !important;
        color: var(--info) !important;
        border-radius: {RADIUS['md']} !important;
    }}

    .stWarning {{
        background-color: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid var(--warning) !important;
        border-left: 3px solid var(--warning) !important;
        color: var(--warning) !important;
        border-radius: {RADIUS['md']} !important;
    }}

    .stError {{
        background-color: {COLORS['danger_glow']} !important;
        border: 1px solid var(--danger) !important;
        border-left: 3px solid var(--danger) !important;
        color: var(--danger) !important;
        border-radius: {RADIUS['md']} !important;
    }}

    /* =======================================================================
       TABS
       ======================================================================= */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0 !important;
        background-color: var(--bg-secondary) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        padding: 0 !important;
        border-radius: {RADIUS['lg']} {RADIUS['lg']} 0 0 !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: var(--text-muted) !important;
        border: none !important;
        padding: {SPACING['md']} {SPACING['xl']} !important;
        font-size: {FONT_SIZES['sm']} !important;
        font-weight: 500 !important;
        border-radius: 0 !important;
        transition: all 0.15s ease !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--text-primary) !important;
        background-color: var(--bg-surface) !important;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: var(--primary) !important;
        color: var(--bg-primary) !important;
        font-weight: 600 !important;
    }}

    .stTabs [data-baseweb="tab-panel"] {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 {RADIUS['lg']} {RADIUS['lg']} !important;
        padding: {SPACING['xl']} !important;
    }}

    /* =======================================================================
       EXPANDERS
       ======================================================================= */

    .streamlit-expanderHeader {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-primary) !important;
        font-size: {FONT_SIZES['sm']} !important;
        font-weight: 500 !important;
        border-radius: {RADIUS['md']} !important;
        padding: {SPACING['md']} {SPACING['lg']} !important;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: var(--primary) !important;
    }}

    .streamlit-expanderContent {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 {RADIUS['md']} {RADIUS['md']} !important;
        padding: {SPACING['lg']} !important;
    }}

    /* =======================================================================
       CHARTS (Plotly)
       ======================================================================= */

    .js-plotly-plot {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: {RADIUS['lg']} !important;
    }}

    /* =======================================================================
       SCROLLBAR
       ======================================================================= */

    ::-webkit-scrollbar {{
        width: 8px !important;
        height: 8px !important;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary) !important;
        border-radius: {RADIUS['full']} !important;
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--border-default) !important;
        border-radius: {RADIUS['full']} !important;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--border-strong) !important;
    }}

    /* =======================================================================
       DIVIDERS
       ======================================================================= */

    hr {{
        border: none !important;
        border-top: 1px solid var(--border-subtle) !important;
        margin: {SPACING['xl']} 0 !important;
    }}

    /* =======================================================================
       CUSTOM UTILITY CLASSES
       ======================================================================= */

    .text-success {{ color: var(--success) !important; }}
    .text-danger {{ color: var(--danger) !important; }}
    .text-warning {{ color: var(--warning) !important; }}
    .text-info {{ color: var(--info) !important; }}
    .text-primary {{ color: var(--primary) !important; }}
    .text-secondary {{ color: var(--secondary) !important; }}
    .text-muted {{ color: var(--text-muted) !important; }}

    .bg-surface {{ background-color: var(--bg-surface) !important; }}
    .bg-elevated {{ background-color: var(--bg-elevated) !important; }}

    .font-mono {{ font-family: var(--font-mono) !important; }}

    .border-glow-primary {{ box-shadow: 0 0 15px {COLORS['primary_glow']} !important; }}
    .border-glow-success {{ box-shadow: 0 0 15px {COLORS['success_glow']} !important; }}
    .border-glow-danger {{ box-shadow: 0 0 15px {COLORS['danger_glow']} !important; }}

    </style>
    """, unsafe_allow_html=True)


def get_plotly_theme() -> Dict[str, Any]:
    """
    Get Plotly chart theme matching the terminal style.

    Usage:
        fig.update_layout(**get_plotly_theme())
    """
    return {
        'paper_bgcolor': COLORS['bg_secondary'],
        'plot_bgcolor': COLORS['bg_secondary'],
        'font': {
            'family': FONTS['mono'],
            'color': COLORS['text_primary'],
            'size': 11
        },
        'title': {
            'font': {
                'color': COLORS['text_primary'],
                'size': 14,
                'family': FONTS['heading']
            },
            'x': 0,
            'xanchor': 'left'
        },
        'xaxis': {
            'gridcolor': COLORS['border_subtle'],
            'linecolor': COLORS['border_subtle'],
            'tickfont': {'color': COLORS['text_muted'], 'size': 10},
            'zerolinecolor': COLORS['border_default']
        },
        'yaxis': {
            'gridcolor': COLORS['border_subtle'],
            'linecolor': COLORS['border_subtle'],
            'tickfont': {'color': COLORS['text_muted'], 'size': 10},
            'zerolinecolor': COLORS['border_default']
        },
        'legend': {
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': COLORS['text_secondary'], 'size': 10}
        },
        'colorway': [
            COLORS['primary'],
            COLORS['secondary'],
            COLORS['success'],
            COLORS['danger'],
            COLORS['warning'],
            COLORS['info'],
        ],
        'margin': {'l': 50, 'r': 20, 't': 40, 'b': 40}
    }


def get_chart_colors() -> Dict[str, str]:
    """Get standard chart colors for consistency."""
    return {
        'line': COLORS['primary'],
        'line_secondary': COLORS['secondary'],
        'positive': COLORS['success'],
        'negative': COLORS['danger'],
        'fill': COLORS['primary_glow'],
        'grid': COLORS['border_subtle'],
        'text': COLORS['text_muted'],
    }


# =============================================================================
# PAGE CONFIGURATION HELPER
# =============================================================================

def set_terminal_page_config(
    title: str = "Trading Terminal",
    icon: str = "",
    layout: str = "wide"
):
    """
    Configure page with terminal theme settings.
    Must be called as first Streamlit command.

    Args:
        title: Page title
        icon: Page icon (emoji or path)
        layout: 'wide' or 'centered'
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )


# Export all theme constants for direct access
__all__ = [
    'COLORS',
    'FONTS',
    'FONT_SIZES',
    'SPACING',
    'RADIUS',
    'SHADOWS',
    'apply_terminal_theme',
    'get_plotly_theme',
    'get_chart_colors',
    'set_terminal_page_config',
]
