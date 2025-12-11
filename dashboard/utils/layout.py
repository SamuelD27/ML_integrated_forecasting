"""
Unified Layout System
=====================
Provides consistent layout structure across all dashboard pages.

Components:
- init_page: Page configuration and theme application
- render_sidebar: Consistent navigation sidebar
- page_header: Standardized page header with status indicators
- app_shell: Main content wrapper
- nav_config: Centralized navigation configuration

Usage:
    from dashboard.utils.layout import init_page, render_sidebar, page_header

    # At top of page:
    init_page("Page Title")
    render_sidebar()
    page_header("Page Title", subtitle="Description")

    # Then render your page content
"""

import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict, Literal, Callable
import pytz

from .theme_terminal import (
    apply_terminal_theme,
    COLORS,
    FONTS,
    FONT_SIZES,
    SPACING,
    RADIUS,
)


# =============================================================================
# NAVIGATION CONFIGURATION
# =============================================================================

NAV_CONFIG = {
    'main': [
        {'label': 'Dashboard', 'page': 'app.py', 'icon': ''},
        {'label': 'Bot Control', 'page': 'pages/bot_control.py', 'icon': ''},
        {'label': 'Deep Analysis', 'page': 'pages/deep_analysis.py', 'icon': ''},
    ],
    'analysis': [
        {'label': 'Single Stock', 'page': 'pages/single_stock_analysis.py', 'icon': ''},
        {'label': 'Portfolio Builder', 'page': 'pages/portfolio_from_single_stock.py', 'icon': ''},
        {'label': 'Risk Analysis', 'page': 'pages/portfolio_risk_analysis.py', 'icon': ''},
        {'label': 'ML Portfolio', 'page': 'pages/ml_optimized_portfolio.py', 'icon': ''},
    ],
    'shortcuts': [
        {'label': 'Factor Analysis', 'page': 'pages/factor_analysis.py'},
        {'label': 'Monte Carlo', 'page': 'pages/advanced_monte_carlo.py'},
        {'label': 'Security Pricing', 'page': 'pages/security_pricing.py'},
        {'label': 'Galaxy Portfolio', 'page': 'pages/portfolio_from_galaxy.py'},
    ]
}


def get_current_page() -> str:
    """Get the current page path from query params or default."""
    try:
        # Get the current script path
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_file = frame.f_back.f_back.f_globals.get('__file__', '')
            if 'pages/' in caller_file:
                return 'pages/' + caller_file.split('pages/')[-1]
            elif 'app.py' in caller_file:
                return 'app.py'
    except:
        pass
    return ''


# =============================================================================
# PAGE INITIALIZATION
# =============================================================================

def init_page(
    title: str = "Trading Terminal",
    icon: str = "",
    layout: Literal["wide", "centered"] = "wide",
    sidebar_state: Literal["expanded", "collapsed", "auto"] = "expanded"
) -> None:
    """
    Initialize page with configuration and theme.
    Must be called as the first Streamlit command.

    Args:
        title: Browser tab title
        icon: Page icon (emoji)
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

    apply_terminal_theme()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar(
    current_page: Optional[str] = None,
    show_account: bool = True,
    show_analysis_tools: bool = True
) -> None:
    """
    Render the unified sidebar navigation.

    Args:
        current_page: Current page path for highlighting (auto-detected if None)
        show_account: Show account status at bottom
        show_analysis_tools: Show analysis tools section
    """
    if current_page is None:
        current_page = get_current_page()

    with st.sidebar:
        # Logo / Brand
        st.markdown(f"""
        <div style="
            padding: {SPACING['lg']} {SPACING['md']};
            border-bottom: 1px solid {COLORS['border_subtle']};
            margin-bottom: {SPACING['lg']};
        ">
            <div style="
                font-family: {FONTS['mono']};
                font-size: {FONT_SIZES['lg']};
                font-weight: 700;
                color: {COLORS['primary']};
                letter-spacing: -0.02em;
            ">
                TRADING TERMINAL
            </div>
            <div style="
                font-size: {FONT_SIZES['xs']};
                color: {COLORS['text_muted']};
                margin-top: 4px;
            ">
                Algorithmic Trading System
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main Navigation
        _render_nav_section("Navigation", NAV_CONFIG['main'], current_page)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Analysis Tools (collapsible or direct)
        if show_analysis_tools:
            _render_nav_section("Analysis Tools", NAV_CONFIG['analysis'], current_page)

        # Account status at bottom
        if show_account:
            _render_account_status()


def _render_nav_section(
    title: str,
    items: List[Dict],
    current_page: str
) -> None:
    """Render a navigation section with title and items."""
    st.markdown(f"""
    <div style="
        font-size: {FONT_SIZES['xs']};
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0 {SPACING['md']};
        margin-bottom: {SPACING['sm']};
    ">
        {title}
    </div>
    """, unsafe_allow_html=True)

    for item in items:
        is_active = current_page and item['page'] in current_page
        icon = item.get('icon', '')
        label = f"{icon} {item['label']}" if icon else f"  {item['label']}"

        # Use CSS classes for active state styling
        if is_active:
            st.markdown(f"""
            <div style="
                background: {COLORS['bg_surface']};
                border-left: 3px solid {COLORS['primary']};
                padding: {SPACING['md']} {SPACING['lg']};
                margin: 2px 0;
                color: {COLORS['primary']};
                font-size: {FONT_SIZES['sm']};
                font-weight: 600;
            ">
                {label}
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(
                label,
                key=f"nav_{item['label']}_{title}",
                use_container_width=True,
            ):
                st.switch_page(item['page'])


def _render_account_status() -> None:
    """Render account status section at bottom of sidebar."""
    # Check if Alpaca is connected
    connected = True  # Would check actual connection here
    status_color = COLORS['success'] if connected else COLORS['danger']
    status_text = "Alpaca Paper" if connected else "Disconnected"

    st.markdown(f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        width: inherit;
        max-width: 280px;
        padding: {SPACING['lg']};
        border-top: 1px solid {COLORS['border_subtle']};
        background: {COLORS['bg_secondary']};
    ">
        <div style="
            font-size: {FONT_SIZES['xs']};
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: {SPACING['sm']};
        ">
            Account
        </div>
        <div style="display: flex; align-items: center; gap: {SPACING['sm']};">
            <span style="
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: {status_color};
            "></span>
            <span style="color: {COLORS['text_secondary']}; font-size: {FONT_SIZES['sm']};">
                {status_text}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE HEADER
# =============================================================================

def page_header(
    title: str,
    subtitle: Optional[str] = None,
    show_env_badge: bool = True,
    show_status: bool = True,
    show_time: bool = True,
    show_home_button: bool = True,
    environment: Literal["paper", "live"] = "paper",
    connection_status: Literal["online", "offline", "pending"] = "online",
    extra_actions: Optional[List[Dict]] = None
) -> Optional[str]:
    """
    Render a standardized page header.

    Args:
        title: Main page title
        subtitle: Optional subtitle/description
        show_env_badge: Show environment badge (Paper/Live)
        show_status: Show connection status dot
        show_time: Show current time
        show_home_button: Show home navigation button
        environment: Trading environment
        connection_status: API connection status
        extra_actions: List of extra action buttons [{'label': 'Refresh', 'key': 'refresh'}]

    Returns:
        Key of clicked action button, or None
    """
    # Calculate column widths based on what's shown
    cols_config = [3]  # Title always takes 3

    if show_env_badge:
        cols_config.append(1)
    if show_status:
        cols_config.append(0.8)
    if show_time:
        cols_config.append(0.8)
    if show_home_button:
        cols_config.append(0.6)
    if extra_actions:
        for _ in extra_actions:
            cols_config.append(0.8)

    cols = st.columns(cols_config)
    col_idx = 0

    # Title and subtitle
    with cols[col_idx]:
        title_html = f"""
        <div style="
            font-size: {FONT_SIZES['2xl']};
            font-weight: 700;
            color: {COLORS['primary']};
            padding: {SPACING['md']} 0;
        ">
            {title}
        </div>
        """
        if subtitle:
            title_html += f"""
            <div style="
                font-size: {FONT_SIZES['sm']};
                color: {COLORS['text_muted']};
                margin-top: -{SPACING['sm']};
            ">
                {subtitle}
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)
    col_idx += 1

    # Environment badge
    if show_env_badge:
        with cols[col_idx]:
            env_color = COLORS['warning'] if environment == "paper" else COLORS['danger']
            env_text = "PAPER" if environment == "paper" else "LIVE"
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: center; padding: {SPACING['md']} 0;">
                <span style="
                    padding: 4px 12px;
                    background: {env_color}22;
                    color: {env_color};
                    border: 1px solid {env_color};
                    border-radius: {RADIUS['full']};
                    font-size: {FONT_SIZES['xs']};
                    font-weight: 600;
                ">{env_text}</span>
            </div>
            """, unsafe_allow_html=True)
        col_idx += 1

    # Connection status
    if show_status:
        with cols[col_idx]:
            status_colors = {
                'online': COLORS['success'],
                'offline': COLORS['danger'],
                'pending': COLORS['warning']
            }
            status_color = status_colors.get(connection_status, COLORS['text_muted'])
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: center; gap: 6px; padding: {SPACING['md']} 0;">
                <span style="
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: {status_color};
                    box-shadow: 0 0 6px {status_color};
                "></span>
                <span style="
                    color: {status_color};
                    font-size: {FONT_SIZES['xs']};
                    text-transform: uppercase;
                ">{connection_status}</span>
            </div>
            """, unsafe_allow_html=True)
        col_idx += 1

    # Current time
    if show_time:
        with cols[col_idx]:
            st.markdown(f"""
            <div style="text-align: center; padding: {SPACING['md']} 0;">
                <span style="
                    color: {COLORS['text_secondary']};
                    font-family: {FONTS['mono']};
                    font-size: {FONT_SIZES['sm']};
                ">
                    {datetime.now().strftime('%H:%M:%S')}
                </span>
            </div>
            """, unsafe_allow_html=True)
        col_idx += 1

    # Home button
    if show_home_button:
        with cols[col_idx]:
            if st.button("HOME", use_container_width=True, key="header_home"):
                st.switch_page("app.py")
        col_idx += 1

    # Extra action buttons
    clicked_action = None
    if extra_actions:
        for action in extra_actions:
            with cols[col_idx]:
                if st.button(action['label'], use_container_width=True, key=action.get('key', action['label'])):
                    clicked_action = action.get('key', action['label'])
            col_idx += 1

    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)

    return clicked_action


# =============================================================================
# MARKET CLOCKS
# =============================================================================

def render_market_clocks(
    markets: Optional[List[str]] = None,
    compact: bool = False
) -> None:
    """
    Render market clocks for major financial hubs.

    Args:
        markets: List of market names to show (defaults to all)
        compact: Use compact styling
    """
    market_config = {
        'NEW YORK': {'tz': 'America/New_York', 'hours': (9, 30, 16, 0)},
        'LONDON': {'tz': 'Europe/London', 'hours': (8, 0, 16, 30)},
        'FRANKFURT': {'tz': 'Europe/Berlin', 'hours': (9, 0, 17, 30)},
        'TOKYO': {'tz': 'Asia/Tokyo', 'hours': (9, 0, 15, 0)},
        'HONG KONG': {'tz': 'Asia/Hong_Kong', 'hours': (9, 30, 16, 0)},
    }

    if markets is None:
        markets = list(market_config.keys())

    # Get times and status
    clock_data = []
    for city in markets:
        if city not in market_config:
            continue
        config = market_config[city]
        tz = pytz.timezone(config['tz'])
        local_time = datetime.now(tz)

        # Check if market is open
        is_open = False
        if local_time.weekday() < 5:  # Weekday
            open_h, open_m, close_h, close_m = config['hours']
            market_open = local_time.replace(hour=open_h, minute=open_m, second=0)
            market_close = local_time.replace(hour=close_h, minute=close_m, second=0)
            is_open = market_open <= local_time <= market_close

        clock_data.append({
            'city': city,
            'time': local_time.strftime('%H:%M'),
            'is_open': is_open
        })

    # Render clocks
    cols = st.columns(len(clock_data))
    padding = SPACING['sm'] if compact else SPACING['md']
    font_size = FONT_SIZES['md'] if compact else FONT_SIZES['xl']

    for i, data in enumerate(clock_data):
        with cols[i]:
            status_color = COLORS['success'] if data['is_open'] else COLORS['danger']
            status_text = "OPEN" if data['is_open'] else "CLOSED"

            st.markdown(f"""
            <div style="
                background: {COLORS['bg_surface']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: {RADIUS['md']};
                padding: {padding};
                text-align: center;
            ">
                <div style="
                    font-size: {FONT_SIZES['xs']};
                    color: {COLORS['text_muted']};
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">{data['city']}</div>
                <div style="
                    font-family: {FONTS['mono']};
                    font-size: {font_size};
                    font-weight: 700;
                    color: {COLORS['text_primary']};
                    margin: 4px 0;
                ">{data['time']}</div>
                <div style="
                    font-size: {FONT_SIZES['xs']};
                    color: {status_color};
                ">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# SECTION UTILITIES
# =============================================================================

def section_divider() -> None:
    """Render a visual section divider."""
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


def render_section_title(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    numbered: Optional[int] = None
) -> None:
    """
    Render a section title within a page.

    Args:
        title: Section title
        subtitle: Optional description
        icon: Optional icon
        numbered: Optional section number (e.g., 1, 2, 3)
    """
    prefix = ""
    if numbered:
        prefix = f"{numbered}. "
    if icon:
        prefix = f"{icon} {prefix}"

    st.markdown(f"""
    <div style="margin: {SPACING['lg']} 0 {SPACING['md']} 0;">
        <div style="
            font-size: {FONT_SIZES['md']};
            font-weight: 600;
            color: {COLORS['text_primary']};
        ">
            {prefix}{title}
        </div>
        {f'<div style="font-size: {FONT_SIZES["sm"]}; color: {COLORS["text_muted"]}; margin-top: 2px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================

def render_footer(
    version: str = "3.0.0",
    show_date: bool = True,
    extra_info: Optional[str] = None
) -> None:
    """
    Render page footer.

    Args:
        version: Application version
        show_date: Show current date
        extra_info: Additional info to display
    """
    st.markdown("<hr>", unsafe_allow_html=True)

    info_parts = [f"Trading Terminal v{version}"]
    if show_date:
        info_parts.append(datetime.now().strftime('%Y-%m-%d'))
    if extra_info:
        info_parts.append(extra_info)

    st.markdown(f"""
    <div style="
        text-align: center;
        color: {COLORS['text_muted']};
        font-size: {FONT_SIZES['xs']};
        padding: {SPACING['md']} 0;
    ">
        {' | '.join(info_parts)}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ERROR/EMPTY STATES
# =============================================================================

def render_error_state(
    title: str = "Error",
    message: str = "Something went wrong",
    icon: str = "",
    suggestion: Optional[str] = None
) -> None:
    """Render a friendly error state card."""
    st.markdown(f"""
    <div style="
        background: {COLORS['danger_glow']};
        border: 1px solid {COLORS['danger']};
        border-radius: {RADIUS['lg']};
        padding: {SPACING['xl']};
        text-align: center;
    ">
        <div style="font-size: 2rem; margin-bottom: {SPACING['md']};">{icon}</div>
        <div style="
            font-size: {FONT_SIZES['lg']};
            font-weight: 600;
            color: {COLORS['danger']};
            margin-bottom: {SPACING['sm']};
        ">{title}</div>
        <div style="
            color: {COLORS['text_secondary']};
            font-size: {FONT_SIZES['sm']};
        ">{message}</div>
        {f'<div style="color: {COLORS["text_muted"]}; font-size: {FONT_SIZES["xs"]}; margin-top: {SPACING["md"]};">{suggestion}</div>' if suggestion else ''}
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(
    title: str = "No Data",
    message: str = "Nothing to display",
    icon: str = "",
    action_label: Optional[str] = None,
    action_key: Optional[str] = None
) -> bool:
    """
    Render a friendly empty state card.

    Returns:
        True if action button clicked, False otherwise
    """
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['lg']};
        padding: {SPACING['xl']};
        text-align: center;
    ">
        <div style="font-size: 2rem; margin-bottom: {SPACING['md']}; opacity: 0.5;">{icon}</div>
        <div style="
            font-size: {FONT_SIZES['lg']};
            font-weight: 600;
            color: {COLORS['text_secondary']};
            margin-bottom: {SPACING['sm']};
        ">{title}</div>
        <div style="
            color: {COLORS['text_muted']};
            font-size: {FONT_SIZES['sm']};
        ">{message}</div>
    </div>
    """, unsafe_allow_html=True)

    if action_label:
        return st.button(action_label, key=action_key, use_container_width=True)
    return False


def render_loading_state(message: str = "Loading...") -> None:
    """Render a loading indicator."""
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['lg']};
        padding: {SPACING['xl']};
        text-align: center;
    ">
        <div style="
            color: {COLORS['primary']};
            font-size: {FONT_SIZES['md']};
        ">{message}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'NAV_CONFIG',
    'init_page',
    'render_sidebar',
    'page_header',
    'render_market_clocks',
    'section_divider',
    'render_section_title',
    'render_footer',
    'render_error_state',
    'render_empty_state',
    'render_loading_state',
]
