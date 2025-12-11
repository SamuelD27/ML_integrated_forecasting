"""
Reusable UI Components
======================
Modern terminal-style components for the trading dashboard.

Components:
- terminal_card: Glass panel with optional terminal dots
- stat_tile: Large metric with label and delta
- status_pill: Colored status indicator
- status_dot: Small connection indicator
- metric_row: Compact metric display
- data_table: Styled data table
- log_panel: Terminal-style log viewer
- command_palette: Search/command input
- section_header: Section title with optional action
- position_card: Trading position display
- alert_card: Signal/alert display
- top_bar: Page header bar
- module_card: Navigation card for main modules
- info_card: Information/tip card
- kv_list: Key-value list display
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime
from .theme_terminal import COLORS, FONTS, FONT_SIZES, SPACING, RADIUS, SHADOWS


# =============================================================================
# TERMINAL CARD
# =============================================================================

def terminal_card(
    title: Optional[str] = None,
    content: str = "",
    icon: Optional[str] = None,
    show_dots: bool = True,
    accent_color: str = "primary",
    height: Optional[str] = None,
    key: Optional[str] = None
) -> None:
    """
    Render a terminal-style card with optional title bar.

    Args:
        title: Card title (optional)
        content: HTML content for the card body
        icon: Icon character/emoji (optional)
        show_dots: Show terminal window dots
        accent_color: Color key for accent (primary, secondary, success, danger)
        height: Fixed height (e.g., "300px")
        key: Unique key for the component
    """
    accent = COLORS.get(accent_color, COLORS['primary'])

    dots_html = ""
    if show_dots:
        dots_html = f"""
        <div style="display: flex; gap: 6px; margin-bottom: 8px;">
            <span style="width: 10px; height: 10px; border-radius: 50%; background: {COLORS['danger']};"></span>
            <span style="width: 10px; height: 10px; border-radius: 50%; background: {COLORS['warning']};"></span>
            <span style="width: 10px; height: 10px; border-radius: 50%; background: {COLORS['success']};"></span>
        </div>
        """

    title_html = ""
    if title:
        icon_html = f'<span style="margin-right: 8px;">{icon}</span>' if icon else ''
        title_html = f"""
        <div style="
            font-family: {FONTS['mono']};
            font-size: {FONT_SIZES['sm']};
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: {SPACING['md']};
            padding-bottom: {SPACING['sm']};
            border-bottom: 1px solid {COLORS['border_subtle']};
        ">
            {icon_html}{title}
        </div>
        """

    height_style = f"height: {height}; overflow-y: auto;" if height else ""

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['lg']};
        padding: {SPACING['lg']};
        {height_style}
        box-shadow: {SHADOWS['sm']};
    ">
        {dots_html}
        {title_html}
        <div style="font-family: {FONTS['mono']};">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# STAT TILE
# =============================================================================

def stat_tile(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: Optional[bool] = None,
    icon: Optional[str] = None,
    accent_color: str = "primary",
    size: Literal["sm", "md", "lg"] = "md",
    key: Optional[str] = None
) -> None:
    """
    Render a key metric tile with large number, label, and optional delta.

    Args:
        label: Small uppercase label
        value: Large primary value
        delta: Change/delta value (e.g., "+2.5%")
        delta_positive: True for green, False for red, None for neutral
        icon: Icon character/emoji
        accent_color: Color key for the value
        size: Tile size (sm, md, lg)
        key: Unique key
    """
    accent = COLORS.get(accent_color, COLORS['primary'])

    # Size configurations
    sizes = {
        "sm": {"value": FONT_SIZES['lg'], "label": FONT_SIZES['xs'], "padding": SPACING['md']},
        "md": {"value": FONT_SIZES['2xl'], "label": FONT_SIZES['xs'], "padding": SPACING['lg']},
        "lg": {"value": FONT_SIZES['3xl'], "label": FONT_SIZES['sm'], "padding": SPACING['xl']},
    }
    config = sizes.get(size, sizes["md"])

    # Delta styling
    delta_html = ""
    if delta:
        if delta_positive is True:
            delta_color = COLORS['success']
            delta_icon = "+"
        elif delta_positive is False:
            delta_color = COLORS['danger']
            delta_icon = ""
        else:
            delta_color = COLORS['text_muted']
            delta_icon = ""

        delta_html = f"""
        <div style="
            font-family: {FONTS['mono']};
            font-size: {FONT_SIZES['sm']};
            color: {delta_color};
            margin-top: 4px;
        ">
            {delta_icon}{delta}
        </div>
        """

    icon_html = f'<span style="font-size: 1.2em; margin-right: 8px; opacity: 0.6;">{icon}</span>' if icon else ''

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-left: 3px solid {accent};
        border-radius: {RADIUS['md']};
        padding: {config['padding']};
    ">
        <div style="
            font-size: {config['label']};
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 4px;
        ">
            {icon_html}{label}
        </div>
        <div style="
            font-family: {FONTS['mono']};
            font-size: {config['value']};
            font-weight: 700;
            color: {accent};
            line-height: 1.2;
        ">
            {value}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# STATUS PILL
# =============================================================================

def status_pill(
    label: str,
    status: Literal["success", "danger", "warning", "info", "neutral", "live", "paper"] = "neutral",
    size: Literal["sm", "md"] = "sm",
    key: Optional[str] = None
) -> None:
    """
    Render a colored status pill/chip.

    Args:
        label: Pill text
        status: Status type determining color
        size: Pill size
        key: Unique key
    """
    status_colors = {
        "success": (COLORS['success'], COLORS['success_glow']),
        "danger": (COLORS['danger'], COLORS['danger_glow']),
        "warning": (COLORS['warning'], f"rgba(245, 158, 11, 0.15)"),
        "info": (COLORS['info'], f"rgba(59, 130, 246, 0.15)"),
        "neutral": (COLORS['text_muted'], COLORS['bg_elevated']),
        "live": (COLORS['danger'], COLORS['danger_glow']),
        "paper": (COLORS['warning'], f"rgba(245, 158, 11, 0.15)"),
    }

    color, bg = status_colors.get(status, status_colors["neutral"])

    sizes = {
        "sm": {"font": FONT_SIZES['xs'], "padding": f"2px {SPACING['sm']}"},
        "md": {"font": FONT_SIZES['sm'], "padding": f"{SPACING['xs']} {SPACING['md']}"},
    }
    config = sizes.get(size, sizes["sm"])

    # Add pulse animation for live status
    animation = ""
    if status == "live":
        animation = f"""
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        animation: pulse 2s ease-in-out infinite;
        """

    st.markdown(f"""
    <style>
        .status-pill-{label.lower().replace(' ', '-')} {{ {animation} }}
    </style>
    <span class="status-pill-{label.lower().replace(' ', '-')}" style="
        display: inline-block;
        background: {bg};
        color: {color};
        border: 1px solid {color};
        border-radius: {RADIUS['full']};
        padding: {config['padding']};
        font-family: {FONTS['mono']};
        font-size: {config['font']};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">
        {label}
    </span>
    """, unsafe_allow_html=True)


# =============================================================================
# STATUS DOT
# =============================================================================

def status_dot(
    status: Literal["online", "offline", "warning", "pending"] = "online",
    label: Optional[str] = None,
    key: Optional[str] = None
) -> None:
    """
    Render a small status indicator dot.

    Args:
        status: Connection status
        label: Optional label text
        key: Unique key
    """
    status_colors = {
        "online": COLORS['success'],
        "offline": COLORS['danger'],
        "warning": COLORS['warning'],
        "pending": COLORS['text_muted'],
    }
    color = status_colors.get(status, COLORS['text_muted'])

    label_html = f'<span style="margin-left: 6px; color: {COLORS["text_secondary"]}; font-size: {FONT_SIZES["xs"]};">{label}</span>' if label else ''

    # Pulse for online status
    pulse = ""
    if status == "online":
        pulse = f"""
        box-shadow: 0 0 0 0 {color};
        animation: dotPulse 2s infinite;
        """

    st.markdown(f"""
    <style>
        @keyframes dotPulse {{
            0% {{ box-shadow: 0 0 0 0 {color}66; }}
            70% {{ box-shadow: 0 0 0 6px {color}00; }}
            100% {{ box-shadow: 0 0 0 0 {color}00; }}
        }}
    </style>
    <span style="display: inline-flex; align-items: center;">
        <span style="
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {color};
            {pulse}
        "></span>
        {label_html}
    </span>
    """, unsafe_allow_html=True)


# =============================================================================
# METRIC ROW
# =============================================================================

def metric_row(
    label: str,
    value: str,
    value_color: Optional[str] = None,
    key: Optional[str] = None
) -> None:
    """
    Render a compact metric row (label: value).

    Args:
        label: Left-aligned label
        value: Right-aligned value
        value_color: Color key for value
        key: Unique key
    """
    color = COLORS.get(value_color, COLORS['text_primary']) if value_color else COLORS['text_primary']

    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: {SPACING['sm']} 0;
        border-bottom: 1px solid {COLORS['border_subtle']};
        font-family: {FONTS['mono']};
        font-size: {FONT_SIZES['sm']};
    ">
        <span style="color: {COLORS['text_muted']};">{label}</span>
        <span style="color: {color}; font-weight: 500;">{value}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# LOG PANEL
# =============================================================================

def log_panel(
    logs: List[str],
    height: str = "400px",
    show_timestamps: bool = True,
    key: Optional[str] = None
) -> None:
    """
    Render a terminal-style log viewer with auto-scroll.

    Args:
        logs: List of log lines
        height: Panel height
        show_timestamps: Show line timestamps
        key: Unique key
    """
    def format_line(line: str) -> str:
        """Format log line with colors based on content."""
        line = line.strip()
        if not line:
            return ""

        # Color based on log level/content
        if "ERROR" in line or "CRITICAL" in line:
            color = COLORS['danger']
        elif "WARNING" in line or "WARN" in line:
            color = COLORS['warning']
        elif "SUCCESS" in line or "CONNECTED" in line:
            color = COLORS['success']
        elif "BUY" in line:
            color = COLORS['success']
        elif "SELL" in line:
            color = COLORS['danger']
        elif "INFO" in line:
            color = COLORS['text_secondary']
        else:
            color = COLORS['text_muted']

        return f'<div style="color: {color}; padding: 2px 0; font-size: {FONT_SIZES["xs"]}; line-height: 1.6; word-break: break-all;">{line}</div>'

    formatted = [format_line(log) for log in logs if log.strip()]
    log_html = "\n".join(formatted)

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['md']};
        padding: {SPACING['md']};
        height: {height};
        overflow-y: auto;
        font-family: {FONTS['mono']};
    ">
        {log_html}
    </div>
    <script>
        // Auto-scroll to bottom
        const logPanel = document.currentScript.previousElementSibling;
        logPanel.scrollTop = logPanel.scrollHeight;
    </script>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA TABLE WRAPPER
# =============================================================================

def data_table(
    data: List[Dict],
    columns: Optional[List[str]] = None,
    highlight_column: Optional[str] = None,
    positive_color: str = "success",
    negative_color: str = "danger",
    key: Optional[str] = None
) -> None:
    """
    Render a styled data table.

    Args:
        data: List of row dictionaries
        columns: Column order (optional, defaults to dict keys)
        highlight_column: Column to apply P&L coloring
        positive_color: Color for positive values
        negative_color: Color for negative values
        key: Unique key
    """
    if not data:
        st.markdown(f'<div style="color: {COLORS["text_muted"]}; padding: {SPACING["lg"]};">No data</div>', unsafe_allow_html=True)
        return

    cols = columns or list(data[0].keys())

    # Header
    header_cells = "".join([f"""
        <th style="
            background: {COLORS['bg_surface']};
            color: {COLORS['text_muted']};
            padding: {SPACING['sm']} {SPACING['md']};
            text-align: left;
            font-size: {FONT_SIZES['xs']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid {COLORS['border_default']};
        ">{col}</th>
    """ for col in cols])

    # Rows
    rows_html = ""
    for row in data:
        cells = ""
        for col in cols:
            value = row.get(col, "")
            cell_color = COLORS['text_primary']

            # Apply highlighting
            if col == highlight_column and isinstance(value, (int, float)):
                if value > 0:
                    cell_color = COLORS[positive_color]
                elif value < 0:
                    cell_color = COLORS[negative_color]

            cells += f"""
            <td style="
                padding: {SPACING['sm']} {SPACING['md']};
                color: {cell_color};
                font-family: {FONTS['mono']};
                font-size: {FONT_SIZES['sm']};
                border-bottom: 1px solid {COLORS['border_subtle']};
            ">{value}</td>
            """

        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div style="overflow-x: auto;">
        <table style="
            width: 100%;
            border-collapse: collapse;
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border_subtle']};
            border-radius: {RADIUS['md']};
        ">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# COMMAND PALETTE / SEARCH INPUT
# =============================================================================

def command_palette(
    placeholder: str = "Search or type command... (Ctrl+K)",
    key: str = "cmd_palette"
) -> Optional[str]:
    """
    Render a command palette / search input.

    Args:
        placeholder: Input placeholder text
        key: Unique key

    Returns:
        User input string or None
    """
    st.markdown(f"""
    <style>
        .command-palette-wrapper input {{
            background: {COLORS['bg_surface']} !important;
            border: 1px solid {COLORS['border_default']} !important;
            border-radius: {RADIUS['lg']} !important;
            color: {COLORS['text_primary']} !important;
            font-family: {FONTS['mono']} !important;
            font-size: {FONT_SIZES['sm']} !important;
            padding: {SPACING['md']} {SPACING['lg']} !important;
        }}
        .command-palette-wrapper input:focus {{
            border-color: {COLORS['primary']} !important;
            box-shadow: 0 0 0 2px {COLORS['primary_glow']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="command-palette-wrapper">', unsafe_allow_html=True)
        value = st.text_input(
            label="",
            placeholder=placeholder,
            key=key,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    return value if value else None


# =============================================================================
# SECTION HEADER
# =============================================================================

def section_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    action_label: Optional[str] = None,
    key: Optional[str] = None
) -> bool:
    """
    Render a section header with optional action button.

    Args:
        title: Section title
        subtitle: Optional subtitle/description
        icon: Icon character/emoji
        action_label: Optional action button label
        key: Unique key

    Returns:
        True if action button was clicked, False otherwise
    """
    icon_html = f'<span style="margin-right: 8px;">{icon}</span>' if icon else ''
    subtitle_html = f'<div style="color: {COLORS["text_muted"]}; font-size: {FONT_SIZES["sm"]}; margin-top: 4px;">{subtitle}</div>' if subtitle else ''

    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(f"""
        <div style="margin-bottom: {SPACING['md']};">
            <div style="
                font-size: {FONT_SIZES['lg']};
                font-weight: 600;
                color: {COLORS['text_primary']};
            ">
                {icon_html}{title}
            </div>
            {subtitle_html}
        </div>
        """, unsafe_allow_html=True)

    clicked = False
    if action_label:
        with col2:
            clicked = st.button(action_label, key=key or f"section_{title}")

    return clicked


# =============================================================================
# POSITION CARD
# =============================================================================

def position_card(
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    current_price: float,
    pnl: float,
    pnl_pct: float,
    exchange: str = "NASDAQ",
    key: Optional[str] = None
) -> None:
    """
    Render a position card with P&L visualization.

    Args:
        symbol: Ticker symbol
        side: "LONG" or "SHORT"
        qty: Position quantity
        entry_price: Entry price
        current_price: Current market price
        pnl: Absolute P&L
        pnl_pct: P&L percentage
        exchange: Exchange name
        key: Unique key
    """
    is_positive = pnl >= 0
    pnl_color = COLORS['success'] if is_positive else COLORS['danger']
    side_color = COLORS['success'] if side.upper() == "LONG" else COLORS['danger']
    sign = "+" if is_positive else ""

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-left: 3px solid {pnl_color};
        border-radius: {RADIUS['md']};
        padding: {SPACING['md']};
        margin-bottom: {SPACING['sm']};
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div>
                <span style="
                    font-family: {FONTS['mono']};
                    font-size: {FONT_SIZES['md']};
                    font-weight: 700;
                    color: {COLORS['primary']};
                ">{symbol}</span>
                <span style="
                    margin-left: 8px;
                    padding: 2px 6px;
                    background: {side_color}22;
                    color: {side_color};
                    border-radius: {RADIUS['sm']};
                    font-size: {FONT_SIZES['xs']};
                    font-weight: 600;
                ">{side.upper()}</span>
            </div>
            <span style="color: {COLORS['text_muted']}; font-size: {FONT_SIZES['xs']};">{exchange}</span>
        </div>

        <div style="display: flex; justify-content: space-between; font-family: {FONTS['mono']}; font-size: {FONT_SIZES['sm']};">
            <div>
                <span style="color: {COLORS['text_muted']};">QTY:</span>
                <span style="color: {COLORS['text_primary']};">{qty:,.0f}</span>
            </div>
            <div>
                <span style="color: {COLORS['text_muted']};">Entry:</span>
                <span style="color: {COLORS['text_primary']};">${entry_price:,.2f}</span>
            </div>
            <div>
                <span style="color: {COLORS['text_muted']};">Current:</span>
                <span style="color: {COLORS['text_primary']};">${current_price:,.2f}</span>
            </div>
        </div>

        <div style="
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid {COLORS['border_subtle']};
            text-align: right;
        ">
            <span style="
                font-family: {FONTS['mono']};
                font-size: {FONT_SIZES['lg']};
                font-weight: 700;
                color: {pnl_color};
            ">
                {sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ALERT CARD
# =============================================================================

def alert_card(
    timestamp: str,
    symbol: str,
    signal: str,
    confidence: float,
    message: Optional[str] = None,
    key: Optional[str] = None
) -> None:
    """
    Render a signal/alert card.

    Args:
        timestamp: Alert timestamp
        symbol: Related symbol
        signal: Signal type (BUY, SELL, HOLD, etc.)
        confidence: Confidence score (0-1)
        message: Optional message
        key: Unique key
    """
    signal_colors = {
        "BUY": COLORS['success'],
        "SELL": COLORS['danger'],
        "HOLD": COLORS['text_muted'],
        "LONG": COLORS['success'],
        "SHORT": COLORS['danger'],
    }
    signal_color = signal_colors.get(signal.upper(), COLORS['text_secondary'])

    confidence_color = COLORS['success'] if confidence >= 0.7 else COLORS['warning'] if confidence >= 0.5 else COLORS['danger']

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['md']};
        padding: {SPACING['md']};
        margin-bottom: {SPACING['sm']};
        display: flex;
        align-items: center;
        gap: {SPACING['md']};
    ">
        <div style="
            width: 4px;
            height: 40px;
            background: {signal_color};
            border-radius: {RADIUS['sm']};
        "></div>

        <div style="flex: 1;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="
                    font-family: {FONTS['mono']};
                    font-size: {FONT_SIZES['md']};
                    font-weight: 700;
                    color: {COLORS['primary']};
                ">{symbol}</span>
                <span style="
                    padding: 2px 8px;
                    background: {signal_color}22;
                    color: {signal_color};
                    border-radius: {RADIUS['sm']};
                    font-size: {FONT_SIZES['xs']};
                    font-weight: 700;
                ">{signal.upper()}</span>
            </div>

            <div style="
                display: flex;
                justify-content: space-between;
                margin-top: 4px;
                font-size: {FONT_SIZES['xs']};
            ">
                <span style="color: {COLORS['text_muted']};">{timestamp}</span>
                <span style="color: {confidence_color};">
                    Confidence: {confidence:.0%}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TOP BAR
# =============================================================================

def top_bar(
    page_title: str,
    environment: Literal["paper", "live"] = "paper",
    connection_status: Literal["online", "offline", "pending"] = "online",
    show_deploy_button: bool = True,
    key: Optional[str] = None
) -> Optional[bool]:
    """
    Render the top bar with page title, environment, and status.

    Args:
        page_title: Current page title
        environment: Trading environment
        connection_status: API connection status
        show_deploy_button: Show the deploy/run button
        key: Unique key

    Returns:
        True if deploy button clicked, None otherwise
    """
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="
            font-size: {FONT_SIZES['xl']};
            font-weight: 600;
            color: {COLORS['text_primary']};
            padding: {SPACING['sm']} 0;
        ">
            {page_title}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        env_color = COLORS['warning'] if environment == "paper" else COLORS['danger']
        st.markdown(f"""
        <div style="text-align: center; padding: {SPACING['sm']};">
            <span style="
                padding: 4px 12px;
                background: {env_color}22;
                color: {env_color};
                border: 1px solid {env_color};
                border-radius: {RADIUS['full']};
                font-size: {FONT_SIZES['xs']};
                font-weight: 600;
                text-transform: uppercase;
            ">{environment.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status_color = COLORS['success'] if connection_status == "online" else COLORS['danger'] if connection_status == "offline" else COLORS['warning']
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: center; padding: {SPACING['sm']}; gap: 6px;">
            <span style="
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: {status_color};
            "></span>
            <span style="
                color: {status_color};
                font-size: {FONT_SIZES['xs']};
                text-transform: uppercase;
            ">{connection_status}</span>
        </div>
        """, unsafe_allow_html=True)

    clicked = None
    if show_deploy_button:
        with col4:
            clicked = st.button("RUN", key=key or "deploy_btn", type="primary", use_container_width=True)

    return clicked


# =============================================================================
# MODULE CARD
# =============================================================================

def module_card(
    title: str,
    description: str,
    number: Optional[str] = None,
    status_label: Optional[str] = None,
    status_type: Literal["success", "warning", "info", "neutral"] = "success",
    accent_color: str = "primary",
    key: Optional[str] = None
) -> bool:
    """
    Render a large module selection card.

    Args:
        title: Module title
        description: Module description
        number: Optional number prefix (e.g., "01")
        status_label: Optional status badge text (e.g., "READY", "8 TOOLS")
        status_type: Status badge color type
        accent_color: Title accent color
        key: Unique key for the button

    Returns:
        True if card was clicked
    """
    accent = COLORS.get(accent_color, COLORS['primary'])
    status_colors = {
        "success": (COLORS['success'], f"{COLORS['success']}22"),
        "warning": (COLORS['warning'], f"{COLORS['warning']}22"),
        "info": (COLORS['info'], f"{COLORS['info']}22"),
        "neutral": (COLORS['text_muted'], COLORS['bg_elevated']),
    }
    status_color, status_bg = status_colors.get(status_type, status_colors["success"])

    number_prefix = f"{number} / " if number else ""
    status_html = ""
    if status_label:
        status_html = f"""
        <span style="
            padding: 4px 8px;
            background: {status_bg};
            color: {status_color};
            border-radius: {RADIUS['sm']};
            font-size: {FONT_SIZES['xs']};
            font-weight: 600;
        ">{status_label}</span>
        """

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['lg']};
        padding: {SPACING['xl']};
        margin-bottom: {SPACING['md']};
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="
                    font-size: {FONT_SIZES['lg']};
                    font-weight: 600;
                    color: {accent};
                    margin-bottom: {SPACING['sm']};
                ">
                    {number_prefix}{title}
                </div>
                <div style="color: {COLORS['text_muted']}; font-size: {FONT_SIZES['sm']}; line-height: 1.6;">
                    {description}
                </div>
            </div>
            {status_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.button(
        f"ENTER {title.upper()}",
        use_container_width=True,
        key=key or f"module_{title}",
        type="primary"
    )


# =============================================================================
# INFO CARD
# =============================================================================

def info_card(
    content: str,
    title: Optional[str] = None,
    card_type: Literal["info", "success", "warning", "error", "tip"] = "info",
    icon: Optional[str] = None,
    key: Optional[str] = None
) -> None:
    """
    Render an information/tip card.

    Args:
        content: Card content text
        title: Optional title
        card_type: Type determining color scheme
        icon: Optional icon
        key: Unique key
    """
    type_styles = {
        "info": (COLORS['info'], f"rgba(59, 130, 246, 0.1)", ""),
        "success": (COLORS['success'], COLORS['success_glow'], ""),
        "warning": (COLORS['warning'], f"rgba(245, 158, 11, 0.1)", ""),
        "error": (COLORS['danger'], COLORS['danger_glow'], ""),
        "tip": (COLORS['secondary'], COLORS['secondary_glow'], ""),
    }

    color, bg, default_icon = type_styles.get(card_type, type_styles["info"])
    display_icon = icon or default_icon

    title_html = ""
    if title:
        title_html = f"""
        <div style="
            font-weight: 600;
            color: {color};
            margin-bottom: {SPACING['xs']};
        ">{display_icon} {title if display_icon else title}</div>
        """

    st.markdown(f"""
    <div style="
        background: {bg};
        border: 1px solid {color};
        border-left: 3px solid {color};
        border-radius: {RADIUS['md']};
        padding: {SPACING['md']};
        margin-bottom: {SPACING['sm']};
    ">
        {title_html}
        <div style="
            color: {COLORS['text_secondary']};
            font-size: {FONT_SIZES['sm']};
            line-height: 1.6;
        ">{content}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# KEY-VALUE LIST
# =============================================================================

def kv_list(
    items: List[Dict[str, Any]],
    title: Optional[str] = None,
    show_dividers: bool = True,
    key: Optional[str] = None
) -> None:
    """
    Render a key-value list.

    Args:
        items: List of dicts with 'label', 'value', and optional 'value_color'
        title: Optional section title
        show_dividers: Show line dividers between items
        key: Unique key
    """
    if title:
        st.markdown(f"""
        <div style="
            font-size: {FONT_SIZES['sm']};
            font-weight: 600;
            color: {COLORS['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: {SPACING['md']};
        ">
            {title}
        </div>
        """, unsafe_allow_html=True)

    for i, item in enumerate(items):
        label = item.get('label', '')
        value = item.get('value', '')
        value_color = COLORS.get(item.get('value_color'), COLORS['text_primary'])

        border = f"border-bottom: 1px solid {COLORS['border_subtle']};" if show_dividers and i < len(items) - 1 else ""

        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: {SPACING['sm']} 0;
            {border}
            font-family: {FONTS['mono']};
            font-size: {FONT_SIZES['sm']};
        ">
            <span style="color: {COLORS['text_muted']};">{label}</span>
            <span style="color: {value_color}; font-weight: 500;">{value}</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# QUICK STATS ROW
# =============================================================================

def quick_stats(
    stats: List[Dict[str, Any]],
    columns: int = 4,
    key: Optional[str] = None
) -> None:
    """
    Render a row of quick stat tiles.

    Args:
        stats: List of stat configs, each with:
            - label: Stat label
            - value: Main value
            - delta: Optional delta value
            - delta_positive: True/False/None for delta color
            - accent_color: Color key (default 'primary')
            - size: 'sm', 'md', 'lg'
        columns: Number of columns
        key: Unique key
    """
    cols = st.columns(columns)

    for i, stat in enumerate(stats):
        with cols[i % columns]:
            stat_tile(
                label=stat.get('label', ''),
                value=stat.get('value', ''),
                delta=stat.get('delta'),
                delta_positive=stat.get('delta_positive'),
                accent_color=stat.get('accent_color', 'primary'),
                size=stat.get('size', 'md'),
                key=f"{key}_{i}" if key else None
            )

        # Add spacer between rows if needed
        if (i + 1) % columns == 0 and i < len(stats) - 1:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# =============================================================================
# TAB DESCRIPTION CARD
# =============================================================================

def tab_intro_card(
    title: str,
    description: str,
    accent_color: str = "text_primary",
    key: Optional[str] = None
) -> None:
    """
    Render an intro card for tab content.

    Args:
        title: Tool/section title
        description: Brief description
        accent_color: Title color key
        key: Unique key
    """
    accent = COLORS.get(accent_color, COLORS['text_primary'])

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['md']};
        padding: {SPACING['lg']};
        margin-bottom: {SPACING['lg']};
    ">
        <div style="
            font-size: {FONT_SIZES['md']};
            font-weight: 600;
            color: {accent};
            margin-bottom: {SPACING['sm']};
        ">
            {title}
        </div>
        <div style="
            font-size: {FONT_SIZES['sm']};
            color: {COLORS['text_muted']};
        ">
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TV PANEL - TradingView Style Widget
# =============================================================================

def tv_panel(
    title: str,
    content: str = "",
    height: Optional[str] = None,
    accent_color: str = "primary",
    show_header: bool = True,
    actions: Optional[List[Dict]] = None,
    key: Optional[str] = None
) -> None:
    """
    Render TradingView-style widget panel.

    Features:
    - Professional header with accent bar
    - Optional action buttons
    - Configurable height with scroll
    - Multiple accent colors

    Args:
        title: Panel header title
        content: HTML content for panel body
        height: Fixed height (e.g., "300px")
        accent_color: Accent color key (primary, success, danger, warning)
        show_header: Show the panel header
        actions: Optional header action buttons [{'icon': '', 'label': 'Close'}]
        key: Unique key
    """
    accent = COLORS.get(accent_color, COLORS['primary'])
    height_style = f"height: {height}; overflow-y: auto;" if height else ""

    # Action buttons HTML
    actions_html = ""
    if actions:
        btns = "".join([f"""
        <button style="
            background: transparent;
            border: none;
            color: {COLORS['text_tertiary']};
            cursor: pointer;
            padding: {SPACING['xs']};
            font-size: {FONT_SIZES['sm']};
            transition: color 0.15s;
        " onmouseover="this.style.color='{COLORS['text_primary']}'"
           onmouseout="this.style.color='{COLORS['text_tertiary']}'">
            {a.get('icon', '')} {a.get('label', '')}
        </button>
        """ for a in actions])
        actions_html = f"<div style='display: flex; gap: {SPACING['sm']};'>{btns}</div>"

    header_html = ""
    if show_header:
        header_html = f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: {SPACING['sm']} {SPACING['md']};
            background: {COLORS['bg_surface']};
            border-bottom: 1px solid {COLORS['border_default']};
        ">
            <div style="display: flex; align-items: center; gap: {SPACING['sm']};">
                <div style="
                    width: 3px;
                    height: 14px;
                    background: {accent};
                    border-radius: 2px;
                "></div>
                <span style="
                    font-size: {FONT_SIZES['sm']};
                    font-weight: 600;
                    color: {COLORS['text_secondary']};
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">{title}</span>
            </div>
            {actions_html}
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
        {header_html}
        <div style="padding: {SPACING['md']};">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# STAT TILE V2 - Enhanced KPI Tile
# =============================================================================

def stat_tile_v2(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: Optional[bool] = None,
    sparkline_data: Optional[List[float]] = None,
    accent: str = "primary",
    size: Literal["sm", "md", "lg"] = "md",
    key: Optional[str] = None
) -> None:
    """
    Render enhanced stat tile with optional mini sparkline.

    Args:
        label: Metric label
        value: Main display value
        delta: Optional change indicator
        delta_positive: True/False/None for delta coloring
        sparkline_data: Optional list of values for mini chart
        accent: Accent color key
        size: Tile size
        key: Unique key
    """
    accent_color = COLORS.get(accent, COLORS['primary'])

    size_config = {
        'sm': {'value': FONT_SIZES['lg'], 'label': FONT_SIZES['xs'], 'padding': SPACING['sm']},
        'md': {'value': FONT_SIZES['2xl'], 'label': FONT_SIZES['xs'], 'padding': SPACING['md']},
        'lg': {'value': FONT_SIZES['3xl'], 'label': FONT_SIZES['sm'], 'padding': SPACING['lg']},
    }
    config = size_config.get(size, size_config['md'])

    # Delta HTML
    delta_html = ""
    if delta:
        if delta_positive is True:
            d_color = COLORS['success']
        elif delta_positive is False:
            d_color = COLORS['danger']
        else:
            d_color = COLORS['text_muted']

        delta_html = f"""
        <div style="
            font-family: {FONTS['mono']};
            font-size: {FONT_SIZES['sm']};
            color: {d_color};
            margin-top: {SPACING['xs']};
        ">{delta}</div>
        """

    # Sparkline SVG (simple line)
    sparkline_html = ""
    if sparkline_data and len(sparkline_data) > 1:
        min_v = min(sparkline_data)
        max_v = max(sparkline_data)
        range_v = max_v - min_v if max_v != min_v else 1

        # Normalize to 0-20 height, 60 width
        points = []
        for i, v in enumerate(sparkline_data):
            x = (i / (len(sparkline_data) - 1)) * 60
            y = 20 - ((v - min_v) / range_v) * 20
            points.append(f"{x},{y}")

        path_d = "M" + " L".join(points)
        line_color = COLORS['success'] if sparkline_data[-1] >= sparkline_data[0] else COLORS['danger']

        sparkline_html = f"""
        <svg width="60" height="24" style="
            position: absolute;
            right: {SPACING['sm']};
            bottom: {SPACING['sm']};
            opacity: 0.5;
        ">
            <path d="{path_d}" fill="none" stroke="{line_color}" stroke-width="1.5"/>
        </svg>
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
            margin-bottom: {SPACING['xs']};
        ">{label}</div>
        <div style="
            font-family: {FONTS['mono']};
            font-size: {config['value']};
            font-weight: 700;
            color: {accent_color};
            line-height: 1.2;
        ">{value}</div>
        {delta_html}
        {sparkline_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MINI LOG TERMINAL - Compact Log Viewer
# =============================================================================

def mini_log_terminal(
    logs: List[str],
    height: str = "200px",
    key: Optional[str] = None
) -> None:
    """
    Render a compact log terminal with color-coded severity.

    Auto-colors log lines based on content:
    - ERROR/CRITICAL: Red
    - WARNING/WARN: Amber
    - SUCCESS/CONNECTED/BUY: Green
    - SELL: Red
    - INFO: Secondary text

    Args:
        logs: List of log lines
        height: Terminal height
        key: Unique key
    """
    def classify_line(line: str) -> str:
        line_upper = line.upper()
        if "ERROR" in line_upper or "CRITICAL" in line_upper:
            return "error"
        if "WARNING" in line_upper or "WARN" in line_upper:
            return "warning"
        if "SUCCESS" in line_upper or "CONNECTED" in line_upper:
            return "success"
        if "BUY" in line_upper:
            return "success"
        if "SELL" in line_upper:
            return "error"
        if "INFO" in line_upper:
            return "info"
        return ""

    lines_html = ""
    for log in logs[-50:]:  # Last 50 lines
        line = log.strip()
        if not line:
            continue

        severity = classify_line(line)
        color = {
            "error": COLORS['danger'],
            "warning": COLORS['warning'],
            "success": COLORS['success'],
            "info": COLORS['text_secondary'],
        }.get(severity, COLORS['text_tertiary'])

        lines_html += f'<div style="color: {color}; padding: 1px 0;">{line}</div>'

    st.markdown(f"""
    <div class="mini-log" style="
        background: {COLORS['bg_base']};
        border: 1px solid {COLORS['border_default']};
        border-radius: {RADIUS['md']};
        padding: {SPACING['sm']};
        height: {height};
        overflow-y: auto;
        font-family: {FONTS['mono']};
        font-size: {FONT_SIZES['xs']};
        line-height: 1.6;
    ">
        {lines_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# KPI STRIPE - Horizontal Stats Row
# =============================================================================

def kpi_stripe(
    kpis: List[Dict[str, Any]],
    key: Optional[str] = None
) -> None:
    """
    Render horizontal KPI stripe (Binance market stats style).

    Args:
        kpis: List of KPI dicts with keys:
            - label: Metric label
            - value: Display value
            - delta: Optional change value
            - delta_positive: True/False/None for coloring
        key: Optional unique key
    """
    items_html = ""

    for i, kpi in enumerate(kpis):
        # Delta coloring
        delta_html = ""
        if kpi.get('delta'):
            if kpi.get('delta_positive') is True:
                delta_color = COLORS['success']
            elif kpi.get('delta_positive') is False:
                delta_color = COLORS['danger']
            else:
                delta_color = COLORS['text_muted']

            delta_html = f"""
            <div style="
                font-family: {FONTS['mono']};
                font-size: {FONT_SIZES['xs']};
                color: {delta_color};
            ">{kpi['delta']}</div>
            """

        # Value color
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
                font-size: {FONT_SIZES['xs']};
                color: {COLORS['text_muted']};
                text-transform: uppercase;
                letter-spacing: 0.05em;
            ">{kpi['label']}</div>
            <div style="
                font-family: {FONTS['mono']};
                font-size: {FONT_SIZES['md']};
                font-weight: 600;
                color: {value_color};
            ">{kpi['value']}</div>
            {delta_html}
        </div>
        """

        # Divider
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
        gap: {SPACING['lg']};
        padding: {SPACING['sm']} {SPACING['md']};
        background: {COLORS['bg_primary']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['lg']};
        overflow-x: auto;
        margin-bottom: {SPACING['md']};
    ">
        {items_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# METRIC ROW PRO - Dense Horizontal KPI Row
# =============================================================================

def metric_row_pro(
    metrics: List[Dict[str, Any]],
    key: Optional[str] = None
) -> None:
    """
    Render dense horizontal metric row (Binance orderbook summary style).

    Args:
        metrics: List of metric dicts with 'label', 'value', and optional 'color'
        key: Optional key
    """
    cells = ""
    for m in metrics:
        color = COLORS.get(m.get('color'), COLORS['text_primary'])
        cells += f"""
        <div style="display: flex; flex-direction: column; gap: 2px;">
            <span style="
                font-size: {FONT_SIZES['xs']};
                color: {COLORS['text_muted']};
                text-transform: uppercase;
            ">{m['label']}</span>
            <span style="
                font-family: {FONTS['mono']};
                font-size: {FONT_SIZES['sm']};
                font-weight: 600;
                color: {color};
            ">{m['value']}</span>
        </div>
        """

    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding: {SPACING['sm']} {SPACING['md']};
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: {RADIUS['md']};
        margin-bottom: {SPACING['sm']};
    ">
        {cells}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SECTION CARD - General Layout Block
# =============================================================================

def section_card(
    title: str,
    content: str = "",
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    accent: str = "primary",
    key: Optional[str] = None
) -> None:
    """
    Render a general section card with header and body.

    Args:
        title: Section title
        content: HTML content for body
        subtitle: Optional subtitle
        icon: Optional icon
        accent: Accent color key
        key: Unique key
    """
    accent_color = COLORS.get(accent, COLORS['primary'])

    icon_html = f'<span style="margin-right: {SPACING["sm"]};">{icon}</span>' if icon else ''
    subtitle_html = f'<div style="font-size: {FONT_SIZES["xs"]}; color: {COLORS["text_muted"]}; margin-top: 2px;">{subtitle}</div>' if subtitle else ''

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border_default']};
        border-radius: {RADIUS['lg']};
        overflow: hidden;
    ">
        <div style="
            padding: {SPACING['md']};
            background: {COLORS['bg_surface']};
            border-bottom: 1px solid {COLORS['border_default']};
            border-left: 3px solid {accent_color};
        ">
            <div style="
                font-size: {FONT_SIZES['md']};
                font-weight: 600;
                color: {COLORS['text_primary']};
            ">{icon_html}{title}</div>
            {subtitle_html}
        </div>
        <div style="padding: {SPACING['md']};">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR ITEM - Navigation Item
# =============================================================================

def sidebar_item(
    label: str,
    icon: str = "",
    active: bool = False,
    key: Optional[str] = None
) -> bool:
    """
    Render a sidebar navigation item.

    Args:
        label: Item label
        icon: Item icon
        active: Whether item is currently active
        key: Unique key

    Returns:
        True if clicked
    """
    if active:
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            gap: {SPACING['sm']};
            padding: {SPACING['sm']} {SPACING['md']};
            background: {COLORS['primary_bg']};
            color: {COLORS['primary']};
            border-left: 2px solid {COLORS['primary']};
            font-size: {FONT_SIZES['sm']};
            font-weight: 500;
        ">
            <span style="width: 18px; text-align: center;">{icon}</span>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)
        return False

    return st.button(
        f"{icon} {label}",
        key=key or f"sidebar_{label}",
        use_container_width=True
    )


# =============================================================================
# QUICK ACTION BAR
# =============================================================================

def quick_action_bar(
    actions: List[Dict[str, str]],
    key: Optional[str] = None
) -> Optional[str]:
    """
    Render a row of quick action buttons.

    Args:
        actions: List of action dicts with 'icon', 'label', 'key'
        key: Optional key prefix

    Returns:
        Key of clicked action, or None
    """
    cols = st.columns(len(actions))

    for i, action in enumerate(actions):
        with cols[i]:
            if st.button(
                f"{action.get('icon', '')} {action['label']}",
                key=action.get('key', f"action_{i}"),
                use_container_width=True
            ):
                return action.get('key', action['label'])

    return None


# Export all components
__all__ = [
    'terminal_card',
    'stat_tile',
    'status_pill',
    'status_dot',
    'metric_row',
    'log_panel',
    'data_table',
    'command_palette',
    'section_header',
    'position_card',
    'alert_card',
    'top_bar',
    'module_card',
    'info_card',
    'kv_list',
    'quick_stats',
    'tab_intro_card',
    # New TradingView-style components
    'tv_panel',
    'stat_tile_v2',
    'mini_log_terminal',
    'kpi_stripe',
    'metric_row_pro',
    'section_card',
    'sidebar_item',
    'quick_action_bar',
]
