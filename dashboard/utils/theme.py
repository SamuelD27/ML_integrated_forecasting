"""
VS Code Dark Theme Styling
===========================
Professional, minimalistic theme inspired by VS Code.
"""

import streamlit as st


def apply_vscode_theme():
    """
    Apply VS Code dark theme styling to Streamlit dashboard.

    Theme colors:
    - Background: #1E1E1E (VS Code editor background)
    - Sidebar: #252526 (VS Code sidebar)
    - Accent: #007ACC (VS Code blue)
    - Success: #4EC9B0 (VS Code cyan/teal)
    - Warning: #CE9178 (VS Code orange)
    - Error: #F48771 (VS Code red)
    - Text: #D4D4D4 (VS Code text)
    - Muted: #858585 (VS Code comments)
    """

    st.markdown("""
    <style>
    /* ========================================================================
       VS CODE DARK THEME - Global Styles
       ======================================================================== */

    /* Import VS Code font */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --vscode-bg: #1E1E1E;
        --vscode-sidebar: #252526;
        --vscode-accent: #007ACC;
        --vscode-success: #4EC9B0;
        --vscode-warning: #CE9178;
        --vscode-error: #F48771;
        --vscode-text: #D4D4D4;
        --vscode-muted: #858585;
        --vscode-border: #3E3E42;
        --vscode-hover: #2A2D2E;
    }

    /* ========================================================================
       Main Container
       ======================================================================== */

    .main {
        background-color: var(--vscode-bg);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Remove default Streamlit padding */
    .main > div {
        padding-top: 2rem;
    }

    /* ========================================================================
       Typography
       ======================================================================== */

    h1, h2, h3, h4, h5, h6 {
        color: var(--vscode-text) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        margin-bottom: 1rem !important;
    }

    h1 {
        font-size: 2rem !important;
        border-bottom: 1px solid var(--vscode-border);
        padding-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
        color: var(--vscode-accent) !important;
    }

    p, div, span, label {
        color: var(--vscode-text) !important;
        font-size: 0.95rem;
    }

    /* Code and monospace text */
    code, pre {
        background-color: var(--vscode-sidebar) !important;
        color: var(--vscode-success) !important;
        font-family: 'Fira Code', monospace !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 4px;
        padding: 0.2rem 0.4rem;
    }

    /* ========================================================================
       Sidebar
       ======================================================================== */

    [data-testid="stSidebar"] {
        background-color: var(--vscode-sidebar);
        border-right: 1px solid var(--vscode-border);
    }

    [data-testid="stSidebar"] .css-1d391kg {
        background-color: var(--vscode-sidebar);
    }

    /* Sidebar text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--vscode-text) !important;
    }

    /* ========================================================================
       Buttons
       ======================================================================== */

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: var(--vscode-accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 122, 204, 0.3);
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #005A9E !important;
        box-shadow: 0 4px 8px rgba(0, 122, 204, 0.4);
        transform: translateY(-1px);
    }

    /* Secondary buttons */
    .stButton > button {
        background-color: var(--vscode-sidebar) !important;
        color: var(--vscode-text) !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--vscode-hover) !important;
        border-color: var(--vscode-accent) !important;
    }

    /* ========================================================================
       Input Fields
       ======================================================================== */

    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: var(--vscode-sidebar) !important;
        color: var(--vscode-text) !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 4px;
        padding: 0.5rem 0.75rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--vscode-accent) !important;
        box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.2) !important;
    }

    /* Number inputs */
    .stNumberInput > div > div > input {
        background-color: var(--vscode-sidebar) !important;
        color: var(--vscode-text) !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 4px;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background-color: var(--vscode-sidebar) !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 4px;
    }

    /* Text areas */
    .stTextArea > div > div > textarea {
        background-color: var(--vscode-sidebar) !important;
        color: var(--vscode-text) !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 4px;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: var(--vscode-accent) !important;
    }

    /* ========================================================================
       Metrics and Cards
       ======================================================================== */

    [data-testid="stMetricValue"] {
        color: var(--vscode-text) !important;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        font-family: 'Fira Code', monospace !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--vscode-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'Fira Code', monospace !important;
    }

    /* Metric containers */
    div[data-testid="metric-container"] {
        background-color: var(--vscode-sidebar);
        border: 1px solid var(--vscode-border);
        border-radius: 6px;
        padding: 1rem;
        transition: all 0.2s ease;
    }

    div[data-testid="metric-container"]:hover {
        border-color: var(--vscode-accent);
        box-shadow: 0 4px 12px rgba(0, 122, 204, 0.15);
    }

    /* ========================================================================
       Alert Boxes
       ======================================================================== */

    /* Success */
    .stSuccess {
        background-color: rgba(78, 201, 176, 0.1) !important;
        border-left: 4px solid var(--vscode-success) !important;
        color: var(--vscode-success) !important;
        border-radius: 4px;
        padding: 1rem;
    }

    /* Info */
    .stInfo {
        background-color: rgba(0, 122, 204, 0.1) !important;
        border-left: 4px solid var(--vscode-accent) !important;
        color: var(--vscode-accent) !important;
        border-radius: 4px;
        padding: 1rem;
    }

    /* Warning */
    .stWarning {
        background-color: rgba(206, 145, 120, 0.1) !important;
        border-left: 4px solid var(--vscode-warning) !important;
        color: var(--vscode-warning) !important;
        border-radius: 4px;
        padding: 1rem;
    }

    /* Error */
    .stError {
        background-color: rgba(244, 135, 113, 0.1) !important;
        border-left: 4px solid var(--vscode-error) !important;
        color: var(--vscode-error) !important;
        border-radius: 4px;
        padding: 1rem;
    }

    /* ========================================================================
       Tables and DataFrames
       ======================================================================== */

    .dataframe {
        background-color: var(--vscode-sidebar) !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 4px;
        font-family: 'Fira Code', monospace !important;
        font-size: 0.85rem;
    }

    .dataframe th {
        background-color: var(--vscode-hover) !important;
        color: var(--vscode-accent) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        padding: 0.75rem !important;
        border-bottom: 2px solid var(--vscode-accent) !important;
    }

    .dataframe td {
        background-color: var(--vscode-sidebar) !important;
        color: var(--vscode-text) !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid var(--vscode-border) !important;
    }

    .dataframe tr:hover td {
        background-color: var(--vscode-hover) !important;
    }

    /* ========================================================================
       Expanders
       ======================================================================== */

    .streamlit-expanderHeader {
        background-color: var(--vscode-sidebar) !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 4px;
        color: var(--vscode-text) !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .streamlit-expanderHeader:hover {
        background-color: var(--vscode-hover) !important;
        border-color: var(--vscode-accent) !important;
    }

    .streamlit-expanderContent {
        background-color: var(--vscode-sidebar) !important;
        border: 1px solid var(--vscode-border) !important;
        border-top: none !important;
        border-radius: 0 0 4px 4px;
    }

    /* ========================================================================
       Tabs
       ======================================================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: var(--vscode-sidebar);
        border-bottom: 1px solid var(--vscode-border);
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: var(--vscode-muted);
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--vscode-hover);
        color: var(--vscode-text);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--vscode-accent) !important;
        color: white !important;
        border-radius: 4px;
    }

    /* ========================================================================
       Plotly Charts
       ======================================================================== */

    .js-plotly-plot {
        background-color: var(--vscode-sidebar) !important;
        border: 1px solid var(--vscode-border) !important;
        border-radius: 6px;
        padding: 0.5rem;
    }

    /* ========================================================================
       Spinner
       ======================================================================== */

    .stSpinner > div {
        border-top-color: var(--vscode-accent) !important;
    }

    /* ========================================================================
       Progress Bar
       ======================================================================== */

    .stProgress > div > div > div {
        background-color: var(--vscode-accent) !important;
    }

    /* ========================================================================
       File Uploader
       ======================================================================== */

    [data-testid="stFileUploadDropzone"] {
        background-color: var(--vscode-sidebar) !important;
        border: 2px dashed var(--vscode-border) !important;
        border-radius: 6px;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--vscode-accent) !important;
    }

    /* ========================================================================
       Markdown
       ======================================================================== */

    .stMarkdown {
        color: var(--vscode-text) !important;
    }

    .stMarkdown a {
        color: var(--vscode-accent) !important;
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .stMarkdown a:hover {
        border-bottom-color: var(--vscode-accent);
    }

    /* ========================================================================
       Dividers
       ======================================================================== */

    hr {
        border: none;
        border-top: 1px solid var(--vscode-border) !important;
        margin: 2rem 0;
    }

    /* ========================================================================
       Scrollbar
       ======================================================================== */

    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }

    ::-webkit-scrollbar-track {
        background: var(--vscode-bg);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--vscode-border);
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--vscode-muted);
    }

    /* ========================================================================
       Radio Buttons
       ======================================================================== */

    .stRadio > label {
        color: var(--vscode-text) !important;
    }

    .stRadio > div {
        background-color: var(--vscode-sidebar) !important;
        padding: 0.5rem;
        border-radius: 4px;
    }

    /* ========================================================================
       Checkboxes
       ======================================================================== */

    .stCheckbox > label {
        color: var(--vscode-text) !important;
    }

    /* ========================================================================
       Download Button
       ======================================================================== */

    .stDownloadButton > button {
        background-color: var(--vscode-success) !important;
        color: var(--vscode-bg) !important;
        border: none !important;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stDownloadButton > button:hover {
        background-color: #3DA58C !important;
        box-shadow: 0 4px 8px rgba(78, 201, 176, 0.3);
    }

    /* ========================================================================
       Caption Text
       ======================================================================== */

    .caption {
        color: var(--vscode-muted) !important;
        font-size: 0.85rem;
        font-style: italic;
    }

    /* ========================================================================
       Remove Streamlit Branding
       ======================================================================== */

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    </style>
    """, unsafe_allow_html=True)


def set_page_config(page_title: str, layout: str = "wide"):
    """
    Configure page with VS Code theme settings.

    Args:
        page_title: Title of the page
        layout: Layout mode ("wide" or "centered")
    """
    st.set_page_config(
        page_title=f"{page_title} | Stock Analysis",
        page_icon="📊",
        layout=layout,
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Stock Analysis Dashboard - Professional Trading Tools"
        }
    )
