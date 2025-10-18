"""
Stock Search Utility
====================
Search for stocks by company name across major exchanges.

Now includes 500+ stocks from 7 major exchanges worldwide.
"""

import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional
import yfinance as yf

# Import mega stock database
from dashboard.utils.stock_database_mega import MEGA_STOCK_DATABASE

# Use mega database (800+ stocks, targeting 1000+)
STOCK_DATABASE = MEGA_STOCK_DATABASE

# Legacy database kept for backward compatibility (200+ stocks)
STOCK_DATABASE_LEGACY = {
    # ============================================================================
    # NYSE - New York Stock Exchange (USA)
    # ============================================================================
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corporation',
    'META': 'Meta Platforms Inc. (Facebook)',
    'TSLA': 'Tesla Inc.',
    'BRK.B': 'Berkshire Hathaway Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble Company',
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'ABBV': 'AbbVie Inc.',
    'KO': 'The Coca-Cola Company',
    'PEP': 'PepsiCo Inc.',
    'MRK': 'Merck & Co. Inc.',
    'COST': 'Costco Wholesale Corporation',
    'AVGO': 'Broadcom Inc.',
    'LLY': 'Eli Lilly and Company',
    'TMO': 'Thermo Fisher Scientific Inc.',
    'ABT': 'Abbott Laboratories',
    'NKE': 'NIKE Inc.',
    'DIS': 'The Walt Disney Company',
    'ORCL': 'Oracle Corporation',
    'ADBE': 'Adobe Inc.',
    'CRM': 'Salesforce Inc.',
    'CSCO': 'Cisco Systems Inc.',
    'INTC': 'Intel Corporation',
    'AMD': 'Advanced Micro Devices Inc.',
    'NFLX': 'Netflix Inc.',
    'PYPL': 'PayPal Holdings Inc.',
    'QCOM': 'QUALCOMM Incorporated',
    'TXN': 'Texas Instruments Incorporated',
    'IBM': 'International Business Machines',
    'BA': 'The Boeing Company',
    'GE': 'General Electric Company',
    'CAT': 'Caterpillar Inc.',
    'HON': 'Honeywell International Inc.',
    'UPS': 'United Parcel Service Inc.',
    'FDX': 'FedEx Corporation',
    'UNH': 'UnitedHealth Group Inc.',
    'CVS': 'CVS Health Corporation',
    'GILD': 'Gilead Sciences Inc.',
    'BMY': 'Bristol-Myers Squibb Company',
    'AMGN': 'Amgen Inc.',
    'BAC': 'Bank of America Corporation',
    'WFC': 'Wells Fargo & Company',
    'GS': 'The Goldman Sachs Group Inc.',
    'MS': 'Morgan Stanley',
    'C': 'Citigroup Inc.',
    'AXP': 'American Express Company',
    'BLK': 'BlackRock Inc.',
    'SCHW': 'The Charles Schwab Corporation',
    'MA': 'Mastercard Incorporated',
    'SQ': 'Block Inc. (Square)',
    'UBER': 'Uber Technologies Inc.',
    'ABNB': 'Airbnb Inc.',
    'COIN': 'Coinbase Global Inc.',
    'PLTR': 'Palantir Technologies Inc.',
    'RBLX': 'Roblox Corporation',
    'SNAP': 'Snap Inc.',
    'PINS': 'Pinterest Inc.',
    'SPOT': 'Spotify Technology S.A.',
    'DASH': 'DoorDash Inc.',
    'SHOP': 'Shopify Inc.',

    # NASDAQ additions
    'QQQ': 'Invesco QQQ Trust (NASDAQ-100 ETF)',
    'SPY': 'SPDR S&P 500 ETF Trust',
    'IWM': 'iShares Russell 2000 ETF',
    'DIA': 'SPDR Dow Jones Industrial Average ETF',
    'VTI': 'Vanguard Total Stock Market ETF',

    # ============================================================================
    # LSE - London Stock Exchange (UK)
    # ============================================================================
    'BP.L': 'BP p.l.c.',
    'SHEL.L': 'Shell plc',
    'HSBA.L': 'HSBC Holdings plc',
    'AZN.L': 'AstraZeneca PLC',
    'GSK.L': 'GSK plc',
    'ULVR.L': 'Unilever PLC',
    'DGE.L': 'Diageo plc',
    'RIO.L': 'Rio Tinto Group',
    'BHP.L': 'BHP Group Limited',
    'BARC.L': 'Barclays PLC',
    'LLOY.L': 'Lloyds Banking Group plc',
    'VOD.L': 'Vodafone Group Plc',
    'BT.A.L': 'BT Group plc',
    'TSCO.L': 'Tesco PLC',
    'PRU.L': 'Prudential plc',

    # ============================================================================
    # TSE - Tokyo Stock Exchange (Japan)
    # ============================================================================
    '7203.T': 'Toyota Motor Corporation',
    '6758.T': 'Sony Group Corporation',
    '9984.T': 'SoftBank Group Corp.',
    '6861.T': 'Keyence Corporation',
    '8306.T': 'Mitsubishi UFJ Financial Group',
    '7267.T': 'Honda Motor Co., Ltd.',
    '6902.T': 'Denso Corporation',
    '6954.T': 'Fanuc Corporation',
    '4063.T': 'Shin-Etsu Chemical Co., Ltd.',
    '8035.T': 'Tokyo Electron Limited',
    '4502.T': 'Takeda Pharmaceutical Company',
    '9432.T': 'Nippon Telegraph and Telephone',

    # ============================================================================
    # SSE - Shanghai Stock Exchange (China)
    # ============================================================================
    '600519.SS': 'Kweichow Moutai Co., Ltd.',
    '601318.SS': 'Ping An Insurance Group',
    '600036.SS': 'China Merchants Bank',
    '600276.SS': 'Jiangsu Hengrui Medicine Co.',
    '601398.SS': 'Industrial and Commercial Bank of China',
    '600000.SS': 'Shanghai Pudong Development Bank',
    '600887.SS': 'Inner Mongolia Yili Industrial Group',
    '600030.SS': 'CITIC Securities Company Limited',

    # ============================================================================
    # HKEX - Hong Kong Stock Exchange
    # ============================================================================
    '0700.HK': 'Tencent Holdings Limited',
    '0941.HK': 'China Mobile Limited',
    '1299.HK': 'AIA Group Limited',
    '0005.HK': 'HSBC Holdings plc',
    '0939.HK': 'China Construction Bank',
    '1398.HK': 'Industrial and Commercial Bank of China',
    '0388.HK': 'Hong Kong Exchanges and Clearing',
    '9988.HK': 'Alibaba Group Holding Limited',
    '3690.HK': 'Meituan',
    '1810.HK': 'Xiaomi Corporation',

    # ============================================================================
    # TSX - Toronto Stock Exchange (Canada)
    # ============================================================================
    'SHOP.TO': 'Shopify Inc.',
    'RY.TO': 'Royal Bank of Canada',
    'TD.TO': 'The Toronto-Dominion Bank',
    'BNS.TO': 'The Bank of Nova Scotia',
    'BMO.TO': 'Bank of Montreal',
    'ENB.TO': 'Enbridge Inc.',
    'CNQ.TO': 'Canadian Natural Resources',
    'CNR.TO': 'Canadian National Railway',
    'CP.TO': 'Canadian Pacific Railway',
    'SU.TO': 'Suncor Energy Inc.',
    'ABX.TO': 'Barrick Gold Corporation',

    # ============================================================================
    # Euronext (France, Netherlands, Belgium)
    # ============================================================================
    'MC.PA': 'LVMH Moët Hennessy Louis Vuitton',
    'OR.PA': "L'Oréal S.A.",
    'SAN.PA': 'Sanofi',
    'TTE.PA': 'TotalEnergies SE',
    'AIR.PA': 'Airbus SE',
    'ASML.AS': 'ASML Holding N.V.',
    'HEIA.AS': 'Heineken N.V.',
    'ADYEN.AS': 'Adyen N.V.',
    'SAP.DE': 'SAP SE',
    'VOW3.DE': 'Volkswagen AG',
    'SIE.DE': 'Siemens AG',
    'BAYN.DE': 'Bayer AG',
}


def search_stocks(query: str, max_results: int = 10) -> List[Tuple[str, str]]:
    """
    Search for stocks by company name or ticker.

    Args:
        query: Search query (company name or ticker)
        max_results: Maximum number of results to return

    Returns:
        List of (ticker, company_name) tuples
    """
    query = query.lower().strip()

    if not query:
        return []

    results = []

    for ticker, company_name in STOCK_DATABASE.items():
        # Search in ticker
        if query in ticker.lower():
            results.append((ticker, company_name))
            continue

        # Search in company name
        if query in company_name.lower():
            results.append((ticker, company_name))

    # Sort by relevance (exact match first, then starts with, then contains)
    def relevance_score(item):
        ticker, name = item
        ticker_lower = ticker.lower()
        name_lower = name.lower()

        # Exact match
        if query == ticker_lower or query == name_lower:
            return 0
        # Ticker starts with query
        elif ticker_lower.startswith(query):
            return 1
        # Name starts with query
        elif name_lower.startswith(query):
            return 2
        # Contains in ticker
        elif query in ticker_lower:
            return 3
        # Contains in name
        else:
            return 4

    results.sort(key=relevance_score)

    return results[:max_results]


def stock_search_widget(key: str = "stock_search", default_ticker: str = "AAPL") -> str:
    """
    Streamlit widget for stock search with autocomplete.

    Args:
        key: Unique key for the widget
        default_ticker: Default ticker to show

    Returns:
        Selected ticker symbol
    """
    st.markdown("### 🔍 Stock Search")

    # Search box
    search_query = st.text_input(
        "Search by company name or ticker",
        placeholder="e.g., Apple, MSFT, Tesla...",
        key=f"{key}_input",
        help="Type company name or ticker symbol"
    )

    selected_ticker = default_ticker

    if search_query:
        results = search_stocks(search_query, max_results=10)

        if results:
            st.markdown("**Search Results:**")

            # Create selection options
            options = {f"{ticker} - {name}": ticker for ticker, name in results}

            selected = st.radio(
                "Select a stock:",
                options=list(options.keys()),
                key=f"{key}_radio",
                label_visibility="collapsed"
            )

            selected_ticker = options[selected]

            # Show confirmation
            st.success(f"✅ Selected: **{selected_ticker}**")

        else:
            st.warning("No results found. Try a different search term.")
            # Fallback to manual entry
            manual_ticker = st.text_input(
                "Or enter ticker manually:",
                value=default_ticker,
                key=f"{key}_manual"
            )
            selected_ticker = manual_ticker.upper()
    else:
        # No search query - show manual entry
        selected_ticker = st.text_input(
            "Enter ticker symbol:",
            value=default_ticker,
            key=f"{key}_ticker"
        ).upper()

    return selected_ticker


def compact_stock_search(key: str = "search", default: str = "AAPL") -> str:
    """
    Compact version of stock search with autocomplete (for sidebar).
    Shows matching stocks as user types - type "ap" to see Apple, etc.

    Args:
        key: Unique key for the widget
        default: Default ticker

    Returns:
        Selected ticker
    """
    # Create searchable list of all stocks in format "AAPL - Apple Inc."
    all_options = [f"{ticker} - {name}" for ticker, name in STOCK_DATABASE.items()]

    # Find default option
    default_option = None
    for option in all_options:
        if option.startswith(f"{default} -"):
            default_option = option
            break

    # If default not found, use first option
    if default_option is None:
        default_option = all_options[0]

    default_index = all_options.index(default_option)

    # Selectbox with autocomplete - user can type to filter
    selected = st.selectbox(
        "Select Stock",
        options=all_options,
        index=default_index,
        key=key,
        help="Type company name or ticker to filter (e.g., 'ap' shows Apple, 'mic' shows Microsoft)"
    )

    # Extract ticker from "AAPL - Apple Inc." format
    if selected:
        ticker = selected.split(" - ")[0]
        return ticker

    return default


def get_stock_info(ticker: str) -> Optional[dict]:
    """
    Get detailed stock information from yfinance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with stock info or None if error
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'exchange': info.get('exchange', 'N/A'),
            'currency': info.get('currency', 'USD'),
            'market_cap': info.get('marketCap', 0),
        }
    except Exception as e:
        return None
