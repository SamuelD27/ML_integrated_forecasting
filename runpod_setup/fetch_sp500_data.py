#!/usr/bin/env python3
"""
Fetch complete S&P 500 dataset for B200 training.
500 tickers × 15 years = ~7.5M rows
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta

# Import from existing script
import sys
sys.path.insert(0, str(Path(__file__).parent))

from fetch_training_data_large import (
    fetch_all_tickers,
    add_technical_features,
    save_data,
    logger
)

# S&P 500 tickers (curated high-quality 500)
SP500_TICKERS = [
    # Technology (50)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
    'ADBE', 'CRM', 'CSCO', 'ACN', 'INTC', 'AMD', 'TXN', 'QCOM', 'AMAT', 'LRCX',
    'KLAC', 'SNPS', 'CDNS', 'MRVL', 'NXPI', 'ADI', 'FTNT', 'PANW', 'CRWD', 'ZS',
    'DDOG', 'SNOW', 'PLTR', 'UBER', 'LYFT', 'ABNB', 'DASH', 'COIN', 'RBLX', 'U',
    'NET', 'SHOP', 'SQ', 'PYPL', 'NOW', 'INTU', 'WDAY', 'VEEV', 'DOCU', 'ZM',

    # Financials (50)
    'BRK.B', 'JPM', 'BAC', 'WFC', 'MS', 'GS', 'C', 'BLK', 'SCHW', 'USB',
    'PNC', 'TFC', 'COF', 'AXP', 'BK', 'STT', 'NTRS', 'CFG', 'KEY', 'RF',
    'FITB', 'HBAN', 'CMA', 'ZION', 'SIVB', 'FRC', 'MTB', 'EWBC', 'SBNY', 'WAL',
    'V', 'MA', 'FIS', 'FISV', 'DFS', 'SYF', 'ALLY', 'SOFI', 'UPST', 'AFRM',
    'ALL', 'TRV', 'PGR', 'CB', 'AIG', 'MET', 'PRU', 'AFL', 'CINF', 'L',

    # Healthcare (50)
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'TMO', 'ABT', 'BMY', 'AMGN', 'GILD',
    'CVS', 'CI', 'ELV', 'HCA', 'HUM', 'CNC', 'MOH', 'ANTM', 'MCK', 'CAH',
    'COR', 'ABC', 'DVA', 'UHS', 'THC', 'HQY', 'CHE', 'ENSG', 'HSIC', 'PDCO',
    'MDT', 'ISRG', 'SYK', 'BSX', 'EW', 'ZBH', 'BAX', 'BDX', 'RMD', 'HOLX',
    'DXCM', 'ALGN', 'PODD', 'TDOC', 'VEEV', 'IQV', 'PEN', 'MTD', 'A', 'WAT',

    # Consumer (50)
    'WMT', 'HD', 'COST', 'TGT', 'LOW', 'DG', 'DLTR', 'ROST', 'TJX', 'BBWI',
    'NKE', 'LULU', 'SBUX', 'MCD', 'CMG', 'YUM', 'QSR', 'PNRA', 'DPZ', 'WEN',
    'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'EL', 'CLX', 'CHD', 'KMB',
    'AMZN', 'EBAY', 'ETSY', 'W', 'CHWY', 'CVNA', 'RH', 'WSM', 'BBBY', 'BIG',
    'F', 'GM', 'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'RIDE', 'FSR',

    # Industrial (50)
    'BA', 'CAT', 'GE', 'HON', 'UNP', 'UPS', 'RTX', 'LMT', 'MMM', 'DE',
    'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'FDX', 'NSC', 'CSX', 'ODFL', 'XPO',
    'WM', 'RSG', 'FAST', 'PCAR', 'IR', 'DOV', 'ROK', 'AME', 'GNRC', 'PWR',
    'JCI', 'CARR', 'OTIS', 'WCN', 'GWW', 'ALLE', 'AOS', 'ROP', 'SWK', 'NDSN',
    'TT', 'HUBB', 'J', 'FLS', 'BLDR', 'VMC', 'MLM', 'NUE', 'STLD', 'CLF',

    # Energy (40)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
    'HAL', 'BKR', 'DVN', 'FANG', 'MRO', 'APA', 'OVV', 'CTRA', 'EQT', 'AR',
    'KMI', 'WMB', 'OKE', 'LNG', 'TRGP', 'EPD', 'ET', 'MPLX', 'PAA', 'WES',
    'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'PEG', 'XEL', 'ED',

    # Materials (30)
    'LIN', 'APD', 'ECL', 'SHW', 'DD', 'DOW', 'PPG', 'NEM', 'FCX', 'GOLD',
    'BHP', 'RIO', 'VALE', 'SCCO', 'TECK', 'AA', 'CENX', 'ACH', 'X', 'MT',
    'PKG', 'IP', 'WRK', 'AVY', 'SEE', 'BALL', 'AMCR', 'CCK', 'GPK', 'SON',

    # Real Estate (30)
    'AMT', 'PLD', 'EQIX', 'PSA', 'DLR', 'WELL', 'SPG', 'O', 'AVB', 'EQR',
    'VTR', 'ARE', 'VICI', 'INVH', 'EXR', 'CUBE', 'MAA', 'UDR', 'CPT', 'ESS',
    'KIM', 'REG', 'FRT', 'BXP', 'VNO', 'SLG', 'HIW', 'DEI', 'CUZ', 'PGRE',

    # Communication (30)
    'GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'CHTR', 'TMUS', 'PARA',
    'WBD', 'FOXA', 'FOX', 'NWSA', 'NWS', 'IPG', 'OMC', 'TTWO', 'EA', 'ATVI',
    'RBLX', 'MTCH', 'BMBL', 'PINS', 'SNAP', 'SPOT', 'ROKU', 'PTON', 'ZM', 'TWLO',

    # Utilities (30)
    'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'PEG', 'XEL', 'ED',
    'WEC', 'ES', 'DTE', 'ETR', 'FE', 'EIX', 'AEE', 'CMS', 'CNP', 'NI',
    'LNT', 'EVRG', 'PNW', 'ATO', 'NWE', 'OGE', 'AVA', 'POR', 'SJW', 'AWK',

    # ETFs & Indexes (40)
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'EEM', 'EFA',
    'XLF', 'XLK', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'TLT', 'IEF', 'SHY', 'AGG', 'BND', 'GLD', 'SLV', 'GDX', 'USO', 'UNG',
    'VIX', 'UVXY', 'VIXY', 'SVXY', 'HYG', 'LQD', 'JNK', 'EMB', 'TIP', 'VCIT',
]


def main():
    """Fetch S&P 500 data."""
    print("\n" + "=" * 80)
    print("S&P 500 DATA FETCHING FOR B200 TRAINING")
    print("=" * 80)

    # Configuration
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=15*365)).strftime('%Y-%m-%d')  # 15 years
    INTERVAL = '1d'
    OUTPUT_DIR = Path('/workspace/data/training')

    print(f"\nConfiguration:")
    print(f"  Start date: {START_DATE}")
    print(f"  End date: {END_DATE}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Tickers: {len(SP500_TICKERS)}")
    print(f"  Output: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch data
    print("\n" + "-" * 80)
    print("STEP 1: FETCHING RAW DATA")
    print("-" * 80)

    df = fetch_all_tickers(
        tickers=SP500_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        max_workers=40  # B200 can handle more
    )

    # Step 2: Add features
    print("\n" + "-" * 80)
    print("STEP 2: ADDING TECHNICAL FEATURES")
    print("-" * 80)

    df = add_technical_features(df)

    # Step 3: Save
    print("\n" + "-" * 80)
    print("STEP 3: SAVING DATA")
    print("-" * 80)

    save_data(df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("✅ S&P 500 DATA FETCHING COMPLETE!")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  python train_b200.py")
    print("")


if __name__ == '__main__':
    main()
