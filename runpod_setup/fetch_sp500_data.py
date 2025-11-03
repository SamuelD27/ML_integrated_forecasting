#!/usr/bin/env python3
"""
Fetch complete S&P 500 dataset for B200 training.
500 tickers × 25 years = ~12.5M rows (compressed with smart downsampling)

Features:
- Extended historical data (25 years)
- Smart downsampling (daily/weekly/monthly)
- Zstd level 9 compression
- 3-5x storage reduction
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Import from existing script
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fetch_training_data_large import (
    fetch_all_tickers,
    add_technical_features,
    logger
)
from data.storage.compression_utils import DataCompressor, compress_training_data

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


def save_compressed_data(df, output_dir: Path, apply_downsampling: bool = True):
    """
    Save data with compression and optional downsampling.

    Args:
        df: DataFrame to save
        output_dir: Output directory
        apply_downsampling: Whether to apply smart downsampling
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    compressor = DataCompressor()

    # Apply downsampling if requested
    if apply_downsampling:
        logger.info("Applying smart downsampling (daily/weekly/monthly)...")
        df = compressor.apply_smart_downsampling(df, date_column='Date')

    # Compress and save
    output_path = output_dir / 'training_data_compressed.parquet'
    stats = compressor.compress_and_save(
        df,
        output_path,
        validate=True,
        metadata={
            'n_tickers': df['ticker'].nunique(),
            'tickers': sorted(df['ticker'].unique().tolist()),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'downsampled': apply_downsampling,
        }
    )

    # Save metadata
    import json
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"Saved compressed data: {output_path}")
    logger.info(f"Compression ratio: {stats['compression_ratio']}x")

    # Also save a legacy uncompressed version for compatibility
    legacy_path = output_dir / 'training_data.parquet'
    df.to_parquet(legacy_path, compression='snappy', index=False)
    logger.info(f"Saved legacy format: {legacy_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total rows: {stats['rows']:,}")
    print(f"Total tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Columns: {stats['columns']}")
    print(f"\nCompression:")
    print(f"  Original size: {stats['original_size_mb']} MB")
    print(f"  Compressed size: {stats['compressed_size_mb']} MB")
    print(f"  Compression ratio: {stats['compression_ratio']}x")
    print(f"  Downsampling: {'Yes' if apply_downsampling else 'No'}")
    print("=" * 80)


def main():
    """Fetch S&P 500 data with extended history and compression."""
    print("\n" + "=" * 80)
    print("S&P 500 DATA FETCHING FOR B200 TRAINING (EXTENDED)")
    print("=" * 80)

    # Configuration
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=25*365 + 6)).strftime('%Y-%m-%d')  # 25 years + leap days
    INTERVAL = '1d'
    OUTPUT_DIR = Path('/workspace/data/training')
    APPLY_DOWNSAMPLING = True  # Enable smart downsampling

    print(f"\nConfiguration:")
    print(f"  Start date: {START_DATE} (25 years)")
    print(f"  End date: {END_DATE}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Tickers: {len(SP500_TICKERS)}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Smart downsampling: {APPLY_DOWNSAMPLING}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch data
    print("\n" + "-" * 80)
    print("STEP 1: FETCHING RAW DATA (25 YEARS)")
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

    # Drop NaN rows
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    logger.info(f"Dropped {dropped_rows:,} rows with NaN values ({dropped_rows/initial_rows*100:.1f}%)")

    # Step 3: Compress and save
    print("\n" + "-" * 80)
    print("STEP 3: COMPRESSING AND SAVING DATA")
    print("-" * 80)

    save_compressed_data(df, OUTPUT_DIR, apply_downsampling=APPLY_DOWNSAMPLING)

    print("\n" + "=" * 80)
    print("✅ S&P 500 DATA FETCHING COMPLETE!")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  python train_b200.py")
    print("")


if __name__ == '__main__':
    main()
