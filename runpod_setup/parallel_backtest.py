"""
Parallel Backtesting Across Multiple GPUs

Splits tickers across 3 GPUs for parallel walk-forward backtesting.
Aggregates results and generates comprehensive reports.

Usage:
    python runpod_setup/parallel_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest_on_gpu(
    gpu_id: int,
    tickers: List[str],
    data_path: Path,
    output_dir: Path,
    config: Dict
) -> Dict:
    """
    Run backtest for a subset of tickers on a specific GPU.

    Args:
        gpu_id: GPU device ID
        tickers: List of tickers to backtest
        data_path: Path to data file
        output_dir: Output directory
        config: Backtest configuration

    Returns:
        Aggregated results
    """
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        logger.info(f"GPU {gpu_id}: Using device {device}")
    else:
        device = 'cpu'
        logger.warning(f"GPU {gpu_id}: CUDA not available, using CPU")

    # Load data
    logger.info(f"GPU {gpu_id}: Loading data for {len(tickers)} tickers...")

    try:
        from data.storage.compression_utils import DataCompressor
        compressor = DataCompressor()
        df = compressor.load_compressed(data_path)
    except:
        df = pd.read_parquet(data_path)

    # Filter to assigned tickers
    df = df[df['ticker'].isin(tickers)]

    logger.info(f"GPU {gpu_id}: Loaded {len(df):,} rows")

    # Import backtest module
    from run_backtest import WalkForwardBacktest

    # Initialize backtest
    backtest = WalkForwardBacktest(
        train_window_months=config.get('train_window_months', 24),
        test_window_months=config.get('test_window_months', 6),
        step_months=config.get('step_months', 1),
        initial_capital=config.get('initial_capital', 100000.0),
        commission=config.get('commission', 0.001),
        slippage=config.get('slippage', 0.0005),
    )

    # Prepare features
    df['returns'] = df.groupby('ticker')['close'].pct_change()
    df = df.dropna()

    feature_cols = [
        col for col in df.columns
        if col not in ['date', 'ticker', 'returns', 'Date']
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    # Initialize model (tree ensemble for speed)
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=4  # Use multiple CPU cores per GPU
    )

    # Run backtest
    logger.info(f"GPU {gpu_id}: Starting backtest for {len(tickers)} tickers...")

    results = backtest.run(
        data=df,
        model=model,
        feature_cols=feature_cols,
        target_col='returns',
        date_col='date'
    )

    # Save results
    output_file = output_dir / f'backtest_gpu{gpu_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results.to_csv(output_file, index=False)

    logger.info(f"GPU {gpu_id}: Backtest complete. Results saved to {output_file}")

    # Aggregate metrics
    agg_metrics = {
        'gpu_id': gpu_id,
        'n_tickers': len(tickers),
        'n_windows': len(results),
        'avg_sharpe': results['sharpe_ratio'].mean(),
        'avg_return': results['total_return'].mean(),
        'avg_drawdown': results['max_drawdown'].mean(),
        'avg_win_rate': results['win_rate'].mean(),
        'results_file': str(output_file),
    }

    return agg_metrics


def parallel_backtest(
    tickers: List[str],
    n_gpus: int,
    data_path: Path,
    output_dir: Path,
    config: Dict
) -> pd.DataFrame:
    """
    Run parallel backtesting across multiple GPUs.

    Args:
        tickers: List of all tickers
        n_gpus: Number of GPUs to use
        data_path: Path to data file
        output_dir: Output directory
        config: Backtest configuration

    Returns:
        Aggregated results DataFrame
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split tickers across GPUs
    ticker_chunks = np.array_split(tickers, n_gpus)

    logger.info(f"Splitting {len(tickers)} tickers across {n_gpus} GPUs")
    for i, chunk in enumerate(ticker_chunks):
        logger.info(f"  GPU {i}: {len(chunk)} tickers")

    # Create process pool
    mp.set_start_method('spawn', force=True)

    # Run backtests in parallel
    with mp.Pool(processes=n_gpus) as pool:
        results = pool.starmap(
            run_backtest_on_gpu,
            [
                (gpu_id, ticker_chunks[gpu_id].tolist(), data_path, output_dir, config)
                for gpu_id in range(n_gpus)
            ]
        )

    # Aggregate results
    agg_df = pd.DataFrame(results)

    logger.info("\n" + "=" * 80)
    logger.info("PARALLEL BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Total tickers: {agg_df['n_tickers'].sum()}")
    logger.info(f"  Total windows: {agg_df['n_windows'].sum()}")
    logger.info(f"  Average Sharpe: {agg_df['avg_sharpe'].mean():.3f}")
    logger.info(f"  Average Return: {agg_df['avg_return'].mean()*100:.2f}%")
    logger.info(f"  Average Drawdown: {agg_df['avg_drawdown'].mean()*100:.2f}%")
    logger.info(f"  Average Win Rate: {agg_df['avg_win_rate'].mean()*100:.2f}%")
    logger.info("=" * 80)

    return agg_df


def main():
    """Main parallel backtesting pipeline."""

    print("\n" + "=" * 80)
    print("PARALLEL BACKTESTING ACROSS MULTIPLE GPUS")
    print("=" * 80)

    # Configuration
    DATA_PATH = Path('/workspace/data/training/training_data_compressed.parquet')
    OUTPUT_DIR = Path('/workspace/results/backtest_parallel')

    # Backtest configuration
    config = {
        'train_window_months': 24,
        'test_window_months': 6,
        'step_months': 1,
        'initial_capital': 100000.0,
        'commission': 0.001,
        'slippage': 0.0005,
    }

    # Get available GPUs
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"\nDetected {n_gpus} GPUs")

    # Load ticker list
    print(f"\nLoading ticker list from {DATA_PATH}...")

    try:
        from data.storage.compression_utils import DataCompressor
        compressor = DataCompressor()
        df = compressor.load_compressed(DATA_PATH, columns=['ticker'])
    except:
        df = pd.read_parquet(DATA_PATH, columns=['ticker'])

    tickers = df['ticker'].unique().tolist()
    print(f"Found {len(tickers)} unique tickers")

    # Run parallel backtest
    print(f"\nStarting parallel backtest on {n_gpus} GPUs...")

    results = parallel_backtest(
        tickers=tickers,
        n_gpus=n_gpus,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        config=config
    )

    # Save aggregated results
    agg_path = OUTPUT_DIR / f'parallel_backtest_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results.to_csv(agg_path, index=False)

    print(f"\n✅ Parallel backtest complete!")
    print(f"Aggregated results saved to: {agg_path}")


if __name__ == '__main__':
    main()
