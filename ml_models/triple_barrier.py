"""
Triple Barrier Labeling
=======================
Implements the triple barrier labeling method for meta-labeling.

Triple barrier method labels trades based on three conditions:
1. Take profit barrier (upper): Price reaches take profit level
2. Stop loss barrier (lower): Price reaches stop loss level
3. Time barrier (horizontal): Maximum holding period expires

Labels:
- 1: Take profit hit (successful trade)
- 0: Stop loss hit or time barrier reached without profit

Reference:
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def label_triple_barrier(
    price_series: pd.Series,
    tp_pct: float = 0.05,
    sl_pct: float = 0.02,
    max_holding_days: int = 10,
    neutral_label: int = 0,
    use_log_returns: bool = True,
) -> pd.DataFrame:
    """
    Apply triple barrier labeling to a price series.

    Args:
        price_series: Series of prices (indexed by date)
        tp_pct: Take profit percentage (e.g., 0.05 = 5%)
        sl_pct: Stop loss percentage (e.g., 0.02 = 2%)
        max_holding_days: Maximum holding period in trading days
        neutral_label: Label for time barrier hit (0 = loss, -1 = neutral)
        use_log_returns: Whether to use log returns for barrier calculation

    Returns:
        DataFrame with columns:
        - entry_date: Entry date
        - exit_date: Exit date
        - holding_days: Days held
        - return_pct: Actual return
        - barrier_hit: Which barrier was hit ('tp', 'sl', 'time')
        - label: Binary label (1=success, 0=failure)

    Example:
        >>> prices = yf.download('AAPL', start='2023-01-01', end='2024-01-01')['Close']
        >>> labels = label_triple_barrier(prices, tp_pct=0.05, sl_pct=0.02, max_holding_days=10)
        >>> print(f"Win rate: {labels['label'].mean():.2%}")
    """
    if len(price_series) < max_holding_days:
        logger.warning(f"Price series too short ({len(price_series)} < {max_holding_days})")
        return pd.DataFrame()

    results = []

    for i in range(len(price_series) - max_holding_days):
        entry_date = price_series.index[i]
        entry_price = price_series.iloc[i]

        # Calculate barriers
        if use_log_returns:
            upper_barrier = entry_price * np.exp(tp_pct)
            lower_barrier = entry_price * np.exp(-sl_pct)
        else:
            upper_barrier = entry_price * (1 + tp_pct)
            lower_barrier = entry_price * (1 - sl_pct)

        # Get prices for holding period
        future_prices = price_series.iloc[i + 1 : i + 1 + max_holding_days]

        # Find first barrier hit
        exit_date = None
        exit_price = None
        barrier_hit = None
        holding_days = 0

        for j, (date, price) in enumerate(future_prices.items()):
            holding_days = j + 1

            # Check upper barrier (take profit)
            if price >= upper_barrier:
                exit_date = date
                exit_price = price
                barrier_hit = 'tp'
                break

            # Check lower barrier (stop loss)
            if price <= lower_barrier:
                exit_date = date
                exit_price = price
                barrier_hit = 'sl'
                break

        # If no barrier hit, use time barrier
        if barrier_hit is None:
            exit_date = future_prices.index[-1]
            exit_price = future_prices.iloc[-1]
            barrier_hit = 'time'
            holding_days = max_holding_days

        # Calculate return
        if use_log_returns:
            return_pct = np.log(exit_price / entry_price)
        else:
            return_pct = (exit_price - entry_price) / entry_price

        # Assign label
        if barrier_hit == 'tp':
            label = 1
        elif barrier_hit == 'sl':
            label = 0
        else:  # time barrier
            label = 1 if return_pct > 0 else neutral_label

        results.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'holding_days': holding_days,
            'return_pct': return_pct,
            'barrier_hit': barrier_hit,
            'label': label,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.set_index('entry_date')

    return df


def get_triple_barrier_labels(
    prices: pd.Series,
    config: Optional[dict] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Get triple barrier labels with default config from trading.yaml.

    Args:
        prices: Price series
        config: Optional configuration dict

    Returns:
        Tuple of (labels, full_results_df)
    """
    # Default config
    default_config = {
        'take_profit_pct': 0.05,
        'stop_loss_pct': 0.02,
        'max_holding_days': 10,
        'neutral_label': 0,
    }

    # Override with provided config
    if config:
        meta_config = config.get('meta_labeling', config.get('triple_barrier', {}))
        if 'triple_barrier' in meta_config:
            meta_config = meta_config['triple_barrier']

        default_config.update({
            'take_profit_pct': meta_config.get('take_profit_pct', default_config['take_profit_pct']),
            'stop_loss_pct': meta_config.get('stop_loss_pct', default_config['stop_loss_pct']),
            'max_holding_days': meta_config.get('max_holding_days', default_config['max_holding_days']),
            'neutral_label': meta_config.get('neutral_label', default_config['neutral_label']),
        })

    results_df = label_triple_barrier(
        price_series=prices,
        tp_pct=default_config['take_profit_pct'],
        sl_pct=default_config['stop_loss_pct'],
        max_holding_days=default_config['max_holding_days'],
        neutral_label=default_config['neutral_label'],
    )

    labels = results_df['label'] if not results_df.empty else pd.Series(dtype=int)

    return labels, results_df


def calculate_barrier_stats(results_df: pd.DataFrame) -> dict:
    """
    Calculate statistics from triple barrier labeling results.

    Args:
        results_df: Results from label_triple_barrier

    Returns:
        Dict with statistics:
        - win_rate: Percentage of winning trades
        - avg_return: Average return per trade
        - avg_holding: Average holding period
        - tp_rate: Take profit hit rate
        - sl_rate: Stop loss hit rate
        - time_rate: Time barrier hit rate
    """
    if results_df.empty:
        return {
            'win_rate': 0.0,
            'avg_return': 0.0,
            'avg_holding': 0.0,
            'tp_rate': 0.0,
            'sl_rate': 0.0,
            'time_rate': 0.0,
        }

    n = len(results_df)

    return {
        'win_rate': float(results_df['label'].mean()),
        'avg_return': float(results_df['return_pct'].mean()),
        'avg_holding': float(results_df['holding_days'].mean()),
        'tp_rate': float((results_df['barrier_hit'] == 'tp').sum() / n),
        'sl_rate': float((results_df['barrier_hit'] == 'sl').sum() / n),
        'time_rate': float((results_df['barrier_hit'] == 'time').sum() / n),
    }


def label_with_volatility_scaling(
    price_series: pd.Series,
    vol_window: int = 20,
    vol_mult_tp: float = 2.0,
    vol_mult_sl: float = 1.0,
    max_holding_days: int = 10,
) -> pd.DataFrame:
    """
    Triple barrier labeling with volatility-scaled barriers.

    Instead of fixed percentages, scales barriers based on recent volatility.
    More appropriate for varying market conditions.

    Args:
        price_series: Price series
        vol_window: Window for volatility calculation
        vol_mult_tp: Multiplier for take profit (e.g., 2 = 2x daily vol)
        vol_mult_sl: Multiplier for stop loss (e.g., 1 = 1x daily vol)
        max_holding_days: Maximum holding period

    Returns:
        DataFrame with labeling results
    """
    # Calculate daily volatility
    returns = np.log(price_series / price_series.shift(1))
    rolling_vol = returns.rolling(vol_window).std()

    results = []

    for i in range(vol_window, len(price_series) - max_holding_days):
        entry_date = price_series.index[i]
        entry_price = price_series.iloc[i]
        daily_vol = rolling_vol.iloc[i]

        if pd.isna(daily_vol) or daily_vol < 1e-8:
            continue

        # Scale barriers by volatility
        tp_pct = daily_vol * vol_mult_tp * np.sqrt(max_holding_days)
        sl_pct = daily_vol * vol_mult_sl * np.sqrt(max_holding_days)

        upper_barrier = entry_price * np.exp(tp_pct)
        lower_barrier = entry_price * np.exp(-sl_pct)

        future_prices = price_series.iloc[i + 1 : i + 1 + max_holding_days]

        exit_date = None
        exit_price = None
        barrier_hit = None
        holding_days = 0

        for j, (date, price) in enumerate(future_prices.items()):
            holding_days = j + 1

            if price >= upper_barrier:
                exit_date = date
                exit_price = price
                barrier_hit = 'tp'
                break

            if price <= lower_barrier:
                exit_date = date
                exit_price = price
                barrier_hit = 'sl'
                break

        if barrier_hit is None:
            exit_date = future_prices.index[-1]
            exit_price = future_prices.iloc[-1]
            barrier_hit = 'time'
            holding_days = max_holding_days

        return_pct = np.log(exit_price / entry_price)

        if barrier_hit == 'tp':
            label = 1
        elif barrier_hit == 'sl':
            label = 0
        else:
            label = 1 if return_pct > 0 else 0

        results.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'holding_days': holding_days,
            'return_pct': return_pct,
            'barrier_hit': barrier_hit,
            'label': label,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'daily_vol': daily_vol,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.set_index('entry_date')

    return df
