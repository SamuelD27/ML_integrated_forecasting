"""
Temporal Fusion Transformer (TFT) for Stock Forecasting
========================================================
State-of-the-art time series forecasting with attention and interpretability.

Key Features:
1. Variable Selection Networks (VSN) for feature importance
2. Multi-horizon forecasting (1-day to 20-day ahead)
3. Quantile regression for uncertainty estimation
4. Temporal attention for interpretable predictions
5. Static, known, and unknown covariates

Reference:
- Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- International Journal of Forecasting
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""

    # Model architecture
    hidden_size: int = 160
    lstm_layers: int = 2
    dropout: float = 0.1
    attention_head_size: int = 4
    hidden_continuous_size: int = 8

    # Training
    max_encoder_length: int = 60  # Lookback window
    max_prediction_length: int = 20  # Forecast horizon
    batch_size: int = 128
    learning_rate: float = 0.001
    max_epochs: int = 50
    gradient_clip_val: float = 0.1

    # Quantile regression
    quantiles: List[float] = None  # e.g., [0.1, 0.5, 0.9]

    # Data
    time_idx_col: str = 'time_idx'
    target_col: str = 'target'
    group_ids: List[str] = None  # e.g., ['ticker']

    # Features
    static_categoricals: List[str] = None  # e.g., ['sector']
    static_reals: List[str] = None
    time_varying_known_categoricals: List[str] = None
    time_varying_known_reals: List[str] = None  # e.g., ['day_of_week', 'month']
    time_varying_unknown_categoricals: List[str] = None
    time_varying_unknown_reals: List[str] = None  # e.g., ['returns', 'volume']

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        if self.group_ids is None:
            self.group_ids = ['ticker']
        if self.static_categoricals is None:
            self.static_categoricals = []
        if self.static_reals is None:
            self.static_reals = []
        if self.time_varying_known_categoricals is None:
            self.time_varying_known_categoricals = []
        if self.time_varying_known_reals is None:
            self.time_varying_known_reals = ['day_of_week', 'month']
        if self.time_varying_unknown_categoricals is None:
            self.time_varying_unknown_categoricals = []
        if self.time_varying_unknown_reals is None:
            self.time_varying_unknown_reals = []


class TFTStockForecaster:
    """
    Temporal Fusion Transformer for multi-stock forecasting.

    Example:
        >>> config = TFTConfig(max_encoder_length=60, max_prediction_length=20)
        >>> forecaster = TFTStockForecaster(config)
        >>> forecaster.fit(train_data)
        >>> predictions = forecaster.predict(test_data)
        >>> forecaster.plot_feature_importance()
    """

    def __init__(self, config: TFTConfig):
        """
        Initialize TFT forecaster.

        Args:
            config: TFT configuration
        """
        self.config = config
        self.model = None
        self.training_dataset = None
        self.trainer = None
        self.best_model_path = None

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = 'forward_returns',
        add_time_features: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for TFT training.

        Args:
            data: DataFrame with columns: ticker, date, features, target
            target_col: Name of target column
            add_time_features: Add temporal features (day_of_week, month, etc.)

        Returns:
            Prepared DataFrame with time_idx and all required columns
        """
        df = data.copy()

        # Ensure date column
        if 'date' not in df.columns and df.index.name == 'date':
            df = df.reset_index()

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Add time index (sequential integer per group)
        df = df.sort_values(['ticker', 'date'])
        df['time_idx'] = df.groupby('ticker').cumcount()

        # Add temporal features
        if add_time_features:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_month'] = df['date'].dt.day
            df['week_of_year'] = df['date'].dt.isocalendar().week

        # Rename target column
        if target_col in df.columns and target_col != self.config.target_col:
            df[self.config.target_col] = df[target_col]

        # Ensure group ID columns exist
        for group_id in self.config.group_ids:
            if group_id not in df.columns and group_id == 'ticker':
                # Already exists, skip
                pass

        logger.info(f"Prepared {len(df)} rows for {df['ticker'].nunique()} tickers")
        logger.info(f"Time range: {df['date'].min()} to {df['date'].max()}")

        return df

    def _classify_features(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Explicitly classify features as known vs unknown.

        Known features: We know them in advance (calendar, scheduled events)
        Unknown features: We don't know until they happen (prices, volumes, technical indicators)

        Returns:
            Tuple of (known_features, unknown_features)
        """
        # Known features - we know these in advance
        known_patterns = [
            'day_of_week', 'month', 'quarter', 'year',
            'is_month_end', 'is_month_start', 'is_quarter_end',
            'is_year_end', 'days_in_month', 'week_of_year',
            'day_of_month', 'day_of_year', 'is_weekend',
            'is_holiday', 'time_idx',  # Explicit time features
        ]

        # Unknown features - we don't know these until they happen
        unknown_patterns = [
            'open', 'high', 'low', 'close', 'volume', 'adj_close', 'adj close',
            'return', 'log_return', 'volatility', 'momentum', 'vwap',
            'rsi', 'macd', 'bollinger', 'bb_', 'atr', 'obv',
            'ema', 'sma', 'wma', 'stdev', 'std', 'var',
            'beta', 'alpha', 'sharpe', 'sortino',
            'drawdown', 'skew', 'kurt', 'corr',
            'spread', 'bid', 'ask', 'trade',
            'norm_', 'z_score', 'zscore',  # Normalized/standardized features
            'fft_', 'wavelet_',  # Frequency domain features
            'hl_', 'oc_', 'price_', 'vol_',  # Composite features
        ]

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Exclude target, time index, group ids
        exclude_cols = [self.config.target_col, self.config.time_idx_col] + self.config.group_ids

        known_features = []
        unknown_features = []

        for col in numeric_cols:
            if col in exclude_cols:
                continue

            col_lower = col.lower()

            # Check if it's a known feature
            if any(pattern in col_lower for pattern in known_patterns):
                known_features.append(col)
            # Check if it's an unknown feature
            elif any(pattern in col_lower for pattern in unknown_patterns):
                unknown_features.append(col)
            else:
                # Default: treat as unknown (safer assumption)
                logger.warning(f"Feature '{col}' classification unclear, treating as unknown")
                unknown_features.append(col)

        return known_features, unknown_features

    def create_dataset(
        self,
        data: pd.DataFrame,
        training: bool = True
    ) -> TimeSeriesDataSet:
        """
        Create TimeSeriesDataSet for TFT.

        Args:
            data: Prepared DataFrame
            training: If True, create training dataset; else validation dataset

        Returns:
            TimeSeriesDataSet
        """
        # Update config with explicit feature classification
        if not self.config.time_varying_unknown_reals or not self.config.time_varying_known_reals:
            known_features, unknown_features = self._classify_features(data)

            # Update config
            if not self.config.time_varying_known_reals:
                self.config.time_varying_known_reals = known_features

            if not self.config.time_varying_unknown_reals:
                # Limit to top 30 unknown features to avoid model complexity
                self.config.time_varying_unknown_reals = unknown_features[:30]

            logger.info(f"Classified features:")
            logger.info(f"  Known features ({len(self.config.time_varying_known_reals)}): "
                       f"{self.config.time_varying_known_reals}")
            logger.info(f"  Unknown features ({len(self.config.time_varying_unknown_reals)}): "
                       f"{self.config.time_varying_unknown_reals[:10]}..."  # Show first 10
                       f"(total: {len(self.config.time_varying_unknown_reals)})")

        if training:
            dataset = TimeSeriesDataSet(
                data,
                time_idx=self.config.time_idx_col,
                target=self.config.target_col,
                group_ids=self.config.group_ids,
                min_encoder_length=self.config.max_encoder_length // 2,
                max_encoder_length=self.config.max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=self.config.max_prediction_length,
                static_categoricals=self.config.static_categoricals,
                static_reals=self.config.static_reals,
                time_varying_known_categoricals=self.config.time_varying_known_categoricals,
                time_varying_known_reals=self.config.time_varying_known_reals,
                time_varying_unknown_categoricals=self.config.time_varying_unknown_categoricals,
                time_varying_unknown_reals=self.config.time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(
                    groups=self.config.group_ids,
                    transformation="softplus"
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

            self.training_dataset = dataset

        else:
            # Validation dataset (uses training dataset's encoders)
            dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                data,
                predict=True,
                stop_randomization=True
            )

        return dataset

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        accelerator: str = 'auto',
        devices: Union[int, str] = 'auto'
    ) -> Dict:
        """
        Train TFT model.

        Args:
            train_data: Training data (prepared with prepare_data)
            val_data: Validation data (optional)
            accelerator: 'gpu', 'cpu', or 'auto'
            devices: Number of devices or 'auto'

        Returns:
            Dictionary with training history
        """
        logger.info("Creating training dataset...")
        train_dataset = self.create_dataset(train_data, training=True)
        train_dataloader = train_dataset.to_dataloader(
            train=True,
            batch_size=self.config.batch_size,
            num_workers=0
        )

        # Validation dataset
        if val_data is not None:
            val_dataset = self.create_dataset(val_data, training=False)
            val_dataloader = val_dataset.to_dataloader(
                train=False,
                batch_size=self.config.batch_size * 2,
                num_workers=0
            )
        else:
            val_dataloader = None

        # Configure trainer
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=10,
            verbose=True,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            dirpath='checkpoints/tft',
            filename='tft-{epoch:02d}-{val_loss:.4f}'
        )

        self.trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=accelerator,
            devices=devices,
            gradient_clip_val=self.config.gradient_clip_val,
            callbacks=[early_stop_callback, checkpoint_callback],
            enable_progress_bar=True,
            logger=True
        )

        # Create model
        logger.info("Creating TFT model...")
        self.model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_continuous_size,
            loss=QuantileLoss(quantiles=self.config.quantiles),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        # Train
        logger.info(f"Training TFT for {self.config.max_epochs} epochs...")
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        self.best_model_path = checkpoint_callback.best_model_path

        return {
            'best_model_path': self.best_model_path,
            'best_val_loss': checkpoint_callback.best_model_score.item()
        }

    def predict(
        self,
        data: pd.DataFrame,
        mode: str = 'quantiles',
        return_index: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Generate predictions.

        Args:
            data: Data to predict on
            mode: 'quantiles' or 'mean'
            return_index: Return predictions as DataFrame with index

        Returns:
            Predictions (quantiles or mean)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Create dataset
        pred_dataset = self.create_dataset(data, training=False)
        pred_dataloader = pred_dataset.to_dataloader(
            train=False,
            batch_size=self.config.batch_size * 2,
            num_workers=0
        )

        # Predict
        raw_predictions = self.model.predict(pred_dataloader, mode='raw', return_x=True)

        if mode == 'quantiles':
            # Return all quantiles
            predictions = raw_predictions.output['prediction']  # Shape: (n_samples, pred_len, n_quantiles)

            if return_index:
                # Convert to DataFrame
                # TODO: Add proper index mapping
                return pd.DataFrame(
                    predictions[:, :, len(self.config.quantiles) // 2].numpy(),  # Median
                    columns=[f't+{i}' for i in range(1, self.config.max_prediction_length + 1)]
                )
            else:
                return predictions.numpy()

        elif mode == 'mean':
            # Return mean prediction (median quantile)
            median_idx = len(self.config.quantiles) // 2
            predictions = raw_predictions.output['prediction'][:, :, median_idx]

            if return_index:
                return pd.DataFrame(
                    predictions.numpy(),
                    columns=[f't+{i}' for i in range(1, self.config.max_prediction_length + 1)]
                )
            else:
                return predictions.numpy()

    def predict_with_uncertainty(
        self,
        data: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions with uncertainty intervals.

        Args:
            data: Data to predict on
            confidence_level: Confidence level (default: 95%)

        Returns:
            Dictionary with 'mean', 'lower', 'upper' DataFrames
        """
        # Get quantile predictions
        predictions = self.predict(data, mode='quantiles', return_index=False)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        # Find closest quantiles
        lower_idx = np.argmin(np.abs(np.array(self.config.quantiles) - lower_q))
        upper_idx = np.argmin(np.abs(np.array(self.config.quantiles) - upper_q))
        median_idx = len(self.config.quantiles) // 2

        columns = [f't+{i}' for i in range(1, self.config.max_prediction_length + 1)]

        return {
            'mean': pd.DataFrame(predictions[:, :, median_idx], columns=columns),
            'lower': pd.DataFrame(predictions[:, :, lower_idx], columns=columns),
            'upper': pd.DataFrame(predictions[:, :, upper_idx], columns=columns)
        }

    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """Plot feature importance from Variable Selection Networks."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        interpretation = self.model.interpret_output(
            self.training_dataset.to_dataloader(train=False, batch_size=128, num_workers=0),
            reduction="sum"
        )

        # Encoder variable importance
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Static variables
        if len(interpretation['static_variables']) > 0:
            static_importance = interpretation['static_variables'].mean(0)
            top_static = static_importance.nlargest(top_n)

            axes[0].barh(range(len(top_static)), top_static.values)
            axes[0].set_yticks(range(len(top_static)))
            axes[0].set_yticklabels(top_static.index)
            axes[0].set_xlabel('Importance')
            axes[0].set_title('Static Variable Importance')

        # Encoder variables
        encoder_importance = interpretation['encoder_variables'].mean(0)
        top_encoder = encoder_importance.nlargest(top_n)

        axes[1].barh(range(len(top_encoder)), top_encoder.values)
        axes[1].set_yticks(range(len(top_encoder)))
        axes[1].set_yticklabels(top_encoder.index)
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Encoder Variable Importance')

        plt.tight_layout()
        plt.savefig('reports/tft_feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info("Feature importance plot saved to reports/tft_feature_importance.png")

        return interpretation

    def plot_attention(self, sample_idx: int = 0):
        """Plot temporal attention weights."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Get attention weights
        attention = self.model.interpret_output(
            self.training_dataset.to_dataloader(train=False, batch_size=1, num_workers=0),
            reduction=None
        )['attention'][sample_idx]

        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention.T,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.xlabel('Prediction Horizon')
        plt.ylabel('Encoder Time Step')
        plt.title('Temporal Attention Weights')
        plt.tight_layout()
        plt.savefig('reports/tft_attention.png', dpi=300, bbox_inches='tight')
        logger.info("Attention plot saved to reports/tft_attention.png")


def create_tft_forecaster(
    lookback_window: int = 60,
    forecast_horizon: int = 20,
    hidden_size: int = 160,
    learning_rate: float = 0.001,
    max_epochs: int = 50,
    **kwargs
) -> TFTStockForecaster:
    """
    Convenience function to create TFT forecaster with common defaults.

    Args:
        lookback_window: Number of historical time steps
        forecast_horizon: Number of future time steps to predict
        hidden_size: Hidden layer size
        learning_rate: Learning rate
        max_epochs: Maximum training epochs
        **kwargs: Additional config parameters

    Returns:
        TFTStockForecaster instance

    Example:
        >>> forecaster = create_tft_forecaster(lookback_window=60, forecast_horizon=20)
        >>> forecaster.fit(train_data)
    """
    config = TFTConfig(
        max_encoder_length=lookback_window,
        max_prediction_length=forecast_horizon,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        **kwargs
    )

    return TFTStockForecaster(config)


if __name__ == '__main__':
    # Example usage
    print("TFT Stock Forecaster Example")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    data_list = []
    for ticker in tickers:
        for date in dates:
            data_list.append({
                'ticker': ticker,
                'date': date,
                'returns': np.random.normal(0.001, 0.02),
                'volume': np.random.lognormal(15, 1),
                'momentum_20': np.random.normal(0, 0.1),
                'volatility_20': np.random.uniform(0.1, 0.3),
                'rsi_14': np.random.uniform(30, 70),
                'forward_returns': np.random.normal(0.001, 0.02)  # Target
            })

    data = pd.DataFrame(data_list)

    print(f"Generated {len(data)} samples for {len(tickers)} tickers")

    # Create forecaster
    forecaster = create_tft_forecaster(
        lookback_window=60,
        forecast_horizon=20,
        max_epochs=5  # Quick test
    )

    # Prepare data
    data_prepared = forecaster.prepare_data(data, target_col='forward_returns')

    # Split train/val
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]

    train_data = data_prepared[data_prepared['date'] < split_date]
    val_data = data_prepared[data_prepared['date'] >= split_date]

    print(f"\nTrain: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")

    # Train (commented out for quick import test)
    # print("\nTraining TFT...")
    # forecaster.fit(train_data, val_data)

    # Predict
    # predictions = forecaster.predict_with_uncertainty(val_data)
    # print("\nPredictions shape:", predictions['mean'].shape)

    print("\n✓ TFT module ready")
