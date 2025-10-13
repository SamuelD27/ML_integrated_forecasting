"""
Training Utilities
==================

Utility functions for data loading, preprocessing, metrics calculation,
and visualization for the hybrid model training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class TradingDataset(Dataset):
    """
    Custom dataset for trading data with sequences.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                sequence_length: int, transform: Optional[Any] = None,
                augment: bool = False, noise_level: float = 0.01):
        """
        Initialize trading dataset.

        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples,) or (n_samples, n_targets)
            sequence_length: Length of sequences
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
            noise_level: Noise level for augmentation
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.transform = transform
        self.augment = augment
        self.noise_level = noise_level

        # Calculate valid indices
        self.valid_indices = list(range(len(features) - sequence_length))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        # Get sequence
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        sequence = self.features[start_idx:end_idx].copy()
        target = self.targets[end_idx] if len(self.targets) > end_idx else self.targets[-1]

        # Apply augmentation
        if self.augment and np.random.random() < 0.5:
            # Add gaussian noise
            noise = np.random.normal(0, self.noise_level, sequence.shape)
            sequence = sequence + noise

            # Random scaling
            scale = np.random.uniform(0.95, 1.05)
            sequence = sequence * scale

        # Apply transform
        if self.transform:
            sequence = self.transform(sequence)

        # Convert to tensors
        sequence_tensor = torch.FloatTensor(sequence)
        target_tensor = torch.FloatTensor([target]) if isinstance(target, (int, float)) else torch.FloatTensor(target)

        return sequence_tensor, target_tensor


class WalkForwardSplitter:
    """
    Walk-forward cross-validation splitter for time series.
    """

    def __init__(self, n_splits: int = 5, train_size: int = 252,
                val_size: int = 63, test_size: int = 63, gap: int = 0):
        """
        Initialize walk-forward splitter.

        Args:
            n_splits: Number of splits
            train_size: Training window size
            val_size: Validation window size
            test_size: Test window size
            gap: Gap between train and validation
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate train/val/test indices.

        Args:
            X: Feature array
            y: Target array (optional)

        Returns:
            List of (train_idx, val_idx, test_idx) tuples
        """
        n_samples = len(X)
        splits = []

        # Calculate step size
        total_window = self.train_size + self.gap + self.val_size + self.test_size
        step_size = max(1, (n_samples - total_window) // (self.n_splits - 1))

        for i in range(self.n_splits):
            start = i * step_size

            # Train indices
            train_start = start
            train_end = start + self.train_size
            if train_end >= n_samples:
                break

            # Validation indices
            val_start = train_end + self.gap
            val_end = val_start + self.val_size
            if val_end >= n_samples:
                break

            # Test indices
            test_start = val_end
            test_end = min(test_start + self.test_size, n_samples)

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, val_idx, test_idx))

        return splits


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 32, num_workers: int = 4,
                       augment: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        augment: Whether to augment training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def compute_trading_metrics(predictions: np.ndarray, targets: np.ndarray,
                           initial_capital: float = 100000) -> Dict[str, float]:
    """
    Compute comprehensive trading metrics.

    Args:
        predictions: Predicted values
        targets: True values
        initial_capital: Initial capital for backtesting

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic regression metrics
    metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
    metrics['mae'] = np.mean(np.abs(predictions - targets))
    metrics['mape'] = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

    # Directional accuracy
    if len(predictions) > 1:
        pred_direction = np.sign(np.diff(predictions))
        true_direction = np.sign(np.diff(targets))
        metrics['directional_accuracy'] = np.mean(pred_direction == true_direction) * 100
    else:
        metrics['directional_accuracy'] = 0

    # Trading metrics (assuming predictions are returns)
    if len(predictions) > 1:
        # Calculate returns if predictions are prices
        if np.all(predictions > 0) and np.all(targets > 0):
            returns_pred = np.diff(predictions) / predictions[:-1]
            returns_true = np.diff(targets) / targets[:-1]
        else:
            returns_pred = predictions[1:]
            returns_true = targets[1:]

        # Sharpe ratio
        if returns_pred.std() > 0:
            metrics['sharpe_ratio'] = np.sqrt(252) * returns_pred.mean() / returns_pred.std()
        else:
            metrics['sharpe_ratio'] = 0

        # Sortino ratio
        downside_returns = returns_pred[returns_pred < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics['sortino_ratio'] = np.sqrt(252) * returns_pred.mean() / downside_returns.std()
        else:
            metrics['sortino_ratio'] = 0

        # Maximum drawdown
        cumulative_returns = (1 + returns_pred).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = abs(drawdown.min())

        # Calmar ratio
        if metrics['max_drawdown'] > 0:
            annual_return = (1 + returns_pred.mean()) ** 252 - 1
            metrics['calmar_ratio'] = annual_return / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0

        # Win rate
        profitable_trades = returns_pred > 0
        metrics['win_rate'] = np.mean(profitable_trades) * 100

        # Profit factor
        gross_profit = np.sum(returns_pred[returns_pred > 0])
        gross_loss = abs(np.sum(returns_pred[returns_pred < 0]))
        if gross_loss > 0:
            metrics['profit_factor'] = gross_profit / gross_loss
        else:
            metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0

        # Average win/loss
        winning_trades = returns_pred[returns_pred > 0]
        losing_trades = returns_pred[returns_pred < 0]

        metrics['avg_win'] = np.mean(winning_trades) * 100 if len(winning_trades) > 0 else 0
        metrics['avg_loss'] = np.mean(losing_trades) * 100 if len(losing_trades) > 0 else 0

        # Final portfolio value (simple strategy: long when prediction > 0)
        portfolio_value = initial_capital
        position = 0

        for i in range(len(returns_pred)):
            if predictions[i] > 0 and position == 0:
                # Buy
                position = portfolio_value
                portfolio_value = 0
            elif predictions[i] <= 0 and position > 0:
                # Sell
                portfolio_value = position * (1 + returns_true[i])
                position = 0
            elif position > 0:
                # Update position value
                position = position * (1 + returns_true[i])

        # Close final position
        if position > 0:
            portfolio_value = position

        metrics['final_value'] = portfolio_value
        metrics['total_return'] = (portfolio_value - initial_capital) / initial_capital * 100

    return metrics


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # RMSE curves
    axes[0, 1].plot(epochs, history['train_rmse'], label='Train RMSE')
    axes[0, 1].plot(epochs, history['val_rmse'], label='Val RMSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Directional accuracy
    if 'val_dir_acc' in history:
        axes[1, 0].plot(epochs, history['val_dir_acc'], label='Val Dir Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Directional Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Learning rate
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")

    plt.show()


def plot_predictions(predictions: np.ndarray, targets: np.ndarray,
                    dates: Optional[pd.DatetimeIndex] = None,
                    save_path: Optional[str] = None):
    """
    Plot predictions vs targets.

    Args:
        predictions: Predicted values
        targets: True values
        dates: Optional date index
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series plot
    if dates is not None:
        axes[0, 0].plot(dates, targets, label='True', alpha=0.7)
        axes[0, 0].plot(dates, predictions, label='Predicted', alpha=0.7)
        axes[0, 0].set_xlabel('Date')
    else:
        axes[0, 0].plot(targets, label='True', alpha=0.7)
        axes[0, 0].plot(predictions, label='Predicted', alpha=0.7)
        axes[0, 0].set_xlabel('Sample')

    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Predictions vs Targets')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Scatter plot
    axes[0, 1].scatter(targets, predictions, alpha=0.5)
    axes[0, 1].plot([targets.min(), targets.max()],
                   [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predictions')
    axes[0, 1].set_title('Prediction Scatter Plot')
    axes[0, 1].grid(True)

    # Error distribution
    errors = predictions - targets
    axes[1, 0].hist(errors, bins=50, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Error Distribution (Mean: {errors.mean():.4f})')
    axes[1, 0].grid(True)

    # Returns correlation (if applicable)
    if len(predictions) > 1:
        pred_returns = np.diff(predictions) / predictions[:-1]
        true_returns = np.diff(targets) / targets[:-1]

        axes[1, 1].scatter(true_returns, pred_returns, alpha=0.5)
        axes[1, 1].plot([true_returns.min(), true_returns.max()],
                       [true_returns.min(), true_returns.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('True Returns')
        axes[1, 1].set_ylabel('Predicted Returns')
        axes[1, 1].set_title('Returns Correlation')
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved predictions plot to {save_path}")

    plt.show()


def save_results(results: Dict[str, Any], save_dir: str):
    """
    Save training results to disk.

    Args:
        results: Dictionary of results
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = save_dir / 'results.json'
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                json_results[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                json_results[key] = int(value)
            else:
                json_results[key] = value

        json.dump(json_results, f, indent=2)

    logger.info(f"Saved results to {json_path}")

    # Save as CSV for metrics
    if 'metrics' in results:
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_path = save_dir / 'metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0,
                mode: str = 'min', verbose: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Check if should stop.

        Args:
            current_value: Current metric value

        Returns:
            True if should stop
        """
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
            if self.verbose:
                logger.info(f"EarlyStopping: improved to {current_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: {self.counter}/{self.patience} patience")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("EarlyStopping: Stopping training")

        return self.early_stop


def create_ensemble_predictions(models: List[Any], X: np.ndarray,
                               weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Create ensemble predictions from multiple models.

    Args:
        models: List of trained models
        X: Input features
        weights: Optional weights for each model

    Returns:
        Ensemble predictions
    """
    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            if hasattr(model, 'device'):
                X_tensor = X_tensor.to(model.device)

            pred = model(X_tensor).cpu().numpy()
            predictions.append(pred)

    predictions = np.array(predictions)

    if weights is None:
        # Simple average
        ensemble_pred = np.mean(predictions, axis=0)
    else:
        # Weighted average
        weights = np.array(weights).reshape(-1, 1, 1)
        ensemble_pred = np.sum(predictions * weights, axis=0)

    return ensemble_pred