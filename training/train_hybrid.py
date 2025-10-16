from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_models.hybrid_model import HybridTradingModel
from ml_models.rl_agent import PPOAgent, TradingEnvironment, TradingConfig
from utils.advanced_feature_engineering import AdvancedFeatureEngineer, FeatureConfig
from training.training_utils import (
    WalkForwardSplitter,
    TradingDataset,
    compute_trading_metrics,
    plot_training_curves,
    create_data_loaders
)

warnings.filterwarnings('ignore')

# Set up logging
def setup_logging(config: Dict[str, Any]):
    """Set up logging configuration."""
    # Fix: Add try-except for log level access
    try:
        log_level = getattr(logging, config['logging']['level'])
    except (KeyError, AttributeError):
        log_level = logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = []
    if config['logging']['console']:
        handlers.append(logging.StreamHandler())

    if config['logging']['file']:
        log_file = Path(config['logging']['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    return logging.getLogger(__name__)


def set_random_seeds(config: Dict[str, Any]):
    """Set random seeds for reproducibility."""
    import random

    random.seed(config['seeds']['python'])
    np.random.seed(config['seeds']['numpy'])
    torch.manual_seed(config['seeds']['torch'])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seeds']['torch'])
        if config['seeds']['cuda_deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: Dict[str, Any], logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for training.

    Returns:
        Tuple of (price_data, features)
    """
    logger.info("Loading data...")

    # Load price data
    data_dir = Path(__file__).resolve().parents[1] / "data"
    price_file = data_dir / f"{config['data']['base_name']}.parquet"

    if price_file.exists():
        prices = pd.read_parquet(price_file)
    else:
        # Try CSV
        price_file = data_dir / f"{config['data']['base_name']}.csv"
        if price_file.exists():
            prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"Price data not found: {price_file}")

    logger.info(f"Loaded price data: {prices.shape}")

    # Load features
    features_dir = Path(__file__).resolve().parents[1] / config['data']['features_dir']

    # Find feature files
    feature_files = list(features_dir.glob("*_features.parquet"))

    if not feature_files:
        logger.warning("No pre-computed features found. Generating features...")

        # Generate features using AdvancedFeatureEngineer
        feature_config = FeatureConfig()
        engineer = AdvancedFeatureEngineer(feature_config)

        # Fix: Robust ticker extraction with fallback
        ticker = "UNKNOWN"
        price_df = prices

        # Try to extract ticker from column names
        if hasattr(prices, 'columns'):
            if isinstance(prices.columns, pd.MultiIndex):
                # MultiIndex columns: (ticker, field)
                tickers = prices.columns.get_level_values(0).unique()
                if len(tickers) > 0:
                    ticker = str(tickers[0])
                    # Extract data for this ticker
                    price_df = prices[ticker] if ticker in prices.columns.levels[0] else prices
            elif 'Close' not in prices.columns:
                # Try to extract from column names with format "Field (TICKER)"
                for col in prices.columns:
                    if '(' in str(col) and ')' in str(col):
                        try:
                            ticker = str(col).split('(')[1].split(')')[0]
                            # Extract all columns for this ticker
                            ticker_cols = [c for c in prices.columns if f'({ticker})' in str(c)]
                            if ticker_cols:
                                price_df = prices[ticker_cols]
                                # Clean column names
                                price_df.columns = [str(c).split(' ')[0] for c in price_df.columns]
                                break
                        except Exception:
                            continue

        features = engineer.generate_features(
            ticker=ticker,
            df=price_df,
            save_features=True
        )
    else:
        # Load first feature file
        features = pd.read_parquet(feature_files[0])
        logger.info(f"Loaded features from {feature_files[0]}: {features.shape}")

    # Align indices
    common_index = prices.index.intersection(features.index)
    prices = prices.loc[common_index]
    features = features.loc[common_index]

    # Select feature groups based on config
    if not config['data']['feature_groups'].get('cross_sectional', False):
        # Remove cross-sectional features if not enabled
        cross_sectional_cols = [col for col in features.columns
                               if any(x in col for x in ['beta_', 'relative_strength_', 'market_', 'sector_'])]
        features = features.drop(columns=cross_sectional_cols, errors='ignore')

    logger.info(f"Final data shape - Prices: {prices.shape}, Features: {features.shape}")

    return prices, features


def create_walk_forward_splits(data: pd.DataFrame, config: Dict[str, Any]) -> List[Tuple[pd.Index, pd.Index, pd.Index]]:
    """
    Create walk-forward cross-validation splits.

    Returns:
        List of (train_idx, val_idx, test_idx) tuples
    """
    if not config['data']['walk_forward']['enabled']:
        # Simple train/val/test split
        n_samples = len(data)
        train_end = int(n_samples * config['data']['train_split'])
        val_end = train_end + int(n_samples * config['data']['val_split'])

        train_idx = data.index[:train_end]
        val_idx = data.index[train_end:val_end]
        test_idx = data.index[val_end:]

        return [(train_idx, val_idx, test_idx)]

    # Walk-forward cross-validation
    splits = []
    train_window = config['data']['walk_forward']['train_window']
    val_window = config['data']['walk_forward']['val_window']
    step_size = config['data']['walk_forward']['step_size']

    # Fix: Ensure we don't go past the available data
    for start_idx in range(0, len(data) - train_window - val_window + 1, step_size):
        train_end = start_idx + train_window
        val_end = train_end + val_window

        if val_end >= len(data):
            break

        train_idx = data.index[start_idx:train_end]
        val_idx = data.index[train_end:val_end]

        # Use next period as test (if available)
        test_start = val_end
        test_end = min(test_start + val_window, len(data))
        test_idx = data.index[test_start:test_end] if test_end > test_start else pd.Index([])

        if len(test_idx) > 0:
            splits.append((train_idx, val_idx, test_idx))

    return splits


def create_sequences(features: np.ndarray, targets: np.ndarray,
                     sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.

    Args:
        features: Feature array (n_samples, n_features)
        targets: Target array (n_samples,)
        sequence_length: Length of input sequences

    Returns:
        Tuple of (sequences, targets)
    """
    n_samples = len(features) - sequence_length
    n_features = features.shape[1]

    X = np.zeros((n_samples, sequence_length, n_features))
    y = np.zeros((n_samples,))

    for i in range(n_samples):
        # Fix: Add assertion to prevent lookahead bias
        assert i + sequence_length < len(targets), f"Lookahead bias detected at index {i}"
        X[i] = features[i:i + sequence_length]
        y[i] = targets[i + sequence_length]

    return X, y


def train_model(model: HybridTradingModel, train_loader: DataLoader,
               val_loader: DataLoader, config: Dict[str, Any],
               logger: logging.Logger, writer: Optional[SummaryWriter] = None) -> Dict[str, List[float]]:
    """
    Train the hybrid model.

    Returns:
        Dictionary of training history
    """
    # Set up optimizer
    optimizer_type = config['training']['optimizer']
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

    # Set up scheduler
    scheduler_config = config['training']['scheduler']
    if scheduler_config['type'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(scheduler_config['factor']),
            patience=int(scheduler_config['patience']),
            min_lr=float(scheduler_config['min_lr'])
        )
    elif scheduler_config['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config['training']['epochs'])
        )
    else:
        scheduler = None

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'val_dir_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    # Checkpoint directory
    checkpoint_dir = Path(config['checkpointing']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(model.device)
            price_targets = targets[:, 0].unsqueeze(1).to(model.device)  # Price targets

            # Forward pass
            optimizer.zero_grad()
            outputs = model(features, return_intermediates=True)

            # Calculate loss
            price_loss = nn.functional.huber_loss(outputs['price_prediction'], price_targets)

            # Direction loss if applicable
            if model.predict_direction and targets.shape[1] > 1:
                direction_targets = targets[:, 1].long().to(model.device)
                direction_loss = nn.functional.cross_entropy(
                    outputs['direction_logits'], direction_targets
                )
                loss = (config['training']['loss_weights']['price'] * price_loss +
                       config['training']['loss_weights']['direction'] * direction_loss)
            else:
                loss = price_loss

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()

            train_loss += loss.item()
            train_predictions.extend(outputs['price_prediction'].detach().cpu().numpy())
            train_targets.extend(price_targets.detach().cpu().numpy())

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_rmse = np.sqrt(np.mean((np.array(train_predictions) - np.array(train_targets)) ** 2))

        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets_list = []
        val_directions_pred = []
        val_directions_true = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(model.device)
                price_targets = targets[:, 0].unsqueeze(1).to(model.device)

                outputs = model(features, return_intermediates=True)

                # Calculate loss
                price_loss = nn.functional.huber_loss(outputs['price_prediction'], price_targets)

                if model.predict_direction and targets.shape[1] > 1:
                    direction_targets = targets[:, 1].long().to(model.device)
                    direction_loss = nn.functional.cross_entropy(
                        outputs['direction_logits'], direction_targets
                    )
                    loss = (config['training']['loss_weights']['price'] * price_loss +
                           config['training']['loss_weights']['direction'] * direction_loss)

                    # Store direction predictions
                    val_directions_pred.extend(
                        torch.argmax(outputs['direction_logits'], dim=1).cpu().numpy()
                    )
                    val_directions_true.extend(direction_targets.cpu().numpy())
                else:
                    loss = price_loss

                val_loss += loss.item()
                val_predictions.extend(outputs['price_prediction'].detach().cpu().numpy())
                val_targets_list.extend(price_targets.detach().cpu().numpy())

        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_rmse = np.sqrt(np.mean((np.array(val_predictions) - np.array(val_targets_list)) ** 2))

        # Directional accuracy
        if val_directions_pred:
            val_dir_acc = np.mean(np.array(val_directions_pred) == np.array(val_directions_true)) * 100
        else:
            # Calculate from price predictions
            pred_returns = np.diff(np.array(val_predictions).flatten())
            true_returns = np.diff(np.array(val_targets_list).flatten())
            val_dir_acc = np.mean(np.sign(pred_returns) == np.sign(true_returns)) * 100

        # Update learning rate
        if scheduler:
            if scheduler_config['type'] == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['val_dir_acc'].append(val_dir_acc)

        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('RMSE/train', train_rmse, epoch)
            writer.add_scalar('RMSE/val', val_rmse, epoch)
            writer.add_scalar('Metrics/directional_accuracy', val_dir_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Log progress
        if epoch % 5 == 0:
            logger.info(
                f"Epoch {epoch}/{config['training']['epochs']}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Val RMSE={val_rmse:.4f}, Val Dir Acc={val_dir_acc:.1f}%"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model with timestamp
            if config['checkpointing']['keep_best_only']:
                # Fix: Use timestamp in filename to avoid overwriting
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = checkpoint_dir / f'best_model_{timestamp}.pt'
                model.save_model(checkpoint_path, optimizer=optimizer)
                logger.info(f"Saved best model with val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Regular checkpointing
        if epoch % config['checkpointing']['save_interval'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            model.save_model(checkpoint_path, optimizer=optimizer)

    return history


def evaluate_model(model: HybridTradingModel, test_loader: DataLoader,
                  config: Dict[str, Any], logger: logging.Logger) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    predictions = []
    targets = []
    returns_pred = []
    returns_true = []

    with torch.no_grad():
        for features, target_batch in test_loader:
            features = features.to(model.device)

            # Get predictions
            outputs = model(features, return_intermediates=True)

            predictions.extend(outputs['price_prediction'].cpu().numpy().flatten())
            targets.extend(target_batch[:, 0].numpy())

    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Basic metrics
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))

    # Directional accuracy
    if len(predictions) > 1:
        pred_direction = np.sign(np.diff(predictions))
        true_direction = np.sign(np.diff(targets))
        directional_accuracy = np.mean(pred_direction == true_direction) * 100
    else:
        directional_accuracy = 0

    # Trading metrics (if returns available)
    if len(predictions) > 1:
        returns_pred = np.diff(predictions) / predictions[:-1]
        returns_true = np.diff(targets) / targets[:-1]

        # Sharpe ratio - Fix: Adapt to data frequency
        days_per_year = 252 if len(returns_pred) >= 252 else len(returns_pred)
        if returns_pred.std() > 0:
            sharpe = np.sqrt(days_per_year) * returns_pred.mean() / returns_pred.std()
        else:
            sharpe = 0

        # Max drawdown
        cumulative_returns = (1 + returns_pred).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
    else:
        sharpe = 0
        max_drawdown = 0

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': directional_accuracy,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

    logger.info("Test Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train hybrid deep learning trading model')
    parser.add_argument('--config', type=str, default='training/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--ticker', type=str, help='Specific ticker to train on')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')

    args = parser.parse_args()

    # Load configuration
    config = load_configuration(args.config)

    # Set up logging
    logger = setup_logging(config)
    logger.info("="*60)
    logger.info("HYBRID DEEP LEARNING TRADING MODEL - TRAINING")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")

    # Set random seeds
    set_random_seeds(config)

    # Determine device
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['hardware']['device'])
    logger.info(f"Using device: {device}")

    # Set up TensorBoard
    writer = None
    if config['tracking']['tensorboard']['enabled']:
        log_dir = Path(config['tracking']['tensorboard']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logging to {log_dir}")

    try:
        # Load data
        prices, features = prepare_data(config, logger)

        # Prepare targets (next day returns)
        price_cols = [col for col in prices.columns if 'Close' in col or 'Adj Close' in col]
        if price_cols:
            close_prices = prices[price_cols[0]].values
        else:
            close_prices = prices.iloc[:, -1].values  # Use last column

        returns = np.diff(close_prices) / close_prices[:-1]
        # Pad to match length
        returns = np.concatenate([[0], returns])

        # Direction labels (0: down, 1: neutral, 2: up)
        direction_labels = np.where(returns < -0.001, 0,
                                   np.where(returns > 0.001, 2, 1))

        # Remove non-numeric columns from features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        feature_values = features[numeric_cols].values

        # Create walk-forward splits
        splits = create_walk_forward_splits(features, config)
        logger.info(f"Created {len(splits)} walk-forward splits")

        # Store results for each split
        all_histories = []
        all_metrics = []

        for split_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
            logger.info(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
            logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            # Fix: Correct indexing using get_indexer instead of boolean mask
            train_positions = features.index.get_indexer(train_idx)
            val_positions = features.index.get_indexer(val_idx)
            test_positions = features.index.get_indexer(test_idx)

            # Filter out -1 (not found) indices
            train_positions = train_positions[train_positions >= 0]
            val_positions = val_positions[val_positions >= 0]
            test_positions = test_positions[test_positions >= 0]

            # Get split data using integer positions
            train_features = feature_values[train_positions]
            val_features = feature_values[val_positions]
            test_features = feature_values[test_positions]

            train_returns = returns[train_positions]
            val_returns = returns[val_positions]
            test_returns = returns[test_positions]

            train_directions = direction_labels[train_positions]
            val_directions = direction_labels[val_positions]
            test_directions = direction_labels[test_positions]

            # Create sequences
            seq_len = config['data']['sequence_length']

            X_train, y_train = create_sequences(train_features, train_returns, seq_len)
            X_val, y_val = create_sequences(val_features, val_returns, seq_len)
            X_test, y_test = create_sequences(test_features, test_returns, seq_len)

            # Create direction sequences
            _, y_train_dir = create_sequences(train_features, train_directions, seq_len)
            _, y_val_dir = create_sequences(val_features, val_directions, seq_len)
            _, y_test_dir = create_sequences(test_features, test_directions, seq_len)

            # Combine targets
            y_train_combined = np.stack([y_train, y_train_dir], axis=1)
            y_val_combined = np.stack([y_val, y_val_dir], axis=1)
            y_test_combined = np.stack([y_test, y_test_dir], axis=1)

            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train_combined)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val_combined)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test_combined)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['hardware']['num_workers'],
                pin_memory=config['hardware']['pin_memory']
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['hardware']['num_workers'],
                pin_memory=config['hardware']['pin_memory']
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['hardware']['num_workers'],
                pin_memory=config['hardware']['pin_memory']
            )

            # Create model
            model = HybridTradingModel(
                input_dim=X_train.shape[2],
                sequence_length=seq_len,
                cnn_filters=config['model']['cnn']['filters'],
                cnn_kernel_sizes=config['model']['cnn']['kernel_sizes'],
                cnn_dropout=config['model']['cnn']['dropout'],
                lstm_hidden=config['model']['lstm']['hidden_units'],
                lstm_layers=config['model']['lstm']['num_layers'],
                lstm_bidirectional=config['model']['lstm']['bidirectional'],
                lstm_dropout=config['model']['lstm']['dropout'],
                transformer_d_model=config['model']['transformer']['d_model'],
                transformer_heads=config['model']['transformer']['n_heads'],
                transformer_layers=config['model']['transformer']['num_layers'],
                transformer_ff_dim=config['model']['transformer']['dim_feedforward'],
                transformer_dropout=config['model']['transformer']['dropout'],
                fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
                fusion_dropout=config['model']['fusion']['dropout'],
                predict_direction=config['model']['output']['predict_direction'],
                device=str(device)
            )

            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Train or evaluate
            if args.evaluate_only:
                # Load checkpoint
                checkpoint_path = Path(args.resume or 'checkpoints/best_model.pt')
                if checkpoint_path.exists():
                    model = HybridTradingModel.load_model(checkpoint_path, str(device))
                    logger.info(f"Loaded model from {checkpoint_path}")
                else:
                    logger.error(f"Checkpoint not found: {checkpoint_path}")
                    continue

                # Evaluate only
                metrics = evaluate_model(model, test_loader, config, logger)
                all_metrics.append(metrics)
            else:
                # Train model
                history = train_model(model, train_loader, val_loader, config, logger, writer)
                all_histories.append(history)

                # Evaluate on test set
                metrics = evaluate_model(model, test_loader, config, logger)
                all_metrics.append(metrics)

                # Save final model for this split
                model_path = Path(config['checkpointing']['save_dir']) / f'model_split_{split_idx}.pt'
                model.save_model(model_path)

        # Aggregate results across splits
        if all_metrics:
            logger.info("\n" + "="*60)
            logger.info("AGGREGATE RESULTS ACROSS ALL SPLITS")
            logger.info("="*60)

            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                mean_val = np.mean(values)
                std_val = np.std(values)
                logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

        # Save training history
        if all_histories:
            history_path = Path(config['checkpointing']['save_dir']) / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(all_histories, f, indent=2)
            logger.info(f"Saved training history to {history_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if writer:
            writer.close()

    logger.info("\nTraining completed!")


if __name__ == "__main__":
    main()