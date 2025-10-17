"""
RunPod Training Script - Optimized for 3x RTX 5090
===================================================
High-performance distributed training for stock forecasting models.

Hardware: 3x RTX 5090 (24GB each = 72GB total VRAM)
Strategy: Minimal checkpoints, maximum performance

Run: python runpod_setup/train_runpod.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# OPTIMAL CONFIGURATION FOR 3x RTX 5090
# ============================================================================

CONFIG = {
    # Hardware
    'gpus': 3,
    'gpu_type': 'RTX 5090',
    'total_vram': '72GB',

    # Model Architecture (LARGE - utilize GPU power)
    'model': {
        'hidden_size': 512,           # Large hidden size
        'num_layers': 6,              # Deep network
        'dropout': 0.2,
        'attention_heads': 8,
        'feed_forward_dim': 2048,
    },

    # Training (Optimized for speed)
    'training': {
        'batch_size': 1024,           # Large batch for 3 GPUs
        'accumulate_grad_batches': 4,  # Effective batch = 4096
        'max_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'gradient_clip_val': 1.0,
    },

    # Data
    'data': {
        'sequence_length': 120,        # 120 days lookback
        'forecast_horizon': 20,        # 20 days ahead
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
    },

    # Checkpointing (MINIMAL)
    'checkpointing': {
        'save_top_k': 1,              # Only save best model
        'monitor': 'val_loss',
        'mode': 'min',
        'save_last': False,           # Don't save last checkpoint
        'every_n_epochs': 10,         # Only check every 10 epochs
    },

    # Optimization
    'optimization': {
        'precision': '16-mixed',       # Mixed precision (faster)
        'num_workers': 16,            # Data loading workers
        'pin_memory': True,
        'persistent_workers': True,
    }
}


# ============================================================================
# DATASET
# ============================================================================

class StockDataset(Dataset):
    """
    Time series dataset for stock forecasting.
    Optimized for GPU training.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 120,
        forecast_horizon: int = 20,
        features: list = None
    ):
        """
        Args:
            data: DataFrame with OHLCV and features
            sequence_length: Lookback window
            forecast_horizon: Days ahead to forecast
            features: List of feature columns to use
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # Default features if not specified
        if features is None:
            features = [
                'returns', 'log_returns', 'volatility_20',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'rsi', 'volume_ratio'
            ]

        self.features = [f for f in features if f in data.columns]

        # Extract feature data
        self.data = data[self.features].values.astype(np.float32)

        # Target: future return
        self.targets = data['returns'].shift(-forecast_horizon).values.astype(np.float32)

        # Valid indices (have enough history and future data)
        self.valid_indices = np.arange(
            sequence_length,
            len(self.data) - forecast_horizon
        )

        logger.info(f"Dataset created: {len(self.valid_indices):,} samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get sequence and target."""
        actual_idx = self.valid_indices[idx]

        # Sequence: [sequence_length, n_features]
        sequence = self.data[actual_idx - self.sequence_length:actual_idx]

        # Target: scalar (future return)
        target = self.targets[actual_idx]

        return {
            'sequence': torch.FloatTensor(sequence),
            'target': torch.FloatTensor([target])
        }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TransformerForecastModel(pl.LightningModule):
    """
    Transformer-based forecasting model.
    Optimized for multi-GPU training.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.n_features = 9  # Will be set dynamically

        # Input projection
        self.input_projection = nn.Linear(
            self.n_features,
            config['model']['hidden_size']
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 120, config['model']['hidden_size'])
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['hidden_size'],
            nhead=config['model']['attention_heads'],
            dim_feedforward=config['model']['feed_forward_dim'],
            dropout=config['model']['dropout'],
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['num_layers']
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config['model']['hidden_size'], 256),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(64, 1)  # Single output (regression)
        )

        # Loss
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Args:
            x: [batch, sequence_length, n_features]

        Returns:
            [batch, 1] predictions
        """
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Transformer
        x = self.transformer(x)

        # Take last time step
        x = x[:, -1, :]

        # Output
        return self.output_head(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        sequences = batch['sequence']
        targets = batch['target']

        predictions = self(sequences)
        loss = self.criterion(predictions, targets)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        sequences = batch['sequence']
        targets = batch['target']

        predictions = self(sequences)
        loss = self.criterion(predictions, targets)

        # Calculate directional accuracy
        pred_direction = (predictions > 0).float()
        true_direction = (targets > 0).float()
        accuracy = (pred_direction == true_direction).float().mean()

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', accuracy, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['max_epochs'],
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """Load training data."""
    logger.info(f"Loading data from {data_path}")

    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info(f"Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")
    return df


def prepare_dataloaders(
    data: pd.DataFrame,
    config: dict
) -> tuple:
    """Prepare train/val/test dataloaders."""

    # Sort by ticker and date
    data = data.sort_values(['ticker', 'Date']).reset_index(drop=True)

    # Split by time (not random!)
    n = len(data)
    train_end = int(n * config['data']['train_split'])
    val_end = int(n * (config['data']['train_split'] + config['data']['val_split']))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    logger.info(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    # Create datasets
    train_dataset = StockDataset(
        train_data,
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon']
    )

    val_dataset = StockDataset(
        val_data,
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon']
    )

    test_dataset = StockDataset(
        test_data,
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['optimization']['num_workers'],
        pin_memory=config['optimization']['pin_memory'],
        persistent_workers=config['optimization']['persistent_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['optimization']['num_workers'],
        pin_memory=config['optimization']['pin_memory'],
        persistent_workers=config['optimization']['persistent_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['optimization']['num_workers'],
        pin_memory=config['optimization']['pin_memory'],
        persistent_workers=config['optimization']['persistent_workers']
    )

    return train_loader, val_loader, test_loader


def train(config: dict, data_path: Path, output_dir: Path):
    """Main training pipeline."""

    print("\n" + "=" * 80)
    print("RUNPOD TRAINING - 3x RTX 5090")
    print("=" * 80)

    # Print configuration
    print("\nHardware Configuration:")
    print(f"  GPUs: {config['gpus']}x {config['gpu_type']}")
    print(f"  Total VRAM: {config['total_vram']}")
    print(f"  Precision: {config['optimization']['precision']}")

    print("\nModel Configuration:")
    for key, value in config['model'].items():
        print(f"  {key}: {value}")

    print("\nTraining Configuration:")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Gradient accumulation: {config['training']['accumulate_grad_batches']}")
    print(f"  Effective batch size: {config['training']['batch_size'] * config['training']['accumulate_grad_batches']}")
    print(f"  Max epochs: {config['training']['max_epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")

    print("\nCheckpoint Strategy:")
    print(f"  Save top K: {config['checkpointing']['save_top_k']} (MINIMAL)")
    print(f"  Save every N epochs: {config['checkpointing']['every_n_epochs']}")
    print(f"  Save last: {config['checkpointing']['save_last']}")

    # Load data
    print("\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)

    data = load_data(data_path)

    # Prepare dataloaders
    print("\n" + "-" * 80)
    print("PREPARING DATALOADERS")
    print("-" * 80)

    train_loader, val_loader, test_loader = prepare_dataloaders(data, config)

    # Initialize model
    print("\n" + "-" * 80)
    print("INITIALIZING MODEL")
    print("-" * 80)

    model = TransformerForecastModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1e6:.1f} MB (fp32)")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor=config['checkpointing']['monitor'],
        mode=config['checkpointing']['mode'],
        save_top_k=config['checkpointing']['save_top_k'],
        save_last=config['checkpointing']['save_last'],
        every_n_epochs=config['checkpointing']['every_n_epochs']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True
    )

    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir / 'logs',
        name='transformer_forecaster'
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu',
        devices=config['gpus'],
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=config['optimization']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    start_time = datetime.now()

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration/3600:.2f} hours")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Test
    print("\n" + "-" * 80)
    print("TESTING ON HELD-OUT DATA")
    print("-" * 80)

    test_results = trainer.test(
        model,
        dataloaders=test_loader,
        ckpt_path='best'
    )

    # Save results
    results = {
        'config': config,
        'training_duration_hours': duration / 3600,
        'best_checkpoint': str(checkpoint_callback.best_model_path),
        'test_results': test_results,
        'total_parameters': total_params,
        'completed_at': datetime.now().isoformat()
    }

    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_path}")


def main():
    """Main entry point."""

    # Paths
    DATA_PATH = Path('/workspace/data/training/training_data.parquet')
    OUTPUT_DIR = Path('/workspace/output')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    train(CONFIG, DATA_PATH, OUTPUT_DIR)


if __name__ == '__main__':
    main()
