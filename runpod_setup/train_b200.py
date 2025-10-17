#!/usr/bin/env python3
"""
Training script optimized for NVIDIA B200 GPU (192GB VRAM)
Designed for massive batch sizes and large models.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION FOR B200 (192GB VRAM)
# ============================================================================

CONFIG = {
    # Hardware
    'gpus': 1,
    'gpu_type': 'B200',
    'total_vram': '192GB',

    # Model Architecture (4x larger than baseline)
    'model': {
        'input_dim': 9,  # OHLCV + 4 features
        'hidden_size': 1024,  # 512 -> 1024 (4x params)
        'num_layers': 12,  # 6 -> 12 (deeper)
        'attention_heads': 16,  # 8 -> 16 (more attention)
        'feed_forward_dim': 4096,  # 2048 -> 4096 (wider)
        'dropout': 0.2,
        'sequence_length': 120,
        'forecast_horizon': 20,
    },

    # Training (B200 optimized)
    'training': {
        'batch_size': 4096,  # Large batch (reduced from 16384 due to model size)
        'accumulate_grad_batches': 4,  # Effective batch = 16384
        'max_epochs': 150,
        'learning_rate': 0.0005,  # Lower for large batch
        'weight_decay': 0.01,
        'warmup_epochs': 5,
        'patience': 25,  # More patience for large model
    },

    # Data
    'data': {
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'num_workers': 32,  # B200 can handle many workers
        'pin_memory': True,
        'persistent_workers': True,
    },

    # Checkpointing (MINIMAL)
    'checkpointing': {
        'save_top_k': 1,
        'save_last': False,
        'every_n_epochs': 10,
        'monitor': 'val_loss',
        'mode': 'min',
    },

    # Optimization
    'optimization': {
        'precision': 'bf16-mixed',  # BF16 better on B200
        'compile': False,  # Disabled - causes OOM with large model
        'gradient_clip_val': 1.0,
        'detect_anomaly': False,
    },
}


# ============================================================================
# DATASET
# ============================================================================

class StockDataset(Dataset):
    """Time series dataset for stock price prediction."""

    def __init__(self, data: pd.DataFrame, seq_len: int = 120, forecast_horizon: int = 20):
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        # Feature columns
        self.feature_cols = ['returns', 'sma_5', 'sma_20', 'volatility_20',
                            'volume_ratio', 'rsi', 'log_returns', 'volatility_5', 'sma_10']

        # Prepare sequences
        self.sequences = []
        self.targets = []

        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].sort_values('Date')

            if len(ticker_data) < seq_len + forecast_horizon:
                continue

            features = ticker_data[self.feature_cols].values
            prices = ticker_data['Close'].values

            for i in range(len(ticker_data) - seq_len - forecast_horizon + 1):
                seq = features[i:i + seq_len]

                # Target: future return
                current_price = prices[i + seq_len - 1]
                future_price = prices[i + seq_len + forecast_horizon - 1]
                target_return = (future_price - current_price) / current_price

                self.sequences.append(seq)
                self.targets.append(target_return)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        logger.info(f"Dataset created: {len(self.sequences)} samples")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.targets[idx])


# ============================================================================
# MODEL
# ============================================================================

class TransformerForecastModel(pl.LightningModule):
    """Large Transformer model for B200 GPU."""

    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        model_cfg = config['model']
        train_cfg = config['training']

        # Input projection
        self.input_projection = nn.Linear(model_cfg['input_dim'], model_cfg['hidden_size'])

        # Positional encoding
        self.register_buffer('positional_encoding',
                            self._create_positional_encoding(model_cfg['sequence_length'],
                                                             model_cfg['hidden_size']))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg['hidden_size'],
            nhead=model_cfg['attention_heads'],
            dim_feedforward=model_cfg['feed_forward_dim'],
            dropout=model_cfg['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_cfg['num_layers'],
            norm=nn.LayerNorm(model_cfg['hidden_size'])
        )

        # Output head (3-layer MLP)
        self.output_head = nn.Sequential(
            nn.Linear(model_cfg['hidden_size'], model_cfg['hidden_size'] // 2),
            nn.GELU(),
            nn.Dropout(model_cfg['dropout']),
            nn.Linear(model_cfg['hidden_size'] // 2, model_cfg['hidden_size'] // 4),
            nn.GELU(),
            nn.Dropout(model_cfg['dropout']),
            nn.Linear(model_cfg['hidden_size'] // 4, 1)
        )

        self.criterion = nn.MSELoss()
        self.learning_rate = train_cfg['learning_rate']
        self.warmup_epochs = train_cfg['warmup_epochs']

    def _create_positional_encoding(self, seq_len: int, hidden_size: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-np.log(10000.0) / hidden_size))

        pe = torch.zeros(1, seq_len, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer
        x = self.transformer(x)

        # Use last token for prediction
        x = x[:, -1, :]

        # Output
        x = self.output_head(x)

        return x.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Accuracy (directional)
        accuracy = ((y_hat > 0) == (y > 0)).float().mean()

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', accuracy, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.trainer.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }


# ============================================================================
# TRAINING
# ============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """Load training data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Drop NaN
    df = df.dropna()

    logger.info(f"Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")
    return df


def create_dataloaders(df: pd.DataFrame, config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    data_cfg = config['data']
    model_cfg = config['model']

    # Split data
    train_size = int(len(df) * data_cfg['train_split'])
    val_size = int(len(df) * data_cfg['val_split'])

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Create datasets
    train_dataset = StockDataset(train_df, model_cfg['sequence_length'], model_cfg['forecast_horizon'])
    val_dataset = StockDataset(val_df, model_cfg['sequence_length'], model_cfg['forecast_horizon'])
    test_dataset = StockDataset(test_df, model_cfg['sequence_length'], model_cfg['forecast_horizon'])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        persistent_workers=data_cfg['persistent_workers'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        persistent_workers=data_cfg['persistent_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        persistent_workers=data_cfg['persistent_workers']
    )

    return train_loader, val_loader, test_loader


def train(config: Dict, data_path: Path, output_dir: Path):
    """Main training function."""

    # Print configuration
    print("\n" + "="*80)
    print("B200 TRAINING CONFIGURATION")
    print("="*80)
    print(f"GPU: {config['gpu_type']} ({config['total_vram']} VRAM)")
    print(f"Model: {config['model']['hidden_size']}H × {config['model']['num_layers']}L × {config['model']['attention_heads']}A")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Effective batch: {config['training']['batch_size']}")
    print(f"Max epochs: {config['training']['max_epochs']}")
    print(f"Precision: {config['optimization']['precision']}")
    print("="*80 + "\n")

    # Load data
    print("-" * 80)
    print("LOADING DATA")
    print("-" * 80)
    df = load_data(data_path)

    # Create dataloaders
    print("\n" + "-" * 80)
    print("PREPARING DATALOADERS")
    print("-" * 80)
    train_loader, val_loader, test_loader = create_dataloaders(df, config)

    # Initialize model
    print("\n" + "-" * 80)
    print("INITIALIZING MODEL")
    print("-" * 80)
    model = TransformerForecastModel(config)

    # Compile model for speedup (30% faster)
    if config['optimization']['compile']:
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor=config['checkpointing']['monitor'],
        mode=config['checkpointing']['mode'],
        save_top_k=config['checkpointing']['save_top_k'],
        save_last=config['checkpointing']['save_last'],
        every_n_epochs=config['checkpointing']['every_n_epochs'],
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['patience'],
        mode='min',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Logger
    logger_tb = TensorBoardLogger(
        save_dir=output_dir / 'logs',
        name='transformer_b200',
        version=0
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu',
        devices=config['gpus'],
        precision=config['optimization']['precision'],
        gradient_clip_val=config['optimization']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger_tb,
        enable_progress_bar=True,
        log_every_n_steps=10,
        detect_anomaly=config['optimization']['detect_anomaly'],
        benchmark=True,  # cudnn benchmark for speed
    )

    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    start_time = datetime.now()
    trainer.fit(model, train_loader, val_loader)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds() / 3600

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Duration: {duration:.2f} hours")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Save results
    results = {
        'config': config,
        'training_duration_hours': duration,
        'best_checkpoint': str(checkpoint_callback.best_model_path),
        'best_val_loss': float(checkpoint_callback.best_model_score),
        'total_parameters': total_params,
        'completed_at': datetime.now().isoformat(),
    }

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_dir / 'training_results.json'}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    DATA_PATH = Path('/workspace/data/training/training_data.parquet')
    OUTPUT_DIR = Path('/workspace/output')

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'checkpoints').mkdir(exist_ok=True)
    (OUTPUT_DIR / 'logs').mkdir(exist_ok=True)

    # Train
    train(CONFIG, DATA_PATH, OUTPUT_DIR)


if __name__ == '__main__':
    main()
