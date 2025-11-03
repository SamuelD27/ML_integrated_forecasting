#!/usr/bin/env python3
"""
Training script optimized for 8x NVIDIA B200 GPUs (1.4TB total VRAM)
Maximizes multi-GPU efficiency with optimized DDP, larger batches, and minimal overhead.

Expected speedup: 5.5-6x vs single GPU
Cost: $45.52/hr (8x $5.69/hr)
Training time: ~3.5 hours (vs 20 hours on 1 GPU)
Total cost: ~$159
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
from pytorch_lightning.strategies import DDPStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION FOR 8x B200 (1.4TB TOTAL VRAM)
# ============================================================================

CONFIG = {
    # Hardware
    'num_gpus': 8,
    'gpu_type': 'B200',
    'vram_per_gpu': '178GB',
    'total_vram': '1.4TB',

    # Model Architecture (Optimized for multi-GPU)
    'model': {
        'input_dim': 9,  # OHLCV + 4 features
        'hidden_size': 1024,  # Large model
        'num_layers': 12,  # Deep network
        'attention_heads': 16,  # Multi-head attention
        'feed_forward_dim': 4096,  # Wide feed-forward
        'dropout': 0.2,
        'sequence_length': 120,
        'forecast_horizon': 20,
    },

    # Training (8-GPU optimized)
    'training': {
        'batch_size_per_gpu': 2048,  # 2048 × 8 = 16,384 effective batch
        'accumulate_grad_batches': 1,  # No accumulation needed with 8 GPUs
        'max_epochs': 150,
        'learning_rate': 0.001,  # Higher LR for large batch (16K)
        'weight_decay': 0.01,
        'warmup_epochs': 10,  # Longer warmup for large batch
        'patience': 25,
    },

    # Data (Optimized for multi-GPU)
    'data': {
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'num_workers_per_gpu': 8,  # 8 workers × 8 GPUs = 64 total
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,  # Prefetch 4 batches per worker
    },

    # Checkpointing (Minimal to reduce I/O)
    'checkpointing': {
        'save_top_k': 1,
        'save_last': False,
        'every_n_epochs': 15,  # Less frequent (I/O bottleneck with 8 GPUs)
        'monitor': 'val_loss',
        'mode': 'min',
    },

    # Optimization (8-GPU specific)
    'optimization': {
        'precision': 'bf16-mixed',  # BF16 for B200
        'compile': False,  # Disable torch.compile (compatibility issues)
        'gradient_clip_val': 1.0,
        'detect_anomaly': False,
        'sync_batchnorm': True,  # Sync batch norm across GPUs
        'find_unused_parameters': False,  # Speed optimization
    },

    # DDP Strategy
    'ddp': {
        'backend': 'nccl',  # Fastest for NVIDIA GPUs
        'gradient_as_bucket_view': True,  # Memory optimization
        'static_graph': True,  # Speed optimization (no dynamic graph)
        'ddp_comm_hook': 'fp16_compress',  # Compress gradients to FP16 for sync
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

        logger.info(f"Dataset created: {len(self.sequences):,} samples")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.targets[idx])


# ============================================================================
# MODEL
# ============================================================================

class TransformerForecastModel(pl.LightningModule):
    """Large Transformer model optimized for 8x B200 GPUs."""

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

        # Cosine annealing with warmup (better for large batches)
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=self.warmup_epochs
        )

        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_path: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and split data into train/val/test sets."""
    logger.info(f"Loading data from {data_path}")

    # Load parquet
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")

    # Create dataset
    dataset = StockDataset(df, seq_len=CONFIG['model']['sequence_length'],
                          forecast_horizon=CONFIG['model']['forecast_horizon'])

    # Split
    train_size = int(CONFIG['data']['train_split'] * len(dataset))
    val_size = int(CONFIG['data']['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    num_workers = CONFIG['data']['num_workers_per_gpu']

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size_per_gpu'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=CONFIG['data']['pin_memory'],
        persistent_workers=CONFIG['data']['persistent_workers'],
        prefetch_factor=CONFIG['data']['prefetch_factor'],
        drop_last=True,  # Important for DDP
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size_per_gpu'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=CONFIG['data']['pin_memory'],
        persistent_workers=CONFIG['data']['persistent_workers'],
        prefetch_factor=CONFIG['data']['prefetch_factor'],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['training']['batch_size_per_gpu'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=CONFIG['data']['pin_memory'],
    )

    logger.info(f"Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
    logger.info(f"Test: {len(test_dataset):,} samples, {len(test_loader)} batches")
    logger.info(f"Effective batch size: {CONFIG['training']['batch_size_per_gpu']} × {CONFIG['num_gpus']} = "
                f"{CONFIG['training']['batch_size_per_gpu'] * CONFIG['num_gpus']:,}")

    return train_loader, val_loader, test_loader


# ============================================================================
# TRAINING
# ============================================================================

def train(config: Dict, data_path: Path, output_dir: Path):
    """Train the model with 8-GPU DDP."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("8x B200 TRAINING - MULTI-GPU OPTIMIZED")
    logger.info("=" * 80)
    logger.info(f"GPUs: {config['num_gpus']}x {config['gpu_type']}")
    logger.info(f"Total VRAM: {config['total_vram']}")
    logger.info(f"Batch size per GPU: {config['training']['batch_size_per_gpu']:,}")
    logger.info(f"Effective batch size: {config['training']['batch_size_per_gpu'] * config['num_gpus']:,}")
    logger.info(f"Model: {config['model']['num_layers']}L × {config['model']['hidden_size']}H")
    logger.info("=" * 80)

    # Load data
    train_loader, val_loader, test_loader = load_data(data_path)

    # Create model
    model = TransformerForecastModel(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor=config['checkpointing']['monitor'],
        mode=config['checkpointing']['mode'],
        save_top_k=config['checkpointing']['save_top_k'],
        save_last=config['checkpointing']['save_last'],
        every_n_epochs=config['checkpointing']['every_n_epochs'],
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['patience'],
        mode='min',
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir / 'logs',
        name='8gpu_training'
    )

    # DDP Strategy (OPTIMIZED)
    ddp_strategy = DDPStrategy(
        process_group_backend=config['ddp']['backend'],
        gradient_as_bucket_view=config['ddp']['gradient_as_bucket_view'],
        static_graph=config['ddp']['static_graph'],
        find_unused_parameters=config['optimization']['find_unused_parameters'],
    )

    # Trainer
    trainer = pl.Trainer(
        # Hardware
        accelerator='gpu',
        devices=config['num_gpus'],
        strategy=ddp_strategy,

        # Training
        max_epochs=config['training']['max_epochs'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        gradient_clip_val=config['optimization']['gradient_clip_val'],
        precision=config['optimization']['precision'],

        # Callbacks & Logging
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=tb_logger,

        # Performance
        enable_progress_bar=True,
        enable_model_summary=True,
        sync_batchnorm=config['optimization']['sync_batchnorm'],
        detect_anomaly=config['optimization']['detect_anomaly'],

        # Determinism
        deterministic=False,  # Faster but non-deterministic
    )

    # Print model summary
    logger.info("\n" + "=" * 80)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 80)

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    start_time = datetime.now()
    trainer.fit(model, train_loader, val_loader)
    end_time = datetime.now()

    training_time = (end_time - start_time).total_seconds() / 3600  # hours

    # Test
    logger.info("\n" + "=" * 80)
    logger.info("TESTING")
    logger.info("=" * 80)
    test_results = trainer.test(model, test_loader)

    # Save results
    results = {
        'config': config,
        'training_time_hours': training_time,
        'test_results': test_results,
        'best_checkpoint': str(checkpoint_callback.best_model_path),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
    }

    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Training time: {training_time:.2f} hours")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Paths
    DATA_PATH = Path('/workspace/data/training/training_data_compressed.parquet')
    OUTPUT_DIR = Path('/workspace/output')

    # Verify CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    num_gpus = torch.cuda.device_count()
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info(f"GPU type: {torch.cuda.get_device_name(0)}")

    if num_gpus != CONFIG['num_gpus']:
        logger.warning(f"Expected {CONFIG['num_gpus']} GPUs but found {num_gpus}")
        CONFIG['num_gpus'] = num_gpus

    # Set environment variables for optimal performance
    os.environ['NCCL_DEBUG'] = 'WARN'  # Reduce NCCL logging
    os.environ['NCCL_IB_DISABLE'] = '0'  # Enable InfiniBand if available
    os.environ['NCCL_P2P_DISABLE'] = '0'  # Enable P2P transfers
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Better memory management

    # Train
    train(CONFIG, DATA_PATH, OUTPUT_DIR)


if __name__ == '__main__':
    main()
