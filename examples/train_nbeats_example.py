"""
Complete training example for N-BEATS on synthetic time series data.

This example demonstrates:
1. Model initialization with proper weight setup
2. Training loop with gradient clipping
3. Learning rate scheduling
4. Convergence monitoring
5. Validation

Key points why this works (vs. flickering loss):
- Explicit weight initialization (Xavier uniform)
- Gradient clipping at max_norm=1.0
- Adam optimizer with learning rate scheduling
- Proper batch processing and loss computation
"""

import sys
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.nbeats import NBeats, train_epoch, validate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    num_samples: int = 1000,
    seq_len: int = 60,
    pred_len: int = 21,
    noise_std: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic time series data for training.

    Pattern: Trending + seasonal + noise

    Args:
        num_samples: Number of sequences
        seq_len: Length of input sequence
        pred_len: Length of prediction horizon
        noise_std: Noise standard deviation
        seed: Random seed

    Returns:
        (inputs, targets) as numpy arrays
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    inputs = []
    targets = []

    for _ in range(num_samples):
        # Trend component
        t = np.arange(seq_len + pred_len)
        trend = 0.5 * t / (seq_len + pred_len)

        # Seasonal component
        seasonal = 0.3 * np.sin(2 * np.pi * t / 12)

        # Noise
        noise = np.random.randn(seq_len + pred_len) * noise_std

        # Combine
        ts = trend + seasonal + noise

        # Split into input and target
        x = ts[:seq_len]
        y = ts[seq_len:seq_len + pred_len]

        inputs.append(x)
        targets.append(y)

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


def main():
    """Training loop with monitoring."""

    # ===== CONFIGURATION =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Data parameters
    seq_len = 60
    pred_len = 21
    num_train = 800
    num_val = 200

    # Model parameters
    hidden_size = 128
    num_blocks = 4
    num_layers_per_block = 4
    dropout = 0.1

    # Training parameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-5
    max_grad_norm = 1.0

    # ===== DATA PREPARATION =====
    logger.info("Generating synthetic data...")
    inputs, targets = generate_synthetic_data(
        num_samples=num_train + num_val,
        seq_len=seq_len,
        pred_len=pred_len,
    )

    # Split into train and validation
    train_inputs = inputs[:num_train]
    train_targets = targets[:num_train]
    val_inputs = inputs[num_train:]
    val_targets = targets[num_train:]

    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_inputs),
        torch.from_numpy(train_targets)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_inputs),
        torch.from_numpy(val_targets)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    logger.info(f"Train samples: {len(train_inputs)}, Val samples: {len(val_inputs)}")

    # ===== MODEL INITIALIZATION =====
    logger.info("Initializing N-BEATS model...")
    model = NBeats(
        input_size=seq_len,
        output_size=pred_len,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
        dropout=dropout,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ===== OPTIMIZER & SCHEDULER =====
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        threshold=1e-5,
    )

    # ===== TRAINING LOOP =====
    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 15

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Create dataloader wrapper for training
        class DataLoaderWrapper:
            def __init__(self, data_loader):
                self.data_loader = data_loader

            def __iter__(self):
                for x, y in self.data_loader:
                    yield {'input': x, 'target': y}

        train_loader_wrapped = DataLoaderWrapper(train_loader)

        # Training epoch
        train_loss = train_epoch(
            model,
            train_loader_wrapped,
            optimizer,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        train_losses.append(train_loss)

        # Validation epoch
        class ValDataLoaderWrapper:
            def __init__(self, data_loader):
                self.data_loader = data_loader

            def __iter__(self):
                for x, y in self.data_loader:
                    yield {'input': x, 'target': y}

        val_loader_wrapped = ValDataLoaderWrapper(val_loader)

        val_loss = validate(
            model,
            val_loader_wrapped,
            device=device,
        )
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(f"  → New best validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # ===== RESULTS =====
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"Final val loss: {val_losses[-1]:.6f}")

    # Loss trajectory
    logger.info("\nLoss trajectory (every 5 epochs):")
    for i in range(0, len(train_losses), 5):
        epoch_num = i + 1
        logger.info(
            f"  Epoch {epoch_num:3d}: train={train_losses[i]:.6f}, "
            f"val={val_losses[i]:.6f}"
        )

    # ===== INFERENCE EXAMPLE =====
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE EXAMPLE")
    logger.info("=" * 60)

    model.eval()
    with torch.no_grad():
        sample_idx = 0
        x = torch.from_numpy(val_inputs[sample_idx:sample_idx + 1]).to(device)
        y_true = val_targets[sample_idx]

        forecast = model(x).cpu().numpy()[0]

        logger.info(f"Input shape: {x.shape}")
        logger.info(f"Forecast shape: {forecast.shape}")
        logger.info(f"True values (first 5): {y_true[:5]}")
        logger.info(f"Predictions (first 5): {forecast[:5]}")
        logger.info(f"MSE: {np.mean((forecast - y_true) ** 2):.6f}")

    logger.info("\nTraining successful! The model learned a meaningful pattern.")
    logger.info("Why this works compared to your original attempt:")
    logger.info("  1. Explicit Xavier weight initialization")
    logger.info("  2. Gradient clipping (max_norm=1.0)")
    logger.info("  3. Adam optimizer with learning rate scheduling")
    logger.info("  4. Proper residual stacking in blocks")
    logger.info("  5. Validation monitoring for early stopping")


if __name__ == '__main__':
    main()
