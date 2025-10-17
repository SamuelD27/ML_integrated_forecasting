"""
Advanced Training Techniques
==============================
Production-grade training enhancements for deep learning models.

Techniques:
1. Mixed Precision Training (FP16 + FP32) - 2-3x speedup
2. Stochastic Weight Averaging (SWA) - Better generalization
3. Cosine Annealing with Warm Restarts - Improved convergence
4. Gradient Clipping & Accumulation - Stability
5. Label Smoothing - Robustness to noisy labels
6. Early Stopping with Patience - Prevent overfitting
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdvancedTrainingConfig:
    """Configuration for advanced training techniques."""

    # Mixed Precision
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = 'float16'  # 'float16' or 'bfloat16'

    # Stochastic Weight Averaging
    use_swa: bool = True
    swa_start_epoch: int = 40  # Start SWA after 80% of training
    swa_lr: float = 0.0001  # SWA learning rate (typically 0.1-0.01 of base LR)
    swa_freq: int = 5  # Update SWA every N epochs

    # Cosine Annealing
    use_cosine_annealing: bool = True
    T_max: int = 50  # Half-period for cosine annealing
    eta_min: float = 1e-6  # Minimum learning rate
    restart_period: int = 50  # Restart period (0 = no restarts)
    restart_mult: int = 2  # Multiply restart period by this after each restart

    # Gradient Clipping & Accumulation
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = 'norm'  # 'norm' or 'value'
    gradient_accumulation_steps: int = 1  # Accumulate over N batches

    # Label Smoothing
    label_smoothing: float = 0.1  # 0.0 = no smoothing, 0.1 = typical

    # Early Stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    early_stopping_mode: str = 'min'  # 'min' or 'max'

    # Checkpoint
    save_top_k: int = 3
    save_last: bool = True


class MixedPrecisionTrainer:
    """
    Mixed precision training for 2-3x speedup with minimal accuracy loss.

    Uses FP16 for forward/backward passes, FP32 for optimizer updates.
    """

    def __init__(self, use_mixed_precision: bool = True):
        """
        Initialize mixed precision trainer.

        Args:
            use_mixed_precision: Enable mixed precision (requires CUDA)
        """
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_mixed_precision else None

        if self.use_mixed_precision:
            logger.info("Mixed precision training enabled (FP16 + FP32)")
        else:
            logger.info("Mixed precision training disabled (FP32 only)")

    def forward(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mixed precision.

        Args:
            model: PyTorch model
            inputs: Input tensor

        Returns:
            Model output
        """
        if self.use_mixed_precision:
            with autocast():
                return model(inputs)
        else:
            return model(inputs)

    def backward(self, loss: torch.Tensor, optimizer: optim.Optimizer):
        """
        Backward pass with mixed precision.

        Args:
            loss: Loss tensor
            optimizer: Optimizer
        """
        if self.use_mixed_precision:
            # Scale loss for FP16
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, optimizer: optim.Optimizer):
        """
        Optimizer step with mixed precision.

        Args:
            optimizer: Optimizer
        """
        if self.use_mixed_precision:
            # Unscales gradients and calls optimizer.step()
            self.scaler.step(optimizer)
            # Updates scaler for next iteration
            self.scaler.update()
        else:
            optimizer.step()


class SWAWrapper:
    """
    Stochastic Weight Averaging for better generalization.

    Averages model weights over training for more robust model.
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start: int = 40,
        swa_freq: int = 5,
        swa_lr: float = 0.0001
    ):
        """
        Initialize SWA.

        Args:
            model: PyTorch model
            swa_start: Start SWA after this epoch
            swa_freq: Update SWA every N epochs
            swa_lr: SWA learning rate
        """
        self.swa_model = AveragedModel(model)
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.swa_scheduler = None

        logger.info(f"SWA initialized: start={swa_start}, freq={swa_freq}, lr={swa_lr}")

    def update(self, epoch: int):
        """
        Update SWA model if conditions met.

        Args:
            epoch: Current epoch
        """
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            self.swa_model.update_parameters(self.swa_model.module)
            logger.info(f"SWA updated at epoch {epoch}")

    def finalize(self, dataloader: torch.utils.data.DataLoader):
        """
        Finalize SWA by updating batch norm statistics.

        Args:
            dataloader: Training dataloader
        """
        logger.info("Finalizing SWA: updating batch norm statistics")
        torch.optim.swa_utils.update_bn(dataloader, self.swa_model)

    def get_model(self) -> nn.Module:
        """Get averaged model."""
        return self.swa_model


class CosineAnnealingWarmRestarts:
    """
    Cosine annealing learning rate with warm restarts.

    LR follows cosine curve, restarting periodically for better exploration.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int = 50,
        T_mult: int = 2,
        eta_min: float = 1e-6
    ):
        """
        Initialize cosine annealing with restarts.

        Args:
            optimizer: PyTorch optimizer
            T_0: Initial restart period
            T_mult: Multiply restart period by this after each restart
            eta_min: Minimum learning rate
        """
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )

        logger.info(f"Cosine annealing with restarts: T_0={T_0}, T_mult={T_mult}, "
                   f"eta_min={eta_min}")

    def step(self, epoch: Optional[int] = None):
        """Step scheduler."""
        self.scheduler.step(epoch)

    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()


class GradientHandler:
    """
    Gradient clipping and accumulation for training stability.
    """

    def __init__(
        self,
        clip_val: float = 1.0,
        clip_algorithm: str = 'norm',
        accumulation_steps: int = 1
    ):
        """
        Initialize gradient handler.

        Args:
            clip_val: Gradient clipping value
            clip_algorithm: 'norm' (clip by global norm) or 'value' (clip by value)
            accumulation_steps: Accumulate gradients over N steps
        """
        self.clip_val = clip_val
        self.clip_algorithm = clip_algorithm
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0

        logger.info(f"Gradient handler: clip={clip_val} ({clip_algorithm}), "
                   f"accumulation={accumulation_steps}")

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients.

        Args:
            model: PyTorch model

        Returns:
            Total gradient norm
        """
        if self.clip_algorithm == 'norm':
            total_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                self.clip_val
            )
        else:  # 'value'
            nn.utils.clip_grad_value_(
                model.parameters(),
                self.clip_val
            )
            total_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5

        return total_norm

    def should_step(self) -> bool:
        """
        Check if optimizer should step (for gradient accumulation).

        Returns:
            True if accumulated enough steps
        """
        self.accumulation_counter += 1

        if self.accumulation_counter >= self.accumulation_steps:
            self.accumulation_counter = 0
            return True
        return False


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Prevents overconfidence by smoothing labels:
    - True class: 1 - smoothing
    - Other classes: smoothing / (n_classes - 1)
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing loss.

        Args:
            smoothing: Smoothing parameter (0.0 = no smoothing, 0.1 = typical)
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate label smoothing loss.

        Args:
            pred: Predictions (logits)
            target: Targets (class indices)

        Returns:
            Loss
        """
        n_classes = pred.size(-1)

        # Convert to one-hot
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

        # Smooth labels
        smooth_one_hot = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (n_classes - 1)

        # Cross entropy with smooth labels
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -(smooth_one_hot * log_prob).sum(dim=-1).mean()

        return loss


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Stops training if validation metric doesn't improve for N epochs.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' (loss) or 'max' (accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}, mode={mode}")

    def __call__(self, metric: float) -> bool:
        """
        Check if should stop training.

        Args:
            metric: Current validation metric

        Returns:
            True if should stop training
        """
        score = -metric if self.mode == 'min' else metric

        if self.best_score is None:
            self.best_score = score
            return False

        # Check if improved
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                logger.info("Early stopping triggered")
                self.early_stop = True
                return True

        return False


class AdvancedTrainer:
    """
    Complete advanced training wrapper with all techniques.

    Example:
        >>> config = AdvancedTrainingConfig()
        >>> trainer = AdvancedTrainer(model, config)
        >>> trainer.train(train_loader, val_loader, epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdvancedTrainingConfig,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = 'cuda'
    ):
        """
        Initialize advanced trainer.

        Args:
            model: PyTorch model
            config: Training configuration
            optimizer: Optimizer (defaults to AdamW)
            device: Device ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        # Mixed precision
        self.mp_trainer = MixedPrecisionTrainer(config.use_mixed_precision)

        # SWA
        if config.use_swa:
            self.swa = SWAWrapper(
                model,
                swa_start=config.swa_start_epoch,
                swa_freq=config.swa_freq,
                swa_lr=config.swa_lr
            )
        else:
            self.swa = None

        # Scheduler
        if config.use_cosine_annealing:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.T_max,
                T_mult=config.restart_mult,
                eta_min=config.eta_min
            )
        else:
            self.scheduler = None

        # Gradient handler
        self.grad_handler = GradientHandler(
            clip_val=config.gradient_clip_val,
            clip_algorithm=config.gradient_clip_algorithm,
            accumulation_steps=config.gradient_accumulation_steps
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode=config.early_stopping_mode
        )

        # Loss function
        if config.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass (mixed precision)
            outputs = self.mp_trainer.forward(self.model, inputs)
            loss = self.criterion(outputs, targets)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass (mixed precision)
            self.mp_trainer.backward(loss, self.optimizer)

            # Gradient clipping
            if self.grad_handler.should_step():
                grad_norm = self.grad_handler.clip_gradients(self.model)

                # Optimizer step
                self.mp_trainer.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps

        return total_loss / len(dataloader)

    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50
    ) -> Dict:
        """
        Train model with all advanced techniques.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs

        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # Learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch)
                lr = self.scheduler.get_last_lr()[0]
                self.history['learning_rate'].append(lr)

            # SWA
            if self.swa is not None:
                self.swa.update(epoch)

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Logging
            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"lr={lr:.6f}"
                )

        # Finalize SWA
        if self.swa is not None:
            self.swa.finalize(train_loader)
            logger.info("SWA finalized")

        return self.history


if __name__ == '__main__':
    # Example usage
    print("Advanced Training Techniques Example")
    print("=" * 60)

    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 2)
    )

    # Configuration
    config = AdvancedTrainingConfig(
        use_mixed_precision=True,
        use_swa=True,
        use_cosine_annealing=True,
        gradient_clip_val=1.0,
        gradient_accumulation_steps=2,
        label_smoothing=0.1,
        early_stopping_patience=10
    )

    print("\nConfiguration:")
    print(f"  Mixed Precision: {config.use_mixed_precision}")
    print(f"  SWA: {config.use_swa}")
    print(f"  Cosine Annealing: {config.use_cosine_annealing}")
    print(f"  Gradient Clipping: {config.gradient_clip_val}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Label Smoothing: {config.label_smoothing}")
    print(f"  Early Stopping Patience: {config.early_stopping_patience}")

    print("\n✓ Advanced training module ready")
