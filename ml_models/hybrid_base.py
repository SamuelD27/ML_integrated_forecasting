"""
Base class for hybrid deep learning models.
===========================================

Provides common interface, training loops, checkpointing, and metrics tracking
for all hybrid model components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_rmse: float
    val_rmse: float
    train_mae: float
    val_mae: float
    directional_accuracy: float
    learning_rate: float


class HybridModelBase(nn.Module):
    """
    Base class for all hybrid trading models.

    Provides:
    - Training loop abstraction
    - Checkpointing and model persistence
    - Metrics tracking (RMSE, MAE, Directional Accuracy, Sharpe)
    - Device management (CPU/GPU)
    """

    def __init__(self, input_dim: int, output_dim: int = 1, device: str = None):
        """
        Initialize base model.

        Args:
            input_dim: Number of input features
            output_dim: Number of output predictions
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Device management
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Training history
        self.training_history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Model metadata
        self.model_name = self.__class__.__name__
        self.created_at = datetime.now().isoformat()

        logger.info(f"Initialized {self.model_name} on {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    loss_fn: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute loss between predictions and targets.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss_fn: Loss function (default: MSE for regression)

        Returns:
            Loss tensor
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        return loss_fn(predictions, targets)

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   loss_fn: Optional[nn.Module] = None,
                   gradient_clip: float = 1.0) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            gradient_clip: Maximum gradient norm for clipping

        Returns:
            Dictionary of training metrics
        """
        self.train()
        total_loss = 0
        predictions_list = []
        targets_list = []

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            predictions = self(features)

            # Compute loss
            loss = self.compute_loss(predictions, targets, loss_fn)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)

            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions_list.append(predictions.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        all_predictions = np.concatenate(predictions_list)
        all_targets = np.concatenate(targets_list)

        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)

        return {
            'loss': avg_loss,
            'rmse': rmse,
            'mae': mae
        }

    def validate(self, val_loader: DataLoader,
                loss_fn: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Validate model on validation set.

        Args:
            val_loader: Validation data loader
            loss_fn: Loss function

        Returns:
            Dictionary of validation metrics
        """
        self.eval()
        total_loss = 0
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                predictions = self(features)
                loss = self.compute_loss(predictions, targets, loss_fn)

                total_loss += loss.item()
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(targets.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        all_predictions = np.concatenate(predictions_list)
        all_targets = np.concatenate(targets_list)

        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)

        # Directional accuracy (for price predictions)
        if all_predictions.shape[-1] == 1:  # Single output
            directional_accuracy = self.compute_directional_accuracy(
                all_predictions.flatten(), all_targets.flatten()
            )
        else:
            directional_accuracy = 0.0

        return {
            'loss': avg_loss,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }

    def compute_directional_accuracy(self, predictions: np.ndarray,
                                    targets: np.ndarray) -> float:
        """
        Compute directional accuracy (correct sign of change).

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            Directional accuracy percentage
        """
        # Assuming these are returns or price changes
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)

        correct = (pred_direction == true_direction).sum()
        total = len(predictions)

        return (correct / total) * 100 if total > 0 else 0.0

    def compute_sharpe_ratio(self, returns: np.ndarray,
                           risk_free_rate: float = 0.02) -> float:
        """
        Compute Sharpe ratio from returns.

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        # Assuming daily returns
        excess_returns = returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-8)

        return sharpe

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
           epochs: int = 100, learning_rate: float = 0.001,
           optimizer_type: str = 'adamw', scheduler_type: str = 'reduce_on_plateau',
           patience: int = 15, gradient_clip: float = 1.0,
           weight_decay: float = 1e-5, checkpoint_dir: Optional[Path] = None,
           verbose: bool = True) -> Dict[str, Any]:
        """
        Full training pipeline with validation and early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            learning_rate: Initial learning rate
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
            scheduler_type: Type of LR scheduler ('reduce_on_plateau', 'cosine')
            patience: Early stopping patience
            gradient_clip: Maximum gradient norm
            weight_decay: L2 regularization weight
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to print training progress

        Returns:
            Dictionary containing training history and best metrics
        """
        # Setup optimizer
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Setup learning rate scheduler
        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        else:
            scheduler = None

        # Setup checkpointing
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        no_improve_count = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, gradient_clip=gradient_clip
            )

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                if scheduler_type == 'reduce_on_plateau':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()

            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_rmse=train_metrics['rmse'],
                val_rmse=val_metrics['rmse'],
                train_mae=train_metrics['mae'],
                val_mae=val_metrics['mae'],
                directional_accuracy=val_metrics['directional_accuracy'],
                learning_rate=current_lr
            )
            self.training_history.append(metrics)

            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                no_improve_count = 0

                # Save best model
                if checkpoint_dir:
                    self.save_checkpoint(
                        checkpoint_dir / 'best_model.pt',
                        optimizer, epoch, val_metrics
                    )
            else:
                no_improve_count += 1

            # Print progress
            if verbose and epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val RMSE: {val_metrics['rmse']:.4f}, "
                    f"Dir Acc: {val_metrics['directional_accuracy']:.1f}%, "
                    f"LR: {current_lr:.6f}"
                )

            # Early stopping
            if no_improve_count >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final model
        if checkpoint_dir:
            self.save_checkpoint(
                checkpoint_dir / 'final_model.pt',
                optimizer, epoch, val_metrics
            )

        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'final_metrics': val_metrics
        }

    def save_checkpoint(self, path: Path, optimizer: optim.Optimizer,
                       epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state to save
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'model_config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'model_name': self.model_name
            },
            'training_history': [
                {
                    'epoch': m.epoch,
                    'train_loss': m.train_loss,
                    'val_loss': m.val_loss,
                    'train_rmse': m.train_rmse,
                    'val_rmse': m.val_rmse,
                    'directional_accuracy': m.directional_accuracy
                } for m in self.training_history
            ]
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path, optimizer: Optional[optim.Optimizer] = None):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            optimizer: Optimizer to load state into (optional)
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded checkpoint from {path}")

        return checkpoint.get('epoch', 0)

    def predict(self, features: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            features: Input features

        Returns:
            Predictions as numpy array
        """
        self.eval()

        # Convert to tensor if needed
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)

        features = features.to(self.device)

        with torch.no_grad():
            if len(features.shape) == 2:
                # Add batch dimension if needed
                features = features.unsqueeze(0) if features.shape[0] != self.input_dim else features.unsqueeze(0)

            predictions = self(features)

        return predictions.cpu().numpy()

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary including architecture and parameters.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'created_at': self.created_at,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }