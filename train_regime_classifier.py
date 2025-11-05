"""
Training script for market regime classifier.

Quick training pipeline with validation and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Optional
import logging

from ml_models.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)


class RegimeClassifierTrainer:
    """
    Trainer for regime classifier.

    Standard supervised learning loop:
    1. Create train/val dataloaders
    2. Loop through epochs
    3. Update weights with CrossEntropyLoss
    4. Validate and checkpoint best model
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden1_size: int = 64,
        hidden2_size: int = 32,
        output_size: int = 4,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            input_size: Number of input features
            hidden1_size: First hidden layer size
            hidden2_size: Second hidden layer size
            output_size: Number of output classes
            learning_rate: Learning rate
            batch_size: Batch size
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Model
        self.model = RegimeClassifier(
            input_size=input_size,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            output_size=output_size,
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        logger.info(f"Trainer initialized on {self.device}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Run one training epoch.

        Args:
            train_loader: Training dataloader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx + 1}: "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.

        Args:
            val_loader: Validation dataloader

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                # Accuracy
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        checkpoint_path: Optional[str] = None,
        early_stopping_patience: int = 10,
    ) -> Dict:
        """
        Train model.

        Args:
            X_train: Training features (N, 10)
            y_train: Training labels (N,)
            X_val: Validation features (M, 10)
            y_val: Validation labels (M,)
            epochs: Number of epochs
            checkpoint_path: Path to save best model
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history dict
        """
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        patience_counter = 0

        logger.info(
            f"Starting training: {epochs} epochs, "
            f"{len(train_loader)} batches/epoch"
        )

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.2%}"
            )

            # Early stopping and checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
                    logger.info(f"Best model saved to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping after {epoch + 1} epochs"
                    )
                    break

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': epoch + 1,
        }

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden1_size': self.model.hidden1_size,
                'hidden2_size': self.model.hidden2_size,
                'output_size': self.model.output_size,
            },
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def create_sample_dataset(
    n_samples: int = 1000,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic training data for testing.

    Args:
        n_samples: Number of samples
        random_seed: Random seed

    Returns:
        (X_train, y_train, X_val, y_val)
    """
    np.random.seed(random_seed)

    # Generate synthetic features and labels
    X = np.random.randn(n_samples, 10).astype(np.float32)
    y = np.random.randint(0, 4, n_samples)

    # Make labels somewhat related to features (for learning)
    X[:100, 0] = np.abs(X[:100, 0]) + 1  # Bull market has high volatility
    y[:100] = 0

    X[100:200, 0] = np.abs(X[100:200, 0]) + 2  # Bear market
    y[100:200] = 1

    # Train/val split (80/20)
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


if __name__ == '__main__':
    # Example training run
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    X_train, y_train, X_val, y_val = create_sample_dataset(n_samples=1000)

    # Initialize trainer
    trainer = RegimeClassifierTrainer(
        learning_rate=0.001,
        batch_size=32,
    )

    # Train
    history = trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        checkpoint_path='/tmp/regime_model_best.pt',
        early_stopping_patience=10,
    )

    print("\nTraining Summary:")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Epochs trained: {history['epochs_trained']}")
    print("✓ Training complete!")
