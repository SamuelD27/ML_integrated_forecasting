"""
Fast market regime classifier using PyTorch.

Architecture: 3-layer fully connected neural network
Input: 10 features -> Hidden: 64 -> 32 -> Output: 4 regimes
Regimes: 0=Bull, 1=Bear, 2=Neutral, 3=Crisis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Regime mapping
REGIME_CLASSES = {
    0: 'bull',
    1: 'bear',
    2: 'neutral',
    3: 'crisis'
}

REGIME_TO_CLASS = {v: k for k, v in REGIME_CLASSES.items()}


class RegimeClassifier(nn.Module):
    """
    Fast market regime classifier.

    Simple 3-layer FC network:
    - Input: 10 features
    - Hidden1: 64 neurons (ReLU)
    - Hidden2: 32 neurons (ReLU)
    - Output: 4 classes (bull/bear/neutral/crisis)

    Args:
        input_size: Number of input features (default: 10)
        hidden1_size: First hidden layer size (default: 64)
        hidden2_size: Second hidden layer size (default: 32)
        output_size: Number of regime classes (default: 4)
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden1_size: int = 64,
        hidden2_size: int = 32,
        output_size: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        # Layer 1: 10 -> 64
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size)
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2: 64 -> 32
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size)
        self.dropout2 = nn.Dropout(dropout)

        # Layer 3: 32 -> 4 (output)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 10)

        Returns:
            Logits of shape (batch_size, 4)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Layer 3 (logits)
        x = self.fc3(x)

        return x


class RegimeDetectorNN:
    """
    Inference wrapper for regime classification.

    Handles data normalization, device management, and batch prediction.

    Attributes:
        model: PyTorch model
        device: torch.device (cpu or cuda)
        feature_names: List of 10 feature names
        feature_stats: Dict with mean/std for normalization
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize regime detector.

        Args:
            model_path: Path to saved model checkpoint (optional)
            device: 'cpu' or 'cuda' (auto-detect if None)
        """
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        logger.info(f"Regime detector using device: {self.device}")

        # Initialize model
        self.model = RegimeClassifier()
        self.model.to(self.device)
        self.model.eval()

        # Load checkpoint if provided
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.feature_stats = checkpoint.get('feature_stats', None)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
                self.feature_stats = None
        else:
            self.feature_stats = None

        # Feature names (in order expected by model)
        self.feature_names = [
            'vol_20d',
            'ret_20d',
            'ret_60d',
            'vol_ratio',
            'rsi',
            'macd',
            'adx',
            'volume_ratio',
            'price_momentum',
            'volatility_trend',
        ]

    def set_feature_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        Set feature normalization statistics.

        Args:
            mean: Array of shape (10,)
            std: Array of shape (10,)
        """
        self.feature_stats = {'mean': mean, 'std': std}
        logger.info("Feature stats set for normalization")

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using stored statistics.

        Args:
            features: Array of shape (10,) or (batch_size, 10)

        Returns:
            Normalized features
        """
        if self.feature_stats is None:
            logger.warning("No feature stats set - returning unnormalized features")
            return features

        mean = self.feature_stats['mean']
        std = self.feature_stats['std']

        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)

        return (features - mean) / std

    @torch.no_grad()
    def predict(
        self,
        features: np.ndarray,
        normalize: bool = True,
        return_confidence: bool = True,
    ) -> Dict:
        """
        Predict regime for given features.

        Args:
            features: Array of shape (10,) or (batch_size, 10)
            normalize: Whether to normalize features
            return_confidence: Whether to return probability distribution

        Returns:
            Dict with keys:
                - regime: Regime name ('bull', 'bear', 'neutral', 'crisis')
                - class_id: Integer class (0, 1, 2, 3)
                - confidence: Prediction confidence (0-1)
                - probabilities: Array of class probabilities (if return_confidence=True)
        """
        # Validate input shape
        if features.ndim == 1:
            if len(features) != 10:
                raise ValueError(
                    f"Expected 10 features, got {len(features)}"
                )
            features = features.reshape(1, -1)
        elif features.ndim == 2:
            if features.shape[1] != 10:
                raise ValueError(
                    f"Expected 10 features, got {features.shape[1]}"
                )
        else:
            raise ValueError(f"Expected 1D or 2D array, got {features.ndim}D")

        # Normalize if stats available
        if normalize:
            features = self.normalize_features(features)

        # Convert to torch tensor
        x = torch.from_numpy(features).float().to(self.device)

        # Forward pass
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        # Get predictions
        class_ids = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()

        # Single sample or batch?
        if features.shape[0] == 1:
            result = {
                'regime': REGIME_CLASSES[class_ids[0]],
                'class_id': int(class_ids[0]),
                'confidence': float(confidences[0]),
            }
            if return_confidence:
                result['probabilities'] = {
                    REGIME_CLASSES[i]: float(p)
                    for i, p in enumerate(probs[0].cpu().numpy())
                }
        else:
            # Batch results
            regimes = [REGIME_CLASSES[cid] for cid in class_ids]
            result = {
                'regimes': regimes,
                'class_ids': class_ids.tolist(),
                'confidences': confidences.tolist(),
            }
            if return_confidence:
                result['probabilities_batch'] = probs.cpu().numpy()

        return result

    def predict_from_dict(
        self,
        feature_dict: Dict[str, float],
        normalize: bool = True,
        return_confidence: bool = True,
    ) -> Dict:
        """
        Predict regime from dictionary of features.

        Args:
            feature_dict: Dict mapping feature names to values
            normalize: Whether to normalize
            return_confidence: Whether to return probabilities

        Returns:
            Prediction dict
        """
        # Extract features in order
        features = np.array(
            [feature_dict[name] for name in self.feature_names]
        )
        return self.predict(features, normalize=normalize,
                          return_confidence=return_confidence)

    def save_model(self, checkpoint_path: str,
                   feature_stats: Optional[Dict] = None) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            feature_stats: Feature normalization stats to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': 10,
                'hidden1_size': 64,
                'hidden2_size': 32,
                'output_size': 4,
            },
            'feature_names': self.feature_names,
            'feature_stats': feature_stats or self.feature_stats,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

    def load_model(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_stats = checkpoint.get('feature_stats', None)
        logger.info(f"Model loaded from {checkpoint_path}")
