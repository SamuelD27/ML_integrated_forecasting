"""
Hybrid Trading Model
====================

This module integrates CNN, LSTM, and Transformer components into a unified
hybrid deep learning model for financial time series prediction.

The model implements:
- Three-stage pipeline (parallel feature extraction → fusion → prediction)
- Multi-task learning (price prediction + direction classification)
- Comprehensive regularization techniques
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import json

from .hybrid_base import HybridModelBase
from .cnn_module import CNN1DFeatureExtractor
from .lstm_module import LSTMEncoder
from .transformer_module import TransformerEncoder
from .fusion_layer import AttentionFusion

logger = logging.getLogger(__name__)


class HybridTradingModel(HybridModelBase):
    """
    Hybrid deep learning model for trading that combines CNN, LSTM, and Transformer.

    Architecture:
    1. Parallel feature extraction using CNN, LSTM, and Transformer
    2. Attention-based fusion of extracted features
    3. Multi-task prediction head for price and direction

    The model is designed to capture:
    - Short-term patterns (CNN)
    - Sequential dependencies (LSTM)
    - Global context (Transformer)
    """

    def __init__(self, input_dim: int, sequence_length: int,
                # CNN parameters
                cnn_filters: List[int] = [64, 128, 256],
                cnn_kernel_sizes: List[int] = [3, 5, 7],
                cnn_dropout: float = 0.3,
                # LSTM parameters
                lstm_hidden: int = 256,
                lstm_layers: int = 2,
                lstm_bidirectional: bool = True,
                lstm_dropout: float = 0.2,
                # Transformer parameters
                transformer_d_model: int = 512,
                transformer_heads: int = 8,
                transformer_layers: int = 4,
                transformer_ff_dim: int = 2048,
                transformer_dropout: float = 0.1,
                # Fusion parameters
                fusion_hidden_dim: int = 256,
                fusion_dropout: float = 0.1,
                # Output parameters
                output_dim: int = 1,
                predict_direction: bool = True,
                device: str = None):
        """
        Initialize hybrid trading model.

        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            cnn_filters: Filter counts for CNN layers
            cnn_kernel_sizes: Kernel sizes for multi-scale CNN
            cnn_dropout: CNN dropout rate
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            lstm_bidirectional: Whether to use bidirectional LSTM
            lstm_dropout: LSTM dropout rate
            transformer_d_model: Transformer model dimension
            transformer_heads: Number of attention heads
            transformer_layers: Number of transformer layers
            transformer_ff_dim: Feed-forward dimension
            transformer_dropout: Transformer dropout rate
            fusion_hidden_dim: Hidden dimension for fusion layer
            fusion_dropout: Fusion layer dropout
            output_dim: Final output dimension (1 for regression)
            predict_direction: Whether to predict price direction
            device: Device to use (cuda/cpu)
        """
        super().__init__(input_dim, output_dim, device)

        self.sequence_length = sequence_length
        self.predict_direction = predict_direction

        # Store hyperparameters
        self.hparams = {
            'input_dim': input_dim,
            'sequence_length': sequence_length,
            'cnn_filters': cnn_filters,
            'cnn_kernel_sizes': cnn_kernel_sizes,
            'lstm_hidden': lstm_hidden,
            'lstm_layers': lstm_layers,
            'transformer_d_model': transformer_d_model,
            'transformer_heads': transformer_heads,
            'transformer_layers': transformer_layers
        }

        # 1. Feature Extractors
        logger.info("Initializing feature extractors...")

        # CNN for short-term patterns
        self.cnn = CNN1DFeatureExtractor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            filters=cnn_filters,
            kernel_sizes=cnn_kernel_sizes,
            dropout=cnn_dropout,
            output_dim=fusion_hidden_dim
        )

        # LSTM for sequential patterns
        self.lstm = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout,
            use_attention=True,
            output_dim=fusion_hidden_dim
        )

        # Transformer for global context
        self.transformer = TransformerEncoder(
            input_dim=input_dim,
            d_model=transformer_d_model,
            n_heads=transformer_heads,
            num_layers=transformer_layers,
            d_ff=transformer_ff_dim,
            dropout=transformer_dropout,
            max_seq_len=sequence_length,
            use_dual_attention=True,
            output_dim=fusion_hidden_dim
        )

        # 2. Fusion Layer
        logger.info("Initializing fusion layer...")

        encoder_dims = {
            'cnn': fusion_hidden_dim,
            'lstm': fusion_hidden_dim,
            'transformer': fusion_hidden_dim
        }

        self.fusion = AttentionFusion(
            input_dims=encoder_dims,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            use_gating=True
        )

        # 3. Prediction Heads
        logger.info("Initializing prediction heads...")

        # Price prediction head (regression)
        self.price_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim // 2, output_dim)
        )

        # Direction prediction head (classification)
        if predict_direction:
            self.direction_head = nn.Sequential(
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden_dim // 2, 3)  # Up, Down, Neutral
            )

        # Initialize weights
        self._initialize_weights()

        # Move to device
        self.to(self.device)

        logger.info(f"Initialized HybridTradingModel with {self._count_parameters():,} parameters")

    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor,
               return_intermediates: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hybrid model.

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            return_intermediates: Whether to return intermediate outputs

        Returns:
            If return_intermediates=False: predictions tensor
            If return_intermediates=True: Dictionary with all outputs
        """
        batch_size, seq_len, features = x.shape

        # Ensure correct sequence length
        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {seq_len}")

        # 1. Parallel Feature Extraction
        # CNN features (short-term patterns)
        cnn_features, cnn_attention = self.cnn(x)

        # LSTM features (sequential patterns)
        lstm_features, lstm_hidden_states = self.lstm(x)

        # Transformer features (global context)
        transformer_features, transformer_attention = self.transformer(x)

        # 2. Feature Fusion
        encoder_outputs = {
            'cnn': cnn_features,
            'lstm': lstm_features,
            'transformer': transformer_features
        }

        fused_features, fusion_weights = self.fusion(encoder_outputs, return_weights=True)

        # 3. Prediction
        price_prediction = self.price_head(fused_features)

        if self.predict_direction:
            direction_logits = self.direction_head(fused_features)
            direction_probs = F.softmax(direction_logits, dim=-1)
        else:
            direction_logits = None
            direction_probs = None

        # Return based on mode
        if return_intermediates:
            return {
                'price_prediction': price_prediction,
                'direction_logits': direction_logits,
                'direction_probs': direction_probs,
                'fused_features': fused_features,
                'cnn_features': cnn_features,
                'lstm_features': lstm_features,
                'transformer_features': transformer_features,
                'cnn_attention': cnn_attention,
                'transformer_attention': transformer_attention,
                'fusion_weights': fusion_weights
            }
        else:
            return price_prediction

    def compute_multi_task_loss(self, predictions: Dict[str, torch.Tensor],
                              targets: Dict[str, torch.Tensor],
                              price_weight: float = 1.0,
                              direction_weight: float = 0.5) -> torch.Tensor:
        """
        Compute multi-task loss for price and direction prediction.

        Args:
            predictions: Dictionary with 'price' and 'direction_logits'
            targets: Dictionary with 'price' and 'direction' targets
            price_weight: Weight for price prediction loss
            direction_weight: Weight for direction classification loss

        Returns:
            Combined loss
        """
        total_loss = 0

        # Price prediction loss (Huber loss for robustness)
        if 'price' in targets:
            price_loss = F.huber_loss(predictions['price_prediction'], targets['price'])
            total_loss += price_weight * price_loss

        # Direction classification loss
        if self.predict_direction and 'direction' in targets:
            direction_loss = F.cross_entropy(predictions['direction_logits'], targets['direction'])
            total_loss += direction_weight * direction_loss

        return total_loss

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimation using dropout sampling.

        Args:
            x: Input tensor
            n_samples: Number of forward passes for uncertainty estimation

        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        self.train()  # Enable dropout for uncertainty estimation

        price_predictions = []
        direction_probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self(x, return_intermediates=True)
                price_predictions.append(outputs['price_prediction'].cpu().numpy())

                if self.predict_direction:
                    direction_probs.append(outputs['direction_probs'].cpu().numpy())

        price_predictions = np.array(price_predictions)
        price_mean = price_predictions.mean(axis=0)
        price_std = price_predictions.std(axis=0)

        results = {
            'price_mean': price_mean,
            'price_std': price_std,
            'price_lower': price_mean - 2 * price_std,
            'price_upper': price_mean + 2 * price_std
        }

        if self.predict_direction:
            direction_probs = np.array(direction_probs)
            results['direction_probs'] = direction_probs.mean(axis=0)
            results['direction_uncertainty'] = direction_probs.std(axis=0)

        self.eval()  # Reset to eval mode
        return results

    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Get feature importance scores from the model.

        Args:
            x: Input tensor

        Returns:
            Dictionary with importance scores for each component
        """
        self.eval()

        with torch.no_grad():
            outputs = self(x, return_intermediates=True)

        # Get fusion weights (attention-based importance)
        fusion_weights = outputs.get('fusion_weights', {})

        # Compute gradient-based importance (optional)
        x.requires_grad_(True)
        price_pred = self(x)
        price_pred.sum().backward()

        feature_gradients = x.grad.abs().mean(dim=(0, 1)).cpu().numpy()

        return {
            'fusion_weights': fusion_weights,
            'feature_gradients': feature_gradients.tolist()
        }

    def save_model(self, path: Path, include_optimizer: bool = True,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  additional_info: Dict[str, Any] = None):
        """
        Save model with all necessary information.

        Args:
            path: Path to save model
            include_optimizer: Whether to save optimizer state
            optimizer: Optimizer instance
            additional_info: Additional information to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hparams': self.hparams,
            'model_class': self.__class__.__name__,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'sequence_length': self.sequence_length,
            'best_val_loss': self.best_val_loss,
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

        if include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if additional_info:
            checkpoint['additional_info'] = additional_info

        torch.save(checkpoint, path)
        logger.info(f"Saved model to {path}")

        # Also save hyperparameters as JSON for reference
        hparams_path = path.parent / f"{path.stem}_hparams.json"
        with open(hparams_path, 'w') as f:
            json.dump(self.hparams, f, indent=2)

    @classmethod
    def load_model(cls, path: Path, device: Optional[str] = None) -> 'HybridTradingModel':
        """
        Load saved model.

        Args:
            path: Path to saved model
            device: Device to load model on

        Returns:
            Loaded HybridTradingModel instance
        """
        checkpoint = torch.load(path, map_location=device or 'cpu')

        # Create model instance
        model = cls(
            input_dim=checkpoint['input_dim'],
            sequence_length=checkpoint['sequence_length'],
            **checkpoint.get('hparams', {}),
            device=device
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Loaded model from {path}")
        return model