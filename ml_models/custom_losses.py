"""
Custom Loss Functions for Trading

Implements specialized loss functions that prioritize directional accuracy:
- DirectionalLoss: Penalizes wrong direction predictions more than magnitude errors
- AsymmetricLoss: Different penalties for under/over-prediction
- SharpeL loss: Optimizes for Sharpe ratio
- Combined losses for multi-objective training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DirectionalLoss(nn.Module):
    """
    Loss function that heavily penalizes wrong direction predictions.

    Combines MSE loss with directional penalty.
    """

    def __init__(
        self,
        direction_weight: float = 2.0,
        magnitude_weight: float = 1.0,
        epsilon: float = 1e-8
    ):
        """
        Initialize directional loss.

        Args:
            direction_weight: Weight for directional penalty
            magnitude_weight: Weight for magnitude (MSE) loss
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.epsilon = epsilon

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prev_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute directional loss.

        Args:
            predictions: Predicted values (batch_size, 1)
            targets: Target values (batch_size, 1)
            prev_values: Previous values for computing direction (batch_size, 1)

        Returns:
            Combined loss
        """
        # Magnitude loss (MSE)
        mse_loss = F.mse_loss(predictions, targets)

        # Directional loss
        if prev_values is not None:
            # Compute actual and predicted directions
            true_direction = torch.sign(targets - prev_values)
            pred_direction = torch.sign(predictions - prev_values)

            # Directional accuracy (1 if correct, 0 if wrong)
            direction_correct = (true_direction == pred_direction).float()

            # Penalty for wrong direction (0 if correct, 1 if wrong)
            direction_penalty = 1.0 - direction_correct

            # Average directional penalty
            direction_loss = direction_penalty.mean()
        else:
            # If no previous values, use sign agreement
            direction_loss = 1.0 - torch.mean(
                (torch.sign(predictions) == torch.sign(targets)).float()
            )

        # Combined loss
        total_loss = (
            self.magnitude_weight * mse_loss +
            self.direction_weight * direction_loss
        )

        return total_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss with different penalties for over/under-prediction.

    Useful when one type of error is more costly (e.g., underestimating risk).
    """

    def __init__(
        self,
        over_penalty: float = 1.0,
        under_penalty: float = 1.5
    ):
        """
        Initialize asymmetric loss.

        Args:
            over_penalty: Weight for over-prediction errors
            under_penalty: Weight for under-prediction errors
        """
        super().__init__()
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Weighted loss
        """
        errors = predictions - targets

        # Over-prediction (predicted > actual)
        over_errors = torch.where(errors > 0, errors, torch.zeros_like(errors))
        over_loss = self.over_penalty * (over_errors ** 2).mean()

        # Under-prediction (predicted < actual)
        under_errors = torch.where(errors < 0, errors, torch.zeros_like(errors))
        under_loss = self.under_penalty * (under_errors ** 2).mean()

        return over_loss + under_loss


class SharpeLoss(nn.Module):
    """
    Loss function that optimizes for Sharpe ratio.

    Encourages predictions that lead to high risk-adjusted returns.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        epsilon: float = 1e-8
    ):
        """
        Initialize Sharpe loss.

        Args:
            risk_free_rate: Annual risk-free rate
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate / 252  # Daily rate
        self.epsilon = epsilon

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sharpe loss.

        Args:
            predictions: Predicted returns
            targets: Actual returns

        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Use predictions to simulate returns
        # Assume we trade based on predicted direction
        predicted_positions = torch.sign(predictions)
        realized_returns = predicted_positions * targets

        # Calculate Sharpe ratio
        mean_return = realized_returns.mean()
        std_return = realized_returns.std() + self.epsilon

        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return

        # Return negative Sharpe (we want to minimize loss, maximize Sharpe)
        return -sharpe_ratio


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions.

    Allows multi-objective optimization.
    """

    def __init__(
        self,
        loss_weights: Optional[dict] = None,
        use_directional: bool = True,
        use_mse: bool = True,
        use_sharpe: bool = False
    ):
        """
        Initialize combined loss.

        Args:
            loss_weights: Weights for each loss component
            use_directional: Whether to include directional loss
            use_mse: Whether to include MSE loss
            use_sharpe: Whether to include Sharpe loss
        """
        super().__init__()

        self.loss_weights = loss_weights or {
            'mse': 1.0,
            'directional': 2.0,
            'sharpe': 0.5,
        }

        self.use_directional = use_directional
        self.use_mse = use_mse
        self.use_sharpe = use_sharpe

        # Initialize loss components
        self.mse_loss = nn.MSELoss()
        self.directional_loss = DirectionalLoss()
        self.sharpe_loss = SharpeLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prev_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            predictions: Predicted values
            targets: Target values
            prev_values: Previous values for directional loss

        Returns:
            Tuple of (total_loss, loss_components)
        """
        loss_components = {}
        total_loss = 0.0

        # MSE loss
        if self.use_mse:
            mse = self.mse_loss(predictions, targets)
            loss_components['mse'] = mse.item()
            total_loss += self.loss_weights['mse'] * mse

        # Directional loss
        if self.use_directional:
            directional = self.directional_loss(predictions, targets, prev_values)
            loss_components['directional'] = directional.item()
            total_loss += self.loss_weights['directional'] * directional

        # Sharpe loss
        if self.use_sharpe:
            sharpe = self.sharpe_loss(predictions, targets)
            loss_components['sharpe'] = sharpe.item()
            total_loss += self.loss_weights['sharpe'] * sharpe

        return total_loss, loss_components


class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic forecasting.

    Useful for predicting confidence intervals.
    """

    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        """
        Initialize quantile loss.

        Args:
            quantiles: List of quantiles to predict
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Predicted quantiles (batch_size, n_quantiles)
            targets: Target values (batch_size, 1)

        Returns:
            Quantile loss
        """
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i:i+1]
            loss = torch.max(
                (q - 1) * errors,
                q * errors
            )
            losses.append(loss.mean())

        return torch.stack(losses).mean()


class HuberLoss(nn.Module):
    """
    Huber loss - robust to outliers.

    Behaves like MSE for small errors, MAE for large errors.
    """

    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss.

        Args:
            delta: Threshold for switching from quadratic to linear
        """
        super().__init__()
        self.delta = delta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Huber loss.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Huber loss
        """
        errors = predictions - targets
        abs_errors = torch.abs(errors)

        # Quadratic for small errors
        quadratic = 0.5 * (errors ** 2)

        # Linear for large errors
        linear = self.delta * abs_errors - 0.5 * (self.delta ** 2)

        # Choose based on error magnitude
        loss = torch.where(
            abs_errors <= self.delta,
            quadratic,
            linear
        )

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance in classification.

    Down-weights easy examples, focuses on hard examples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Predicted logits (batch_size, n_classes)
            targets: Target class indices (batch_size,)

        Returns:
            Focal loss
        """
        # Convert to probabilities
        probs = F.softmax(predictions, dim=1)

        # Get probabilities for target class
        targets_one_hot = F.one_hot(targets, num_classes=predictions.shape[1])
        pt = (probs * targets_one_hot).sum(dim=1)

        # Focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(pt + 1e-8)

        return loss.mean()


def create_loss_function(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss ('mse', 'directional', 'combined', 'sharpe', etc.)
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function module
    """
    loss_functions = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'directional': DirectionalLoss,
        'asymmetric': AsymmetricLoss,
        'sharpe': SharpeLoss,
        'combined': CombinedLoss,
        'quantile': QuantileLoss,
        'huber': HuberLoss,
        'focal': FocalLoss,
    }

    if loss_type not in loss_functions:
        logger.warning(f"Unknown loss type '{loss_type}', using MSE")
        loss_type = 'mse'

    return loss_functions[loss_type](**kwargs)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    batch_size = 32
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)
    prev_values = torch.randn(batch_size, 1)

    # Test directional loss
    print("\n=== Directional Loss ===")
    dir_loss = DirectionalLoss(direction_weight=2.0)
    loss = dir_loss(predictions, targets, prev_values)
    print(f"Loss: {loss.item():.4f}")

    # Test combined loss
    print("\n=== Combined Loss ===")
    combined_loss = CombinedLoss()
    loss, components = combined_loss(predictions, targets, prev_values)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Components: {components}")

    # Test Sharpe loss
    print("\n=== Sharpe Loss ===")
    sharpe_loss = SharpeLoss()
    loss = sharpe_loss(predictions, targets)
    print(f"Loss (negative Sharpe): {loss.item():.4f}")

    # Test Huber loss
    print("\n=== Huber Loss ===")
    huber_loss = HuberLoss(delta=1.0)
    loss = huber_loss(predictions, targets)
    print(f"Loss: {loss.item():.4f}")
