"""
Uncertainty Quantification for ML Models
=========================================
Bayesian uncertainty estimation using Monte Carlo Dropout.

Key Concepts:
1. Epistemic Uncertainty (model uncertainty): Dropout at test time
2. Aleatoric Uncertainty (data uncertainty): Heteroscedastic regression
3. Prediction Intervals: Quantile-based confidence intervals
4. Calibration: Ensure predicted probabilities match true frequencies

Reference:
- Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation"
- Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning?"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""

    # Monte Carlo Dropout
    n_samples: int = 100  # Number of forward passes
    dropout_rate: float = 0.2

    # Prediction intervals
    confidence_levels: List[float] = None  # e.g., [0.68, 0.95, 0.99]

    # Calibration
    n_bins: int = 10

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.95, 0.99]


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer that stays active during inference.

    Unlike standard dropout, this layer remains active at test time
    to enable uncertainty estimation via stochastic predictions.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize MC Dropout.

        Args:
            p: Dropout probability
        """
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout regardless of training mode."""
        return F.dropout(x, p=self.p, training=True)


class BayesianLinear(nn.Module):
    """
    Linear layer with Monte Carlo Dropout for uncertainty.

    Replaces standard dropout with MC dropout for Bayesian inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = MCDropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(x))


class UncertaintyEstimator:
    """
    Estimate epistemic uncertainty using Monte Carlo Dropout.

    Example:
        >>> model = MyNeuralNetwork()
        >>> estimator = UncertaintyEstimator(model, n_samples=100)
        >>> predictions = estimator.predict_with_uncertainty(X_test)
        >>> print(predictions['mean'], predictions['std'])
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[UncertaintyConfig] = None
    ):
        """
        Initialize uncertainty estimator.

        Args:
            model: PyTorch model with dropout layers
            config: Uncertainty configuration
        """
        self.model = model
        self.config = config or UncertaintyConfig()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        X: Union[torch.Tensor, np.ndarray],
        return_samples: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.

        Args:
            X: Input features
            return_samples: Return all MC samples (memory intensive)

        Returns:
            Dictionary with:
                - mean: Mean prediction
                - std: Standard deviation (epistemic uncertainty)
                - lower_XX: Lower confidence bound (XX% confidence)
                - upper_XX: Upper confidence bound
                - samples: All MC samples (if return_samples=True)
        """
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)

        # Enable dropout at test time
        self.model.train()

        # Multiple forward passes with dropout
        samples = []
        for _ in range(self.config.n_samples):
            pred = self.model(X)
            if isinstance(pred, tuple):
                pred = pred[0]  # Handle models that return (pred, hidden)
            samples.append(pred.cpu().numpy())

        samples = np.array(samples)  # Shape: (n_samples, batch_size, output_dim)

        # Calculate statistics
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)

        result = {
            'mean': mean,
            'std': std,
            'epistemic_uncertainty': std,  # Alias
        }

        # Confidence intervals
        for confidence in self.config.confidence_levels:
            alpha = 1 - confidence
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            lower = np.quantile(samples, lower_q, axis=0)
            upper = np.quantile(samples, upper_q, axis=0)

            conf_str = f"{int(confidence * 100)}"
            result[f'lower_{conf_str}'] = lower
            result[f'upper_{conf_str}'] = upper

        if return_samples:
            result['samples'] = samples

        # Restore model to eval mode
        self.model.eval()

        return result

    def estimate_aleatoric_uncertainty(
        self,
        X: Union[torch.Tensor, np.ndarray],
        heteroscedastic_model: Optional[nn.Module] = None
    ) -> np.ndarray:
        """
        Estimate aleatoric (data) uncertainty.

        Args:
            X: Input features
            heteroscedastic_model: Optional model that predicts variance

        Returns:
            Aleatoric uncertainty (std) per sample
        """
        if heteroscedastic_model is not None:
            # Use dedicated variance model
            self.model.eval()
            heteroscedastic_model.eval()

            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float().to(self.device)

            with torch.no_grad():
                log_variance = heteroscedastic_model(X)
                variance = torch.exp(log_variance)

            return variance.cpu().numpy().squeeze()

        else:
            # Approximate from MC samples
            logger.warning("No heteroscedastic model provided, using MC approximation")
            result = self.predict_with_uncertainty(X, return_samples=True)

            # Aleatoric ≈ variance of samples (rough approximation)
            samples = result['samples']
            aleatoric = samples.var(axis=0)

            return aleatoric

    def total_uncertainty(
        self,
        X: Union[torch.Tensor, np.ndarray],
        heteroscedastic_model: Optional[nn.Module] = None
    ) -> Dict[str, np.ndarray]:
        """
        Decompose total uncertainty into epistemic and aleatoric.

        Total Uncertainty² = Epistemic² + Aleatoric²

        Args:
            X: Input features
            heteroscedastic_model: Optional model for aleatoric uncertainty

        Returns:
            Dictionary with epistemic, aleatoric, and total uncertainties
        """
        # Epistemic uncertainty (model uncertainty)
        result = self.predict_with_uncertainty(X)
        epistemic = result['std']

        # Aleatoric uncertainty (data uncertainty)
        aleatoric = self.estimate_aleatoric_uncertainty(X, heteroscedastic_model)

        # Total uncertainty
        total = np.sqrt(epistemic ** 2 + aleatoric ** 2)

        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total,
            'mean': result['mean']
        }


class CalibrationAnalyzer:
    """
    Analyze prediction calibration.

    Well-calibrated model: predicted probability matches true frequency.
    E.g., 80% confidence predictions should be correct 80% of the time.
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration analyzer.

        Args:
            n_bins: Number of bins for calibration curve
        """
        self.n_bins = n_bins

    def calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate calibration curve.

        Args:
            y_true: True values
            y_pred_mean: Predicted means
            y_pred_std: Predicted standard deviations

        Returns:
            Tuple of (expected_freq, observed_freq, bin_counts)
        """
        # Convert to standardized residuals
        residuals = (y_true - y_pred_mean) / (y_pred_std + 1e-8)

        # Calculate cumulative probabilities (assuming Gaussian)
        cdf = stats.norm.cdf(residuals)

        # Bin the probabilities
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(cdf, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Calculate observed frequencies
        expected_freq = []
        observed_freq = []
        bin_counts = []

        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                expected_freq.append((bins[i] + bins[i + 1]) / 2)
                observed_freq.append(cdf[mask].mean())
                bin_counts.append(mask.sum())
            else:
                expected_freq.append((bins[i] + bins[i + 1]) / 2)
                observed_freq.append(np.nan)
                bin_counts.append(0)

        return np.array(expected_freq), np.array(observed_freq), np.array(bin_counts)

    def expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures average calibration error across bins.

        Returns:
            ECE (lower is better, 0 = perfect calibration)
        """
        expected_freq, observed_freq, bin_counts = self.calibration_curve(
            y_true, y_pred_mean, y_pred_std
        )

        # Remove NaN bins
        mask = ~np.isnan(observed_freq)
        expected_freq = expected_freq[mask]
        observed_freq = observed_freq[mask]
        bin_counts = bin_counts[mask]

        # Weighted average of |expected - observed|
        weights = bin_counts / bin_counts.sum()
        ece = np.sum(weights * np.abs(expected_freq - observed_freq))

        return ece

    def plot_calibration(
        self,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        save_path: str = 'reports/calibration.png'
    ):
        """Plot calibration curve."""
        expected_freq, observed_freq, bin_counts = self.calibration_curve(
            y_true, y_pred_mean, y_pred_std
        )

        ece = self.expected_calibration_error(y_true, y_pred_mean, y_pred_std)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Calibration curve
        mask = ~np.isnan(observed_freq)
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.plot(expected_freq[mask], observed_freq[mask], 'o-', label='Model')
        ax1.set_xlabel('Expected Frequency')
        ax1.set_ylabel('Observed Frequency')
        ax1.set_title(f'Calibration Curve (ECE={ece:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bin counts
        ax2.bar(range(len(bin_counts)), bin_counts)
        ax2.set_xlabel('Bin')
        ax2.set_ylabel('Count')
        ax2.set_title('Samples per Bin')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration plot saved to {save_path}")


class UncertaintyBasedSelector:
    """
    Stock selector using uncertainty for risk management.

    Strategy: Prefer stocks with:
    - High predicted return (mean)
    - Low uncertainty (std)
    - Good calibration
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.3,
        return_weight: float = 0.7
    ):
        """
        Initialize uncertainty-based selector.

        Args:
            uncertainty_weight: Weight for uncertainty penalty
            return_weight: Weight for expected return
        """
        self.uncertainty_weight = uncertainty_weight
        self.return_weight = return_weight

    def score(
        self,
        pred_mean: np.ndarray,
        pred_std: np.ndarray
    ) -> np.ndarray:
        """
        Calculate uncertainty-adjusted score.

        Score = return_weight * mean - uncertainty_weight * std

        Args:
            pred_mean: Predicted returns
            pred_std: Prediction uncertainty

        Returns:
            Adjusted scores (higher = better)
        """
        # Normalize both metrics
        mean_norm = (pred_mean - pred_mean.mean()) / (pred_mean.std() + 1e-8)
        std_norm = (pred_std - pred_std.mean()) / (pred_std.std() + 1e-8)

        # Combined score (high return, low uncertainty)
        score = (
            self.return_weight * mean_norm -
            self.uncertainty_weight * std_norm
        )

        return score

    def select_top_k(
        self,
        tickers: List[str],
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float, float, float]]:
        """
        Select top K stocks based on uncertainty-adjusted score.

        Args:
            tickers: Stock tickers
            pred_mean: Predicted returns
            pred_std: Prediction uncertainty
            k: Number of stocks to select

        Returns:
            List of (ticker, score, pred_mean, pred_std) sorted by score
        """
        scores = self.score(pred_mean, pred_std)

        # Create ranking
        ranking = [
            (ticker, score, mean, std)
            for ticker, score, mean, std in zip(tickers, scores, pred_mean, pred_std)
        ]

        # Sort by score descending
        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking[:k]


def add_uncertainty_to_predictions(
    model: nn.Module,
    X: Union[torch.Tensor, np.ndarray],
    n_samples: int = 100,
    confidence_levels: List[float] = [0.68, 0.95, 0.99]
) -> pd.DataFrame:
    """
    Convenience function to add uncertainty estimates to predictions.

    Args:
        model: PyTorch model with dropout
        X: Input features
        n_samples: Number of MC samples
        confidence_levels: Confidence levels for intervals

    Returns:
        DataFrame with mean, std, and confidence intervals

    Example:
        >>> predictions_df = add_uncertainty_to_predictions(model, X_test)
        >>> print(predictions_df[['mean', 'std', 'lower_95', 'upper_95']])
    """
    config = UncertaintyConfig(
        n_samples=n_samples,
        confidence_levels=confidence_levels
    )

    estimator = UncertaintyEstimator(model, config)
    result = estimator.predict_with_uncertainty(X)

    # Convert to DataFrame
    df = pd.DataFrame()
    df['mean'] = result['mean'].squeeze()
    df['std'] = result['std'].squeeze()

    for confidence in confidence_levels:
        conf_str = f"{int(confidence * 100)}"
        df[f'lower_{conf_str}'] = result[f'lower_{conf_str}'].squeeze()
        df[f'upper_{conf_str}'] = result[f'upper_{conf_str}'].squeeze()

    return df


if __name__ == '__main__':
    # Example usage
    print("Uncertainty Quantification Example")
    print("=" * 60)

    # Create simple model with MC Dropout
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64, dropout_rate=0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.dropout1 = MCDropout(p=dropout_rate)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.dropout2 = MCDropout(p=dropout_rate)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    # Create model and dummy data
    model = SimpleModel(input_dim=10, dropout_rate=0.2)
    X_test = np.random.randn(100, 10).astype(np.float32)

    print("\nGenerating predictions with uncertainty...")
    estimator = UncertaintyEstimator(model, UncertaintyConfig(n_samples=50))
    result = estimator.predict_with_uncertainty(X_test)

    print(f"Mean shape: {result['mean'].shape}")
    print(f"Std shape: {result['std'].shape}")
    print(f"Mean uncertainty: {result['std'].mean():.4f}")

    # Calibration analysis
    print("\nCalibration Analysis:")
    y_true = np.random.randn(100)  # Dummy true values
    analyzer = CalibrationAnalyzer(n_bins=10)
    ece = analyzer.expected_calibration_error(
        y_true,
        result['mean'].squeeze(),
        result['std'].squeeze()
    )
    print(f"Expected Calibration Error: {ece:.4f}")

    # Uncertainty-based selection
    print("\nUncertainty-Based Stock Selection:")
    tickers = [f'STOCK{i}' for i in range(100)]
    selector = UncertaintyBasedSelector(uncertainty_weight=0.3, return_weight=0.7)
    top_10 = selector.select_top_k(
        tickers,
        result['mean'].squeeze(),
        result['std'].squeeze(),
        k=10
    )

    print("\nTop 10 stocks (high return, low uncertainty):")
    for i, (ticker, score, mean, std) in enumerate(top_10, 1):
        print(f"{i:2d}. {ticker}: score={score:.3f}, mean={mean:.4f}, std={std:.4f}")

    print("\n✓ Uncertainty module ready")
