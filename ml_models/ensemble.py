"""
Model Ensemble System
=====================
Combine multiple models for improved predictions.

Ensemble Methods:
1. Simple Averaging: Equal weight to all models
2. Weighted Averaging: Weight by validation performance
3. Stacking: Meta-model learns to combine predictions
4. Boosting: Sequential ensemble (already in LightGBM/XGBoost)

Models to Ensemble:
- Temporal Fusion Transformer (TFT)
- Hybrid CNN-LSTM-Transformer
- LightGBM
- XGBoost
- Ridge Regression (baseline)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble."""

    # Ensemble method
    method: str = 'weighted'  # 'simple', 'weighted', 'stacking', 'rank'

    # Weights (for weighted averaging)
    weights: Optional[Dict[str, float]] = None

    # Stacking
    meta_model: str = 'ridge'  # 'ridge', 'lasso', 'lgb'
    use_original_features: bool = True  # Include original features in stacking

    # Diversity
    min_correlation: float = 0.0  # Minimum correlation to include model
    max_correlation: float = 0.95  # Maximum correlation (remove too similar models)

    # Uncertainty
    estimate_uncertainty: bool = True
    n_bootstrap: int = 100  # Bootstrap for uncertainty estimation


class ModelEnsemble:
    """
    Ensemble of multiple models with various combination strategies.

    Example:
        >>> ensemble = ModelEnsemble(config)
        >>> ensemble.add_model('tft', tft_model)
        >>> ensemble.add_model('hybrid', hybrid_model)
        >>> ensemble.add_model('lgb', lgb_model)
        >>> ensemble.fit(X_train, y_train, X_val, y_val)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize ensemble.

        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.models = {}
        self.model_predictions = {}
        self.model_weights = {}
        self.meta_model = None
        self.is_fitted = False

    def add_model(
        self,
        name: str,
        model: Union[nn.Module, object],
        weight: Optional[float] = None
    ):
        """
        Add model to ensemble.

        Args:
            name: Model name
            model: Model instance
            weight: Optional weight for weighted averaging
        """
        self.models[name] = model

        if weight is not None:
            self.model_weights[name] = weight

        logger.info(f"Added model '{name}' to ensemble")

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Fit ensemble (calculate weights or train meta-model).

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (required for weighted/stacking)
            y_val: Validation targets (required for weighted/stacking)
        """
        if len(self.models) == 0:
            raise ValueError("No models added to ensemble")

        # Generate predictions from all models
        logger.info("Generating predictions from base models...")

        train_preds = {}
        val_preds = {}

        for name, model in self.models.items():
            # Train predictions
            train_preds[name] = self._predict_single(model, X_train)

            # Validation predictions
            if X_val is not None:
                val_preds[name] = self._predict_single(model, X_val)

        self.model_predictions['train'] = train_preds
        if X_val is not None:
            self.model_predictions['val'] = val_preds

        # Fit ensemble based on method
        if self.config.method == 'simple':
            self._fit_simple_averaging()
        elif self.config.method == 'weighted':
            if X_val is None or y_val is None:
                raise ValueError("Weighted averaging requires validation data")
            self._fit_weighted_averaging(val_preds, y_val)
        elif self.config.method == 'stacking':
            if X_val is None or y_val is None:
                raise ValueError("Stacking requires validation data")
            self._fit_stacking(train_preds, y_train, X_train)
        elif self.config.method == 'rank':
            self._fit_rank_averaging()
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.method}")

        self.is_fitted = True
        logger.info(f"Ensemble fitted using '{self.config.method}' method")

    def _fit_simple_averaging(self):
        """Simple averaging: equal weights."""
        n_models = len(self.models)
        self.model_weights = {name: 1.0 / n_models for name in self.models.keys()}

        logger.info("Using simple averaging (equal weights)")

    def _fit_weighted_averaging(self, val_preds: Dict[str, np.ndarray], y_val: np.ndarray):
        """
        Weighted averaging: weight by validation performance.

        Weight proportional to inverse MSE.
        """
        # Calculate MSE for each model
        mse_scores = {}
        for name, preds in val_preds.items():
            mse = np.mean((preds - y_val) ** 2)
            mse_scores[name] = mse

        # Calculate weights (inverse MSE)
        inv_mse = {name: 1.0 / (mse + 1e-8) for name, mse in mse_scores.items()}
        total = sum(inv_mse.values())
        self.model_weights = {name: weight / total for name, weight in inv_mse.items()}

        logger.info("Model weights (by validation performance):")
        for name, weight in self.model_weights.items():
            mse = mse_scores[name]
            logger.info(f"  {name}: weight={weight:.4f}, val_mse={mse:.6f}")

    def _fit_stacking(
        self,
        train_preds: Dict[str, np.ndarray],
        y_train: np.ndarray,
        X_train: Optional[np.ndarray] = None
    ):
        """
        Stacking: train meta-model on base model predictions.

        Args:
            train_preds: Predictions from base models
            y_train: Training targets
            X_train: Original features (optional, for feature stacking)
        """
        # Create meta-features (base model predictions)
        meta_X = np.column_stack([train_preds[name] for name in sorted(train_preds.keys())])

        # Optionally include original features
        if self.config.use_original_features and X_train is not None:
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            meta_X = np.column_stack([meta_X, X_train])

        # Train meta-model
        if self.config.meta_model == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        elif self.config.meta_model == 'lasso':
            self.meta_model = Lasso(alpha=0.01)
        elif self.config.meta_model == 'lgb':
            self.meta_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown meta-model: {self.config.meta_model}")

        self.meta_model.fit(meta_X, y_train)

        logger.info(f"Stacking meta-model trained: {self.config.meta_model}")

        # Log feature importance if available
        if hasattr(self.meta_model, 'coef_'):
            model_names = sorted(train_preds.keys())
            for i, name in enumerate(model_names):
                logger.info(f"  {name}: coef={self.meta_model.coef_[i]:.4f}")

    def _fit_rank_averaging(self):
        """Rank averaging: convert to ranks then average."""
        # Equal weights (rank conversion done in predict)
        n_models = len(self.models)
        self.model_weights = {name: 1.0 / n_models for name in self.models.keys()}

        logger.info("Using rank averaging")

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate ensemble predictions.

        Args:
            X: Input features
            return_uncertainty: Return prediction uncertainty

        Returns:
            Predictions (or tuple of predictions and uncertainty)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all models
        preds = {}
        for name, model in self.models.items():
            preds[name] = self._predict_single(model, X)

        # Combine predictions
        if self.config.method == 'stacking':
            ensemble_pred = self._predict_stacking(preds, X)
        elif self.config.method == 'rank':
            ensemble_pred = self._predict_rank_averaging(preds)
        else:  # simple or weighted
            ensemble_pred = self._predict_weighted_averaging(preds)

        if return_uncertainty and self.config.estimate_uncertainty:
            uncertainty = self._estimate_uncertainty(preds)
            return ensemble_pred, uncertainty
        else:
            return ensemble_pred

    def _predict_weighted_averaging(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average of predictions."""
        weighted_sum = np.zeros_like(next(iter(preds.values())))

        for name, pred in preds.items():
            weight = self.model_weights.get(name, 1.0 / len(preds))
            weighted_sum += weight * pred

        return weighted_sum

    def _predict_rank_averaging(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Rank-based averaging."""
        # Convert to ranks
        rank_preds = {}
        for name, pred in preds.items():
            # Use scipy rankdata for proper handling
            from scipy.stats import rankdata
            rank_preds[name] = rankdata(pred, method='average')

        # Average ranks
        avg_ranks = self._predict_weighted_averaging(rank_preds)

        # Convert back to original scale (approximate)
        # Use percentile mapping
        sorted_pred = np.sort(next(iter(preds.values())))
        percentiles = avg_ranks / len(avg_ranks) * 100
        result = np.percentile(sorted_pred, percentiles)

        return result

    def _predict_stacking(
        self,
        preds: Dict[str, np.ndarray],
        X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Stacking predictions."""
        # Create meta-features
        meta_X = np.column_stack([preds[name] for name in sorted(preds.keys())])

        # Include original features if used during training
        if self.config.use_original_features and X is not None:
            if isinstance(X, pd.DataFrame):
                X = X.values
            meta_X = np.column_stack([meta_X, X])

        # Predict with meta-model
        return self.meta_model.predict(meta_X)

    def _predict_single(
        self,
        model: Union[nn.Module, object],
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Generate predictions from single model.

        Handles PyTorch and sklearn-like models.
        """
        # PyTorch model
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                if isinstance(X, pd.DataFrame):
                    X = X.values
                X_tensor = torch.from_numpy(X).float()
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                pred = model(X_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                return pred.cpu().numpy().squeeze()

        # Sklearn-like model
        else:
            return model.predict(X).squeeze()

    def _estimate_uncertainty(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimate prediction uncertainty from model disagreement.

        Args:
            preds: Predictions from each model

        Returns:
            Standard deviation across models
        """
        # Stack predictions
        all_preds = np.column_stack([preds[name] for name in sorted(preds.keys())])

        # Standard deviation across models
        uncertainty = all_preds.std(axis=1)

        return uncertainty

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.

        Args:
            X: Features
            y: Targets

        Returns:
            Dictionary with metrics
        """
        pred = self.predict(X)

        # Calculate metrics
        mse = np.mean((pred - y) ** 2)
        mae = np.mean(np.abs(pred - y))
        rmse = np.sqrt(mse)

        # R²
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Information Coefficient (Spearman correlation)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(pred, y)

        # Directional accuracy
        direction_correct = np.sign(pred) == np.sign(y)
        directional_accuracy = direction_correct.mean()

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'ic': ic,
            'directional_accuracy': directional_accuracy
        }


def create_ensemble(
    models: Dict[str, object],
    method: str = 'weighted',
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> ModelEnsemble:
    """
    Convenience function to create and fit ensemble.

    Args:
        models: Dictionary of {name: model}
        method: Ensemble method ('simple', 'weighted', 'stacking', 'rank')
        X_val: Validation features (required for weighted/stacking)
        y_val: Validation targets (required for weighted/stacking)

    Returns:
        Fitted ModelEnsemble

    Example:
        >>> models = {
        ...     'lgb': lgb_model,
        ...     'xgb': xgb_model,
        ...     'ridge': ridge_model
        ... }
        >>> ensemble = create_ensemble(models, method='weighted', X_val=X_val, y_val=y_val)
        >>> predictions = ensemble.predict(X_test)
    """
    config = EnsembleConfig(method=method)
    ensemble = ModelEnsemble(config)

    # Add models
    for name, model in models.items():
        ensemble.add_model(name, model)

    # Fit ensemble
    if method in ['weighted', 'stacking']:
        if X_val is None or y_val is None:
            raise ValueError(f"{method} method requires validation data")
        ensemble.fit(None, None, X_val, y_val)
    else:
        ensemble.fit(None, None)

    return ensemble


if __name__ == '__main__':
    # Example usage
    print("Model Ensemble Example")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X_train = np.random.randn(n_samples, n_features)
    X_val = np.random.randn(200, n_features)
    X_test = np.random.randn(200, n_features)

    # True function with noise
    y_train = X_train[:, 0] * 2 + X_train[:, 1] - X_train[:, 2] * 0.5 + np.random.randn(n_samples) * 0.5
    y_val = X_val[:, 0] * 2 + X_val[:, 1] - X_val[:, 2] * 0.5 + np.random.randn(200) * 0.5
    y_test = X_test[:, 0] * 2 + X_test[:, 1] - X_test[:, 2] * 0.5 + np.random.randn(200) * 0.5

    # Train base models
    print("\nTraining base models...")

    model1 = Ridge(alpha=1.0)
    model1.fit(X_train, y_train)

    model2 = lgb.LGBMRegressor(n_estimators=100, verbose=-1)
    model2.fit(X_train, y_train)

    model3 = xgb.XGBRegressor(n_estimators=100, verbosity=0)
    model3.fit(X_train, y_train)

    # Evaluate individual models
    print("\nIndividual Model Performance:")
    for name, model in [('Ridge', model1), ('LightGBM', model2), ('XGBoost', model3)]:
        pred = model.predict(X_val)
        mse = np.mean((pred - y_val) ** 2)
        print(f"  {name}: val_mse={mse:.6f}")

    # Create ensemble
    print("\nCreating ensemble...")

    models = {
        'ridge': model1,
        'lgb': model2,
        'xgb': model3
    }

    # Test different ensemble methods
    for method in ['simple', 'weighted', 'stacking']:
        print(f"\n{method.upper()} Ensemble:")

        config = EnsembleConfig(method=method)
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)

        # Evaluate
        metrics = ensemble.evaluate(X_test, y_test)
        print(f"  Test MSE: {metrics['mse']:.6f}")
        print(f"  Test R²: {metrics['r2']:.4f}")
        print(f"  IC: {metrics['ic']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")

    print("\n✓ Ensemble module ready")
