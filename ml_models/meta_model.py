"""
Meta-Model for Trade Filtering
==============================
XGBoost/Random Forest model that predicts whether a primary forecast
will result in a profitable trade.

The meta-model acts as a secondary filter:
- Primary model generates directional forecasts
- Meta-model predicts probability of success
- Trades only executed if meta_prob >= threshold

Reference:
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

import pickle
import logging
from typing import Dict, Optional, List, Any, Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import ML libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not available")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available")


class MetaModel:
    """
    Meta-model for predicting trade success probability.

    Supports XGBoost or Random Forest classifiers.
    Predicts probability of a forecasted trade being profitable.

    Example:
        >>> meta_model = MetaModel(model_type='xgboost')
        >>> meta_model.fit(X_train, y_train)
        >>> proba = meta_model.predict_proba(features)
        >>> if proba >= 0.65:
        ...     execute_trade()
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize meta-model.

        Args:
            model_type: 'xgboost' or 'random_forest'
            config: Model hyperparameters
            model_path: Path to saved model
        """
        self.model_type = model_type.lower()
        self.config = config or {}
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False

        # Initialize model
        self._init_model()

        # Load if path provided
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _init_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'xgboost' and HAS_XGB:
            # XGBoost parameters
            xgb_params = self.config.get('xgboost', {})
            self.model = xgb.XGBClassifier(
                n_estimators=xgb_params.get('n_estimators', 100),
                max_depth=xgb_params.get('max_depth', 5),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                min_child_weight=xgb_params.get('min_child_weight', 3),
                subsample=xgb_params.get('subsample', 0.8),
                colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
            )

        elif self.model_type == 'random_forest' and HAS_SKLEARN:
            # Random Forest parameters
            rf_params = self.config.get('random_forest', {})
            self.model = RandomForestClassifier(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', 10),
                min_samples_split=rf_params.get('min_samples_split', 5),
                min_samples_leaf=rf_params.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1,
            )

        elif HAS_SKLEARN:
            # Fallback to Random Forest
            logger.warning(f"Model type {self.model_type} not available, using Random Forest")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )

        else:
            logger.error("No ML library available for meta-model")
            self.model = None

        # Initialize scaler
        if HAS_SKLEARN:
            self.scaler = StandardScaler()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Fit the meta-model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (1=success, 0=failure)
            feature_names: Optional list of feature names
            validation_split: Fraction for validation

        Returns:
            Dict with training metrics
        """
        if self.model is None:
            raise RuntimeError("No model available")

        if len(X) < 10:
            raise ValueError("Insufficient training data")

        # Store feature names
        self.feature_names = feature_names

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Split for validation
        n_val = int(len(X) * validation_split)
        if n_val > 0:
            X_train, X_val = X_scaled[:-n_val], X_scaled[-n_val:]
            y_train, y_val = y[:-n_val], y[-n_val:]
        else:
            X_train, X_val = X_scaled, X_scaled
            y_train, y_val = y, y

        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Calculate metrics
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]

        metrics = {
            'accuracy': float(accuracy_score(y_val, y_pred)),
            'precision': float(precision_score(y_val, y_pred, zero_division=0)),
            'recall': float(recall_score(y_val, y_pred, zero_division=0)),
            'f1': float(f1_score(y_val, y_pred, zero_division=0)),
        }

        try:
            metrics['auc'] = float(roc_auc_score(y_val, y_proba))
        except:
            metrics['auc'] = 0.5

        logger.info(f"Meta-model trained: accuracy={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")

        return metrics

    def predict_proba(
        self,
        features: Union[np.ndarray, 'MetaFeatures'],
    ) -> float:
        """
        Predict probability of trade success.

        Args:
            features: Feature array or MetaFeatures object

        Returns:
            Probability of success (0-1)
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, returning default probability")
            return 0.5

        # Convert MetaFeatures to array
        if hasattr(features, 'to_array'):
            features = features.to_array()

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale
        if self.scaler is not None:
            try:
                features = self.scaler.transform(features)
            except:
                pass  # Use unscaled if scaler fails

        # Predict
        try:
            proba = self.model.predict_proba(features)[:, 1]
            return float(proba[0])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for multiple samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            return np.full(len(X), 0.5)

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except:
                pass

        try:
            return self.model.predict_proba(X)[:, 1]
        except:
            return np.full(len(X), 0.5)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_fitted:
            return {}

        try:
            importances = self.model.feature_importances_
        except:
            return {}

        if self.feature_names is None:
            names = [f'feature_{i}' for i in range(len(importances))]
        else:
            names = self.feature_names

        return dict(zip(names, importances.tolist()))

    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        save_dict = {
            'model_type': self.model_type,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"Meta-model saved to {path}")

    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.model_type = save_dict['model_type']
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.feature_names = save_dict['feature_names']
        self.config = save_dict['config']
        self.is_fitted = save_dict['is_fitted']

        logger.info(f"Meta-model loaded from {path}")


def train_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'xgboost',
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> Tuple[MetaModel, Dict[str, float]]:
    """
    Train a meta-model.

    Args:
        X: Feature matrix
        y: Binary labels
        model_type: 'xgboost' or 'random_forest'
        config: Model configuration
        save_path: Path to save trained model

    Returns:
        Tuple of (trained MetaModel, metrics dict)
    """
    model = MetaModel(model_type=model_type, config=config)
    metrics = model.fit(X, y)

    if save_path:
        model.save(save_path)

    return model, metrics


def cross_validate_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'xgboost',
    config: Optional[Dict[str, Any]] = None,
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    Cross-validate meta-model.

    Args:
        X: Feature matrix
        y: Binary labels
        model_type: Model type
        config: Model configuration
        n_folds: Number of CV folds

    Returns:
        Dict with mean and std of CV scores
    """
    if not HAS_SKLEARN:
        return {'error': 'sklearn not available'}

    model = MetaModel(model_type=model_type, config=config)

    if model.model is None:
        return {'error': 'No model available'}

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    scores = cross_val_score(model.model, X, y, cv=n_folds, scoring='roc_auc')

    return {
        'mean_auc': float(np.mean(scores)),
        'std_auc': float(np.std(scores)),
        'min_auc': float(np.min(scores)),
        'max_auc': float(np.max(scores)),
    }


# Global singleton for reuse
_global_meta_model: Optional[MetaModel] = None


def get_meta_model(
    model_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    reload: bool = False,
) -> MetaModel:
    """
    Get global meta-model instance.

    Args:
        model_path: Path to model file
        config: Model configuration
        reload: Force reload

    Returns:
        MetaModel instance
    """
    global _global_meta_model

    if _global_meta_model is None or reload:
        _global_meta_model = MetaModel(
            model_type='xgboost',
            config=config,
            model_path=model_path,
        )

    return _global_meta_model
