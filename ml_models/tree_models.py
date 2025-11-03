"""
Tree-Based Models Ensemble

Implements gradient boosting models for trading:
- XGBoost: Extreme Gradient Boosting
- LightGBM: Light Gradient Boosting Machine
- CatBoost: Categorical Boosting

Features:
- Unified interface for all tree models
- Hyperparameter optimization
- Feature importance tracking
- Early stopping
- GPU acceleration support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import json
import joblib

logger = logging.getLogger(__name__)


class TreeEnsemble:
    """Ensemble of gradient boosting models."""

    def __init__(
        self,
        models: Optional[List[str]] = None,
        use_gpu: bool = False,
        random_state: int = 42
    ):
        """
        Initialize tree ensemble.

        Args:
            models: List of models to include ['xgboost', 'lightgbm', 'catboost']
            use_gpu: Whether to use GPU acceleration
            random_state: Random seed for reproducibility
        """
        self.models_to_use = models or ['xgboost', 'lightgbm', 'catboost']
        self.use_gpu = use_gpu
        self.random_state = random_state

        self.models = {}
        self.feature_importances = {}
        self.training_history = {}

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize gradient boosting models."""

        # XGBoost
        if 'xgboost' in self.models_to_use:
            try:
                import xgboost as xgb

                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'n_jobs': -1,
                }

                if self.use_gpu:
                    params['tree_method'] = 'gpu_hist'
                    params['gpu_id'] = 0

                self.models['xgboost'] = xgb.XGBRegressor(**params)
                logger.info("XGBoost initialized")

            except ImportError:
                logger.warning("XGBoost not installed, skipping")

        # LightGBM
        if 'lightgbm' in self.models_to_use:
            try:
                import lightgbm as lgb

                params = {
                    'objective': 'regression',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'verbose': -1,
                }

                if self.use_gpu:
                    params['device'] = 'gpu'
                    params['gpu_platform_id'] = 0
                    params['gpu_device_id'] = 0

                self.models['lightgbm'] = lgb.LGBMRegressor(**params)
                logger.info("LightGBM initialized")

            except ImportError:
                logger.warning("LightGBM not installed, skipping")

        # CatBoost
        if 'catboost' in self.models_to_use:
            try:
                from catboost import CatBoostRegressor

                params = {
                    'iterations': 100,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_state': self.random_state,
                    'verbose': False,
                }

                if self.use_gpu:
                    params['task_type'] = 'GPU'
                    params['devices'] = '0'

                self.models['catboost'] = CatBoostRegressor(**params)
                logger.info("CatBoost initialized")

            except ImportError:
                logger.warning("CatBoost not installed, skipping")

        if not self.models:
            raise ValueError("No gradient boosting libraries installed!")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 10
    ) -> Dict[str, Any]:
        """
        Train all models in ensemble.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names for importance tracking
            early_stopping_rounds: Early stopping patience

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {len(self.models)} tree models...")

        metrics = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            try:
                # Prepare validation set
                if X_val is not None and y_val is not None:
                    eval_set = [(X_val, y_val)]

                    # XGBoost
                    if name == 'xgboost':
                        model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose=False
                        )

                    # LightGBM
                    elif name == 'lightgbm':
                        model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            callbacks=[
                                model.__class__.early_stopping(early_stopping_rounds),
                                model.__class__.log_evaluation(0)
                            ]
                        )

                    # CatBoost
                    elif name == 'catboost':
                        model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose=False
                        )

                else:
                    # No validation set
                    model.fit(X_train, y_train)

                # Get feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_

                    if feature_names:
                        self.feature_importances[name] = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                    else:
                        self.feature_importances[name] = importances

                # Get training metrics
                train_pred = model.predict(X_train)
                train_score = self._compute_metrics(y_train, train_pred)

                if X_val is not None:
                    val_pred = model.predict(X_val)
                    val_score = self._compute_metrics(y_val, val_pred)
                else:
                    val_score = {}

                metrics[name] = {
                    'train': train_score,
                    'val': val_score,
                }

                logger.info(f"{name} trained - Train R²: {train_score.get('r2', 0):.4f}, "
                           f"Val R²: {val_score.get('r2', 0):.4f}")

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                metrics[name] = {'error': str(e)}

        return metrics

    def predict(
        self,
        X: np.ndarray,
        ensemble_method: str = 'mean'
    ) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Features
            ensemble_method: How to combine predictions ('mean', 'median', 'weighted')

        Returns:
            Ensemble predictions
        """
        predictions = {}

        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.error(f"Failed to predict with {name}: {e}")

        if not predictions:
            raise ValueError("No models available for prediction!")

        # Combine predictions
        pred_array = np.array(list(predictions.values()))

        if ensemble_method == 'mean':
            return np.mean(pred_array, axis=0)
        elif ensemble_method == 'median':
            return np.median(pred_array, axis=0)
        elif ensemble_method == 'weighted':
            # Weight by validation performance (if available)
            weights = self._get_model_weights()
            return np.average(pred_array, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    def predict_proba(
        self,
        X: np.ndarray,
        bins: int = 3
    ) -> np.ndarray:
        """
        Generate probabilistic predictions (discretized into bins).

        Args:
            X: Features
            bins: Number of bins for discretization

        Returns:
            Probability distributions
        """
        # Get continuous predictions
        predictions = self.predict(X)

        # Discretize into bins (e.g., down, neutral, up)
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(predictions, percentiles)

        # Create probability distribution
        probs = np.zeros((len(predictions), bins))

        for i, pred in enumerate(predictions):
            # Find which bin this prediction falls into
            bin_idx = np.digitize(pred, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, bins - 1)

            # Assign probability (could be softened with uncertainty estimate)
            probs[i, bin_idx] = 1.0

        return probs

    def get_feature_importance(
        self,
        top_n: Optional[int] = None,
        aggregate: str = 'mean'
    ) -> pd.DataFrame:
        """
        Get aggregated feature importances across models.

        Args:
            top_n: Number of top features to return
            aggregate: How to aggregate ('mean', 'max', 'min')

        Returns:
            DataFrame with feature importances
        """
        if not self.feature_importances:
            logger.warning("No feature importances available")
            return pd.DataFrame()

        # Collect all importances
        all_importances = []

        for name, importance_df in self.feature_importances.items():
            if isinstance(importance_df, pd.DataFrame):
                df = importance_df.copy()
                df['model'] = name
                all_importances.append(df)

        if not all_importances:
            return pd.DataFrame()

        # Combine
        combined = pd.concat(all_importances, ignore_index=True)

        # Aggregate
        if aggregate == 'mean':
            result = combined.groupby('feature')['importance'].mean()
        elif aggregate == 'max':
            result = combined.groupby('feature')['importance'].max()
        elif aggregate == 'min':
            result = combined.groupby('feature')['importance'].min()
        else:
            result = combined.groupby('feature')['importance'].mean()

        result = result.sort_values(ascending=False)

        if top_n:
            result = result.head(top_n)

        return pd.DataFrame({
            'feature': result.index,
            'importance': result.values
        })

    def save(self, save_dir: Union[str, Path]):
        """Save ensemble models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model_path = save_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")

        # Save feature importances
        for name, importances in self.feature_importances.items():
            if isinstance(importances, pd.DataFrame):
                importances_path = save_dir / f"{name}_importances.csv"
                importances.to_csv(importances_path, index=False)

        logger.info(f"Saved tree ensemble to {save_dir}")

    def load(self, save_dir: Union[str, Path]):
        """Load ensemble models."""
        save_dir = Path(save_dir)

        for name in self.models_to_use:
            model_path = save_dir / f"{name}_model.pkl"

            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} from {model_path}")

        logger.info(f"Loaded tree ensemble from {save_dir}")

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }

        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)

        return metrics

    def _get_model_weights(self) -> List[float]:
        """Get weights for ensemble based on validation performance."""
        if not self.training_history:
            # Equal weights
            return [1.0 / len(self.models)] * len(self.models)

        # Weight by R² score (if available)
        weights = []
        for name in self.models.keys():
            if name in self.training_history:
                val_metrics = self.training_history[name].get('val', {})
                r2 = val_metrics.get('r2', 0.0)
                weights.append(max(0, r2))  # Clip to non-negative

        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(self.models)] * len(self.models)

        return weights


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    # Train/test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create ensemble
    ensemble = TreeEnsemble(use_gpu=False)

    # Train
    print("\nTraining ensemble...")
    metrics = ensemble.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        early_stopping_rounds=10
    )

    print("\nTraining metrics:")
    for model, model_metrics in metrics.items():
        print(f"\n{model}:")
        for split, split_metrics in model_metrics.items():
            print(f"  {split}: {split_metrics}")

    # Predict
    print("\nMaking predictions...")
    predictions = ensemble.predict(X_test)

    # Feature importance
    print("\nTop 10 features:")
    importance = ensemble.get_feature_importance(top_n=10)
    print(importance)
