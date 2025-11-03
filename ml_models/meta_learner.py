"""
Meta-Learner Stacking

Implements stacking ensemble that combines predictions from:
- Tree models (XGBoost, LightGBM, CatBoost)
- Deep learning models (CNN-LSTM-Transformer hybrid)
- Using a meta-learner (neural network or linear model)

Advantages over simple averaging:
- Learns optimal weighting for each base model
- Can capture non-linear interactions
- Adaptive to different market regimes
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import joblib

logger = logging.getLogger(__name__)


class MetaLearnerNN(nn.Module):
    """Neural network meta-learner."""

    def __init__(
        self,
        n_base_models: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize meta-learner neural network.

        Args:
            n_base_models: Number of base model predictions
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()

        self.n_base_models = n_base_models

        # Build network
        layers = []
        in_dim = n_base_models

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, base_predictions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            base_predictions: Predictions from base models (batch_size, n_base_models)

        Returns:
            Meta-predictions (batch_size, 1)
        """
        return self.network(base_predictions)


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""

    def __init__(
        self,
        base_models: Optional[Dict[str, any]] = None,
        meta_learner_type: str = 'nn',
        meta_learner_params: Optional[dict] = None,
        use_gpu: bool = False
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: Dictionary of base models {name: model}
            meta_learner_type: Type of meta-learner ('nn', 'linear', 'ridge')
            meta_learner_params: Parameters for meta-learner
            use_gpu: Whether to use GPU for meta-learner
        """
        self.base_models = base_models or {}
        self.meta_learner_type = meta_learner_type
        self.meta_learner_params = meta_learner_params or {}
        self.use_gpu = use_gpu and torch.cuda.is_available()

        self.meta_learner = None
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # Training history
        self.training_history = {
            'base_models': {},
            'meta_learner': {},
        }

    def add_base_model(self, name: str, model: any):
        """Add a base model to the ensemble."""
        self.base_models[name] = model
        logger.info(f"Added base model: {name}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_base_models: bool = True,
        cv_folds: int = 5,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train stacking ensemble.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            train_base_models: Whether to train base models
            cv_folds: Number of CV folds for out-of-fold predictions
            epochs: Epochs for meta-learner training
            batch_size: Batch size for meta-learner
            learning_rate: Learning rate for meta-learner

        Returns:
            Training metrics
        """
        logger.info(f"Training stacking ensemble with {len(self.base_models)} base models")

        # Step 1: Generate out-of-fold predictions from base models
        if train_base_models:
            logger.info("Training base models with cross-validation...")
            oof_predictions = self._train_base_models_cv(
                X_train, y_train, cv_folds
            )
        else:
            logger.info("Generating predictions from pre-trained base models...")
            oof_predictions = self._get_base_predictions(X_train)

        # Step 2: Train meta-learner on out-of-fold predictions
        logger.info("Training meta-learner...")
        self._train_meta_learner(
            oof_predictions,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Step 3: Evaluate on validation set
        metrics = {}

        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_metrics = self._compute_metrics(y_val, val_predictions)
            metrics['validation'] = val_metrics

            logger.info(f"Validation R²: {val_metrics['r2']:.4f}, "
                       f"RMSE: {val_metrics['rmse']:.4f}")

        return metrics

    def _train_base_models_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> np.ndarray:
        """
        Train base models with cross-validation and generate OOF predictions.

        Args:
            X: Features
            y: Targets
            cv_folds: Number of CV folds

        Returns:
            Out-of-fold predictions (n_samples, n_base_models)
        """
        from sklearn.model_selection import KFold

        n_samples = len(X)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))

        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"  Training {name} with {cv_folds}-fold CV...")

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_fold_train = X[train_idx]
                y_fold_train = y[train_idx]
                X_fold_val = X[val_idx]

                # Train model on this fold
                if hasattr(model, 'fit'):
                    model.fit(X_fold_train, y_fold_train)

                # Predict on validation fold
                if hasattr(model, 'predict'):
                    oof_predictions[val_idx, model_idx] = model.predict(X_fold_val).flatten()

            logger.info(f"  {name} CV complete")

        return oof_predictions

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models."""
        n_samples = len(X)
        n_models = len(self.base_models)
        predictions = np.zeros((n_samples, n_models))

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            if hasattr(model, 'predict'):
                predictions[:, model_idx] = model.predict(X).flatten()

        return predictions

    def _train_meta_learner(
        self,
        base_predictions: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ):
        """Train meta-learner on base model predictions."""

        n_base_models = base_predictions.shape[1]

        if self.meta_learner_type == 'nn':
            # Neural network meta-learner
            self.meta_learner = MetaLearnerNN(
                n_base_models=n_base_models,
                **self.meta_learner_params
            ).to(self.device)

            # Training
            optimizer = torch.optim.Adam(
                self.meta_learner.parameters(),
                lr=learning_rate
            )
            criterion = nn.MSELoss()

            # Convert to tensors
            X_tensor = torch.FloatTensor(base_predictions).to(self.device)
            y_tensor = torch.FloatTensor(targets).reshape(-1, 1).to(self.device)

            # Create dataset
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )

            # Training loop
            self.meta_learner.train()
            for epoch in range(epochs):
                epoch_loss = 0.0

                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()

                    predictions = self.meta_learner(batch_X)
                    loss = criterion(predictions, batch_y)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / len(dataloader)
                    logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            logger.info("Meta-learner training complete")

        elif self.meta_learner_type == 'linear':
            # Linear regression meta-learner
            from sklearn.linear_model import LinearRegression

            self.meta_learner = LinearRegression()
            self.meta_learner.fit(base_predictions, targets)

            logger.info("Linear meta-learner trained")

        elif self.meta_learner_type == 'ridge':
            # Ridge regression meta-learner
            from sklearn.linear_model import Ridge

            alpha = self.meta_learner_params.get('alpha', 1.0)
            self.meta_learner = Ridge(alpha=alpha)
            self.meta_learner.fit(base_predictions, targets)

            logger.info("Ridge meta-learner trained")

        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        # Get base model predictions
        base_predictions = self._get_base_predictions(X)

        # Meta-learner prediction
        if self.meta_learner_type == 'nn':
            self.meta_learner.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(base_predictions).to(self.device)
                predictions = self.meta_learner(X_tensor).cpu().numpy()

        else:
            predictions = self.meta_learner.predict(base_predictions)

        return predictions.flatten()

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get learned weights for each base model (for linear meta-learners).

        Returns:
            Dictionary of model weights
        """
        if self.meta_learner_type in ['linear', 'ridge']:
            if hasattr(self.meta_learner, 'coef_'):
                weights = {}
                for idx, (name, _) in enumerate(self.base_models.items()):
                    weights[name] = self.meta_learner.coef_[idx]
                return weights

        logger.warning("Model weights only available for linear meta-learners")
        return {}

    def save(self, save_dir: Union[str, Path]):
        """Save stacking ensemble."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save base models
        for name, model in self.base_models.items():
            model_path = save_dir / f"base_{name}.pkl"
            joblib.dump(model, model_path)

        # Save meta-learner
        if self.meta_learner_type == 'nn':
            meta_path = save_dir / "meta_learner.pth"
            torch.save(self.meta_learner.state_dict(), meta_path)
        else:
            meta_path = save_dir / "meta_learner.pkl"
            joblib.dump(self.meta_learner, meta_path)

        logger.info(f"Saved stacking ensemble to {save_dir}")

    def load(self, save_dir: Union[str, Path]):
        """Load stacking ensemble."""
        save_dir = Path(save_dir)

        # Load base models
        for name in list(self.base_models.keys()):
            model_path = save_dir / f"base_{name}.pkl"
            if model_path.exists():
                self.base_models[name] = joblib.load(model_path)

        # Load meta-learner
        if self.meta_learner_type == 'nn':
            meta_path = save_dir / "meta_learner.pth"
            if meta_path.exists():
                n_base_models = len(self.base_models)
                self.meta_learner = MetaLearnerNN(
                    n_base_models=n_base_models,
                    **self.meta_learner_params
                ).to(self.device)
                self.meta_learner.load_state_dict(torch.load(meta_path))
        else:
            meta_path = save_dir / "meta_learner.pkl"
            if meta_path.exists():
                self.meta_learner = joblib.load(meta_path)

        logger.info(f"Loaded stacking ensemble from {save_dir}")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute evaluation metrics."""
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


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create base models
    base_models = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'ridge': Ridge(alpha=1.0),
    }

    # Create stacking ensemble
    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_learner_type='nn',
        meta_learner_params={'hidden_dims': [32, 16]}
    )

    # Train
    print("\nTraining stacking ensemble...")
    metrics = ensemble.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        train_base_models=True,
        cv_folds=3,
        epochs=50
    )

    print("\nMetrics:")
    print(metrics)

    # Predict
    print("\nMaking predictions...")
    predictions = ensemble.predict(X_test)

    print(f"Test predictions shape: {predictions.shape}")
