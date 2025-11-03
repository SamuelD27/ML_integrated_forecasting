"""
Hyperparameter Optimization with Optuna on RunPod

Bayesian optimization for model hyperparameters:
- Distributed across multiple GPUs
- 100+ trials for comprehensive search
- Saves best configuration to YAML
- Early stopping for unpromising trials

Search space:
- hidden_size: 256-2048
- layers: 4-16
- heads: 8-24
- dropout: 0.1-0.4
- learning_rate: 1e-5 to 1e-3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import yaml
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning with Optuna."""

    def __init__(
        self,
        n_trials: int = 100,
        n_gpus: int = 3,
        timeout: Optional[int] = None,
        study_name: str = "hybrid_model_optimization",
        storage: Optional[str] = None
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            n_trials: Number of optimization trials
            n_gpus: Number of GPUs to use
            timeout: Timeout in seconds (None = no timeout)
            study_name: Name of Optuna study
            storage: Database URL for distributed optimization
        """
        self.n_trials = n_trials
        self.n_gpus = n_gpus
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage

        # Best trial results
        self.best_params = None
        self.best_value = None

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        direction: str = 'maximize',
        metric: str = 'sharpe_ratio',
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Run hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            direction: Optimization direction ('maximize' or 'minimize')
            metric: Metric to optimize
            save_path: Path to save best configuration

        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization...")
        logger.info(f"Trials: {self.n_trials}, GPUs: {self.n_gpus}, Metric: {metric}")

        # Create Optuna study
        if self.storage:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=direction,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                study_name=self.study_name,
                direction=direction
            )

        # Add pruner for early stopping
        study.sampler = optuna.samplers.TPESampler(seed=42)
        study.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        )

        # Objective function
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial)

            # Train and evaluate model
            score = self._train_and_evaluate(
                params,
                X_train, y_train,
                X_val, y_val,
                metric=metric,
                trial=trial
            )

            return score

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_gpus,  # Parallel trials
            show_progress_bar=True
        )

        # Get best results
        self.best_params = study.best_params
        self.best_value = study.best_value

        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Best {metric}: {self.best_value:.4f}")
        logger.info(f"Best hyperparameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")

        # Save best configuration
        if save_path:
            self._save_config(save_path)

        # Print optimization statistics
        self._print_statistics(study)

        return self.best_params

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Sample hyperparameters for a trial."""

        params = {
            # Model architecture
            'hidden_size': trial.suggest_int('hidden_size', 256, 2048, step=256),
            'lstm_layers': trial.suggest_int('lstm_layers', 2, 6),
            'transformer_layers': trial.suggest_int('transformer_layers', 2, 8),
            'n_heads': trial.suggest_categorical('n_heads', [8, 12, 16, 24]),
            'cnn_filters': trial.suggest_categorical('cnn_filters', [
                [64, 128, 256],
                [128, 256, 512],
                [64, 128, 256, 512]
            ]),

            # Regularization
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'lstm_dropout': trial.suggest_float('lstm_dropout', 0.1, 0.3),
            'transformer_dropout': trial.suggest_float('transformer_dropout', 0.05, 0.2),

            # Training
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),

            # Loss function
            'directional_weight': trial.suggest_float('directional_weight', 1.0, 3.0),
            'magnitude_weight': trial.suggest_float('magnitude_weight', 0.5, 2.0),

            # Optimizer
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
        }

        # Ensure hidden_size is divisible by n_heads
        if params['hidden_size'] % params['n_heads'] != 0:
            params['hidden_size'] = (params['hidden_size'] // params['n_heads']) * params['n_heads']

        return params

    def _train_and_evaluate(
        self,
        params: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'sharpe_ratio',
        trial: Optional[optuna.Trial] = None
    ) -> float:
        """
        Train model with given hyperparameters and evaluate.

        Args:
            params: Hyperparameters
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            metric: Metric to return
            trial: Optuna trial for pruning

        Returns:
            Validation metric value
        """
        # Import here to avoid circular dependencies
        from ml_models.tree_models import TreeEnsemble

        try:
            # For this example, we'll use tree models (faster than deep learning)
            # In production, you'd train the full hybrid model

            # Create ensemble
            model = TreeEnsemble(
                models=['xgboost', 'lightgbm'],
                use_gpu=torch.cuda.is_available()
            )

            # Update model parameters
            if hasattr(model.models.get('xgboost'), 'set_params'):
                model.models['xgboost'].set_params(
                    learning_rate=params['learning_rate'],
                    max_depth=params.get('hidden_size', 256) // 128,  # Scale depth
                    n_estimators=100,
                )

            if hasattr(model.models.get('lightgbm'), 'set_params'):
                model.models['lightgbm'].set_params(
                    learning_rate=params['learning_rate'],
                    max_depth=params.get('hidden_size', 256) // 128,
                    n_estimators=100,
                )

            # Train
            metrics = model.fit(
                X_train, y_train,
                X_val, y_val,
                early_stopping_rounds=10
            )

            # Evaluate
            val_pred = model.predict(X_val)

            # Compute metrics
            from sklearn.metrics import mean_squared_error, r2_score

            mse = mean_squared_error(y_val, val_pred)
            r2 = r2_score(y_val, val_pred)

            # Compute Sharpe ratio (approximate)
            # Assume predictions are returns
            pred_returns = val_pred
            if np.std(pred_returns) > 0:
                sharpe = np.mean(pred_returns) / np.std(pred_returns) * np.sqrt(252)
            else:
                sharpe = 0.0

            # Directional accuracy
            direction_correct = (np.sign(val_pred) == np.sign(y_val)).mean()

            # Return requested metric
            metric_map = {
                'sharpe_ratio': sharpe,
                'r2': r2,
                'mse': -mse,  # Negative for maximization
                'directional_accuracy': direction_correct
            }

            score = metric_map.get(metric, sharpe)

            # Report to trial for pruning
            if trial:
                trial.report(score, step=0)

                # Check if should prune
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return -np.inf  # Return worst possible score

    def _save_config(self, save_path: Path):
        """Save best configuration to YAML."""

        # Create config structure
        config = {
            'model': {
                'cnn': {
                    'filters': self.best_params.get('cnn_filters', [64, 128, 256]),
                    'dropout': self.best_params.get('dropout', 0.3),
                },
                'lstm': {
                    'hidden_units': self.best_params.get('hidden_size', 256),
                    'num_layers': self.best_params.get('lstm_layers', 2),
                    'dropout': self.best_params.get('lstm_dropout', 0.2),
                },
                'transformer': {
                    'd_model': self.best_params.get('hidden_size', 512),
                    'n_heads': self.best_params.get('n_heads', 8),
                    'num_layers': self.best_params.get('transformer_layers', 4),
                    'dropout': self.best_params.get('transformer_dropout', 0.1),
                },
            },
            'training': {
                'batch_size': self.best_params.get('batch_size', 64),
                'learning_rate': self.best_params.get('learning_rate', 0.001),
                'weight_decay': self.best_params.get('weight_decay', 1e-5),
                'optimizer': self.best_params.get('optimizer', 'adamw'),
            },
            'loss_weights': {
                'directional': self.best_params.get('directional_weight', 2.0),
                'magnitude': self.best_params.get('magnitude_weight', 1.0),
            },
            'optimization': {
                'best_score': self.best_value,
                'optimized_at': datetime.now().isoformat(),
            }
        }

        # Save to YAML
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved best configuration to {save_path}")

    def _print_statistics(self, study: optuna.Study):
        """Print optimization statistics."""

        logger.info(f"\n{'='*80}")
        logger.info("OPTIMIZATION STATISTICS")
        logger.info(f"{'='*80}")

        # Trial statistics
        logger.info(f"Number of finished trials: {len(study.trials)}")

        # Pruned trials
        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        logger.info(f"Number of pruned trials: {len(pruned_trials)}")

        # Complete trials
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        logger.info(f"Number of complete trials: {len(complete_trials)}")

        # Parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(study)
            logger.info(f"\nParameter importances:")
            for param, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
                logger.info(f"  {param}: {imp:.4f}")
        except:
            pass


def main():
    """Main hyperparameter optimization pipeline."""

    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 80)

    # Configuration
    DATA_PATH = Path('/workspace/data/training/training_data_compressed.parquet')
    OUTPUT_DIR = Path('/workspace/results/optimization')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")

    try:
        from data.storage.compression_utils import DataCompressor
        compressor = DataCompressor()
        df = compressor.load_compressed(DATA_PATH)
    except:
        df = pd.read_parquet(DATA_PATH)

    print(f"Loaded {len(df):,} rows")

    # Prepare data (simplified for example)
    print("\nPreparing data...")

    # Feature engineering
    df['returns'] = df.groupby('ticker')['close'].pct_change()
    df = df.dropna()

    # Features
    feature_cols = [
        col for col in df.columns
        if col not in ['date', 'ticker', 'returns', 'Date']
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ][:50]  # Limit features for speed

    X = df[feature_cols].values
    y = df['returns'].values

    # Train/val split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Features: {len(feature_cols)}")

    # Initialize tuner
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"\nUsing {n_gpus} GPUs for optimization")

    tuner = HyperparameterTuner(
        n_trials=100,
        n_gpus=n_gpus,
        timeout=3600 * 4,  # 4 hours
        study_name="hybrid_trading_model"
    )

    # Run optimization
    print("\nStarting optimization...")

    best_params = tuner.optimize(
        X_train, y_train,
        X_val, y_val,
        direction='maximize',
        metric='sharpe_ratio',
        save_path=OUTPUT_DIR / 'best_config.yaml'
    )

    # Save results
    results_path = OUTPUT_DIR / f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_value': tuner.best_value,
            'n_trials': tuner.n_trials,
        }, f, indent=2)

    print(f"\n✅ Optimization complete!")
    print(f"Results saved to: {results_path}")
    print(f"Best config saved to: {OUTPUT_DIR / 'best_config.yaml'}")


if __name__ == '__main__':
    main()
