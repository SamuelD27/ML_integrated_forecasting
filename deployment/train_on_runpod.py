#!/usr/bin/env python3

import os
import sys
import json
import time
import yaml
import boto3
import wandb
import optuna
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import tempfile
import zipfile
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RunPodConfig:
    """RunPod instance configuration"""
    instance_type: str = "RTX A6000"  # RTX 3090, RTX A6000, A100-40GB, A100-80GB
    volume_size: int = 100  # GB
    docker_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"
    timeout_hours: int = 6
    spot_instance: bool = True
    region: str = "US"

@dataclass
class TrainingConfig:
    """Training configuration"""
    config_path: str = "training/config.yaml"
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 4

@dataclass
class StorageConfig:
    """Cloud storage configuration"""
    s3_bucket: Optional[str] = None
    gcs_bucket: Optional[str] = None
    azure_container: Optional[str] = None
    local_backup: str = "checkpoints/runpod_backup"

class CloudStorageManager:
    """Manages model artifact storage across cloud providers"""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.s3_client = None
        self.gcs_client = None
        self.azure_client = None

        # Initialize cloud clients if credentials exist
        if config.s3_bucket and os.environ.get('AWS_ACCESS_KEY_ID'):
            self.s3_client = boto3.client('s3')
            logger.info(f"Initialized S3 client for bucket: {config.s3_bucket}")

    def upload_artifacts(self, local_path: Path, remote_prefix: str) -> Dict[str, str]:
        """Upload training artifacts to cloud storage"""
        uploaded = {}

        # Create archive of artifacts
        archive_path = Path(tempfile.gettempdir()) / f"artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(local_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(local_path)
                    zipf.write(file_path, arcname)

        # Upload to S3
        if self.s3_client and self.config.s3_bucket:
            try:
                s3_key = f"{remote_prefix}/{archive_path.name}"
                self.s3_client.upload_file(
                    str(archive_path),
                    self.config.s3_bucket,
                    s3_key
                )
                uploaded['s3'] = f"s3://{self.config.s3_bucket}/{s3_key}"
                logger.info(f"Uploaded to S3: {uploaded['s3']}")
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")

        # Local backup
        if self.config.local_backup:
            backup_dir = Path(self.config.local_backup)
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / archive_path.name
            archive_path.rename(backup_path)
            uploaded['local'] = str(backup_path)
            logger.info(f"Local backup saved: {backup_path}")

        return uploaded

class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization"""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.study = None

    def create_study(self, study_name: str, storage: Optional[str] = None):
        """Create Optuna study for hyperparameter search"""
        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize",  # Minimize validation loss
            storage=storage,  # e.g., "postgresql://user:password@localhost/optuna"
            load_if_exists=True
        )

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial"""
        config = self.base_config.copy()

        # Model architecture
        config['model']['cnn']['filters'] = [
            trial.suggest_int('cnn_filter_1', 32, 128, step=32),
            trial.suggest_int('cnn_filter_2', 64, 256, step=32),
            trial.suggest_int('cnn_filter_3', 128, 512, step=64),
        ]
        config['model']['lstm']['hidden_dim'] = trial.suggest_int('lstm_hidden', 128, 512, step=64)
        config['model']['lstm']['num_layers'] = trial.suggest_int('lstm_layers', 1, 4)
        config['model']['transformer']['d_model'] = trial.suggest_categorical(
            'transformer_d_model', [256, 512, 768, 1024]
        )
        config['model']['transformer']['n_heads'] = trial.suggest_categorical(
            'transformer_heads', [4, 8, 12, 16]
        )
        config['model']['transformer']['num_layers'] = trial.suggest_int('transformer_layers', 2, 8)

        # Training
        config['training']['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        config['training']['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        config['training']['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
        config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        # Optimizer
        config['training']['optimizer'] = trial.suggest_categorical(
            'optimizer', ['adam', 'adamw', 'sgd', 'rmsprop']
        )

        return config

class RunPodTrainer:
    """Main training orchestrator for RunPod"""

    def __init__(self,
                 runpod_config: RunPodConfig,
                 training_config: TrainingConfig,
                 storage_config: StorageConfig):
        self.runpod = runpod_config
        self.training = training_config
        self.storage_manager = CloudStorageManager(storage_config)
        self.optimizer = None

    def setup_wandb(self, project_name: str, run_name: Optional[str] = None):
        """Initialize Weights & Biases tracking"""
        run_name = run_name or f"hybrid_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                'runpod': asdict(self.runpod),
                'training': asdict(self.training),
            }
        )
        logger.info(f"W&B run initialized: {run_name}")

    def train_single_run(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training run"""
        # Save config to temporary file
        config_path = Path(tempfile.gettempdir()) / "temp_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Build training command
        cmd = [
            "python", "training/train_hybrid.py",
            "--config", str(config_path),
            "--epochs", str(self.training.epochs),
            "--batch-size", str(config.get('training', {}).get('batch_size', self.training.batch_size)),
            "--learning-rate", str(config.get('training', {}).get('learning_rate', self.training.learning_rate)),
            "--num-workers", str(self.training.num_workers),
        ]

        if self.training.use_mixed_precision:
            cmd.append("--mixed-precision")
        if self.training.gradient_checkpointing:
            cmd.append("--gradient-checkpointing")

        # Run training
        logger.info(f"Starting training with command: {' '.join(cmd)}")
        start_time = time.time()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            training_time = time.time() - start_time

            # Parse results from output
            metrics = self._parse_training_output(result.stdout)
            metrics['training_time_seconds'] = training_time

            # Log to W&B
            if wandb.run:
                wandb.log(metrics)

            logger.info(f"Training completed in {training_time/3600:.2f} hours")
            logger.info(f"Metrics: {metrics}")

            return metrics

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")
            raise

    def _parse_training_output(self, output: str) -> Dict[str, float]:
        """Parse metrics from training output"""
        metrics = {}

        # Look for specific patterns in output
        patterns = {
            'val_loss': r'Validation Loss: ([\d.]+)',
            'val_rmse': r'Validation RMSE: ([\d.]+)',
            'val_mae': r'Validation MAE: ([\d.]+)',
            'val_direction_acc': r'Direction Accuracy: ([\d.]+)',
            'val_sharpe': r'Sharpe Ratio: ([\d.]+)',
            'best_epoch': r'Best Epoch: (\d+)',
        }

        import re
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[key] = float(match.group(1))

        return metrics

    def run_hyperparameter_search(self,
                                 n_trials: int = 50,
                                 study_name: str = "hybrid_model_optimization"):
        """Run Optuna hyperparameter optimization"""
        # Load base config
        with open(self.training.config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        self.optimizer = HyperparameterOptimizer(base_config)
        self.optimizer.create_study(study_name)

        def objective(trial):
            # Suggest hyperparameters
            config = self.optimizer.suggest_hyperparameters(trial)

            # Train with suggested config
            try:
                metrics = self.train_single_run(config)

                # Return validation loss for optimization
                return metrics.get('val_loss', float('inf'))

            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf')

        # Run optimization
        logger.info(f"Starting hyperparameter search with {n_trials} trials")
        self.optimizer.study.optimize(objective, n_trials=n_trials)

        # Log best parameters
        best_params = self.optimizer.study.best_params
        best_value = self.optimizer.study.best_value

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation loss: {best_value}")

        # Save best config
        best_config = base_config.copy()
        for key, value in best_params.items():
            # Parse nested keys (e.g., 'cnn_filter_1' -> config['model']['cnn']['filters'][0])
            if key.startswith('cnn_filter_'):
                idx = int(key.split('_')[-1]) - 1
                best_config['model']['cnn']['filters'][idx] = value
            elif key == 'lstm_hidden':
                best_config['model']['lstm']['hidden_dim'] = value
            elif key == 'lstm_layers':
                best_config['model']['lstm']['num_layers'] = value
            elif key == 'transformer_d_model':
                best_config['model']['transformer']['d_model'] = value
            elif key == 'transformer_heads':
                best_config['model']['transformer']['n_heads'] = value
            elif key == 'transformer_layers':
                best_config['model']['transformer']['num_layers'] = value
            elif key in ['lr', 'learning_rate']:
                best_config['training']['learning_rate'] = value
            elif key == 'batch_size':
                best_config['training']['batch_size'] = value
            elif key == 'dropout':
                best_config['training']['dropout'] = value
            elif key == 'weight_decay':
                best_config['training']['weight_decay'] = value
            elif key == 'optimizer':
                best_config['training']['optimizer'] = value

        # Save best config
        best_config_path = Path('training/config_best.yaml')
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f)
        logger.info(f"Best config saved to {best_config_path}")

        return best_params, best_value

    def deploy_and_train(self):
        """Main deployment and training workflow"""
        logger.info("Starting RunPod deployment and training")

        # Setup W&B
        if os.environ.get('WANDB_API_KEY'):
            self.setup_wandb(
                project_name="hybrid-trading-model",
                run_name=f"runpod_{self.runpod.instance_type.replace(' ', '_')}"
            )

        # Train model
        with open(self.training.config_path, 'r') as f:
            config = yaml.safe_load(f)

        metrics = self.train_single_run(config)

        # Upload artifacts
        artifacts_path = Path('checkpoints')
        if artifacts_path.exists():
            remote_prefix = f"training_runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            uploaded = self.storage_manager.upload_artifacts(artifacts_path, remote_prefix)
            logger.info(f"Artifacts uploaded: {uploaded}")

            # Log artifact locations to W&B
            if wandb.run:
                wandb.log({'artifacts': uploaded})

        # Finish W&B run
        if wandb.run:
            wandb.finish()

        return metrics

def main():
    parser = argparse.ArgumentParser(description="Train hybrid model on RunPod GPU")

    # RunPod configuration
    parser.add_argument('--instance-type', type=str, default='RTX A6000',
                      choices=['RTX 3090', 'RTX A6000', 'A100-40GB', 'A100-80GB'],
                      help='RunPod GPU instance type')
    parser.add_argument('--volume-size', type=int, default=100,
                      help='Volume size in GB')
    parser.add_argument('--spot', action='store_true',
                      help='Use spot instances for cost savings')
    parser.add_argument('--timeout-hours', type=int, default=6,
                      help='Maximum training time in hours')

    # Training configuration
    parser.add_argument('--config', type=str, default='training/config.yaml',
                      help='Path to training config file')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--mixed-precision', action='store_true',
                      help='Use mixed precision training')

    # Hyperparameter optimization
    parser.add_argument('--optimize', action='store_true',
                      help='Run hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=50,
                      help='Number of Optuna trials')

    # Storage configuration
    parser.add_argument('--s3-bucket', type=str,
                      help='S3 bucket for artifact storage')
    parser.add_argument('--gcs-bucket', type=str,
                      help='GCS bucket for artifact storage')

    # Monitoring
    parser.add_argument('--wandb-project', type=str, default='hybrid-trading-model',
                      help='Weights & Biases project name')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable W&B tracking')

    args = parser.parse_args()

    # Create configurations
    runpod_config = RunPodConfig(
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        spot_instance=args.spot,
        timeout_hours=args.timeout_hours
    )

    training_config = TrainingConfig(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_mixed_precision=args.mixed_precision
    )

    storage_config = StorageConfig(
        s3_bucket=args.s3_bucket,
        gcs_bucket=args.gcs_bucket
    )

    # Disable W&B if requested
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    # Create trainer
    trainer = RunPodTrainer(runpod_config, training_config, storage_config)

    # Run training
    if args.optimize:
        # Run hyperparameter search
        best_params, best_value = trainer.run_hyperparameter_search(
            n_trials=args.n_trials,
            study_name=f"hybrid_optuna_{datetime.now().strftime('%Y%m%d')}"
        )
        print(f"\nBest parameters found:")
        print(json.dumps(best_params, indent=2))
        print(f"Best validation loss: {best_value:.4f}")
    else:
        # Run single training
        metrics = trainer.deploy_and_train()
        print(f"\nTraining completed successfully!")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    main()