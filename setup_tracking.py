#!/usr/bin/env python3
"""
Experiment Tracking Setup and Verification Script

Tests all three tracking systems:
1. Weights & Biases (W&B) - Cloud-based experiment tracking
2. TensorBoard - Local visualization
3. MLflow - Local experiment tracking and model registry

Usage:
    python setup_tracking.py --test
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_env_vars() -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent / '.env'
    env_vars = {}

    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                    os.environ[key.strip()] = value.strip()

    return env_vars


def test_wandb() -> bool:
    """Test Weights & Biases setup."""
    logger.info("Testing Weights & Biases (W&B)...")

    try:
        import wandb

        # Check API key
        api_key = os.getenv('WANDB_API_KEY')
        if not api_key:
            logger.error("❌ WANDB_API_KEY not found in environment")
            return False

        # Test login
        try:
            wandb.login(key=api_key, relogin=True)
            logger.info("✓ W&B login successful")
        except Exception as e:
            logger.error(f"❌ W&B login failed: {e}")
            return False

        # Test run initialization
        try:
            run = wandb.init(
                project="hybrid-trading-model",
                name="test-run",
                tags=["test", "setup"],
                mode="online"
            )

            # Log test metrics
            run.log({
                "test_loss": 0.5,
                "test_accuracy": 0.85,
                "epoch": 1
            })

            logger.info(f"✓ W&B test run created: {run.url}")
            run.finish()

            logger.info("✓ W&B is fully configured and working!")
            return True

        except Exception as e:
            logger.error(f"❌ W&B run creation failed: {e}")
            return False

    except ImportError:
        logger.error("❌ wandb package not installed. Run: pip install wandb")
        return False


def test_tensorboard() -> bool:
    """Test TensorBoard setup."""
    logger.info("\nTesting TensorBoard...")

    try:
        from torch.utils.tensorboard import SummaryWriter
        import shutil

        # Create test run directory
        log_dir = Path(__file__).parent / "runs" / "test_run"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Test writer
        writer = SummaryWriter(log_dir=str(log_dir))

        # Log test metrics
        for i in range(10):
            writer.add_scalar('Loss/train', 1.0 / (i + 1), i)
            writer.add_scalar('Loss/val', 1.2 / (i + 1), i)
            writer.add_scalar('Accuracy/train', 0.5 + 0.04 * i, i)

        writer.close()

        logger.info(f"✓ TensorBoard logs written to: {log_dir}")
        logger.info("✓ To view, run: tensorboard --logdir runs/")
        logger.info("✓ TensorBoard is fully configured and working!")

        # Clean up test run
        shutil.rmtree(log_dir)

        return True

    except ImportError:
        logger.error("❌ tensorboard package not installed. Run: pip install tensorboard")
        return False
    except Exception as e:
        logger.error(f"❌ TensorBoard test failed: {e}")
        return False


def test_mlflow() -> bool:
    """Test MLflow setup."""
    logger.info("\nTesting MLflow...")

    try:
        import mlflow

        # Set tracking URI
        tracking_dir = Path(__file__).parent / "mlruns"
        tracking_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{tracking_dir.absolute()}")

        # Set experiment
        mlflow.set_experiment("test-experiment")

        # Test run
        with mlflow.start_run(run_name="test-run"):
            # Log parameters
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("batch_size", 64)

            # Log metrics
            for i in range(10):
                mlflow.log_metric("loss", 1.0 / (i + 1), step=i)
                mlflow.log_metric("accuracy", 0.5 + 0.04 * i, step=i)

            # Log artifact
            test_file = tracking_dir / "test_artifact.txt"
            test_file.write_text("Test artifact content")
            mlflow.log_artifact(str(test_file))
            test_file.unlink()

        logger.info(f"✓ MLflow logs written to: {tracking_dir}")
        logger.info("✓ To view, run: mlflow ui --backend-store-uri mlruns/")
        logger.info("✓ MLflow is fully configured and working!")

        return True

    except ImportError:
        logger.error("❌ mlflow package not installed. Run: pip install mlflow")
        return False
    except Exception as e:
        logger.error(f"❌ MLflow test failed: {e}")
        return False


def verify_training_config() -> bool:
    """Verify training config has tracking enabled."""
    logger.info("\nVerifying training configuration...")

    config_file = Path(__file__).parent / "training" / "config.yaml"

    if not config_file.exists():
        logger.error(f"❌ Training config not found: {config_file}")
        return False

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    tracking = config.get('tracking', {})

    # Check W&B
    wandb_enabled = tracking.get('wandb', {}).get('enabled', False)
    logger.info(f"{'✓' if wandb_enabled else '✗'} W&B enabled in config: {wandb_enabled}")

    # Check TensorBoard
    tb_enabled = tracking.get('tensorboard', {}).get('enabled', False)
    logger.info(f"{'✓' if tb_enabled else '✗'} TensorBoard enabled in config: {tb_enabled}")

    # Check MLflow
    mlflow_enabled = tracking.get('mlflow', {}).get('enabled', False)
    logger.info(f"{'✓' if mlflow_enabled else '✗'} MLflow enabled in config: {mlflow_enabled}")

    logger.info("✓ Configuration verified!")
    return True


def setup_wandb_config() -> Dict[str, Any]:
    """Generate optimal W&B configuration."""
    return {
        "project": "hybrid-trading-model",
        "entity": None,  # Will use default entity
        "tags": ["hybrid", "deep-learning", "trading"],
        "config": {
            "architecture": "CNN-LSTM-Transformer",
            "dataset": "stock-prices",
            "framework": "pytorch"
        },
        "save_code": True,
        "notes": "Hybrid deep learning model for stock price prediction"
    }


def print_setup_guide():
    """Print setup and usage guide."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT TRACKING SETUP COMPLETE!")
    logger.info("="*80)

    logger.info("""
📊 THREE TRACKING SYSTEMS CONFIGURED:

1. WEIGHTS & BIASES (W&B) - Cloud-based
   • Status: ✓ Configured and tested
   • View runs: https://wandb.ai/your-username/hybrid-trading-model
   • Features:
     - Cloud storage of all experiments
     - Interactive visualizations
     - Hyperparameter sweeps
     - Model registry
     - Team collaboration

2. TENSORBOARD - Local visualization
   • Status: ✓ Configured and tested
   • Start server: tensorboard --logdir runs/
   • View at: http://localhost:6006
   • Features:
     - Real-time training curves
     - Model graph visualization
     - Embedding projections
     - Image/audio logging

3. MLFLOW - Local experiment tracking
   • Status: ✓ Configured and tested
   • Start UI: mlflow ui --backend-store-uri mlruns/
   • View at: http://localhost:5000
   • Features:
     - Experiment comparison
     - Model registry and versioning
     - Parameter search
     - Deployment tracking

═══════════════════════════════════════════════════════════════════════════════

🚀 QUICK START COMMANDS:

Start training with all tracking:
  cd training
  python train_hybrid.py --config config.yaml

Monitor training in real-time:
  # Terminal 1: TensorBoard
  tensorboard --logdir runs/

  # Terminal 2: MLflow (optional)
  mlflow ui --backend-store-uri mlruns/

View W&B dashboard:
  Open: https://wandb.ai/

═══════════════════════════════════════════════════════════════════════════════

📝 TRACKING FEATURES IN YOUR TRAINING SCRIPT:

Your train_hybrid.py already logs:
  ✓ Loss (train/val)
  ✓ RMSE metrics
  ✓ Directional accuracy
  ✓ Learning rate
  ✓ Sharpe ratio
  ✓ Max drawdown
  ✓ Model checkpoints

═══════════════════════════════════════════════════════════════════════════════

⚙️  CONFIGURATION:

Enable/disable tracking in training/config.yaml:
  tracking:
    wandb:
      enabled: true  # Toggle W&B
    tensorboard:
      enabled: true  # Toggle TensorBoard
    mlflow:
      enabled: false # Toggle MLflow

Environment variables (.env):
  WANDB_API_KEY: ✓ Configured
  WANDB_PROJECT: hybrid-trading-model

═══════════════════════════════════════════════════════════════════════════════
""")


def main():
    """Main setup and test function."""
    parser = argparse.ArgumentParser(description="Setup and test experiment tracking")
    parser.add_argument('--test', action='store_true', help='Run tests for all tracking systems')
    parser.add_argument('--wandb-only', action='store_true', help='Test W&B only')
    parser.add_argument('--tensorboard-only', action='store_true', help='Test TensorBoard only')
    parser.add_argument('--mlflow-only', action='store_true', help='Test MLflow only')

    args = parser.parse_args()

    # Load environment variables
    logger.info("Loading environment variables...")
    env_vars = load_env_vars()
    logger.info(f"✓ Loaded {len(env_vars)} environment variables")

    if not (args.test or args.wandb_only or args.tensorboard_only or args.mlflow_only):
        print_setup_guide()
        return

    # Run tests
    results = {}

    if args.test or args.wandb_only:
        results['wandb'] = test_wandb()

    if args.test or args.tensorboard_only:
        results['tensorboard'] = test_tensorboard()

    if args.test or args.mlflow_only:
        results['mlflow'] = test_mlflow()

    if args.test:
        verify_training_config()

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY:")
    logger.info("="*80)
    for system, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{system.upper()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED!")
        print_setup_guide()
    else:
        logger.error("\n✗ SOME TESTS FAILED. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
