#!/usr/bin/env python3
"""
Pre-Flight Validation Script for RunPod Training
=================================================
Validates all dependencies, imports, and configurations before launching RunPod.

Run this script to ensure everything is ready:
    python3 validate_training_setup.py --full
"""

import sys
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# ANSI color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class ValidationResult:
    """Container for validation results"""
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.failures = []
        self.critical_failures = []

    def add_pass(self, message: str):
        self.passed.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def add_failure(self, message: str, critical: bool = False):
        if critical:
            self.critical_failures.append(message)
        else:
            self.failures.append(message)

    def has_critical_failures(self) -> bool:
        return len(self.critical_failures) > 0

    def print_summary(self):
        """Print colored summary of results"""
        print("\n" + "=" * 80)
        print(f"{BOLD}VALIDATION SUMMARY{RESET}")
        print("=" * 80)

        if self.passed:
            print(f"\n{GREEN}✓ PASSED ({len(self.passed)}){RESET}")
            for msg in self.passed:
                print(f"  {GREEN}✓{RESET} {msg}")

        if self.warnings:
            print(f"\n{YELLOW}⚠ WARNINGS ({len(self.warnings)}){RESET}")
            for msg in self.warnings:
                print(f"  {YELLOW}⚠{RESET} {msg}")

        if self.failures:
            print(f"\n{RED}✗ FAILURES ({len(self.failures)}){RESET}")
            for msg in self.failures:
                print(f"  {RED}✗{RESET} {msg}")

        if self.critical_failures:
            print(f"\n{RED}{BOLD}✗✗ CRITICAL FAILURES ({len(self.critical_failures)}){RESET}")
            for msg in self.critical_failures:
                print(f"  {RED}{BOLD}✗✗{RESET} {msg}")

        print("\n" + "=" * 80)

        # Final verdict
        if self.critical_failures:
            print(f"{RED}{BOLD}VALIDATION FAILED - DO NOT LAUNCH RUNPOD{RESET}")
            print(f"Fix critical issues before proceeding.")
            return False
        elif self.failures:
            print(f"{YELLOW}VALIDATION PASSED WITH WARNINGS{RESET}")
            print(f"Training may work but some features might fail.")
            return True
        else:
            print(f"{GREEN}{BOLD}✓ ALL VALIDATIONS PASSED - READY FOR RUNPOD{RESET}")
            return True


def check_python_version(result: ValidationResult):
    """Check Python version compatibility"""
    print(f"\n{BLUE}Checking Python version...{RESET}")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and 8 <= version.minor < 12:
        result.add_pass(f"Python version {version_str} is compatible")
    elif version.major == 3 and version.minor >= 12:
        result.add_failure(
            f"Python {version_str} detected. PyTorch training requires Python <3.12. "
            f"Use Python 3.11.x",
            critical=True
        )
    else:
        result.add_failure(
            f"Python {version_str} detected. Requires Python 3.8-3.11",
            critical=True
        )


def check_core_dependencies(result: ValidationResult):
    """Check core Python packages"""
    print(f"\n{BLUE}Checking core dependencies...{RESET}")

    core_packages = {
        'pandas': '2.0.0',
        'numpy': '1.24.0',
        'scipy': '1.10.0',
        'matplotlib': '3.7.0',
        'yfinance': '0.2.0',
        'statsmodels': '0.14.0',
        'openpyxl': '3.1.0',
        'pyarrow': '14.0.0',
    }

    for package, min_version in core_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            result.add_pass(f"{package} {version} installed")
        except ImportError:
            result.add_failure(f"{package} not installed (required: >={min_version})", critical=True)


def check_ml_dependencies(result: ValidationResult):
    """Check ML/DL framework dependencies"""
    print(f"\n{BLUE}Checking ML/DL dependencies...{RESET}")

    ml_packages = {
        'torch': '2.1.0',
        'torchvision': '0.16.0',
        'transformers': '4.36.0',
        'pytorch_lightning': '2.1.0',
    }

    for package, min_version in ml_packages.items():
        try:
            # Handle special cases
            if package == 'pytorch_lightning':
                mod = importlib.import_module('pytorch_lightning')
            else:
                mod = importlib.import_module(package)

            version = getattr(mod, '__version__', 'unknown')
            result.add_pass(f"{package} {version} installed")

            # Check CUDA availability for PyTorch
            if package == 'torch':
                import torch
                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda
                    result.add_pass(f"CUDA {cuda_version} available with {torch.cuda.device_count()} GPU(s)")
                else:
                    result.add_warning("CUDA not available - training will use CPU (slow)")

        except ImportError:
            result.add_failure(f"{package} not installed (required: >={min_version})", critical=True)


def check_optional_dependencies(result: ValidationResult):
    """Check optional packages"""
    print(f"\n{BLUE}Checking optional dependencies...{RESET}")

    optional_packages = [
        'gymnasium',
        'stable_baselines3',
        'optuna',
        'tensorboard',
        'wandb',
        'mlflow',
        'arch',
        'h5py',
    ]

    for package in optional_packages:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            result.add_pass(f"{package} {version} installed")
        except ImportError:
            result.add_warning(f"{package} not installed (optional, some features may not work)")


def check_project_structure(result: ValidationResult):
    """Check project directory structure"""
    print(f"\n{BLUE}Checking project structure...{RESET}")

    project_root = Path(__file__).parent
    required_dirs = [
        'ml_models',
        'training',
        'data',
        'utils',
        'single_stock',
        'portfolio',
    ]

    required_files = [
        'training/config.yaml',
        'training/train_hybrid.py',
        'prepare_training_data.py',
        'data_fetching.py',
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            result.add_pass(f"Directory exists: {dir_name}/")
        else:
            result.add_failure(f"Missing directory: {dir_name}/", critical=True)

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            result.add_pass(f"File exists: {file_path}")
        else:
            result.add_failure(f"Missing file: {file_path}", critical=True)


def check_ml_modules(result: ValidationResult):
    """Check ML module imports"""
    print(f"\n{BLUE}Checking ML module imports...{RESET}")

    # Add project to path
    sys.path.insert(0, str(Path(__file__).parent))

    modules_to_check = [
        ('ml_models.hybrid_model', 'HybridTradingModel'),
        ('ml_models.cnn_module', 'CNN1DFeatureExtractor'),
        ('ml_models.lstm_module', 'LSTMEncoder'),
        ('ml_models.transformer_module', 'TransformerEncoder'),
        ('ml_models.fusion_layer', 'AttentionFusion'),
        ('ml_models.features', 'FeatureEngineer'),
        ('utils.advanced_feature_engineering', 'AdvancedFeatureEngineer'),
        ('training.utils', 'WalkForwardSplitter'),
    ]

    for module_name, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                result.add_pass(f"Module import OK: {module_name}.{class_name}")
            else:
                result.add_failure(f"Class not found: {module_name}.{class_name}", critical=True)
        except ImportError as e:
            result.add_failure(f"Import failed: {module_name} - {e}", critical=True)
        except Exception as e:
            result.add_failure(f"Error loading {module_name}: {e}", critical=False)


def check_training_config(result: ValidationResult):
    """Validate training configuration"""
    print(f"\n{BLUE}Checking training configuration...{RESET}")

    config_path = Path(__file__).parent / 'training' / 'config.yaml'

    if not config_path.exists():
        result.add_failure("training/config.yaml not found", critical=True)
        return

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ['data', 'model', 'training', 'hardware', 'logging', 'seeds']
        for section in required_sections:
            if section in config:
                result.add_pass(f"Config section exists: {section}")
            else:
                result.add_failure(f"Missing config section: {section}", critical=True)

        # Check specific important settings
        if 'data' in config:
            if 'sequence_length' in config['data']:
                result.add_pass(f"Sequence length: {config['data']['sequence_length']}")
            if 'train_split' in config['data']:
                result.add_pass(f"Train/val/test split: {config['data']['train_split']}/{config['data']['val_split']}/{config['data']['test_split']}")

        if 'training' in config:
            if 'batch_size' in config['training']:
                result.add_pass(f"Batch size: {config['training']['batch_size']}")
            if 'epochs' in config['training']:
                result.add_pass(f"Max epochs: {config['training']['epochs']}")

    except Exception as e:
        result.add_failure(f"Failed to parse config.yaml: {e}", critical=True)


def check_data_availability(result: ValidationResult):
    """Check if training data exists"""
    print(f"\n{BLUE}Checking data availability...{RESET}")

    data_dir = Path(__file__).parent / 'data'

    if not data_dir.exists():
        result.add_warning("data/ directory not found - will need to fetch data")
        return

    # Check for existing data files
    data_files = [
        'last_fetch.parquet',
        'last_fetch.csv',
        'training/training_data.parquet',
    ]

    found_data = False
    for file_path in data_files:
        full_path = data_dir / file_path if '/' not in file_path else data_dir.parent / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            result.add_pass(f"Data file exists: {file_path} ({size_mb:.1f} MB)")
            found_data = True

    if not found_data:
        result.add_warning("No training data found - run prepare_training_data.py first")


def check_runpod_script(result: ValidationResult):
    """Check RunPod deployment script"""
    print(f"\n{BLUE}Checking RunPod deployment script...{RESET}")

    runpod_script = Path(__file__).parent / 'deployment' / 'train_on_runpod.py'

    if runpod_script.exists():
        result.add_pass("RunPod training script exists")

        # Check for required environment variables
        env_vars = ['RUNPOD_API_KEY', 'WANDB_API_KEY']
        for var in env_vars:
            import os
            if os.environ.get(var):
                result.add_pass(f"Environment variable set: {var}")
            else:
                result.add_warning(f"Environment variable not set: {var} (optional)")
    else:
        result.add_warning("deployment/train_on_runpod.py not found")


def check_disk_space(result: ValidationResult):
    """Check available disk space"""
    print(f"\n{BLUE}Checking disk space...{RESET}")

    import shutil

    try:
        stats = shutil.disk_usage(Path(__file__).parent)
        free_gb = stats.free / (1024 ** 3)

        if free_gb > 10:
            result.add_pass(f"Available disk space: {free_gb:.1f} GB")
        elif free_gb > 5:
            result.add_warning(f"Low disk space: {free_gb:.1f} GB (recommended: >10 GB)")
        else:
            result.add_failure(f"Critically low disk space: {free_gb:.1f} GB", critical=True)
    except Exception as e:
        result.add_warning(f"Could not check disk space: {e}")


def generate_report(result: ValidationResult, output_file: str = None):
    """Generate detailed validation report"""
    report = {
        'timestamp': str(Path(__file__).parent),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'passed': len(result.passed),
        'warnings': len(result.warnings),
        'failures': len(result.failures),
        'critical_failures': len(result.critical_failures),
        'ready_for_training': not result.has_critical_failures(),
        'details': {
            'passed': result.passed,
            'warnings': result.warnings,
            'failures': result.failures,
            'critical_failures': result.critical_failures,
        }
    }

    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n{GREEN}Report saved to: {output_path}{RESET}")

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate training setup before launching RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 validate_training_setup.py              # Quick validation
  python3 validate_training_setup.py --full       # Full validation with optional deps
  python3 validate_training_setup.py --report     # Generate JSON report
        """
    )

    parser.add_argument('--full', action='store_true',
                       help='Run full validation including optional dependencies')
    parser.add_argument('--report', type=str, default=None,
                       help='Save validation report to JSON file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick validation (skip optional checks)')

    args = parser.parse_args()

    print(f"{BOLD}{BLUE}=" * 80)
    print("RUNPOD TRAINING PRE-FLIGHT VALIDATION")
    print("=" * 80 + f"{RESET}")

    result = ValidationResult()

    # Core validations (always run)
    check_python_version(result)
    check_core_dependencies(result)
    check_ml_dependencies(result)
    check_project_structure(result)
    check_training_config(result)

    # Full validation
    if args.full or not args.quick:
        check_ml_modules(result)
        check_optional_dependencies(result)
        check_data_availability(result)
        check_runpod_script(result)
        check_disk_space(result)

    # Print summary
    result.print_summary()

    # Generate report if requested
    if args.report or args.full:
        report_file = args.report or 'validation_report.json'
        generate_report(result, report_file)

    # Exit with appropriate code
    if result.has_critical_failures():
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
