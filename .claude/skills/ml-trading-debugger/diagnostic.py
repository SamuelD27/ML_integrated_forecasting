#!/usr/bin/env python3
"""
Automated diagnostic script for ML trading system debugging.

Usage:
    python diagnostic.py --full                    # Run all diagnostics
    python diagnostic.py --component training      # Check specific component
    python diagnostic.py --validate-data           # Data pipeline validation
    python diagnostic.py --system-health           # GPU and storage check
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

try:
    import torch
    import pandas as pd
    import numpy as np
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: Some imports not available. Install requirements.")


class DiagnosticRunner:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }
        
    def add_check(self, name, status, message, details=None):
        """Add a diagnostic check result."""
        self.results['checks'].append({
            'name': name,
            'status': status,  # 'pass', 'fail', 'warning'
            'message': message,
            'details': details or {}
        })
        
        # Print result
        status_symbol = {
            'pass': '✓',
            'fail': '✗',
            'warning': '⚠'
        }
        print(f"{status_symbol.get(status, '?')} {name}: {message}")
        if details:
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    def check_system_health(self):
        """Check GPU memory and storage."""
        print("\n=== System Health ===")
        
        # GPU Check
        if not IMPORTS_AVAILABLE or not torch.cuda.is_available():
            self.add_check(
                "GPU Availability",
                "fail",
                "CUDA not available or PyTorch not installed"
            )
        else:
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                status = 'pass'
            elif gpu_count >= 1:
                status = 'warning'
            else:
                status = 'fail'
                
            details = {'gpu_count': gpu_count}
            
            # Memory details for each GPU
            for i in range(gpu_count):
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                details[f'gpu_{i}_memory_total_GB'] = f"{total:.1f}"
                details[f'gpu_{i}_memory_allocated_GB'] = f"{allocated:.1f}"
                details[f'gpu_{i}_memory_reserved_GB'] = f"{reserved:.1f}"
            
            self.add_check(
                "GPU Status",
                status,
                f"{gpu_count} GPUs detected (target: 8)",
                details
            )
        
        # Storage Check
        if sys.platform != 'win32':
            import shutil
            total, used, free = shutil.disk_usage('/')
            free_gb = free / 1e9
            total_gb = total / 1e9
            used_pct = (used / total) * 100
            
            if used_pct > 90:
                status = 'fail'
                message = f"Storage critical: {used_pct:.1f}% used"
            elif used_pct > 75:
                status = 'warning'
                message = f"Storage high: {used_pct:.1f}% used"
            else:
                status = 'pass'
                message = f"Storage healthy: {used_pct:.1f}% used"
            
            self.add_check(
                "Storage",
                status,
                message,
                {
                    'total_GB': f"{total_gb:.1f}",
                    'free_GB': f"{free_gb:.1f}",
                    'used_percent': f"{used_pct:.1f}"
                }
            )
    
    def check_training_setup(self):
        """Check training configuration and environment."""
        print("\n=== Training Setup ===")
        
        if not IMPORTS_AVAILABLE:
            self.add_check(
                "Training Dependencies",
                "fail",
                "PyTorch or pandas not available"
            )
            return
        
        # Check PyTorch version
        torch_version = torch.__version__
        if int(torch_version.split('.')[0]) >= 2:
            status = 'pass'
        else:
            status = 'warning'
        
        self.add_check(
            "PyTorch Version",
            status,
            f"PyTorch {torch_version} (recommended: 2.x+)",
            {'version': torch_version}
        )
        
        # Check for mixed precision support
        if torch.cuda.is_available():
            amp_available = hasattr(torch.cuda, 'amp')
            self.add_check(
                "Mixed Precision",
                'pass' if amp_available else 'warning',
                "AMP available" if amp_available else "AMP not detected",
                {'automatic_mixed_precision': str(amp_available)}
            )
            
            # Check TF32
            tf32_enabled = torch.backends.cuda.matmul.allow_tf32
            self.add_check(
                "TF32 Acceleration",
                'pass' if tf32_enabled else 'warning',
                "TF32 enabled" if tf32_enabled else "TF32 disabled (enable for speed)",
                {'tf32_matmul': str(tf32_enabled)}
            )
    
    def check_data_pipeline(self):
        """Validate data pipeline and sources."""
        print("\n=== Data Pipeline ===")
        
        # Check common data directories
        data_paths = {
            'uploads': Path('/mnt/user-data/uploads'),
            'processed': Path('/data/processed'),
            'raw': Path('/data/raw')
        }
        
        for name, path in data_paths.items():
            if path.exists():
                file_count = len(list(path.glob('*')))
                self.add_check(
                    f"Data Directory: {name}",
                    'pass',
                    f"Directory exists with {file_count} files",
                    {'path': str(path), 'file_count': file_count}
                )
            else:
                self.add_check(
                    f"Data Directory: {name}",
                    'warning',
                    "Directory not found (may be expected)",
                    {'path': str(path)}
                )
        
        # Check for Parquet files
        parquet_files = []
        for path in data_paths.values():
            if path.exists():
                parquet_files.extend(list(path.glob('**/*.parquet')))
        
        if parquet_files:
            total_size = sum(f.stat().st_size for f in parquet_files) / 1e9
            self.add_check(
                "Parquet Data Files",
                'pass',
                f"Found {len(parquet_files)} Parquet files",
                {
                    'file_count': len(parquet_files),
                    'total_size_GB': f"{total_size:.2f}"
                }
            )
        else:
            self.add_check(
                "Parquet Data Files",
                'warning',
                "No Parquet files found in common locations"
            )
        
        # Check if pandas can read a sample Parquet file
        if parquet_files and IMPORTS_AVAILABLE:
            try:
                sample_file = parquet_files[0]
                df = pd.read_parquet(sample_file, nrows=5)
                self.add_check(
                    "Parquet Read Test",
                    'pass',
                    f"Successfully read {sample_file.name}",
                    {
                        'columns': str(df.columns.tolist()),
                        'shape': str(df.shape)
                    }
                )
            except Exception as e:
                self.add_check(
                    "Parquet Read Test",
                    'fail',
                    f"Failed to read Parquet file: {str(e)}"
                )
    
    def check_model_checkpoints(self):
        """Check model checkpoint status."""
        print("\n=== Model Checkpoints ===")
        
        checkpoint_paths = [
            Path('/checkpoints'),
            Path('/models/checkpoints'),
            Path('/home/claude/checkpoints')
        ]
        
        checkpoint_files = []
        for path in checkpoint_paths:
            if path.exists():
                checkpoint_files.extend(list(path.glob('**/*.pt')))
                checkpoint_files.extend(list(path.glob('**/*.pth')))
        
        if checkpoint_files:
            total_size = sum(f.stat().st_size for f in checkpoint_files) / 1e9
            largest = max(checkpoint_files, key=lambda f: f.stat().st_size)
            largest_size = largest.stat().st_size / 1e9
            
            if total_size > 100:  # >100GB is concerning
                status = 'warning'
                message = f"Large checkpoint storage: {total_size:.1f}GB"
            else:
                status = 'pass'
                message = f"Found {len(checkpoint_files)} checkpoints"
            
            self.add_check(
                "Model Checkpoints",
                status,
                message,
                {
                    'checkpoint_count': len(checkpoint_files),
                    'total_size_GB': f"{total_size:.1f}",
                    'largest_file': largest.name,
                    'largest_size_GB': f"{largest_size:.1f}"
                }
            )
            
            # Check for old checkpoints
            old_threshold = datetime.now() - timedelta(days=7)
            old_checkpoints = [
                f for f in checkpoint_files 
                if datetime.fromtimestamp(f.stat().st_mtime) < old_threshold
            ]
            
            if old_checkpoints:
                old_size = sum(f.stat().st_size for f in old_checkpoints) / 1e9
                self.add_check(
                    "Old Checkpoints",
                    'warning',
                    f"{len(old_checkpoints)} checkpoints >7 days old",
                    {
                        'old_count': len(old_checkpoints),
                        'old_size_GB': f"{old_size:.1f}",
                        'recommendation': 'Consider cleanup to free storage'
                    }
                )
        else:
            self.add_check(
                "Model Checkpoints",
                'warning',
                "No checkpoint files found"
            )
    
    def check_integration(self):
        """Check integration between components."""
        print("\n=== Integration Checks ===")
        
        # Check if prediction files exist
        prediction_paths = [
            Path('/data/predictions'),
            Path('/mnt/user-data/outputs/predictions')
        ]
        
        prediction_files = []
        for path in prediction_paths:
            if path.exists():
                prediction_files.extend(list(path.glob('predictions_*.parquet')))
        
        if prediction_files:
            latest = max(prediction_files, key=lambda f: f.stat().st_mtime)
            age_days = (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).days
            
            if age_days > 1:
                status = 'warning'
                message = f"Predictions stale ({age_days} days old)"
            else:
                status = 'pass'
                message = f"Recent predictions found ({age_days} days old)"
            
            self.add_check(
                "ML Predictions Available",
                status,
                message,
                {
                    'prediction_files': len(prediction_files),
                    'latest_file': latest.name,
                    'age_days': age_days
                }
            )
        else:
            self.add_check(
                "ML Predictions Available",
                'fail',
                "No prediction files found - model-to-dashboard disconnect",
                {'recommendation': 'Run inference pipeline to generate predictions'}
            )
    
    def generate_report(self):
        """Generate summary report."""
        print("\n" + "="*50)
        print("DIAGNOSTIC SUMMARY")
        print("="*50)
        
        pass_count = sum(1 for c in self.results['checks'] if c['status'] == 'pass')
        warning_count = sum(1 for c in self.results['checks'] if c['status'] == 'warning')
        fail_count = sum(1 for c in self.results['checks'] if c['status'] == 'fail')
        total = len(self.results['checks'])
        
        print(f"Total Checks: {total}")
        print(f"✓ Passed: {pass_count}")
        print(f"⚠ Warnings: {warning_count}")
        print(f"✗ Failed: {fail_count}")
        
        if fail_count > 0:
            print("\nCritical Issues:")
            for check in self.results['checks']:
                if check['status'] == 'fail':
                    print(f"  - {check['name']}: {check['message']}")
        
        if warning_count > 0:
            print("\nWarnings:")
            for check in self.results['checks']:
                if check['status'] == 'warning':
                    print(f"  - {check['name']}: {check['message']}")
        
        # Save results to JSON
        output_file = Path('diagnostic_results.json')
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
        
        return fail_count == 0


def main():
    parser = argparse.ArgumentParser(description='ML Trading System Diagnostics')
    parser.add_argument('--full', action='store_true', 
                       help='Run all diagnostic checks')
    parser.add_argument('--component', choices=['training', 'data', 'infrastructure', 'integration'],
                       help='Check specific component')
    parser.add_argument('--validate-data', action='store_true',
                       help='Validate data pipeline')
    parser.add_argument('--system-health', action='store_true',
                       help='Check GPU memory and storage')
    
    args = parser.parse_args()
    
    runner = DiagnosticRunner()
    
    if args.full or not any([args.component, args.validate_data, args.system_health]):
        # Run all checks
        runner.check_system_health()
        runner.check_training_setup()
        runner.check_data_pipeline()
        runner.check_model_checkpoints()
        runner.check_integration()
    else:
        # Run specific checks
        if args.system_health:
            runner.check_system_health()
        
        if args.validate_data:
            runner.check_data_pipeline()
        
        if args.component:
            if args.component == 'training':
                runner.check_training_setup()
                runner.check_model_checkpoints()
            elif args.component == 'data':
                runner.check_data_pipeline()
            elif args.component == 'infrastructure':
                runner.check_system_health()
            elif args.component == 'integration':
                runner.check_integration()
    
    success = runner.generate_report()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
