#!/bin/bash
# Setup Python path for training on RunPod
# This ensures all modules are importable

# Get the project root directory
PROJECT_ROOT="/workspace/ML_integrated_forecasting"

# Export PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "✓ PYTHONPATH set to: $PYTHONPATH"
echo ""
echo "Now you can run:"
echo "  python training/train_hybrid.py --config training/config.yaml"
