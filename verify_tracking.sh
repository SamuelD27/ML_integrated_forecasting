#!/bin/bash
# Quick verification script for experiment tracking setup

echo "=========================================="
echo "EXPERIMENT TRACKING VERIFICATION"
echo "=========================================="
echo ""

# Check W&B
echo "1. Checking Weights & Biases..."
if grep -q "WANDB_API_KEY" .env; then
    echo "   ✓ W&B API key found in .env"
    if command -v wandb &> /dev/null; then
        echo "   ✓ wandb CLI installed"
        wandb status 2>&1 | grep -q "Logged in" && echo "   ✓ W&B logged in" || echo "   ⚠ Run: wandb login"
    else
        echo "   ⚠ wandb not installed (pip install wandb)"
    fi
else
    echo "   ✗ WANDB_API_KEY not found in .env"
fi
echo ""

# Check TensorBoard
echo "2. Checking TensorBoard..."
if python3 -c "import tensorboard" 2>/dev/null; then
    echo "   ✓ TensorBoard installed"
else
    echo "   ⚠ TensorBoard not installed (pip install tensorboard)"
fi
if [ -d "runs" ]; then
    echo "   ✓ runs/ directory exists"
else
    echo "   ⚠ runs/ directory missing (will be created on first run)"
    mkdir -p runs
fi
echo ""

# Check MLflow
echo "3. Checking MLflow..."
if python3 -c "import mlflow" 2>/dev/null; then
    echo "   ✓ MLflow installed"
else
    echo "   ⚠ MLflow not installed (pip install mlflow)"
fi
if [ -d "mlruns" ]; then
    echo "   ✓ mlruns/ directory exists"
else
    echo "   ℹ mlruns/ directory will be created on first use"
fi
echo ""

# Check config
echo "4. Checking training config..."
if [ -f "training/config.yaml" ]; then
    echo "   ✓ training/config.yaml exists"

    # Check W&B enabled
    if grep -A2 "wandb:" training/config.yaml | grep -q "enabled: true"; then
        echo "   ✓ W&B enabled in config"
    else
        echo "   ⚠ W&B disabled in config"
    fi

    # Check TensorBoard enabled
    if grep -A2 "tensorboard:" training/config.yaml | grep -q "enabled: true"; then
        echo "   ✓ TensorBoard enabled in config"
    else
        echo "   ⚠ TensorBoard disabled in config"
    fi
else
    echo "   ✗ training/config.yaml not found"
fi
echo ""

# Check directories
echo "5. Checking required directories..."
for dir in checkpoints data/features logs; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/ exists"
    else
        echo "   ⚠ $dir/ missing (creating...)"
        mkdir -p "$dir"
        echo "   ✓ $dir/ created"
    fi
done
echo ""

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Ready to start training with tracking!"
echo ""
echo "Commands:"
echo "  Start training:  cd training && python train_hybrid.py --config config.yaml"
echo "  View TensorBoard: tensorboard --logdir runs/"
echo "  View W&B:        https://wandb.ai/samsamd2704-nus/hybrid-trading-model"
echo "  View MLflow:     mlflow ui --backend-store-uri mlruns/"
echo ""
echo "Documentation: See TRACKING_SETUP.md for full guide"
echo ""
