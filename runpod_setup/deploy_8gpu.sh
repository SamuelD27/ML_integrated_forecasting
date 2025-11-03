#!/bin/bash
# ============================================================================
# 8x B200 GPU Training - Automated Deployment
# ============================================================================
# Hardware: 8x NVIDIA B200 (1.4TB total VRAM)
# Cost: $45.52/hr (8x $5.69/hr per GPU)
# Expected time: ~3.5 hours
# Total cost: ~$159
# Speedup: 5.5-6x vs single GPU
# ============================================================================

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "                    8x B200 GPU TRAINING - DEPLOYMENT"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  GPUs: 8x NVIDIA B200"
echo "  Total VRAM: 1.4TB"
echo "  Batch size: 2048 per GPU (16,384 effective)"
echo "  Cost: \$45.52/hr"
echo "  Expected time: ~3.5 hours"
echo "  Total cost: ~\$159"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

# ============================================================================
# STEP 1: VERIFY HARDWARE
# ============================================================================

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 1: VERIFYING HARDWARE"
echo "─────────────────────────────────────────────────────────────────────────────"

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "✓ Found $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -lt 8 ]; then
    echo "⚠️  WARNING: Expected 8 GPUs but found $GPU_COUNT"
    echo "   Training will proceed with $GPU_COUNT GPUs (slower)"
    read -p "   Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Display GPU info
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  WARNING: nvcc not found, but CUDA may still work"
else
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "✓ CUDA version: $CUDA_VERSION"
fi

# ============================================================================
# STEP 2: SETUP ENVIRONMENT
# ============================================================================

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 2: SETTING UP ENVIRONMENT"
echo "─────────────────────────────────────────────────────────────────────────────"

# Set environment variables for optimal multi-GPU performance
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "✓ Environment variables set for multi-GPU training"
echo "  NCCL_DEBUG=WARN (reduced logging)"
echo "  NCCL_IB_DISABLE=0 (InfiniBand enabled)"
echo "  NCCL_P2P_DISABLE=0 (P2P transfers enabled)"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# ============================================================================
# STEP 3: INSTALL DEPENDENCIES
# ============================================================================

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 3: INSTALLING DEPENDENCIES"
echo "─────────────────────────────────────────────────────────────────────────────"

# Check if PyTorch is installed
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch already installed: $TORCH_VERSION"

    # Check CUDA support
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        echo "✓ PyTorch CUDA support: Enabled"
    else
        echo "✗ PyTorch CUDA support: DISABLED - Reinstalling PyTorch"
        pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -q pytorch-lightning pandas numpy pyarrow yfinance \
            scikit-learn tensorboard tqdm pyyaml

echo "✓ All dependencies installed"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import pytorch_lightning as pl
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ PyTorch Lightning version: {pl.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ CUDA version: {torch.version.cuda}')
print(f'✓ Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)')
"

# ============================================================================
# STEP 4: FETCH DATA
# ============================================================================

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 4: FETCHING TRAINING DATA"
echo "─────────────────────────────────────────────────────────────────────────────"

DATA_PATH="/workspace/data/training/training_data_compressed.parquet"

if [ -f "$DATA_PATH" ]; then
    echo "✓ Training data already exists: $DATA_PATH"

    # Show file info
    FILE_SIZE=$(du -h "$DATA_PATH" | cut -f1)
    echo "  File size: $FILE_SIZE"

    # Count rows
    ROW_COUNT=$(python -c "import pandas as pd; df = pd.read_parquet('$DATA_PATH'); print(f'{len(df):,}')")
    TICKER_COUNT=$(python -c "import pandas as pd; df = pd.read_parquet('$DATA_PATH'); print(df['ticker'].nunique())")
    echo "  Rows: $ROW_COUNT"
    echo "  Tickers: $TICKER_COUNT"

    read -p "Re-fetch data? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Fetching fresh data..."
        python fetch_sp500_data.py
    fi
else
    echo "Training data not found. Fetching now..."
    echo "This will take 30-45 minutes..."
    echo ""

    START_TIME=$(date +%s)
    python fetch_sp500_data.py
    END_TIME=$(date +%s)

    FETCH_TIME=$((($END_TIME - $START_TIME) / 60))
    echo ""
    echo "✓ Data fetched in $FETCH_TIME minutes"
fi

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "STEP 5: SELECT TRAINING MODE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Choose training mode:"
echo ""
echo "  [1] STANDARD - Fast training without backtesting"
echo "      Time: ~3.5 hours | Cost: ~\$159"
echo "      Optimizes: Validation loss (MSE)"
echo "      Use when: Quick iteration, testing code changes"
echo ""
echo "  [2] BACKTEST - Training with integrated backtesting (RECOMMENDED)"
echo "      Time: ~4 hours | Cost: ~\$182"
echo "      Optimizes: Sharpe ratio (risk-adjusted returns)"
echo "      Includes: Walk-forward validation, transaction costs, trading metrics"
echo "      Use when: Final model training, production deployment"
echo ""
read -p "Select mode (1 or 2): " -n 1 -r MODE
echo ""
echo ""

if [[ $MODE == "1" ]]; then
    TRAIN_SCRIPT="train_b200_8gpu.py"
    OUTPUT_DIR="/workspace/output"
    EXPECTED_TIME="~3.5 hours"
    COST="~\$159"
    echo "Selected: STANDARD MODE"
elif [[ $MODE == "2" ]]; then
    TRAIN_SCRIPT="train_b200_8gpu_backtest.py"
    OUTPUT_DIR="/workspace/output_backtest"
    EXPECTED_TIME="~4 hours"
    COST="~\$182"
    echo "Selected: BACKTEST MODE (with walk-forward validation)"
else
    echo "Invalid selection. Exiting..."
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "TRAINING CONFIGURATION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  GPUs: $GPU_COUNT"
echo "  Batch size per GPU: 2,048"
echo "  Effective batch size: $(($GPU_COUNT * 2048))"
echo "  Model: 151M parameters"
echo "  Expected time: $EXPECTED_TIME"
echo "  Estimated cost: $COST"
echo "  Training script: $TRAIN_SCRIPT"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Monitoring:"
echo "  GPU usage: watch -n 1 nvidia-smi"
echo "  Training log: tail -f $OUTPUT_DIR/training.log"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting..."
    exit 0
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Start training with timing
echo ""
echo "Starting training at $(date)..."
echo ""

START_TIME=$(date +%s)

# Run training (with output to both console and log)
python $TRAIN_SCRIPT 2>&1 | tee $OUTPUT_DIR/training.log

END_TIME=$(date +%s)

# ============================================================================
# STEP 6: TRAINING COMPLETE
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "TRAINING COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"

# Calculate time and cost
TRAINING_TIME_SEC=$((END_TIME - START_TIME))
TRAINING_TIME_MIN=$((TRAINING_TIME_SEC / 60))
TRAINING_TIME_HR=$(echo "scale=2; $TRAINING_TIME_SEC / 3600" | bc)
TOTAL_COST=$(echo "scale=2; $TRAINING_TIME_HR * 45.52" | bc)

echo ""
echo "Training time: ${TRAINING_TIME_HR} hours (${TRAINING_TIME_MIN} minutes)"
echo "Total cost: \$${TOTAL_COST} (at \$45.52/hr)"
echo ""

# Find best checkpoint
BEST_CHECKPOINT=$(find $OUTPUT_DIR/checkpoints -name "best-*.ckpt" 2>/dev/null | head -1)

if [ -n "$BEST_CHECKPOINT" ]; then
    CHECKPOINT_SIZE=$(du -h "$BEST_CHECKPOINT" | cut -f1)
    echo "Best checkpoint: $BEST_CHECKPOINT"
    echo "Checkpoint size: $CHECKPOINT_SIZE"
else
    echo "⚠️  No checkpoint found"
fi

# Show results
RESULTS_FILE="$OUTPUT_DIR/training_results.json"
if [[ $MODE == "2" ]]; then
    RESULTS_FILE="$OUTPUT_DIR/training_results_backtest.json"
fi

if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Training results:"
    python -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)
    print(f\"  Best val loss: {results.get('test_results', [{}])[0].get('val_loss', 'N/A')}\")
    print(f\"  Test accuracy: {results.get('test_results', [{}])[0].get('val_accuracy', 'N/A')}\")
"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "NEXT STEPS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Download checkpoint to your local machine:"
echo "   scp root@<runpod-ip>:$BEST_CHECKPOINT ./"
echo ""
echo "2. Download all results:"
echo "   scp -r root@<runpod-ip>:$OUTPUT_DIR ./"
echo ""
echo "3. View TensorBoard logs locally:"
echo "   tensorboard --logdir ./${OUTPUT_DIR##*/}/logs"
echo ""
echo "4. TERMINATE THE POD to stop charges!"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Training completed at $(date)"
echo ""
