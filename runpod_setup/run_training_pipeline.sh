#!/bin/bash
# RunPod Complete Training Pipeline
# ==================================
# Orchestrates full training workflow:
# 1. Data fetching (100+ tickers, 10 years)
# 2. Data validation
# 3. Model training (3x RTX 5090)
# 4. Results download
#
# Usage: bash run_training_pipeline.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - Auto-detect code directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/output"
TRAINING_DATA="${DATA_DIR}/training/training_data.parquet"

echo "Code directory: ${CODE_DIR}"

echo ""
echo "=========================================="
echo "RunPod ML Training Pipeline"
echo "=========================================="
echo "Hardware: 3x RTX 5090 (72GB VRAM)"
echo "Strategy: Minimal checkpoints, maximum performance"
echo "=========================================="
echo ""

# Step 1: Data Fetching
echo -e "${BLUE}[STEP 1/4]${NC} Fetching training data..."
echo "  • Tickers: 100+"
echo "  • Timeframe: 10 years"
echo "  • Parallel workers: 20"
echo ""

START_TIME=$(date +%s)

python ${CODE_DIR}/runpod_setup/fetch_training_data_large.py

FETCH_END=$(date +%s)
FETCH_DURATION=$((FETCH_END - START_TIME))

if [ -f "$TRAINING_DATA" ]; then
    FILE_SIZE=$(du -h "$TRAINING_DATA" | cut -f1)
    echo -e "${GREEN}✅ Data fetching complete${NC}"
    echo "  • Duration: ${FETCH_DURATION}s"
    echo "  • File: ${TRAINING_DATA}"
    echo "  • Size: ${FILE_SIZE}"
else
    echo -e "${RED}❌ ERROR: Training data not found at ${TRAINING_DATA}${NC}"
    exit 1
fi
echo ""

# Step 2: Data Validation
echo -e "${BLUE}[STEP 2/4]${NC} Validating data..."

python -c "
import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('${TRAINING_DATA}')

print(f'  • Total rows: {len(df):,}')
print(f'  • Unique tickers: {df[\"ticker\"].nunique()}')
print(f'  • Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')
print(f'  • Features: {len(df.columns)}')

# Check for issues
nan_count = df.isna().sum().sum()
if nan_count > 0:
    print(f'  ⚠️  WARNING: {nan_count:,} NaN values found')
else:
    print('  • No NaN values ✓')

# Check feature columns
required_features = ['returns', 'volatility_20', 'sma_5', 'rsi']
missing = [f for f in required_features if f not in df.columns]
if missing:
    print(f'  ⚠️  WARNING: Missing features: {missing}')
else:
    print('  • All required features present ✓')

print('')
print('${GREEN}✅ Data validation complete${NC}')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Data validation failed${NC}"
    exit 1
fi
echo ""

# Step 3: GPU Check
echo -e "${BLUE}[STEP 3/4]${NC} Pre-training GPU check..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=',' read -r idx name total free; do
    echo "  • GPU ${idx}: ${name} (${free}MB / ${total}MB free)"
done
echo ""

# Step 4: Model Training
echo -e "${BLUE}[STEP 4/4]${NC} Starting model training..."
echo "  • Model: Transformer (512 hidden, 6 layers, 8 heads)"
echo "  • Batch size: 1024 × 4 accumulation = 4096 effective"
echo "  • Precision: Mixed 16-bit"
echo "  • Max epochs: 100"
echo "  • Checkpoints: MINIMAL (top-1 only, every 10 epochs)"
echo ""
echo "This will take several hours. TensorBoard logs at:"
echo "  ${OUTPUT_DIR}/logs/transformer_forecaster"
echo ""
echo "To monitor in another terminal:"
echo "  watch -n 1 nvidia-smi"
echo ""

read -p "Press ENTER to start training (or Ctrl+C to cancel)..."
echo ""

TRAIN_START=$(date +%s)

# Run training with output logging
python ${CODE_DIR}/runpod_setup/train_runpod.py 2>&1 | tee ${OUTPUT_DIR}/training.log

TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
TRAIN_HOURS=$(echo "scale=2; $TRAIN_DURATION / 3600" | bc)

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo -e "${GREEN}✅ TRAINING COMPLETE${NC}"
    echo "=========================================="
    echo ""
    echo "⏱️  Timing:"
    echo "  • Data fetch: ${FETCH_DURATION}s"
    echo "  • Training: ${TRAIN_DURATION}s (${TRAIN_HOURS}h)"
    echo "  • Total: $((TRAIN_END - START_TIME))s"
    echo ""
    echo "📁 Outputs:"
    echo "  • Checkpoints: ${OUTPUT_DIR}/checkpoints/"
    echo "  • Logs: ${OUTPUT_DIR}/logs/"
    echo "  • Results: ${OUTPUT_DIR}/training_results.json"
    echo "  • Full log: ${OUTPUT_DIR}/training.log"
    echo ""
    echo "📊 Results summary:"
    python -c "
import json
with open('${OUTPUT_DIR}/training_results.json', 'r') as f:
    results = json.load(f)
print(f'  • Best checkpoint: {results[\"best_checkpoint\"]}')
print(f'  • Total parameters: {results[\"total_parameters\"]:,}')
print(f'  • Completed at: {results[\"completed_at\"]}')
if 'test_results' in results and len(results['test_results']) > 0:
    test = results['test_results'][0]
    if 'test_loss' in test:
        print(f'  • Test loss: {test[\"test_loss\"]:.4f}')
    if 'test_accuracy' in test:
        print(f'  • Test accuracy: {test[\"test_accuracy\"]:.2%}')
"
    echo ""
    echo "🔥 GPU utilization summary:"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits | while IFS=',' read -r idx gpu_util mem_util mem_used; do
        echo "  • GPU ${idx}: ${gpu_util}% compute, ${mem_util}% memory (${mem_used}MB used)"
    done
    echo ""
    echo "💾 To download results:"
    echo "  scp -r runpod:/workspace/output ./runpod_output"
    echo ""
else
    echo ""
    echo -e "${RED}❌ TRAINING FAILED${NC}"
    echo "Check logs at: ${OUTPUT_DIR}/training.log"
    exit 1
fi
