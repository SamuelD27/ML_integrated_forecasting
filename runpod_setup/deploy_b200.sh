#!/bin/bash
# ============================================================================
# B200 COMPLETE DEPLOYMENT SCRIPT
# ============================================================================
# Run this script once on RunPod B200 instance
# It will:
# 1. Setup environment
# 2. Fetch S&P 500 data (500 tickers × 15 years)
# 3. Train 100M parameter model
# 4. Save results
#
# Usage: bash deploy_b200.sh
# ============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
WORKSPACE="/workspace"
CODE_DIR="${WORKSPACE}/code"
DATA_DIR="${WORKSPACE}/data/training"
OUTPUT_DIR="${WORKSPACE}/output"

echo ""
echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}                  NVIDIA B200 TRAINING DEPLOYMENT${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo -e "${YELLOW}Hardware: 1x B200 (192GB VRAM)${NC}"
echo -e "${YELLOW}Dataset: 500 S&P 500 tickers × 15 years${NC}"
echo -e "${YELLOW}Model: 100M parameters (Transformer)${NC}"
echo -e "${YELLOW}Expected time: 8-10 hours${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""

# ============================================================================
# STEP 1: ENVIRONMENT SETUP
# ============================================================================

echo -e "${BLUE}[STEP 1/5]${NC} Setting up environment..."
echo ""

# Check GPU
echo "🔍 Checking B200 GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo -e "${RED}❌ ERROR: nvidia-smi not found!${NC}"
    exit 1
fi

# Check Python
echo "🐍 Checking Python..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: ${python_version}"

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo -e "${RED}❌ ERROR: Python 3.11+ required${NC}"
    exit 1
fi
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p ${DATA_DIR}
mkdir -p ${OUTPUT_DIR}/{checkpoints,logs}
echo "✅ Directories created"
echo ""

# Install PyTorch with CUDA 12.1
echo "🔥 Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
echo "✅ PyTorch installed"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
cd ${CODE_DIR}
if [ -f "runpod_setup/requirements_runpod.txt" ]; then
    pip install -r runpod_setup/requirements_runpod.txt --quiet
else
    pip install pytorch-lightning pandas numpy pyarrow yfinance lightgbm scikit-learn tensorboard tqdm pyyaml --quiet
fi
echo "✅ Dependencies installed"
echo ""

# Verify PyTorch + CUDA
echo "🔥 Verifying PyTorch CUDA..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('❌ ERROR: CUDA not available!')
    exit(1)
"
echo ""

echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo ""

# ============================================================================
# STEP 2: CLONE/UPDATE CODE
# ============================================================================

echo -e "${BLUE}[STEP 2/5]${NC} Checking code..."
echo ""

if [ ! -d "${CODE_DIR}/.git" ]; then
    echo "📥 Cloning repository..."
    cd ${WORKSPACE}
    git clone https://github.com/SamuelD27/ML_integrated_forecasting.git code
else
    echo "📥 Updating repository..."
    cd ${CODE_DIR}
    git pull origin main
fi

echo -e "${GREEN}✅ Code ready!${NC}"
echo ""

# ============================================================================
# STEP 3: FETCH S&P 500 DATA
# ============================================================================

echo -e "${BLUE}[STEP 3/5]${NC} Fetching S&P 500 data (500 tickers × 15 years)..."
echo ""
echo -e "${YELLOW}⏱️  Expected time: 30-45 minutes${NC}"
echo ""

START_FETCH=$(date +%s)

cd ${CODE_DIR}
python runpod_setup/fetch_sp500_data.py

END_FETCH=$(date +%s)
FETCH_DURATION=$((END_FETCH - START_FETCH))
FETCH_MINUTES=$((FETCH_DURATION / 60))

echo ""
echo -e "${GREEN}✅ Data fetching complete! (${FETCH_MINUTES} minutes)${NC}"
echo ""

# Verify data
if [ -f "${DATA_DIR}/training_data.parquet" ]; then
    FILE_SIZE=$(du -h "${DATA_DIR}/training_data.parquet" | cut -f1)
    echo "📊 Data file: ${FILE_SIZE}"

    # Show data summary
    python -c "
import pandas as pd
df = pd.read_parquet('${DATA_DIR}/training_data.parquet')
print(f'Total rows: {len(df):,}')
print(f'Tickers: {df[\"ticker\"].nunique()}')
print(f'Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')
print(f'Features: {len(df.columns)}')
"
    echo ""
else
    echo -e "${RED}❌ ERROR: Data file not found!${NC}"
    exit 1
fi

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================

echo -e "${BLUE}[STEP 4/5]${NC} Training model on B200..."
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: 100M parameters"
echo "  Architecture: 1024H × 12L × 16A"
echo "  Batch size: 16,384"
echo "  Epochs: 150"
echo "  Precision: BF16-mixed"
echo ""
echo -e "${YELLOW}⏱️  Expected time: 8-10 hours${NC}"
echo ""
echo -e "${CYAN}To monitor in another terminal:${NC}"
echo "  watch -n 1 nvidia-smi"
echo "  tail -f ${OUTPUT_DIR}/training.log"
echo ""

read -p "Press ENTER to start training (or Ctrl+C to cancel)..."
echo ""

START_TRAIN=$(date +%s)

# Run training with output logging
cd ${CODE_DIR}
python runpod_setup/train_b200.py 2>&1 | tee ${OUTPUT_DIR}/training.log

END_TRAIN=$(date +%s)
TRAIN_DURATION=$((END_TRAIN - START_TRAIN))
TRAIN_HOURS=$(echo "scale=2; ${TRAIN_DURATION} / 3600" | bc)

echo ""
echo -e "${GREEN}✅ Training complete! (${TRAIN_HOURS} hours)${NC}"
echo ""

# ============================================================================
# STEP 5: RESULTS SUMMARY
# ============================================================================

echo -e "${BLUE}[STEP 5/5]${NC} Generating results summary..."
echo ""

TOTAL_DURATION=$((END_TRAIN - START_FETCH))
TOTAL_HOURS=$(echo "scale=2; ${TOTAL_DURATION} / 3600" | bc)

echo ""
echo -e "${CYAN}============================================================================${NC}"
echo -e "${GREEN}                    ✅ DEPLOYMENT COMPLETE${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""
echo -e "${YELLOW}⏱️  Timing:${NC}"
echo "  Data fetch: ${FETCH_MINUTES} minutes"
echo "  Training: ${TRAIN_HOURS} hours"
echo "  Total: ${TOTAL_HOURS} hours"
echo ""
echo -e "${YELLOW}📁 Outputs:${NC}"
echo "  Checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "  Logs: ${OUTPUT_DIR}/logs/"
echo "  Results: ${OUTPUT_DIR}/training_results.json"
echo "  Training log: ${OUTPUT_DIR}/training.log"
echo ""

# Show results if available
if [ -f "${OUTPUT_DIR}/training_results.json" ]; then
    echo -e "${YELLOW}📊 Training Results:${NC}"
    python -c "
import json
with open('${OUTPUT_DIR}/training_results.json', 'r') as f:
    results = json.load(f)
print(f'  Best checkpoint: {results.get(\"best_checkpoint\", \"N/A\")}')
print(f'  Best val loss: {results.get(\"best_val_loss\", \"N/A\")}')
print(f'  Total parameters: {results.get(\"total_parameters\", 0):,}')
print(f'  Completed at: {results.get(\"completed_at\", \"N/A\")}')
"
    echo ""
fi

echo -e "${YELLOW}🔥 GPU Stats:${NC}"
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
echo ""

echo -e "${YELLOW}💾 Download Results:${NC}"
echo "  From your local machine:"
echo "  scp -r runpod@<ip>:${OUTPUT_DIR} ./b200_results"
echo ""

echo -e "${YELLOW}🎯 Next Steps:${NC}"
echo "  1. Download checkpoint: best-*.ckpt"
echo "  2. Integrate into dashboard"
echo "  3. Run backtests on held-out data"
echo "  4. Monitor prediction accuracy"
echo ""

echo -e "${CYAN}============================================================================${NC}"
echo -e "${GREEN}Training complete! You can now stop the RunPod instance.${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""
