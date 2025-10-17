#!/bin/bash
# ============================================================================
# B200 TRAINING - SIMPLIFIED (runs from current directory)
# ============================================================================
# Usage: cd /your/repo/runpod_setup && bash run_b200_training.sh
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get the repo root (parent of runpod_setup)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo ""
echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}                  NVIDIA B200 TRAINING${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo -e "${YELLOW}Repo: ${REPO_ROOT}${NC}"
echo -e "${YELLOW}Dataset: 500 S&P 500 tickers × 15 years${NC}"
echo -e "${YELLOW}Model: 100M parameters${NC}"
echo -e "${YELLOW}Time: 8-10 hours${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""

# Check GPU
echo -e "${BLUE}[1/3]${NC} Checking B200 GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install dependencies
echo -e "${BLUE}[2/3]${NC} Installing dependencies (if needed)..."
pip install torch pytorch-lightning pandas numpy pyarrow yfinance \
            lightgbm scikit-learn tensorboard tqdm pyyaml --quiet
echo -e "${GREEN}✅ Dependencies ready${NC}"
echo ""

# Fetch data
echo -e "${BLUE}[3/3]${NC} Fetching S&P 500 data..."
echo -e "${YELLOW}⏱️  This will take 30-45 minutes${NC}"
echo ""

cd "${REPO_ROOT}"
python runpod_setup/fetch_sp500_data.py

echo ""
echo -e "${GREEN}✅ Data ready!${NC}"
echo ""

# Train
echo -e "${CYAN}Starting training...${NC}"
echo -e "${YELLOW}⏱️  This will take 8-10 hours${NC}"
echo ""
echo "To monitor in another terminal:"
echo "  watch -n 1 nvidia-smi"
echo ""

python runpod_setup/train_b200.py

echo ""
echo -e "${GREEN}✅ TRAINING COMPLETE!${NC}"
echo ""
echo "Results saved to: /workspace/output/"
echo ""
