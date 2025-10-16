#!/bin/bash
# RunPod Setup Commands - Python 3.11 Installation + Training Setup
# Copy and paste these commands one section at a time into your RunPod terminal

echo "=================================="
echo "RunPod Training Setup - Python 3.11"
echo "=================================="

# STEP 1: Install Python 3.11 (RunPod comes with 3.12.3 which won't work)
echo ""
echo "STEP 1: Installing Python 3.11..."
apt update
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.11 python3.11-venv python3.11-dev

# Verify Python 3.11 is installed
python3.11 --version

# STEP 2: Clone your repository
echo ""
echo "STEP 2: Cloning repository..."
cd /workspace
git clone https://github.com/SamuelD27/ML_integrated_forecasting.git
cd ML_integrated_forecasting

# STEP 3: Create virtual environment with Python 3.11
echo ""
echo "STEP 3: Creating virtual environment with Python 3.11..."
python3.11 -m venv venv_runpod
source venv_runpod/bin/activate

# Verify we're using Python 3.11 in venv
python --version  # Should show Python 3.11.x

# STEP 4: Upgrade pip
echo ""
echo "STEP 4: Upgrading pip..."
pip install --upgrade pip

# STEP 5: Install PyTorch with CUDA support
echo ""
echo "STEP 5: Installing PyTorch 2.1.0 with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# STEP 6: Install all training requirements (using simplified RunPod requirements)
echo ""
echo "STEP 6: Installing training dependencies..."
pip install -r requirements_runpod.txt

# STEP 7: Set up environment variables
echo ""
echo "STEP 7: Setting up API keys..."
export WANDB_API_KEY=4e6f46e0f12cb3edced9a0b4d94ba315c4be7b59
export PROJECT_NAME=stock_analysis
export WANDB_PROJECT=hybrid-trading-model

# Verify wandb login
wandb login 4e6f46e0f12cb3edced9a0b4d94ba315c4be7b59

# STEP 8: Verify CUDA is available
echo ""
echo "STEP 8: Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# STEP 9: Prepare training data (if needed)
echo ""
echo "STEP 9: Preparing training data..."
# python prepare_training_data.py  # Uncomment if you need to fetch fresh data

# STEP 10: Set up Python path (CRITICAL - fixes import errors)
echo ""
echo "STEP 10: Setting up Python path..."
export PYTHONPATH="/workspace/ML_integrated_forecasting:${PYTHONPATH}"
echo "✓ PYTHONPATH configured"

# STEP 11: Launch training!
echo ""
echo "STEP 11: Starting training..."
echo "=================================="
python training/train_hybrid.py --config training/config.yaml

# MONITORING:
# - Weights & Biases: https://wandb.ai/your-username/hybrid-trading-model
# - TensorBoard: tensorboard --logdir=checkpoints/tensorboard --host=0.0.0.0 --port=6006
# - Check logs: tail -f training/training.log
