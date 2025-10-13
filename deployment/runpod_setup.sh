#!/bin/bash
# RunPod GPU Instance Setup Script
# This script sets up a RunPod instance with all required dependencies
# for training the hybrid deep learning trading model

set -e  # Exit on error

# Configuration with environment variable support
CUDA_VERSION="${CUDA_VERSION:-cu118}"

echo "================================================"
echo "RunPod GPU Instance Setup for Hybrid Trading Model"
echo "================================================"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
if [ $? -eq 0 ]; then
    echo "✓ GPU detected successfully"
else
    echo "⚠ Warning: No GPU detected. Training will be slow."
fi

# Update system packages
echo "Updating system packages..."
apt-get update && apt-get upgrade -y
apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    python3-dev \
    libta-lib-dev \
    libhdf5-dev

# Install Python dependencies
echo "Installing PyTorch with CUDA support (version: $CUDA_VERSION)..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install TA-Lib (required for technical indicators)
echo "Installing TA-Lib..."
wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
pip install TA-Lib

# Install project requirements
echo "Installing project requirements..."
pip install -r requirements_training.txt

# Setup directories
echo "Setting up project directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p reports
mkdir -p cache

# Configure environment variables
echo "Configuring environment..."
export CUDA_LAUNCH_BLOCKING=1  # Better error messages
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE=online  # Enable Weights & Biases

# Download data if specified
if [ ! -z "$DOWNLOAD_DATA" ]; then
    # Validate TICKERS format
    if [[ -z "$TICKERS" ]]; then
        echo "Error: TICKERS not set but DOWNLOAD_DATA requested"
        exit 1
    fi

    echo "Downloading training data..."
    python data_fetching.py --tickers "$TICKERS" --start "$START_DATE" --end "$END_DATE"
    echo "Generating advanced features..."
    python utils/advanced_feature_engineering.py --base last_fetch
fi

# Verify installation with error handling
echo "Verifying installation..."
python -c "
import sys
try:
    import torch
    import gymnasium as gym
    import stable_baselines3
    import optuna
    import wandb
    import transformers
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA device count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
    print('All dependencies installed successfully!')
except ImportError as e:
    print(f'Error importing module: {e}', file=sys.stderr)
    sys.exit(1)
" || {
    echo "Installation verification failed!"
    exit 1
}

# Setup Weights & Biases (if API key provided)
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Setting up Weights & Biases..."
    wandb login "$WANDB_API_KEY"
fi

# Setup AWS credentials (if provided)
if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Configuring AWS credentials..."
    aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
    aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
    aws configure set default.region "${AWS_DEFAULT_REGION:-us-east-1}"
fi

echo "================================================"
echo "Setup complete! You can now run training with:"
echo "python training/train_hybrid.py --config training/config.yaml"
echo "================================================"