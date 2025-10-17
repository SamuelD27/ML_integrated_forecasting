#!/bin/bash
# RunPod Environment Setup Script
# ================================
# Prepares RunPod instance for ML training with 3x RTX 5090
#
# Usage: bash setup_runpod.sh

set -e  # Exit on error

echo "=========================================="
echo "RunPod Training Environment Setup"
echo "=========================================="
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --list-gpus
    echo ""
    nvidia-smi
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not found. Are you on a GPU instance?"
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p /workspace/data/training
mkdir -p /workspace/data/raw
mkdir -p /workspace/output/checkpoints
mkdir -p /workspace/output/logs
mkdir -p /workspace/code
echo "✅ Directories created"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python --version
if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "✅ Python 3.11+ detected"
else
    echo "❌ ERROR: Python 3.11+ required (found $(python --version))"
    exit 1
fi
echo ""

# Install pip if needed
echo "📦 Ensuring pip is up-to-date..."
python -m pip install --upgrade pip --quiet
echo "✅ pip updated"
echo ""

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
echo "✅ PyTorch installed"
echo ""

# Install remaining requirements
echo "📦 Installing Python dependencies..."
if [ -f "/workspace/code/runpod_setup/requirements_runpod.txt" ]; then
    pip install -r /workspace/code/runpod_setup/requirements_runpod.txt --quiet
    echo "✅ All dependencies installed"
else
    echo "⚠️  WARNING: requirements_runpod.txt not found at /workspace/code/runpod_setup/"
    echo "    Installing core dependencies manually..."
    pip install pytorch-lightning pandas numpy pyarrow yfinance lightgbm scikit-learn tensorboard tqdm pyyaml --quiet
    echo "✅ Core dependencies installed"
fi
echo ""

# Verify PyTorch CUDA
echo "🔥 Verifying PyTorch CUDA setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ ERROR: CUDA not available!')
    exit(1)
"
echo "✅ CUDA verification complete"
echo ""

# Test PyTorch Lightning
echo "⚡ Testing PyTorch Lightning..."
python -c "
import pytorch_lightning as pl
print(f'PyTorch Lightning version: {pl.__version__}')
print('✅ PyTorch Lightning working')
"
echo ""

# Display disk space
echo "💾 Disk space:"
df -h /workspace
echo ""

# Display memory
echo "🧠 Available memory:"
free -h
echo ""

# Environment summary
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📊 Environment Summary:"
echo "  • Python: $(python --version | cut -d' ' -f2)"
echo "  • PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  • CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
echo "  • GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""
echo "📁 Directory Structure:"
echo "  • Data: /workspace/data/"
echo "  • Output: /workspace/output/"
echo "  • Code: /workspace/code/"
echo ""
echo "🚀 Next Steps:"
echo "  1. Copy your code to /workspace/code/"
echo "  2. Run: bash run_training_pipeline.sh"
echo ""
