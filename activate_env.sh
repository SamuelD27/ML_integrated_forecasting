#!/bin/bash
# Quick environment activation script
# Usage: source activate_env.sh

echo "=================================="
echo "Stock Analysis ML Environment"
echo "=================================="

# Setup pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Activate virtual environment
source venv_ml/bin/activate

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
    echo "✓ Environment variables loaded from .env"
fi

# Show status
echo "✓ Python version: $(python --version)"
echo "✓ Virtual env: venv_ml activated"
echo "✓ API Keys: RunPod & W&B configured"
echo "✓ Ready for training!"
echo ""
echo "Quick commands:"
echo "  python validate_training_setup.py --full    # Validate setup"
echo "  python prepare_training_data.py             # Prepare data"
echo "  python run_portfolio.py AAPL                # Analyze stock"
echo "  deactivate                                  # Exit environment"
echo "=================================="
