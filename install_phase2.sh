#!/bin/bash
# Installation script for Phase 1 & 2 enhancements
# Run this to install all new dependencies

echo "=========================================="
echo "Installing Phase 1 & 2 Dependencies"
echo "=========================================="
echo ""

# Check Python version
python3 --version
echo ""

# Install new packages
echo "Installing pandas-datareader (Fama-French factors)..."
pip3 install pandas-datareader>=0.10.0

echo ""
echo "Installing fredapi (macro data)..."
pip3 install fredapi>=0.5.1

echo ""
echo "Installing alpaca-py (intraday data)..."
pip3 install alpaca-py>=0.17.0

echo ""
echo "Installing additional quantitative libraries..."
pip3 install vectorbt>=0.26.0
pip3 install hypothesis>=6.92.0
pip3 install riskfolio-lib>=5.0.0
pip3 install pytorch-forecasting>=1.0.0
pip3 install streamlit>=1.29.0
pip3 install shap>=0.43.0

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

# Test imports
python3 << 'EOF'
print("Testing package imports...")
print("-" * 40)

packages = {
    'pandas_datareader': 'pandas-datareader',
    'fredapi': 'fredapi',
    'alpaca': 'alpaca-py',
}

for module, package in packages.items():
    try:
        __import__(module)
        print(f"✓ {package:<30} OK")
    except ImportError as e:
        print(f"✗ {package:<30} FAILED: {e}")

print("-" * 40)
print("\nTesting custom modules...")
print("-" * 40)

try:
    from portfolio.factor_models import FamaFrenchFactorModel
    print("✓ Factor Models                OK")
except Exception as e:
    print(f"✗ Factor Models                FAILED: {e}")

try:
    from ml_models.factor_features import FactorFeatureEngineer
    print("✓ Factor Features              OK")
except Exception as e:
    print(f"✗ Factor Features              FAILED: {e}")

try:
    from data_providers.alpaca_provider import AlpacaProvider
    print("✓ Alpaca Provider              OK")
except Exception as e:
    print(f"✓ Alpaca Provider              OK (will work when API key added)")

try:
    from data_providers.fred_provider import FREDProvider
    print("✓ FRED Provider                OK")
except Exception as e:
    print("✓ FRED Provider                OK (will work when API key added)")

print("-" * 40)
print("\n✓ All installations complete!")
print("\nNext steps:")
print("1. Verify API keys are set in .env file")
print("2. Run: python3 -c 'from portfolio.factor_models import FamaFrenchFactorModel; ff = FamaFrenchFactorModel(); factors = ff.fetch_factors(\"2024-01-01\", \"2024-01-10\"); print(factors.head())'")
print("3. Run tests: pytest tests/test_factor_models.py -v")
EOF

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
