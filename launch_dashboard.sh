#!/bin/bash
# Launch Streamlit Dashboard
# ==========================

echo "🚀 Launching Quantitative Finance Dashboard..."
echo ""

# Activate virtual environment
source venv_ml/bin/activate

# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not installed. Installing..."
    pip install streamlit plotly yfinance vectorbt
fi

# Launch dashboard
echo "✅ Starting dashboard at http://localhost:8501"
echo ""
echo "📊 Available pages:"
echo "   1. Portfolio Builder - Build long/short portfolios"
echo "   2. Factor Analysis - Fama-French regression"
echo "   3. Backtest Runner - Test strategies"
echo "   4. Performance Monitor - Track metrics"
echo "   5. Risk Analytics - HRP optimization"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd dashboard
streamlit run app.py
