# Stock Analysis Platform

**Professional Quantitative Finance System with ML-Enhanced Portfolio Optimization**

A comprehensive platform for stock analysis, portfolio construction, and ML-powered trading forecasts. Built for both beginners and advanced users.

---

## Quick Start

### Launch the Dashboard (Easiest Way)

```bash
# 1. Navigate to the project
cd stock_analysis

# 2. Activate virtual environment
source venv/bin/activate

# 3. Launch the GUI
streamlit run dashboard/app.py
```

A browser window will open with the interactive dashboard!

### Command Line Analysis

```bash
# Analyze any stock
python run_portfolio.py AAPL

# With custom options
python run_portfolio.py TSLA --years 5 --benchmark QQQ
```

---

## Features

### Interactive Dashboard
- **Single Stock Analysis** - Comprehensive quantitative analysis with ML forecasts
- **Portfolio Builder** - Create optimized portfolios (long/short strategies)
- **Risk Analysis** - VaR, CVaR, drawdown, stress testing
- **Factor Analysis** - Fama-French factor exposure
- **Monte Carlo Simulations** - Thousands of future price paths
- **Security Pricing** - Valuation for stocks, bonds, options

### ML-Powered Forecasting
- **3-Model Ensemble** - LightGBM + Ridge + Momentum
- **20-Day Forecasts** - With confidence intervals
- **Trading Signals** - STRONG BUY/BUY/HOLD/SELL/STRONG SELL
- **Black-Litterman Integration** - ML-enhanced portfolio optimization

### Professional Tools
- **CVaR Optimization** - Tail risk management
- **Peer Discovery** - Auto-find similar stocks
- **Options Hedging** - Greeks-based strategies
- **Excel Reports** - Comprehensive workbooks
- **1000+ Stock Database** - Searchable tickers

---

## Installation

### Prerequisites
- **Python 3.11** (IMPORTANT: NOT 3.12+ due to compatibility issues)
- macOS, Linux, or Windows
- Internet connection

### Setup

1. **Clone repository**
   ```bash
   git clone <repository-url> stock_analysis
   cd stock_analysis
   ```

2. **Check Python version**
   ```bash
   python3 --version
   # Must show 3.11.x

   # If you have 3.12+, install 3.11:
   # pyenv install 3.11.7
   # pyenv local 3.11.7
   ```

3. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # OR
   venv\Scripts\activate     # Windows
   ```

4. **Install dependencies**
   ```bash
   # Basic requirements (dashboard + analysis)
   pip install -r requirements.txt

   # Optional: ML training requirements
   pip install -r requirements_training.txt
   ```

5. **Verify installation**
   ```bash
   streamlit run dashboard/app.py
   ```

---

## Documentation

### For Users (Non-Technical)
- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete beginner-friendly guide (START HERE!)
  - Dashboard walkthrough
  - Understanding results
  - Common use cases
  - Troubleshooting

### For Power Users
- **[USAGE.md](USAGE.md)** - Command line reference
- **[QUICK_START.md](QUICK_START.md)** - ML features and code examples
- **[HOW_TO_READ_SINGLE_STOCK_ANALYSIS.md](HOW_TO_READ_SINGLE_STOCK_ANALYSIS.md)** - Metric explanations

### For Developers
- **[CLAUDE.md](CLAUDE.md)** - System architecture and design decisions
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Implementation details
- **[READY_FOR_RUNPOD.md](READY_FOR_RUNPOD.md)** - Cloud deployment guide

---

## Quick Examples

### Example 1: Analyze Apple Stock

**Dashboard:**
1. Launch: `streamlit run dashboard/app.py`
2. Go to "Single Stock Analysis"
3. Enter `AAPL`
4. Click "Run Complete Analysis"

**Command Line:**
```bash
python run_portfolio.py AAPL --years 3
```

**What you get:**
- Current price and change
- ML 20-day forecast with confidence
- Valuation (undervalued/overvalued)
- Risk metrics (VaR, Sharpe, drawdown)
- Technical indicators (RSI, MACD)
- Final recommendation (LONG/SHORT)

### Example 2: Build Diversified Portfolio

**Dashboard:**
1. Go to "Portfolio from Single Stock"
2. Enter `AAPL`
3. Set peers: 5
4. Set capital: $100,000
5. Click "Build Portfolio"

**What you get:**
- 5-6 auto-discovered similar stocks
- Optimal allocation (long/short positions)
- Portfolio metrics (return, volatility, Sharpe)
- Hedging strategy
- Excel report download

### Example 3: Simulate Future Prices

**Dashboard:**
1. Go to "Advanced Monte Carlo"
2. Enter `TSLA`
3. Set simulations: 5000
4. Set horizon: 60 days
5. Click "Run Simulation"

**What you get:**
- Fan chart of 5000 possible futures
- Probability distribution
- Percentiles (5th, 25th, 50th, 75th, 95th)
- Worst/best case scenarios

---

## System Requirements

### Minimum
- Python 3.11
- 4GB RAM
- 2GB disk space
- Internet connection

### Recommended
- Python 3.11.7
- 8GB+ RAM (for large portfolios)
- SSD storage
- Stable internet

### For ML Training (Optional)
- 16GB+ RAM
- NVIDIA GPU with CUDA (RTX 3060 or better)
- 10GB+ disk space

---

## Project Structure

```
stock_analysis/
├── dashboard/              # Streamlit web interface
│   ├── app.py             # Main dashboard entry point
│   ├── pages/             # Dashboard pages
│   └── utils/             # UI utilities, stock database
├── ml_models/             # ML models and forecasting
│   ├── hybrid_model.py    # CNN-LSTM-Transformer
│   ├── practical_ensemble.py  # 3-model ensemble
│   ├── features.py        # Feature engineering
│   └── selection.py       # Stock selection
├── portfolio/             # Portfolio optimization
│   ├── cvar_allocator.py  # CVaR optimization
│   ├── black_litterman.py # ML-enhanced optimizer
│   └── peer_discovery.py  # Peer stock finder
├── single_stock/          # Individual stock analysis
│   ├── forecasting.py     # ARIMA forecasting
│   ├── valuation.py       # Valuation models
│   └── risk_metrics.py    # Risk calculations
├── training/              # ML training infrastructure
├── backtesting/           # Backtesting engine
├── data/                  # Data cache
└── reports/               # Generated reports
```

---

## Common Issues & Solutions

### "AttributeError: module 'pkgutil' has no attribute 'ImpImporter'"
**Solution:** Use Python 3.11 instead of 3.12+
```bash
pyenv install 3.11.7
pyenv local 3.11.7
```

### "Module not found: streamlit"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Dashboard won't load
**Solution:** Clear cache and restart
```bash
rm -rf ~/.streamlit/cache
streamlit run dashboard/app.py
```

### "No data found" for ticker
**Solution:**
- Check ticker spelling (use search dropdown in dashboard)
- Try again in a few minutes (Yahoo Finance rate limit)
- Use correct format: `AAPL` not `APPLE`

See **[USER_GUIDE.md](USER_GUIDE.md)** for complete troubleshooting guide.

---

## Performance

### Portfolio Construction
- Small (4 assets): < 3 seconds
- Medium (8-10 assets): < 15 seconds
- Large (15+ assets): < 30 seconds

### ML Forecasting
- Single stock forecast: < 2 seconds (cached)
- Portfolio of 10 stocks: < 10 seconds

### Monte Carlo Simulations
- 1,000 simulations: < 5 seconds
- 10,000 simulations: < 30 seconds

---

## Technology Stack

**Core:**
- Python 3.11
- PyTorch 2.1 (deep learning)
- LightGBM/XGBoost (gradient boosting)
- CVXPY (optimization)

**Data:**
- yfinance (market data)
- pandas/numpy (data processing)
- pyarrow (caching)

**UI:**
- Streamlit (dashboard)
- Plotly (interactive charts)
- matplotlib/seaborn (static plots)

**ML:**
- scikit-learn (traditional ML)
- statsmodels (time series)
- PyTorch Lightning (training)

---

## Contributing

This is currently a private project. If you have access:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Make your changes**
4. **Test thoroughly**
   ```bash
   pytest tests/ -v
   ```
5. **Submit a pull request**

---

## License

This project is proprietary. Unauthorized distribution or commercial use is prohibited.

---

## Disclaimer

**This platform is for educational and analytical purposes only.**

- NOT financial advice
- NOT guaranteed to be profitable
- NOT suitable for all investors
- Past performance ≠ future results

Always:
- Do your own research
- Consult a financial advisor
- Understand the risks
- Never invest more than you can afford to lose

---

## Support

- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
- **Technical Docs**: [CLAUDE.md](CLAUDE.md)
- **Issues**: Contact repository maintainer

---

## What's New

**Latest Updates:**
- Interactive Streamlit dashboard with 7 tools
- ML 3-model ensemble forecasting (LightGBM + Ridge + Momentum)
- Black-Litterman ML-enhanced optimization
- 1000+ stock searchable database
- VS Code dark theme
- CSV export functionality
- Comprehensive user documentation

See [PROJECT_COMPLETE_SUMMARY.md](PROJECT_COMPLETE_SUMMARY.md) for full changelog.

---

**Built with Python + PyTorch + Love**

Get started now: **[USER_GUIDE.md](USER_GUIDE.md)**
