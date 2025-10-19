# Stock Analysis Platform - User Guide

**Welcome!** This guide will help you get started with the Stock Analysis Platform - a professional quantitative finance system for analyzing stocks, building portfolios, and generating ML-powered forecasts.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Using the Dashboard (GUI)](#using-the-dashboard-gui)
3. [Dashboard Features](#dashboard-features)
4. [Command Line Tools](#command-line-tools)
5. [Understanding the Results](#understanding-the-results)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)
8. [Tips & Best Practices](#tips--best-practices)

---

## Getting Started

### Prerequisites

**Required:**
- Python 3.11 (IMPORTANT: NOT Python 3.12+ - see troubleshooting)
- macOS, Linux, or Windows
- Internet connection (for downloading stock data)

**Optional:**
- Basic understanding of stocks and investing
- Familiarity with terminal/command line

### Installation

1. **Download or clone the repository**
   ```bash
   cd ~/Desktop
   git clone <repository-url> stock_analysis
   cd stock_analysis
   ```

2. **Set up Python environment**
   ```bash
   # Check your Python version (MUST be 3.11.x)
   python3 --version

   # If you have Python 3.12+, install 3.11 using pyenv:
   # pyenv install 3.11.7
   # pyenv local 3.11.7

   # Create virtual environment
   python3 -m venv venv

   # Activate virtual environment
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   # Install basic requirements (for dashboard)
   pip install -r requirements.txt

   # Optional: Install ML training requirements (only if you want to train models)
   pip install -r requirements_training.txt
   ```

4. **Verify installation**
   ```bash
   # This should show the help message
   python run_portfolio.py --help
   ```

You're ready to go!

---

## Using the Dashboard (GUI)

The **easiest way** to use this platform is through the interactive dashboard.

### Launch the Dashboard

```bash
# Make sure you're in the stock_analysis directory
cd ~/Desktop/stock_analysis

# Activate virtual environment
source venv/bin/activate

# Launch dashboard
streamlit run dashboard/app.py
```

**What happens:**
- A browser window will open automatically
- You'll see "Stock Analysis | Quantitative Trading Platform"
- The dashboard runs at `http://localhost:8501`

### Dashboard Navigation

The dashboard has **7 main features** accessible from the sidebar:

1. **Single Stock Analysis** - Analyze any stock (AAPL, TSLA, etc.)
2. **Portfolio from Galaxy** - Build portfolio from multiple stocks
3. **Portfolio from Single Stock** - Auto-discover similar stocks
4. **Portfolio Risk Analysis** - Deep dive into risk metrics
5. **Factor Analysis** - Fama-French factor exposure
6. **Advanced Monte Carlo** - Simulate future price paths
7. **Security Pricing** - Value stocks, bonds, options

---

## Dashboard Features

### 1. Single Stock Analysis

**Use this to:** Get comprehensive analysis on any stock

**Steps:**
1. Click "Single Stock Analysis" in sidebar
2. Enter ticker symbol (e.g., `AAPL`, `MSFT`, `TSLA`)
   - Not sure about the ticker? Use the searchable dropdown with 1000+ stocks!
3. Select time period (1y, 2y, 3y, 5y, or max)
4. Click "Run Complete Analysis"

**What you get:**
- **Current price & change** (real-time)
- **Valuation** - Is the stock overvalued or undervalued?
- **ML Forecast** - 20-day price prediction with confidence
- **Technical Analysis** - RSI, MACD, moving averages
- **Risk Metrics** - How risky is this stock?
- **Factor Analysis** - Fama-French factors
- **Final Recommendation** - Should you LONG or SHORT?

**Example:**
```
Ticker: AAPL
Period: 3y
→ Click "Run Complete Analysis"

Results:
✓ Current Price: $175.50 (+2.3%)
✓ ML Forecast (20 days): $182.30 (3.87% return, 68% confidence)
✓ Signal: STRONG BUY
✓ Valuation: UNDERVALUED (Target: $195.00)
✓ RSI: 58 (Neutral)
✓ Recommendation: LONG
```

### 2. Portfolio from Galaxy

**Use this to:** Build a custom portfolio from stocks you choose

**Steps:**
1. Click "Portfolio from Galaxy" in sidebar
2. Select multiple stocks (use Ctrl/Cmd + Click)
   - Choose 4-10 stocks for best results
3. Set your capital (e.g., $100,000)
4. Set target long/short ratio (e.g., 0.6 = 60% long, 40% short)
5. Click "Build Portfolio"

**What you get:**
- **Optimal allocation** - How much of each stock to buy/short
- **Portfolio metrics** - Expected return, volatility, Sharpe ratio
- **Risk breakdown** - VaR, CVaR, max drawdown
- **Hedging strategy** - ETF hedges (if applicable)
- **Excel report** - Download detailed workbook

**Example:**
```
Stocks: AAPL, MSFT, GOOGL, AMZN, TSLA
Capital: $100,000
Long/Short Ratio: 0.6

Results:
✓ AAPL: $25,000 (LONG)
✓ MSFT: $20,000 (LONG)
✓ GOOGL: $15,000 (LONG)
✓ AMZN: -$10,000 (SHORT)
✓ TSLA: -$5,000 (SHORT)

Expected Return: 12.5% annually
Volatility: 18.3%
Sharpe Ratio: 0.68
```

### 3. Portfolio from Single Stock

**Use this to:** Start with one stock and auto-discover similar stocks

**Steps:**
1. Click "Portfolio from Single Stock" in sidebar
2. Enter starting ticker (e.g., `AAPL`)
3. Set number of peers (4-10 recommended)
4. Set your capital
5. Click "Build Portfolio"

**What you get:**
- **Auto-discovered peers** - Stocks similar to your starting stock
- **Optimal allocation** - Diversified portfolio
- **ML-enhanced optimization** - Uses ML forecasts for better weights
- **Full portfolio metrics**
- **Excel report**

**Example:**
```
Starting Stock: AAPL
Number of Peers: 5
Capital: $100,000

Auto-discovered:
✓ MSFT (Peer score: 0.92)
✓ GOOGL (Peer score: 0.88)
✓ NVDA (Peer score: 0.85)
✓ META (Peer score: 0.82)
✓ AMZN (Peer score: 0.79)

Portfolio built with ML-enhanced optimization!
```

### 4. Portfolio Risk Analysis

**Use this to:** Deep dive into portfolio risk

**Steps:**
1. Click "Portfolio Risk Analysis" in sidebar
2. Enter tickers and weights (or use equal weights)
3. Set time period
4. Click "Analyze Risk"

**What you get:**
- **VaR (Value at Risk)** - Maximum expected loss
- **CVaR (Conditional VaR)** - Average loss in worst scenarios
- **Stress testing** - How portfolio performs in crashes
- **Correlation matrix** - How stocks move together
- **Risk decomposition** - Which stocks contribute most risk

### 5. Factor Analysis

**Use this to:** Understand factor exposure (Fama-French)

**Steps:**
1. Click "Factor Analysis" in sidebar
2. Enter ticker
3. Set time period
4. Click "Analyze Factors"

**What you get:**
- **Market Beta** - Sensitivity to overall market
- **Size Factor (SMB)** - Small vs large cap exposure
- **Value Factor (HML)** - Value vs growth exposure
- **Momentum Factor (MOM)** - Momentum exposure
- **Alpha** - Excess return after factors

### 6. Advanced Monte Carlo

**Use this to:** Simulate thousands of future price paths

**Steps:**
1. Click "Advanced Monte Carlo" in sidebar
2. Enter ticker
3. Set number of simulations (1000-10000)
4. Set forecast horizon (days)
5. Click "Run Simulation"

**What you get:**
- **Price path fan chart** - Possible future prices
- **Probability distribution** - Likelihood of outcomes
- **Percentiles** - 5th, 25th, 50th, 75th, 95th
- **Risk metrics** - Worst-case scenarios

### 7. Security Pricing

**Use this to:** Value different types of securities

**Steps:**
1. Click "Security Pricing" in sidebar
2. Choose security type (stock, bond, option)
3. Enter parameters
4. Click "Calculate Value"

**What you get:**
- **Fair value** - Theoretical price
- **Greeks** (for options) - Delta, gamma, theta, vega
- **Sensitivity analysis** - How value changes with inputs

---

## Command Line Tools

If you prefer command line or want to automate tasks:

### Quick Stock Analysis

**Simplest way to analyze a stock:**

```bash
python run_portfolio.py AAPL
```

This will:
- Analyze Apple (AAPL)
- Use 2 years of historical data
- Compare against S&P 500
- Generate comprehensive Excel report in `reports/` folder

**With options:**

```bash
# 5 years of data
python run_portfolio.py TSLA --years 5

# Custom benchmark (NASDAQ instead of S&P 500)
python run_portfolio.py MSFT --benchmark QQQ

# Both options
python run_portfolio.py NVDA --years 3 --benchmark QQQ
```

### ML-Enhanced Portfolio

**Build portfolio with ML forecasts:**

```bash
python portfolio_creation_ml.py AAPL --capital 100000 --rrr 0.6 --enable-ml
```

**Parameters:**
- `--capital`: Your investment amount (default: 100000)
- `--rrr`: Long/short ratio, 0-1 (default: 0.6 = 60% long)
- `--enable-ml`: Use ML forecasts for optimization
- `--ml-top-n`: Number of stocks in portfolio (default: 15)

---

## Understanding the Results

### ML Forecast Section

When you see an ML forecast, here's what it means:

```
Current Price: $175.50
20-Day Forecast: $182.30
Expected Return: +3.87%
Confidence: 68%
95% Confidence Interval: $170.20 - $194.40
Signal: STRONG BUY
```

**Interpretation:**
- **Current Price**: Latest price from market
- **20-Day Forecast**: Predicted price 20 days from now
- **Expected Return**: Percentage gain/loss expected
- **Confidence**: How confident the model is (0-100%)
  - >70%: High confidence, strong signal
  - 40-70%: Moderate confidence
  - <40%: Low confidence, be cautious
- **95% CI**: Range where price will likely be (95% probability)
- **Signal**: Trading recommendation
  - STRONG BUY: >5% return + >70% confidence
  - BUY: >2% return + >50% confidence
  - HOLD: -2% to +2% return
  - SELL: <-2% return + >50% confidence
  - STRONG SELL: <-5% return + >70% confidence

### Risk Metrics

```
Sharpe Ratio: 1.45
Sortino Ratio: 2.12
Max Drawdown: -18.5%
VaR (95%): -$2,450
CVaR (95%): -$3,890
Beta: 1.12
```

**Interpretation:**
- **Sharpe Ratio**: Risk-adjusted return
  - >1.0: Good
  - >2.0: Very good
  - >3.0: Excellent
- **Sortino Ratio**: Like Sharpe but only downside risk
  - Usually higher than Sharpe
  - >2.0 is good
- **Max Drawdown**: Worst peak-to-trough loss
  - Lower is better
  - -20% to -30% is typical for stocks
- **VaR (95%)**: Maximum expected loss (95% of time)
  - "$2,450" means you could lose up to $2,450
- **CVaR (95%)**: Average loss in worst 5% of cases
  - Always worse than VaR
- **Beta**: Market sensitivity
  - 1.0 = moves with market
  - >1.0 = more volatile than market
  - <1.0 = less volatile than market

### Portfolio Allocation

```
AAPL: $25,000 (25.0%) - LONG
MSFT: $20,000 (20.0%) - LONG
GOOGL: $15,000 (15.0%) - LONG
AMZN: -$10,000 (-10.0%) - SHORT
TSLA: -$5,000 (-5.0%) - SHORT
Cash: $5,000 (5.0%)
```

**Interpretation:**
- **Positive amounts**: LONG (buying the stock)
  - You profit if price goes UP
- **Negative amounts**: SHORT (selling/shorting the stock)
  - You profit if price goes DOWN
- **Cash**: Uninvested capital (safety buffer)

---

## Common Use Cases

### Use Case 1: "Should I buy this stock?"

**Steps:**
1. Launch dashboard: `streamlit run dashboard/app.py`
2. Go to "Single Stock Analysis"
3. Enter ticker (e.g., `AAPL`)
4. Select period (recommend 3y or 5y)
5. Click "Run Complete Analysis"
6. Look at:
   - **ML Forecast**: Is it BUY or SELL?
   - **Valuation**: Is it undervalued?
   - **Final Recommendation**: LONG or SHORT?

**Decision:**
- If 2+ of 3 say BUY/LONG → Consider buying
- If 2+ of 3 say SELL/SHORT → Avoid or wait
- If mixed signals → Do more research

### Use Case 2: "Build me a diversified portfolio"

**Steps:**
1. Launch dashboard
2. Go to "Portfolio from Single Stock"
3. Enter your favorite stock (e.g., `AAPL`)
4. Set peers to 5-10
5. Set your capital
6. Click "Build Portfolio"
7. Download Excel report

**Result:**
- Diversified portfolio with similar stocks
- Optimal weights (not equal)
- Risk metrics
- Hedging strategy

### Use Case 3: "How risky is my portfolio?"

**Steps:**
1. Launch dashboard
2. Go to "Portfolio Risk Analysis"
3. Enter your current holdings (tickers + weights)
4. Set time period
5. Click "Analyze Risk"

**Look for:**
- **VaR/CVaR**: How much could you lose?
- **Max Drawdown**: Worst historical loss
- **Correlation**: Are stocks too similar?
- **Stress Test**: Performance in crashes

### Use Case 4: "What could happen to this stock?"

**Steps:**
1. Launch dashboard
2. Go to "Advanced Monte Carlo"
3. Enter ticker
4. Set simulations to 5000-10000
5. Set horizon (e.g., 60 days)
6. Click "Run Simulation"

**Result:**
- Fan chart of possible futures
- Probability of gains/losses
- Worst-case scenarios
- Best-case scenarios

### Use Case 5: "Get a quick report (no GUI)"

**Command line:**
```bash
python run_portfolio.py TSLA --years 3
```

**Result:**
- Analysis runs in terminal
- Excel report saved to `reports/TSLA_analysis_<timestamp>.xlsx`
- Open Excel file to see all details

---

## Troubleshooting

### Problem: "AttributeError: module 'pkgutil' has no attribute 'ImpImporter'"

**Cause:** You're using Python 3.12+ (not supported)

**Solution:** Install and use Python 3.11

```bash
# Check version
python3 --version

# Install Python 3.11 with pyenv
pyenv install 3.11.7
pyenv local 3.11.7

# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: "Module not found: streamlit" or "Module not found: lightgbm"

**Cause:** Missing dependencies

**Solution:** Install requirements

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# If still missing:
pip install streamlit lightgbm yfinance pandas numpy scipy scikit-learn
```

### Problem: Dashboard won't load or shows errors

**Cause:** Cache issues or port conflict

**Solution:**

```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Try different port
streamlit run dashboard/app.py --server.port 8502

# Or restart from scratch
pkill -f streamlit
streamlit run dashboard/app.py
```

### Problem: "No data found" or "Download failed"

**Cause:** Invalid ticker or Yahoo Finance issues

**Solution:**

1. **Check ticker symbol**
   - Use correct format: `AAPL` (not `APPLE`)
   - For international stocks: `9992.HK`, `VOD.L`, etc.

2. **Try again in a few minutes**
   - Yahoo Finance occasionally has rate limits

3. **Use search feature**
   - In dashboard, use searchable dropdown with 1000+ pre-validated tickers

### Problem: Slow performance

**Cause:** First run or large dataset

**Solution:**

- **First run is always slower** (building cache)
- **Subsequent runs are much faster** (using cache)
- **Reduce time period** (use 2y instead of 5y)
- **Reduce simulations** (use 1000 instead of 10000 for Monte Carlo)

**Cache info:**
- Cache location: `data/` folder
- Cache TTL: 24 hours
- Clear cache: Delete `data/last_fetch.parquet`

### Problem: Excel file won't open

**Cause:** Missing openpyxl or xlsxwriter

**Solution:**

```bash
pip install openpyxl xlsxwriter
```

### Problem: "CUDA out of memory" (when training models)

**Cause:** GPU memory overflow

**Solution:**

- This only happens if you're training ML models
- Reduce batch size in `training/config.yaml`
- Enable gradient accumulation
- Or use CPU only (slower but works)

---

## Tips & Best Practices

### For Analysis

1. **Use 2-5 years of data**
   - 1 year: Too short, noisy
   - 2-3 years: Good balance
   - 5+ years: More robust, slower

2. **Check multiple signals**
   - Don't rely on just ML forecast
   - Look at valuation, technicals, risk
   - 2+ agreeing signals = stronger conviction

3. **Understand confidence levels**
   - High confidence (>70%): Trust the forecast
   - Low confidence (<40%): Be cautious
   - Model disagreement = uncertainty

### For Portfolio Building

1. **Diversify (5-10 stocks)**
   - Too few (<4): Not diversified
   - Good range (5-10): Balanced
   - Too many (>15): Diminishing returns

2. **Use ML-enhanced optimization**
   - In "Portfolio from Single Stock", ML is automatic
   - In "Portfolio from Galaxy", enable ML mode
   - ML forecasts improve allocation

3. **Check correlation**
   - If stocks are highly correlated (>0.8), reduce
   - Aim for mix of correlations
   - Diversification works when correlations are <0.7

4. **Set appropriate long/short ratio**
   - Conservative: 0.8-1.0 (mostly long)
   - Moderate: 0.6-0.7 (balanced)
   - Aggressive: 0.4-0.5 (more short)

### For Risk Management

1. **Know your VaR/CVaR**
   - VaR: Typical worst case
   - CVaR: Average of worst cases
   - Size positions so you can handle CVaR

2. **Monitor max drawdown**
   - If max drawdown is -30%, can you handle -30% loss?
   - Historical drawdown doesn't mean it can't go worse

3. **Use stop losses**
   - Set based on VaR or technical levels
   - Don't let losses run unchecked

4. **Rebalance regularly**
   - Weekly/monthly for active strategy
   - Quarterly for longer-term
   - After major market moves

### General Tips

1. **Start small**
   - Learn the platform with small amounts
   - Understand the metrics
   - Build confidence

2. **Export reports**
   - All dashboards have Excel export
   - Save reports for your records
   - Track performance over time

3. **Compare strategies**
   - Traditional vs ML-enhanced
   - Equal weight vs optimized
   - See what works better

4. **Stay updated**
   - Clear cache daily for fresh data
   - Re-run analysis weekly
   - Markets change, so should your analysis

---

## Getting Help

### Documentation

- **This guide**: General usage
- **USAGE.md**: Original quick start
- **QUICK_START.md**: ML features
- **CLAUDE.md**: Technical architecture (for developers)

### Common Questions

**Q: How accurate are the ML forecasts?**
A: Directional accuracy is typically 55-65% (better than random). Use confidence levels to gauge reliability.

**Q: Can I use this for crypto?**
A: Not currently. Platform uses Yahoo Finance which focuses on stocks.

**Q: Can I add my own stocks to the search?**
A: Yes! Edit `dashboard/utils/stock_database_mega.py` and add your ticker.

**Q: How often should I re-run analysis?**
A: Weekly for active trading, monthly for long-term investing.

**Q: Can I backtest strategies?**
A: Yes, but requires code. See `backtesting/backtest_engine.py` for developers.

**Q: Is this financial advice?**
A: NO. This is an analytical tool. Always do your own research and consult a financial advisor.

---

## Next Steps

Now that you understand the platform:

1. **Launch the dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

2. **Try analyzing a stock you know**
   - Start with something familiar (AAPL, MSFT, etc.)
   - Explore all sections
   - Understand the metrics

3. **Build a test portfolio**
   - Use "Portfolio from Single Stock"
   - Try different parameters
   - Download the Excel report

4. **Compare strategies**
   - Traditional vs ML-enhanced
   - See which performs better
   - Learn from the differences

5. **Read more documentation**
   - QUICK_START.md for ML details
   - USAGE.md for command line options
   - HOW_TO_READ_SINGLE_STOCK_ANALYSIS.md for metric explanations

---

## Disclaimer

This platform is for educational and analytical purposes only. It is NOT financial advice. Always:

- Do your own research
- Consult with financial professionals
- Understand the risks
- Never invest more than you can afford to lose
- Past performance does not guarantee future results

**Trade responsibly!**

---

**Happy analyzing!** If you have questions, check the documentation or reach out to the repository maintainer.
