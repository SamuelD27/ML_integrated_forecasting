# Technology Stack Reference

## Core ML Framework

### PyTorch
- Version: Latest stable (2.x+)
- Purpose: Primary deep learning framework
- Key features used:
  - Mixed precision training (automatic TF32 on B200)
  - Distributed data parallelism across 8 GPUs
  - Gradient checkpointing for memory efficiency
  - Custom loss functions and optimizers

**Common imports**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
```

**GPU management**:
```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU count: {torch.cuda.device_count()}")

# Monitor memory
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
```

## Model Architectures

### Temporal Fusion Transformer (TFT)
- Library: Custom implementation or PyTorch Forecasting
- Architecture: Multi-head attention + LSTM + gating mechanisms
- Input: Multi-variate time series
- Output: Multi-horizon probabilistic forecasts
- Key parameters:
  - Hidden size: 256-512
  - Attention heads: 4-8
  - Dropout: 0.1-0.3
  - LSTM layers: 2-4

**Training config**:
```python
config = {
    'hidden_size': 512,
    'attention_heads': 8,
    'lstm_layers': 2,
    'dropout': 0.2,
    'learning_rate': 1e-4,
    'batch_size': 256,
    'epochs': 100,
    'gradient_clip': 1.0,
    'early_stopping_patience': 10
}
```

### LSTM Models
- Purpose: Baseline and ensemble member
- Variants: Vanilla LSTM, Bidirectional LSTM, Stacked LSTM
- Architecture: 2-4 LSTM layers + fully connected output
- Best for: Capturing temporal dependencies

### CNN-LSTM-Transformer Hybrid
- Architecture:
  1. Conv1D layers: Extract local patterns (3-5 layers)
  2. LSTM layers: Sequence modeling (2-3 layers)
  3. Transformer: Long-range dependencies (4-8 layers)
- Current: 100M parameters
- Target: 500M+ parameters
- Scaling strategy: Increase layer widths and depths proportionally

## Hyperparameter Optimization

### Optuna
- Purpose: Automated hyperparameter tuning
- Search strategies: TPE (Tree-structured Parzen Estimator)
- Parallelization: Multi-GPU trials

**Example usage**:
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int('hidden_size', 128, 1024, step=128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Train model with suggested params
    model = build_model(hidden_size, dropout)
    val_loss = train(model, lr)
    
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, n_jobs=8)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

## Data Stack

### Data Sources

#### yfinance
- Purpose: Primary price and volume data
- Rate limit: ~2000 requests/hour
- Data: OHLCV, splits, dividends
- Limitations: Occasional missing data, rate limits

**Usage**:
```python
import yfinance as yf

ticker = yf.Ticker("AAPL")
hist = ticker.history(period="max")  # All available history
info = ticker.info  # Company info
```

#### Financial Modeling Prep API
- Purpose: Fundamental data
- Data types:
  - Income statements (quarterly, annual)
  - Balance sheets
  - Cash flow statements
  - Financial ratios (P/E, P/B, ROE, etc.)
- Rate limit: Depends on plan (typically 250-750 calls/day)

**Usage**:
```python
import requests

def get_fundamentals(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
    params = {'apikey': api_key, 'limit': 100}
    response = requests.get(url, params=params)
    return response.json()
```

#### NewsAPI
- Purpose: News articles for sentiment analysis
- Coverage: Major financial news sources
- Rate limit: 100 requests/day (free tier)

#### Reddit API (PRAW)
- Purpose: Social sentiment from r/wallstreetbets, r/investing
- Data: Post titles, comments, scores
- Rate limit: 60 requests/minute

#### FRED (Federal Reserve Economic Data)
- Purpose: Macro indicators
- Key series:
  - Interest rates (DFF, DGS10)
  - GDP (GDPC1)
  - Inflation (CPIAUCSL)
  - Unemployment (UNRATE)
- No rate limit (government API)

**Usage**:
```python
from fredapi import Fred

fred = Fred(api_key='your_key')
data = fred.get_series('DFF')  # Federal funds rate
```

### Data Processing

#### pandas
- Purpose: Data manipulation and analysis
- Key operations:
  - Time series resampling
  - Rolling calculations
  - Merging multiple data sources
  - Handling missing values

**Common patterns**:
```python
import pandas as pd

# Resample to daily frequency
df = df.resample('D').ffill()

# Calculate returns
df['returns'] = df['close'].pct_change()

# Rolling statistics
df['sma_20'] = df['close'].rolling(20).mean()
df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)

# Handle missing values
df = df.fillna(method='ffill').dropna()
```

#### NumPy
- Purpose: Numerical computations
- Use cases:
  - Array operations
  - Statistical calculations
  - Linear algebra (factor models)

### Data Storage

#### Parquet
- Library: pyarrow or fastparquet
- Compression: Snappy (fast) or Gzip (high compression)
- Benefits:
  - Columnar storage (fast reads)
  - Efficient compression (70-80% reduction)
  - Schema preservation
  - Fast filtering

**Usage**:
```python
# Write
df.to_parquet('data.parquet', compression='snappy', engine='pyarrow')

# Read
df = pd.read_parquet('data.parquet')

# Read specific columns
df = pd.read_parquet('data.parquet', columns=['date', 'close', 'volume'])

# Read with filter
df = pd.read_parquet('data.parquet', filters=[('date', '>=', '2020-01-01')])
```

## Sentiment Analysis

### FinBERT
- Purpose: Financial sentiment analysis from text
- Model: BERT fine-tuned on financial news
- Output: Positive/Negative/Neutral scores
- Use cases: News articles, Reddit posts, earnings calls

**Usage**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # scores: [positive, negative, neutral]
    return scores.detach().numpy()[0]
```

## Portfolio Optimization

### Hierarchical Risk Parity (HRP)
- Library: PyPortfolioOpt or custom implementation
- Purpose: Risk-based portfolio allocation
- Benefits: More stable than mean-variance, robust to estimation error

**Usage**:
```python
from pypfopt import HRPOpt

returns = pd.DataFrame(...)  # Asset returns
hrp = HRPOpt(returns)
weights = hrp.optimize()
```

### Mean-Variance Optimization
- Library: cvxpy or PyPortfolioOpt
- Methods: Efficient frontier, max Sharpe ratio
- Constraints: Long-only, sector limits, turnover

### Risk Models

#### GARCH
- Library: arch
- Purpose: Volatility forecasting
- Variants: GARCH(1,1), EGARCH, GJR-GARCH

**Usage**:
```python
from arch import arch_model

model = arch_model(returns, vol='Garch', p=1, q=1)
results = model.fit()
forecast = results.forecast(horizon=5)
```

#### VaR and CVaR
- Purpose: Risk measurement
- Calculation methods:
  - Historical simulation
  - Parametric (normal distribution)
  - Monte Carlo simulation

```python
def calculate_var(returns, confidence=0.95):
    """Historical VaR"""
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_cvar(returns, confidence=0.95):
    """Conditional VaR (Expected Shortfall)"""
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()
```

## Backtesting

### vectorbt
- Purpose: Fast vectorized backtesting
- Features:
  - Portfolio simulation
  - Performance metrics
  - Visualization

**Usage**:
```python
import vectorbt as vbt

# Backtest simple strategy
price = vbt.YFData.download('AAPL').get('Close')
fast_ma = vbt.MA.run(price, 10)
slow_ma = vbt.MA.run(price, 50)

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

portfolio = vbt.Portfolio.from_signals(price, entries, exits)
print(portfolio.stats())
```

## Dashboard

### Streamlit
- Purpose: Interactive web dashboard
- Features:
  - Real-time updates
  - Interactive charts
  - User inputs for analysis
  - File uploads

**Common components**:
```python
import streamlit as st
import plotly.express as px

# Layout
st.title("ML Trading Dashboard")
ticker = st.selectbox("Select Ticker", tickers)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Return", f"{return_pct:.2f}%")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Max Drawdown", f"{max_dd:.2f}%")

# Interactive chart
fig = px.line(df, x='date', y='portfolio_value')
st.plotly_chart(fig)
```

## Infrastructure

### RunPod
- GPUs: 8x NVIDIA B200 (24GB VRAM each = 192GB total)
- CPU: High core count for data processing
- Storage: Limited, requires optimization
- Network: Bandwidth for data downloads

**Key commands**:
```bash
# Check GPU status
nvidia-smi

# Monitor GPU utilization
nvidia-smi dmon -s u

# Check storage
df -h

# Process monitoring
htop

# Network speed test
speedtest-cli
```

### Docker (if used)
- Base image: pytorch/pytorch:latest or nvidia/cuda:latest
- Mounted volumes: Data, checkpoints, code
- Environment variables: API keys, paths

## Development Tools

### Version Control
- Git for code versioning
- .gitignore: Exclude data/, checkpoints/, __pycache__/

### Environment Management
- conda or venv for Python environments
- requirements.txt or environment.yml
- Keep separate dev and production environments

### Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Training started")
```

### Monitoring
- TensorBoard for training metrics
- Weights & Biases (wandb) for experiment tracking
- Custom dashboards for production monitoring

**TensorBoard usage**:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment1')

for epoch in range(epochs):
    # Training
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

writer.close()
```

**View**: `tensorboard --logdir=runs`

## Common Library Versions

```txt
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
optuna>=3.0.0
yfinance>=0.2.0
requests>=2.28.0
pyarrow>=12.0.0
streamlit>=1.25.0
plotly>=5.14.0
scikit-learn>=1.3.0
transformers>=4.30.0
arch>=5.6.0
vectorbt>=0.25.0
fredapi>=0.5.0
pypfopt>=1.5.0
```

## Performance Optimization Tips

### Training Speed
1. Use DataLoader with num_workers=8+
2. Enable pin_memory=True
3. Increase batch size to GPU limit
4. Use mixed precision training
5. Profile to find bottlenecks

### Memory Efficiency
1. Gradient checkpointing for large models
2. Clear cache periodically: `torch.cuda.empty_cache()`
3. Delete unnecessary tensors: `del tensor`
4. Use in-place operations where possible
5. Reduce precision (fp16/bf16 instead of fp32)

### Data Loading
1. Preprocess data once, save to Parquet
2. Use columnar storage (Parquet, not CSV)
3. Load only needed columns
4. Implement data caching
5. Parallel data fetching

### Inference Speed
1. Model quantization (int8)
2. Batch predictions when possible
3. Cache frequent predictions
4. Use ONNX runtime for deployment
5. Torch.jit.script for production
