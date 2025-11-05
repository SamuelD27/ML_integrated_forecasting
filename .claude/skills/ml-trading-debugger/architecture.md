# ML Trading System Architecture

## System Overview

The ML_integrated_forecasting system is organized into distinct components:

```
ml_integrated_forecasting/
├── data/                      # Data collection and processing
│   ├── collectors/            # API integrations (yfinance, FMP, FRED)
│   ├── processors/            # Data cleaning, normalization
│   └── storage/               # Parquet storage management
├── models/                    # ML model implementations
│   ├── tft/                   # Temporal Fusion Transformer
│   ├── lstm/                  # LSTM variants
│   ├── hybrid/                # CNN-LSTM-Transformer hybrid
│   └── ensemble/              # Ensemble strategies
├── training/                  # Training infrastructure
│   ├── trainers/              # Training loops and logic
│   ├── optimizers/            # Hyperparameter tuning (Optuna)
│   └── checkpoints/           # Model checkpoint management
├── inference/                 # Prediction generation
│   ├── predictors/            # Model loading and inference
│   └── cache/                 # Prediction caching layer
├── portfolio/                 # Portfolio management
│   ├── optimization/          # HRP, mean-variance optimization
│   ├── risk/                  # GARCH, VaR, CVaR calculations
│   └── execution/             # Order generation
├── analysis/                  # Investment analysis
│   ├── fundamental/           # DCF, factor models
│   ├── technical/             # Technical indicators
│   ├── sentiment/             # FinBERT sentiment analysis
│   └── decision_engine/       # InvestmentDecisionEngine
└── dashboard/                 # Streamlit visualization
    ├── views/                 # Dashboard pages
    ├── components/            # Reusable UI components
    └── data_loaders/          # Data fetching for dashboard
```

## Component Interactions

### Data Flow

1. **Data Collection**:
   - Multiple APIs fetch historical and real-time data
   - yfinance: Price and volume data (primary source)
   - Financial Modeling Prep: Fundamentals (income statement, balance sheet, ratios)
   - NewsAPI + Reddit: Text for sentiment analysis
   - FRED: Macro indicators (interest rates, GDP, inflation)

2. **Data Processing**:
   - Raw data → cleaning → normalization → feature engineering
   - Storage in compressed Parquet format
   - Time series alignment across all sources
   - Target: 25 years × 500 tickers

3. **Model Training**:
   - Load processed data from Parquet files
   - Train ensemble of models (TFT, LSTM, CNN-LSTM-Transformer)
   - Hyperparameter tuning with Optuna
   - Distributed training on 8xB200 GPUs
   - Save checkpoints to disk

4. **Inference**:
   - Load trained model checkpoints
   - Generate predictions for next 1, 5, 20, 60 day horizons
   - Cache predictions for dashboard access
   - **CRITICAL INTEGRATION GAP**: Predictions not automatically flowing to dashboard

5. **Portfolio Construction**:
   - InvestmentDecisionEngine scores stocks across factors:
     - Valuation (P/E, P/B, DCF-based fair value)
     - Quality (ROE, debt ratios, margins)
     - Momentum (price trends, technical signals)
     - Risk (volatility, beta, liquidity)
   - Multi-factor scoring → portfolio weights
   - HRP optimization for risk-adjusted allocation
   - **CRITICAL INTEGRATION GAP**: ML predictions not connected to factor scoring

6. **Dashboard Visualization**:
   - Streamlit interface for analysis presentation
   - Portfolio performance tracking
   - Individual stock analysis
   - Risk metrics and scenario analysis
   - **CRITICAL INTEGRATION GAP**: Analysis functions lack ML model outputs

## Model Architecture Details

### Temporal Fusion Transformer (TFT)

- Architecture: Multi-head attention with temporal fusion
- Input: Multiple time series features (OHLCV, fundamentals, sentiment, macro)
- Output: Multi-horizon predictions (1d, 5d, 20d, 60d)
- Target size: 500M+ parameters
- Training: Mixed precision (TF32), distributed across 8 GPUs

### CNN-LSTM-Transformer Hybrid

- CNN layers: Extract local patterns from price sequences
- LSTM layers: Capture temporal dependencies
- Transformer layers: Model long-range interactions
- Current size: 100M parameters → scaling to 500M+
- Ensemble member alongside TFT

### Ensemble Strategy

- Models: TFT, LSTM variants, CNN-LSTM-Transformer hybrid
- Weighting: Optuna-tuned based on validation performance
- Target: 15-41% Sharpe ratio improvement over individual models
- Directional accuracy target: 70-75% (currently ~65%)

## Integration Points

### CRITICAL: Model-to-Dashboard Connection

**Current State**: Models train successfully but predictions don't reach analysis functions

**Required Integration**:
1. Standardized prediction output format (Parquet or pickle)
2. Prediction caching layer accessible to dashboard
3. Dashboard data loaders updated to read prediction cache
4. InvestmentDecisionEngine updated to incorporate ML predictions

**Expected Interface**:
```python
# predictions.parquet schema
{
    'ticker': str,
    'date': datetime,
    'horizon': int,  # 1, 5, 20, 60 days
    'prediction': float,  # predicted return
    'confidence': float,  # model confidence [0, 1]
    'model': str,  # 'tft', 'lstm', 'hybrid', 'ensemble'
}
```

### InvestmentDecisionEngine Integration

**Purpose**: Multi-factor stock scoring incorporating ML predictions

**Factors**:
1. **Valuation** (30%): P/E, P/B, DCF vs current price
2. **Quality** (25%): ROE, debt ratios, profit margins
3. **Momentum** (25%): Technical indicators + **ML predictions**
4. **Risk** (20%): Volatility, beta, drawdown metrics

**ML Integration Point**:
- Use ensemble predictions as momentum signal
- Weight by prediction confidence
- Combine with traditional technical indicators
- Risk-adjust based on prediction uncertainty

## Storage Management

### Parquet Storage Strategy

- Dataset size: 25 years × 500 tickers × multiple features
- Compression: Snappy or Gzip (target 70-80% reduction)
- Partitioning: By year and ticker for efficient loading
- Location: `/data/processed/` on RunPod instance

### Checkpoint Management

**Problem**: Previous pod failures from excessive checkpoint saving

**Solution**:
- Save only top 3 checkpoints by validation loss
- Implement checkpoint rotation (delete old on new save)
- Save frequency: Only when validation loss improves
- Validation before save: Ensure checkpoint is loadable

## Infrastructure Details

### RunPod Configuration

- Instance: 8xB200 GPUs
- Total VRAM: 192GB (24GB × 8)
- Storage: Limited, requires optimization
- Training mode: Distributed data parallelism
- Precision: Mixed precision (TF32 automatic)

### GPU Memory Optimization

- Gradient checkpointing for large models
- Batch size tuning per GPU memory
- Clear cache between training stages
- Monitor VRAM usage with nvidia-smi

## Performance Targets

### Model Performance
- Directional accuracy: 70-75% (current: ~65%)
- Sharpe ratio improvement: 15-41% vs baseline
- Multi-horizon consistency: Maintain accuracy across 1d-60d

### System Performance
- Training time: <24 hours for 500M parameter model
- Inference latency: <1 second per ticker
- Dashboard load time: <3 seconds for portfolio view

### Production Standards
- Uptime: 99%+ reliability
- Data freshness: Daily updates
- Backup strategy: Checkpoints and data backups
- Monitoring: Automated alerts for failures
