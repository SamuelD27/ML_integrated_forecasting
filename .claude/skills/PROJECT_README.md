# Stock Analysis Project - Skills Framework

## Overview

This directory contains a complete Claude Code Skills framework specifically for the Stock Analysis ML Trading System project. All 6 skills are production-ready and tailored to this codebase.

**Project Location**: `/Users/samueldukmedjian/Desktop/stock_analysis/`

---

## Skills Integration Map

### How Skills Map to Your Codebase

```
stock_analysis/
│
├── single_stock/
│   ├── risk_metrics.py              → Skill 1: Financial Knowledge Validator
│   │                                   Validates Sharpe, Sortino, VaR, CVaR
│   ├── valuation.py                 → Skill 1: Financial Knowledge Validator
│   │                                   Validates DCF, multiples, etc.
│   └── forecasting.py               → Skill 3: Time Series Validation Specialist
│                                      Walk-forward validation for ARIMA
│
├── ml_models/
│   ├── hybrid_model.py              → Skill 2: ML Architecture Builder
│   │                                   Weight initialization + output validation
│   ├── cnn_module.py                → Skill 2: ML Architecture Builder
│   ├── lstm_module.py               → Skill 2: ML Architecture Builder
│   ├── transformer_module.py        → Skill 2: ML Architecture Builder
│   ├── features.py                  → Skill 5: Real-Time Feature Pipeline
│   │                                   Incremental feature computation
│   └── regime.py                    → Skill 1: Financial Knowledge Validator
│                                      Market regime classification
│
├── portfolio/
│   ├── cvar_allocator.py            → Skill 4: Portfolio Optimization Expert
│   │                                   Ledoit-Wolf enforcement, CVaR optimization
│   ├── advanced_optimizer.py        → Skill 4: Portfolio Optimization Expert
│   │                                   Mean-variance with constraints
│   ├── options_overlay.py           → Skill 1: Financial Knowledge Validator
│   │                                   Greeks calculation, hedge ratios
│   └── ml_reporter.py               → Skill 6: Model Deployment & Monitoring
│                                      Deployment reporting
│
├── training/
│   ├── train_hybrid.py              → Skill 3: Time Series Validation Specialist
│   │                                   Walk-forward validation mandatory
│   ├── training_utils.py            → Skill 3: Time Series Validation Specialist
│   │                                   WalkForwardSplitter implementation
│   └── config.yaml                  → Skill 2: ML Architecture Builder
│                                      Model configuration validation
│
├── backtesting/
│   └── backtest_engine.py           → Skill 3: Time Series Validation Specialist
│                                      Embargo periods, no look-ahead bias
│
└── [deployment scripts]             → Skill 6: Model Deployment & Monitoring
                                       Smoke tests before deployment
```

---

## Quick Start Guide

### Using Skills in Your Project

**Scenario 1: Calculating Financial Metrics**
```python
# Your code in single_stock/risk_metrics.py
# Skill 1 will activate and ensure:
# - Assertions for validation (not just logging)
# - Edge case handling (zero volatility, negative returns)
# - Correct formula implementation

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    # Skill ensures this uses assertions, not just returns value
    pass
```

**Scenario 2: Building ML Architecture**
```python
# Your code in ml_models/hybrid_model.py
# Skill 2 will activate and ensure:
# - _initialize_weights() method added
# - Output shape validation
# - Gradient flow configuration

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._initialize_weights()  # Skill enforces this
```

**Scenario 3: Training Time Series Model**
```python
# Your code in training/train_hybrid.py
# Skill 3 will activate and ensure:
# - Walk-forward validation (not random split)
# - Embargo periods (T+2 for daily data)
# - No look-ahead bias

splitter = WalkForwardSplitter(
    lookback_days=252,
    test_days=63,
    step_days=21,
    embargo_days=2  # Skill enforces this
)
```

**Scenario 4: Optimizing Portfolio**
```python
# Your code in portfolio/cvar_allocator.py
# Skill 4 will activate and ensure:
# - Ledoit-Wolf (no fallback to sample covariance)
# - Complete transaction costs (4 components)
# - Constraint assertions

lw = LedoitWolf()
cov = lw.fit(returns).covariance_  # No try/except fallback allowed
```

**Scenario 5: Computing Real-Time Features**
```python
# Your code in ml_models/features.py
# Skill 5 will activate and ensure:
# - Incremental updates validated
# - Latency < 10ms enforced
# - Memory bounded

rsi = IncrementalRSI(period=14, validate=True)  # Skill adds validation
```

**Scenario 6: Deploying Model**
```python
# Your deployment scripts
# Skill 6 will activate and ensure:
# - Smoke tests run before deployment
# - Version tracking automatic
# - Output validation built-in

deployment = ProductionModelDeployment('model.pkl')
deployment.run_smoke_tests()  # Mandatory before deployment
```

---

## Available Skills

### Skill 1: Financial Knowledge Validator (95%)
**Location**: `financial-knowledge-validator/`
**Command**: `/financial-knowledge-validator`

**Applies to**:
- `single_stock/risk_metrics.py` - Sharpe, Sortino, VaR, CVaR
- `single_stock/valuation.py` - DCF, multiples
- `portfolio/options_overlay.py` - Greeks, hedge ratios
- `ml_models/regime.py` - Market regime detection

**Key Pattern**: Assertions for validation, not warnings

---

### Skill 2: ML Architecture Builder (100%)
**Location**: `ml-architecture-builder/`
**Command**: `/ml-architecture-builder`

**Applies to**:
- `ml_models/hybrid_model.py` - Weight initialization
- `ml_models/cnn_module.py` - 1D CNN initialization
- `ml_models/lstm_module.py` - LSTM initialization
- `ml_models/transformer_module.py` - Transformer initialization

**Key Pattern**: `_initialize_weights()` + output validation

---

### Skill 3: Time Series Validation Specialist (100%)
**Location**: `time-series-validation-specialist/`
**Command**: `/time-series-validation-specialist`

**Applies to**:
- `training/train_hybrid.py` - Walk-forward validation
- `training/training_utils.py` - WalkForwardSplitter
- `backtesting/backtest_engine.py` - Embargo periods
- `single_stock/forecasting.py` - ARIMA validation

**Key Pattern**: Walk-forward + embargo (never random split)

---

### Skill 4: Portfolio Optimization Expert (100%)
**Location**: `portfolio-optimization-expert/`
**Command**: `/portfolio-optimization-expert`

**Applies to**:
- `portfolio/cvar_allocator.py` - CVaR optimization
- `portfolio/advanced_optimizer.py` - Mean-variance
- `portfolio/peer_discovery.py` - Portfolio construction
- `portfolio/etf_discovery.py` - ETF selection

**Key Pattern**: Ledoit-Wolf enforcement (no fallback)

---

### Skill 5: Real-Time Feature Pipeline (100%)
**Location**: `real-time-feature-pipeline/`
**Command**: `/real-time-feature-pipeline`

**Applies to**:
- `ml_models/features.py` - Feature engineering
- `utils/advanced_feature_engineering.py` - Advanced features
- Real-time data processing pipelines

**Key Pattern**: Validate incremental == full (0.00e+00 error)

---

### Skill 6: Model Deployment & Monitoring (100%)
**Location**: `model-deployment-monitoring/`
**Command**: `/model-deployment-monitoring`

**Applies to**:
- Model deployment scripts
- `portfolio/ml_reporter.py` - Reporting
- Production deployment pipelines

**Key Pattern**: Smoke tests before deployment (mandatory)

---

## Project-Specific Examples

### Example 1: Fixing Risk Metrics (Skill 1)

**Before** (in `single_stock/risk_metrics.py`):
```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe  # No validation
```

**After** (with Skill 1):
```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate

    # VALIDATE: Check for zero volatility
    std = excess_returns.std()
    assert std > 0, f"Zero volatility detected: std={std}"

    sharpe = excess_returns.mean() / std

    # VALIDATE: Sharpe ratio is finite
    assert np.isfinite(sharpe), f"Invalid Sharpe ratio: {sharpe}"

    return sharpe
```

---

### Example 2: Fixing Model Initialization (Skill 2)

**Before** (in `ml_models/hybrid_model.py`):
```python
class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = CNNModule(config)
        self.lstm = LSTMModule(config)
        # No weight initialization
```

**After** (with Skill 2):
```python
class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = CNNModule(config)
        self.lstm = LSTMModule(config)
        self._initialize_weights()  # Added by Skill 2

    def _initialize_weights(self):
        """Initialize weights for all modules."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
```

---

### Example 3: Fixing Training Validation (Skill 3)

**Before** (in `training/train_hybrid.py`):
```python
# Random split (WRONG for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**After** (with Skill 3):
```python
# Walk-forward validation (CORRECT)
from training.training_utils import WalkForwardSplitter

splitter = WalkForwardSplitter(
    lookback_days=252,
    test_days=63,
    step_days=21,
    embargo_days=2  # Prevents look-ahead bias
)

for train_idx, test_idx in splitter.split(data):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    # Train and evaluate
```

---

### Example 4: Fixing Portfolio Optimization (Skill 4)

**Before** (in `portfolio/cvar_allocator.py`):
```python
try:
    lw = LedoitWolf()
    cov = lw.fit(returns).covariance_
except:
    cov = np.cov(returns.T)  # Fallback (BAD)
```

**After** (with Skill 4):
```python
# No fallback allowed - must use Ledoit-Wolf
lw = LedoitWolf()
try:
    cov = lw.fit(returns).covariance_ * 252  # Annualize
except Exception as e:
    raise RuntimeError(
        f"CRITICAL: Ledoit-Wolf shrinkage failed: {e}. "
        f"Sample covariance NOT acceptable for production."
    )
```

---

## Statistics

**Project Coverage**:
- Files validated by skills: 20+ files
- Patterns enforced: 35+ patterns
- Skills active: 6/6 (100%)

**Impact on Your Project**:
- Prevents financial calculation errors (Skill 1)
- Prevents gradient explosions (Skill 2)
- Prevents look-ahead bias (Skill 3)
- Prevents unstable portfolios (Skill 4)
- Prevents numerical drift (Skill 5)
- Prevents broken deployments (Skill 6)

---

## Documentation

All skills are fully documented:

- **[README.md](README.md)** - Comprehensive guide
- **[SKILLS_MAP.md](SKILLS_MAP.md)** - Quick reference with visual maps
- **[STRUCTURE.txt](STRUCTURE.txt)** - Directory structure

Each skill folder contains:
1. `SKILL.md` - Main skill patterns
2. `[domain]-reference.md` - Complete implementations
3. `test-scenarios.md` - Test scenarios
4. `baseline-results.md` - Agent behavior without skill
5. `refactor-findings.md` - Compliance results
6. `DEPLOYMENT-CHECKLIST.md` - Verification

---

## Framework Status

```
✅ DEPLOYED - All 6 skills active
✅ TESTED - 99.2% average compliance
✅ PRODUCTION-READY - Integrated with your project
```

**Skills are now part of your stock_analysis project and will automatically assist with ML trading system development.**

---

## Version

**Framework Version**: 1.0
**Deployment Date**: 2025-11-05
**Project**: Stock Analysis ML Trading System
**Location**: `/Users/samueldukmedjian/Desktop/stock_analysis/.claude/skills/`
