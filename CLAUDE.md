# CLAUDE.md - AI Assistant Guide for Stock Analysis Platform

> **Last Updated:** December 3, 2025
> **Version:** 2.1.0
> **Status:** Production Ready with Alpaca Paper Trading

---

## Quick Start for New Claude Sessions

**BEFORE DOING ANYTHING, READ THESE:**
1. `~/.claude_knowledge/` - Global knowledge base with protocols, skills, and learnings
2. `.claude/skills/` - Project-specific AI skills (6 specialized skills)
3. This file for project context

---

## Project Summary

**Professional Quantitative Finance System** - ML-enhanced platform for:
- Stock analysis and forecasting (3-model ensemble: LightGBM + Ridge + Momentum)
- Portfolio optimization (Black-Litterman, HRP, CVaR with Ledoit-Wolf shrinkage)
- Paper trading via Alpaca API
- Interactive Streamlit dashboard

**Recent Changes (Dec 2025):**
- Fixed critical log-returns index bug in data_fetching.py
- Added CVaR fallback for small universes
- Portfolio-wide hedge coverage with index options
- International peer discovery + multi-provider fallback
- Alpaca paper trading pipeline integrated

---

## Tech Stack

```
Python 3.11 (REQUIRED - NOT 3.12+)
├── ML/DL: PyTorch 2.1, LightGBM, XGBoost, scikit-learn
├── Portfolio: CVXPY, pyportfolioopt
├── Data: yfinance, Alpaca API, pandas, numpy
├── UI: Streamlit, Plotly
└── Testing: pytest
```

---

## Architecture

```
stock_analysis/
├── dashboard/              # Streamlit UI (streamlit run dashboard/app.py)
├── ml_models/              # ML models
│   ├── practical_ensemble.py   # 3-model ensemble (USE THIS)
│   ├── hybrid_model.py         # CNN-LSTM-Transformer
│   └── tft_model.py            # Temporal Fusion Transformer
├── portfolio/              # Portfolio optimization
│   ├── black_litterman.py      # ML-enhanced B-L
│   ├── hrp_optimizer.py        # Hierarchical Risk Parity
│   ├── cvar_allocator.py       # CVaR with Ledoit-Wolf
│   └── options_overlay.py      # Portfolio-wide hedging
├── data_providers/         # Multi-provider data
│   ├── yahoo_provider.py       # Default
│   ├── alpaca_provider.py      # Real-time + paper trading
│   └── manager.py              # Fallback chain
├── single_stock/           # Individual analysis
├── training/               # ML training infrastructure
├── tests/                  # pytest suite (22 files)
├── configs/                # YAML configs
└── .claude/skills/         # 6 AI skills
```

---

## Key Commands

```bash
# Activate environment
source venv_ml/bin/activate

# Launch dashboard
streamlit run dashboard/app.py

# Run tests
make test-fast
# OR
pytest tests/test_practical_ensemble.py tests/test_hrp_optimizer.py -v

# Stock analysis CLI
python run_portfolio.py AAPL --years 3

# ML portfolio construction
python portfolio_creation_ml.py AAPL --capital 100000 --enable-ml
```

---

## Alpaca Paper Trading

**Status:** Integrated and ready for testing

**Setup:**
```bash
# Ensure .env has:
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Key Files:**
- `data_providers/alpaca_provider.py` - Data + order submission
- `data_providers/manager.py` - Multi-provider with Alpaca fallback
- `test_alpaca_pipeline.py` - Integration tests

**Test the Pipeline:**
```bash
python test_alpaca_pipeline.py
```

---

## AI Skills Available (.claude/skills/)

| Skill | Use When |
|-------|----------|
| `financial-knowledge-validator` | Implementing financial calculations |
| `ml-architecture-builder` | Building neural network architectures |
| `time-series-validation-specialist` | Backtesting, walk-forward validation |
| `portfolio-optimization-expert` | Portfolio construction, covariance |
| `real-time-feature-pipeline` | Feature engineering, latency |
| `model-deployment-monitoring` | Deploying models to production |

---

## Critical Rules

### DO
- Use Ledoit-Wolf shrinkage for covariance (never raw np.cov)
- Use walk-forward validation for time series (never random split)
- Initialize neural network weights explicitly
- Add assertions for financial calculations
- Run tests after changes: `make test-fast`

### DON'T
- Use Python 3.12+ (compatibility issues)
- Skip weight initialization in ML models
- Use random train/test split on time series
- Commit API keys (use .env)

---

## Known Issues & Fixes

| Issue | Status | Notes |
|-------|--------|-------|
| Log-returns index bug | ✅ FIXED | data_fetching.py preserves datetime index |
| CVaR fails on small universe | ✅ FIXED | Auto-reduces universe with warning |
| Hedge covers only primary ticker | ✅ FIXED | Portfolio-wide index options |
| Peer discovery US-only | ✅ FIXED | International + ETF-based fallback |
| Yahoo rate limits | ✅ FIXED | Multi-provider fallback to Alpaca |
| ML picker is heuristic | 🟡 PENDING | Currently "Feature-Based Scoring" |
| Risk metrics assume Rf=0 | 🟡 PENDING | Need FRED integration |

---

## Testing

```bash
# Fast tests (no API calls)
make test-fast

# Full test suite
pytest tests/ -v

# Specific modules
pytest tests/test_practical_ensemble.py -v
pytest tests/test_hrp_optimizer.py -v
pytest tests/test_alpaca_pipeline.py -v  # Requires API keys
```

---

## Environment Variables (.env)

```bash
# Required for Alpaca
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional
FRED_API_KEY=xxx          # Economic data
WANDB_API_KEY=xxx         # Experiment tracking
POLYGON_API_KEY=xxx       # Additional data provider
```

---

## Global Knowledge Base (~/.claude_knowledge/)

**IMPORTANT:** Check this before starting any task:

```
~/.claude_knowledge/
├── core/                   # Mandatory protocols
│   ├── interaction_protocols.md
│   ├── efficiency_rules.md
│   ├── testing_protocols.md
│   └── help_seeking_rules.md
├── platforms/              # Platform docs (RunPod, macOS)
├── skills/                 # Reusable patterns
├── learning/               # Compound learnings database
└── AUTO_CONFIG.md          # Full permissions granted
```

**You have FULL CLEARANCE** - no confirmation needed for standard operations.

---

## Current State Summary

**What's Working:**
- Dashboard with 7 analysis tools
- 3-model ML ensemble forecasting
- Portfolio optimization (B-L, HRP, CVaR)
- Alpaca paper trading integration
- Multi-provider data fallback
- Portfolio-wide hedging

**What's Pending:**
- ML picker honest labeling (rename to Feature-Based Scoring)
- Risk-free rate integration from FRED
- Integration tests for main workflows
- Interactive Brokers research/integration

---

## For This Session

If testing Alpaca integration:
1. Verify .env has Alpaca credentials
2. Run `python test_alpaca_pipeline.py`
3. Check `data_providers/alpaca_provider.py` for order submission
4. Test with small orders ($100) on paper account

---

**This document is the single source of truth for AI assistants.**
