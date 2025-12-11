# CLAUDE.md - AI Assistant Guide for Stock Analysis Platform

> **Last Updated:** December 4, 2025
> **Version:** 2.4.0
> **Status:** Production Ready - Bot Running Locally
> **Current Test Version:** v1

---

## Test Session Bookkeeping (MANDATORY)

### Location
All test session reports are stored in: `test_reports/`

### File Naming Convention

| File | Purpose |
|------|---------|
| `Base.md` | Baseline documentation of all models and equations (reference only) |
| `session_v{VERSION}_{SESSION}.md` | Individual session reports |

**Naming Format:** `session_v1_001.md`, `session_v1_002.md`, ..., `session_v2_001.md`

- **VERSION:** Major version of the trading strategy (increment when models change significantly)
- **SESSION:** Three-digit session number within that version (001, 002, 003, ...)

### Current State
- **Version:** v1
- **Next Session:** 001
- **Baseline:** `test_reports/Base.md`

### Report Template (MUST FOLLOW)

Every session report MUST include:

```markdown
# Session Report: v{VERSION}_{SESSION}

> **Date:** YYYY-MM-DD
> **Duration:** HH:MM - HH:MM
> **Account:** Alpaca Paper Trading

---

## 1. Models Used This Session

### Active Models
| Model | Status | Parameters |
|-------|--------|------------|
| Regime Detection | ON/OFF | {params} |
| Feature Scoring | ON/OFF | {weights if modified} |
| CVaR Optimization | ON/OFF | risk_aversion={X} |
| ML Ensemble | ON/OFF | {model details} |
| Options Hedging | ON/OFF | {strategy} |

### Changes from Baseline
- List any modifications to models
- List any new parameters tested

---

## 2. Session Configuration

- **RRR:** {value}
- **Capital Deployed:** ${amount}
- **Symbols Traded:** {list}
- **Rebalance Frequency:** {frequency}

---

## 3. Performance Metrics

### Returns
| Metric | Value |
|--------|-------|
| Session P&L | ${X} ({Y}%) |
| Max Drawdown | {Z}% |
| Sharpe (session) | {value} |

### Trades Executed
| Time | Symbol | Side | Qty | Price | P&L |
|------|--------|------|-----|-------|-----|
| ... | ... | ... | ... | ... | ... |

---

## 4. Observations

### What Worked
- ...

### What Didn't Work
- ...

### Anomalies/Issues
- ...

---

## 5. Next Steps
- ...
```

### When to Increment Version

Increment VERSION (v1 → v2) when:
- New ML model is trained and connected
- Fundamental change to optimization objective
- Major change to risk parameters
- New asset classes added

Increment SESSION (001 → 002) for:
- Each trading session on Alpaca
- Different market conditions tested
- Parameter tuning within same model

### AI Assistant Instructions

**BEFORE each trading session:**
1. Check `test_reports/` for the latest session number
2. Create new report file with correct naming
3. Document all active models and parameters

**AFTER each trading session:**
1. Complete the performance metrics section
2. Document observations
3. Update "Next Session" in this CLAUDE.md file

---

## Health / Status

### Recent Fixes (Dec 4, 2025)

| Issue | Status | Resolution |
|-------|--------|------------|
| Missing reconnect logic | ✅ FIXED | `bot/trader.py` has exponential backoff reconnection |
| No structured logging | ✅ FIXED | `bot/logging_config.py` with rotating file handlers |
| Hardcoded config | ✅ FIXED | `bot/config.py` reads all settings from env vars |
| No trade persistence | ✅ FIXED | `bot/trade_store.py` SQLite storage |
| No monitoring | ✅ FIXED | Telegram integration with 2-hourly reports |
| No Docker deployment | ✅ FIXED | `Dockerfile` + `docker-compose.yml` added |
| VPS provisioning | ✅ FIXED | `infra/create_vps_vpsserver.py` script created |
| Bot not running | ✅ RUNNING | Bot running locally (lightweight, ~0% CPU) |
| Sell orders not tracked | ✅ FIXED | `bot/main.py` now calls `track_position_close` for sells |
| Stale yfinance prices | ✅ FIXED | Live Alpaca quotes used for order sizing |
| No market hours check | ✅ FIXED | `is_market_open()` guards strategy execution |
| No fill reconciliation | ✅ FIXED | `wait_for_fill()` polls orders for actual fill price |
| Heavy data fetching | ✅ FIXED | 1-hour cache for historical price data |
| No strategy unit tests | ✅ FIXED | 18 tests in `tests/test_bot_strategy.py` |

### Current Architecture

```
bot/                    # Production trading bot
├── config.py           # Centralized configuration from env
├── logging_config.py   # Rotating file logging
├── trader.py           # Enhanced trader with reconnect
├── trade_store.py      # SQLite trade persistence
└── main.py             # Bot entry point

infra/                  # Infrastructure provisioning
└── create_vps_vpsserver.py  # VPSServer.com API client

reports/                # Performance analytics
└── performance_report.py

notifications/          # Telegram integration
└── telegram_bot.py

scripts/                # Deployment scripts
└── send_report.sh      # Cron script for reports

logs/                   # Runtime logs (gitignored)
└── bot.log             # Main bot log file

data/                   # Persistent data (gitignored)
└── trades.db           # SQLite trade/equity database
```

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
- **Trading bot running locally** with Telegram notifications
- VPS provisioning script for VPSServer.com

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

# Trading bot commands
nohup python -m bot.main > logs/bot.log 2>&1 &  # Start in background
tail -f logs/bot.log                             # View live logs
ps aux | grep bot.main                           # Check if running
kill <PID>                                       # Stop the bot
```

---

## Alpaca Paper Trading

**Status:** ✅ LIVE - Bot running locally connected to paper account

**Account Info (as of Dec 4, 2025):**
- Equity: $100,000.00
- Buying Power: $200,000.00 (2x margin)
- Connection: Active with auto-reconnect

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
- `bot/trader.py` - Enhanced trading client with risk checks
- `bot/main.py` - Main bot entry point
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
| Sell orders not tracked | ✅ FIXED | `track_position_close` wired for sells |
| Order sizing uses stale data | ✅ FIXED | Live Alpaca quotes for sizing |
| Trading outside market hours | ✅ FIXED | `is_market_open()` guards strategy |
| No fill reconciliation | ✅ FIXED | `wait_for_fill()` with 30s timeout |
| Heavy yfinance fetching | ✅ FIXED | 1-hour cache for price data |
| Position reconciliation | ✅ FIXED | `reconcile_positions()` + `sync_from_alpaca()` |
| Portfolio-level kill switches | ✅ FIXED | 3% daily loss / 10% drawdown triggers |
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
pytest tests/test_bot_strategy.py -v     # Bot strategy/sizing tests (18 tests)
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
- **Trading bot running locally** (PID tracked, ~0% CPU usage)
- Telegram notifications for bot events
- SQLite trade/equity persistence
- VPS provisioning script ready

**What's Pending:**
- ML picker honest labeling (rename to Feature-Based Scoring)
- Risk-free rate integration from FRED
- Integration tests for main workflows
- Interactive Brokers research/integration
- Add actual trading strategy to `bot/main.py:191-195`

---

## For This Session

**Bot is currently running locally.** To check status:
```bash
ps aux | grep bot.main          # Check if running
tail -f logs/bot.log            # View live logs
```

If testing Alpaca integration:
1. Verify .env has Alpaca credentials
2. Run `python test_alpaca_pipeline.py`
3. Check `data_providers/alpaca_provider.py` for order submission
4. Test with small orders ($100) on paper account

---

## Local Bot Deployment (Recommended)

The trading bot is lightweight (~0% CPU, ~8MB RAM) and runs perfectly on local machine.

### Start Bot
```bash
cd /Users/samueldukmedjian/Desktop/stock_analysis
source venv_ml/bin/activate
nohup python -m bot.main > logs/bot.log 2>&1 &
```

### Monitor
```bash
tail -f logs/bot.log                    # Live logs
ps aux | grep bot.main                  # Process status
sqlite3 data/trades.db "SELECT * FROM equity_snapshots ORDER BY id DESC LIMIT 5;"
```

### Stop
```bash
kill $(pgrep -f "bot.main")
```

### What the Bot Does
- Connects to Alpaca paper trading account
- Records equity snapshots every 5 minutes to SQLite
- Sends Telegram notifications on startup/shutdown/errors
- Auto-reconnects with exponential backoff on connection loss
- **TODO:** Add trading strategy at `bot/main.py:191-195`

---

## VPS Deployment (Docker)

### Quick Deploy

```bash
# On VPS
git clone <your-repo> stock_analysis
cd stock_analysis
cp .env.example .env
# Edit .env with your credentials
docker compose up -d
```

### Key Files

- `Dockerfile` - Bot container image
- `docker-compose.yml` - Container orchestration
- `.env.example` - All required environment variables
- `requirements_bot.txt` - Minimal dependencies

### Commands

```bash
# Start bot
docker compose up -d

# View logs
docker compose logs -f trading-bot

# Stop
docker compose down

# Rebuild after code changes
docker compose up -d --build
```

### 2-Hourly Reports

Option 1 (Docker):
```bash
docker compose --profile with-reports up -d
```

Option 2 (Cron):
```bash
# Add to crontab -e
0 */2 * * * /path/to/stock_analysis/scripts/send_report.sh
```

---

## VPS Provisioning (VPSServer.com)

Script created at `infra/create_vps_vpsserver.py` for programmatic VPS creation.

**Note:** VPSServer.com API requires IP whitelisting in their console. The API schema validation is strict - script may need updates if API changes.

### Usage
```bash
export VPSSERVER_CLIENT_ID="your_client_id"
export VPSSERVER_SECRET="your_secret"
export VPS_ROOT_PASSWORD="YourSecurePassword123"

python infra/create_vps_vpsserver.py --list      # List servers
python infra/create_vps_vpsserver.py --options   # Show available configs
python infra/create_vps_vpsserver.py --create    # Create VPS (US-NY2, Ubuntu 24.04)
```

**Default Config:** US-NY2 datacenter, Ubuntu 24.04, 2 vCPU, 4GB RAM, 20GB SSD

---

**This document is the single source of truth for AI assistants.**
