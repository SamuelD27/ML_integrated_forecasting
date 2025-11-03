# Crypto Trading MVP

Automated cryptocurrency trading system with Kraken WebSocket integration, EMA crossover signals, and paper trading simulation.

## Features

- ✅ Kraken WebSocket connection with automatic reconnection
- ✅ TimescaleDB for tick storage with OHLCV aggregations
- ✅ EMA 20/50 crossover signal generation
- ✅ Volatility-based position sizing (ATR-adjusted)
- ✅ Paper trading simulation with realistic slippage
- ✅ Console logging for all trading activities

## Prerequisites

- Python 3.11 (NOT 3.12+)
- TimescaleDB 2.x
- PostgreSQL 14+

## Setup

### 1. Install TimescaleDB

**macOS (Homebrew):**
```bash
brew install timescaledb
timescaledb-tune --quiet --yes
brew services start postgresql@14
```

**Ubuntu:**
```bash
sudo apt install postgresql-14-timescaledb
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql
```

### 2. Create Database

```bash
psql -U postgres -c "CREATE DATABASE crypto_trading;"
psql -U postgres -d crypto_trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
psql -U postgres -d crypto_trading -f crypto_trading/schema.sql
```

### 3. Install Python Dependencies

```bash
cd crypto_trading
pip install -r requirements.txt
```

**Note:** TA-Lib requires system library:
```bash
# macOS
brew install ta-lib

# Ubuntu
sudo apt-get install libta-lib0-dev
```

### 4. Configure Credentials

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 5. Run System

```bash
python -m crypto_trading.main
```

## Configuration

Edit `config.yaml` to customize:
- Trading pairs
- Initial balance ($100 test, $1000 production)
- EMA periods (default: 20/50)
- Risk parameters (position limits, portfolio heat)
- Slippage and commission rates

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=crypto_trading --cov-report=html
```

## Architecture

```
crypto_trading/
├── main.py                 # Main orchestrator
├── websocket_manager.py    # Kraken WebSocket client
├── database.py             # TimescaleDB interface
├── signal_engine.py        # EMA crossover detection
├── risk_manager.py         # Position sizing
├── paper_trader.py         # Virtual trading
├── config.yaml             # Configuration
└── requirements.txt        # Dependencies
```

## Logs

All trading activity is logged to console:

```
[2025-11-03 14:32:15] INFO main: ✅ Database connected
[2025-11-03 14:32:16] INFO websocket_manager: Connected to Kraken WebSocket
[2025-11-03 14:35:22] INFO signal_engine: Signal generated: BUY XBT/USD @ $67450.00
[2025-11-03 14:35:22] INFO paper_trader: BUY 0.007421 XBT/USD @ $67450.00
[2025-11-03 14:35:22] INFO main: ✅ Trade executed: BUY 0.007421 XBT/USD @ $67450.00
[2025-11-03 14:35:22] INFO main: 💰 Balance: $500.13, Positions: 1
```

## Monitoring

Check portfolio status:
```sql
-- Open positions
SELECT * FROM crypto_trading.positions;

-- Recent trades
SELECT * FROM crypto_trading.trade_history ORDER BY timestamp DESC LIMIT 10;

-- OHLCV data
SELECT * FROM ohlcv_1m WHERE symbol = 'XBT/USD' ORDER BY bucket DESC LIMIT 20;
```

## Future Enhancements

- [ ] Additional strategies (Bollinger Bands, RSI, MACD)
- [ ] Machine learning models (PyTorch LSTM)
- [ ] Advanced risk management (correlation, drawdown limits)
- [ ] Web dashboard (Grafana/Streamlit)
- [ ] Microservices architecture (RabbitMQ)
- [ ] Live trading mode
- [ ] Multi-exchange support

## Troubleshooting

**TimescaleDB connection fails:**
- Check PostgreSQL is running: `pg_isready`
- Verify credentials in `.env`
- Check database exists: `psql -U postgres -l`

**TA-Lib import error:**
- Install system library first (see Setup step 3)
- Rebuild: `pip uninstall TA-Lib && pip install TA-Lib`

**WebSocket disconnects frequently:**
- Check internet connection stability
- Verify Kraken API status
- System will auto-reconnect with exponential backoff

## License

MIT
