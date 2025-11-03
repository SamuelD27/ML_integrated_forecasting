# Crypto Trading Pipeline MVP - Design Document

**Date**: 2025-11-03
**Status**: Approved
**Architecture**: Modular Monolith

## Executive Summary

Building a production-grade cryptocurrency automated trading pipeline for Kraken exchange implementing institutional-quality patterns from day one while maintaining MVP simplicity. The system combines real-time WebSocket data ingestion, TimescaleDB storage, EMA crossover signal generation, volatility-based position sizing, and paper trading simulation.

**Key Design Decision**: Modular monolith architecture balances rapid MVP delivery with production-ready patterns, enabling seamless extraction to microservices as complexity grows.

## Requirements

### Functional Requirements

1. **Data Ingestion**: Kraken WebSocket connection with automatic reconnection logic
2. **Data Storage**: TimescaleDB for tick-level data with OHLCV aggregations
3. **Signal Generation**: EMA 20/50 crossover detection
4. **Position Sizing**: Simple volatility-based allocation (ATR-adjusted)
5. **Paper Trading**: Virtual balance simulation ($100 test, $1000 production)
6. **Logging**: Console output for all trading activities

### Non-Functional Requirements

1. **Reliability**: 24/7 operation with automatic recovery from network failures
2. **Scalability**: Handle all Kraken spot pairs (100+ concurrent subscriptions)
3. **Extensibility**: Clean module boundaries for future ML/strategy enhancements
4. **Security**: API credentials in environment variables, never committed to git
5. **Performance**: Sub-second signal generation latency

### Configuration

- **Testing**: $100 virtual balance, 5-10 configurable pairs
- **Production**: $1000 virtual balance, expandable to all pairs
- **API Credentials**: Provided (stored in .env)

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Crypto Trading System                    │
│                      (Single Process)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   WebSocket  │─────▶│  Database    │                    │
│  │   Manager    │      │  Layer       │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                      │                            │
│         │                      ▼                            │
│         │              ┌──────────────┐                    │
│         └─────────────▶│   Signal     │                    │
│                        │   Engine     │                    │
│                        └──────────────┘                    │
│                               │                            │
│                               ▼                            │
│                        ┌──────────────┐                    │
│                        │     Risk     │                    │
│                        │   Manager    │                    │
│                        └──────────────┘                    │
│                               │                            │
│                               ▼                            │
│                        ┌──────────────┐                    │
│                        │    Paper     │                    │
│                        │   Trader     │                    │
│                        └──────────────┘                    │
│                               │                            │
│                               ▼                            │
│                        ┌──────────────┐                    │
│                        │   Console    │                    │
│                        │   Logger     │                    │
│                        └──────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ▲                                          │
         │                                          │
    Kraken API                              TimescaleDB
```

### Module Design

#### 1. WebSocket Manager (`websocket_manager.py`)

**Responsibilities**:
- Establish and maintain persistent WebSocket connection to Kraken
- Subscribe to ticker/trade feeds for configured pairs
- Handle disconnections with exponential backoff (5s, 10s, 20s, 60s max)
- Validate incoming data (price > 0, timestamp fresh, checksum validation)
- Emit tick events to internal event bus
- Send heartbeat/ping to prevent 60s timeout

**Key Classes**:
- `KrakenWebSocketClient`: Main client with async event loop
- `TickEvent`: Data class for validated tick data
- `ReconnectionHandler`: Exponential backoff logic

**External Dependencies**: `websockets`, `asyncio`

#### 2. Database Layer (`database.py`)

**Responsibilities**:
- TimescaleDB connection management with pooling
- Create hypertable for tick storage
- Write tick data asynchronously (non-blocking)
- Maintain continuous aggregate views for OHLCV (1m, 5m, 1h, 1d)
- Provide query interface for historical data (signal engine)

**Schema**:
```sql
CREATE TABLE ticks (
    time TIMESTAMPTZ NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION
);

SELECT create_hypertable('ticks', 'time');

CREATE MATERIALIZED VIEW ohlcv_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(volume) AS volume
FROM ticks
GROUP BY bucket, symbol;
```

**External Dependencies**: `psycopg2`, `asyncpg`

#### 3. Signal Engine (`signal_engine.py`)

**Responsibilities**:
- Consume tick events from WebSocket manager
- Fetch historical OHLCV data from database for context
- Calculate EMA 20 and EMA 50 using pandas/talib
- Detect crossovers:
  - **Bullish**: EMA20 crosses above EMA50 (previous: 20 < 50, current: 20 > 50)
  - **Bearish**: EMA20 crosses below EMA50 (previous: 20 > 50, current: 20 < 50)
- Emit trade signals with metadata (symbol, direction, timestamp, EMA values)

**Key Classes**:
- `SignalEngine`: Main signal processor
- `EMACalculator`: EMA computation with pandas
- `CrossoverDetector`: Stateful crossover detection
- `TradeSignal`: Data class for signals

**Parameters** (from config.yaml):
- `ema_fast_period: 20`
- `ema_slow_period: 50`
- `min_bars_required: 51` (need 51 bars minimum for EMA50)

**External Dependencies**: `pandas`, `talib`, `numpy`

#### 4. Risk Manager (`risk_manager.py`)

**Responsibilities**:
- Calculate ATR (Average True Range) per pair for volatility measurement
- Compute allocation weights using volatility-adjusted formula
- Enforce portfolio heat limits (max 10% total capital at risk)
- Validate position sizes before execution
- Track correlation between assets (future enhancement)

**Allocation Formula** (MVP):
```python
# Simple volatility-based with bounds
base_allocation = 1.0 / num_signals  # Equal weight baseline
volatility_factor = mean_atr / pair_atr  # Inverse volatility
weight = base_allocation * volatility_factor
weight = clip(weight, 0.02, 0.20)  # 2-20% bounds
normalized_weight = weight / sum(all_weights)
position_size = balance * normalized_weight
```

**Key Classes**:
- `RiskManager`: Main risk coordination
- `VolatilityCalculator`: ATR computation
- `PositionSizer`: Allocation weight calculation
- `PortfolioHeatTracker`: Total risk monitoring

**Parameters**:
- `atr_period: 14`
- `max_portfolio_heat: 0.10` (10%)
- `min_position_pct: 0.02` (2%)
- `max_position_pct: 0.20` (20%)

**External Dependencies**: `pandas`, `talib`

#### 5. Paper Trader (`paper_trader.py`)

**Responsibilities**:
- Maintain virtual balance (starts at $100 test / $1000 prod)
- Simulate order execution with realistic slippage
- Track open positions (entry price, quantity, unrealized P&L)
- Close positions on opposite signals
- Calculate realized P&L on position close
- Generate trade confirmations

**Slippage Model**:
```python
slippage_bps = base_slippage + (order_size_usd / 100000 * 0.1)
slippage_bps = min(slippage_bps, 1.0)  # Cap at 1%
execution_price = price * (1 + slippage_bps) if BUY else price * (1 - slippage_bps)
```

**Key Classes**:
- `PaperTradingEngine`: Main trading simulator
- `VirtualBalance`: Cash + positions tracking
- `Position`: Open position with P&L calculation
- `TradeExecution`: Simulated fill details

**Parameters**:
- `initial_balance: 100` (test) / `1000` (prod)
- `base_slippage: 0.001` (0.1%)
- `commission: 0.0026` (0.26% Kraken taker fee)

**External Dependencies**: `dataclasses`, `datetime`

#### 6. Main Orchestrator (`main.py`)

**Responsibilities**:
- Load configuration from config.yaml and .env
- Initialize all modules with dependency injection
- Start WebSocket manager event loop
- Wire event handlers (tick → signal → risk → trade)
- Handle graceful shutdown (SIGINT, SIGTERM)
- Coordinate logging across modules

**Event Flow**:
1. WebSocket receives tick → emit TickEvent
2. Database writes tick asynchronously
3. Signal Engine processes tick → emit TradeSignal (if crossover)
4. Risk Manager sizes position → emit SizedOrder
5. Paper Trader executes → emit TradeConfirmation
6. Logger writes to console

**Key Classes**:
- `Application`: Main orchestrator
- `EventBus`: Internal pub/sub for module communication
- `ConfigLoader`: YAML + env parsing

**External Dependencies**: `asyncio`, `signal`, `pyyaml`, `python-dotenv`

### Configuration Files

#### `config.yaml`
```yaml
kraken:
  websocket_url: "wss://ws.kraken.com/"
  rest_api_url: "https://api.kraken.com/0/"

trading:
  pairs:
    - "XBT/USD"  # Bitcoin
    - "ETH/USD"  # Ethereum
    - "SOL/USD"  # Solana
    - "MATIC/USD"  # Polygon
    - "AVAX/USD"  # Avalanche

  initial_balance: 100  # $100 for testing

strategy:
  ema_fast_period: 20
  ema_slow_period: 50
  min_bars_required: 51

risk:
  atr_period: 14
  max_portfolio_heat: 0.10
  min_position_pct: 0.02
  max_position_pct: 0.20

execution:
  base_slippage: 0.001
  commission: 0.0026

database:
  host: "${DB_HOST:localhost}"
  port: 5432
  database: "crypto_trading"
  user: "${DB_USER}"
  password: "${DB_PASSWORD}"
```

#### `.env` (gitignored)
```bash
# Kraken API Credentials
KRAKEN_API_KEY=RQkZUycpNOYWqL84G1sVDtVD3FyrY9a0JqXV8DxtcsUEW04Ks9qUJ5Fh
KRAKEN_API_SECRET=XFO14t9c84tdWb+mPQWiK+ulio1/+ugiQhQep8kNgy18gPQSPmJQ34Q87diRUj0RKdnWGzeP+nzPOX08ZxmhZw==

# TimescaleDB
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=your_secure_password_here
```

### Directory Structure

```
stock_analysis/
├── crypto_trading/              # NEW: Crypto trading system
│   ├── __init__.py
│   ├── main.py                  # Orchestrator
│   ├── websocket_manager.py     # Kraken WebSocket client
│   ├── database.py              # TimescaleDB interface
│   ├── signal_engine.py         # EMA crossover detection
│   ├── risk_manager.py          # Volatility-based sizing
│   ├── paper_trader.py          # Virtual trading simulation
│   ├── config.yaml              # Configuration
│   └── requirements.txt         # Dependencies
├── docs/
│   └── plans/
│       └── 2025-11-03-crypto-trading-mvp-design.md  # This document
├── .env                         # API credentials (gitignored)
└── .gitignore                   # Updated to exclude .env
```

## Data Flow Detailed

### 1. Tick Ingestion Flow

```
Kraken WebSocket
  ↓ (ticker update for XBT/USD)
WebSocket Manager
  ↓ validate (price > 0, timestamp fresh)
  ├─→ Database Layer (async write to TimescaleDB)
  └─→ Signal Engine (trigger EMA calculation)
```

### 2. Signal Generation Flow

```
Signal Engine receives tick
  ↓ query last 51 bars from TimescaleDB
  ↓ calculate EMA20 and EMA50
  ↓ detect crossover
  ├─→ NO CROSSOVER: ignore
  └─→ CROSSOVER DETECTED:
       ↓ emit TradeSignal(symbol, direction, timestamp)
       ↓
Risk Manager
  ↓ calculate ATR for all pairs
  ↓ compute allocation weights
  ↓ validate portfolio heat
  ↓ emit SizedOrder(symbol, direction, quantity, price)
  ↓
Paper Trader
  ↓ check balance sufficient
  ↓ apply slippage model
  ↓ update virtual balance
  ↓ track position
  ↓ emit TradeConfirmation
  ↓
Console Logger
  └─→ "[2025-11-03 14:32:15] BUY 0.0123 XBT/USD @ $67,450 (EMA20: 67200, EMA50: 67000)"
```

### 3. Position Management Flow

```
Open Positions: {XBT/USD: LONG 0.0123 @ $67,450}

New Signal: SELL XBT/USD (bearish crossover)
  ↓
Paper Trader detects opposite signal
  ↓ close existing LONG position
  ↓ calculate P&L = (exit_price - entry_price) * quantity - fees
  ↓ update balance
  ↓ emit PositionClosed
  ↓
Console Logger
  └─→ "[2025-11-03 18:45:22] CLOSE LONG XBT/USD: +$24.50 (+3.6%)"
```

## Error Handling & Recovery

### WebSocket Disconnection

```python
try:
    await websocket.recv()
except ConnectionClosed:
    logger.warning("WebSocket disconnected, reconnecting...")
    await reconnect_with_backoff()
```

**Backoff Strategy**: 5s → 10s → 20s → 40s → 60s (max)

### Database Failures

```python
try:
    await db.write_tick(tick)
except DatabaseError as e:
    logger.error(f"DB write failed: {e}")
    # Continue processing (don't block trading on DB issues)
    # Store in local buffer for retry
```

### Missing Data

```python
if len(bars) < min_bars_required:
    logger.debug(f"Insufficient bars for {symbol}: {len(bars)}/{min_bars_required}")
    return None  # Skip signal generation
```

## Security Considerations

1. **API Credentials**: Never commit to git, use .env file, rotate quarterly
2. **Database Access**: Use connection pooling with max connections limit
3. **WebSocket Auth**: Use token-based auth (Kraken GetWebSocketsToken)
4. **Input Validation**: Validate all incoming data (price > 0, volume >= 0)
5. **Rate Limiting**: Respect Kraken rate limits (built into CCXT if used)

## Performance Targets

- **WebSocket Latency**: < 100ms from Kraken tick to database write
- **Signal Generation**: < 500ms from tick to signal emission
- **Memory Usage**: < 500MB for 100 pairs
- **Database Write**: < 50ms per tick (async, non-blocking)
- **Reconnection**: < 5s average downtime on disconnect

## Testing Strategy

### Unit Tests
- `test_ema_calculator.py`: Verify EMA calculations match talib
- `test_crossover_detector.py`: Test all crossover scenarios
- `test_position_sizer.py`: Validate allocation bounds
- `test_paper_trader.py`: Verify P&L calculations

### Integration Tests
- `test_websocket_to_db.py`: End-to-end tick ingestion
- `test_signal_to_trade.py`: Full signal → execution flow
- `test_reconnection.py`: WebSocket recovery

### Manual Testing
- Paper trade 5 pairs for 7 days
- Verify against manual EMA calculations
- Test forced disconnection recovery
- Validate balance reconciliation

## Deployment

### Prerequisites
- Python 3.11 (NOT 3.12+ per CLAUDE.md)
- TimescaleDB 2.x installed and running
- Kraken account with API keys

### Setup Steps
```bash
# 1. Install dependencies
cd crypto_trading
pip install -r requirements.txt

# 2. Set up database
psql -U postgres -c "CREATE DATABASE crypto_trading;"
psql -U postgres -d crypto_trading -f schema.sql

# 3. Configure credentials
cp .env.example .env
# Edit .env with actual credentials

# 4. Run system
python main.py
```

### Monitoring
- Console logs for all trading activity
- Database queries for historical performance
- Manual P&L reconciliation daily

## Future Enhancements (Out of MVP Scope)

1. **Advanced Strategies**: Bollinger Bands, RSI, MACD, Donchian Channels
2. **Machine Learning**: PyTorch LSTM for price prediction
3. **Risk Enhancements**: Correlation tracking, drawdown limits, circuit breakers
4. **Microservices**: Extract modules to independent services with RabbitMQ
5. **Web Dashboard**: Real-time Grafana/Streamlit monitoring
6. **Live Trading**: Transition from paper to real execution
7. **Multi-Exchange**: Add Binance, Coinbase support
8. **On-Chain Data**: Integrate Glassnode metrics

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| WebSocket connection drops | No data ingestion | Exponential backoff reconnection |
| Database unavailable | Data loss | Local buffer with retry queue |
| Insufficient historical data | No signals | Require 51 bars minimum |
| High volatility causing losses | Capital depletion | Portfolio heat limits, position caps |
| API rate limiting | Throttled operations | Built-in rate limiting, request batching |
| Slippage worse than modeled | Lower returns | Conservative slippage assumptions |

## Success Metrics

### Technical Metrics
- WebSocket uptime: > 99.5%
- Signal generation latency: < 500ms
- Database write success: > 99.9%
- Reconnection time: < 10s average

### Trading Metrics
- Paper trading runs 7+ days without errors
- Position sizing stays within bounds (2-20%)
- Portfolio heat never exceeds 10%
- Slippage < 0.5% average
- P&L tracking accuracy: 100% vs manual calculation

## Conclusion

This modular monolith design delivers production-grade architecture while maintaining MVP simplicity. Clean module boundaries enable rapid iteration and future microservices extraction. The system balances institutional patterns (event-driven, risk management, proper error handling) with pragmatic implementation for rapid deployment.

**Next Steps**: Set up git worktree, create detailed implementation plan, begin coding.
