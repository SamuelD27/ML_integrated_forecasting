# Alpaca Paper Trading Pipeline

This document describes how to use the Alpaca paper trading integration for testing trading strategies with simulated money.

## Prerequisites

1. **Alpaca Account**: Sign up for a free account at [alpaca.markets](https://alpaca.markets)
2. **API Keys**: Get your paper trading API keys from the Alpaca dashboard
3. **Python Package**: Install `alpaca-py`:
   ```bash
   pip install alpaca-py
   ```

## Configuration

### Environment Variables

Add your Alpaca credentials to `.env`:

```bash
ALPACA_API_KEY=your_paper_api_key
ALPACA_API_SECRET=your_paper_api_secret
```

**Important**: Use PAPER trading keys, not live trading keys. Paper keys can be found in your Alpaca dashboard under "Paper Trading".

## Quick Start

```python
from deployment.alpaca_paper_trader import AlpacaPaperTrader

# Initialize (uses env vars automatically)
trader = AlpacaPaperTrader()

# Check account
account = trader.get_account()
print(f"Equity: ${account.equity:,.2f}")

# Get quotes
quote = trader.get_quote('AAPL')
print(f"AAPL: ${quote['mid']:.2f}")

# Submit a market order
order = trader.submit_order(
    symbol='AAPL',
    qty=10,
    side='buy',
    order_type='market'
)
print(f"Order ID: {order.order_id}")

# Check positions
positions = trader.get_positions()
for pos in positions:
    print(f"{pos.symbol}: {pos.qty} shares")

# Close a position
trader.close_position('AAPL')
```

## API Reference

### AlpacaPaperTrader

Main trading client class.

#### Initialization

```python
from deployment.alpaca_paper_trader import AlpacaPaperTrader, RiskLimits

# Default risk limits
trader = AlpacaPaperTrader()

# Custom risk limits
limits = RiskLimits(
    max_position_pct=0.20,      # Max 20% in one position
    max_order_value=10000.0,    # Max $10k per order
    max_daily_loss_pct=0.03,    # Max 3% daily drawdown
    min_buying_power_reserve=0.15,  # Keep 15% cash
    max_positions=15,           # Max 15 positions
)
trader = AlpacaPaperTrader(risk_limits=limits)
```

#### Account Methods

```python
# Get account snapshot
account = trader.get_account()
# Returns: AccountSnapshot with equity, cash, buying_power, daily_pnl

# Check connection
is_connected = trader.is_connected()
```

#### Quote Methods

```python
# Single quote
quote = trader.get_quote('AAPL')
# Returns: {'bid': 175.50, 'ask': 175.52, 'mid': 175.51, ...}

# Multiple quotes
quotes = trader.get_quotes(['AAPL', 'MSFT', 'GOOGL'])
# Returns: {'AAPL': {...}, 'MSFT': {...}, 'GOOGL': {...}}
```

#### Order Methods

```python
# Market order
order = trader.submit_order(
    symbol='AAPL',
    qty=10,
    side='buy',  # or 'sell'
    order_type='market'
)

# Limit order
order = trader.submit_order(
    symbol='AAPL',
    qty=10,
    side='buy',
    order_type='limit',
    limit_price=170.00,
    time_in_force='gtc'  # 'day', 'gtc', 'ioc', 'fok'
)

# Get order status
order = trader.get_order(order_id)

# Cancel order
success = trader.cancel_order(order_id)

# Cancel all orders
count = trader.cancel_all_orders()

# Get pending orders
orders = trader.get_pending_orders()
```

#### Position Methods

```python
# All positions
positions = trader.get_positions()

# Single position
position = trader.get_position('AAPL')
# Returns: Position with qty, avg_entry_price, unrealized_pnl, etc.

# Close position
order = trader.close_position('AAPL')

# Close all positions
orders = trader.close_all_positions()
```

#### Risk Checks

```python
# Check before order
result, message = trader.check_risk(
    symbol='AAPL',
    qty=100,
    side='buy',
    limit_price=175.00  # optional
)

if result == RiskCheckResult.PASSED:
    print("Order allowed")
else:
    print(f"Blocked: {message}")
```

Risk check failures:
- `FAILED_INSUFFICIENT_BUYING_POWER` - Not enough cash
- `FAILED_POSITION_LIMIT` - Too many positions
- `FAILED_ORDER_SIZE` - Order too large
- `FAILED_CONCENTRATION` - Position would be too large % of portfolio
- `FAILED_DAILY_LOSS` - Daily loss limit exceeded

#### Portfolio Allocation

```python
# Execute target allocation from portfolio optimizer
allocation = {
    'AAPL': 0.25,  # 25% weight
    'MSFT': 0.25,
    'GOOGL': 0.25,
    'AMZN': 0.25,
}

orders = trader.execute_portfolio_allocation(
    target_allocation=allocation,
    rebalance=True  # Sell over-allocated positions
)
```

## Integration with Portfolio Optimizer

```python
from portfolio.cvar_allocator import CVaRAllocator
from deployment.alpaca_paper_trader import AlpacaPaperTrader

# 1. Run portfolio optimization
allocator = CVaRAllocator(tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
weights = allocator.optimize()

# 2. Convert to allocation dict
allocation = dict(zip(allocator.tickers, weights))

# 3. Execute via paper trader
trader = AlpacaPaperTrader()
orders = trader.execute_portfolio_allocation(allocation)

# 4. Verify positions
for pos in trader.get_positions():
    print(f"{pos.symbol}: {pos.qty} shares, P&L: ${pos.unrealized_pnl:,.2f}")
```

## Running Tests

### Quick Test

```bash
python test_alpaca_pipeline.py
```

### Full Test Suite

```bash
pytest test_alpaca_pipeline.py -v -s
```

### Test Output Example

```
=== Account Info ===
Equity:       $100,000.00
Cash:         $100,000.00
Buying Power: $400,000.00
Daily P&L:    $0.00 (+0.00%)

=== Current Positions ===
  No open positions

=== Sample Quotes ===
  AAPL: $175.50
  MSFT: $415.30
  GOOGL: $175.80

Paper trading pipeline ready!
```

## Risk Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_pct` | 25% | Max % of portfolio in single position |
| `max_order_value` | $50,000 | Max value per order |
| `max_daily_loss_pct` | 5% | Stop trading after this daily loss |
| `min_buying_power_reserve` | 10% | Keep this % in cash |
| `max_positions` | 20 | Maximum concurrent positions |

## Pipeline Flow

```
Portfolio Optimizer Output
          ↓
    Risk Checks
          ↓
   Order Generation
          ↓
  Alpaca Submission → Order Status
          ↓
   Position Tracker ← Fills
          ↓
    P&L Tracking
```

## Troubleshooting

### Connection Issues

```python
# Check credentials
import os
print(f"API Key: {os.getenv('ALPACA_API_KEY')[:8]}...")
print(f"Secret: {os.getenv('ALPACA_API_SECRET')[:8]}...")
```

### Order Rejections

```python
# Check buying power
account = trader.get_account()
print(f"Available: ${account.buying_power:,.2f}")

# Run risk check first
result, msg = trader.check_risk('AAPL', 100, 'buy')
print(f"Risk: {result.value} - {msg}")
```

### Market Hours

Paper trading follows real market hours (9:30 AM - 4:00 PM ET). Orders submitted outside market hours will queue until open.

## Best Practices

1. **Always use paper trading for testing** - Never test with real money
2. **Run risk checks before orders** - Catch issues before submission
3. **Start with small orders** - Test pipeline with minimal amounts
4. **Monitor daily P&L** - Stop trading if hitting loss limits
5. **Log all trades** - Use `trader.get_trade_log()` for debugging
