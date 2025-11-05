# Test Scenarios - Real-Time Feature Pipeline

## Scenario 1: Sub-10ms Feature Computation

**Pressure Type**: Time Pressure + Performance Requirements

**User Message**:
```
We need to compute 50+ technical indicators in real-time for our high-frequency trading system. The ML model expects features every time a new price tick arrives (100-1000 ticks/second).

Current problem: Our feature computation takes 150ms per tick, which is WAY too slow. We're missing trades because predictions arrive late.

Requirements:
- Must compute all features in < 10ms (sub-10ms latency target)
- Features: 20-day SMA, 50-day SMA, RSI(14), MACD, Bollinger Bands, ATR(14), volume profile, price momentum (5 timeframes), correlation to SPY
- New price tick arrives, need features IMMEDIATELY
- Cannot miss any ticks (must handle 1000 ticks/second = 1 tick every 1ms)

Can you optimize the feature pipeline to meet this latency requirement? This is blocking production deployment.
```

**What We're Testing**:
1. Does agent use incremental computation (not recalculate from scratch)?
2. Does agent cache expensive computations (moving averages, indicators)?
3. Does agent vectorize operations (numpy, not Python loops)?
4. Does agent measure and validate latency (< 10ms)?
5. Does agent use efficient data structures (ring buffers, not lists)?

**Expected Baseline Behavior** (without skill):
- ❌ Recalculates everything from scratch on each tick
- ❌ Uses Python loops instead of vectorized operations
- ❌ Doesn't cache intermediate results (moving sums, EWMAs)
- ❌ No latency measurement or validation
- ❌ Inefficient data structures (appending to lists)

**With Skill**:
- ✅ Incremental updates (O(1) updates, not O(n) recalculation)
- ✅ Caching strategy for expensive features
- ✅ Vectorized numpy operations
- ✅ Latency measurement and assertion (< 10ms)
- ✅ Efficient data structures (ring buffers for rolling windows)

---

## Scenario 2: Caching Strategy for Expensive Features

**Pressure Type**: Complexity + Performance

**User Message**:
```
Our feature pipeline computes 100+ features, but many are VERY expensive:
- Correlation matrix (20 stocks): 500ms
- Covariance matrix: 300ms
- PCA components: 200ms
- Factor exposures (Fama-French): 150ms

These don't change on every tick - correlation updates maybe every 5 minutes, PCA every hour. But currently we recompute them EVERY TIME.

Requirements:
- Cache expensive features with TTL (time-to-live)
- Invalidate cache when underlying data changes significantly
- Track cache hit rate (should be >90%)
- Ensure cache doesn't grow unbounded (memory limit)

Implement a smart caching layer that reduces computation time from 1150ms to < 50ms.
```

**What We're Testing**:
1. Does agent implement TTL-based caching?
2. Does agent track cache hit rates?
3. Does agent implement cache invalidation logic?
4. Does agent handle memory limits (LRU eviction)?
5. Does agent measure cache effectiveness?

**Expected Baseline Behavior** (without skill):
- ❌ No caching (recalculates everything)
- ❌ Or simple dict caching (no TTL, no eviction)
- ❌ No cache hit rate tracking
- ❌ No validation that cache is working
- ❌ Memory grows unbounded (no LRU eviction)

**With Skill**:
- ✅ TTL-based cache with configurable expiry
- ✅ Cache hit rate tracking and logging
- ✅ Automatic invalidation on data changes
- ✅ LRU eviction for memory management
- ✅ Assertions that cache hit rate > threshold

---

## Scenario 3: Incremental Feature Updates

**Pressure Type**: Authority + Complexity

**User Message**:
```
Our quant researcher says we MUST use incremental feature updates, not full recalculation. He showed me this paper where they update moving averages in O(1) time instead of O(n).

Current implementation:
- 20-day SMA: Recalculates sum of last 20 values every tick → O(n)
- RSI: Recalculates gains/losses over 14 periods → O(n)
- EMA: Uses full history → O(n)

Target implementation:
- SMA: Update = (new_value - old_value) / window_size → O(1)
- RSI: Track running gains/losses → O(1)
- EMA: Update = alpha * new_value + (1-alpha) * old_ema → O(1)

Implement incremental updates for ALL rolling window features. The researcher says this is standard in HFT, so we need to get it right.
```

**What We're Testing**:
1. Does agent implement O(1) updates for moving averages?
2. Does agent implement O(1) updates for RSI?
3. Does agent implement O(1) updates for EMA/EWMA?
4. Does agent maintain running sums/state for incremental updates?
5. Does agent validate that updates match full calculation (within numerical tolerance)?

**Expected Baseline Behavior** (without skill):
- ❌ Recalculates from scratch (O(n))
- ❌ Doesn't maintain running state
- ❌ No validation of incremental vs full calculation
- ❌ Uses pandas .rolling() which recalculates

**With Skill**:
- ✅ O(1) incremental updates for SMA, EMA, RSI
- ✅ Maintains running sums/state
- ✅ Validates incremental == full calculation (within tolerance)
- ✅ Benchmarks show O(1) scaling

---

## Scenario 4: Vectorization for Batch Processing

**Pressure Type**: Sunk Cost + Time Pressure

**User Message**:
```
I spent 2 days trying to optimize our feature pipeline but it's still too slow. The bottleneck is computing features for 50 stocks in parallel.

Current implementation (Python loops):
```python
for stock in stocks:
    features[stock] = {
        'sma_20': calculate_sma(prices[stock], 20),
        'rsi': calculate_rsi(prices[stock], 14),
        'macd': calculate_macd(prices[stock]),
        # ... 47 more features
    }
```
This takes 500ms for 50 stocks (10ms per stock).

Target: Use numpy vectorization to compute all 50 stocks in < 50ms (100x speedup).

Can you vectorize this? I've already invested 2 days and my boss expects this done by EOD.
```

**What We're Testing**:
1. Does agent vectorize feature computation across stocks?
2. Does agent use numpy broadcasting for multi-stock operations?
3. Does agent avoid Python loops (for stocks, for features)?
4. Does agent benchmark vectorized vs loop implementation?
5. Does agent validate that vectorized output matches loop output?

**Expected Baseline Behavior** (without skill):
- ❌ Uses Python loops for stocks
- ❌ Uses Python loops for features
- ❌ Doesn't leverage numpy vectorization
- ❌ No benchmarking of speedup

**With Skill**:
- ✅ Vectorized across stocks (processes all 50 simultaneously)
- ✅ Numpy broadcasting for features
- ✅ No Python loops
- ✅ Benchmark shows 10-100x speedup
- ✅ Validation that outputs match

---

## Scenario 5: Memory-Efficient Rolling Windows

**Pressure Type**: Complexity + Authority

**User Message**:
```
Our production system ran out of memory after 3 hours. Turns out we're storing EVERY historical price to calculate rolling features. After 3 hours at 1 tick/second, that's 10,800 prices per stock × 100 stocks = 1M+ data points in memory.

Senior engineer says we should use ring buffers (circular buffers) that only keep the last N values needed for features. For a 50-period SMA, we only need 50 values, not 10,800.

Requirements:
- Implement ring buffer for rolling windows
- Maximum memory = window_size, not unlimited growth
- O(1) append and O(1) access to rolling window
- Works with all features (SMA, RSI, Bollinger Bands, etc.)

This is critical - system keeps crashing in production. Need this fixed ASAP.
```

**What We're Testing**:
1. Does agent implement ring buffer (circular buffer)?
2. Does agent limit memory to window_size?
3. Does agent maintain O(1) append and access?
4. Does agent integrate ring buffer with feature computation?
5. Does agent validate memory usage doesn't grow unbounded?

**Expected Baseline Behavior** (without skill):
- ❌ Appends to list (grows unbounded)
- ❌ Or uses deque but doesn't optimize for rolling windows
- ❌ No memory usage tracking
- ❌ No validation that memory is bounded

**With Skill**:
- ✅ Ring buffer implementation (fixed-size array)
- ✅ O(1) append, O(1) access to window
- ✅ Memory bounded at window_size × dtype_size
- ✅ Integrated with all rolling features
- ✅ Memory usage assertions (bounded)

---

## Pressure Analysis

### Pressure Type Distribution

1. **Time Pressure**: Scenarios 1, 4 (need optimization NOW)
2. **Performance Requirements**: Scenarios 1, 2 (< 10ms, > 90% cache hit)
3. **Complexity**: Scenarios 2, 3, 5 (caching, incremental, ring buffers)
4. **Authority**: Scenarios 3, 5 (researcher says, senior engineer says)
5. **Sunk Cost**: Scenario 4 (spent 2 days already)

### Rationalization Predictions

| Excuse | Scenario | Reality |
|--------|----------|---------|
| "Recalculation is simple, fast enough" | 1 | 150ms >> 10ms, fails requirement |
| "Simple caching with dict works" | 2 | No TTL, no eviction, memory leak |
| "Pandas .rolling() is efficient" | 3 | Recalculates O(n), not incremental O(1) |
| "For loops are readable" | 4 | 10ms/stock vs 1ms/stock with vectorization |
| "Deque handles rolling windows" | 5 | Doesn't optimize memory, no ring buffer |

### Expected Failure Modes

**Without Skill**:
1. **Latency violations**: Features take 50-150ms, not < 10ms
2. **Memory leaks**: Unbounded growth from storing all history
3. **O(n) recalculation**: No incremental updates
4. **Cache misses**: No caching or ineffective caching
5. **No vectorization**: Python loops for batch processing

**With Skill**:
- Sub-10ms latency with incremental updates + caching
- Bounded memory with ring buffers
- O(1) updates for rolling features
- >90% cache hit rates for expensive features
- 10-100x speedup from vectorization

---

## Success Criteria

### Baseline Tests (RED Phase)

For each scenario, agent should:
- ❌ Use full recalculation (O(n)), not incremental (O(1))
- ❌ No caching or ineffective caching
- ❌ Python loops instead of vectorization
- ❌ Unbounded memory growth
- ❌ No latency measurement or validation

### With Skill (GREEN Phase)

Agent should:
- ✅ Incremental O(1) updates for rolling features
- ✅ TTL-based caching with LRU eviction
- ✅ Numpy vectorization for batch processing
- ✅ Ring buffers for memory-efficient rolling windows
- ✅ Latency measurement and assertion (< threshold)

### Refactor Phase (REFACTOR)

Test that skill prevents:
- ❌ "Recalculation is fast enough" rationalization
- ❌ "Simple dict caching works" excuse
- ❌ "Pandas handles it" assumption
- ❌ "For loops are clearer" preference
- ❌ "Memory isn't an issue" dismissal

---

## File Outputs Expected

For baseline tests, each scenario should generate:
- `feature_pipeline_baseline_scenario[1-5].py`

For skill tests, each scenario should generate:
- `feature_pipeline_with_skill_scenario[1-5].py`

Compare implementations to measure skill effectiveness.

---

## Notes for Skill Design

Based on these scenarios, the skill must teach:

1. **Incremental Updates**: O(1) updates for SMA, EMA, RSI, Bollinger Bands
2. **Caching Strategy**: TTL-based cache with LRU eviction, hit rate tracking
3. **Vectorization**: Numpy broadcasting for multi-stock computation
4. **Ring Buffers**: Fixed-size circular buffers for rolling windows
5. **Latency Measurement**: Benchmark and assert < threshold
6. **Memory Profiling**: Track and bound memory usage

Each pattern must include:
- Complete implementation code
- Benchmarking (latency, cache hit rate, memory)
- Validation (incremental == full, memory bounded)
- Common mistakes (Bad vs Good)
- Complexity analysis (O(1) vs O(n))
