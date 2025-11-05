---
name: real-time-feature-pipeline
description: Use when building real-time feature pipelines for ML trading systems - enforces incremental update validation, built-in latency assertions, memory profiling, and systematic application of O(1) patterns to ensure sub-10ms inference with bounded memory, preventing numerical drift and performance degradation
---

# Real-Time Feature Pipeline

## Overview

**Real-time feature engineering is optimization + validation + enforcement.** Every feature pipeline must validate incremental updates match full calculations, assert latency < threshold, profile memory usage, and apply O(1) patterns systematically. Implementation alone causes numerical drift and performance degradation.

**Core principle:** Achieving fast performance ≠ maintaining correctness. Always validate incremental updates and enforce latency/memory bounds.

## When to Use

Use this skill when:
- Building real-time feature pipelines (sub-10ms latency)
- Implementing incremental indicator updates (RSI, MACD, Moving Averages)
- Optimizing feature computation for streaming data
- Managing memory for long-running systems
- Caching expensive feature calculations
- Vectorizing batch feature computation

**Don't skip validation because:**
- "Performance target met" (numerical errors can accumulate)
- "Circular buffer works" (others don't know why you chose it)
- "Benchmarked externally" (can degrade without built-in assertions)
- "Memory efficient in theory" (should measure actual usage)
- "Only optimize when explicitly required" (real-time ALWAYS needs it)

## Implementation Checklist

Before deploying ANY real-time feature pipeline:

- [ ] Incremental updates validated against full calculation (< 1e-6 tolerance)
- [ ] Latency assertion built into pipeline (< threshold, not external benchmark)
- [ ] Memory profiled and bounded (assert < max_bytes)
- [ ] O(1) patterns for all rolling windows (circular buffers, running sums)
- [ ] Decision framework documented (why circular buffer vs deque vs list)
- [ ] Cache hit rate tracked for expensive features (> 90%)

---

## Pattern 1: Validated Incremental Updates

### Always Validate Incremental == Full Calculation

**CRITICAL: Incremental updates can accumulate numerical errors. Validate during testing.**

```python
import numpy as np
from typing import Optional


class ValidatedIncrementalRSI:
    """
    RSI with incremental updates + validation.

    Validates that incremental calculation matches full calculation
    within numerical tolerance.
    """

    def __init__(self, period: int = 14, validate: bool = False):
        self.period = period
        self.validate = validate  # Enable during testing

        # Incremental state
        self.prev_price: Optional[float] = None
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None

        # History for full calculation (validation only)
        if self.validate:
            self.price_history: list = []

    def update(self, price: float) -> Optional[float]:
        """
        Update RSI with new price.

        Args:
            price: New price value

        Returns:
            RSI value (0-100), or None if not enough data

        Raises:
            AssertionError: If incremental != full (when validate=True)
        """
        if self.validate:
            self.price_history.append(price)

        if self.prev_price is None:
            self.prev_price = price
            return None

        # Calculate price change
        change = price - self.prev_price
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        self.prev_price = price

        # Initialize or update Wilder's smoothing
        if self.avg_gain is None:
            # Need period+1 prices for initialization
            if not self.validate or len(self.price_history) < self.period + 1:
                return None

            # Calculate initial averages
            changes = np.diff(self.price_history)
            gains = np.maximum(changes, 0)
            losses = np.maximum(-changes, 0)
            self.avg_gain = gains[:self.period].mean()
            self.avg_loss = losses[:self.period].mean()
        else:
            # Incremental update (Wilder's smoothing)
            alpha = 1.0 / self.period
            self.avg_gain = (1 - alpha) * self.avg_gain + alpha * gain
            self.avg_loss = (1 - alpha) * self.avg_loss + alpha * loss

        # Calculate RSI
        if self.avg_loss == 0:
            rsi_incremental = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi_incremental = 100.0 - (100.0 / (1.0 + rs))

        # VALIDATE: Incremental matches full calculation
        if self.validate and len(self.price_history) > self.period:
            rsi_full = self._calculate_full_rsi()
            assert abs(rsi_incremental - rsi_full) < 1e-4, \
                f"Incremental RSI {rsi_incremental:.6f} != full {rsi_full:.6f}"

        return rsi_incremental

    def _calculate_full_rsi(self) -> float:
        """Calculate RSI from scratch (for validation)."""
        prices = np.array(self.price_history)
        changes = np.diff(prices)
        gains = np.maximum(changes, 0)
        losses = np.maximum(-changes, 0)

        # Wilder's smoothing
        avg_gain = gains[:self.period].mean()
        avg_loss = losses[:self.period].mean()

        for i in range(self.period, len(gains)):
            alpha = 1.0 / self.period
            avg_gain = (1 - alpha) * avg_gain + alpha * gains[i]
            avg_loss = (1 - alpha) * avg_loss + alpha * losses[i]

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
```

**Testing**:
```python
# Enable validation during testing
rsi = ValidatedIncrementalRSI(period=14, validate=True)

# Process ticks
for price in price_stream:
    rsi_value = rsi.update(price)
    # Automatically validates incremental == full
    # Raises AssertionError if mismatch
```

---

## Pattern 2: Built-in Latency Enforcement

### Assert Latency Requirements in Production

**Don't just measure latency externally—enforce it within the pipeline.**

```python
import time
from typing import Dict


class LatencyEnforcedPipeline:
    """
    Feature pipeline with built-in latency enforcement.

    Raises AssertionError if latency exceeds threshold.
    """

    def __init__(
        self,
        max_latency_ms: float = 10.0,
        enable_assertions: bool = True
    ):
        self.max_latency_ms = max_latency_ms
        self.enable_assertions = enable_assertions

        # Latency tracking
        self.latency_history: list = []
        self.max_latency_observed: float = 0.0

        # Initialize indicators
        self._initialize_indicators()

    def process_tick(self, tick: Dict) -> Dict:
        """
        Process tick and compute features.

        Args:
            tick: Dict with 'price', 'volume', 'high', 'low'

        Returns:
            Dict of computed features

        Raises:
            AssertionError: If latency > max_latency_ms
        """
        start = time.perf_counter()

        # Compute features
        features = self._compute_features(tick)

        # Measure latency
        latency_sec = time.perf_counter() - start
        latency_ms = latency_sec * 1000.0

        # Track latency
        self.latency_history.append(latency_ms)
        self.max_latency_observed = max(self.max_latency_observed, latency_ms)

        # ENFORCE: Latency requirement
        if self.enable_assertions:
            assert latency_ms < self.max_latency_ms, \
                f"Latency {latency_ms:.3f}ms exceeds threshold {self.max_latency_ms}ms"

        return features

    def get_latency_stats(self) -> Dict:
        """Get latency statistics."""
        if not self.latency_history:
            return {}

        latencies = np.array(self.latency_history)
        return {
            'mean_ms': float(np.mean(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'max_ms': float(np.max(latencies)),
            'num_violations': int((latencies > self.max_latency_ms).sum())
        }
```

**Usage**:
```python
# Pipeline fails if latency > 10ms
pipeline = LatencyEnforcedPipeline(max_latency_ms=10.0)

for tick in tick_stream:
    features = pipeline.process_tick(tick)
    # Automatically enforces latency
    # Raises AssertionError if > 10ms
```

---

## Pattern 3: Memory Profiling and Bounds

### Profile Memory and Assert Bounds

**Measure actual memory usage, don't just claim efficiency.**

```python
import sys
from typing import List


class MemoryProfiledPipeline:
    """
    Feature pipeline with memory profiling and bounds.

    Tracks memory usage and enforces maximum.
    """

    def __init__(self, max_memory_kb: float = 10.0):
        self.max_memory_kb = max_memory_kb
        self.buffers: List = []

        # Initialize circular buffers
        self._initialize_buffers()

    def measure_memory_footprint(self) -> Dict[str, float]:
        """
        Measure actual memory usage in KB.

        Returns:
            Dict with memory breakdown
        """
        total_bytes = 0
        breakdown = {}

        # Measure each buffer
        for i, buffer in enumerate(self.buffers):
            buffer_bytes = sys.getsizeof(buffer.buffer)
            total_bytes += buffer_bytes
            breakdown[f'buffer_{i}'] = buffer_bytes / 1024.0

        # Measure indicator state
        indicator_bytes = sum(
            sys.getsizeof(getattr(self, attr))
            for attr in dir(self)
            if not attr.startswith('_')
        )
        total_bytes += indicator_bytes
        breakdown['indicators'] = indicator_bytes / 1024.0

        breakdown['total_kb'] = total_bytes / 1024.0

        return breakdown

    def validate_memory_bounds(self) -> None:
        """
        Validate memory usage is within bounds.

        Raises:
            AssertionError: If memory exceeds max_memory_kb
        """
        memory = self.measure_memory_footprint()
        total_kb = memory['total_kb']

        assert total_kb < self.max_memory_kb, \
            f"Memory {total_kb:.2f}KB exceeds {self.max_memory_kb}KB"

        print(f"✓ Memory validated: {total_kb:.2f}KB (< {self.max_memory_kb}KB)")


# Validate memory periodically
pipeline = MemoryProfiledPipeline(max_memory_kb=10.0)

# After processing many ticks
pipeline.validate_memory_bounds()  # Asserts memory < 10KB
```

---

## Pattern 4: Circular Buffer with Running Sum

### O(1) Rolling Window Operations

```python
class CircularBuffer:
    """
    Fixed-size circular buffer with O(1) operations.

    Maintains running sum for O(1) mean calculation.
    """

    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float64)
        self.index = 0
        self.count = 0
        self.sum = 0.0

    def append(self, value: float) -> None:
        """
        Add value and update running sum.

        Time complexity: O(1)
        """
        if self.count == self.size:
            # Remove old value from sum
            self.sum -= self.buffer[self.index]
        else:
            self.count += 1

        # Add new value
        self.buffer[self.index] = value
        self.sum += value

        # Circular increment
        self.index = (self.index + 1) % self.size

    def get_mean(self) -> float:
        """
        Get mean value.

        Time complexity: O(1) via running sum
        """
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def get_std(self) -> float:
        """
        Get standard deviation.

        Time complexity: O(n) where n = window size
        """
        if self.count < 2:
            return 0.0

        arr = self.get_array()
        return float(np.std(arr, dtype=np.float64))

    def get_array(self) -> np.ndarray:
        """Get ordered array (oldest to newest)."""
        if self.count < self.size:
            return self.buffer[:self.count]

        # Reorder circular buffer
        return np.concatenate([
            self.buffer[self.index:],
            self.buffer[:self.index]
        ])
```

---

## Pattern 5: Decision Framework Documentation

### Document Tradeoffs for Each Choice

**Always explain WHY you chose a data structure.**

```python
"""
Data Structure Selection for Rolling Windows:

┌────────────────┬──────────┬──────────┬───────────┬──────────────────────┐
│ Structure      │ Append   │ Mean     │ Memory    │ Use When             │
├────────────────┼──────────┼──────────┼───────────┼──────────────────────┤
│ Circular Buffer│ O(1)     │ O(1)*    │ Fixed     │ Frequent mean/sum    │
│                │          │          │           │ Real-time indicators │
├────────────────┼──────────┼──────────┼───────────┼──────────────────────┤
│ collections    │ O(1)     │ O(n)     │ Variable  │ FIFO without stats   │
│ .deque         │          │          │           │ Simple queues        │
├────────────────┼──────────┼──────────┼───────────┼──────────────────────┤
│ List           │ O(1)†    │ O(n)     │ Unbounded │ Temporary storage    │
│                │          │          │           │ NOT for rolling      │
└────────────────┴──────────┴──────────┴───────────┴──────────────────────┘

*With running sum (add O(n) for std dev)
†Amortized, but requires .pop(0) which is O(n)

Chose Circular Buffer because:
- Need O(1) mean for SMA indicators
- Fixed memory is critical for long-running systems
- Reordering overhead (O(n)) acceptable for infrequent full array access
"""
```

---

## Pattern 6: TTL-based Caching with Hit Rate Tracking

### Cache Expensive Features with Metrics

```python
from typing import Any, Optional
from dataclasses import dataclass
from functools import wraps
import time


@dataclass
class CacheEntry:
    value: Any
    timestamp: float
    ttl: float

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class TTLCache:
    """
    Time-to-live cache with LRU eviction and hit rate tracking.

    Tracks cache hit rate to validate effectiveness.
    """

    def __init__(self, max_size: int = 100, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}

        # Metrics
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Returns:
            Cached value if valid, None otherwise
        """
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                self.hits += 1
                return entry.value
            else:
                # Expired, remove
                del self.cache[key]

        self.misses += 1
        return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl

        # LRU eviction if full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]

        self.cache[key] = CacheEntry(
            value=value,
            timestamp=time.time(),
            ttl=ttl
        )

    def get_hit_rate(self) -> float:
        """
        Get cache hit rate.

        Returns:
            Hit rate (0.0 to 1.0)
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def validate_hit_rate(self, min_hit_rate: float = 0.9) -> None:
        """
        Validate cache hit rate meets minimum.

        Args:
            min_hit_rate: Minimum acceptable hit rate (default 90%)

        Raises:
            AssertionError: If hit rate < min_hit_rate
        """
        hit_rate = self.get_hit_rate()
        assert hit_rate >= min_hit_rate, \
            f"Cache hit rate {hit_rate:.2%} < minimum {min_hit_rate:.2%}"


# Usage
cache = TTLCache(max_size=100, default_ttl=300.0)  # 5 min TTL

def expensive_feature(symbol: str) -> float:
    """Compute expensive feature (e.g., correlation matrix)."""
    # Check cache first
    cached = cache.get(f'feature_{symbol}')
    if cached is not None:
        return cached

    # Compute expensive feature
    result = compute_expensive_feature(symbol)

    # Cache result
    cache.put(f'feature_{symbol}', result, ttl=300.0)

    return result

# Validate cache effectiveness
cache.validate_hit_rate(min_hit_rate=0.9)  # Assert hit rate > 90%
```

---

## Common Mistakes

### Mistake 1: No Validation of Incremental Updates
❌ **Bad:**
```python
# Incremental RSI without validation
self.avg_gain = (self.avg_gain * 13 + gain) / 14
return rsi  # Hope it's correct!
```

✅ **Good:**
```python
rsi_incremental = self.update_incremental(price)

if self.validate:
    rsi_full = self.calculate_full(self.price_history)
    assert abs(rsi_incremental - rsi_full) < 1e-6
```

### Mistake 2: External Latency Measurement Only
❌ **Bad:**
```python
# Measure latency in separate benchmark file
start = time.perf_counter()
features = pipeline.process_tick(tick)
latency = time.perf_counter() - start
print(f"Latency: {latency}ms")  # Just logs
```

✅ **Good:**
```python
# Built into pipeline
def process_tick(self, tick):
    start = time.perf_counter()
    features = self._compute(tick)
    latency_ms = (time.perf_counter() - start) * 1000

    assert latency_ms < self.max_latency_ms  # Enforces
```

### Mistake 3: No Memory Profiling
❌ **Bad:**
```python
# Claim efficiency without measurement
# "Uses circular buffer, so memory is bounded"
```

✅ **Good:**
```python
def validate_memory(self):
    memory_kb = self.measure_memory_footprint()['total_kb']
    assert memory_kb < self.max_memory_kb, \
        f"Memory {memory_kb:.2f}KB > {self.max_memory_kb}KB"
```

### Mistake 4: No Cache Hit Rate Tracking
❌ **Bad:**
```python
# Simple dict caching
cache = {}
if key in cache:
    return cache[key]  # No idea if effective
```

✅ **Good:**
```python
cache = TTLCache()
result = cache.get(key)
# ...later...
cache.validate_hit_rate(min_hit_rate=0.9)  # Assert effectiveness
```

---

## ANTI-PATTERNS (What NOT To Do)

### ❌ ANTI-PATTERN 1: External Validation Function

**WRONG: Validation as separate function**
```python
class IncrementalRSI:
    def update(self, price: float) -> float:
        # Incremental calculation
        rsi = self._compute_rsi_incremental(price)
        return rsi  # NO VALIDATION HERE


# Validation in separate function (CAN BE SKIPPED!)
def validate_rsi(prices: List[float]) -> bool:
    """Validate RSI calculation (external)."""
    rsi_calc = IncrementalRSI()

    for price in prices:
        rsi_incremental = rsi_calc.update(price)

    # Calculate full RSI
    rsi_full = calculate_full_rsi(prices)

    # Compare
    return abs(rsi_incremental - rsi_full) < 1e-4
```

**Why this is WRONG:**
- Developer can forget to call `validate_rsi()`
- Validation only happens if test is explicitly run
- No enforcement during development
- Risk of deploying without validation

---

**✅ CORRECT: Built-in validation with flag**
```python
class IncrementalRSI:
    def __init__(self, period: int = 14, validate: bool = False):
        """
        Initialize RSI.

        Args:
            validate: If True, validate incremental == full (TESTING ONLY)
        """
        self.period = period
        self.validate = validate

        # State for incremental calculation
        self.avg_gain = None
        self.avg_loss = None

        # History for validation (only if validate=True)
        if self.validate:
            self.price_history = []

    def update(self, price: float) -> Optional[float]:
        """Update RSI with validation."""
        if self.validate:
            self.price_history.append(price)

        # Incremental calculation
        rsi_incremental = self._compute_incremental(price)

        # VALIDATE: Built-in assertion (when validate=True)
        if self.validate and rsi_incremental is not None:
            rsi_full = self._compute_full()
            assert abs(rsi_incremental - rsi_full) < 1e-4, \
                f"Incremental {rsi_incremental:.6f} != full {rsi_full:.6f}"

        return rsi_incremental

    def _compute_full(self) -> float:
        """Full calculation from history (validation only)."""
        # ... calculate from self.price_history
        return rsi_full


# Usage:
# TESTING: Validation enabled (catches errors immediately)
rsi_test = IncrementalRSI(period=14, validate=True)

# PRODUCTION: Validation disabled (zero overhead)
rsi_prod = IncrementalRSI(period=14, validate=False)
```

**Why this is CORRECT:**
- ✅ Validation runs automatically when `validate=True`
- ✅ Cannot be forgotten or skipped in testing
- ✅ Raises AssertionError immediately if mismatch
- ✅ Zero overhead in production (`validate=False`)
- ✅ Clear separation: testing (with validation) vs production (optimized)

---

### ❌ ANTI-PATTERN 2: External Benchmark Function

**WRONG: Latency measured externally**
```python
class FeaturePipeline:
    def process_tick(self, tick) -> Dict:
        # Process tick
        features = self._compute_features(tick)
        return features  # NO LATENCY CHECK


# Benchmark in separate file (NOT ENFORCED!)
def benchmark_pipeline():
    pipeline = FeaturePipeline()

    start = time.perf_counter()
    features = pipeline.process_tick(tick)
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"Latency: {latency_ms:.3f}ms")  # Just prints, doesn't enforce
```

**Why this is WRONG:**
- Pipeline can degrade without detection
- No enforcement in production
- Latency only measured if benchmark is run
- Performance regression discovered too late

---

**✅ CORRECT: Built-in latency assertions**
```python
class FeaturePipeline:
    def __init__(
        self,
        max_latency_ms: float = 10.0,
        enable_assertions: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            max_latency_ms: Maximum allowed latency
            enable_assertions: If True, assert latency < max (TESTING/PRODUCTION)
        """
        self.max_latency_ms = max_latency_ms
        self.enable_assertions = enable_assertions
        self.latency_history = []

    def process_tick(self, tick) -> Dict:
        """Process tick with built-in latency enforcement."""
        start = time.perf_counter()

        # Compute features
        features = self._compute_features(tick)

        # Measure latency
        latency_ms = (time.perf_counter() - start) * 1000
        self.latency_history.append(latency_ms)

        # ENFORCE: Latency requirement (when enabled)
        if self.enable_assertions:
            assert latency_ms < self.max_latency_ms, \
                f"Latency {latency_ms:.3f}ms exceeds limit {self.max_latency_ms}ms"

        return features


# Usage:
# TESTING/STAGING: Assertions enabled (catch degradation)
pipeline_test = FeaturePipeline(max_latency_ms=10.0, enable_assertions=True)

# PRODUCTION: Assertions disabled OR keep enabled for monitoring
pipeline_prod = FeaturePipeline(max_latency_ms=10.0, enable_assertions=True)
```

**Why this is CORRECT:**
- ✅ Latency checked on every call
- ✅ Performance degradation detected immediately
- ✅ Assertions can stay enabled in production (for monitoring)
- ✅ Latency history tracked automatically

---

### ❌ ANTI-PATTERN 3: Memory Profiling Only in Tests

**WRONG: Memory measured externally**
```python
class FeaturePipeline:
    def __init__(self):
        self.buffer_20 = CircularBuffer(size=20)
        self.buffer_50 = CircularBuffer(size=50)
        # No memory tracking


# Memory test in separate file
def test_memory_usage():
    pipeline = FeaturePipeline()

    # Process ticks
    for i in range(1000):
        pipeline.process_tick(price)

    # Measure memory (manually)
    memory_kb = sys.getsizeof(pipeline.buffer_20) / 1024
    print(f"Memory: {memory_kb:.2f}KB")
```

**Why this is WRONG:**
- Memory only measured if test is run
- No enforcement of bounds
- Memory leak discovered too late
- Manual calculation of memory usage

---

**✅ CORRECT: Built-in memory profiling**
```python
class FeaturePipeline:
    def __init__(self, max_memory_kb: float = 10.0):
        """
        Initialize pipeline.

        Args:
            max_memory_kb: Maximum allowed memory usage
        """
        self.max_memory_kb = max_memory_kb
        self.buffer_20 = CircularBuffer(size=20)
        self.buffer_50 = CircularBuffer(size=50)

    def measure_memory(self) -> Dict[str, float]:
        """Measure actual memory usage."""
        total_bytes = 0
        total_bytes += sys.getsizeof(self.buffer_20.buffer)
        total_bytes += sys.getsizeof(self.buffer_50.buffer)

        return {
            'total_kb': total_bytes / 1024.0,
            'buffer_20_kb': sys.getsizeof(self.buffer_20.buffer) / 1024.0,
            'buffer_50_kb': sys.getsizeof(self.buffer_50.buffer) / 1024.0
        }

    def assert_memory_bounds(self) -> None:
        """Assert memory usage is within bounds."""
        memory = self.measure_memory()
        total_kb = memory['total_kb']

        assert total_kb < self.max_memory_kb, \
            f"Memory {total_kb:.2f}KB exceeds limit {self.max_memory_kb}KB"


# Usage:
pipeline = FeaturePipeline(max_memory_kb=10.0)

# Process ticks
for tick in tick_stream:
    pipeline.process_tick(tick)

# Validate memory (periodically or at end)
pipeline.assert_memory_bounds()  # Fails if memory exceeded
```

**Why this is CORRECT:**
- ✅ Memory profiling built into class
- ✅ Easy to assert bounds at any time
- ✅ Memory breakdown shows which components use most memory
- ✅ Can be called periodically in production

---

## Summary: Always Use Built-In Validation

**Key Principle**: Validation, assertions, and profiling should be **built into the class**, not external functions.

| Aspect | ❌ External (Wrong) | ✅ Built-In (Correct) |
|--------|---------------------|----------------------|
| Validation | Separate function `validate_rsi()` | `validate` flag in `__init__`, assert in `update()` |
| Latency | Benchmark script | `enable_assertions` flag, assert in `process_tick()` |
| Memory | Test function | `measure_memory()` method, `assert_memory_bounds()` |
| Can Skip? | ❌ Yes (forget to call) | ✅ No (runs automatically when enabled) |
| Overhead | N/A | Zero (disabled in production) |

**Testing vs Production**:
```python
# TESTING: All validations enabled
rsi = IncrementalRSI(period=14, validate=True)
pipeline = FeaturePipeline(
    max_latency_ms=10.0,
    enable_assertions=True
)

# PRODUCTION: Validations disabled (or keep assertions for monitoring)
rsi = IncrementalRSI(period=14, validate=False)
pipeline = FeaturePipeline(
    max_latency_ms=10.0,
    enable_assertions=False  # Or True for monitoring
)
```

---

## Rationalization Table

| Excuse | Reality | Fix |
|--------|---------|-----|
| "Performance target met" | Numerical errors accumulate | Validate incremental == full |
| "Circular buffer chosen" | Others don't know tradeoffs | Document decision framework |
| "Benchmarked externally" | Can degrade without detection | Build latency assertions in |
| "Memory efficient in theory" | Should measure actual usage | Profile memory systematically |
| "Only optimize when needed" | Real-time ALWAYS needs it | Apply patterns proactively |
| "Cache improves performance" | Without metrics, don't know | Track and validate hit rate |

---

## Real-World Impact

**Proper validation prevents:**
- Incremental RSI drifting from true value (numerical errors)
- Pipeline degrading from 1ms to 50ms without detection
- Memory growing from 2KB to 2GB (unbounded buffers)
- Cache adding latency instead of reducing it (low hit rate)
- Choosing wrong data structure (list when circular buffer needed)

**Time investment:**
- Add incremental validation: 5 minutes
- Build latency assertions: 3 minutes
- Add memory profiling: 5 minutes
- Implement TTL cache: 10 minutes
- Debug numerical drift without validation: 4+ hours

---

## Data Structure Selection Guide

| Use Case | Structure | Why |
|----------|-----------|-----|
| Rolling mean (frequent) | Circular Buffer | O(1) mean via running sum |
| Rolling std (infrequent) | Circular Buffer | O(1) append, O(n) std acceptable |
| FIFO queue (no stats) | collections.deque | O(1) append/popleft, simpler |
| Temporary storage | List | Simple, but NOT for rolling |
| Caching with TTL | TTLCache | Automatic expiry, hit tracking |
| LRU cache | functools.lru_cache | Built-in, simple |

---

## Bottom Line

**Real-time feature engineering requires three components:**
1. Optimization (circular buffers, incremental updates) ← AGENTS KNOW
2. Validation (incremental == full, latency < threshold) ← AGENTS SKIP
3. Enforcement (assertions in production, not just tests) ← AGENTS SKIP

Don't just optimize—validate and enforce.

Every incremental update gets validation test. Every pipeline gets latency assertions. Every system gets memory profiling. No exceptions.
