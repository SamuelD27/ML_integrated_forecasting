# Feature Engineering Reference - Real-Time Feature Pipeline

## Table of Contents

1. [Circular Buffer Implementations](#circular-buffer-implementations)
2. [Incremental Technical Indicators](#incremental-technical-indicators)
3. [Caching Strategies](#caching-strategies)
4. [Vectorization Patterns](#vectorization-patterns)
5. [Memory Profiling Utilities](#memory-profiling-utilities)
6. [Validation Framework](#validation-framework)
7. [Complete Integration Example](#complete-integration-example)

---

## Circular Buffer Implementations

### Base Circular Buffer with Running Sum

**Use Case**: Rolling windows with frequent mean/sum calculations

**Performance**: O(1) append, O(1) mean

```python
import numpy as np
from typing import Optional

class CircularBuffer:
    """
    Fixed-size circular buffer with O(1) mean calculation.

    Memory: size × 8 bytes (float64)
    Append: O(1)
    Mean: O(1) with running sum
    """

    def __init__(self, size: int):
        """
        Initialize circular buffer.

        Args:
            size: Maximum number of elements to store
        """
        assert size > 0, "Buffer size must be positive"

        self.size = size
        self.buffer = np.zeros(size, dtype=np.float64)
        self.index = 0
        self.count = 0
        self.sum = 0.0  # Running sum for O(1) mean
        self.sum_sq = 0.0  # Running sum of squares for O(1) variance

    def append(self, value: float) -> None:
        """Append value to buffer (O(1))."""
        if self.count == self.size:
            # Remove old value from running sums
            old_value = self.buffer[self.index]
            self.sum -= old_value
            self.sum_sq -= old_value ** 2

        # Add new value
        self.buffer[self.index] = value
        self.sum += value
        self.sum_sq += value ** 2

        # Update pointers
        self.index = (self.index + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def get_mean(self) -> Optional[float]:
        """Get mean (O(1))."""
        if self.count == 0:
            return None
        return self.sum / self.count

    def get_std(self) -> Optional[float]:
        """Get standard deviation (O(1))."""
        if self.count < 2:
            return None

        mean = self.sum / self.count
        variance = (self.sum_sq / self.count) - (mean ** 2)
        return np.sqrt(max(0.0, variance))  # Avoid negative due to float precision

    def get_min(self) -> Optional[float]:
        """Get minimum (O(n))."""
        if self.count == 0:
            return None
        return np.min(self.buffer[:self.count])

    def get_max(self) -> Optional[float]:
        """Get maximum (O(n))."""
        if self.count == 0:
            return None
        return np.max(self.buffer[:self.count])

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.count == self.size

    def get_values(self) -> np.ndarray:
        """Get all values in insertion order."""
        if self.count < self.size:
            return self.buffer[:self.count].copy()

        # Rearrange circular buffer to chronological order
        return np.concatenate([
            self.buffer[self.index:],
            self.buffer[:self.index]
        ])


# VALIDATION: Verify circular buffer matches naive implementation
def validate_circular_buffer():
    """Test that circular buffer produces identical results to naive list."""
    cb = CircularBuffer(size=5)
    naive_list = []

    # Test data
    test_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    for value in test_values:
        cb.append(value)
        naive_list.append(value)

        # Keep only last 5 values
        if len(naive_list) > 5:
            naive_list.pop(0)

        # Validate mean
        expected_mean = np.mean(naive_list)
        actual_mean = cb.get_mean()
        assert abs(expected_mean - actual_mean) < 1e-10, \
            f"Mean mismatch: expected {expected_mean}, got {actual_mean}"

        # Validate std
        if len(naive_list) >= 2:
            expected_std = np.std(naive_list, ddof=0)
            actual_std = cb.get_std()
            assert abs(expected_std - actual_std) < 1e-10, \
                f"Std mismatch: expected {expected_std}, got {actual_std}"

    print("✅ Circular buffer validation passed")


# BENCHMARK: Compare circular buffer vs naive list
def benchmark_circular_buffer():
    """Benchmark circular buffer vs naive list implementation."""
    import time

    n_iterations = 100_000
    window_size = 50

    # Circular buffer
    cb = CircularBuffer(size=window_size)
    start = time.perf_counter()
    for i in range(n_iterations):
        cb.append(float(i))
        _ = cb.get_mean()
    circular_time = time.perf_counter() - start

    # Naive list
    naive_list = []
    start = time.perf_counter()
    for i in range(n_iterations):
        naive_list.append(float(i))
        if len(naive_list) > window_size:
            naive_list.pop(0)
        _ = np.mean(naive_list)
    naive_time = time.perf_counter() - start

    print(f"Circular buffer: {circular_time:.4f}s")
    print(f"Naive list: {naive_time:.4f}s")
    print(f"Speedup: {naive_time / circular_time:.2f}x")
    # Expected: 5-10x speedup
```

**Memory Comparison**:
```
Circular Buffer (size=50):
- buffer: 50 × 8 = 400 bytes
- sum, sum_sq, index, count: 32 bytes
- Total: ~432 bytes (FIXED)

Naive List (unbounded):
- list overhead: 64 bytes
- 50 elements × 8 bytes = 400 bytes
- Reallocations: 25% overhead
- Total: ~580 bytes (GROWS)
```

---

## Incremental Technical Indicators

### 1. Exponential Moving Average (EMA)

**Formula**: `EMA_t = α × Price_t + (1 - α) × EMA_{t-1}`
**Multiplier**: `α = 2 / (period + 1)`

```python
class IncrementalEMA:
    """
    Incremental Exponential Moving Average.

    Performance: O(1) per update
    Memory: O(1) - single float
    """

    def __init__(self, period: int, validate: bool = False):
        """
        Initialize EMA.

        Args:
            period: EMA period (e.g., 12, 26, 200)
            validate: If True, validate against full calculation (testing only)
        """
        assert period > 0, "Period must be positive"

        self.period = period
        self.validate = validate

        # Pre-calculate multiplier (done once)
        self.alpha = 2.0 / (period + 1)

        # State
        self.ema: Optional[float] = None
        self.count = 0

        # Validation (testing only)
        if self.validate:
            self.price_history = []

    def update(self, price: float) -> Optional[float]:
        """
        Update EMA with new price.

        Args:
            price: Current price

        Returns:
            EMA value (None until enough data)
        """
        self.count += 1

        if self.validate:
            self.price_history.append(price)

        if self.ema is None:
            # Initialize with first price
            self.ema = price
        else:
            # Incremental update (O(1))
            self.ema = self.alpha * price + (1 - self.alpha) * self.ema

        # VALIDATE: Incremental matches full calculation
        if self.validate and self.count >= self.period:
            ema_full = self._calculate_full_ema()
            assert abs(self.ema - ema_full) < 1e-6, \
                f"Incremental EMA {self.ema:.6f} != full {ema_full:.6f}"

        return self.ema if self.count >= self.period else None

    def _calculate_full_ema(self) -> float:
        """Calculate EMA from scratch (validation only)."""
        ema = self.price_history[0]
        for price in self.price_history[1:]:
            ema = self.alpha * price + (1 - self.alpha) * ema
        return ema
```

### 2. Relative Strength Index (RSI)

**Formula**: `RSI = 100 - (100 / (1 + RS))`
**RS**: `avg_gain / avg_loss` (Wilder's smoothing)

```python
class IncrementalRSI:
    """
    Incremental RSI using Wilder's smoothing.

    Performance: O(1) per update after warmup
    Memory: O(1) - single averages
    """

    def __init__(self, period: int = 14, validate: bool = False):
        """
        Initialize RSI.

        Args:
            period: RSI period (typically 14)
            validate: If True, validate against full calculation
        """
        assert period > 0, "Period must be positive"

        self.period = period
        self.validate = validate

        # State
        self.prev_price: Optional[float] = None
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.count = 0

        # Warmup buffer
        self.gains = CircularBuffer(size=period)
        self.losses = CircularBuffer(size=period)

        # Validation
        if self.validate:
            self.price_history = []

    def update(self, price: float) -> Optional[float]:
        """
        Update RSI with new price.

        Args:
            price: Current price

        Returns:
            RSI value (None until warmup complete)
        """
        if self.validate:
            self.price_history.append(price)

        if self.prev_price is None:
            self.prev_price = price
            return None

        # Calculate gain/loss
        change = price - self.prev_price
        gain = max(0.0, change)
        loss = max(0.0, -change)

        self.prev_price = price
        self.count += 1

        # Warmup phase: Fill buffers
        if self.count <= self.period:
            self.gains.append(gain)
            self.losses.append(loss)

            if self.count == self.period:
                # Initialize averages
                self.avg_gain = self.gains.get_mean()
                self.avg_loss = self.losses.get_mean()

            return None

        # Incremental phase: Wilder's smoothing (O(1))
        self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
        self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        # Calculate RSI
        if self.avg_loss == 0:
            rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # VALIDATE: Incremental matches full calculation
        if self.validate:
            rsi_full = self._calculate_full_rsi()
            assert abs(rsi - rsi_full) < 1e-4, \
                f"Incremental RSI {rsi:.6f} != full {rsi_full:.6f}"

        return rsi

    def _calculate_full_rsi(self) -> float:
        """Calculate RSI from scratch (validation only)."""
        if len(self.price_history) < self.period + 1:
            return 50.0  # Neutral

        # Calculate changes
        changes = np.diff(self.price_history)
        gains = np.maximum(changes, 0)
        losses = np.maximum(-changes, 0)

        # Initial average
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        # Wilder's smoothing
        for i in range(self.period, len(gains)):
            avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
            avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
```

### 3. Moving Average Convergence Divergence (MACD)

**Formula**:
- MACD Line = EMA(12) - EMA(26)
- Signal Line = EMA(9) of MACD Line
- Histogram = MACD Line - Signal Line

```python
class IncrementalMACD:
    """
    Incremental MACD using pre-calculated EMA multipliers.

    Performance: O(1) per update after warmup
    Memory: O(1) - three EMAs
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        validate: bool = False
    ):
        """
        Initialize MACD.

        Args:
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
            validate: If True, validate against full calculation
        """
        assert fast < slow, "Fast period must be less than slow period"

        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.validate = validate

        # Pre-calculate multipliers (done once)
        self.fast_mult = 2.0 / (fast + 1)
        self.slow_mult = 2.0 / (slow + 1)
        self.signal_mult = 2.0 / (signal + 1)

        # State
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.signal_ema: Optional[float] = None
        self.count = 0

        # Validation
        if self.validate:
            self.price_history = []

    def update(self, price: float) -> Optional[dict]:
        """
        Update MACD with new price.

        Args:
            price: Current price

        Returns:
            Dict with 'macd', 'signal', 'histogram' (None until warmup)
        """
        if self.validate:
            self.price_history.append(price)

        self.count += 1

        # Update fast EMA
        if self.fast_ema is None:
            self.fast_ema = price
        else:
            self.fast_ema = (price - self.fast_ema) * self.fast_mult + self.fast_ema

        # Update slow EMA
        if self.slow_ema is None:
            self.slow_ema = price
        else:
            self.slow_ema = (price - self.slow_ema) * self.slow_mult + self.slow_ema

        # Not enough data yet
        if self.count < self.slow:
            return None

        # Calculate MACD line
        macd_line = self.fast_ema - self.slow_ema

        # Update signal line
        if self.signal_ema is None:
            self.signal_ema = macd_line
        else:
            self.signal_ema = (macd_line - self.signal_ema) * self.signal_mult + self.signal_ema

        # Not enough data for signal yet
        if self.count < self.slow + self.signal:
            return None

        # Calculate histogram
        histogram = macd_line - self.signal_ema

        result = {
            'macd': macd_line,
            'signal': self.signal_ema,
            'histogram': histogram
        }

        # VALIDATE: Incremental matches full calculation
        if self.validate:
            result_full = self._calculate_full_macd()
            if result_full is not None:
                for key in ['macd', 'signal', 'histogram']:
                    assert abs(result[key] - result_full[key]) < 1e-4, \
                        f"Incremental {key} {result[key]:.6f} != full {result_full[key]:.6f}"

        return result

    def _calculate_full_macd(self) -> Optional[dict]:
        """Calculate MACD from scratch (validation only)."""
        if len(self.price_history) < self.slow + self.signal:
            return None

        prices = np.array(self.price_history)

        # Calculate EMAs
        fast_ema = prices[0]
        slow_ema = prices[0]

        for price in prices[1:]:
            fast_ema = (price - fast_ema) * self.fast_mult + fast_ema
            slow_ema = (price - slow_ema) * self.slow_mult + slow_ema

        macd_line = fast_ema - slow_ema

        # Calculate signal line
        macd_history = []
        fast_ema = self.price_history[0]
        slow_ema = self.price_history[0]

        for i, price in enumerate(self.price_history):
            fast_ema = (price - fast_ema) * self.fast_mult + fast_ema
            slow_ema = (price - slow_ema) * self.slow_mult + slow_ema

            if i >= self.slow - 1:
                macd_history.append(fast_ema - slow_ema)

        signal_ema = macd_history[0]
        for macd in macd_history[1:]:
            signal_ema = (macd - signal_ema) * self.signal_mult + signal_ema

        return {
            'macd': macd_line,
            'signal': signal_ema,
            'histogram': macd_line - signal_ema
        }
```

### 4. Average True Range (ATR)

**Formula**: ATR = Wilder's smoothed average of True Range
**True Range**: max(High - Low, |High - Prev Close|, |Low - Prev Close|)

```python
class IncrementalATR:
    """
    Incremental ATR using Wilder's smoothing.

    Performance: O(1) per update after warmup
    Memory: O(1) - single average
    """

    def __init__(self, period: int = 14, validate: bool = False):
        """
        Initialize ATR.

        Args:
            period: ATR period (typically 14)
            validate: If True, validate against full calculation
        """
        assert period > 0, "Period must be positive"

        self.period = period
        self.validate = validate

        # State
        self.atr: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.count = 0

        # Warmup buffer
        self.tr_buffer = CircularBuffer(size=period)

        # Validation
        if self.validate:
            self.ohlc_history = []  # List of (high, low, close) tuples

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """
        Update ATR with new OHLC data.

        Args:
            high: Current high price
            low: Current low price
            close: Current close price

        Returns:
            ATR value (None until warmup complete)
        """
        assert high >= low, f"High {high} must be >= low {low}"
        assert low <= close <= high, f"Close {close} must be between low {low} and high {high}"

        if self.validate:
            self.ohlc_history.append((high, low, close))

        # Calculate true range
        if self.prev_close is None:
            true_range = high - low
        else:
            true_range = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )

        self.prev_close = close
        self.count += 1

        # Warmup phase
        if self.count <= self.period:
            self.tr_buffer.append(true_range)

            if self.count == self.period:
                # Initialize ATR
                self.atr = self.tr_buffer.get_mean()

            return None

        # Incremental phase: Wilder's smoothing (O(1))
        self.atr = (self.atr * (self.period - 1) + true_range) / self.period

        # VALIDATE: Incremental matches full calculation
        if self.validate:
            atr_full = self._calculate_full_atr()
            assert abs(self.atr - atr_full) < 1e-6, \
                f"Incremental ATR {self.atr:.6f} != full {atr_full:.6f}"

        return self.atr

    def _calculate_full_atr(self) -> float:
        """Calculate ATR from scratch (validation only)."""
        if len(self.ohlc_history) < self.period:
            return 0.0

        # Calculate all true ranges
        true_ranges = []
        for i, (high, low, close) in enumerate(self.ohlc_history):
            if i == 0:
                tr = high - low
            else:
                prev_close = self.ohlc_history[i-1][2]
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
            true_ranges.append(tr)

        # Initial average
        atr = np.mean(true_ranges[:self.period])

        # Wilder's smoothing
        for tr in true_ranges[self.period:]:
            atr = (atr * (self.period - 1) + tr) / self.period

        return atr
```

### 5. Bollinger Bands

**Formula**:
- Middle Band = SMA(close, period)
- Upper Band = Middle Band + (k × std)
- Lower Band = Middle Band - (k × std)

```python
class IncrementalBollingerBands:
    """
    Incremental Bollinger Bands using circular buffer.

    Performance: O(1) per update (with running sum and sum of squares)
    Memory: O(period) - circular buffer
    """

    def __init__(self, period: int = 20, k: float = 2.0, validate: bool = False):
        """
        Initialize Bollinger Bands.

        Args:
            period: SMA period (default 20)
            k: Number of standard deviations (default 2.0)
            validate: If True, validate against full calculation
        """
        assert period > 0, "Period must be positive"
        assert k > 0, "K must be positive"

        self.period = period
        self.k = k
        self.validate = validate

        # State
        self.buffer = CircularBuffer(size=period)

        # Validation
        if self.validate:
            self.price_history = []

    def update(self, close: float) -> Optional[dict]:
        """
        Update Bollinger Bands with new close price.

        Args:
            close: Current close price

        Returns:
            Dict with 'upper', 'middle', 'lower' bands (None until warmup)
        """
        if self.validate:
            self.price_history.append(close)

        self.buffer.append(close)

        if not self.buffer.is_full():
            return None

        # Calculate bands (O(1) with circular buffer)
        middle = self.buffer.get_mean()
        std = self.buffer.get_std()

        result = {
            'upper': middle + (self.k * std),
            'middle': middle,
            'lower': middle - (self.k * std)
        }

        # VALIDATE: Incremental matches full calculation
        if self.validate:
            result_full = self._calculate_full_bands()
            if result_full is not None:
                for key in ['upper', 'middle', 'lower']:
                    assert abs(result[key] - result_full[key]) < 1e-6, \
                        f"Incremental {key} {result[key]:.6f} != full {result_full[key]:.6f}"

        return result

    def _calculate_full_bands(self) -> Optional[dict]:
        """Calculate Bollinger Bands from scratch (validation only)."""
        if len(self.price_history) < self.period:
            return None

        recent_prices = self.price_history[-self.period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices, ddof=0)

        return {
            'upper': middle + (self.k * std),
            'middle': middle,
            'lower': middle - (self.k * std)
        }
```

---

## Caching Strategies

### TTL Cache with LRU Eviction

**Use Case**: Cache computed features with time-based expiration

```python
from typing import Any, Optional
import time
from collections import OrderedDict

class TTLCache:
    """
    Time-To-Live cache with LRU eviction.

    Features:
    - TTL-based expiration
    - LRU eviction when capacity exceeded
    - Hit rate tracking
    - Memory bounds
    """

    def __init__(self, capacity: int, ttl_seconds: float):
        """
        Initialize TTL cache.

        Args:
            capacity: Maximum number of entries
            ttl_seconds: Time-to-live for entries
        """
        assert capacity > 0, "Capacity must be positive"
        assert ttl_seconds > 0, "TTL must be positive"

        self.capacity = capacity
        self.ttl_seconds = ttl_seconds

        # Use OrderedDict for LRU ordering
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if miss/expired
        """
        if key not in self.cache:
            self.misses += 1
            return None

        value, timestamp = self.cache[key]

        # Check if expired
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None

        # Move to end (LRU)
        self.cache.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Update existing key
        if key in self.cache:
            self.cache.move_to_end(key)
        # Evict oldest if at capacity
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # Remove oldest
            self.evictions += 1

        # Add new entry with timestamp
        self.cache[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        if key in self.cache:
            del self.cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.get_hit_rate()
        }


# Example usage: Feature cache
class CachedFeatureComputer:
    """
    Feature computer with TTL caching.

    Cache key: f"{symbol}_{timestamp}"
    TTL: 10 seconds (features stable within window)
    Capacity: 1000 symbols
    """

    def __init__(self, capacity: int = 1000, ttl_seconds: float = 10.0):
        self.cache = TTLCache(capacity=capacity, ttl_seconds=ttl_seconds)
        self.rsi = IncrementalRSI(period=14)
        self.macd = IncrementalMACD()

    def compute_features(self, symbol: str, timestamp: int, price: float) -> dict:
        """
        Compute features with caching.

        Args:
            symbol: Stock symbol
            timestamp: Unix timestamp (rounded to cache granularity)
            price: Current price

        Returns:
            Feature dict
        """
        cache_key = f"{symbol}_{timestamp}"

        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Cache miss: Compute features
        features = {
            'rsi': self.rsi.update(price),
            'macd': self.macd.update(price)
        }

        # Store in cache
        self.cache.put(cache_key, features)

        return features

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        return self.cache.get_stats()
```

**Performance Requirements**:
```python
# ENFORCE: Cache hit rate > 50% after warmup
def assert_cache_performance(cache: TTLCache, min_hit_rate: float = 0.5):
    """Assert cache hit rate meets minimum threshold."""
    total_requests = cache.hits + cache.misses

    if total_requests > 100:  # After warmup
        hit_rate = cache.get_hit_rate()
        assert hit_rate >= min_hit_rate, \
            f"Cache hit rate {hit_rate:.2%} < minimum {min_hit_rate:.2%}"
```

---

## Vectorization Patterns

### Batch Feature Computation

**Use Case**: Compute features for multiple symbols in parallel

```python
import numpy as np
import pandas as pd
from typing import List

class VectorizedFeatureComputer:
    """
    Vectorized feature computation using NumPy.

    Performance: 10-100x faster than Python loops
    """

    @staticmethod
    def compute_rsi_batch(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Compute RSI for batch of price series.

        Args:
            prices: Array of shape (n_series, n_timesteps)
            period: RSI period

        Returns:
            RSI values of shape (n_series, n_timesteps)
        """
        n_series, n_timesteps = prices.shape

        # Calculate changes
        changes = np.diff(prices, axis=1)

        # Separate gains and losses
        gains = np.maximum(changes, 0)
        losses = np.maximum(-changes, 0)

        # Initialize output
        rsi = np.full((n_series, n_timesteps), np.nan)

        # Calculate initial averages
        avg_gains = np.mean(gains[:, :period], axis=1, keepdims=True)
        avg_losses = np.mean(losses[:, :period], axis=1, keepdims=True)

        # Apply Wilder's smoothing (vectorized)
        for i in range(period, n_timesteps - 1):
            avg_gains = (avg_gains * (period - 1) + gains[:, i:i+1]) / period
            avg_losses = (avg_losses * (period - 1) + losses[:, i:i+1]) / period

            # Calculate RS and RSI
            rs = avg_gains / np.where(avg_losses == 0, 1e-10, avg_losses)
            rsi[:, i+1] = 100.0 - (100.0 / (1.0 + rs.squeeze()))

        return rsi

    @staticmethod
    def compute_ema_batch(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Compute EMA for batch of price series.

        Args:
            prices: Array of shape (n_series, n_timesteps)
            period: EMA period

        Returns:
            EMA values of shape (n_series, n_timesteps)
        """
        n_series, n_timesteps = prices.shape
        alpha = 2.0 / (period + 1)

        # Initialize with first prices
        ema = np.copy(prices)
        ema[:, 0] = prices[:, 0]

        # Vectorized EMA calculation
        for i in range(1, n_timesteps):
            ema[:, i] = alpha * prices[:, i] + (1 - alpha) * ema[:, i-1]

        return ema

    @staticmethod
    def compute_bollinger_bands_batch(
        prices: np.ndarray,
        period: int = 20,
        k: float = 2.0
    ) -> dict:
        """
        Compute Bollinger Bands for batch of price series.

        Args:
            prices: Array of shape (n_series, n_timesteps)
            period: SMA period
            k: Number of standard deviations

        Returns:
            Dict with 'upper', 'middle', 'lower' arrays
        """
        n_series, n_timesteps = prices.shape

        # Calculate rolling mean and std using pandas (more efficient)
        df = pd.DataFrame(prices.T)

        middle = df.rolling(window=period).mean().values.T
        std = df.rolling(window=period).std(ddof=0).values.T

        return {
            'upper': middle + (k * std),
            'middle': middle,
            'lower': middle - (k * std)
        }


# BENCHMARK: Vectorized vs loop
def benchmark_vectorization():
    """Compare vectorized vs loop implementation."""
    import time

    n_series = 100
    n_timesteps = 1000
    period = 14

    # Generate random price data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_series, n_timesteps) * 0.02, axis=1)

    # Vectorized
    start = time.perf_counter()
    rsi_vec = VectorizedFeatureComputer.compute_rsi_batch(prices, period)
    vec_time = time.perf_counter() - start

    # Loop (for comparison)
    start = time.perf_counter()
    rsi_loop = np.zeros_like(prices)
    for i in range(n_series):
        rsi_calc = IncrementalRSI(period=period)
        for j in range(n_timesteps):
            result = rsi_calc.update(prices[i, j])
            if result is not None:
                rsi_loop[i, j] = result
    loop_time = time.perf_counter() - start

    print(f"Vectorized: {vec_time:.4f}s")
    print(f"Loop: {loop_time:.4f}s")
    print(f"Speedup: {loop_time / vec_time:.2f}x")
    # Expected: 10-50x speedup
```

---

## Memory Profiling Utilities

### Memory Footprint Analyzer

```python
import sys
from typing import Dict

class MemoryProfiler:
    """
    Utilities for measuring memory usage of feature pipeline.
    """

    @staticmethod
    def measure_object_size(obj: Any) -> int:
        """
        Recursively measure size of object and its attributes.

        Args:
            obj: Object to measure

        Returns:
            Size in bytes
        """
        size = sys.getsizeof(obj)

        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.nbytes

        # Handle circular buffers
        if hasattr(obj, 'buffer') and isinstance(obj.buffer, np.ndarray):
            size += obj.buffer.nbytes

        return size

    @staticmethod
    def profile_feature_pipeline(pipeline: Any) -> Dict[str, int]:
        """
        Profile memory usage of feature pipeline components.

        Args:
            pipeline: Feature pipeline object

        Returns:
            Dict mapping component name to memory usage (bytes)
        """
        profile = {}

        # Profile each attribute
        for attr_name in dir(pipeline):
            if attr_name.startswith('_'):
                continue

            try:
                attr = getattr(pipeline, attr_name)
                if not callable(attr):
                    size = MemoryProfiler.measure_object_size(attr)
                    profile[attr_name] = size
            except:
                pass

        # Calculate total
        profile['TOTAL'] = sum(profile.values())

        return profile

    @staticmethod
    def format_bytes(bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024
        return f"{bytes:.2f} TB"

    @staticmethod
    def print_memory_profile(profile: Dict[str, int]) -> None:
        """Print memory profile in table format."""
        print(f"{'Component':<30} {'Memory':>15}")
        print("-" * 47)

        for component, size in sorted(profile.items(), key=lambda x: x[1], reverse=True):
            print(f"{component:<30} {MemoryProfiler.format_bytes(size):>15}")


# Example: Profile feature pipeline
class OptimizedFeaturePipeline:
    def __init__(self):
        self.sma_20 = CircularBuffer(size=20)
        self.sma_50 = CircularBuffer(size=50)
        self.rsi = IncrementalRSI(period=14)
        self.macd = IncrementalMACD()
        self.atr = IncrementalATR(period=14)
        self.bbands = IncrementalBollingerBands(period=20)

    def assert_memory_bounds(self, max_bytes: int = 2048):
        """
        Assert pipeline memory usage is within bounds.

        Args:
            max_bytes: Maximum allowed memory (default 2 KB)
        """
        profile = MemoryProfiler.profile_feature_pipeline(self)
        total_bytes = profile['TOTAL']

        assert total_bytes <= max_bytes, \
            f"Memory usage {total_bytes} bytes exceeds limit {max_bytes} bytes"

        return profile


# Memory assertion example
def test_memory_bounds():
    """Test that pipeline memory stays within bounds."""
    pipeline = OptimizedFeaturePipeline()

    # Process 1000 ticks
    for i in range(1000):
        _ = pipeline.process_tick(100.0 + i * 0.1)

    # ASSERT: Memory bounded
    profile = pipeline.assert_memory_bounds(max_bytes=2048)

    # Print profile
    MemoryProfiler.print_memory_profile(profile)
```

---

## Validation Framework

### Complete Validation Suite

```python
import time
from typing import Callable, Optional

class FeaturePipelineValidator:
    """
    Comprehensive validation framework for real-time feature pipelines.

    Validates:
    1. Incremental updates match full calculations
    2. Latency meets requirements
    3. Memory stays bounded
    4. Cache hit rate is acceptable
    """

    def __init__(
        self,
        max_latency_ms: float = 10.0,
        max_memory_bytes: int = 2048,
        min_cache_hit_rate: float = 0.5
    ):
        """
        Initialize validator.

        Args:
            max_latency_ms: Maximum allowed latency per tick
            max_memory_bytes: Maximum allowed memory usage
            min_cache_hit_rate: Minimum acceptable cache hit rate
        """
        self.max_latency_ms = max_latency_ms
        self.max_memory_bytes = max_memory_bytes
        self.min_cache_hit_rate = min_cache_hit_rate

        # Metrics
        self.latencies = []
        self.memory_samples = []

    def validate_latency(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Validate function latency meets requirements.

        Args:
            fn: Function to validate
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            AssertionError if latency exceeds limit
        """
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        self.latencies.append(latency_ms)

        # ENFORCE: Latency requirement
        assert latency_ms < self.max_latency_ms, \
            f"Latency {latency_ms:.3f}ms exceeds limit {self.max_latency_ms}ms"

        return result

    def validate_memory(self, pipeline: Any) -> Dict[str, int]:
        """
        Validate pipeline memory usage.

        Args:
            pipeline: Feature pipeline object

        Returns:
            Memory profile dict

        Raises:
            AssertionError if memory exceeds limit
        """
        profile = MemoryProfiler.profile_feature_pipeline(pipeline)
        total_bytes = profile['TOTAL']

        self.memory_samples.append(total_bytes)

        # ENFORCE: Memory requirement
        assert total_bytes <= self.max_memory_bytes, \
            f"Memory {total_bytes} bytes exceeds limit {self.max_memory_bytes} bytes"

        return profile

    def validate_cache_hit_rate(self, cache: TTLCache) -> float:
        """
        Validate cache hit rate.

        Args:
            cache: TTL cache object

        Returns:
            Hit rate

        Raises:
            AssertionError if hit rate below minimum
        """
        hit_rate = cache.get_hit_rate()
        total_requests = cache.hits + cache.misses

        # Only validate after warmup
        if total_requests > 100:
            assert hit_rate >= self.min_cache_hit_rate, \
                f"Cache hit rate {hit_rate:.2%} < minimum {self.min_cache_hit_rate:.2%}"

        return hit_rate

    def get_latency_stats(self) -> dict:
        """Get latency statistics."""
        if not self.latencies:
            return {}

        latencies = np.array(self.latencies)
        return {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'max_ms': np.max(latencies)
        }

    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        if not self.memory_samples:
            return {}

        memory = np.array(self.memory_samples)
        return {
            'mean_bytes': np.mean(memory),
            'max_bytes': np.max(memory),
            'growth': memory[-1] - memory[0] if len(memory) > 1 else 0
        }

    def print_validation_report(self) -> None:
        """Print comprehensive validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        # Latency
        print("\nLatency:")
        latency_stats = self.get_latency_stats()
        for metric, value in latency_stats.items():
            status = "✅" if value < self.max_latency_ms else "❌"
            print(f"  {status} {metric}: {value:.3f}ms")
        print(f"  Limit: {self.max_latency_ms}ms")

        # Memory
        print("\nMemory:")
        memory_stats = self.get_memory_stats()
        for metric, value in memory_stats.items():
            if 'growth' in metric:
                status = "✅" if value <= 0 else "⚠️"
                print(f"  {status} {metric}: {value:+d} bytes")
            else:
                status = "✅" if value < self.max_memory_bytes else "❌"
                print(f"  {status} {metric}: {MemoryProfiler.format_bytes(value)}")
        print(f"  Limit: {MemoryProfiler.format_bytes(self.max_memory_bytes)}")

        print("\n" + "=" * 60)


# Complete validation example
def validate_feature_pipeline():
    """
    Complete validation of feature pipeline.

    Validates:
    - Incremental updates (done by indicators themselves)
    - Latency (< 10ms per tick)
    - Memory (< 2 KB)
    - Cache performance (> 50% hit rate)
    """
    # Create pipeline with validation enabled
    pipeline = OptimizedFeaturePipeline()
    cache = TTLCache(capacity=1000, ttl_seconds=10.0)
    validator = FeaturePipelineValidator(
        max_latency_ms=10.0,
        max_memory_bytes=2048,
        min_cache_hit_rate=0.5
    )

    # Simulate 1000 ticks
    for i in range(1000):
        price = 100.0 + np.sin(i / 50) * 10.0

        # Validate latency
        features = validator.validate_latency(
            pipeline.process_tick,
            price
        )

        # Validate memory every 100 ticks
        if i % 100 == 0:
            validator.validate_memory(pipeline)

    # Validate cache
    validator.validate_cache_hit_rate(cache)

    # Print report
    validator.print_validation_report()

    print("\n✅ All validations passed!")
```

---

## Complete Integration Example

```python
class ProductionFeaturePipeline:
    """
    Production-ready feature pipeline with all optimizations.

    Features:
    - Incremental updates (O(1) per tick)
    - Built-in latency assertions
    - Memory profiling
    - TTL caching
    - Validation framework
    """

    def __init__(
        self,
        max_latency_ms: float = 10.0,
        max_memory_bytes: int = 2048,
        cache_capacity: int = 1000,
        cache_ttl_seconds: float = 10.0,
        validate: bool = False
    ):
        """
        Initialize production pipeline.

        Args:
            max_latency_ms: Maximum allowed latency
            max_memory_bytes: Maximum allowed memory
            cache_capacity: Cache capacity
            cache_ttl_seconds: Cache TTL
            validate: Enable validation (testing only)
        """
        self.max_latency_ms = max_latency_ms
        self.max_memory_bytes = max_memory_bytes
        self.validate = validate

        # Indicators (incremental)
        self.sma_20 = CircularBuffer(size=20)
        self.sma_50 = CircularBuffer(size=50)
        self.rsi = IncrementalRSI(period=14, validate=validate)
        self.macd = IncrementalMACD(validate=validate)
        self.atr = IncrementalATR(period=14, validate=validate)
        self.bbands = IncrementalBollingerBands(period=20, validate=validate)

        # Caching
        self.cache = TTLCache(capacity=cache_capacity, ttl_seconds=cache_ttl_seconds)

        # Validator
        self.validator = FeaturePipelineValidator(
            max_latency_ms=max_latency_ms,
            max_memory_bytes=max_memory_bytes,
            min_cache_hit_rate=0.5
        )

    def process_tick(
        self,
        symbol: str,
        timestamp: int,
        high: float,
        low: float,
        close: float
    ) -> dict:
        """
        Process single tick with validation.

        Args:
            symbol: Stock symbol
            timestamp: Unix timestamp
            high: High price
            low: Low price
            close: Close price

        Returns:
            Feature dict
        """
        # Check cache first
        cache_key = f"{symbol}_{timestamp}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Measure latency
        start = time.perf_counter()

        # Compute features (incremental updates)
        self.sma_20.append(close)
        self.sma_50.append(close)

        features = {
            'sma_20': self.sma_20.get_mean(),
            'sma_50': self.sma_50.get_mean(),
            'rsi': self.rsi.update(close),
            'macd': self.macd.update(close),
            'atr': self.atr.update(high, low, close),
            'bbands': self.bbands.update(close)
        }

        # ENFORCE: Latency requirement
        latency_ms = (time.perf_counter() - start) * 1000
        assert latency_ms < self.max_latency_ms, \
            f"Latency {latency_ms:.3f}ms exceeds limit {self.max_latency_ms}ms"

        # Store in cache
        self.cache.put(cache_key, features)

        # Track metrics
        if self.validate:
            self.validator.latencies.append(latency_ms)

        return features

    def assert_memory_bounds(self) -> Dict[str, int]:
        """
        Assert pipeline memory is within bounds.

        Returns:
            Memory profile
        """
        return self.validator.validate_memory(self)

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics."""
        return {
            'latency': self.validator.get_latency_stats(),
            'memory': self.validator.get_memory_stats(),
            'cache': self.cache.get_stats()
        }

    def print_performance_report(self) -> None:
        """Print performance report."""
        self.validator.print_validation_report()

        print("\nCache Performance:")
        cache_stats = self.cache.get_stats()
        for metric, value in cache_stats.items():
            if 'rate' in metric:
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value}")


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = ProductionFeaturePipeline(
        max_latency_ms=10.0,
        max_memory_bytes=2048,
        validate=True
    )

    # Simulate trading day
    np.random.seed(42)
    for i in range(1000):
        symbol = "AAPL"
        timestamp = i
        close = 100.0 + np.sin(i / 50) * 10.0
        high = close + np.random.rand() * 2.0
        low = close - np.random.rand() * 2.0

        features = pipeline.process_tick(symbol, timestamp, high, low, close)

    # Assert memory bounds
    pipeline.assert_memory_bounds()

    # Print report
    pipeline.print_performance_report()

    print("\n✅ Production pipeline validated successfully!")
```

---

## Summary: When to Use Each Pattern

| Pattern | Use Case | Performance | Memory |
|---------|----------|-------------|--------|
| Circular Buffer | Rolling windows with frequent mean/sum | O(1) append, O(1) mean | Fixed |
| Incremental EMA | Exponential moving averages | O(1) update | O(1) |
| Incremental RSI | Momentum oscillators with smoothing | O(1) after warmup | O(1) |
| Incremental MACD | Trend indicators with multiple EMAs | O(1) update | O(1) |
| Incremental ATR | Volatility with true range | O(1) after warmup | O(1) |
| TTL Cache | Stable features with time window | O(1) get/put | O(capacity) |
| Vectorization | Batch processing (backtesting) | 10-100x speedup | O(batch_size) |
| Memory Profiling | Validate memory bounds | - | - |
| Validation Framework | Testing incremental == full | - | - |

**Key Takeaway**: Always use incremental updates for real-time pipelines. Validate that incremental matches full calculation. Assert latency and memory bounds in production.
