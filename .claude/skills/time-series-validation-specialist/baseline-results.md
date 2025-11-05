# Baseline Test Results - Time Series Validation Specialist

## Test Date
2025-11-05

## Key Finding
**Agents catch OBVIOUS leakage but may not implement systematic validation without existing code patterns.**

Agents will:
- ✅ Identify blatant look-ahead bias (future volume, RSI with future prices)
- ✅ Question suspiciously high performance (Sharpe 8.5, accuracy 95%)
- ✅ Use existing walk-forward patterns when available in codebase
- ✅ Explain why random split is wrong for time series

Agents will NOT:
- ❌ Implement walk-forward validation from scratch (without example)
- ❌ Add embargo periods unless prompted
- ❌ Check for SUBTLE leakage (close_t when predicting price_t+1)
- ❌ Validate across multiple market regimes
- ❌ Implement purged k-fold systematically

## Test Results by Scenario

### Scenario 2: Look-Ahead Bias Review (TESTED)
**Agent Response:** Caught ALL major leakage issues
**Key Findings:**
- ✅ Identified volume spike using future data
- ✅ Identified RSI using next 14 days
- ✅ Identified return feature containing target
- ✅ Questioned Sharpe 8.5 as unrealistic
- ✅ Explained why each feature causes leakage

**Direct Quote:**
> "This is a direct look-ahead violation. You're using future volume data (the next 5 days) to predict today's return."

**Conclusion:** Agents successfully catch OBVIOUS leakage when reviewing features.

### Scenario 1: Walk-Forward Implementation (TESTED)
**Agent Response:** Used existing WalkForwardSplitter from codebase
**Key Findings:**
- ✅ Recognized walk-forward as correct approach
- ✅ Used existing implementation (training/training_utils.py)
- ✅ Created comparison showing why random split is wrong
- ✅ Explained look-ahead bias from shuffling

**However:**
- ⚠️ Agent used EXISTING code, not implemented from scratch
- ⚠️ Success may be due to codebase having correct patterns
- ⚠️ Unclear if agent would implement walk-forward without example

**Quote:**
> "Your CLAUDE.md explicitly specifies ADR-004: Walk-Forward Validation for this exact reason."

**Conclusion:** Agents will follow existing patterns but may not create them independently.

## Critical Gap Identified

**The skill gap is not about DETECTING leakage—it's about SYSTEMATICALLY PREVENTING it.**

Agents will:
- Catch obvious mistakes during review ✅
- Use existing correct patterns ✅
- Explain concepts clearly ✅

But won't:
- Implement validation systematically from scratch ❌
- Add embargo periods without prompting ❌
- Check for subtle leakage programmatically ❌
- Validate across regimes automatically ❌

## Comparison: Review vs Implementation

**When REVIEWING features (Scenario 2):**
```
Agent behavior: Excellent
- Found all 3 leakage issues
- Explained each clearly
- Questioned suspicious metrics
- Provided fixes
```

**When IMPLEMENTING from scratch (Scenario 1):**
```
Agent behavior: Good but dependent on examples
- Used existing WalkForwardSplitter
- Didn't implement from scratch
- May default to random split without example
```

## What Agents Need

### Pattern 1: Walk-Forward Template
Agents need a clear template to implement even without codebase examples:

```python
# What agents DO when they have example:
from training.training_utils import WalkForwardSplitter  # Uses existing

# What agents SHOULD do without example:
def walk_forward_split(data, lookback=252, test_size=63, step=30):
    """Proper walk-forward without external dependencies."""
    # ... implementation from scratch
```

### Pattern 2: Leakage Detection Checklist
Agents need systematic checks, not ad-hoc review:

```python
# What agents do now: Review and spot obvious issues

# What agents need: Systematic validation
def check_feature_leakage(features_df, target_series, feature_timestamps):
    """Automatically detect look-ahead bias."""
    for col in features_df.columns:
        # Check timestamps
        assert (feature_timestamps[col] <= target_timestamps).all()

        # Check correlation (>0.95 suspicious)
        corr = features_df[col].corr(target_series)
        if abs(corr) > 0.95:
            warnings.warn(f"{col} has correlation {corr:.3f} with target")
```

### Pattern 3: Embargo Period Calculation
Agents don't add embargo periods automatically:

```python
# What agents skip: Embargo gaps

# What skill should teach:
def calculate_embargo_period(data_frequency, settlement_days=2):
    """
    Calculate minimum embargo period.

    Args:
        data_frequency: 'daily', 'hourly', '5min'
        settlement_days: Settlement lag (T+2 for US stocks)
    """
    if data_frequency == 'daily':
        return settlement_days  # 2 days
    elif data_frequency == 'hourly':
        return settlement_days * 6.5  # Trading hours
    # ... etc
```

## Subtle Leakage Agents Might Miss

### Example 1: Current close in features
```python
# Agents may not catch this:
features['close'] = data['close'].values  # Uses close_t
target = data['close'].shift(-1).values   # Predicts close_t+1

# Problem: When predicting close_t+1, close_t contains information
# Should use close_t-1 or earlier only
```

### Example 2: Intraday data timing
```python
# Agents may not validate this:
features['volume_10am'] = data.loc[data.time == '10:00', 'volume']
target_time = '09:30'  # Market open

# Problem: Can't use 10am volume to predict 9:30am price
# Need timestamp validation
```

### Example 3: Cross-asset leakage
```python
# Agents may not check this:
features['spy_return'] = spy_data['return'].values  # S&P 500 return
target = aapl_data['return'].values  # Apple return

# Problem: If timestamps don't align, may use future S&P to predict Apple
# Need cross-asset timestamp synchronization
```

## Rationalization Patterns Observed

### Pattern 1: "Obvious leakage is caught, subtle is fine"
- Agent identifies blatant look-ahead bias
- But doesn't implement SYSTEMATIC checks
- Assumes if obvious cases are caught, all is well

**Reality:** Subtle leakage is more dangerous (goes unnoticed)

### Pattern 2: "Existing code means validation is correct"
- Agent sees WalkForwardSplitter in codebase
- Uses it without verifying it's actually used correctly
- Trusts that existing patterns are sufficient

**Reality:** Need to verify usage, not just existence

### Pattern 3: "High performance is questioned but not validated"
- Agent says "Sharpe 8.5 is too good to be true"
- But doesn't implement CHECK that would catch it
- Relies on human judgment, not automated validation

**Reality:** Should implement `assert sharpe < 5` as code

## Skill Design Implications

Based on findings, the Time Series Validation Specialist skill must:

### 1. Provide Implementation Templates (Not Just Concepts)
**Don't:**
```markdown
Use walk-forward validation to prevent look-ahead bias.
```

**Do:**
```markdown
ALWAYS implement walk-forward validation:
```python
def walk_forward_split(data, lookback=252, test_size=63, step=30):
    """Walk-forward split preserving temporal order."""
    train_start = 0
    train_end = lookback
    test_end = train_end + test_size

    while test_end <= len(data):
        train_idx = range(train_start, train_end)
        test_idx = range(train_end, test_end)

        yield train_idx, test_idx

        train_start += step
        train_end += step
        test_end += step
```

### 2. Make Leakage Checks MANDATORY
Every time series model must include:
```python
def validate_no_leakage(features, target, feature_times, target_times):
    """Check for look-ahead bias and target leakage."""
    # Timestamp check
    for col in features.columns:
        assert (feature_times[col] <= target_times).all(), \
            f"{col} contains future data"

    # Correlation check
    for col in features.columns:
        corr = features[col].corr(target)
        assert abs(corr) < 0.95, \
            f"{col} correlation {corr:.3f} indicates leakage"
```

### 3. Embargo Periods REQUIRED
**Don't:**
```python
train_idx = range(0, 1000)
test_idx = range(1000, 1250)  # No gap
```

**Do:**
```python
embargo_days = 2  # T+2 settlement
train_idx = range(0, 1000)
test_idx = range(1000 + embargo_days, 1250)  # Gap enforced
```

### 4. Regime Validation CHECKLIST
Before deploying any backtest:
- [ ] Walk-forward spans bull AND bear markets
- [ ] At least 2 complete market cycles
- [ ] Performance consistent across regimes
- [ ] Out-of-sample Sharpe within 30% of in-sample
- [ ] Maximum 1-year test windows (prevent overfitting)

### 5. Rationalization Table

| Excuse | Reality | Fix |
|--------|---------|-----|
| "Obvious leakage is caught" | Subtle leakage is more dangerous | Implement systematic checks |
| "Code runs without errors" | Errors are silent in leakage | Add validation assertions |
| "Existing patterns are used" | Need to verify correct usage | Check implementation, not just existence |
| "High performance questioned" | Need automated validation | Add `assert sharpe < 5` as code |
| "Random split is standard ML" | Wrong for time series | Use walk-forward always |

## Scenarios Still Needed

**Tested:**
- ✅ Scenario 2: Look-ahead bias review (agents catch it)
- ✅ Scenario 1: Walk-forward implementation (agents use existing)

**Still need to test:**
- ⏳ Scenario 3: K-fold on time series (do agents reject it?)
- ⏳ Scenario 4: Subtle target leakage (close_t predicting price_t+1)
- ⏳ Scenario 5: Regime overfitting (bull market backtest fails in bear)

**Recommendation:** Test Scenario 4 to see if agents catch SUBTLE leakage.

## Conclusion

**The skill gap is implementation and systematization, not understanding.**

**Agents understand:**
- Walk-forward is correct for time series ✅
- Look-ahead bias is bad ✅
- Random split causes problems ✅

**But agents don't:**
- Implement walk-forward from scratch ❌
- Add systematic leakage checks ❌
- Include embargo periods ❌
- Validate across regimes ❌

**The Time Series Validation Specialist skill must provide:**
1. Implementation templates (copy-paste ready)
2. Mandatory validation functions
3. Leakage detection automation
4. Embargo period calculators
5. Regime validation checklists

Transform understanding into enforceable, systematic validation.
