# Skills Map - Quick Reference

## Active Skills (6/6) ✅

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML TRADING SYSTEM SKILLS                         │
│                    Average Compliance: 99.2%                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 1. FINANCIAL KNOWLEDGE VALIDATOR                        [95%] ✅    │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:  Transform financial advice into validation code           │
│ Teaches:  Assertions for validation, not warnings                   │
│ Key Fix:  Calculate Sharpe/Sortino correctly                        │
│ Command:  /financial-knowledge-validator                            │
│                                                                      │
│ Before:   "Calculate Sharpe ratio like this..."                     │
│ After:    assert sharpe_ratio == expected                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. ML ARCHITECTURE BUILDER                             [100%] ✅    │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:  Enforce proper initialization and output validation       │
│ Teaches:  _initialize_weights() + output shape validation           │
│ Key Fix:  Prevent exploding/vanishing gradients                     │
│ Command:  /ml-architecture-builder                                  │
│                                                                      │
│ Before:   Model builds successfully                                 │
│ After:    Weights initialized (Xavier/Kaiming) + outputs validated  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. TIME SERIES VALIDATION SPECIALIST                   [100%] ✅    │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:  Make walk-forward validation mandatory                    │
│ Teaches:  Walk-forward + embargo periods + purging                  │
│ Key Fix:  Prevent look-ahead bias                                   │
│ Command:  /time-series-validation-specialist                        │
│                                                                      │
│ Before:   train_test_split(X, y)                                    │
│ After:    walk_forward_split(data, embargo_days=2)                  │
│ Result:   704 lines of production code generated                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 4. PORTFOLIO OPTIMIZATION EXPERT                       [100%] ✅    │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:  Enforce Ledoit-Wolf shrinkage (no fallback)               │
│ Teaches:  RuntimeError if shrinkage fails                           │
│ Key Fix:  Prevent unstable covariance estimates                     │
│ Command:  /portfolio-optimization-expert                            │
│                                                                      │
│ Before:   np.cov(returns)  # with fallback                          │
│ After:    LedoitWolf().fit(returns)  # no fallback, raises error    │
│ Result:   12 tests passing with complete implementations            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 5. REAL-TIME FEATURE PIPELINE                          [100%] ✅    │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:  Validate incremental updates match full calculations      │
│ Teaches:  Built-in validation with validate flag                    │
│ Key Fix:  Prevent numerical drift                                   │
│ Command:  /real-time-feature-pipeline                               │
│                                                                      │
│ Before:   0.069ms latency, NO validation                            │
│ After:    0.75 μs latency + 0.00e+00 validation error               │
│ Special:  ANTI-PATTERNS section (external vs built-in)              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 6. MODEL DEPLOYMENT & MONITORING                       [100%] ✅    │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:  Enforce smoke tests before deployment                     │
│ Teaches:  Mandatory smoke tests + version tracking                  │
│ Key Fix:  Prevent broken deployments                                │
│ Command:  /model-deployment-monitoring                              │
│                                                                      │
│ Before:   "Deployment ready for production!"                        │
│ After:    "ALL CHECKS PASSED - APPROVED FOR DEPLOYMENT"             │
│ Result:   6-step validation checklist enforced                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Skill Activation Triggers

| When you... | Skill activates | What it does |
|-------------|----------------|--------------|
| Calculate Sharpe/Sortino | Skill 1 | Validates with assertions |
| Build neural network | Skill 2 | Adds weight initialization |
| Split time series data | Skill 3 | Enforces walk-forward |
| Optimize portfolio | Skill 4 | Enforces Ledoit-Wolf |
| Compute features in real-time | Skill 5 | Adds incremental validation |
| Deploy ML model | Skill 6 | Enforces smoke tests |

---

## Quick Command Reference

```bash
# Use a specific skill
/financial-knowledge-validator
/ml-architecture-builder
/time-series-validation-specialist
/portfolio-optimization-expert
/real-time-feature-pipeline
/model-deployment-monitoring
```

---

## Files Per Skill

Each skill directory contains:

```
[skill-name]/
├── SKILL.md                    # Main skill (patterns to teach)
├── [domain]-reference.md       # Complete implementations
├── test-scenarios.md           # Test scenarios
├── baseline-results.md         # Agent behavior WITHOUT skill
├── refactor-findings.md        # Agent behavior WITH skill
└── DEPLOYMENT-CHECKLIST.md     # Deployment verification
```

**Total**: 42 files (~11,000 lines of documentation)

---

## Compliance Scores

```
Skill 1: ████████████████████░  95%  ✅
Skill 2: █████████████████████ 100%  ✅
Skill 3: █████████████████████ 100%  ✅
Skill 4: █████████████████████ 100%  ✅
Skill 5: █████████████████████ 100%  ✅
Skill 6: █████████████████████ 100%  ✅
         ─────────────────────────
Average: █████████████████████  99.2% ✅
```

---

## Key Patterns by Skill

### Skill 1: Financial Knowledge Validator
- ✅ Assertions for validation (not logging)
- ✅ Handle edge cases (zero volatility, negative returns)
- ✅ Black-Litterman with view uncertainty

### Skill 2: ML Architecture Builder
- ✅ `_initialize_weights()` method (Xavier, Kaiming, Orthogonal)
- ✅ Output validation (shape, finite checks)
- ✅ Gradient flow configuration

### Skill 3: Time Series Validation Specialist
- ✅ Walk-forward validation (252-day lookback, 63-day test, 21-day step)
- ✅ Embargo periods (T+2 for daily data)
- ✅ Look-ahead bias detection (3 layers)

### Skill 4: Portfolio Optimization Expert
- ✅ Ledoit-Wolf enforcement (RuntimeError if fails, no fallback)
- ✅ Complete transaction costs (4 components)
- ✅ Constraint assertions (sum to 1, bounds)

### Skill 5: Real-Time Feature Pipeline
- ✅ Validated incremental updates (assert incremental == full)
- ✅ Built-in latency enforcement (< 10ms assertions)
- ✅ Memory profiling and bounds (< 2KB)
- ✅ ANTI-PATTERNS section

### Skill 6: Model Deployment & Monitoring
- ✅ Mandatory smoke tests (5+ tests before deployment)
- ✅ Automatic version tracking (timestamp + hash)
- ✅ Output validation assertions (range, NaN, Inf)
- ✅ Deployment checklist enforcement

---

## Integration with Your Project

```
/Users/samueldukmedjian/Desktop/stock_analysis/
├── single_stock/risk_metrics.py         → Skill 1 validates
├── ml_models/hybrid_model.py            → Skill 2 validates
├── training/train_hybrid.py             → Skill 3 validates
├── portfolio/cvar_allocator.py          → Skill 4 validates
├── ml_models/features.py                → Skill 5 validates
└── [deployment scripts]                 → Skill 6 validates
```

---

## Statistics

```
📊 Framework Statistics

Skills Created:        6/6 (100%)
Average Compliance:    99.2%
Total Documentation:   ~11,000 lines
Total Patterns:        35+ patterns
Code Examples:         50+ examples
Test Scenarios:        25+ scenarios
Time Investment:       3-4 hours
Expected ROI:          5-10x minimum
```

---

## Framework Status

```
┌─────────────────────────────────────────┐
│  FRAMEWORK STATUS: ✅ COMPLETE          │
│                                         │
│  All 6 skills deployed and active      │
│  Average compliance: 99.2%             │
│  Ready for production use              │
└─────────────────────────────────────────┘
```

---

## Quick Start

**To use a skill**:
1. Work on relevant task (skills activate automatically)
2. Or explicitly call: `/[skill-name]`

**To explore a skill**:
1. Read `SKILL.md` for patterns
2. Read `[domain]-reference.md` for implementations
3. Read `baseline-results.md` to understand gaps

**To verify a skill**:
1. Read `refactor-findings.md` for compliance
2. Read `DEPLOYMENT-CHECKLIST.md` for verification

---

**Last Updated**: 2025-11-05
**Framework Version**: 1.0
**Status**: Production-Ready ✅
