# Test Scenarios for Time Series Validation Specialist

## Purpose
Test whether agents implement proper time series validation to prevent look-ahead bias, data leakage, and overfitting.

## Scenario 1: Walk-Forward Validation with Time Pressure

**Setup:** User needs backtesting for trading strategy, wants quick results.

**Pressures Applied:**
- Time: "Need backtest results today for investor meeting"
- Simplicity bias: "Just split 80/20 train/test"
- Authority: "Standard ML approach should work fine"
- Urgency: "Already spent hours on strategy code"

**User Prompt:**
```
I need to backtest my trading strategy ASAP. I have 5 years of daily stock data.

Requirements:
- Train model on historical data
- Test on recent data
- Report Sharpe ratio and returns

Just use standard train_test_split (80/20). I have an investor meeting this afternoon
and need to show backtest results.

Data: 2018-01-01 to 2023-12-31 (5 years, ~1250 trading days)
Strategy: ML model predicts next-day returns based on technical indicators
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Uses sklearn's train_test_split (random shuffle)
- Doesn't preserve temporal order
- No walk-forward validation
- Single train/test split (no cross-validation)
- Doesn't account for regime changes over 5 years
- Reports inflated performance (look-ahead from random split)

**Expected Rationalizations:**
- "train_test_split is standard ML practice"
- "80/20 split is common rule of thumb"
- "Time pressure, can't do complex validation"
- "Model performance will be similar on any split"

**Success Criteria WITH Skill:**
- Implements walk-forward validation (rolling window)
- Preserves temporal order (no shuffling)
- Multiple test windows (not single split)
- Reports performance degradation over time
- Includes embargo period between train/test
- Warns about regime-specific overfitting

---

## Scenario 2: Feature Engineering with Look-Ahead Bias

**Setup:** User creating features for price prediction, accidentally uses future data.

**Pressures Applied:**
- Complexity: Many features being computed
- Overconfidence: "I know these are good predictors"
- Technical: Complex feature calculations
- Results: "Model performs great!" (due to leak)

**User Prompt:**
```
My model is performing amazing - 95% accuracy and Sharpe ratio of 8.5!

Here are my features:
- Current price
- 20-day moving average
- Volume spike: current volume vs future 5-day avg volume  [LEAK!]
- Return: (future close - current close) / current close  [LEAK!]
- RSI using next 14 days  [LEAK!]
- Target: next day return

Something seems too good to be true, but the code runs without errors.
Can you review my feature engineering and make sure it's correct?
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Doesn't identify look-ahead bias in features
- Accepts "95% accuracy" at face value
- Doesn't check feature timestamps
- Trusts that "no errors" means correct
- Doesn't question suspiciously high performance

**Expected Rationalizations:**
- "Code runs without errors, must be correct"
- "High performance means good features"
- "User says features are good predictors"
- "Focus on fixing bugs, not validation"

**Success Criteria WITH Skill:**
- Identifies all look-ahead bias (volume spike, return calculation, RSI)
- Explains why Sharpe 8.5 is unrealistic (indicates leakage)
- Checks that feature timestamps < target timestamps
- Requires features to use only past data
- Suggests proper feature computation (lag by 1 day minimum)

---

## Scenario 3: K-Fold Cross-Validation on Time Series

**Setup:** User wants to use k-fold CV for model selection, doesn't understand time series specifics.

**Pressures Applied:**
- Best practices: "k-fold is gold standard for CV"
- Authority: "Learned this in ML course"
- Simplicity: "sklearn makes it easy"
- Confidence: "5-fold CV is industry standard"

**User Prompt:**
```
I'm tuning hyperparameters for my time series forecasting model.

Plan:
1. Use sklearn KFold with 5 splits
2. Train on 4 folds, test on 1 fold
3. Repeat 5 times
4. Average scores to select best hyperparameters

This is standard ML best practice from my course. Implement this for me.

Data: Stock prices 2020-2023
Model: LSTM predicting next 5 days
Hyperparameters: learning rate, hidden size, num layers
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Implements standard KFold (shuffles or assigns randomly)
- Doesn't preserve temporal order in splits
- Training on future data, testing on past data
- No embargo period between folds
- Doesn't recognize time series violation

**Expected Rationalizations:**
- "KFold is standard best practice"
- "sklearn's KFold is battle-tested"
- "Course taught this method"
- "Cross-validation prevents overfitting"

**Success Criteria WITH Skill:**
- Rejects standard KFold for time series
- Implements TimeSeriesSplit or purged k-fold
- Ensures training only on past, testing on future
- Adds embargo period (gap between train/test)
- Explains why standard k-fold fails for time series

---

## Scenario 4: Target Leakage in Feature Set

**Setup:** User's features contain information about the target variable.

**Pressures Applied:**
- Sunk cost: "Already trained for 2 hours"
- Results: "Model converged nicely"
- Complexity: "50+ features, hard to check all"
- Urgency: "Need to deploy tomorrow"

**User Prompt:**
```
My model trained successfully and converged nicely. Here are my features:

Technical indicators:
- RSI, MACD, Bollinger Bands (all standard)
- Volume ratios
- Price momentum

Derived features:
- Daily return: (close - open) / open
- Intraday high-low spread: (high - low) / close
- Close-to-close return: (close_t - close_t-1) / close_t-1  [LEAK if target is price_t]

Target: next day's close price

Model: Random Forest, trained for 2 hours
Performance: R² = 0.94, MAE = 0.3%

Looks good to deploy tomorrow. Can you review?
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Doesn't identify close_t in features when target is price_t+1
- Accepts R² = 0.94 as realistic
- Doesn't check for feature-target correlation > 0.9
- Trusts that "trained successfully" means correct
- Doesn't question suspiciously low MAE

**Expected Rationalizations:**
- "All features are from technical indicators"
- "Model trained without errors"
- "High R² means good fit"
- "Standard features shouldn't have leakage"

**Success Criteria WITH Skill:**
- Identifies close_t in features as leakage source
- Checks feature-target correlations
- Questions R² = 0.94 as too good (typical 0.1-0.3 for prices)
- Requires features to use t-1 or earlier only
- Suggests proper feature lagging

---

## Scenario 5: Overfitting to Historical Regime

**Setup:** User backtests on bull market, model fails in bear market.

**Pressures Applied:**
- Success bias: "Great backtest results!"
- Single regime: Only tested on 2020-2021 bull market
- Overconfidence: "Model learned the patterns"
- Urgency: "Ready to deploy"

**User Prompt:**
```
My model has amazing backtest results:
- Sharpe ratio: 3.2
- Max drawdown: 5%
- Win rate: 68%

Backtested on 2020-2021 data (S&P 500 bull market).

Now deploying to production, but performance is terrible in Q1 2022:
- Sharpe: -0.5
- Drawdown: 18%
- Win rate: 38%

What's going wrong? The model worked great in backtest!
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Doesn't identify single-regime overfitting
- Doesn't suggest walk-forward through multiple regimes
- Doesn't recommend regime-adaptive parameters
- Focuses on tweaking model instead of validation approach
- Doesn't check if backtest period representative

**Expected Rationalizations:**
- "Backtest was thorough (2 years of data)"
- "Model just needs retraining"
- "Market changed, not a validation issue"
- "High Sharpe in backtest proves model quality"

**Success Criteria WITH Skill:**
- Identifies regime-specific overfitting (only bull market)
- Requires walk-forward through multiple regimes (bull/bear/neutral)
- Suggests expanding backtest to 2008-2021 (includes crashes)
- Recommends regime detection and adaptive parameters
- Explains that 2020-2021 is non-representative period

---

## Testing Protocol

### Phase 1: Baseline (RED)
1. Run each scenario with general-purpose agent WITHOUT skill
2. Document EXACT approaches provided
3. Note which validation steps are missing
4. Record rationalizations verbatim

### Phase 2: Skill Implementation (GREEN)
1. Write skill addressing specific gaps:
   - Walk-forward validation implementation
   - Look-ahead bias detection methods
   - Purged k-fold for time series
   - Target leakage checks
   - Regime-aware validation
2. Include code utilities for common validations

### Phase 3: Verification
1. Re-run all scenarios WITH skill
2. Agent should implement proper time series validation
3. Document any NEW rationalizations

### Phase 4: Loophole Closing (REFACTOR)
1. Add explicit counters for new rationalizations
2. Build comprehensive validation checklist
3. Re-test until bulletproof

---

## Metrics for Success

**Baseline (WITHOUT skill) should show:**
- 80%+ scenarios use random train/test split
- 90%+ don't check for look-ahead bias
- 100% don't implement embargo periods
- 80%+ don't validate across multiple regimes
- 70%+ don't detect target leakage

**With skill should show:**
- 100% use temporal train/test splits
- 100% check for look-ahead bias
- 90%+ implement embargo periods
- 90%+ validate across regimes
- 100% detect target leakage

---

## Key Validation Patterns to Test

1. **Temporal Ordering:**
   - Training always on past
   - Testing always on future
   - No shuffling of data

2. **Look-Ahead Bias:**
   - Features only use t-1 or earlier
   - No future data in calculations
   - Timestamp validation

3. **Target Leakage:**
   - Features don't contain target
   - Correlation checks (features vs target)
   - Information flow analysis

4. **Embargo Periods:**
   - Gap between train and test
   - Accounts for settlement lag
   - Prevents overlapping samples

5. **Regime Validation:**
   - Multiple market conditions
   - Walk-forward through regimes
   - Performance consistency checks

---

## Notes for Test Runners

- Use Task tool with general-purpose subagent
- Provide scenarios verbatim (include subtle leaks)
- Record whether agent catches leakage
- Check if validation is systematic (not ad-hoc fixes)
- Test one scenario at a time
- Allow natural failures before intervention
