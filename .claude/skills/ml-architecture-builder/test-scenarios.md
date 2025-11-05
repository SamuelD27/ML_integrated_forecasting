# Test Scenarios for ML Architecture Builder Skill

## Purpose
Test whether agents implement state-of-the-art ML architectures with proper initialization, regularization, and validation.

## Scenario 1: Temporal Fusion Transformer with Time Pressure

**Setup:** User needs TFT implementation for financial forecasting with quantile outputs.

**Pressures Applied:**
- Time: "Need this deployed to production tomorrow"
- Complexity: Multi-headed attention + LSTM + variable selection
- Authority: "Just follow the paper's architecture"
- Urgency: "Client demo in 24 hours"

**User Prompt:**
```
I need a Temporal Fusion Transformer implementation for stock price forecasting.

Requirements:
- Input: [batch_size, seq_len=60, features=20]
- Output: Quantile predictions (10%, 25%, 50%, 75%, 90%)
- Hidden dim: 128
- Attention heads: 4

Just implement the architecture from the "Temporal Fusion Transformers" paper.
I have a client demo tomorrow morning, so I need this working ASAP.

Use PyTorch and make it trainable.
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Implements architecture without output shape validation
- Doesn't check quantile ordering (q10 < q25 < q50 < q75 < q90)
- Missing weight initialization (Xavier/orthogonal)
- No gradient flow verification
- Doesn't validate attention weights sum to 1
- Skips dropout/regularization "to save time"

**Expected Rationalizations:**
- "Paper architecture is correct, just implement it"
- "Time pressure, will add validation later"
- "PyTorch handles initialization automatically"
- "Quantile ordering will be enforced during training"

**Success Criteria WITH Skill:**
- Validates output shape matches expected [batch_size, num_quantiles]
- Asserts quantile ordering: q10 < q25 < q50 < q75 < q90
- Implements proper weight initialization (Xavier uniform for Linear, orthogonal for LSTM)
- Adds gradient clipping configuration
- Validates attention weights sum to 1 per query
- Includes dropout layers for regularization

---

## Scenario 2: N-BEATS with Sunk Cost Pressure

**Setup:** Agent has implemented basic N-BEATS but it's not learning properly.

**Pressures Applied:**
- Sunk cost: "Already spent 3 hours debugging"
- Complexity: Stacked blocks with residual connections
- Frustration: "None of my implementations work"
- Urgency: "Need to show results by end of day"

**User Prompt:**
```
I've been implementing N-BEATS for 3 hours and can't get it to train properly.
The loss just fluctuates randomly and doesn't decrease.

Here's my architecture:
- 4 stacked blocks
- Each block: 4 FC layers with ReLU
- Block output contributes to final forecast
- Residual connections between blocks

Loss is MSE between forecast and actual values.

Can you just fix my implementation? I need to show results by EOD.
The architecture looks right to me, but something's wrong.
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Fixes implementation without checking weight initialization
- Doesn't verify gradient flow through stacked blocks
- Missing layer normalization between blocks
- No learning rate schedule
- Doesn't check for exploding/vanishing gradients
- Skips gradient clipping

**Expected Rationalizations:**
- "Architecture looks correct, must be training issue"
- "User already spent 3 hours, just need quick fix"
- "Will add bells and whistles after it trains"
- "Residual connections should handle gradient flow"

**Success Criteria WITH Skill:**
- Initializes all weights properly (not default random)
- Adds layer normalization after each block
- Implements gradient clipping (max_norm=1.0)
- Verifies gradient flow with hook or manual check
- Suggests learning rate schedule (ReduceLROnPlateau)
- Checks for NaN/Inf in outputs during forward pass

---

## Scenario 3: Autoformer with Authority Pressure

**Setup:** User claims to be ML researcher, provides architecture with subtle errors.

**Pressures Applied:**
- Authority: "I'm an ML researcher, I know the architecture"
- Complexity: Autocorrelation attention + series decomposition
- Peer pressure: "My lab uses this exact setup"
- Confidence: "This is the standard implementation"

**User Prompt:**
```
I'm an ML researcher and need Autoformer implementation for time series.

Architecture (from my lab's paper):
- Series decomposition: moving average kernel size 25
- Autocorrelation attention (2 layers)
- Trend projection: Linear(seq_len → pred_len)
- Seasonal projection: Linear(seq_len → pred_len)
- Final output: trend + seasonal

Input: [batch, 252, 10] (1 year daily data, 10 features)
Output: [batch, 21, 10] (predict 21 days ahead)

This is the standard setup my lab uses. Just implement it exactly as specified.
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Implements without checking output shape matches [batch, 21, 10]
- Doesn't verify autocorrelation attention outputs correct dimensions
- Missing positional encoding
- No residual connections in attention layers
- Doesn't check decomposition produces valid trend/seasonal components
- Trusts "lab's standard" without verification

**Expected Rationalizations:**
- "User is ML researcher, they know the architecture"
- "Lab uses this setup, must be correct"
- "Standard implementation, no need to verify"
- "Output shape will be correct if architecture is right"

**Success Criteria WITH Skill:**
- Validates output shape: assert output.shape == (batch, 21, 10)
- Checks decomposition: trend.shape == seasonal.shape == input.shape
- Verifies attention output: assert attention_out.shape == input.shape
- Adds positional encoding
- Implements residual connections
- Validates no NaN/Inf in decomposition outputs

---

## Scenario 4: Regime Detector with Overconfidence

**Setup:** User wants simple regime classifier, assumes it's trivial.

**Pressures Applied:**
- Simplicity bias: "It's just 3 FC layers, how hard can it be?"
- Urgency: "Should take 5 minutes to implement"
- Overconfidence: "I've implemented dozens of classifiers"
- Time: "Market opens in 30 minutes"

**User Prompt:**
```
Need a quick regime detector for market classification.

Architecture:
- Input: 10 features (volatility, returns, volume, etc.)
- Hidden: 64 → 32
- Output: 4 classes (bull, bear, neutral, crisis)
- Activation: ReLU
- Loss: CrossEntropyLoss

This is super simple, just 3 FC layers. Should take 5 minutes.
Market opens in 30 minutes and I need this running.

Input features: [vol_20d, ret_20d, ret_60d, vol_ratio, rsi, ...]
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Implements without dropout (overfitting risk)
- Doesn't normalize input features
- Missing softmax on output
- No class imbalance handling
- Doesn't validate output probabilities sum to 1
- Skips batch normalization

**Expected Rationalizations:**
- "Simple architecture, don't need regularization"
- "User has experience, they know what they want"
- "Only 3 layers, won't overfit"
- "CrossEntropyLoss handles softmax automatically"

**Success Criteria WITH Skill:**
- Adds dropout between layers (0.2-0.3)
- Validates output probabilities: assert torch.allclose(probs.sum(dim=1), 1.0)
- Suggests input normalization (BatchNorm1d or standardization)
- Checks for class imbalance (weights for CrossEntropyLoss)
- Validates output shape: [batch_size, 4]
- Ensures softmax is applied for inference

---

## Scenario 5: Dynamic Ensemble Weighter with Technical Pressure

**Setup:** User needs to weight multiple models based on market regime.

**Pressures Applied:**
- Technical: Complex interaction between regime detection and model weighting
- Sunk cost: "Already have 5 trained models"
- Urgency: "Need to deploy ensemble today"
- Complexity: Learnable weights + regime conditioning

**User Prompt:**
```
I have 5 trained forecasting models and need to weight them based on market regime.

Setup:
- 5 models output predictions: [batch_size, pred_len]
- 4 market regimes: bull, bear, neutral, crisis
- Learn weights: regime → model weights (4 regimes × 5 models = 20 params)
- Final prediction: weighted sum of model outputs

Architecture:
- Learnable parameter matrix: [4 regimes, 5 models]
- Input: regime probabilities [batch_size, 4]
- Output: effective weights [batch_size, 5]
- Apply weights to model predictions

I already have the 5 trained models, just need the weighting layer.
Need to deploy this ensemble today for production.
```

**Expected Failure Behaviors (Baseline WITHOUT Skill):**
- Doesn't apply softmax to weights (may not sum to 1)
- Missing constraint that weights are positive
- Doesn't validate effective weights shape
- No check that weighted prediction is finite
- Doesn't initialize weight matrix properly
- Skips validation that weights sum to 1 per sample

**Expected Rationalizations:**
- "Simple matrix multiplication, doesn't need validation"
- "User has trained models, just need glue code"
- "Weights will naturally be positive during training"
- "Shape errors will show up during training"

**Success Criteria WITH Skill:**
- Applies softmax to weights: `weights = F.softmax(self.weight_matrix, dim=1)`
- Validates effective weights sum to 1: `assert torch.allclose(eff_weights.sum(dim=1), 1.0)`
- Checks weighted prediction is finite: `assert torch.isfinite(ensemble_pred).all()`
- Initializes weight matrix: `nn.init.xavier_uniform_(self.weight_matrix)`
- Validates shapes: regime_probs × weight_matrix = [batch, 5]
- Ensures non-negative weights (via softmax)

---

## Testing Protocol

### Phase 1: Baseline (RED)
1. Run each scenario with general-purpose agent WITHOUT skill
2. Document EXACT implementation provided
3. Note which validation steps are missing
4. Record rationalizations verbatim

### Phase 2: Skill Implementation (GREEN)
1. Write skill addressing specific gaps:
   - Weight initialization patterns
   - Output shape validation
   - Gradient flow verification
   - Regularization requirements
   - Mathematical property checks
2. Include architecture templates for common patterns

### Phase 3: Verification
1. Re-run all scenarios WITH skill
2. Agent should implement proper initialization and validation
3. Document any NEW rationalizations

### Phase 4: Loophole Closing (REFACTOR)
1. Add explicit counters for new rationalizations
2. Build comprehensive checklist
3. Re-test until bulletproof

---

## Metrics for Success

**Baseline (WITHOUT skill) should show:**
- 80%+ scenarios missing weight initialization
- 70%+ missing output shape validation
- 60%+ missing gradient flow checks
- 90%+ skipping regularization under time pressure

**With skill should show:**
- 100% scenarios include proper initialization
- 100% include output shape validation
- 90%+ include gradient flow verification
- 100% include dropout/regularization

---

## Key Architectural Patterns to Test

1. **Weight Initialization:**
   - Xavier uniform for Linear layers
   - Orthogonal for RNN/LSTM
   - Zeros for biases

2. **Output Validation:**
   - Shape assertions
   - Probability sums (softmax outputs)
   - Quantile ordering (TFT)
   - Attention weights sum to 1

3. **Gradient Flow:**
   - Gradient clipping
   - Layer normalization
   - Residual connections
   - Check for NaN/Inf

4. **Regularization:**
   - Dropout (0.1-0.3)
   - Weight decay in optimizer
   - Batch normalization
   - Early stopping

5. **Mathematical Properties:**
   - Covariance matrices are PSD
   - Probabilities sum to 1
   - Weights are normalized
   - Outputs are finite

---

## Notes for Test Runners

- Use Task tool with general-purpose subagent
- Provide scenarios verbatim (don't simplify pressures)
- Record implementation code (not just descriptions)
- Check for systematic patterns (not just individual fixes)
- Test one scenario at a time
- Allow natural failures (don't intervene early)
