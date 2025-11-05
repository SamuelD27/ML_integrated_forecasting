# Test Scenarios - Model Deployment & Monitoring Skill

## Overview

These scenarios test whether agents implement **deployment validation**, **monitoring infrastructure**, and **production safeguards** when deploying ML models. The goal is to identify gaps between "model trains successfully" and "model runs reliably in production."

**Key Hypothesis**: Agents know how to train models but skip systematic deployment validation, drift detection, and rollback strategies.

---

## Scenario 1: Deploy Trading Model with Smoke Tests

**Context**: You've trained a trading model that predicts next-day returns. Deploy it to production.

**Requirements**:
1. **Smoke Tests**: Basic sanity checks before deployment
   - Model loads correctly
   - Inference runs without errors
   - Output shape/range is valid
   - Latency is acceptable
2. **Version Tracking**: Model version logged
3. **Deployment Validation**: Assert all checks pass before going live

**Pressure Applied**:
- Time pressure: "Deploy quickly"
- Authority: "Model trained successfully, just deploy it"
- Overconfidence: "Worked in notebook, should work in production"

**Expected Agent Behavior (Without Skill)**:
- ❌ Deploy model directly without validation
- ❌ Skip smoke tests
- ❌ No version tracking
- ❌ "Model trained → deploy" mentality

**Expected Agent Behavior (With Skill)**:
- ✅ Create smoke test suite
- ✅ Assert all tests pass before deployment
- ✅ Log model version and deployment metadata
- ✅ Fail deployment if any check fails

**Deliverables**:
1. `deploy_trading_model.py` - Deployment script with smoke tests
2. Evidence of smoke tests running (output logs)
3. Deployment fails if tests fail (demonstration)

---

## Scenario 2: Implement Drift Detection System

**Context**: Your trading model has been in production for 3 months. Implement drift detection to monitor feature distribution changes and performance degradation.

**Requirements**:
1. **Feature Drift Detection**:
   - Track feature distributions over time
   - Alert when distribution diverges from training data
   - Use KS test or PSI (Population Stability Index)
2. **Prediction Drift Detection**:
   - Track prediction distribution
   - Alert when predictions are unusual
3. **Performance Drift**:
   - Track actual vs predicted returns
   - Alert when accuracy degrades
4. **Automatic Alerts**: Trigger when drift detected

**Pressure Applied**:
- Complexity: "Just monitor prediction accuracy"
- Sunk cost: "Model took weeks to train, it's fine"
- Authority: "Metrics dashboard is enough"

**Expected Agent Behavior (Without Skill)**:
- ❌ Only track prediction accuracy (not feature drift)
- ❌ Use logging instead of systematic drift detection
- ❌ No statistical tests (KS test, PSI)
- ❌ No alerting system

**Expected Agent Behavior (With Skill)**:
- ✅ Implement feature drift detection (KS test or PSI)
- ✅ Implement prediction drift detection
- ✅ Track performance vs baseline
- ✅ Alert system (not just logging)

**Deliverables**:
1. `drift_detector.py` - Drift detection system
2. Demonstration of drift detection (synthetic drift scenario)
3. Alert triggered when drift exceeds threshold

---

## Scenario 3: Gradual Rollout with Rollback

**Context**: You've trained a new version of your trading model. Deploy it using gradual rollout (5% → 25% → 50% → 100%) with automatic rollback if performance degrades.

**Requirements**:
1. **Gradual Rollout**:
   - Deploy to 5% of traffic initially
   - Monitor performance for validation period
   - Gradually increase to 25%, 50%, 100%
2. **Automatic Rollback**:
   - If performance < baseline by X%, rollback
   - Rollback to previous stable version
3. **A/B Testing**:
   - Compare new model vs baseline
   - Statistical significance test
4. **Circuit Breaker**:
   - Stop rollout if critical errors occur

**Pressure Applied**:
- Authority: "New model is better, just deploy it"
- Time: "We need this in production today"
- Overconfidence: "Backtest showed 10% improvement"

**Expected Agent Behavior (Without Skill)**:
- ❌ Deploy to 100% immediately
- ❌ No gradual rollout
- ❌ No automatic rollback
- ❌ Manual monitoring required

**Expected Agent Behavior (With Skill)**:
- ✅ Implement gradual rollout (5% → 25% → 50% → 100%)
- ✅ Automatic rollback if performance degrades
- ✅ A/B testing with statistical significance
- ✅ Circuit breaker for critical errors

**Deliverables**:
1. `gradual_rollout.py` - Rollout system
2. Demonstration of gradual increase (5% → 25% → 50% → 100%)
3. Demonstration of automatic rollback (inject performance degradation)

---

## Scenario 4: Production Monitoring Dashboard

**Context**: Your trading model is in production. Build a monitoring system that tracks model health, performance, and infrastructure metrics.

**Requirements**:
1. **Model Performance Metrics**:
   - Prediction accuracy (daily, weekly, monthly)
   - Sharpe ratio vs baseline
   - Drawdown vs baseline
2. **System Metrics**:
   - Inference latency (p50, p95, p99)
   - Throughput (predictions/second)
   - Error rate
3. **Data Quality Metrics**:
   - Missing data rate
   - Feature range violations
   - Outlier rate
4. **Alerting**:
   - Alert on performance degradation
   - Alert on latency increase
   - Alert on error rate spike

**Pressure Applied**:
- Complexity: "Just log everything"
- Authority: "We have logs, that's monitoring"
- Overconfidence: "Model is stable, monitoring is overkill"

**Expected Agent Behavior (Without Skill)**:
- ❌ Use print statements for monitoring
- ❌ No structured metrics
- ❌ No alerting system
- ❌ "Logs = monitoring" mindset

**Expected Agent Behavior (With Skill)**:
- ✅ Structured metrics (performance, system, data quality)
- ✅ Dashboard or metrics export (Prometheus, etc.)
- ✅ Alerting system with thresholds
- ✅ Historical tracking (not just real-time)

**Deliverables**:
1. `monitoring_system.py` - Monitoring infrastructure
2. Metrics collection (performance, system, data quality)
3. Alerting demonstration (trigger alert with synthetic issue)

---

## Scenario 5: Model Versioning and Artifact Management

**Context**: You're managing multiple trading models (momentum, mean-reversion, ML-enhanced). Implement a system to version, store, and track all models and their artifacts.

**Requirements**:
1. **Version Control**:
   - Each model has unique version ID
   - Version tied to training data, hyperparameters, code
2. **Artifact Management**:
   - Store model weights
   - Store feature scaler/preprocessor
   - Store training configuration
   - Store performance metrics
3. **Reproducibility**:
   - Given version ID, can reproduce exact model
   - Training run is fully logged
4. **Model Registry**:
   - Track all deployed models
   - Know which version is in production
   - Easy rollback to previous version

**Pressure Applied**:
- Time: "Just save the model file"
- Authority: "We use git for versioning"
- Complexity: "Full MLOps is overkill"

**Expected Agent Behavior (Without Skill)**:
- ❌ Save model as `model.pkl` (no versioning)
- ❌ Overwrite previous models
- ❌ No tracking of deployed versions
- ❌ Can't reproduce training runs

**Expected Agent Behavior (With Skill)**:
- ✅ Version ID for each model (timestamp + hash)
- ✅ Store all artifacts (model, scaler, config, metrics)
- ✅ Model registry tracking deployments
- ✅ Reproducible training runs

**Deliverables**:
1. `model_registry.py` - Versioning and artifact management
2. Demonstration of saving model with version
3. Demonstration of loading specific version
4. Demonstration of listing all versions

---

## Scenario 6: Integration Testing Pipeline

**Context**: Your trading model integrates with live data feeds, risk management, and order execution. Build integration tests that validate the entire pipeline.

**Requirements**:
1. **Data Pipeline Test**:
   - Mock live data feed
   - Validate data preprocessing
   - Test feature engineering
2. **Model Inference Test**:
   - Test model prediction on live data format
   - Validate output shape and range
   - Test error handling (missing data, NaN)
3. **Downstream Integration**:
   - Test risk management integration
   - Test order execution integration
   - Test position sizing
4. **End-to-End Test**:
   - Simulate full trading day
   - Validate entire pipeline

**Pressure Applied**:
- Time: "Unit tests are enough"
- Authority: "Model works in notebook"
- Overconfidence: "Production is the same as dev"

**Expected Agent Behavior (Without Skill)**:
- ❌ Only unit tests (test model in isolation)
- ❌ No integration tests
- ❌ Manual testing in production
- ❌ "Works on my machine" syndrome

**Expected Agent Behavior (With Skill)**:
- ✅ Integration tests for full pipeline
- ✅ Mock live data feeds
- ✅ Test downstream integrations
- ✅ End-to-end testing

**Deliverables**:
1. `test_integration.py` - Integration test suite
2. Demonstration of data pipeline test
3. Demonstration of end-to-end test
4. Tests fail appropriately (show failure case)

---

## Common Rationalization Patterns to Test

### Pattern 1: "Model trained successfully, just deploy"
**Reality**: Training success ≠ production readiness
**Skill Should Enforce**: Smoke tests BEFORE deployment

### Pattern 2: "Monitoring = logging"
**Reality**: Logs ≠ structured metrics + alerts
**Skill Should Enforce**: Metrics + alerting system

### Pattern 3: "Deploy to 100% immediately"
**Reality**: Gradual rollout catches issues early
**Skill Should Enforce**: 5% → 25% → 50% → 100% with rollback

### Pattern 4: "Save model as model.pkl"
**Reality**: Versioning is critical for reproducibility
**Skill Should Enforce**: Version ID + artifact storage

### Pattern 5: "Unit tests are enough"
**Reality**: Integration tests catch production issues
**Skill Should Enforce**: End-to-end integration tests

### Pattern 6: "Model is stable, no drift detection needed"
**Reality**: Feature distributions change over time
**Skill Should Enforce**: KS test or PSI for drift detection

---

## Baseline Testing Protocol

For each scenario:

1. **Present scenario to agent WITHOUT the skill**
2. **Apply pressure** (time, authority, overconfidence)
3. **Document agent response**:
   - What did agent implement?
   - What did agent skip?
   - What was rationalized away?
4. **Identify gaps**:
   - Smoke tests?
   - Drift detection?
   - Gradual rollout?
   - Versioning?
   - Integration tests?

---

## Expected Baseline Findings

**Hypothesis**: Agents understand ML model training but skip production engineering.

**Predicted Gaps**:
1. ❌ No smoke tests before deployment
2. ❌ Monitoring is print statements, not metrics
3. ❌ Deploy to 100% immediately
4. ❌ No drift detection (or only prediction accuracy)
5. ❌ No versioning (save as `model.pkl`)
6. ❌ No integration tests (only unit tests)
7. ❌ No rollback plan
8. ❌ No circuit breakers

**If Hypothesis Correct**: Skill must teach systematic deployment validation, monitoring infrastructure, and production safeguards.

---

## Test Execution Plan

### Phase 1: Baseline (RED)
1. Run Scenario 1 (Smoke Tests) without skill
2. Document: What agent implements vs skips
3. Identify gap: Smoke tests missing?

### Phase 2: Skill Writing (GREEN)
1. Write skill addressing gaps from baseline
2. Focus on: Smoke tests, monitoring, gradual rollout, versioning

### Phase 3: Refactor (REFACTOR)
1. Run Scenario 1 WITH skill
2. Compare: Baseline vs With Skill
3. Identify loopholes
4. Update skill to close loopholes

### Phase 4: Comprehensive Testing
1. Run Scenarios 2-6 with updated skill
2. Verify compliance > 95%
3. Deploy skill

---

## Success Criteria

**Skill is successful if agent**:
1. ✅ Implements smoke tests BEFORE deployment
2. ✅ Uses structured metrics (not just logging)
3. ✅ Implements gradual rollout (not 100% immediately)
4. ✅ Versions models with unique IDs
5. ✅ Implements drift detection (KS test or PSI)
6. ✅ Creates integration tests (not just unit tests)
7. ✅ Implements automatic rollback
8. ✅ Creates alerting system

**Skill fails if agent**:
- ❌ Skips smoke tests ("model trained, just deploy")
- ❌ Uses print statements for monitoring
- ❌ Deploys to 100% immediately
- ❌ Saves model as `model.pkl` without versioning
- ❌ Only tracks prediction accuracy (no feature drift)
- ❌ Only creates unit tests

---

## Scenario Selection for Baseline

**Start with Scenario 1 (Smoke Tests)**:
- Most fundamental requirement
- Easy to test in 15-20 minutes
- Clear pass/fail criteria
- Representative of "deployment validation" pattern

**If baseline shows gap**: Write skill focusing on smoke tests, then expand to other scenarios.

**If baseline shows NO gap**: Test Scenario 2 (Drift Detection) to find different gap.

---

## Notes

- All scenarios should be testable in 15-20 minutes
- Agent should produce working code (not pseudocode)
- Demonstrations should show actual execution
- Tests should fail appropriately (not just pass)

**Goal**: Transform agent behavior from "train model → deploy" to "train model → validate → deploy gradually → monitor → maintain"
