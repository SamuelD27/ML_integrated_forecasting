---
name: model-deployment-monitoring
description: Use when deploying ML models to production - enforces smoke tests before deployment, version tracking, output validation, latency monitoring, and gradual rollout with automatic rollback, preventing "backtest good → deploy immediately" failures
---

# Model Deployment & Monitoring

## Overview

**Model deployment is validation + monitoring + safeguards.** Every model must pass smoke tests before deployment, track versions, validate outputs, monitor performance, and support gradual rollout with automatic rollback. Training success ≠ production readiness.

**Core principle:** Backtest performance ≠ deployment validation. Always run smoke tests, track versions, and deploy gradually.

## When to Use

Use this skill when:
- Deploying ML models to production
- Creating model deployment scripts
- Implementing model monitoring systems
- Setting up A/B testing infrastructure
- Building rollback mechanisms
- Tracking model versions and artifacts

**Don't skip validation because:**
- "Backtest was good" (backtest ≠ production)
- "Model loads successfully" (loading is ONE check, not sufficient)
- "Logging = monitoring" (logs ≠ structured metrics + alerts)
- "Deploy to 100% immediately" (gradual rollout catches issues early)
- "Clean code = production ready" (code quality ≠ deployment validation)
- "Model worked in notebook" (notebook ≠ production environment)

## Implementation Checklist

Before deploying ANY ML model:

- [ ] Smoke tests passed (model loads, inference works, outputs valid, latency OK)
- [ ] Version tracked (timestamp + hash, deployment metadata logged)
- [ ] Output validated (range checks, NaN/Inf detection, shape validation)
- [ ] Latency measured (p50, p95, p99 tracked)
- [ ] Error scenarios tested (missing file, wrong features, NaN inputs)
- [ ] Deployment checklist enforced (ALL checks must pass)
- [ ] Gradual rollout plan (5% → 25% → 50% → 100%)
- [ ] Rollback strategy (automatic if performance degrades)
- [ ] Monitoring configured (drift detection, performance tracking, alerting)

---

## Pattern 1: Mandatory Smoke Test Suite

### Run 5+ Smoke Tests BEFORE Deployment

**CRITICAL: Never deploy without smoke tests. Backtest performance doesn't validate deployment.**

```python
import time
import numpy as np
import pandas as pd
from typing import Any


def run_smoke_tests(deployment: Any) -> bool:
    """
    Run mandatory smoke tests before deployment.

    Args:
        deployment: Model deployment object

    Returns:
        True if all tests pass

    Raises:
        AssertionError: If any smoke test fails
    """
    print("\n" + "="*60)
    print("RUNNING SMOKE TESTS (MANDATORY)")
    print("="*60)

    # Test 1: Model loads without errors
    print("\n[1/5] Testing model load...")
    try:
        deployment.load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        raise AssertionError(f"SMOKE TEST FAILED: Model load - {e}")

    # Test 2: Inference works
    print("\n[2/5] Testing inference...")
    test_features = {
        'returns_1d': 0.01,
        'returns_5d': 0.02,
        'returns_20d': 0.05,
        'volume_ratio': 1.1,
        'volume_change': 0.05,
        'volatility_20d': 0.02,
        'volatility_60d': 0.018,
        'rsi': 55.0,
        'macd': 0.005,
        'bollinger_position': 0.6
    }

    try:
        prob = deployment.predict_single(test_features)
        print(f"✓ Inference successful: {prob:.4f}")
    except Exception as e:
        raise AssertionError(f"SMOKE TEST FAILED: Inference - {e}")

    # Test 3: Output is valid (range, type, no NaN/Inf)
    print("\n[3/5] Testing output validity...")

    # Check type
    assert isinstance(prob, (float, np.floating)), \
        f"Expected float, got {type(prob)}"

    # Check range [0, 1] for probability
    assert 0.0 <= prob <= 1.0, \
        f"Probability {prob} not in [0, 1]"

    # Check not NaN or Inf
    assert not np.isnan(prob), \
        "Prediction is NaN"
    assert not np.isinf(prob), \
        "Prediction is Inf"

    print(f"✓ Output valid: type=float, range=[0,1], no NaN/Inf")

    # Test 4: Batch prediction works
    print("\n[4/5] Testing batch prediction...")
    batch_features = [test_features, test_features]

    try:
        batch_probs = deployment.batch_predict(batch_features)
        assert len(batch_probs) == 2, \
            f"Expected 2 predictions, got {len(batch_probs)}"
        assert np.all((batch_probs >= 0.0) & (batch_probs <= 1.0)), \
            f"Batch predictions out of range: {batch_probs}"
        print(f"✓ Batch prediction successful: {batch_probs}")
    except Exception as e:
        raise AssertionError(f"SMOKE TEST FAILED: Batch prediction - {e}")

    # Test 5: Inference latency is acceptable
    print("\n[5/5] Testing inference latency...")
    n_iterations = 100
    max_latency_ms = 50.0  # Threshold

    start = time.perf_counter()
    for _ in range(n_iterations):
        deployment.predict_single(test_features)
    avg_latency_ms = (time.perf_counter() - start) * 1000 / n_iterations

    assert avg_latency_ms < max_latency_ms, \
        f"Latency {avg_latency_ms:.2f}ms exceeds {max_latency_ms}ms"
    print(f"✓ Latency: {avg_latency_ms:.2f}ms (< {max_latency_ms}ms)")

    print("\n" + "="*60)
    print("ALL SMOKE TESTS PASSED ✓")
    print("="*60 + "\n")

    return True


# ENFORCE: Run smoke tests before deployment
deployment = TradingModelDeployment('trained_model.pkl')
run_smoke_tests(deployment)  # Raises if any test fails

# Only deploy if smoke tests pass
logger.info("✓ Smoke tests passed - APPROVED FOR DEPLOYMENT")
```

**Why This Pattern**:
- ✅ Catches broken models before production
- ✅ Validates full inference pipeline (not just loading)
- ✅ Tests output validity (range, NaN, Inf)
- ✅ Measures latency (not just functionality)
- ✅ Fails loudly (raises AssertionError)

---

## Pattern 2: Automatic Version Tracking

### Every Model Gets Unique Version ID

**Don't save as `model.pkl` — track version (timestamp + hash).**

```python
import hashlib
from datetime import datetime
from typing import Dict


class VersionTrackedDeployment:
    """Model deployment with automatic version tracking."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

        # Version tracking
        self.model_version: str = None
        self.deployment_timestamp: str = None
        self.model_hash: str = None
        self.training_metadata: Dict = {}

    def load_model(self) -> None:
        """Load model and automatically track version."""
        logger.info(f"Loading model from {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_bytes = f.read()
            self.model = pickle.loads(model_bytes)

            # Calculate hash for version tracking
            self.model_hash = hashlib.sha256(model_bytes).hexdigest()[:8]

        # Timestamp deployment
        self.deployment_timestamp = datetime.now().isoformat()

        # Create version ID
        self.model_version = f"v_{self.deployment_timestamp}_{self.model_hash}"

        logger.info(f"✓ Model loaded: version {self.model_version}")

    def get_model_metadata(self) -> Dict:
        """
        Get complete deployment metadata.

        Returns:
            Dict with version, timestamp, hash, path
        """
        return {
            'model_version': self.model_version,
            'deployment_timestamp': self.deployment_timestamp,
            'model_hash': self.model_hash,
            'model_path': self.model_path,
            'training_metadata': self.training_metadata
        }

    def log_deployment(self) -> None:
        """Log deployment to registry."""
        metadata = self.get_model_metadata()

        # Log to file
        with open('model_deployments.jsonl', 'a') as f:
            import json
            f.write(json.dumps(metadata) + '\n')

        logger.info(f"✓ Deployment logged: {self.model_version}")


# Usage:
deployment = VersionTrackedDeployment('trained_model.pkl')
deployment.load_model()
deployment.log_deployment()

# Know exactly what's deployed
print(f"Deployed version: {deployment.model_version}")
print(f"Model hash: {deployment.model_hash}")
print(f"Timestamp: {deployment.deployment_timestamp}")
```

**Why This Pattern**:
- ✅ Every model has unique ID
- ✅ Can reproduce exact model from version ID
- ✅ Know which version is in production
- ✅ Easy rollback (load previous version)
- ✅ Audit trail of all deployments

---

## Pattern 3: Output Validation Assertions

### Assert Output Validity (Don't Just Log)

**Validate predictions are in valid range, no NaN/Inf.**

```python
import numpy as np


class ValidatedDeployment:
    """Model deployment with output validation."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with output validation.

        Args:
            features: Input features

        Returns:
            Validated predictions

        Raises:
            AssertionError: If predictions invalid
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Make prediction
        probabilities = self.model.predict_proba(features)[:, 1]

        # VALIDATE: Output shape
        expected_shape = (len(features),)
        assert probabilities.shape == expected_shape, \
            f"Output shape {probabilities.shape} != expected {expected_shape}"

        # VALIDATE: Output range [0, 1]
        out_of_range = ~((probabilities >= 0.0) & (probabilities <= 1.0))
        assert not np.any(out_of_range), \
            f"Predictions out of [0,1]: {probabilities[out_of_range]}"

        # VALIDATE: No NaN
        assert not np.any(np.isnan(probabilities)), \
            f"NaN predictions detected at indices: {np.where(np.isnan(probabilities))[0]}"

        # VALIDATE: No Inf
        assert not np.any(np.isinf(probabilities)), \
            f"Inf predictions detected at indices: {np.where(np.isinf(probabilities))[0]}"

        logger.debug(f"✓ Generated {len(probabilities)} valid predictions")
        return probabilities


# ENFORCE: All predictions validated
deployment = ValidatedDeployment()
probabilities = deployment.predict(features)  # Raises if invalid
```

**Why This Pattern**:
- ✅ Catches invalid predictions immediately
- ✅ Prevents NaN/Inf from reaching production
- ✅ Validates shape (catches feature engineering bugs)
- ✅ Fails loudly (raises, doesn't just log warning)

---

## Pattern 4: Built-in Latency Tracking

### Measure Latency on Every Inference

**Track p50, p95, p99 latency (not just external benchmarks).**

```python
import time
from collections import deque


class LatencyTrackedDeployment:
    """Model deployment with latency tracking."""

    def __init__(
        self,
        model_path: str,
        max_latency_ms: float = 50.0,
        latency_history_size: int = 1000
    ):
        self.model_path = model_path
        self.model = None

        # Latency tracking
        self.max_latency_ms = max_latency_ms
        self.latency_history = deque(maxlen=latency_history_size)
        self.slow_inference_count = 0

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions with latency tracking."""
        # Measure latency
        start = time.perf_counter()

        probabilities = self.model.predict_proba(features)[:, 1]

        latency_ms = (time.perf_counter() - start) * 1000

        # Track latency
        self.latency_history.append(latency_ms)

        # WARN: If latency exceeds threshold
        if latency_ms > self.max_latency_ms:
            self.slow_inference_count += 1
            logger.warning(
                f"Slow inference: {latency_ms:.2f}ms > {self.max_latency_ms}ms "
                f"(count: {self.slow_inference_count})"
            )

        return probabilities

    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics.

        Returns:
            Dict with mean, p50, p95, p99, max latency
        """
        if not self.latency_history:
            return {}

        latencies = np.array(self.latency_history)
        return {
            'mean_ms': float(np.mean(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'max_ms': float(np.max(latencies)),
            'slow_count': self.slow_inference_count
        }

    def assert_latency_acceptable(self) -> None:
        """
        Assert latency is acceptable.

        Raises:
            AssertionError: If p99 latency exceeds threshold
        """
        stats = self.get_latency_stats()
        p99_latency = stats.get('p99_ms', 0)

        assert p99_latency < self.max_latency_ms, \
            f"P99 latency {p99_latency:.2f}ms exceeds {self.max_latency_ms}ms"

        logger.info(f"✓ Latency acceptable: P99={p99_latency:.2f}ms")


# Usage:
deployment = LatencyTrackedDeployment('trained_model.pkl', max_latency_ms=50.0)
deployment.load_model()

# Make predictions (latency tracked automatically)
for features in feature_stream:
    probs = deployment.predict(features)

# Validate latency
print(deployment.get_latency_stats())
deployment.assert_latency_acceptable()  # Raises if too slow
```

**Why This Pattern**:
- ✅ Latency measured on every call
- ✅ Historical tracking (not just real-time)
- ✅ Detects performance degradation
- ✅ Can assert latency requirements

---

## Pattern 5: Error Scenario Testing

### Test Failure Modes Before Deployment

**Test missing files, wrong features, NaN inputs, etc.**

```python
def test_error_scenarios(DeploymentClass: type) -> bool:
    """
    Test error scenarios before deployment.

    Args:
        DeploymentClass: Deployment class to test

    Returns:
        True if all error scenarios handled correctly

    Raises:
        AssertionError: If error handling insufficient
    """
    print("\n" + "="*60)
    print("TESTING ERROR SCENARIOS")
    print("="*60)

    # Test 1: Missing model file
    print("\n[1/5] Testing missing model file...")
    try:
        deployment = DeploymentClass('nonexistent_model.pkl')
        deployment.load_model()
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✓ Correctly handles missing model file")

    # Test 2: Wrong feature names
    print("\n[2/5] Testing wrong feature names...")
    deployment = DeploymentClass('trained_model.pkl')
    deployment.load_model()

    try:
        wrong_features = {'wrong_feature_name': 1.0}
        deployment.predict_single(wrong_features)
        raise AssertionError("Should have raised KeyError")
    except KeyError:
        print("✓ Correctly handles wrong feature names")

    # Test 3: Missing features
    print("\n[3/5] Testing missing features...")
    try:
        incomplete_features = {'returns_1d': 0.01}  # Only 1 of 10
        deployment.predict_single(incomplete_features)
        raise AssertionError("Should have raised KeyError")
    except KeyError:
        print("✓ Correctly handles missing features")

    # Test 4: NaN inputs
    print("\n[4/5] Testing NaN inputs...")
    try:
        nan_features = {
            'returns_1d': np.nan,
            'returns_5d': 0.02,
            # ... rest of features
        }
        deployment.predict_single(nan_features)
        # Should either handle gracefully or raise clear error
        print("✓ Handles NaN inputs (either imputes or raises clear error)")
    except (ValueError, AssertionError) as e:
        print(f"✓ Raises clear error on NaN: {type(e).__name__}")

    # Test 5: Prediction before loading model
    print("\n[5/5] Testing prediction before model loaded...")
    unloaded_deployment = DeploymentClass('trained_model.pkl')
    try:
        unloaded_deployment.predict_single({'feature': 1.0})
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        if "not loaded" in str(e).lower():
            print("✓ Raises clear error when model not loaded")
        else:
            raise

    print("\n" + "="*60)
    print("ALL ERROR SCENARIO TESTS PASSED ✓")
    print("="*60 + "\n")

    return True


# ENFORCE: Test error scenarios before deployment
test_error_scenarios(TradingModelDeployment)
```

**Why This Pattern**:
- ✅ Validates error handling works
- ✅ Catches edge cases before production
- ✅ Ensures clear error messages
- ✅ Tests failure modes systematically

---

## Pattern 6: Deployment Checklist Enforcement

### All Checks Must Pass Before Deployment

**Enforce systematic validation (don't just suggest).**

```python
def deployment_checklist(deployment: Any) -> bool:
    """
    Run complete deployment checklist.

    Args:
        deployment: Model deployment object

    Returns:
        True if all checks pass

    Raises:
        AssertionError: If any check fails
    """
    print("\n" + "="*60)
    print("DEPLOYMENT CHECKLIST")
    print("="*60)

    checks = []

    # Check 1: Model version tracked
    print("\n[1/6] Verifying model version tracked...")
    version_tracked = hasattr(deployment, 'model_version') and deployment.model_version is not None
    checks.append(("Model version tracked", version_tracked))
    print(f"  {'✓' if version_tracked else '✗'} Version: {deployment.model_version if version_tracked else 'MISSING'}")

    # Check 2: Smoke tests passed
    print("\n[2/6] Running smoke tests...")
    try:
        run_smoke_tests(deployment)
        checks.append(("Smoke tests passed", True))
    except AssertionError as e:
        checks.append(("Smoke tests passed", False))
        print(f"  ✗ Smoke tests failed: {e}")

    # Check 3: Error scenarios tested
    print("\n[3/6] Testing error scenarios...")
    try:
        test_error_scenarios(type(deployment))
        checks.append(("Error scenarios tested", True))
    except AssertionError as e:
        checks.append(("Error scenarios tested", False))
        print(f"  ✗ Error scenarios failed: {e}")

    # Check 4: Latency acceptable
    print("\n[4/6] Verifying latency acceptable...")
    if hasattr(deployment, 'get_latency_stats'):
        stats = deployment.get_latency_stats()
        p99 = stats.get('p99_ms', 0)
        latency_ok = p99 < deployment.max_latency_ms if hasattr(deployment, 'max_latency_ms') else True
        checks.append(("Latency < threshold", latency_ok))
        print(f"  {'✓' if latency_ok else '✗'} P99: {p99:.2f}ms")
    else:
        checks.append(("Latency < threshold", False))
        print("  ✗ No latency tracking")

    # Check 5: Output validation enabled
    print("\n[5/6] Verifying output validation...")
    has_validation = 'assert' in str(deployment.predict.__code__.co_consts)
    checks.append(("Output validation enabled", has_validation))
    print(f"  {'✓' if has_validation else '✗'} Output validation")

    # Check 6: Deployment metadata logged
    print("\n[6/6] Verifying deployment logged...")
    has_metadata = hasattr(deployment, 'get_model_metadata')
    checks.append(("Deployment metadata logged", has_metadata))
    print(f"  {'✓' if has_metadata else '✗'} Metadata logging")

    # Print summary
    print("\n" + "="*60)
    print("CHECKLIST SUMMARY")
    print("="*60)

    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")

    # All must pass
    all_passed = all(passed for _, passed in checks)
    failed_count = sum(1 for _, passed in checks if not passed)

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - APPROVED FOR DEPLOYMENT")
    else:
        print(f"✗ {failed_count} CHECK(S) FAILED - DEPLOYMENT BLOCKED")
        print("Fix issues above before deploying")
    print("="*60 + "\n")

    if not all_passed:
        raise AssertionError(f"Deployment checklist failed: {failed_count} checks")

    return True


# ENFORCE: Run checklist before deployment
deployment = TradingModelDeployment('trained_model.pkl')
deployment_checklist(deployment)  # Raises if any check fails

# Only deploy if ALL checks pass
logger.info("✓ Deployment checklist passed - DEPLOYING TO PRODUCTION")
```

**Why This Pattern**:
- ✅ Systematic validation (not ad-hoc)
- ✅ All checks enforced (not optional)
- ✅ Clear pass/fail (not subjective)
- ✅ Blocks deployment if fails

---

## Pattern 7: Gradual Rollout with Automatic Rollback

### Deploy Gradually (5% → 25% → 50% → 100%)

**Don't deploy to 100% immediately. Monitor and rollback if needed.**

```python
import random
from typing import Optional


class GradualRolloutDeployment:
    """
    Model deployment with gradual rollout and automatic rollback.

    Rollout stages: 5% → 25% → 50% → 100%
    """

    def __init__(
        self,
        new_model_path: str,
        baseline_model_path: str,
        rollout_percentage: float = 5.0,
        performance_threshold: float = 0.95  # New model must be >= 95% of baseline
    ):
        self.new_model = self._load_model(new_model_path)
        self.baseline_model = self._load_model(baseline_model_path)

        # Rollout config
        self.rollout_percentage = rollout_percentage
        self.performance_threshold = performance_threshold

        # Metrics
        self.new_model_predictions = []
        self.baseline_predictions = []
        self.new_model_errors = 0
        self.baseline_errors = 0

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict using gradual rollout.

        Routes traffic to new model based on rollout percentage.
        Tracks performance for comparison.
        """
        # Determine which model to use
        use_new_model = random.random() < (self.rollout_percentage / 100.0)

        try:
            if use_new_model:
                predictions = self.new_model.predict_proba(features)[:, 1]
                self.new_model_predictions.extend(predictions)
                return predictions
            else:
                predictions = self.baseline_model.predict_proba(features)[:, 1]
                self.baseline_predictions.extend(predictions)
                return predictions

        except Exception as e:
            # Track errors
            if use_new_model:
                self.new_model_errors += 1
                logger.error(f"New model error: {e}")
                # Fallback to baseline
                return self.baseline_model.predict_proba(features)[:, 1]
            else:
                self.baseline_errors += 1
                logger.error(f"Baseline model error: {e}")
                raise

    def evaluate_rollout(self) -> Dict[str, float]:
        """
        Evaluate rollout performance.

        Returns:
            Dict with performance metrics
        """
        # Calculate metrics
        new_model_mean = np.mean(self.new_model_predictions) if self.new_model_predictions else 0
        baseline_mean = np.mean(self.baseline_predictions) if self.baseline_predictions else 0

        relative_performance = new_model_mean / baseline_mean if baseline_mean > 0 else 0

        return {
            'rollout_percentage': self.rollout_percentage,
            'new_model_mean': new_model_mean,
            'baseline_mean': baseline_mean,
            'relative_performance': relative_performance,
            'new_model_errors': self.new_model_errors,
            'baseline_errors': self.baseline_errors,
            'new_model_samples': len(self.new_model_predictions),
            'baseline_samples': len(self.baseline_predictions)
        }

    def should_rollback(self) -> bool:
        """
        Determine if rollout should be rolled back.

        Returns:
            True if new model underperforms
        """
        metrics = self.evaluate_rollout()

        # Check: Performance degradation
        if metrics['relative_performance'] < self.performance_threshold:
            logger.warning(
                f"Performance below threshold: "
                f"{metrics['relative_performance']:.2%} < {self.performance_threshold:.2%}"
            )
            return True

        # Check: Error rate
        new_error_rate = metrics['new_model_errors'] / max(metrics['new_model_samples'], 1)
        baseline_error_rate = metrics['baseline_errors'] / max(metrics['baseline_samples'], 1)

        if new_error_rate > 2 * baseline_error_rate:  # 2x more errors
            logger.warning(
                f"Error rate too high: new={new_error_rate:.2%}, baseline={baseline_error_rate:.2%}"
            )
            return True

        return False

    def advance_rollout(self) -> bool:
        """
        Advance to next rollout stage if performance acceptable.

        Returns:
            True if advanced, False if rolled back
        """
        metrics = self.evaluate_rollout()

        print("\n" + "="*60)
        print(f"ROLLOUT EVALUATION ({self.rollout_percentage}%)")
        print("="*60)
        print(f"New model mean: {metrics['new_model_mean']:.4f}")
        print(f"Baseline mean: {metrics['baseline_mean']:.4f}")
        print(f"Relative performance: {metrics['relative_performance']:.2%}")
        print(f"New model errors: {metrics['new_model_errors']}")
        print(f"Baseline errors: {metrics['baseline_errors']}")

        # Check if should rollback
        if self.should_rollback():
            print("\n✗ ROLLING BACK - Performance below threshold")
            print("="*60 + "\n")
            self.rollout_percentage = 0.0
            return False

        # Advance to next stage
        if self.rollout_percentage < 100:
            rollout_stages = [5, 25, 50, 100]
            current_idx = rollout_stages.index(self.rollout_percentage) if self.rollout_percentage in rollout_stages else 0
            next_idx = min(current_idx + 1, len(rollout_stages) - 1)
            self.rollout_percentage = rollout_stages[next_idx]

            print(f"\n✓ ADVANCING ROLLOUT to {self.rollout_percentage}%")
            print("="*60 + "\n")
        else:
            print("\n✓ ROLLOUT COMPLETE (100%)")
            print("="*60 + "\n")

        return True


# Usage:
rollout = GradualRolloutDeployment(
    new_model_path='new_model.pkl',
    baseline_model_path='baseline_model.pkl',
    rollout_percentage=5.0,  # Start at 5%
    performance_threshold=0.95  # New model must be >= 95% of baseline
)

# Process traffic
for features in feature_stream:
    predictions = rollout.predict(features)

# Evaluate and advance/rollback
if rollout.should_rollback():
    logger.error("Rollout failed - rolling back to baseline")
else:
    rollout.advance_rollout()  # Move to 25%
```

**Why This Pattern**:
- ✅ Catches issues early (5% exposure)
- ✅ Automatic rollback if performance degrades
- ✅ Gradual increase (5% → 25% → 50% → 100%)
- ✅ Compares to baseline (not just absolute performance)

---

## Rationalization Table

| Excuse | Reality | Fix |
|--------|---------|-----|
| "Backtest was good" | Backtest ≠ production | Run smoke tests before deployment |
| "Model loads successfully" | Loading is ONE check | Run 5+ smoke tests (load, inference, output, batch, latency) |
| "Logging = monitoring" | Logs ≠ structured metrics | Build metrics + alerting system |
| "Deploy to 100% immediately" | Catches issues too late | Gradual rollout (5% → 25% → 50% → 100%) |
| "Clean code = production ready" | Code quality ≠ validation | Enforce deployment checklist |
| "Save as model.pkl" | No version tracking | Version ID (timestamp + hash) |

---

## Real-World Impact

**Proper deployment prevents:**
- Models with NaN outputs reaching production
- Slow models (>100ms) degrading user experience
- Version confusion ("which model is deployed?")
- Silent performance degradation
- Unrecoverable failures (no rollback)
- Feature engineering bugs (wrong feature names)

**Time investment:**
- Add smoke tests: 10 minutes
- Add version tracking: 5 minutes
- Add output validation: 5 minutes
- Add latency tracking: 5 minutes
- Test error scenarios: 10 minutes
- Build deployment checklist: 10 minutes
- **Total: 45 minutes**

**Cost of skipping validation:**
- Debug production NaN issue: 4+ hours
- Track down "which model version": 2+ hours
- Manual rollback without version: 1+ hour
- Reproduce training without version: Impossible

---

## Bottom Line

**Model deployment requires three components:**
1. Validation (smoke tests, output checks, error scenarios) ← AGENTS SKIP
2. Monitoring (latency, performance, drift) ← AGENTS SKIP
3. Safeguards (gradual rollout, rollback, fallbacks) ← AGENTS SKIP

Don't just load and predict—validate, version, and deploy gradually.

Every model gets smoke tests. Every deployment gets version tracking. Every rollout gets monitoring. No exceptions.
