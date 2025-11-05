# Deployment Patterns Reference - Model Deployment & Monitoring

## Table of Contents

1. [Complete Production Deployment Class](#complete-production-deployment-class)
2. [Drift Detection System](#drift-detection-system)
3. [A/B Testing Framework](#ab-testing-framework)
4. [Monitoring Dashboard](#monitoring-dashboard)
5. [Model Registry](#model-registry)
6. [Integration Testing](#integration-testing)

---

## Complete Production Deployment Class

### Full-Featured Model Deployment

```python
import pickle
import hashlib
import time
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


class ProductionModelDeployment:
    """
    Production-ready model deployment with all safeguards.

    Features:
    - Automatic version tracking
    - Smoke tests before deployment
    - Output validation
    - Latency tracking
    - Error scenario handling
    - Deployment checklist
    """

    def __init__(
        self,
        model_path: str,
        max_latency_ms: float = 50.0,
        latency_history_size: int = 1000,
        enable_output_validation: bool = True
    ):
        """
        Initialize production deployment.

        Args:
            model_path: Path to trained model
            max_latency_ms: Maximum acceptable latency
            latency_history_size: Size of latency history buffer
            enable_output_validation: Enable output validation
        """
        self.model_path = model_path
        self.model = None

        # Version tracking
        self.model_version: Optional[str] = None
        self.deployment_timestamp: Optional[str] = None
        self.model_hash: Optional[str] = None

        # Latency tracking
        self.max_latency_ms = max_latency_ms
        self.latency_history = deque(maxlen=latency_history_size)
        self.slow_inference_count = 0

        # Validation
        self.enable_output_validation = enable_output_validation

        # Feature names (trading model example)
        self.feature_names = [
            'returns_1d', 'returns_5d', 'returns_20d',
            'volume_ratio', 'volume_change',
            'volatility_20d', 'volatility_60d',
            'rsi', 'macd', 'bollinger_position'
        ]

        # Deployment metadata
        self.deployment_metadata: Dict[str, Any] = {}

    def load_model(self) -> None:
        """Load model and track version."""
        logger.info(f"Loading model from {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_bytes = f.read()
            self.model = pickle.loads(model_bytes)

            # Calculate hash
            self.model_hash = hashlib.sha256(model_bytes).hexdigest()[:8]

        # Timestamp deployment
        self.deployment_timestamp = datetime.now().isoformat()

        # Create version ID
        self.model_version = f"v_{self.deployment_timestamp}_{self.model_hash}"

        logger.info(f"✓ Model loaded: version {self.model_version}")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with validation and tracking.

        Args:
            features: Input features DataFrame

        Returns:
            Validated predictions

        Raises:
            ValueError: If model not loaded
            AssertionError: If output validation fails
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Measure latency
        start = time.perf_counter()

        # Make prediction
        probabilities = self.model.predict_proba(features)[:, 1]

        # Track latency
        latency_ms = (time.perf_counter() - start) * 1000
        self.latency_history.append(latency_ms)

        # Warn if slow
        if latency_ms > self.max_latency_ms:
            self.slow_inference_count += 1
            logger.warning(
                f"Slow inference: {latency_ms:.2f}ms > {self.max_latency_ms}ms "
                f"(count: {self.slow_inference_count})"
            )

        # Validate output
        if self.enable_output_validation:
            self._validate_output(probabilities, len(features))

        return probabilities

    def _validate_output(self, predictions: np.ndarray, expected_length: int) -> None:
        """
        Validate model output.

        Args:
            predictions: Model predictions
            expected_length: Expected number of predictions

        Raises:
            AssertionError: If validation fails
        """
        # Check shape
        assert len(predictions) == expected_length, \
            f"Output length {len(predictions)} != expected {expected_length}"

        # Check range [0, 1]
        out_of_range = ~((predictions >= 0.0) & (predictions <= 1.0))
        assert not np.any(out_of_range), \
            f"Predictions out of [0,1]: {predictions[out_of_range]}"

        # Check NaN
        assert not np.any(np.isnan(predictions)), \
            f"NaN at indices: {np.where(np.isnan(predictions))[0]}"

        # Check Inf
        assert not np.any(np.isinf(predictions)), \
            f"Inf at indices: {np.where(np.isinf(predictions))[0]}"

    def predict_single(self, feature_dict: Dict[str, float]) -> float:
        """
        Make prediction for single data point.

        Args:
            feature_dict: Dictionary of features

        Returns:
            Prediction probability
        """
        # Convert to DataFrame
        features_df = pd.DataFrame([feature_dict])

        # Ensure correct feature order
        features_df = features_df[self.feature_names]

        # Predict
        prob = self.predict(features_df)[0]

        return prob

    def batch_predict(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Make predictions for multiple data points.

        Args:
            features_list: List of feature dictionaries

        Returns:
            Array of predictions
        """
        features_df = pd.DataFrame(features_list)
        features_df = features_df[self.feature_names]
        return self.predict(features_df)

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
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

    def get_model_metadata(self) -> Dict[str, Any]:
        """Get deployment metadata."""
        return {
            'model_version': self.model_version,
            'deployment_timestamp': self.deployment_timestamp,
            'model_hash': self.model_hash,
            'model_path': self.model_path,
            'latency_stats': self.get_latency_stats(),
            'deployment_metadata': self.deployment_metadata
        }

    def run_smoke_tests(self) -> bool:
        """
        Run smoke tests before deployment.

        Returns:
            True if all tests pass

        Raises:
            AssertionError: If any test fails
        """
        print("\n" + "="*60)
        print("RUNNING SMOKE TESTS")
        print("="*60)

        # Test 1: Model loads
        print("\n[1/5] Testing model load...")
        try:
            if self.model is None:
                self.load_model()
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
            prob = self.predict_single(test_features)
            print(f"✓ Inference successful: {prob:.4f}")
        except Exception as e:
            raise AssertionError(f"SMOKE TEST FAILED: Inference - {e}")

        # Test 3: Output valid
        print("\n[3/5] Testing output validity...")
        assert 0.0 <= prob <= 1.0, f"Probability {prob} not in [0, 1]"
        assert not np.isnan(prob), "Prediction is NaN"
        assert not np.isinf(prob), "Prediction is Inf"
        print("✓ Output valid: range=[0,1], no NaN/Inf")

        # Test 4: Batch prediction
        print("\n[4/5] Testing batch prediction...")
        try:
            batch_probs = self.batch_predict([test_features, test_features])
            assert len(batch_probs) == 2
            assert np.all((batch_probs >= 0.0) & (batch_probs <= 1.0))
            print(f"✓ Batch prediction successful: {batch_probs}")
        except Exception as e:
            raise AssertionError(f"SMOKE TEST FAILED: Batch - {e}")

        # Test 5: Latency
        print("\n[5/5] Testing latency...")
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            self.predict_single(test_features)
        avg_latency_ms = (time.perf_counter() - start) * 1000 / n_iterations

        assert avg_latency_ms < self.max_latency_ms, \
            f"Latency {avg_latency_ms:.2f}ms > {self.max_latency_ms}ms"
        print(f"✓ Latency: {avg_latency_ms:.2f}ms (< {self.max_latency_ms}ms)")

        print("\n" + "="*60)
        print("ALL SMOKE TESTS PASSED ✓")
        print("="*60 + "\n")

        return True

    def deployment_checklist(self) -> bool:
        """
        Run deployment checklist.

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
        version_ok = self.model_version is not None
        checks.append(("Model version tracked", version_ok))
        print(f"\n[1/4] {'✓' if version_ok else '✗'} Model version: {self.model_version}")

        # Check 2: Smoke tests
        print("\n[2/4] Running smoke tests...")
        try:
            self.run_smoke_tests()
            checks.append(("Smoke tests passed", True))
        except AssertionError as e:
            checks.append(("Smoke tests passed", False))
            print(f"✗ Failed: {e}")

        # Check 3: Latency acceptable
        print("\n[3/4] Checking latency...")
        stats = self.get_latency_stats()
        p99 = stats.get('p99_ms', 0)
        latency_ok = p99 < self.max_latency_ms
        checks.append(("Latency acceptable", latency_ok))
        print(f"{'✓' if latency_ok else '✗'} P99: {p99:.2f}ms")

        # Check 4: Output validation enabled
        checks.append(("Output validation enabled", self.enable_output_validation))
        print(f"\n[4/4] {'✓' if self.enable_output_validation else '✗'} Output validation")

        # Summary
        print("\n" + "="*60)
        for name, passed in checks:
            print(f"{'✓' if passed else '✗'} {name}")

        all_passed = all(passed for _, passed in checks)
        print("="*60)
        if all_passed:
            print("✓ ALL CHECKS PASSED - APPROVED FOR DEPLOYMENT")
        else:
            print("✗ DEPLOYMENT BLOCKED - FIX ISSUES ABOVE")
        print("="*60 + "\n")

        if not all_passed:
            raise AssertionError("Deployment checklist failed")

        return True


# Example usage
if __name__ == "__main__":
    # Create deployment
    deployment = ProductionModelDeployment(
        model_path='trained_model.pkl',
        max_latency_ms=50.0,
        enable_output_validation=True
    )

    # Load model
    deployment.load_model()

    # Run deployment checklist
    deployment.deployment_checklist()

    # Make predictions
    features = {
        'returns_1d': 0.015,
        'returns_5d': 0.032,
        'returns_20d': 0.085,
        'volume_ratio': 1.25,
        'volume_change': 0.18,
        'volatility_20d': 0.022,
        'volatility_60d': 0.019,
        'rsi': 62.5,
        'macd': 0.008,
        'bollinger_position': 0.65
    }

    prob = deployment.predict_single(features)
    print(f"Prediction: {prob:.4f}")

    # Get metadata
    print(f"\nDeployment metadata:")
    import json
    print(json.dumps(deployment.get_model_metadata(), indent=2))
```

---

## Drift Detection System

### Feature and Prediction Drift Detection

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from collections import deque


class DriftDetector:
    """
    Detect feature and prediction drift.

    Methods:
    - KS test for continuous features
    - Chi-square test for categorical features
    - PSI (Population Stability Index) for overall drift
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_threshold: float = 0.1,
        window_size: int = 1000
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Training data distribution
            drift_threshold: Threshold for drift alert
            window_size: Size of sliding window for current data
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.window_size = window_size

        # Current data window
        self.current_data = deque(maxlen=window_size)

        # Drift history
        self.drift_history: List[Dict] = []

    def add_sample(self, sample: Dict[str, float]) -> None:
        """Add sample to current data window."""
        self.current_data.append(sample)

    def kolmogorov_smirnov_test(
        self,
        feature_name: str
    ) -> Tuple[float, float]:
        """
        KS test for feature drift.

        Args:
            feature_name: Name of feature to test

        Returns:
            (ks_statistic, p_value)
        """
        if len(self.current_data) < 30:  # Need minimum samples
            return 0.0, 1.0

        # Extract feature values
        reference_values = self.reference_data[feature_name].values
        current_values = [sample[feature_name] for sample in self.current_data]

        # KS test
        ks_stat, p_value = stats.ks_2samp(reference_values, current_values)

        return ks_stat, p_value

    def population_stability_index(
        self,
        feature_name: str,
        n_bins: int = 10
    ) -> float:
        """
        Calculate PSI (Population Stability Index).

        PSI < 0.1: No significant drift
        PSI 0.1-0.2: Moderate drift
        PSI > 0.2: Significant drift

        Args:
            feature_name: Feature to calculate PSI for
            n_bins: Number of bins for discretization

        Returns:
            PSI value
        """
        if len(self.current_data) < 30:
            return 0.0

        # Extract values
        reference_values = self.reference_data[feature_name].values
        current_values = np.array([sample[feature_name] for sample in self.current_data])

        # Create bins based on reference data
        bins = np.percentile(reference_values, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calculate distributions
        ref_counts, _ = np.histogram(reference_values, bins=bins)
        curr_counts, _ = np.histogram(current_values, bins=bins)

        # Normalize
        ref_dist = ref_counts / len(reference_values)
        curr_dist = curr_counts / len(current_values)

        # Avoid division by zero
        ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
        curr_dist = np.where(curr_dist == 0, 0.0001, curr_dist)

        # Calculate PSI
        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))

        return psi

    def detect_drift(self) -> Dict[str, Any]:
        """
        Detect drift across all features.

        Returns:
            Dict with drift metrics and alerts
        """
        if len(self.current_data) < 30:
            return {'status': 'insufficient_data', 'features': {}}

        drift_results = {}
        alerts = []

        for feature in self.reference_data.columns:
            # KS test
            ks_stat, p_value = self.kolmogorov_smirnov_test(feature)

            # PSI
            psi = self.population_stability_index(feature)

            # Check for drift
            has_drift = (ks_stat > self.drift_threshold) or (psi > 0.1)

            drift_results[feature] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'psi': float(psi),
                'has_drift': has_drift
            }

            if has_drift:
                alerts.append(f"DRIFT ALERT: {feature} (KS={ks_stat:.3f}, PSI={psi:.3f})")

        # Store in history
        self.drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': drift_results,
            'alerts': alerts
        })

        return {
            'status': 'drift_detected' if alerts else 'no_drift',
            'features': drift_results,
            'alerts': alerts
        }

    def print_drift_report(self) -> None:
        """Print drift detection report."""
        results = self.detect_drift()

        print("\n" + "="*60)
        print("DRIFT DETECTION REPORT")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Window size: {len(self.current_data)}")

        print("\nFeature Analysis:")
        for feature, metrics in results['features'].items():
            status = "⚠️  DRIFT" if metrics['has_drift'] else "✓ OK"
            print(f"  {status} {feature:20s} | KS={metrics['ks_statistic']:.3f} PSI={metrics['psi']:.3f}")

        if results['alerts']:
            print("\n🚨 ALERTS:")
            for alert in results['alerts']:
                print(f"  {alert}")

        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Create reference data (training distribution)
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'returns_1d': np.random.normal(0.001, 0.02, 1000),
        'volume_ratio': np.random.lognormal(0, 0.3, 1000),
        'volatility': np.random.gamma(2, 0.01, 1000)
    })

    # Initialize detector
    detector = DriftDetector(
        reference_data=reference_data,
        drift_threshold=0.1,
        window_size=500
    )

    # Simulate production data (with drift)
    for i in range(600):
        # Simulate drift after 300 samples
        if i < 300:
            sample = {
                'returns_1d': np.random.normal(0.001, 0.02),
                'volume_ratio': np.random.lognormal(0, 0.3),
                'volatility': np.random.gamma(2, 0.01)
            }
        else:
            # Drift: Distribution shift
            sample = {
                'returns_1d': np.random.normal(0.005, 0.03),  # Mean shift
                'volume_ratio': np.random.lognormal(0.2, 0.4),  # Scale shift
                'volatility': np.random.gamma(3, 0.015)  # Shape shift
            }

        detector.add_sample(sample)

        # Check drift every 100 samples
        if (i + 1) % 100 == 0:
            detector.print_drift_report()
```

---

## A/B Testing Framework

### Statistical A/B Testing for Model Rollout

```python
from scipy import stats
from typing import Optional


class ABTestFramework:
    """
    A/B testing framework for model deployment.

    Compares new model vs baseline with statistical significance.
    """

    def __init__(
        self,
        new_model,
        baseline_model,
        alpha: float = 0.05,  # Significance level
        min_samples: int = 100  # Minimum samples for test
    ):
        """
        Initialize A/B test.

        Args:
            new_model: New model to test
            baseline_model: Baseline model
            alpha: Significance level (default 0.05)
            min_samples: Minimum samples before test
        """
        self.new_model = new_model
        self.baseline_model = baseline_model
        self.alpha = alpha
        self.min_samples = min_samples

        # Collect results
        self.new_model_results = []
        self.baseline_results = []

    def add_result(self, is_new_model: bool, result: float) -> None:
        """
        Add result from either model.

        Args:
            is_new_model: True if from new model
            result: Result value (e.g., return, accuracy)
        """
        if is_new_model:
            self.new_model_results.append(result)
        else:
            self.baseline_results.append(result)

    def t_test(self) -> Dict[str, float]:
        """
        Perform t-test comparing new vs baseline.

        Returns:
            Dict with test statistics
        """
        if len(self.new_model_results) < self.min_samples or \
           len(self.baseline_results) < self.min_samples:
            return {
                'status': 'insufficient_data',
                'new_samples': len(self.new_model_results),
                'baseline_samples': len(self.baseline_results)
            }

        # T-test
        t_stat, p_value = stats.ttest_ind(
            self.new_model_results,
            self.baseline_results
        )

        # Calculate means
        new_mean = np.mean(self.new_model_results)
        baseline_mean = np.mean(self.baseline_results)

        # Determine winner
        is_significant = p_value < self.alpha
        new_is_better = new_mean > baseline_mean

        if is_significant:
            winner = "new_model" if new_is_better else "baseline"
        else:
            winner = "no_significant_difference"

        return {
            'status': 'test_complete',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'winner': winner,
            'new_mean': float(new_mean),
            'baseline_mean': float(baseline_mean),
            'relative_improvement': float((new_mean - baseline_mean) / baseline_mean * 100),
            'new_samples': len(self.new_model_results),
            'baseline_samples': len(self.baseline_results)
        }

    def print_ab_test_report(self) -> None:
        """Print A/B test report."""
        results = self.t_test()

        print("\n" + "="*60)
        print("A/B TEST REPORT")
        print("="*60)

        if results['status'] == 'insufficient_data':
            print(f"Status: Insufficient data")
            print(f"New model samples: {results['new_samples']} (need {self.min_samples})")
            print(f"Baseline samples: {results['baseline_samples']} (need {self.min_samples})")
        else:
            print(f"New model mean: {results['new_mean']:.6f}")
            print(f"Baseline mean: {results['baseline_mean']:.6f}")
            print(f"Relative improvement: {results['relative_improvement']:.2f}%")
            print(f"\nT-statistic: {results['t_statistic']:.3f}")
            print(f"P-value: {results['p_value']:.6f}")
            print(f"Significance level (alpha): {self.alpha}")
            print(f"Is significant: {results['is_significant']}")
            print(f"\n{'='*60}")
            print(f"WINNER: {results['winner'].upper()}")
            print("="*60)

        print()


# Example usage
if __name__ == "__main__":
    # Simulate A/B test
    np.random.seed(42)

    # New model: mean=0.12, std=0.05 (better)
    # Baseline: mean=0.10, std=0.05
    ab_test = ABTestFramework(
        new_model=None,  # Would be actual models
        baseline_model=None,
        alpha=0.05,
        min_samples=100
    )

    # Simulate results
    for _ in range(150):
        # New model
        new_result = np.random.normal(0.12, 0.05)
        ab_test.add_result(is_new_model=True, result=new_result)

        # Baseline
        baseline_result = np.random.normal(0.10, 0.05)
        ab_test.add_result(is_new_model=False, result=baseline_result)

    # Print report
    ab_test.print_ab_test_report()
```

---

## Summary

This reference provides production-ready implementations for:

1. ✅ **Complete Deployment Class** - All patterns in one class
2. ✅ **Drift Detection** - KS test + PSI for feature drift
3. ✅ **A/B Testing** - Statistical comparison of models
4. ✅ **Monitoring** - Latency, performance, error tracking
5. ✅ **Version Tracking** - Automatic version IDs
6. ✅ **Smoke Tests** - 5-step validation before deployment

**Key Takeaway**: Production ML deployment = Model + Validation + Monitoring + Safeguards

Never deploy without smoke tests, version tracking, and monitoring infrastructure.
