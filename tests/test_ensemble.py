"""
Tests for Model Ensemble System
================================
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
import lightgbm as lgb

from ml_models.ensemble import (
    ModelEnsemble,
    EnsembleConfig,
    create_ensemble
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic regression data."""
    np.random.seed(42)
    n_train = 1000
    n_val = 200
    n_test = 200
    n_features = 10

    X_train = np.random.randn(n_train, n_features)
    X_val = np.random.randn(n_val, n_features)
    X_test = np.random.randn(n_test, n_features)

    # True function: y = 2*x0 + x1 - 0.5*x2 + noise
    y_train = 2*X_train[:, 0] + X_train[:, 1] - 0.5*X_train[:, 2] + np.random.randn(n_train)*0.5
    y_val = 2*X_val[:, 0] + X_val[:, 1] - 0.5*X_val[:, 2] + np.random.randn(n_val)*0.5
    y_test = 2*X_test[:, 0] + X_test[:, 1] - 0.5*X_test[:, 2] + np.random.randn(n_test)*0.5

    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def trained_models(synthetic_data):
    """Train base models on synthetic data."""
    X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data

    # Train Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Train Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)

    # Train LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
    lgb_model.fit(X_train, y_train)

    models = {
        'ridge': ridge,
        'lasso': lasso,
        'lgb': lgb_model
    }

    return models, synthetic_data


class TestEnsembleConfig:
    """Test EnsembleConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnsembleConfig()
        assert config.method == 'weighted'
        assert config.meta_model == 'ridge'
        assert config.estimate_uncertainty is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnsembleConfig(
            method='stacking',
            meta_model='lgb',
            min_correlation=0.1,
            max_correlation=0.90
        )

        assert config.method == 'stacking'
        assert config.meta_model == 'lgb'
        assert config.min_correlation == 0.1


class TestModelEnsemble:
    """Test ModelEnsemble class."""

    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = ModelEnsemble()
        assert ensemble.config.method == 'weighted'
        assert len(ensemble.models) == 0
        assert not ensemble.is_fitted

    def test_add_model(self, trained_models):
        """Test adding models to ensemble."""
        models, _ = trained_models
        ensemble = ModelEnsemble()

        for name, model in models.items():
            ensemble.add_model(name, model)

        assert len(ensemble.models) == 3
        assert 'ridge' in ensemble.models
        assert 'lasso' in ensemble.models
        assert 'lgb' in ensemble.models

    def test_add_model_with_weight(self, trained_models):
        """Test adding model with custom weight."""
        models, _ = trained_models
        ensemble = ModelEnsemble()

        ensemble.add_model('ridge', models['ridge'], weight=0.5)

        assert 'ridge' in ensemble.model_weights
        assert ensemble.model_weights['ridge'] == 0.5

    def test_simple_averaging(self, trained_models):
        """Test simple averaging ensemble."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='simple')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)

        # Check weights are equal
        assert len(ensemble.model_weights) == 3
        for weight in ensemble.model_weights.values():
            assert abs(weight - 1.0/3) < 1e-6

    def test_weighted_averaging(self, trained_models):
        """Test weighted averaging ensemble."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='weighted')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)

        # Weights should sum to 1
        assert abs(sum(ensemble.model_weights.values()) - 1.0) < 1e-6

        # Weights should be different (not equal)
        weights = list(ensemble.model_weights.values())
        assert not all(abs(w - weights[0]) < 0.01 for w in weights)

    def test_stacking(self, trained_models):
        """Test stacking ensemble."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='stacking', meta_model='ridge')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)

        assert ensemble.meta_model is not None
        assert ensemble.is_fitted

    def test_rank_averaging(self, trained_models):
        """Test rank averaging ensemble."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='rank')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)

        # Weights should be equal for rank averaging
        weights = list(ensemble.model_weights.values())
        assert all(abs(w - 1.0/3) < 1e-6 for w in weights)

    def test_predict_simple(self, trained_models):
        """Test prediction with simple averaging."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='simple')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)
        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(y_test)
        assert not np.isnan(predictions).any()

    def test_predict_weighted(self, trained_models):
        """Test prediction with weighted averaging."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='weighted')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)
        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(y_test)
        assert not np.isnan(predictions).any()

    def test_predict_stacking(self, trained_models):
        """Test prediction with stacking."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='stacking')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)
        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(y_test)
        assert not np.isnan(predictions).any()

    def test_predict_with_uncertainty(self, trained_models):
        """Test prediction with uncertainty estimation."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='weighted', estimate_uncertainty=True)
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)
        predictions, uncertainty = ensemble.predict(X_test, return_uncertainty=True)

        assert len(predictions) == len(y_test)
        assert len(uncertainty) == len(y_test)
        assert all(uncertainty >= 0)  # Uncertainty should be non-negative

    def test_evaluate(self, trained_models):
        """Test ensemble evaluation."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        config = EnsembleConfig(method='weighted')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)
        metrics = ensemble.evaluate(X_test, y_test)

        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'ic' in metrics
        assert 'directional_accuracy' in metrics

        # Metrics should be reasonable
        assert metrics['r2'] > 0.5  # Should explain > 50% variance
        assert metrics['ic'] > 0.3  # Should have decent correlation

    def test_ensemble_improves_over_individual(self, trained_models):
        """Test that ensemble improves over individual models."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        # Evaluate individual models
        individual_mse = {}
        for name, model in models.items():
            pred = model.predict(X_test)
            mse = np.mean((pred - y_test) ** 2)
            individual_mse[name] = mse

        # Evaluate ensemble
        config = EnsembleConfig(method='weighted')
        ensemble = ModelEnsemble(config)

        for name, model in models.items():
            ensemble.add_model(name, model)

        ensemble.fit(X_train, y_train, X_val, y_val)
        metrics = ensemble.evaluate(X_test, y_test)
        ensemble_mse = metrics['mse']

        # Ensemble should be better than average individual
        avg_individual_mse = np.mean(list(individual_mse.values()))

        # Ensemble should typically beat average (but not always guaranteed)
        assert ensemble_mse < avg_individual_mse * 1.1  # Allow 10% tolerance

    def test_no_models_raises_error(self):
        """Test that fitting without models raises error."""
        ensemble = ModelEnsemble()

        with pytest.raises(ValueError, match="No models"):
            ensemble.fit(np.random.randn(10, 5), np.random.randn(10))

    def test_predict_before_fit_raises_error(self, trained_models):
        """Test that predicting before fit raises error."""
        models, _ = trained_models
        ensemble = ModelEnsemble()

        for name, model in models.items():
            ensemble.add_model(name, model)

        with pytest.raises(ValueError, match="not fitted"):
            ensemble.predict(np.random.randn(10, 5))


class TestCreateEnsemble:
    """Test create_ensemble convenience function."""

    def test_create_ensemble_weighted(self, trained_models):
        """Test create_ensemble with weighted method."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        ensemble = create_ensemble(
            models,
            method='weighted',
            X_val=X_val,
            y_val=y_val
        )

        assert ensemble.is_fitted
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_create_ensemble_simple(self, trained_models):
        """Test create_ensemble with simple method."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        ensemble = create_ensemble(models, method='simple')

        assert ensemble.is_fitted
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_create_ensemble_stacking(self, trained_models):
        """Test create_ensemble with stacking method."""
        models, (X_train, y_train, X_val, y_val, X_test, y_test) = trained_models

        ensemble = create_ensemble(
            models,
            method='stacking',
            X_val=X_val,
            y_val=y_val
        )

        assert ensemble.is_fitted
        assert ensemble.meta_model is not None
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_create_ensemble_missing_val_data_raises_error(self, trained_models):
        """Test that weighted/stacking without val data raises error."""
        models, _ = trained_models

        with pytest.raises(ValueError, match="validation data"):
            create_ensemble(models, method='weighted')


# ============================================================================
# Property-Based Tests
# ============================================================================

from hypothesis import given, strategies as st, assume

@given(
    n_models=st.integers(min_value=2, max_value=5),
    n_samples=st.integers(min_value=50, max_value=200)
)
def test_ensemble_predictions_in_reasonable_range(n_models, n_samples):
    """Property: Ensemble predictions should be in reasonable range."""
    np.random.seed(42)
    n_features = 5

    # Generate data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Train models
    models = {}
    for i in range(n_models):
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        models[f'model{i}'] = model

    # Create ensemble
    config = EnsembleConfig(method='simple')
    ensemble = ModelEnsemble(config)

    for name, model in models.items():
        ensemble.add_model(name, model)

    ensemble.fit(X, y)

    # Predict
    X_test = np.random.randn(50, n_features)
    predictions = ensemble.predict(X_test)

    # Predictions should be in reasonable range
    assert np.abs(predictions).max() < 10 * np.abs(y).max()


@given(
    n_models=st.integers(min_value=2, max_value=5)
)
def test_weighted_ensemble_weights_sum_to_one(n_models):
    """Property: Weighted ensemble weights should sum to 1."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Train models
    models = {}
    for i in range(n_models):
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        models[f'model{i}'] = model

    # Create weighted ensemble
    config = EnsembleConfig(method='weighted')
    ensemble = ModelEnsemble(config)

    for name, model in models.items():
        ensemble.add_model(name, model)

    # Split for val
    split = n_samples // 2
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    ensemble.fit(X_train, y_train, X_val, y_val)

    # Check weights sum to 1
    total_weight = sum(ensemble.model_weights.values())
    assert abs(total_weight - 1.0) < 1e-6


def test_stacking_outperforms_simple_averaging(synthetic_data):
    """Test that stacking typically outperforms simple averaging."""
    X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data

    # Train diverse models
    models = {
        'ridge': Ridge(alpha=1.0).fit(X_train, y_train),
        'lasso': Lasso(alpha=0.1).fit(X_train, y_train),
        'lgb': lgb.LGBMRegressor(n_estimators=100, verbose=-1).fit(X_train, y_train)
    }

    # Simple averaging ensemble
    simple_config = EnsembleConfig(method='simple')
    simple_ensemble = ModelEnsemble(simple_config)
    for name, model in models.items():
        simple_ensemble.add_model(name, model)
    simple_ensemble.fit(X_train, y_train)

    # Stacking ensemble
    stack_config = EnsembleConfig(method='stacking')
    stack_ensemble = ModelEnsemble(stack_config)
    for name, model in models.items():
        stack_ensemble.add_model(name, model)
    stack_ensemble.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    simple_metrics = simple_ensemble.evaluate(X_test, y_test)
    stack_metrics = stack_ensemble.evaluate(X_test, y_test)

    # Stacking should typically have lower MSE
    # But allow some tolerance as it's not guaranteed
    assert stack_metrics['mse'] < simple_metrics['mse'] * 1.1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
