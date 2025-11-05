"""
Unit tests for regime classifier.

Tests:
- Model initialization
- Forward pass shapes
- Prediction API
- Feature normalization
- Model save/load
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

# Import after ensuring path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.regime_classifier import RegimeClassifier, RegimeDetectorNN


class TestRegimeClassifier:
    """Test core PyTorch model."""

    def test_model_initialization(self):
        """Test model can be created."""
        model = RegimeClassifier(
            input_size=10,
            hidden1_size=64,
            hidden2_size=32,
            output_size=4,
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        model = RegimeClassifier()

        # Single sample
        x = torch.randn(1, 10)
        y = model(x)
        assert y.shape == (1, 4)

        # Batch
        x = torch.randn(32, 10)
        y = model(x)
        assert y.shape == (32, 4)

    def test_model_parameters(self):
        """Test model has correct number of parameters."""
        model = RegimeClassifier()

        # Count parameters
        # Layer 1: 10*64 + 64 (bias) = 704
        # Layer 2: 64*64 + 64 (bias) = 4160
        # Layer 3: 64*32 + 32 (bias) = 2080
        # Layer 4: 32*4 + 4 (bias) = 132
        # Total: ~6400+
        total = sum(p.numel() for p in model.parameters())
        assert total > 3000  # Rough check

    def test_model_requires_grad(self):
        """Test model parameters require gradients."""
        model = RegimeClassifier()
        for param in model.parameters():
            assert param.requires_grad


class TestRegimeDetectorNN:
    """Test inference wrapper."""

    def test_initialization_cpu(self):
        """Test detector can be initialized on CPU."""
        detector = RegimeDetectorNN(device='cpu')
        assert detector.device.type == 'cpu'
        assert detector.model is not None

    def test_initialization_with_device(self):
        """Test detector device specification."""
        detector = RegimeDetectorNN(device='cpu')
        assert str(detector.device) == 'cpu'

    def test_predict_single_sample(self):
        """Test prediction on single sample."""
        detector = RegimeDetectorNN(device='cpu')
        features = np.random.randn(10).astype(np.float32)

        result = detector.predict(features)

        assert 'regime' in result
        assert 'class_id' in result
        assert 'confidence' in result
        assert result['regime'] in ['bull', 'bear', 'neutral', 'crisis']
        assert isinstance(result['class_id'], (int, np.integer))
        assert 0 <= result['confidence'] <= 1

    def test_predict_batch(self):
        """Test prediction on batch."""
        detector = RegimeDetectorNN(device='cpu')
        features = np.random.randn(5, 10).astype(np.float32)

        result = detector.predict(features)

        assert 'regimes' in result
        assert 'class_ids' in result
        assert 'confidences' in result
        assert len(result['regimes']) == 5
        assert len(result['class_ids']) == 5
        assert len(result['confidences']) == 5

    def test_predict_with_confidence_probs(self):
        """Test prediction returns probabilities."""
        detector = RegimeDetectorNN(device='cpu')
        features = np.random.randn(10).astype(np.float32)

        result = detector.predict(features, return_confidence=True)

        assert 'probabilities' in result
        probs = result['probabilities']
        assert len(probs) == 4
        assert all(k in probs for k in ['bull', 'bear', 'neutral', 'crisis'])
        # Check probabilities sum to ~1
        total_prob = sum(probs.values())
        assert 0.99 <= total_prob <= 1.01

    def test_predict_invalid_feature_count(self):
        """Test error on wrong feature count."""
        detector = RegimeDetectorNN(device='cpu')
        features = np.random.randn(5)  # Wrong size

        with pytest.raises(ValueError):
            detector.predict(features)

    def test_feature_normalization(self):
        """Test feature normalization."""
        detector = RegimeDetectorNN(device='cpu')

        mean = np.array([0.1] * 10)
        std = np.array([0.05] * 10)
        detector.set_feature_stats(mean, std)

        features = np.array([0.12] * 10)
        normalized = detector.normalize_features(features)

        # Check normalization: (0.12 - 0.1) / 0.05 = 0.4
        expected = np.array([0.4] * 10)
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_feature_normalization_none(self):
        """Test normalization returns features unchanged if no stats."""
        detector = RegimeDetectorNN(device='cpu')
        features = np.random.randn(10)

        # No stats set - should return unchanged
        normalized = detector.normalize_features(features)
        np.testing.assert_array_almost_equal(normalized, features)

    def test_predict_from_dict(self):
        """Test prediction from feature dictionary."""
        detector = RegimeDetectorNN(device='cpu')

        feature_dict = {
            'vol_20d': 0.1,
            'ret_20d': 0.002,
            'ret_60d': 0.005,
            'vol_ratio': 1.0,
            'rsi': 50.0,
            'macd': 0.5,
            'adx': 30.0,
            'volume_ratio': 1.0,
            'price_momentum': 0.01,
            'volatility_trend': 0.0,
        }

        result = detector.predict_from_dict(feature_dict)

        assert 'regime' in result
        assert result['regime'] in ['bull', 'bear', 'neutral', 'crisis']

    def test_save_load_model(self):
        """Test model save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_model.pt'

            # Create and save
            detector1 = RegimeDetectorNN(device='cpu')
            detector1.save_model(str(checkpoint_path))
            assert checkpoint_path.exists()

            # Load and verify
            detector2 = RegimeDetectorNN(
                device='cpu',
                model_path=str(checkpoint_path)
            )

            features = np.random.randn(10).astype(np.float32)
            result1 = detector1.predict(features)
            result2 = detector2.predict(features)

            # Both should give same result
            assert result1['regime'] == result2['regime']
            assert result1['class_id'] == result2['class_id']

    def test_model_eval_mode(self):
        """Test model is in eval mode for inference."""
        detector = RegimeDetectorNN(device='cpu')
        assert not detector.model.training
        # In eval mode: no dropout, BatchNorm uses running stats

    def test_no_grad_inference(self):
        """Test inference doesn't create computation graph."""
        detector = RegimeDetectorNN(device='cpu')
        features = np.random.randn(10).astype(np.float32)

        # No gradients should be computed
        initial_grad = sum(
            p.grad.numel() if p.grad is not None else 0
            for p in detector.model.parameters()
        )

        detector.predict(features)

        final_grad = sum(
            p.grad.numel() if p.grad is not None else 0
            for p in detector.model.parameters()
        )

        # Gradients shouldn't change (no backward pass)
        assert initial_grad == final_grad


class TestFeatureOrdering:
    """Test feature ordering is consistent."""

    def test_feature_names_order(self):
        """Test feature names are in correct order."""
        detector = RegimeDetectorNN(device='cpu')

        expected_order = [
            'vol_20d',
            'ret_20d',
            'ret_60d',
            'vol_ratio',
            'rsi',
            'macd',
            'adx',
            'volume_ratio',
            'price_momentum',
            'volatility_trend',
        ]

        assert detector.feature_names == expected_order

    def test_feature_count(self):
        """Test exactly 10 features."""
        detector = RegimeDetectorNN(device='cpu')
        assert len(detector.feature_names) == 10


class TestRegimeClassMapping:
    """Test regime class mapping."""

    def test_regime_classes(self):
        """Test regime class mapping."""
        from ml_models.regime_classifier import REGIME_CLASSES, REGIME_TO_CLASS

        assert REGIME_CLASSES[0] == 'bull'
        assert REGIME_CLASSES[1] == 'bear'
        assert REGIME_CLASSES[2] == 'neutral'
        assert REGIME_CLASSES[3] == 'crisis'

        assert REGIME_TO_CLASS['bull'] == 0
        assert REGIME_TO_CLASS['bear'] == 1
        assert REGIME_TO_CLASS['neutral'] == 2
        assert REGIME_TO_CLASS['crisis'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
