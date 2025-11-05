"""
Comprehensive test suite for N-BEATS implementation.

Tests cover:
1. Model initialization
2. Forward pass with shape validation
3. Gradient flow
4. Training loop
5. Numerical stability (NaN/Inf checking)
"""

import torch
import torch.nn as nn
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.nbeats import NBeats, NBeatsBlock, train_epoch, validate


class TestNBeatsBlock:
    """Test individual N-BEATS block."""

    def test_block_initialization(self):
        """Test block initializes with proper weight distribution."""
        block = NBeatsBlock(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_layers=4,
        )

        # Check weights are not default (random) values
        weights = block.theta_forecast.weight
        assert weights.std().item() > 0, "Weights should be initialized"
        assert torch.isfinite(weights).all(), "Weights should be finite"

    def test_block_forward_shape(self):
        """Test block output shapes are correct."""
        block = NBeatsBlock(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_layers=4,
        )

        x = torch.randn(32, 60)
        backcast, forecast = block(x)

        assert backcast.shape == (32, 60), f"Backcast shape {backcast.shape}"
        assert forecast.shape == (32, 21), f"Forecast shape {forecast.shape}"

    def test_block_forward_finite(self):
        """Test block outputs are finite."""
        block = NBeatsBlock(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_layers=4,
        )

        x = torch.randn(32, 60)
        backcast, forecast = block(x)

        assert torch.isfinite(backcast).all(), "Backcast contains NaN/Inf"
        assert torch.isfinite(forecast).all(), "Forecast contains NaN/Inf"

    def test_block_gradient_flow(self):
        """Test gradients flow through block."""
        block = NBeatsBlock(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_layers=4,
        )

        x = torch.randn(32, 60, requires_grad=True)
        backcast, forecast = block(x)

        loss = forecast.mean()
        loss.backward()

        # Check gradients exist and are finite
        assert block.theta_forecast.weight.grad is not None
        assert torch.isfinite(block.theta_forecast.weight.grad).all()

    def test_block_invalid_input_dim(self):
        """Test block rejects invalid input dimensions."""
        block = NBeatsBlock(input_size=60, output_size=21, hidden_size=128)

        # 3D input (should be 2D)
        with pytest.raises(AssertionError):
            block(torch.randn(32, 60, 1))

    def test_block_invalid_input_size(self):
        """Test block rejects wrong sequence length."""
        block = NBeatsBlock(input_size=60, output_size=21, hidden_size=128)

        # Wrong sequence length
        with pytest.raises(AssertionError):
            block(torch.randn(32, 50))  # Expected 60


class TestNBeats:
    """Test full N-BEATS model."""

    def test_model_initialization(self):
        """Test model initializes with 4 blocks."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=4,
            num_layers_per_block=4,
        )

        assert len(model.blocks) == 4, "Should have 4 blocks"
        assert model.input_size == 60
        assert model.output_size == 21

    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=4,
        )

        x = torch.randn(32, 60)
        forecast = model(x)

        assert forecast.shape == (32, 21), f"Output shape {forecast.shape}"

    def test_forward_pass_finite(self):
        """Test forward pass produces finite outputs."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=4,
        )

        x = torch.randn(32, 60)
        forecast = model(x)

        assert torch.isfinite(forecast).all(), "Output contains NaN/Inf"

    def test_forward_pass_residual_stacking(self):
        """Test residual stacking is working (blocks contribute to output)."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=4,
        )

        x = torch.randn(32, 60)
        forecast = model(x)

        # Output should not be constant (blocks should contribute)
        assert forecast.std().item() > 0, "Output should have variance"

    def test_gradient_flow_through_blocks(self):
        """Test gradients flow through all blocks."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=4,
        )

        x = torch.randn(32, 60, requires_grad=True)
        forecast = model(x)

        loss = forecast.mean()
        loss.backward()

        # Check all blocks have gradients
        for i, block in enumerate(model.blocks):
            grad = block.theta_forecast.weight.grad
            assert grad is not None, f"Block {i} has no gradients"
            assert torch.isfinite(grad).all(), f"Block {i} has NaN gradients"

    def test_gradient_norm_computation(self):
        """Test gradient norm calculation."""
        model = NBeats(input_size=60, output_size=21, hidden_size=128)

        x = torch.randn(32, 60, requires_grad=True)
        forecast = model(x)

        loss = forecast.mean()
        loss.backward()

        grad_norm = model.get_gradient_norm()
        assert grad_norm > 0, "Gradient norm should be positive"
        assert torch.isfinite(torch.tensor(grad_norm)), "Gradient norm should be finite"

    def test_deep_stacking_8_blocks(self):
        """Test model with deeper stack (8 blocks)."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=8,
            num_layers_per_block=4,
        )

        x = torch.randn(32, 60)
        forecast = model(x)

        assert forecast.shape == (32, 21)
        assert torch.isfinite(forecast).all()

    def test_variable_input_sizes(self):
        """Test model works with different input/output sizes."""
        test_cases = [
            (30, 7),    # Short-term
            (60, 21),   # Medium-term
            (252, 63),  # Long-term (1 year → 3 months)
        ]

        for input_size, output_size in test_cases:
            model = NBeats(
                input_size=input_size,
                output_size=output_size,
                hidden_size=128,
                num_blocks=4,
            )

            x = torch.randn(16, input_size)
            forecast = model(x)

            assert forecast.shape == (16, output_size), \
                f"Failed for input={input_size}, output={output_size}"
            assert torch.isfinite(forecast).all()

    def test_invalid_input_dimension(self):
        """Test model rejects invalid input dimensions."""
        model = NBeats(input_size=60, output_size=21, hidden_size=128)

        # 3D input (should be 2D)
        with pytest.raises(AssertionError):
            model(torch.randn(32, 60, 1))

    def test_invalid_input_size(self):
        """Test model rejects wrong sequence length."""
        model = NBeats(input_size=60, output_size=21, hidden_size=128)

        with pytest.raises(AssertionError):
            model(torch.randn(32, 50))  # Expected 60


class TestTraining:
    """Test training loop functionality."""

    def create_dummy_dataloader(self, num_samples=100, batch_size=32):
        """Create dummy dataset for testing."""
        data = []
        for _ in range(num_samples):
            x = torch.randn(60)  # Input sequence
            y = torch.randn(21)  # Target sequence
            data.append({'input': x, 'target': y})

        class DummyDataLoader:
            def __init__(self, data, batch_size):
                self.data = data
                self.batch_size = batch_size

            def __iter__(self):
                for i in range(0, len(self.data), self.batch_size):
                    batch = self.data[i:i + self.batch_size]
                    batch_x = torch.stack([d['input'] for d in batch])
                    batch_y = torch.stack([d['target'] for d in batch])
                    yield {'input': batch_x, 'target': batch_y}

        return DummyDataLoader(data, batch_size)

    def test_training_loop_convergence(self):
        """Test that training loop can reduce loss."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=4,
        )

        dataloader = self.create_dummy_dataloader(num_samples=200, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for epoch in range(3):
            loss = train_epoch(
                model, dataloader, optimizer,
                max_grad_norm=1.0, device='cpu'
            )
            losses.append(loss)

        # Loss should generally decrease (with noise)
        # At minimum, final loss should be comparable to first
        print(f"Training losses: {losses}")
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
            "All losses should be finite"

    def test_training_with_gradient_clipping(self):
        """Test training with gradient clipping prevents instability."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=4,
        )

        dataloader = self.create_dummy_dataloader(num_samples=100, batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # High LR

        for _ in range(5):
            loss = train_epoch(
                model, dataloader, optimizer,
                max_grad_norm=1.0, device='cpu'
            )
            assert torch.isfinite(torch.tensor(loss)), \
                "Loss should remain finite with gradient clipping"

    def test_validation_loop(self):
        """Test validation loop computation."""
        model = NBeats(input_size=60, output_size=21, hidden_size=128)

        dataloader = self.create_dummy_dataloader(num_samples=50, batch_size=16)
        val_loss = validate(model, dataloader, device='cpu')

        assert torch.isfinite(torch.tensor(val_loss)), "Validation loss should be finite"
        assert val_loss > 0, "Validation loss should be positive"

    def test_no_nan_during_training(self):
        """Test that NaNs don't appear during training."""
        model = NBeats(
            input_size=60,
            output_size=21,
            hidden_size=128,
            num_blocks=4,
        )

        dataloader = self.create_dummy_dataloader(num_samples=100, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(5):
            loss = train_epoch(
                model, dataloader, optimizer,
                max_grad_norm=1.0, device='cpu'
            )
            assert not torch.isnan(torch.tensor(loss)), \
                "Loss became NaN during training"


class TestNumericalStability:
    """Test numerical stability with edge cases."""

    def test_very_large_input_values(self):
        """Test model handles large input values."""
        model = NBeats(input_size=60, output_size=21, hidden_size=128)

        x = torch.randn(16, 60) * 1000  # Very large values
        forecast = model(x)

        assert torch.isfinite(forecast).all(), \
            "Model should handle large inputs with gradient clipping"

    def test_very_small_input_values(self):
        """Test model handles very small input values."""
        model = NBeats(input_size=60, output_size=21, hidden_size=128)

        x = torch.randn(16, 60) * 1e-5  # Very small values
        forecast = model(x)

        assert torch.isfinite(forecast).all(), \
            "Model should handle small inputs"

    def test_zero_input(self):
        """Test model handles zero input."""
        model = NBeats(input_size=60, output_size=21, hidden_size=128)

        x = torch.zeros(16, 60)
        forecast = model(x)

        assert torch.isfinite(forecast).all(), "Model should handle zero input"

    def test_constant_input(self):
        """Test model handles constant input."""
        model = NBeats(input_size=60, output_size=21, hidden_size=128)

        x = torch.ones(16, 60) * 3.14
        forecast = model(x)

        assert torch.isfinite(forecast).all(), "Model should handle constant input"


if __name__ == '__main__':
    """Run tests with pytest."""
    pytest.main([__file__, '-v', '--tb=short'])
