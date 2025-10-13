#!/usr/bin/env python3

import unittest
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.advanced_feature_engineering import AdvancedFeatureEngineer, FeatureConfig
from ml_models.cnn_module import CNN1DFeatureExtractor, MultiScaleConv1D
from ml_models.lstm_module import LSTMEncoder, AttentionLayer
from ml_models.transformer_module import TransformerEncoder, PositionalEncoding
from ml_models.fusion_layer import AttentionFusion
from ml_models.hybrid_model import HybridTradingModel
from ml_models.rl_agent import TradingEnvironment, TradingConfig, PPOAgent
from training.utils import WalkForwardSplitter, TradingDataset, compute_trading_metrics
from backtesting.backtest_engine import BacktestEngine, BacktestConfig


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering components"""

    def setUp(self):
        """Create sample data for testing"""
        np.random.seed(42)
        self.n_samples = 500

        # Create synthetic OHLCV data
        dates = pd.date_range(end='2024-01-01', periods=self.n_samples, freq='D')
        close = 100 + np.cumsum(np.random.randn(self.n_samples) * 2)

        self.df = pd.DataFrame({
            'open': close * (1 + np.random.uniform(-0.01, 0.01, self.n_samples)),
            'high': close * (1 + np.random.uniform(0, 0.02, self.n_samples)),
            'low': close * (1 + np.random.uniform(-0.02, 0, self.n_samples)),
            'close': close,
            'volume': np.random.uniform(1e6, 1e7, self.n_samples)
        }, index=dates)

    def test_feature_generation(self):
        """Test feature generation"""
        config = FeatureConfig(use_all=True)
        engineer = AdvancedFeatureEngineer(config)

        features = engineer.generate_features(
            ticker='TEST',
            df=self.df,
            save_features=False
        )

        # Check features were generated
        self.assertIsNotNone(features)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 50)  # Should have many features
        self.assertLess(len(features), len(self.df))  # Some rows lost due to indicators

        # Check no NaN values (after dropna)
        self.assertEqual(features.isna().sum().sum(), 0)

        # Check feature names were stored
        self.assertGreater(len(engineer.feature_names), 0)

    def test_feature_scaling(self):
        """Test feature scaling"""
        config = FeatureConfig(normalize=True)
        engineer = AdvancedFeatureEngineer(config)

        features = engineer.generate_features(
            ticker='TEST',
            df=self.df,
            save_features=False
        )

        # Check features are scaled
        for col in features.columns:
            if 'return' not in col:  # Returns might have different ranges
                self.assertLessEqual(features[col].max(), 10)
                self.assertGreaterEqual(features[col].min(), -10)

    def test_incremental_features(self):
        """Test incremental feature updates"""
        engineer = AdvancedFeatureEngineer()

        # Generate initial features
        features1 = engineer.generate_features(
            ticker='TEST',
            df=self.df[:250],
            save_features=False
        )

        # Generate features with more data
        features2 = engineer.generate_features(
            ticker='TEST',
            df=self.df,
            save_features=False
        )

        # Should have more samples with more data
        self.assertGreater(len(features2), len(features1))

        # Feature columns should be the same
        self.assertEqual(list(features1.columns), list(features2.columns))


class TestCNNModule(unittest.TestCase):
    """Test CNN module components"""

    def setUp(self):
        self.batch_size = 16
        self.seq_length = 60
        self.input_dim = 78
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_multi_scale_conv(self):
        """Test multi-scale convolution"""
        module = MultiScaleConv1D(
            in_channels=self.input_dim,
            out_channels=128,
            kernel_sizes=[3, 5, 7]
        ).to(self.device)

        x = torch.randn(self.batch_size, self.input_dim, self.seq_length).to(self.device)
        output = module(x)

        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 128)
        self.assertEqual(output.shape[2], self.seq_length)

        # Check no NaN values
        self.assertFalse(torch.isnan(output).any())

    def test_cnn_extractor(self):
        """Test CNN feature extractor"""
        model = CNN1DFeatureExtractor(
            input_dim=self.input_dim,
            sequence_length=self.seq_length,
            filters=[64, 128, 256]
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)
        features, attention_weights = model(x)

        # Check output shape
        self.assertEqual(features.shape[0], self.batch_size)
        self.assertEqual(features.shape[1], 256)  # Final hidden dimension

        # Check attention weights if returned
        if attention_weights is not None:
            self.assertEqual(attention_weights.shape[0], self.batch_size)

        # Check no NaN values
        self.assertFalse(torch.isnan(features).any())

    def test_cnn_gradient_flow(self):
        """Test gradient flow through CNN"""
        model = CNN1DFeatureExtractor(
            input_dim=self.input_dim,
            sequence_length=self.seq_length
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_length, self.input_dim, requires_grad=True).to(self.device)
        features, _ = model(x)
        loss = features.mean()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


class TestLSTMModule(unittest.TestCase):
    """Test LSTM module components"""

    def setUp(self):
        self.batch_size = 16
        self.seq_length = 60
        self.input_dim = 78
        self.hidden_dim = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_attention_layer(self):
        """Test attention mechanism"""
        attention = AttentionLayer(self.hidden_dim).to(self.device)

        x = torch.randn(self.batch_size, self.seq_length, self.hidden_dim).to(self.device)
        output, weights = attention(x)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_length))

        # Check attention weights sum to 1
        weight_sums = weights.sum(dim=1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), rtol=1e-5, atol=1e-5)

    def test_lstm_encoder(self):
        """Test LSTM encoder"""
        model = LSTMEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            use_attention=True
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)
        output, attention_weights = model(x)

        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.hidden_dim * 2)  # Bidirectional

        # Check attention weights if used
        if attention_weights is not None:
            self.assertEqual(attention_weights.shape[0], self.batch_size)

        # Check no NaN values
        self.assertFalse(torch.isnan(output).any())

    def test_lstm_variable_length(self):
        """Test LSTM with variable length sequences"""
        model = LSTMEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # Create sequences with different lengths
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)
        lengths = torch.randint(30, self.seq_length, (self.batch_size,))

        output, _ = model(x, lengths)

        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertFalse(torch.isnan(output).any())


class TestTransformerModule(unittest.TestCase):
    """Test Transformer module components"""

    def setUp(self):
        self.batch_size = 16
        self.seq_length = 60
        self.input_dim = 78
        self.d_model = 512
        self.n_heads = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_positional_encoding(self):
        """Test positional encoding"""
        pe = PositionalEncoding(
            d_model=self.d_model,
            max_len=5000,
            learnable=True
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_length, self.d_model).to(self.device)
        output = pe(x)

        # Check output shape unchanged
        self.assertEqual(output.shape, x.shape)

        # Check encoding was added (output should be different from input)
        self.assertFalse(torch.allclose(output, x))

    def test_transformer_encoder(self):
        """Test Transformer encoder"""
        model = TransformerEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=4,
            use_dual_attention=True
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)
        output, attention_weights = model(x)

        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.d_model)

        # Check attention weights if returned
        if attention_weights is not None:
            self.assertIsInstance(attention_weights, dict)

        # Check no NaN values
        self.assertFalse(torch.isnan(output).any())

    def test_transformer_masking(self):
        """Test Transformer with masking"""
        model = TransformerEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)

        # Create causal mask
        mask = torch.triu(torch.ones(self.seq_length, self.seq_length), diagonal=1).bool().to(self.device)

        output, _ = model(x, mask=mask)

        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertFalse(torch.isnan(output).any())


class TestFusionLayer(unittest.TestCase):
    """Test fusion layer components"""

    def setUp(self):
        self.batch_size = 16
        self.hidden_dim = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_attention_fusion(self):
        """Test attention-based fusion"""
        fusion = AttentionFusion(
            input_dims={'cnn': 256, 'lstm': 512, 'transformer': 512},
            fusion_hidden_dim=self.hidden_dim,
            use_gating=True
        ).to(self.device)

        # Create encoder outputs
        encoder_outputs = {
            'cnn': torch.randn(self.batch_size, 256).to(self.device),
            'lstm': torch.randn(self.batch_size, 512).to(self.device),
            'transformer': torch.randn(self.batch_size, 512).to(self.device)
        }

        fused, weights = fusion(encoder_outputs, return_weights=True)

        # Check output shape
        self.assertEqual(fused.shape, (self.batch_size, self.hidden_dim))

        # Check weights
        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), 3)

        # Check weights sum to 1
        weight_sum = sum(weights.values())
        torch.testing.assert_close(
            weight_sum,
            torch.ones(self.batch_size, 1).to(self.device),
            rtol=1e-5,
            atol=1e-5
        )

    def test_fusion_missing_encoder(self):
        """Test fusion with missing encoder output"""
        fusion = AttentionFusion(
            input_dims={'cnn': 256, 'lstm': 512, 'transformer': 512},
            fusion_hidden_dim=self.hidden_dim
        ).to(self.device)

        # Missing transformer output
        encoder_outputs = {
            'cnn': torch.randn(self.batch_size, 256).to(self.device),
            'lstm': torch.randn(self.batch_size, 512).to(self.device)
        }

        # Should handle gracefully
        fused, _ = fusion(encoder_outputs)
        self.assertEqual(fused.shape, (self.batch_size, self.hidden_dim))


class TestHybridModel(unittest.TestCase):
    """Test complete hybrid model"""

    def setUp(self):
        self.batch_size = 8
        self.seq_length = 60
        self.input_dim = 78
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_model_forward(self):
        """Test forward pass"""
        model = HybridTradingModel(
            input_dim=self.input_dim,
            sequence_length=self.seq_length,
            predict_direction=True
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)

        price_pred, direction_logits = model(x)

        # Check output shapes
        self.assertEqual(price_pred.shape, (self.batch_size, 1))
        self.assertEqual(direction_logits.shape, (self.batch_size, 3))  # 3 classes

        # Check no NaN values
        self.assertFalse(torch.isnan(price_pred).any())
        self.assertFalse(torch.isnan(direction_logits).any())

    def test_model_training_step(self):
        """Test training step"""
        model = HybridTradingModel(
            input_dim=self.input_dim,
            sequence_length=self.seq_length,
            predict_direction=True
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion_price = nn.MSELoss()
        criterion_direction = nn.CrossEntropyLoss()

        # Create batch
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)
        y_price = torch.randn(self.batch_size, 1).to(self.device)
        y_direction = torch.randint(0, 3, (self.batch_size,)).to(self.device)

        # Forward pass
        model.train()
        price_pred, direction_logits = model(x)

        # Compute losses
        loss_price = criterion_price(price_pred, y_price)
        loss_direction = criterion_direction(direction_logits, y_direction)
        total_loss = loss_price + loss_direction

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check loss is finite
        self.assertTrue(torch.isfinite(total_loss))

        # Check gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_model_inference(self):
        """Test model inference with uncertainty"""
        model = HybridTradingModel(
            input_dim=self.input_dim,
            sequence_length=self.seq_length,
            predict_direction=True
        ).to(self.device)

        x = torch.randn(1, self.seq_length, self.input_dim).to(self.device)

        # Get predictions with uncertainty
        predictions = []
        model.train()  # Enable dropout for MC sampling

        with torch.no_grad():
            for _ in range(10):
                price_pred, _ = model(x)
                predictions.append(price_pred.cpu().numpy())

        predictions = np.array(predictions)

        # Check we get variation (uncertainty)
        std = predictions.std()
        self.assertGreater(std, 0)

    def test_model_save_load(self):
        """Test model saving and loading"""
        model1 = HybridTradingModel(
            input_dim=self.input_dim,
            sequence_length=self.seq_length
        ).to(self.device)

        # Create temporary checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save model
            torch.save({
                'model_state_dict': model1.state_dict(),
                'input_dim': self.input_dim,
                'sequence_length': self.seq_length
            }, tmp.name)

            # Load model
            model2 = HybridTradingModel(
                input_dim=self.input_dim,
                sequence_length=self.seq_length
            ).to(self.device)

            checkpoint = torch.load(tmp.name, map_location=self.device)
            model2.load_state_dict(checkpoint['model_state_dict'])

            # Check models produce same output
            model1.eval()
            model2.eval()

            x = torch.randn(1, self.seq_length, self.input_dim).to(self.device)

            with torch.no_grad():
                out1, _ = model1(x)
                out2, _ = model2(x)

            torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)


class TestRLAgent(unittest.TestCase):
    """Test RL trading agent"""

    def setUp(self):
        """Create test environment"""
        np.random.seed(42)

        # Create synthetic data
        dates = pd.date_range(end='2024-01-01', periods=1000, freq='D')
        prices = 100 + np.cumsum(np.random.randn(1000) * 2)

        self.data = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, 1000)
        }, index=dates)

        # Create features
        self.features = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000)
            for i in range(10)
        }, index=dates)

        self.config = TradingConfig(
            initial_capital=100000,
            max_position=1000,
            commission=0.001,
            slippage=0.001
        )

    def test_environment_creation(self):
        """Test environment creation"""
        env = TradingEnvironment(
            data=self.data,
            features=self.features,
            config=self.config
        )

        # Check spaces
        self.assertIsNotNone(env.observation_space)
        self.assertIsNotNone(env.action_space)

        # Check reset
        obs = env.reset()
        self.assertEqual(obs.shape[0], env.observation_space.shape[0])

    def test_environment_step(self):
        """Test environment step"""
        env = TradingEnvironment(
            data=self.data,
            features=self.features,
            config=self.config
        )

        obs = env.reset()

        # Take random actions
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            # Check returns
            self.assertEqual(obs.shape[0], env.observation_space.shape[0])
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)

            if done:
                break

    def test_ppo_agent_init(self):
        """Test PPO agent initialization"""
        env = TradingEnvironment(
            data=self.data,
            features=self.features,
            config=self.config
        )

        agent = PPOAgent(env, self.config)

        # Check network was created
        self.assertIsNotNone(agent.network)
        self.assertIsNotNone(agent.optimizer)


class TestBacktesting(unittest.TestCase):
    """Test backtesting engine"""

    def setUp(self):
        """Create test data"""
        np.random.seed(42)

        dates = pd.date_range(end='2024-01-01', periods=500, freq='D')
        prices = 100 + np.cumsum(np.random.randn(500) * 2)

        self.data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, 500)
        }, index=dates)

        self.config = BacktestConfig(
            initial_capital=100000,
            commission=0.001,
            slippage=0.001
        )

    def test_backtest_engine_creation(self):
        """Test backtest engine creation"""
        engine = BacktestEngine(self.data, self.config)

        self.assertEqual(engine.cash, self.config.initial_capital)
        self.assertEqual(len(engine.positions), 0)
        self.assertEqual(len(engine.trades), 0)

    def test_simple_strategy_backtest(self):
        """Test backtesting with simple strategy"""
        engine = BacktestEngine(self.data, self.config)

        # Simple moving average crossover strategy
        def strategy(data, index):
            if index < 20:
                return 0

            sma_short = data['close'].iloc[index-10:index].mean()
            sma_long = data['close'].iloc[index-20:index].mean()

            if sma_short > sma_long:
                return 1  # Buy signal
            elif sma_short < sma_long:
                return -1  # Sell signal
            else:
                return 0  # Hold

        # Run backtest
        results = engine.run(
            strategy=strategy,
            start_date=self.data.index[20],
            end_date=self.data.index[-1]
        )

        # Check results
        self.assertIsNotNone(results)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('trades', results)


class TestTrainingUtils(unittest.TestCase):
    """Test training utilities"""

    def test_walk_forward_splitter(self):
        """Test walk-forward data splitting"""
        n_samples = 1000
        X = np.random.randn(n_samples, 10)
        y = np.random.randn(n_samples)

        splitter = WalkForwardSplitter(
            n_splits=5,
            train_size=500,
            val_size=100,
            test_size=100
        )

        splits = list(splitter.split(X, y))

        # Check number of splits
        self.assertGreater(len(splits), 0)

        # Check no overlap
        for train_idx, val_idx, test_idx in splits:
            # Check sizes
            self.assertLessEqual(len(train_idx), 500)
            self.assertLessEqual(len(val_idx), 100)
            self.assertLessEqual(len(test_idx), 100)

            # Check no overlap
            train_set = set(train_idx)
            val_set = set(val_idx)
            test_set = set(test_idx)

            self.assertEqual(len(train_set & val_set), 0)
            self.assertEqual(len(train_set & test_set), 0)
            self.assertEqual(len(val_set & test_set), 0)

            # Check temporal order
            self.assertLess(max(train_idx), min(val_idx))
            self.assertLess(max(val_idx), min(test_idx))

    def test_trading_metrics(self):
        """Test trading metrics computation"""
        # Create sample predictions and targets
        predictions = np.array([100, 101, 99, 102, 98])
        targets = np.array([100, 100, 100, 100, 100])

        metrics = compute_trading_metrics(predictions, targets)

        # Check metrics exist
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('directional_accuracy', metrics)
        self.assertIn('sharpe_ratio', metrics)

        # Check metrics are reasonable
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreaterEqual(metrics['directional_accuracy'], 0)
        self.assertLessEqual(metrics['directional_accuracy'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to predictions"""
        np.random.seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Create synthetic data
        dates = pd.date_range(end='2024-01-01', periods=500, freq='D')
        prices = 100 + np.cumsum(np.random.randn(500) * 2)

        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, 500)
        }, index=dates)

        # 2. Generate features
        engineer = AdvancedFeatureEngineer()
        features = engineer.generate_features('TEST', df, save_features=False)

        # 3. Prepare sequences
        seq_length = 30
        sequences = []
        targets = []

        for i in range(len(features) - seq_length - 1):
            sequences.append(features.iloc[i:i+seq_length].values)
            targets.append(prices[i+seq_length+1])

        X = np.array(sequences)
        y = np.array(targets)

        # 4. Create and train model
        model = HybridTradingModel(
            input_dim=X.shape[2],
            sequence_length=seq_length,
            predict_direction=True
        ).to(device)

        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(5):  # Very few epochs for testing
            # Create batch
            batch_idx = np.random.choice(len(X), size=min(16, len(X)), replace=False)
            batch_x = torch.FloatTensor(X[batch_idx]).to(device)
            batch_y = torch.FloatTensor(y[batch_idx].reshape(-1, 1)).to(device)

            # Training step
            optimizer.zero_grad()
            price_pred, _ = model(batch_x)
            loss = criterion(price_pred, batch_y)
            loss.backward()
            optimizer.step()

        # 5. Make predictions
        model.eval()
        test_x = torch.FloatTensor(X[-10:]).to(device)

        with torch.no_grad():
            predictions, directions = model(test_x)

        # 6. Check predictions
        self.assertEqual(predictions.shape[0], 10)
        self.assertEqual(directions.shape[0], 10)
        self.assertFalse(torch.isnan(predictions).any())
        self.assertFalse(torch.isnan(directions).any())


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestCNNModule))
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMModule))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerModule))
    suite.addTests(loader.loadTestsFromTestCase(TestFusionLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridModel))
    suite.addTests(loader.loadTestsFromTestCase(TestRLAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktesting))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)