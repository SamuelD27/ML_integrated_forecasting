import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import warnings
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_feature_engineering():
    """Test the advanced feature engineering module."""
    logger.info("="*60)
    logger.info("TESTING PHASE 1: Advanced Feature Engineering")
    logger.info("="*60)

    try:
        from utils.advanced_feature_engineering import AdvancedFeatureEngineer, FeatureConfig
        logger.info("✓ Successfully imported advanced_feature_engineering module")
    except ImportError as e:
        logger.error(f"✗ Failed to import advanced_feature_engineering: {e}")
        return False

    # Create synthetic data for testing
    logger.info("\nCreating synthetic OHLCV data...")
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    n_days = len(dates)

    # Generate realistic-looking price data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)
    price = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV DataFrame
    df = pd.DataFrame({
        'Open': price * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': price * (1 + np.random.uniform(0, 0.02, n_days)),
        'Low': price * (1 + np.random.uniform(-0.02, 0, n_days)),
        'Close': price,
        'Adj Close': price,
        'Volume': np.random.uniform(1e6, 1e7, n_days)
    }, index=dates)

    logger.info(f"✓ Created synthetic data: {len(df)} days, shape: {df.shape}")

    # Initialize feature engineer
    logger.info("\nInitializing AdvancedFeatureEngineer...")
    config = FeatureConfig(
        lookback_windows=[1, 5, 10, 20],
        ma_periods=[5, 10, 20, 50],
        compute_fft=True,
        fft_components=5,
        compute_microstructure=True,
        compute_volatility=True,
        compute_cross_sectional=False  # Skip for now as we don't have market data
    )

    engineer = AdvancedFeatureEngineer(config)
    logger.info("✓ Feature engineer initialized")

    # Generate features
    logger.info("\nGenerating features...")
    try:
        features = engineer.generate_features(
            ticker='TEST',
            df=df,
            market_df=None,
            sector_df=None,
            save_features=False  # Don't save for test
        )

        logger.info(f"✓ Generated {len(features.columns)} features")
        logger.info(f"  Feature shape: {features.shape}")
        logger.info(f"  Sample features: {features.columns[:10].tolist()}")

        # Check for NaN values
        nan_count = features.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"  Warning: {nan_count} NaN values found (expected some at beginning due to lookback)")

        # Test scaling
        logger.info("\nTesting feature scaling...")
        train_size = int(len(features) * 0.8)
        train_features = features.iloc[:train_size]
        test_features = features.iloc[train_size:]

        scaled_train = engineer.fit_scalers(train_features)
        scaled_test = engineer.transform_features(test_features)

        logger.info(f"✓ Scaling successful")
        logger.info(f"  Scaled train shape: {scaled_train.shape}")
        logger.info(f"  Scaled test shape: {scaled_test.shape}")

    except Exception as e:
        logger.error(f"✗ Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n✓ PHASE 1 TEST COMPLETED SUCCESSFULLY")
    return True


def test_hybrid_model():
    """Test the hybrid model architecture."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PHASE 2: Hybrid Model Architecture")
    logger.info("="*60)

    # Check if PyTorch is available
    try:
        import torch
        logger.info(f"✓ PyTorch version: {torch.__version__}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"✓ Using device: {device}")
    except ImportError:
        logger.error("✗ PyTorch not installed. Please install it first.")
        return False

    # Import model components
    try:
        from ml_models.hybrid_base import HybridModelBase
        from ml_models.cnn_module import CNN1DFeatureExtractor
        from ml_models.lstm_module import LSTMEncoder
        from ml_models.transformer_module import TransformerEncoder
        from ml_models.fusion_layer import AttentionFusion
        from ml_models.hybrid_model import HybridTradingModel
        logger.info("✓ Successfully imported all model components")
    except ImportError as e:
        logger.error(f"✗ Failed to import model components: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test individual components
    logger.info("\nTesting individual components...")

    # Create sample input
    batch_size = 32
    seq_len = 60
    n_features = 50
    sample_input = torch.randn(batch_size, seq_len, n_features)

    # Test CNN module
    try:
        logger.info("\n1. Testing CNN module...")
        cnn = CNN1DFeatureExtractor(
            input_dim=n_features,
            sequence_length=seq_len,
            filters=[32, 64, 128],
            kernel_sizes=[3, 5, 7],
            dropout=0.2,
            output_dim=128
        )
        cnn_output, cnn_attention = cnn(sample_input)
        logger.info(f"   ✓ CNN output shape: {cnn_output.shape}")
        logger.info(f"   ✓ CNN attention shape: {cnn_attention.shape}")
        assert cnn_output.shape == (batch_size, 128)
    except Exception as e:
        logger.error(f"   ✗ CNN test failed: {e}")
        return False

    # Test LSTM module
    try:
        logger.info("\n2. Testing LSTM module...")
        lstm = LSTMEncoder(
            input_dim=n_features,
            hidden_dim=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            output_dim=128
        )
        lstm_output, lstm_hidden = lstm(sample_input)
        logger.info(f"   ✓ LSTM output shape: {lstm_output.shape}")
        logger.info(f"   ✓ LSTM hidden states shape: {lstm_hidden.shape}")
        assert lstm_output.shape == (batch_size, 128)
    except Exception as e:
        logger.error(f"   ✗ LSTM test failed: {e}")
        return False

    # Test Transformer module
    try:
        logger.info("\n3. Testing Transformer module...")
        transformer = TransformerEncoder(
            input_dim=n_features,
            d_model=256,
            n_heads=8,
            num_layers=2,
            d_ff=512,
            dropout=0.1,
            output_dim=128
        )
        transformer_output, transformer_attention = transformer(sample_input)
        logger.info(f"   ✓ Transformer output shape: {transformer_output.shape}")
        logger.info(f"   ✓ Transformer attention shape: {transformer_attention.shape}")
        assert transformer_output.shape == (batch_size, 128)
    except Exception as e:
        logger.error(f"   ✗ Transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Fusion layer
    try:
        logger.info("\n4. Testing Fusion layer...")
        fusion = AttentionFusion(
            input_dims={'cnn': 128, 'lstm': 128, 'transformer': 128},
            hidden_dim=128,
            output_dim=128,
            dropout=0.1
        )
        encoder_outputs = {
            'cnn': cnn_output,
            'lstm': lstm_output,
            'transformer': transformer_output
        }
        fused_output, fusion_weights = fusion(encoder_outputs, return_weights=True)
        logger.info(f"   ✓ Fused output shape: {fused_output.shape}")
        logger.info(f"   ✓ Fusion weights: {fusion_weights}")
        assert fused_output.shape == (batch_size, 128)
    except Exception as e:
        logger.error(f"   ✗ Fusion test failed: {e}")
        return False

    # Test complete hybrid model
    try:
        logger.info("\n5. Testing complete HybridTradingModel...")
        model = HybridTradingModel(
            input_dim=n_features,
            sequence_length=seq_len,
            cnn_filters=[32, 64, 128],
            cnn_kernel_sizes=[3, 5, 7],
            cnn_dropout=0.2,
            lstm_hidden=128,
            lstm_layers=2,
            lstm_bidirectional=True,
            lstm_dropout=0.2,
            transformer_d_model=256,
            transformer_heads=8,
            transformer_layers=2,
            transformer_ff_dim=512,
            transformer_dropout=0.1,
            fusion_hidden_dim=128,
            fusion_dropout=0.1,
            output_dim=1,
            predict_direction=True
        )

        # Forward pass
        outputs = model(sample_input, return_intermediates=True)

        logger.info(f"   ✓ Price prediction shape: {outputs['price_prediction'].shape}")
        logger.info(f"   ✓ Direction logits shape: {outputs['direction_logits'].shape}")
        logger.info(f"   ✓ Fusion weights: {outputs['fusion_weights']}")

        # Test prediction
        model.eval()
        with torch.no_grad():
            price_pred = model(sample_input)
        logger.info(f"   ✓ Simple forward pass shape: {price_pred.shape}")

        # Test multi-task loss
        predictions = {
            'price_prediction': outputs['price_prediction'],
            'direction_logits': outputs['direction_logits']
        }
        targets = {
            'price': torch.randn(batch_size, 1),
            'direction': torch.randint(0, 3, (batch_size,))
        }
        loss = model.compute_multi_task_loss(predictions, targets)
        logger.info(f"   ✓ Multi-task loss computed: {loss.item():.4f}")

        # Test uncertainty estimation
        test_input = sample_input[:1]  # Single sample
        uncertainty_results = model.predict_with_uncertainty(test_input, n_samples=5)
        logger.info(f"   ✓ Uncertainty estimation completed")
        logger.info(f"     Price mean shape: {uncertainty_results['price_mean'].shape}")
        logger.info(f"     Price std shape: {uncertainty_results['price_std'].shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"   ✓ Total parameters: {total_params:,}")
        logger.info(f"   ✓ Trainable parameters: {trainable_params:,}")

    except Exception as e:
        logger.error(f"   ✗ Hybrid model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n✓ PHASE 2 TEST COMPLETED SUCCESSFULLY")
    return True


def test_training_loop():
    """Test a mini training loop."""
    logger.info("\n" + "="*60)
    logger.info("TESTING TRAINING LOOP")
    logger.info("="*60)

    try:
        from ml_models.hybrid_model import HybridTradingModel
        from torch.utils.data import DataLoader, TensorDataset

        # Create small synthetic dataset
        n_samples = 100
        seq_len = 60
        n_features = 20

        X = torch.randn(n_samples, seq_len, n_features)
        y = torch.randn(n_samples, 1)

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Create model
        model = HybridTradingModel(
            input_dim=n_features,
            sequence_length=seq_len,
            cnn_filters=[16, 32],
            lstm_hidden=32,
            lstm_layers=1,
            transformer_d_model=64,
            transformer_heads=4,
            transformer_layers=1,
            fusion_hidden_dim=32,
            predict_direction=False
        )

        logger.info("Training for 3 epochs (mini test)...")

        # Quick training test
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            learning_rate=0.001,
            verbose=False
        )

        logger.info(f"✓ Training completed")
        logger.info(f"  Best epoch: {history['best_epoch']}")
        logger.info(f"  Best validation loss: {history['best_val_loss']:.4f}")
        logger.info(f"  Final RMSE: {history['final_metrics']['rmse']:.4f}")

    except Exception as e:
        logger.error(f"✗ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n✓ TRAINING LOOP TEST COMPLETED SUCCESSFULLY")
    return True


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("HYBRID DEEP LEARNING TRADING SYSTEM - QUICK TEST")
    logger.info("="*60)
    logger.info(f"Test started at: {datetime.now()}")

    # Check Python version
    logger.info(f"\nPython version: {sys.version}")

    # Run tests
    test_results = []

    # Test Phase 1
    test_results.append(("Feature Engineering", test_feature_engineering()))

    # Test Phase 2
    test_results.append(("Hybrid Model", test_hybrid_model()))

    # Test Training
    test_results.append(("Training Loop", test_training_loop()))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<30} {status}")

    all_passed = all(result for _, result in test_results)

    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! The hybrid system is working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Install PyTorch if not already installed:")
        logger.info("   pip install torch torchvision")
        logger.info("2. Fetch real data using data_fetching.py")
        logger.info("3. Generate features: python utils/advanced_feature_engineering.py")
        logger.info("4. Continue with Phase 3 (RL Agent) or Phase 4 (Training Pipeline)")
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
        logger.info("\nCommon issues:")
        logger.info("1. Missing dependencies (install PyTorch)")
        logger.info("2. Import errors (check file paths)")
        logger.info("3. Version incompatibilities")

    logger.info(f"\nTest completed at: {datetime.now()}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)