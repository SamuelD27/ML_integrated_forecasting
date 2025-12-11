"""
Ensemble Forecaster
===================
Pipeline-ready ensemble of N-BEATS, LSTM, and TFT models.

Provides:
- Common forecast() interface for all models
- Parallel model execution
- Weighted ensemble aggregation
- Uncertainty estimation from model disagreement
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Optional torch import
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)

# Import pipeline types
try:
    from pipeline.core_types import Forecast
    HAS_PIPELINE_TYPES = True
except ImportError:
    HAS_PIPELINE_TYPES = False
    logger.warning("Pipeline types not available")


@dataclass
class ForecastResult:
    """Standard forecast result from a single model."""
    mean_return: float
    volatility: float
    p10: float
    p90: float
    confidence_score: float
    raw_predictions: Optional[np.ndarray] = None
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseForecaster(ABC):
    """Abstract base class for forecaster models."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.model_path = model_path
        if HAS_TORCH:
            self.device = torch.device(device)
        else:
            self.device = device  # Store as string when torch not available
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """Load model weights from disk."""
        pass

    @abstractmethod
    def forecast(
        self,
        price_history: pd.DataFrame,
        horizon_days: int = 10,
    ) -> ForecastResult:
        """
        Generate forecast from price history.

        Args:
            price_history: DataFrame with OHLCV columns (indexed by date)
            horizon_days: Forecast horizon in trading days

        Returns:
            ForecastResult with:
            - mean_return: Expected return over horizon
            - volatility: Expected volatility
            - p10: 10th percentile return
            - p90: 90th percentile return
            - confidence_score: Model's confidence (0-1)
            - raw_predictions: Raw model outputs
        """
        pass

    def _prepare_returns(self, price_history: pd.DataFrame) -> np.ndarray:
        """Prepare log returns from price data."""
        # Normalize column names
        df = price_history.copy()
        df.columns = [c.lower() for c in df.columns]

        if 'close' in df.columns:
            prices = df['close']
        elif 'adj close' in df.columns:
            prices = df['adj close']
        else:
            prices = df.iloc[:, 0]

        returns = np.log(prices / prices.shift(1)).dropna().values
        return returns


class NBeatsForecaster(BaseForecaster):
    """N-BEATS model wrapper with forecast() interface."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        input_size: int = 60,
        output_size: int = 21,
    ):
        super().__init__(model_path, device)
        self.input_size = input_size
        self.output_size = output_size
        self.model_name = "nbeats"

    def load_model(self) -> bool:
        """Load N-BEATS model from checkpoint."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available, N-BEATS will use fallback")
            self.is_loaded = True
            return True

        try:
            from ml_models.nbeats import NBeats

            self.model = NBeats(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_size=128,
                num_blocks=4,
            ).to(self.device)

            if self.model_path and Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded N-BEATS from {self.model_path}")

            self.model.eval()
            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load N-BEATS: {e}")
            self.is_loaded = True  # Mark as loaded, will use fallback
            return True

    def forecast(
        self,
        price_history: pd.DataFrame,
        horizon_days: int = 10,
    ) -> ForecastResult:
        """Generate N-BEATS forecast."""
        if not self.is_loaded:
            self.load_model()

        # Prepare input returns
        returns = self._prepare_returns(price_history)

        if len(returns) < self.input_size:
            # Pad with zeros if insufficient history
            padding = np.zeros(self.input_size - len(returns))
            returns = np.concatenate([padding, returns])

        # Use last input_size returns
        input_seq = returns[-self.input_size:]

        # Use model if available, otherwise fallback
        if HAS_TORCH and self.model is not None:
            x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                raw_output = self.model(x).squeeze().cpu().numpy()
        else:
            # Fallback: use momentum-based prediction
            momentum = np.mean(returns[-10:]) if len(returns) >= 10 else 0.0
            raw_output = np.array([momentum] * self.output_size)

        # N-BEATS outputs are scaled returns for each future day
        # Take relevant horizon
        horizon_returns = raw_output[:min(horizon_days, len(raw_output))]

        # Aggregate forecast
        cumulative_return = np.sum(horizon_returns)
        daily_vol = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns) if len(returns) > 0 else 0.01
        annualized_vol = daily_vol * np.sqrt(252)

        # Simple confidence based on recent volatility stability
        if len(returns) >= 20:
            vol_stability = 1.0 / (1.0 + np.std(np.abs(returns[-20:])) * 10)
        else:
            vol_stability = 0.5

        return ForecastResult(
            mean_return=float(cumulative_return),
            volatility=float(annualized_vol),
            p10=float(cumulative_return - 1.28 * daily_vol * np.sqrt(horizon_days)),
            p90=float(cumulative_return + 1.28 * daily_vol * np.sqrt(horizon_days)),
            confidence_score=float(vol_stability),
            raw_predictions=raw_output,
            model_name=self.model_name,
            metadata={'fallback': self.model is None},
        )


class LSTMForecaster(BaseForecaster):
    """LSTM model wrapper with forecast() interface."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        input_dim: int = 1,
        hidden_dim: int = 256,
        seq_length: int = 60,
    ):
        super().__init__(model_path, device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.model_name = "lstm"
        self.scaler_mean = 0.0
        self.scaler_std = 1.0
        self.encoder = None
        self.output_layer = None

    def load_model(self) -> bool:
        """Load LSTM model from checkpoint."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available, LSTM will use fallback")
            self.is_loaded = True
            return True

        try:
            from ml_models.lstm_module import LSTMEncoder

            # Create a simple LSTM-based predictor
            self.encoder = LSTMEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                bidirectional=True,
                use_attention=True,
            ).to(self.device)

            # Output layer
            self.output_layer = nn.Linear(self.encoder.output_dim, 21).to(self.device)

            if self.model_path and Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'encoder_state_dict' in checkpoint:
                    self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if 'output_state_dict' in checkpoint:
                    self.output_layer.load_state_dict(checkpoint['output_state_dict'])
                if 'scaler_mean' in checkpoint:
                    self.scaler_mean = checkpoint['scaler_mean']
                    self.scaler_std = checkpoint['scaler_std']
                logger.info(f"Loaded LSTM from {self.model_path}")

            self.encoder.eval()
            self.output_layer.eval()
            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load LSTM: {e}")
            self.is_loaded = True
            return True

    def forecast(
        self,
        price_history: pd.DataFrame,
        horizon_days: int = 10,
    ) -> ForecastResult:
        """Generate LSTM forecast."""
        if not self.is_loaded:
            self.load_model()

        returns = self._prepare_returns(price_history)

        if len(returns) < self.seq_length:
            padding = np.zeros(self.seq_length - len(returns))
            returns = np.concatenate([padding, returns])

        # Normalize
        input_seq = returns[-self.seq_length:]
        input_normalized = (input_seq - self.scaler_mean) / (self.scaler_std + 1e-8)

        # Use model if available
        if HAS_TORCH and self.encoder is not None and self.output_layer is not None:
            # Shape: (batch=1, seq_len, features=1)
            x = torch.tensor(
                input_normalized.reshape(1, -1, 1),
                dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                encoded, _ = self.encoder(x)
                raw_output = self.output_layer(encoded).squeeze().cpu().numpy()

            # De-normalize
            raw_output = raw_output * (self.scaler_std + 1e-8) + self.scaler_mean
        else:
            # Fallback: use simple MA-based prediction
            momentum = np.mean(returns[-10:]) if len(returns) >= 10 else 0.0
            raw_output = np.array([momentum] * 21)

        horizon_returns = raw_output[:min(horizon_days, len(raw_output))]
        cumulative_return = np.sum(horizon_returns)

        daily_vol = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns) if len(returns) > 0 else 0.01
        annualized_vol = daily_vol * np.sqrt(252)

        # Confidence based on prediction variance
        pred_stability = 1.0 / (1.0 + np.std(raw_output) * 5) if len(raw_output) > 0 else 0.5

        return ForecastResult(
            mean_return=float(cumulative_return),
            volatility=float(annualized_vol),
            p10=float(cumulative_return - 1.28 * daily_vol * np.sqrt(horizon_days)),
            p90=float(cumulative_return + 1.28 * daily_vol * np.sqrt(horizon_days)),
            confidence_score=float(pred_stability),
            raw_predictions=raw_output,
            model_name=self.model_name,
            metadata={'fallback': self.encoder is None},
        )


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer wrapper with forecast() interface."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        seq_length: int = 60,
    ):
        super().__init__(model_path, device)
        self.seq_length = seq_length
        self.model_name = "tft"
        self.tft_model = None

    def load_model(self) -> bool:
        """Load TFT model from checkpoint."""
        try:
            # TFT from pytorch_forecasting is complex to load without dataset
            # Use a simplified approach for inference
            if self.model_path and Path(self.model_path).exists():
                # pytorch_forecasting TFT requires specific loading
                from pytorch_forecasting import TemporalFusionTransformer
                try:
                    self.tft_model = TemporalFusionTransformer.load_from_checkpoint(
                        self.model_path
                    )
                    self.tft_model.eval()
                    logger.info(f"Loaded TFT from {self.model_path}")
                except Exception as e:
                    logger.warning(f"Could not load TFT checkpoint: {e}")
                    self.tft_model = None

            self.is_loaded = True
            return True

        except ImportError:
            logger.warning("pytorch_forecasting not available for TFT")
            self.is_loaded = True  # Mark as loaded, will use fallback
            return True
        except Exception as e:
            logger.error(f"Failed to load TFT: {e}")
            return False

    def forecast(
        self,
        price_history: pd.DataFrame,
        horizon_days: int = 10,
    ) -> ForecastResult:
        """Generate TFT forecast."""
        if not self.is_loaded:
            self.load_model()

        returns = self._prepare_returns(price_history)

        # If no TFT model loaded, use simple quantile estimation
        if self.tft_model is None:
            # Fallback: use returns distribution for quantile estimates
            return self._fallback_forecast(returns, horizon_days)

        # TFT proper inference would require dataset preparation
        # For now, use fallback
        return self._fallback_forecast(returns, horizon_days)

    def _fallback_forecast(
        self,
        returns: np.ndarray,
        horizon_days: int
    ) -> ForecastResult:
        """Fallback forecast using simple statistical methods."""
        # Use recent returns for estimation
        recent_returns = returns[-60:] if len(returns) >= 60 else returns

        # Simple momentum-based forecast
        momentum = np.mean(recent_returns[-10:]) if len(recent_returns) >= 10 else 0.0
        mean_return = momentum * horizon_days

        daily_vol = np.std(recent_returns)
        annualized_vol = daily_vol * np.sqrt(252)

        # Quantile estimates
        sorted_returns = np.sort(recent_returns)
        p10_daily = np.percentile(sorted_returns, 10)
        p90_daily = np.percentile(sorted_returns, 90)

        # Scale to horizon
        p10 = p10_daily * horizon_days
        p90 = p90_daily * horizon_days

        return ForecastResult(
            mean_return=float(mean_return),
            volatility=float(annualized_vol),
            p10=float(p10),
            p90=float(p90),
            confidence_score=0.5,  # Lower confidence for fallback
            raw_predictions=np.array([mean_return / horizon_days] * horizon_days),
            model_name=self.model_name,
            metadata={'fallback': True},
        )


class EnsembleForecaster:
    """
    Ensemble of N-BEATS, LSTM, and TFT models.

    Features:
    - Parallel model execution using ThreadPoolExecutor
    - Weighted averaging with configurable weights
    - Model disagreement as uncertainty measure
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_dir: str = "models/ensemble_forecaster",
        device: str = 'cpu',
    ):
        """
        Initialize ensemble forecaster.

        Args:
            config: Configuration dict with model weights and settings
            model_dir: Directory containing model checkpoints
            device: 'cpu' or 'cuda'
        """
        self.config = config or {}
        self.model_dir = Path(model_dir)
        self.device = device

        # Model weights (configurable)
        forecaster_config = self.config.get('forecaster', {})
        self.weights = {
            'nbeats': 0.35,
            'lstm': 0.35,
            'tft': 0.30,
        }

        # Update from config if provided
        if 'models' in forecaster_config:
            for model_cfg in forecaster_config['models']:
                name = model_cfg.get('name', '')
                if name in self.weights and model_cfg.get('enabled', True):
                    self.weights[name] = model_cfg.get('weight', self.weights[name])

        # Initialize models
        self.models: Dict[str, BaseForecaster] = {}
        self._init_models()

    def _init_models(self):
        """Initialize all forecaster models."""
        # N-BEATS
        nbeats_path = self.model_dir / "nbeats.pt"
        self.models['nbeats'] = NBeatsForecaster(
            model_path=str(nbeats_path) if nbeats_path.exists() else None,
            device=self.device,
        )

        # LSTM
        lstm_path = self.model_dir / "lstm.pt"
        self.models['lstm'] = LSTMForecaster(
            model_path=str(lstm_path) if lstm_path.exists() else None,
            device=self.device,
        )

        # TFT
        tft_path = self.model_dir / "tft.pt"
        self.models['tft'] = TFTForecaster(
            model_path=str(tft_path) if tft_path.exists() else None,
            device=self.device,
        )

        logger.info(f"Initialized ensemble with models: {list(self.models.keys())}")

    def forecast(
        self,
        price_history: pd.DataFrame,
        horizon_days: int = 10,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate ensemble forecast.

        Args:
            price_history: DataFrame with OHLCV columns
            horizon_days: Forecast horizon in trading days
            parallel: Whether to run models in parallel

        Returns:
            Dict with:
            - mean_return: Weighted average expected return
            - volatility: Expected volatility
            - p10: 10th percentile
            - p90: 90th percentile
            - confidence_score: Ensemble confidence
            - model_disagreement: Std of model predictions
            - individual_forecasts: Dict of per-model results
        """
        individual_results: Dict[str, ForecastResult] = {}

        if parallel:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        model.forecast, price_history, horizon_days
                    ): name
                    for name, model in self.models.items()
                    if self.weights.get(name, 0) > 0
                }

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result(timeout=30)
                        individual_results[name] = result
                    except Exception as e:
                        logger.error(f"Model {name} failed: {e}")
        else:
            # Sequential execution
            for name, model in self.models.items():
                if self.weights.get(name, 0) > 0:
                    try:
                        result = model.forecast(price_history, horizon_days)
                        individual_results[name] = result
                    except Exception as e:
                        logger.error(f"Model {name} failed: {e}")

        if not individual_results:
            logger.error("All models failed, returning default forecast")
            return self._default_forecast()

        # Aggregate forecasts
        return self._aggregate_forecasts(individual_results)

    def _aggregate_forecasts(
        self,
        results: Dict[str, ForecastResult]
    ) -> Dict[str, Any]:
        """Aggregate individual model forecasts into ensemble result."""
        # Normalize weights for available models
        total_weight = sum(
            self.weights.get(name, 0) for name in results.keys()
        )
        if total_weight == 0:
            total_weight = len(results)  # Equal weights if not configured

        normalized_weights = {
            name: self.weights.get(name, 1.0 / len(results)) / total_weight
            for name in results.keys()
        }

        # Weighted averages
        mean_return = sum(
            r.mean_return * normalized_weights[name]
            for name, r in results.items()
        )

        volatility = sum(
            r.volatility * normalized_weights[name]
            for name, r in results.items()
        )

        p10 = sum(
            r.p10 * normalized_weights[name]
            for name, r in results.items()
        )

        p90 = sum(
            r.p90 * normalized_weights[name]
            for name, r in results.items()
        )

        # Model disagreement (std of predictions)
        returns = [r.mean_return for r in results.values()]
        model_disagreement = float(np.std(returns)) if len(returns) > 1 else 0.0

        # Ensemble confidence (weighted average of confidences, penalized by disagreement)
        base_confidence = sum(
            r.confidence_score * normalized_weights[name]
            for name, r in results.items()
        )
        # Reduce confidence if models disagree significantly
        disagreement_penalty = min(1.0, model_disagreement * 10)
        confidence = max(0.1, base_confidence * (1 - disagreement_penalty * 0.5))

        # Individual forecasts for inspection
        individual_forecasts = {
            name: {
                'mean_return': r.mean_return,
                'volatility': r.volatility,
                'p10': r.p10,
                'p90': r.p90,
                'confidence': r.confidence_score,
            }
            for name, r in results.items()
        }

        return {
            'mean_return': float(mean_return),
            'volatility': float(volatility),
            'p10': float(p10),
            'p90': float(p90),
            'confidence_score': float(confidence),
            'model_disagreement': float(model_disagreement),
            'individual_forecasts': individual_forecasts,
            'weights_used': normalized_weights,
        }

    def _default_forecast(self) -> Dict[str, Any]:
        """Return default forecast when all models fail."""
        return {
            'mean_return': 0.0,
            'volatility': 0.20,
            'p10': -0.05,
            'p90': 0.05,
            'confidence_score': 0.1,
            'model_disagreement': 0.0,
            'individual_forecasts': {},
            'weights_used': {},
            'error': 'All models failed',
        }

    def to_pipeline_forecast(
        self,
        ticker: str,
        result: Dict[str, Any],
        as_of_date: pd.Timestamp,
        horizon_days: int = 10,
    ) -> 'Forecast':
        """
        Convert ensemble result to pipeline Forecast dataclass.

        Args:
            ticker: Stock ticker
            result: Result from forecast()
            as_of_date: Forecast date
            horizon_days: Forecast horizon

        Returns:
            Forecast dataclass instance
        """
        if not HAS_PIPELINE_TYPES:
            raise ImportError("Pipeline types not available")

        return Forecast(
            ticker=ticker,
            as_of_date=as_of_date,
            horizon_days=horizon_days,
            expected_return=result['mean_return'],
            expected_volatility=result['volatility'],
            p10=result['p10'],
            p90=result['p90'],
            model_disagreement=result['model_disagreement'],
            confidence=result['confidence_score'],
            model_weights=result['weights_used'],
            individual_forecasts=result['individual_forecasts'],
        )


def forecast_ticker(
    ticker: str,
    price_history: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    horizon_days: int = 10,
) -> Dict[str, Any]:
    """
    Convenience function to generate forecast for a single ticker.

    Args:
        ticker: Stock ticker
        price_history: Price data
        config: Configuration
        horizon_days: Forecast horizon

    Returns:
        Forecast result dict
    """
    forecaster = EnsembleForecaster(config=config)
    result = forecaster.forecast(price_history, horizon_days)
    result['ticker'] = ticker
    return result


# Module-level singleton for reuse
_global_forecaster: Optional[EnsembleForecaster] = None


def get_forecaster(
    config: Optional[Dict[str, Any]] = None,
    reload: bool = False
) -> EnsembleForecaster:
    """
    Get global forecaster instance (singleton pattern).

    Args:
        config: Configuration dict
        reload: Force reload models

    Returns:
        EnsembleForecaster instance
    """
    global _global_forecaster

    if _global_forecaster is None or reload:
        _global_forecaster = EnsembleForecaster(config=config)

    return _global_forecaster
