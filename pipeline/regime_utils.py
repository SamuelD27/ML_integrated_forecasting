"""
Regime Utilities
================
Pipeline wrapper for regime classification.

Provides:
- Feature computation from market data
- Regime classification with smoothing
- Regime-based parameter adjustment
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import regime classifier
try:
    from ml_models.regime_classifier import RegimeDetectorNN, REGIME_CLASSES
    HAS_REGIME_MODEL = True
except ImportError:
    HAS_REGIME_MODEL = False
    logger.warning("RegimeDetectorNN not available")
    REGIME_CLASSES = {0: 'bull', 1: 'bear', 2: 'neutral', 3: 'crisis'}

# Import macro loader
try:
    from data.macro_loader import load_macro_features, get_current_macro_snapshot
    HAS_MACRO_LOADER = True
except ImportError:
    HAS_MACRO_LOADER = False
    logger.warning("Macro loader not available")


# Default regime mapping
REGIME_INT_MAP = {
    'bull': 0,
    'bear': 1,
    'neutral': 2,
    'crisis': 3,
}

REGIME_LABEL_MAP = {v: k for k, v in REGIME_INT_MAP.items()}


class RegimeDetector:
    """
    Pipeline-aware regime detection.

    Combines macro data, price features, and neural network classifier
    to determine market regime with configurable smoothing.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        device: str = 'cpu',
    ):
        """
        Initialize regime detector.

        Args:
            config: Trading config (or config['regime'] section)
            model_path: Path to trained regime classifier
            device: 'cpu' or 'cuda'
        """
        self.config = config or {}
        self.device = device

        # Extract regime-specific config
        if 'regime' in self.config:
            self.regime_config = self.config['regime']
        else:
            self.regime_config = self.config

        # Model path
        if model_path is None:
            model_path = self.regime_config.get(
                'model_path',
                'models/regime_classifier/regime_model.pt'
            )

        # Initialize model
        self._model = None
        self._model_path = model_path
        self._feature_stats = None

        # Smoothing parameters
        self.min_persistence = self.regime_config.get('min_persistence_days', 3)
        self.smoothing_window = self.regime_config.get('smoothing_window', 5)

        # Regime history for smoothing
        self._regime_history: List[int] = []
        self._last_confirmed_regime: Optional[int] = None

    @property
    def model(self) -> Optional['RegimeDetectorNN']:
        """Lazy-load the regime model."""
        if self._model is None and HAS_REGIME_MODEL:
            model_path = Path(self._model_path)
            if model_path.exists():
                try:
                    self._model = RegimeDetectorNN(
                        model_path=str(model_path),
                        device=self.device
                    )
                    logger.info(f"Loaded regime model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load regime model: {e}")
                    self._model = RegimeDetectorNN(device=self.device)
            else:
                logger.warning(f"Regime model not found at {model_path}, using untrained model")
                self._model = RegimeDetectorNN(device=self.device)
        return self._model

    def compute_regime_features(
        self,
        price_data: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute features required for regime classification.

        Args:
            price_data: DataFrame with OHLCV columns (indexed by date)
            macro_data: Optional macro features from macro_loader

        Returns:
            DataFrame with regime classification features:
            - vol_20d: 20-day volatility
            - ret_20d: 20-day return
            - ret_60d: 60-day return
            - vol_ratio: Current vol / 60-day vol
            - rsi: 14-day RSI
            - macd: MACD line
            - adx: Average Directional Index
            - volume_ratio: Current volume / 20-day avg
            - price_momentum: Price / 20-day MA
            - volatility_trend: Volatility trend indicator
        """
        df = price_data.copy()

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Ensure we have required columns
        close = df['close'] if 'close' in df.columns else df['adj close']
        volume = df.get('volume', pd.Series(1, index=df.index))

        # Calculate returns
        returns = np.log(close / close.shift(1))

        # Volatility features
        vol_20d = returns.rolling(20, min_periods=5).std() * np.sqrt(252)
        vol_60d = returns.rolling(60, min_periods=10).std() * np.sqrt(252)
        vol_ratio = vol_20d / vol_60d.clip(lower=0.01)

        # Return features
        ret_20d = (close / close.shift(20) - 1).fillna(0)
        ret_60d = (close / close.shift(60) - 1).fillna(0)

        # RSI
        rsi = self._compute_rsi(close, period=14)

        # MACD
        macd = self._compute_macd(close)

        # ADX
        if 'high' in df.columns and 'low' in df.columns:
            adx = self._compute_adx(df['high'], df['low'], close)
        else:
            adx = pd.Series(25.0, index=df.index)  # Default moderate trend

        # Volume ratio
        volume_ma20 = volume.rolling(20, min_periods=5).mean()
        volume_ratio = volume / volume_ma20.clip(lower=1)

        # Price momentum
        price_ma20 = close.rolling(20, min_periods=5).mean()
        price_momentum = close / price_ma20.clip(lower=0.01)

        # Volatility trend
        vol_ma20 = vol_20d.rolling(20, min_periods=5).mean()
        volatility_trend = vol_20d / vol_ma20.clip(lower=0.01)

        # Build feature DataFrame
        features = pd.DataFrame({
            'vol_20d': vol_20d,
            'ret_20d': ret_20d,
            'ret_60d': ret_60d,
            'vol_ratio': vol_ratio,
            'rsi': rsi,
            'macd': macd,
            'adx': adx,
            'volume_ratio': volume_ratio,
            'price_momentum': price_momentum,
            'volatility_trend': volatility_trend,
        }, index=df.index)

        # Fill NaN with neutral values
        features = features.fillna({
            'vol_20d': 0.20,
            'ret_20d': 0.0,
            'ret_60d': 0.0,
            'vol_ratio': 1.0,
            'rsi': 50.0,
            'macd': 0.0,
            'adx': 25.0,
            'volume_ratio': 1.0,
            'price_momentum': 1.0,
            'volatility_trend': 1.0,
        })

        return features

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss.clip(lower=1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _compute_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.Series:
        """Compute MACD line (normalized)."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = (ema_fast - ema_slow) / prices.clip(lower=0.01)
        return macd_line.fillna(0)

    def _compute_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Compute ADX (Average Directional Index)."""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smoothed averages
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.clip(lower=1e-10)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.clip(lower=1e-10)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).clip(lower=1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx.fillna(25)

    def classify_regime(
        self,
        features: Union[pd.Series, np.ndarray, Dict[str, float]],
        apply_smoothing: bool = True,
    ) -> Dict[str, Any]:
        """
        Classify market regime from features.

        Args:
            features: Feature array (10 values), Series, or dict
            apply_smoothing: Whether to apply regime persistence smoothing

        Returns:
            Dict with:
            - regime: str ('bull', 'bear', 'neutral', 'crisis')
            - regime_int: int (0, 1, 2, 3)
            - confidence: float (0-1)
            - probabilities: dict of class probabilities
            - smoothed: bool (whether smoothing was applied)
        """
        # Convert features to numpy array
        if isinstance(features, dict):
            feature_array = np.array([
                features.get('vol_20d', 0.20),
                features.get('ret_20d', 0.0),
                features.get('ret_60d', 0.0),
                features.get('vol_ratio', 1.0),
                features.get('rsi', 50.0),
                features.get('macd', 0.0),
                features.get('adx', 25.0),
                features.get('volume_ratio', 1.0),
                features.get('price_momentum', 1.0),
                features.get('volatility_trend', 1.0),
            ])
        elif isinstance(features, pd.Series):
            feature_array = features.values[:10]
        else:
            feature_array = np.asarray(features)[:10]

        # Use model if available
        if self.model is not None:
            result = self.model.predict(
                feature_array,
                normalize=True,
                return_confidence=True
            )
            raw_regime = result['class_id']
            confidence = result['confidence']
            probabilities = result.get('probabilities', {})
        else:
            # Fallback: rule-based classification
            raw_regime, confidence, probabilities = self._rule_based_classify(feature_array)

        # Apply smoothing if requested
        if apply_smoothing:
            smoothed_regime = self._apply_smoothing(raw_regime)
        else:
            smoothed_regime = raw_regime

        return {
            'regime': REGIME_LABEL_MAP[smoothed_regime],
            'regime_int': smoothed_regime,
            'raw_regime_int': raw_regime,
            'confidence': confidence,
            'probabilities': probabilities,
            'smoothed': apply_smoothing and smoothed_regime != raw_regime,
        }

    def _rule_based_classify(
        self,
        features: np.ndarray
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Rule-based fallback classifier.

        Uses simple thresholds on volatility and returns.
        """
        vol_20d = features[0]
        ret_20d = features[1]
        ret_60d = features[2]
        rsi = features[4]
        adx = features[6]

        # Crisis detection (high volatility + negative returns)
        if vol_20d > 0.40 and ret_20d < -0.10:
            regime = 3  # Crisis
            confidence = min(0.9, vol_20d)
        # Bear market (negative trend)
        elif ret_60d < -0.15 and rsi < 40:
            regime = 1  # Bear
            confidence = 0.7
        # Bull market (positive trend)
        elif ret_60d > 0.10 and rsi > 50 and adx > 20:
            regime = 0  # Bull
            confidence = 0.7
        # Neutral
        else:
            regime = 2  # Neutral
            confidence = 0.5

        # Simple probability distribution
        probs = {k: 0.1 for k in REGIME_CLASSES.values()}
        probs[REGIME_CLASSES[regime]] = confidence
        remaining = 1.0 - confidence
        for k in probs:
            if k != REGIME_CLASSES[regime]:
                probs[k] = remaining / 3

        return regime, confidence, probs

    def _apply_smoothing(self, raw_regime: int) -> int:
        """
        Apply regime persistence smoothing.

        Prevents rapid regime switching by requiring minimum persistence.
        """
        # Add to history
        self._regime_history.append(raw_regime)

        # Keep only recent history
        if len(self._regime_history) > self.smoothing_window:
            self._regime_history = self._regime_history[-self.smoothing_window:]

        # Count regime occurrences in window
        regime_counts = {}
        for r in self._regime_history:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        # Check if current regime has persisted enough
        if len(self._regime_history) >= self.min_persistence:
            # Find most common regime
            most_common = max(regime_counts, key=regime_counts.get)
            most_common_count = regime_counts[most_common]

            # If most common has minimum persistence, update confirmed regime
            if most_common_count >= self.min_persistence:
                self._last_confirmed_regime = most_common

        # Return confirmed regime (or raw if no confirmed yet)
        if self._last_confirmed_regime is not None:
            return self._last_confirmed_regime
        return raw_regime

    def reset_smoothing(self):
        """Reset smoothing state."""
        self._regime_history = []
        self._last_confirmed_regime = None


def classify_regime(
    as_of_date: Union[str, pd.Timestamp],
    price_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classify market regime for a given date.

    Convenience function that creates a RegimeDetector and classifies.

    Args:
        as_of_date: Date for classification
        price_data: Price data (will be fetched if None)
        config: Configuration dict

    Returns:
        Regime classification result dict
    """
    as_of_date = pd.Timestamp(as_of_date)

    # Fetch price data if not provided
    if price_data is None:
        try:
            import yfinance as yf
            start = as_of_date - timedelta(days=120)
            price_data = yf.download(
                'SPY',
                start=start.strftime('%Y-%m-%d'),
                end=(as_of_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
        except Exception as e:
            logger.error(f"Could not fetch price data: {e}")
            return {
                'regime': 'neutral',
                'regime_int': 2,
                'confidence': 0.5,
                'error': str(e)
            }

    # Create detector and classify
    detector = RegimeDetector(config=config)
    features = detector.compute_regime_features(price_data)

    if len(features) == 0:
        return {
            'regime': 'neutral',
            'regime_int': 2,
            'confidence': 0.5,
            'error': 'No features computed'
        }

    # Use latest features
    latest_features = features.iloc[-1]

    return detector.classify_regime(latest_features, apply_smoothing=False)


def get_regime_parameters(
    regime: Union[str, int],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get regime-specific parameters.

    Args:
        regime: Regime name or integer
        config: Configuration dict with regime_params

    Returns:
        Dict with:
        - risk_multiplier
        - position_size_multiplier
        - options_enabled
    """
    # Convert to string if int
    if isinstance(regime, int):
        regime = REGIME_LABEL_MAP.get(regime, 'neutral')

    # Default parameters
    defaults = {
        'bull': {
            'risk_multiplier': 1.0,
            'position_size_multiplier': 1.0,
            'options_enabled': True,
        },
        'bear': {
            'risk_multiplier': 0.5,
            'position_size_multiplier': 0.5,
            'options_enabled': True,
        },
        'neutral': {
            'risk_multiplier': 0.8,
            'position_size_multiplier': 0.8,
            'options_enabled': True,
        },
        'crisis': {
            'risk_multiplier': 0.2,
            'position_size_multiplier': 0.2,
            'options_enabled': False,
        },
    }

    # Get from config if available
    if config:
        regime_config = config.get('regime', config)
        params = regime_config.get('regime_params', {})
        regime_params = params.get(regime.capitalize(), defaults.get(regime, {}))
        if regime_params:
            return regime_params

    return defaults.get(regime, defaults['neutral'])


def compute_regime_time_series(
    price_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Compute regime classification for entire time series.

    Args:
        price_data: DataFrame with OHLCV data
        config: Configuration dict

    Returns:
        DataFrame with columns:
        - regime: str
        - regime_int: int
        - confidence: float
        Plus all computed features
    """
    detector = RegimeDetector(config=config)
    features = detector.compute_regime_features(price_data)

    results = []
    detector.reset_smoothing()  # Start fresh

    for idx, row in features.iterrows():
        result = detector.classify_regime(row, apply_smoothing=True)
        results.append({
            'date': idx,
            'regime': result['regime'],
            'regime_int': result['regime_int'],
            'confidence': result['confidence'],
        })

    result_df = pd.DataFrame(results).set_index('date')

    # Merge with features
    return pd.concat([features, result_df], axis=1)
