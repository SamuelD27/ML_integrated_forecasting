"""
Meta-Labeling Features
=====================
Combines forecast, regime, and market features for the meta-model.

The meta-model predicts whether a primary forecast will be profitable,
using features from:
1. Forecast characteristics (return, vol, model disagreement)
2. Regime information (current regime, regime features)
3. Market context (VIX, recent returns, volatility)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Try to import pipeline types
try:
    from pipeline.core_types import Forecast, TradeSignal
    HAS_PIPELINE_TYPES = True
except ImportError:
    HAS_PIPELINE_TYPES = False


@dataclass
class MetaFeatures:
    """
    Feature vector for meta-model prediction.

    Combines features from forecast, regime, and market data
    into a unified representation for meta-model input.

    Attributes:
        forecast_features: Features from the ensemble forecast
        regime_features: Features from regime classification
        market_features: Features from market data
        feature_names: List of feature names in order
        values: Numpy array of feature values
    """
    forecast_features: Dict[str, float] = field(default_factory=dict)
    regime_features: Dict[str, float] = field(default_factory=dict)
    market_features: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    values: Optional[np.ndarray] = None

    def __post_init__(self):
        """Build feature vector from components."""
        self._build_feature_vector()

    def _build_feature_vector(self):
        """Combine all features into single vector."""
        all_features = {}

        # Add forecast features with prefix
        for k, v in self.forecast_features.items():
            all_features[f'forecast_{k}'] = v

        # Add regime features with prefix
        for k, v in self.regime_features.items():
            all_features[f'regime_{k}'] = v

        # Add market features with prefix
        for k, v in self.market_features.items():
            all_features[f'market_{k}'] = v

        # Build ordered feature names and values
        self.feature_names = sorted(all_features.keys())
        self.values = np.array([all_features[name] for name in self.feature_names])

    def to_array(self) -> np.ndarray:
        """Return feature values as numpy array."""
        return self.values

    def to_dict(self) -> Dict[str, float]:
        """Return features as dictionary."""
        return dict(zip(self.feature_names, self.values))


def extract_forecast_features(
    forecast: Union['Forecast', Dict[str, Any]],
) -> Dict[str, float]:
    """
    Extract features from ensemble forecast.

    Args:
        forecast: Forecast dataclass or dict with forecast data

    Returns:
        Dict of forecast features
    """
    if hasattr(forecast, 'expected_return'):
        # Forecast dataclass
        return {
            'expected_return': float(forecast.expected_return),
            'volatility': float(forecast.volatility),
            'p10': float(forecast.p10),
            'p90': float(forecast.p90),
            'forecast_range': float(forecast.p90 - forecast.p10),
            'model_disagreement': float(forecast.model_disagreement),
            'confidence': float(forecast.confidence),
            'return_vol_ratio': float(
                forecast.expected_return / max(forecast.volatility, 0.01)
            ),
        }
    else:
        # Dict
        expected_return = forecast.get('expected_return', forecast.get('mean_return', 0.0))
        volatility = forecast.get('volatility', forecast.get('expected_volatility', 0.20))
        p10 = forecast.get('p10', expected_return - volatility)
        p90 = forecast.get('p90', expected_return + volatility)

        return {
            'expected_return': float(expected_return),
            'volatility': float(volatility),
            'p10': float(p10),
            'p90': float(p90),
            'forecast_range': float(p90 - p10),
            'model_disagreement': float(forecast.get('model_disagreement', 0.0)),
            'confidence': float(forecast.get('confidence', forecast.get('confidence_score', 0.5))),
            'return_vol_ratio': float(
                expected_return / max(volatility, 0.01)
            ),
        }


def extract_regime_features(
    regime: int,
    regime_label: str,
    regime_confidence: float = 0.5,
) -> Dict[str, float]:
    """
    Extract features from regime classification.

    Args:
        regime: Regime integer (0-3)
        regime_label: Regime name
        regime_confidence: Classification confidence

    Returns:
        Dict of regime features
    """
    return {
        'regime_int': float(regime),
        'regime_bull': 1.0 if regime == 0 else 0.0,
        'regime_bear': 1.0 if regime == 1 else 0.0,
        'regime_neutral': 1.0 if regime == 2 else 0.0,
        'regime_crisis': 1.0 if regime == 3 else 0.0,
        'regime_confidence': float(regime_confidence),
        # Risk adjustment based on regime
        'regime_risk_mult': {0: 1.0, 1: 0.5, 2: 0.8, 3: 0.2}.get(regime, 0.5),
    }


def extract_market_features(
    price_history: Optional[pd.DataFrame] = None,
    macro_data: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Extract market context features.

    Args:
        price_history: Recent price data for the ticker
        macro_data: Macro data (VIX, yields, etc.)

    Returns:
        Dict of market features
    """
    features = {}

    # Market features from price history
    if price_history is not None and len(price_history) > 0:
        # Normalize column names
        df = price_history.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df.get('close', df.get('adj close', df.iloc[:, 0]))
        returns = np.log(close / close.shift(1)).dropna()

        if len(returns) >= 5:
            features['recent_returns_5d'] = float(returns.iloc[-5:].sum())
        else:
            features['recent_returns_5d'] = 0.0

        if len(returns) >= 20:
            features['recent_returns_20d'] = float(returns.iloc[-20:].sum())
            features['realized_vol_20d'] = float(returns.iloc[-20:].std() * np.sqrt(252))
        else:
            features['recent_returns_20d'] = 0.0
            features['realized_vol_20d'] = 0.20

        if len(returns) >= 60:
            features['realized_vol_60d'] = float(returns.iloc[-60:].std() * np.sqrt(252))
        else:
            features['realized_vol_60d'] = features.get('realized_vol_20d', 0.20)

        # Price position relative to moving averages
        if len(close) >= 20:
            ma20 = close.rolling(20).mean().iloc[-1]
            features['price_vs_ma20'] = float(close.iloc[-1] / ma20 - 1)
        else:
            features['price_vs_ma20'] = 0.0

        if len(close) >= 50:
            ma50 = close.rolling(50).mean().iloc[-1]
            features['price_vs_ma50'] = float(close.iloc[-1] / ma50 - 1)
        else:
            features['price_vs_ma50'] = features.get('price_vs_ma20', 0.0)

    else:
        features.update({
            'recent_returns_5d': 0.0,
            'recent_returns_20d': 0.0,
            'realized_vol_20d': 0.20,
            'realized_vol_60d': 0.20,
            'price_vs_ma20': 0.0,
            'price_vs_ma50': 0.0,
        })

    # Macro features
    if macro_data:
        features['vix_level'] = float(macro_data.get('vix', 20.0))
        features['vix_change'] = float(macro_data.get('vix_change', 0.0))
        features['vix_percentile'] = float(macro_data.get('vix_percentile', 0.5))
        features['yield_spread'] = float(macro_data.get('yield_spread', 0.5))
        features['spy_return'] = float(macro_data.get('spy_return', 0.0))
        features['spy_vol_20d'] = float(macro_data.get('spy_vol_20d', 0.15))
    else:
        features.update({
            'vix_level': 20.0,
            'vix_change': 0.0,
            'vix_percentile': 0.5,
            'yield_spread': 0.5,
            'spy_return': 0.0,
            'spy_vol_20d': 0.15,
        })

    return features


def build_meta_features(
    forecast: Union['Forecast', Dict[str, Any]],
    regime: int,
    regime_label: str,
    regime_confidence: float = 0.5,
    price_history: Optional[pd.DataFrame] = None,
    macro_data: Optional[Dict[str, float]] = None,
) -> MetaFeatures:
    """
    Build complete meta-feature vector.

    Combines forecast, regime, and market features into a single
    MetaFeatures object for meta-model input.

    Args:
        forecast: Forecast dataclass or dict
        regime: Regime integer
        regime_label: Regime name
        regime_confidence: Regime classification confidence
        price_history: Recent price data
        macro_data: Macro data dict

    Returns:
        MetaFeatures object with all features
    """
    forecast_features = extract_forecast_features(forecast)
    regime_features = extract_regime_features(regime, regime_label, regime_confidence)
    market_features = extract_market_features(price_history, macro_data)

    return MetaFeatures(
        forecast_features=forecast_features,
        regime_features=regime_features,
        market_features=market_features,
    )


def build_meta_features_from_signal(
    signal: 'TradeSignal',
    price_history: Optional[pd.DataFrame] = None,
    macro_data: Optional[Dict[str, float]] = None,
) -> MetaFeatures:
    """
    Build meta-features from a TradeSignal object.

    Args:
        signal: TradeSignal dataclass
        price_history: Recent price data
        macro_data: Macro data dict

    Returns:
        MetaFeatures object
    """
    if not HAS_PIPELINE_TYPES:
        raise ImportError("Pipeline types not available")

    # Extract forecast features from signal
    if signal.forecast is not None:
        forecast_features = extract_forecast_features(signal.forecast)
    else:
        forecast_features = {
            'expected_return': float(signal.expected_return),
            'volatility': float(signal.expected_vol),
            'p10': float(signal.expected_return - signal.expected_vol),
            'p90': float(signal.expected_return + signal.expected_vol),
            'forecast_range': float(2 * signal.expected_vol),
            'model_disagreement': 0.0,
            'confidence': 0.5,
            'return_vol_ratio': float(
                signal.expected_return / max(signal.expected_vol, 0.01)
            ),
        }

    regime_features = extract_regime_features(
        signal.regime,
        signal.regime_label,
        regime_confidence=0.5  # Default if not available
    )

    market_features = extract_market_features(price_history, macro_data)

    return MetaFeatures(
        forecast_features=forecast_features,
        regime_features=regime_features,
        market_features=market_features,
    )


def get_feature_names() -> List[str]:
    """
    Get ordered list of all meta-feature names.

    Returns:
        List of feature names in consistent order
    """
    # Build dummy features to get names
    dummy_features = build_meta_features(
        forecast={'expected_return': 0, 'volatility': 0, 'p10': 0, 'p90': 0},
        regime=0,
        regime_label='Bull',
    )
    return dummy_features.feature_names


def prepare_training_features(
    forecasts: List[Dict[str, Any]],
    regimes: List[int],
    price_histories: Optional[List[pd.DataFrame]] = None,
    macro_data_list: Optional[List[Dict[str, float]]] = None,
) -> np.ndarray:
    """
    Prepare feature matrix for meta-model training.

    Args:
        forecasts: List of forecast dicts
        regimes: List of regime integers
        price_histories: Optional list of price DataFrames
        macro_data_list: Optional list of macro data dicts

    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    n = len(forecasts)

    if price_histories is None:
        price_histories = [None] * n
    if macro_data_list is None:
        macro_data_list = [None] * n

    features_list = []

    for forecast, regime, prices, macro in zip(
        forecasts, regimes, price_histories, macro_data_list
    ):
        meta_features = build_meta_features(
            forecast=forecast,
            regime=regime,
            regime_label={0: 'Bull', 1: 'Bear', 2: 'Neutral', 3: 'Crisis'}.get(regime, 'Neutral'),
            price_history=prices,
            macro_data=macro,
        )
        features_list.append(meta_features.to_array())

    return np.array(features_list)
