"""
Practical ML Ensemble for Stock Forecasting
===========================================
Simple, working ensemble combining LightGBM + Ridge + Momentum

This ensemble:
1. Trains on historical price data
2. Creates technical features
3. Combines multiple models with weighted averaging
4. Provides confidence estimates from model disagreement

Example:
    >>> ensemble = StockEnsemble()
    >>> results = ensemble.fit(price_series)
    >>> forecast = ensemble.predict(price_series)
    >>> print(f"Forecast: ${forecast['forecast_price']:.2f}")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class StockEnsemble:
    """
    Simple ensemble for stock price forecasting.

    Combines:
    - LightGBM (gradient boosting)
    - Ridge Regression (linear baseline)
    - Momentum Model (trend-following)

    Attributes:
        models: Dictionary of trained models
        weights: Dictionary of model weights (based on validation performance)
        is_fitted: Whether ensemble has been trained
    """

    def __init__(self):
        """Initialize ensemble."""
        self.models = {}
        self.weights = {}
        self.is_fitted = False
        self.feature_names = []

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def prepare_features(
        self,
        prices: pd.Series,
        lookback: int = 60,
        forecast_horizon: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features from price series for ML models.

        Features include:
        - Returns (1, 5, 10, 20 day)
        - Volatility (5, 20 day)
        - Moving averages (10, 20, 50 day)
        - RSI (14 day)
        - Volume-based features (if volume available)

        Args:
            prices: Price series (can be DataFrame with 'Close' column or Series)
            lookback: Minimum history needed for features
            forecast_horizon: Days ahead to forecast

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,) - forward returns
        """
        # Handle both Series and DataFrame input
        if isinstance(prices, pd.DataFrame):
            if 'Close' in prices.columns:
                price_series = prices['Close']
            else:
                price_series = prices.iloc[:, 0]
        else:
            price_series = prices

        df = pd.DataFrame({'price': price_series})

        # Returns at different horizons
        for period in [1, 5, 10, 20]:
            df[f'return_{period}d'] = price_series.pct_change(period)

        # Volatility
        df['vol_5d'] = price_series.pct_change().rolling(5).std()
        df['vol_20d'] = price_series.pct_change().rolling(20).std()

        # Moving averages (as percentage from current price)
        for period in [10, 20, 50]:
            ma = price_series.rolling(period).mean()
            df[f'ma_{period}'] = (price_series - ma) / price_series

        # RSI
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Price momentum
        df['momentum'] = price_series / price_series.shift(10) - 1

        # Target: forward return at forecast_horizon
        df['target'] = price_series.shift(-forecast_horizon) / price_series - 1

        # Drop NaN
        df = df.dropna()

        if len(df) < 100:
            raise ValueError(
                f"Insufficient data after feature engineering: {len(df)} samples. "
                f"Need at least 100 samples."
            )

        feature_cols = [c for c in df.columns if c not in ['price', 'target']]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df['target'].values

        return X, y

    def fit(
        self,
        prices: pd.Series,
        lookback: int = 60,
        forecast_horizon: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Train ensemble on price history.

        Args:
            prices: Historical price series
            lookback: Minimum history for features
            forecast_horizon: Days ahead to forecast
            verbose: Print training progress

        Returns:
            Dictionary with training metrics for each model
        """
        if verbose:
            logger.info(f"Training ensemble on {len(prices)} price observations...")

        # Prepare features
        X, y = self.prepare_features(prices, lookback, forecast_horizon)

        # Split train/val (80/20)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        if verbose:
            logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

        # Model 1: LightGBM
        if verbose:
            logger.info("Training LightGBM...")

        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_val)
        lgb_rmse = np.sqrt(np.mean((lgb_pred - y_val)**2))
        lgb_mae = np.mean(np.abs(lgb_pred - y_val))
        lgb_dir_acc = np.mean(np.sign(lgb_pred) == np.sign(y_val))

        self.models['lgb'] = lgb_model

        if verbose:
            logger.info(f"  LightGBM - RMSE: {lgb_rmse:.4f}, "
                       f"MAE: {lgb_mae:.4f}, Dir Acc: {lgb_dir_acc:.2%}")

        # Model 2: Ridge Regression
        if verbose:
            logger.info("Training Ridge Regression...")

        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_train, y_train)
        ridge_pred = ridge_model.predict(X_val)
        ridge_rmse = np.sqrt(np.mean((ridge_pred - y_val)**2))
        ridge_mae = np.mean(np.abs(ridge_pred - y_val))
        ridge_dir_acc = np.mean(np.sign(ridge_pred) == np.sign(y_val))

        self.models['ridge'] = ridge_model

        if verbose:
            logger.info(f"  Ridge - RMSE: {ridge_rmse:.4f}, "
                       f"MAE: {ridge_mae:.4f}, Dir Acc: {ridge_dir_acc:.2%}")

        # Model 3: Momentum Model (simple baseline)
        if verbose:
            logger.info("Creating Momentum Model...")

        class MomentumModel:
            """Simple momentum-based forecasting."""

            def __init__(self):
                self.mean_return = 0.0

            def fit(self, X, y):
                # Use recent return trend
                self.mean_return = np.mean(y)
                return self

            def predict(self, X):
                # Predict based on recent momentum (last feature)
                if X.shape[1] > 0:
                    # Use momentum feature (last column typically)
                    return X[:, -1] * 0.5  # Damped momentum
                return np.full(len(X), self.mean_return)

        momentum_model = MomentumModel()
        momentum_model.fit(X_train, y_train)
        momentum_pred = momentum_model.predict(X_val)
        momentum_rmse = np.sqrt(np.mean((momentum_pred - y_val)**2))
        momentum_mae = np.mean(np.abs(momentum_pred - y_val))
        momentum_dir_acc = np.mean(np.sign(momentum_pred) == np.sign(y_val))

        self.models['momentum'] = momentum_model

        if verbose:
            logger.info(f"  Momentum - RMSE: {momentum_rmse:.4f}, "
                       f"MAE: {momentum_mae:.4f}, Dir Acc: {momentum_dir_acc:.2%}")

        # Calculate weights based on inverse RMSE
        rmse_scores = {
            'lgb': lgb_rmse,
            'ridge': ridge_rmse,
            'momentum': momentum_rmse
        }

        # Inverse RMSE weighting
        total_inv_rmse = sum([1/rmse for rmse in rmse_scores.values()])
        self.weights = {
            name: (1/rmse) / total_inv_rmse
            for name, rmse in rmse_scores.items()
        }

        if verbose:
            logger.info("Ensemble weights:")
            for name, weight in self.weights.items():
                logger.info(f"  {name}: {weight:.3f}")

        self.is_fitted = True

        return {
            'lgb_rmse': lgb_rmse,
            'lgb_mae': lgb_mae,
            'lgb_dir_acc': lgb_dir_acc,
            'ridge_rmse': ridge_rmse,
            'ridge_mae': ridge_mae,
            'ridge_dir_acc': ridge_dir_acc,
            'momentum_rmse': momentum_rmse,
            'momentum_mae': momentum_mae,
            'momentum_dir_acc': momentum_dir_acc,
            'weights': self.weights,
            'n_train': len(X_train),
            'n_val': len(X_val)
        }

    def predict(
        self,
        prices: pd.Series,
        return_confidence: bool = True
    ) -> Dict:
        """
        Generate forecast with confidence estimation.

        Args:
            prices: Recent price history
            return_confidence: Whether to calculate confidence intervals

        Returns:
            Dictionary containing:
            - current_price: Latest price
            - forecast_price: Predicted price
            - forecast_return: Predicted return
            - lower_bound: 95% CI lower bound
            - upper_bound: 95% CI upper bound
            - confidence: Model confidence (0-1)
            - model_predictions: Individual model predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Handle both Series and DataFrame
        if isinstance(prices, pd.DataFrame):
            if 'Close' in prices.columns:
                price_series = prices['Close']
            else:
                price_series = prices.iloc[:, 0]
        else:
            price_series = prices

        # Prepare features from recent data
        df = pd.DataFrame({'price': price_series})

        # Calculate same features as training
        for period in [1, 5, 10, 20]:
            df[f'return_{period}d'] = price_series.pct_change(period)

        df['vol_5d'] = price_series.pct_change().rolling(5).std()
        df['vol_20d'] = price_series.pct_change().rolling(20).std()

        for period in [10, 20, 50]:
            ma = price_series.rolling(period).mean()
            df[f'ma_{period}'] = (price_series - ma) / price_series

        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['momentum'] = price_series / price_series.shift(10) - 1

        df = df.dropna()

        if len(df) == 0:
            raise ValueError("Insufficient data for prediction after feature engineering")

        # Get latest features
        feature_cols = [c for c in df.columns if c != 'price']
        X = df[feature_cols].iloc[-1:].values

        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            predictions[name] = pred

        # Ensemble prediction (weighted average)
        ensemble_pred = sum([
            pred * self.weights[name]
            for name, pred in predictions.items()
        ])

        # Estimate uncertainty from model disagreement
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)

        # Confidence: higher when models agree, lower when they disagree
        # Scale std by mean to get relative disagreement
        relative_std = pred_std / (abs(pred_mean) + 1e-6)
        confidence = 1.0 / (1.0 + relative_std * 5)  # Scale factor
        confidence = np.clip(confidence, 0.0, 1.0)

        # Calculate confidence intervals (using prediction std)
        current_price = price_series.iloc[-1]
        forecast_price = current_price * (1 + ensemble_pred)
        lower_bound = current_price * (1 + ensemble_pred - 1.96 * pred_std)
        upper_bound = current_price * (1 + ensemble_pred + 1.96 * pred_std)

        return {
            'current_price': current_price,
            'forecast_price': forecast_price,
            'forecast_return': ensemble_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidence,
            'uncertainty': pred_std,
            'model_predictions': predictions,
            'model_weights': self.weights
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from LightGBM model.

        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        if 'lgb' not in self.models:
            raise ValueError("LightGBM model not available")

        lgb_model = self.models['lgb']
        importance = lgb_model.feature_importances_

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df


def generate_trading_signal(
    forecast: Dict,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
    min_confidence: float = 0.5
) -> str:
    """
    Generate trading signal from forecast.

    Args:
        forecast: Forecast dictionary from ensemble.predict()
        buy_threshold: Minimum return for buy signal (default 2%)
        sell_threshold: Maximum return for sell signal (default -2%)
        min_confidence: Minimum confidence to take position

    Returns:
        Trading signal string
    """
    return_forecast = forecast['forecast_return']
    confidence = forecast['confidence']

    if confidence < min_confidence:
        return "HOLD (Low Confidence)"

    if return_forecast > buy_threshold:
        if confidence > 0.7:
            return "STRONG BUY"
        else:
            return "BUY"
    elif return_forecast < sell_threshold:
        if confidence > 0.7:
            return "STRONG SELL"
        else:
            return "SELL"
    else:
        return "HOLD (Neutral)"
