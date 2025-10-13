"""
ML Stock Selection Model
=========================

LightGBM/XGBoost-based stock ranking for cross-sectional selection.
Features → Ranking scores → Top-N selection with time-series CV.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
import warnings

warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class MLStockSelector:
    """ML-based stock selection using ranking models."""

    def __init__(self, model_type: str = 'lightgbm',
                 n_cv_splits: int = 2,
                 lookback_days: int = 252):
        """
        Initialize ML stock selector.

        Parameters
        ----------
        model_type : str
            'lightgbm' or 'xgboost' or 'ensemble'
        n_cv_splits : int
            Number of time-series CV splits
        lookback_days : int
            Lookback period for training data
        """
        self.model_type = model_type
        self.n_cv_splits = n_cv_splits
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.cv_scores = []

    def train(self, features_df: pd.DataFrame, returns_df: pd.DataFrame,
              forward_periods: int = 20) -> Dict:
        """
        Train ranking model with time-series cross-validation.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix with columns: date, ticker, feature1, feature2, ...
        returns_df : pd.DataFrame
            Forward returns with tickers as columns, dates as index
        forward_periods : int
            Forward-looking period for returns (default: 20 days = 1 month)

        Returns
        -------
        dict
            Training metrics and diagnostics
        """
        print(f"\n[ML Selection] Training {self.model_type} model...")

        # Prepare training data
        X, y, dates, tickers = self._prepare_training_data(
            features_df, returns_df, forward_periods
        )

        if len(X) < 20:
            print("  Warning: Insufficient training data. Need more history.")
            return {'status': 'insufficient_data'}

        print(f"  Training samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
                model = self._train_lightgbm(X_train, y_train, X_val, y_val)
            elif self.model_type == 'xgboost' and HAS_XGBOOST:
                model = self._train_xgboost(X_train, y_train, X_val, y_val)
            else:
                # Fallback to simple model
                model = self._train_simple(X_train, y_train)

            # Evaluate
            val_pred = model.predict(X_val)

            # Calculate NDCG (Normalized Discounted Cumulative Gain) for ranking
            # Group by date and calculate NDCG per period
            val_dates = dates[val_idx]
            unique_dates = np.unique(val_dates)

            date_ndcgs = []
            for date in unique_dates:
                date_mask = val_dates == date
                if date_mask.sum() > 1:  # Need at least 2 stocks to rank
                    date_y_true = y_val[date_mask].reshape(1, -1)
                    date_y_pred = val_pred[date_mask].reshape(1, -1)
                    try:
                        ndcg = ndcg_score(date_y_true, date_y_pred)
                        date_ndcgs.append(ndcg)
                    except Exception:
                        pass

            fold_ndcg = np.mean(date_ndcgs) if date_ndcgs else 0.5
            cv_scores.append(fold_ndcg)

            print(f"    Fold {fold+1}/{self.n_cv_splits}: NDCG = {fold_ndcg:.4f}")

            # Keep final model
            if fold == self.n_cv_splits - 1:
                self.model = model

        self.cv_scores = cv_scores
        avg_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        print(f"  ✓ CV NDCG: {avg_score:.4f} ± {std_score:.4f}")

        # Extract feature importance
        self._extract_feature_importance()

        return {
            'status': 'success',
            'cv_score_mean': avg_score,
            'cv_score_std': std_score,
            'cv_scores': cv_scores,
            'n_samples': len(X),
            'n_features': X.shape[1],
        }

    def rank_stocks(self, features_df: pd.DataFrame,
                   top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Rank stocks based on ML model predictions.

        Parameters
        ----------
        features_df : pd.DataFrame
            Latest features for all stocks
        top_n : int, optional
            Return only top N stocks

        Returns
        -------
        pd.DataFrame
            Ranked stocks with scores
        """
        if self.model is None:
            print("  Warning: Model not trained. Returning equal scores.")
            scores_df = features_df[['ticker']].copy()
            scores_df['ml_score'] = 0.5
            scores_df['ml_rank'] = scores_df.index + 1
            return scores_df

        # Prepare features
        X = self._extract_features(features_df)

        if len(X) == 0:
            print("  Warning: No valid features to score.")
            return pd.DataFrame()

        # Normalize
        X_scaled = self.scaler.transform(X)

        # Predict
        scores = self.model.predict(X_scaled)

        # Create results
        results = pd.DataFrame({
            'ticker': features_df['ticker'].values,
            'ml_score': scores
        })

        # Rank by score (higher is better)
        results = results.sort_values('ml_score', ascending=False)
        results['ml_rank'] = range(1, len(results) + 1)

        if top_n:
            results = results.head(top_n)

        print(f"  ✓ Ranked {len(results)} stocks")
        print(f"    Top 5: {results.head()['ticker'].tolist()}")

        return results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances."""
        if self.feature_importance is None:
            return pd.DataFrame()

        return self.feature_importance.head(top_n)

    def _prepare_training_data(self, features_df: pd.DataFrame,
                               returns_df: pd.DataFrame,
                               forward_periods: int) -> Tuple:
        """Prepare X, y, dates, tickers for training."""
        # Calculate forward returns
        forward_returns = returns_df.pct_change(forward_periods).shift(-forward_periods)

        # Merge features with forward returns
        training_data = []

        for idx, row in features_df.iterrows():
            ticker = row['ticker']
            date = row.get('date', features_df.index[idx])

            if ticker in forward_returns.columns:
                try:
                    forward_ret = forward_returns.loc[date, ticker]
                    if pd.notna(forward_ret):
                        row_data = row.to_dict()
                        row_data['forward_return'] = forward_ret
                        row_data['date'] = date
                        training_data.append(row_data)
                except Exception:
                    continue

        if not training_data:
            return np.array([]), np.array([]), np.array([]), np.array([])

        train_df = pd.DataFrame(training_data)

        # Extract features
        X = self._extract_features(train_df)
        y = train_df['forward_return'].values
        dates = train_df['date'].values
        tickers = train_df['ticker'].values

        # Remove rows with NaN
        valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        dates = dates[valid_mask]
        tickers = tickers[valid_mask]

        # Normalize features
        X = self.scaler.fit_transform(X)

        return X, y, dates, tickers

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame."""
        # Exclude non-feature columns
        exclude_cols = ['ticker', 'date', 'forward_return']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        if self.feature_names is None:
            self.feature_names = feature_cols

        X = df[feature_cols].values
        return X

    def _train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model."""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        return model

    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dval, 'validation')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        return model

    def _train_simple(self, X_train, y_train):
        """Simple linear model fallback."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        return model

    def _extract_feature_importance(self):
        """Extract feature importance from trained model."""
        if self.model is None or self.feature_names is None:
            return

        try:
            if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
                importance = self.model.feature_importance(importance_type='gain')
            elif self.model_type == 'xgboost' and HAS_XGBOOST:
                importance = self.model.get_score(importance_type='gain')
                # Convert to array
                importance = np.array([importance.get(f'f{i}', 0)
                                     for i in range(len(self.feature_names))])
            else:
                # Linear model
                importance = np.abs(self.model.coef_)

            # Create DataFrame
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

        except Exception as e:
            print(f"  Warning: Could not extract feature importance: {e}")
            self.feature_importance = None


def select_stocks_ml(features_df: pd.DataFrame,
                    returns_df: pd.DataFrame,
                    top_n: int = 10,
                    model_type: str = 'lightgbm') -> Tuple[List[str], pd.DataFrame, Dict]:
    """
    Convenience function for ML stock selection.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix
    returns_df : pd.DataFrame
        Historical returns
    top_n : int
        Number of stocks to select
    model_type : str
        ML model type

    Returns
    -------
    tuple
        (selected_tickers, rankings, diagnostics)
    """
    selector = MLStockSelector(model_type=model_type)

    # Train
    train_metrics = selector.train(features_df, returns_df)

    # Rank latest
    latest_features = features_df.groupby('ticker').tail(1)
    rankings = selector.rank_stocks(latest_features, top_n=top_n)

    selected_tickers = rankings['ticker'].tolist()

    diagnostics = {
        'model_type': model_type,
        'train_metrics': train_metrics,
        'feature_importance': selector.get_feature_importance(),
    }

    return selected_tickers, rankings, diagnostics
