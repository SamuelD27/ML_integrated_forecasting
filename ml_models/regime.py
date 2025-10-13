import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class RegimeDetector:
    """Detect market regimes based on volatility and returns."""

    def __init__(self, lookback_short: int = 20, lookback_long: int = 60):
        """
        Initialize regime detector.

        Parameters
        ----------
        lookback_short : int
            Short-term lookback for volatility (default: 20 days)
        lookback_long : int
            Long-term lookback for regime classification (default: 60 days)
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.current_regime = None
        self.regime_history = []

    def detect_regime(self, returns: pd.Series,
                     method: str = 'volatility_breakpoint') -> Dict:
        """
        Detect current market regime.

        Parameters
        ----------
        returns : pd.Series
            Market returns (daily)
        method : str
            Detection method: 'volatility_breakpoint', 'returns_based', or 'combined'

        Returns
        -------
        dict
            Regime classification and metrics
        """
        if len(returns) < self.lookback_long:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'metrics': {}
            }

        if method == 'volatility_breakpoint':
            regime_info = self._detect_volatility_breakpoint(returns)
        elif method == 'returns_based':
            regime_info = self._detect_returns_based(returns)
        else:  # combined
            vol_regime = self._detect_volatility_breakpoint(returns)
            ret_regime = self._detect_returns_based(returns)
            regime_info = self._combine_regimes(vol_regime, ret_regime)

        self.current_regime = regime_info['regime']
        self.regime_history.append({
            'date': datetime.now(),
            'regime': regime_info['regime'],
            'confidence': regime_info['confidence']
        })

        return regime_info

    def _detect_volatility_breakpoint(self, returns: pd.Series) -> Dict:
        """
        Detect regime based on volatility breakpoints.

        Regimes:
        - 'low_vol': Volatility < 25th percentile (calm markets)
        - 'normal': Volatility between 25th and 75th percentile
        - 'high_vol': Volatility > 75th percentile (stress/crisis)
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(self.lookback_short).std() * np.sqrt(252)

        # Current volatility
        current_vol = rolling_vol.iloc[-1]

        # Historical volatility distribution (long-term)
        hist_vol = rolling_vol.iloc[-self.lookback_long:-1]

        # Percentiles
        p25 = hist_vol.quantile(0.25)
        p75 = hist_vol.quantile(0.75)
        p90 = hist_vol.quantile(0.90)

        # Classify regime
        if current_vol < p25:
            regime = 'low_vol'
            confidence = 1 - (current_vol / p25)
        elif current_vol > p90:
            regime = 'high_vol'
            confidence = min((current_vol - p75) / (p90 - p75), 1.0)
        elif current_vol > p75:
            regime = 'elevated_vol'
            confidence = (current_vol - p75) / (p90 - p75)
        else:
            regime = 'normal'
            confidence = 1 - abs(current_vol - 0.5 * (p25 + p75)) / (p75 - p25)

        return {
            'regime': regime,
            'confidence': confidence,
            'metrics': {
                'current_volatility': current_vol,
                'vol_p25': p25,
                'vol_p50': hist_vol.median(),
                'vol_p75': p75,
                'vol_p90': p90,
            }
        }

    def _detect_returns_based(self, returns: pd.Series) -> Dict:
        """
        Detect regime based on returns trend.

        Regimes:
        - 'bull': Positive trend with positive returns
        - 'bear': Negative trend with negative returns
        - 'sideways': No clear trend
        """
        # Recent returns
        recent_ret = returns.iloc[-self.lookback_long:]

        # Cumulative return
        cum_ret = (1 + recent_ret).prod() - 1

        # Trend (linear regression slope)
        x = np.arange(len(recent_ret))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_ret.values)

        # Annualized trend
        trend_annual = slope * 252

        # Mean return
        mean_ret = recent_ret.mean() * 252

        # Classify regime
        if trend_annual > 0.05 and mean_ret > 0:
            regime = 'bull'
            confidence = min(abs(r_value), 1.0)
        elif trend_annual < -0.05 and mean_ret < 0:
            regime = 'bear'
            confidence = min(abs(r_value), 1.0)
        else:
            regime = 'sideways'
            confidence = 1 - min(abs(r_value), 1.0)

        return {
            'regime': regime,
            'confidence': confidence,
            'metrics': {
                'cumulative_return': cum_ret,
                'trend_annual': trend_annual,
                'mean_return_annual': mean_ret,
                'r_squared': r_value ** 2,
            }
        }

    def _combine_regimes(self, vol_regime: Dict, ret_regime: Dict) -> Dict:
        """Combine volatility and returns-based regime detection."""
        vol_type = vol_regime['regime']
        ret_type = ret_regime['regime']

        # Priority to volatility regimes (risk management focus)
        if vol_type == 'high_vol':
            regime = 'crisis'
            confidence = vol_regime['confidence']
        elif vol_type == 'elevated_vol' and ret_type == 'bear':
            regime = 'bear_market'
            confidence = 0.5 * (vol_regime['confidence'] + ret_regime['confidence'])
        elif vol_type == 'low_vol' and ret_type == 'bull':
            regime = 'bull_market'
            confidence = 0.5 * (vol_regime['confidence'] + ret_regime['confidence'])
        elif ret_type == 'bull':
            regime = 'bull_market'
            confidence = ret_regime['confidence']
        elif ret_type == 'bear':
            regime = 'bear_market'
            confidence = ret_regime['confidence']
        else:
            regime = 'neutral'
            confidence = 0.5

        # Combine metrics
        metrics = {**vol_regime['metrics'], **ret_regime['metrics']}

        return {
            'regime': regime,
            'confidence': confidence,
            'metrics': metrics,
            'vol_regime': vol_type,
            'ret_regime': ret_type,
        }

    def get_regime_parameters(self, regime: Optional[str] = None) -> Dict:
        """
        Get recommended portfolio parameters for current/specified regime.

        Returns
        -------
        dict
            risk_aversion, max_weight, hedge_multiplier, rebalance_threshold
        """
        if regime is None:
            regime = self.current_regime

        if regime is None:
            regime = 'normal'

        # Define parameters per regime
        regime_params = {
            'bull_market': {
                'risk_aversion': 3.0,      # Lower (more aggressive)
                'max_weight': 0.25,        # Allow larger positions
                'hedge_multiplier': 0.5,   # Reduce hedge budget
                'rebalance_threshold': 0.15,  # Less frequent rebalancing
                'beta_target': 1.1,        # Slightly leveraged to market
            },
            'bear_market': {
                'risk_aversion': 12.0,     # Higher (more defensive)
                'max_weight': 0.15,        # Smaller positions
                'hedge_multiplier': 1.5,   # Increase hedge budget
                'rebalance_threshold': 0.08,  # More frequent rebalancing
                'beta_target': 0.7,        # Defensive positioning
            },
            'crisis': {
                'risk_aversion': 20.0,     # Very high (very defensive)
                'max_weight': 0.10,        # Very small positions
                'hedge_multiplier': 2.0,   # Maximum hedge budget
                'rebalance_threshold': 0.05,  # Very frequent rebalancing
                'beta_target': 0.5,        # Highly defensive
            },
            'low_vol': {
                'risk_aversion': 4.0,
                'max_weight': 0.30,
                'hedge_multiplier': 0.8,
                'rebalance_threshold': 0.12,
                'beta_target': 1.0,
            },
            'normal': {
                'risk_aversion': 6.0,
                'max_weight': 0.20,
                'hedge_multiplier': 1.0,
                'rebalance_threshold': 0.10,
                'beta_target': 0.9,
            },
            'elevated_vol': {
                'risk_aversion': 10.0,
                'max_weight': 0.15,
                'hedge_multiplier': 1.3,
                'rebalance_threshold': 0.08,
                'beta_target': 0.8,
            },
            'neutral': {
                'risk_aversion': 6.0,
                'max_weight': 0.20,
                'hedge_multiplier': 1.0,
                'rebalance_threshold': 0.10,
                'beta_target': 0.9,
            },
            'sideways': {
                'risk_aversion': 7.0,
                'max_weight': 0.18,
                'hedge_multiplier': 1.1,
                'rebalance_threshold': 0.10,
                'beta_target': 0.85,
            },
        }

        return regime_params.get(regime, regime_params['normal'])

    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of recent regime history."""
        if not self.regime_history:
            return pd.DataFrame()

        # Last 30 entries
        recent = self.regime_history[-30:]

        df = pd.DataFrame(recent)

        # Count by regime
        regime_counts = df['regime'].value_counts()

        summary = pd.DataFrame({
            'regime': regime_counts.index,
            'frequency': regime_counts.values,
            'percentage': regime_counts.values / len(recent) * 100
        })

        return summary


def detect_market_regime(market_returns: pd.Series,
                        method: str = 'combined') -> Tuple[str, Dict]:
    """
    Convenience function for regime detection.

    Parameters
    ----------
    market_returns : pd.Series
        Market returns (daily)
    method : str
        Detection method

    Returns
    -------
    tuple
        (regime, full_info)
    """
    detector = RegimeDetector()
    regime_info = detector.detect_regime(market_returns, method=method)

    regime = regime_info['regime']
    return regime, regime_info
