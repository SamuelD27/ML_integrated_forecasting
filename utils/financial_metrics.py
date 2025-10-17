"""
Financial Performance Metrics
==============================
Comprehensive performance evaluation for trading strategies.

Metrics Categories:
1. Return Metrics (total, annual, CAGR)
2. Risk-Adjusted Metrics (Sharpe, Sortino, Calmar)
3. Risk Metrics (volatility, VaR, CVaR, max drawdown)
4. Trading Metrics (win rate, profit factor, avg win/loss)
5. Factor Attribution (IC, directional accuracy, factor contributions)
6. Benchmark Comparison (alpha, beta, information ratio)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class FinancialMetrics:
    """
    Calculate comprehensive financial performance metrics.

    Example:
        >>> metrics = FinancialMetrics(returns=strategy_returns)
        >>> metrics.sharpe_ratio()
        >>> metrics.information_coefficient(predictions, actuals)
        >>> metrics.summary()
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize financial metrics calculator.

        Args:
            returns: Strategy returns (daily)
            benchmark_returns: Benchmark returns (daily, optional)
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate

        # Trading days per year
        self.periods_per_year = 252

    # ========================================================================
    # RETURN METRICS
    # ========================================================================

    def total_return(self) -> float:
        """Total cumulative return."""
        return (1 + self.returns).prod() - 1

    def annualized_return(self) -> float:
        """Annualized return (CAGR)."""
        n_days = len(self.returns)
        n_years = n_days / self.periods_per_year
        total_ret = self.total_return()
        return (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    def monthly_returns(self) -> pd.Series:
        """Calculate monthly returns."""
        return self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    def annual_returns(self) -> pd.Series:
        """Calculate annual returns."""
        return self.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)

    # ========================================================================
    # RISK-ADJUSTED METRICS
    # ========================================================================

    def sharpe_ratio(self) -> float:
        """
        Sharpe Ratio: (return - risk_free) / volatility

        Measures excess return per unit of risk.
        > 1.0 is good, > 2.0 is very good, > 3.0 is excellent
        """
        excess_returns = self.returns - self.risk_free_rate / self.periods_per_year
        return (
            excess_returns.mean() / excess_returns.std() * np.sqrt(self.periods_per_year)
            if excess_returns.std() > 0 else 0
        )

    def sortino_ratio(self, target_return: float = 0) -> float:
        """
        Sortino Ratio: (return - target) / downside_deviation

        Like Sharpe but only penalizes downside volatility.
        More relevant for risk assessment.
        """
        excess_returns = self.returns - target_return / self.periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()

        return (
            excess_returns.mean() / downside_std * np.sqrt(self.periods_per_year)
            if downside_std > 0 else 0
        )

    def calmar_ratio(self) -> float:
        """
        Calmar Ratio: annual_return / max_drawdown

        Return per unit of worst loss.
        > 0.5 is acceptable, > 1.0 is good
        """
        ann_ret = self.annualized_return()
        max_dd = self.max_drawdown()
        return ann_ret / abs(max_dd) if max_dd != 0 else 0

    # ========================================================================
    # RISK METRICS
    # ========================================================================

    def volatility(self, annualized: bool = True) -> float:
        """Return volatility (standard deviation)."""
        vol = self.returns.std()
        return vol * np.sqrt(self.periods_per_year) if annualized else vol

    def max_drawdown(self) -> float:
        """
        Maximum Drawdown: largest peak-to-trough decline.

        Returns negative value (e.g., -0.15 = -15% drawdown)
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Value at Risk (VaR): maximum loss at confidence level.

        Args:
            confidence: Confidence level (0.95 = 95%)

        Returns:
            VaR as negative number (e.g., -0.02 = -2% loss)
        """
        return self.returns.quantile(1 - confidence)

    def conditional_var(self, confidence: float = 0.95) -> float:
        """
        Conditional VaR (CVaR) / Expected Shortfall:
        Average loss beyond VaR threshold.

        More conservative than VaR.
        """
        var = self.value_at_risk(confidence)
        return self.returns[self.returns <= var].mean()

    def drawdown_duration(self) -> Dict[str, int]:
        """
        Calculate drawdown statistics.

        Returns:
            Dictionary with max_duration_days and current_duration_days
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()

        # Calculate durations
        durations = drawdown_periods[in_drawdown].value_counts()
        max_duration = durations.max() if len(durations) > 0 else 0

        # Current drawdown duration
        current_duration = 0
        if in_drawdown.iloc[-1]:
            current_group = drawdown_periods.iloc[-1]
            current_duration = (drawdown_periods == current_group).sum()

        return {
            'max_duration_days': int(max_duration),
            'current_duration_days': int(current_duration)
        }

    # ========================================================================
    # TRADING METRICS
    # ========================================================================

    def win_rate(self) -> float:
        """Percentage of positive return periods."""
        return (self.returns > 0).sum() / len(self.returns)

    def profit_factor(self) -> float:
        """
        Profit Factor: sum(gains) / sum(losses)

        > 1.0 means profitable, > 2.0 is good
        """
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return gains / losses if losses > 0 else np.inf

    def avg_win_loss_ratio(self) -> float:
        """Average win / average loss."""
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        return avg_win / avg_loss if avg_loss > 0 else 0

    # ========================================================================
    # FACTOR ATTRIBUTION
    # ========================================================================

    def information_coefficient(
        self,
        predictions: Union[pd.Series, np.ndarray],
        actuals: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Information Coefficient (IC): correlation between predictions and actuals.

        Measures quality of forecasts.
        IC > 0.03 is good, > 0.05 is very good

        Args:
            predictions: Predicted returns or scores
            actuals: Actual returns

        Returns:
            Correlation coefficient
        """
        # Convert to numpy arrays
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(actuals, pd.Series):
            actuals = actuals.values

        # Remove NaN
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if mask.sum() < 2:
            return 0

        ic, _ = stats.spearmanr(predictions[mask], actuals[mask])
        return ic if not np.isnan(ic) else 0

    def directional_accuracy(
        self,
        predictions: Union[pd.Series, np.ndarray],
        actuals: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Directional Accuracy: percentage of correct direction predictions.

        > 0.50 is better than random, > 0.55 is good

        Args:
            predictions: Predicted direction/returns
            actuals: Actual returns

        Returns:
            Accuracy (0 to 1)
        """
        # Convert to numpy
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(actuals, pd.Series):
            actuals = actuals.values

        # Remove NaN
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if mask.sum() < 1:
            return 0

        # Compare directions
        pred_direction = np.sign(predictions[mask])
        actual_direction = np.sign(actuals[mask])

        accuracy = (pred_direction == actual_direction).sum() / len(pred_direction)
        return accuracy

    def hit_ratio_by_quantile(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        n_quantiles: int = 5
    ) -> pd.DataFrame:
        """
        Calculate hit ratio by prediction quantile.

        Shows if top-ranked predictions actually perform best.

        Args:
            predictions: Predicted scores
            actuals: Actual returns
            n_quantiles: Number of quantiles (default: 5 = quintiles)

        Returns:
            DataFrame with quantile, avg_prediction, avg_actual, hit_ratio
        """
        df = pd.DataFrame({'pred': predictions, 'actual': actuals}).dropna()

        # Assign quantiles (1 = highest predictions)
        df['quantile'] = pd.qcut(df['pred'], n_quantiles, labels=False, duplicates='drop') + 1

        # Calculate metrics by quantile
        results = []
        for q in range(1, n_quantiles + 1):
            subset = df[df['quantile'] == q]
            if len(subset) > 0:
                results.append({
                    'quantile': q,
                    'avg_prediction': subset['pred'].mean(),
                    'avg_actual': subset['actual'].mean(),
                    'hit_ratio': (subset['actual'] > 0).sum() / len(subset),
                    'n_obs': len(subset)
                })

        return pd.DataFrame(results)

    # ========================================================================
    # BENCHMARK COMPARISON
    # ========================================================================

    def alpha_beta(self) -> Tuple[float, float]:
        """
        Calculate alpha and beta vs benchmark.

        Returns:
            Tuple of (alpha, beta)
            Alpha: annualized excess return
            Beta: market sensitivity
        """
        if self.benchmark_returns is None:
            return (0, 0)

        # Align returns
        combined = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        if len(combined) < 2:
            return (0, 0)

        y = combined.iloc[:, 0].values  # Strategy
        X = combined.iloc[:, 1].values  # Benchmark

        # Linear regression
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X.reshape(-1, 1), y)

        beta = reg.coef_[0]
        alpha = reg.intercept_ * self.periods_per_year  # Annualize

        return (alpha, beta)

    def information_ratio(self) -> float:
        """
        Information Ratio: active_return / tracking_error

        Measures excess return per unit of active risk.
        > 0.5 is good, > 1.0 is excellent
        """
        if self.benchmark_returns is None:
            return 0

        # Align returns
        combined = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        if len(combined) < 2:
            return 0

        active_returns = combined.iloc[:, 0] - combined.iloc[:, 1]

        return (
            active_returns.mean() / active_returns.std() * np.sqrt(self.periods_per_year)
            if active_returns.std() > 0 else 0
        )

    def tracking_error(self) -> float:
        """Tracking Error: std(strategy - benchmark)."""
        if self.benchmark_returns is None:
            return 0

        combined = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        if len(combined) < 2:
            return 0

        active_returns = combined.iloc[:, 0] - combined.iloc[:, 1]
        return active_returns.std() * np.sqrt(self.periods_per_year)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def summary(self) -> Dict[str, float]:
        """
        Calculate all metrics at once.

        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {
            # Returns
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),

            # Risk-adjusted
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),

            # Risk
            'volatility': self.volatility(),
            'max_drawdown': self.max_drawdown(),
            'var_95': self.value_at_risk(0.95),
            'cvar_95': self.conditional_var(0.95),

            # Trading
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'avg_win_loss_ratio': self.avg_win_loss_ratio(),
        }

        # Drawdown duration
        dd_stats = self.drawdown_duration()
        metrics.update(dd_stats)

        # Benchmark comparison
        if self.benchmark_returns is not None:
            alpha, beta = self.alpha_beta()
            metrics['alpha'] = alpha
            metrics['beta'] = beta
            metrics['information_ratio'] = self.information_ratio()
            metrics['tracking_error'] = self.tracking_error()

        return metrics

    def report(self) -> str:
        """Generate text report."""
        metrics = self.summary()

        report = f"""
==================================================
         FINANCIAL PERFORMANCE METRICS
==================================================

RETURN METRICS
--------------------------------------------------
Total Return:          {metrics['total_return']:>10.2%}
Annualized Return:     {metrics['annualized_return']:>10.2%}

RISK-ADJUSTED RETURNS
--------------------------------------------------
Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}
Sortino Ratio:         {metrics['sortino_ratio']:>10.2f}
Calmar Ratio:          {metrics['calmar_ratio']:>10.2f}

RISK METRICS
--------------------------------------------------
Volatility (annual):   {metrics['volatility']:>10.2%}
Max Drawdown:          {metrics['max_drawdown']:>10.2%}
VaR (95%):             {metrics['var_95']:>10.2%}
CVaR (95%):            {metrics['cvar_95']:>10.2%}
Max DD Duration:       {metrics['max_duration_days']:>10.0f} days
Current DD Duration:   {metrics['current_duration_days']:>10.0f} days

TRADING METRICS
--------------------------------------------------
Win Rate:              {metrics['win_rate']:>10.2%}
Profit Factor:         {metrics['profit_factor']:>10.2f}
Avg Win/Loss Ratio:    {metrics['avg_win_loss_ratio']:>10.2f}
"""

        if self.benchmark_returns is not None:
            report += f"""
BENCHMARK COMPARISON
--------------------------------------------------
Alpha (annual):        {metrics['alpha']:>10.2%}
Beta:                  {metrics['beta']:>10.2f}
Information Ratio:     {metrics['information_ratio']:>10.2f}
Tracking Error:        {metrics['tracking_error']:>10.2%}
"""

        report += "==================================================\n"

        return report


def compare_strategies(
    strategies: Dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Compare multiple strategies.

    Args:
        strategies: Dictionary mapping strategy_name -> returns
        benchmark: Benchmark returns (optional)
        risk_free_rate: Annual risk-free rate

    Returns:
        DataFrame with comparative metrics

    Example:
        >>> strategies = {
        ...     'Strategy A': returns_a,
        ...     'Strategy B': returns_b,
        ...     'Strategy C': returns_c
        ... }
        >>> comparison = compare_strategies(strategies, benchmark=spy_returns)
        >>> print(comparison)
    """
    results = []

    for name, returns in strategies.items():
        metrics_calc = FinancialMetrics(returns, benchmark, risk_free_rate)
        metrics = metrics_calc.summary()
        metrics['strategy'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index('strategy')

    return df
