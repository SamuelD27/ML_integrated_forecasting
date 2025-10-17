"""
Tooltip Definitions for Dashboard Metrics
==========================================
Comprehensive help text for all quantitative finance metrics.

Usage:
    from dashboard.utils.tooltips import get_tooltip

    st.metric("Sharpe Ratio", f"{sharpe:.2f}", help=get_tooltip('sharpe_ratio'))
"""

TOOLTIPS = {
    # Return Metrics
    'total_return': """
**Total Return**

Formula: (End Value - Start Value) / Start Value

• 0.15 = 15% gain
• Does NOT account for risk
• Good for same time period comparisons

Example: $10,000 → $11,500 = 15% total return
    """,

    'annualized_return': """
**Annualized Return**

Formula: (1 + Total Return)^(252/days) - 1

• Converts any period to annual equivalent
• Accounts for compounding
• Allows fair comparisons across time periods

Example: 20% in 6 months ≈ 44% annualized
    """,

    'cumulative_return': """
**Cumulative Return**

Total wealth growth over time.

• Shows compounding effects
• Path-dependent
• Start from 1.0 (initial investment)

Example: $100 → $150 = 50% cumulative return
    """,

    # Risk Metrics
    'volatility': """
**Volatility (Standard Deviation)**

Formula: std(returns) × sqrt(252)

• Measures price fluctuation
• Higher = riskier
• ~68% of returns fall within ±1σ

Typical ranges:
• Bonds: 5-10%
• Large stocks: 15-20%
• Small stocks: 25-35%
• Crypto: 60-150%+
    """,

    'sharpe_ratio': """
**Sharpe Ratio**

Formula: (Return - RiskFree) / Volatility

Return per unit of risk. Higher is better.

Guide:
• <0: Losing vs safe asset
• 1.0: Good
• 2.0: Very good
• >3.0: Excellent (or suspicious)

Example: S&P 500 long-term ~0.5
    """,

    'sortino_ratio': """
**Sortino Ratio**

Like Sharpe but only counts downside risk.

• Only penalizes bad volatility
• Upside volatility is good!
• Better for asymmetric strategies

Guide: Same as Sharpe, typically higher values
    """,

    'calmar_ratio': """
**Calmar Ratio**

Formula: Annual Return / |Max Drawdown|

Return per unit of worst pain.

Guide:
• >2.0: Excellent
• 0.5-2.0: Average
• <0.5: Poor

Professional managers often target 1.0-2.0
    """,

    'max_drawdown': """
**Maximum Drawdown**

Largest peak-to-trough decline.

• Tests pain tolerance
• Recovery can take years

Typical ranges:
• Conservative: -10% to -20%
• S&P 500: -20% to -30%
• Aggressive: -30% to -50%

Example: 2008 S&P 500 = -56.8%
    """,

    'var_95': """
**Value at Risk (95%)**

"On worst 5% of days, expect to lose this much or more"

• NOT the maximum loss
• 95% of days: lose less than this
• 5% of days: lose this or more

Example: VaR = 5% means 1 in 20 days lose 5%+
    """,

    'cvar_95': """
**Conditional VaR (95%)**

Average loss in worst 5% of cases.

• More informative than VaR
• Captures tail risk severity
• Used in portfolio optimization

Always worse than VaR (as expected)
    """,

    # Factor Analysis
    'alpha': """
**Alpha (α)**

Return NOT explained by market factors.

• Positive alpha = skill (if p<0.05)
• Zero alpha = fair pricing
• Negative alpha = underperformance

Note: Most managers have alpha ≈ 0
Statistically significant alpha is RARE
    """,

    'beta': """
**Market Beta (β)**

Sensitivity to market movements.

• 0.5 = Half as volatile (defensive)
• 1.0 = Moves with market
• 1.5 = 50% more volatile (aggressive)

Market up 10%:
• β=0.5 stock: expect +5%
• β=1.5 stock: expect +15%
    """,

    'beta_smb': """
**Size Beta (SMB)**

Small Minus Big factor exposure.

• Positive = Small-cap characteristics
• Negative = Large-cap characteristics
• Zero = Size-neutral

Small-cap premium historically ~2-3% annually
    """,

    'beta_hml': """
**Value Beta (HML)**

High Minus Low (book-to-market) factor.

• Positive = Value stock characteristics
• Negative = Growth stock characteristics
• Zero = Neither value nor growth

Value premium historically ~3-5% annually
    """,

    'r_squared': """
**R-Squared (R²)**

% of variance explained by factors.

• 0.85 = 85% systematic, 15% idiosyncratic
• Higher = more factor-driven
• Lower = more stock-specific

High R² + low alpha → just buy index!
    """,

    # ML Metrics
    'ml_confidence': """
**Model Confidence**

Based on ensemble agreement.

• >0.7: Trust forecast
• 0.4-0.7: Moderate confidence
• <0.4: Don't trust, use historical

Higher when models agree.
Lower when models disagree.
    """,

    'ml_forecast': """
**ML Ensemble Forecast**

20-day ahead price prediction.

Combines:
• LightGBM (gradient boosting)
• Ridge Regression (linear)
• Momentum Model (trend)

Weighted by validation performance.
    """,

    'directional_accuracy': """
**Directional Accuracy**

% of times model predicts correct direction.

• 50%: Random (useless)
• 55%: Decent (profitable)
• 60%+: Very good
• 70%+: Exceptional

Even 55% can be very profitable with proper risk management!
    """,

    'forecast_return': """
**Forecast Return**

Predicted price change over forecast horizon.

• Positive = bullish forecast
• Negative = bearish forecast

Combined with confidence to generate signals.
    """,

    # Portfolio Metrics
    'portfolio_weight': """
**Portfolio Weight**

% of capital allocated to this asset.

• Sum of all weights = 100%
• Higher weight = more conviction
• Constraints typically 5-40% per asset

Diversification reduces risk.
    """,

    'correlation': """
**Correlation**

Measure of co-movement (-1 to +1).

• +1: Perfect positive correlation
• 0: No relationship
• -1: Perfect negative correlation

Low correlation improves diversification!
    """,

    'tracking_error': """
**Tracking Error**

Standard deviation of active returns.

• Measures deviation from benchmark
• Higher = more active management
• Lower = closer to index

Active managers aim for 2-6% tracking error
    """,

    # Options Greeks
    'delta': """
**Delta (Δ)**

Option price change per $1 stock move.

• Call: 0 to 1
• Put: -1 to 0
• Also ≈ probability of expiring ITM

Hedge ratio: 1/delta options per 100 shares
    """,

    'gamma': """
**Gamma (Γ)**

Rate of change of Delta.

• Higher = Delta changes faster
• Peaks for ATM options near expiration
• Long gamma wants big moves

High gamma = position risk changes rapidly
    """,

    'theta': """
**Theta (Θ)**

Daily time decay.

• Always negative for long options
• Always positive for short options
• Accelerates near expiration

Options are wasting assets!
    """,

    'vega': """
**Vega (ν)**

Price change per 1% IV move.

• Higher for longer-dated options
• Always positive for long options

Long options want IV to increase.
Short options want IV to decrease.
    """,

    # Valuation
    'dcf_value': """
**DCF Intrinsic Value**

Discounted cash flow valuation.

• Based on future cash flow projections
• Discounted to present value
• Extremely sensitive to assumptions

Compare to current price for fair value estimate.
    """,

    'pe_ratio': """
**P/E Ratio**

Price-to-Earnings ratio.

Formula: Price / Earnings Per Share

Typical ranges:
• Value stocks: 8-15
• Market average: 15-20
• Growth stocks: 25-50+

Lower = cheaper (generally)
    """,

    # Monte Carlo
    'probability_profit': """
**Probability of Profit**

% chance of ending positive.

• Based on 10,000 simulations
• Accounts for volatility and drift
• >50% = expected positive outcome

Use for position sizing and risk assessment.
    """,

    'percentile_paths': """
**Percentile Paths**

Range of possible outcomes.

• P5 to P95 = 90% probability range
• P50 = median (most likely)
• Wider range = more uncertainty

Example: P5=$130, P50=$150, P95=$180
    """
}


def get_tooltip(metric_name: str) -> str:
    """
    Get tooltip text for a metric.

    Args:
        metric_name: Name of the metric

    Returns:
        Tooltip text (formatted markdown)
    """
    return TOOLTIPS.get(metric_name, "No description available.")


def get_all_tooltips() -> dict:
    """Get all tooltips as dictionary."""
    return TOOLTIPS.copy()


# Convenient aliases
HELP = {
    'sharpe': get_tooltip('sharpe_ratio'),
    'sortino': get_tooltip('sortino_ratio'),
    'calmar': get_tooltip('calmar_ratio'),
    'volatility': get_tooltip('volatility'),
    'vol': get_tooltip('volatility'),
    'drawdown': get_tooltip('max_drawdown'),
    'var': get_tooltip('var_95'),
    'cvar': get_tooltip('cvar_95'),
    'alpha': get_tooltip('alpha'),
    'beta': get_tooltip('beta'),
    'r2': get_tooltip('r_squared'),
    'ml_conf': get_tooltip('ml_confidence'),
    'dir_acc': get_tooltip('directional_accuracy'),
}


def help(metric: str) -> str:
    """Shortcut to get tooltip."""
    return HELP.get(metric.lower(), get_tooltip(metric))
