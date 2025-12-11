"""
Security Data Configuration
============================
Ultra-comprehensive configuration for security data fetching and analysis.

This module defines ALL data points that can be fetched for each security,
organized by category. The analysis/decision engine uses these specifications
to determine what data to collect before making trading decisions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


# =============================================================================
# ENUMS - Data Categories and Options
# =============================================================================

class UniverseType(Enum):
    """Stock universe types."""
    SP500 = "sp500"
    SP100 = "sp100"
    NASDAQ100 = "nasdaq100"
    DOW30 = "dow30"
    RUSSELL1000 = "russell1000"
    RUSSELL2000 = "russell2000"
    CUSTOM = "custom"


class SelectionMethod(Enum):
    """Stock selection/ranking methods."""
    FEATURE_SCORING = "feature_scoring"      # Momentum + volatility + Sharpe
    MULTI_FACTOR = "multi_factor"            # Quality + Value + Momentum
    MOMENTUM = "momentum"                     # Pure momentum
    VALUE = "value"                           # P/E, P/B, EV/EBITDA
    QUALITY = "quality"                       # ROE, margins, debt
    GROWTH = "growth"                         # Revenue/earnings growth
    DIVIDEND = "dividend"                     # Yield, payout, growth
    LOW_VOLATILITY = "low_volatility"        # Min variance
    COMBINED = "combined"                     # Weighted combination


class DataFrequency(Enum):
    """Data frequency options."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"


# =============================================================================
# UNIVERSE SCREENING CONFIG
# =============================================================================

@dataclass
class UniverseScreeningConfig:
    """
    Configuration for universe screening and filtering.

    These parameters determine which securities make it into the investable universe.
    """
    # Base universe
    universe_type: UniverseType = UniverseType.SP500
    custom_symbols: List[str] = field(default_factory=list)
    exclude_symbols: List[str] = field(default_factory=list)
    max_universe_size: int = 100

    # Price filters
    min_price: float = 5.0                    # Avoid penny stocks
    max_price: Optional[float] = None         # None = no limit

    # Liquidity filters
    min_avg_volume: int = 500_000             # Daily volume (shares)
    min_avg_dollar_volume: float = 10_000_000 # Daily dollar volume
    min_market_cap: float = 10_000_000_000    # $10B minimum
    max_market_cap: Optional[float] = None    # None = no limit

    # Float and tradability
    min_float_shares: int = 50_000_000        # Minimum public float
    min_days_trading: int = 252               # At least 1 year of history

    # Sector filters
    include_sectors: Optional[List[str]] = None  # None = all sectors
    exclude_sectors: List[str] = field(default_factory=list)

    # Exchange filters
    include_exchanges: List[str] = field(default_factory=lambda: ['NYSE', 'NASDAQ', 'AMEX'])
    exclude_adrs: bool = False                # Exclude American Depositary Receipts
    exclude_etfs: bool = True                 # Exclude ETFs (if screening stocks)

    @classmethod
    def from_env(cls) -> 'UniverseScreeningConfig':
        """Load from environment variables."""
        universe_str = os.getenv('UNIVERSE_TYPE', 'sp500').lower()
        universe_type = UniverseType(universe_str) if universe_str in [u.value for u in UniverseType] else UniverseType.SP500

        custom_str = os.getenv('UNIVERSE_CUSTOM_SYMBOLS', '')
        custom_symbols = [s.strip().upper() for s in custom_str.split(',') if s.strip()]

        exclude_str = os.getenv('UNIVERSE_EXCLUDE_SYMBOLS', '')
        exclude_symbols = [s.strip().upper() for s in exclude_str.split(',') if s.strip()]

        exclude_sectors_str = os.getenv('UNIVERSE_EXCLUDE_SECTORS', '')
        exclude_sectors = [s.strip() for s in exclude_sectors_str.split(',') if s.strip()]

        return cls(
            universe_type=universe_type,
            custom_symbols=custom_symbols,
            exclude_symbols=exclude_symbols,
            max_universe_size=int(os.getenv('MAX_UNIVERSE_SIZE', '100')),
            min_price=float(os.getenv('SCREEN_MIN_PRICE', '5.0')),
            max_price=float(os.getenv('SCREEN_MAX_PRICE')) if os.getenv('SCREEN_MAX_PRICE') else None,
            min_avg_volume=int(os.getenv('SCREEN_MIN_AVG_VOLUME', '500000')),
            min_avg_dollar_volume=float(os.getenv('SCREEN_MIN_DOLLAR_VOLUME', '10000000')),
            min_market_cap=float(os.getenv('SCREEN_MIN_MARKET_CAP', '10000000000')),
            max_market_cap=float(os.getenv('SCREEN_MAX_MARKET_CAP')) if os.getenv('SCREEN_MAX_MARKET_CAP') else None,
            min_float_shares=int(os.getenv('SCREEN_MIN_FLOAT', '50000000')),
            min_days_trading=int(os.getenv('SCREEN_MIN_DAYS_TRADING', '252')),
            exclude_sectors=exclude_sectors,
            exclude_adrs=os.getenv('SCREEN_EXCLUDE_ADRS', 'false').lower() == 'true',
            exclude_etfs=os.getenv('SCREEN_EXCLUDE_ETFS', 'true').lower() == 'true',
        )


# =============================================================================
# PRICE DATA CONFIG
# =============================================================================

@dataclass
class PriceDataConfig:
    """
    Configuration for price/OHLCV data fetching.
    """
    # Time series depth
    lookback_days: int = 504                  # ~2 years of trading days
    intraday_lookback_days: int = 5           # Days of intraday data

    # Frequencies to fetch
    fetch_daily: bool = True
    fetch_intraday: bool = True
    intraday_frequency: DataFrequency = DataFrequency.MINUTE_5

    # Price adjustments
    use_adjusted_close: bool = True           # Adjust for splits/dividends
    fill_missing: str = 'ffill'               # Forward fill gaps

    # Derived data to compute
    compute_returns: bool = True              # Daily returns
    compute_log_returns: bool = True          # Log returns
    compute_cumulative_returns: bool = True   # Cumulative returns

    @classmethod
    def from_env(cls) -> 'PriceDataConfig':
        """Load from environment variables."""
        intraday_freq_str = os.getenv('PRICE_INTRADAY_FREQ', '5m')
        try:
            intraday_freq = DataFrequency(intraday_freq_str)
        except ValueError:
            intraday_freq = DataFrequency.MINUTE_5

        return cls(
            lookback_days=int(os.getenv('PRICE_LOOKBACK_DAYS', '504')),
            intraday_lookback_days=int(os.getenv('PRICE_INTRADAY_DAYS', '5')),
            fetch_daily=os.getenv('PRICE_FETCH_DAILY', 'true').lower() == 'true',
            fetch_intraday=os.getenv('PRICE_FETCH_INTRADAY', 'true').lower() == 'true',
            intraday_frequency=intraday_freq,
            use_adjusted_close=os.getenv('PRICE_USE_ADJUSTED', 'true').lower() == 'true',
            compute_returns=os.getenv('PRICE_COMPUTE_RETURNS', 'true').lower() == 'true',
            compute_log_returns=os.getenv('PRICE_COMPUTE_LOG_RETURNS', 'true').lower() == 'true',
            compute_cumulative_returns=os.getenv('PRICE_COMPUTE_CUM_RETURNS', 'true').lower() == 'true',
        )


# =============================================================================
# FUNDAMENTAL DATA CONFIG
# =============================================================================

@dataclass
class FundamentalDataConfig:
    """
    Configuration for fundamental/financial data fetching.

    Controls which fundamental metrics are fetched for analysis.
    """
    enabled: bool = True

    # === VALUATION METRICS ===
    fetch_valuation: bool = True
    valuation_metrics: List[str] = field(default_factory=lambda: [
        'pe_ratio',              # Price to Earnings
        'pe_forward',            # Forward P/E
        'peg_ratio',             # P/E to Growth
        'pb_ratio',              # Price to Book
        'ps_ratio',              # Price to Sales
        'pcf_ratio',             # Price to Cash Flow
        'ev_ebitda',             # Enterprise Value / EBITDA
        'ev_revenue',            # Enterprise Value / Revenue
        'ev_fcf',                # Enterprise Value / Free Cash Flow
        'market_cap',            # Market Capitalization
        'enterprise_value',      # Enterprise Value
    ])

    # === PROFITABILITY METRICS ===
    fetch_profitability: bool = True
    profitability_metrics: List[str] = field(default_factory=lambda: [
        'roe',                   # Return on Equity
        'roa',                   # Return on Assets
        'roic',                  # Return on Invested Capital
        'roce',                  # Return on Capital Employed
        'gross_margin',          # Gross Profit Margin
        'operating_margin',      # Operating Margin
        'net_margin',            # Net Profit Margin
        'ebitda_margin',         # EBITDA Margin
        'fcf_margin',            # Free Cash Flow Margin
        'asset_turnover',        # Asset Turnover Ratio
    ])

    # === GROWTH METRICS ===
    fetch_growth: bool = True
    growth_periods: List[str] = field(default_factory=lambda: ['yoy', '3y_cagr', '5y_cagr'])
    growth_metrics: List[str] = field(default_factory=lambda: [
        'revenue_growth',        # Revenue Growth
        'earnings_growth',       # EPS Growth
        'ebitda_growth',         # EBITDA Growth
        'fcf_growth',            # Free Cash Flow Growth
        'book_value_growth',     # Book Value Growth
        'dividend_growth',       # Dividend Growth
        'operating_income_growth',
    ])

    # === FINANCIAL HEALTH / LEVERAGE ===
    fetch_financial_health: bool = True
    health_metrics: List[str] = field(default_factory=lambda: [
        'debt_to_equity',        # Total Debt / Equity
        'debt_to_assets',        # Total Debt / Assets
        'debt_to_ebitda',        # Net Debt / EBITDA
        'interest_coverage',     # EBIT / Interest Expense
        'current_ratio',         # Current Assets / Current Liabilities
        'quick_ratio',           # (Current Assets - Inventory) / Current Liabilities
        'cash_ratio',            # Cash / Current Liabilities
        'altman_z_score',        # Bankruptcy prediction
        'piotroski_f_score',     # Financial strength (0-9)
    ])

    # === CASH FLOW METRICS ===
    fetch_cash_flow: bool = True
    cash_flow_metrics: List[str] = field(default_factory=lambda: [
        'operating_cash_flow',
        'free_cash_flow',
        'fcf_per_share',
        'capex_to_revenue',
        'capex_to_depreciation',
        'cash_conversion_cycle',
        'fcf_yield',             # FCF / Market Cap
        'buyback_yield',         # Share repurchases / Market Cap
    ])

    # === DIVIDEND DATA ===
    fetch_dividends: bool = True
    dividend_metrics: List[str] = field(default_factory=lambda: [
        'dividend_yield',
        'dividend_yield_forward',
        'payout_ratio',
        'dividend_growth_5y',
        'years_of_dividend_growth',
        'ex_dividend_date',
        'dividend_frequency',
    ])

    # === EARNINGS DATA ===
    fetch_earnings: bool = True
    earnings_metrics: List[str] = field(default_factory=lambda: [
        'eps_ttm',               # Trailing 12 months EPS
        'eps_forward',           # Forward EPS estimate
        'earnings_surprise_pct', # Last earnings surprise
        'earnings_beat_rate',    # % of beats last 4 quarters
        'next_earnings_date',
        'eps_revision_1m',       # EPS estimate revision (1 month)
        'eps_revision_3m',       # EPS estimate revision (3 months)
    ])

    # Historical periods for financials
    financial_periods: int = 12              # Quarters of historical data

    @classmethod
    def from_env(cls) -> 'FundamentalDataConfig':
        """Load from environment variables."""
        return cls(
            enabled=os.getenv('FUNDAMENTAL_ENABLED', 'true').lower() == 'true',
            fetch_valuation=os.getenv('FUNDAMENTAL_VALUATION', 'true').lower() == 'true',
            fetch_profitability=os.getenv('FUNDAMENTAL_PROFITABILITY', 'true').lower() == 'true',
            fetch_growth=os.getenv('FUNDAMENTAL_GROWTH', 'true').lower() == 'true',
            fetch_financial_health=os.getenv('FUNDAMENTAL_HEALTH', 'true').lower() == 'true',
            fetch_cash_flow=os.getenv('FUNDAMENTAL_CASH_FLOW', 'true').lower() == 'true',
            fetch_dividends=os.getenv('FUNDAMENTAL_DIVIDENDS', 'true').lower() == 'true',
            fetch_earnings=os.getenv('FUNDAMENTAL_EARNINGS', 'true').lower() == 'true',
            financial_periods=int(os.getenv('FUNDAMENTAL_PERIODS', '12')),
        )


# =============================================================================
# TECHNICAL INDICATORS CONFIG
# =============================================================================

@dataclass
class TechnicalIndicatorsConfig:
    """
    Configuration for technical indicator computation.
    """
    enabled: bool = True

    # === TREND INDICATORS ===
    compute_trend: bool = True
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 12, 21, 26, 50])
    compute_macd: bool = True
    macd_params: Dict[str, int] = field(default_factory=lambda: {'fast': 12, 'slow': 26, 'signal': 9})
    compute_adx: bool = True                  # Average Directional Index
    adx_period: int = 14

    # === MOMENTUM INDICATORS ===
    compute_momentum: bool = True
    rsi_period: int = 14
    stochastic_params: Dict[str, int] = field(default_factory=lambda: {'k': 14, 'd': 3})
    cci_period: int = 20                      # Commodity Channel Index
    williams_r_period: int = 14               # Williams %R
    roc_periods: List[int] = field(default_factory=lambda: [10, 20, 60])  # Rate of Change

    # === VOLATILITY INDICATORS ===
    compute_volatility: bool = True
    bollinger_params: Dict[str, int] = field(default_factory=lambda: {'period': 20, 'std': 2})
    atr_period: int = 14                      # Average True Range
    keltner_params: Dict[str, int] = field(default_factory=lambda: {'period': 20, 'atr_mult': 2})
    donchian_period: int = 20                 # Donchian Channels

    # === VOLUME INDICATORS ===
    compute_volume: bool = True
    volume_sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    compute_obv: bool = True                  # On-Balance Volume
    compute_vwap: bool = True                 # Volume Weighted Average Price
    compute_ad_line: bool = True              # Accumulation/Distribution
    compute_mfi: bool = True                  # Money Flow Index
    mfi_period: int = 14

    # === SUPPORT/RESISTANCE ===
    compute_pivots: bool = True               # Pivot points
    compute_fibonacci: bool = True            # Fibonacci retracements

    @classmethod
    def from_env(cls) -> 'TechnicalIndicatorsConfig':
        """Load from environment variables."""
        sma_str = os.getenv('TECH_SMA_PERIODS', '10,20,50,100,200')
        sma_periods = [int(p) for p in sma_str.split(',')]

        ema_str = os.getenv('TECH_EMA_PERIODS', '9,12,21,26,50')
        ema_periods = [int(p) for p in ema_str.split(',')]

        return cls(
            enabled=os.getenv('TECHNICAL_ENABLED', 'true').lower() == 'true',
            compute_trend=os.getenv('TECH_TREND', 'true').lower() == 'true',
            sma_periods=sma_periods,
            ema_periods=ema_periods,
            compute_macd=os.getenv('TECH_MACD', 'true').lower() == 'true',
            compute_adx=os.getenv('TECH_ADX', 'true').lower() == 'true',
            compute_momentum=os.getenv('TECH_MOMENTUM', 'true').lower() == 'true',
            rsi_period=int(os.getenv('TECH_RSI_PERIOD', '14')),
            compute_volatility=os.getenv('TECH_VOLATILITY', 'true').lower() == 'true',
            atr_period=int(os.getenv('TECH_ATR_PERIOD', '14')),
            compute_volume=os.getenv('TECH_VOLUME', 'true').lower() == 'true',
            compute_pivots=os.getenv('TECH_PIVOTS', 'true').lower() == 'true',
            compute_fibonacci=os.getenv('TECH_FIBONACCI', 'true').lower() == 'true',
        )


# =============================================================================
# RISK METRICS CONFIG
# =============================================================================

@dataclass
class RiskMetricsConfig:
    """
    Configuration for risk metric computation.
    """
    enabled: bool = True

    # === VOLATILITY MEASURES ===
    compute_volatility: bool = True
    volatility_windows: List[int] = field(default_factory=lambda: [20, 60, 252])  # Trading days
    compute_realized_vol: bool = True
    compute_parkinson_vol: bool = True        # High-low volatility
    compute_garman_klass_vol: bool = True     # OHLC volatility

    # === BETA AND CORRELATION ===
    compute_beta: bool = True
    beta_benchmark: str = 'SPY'               # Market benchmark
    beta_windows: List[int] = field(default_factory=lambda: [60, 252])
    compute_correlation_matrix: bool = True
    correlation_window: int = 60

    # === DRAWDOWN METRICS ===
    compute_drawdowns: bool = True
    max_drawdown_window: int = 252            # 1 year lookback

    # === VALUE AT RISK ===
    compute_var: bool = True
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_methods: List[str] = field(default_factory=lambda: ['historical', 'parametric', 'cornish_fisher'])
    var_window: int = 252

    # === CONDITIONAL VAR (EXPECTED SHORTFALL) ===
    compute_cvar: bool = True
    cvar_confidence: float = 0.95

    # === TAIL RISK ===
    compute_tail_metrics: bool = True
    compute_skewness: bool = True
    compute_kurtosis: bool = True
    compute_sortino_ratio: bool = True        # Downside deviation

    # === RISK-ADJUSTED RETURNS ===
    compute_sharpe: bool = True
    compute_calmar: bool = True               # Return / Max Drawdown
    compute_omega: bool = True                # Omega ratio
    risk_free_rate_source: str = 'FRED'       # FRED, fixed, or None

    @classmethod
    def from_env(cls) -> 'RiskMetricsConfig':
        """Load from environment variables."""
        vol_windows_str = os.getenv('RISK_VOL_WINDOWS', '20,60,252')
        vol_windows = [int(w) for w in vol_windows_str.split(',')]

        var_levels_str = os.getenv('RISK_VAR_LEVELS', '0.95,0.99')
        var_levels = [float(l) for l in var_levels_str.split(',')]

        return cls(
            enabled=os.getenv('RISK_METRICS_ENABLED', 'true').lower() == 'true',
            compute_volatility=os.getenv('RISK_VOLATILITY', 'true').lower() == 'true',
            volatility_windows=vol_windows,
            compute_beta=os.getenv('RISK_BETA', 'true').lower() == 'true',
            beta_benchmark=os.getenv('RISK_BENCHMARK', 'SPY'),
            compute_drawdowns=os.getenv('RISK_DRAWDOWNS', 'true').lower() == 'true',
            compute_var=os.getenv('RISK_VAR', 'true').lower() == 'true',
            var_confidence_levels=var_levels,
            compute_cvar=os.getenv('RISK_CVAR', 'true').lower() == 'true',
            compute_sharpe=os.getenv('RISK_SHARPE', 'true').lower() == 'true',
            risk_free_rate_source=os.getenv('RISK_RF_SOURCE', 'FRED'),
        )


# =============================================================================
# MARKET MICROSTRUCTURE CONFIG
# =============================================================================

@dataclass
class MarketMicrostructureConfig:
    """
    Configuration for market microstructure data.
    """
    enabled: bool = True

    # === LIQUIDITY METRICS ===
    fetch_bid_ask: bool = True                # Real-time bid/ask
    compute_spread_metrics: bool = True
    spread_lookback_days: int = 20

    # === VOLUME ANALYSIS ===
    fetch_volume_profile: bool = True
    compute_volume_at_price: bool = True
    compute_relative_volume: bool = True      # vs 20-day average

    # === SHORT INTEREST ===
    fetch_short_interest: bool = True
    short_metrics: List[str] = field(default_factory=lambda: [
        'short_interest',
        'short_ratio',                        # Days to cover
        'short_pct_float',
        'short_pct_outstanding',
    ])

    # === INSTITUTIONAL OWNERSHIP ===
    fetch_institutional: bool = True
    institutional_metrics: List[str] = field(default_factory=lambda: [
        'institutional_ownership_pct',
        'institutional_holders_count',
        'top_10_holders_pct',
        'insider_ownership_pct',
        'insider_transactions_3m',
    ])

    # === FLOAT AND SHARES ===
    fetch_share_stats: bool = True
    share_metrics: List[str] = field(default_factory=lambda: [
        'shares_outstanding',
        'float_shares',
        'shares_short',
        'avg_volume_10d',
        'avg_volume_3m',
    ])

    @classmethod
    def from_env(cls) -> 'MarketMicrostructureConfig':
        """Load from environment variables."""
        return cls(
            enabled=os.getenv('MICROSTRUCTURE_ENABLED', 'true').lower() == 'true',
            fetch_bid_ask=os.getenv('MICRO_BID_ASK', 'true').lower() == 'true',
            fetch_short_interest=os.getenv('MICRO_SHORT_INTEREST', 'true').lower() == 'true',
            fetch_institutional=os.getenv('MICRO_INSTITUTIONAL', 'true').lower() == 'true',
            fetch_share_stats=os.getenv('MICRO_SHARE_STATS', 'true').lower() == 'true',
        )


# =============================================================================
# ALTERNATIVE DATA CONFIG
# =============================================================================

@dataclass
class AlternativeDataConfig:
    """
    Configuration for alternative/sentiment data.
    """
    enabled: bool = False                     # Disabled by default (requires API keys)

    # === ANALYST DATA ===
    fetch_analyst_ratings: bool = True
    analyst_metrics: List[str] = field(default_factory=lambda: [
        'analyst_rating',                     # Buy/Hold/Sell consensus
        'analyst_target_price',
        'analyst_target_high',
        'analyst_target_low',
        'analyst_count',
        'rating_change_1m',
        'upgrades_downgrades_3m',
    ])

    # === ESG DATA ===
    fetch_esg: bool = False
    esg_metrics: List[str] = field(default_factory=lambda: [
        'esg_score',
        'environmental_score',
        'social_score',
        'governance_score',
        'controversy_score',
    ])

    # === NEWS SENTIMENT ===
    fetch_news_sentiment: bool = False
    sentiment_sources: List[str] = field(default_factory=lambda: ['finnhub', 'alphavantage'])
    sentiment_lookback_days: int = 7

    # === SOCIAL SENTIMENT ===
    fetch_social_sentiment: bool = False
    social_sources: List[str] = field(default_factory=lambda: ['stocktwits', 'reddit'])

    # === OPTIONS FLOW ===
    fetch_options_flow: bool = False
    options_metrics: List[str] = field(default_factory=lambda: [
        'put_call_ratio',
        'options_volume',
        'implied_volatility',
        'iv_percentile',
        'iv_rank',
        'max_pain',
    ])

    @classmethod
    def from_env(cls) -> 'AlternativeDataConfig':
        """Load from environment variables."""
        return cls(
            enabled=os.getenv('ALT_DATA_ENABLED', 'false').lower() == 'true',
            fetch_analyst_ratings=os.getenv('ALT_ANALYST', 'true').lower() == 'true',
            fetch_esg=os.getenv('ALT_ESG', 'false').lower() == 'true',
            fetch_news_sentiment=os.getenv('ALT_NEWS', 'false').lower() == 'true',
            fetch_social_sentiment=os.getenv('ALT_SOCIAL', 'false').lower() == 'true',
            fetch_options_flow=os.getenv('ALT_OPTIONS', 'false').lower() == 'true',
        )


# =============================================================================
# CORPORATE EVENTS CONFIG
# =============================================================================

@dataclass
class CorporateEventsConfig:
    """
    Configuration for corporate event data.
    """
    enabled: bool = True

    # === EARNINGS CALENDAR ===
    fetch_earnings_calendar: bool = True
    earnings_lookahead_days: int = 30
    earnings_lookback_days: int = 90

    # === DIVIDEND CALENDAR ===
    fetch_dividend_calendar: bool = True
    dividend_lookahead_days: int = 60

    # === CORPORATE ACTIONS ===
    fetch_splits: bool = True
    fetch_spinoffs: bool = True
    fetch_mergers: bool = True
    corporate_actions_lookback: int = 365

    # === INSIDER TRADING ===
    fetch_insider_trades: bool = True
    insider_lookback_days: int = 90
    insider_metrics: List[str] = field(default_factory=lambda: [
        'insider_buys_3m',
        'insider_sells_3m',
        'insider_net_shares',
        'insider_net_value',
    ])

    # === SEC FILINGS ===
    fetch_sec_filings: bool = False           # 10-K, 10-Q, 8-K
    sec_filing_types: List[str] = field(default_factory=lambda: ['10-K', '10-Q', '8-K'])

    @classmethod
    def from_env(cls) -> 'CorporateEventsConfig':
        """Load from environment variables."""
        return cls(
            enabled=os.getenv('EVENTS_ENABLED', 'true').lower() == 'true',
            fetch_earnings_calendar=os.getenv('EVENTS_EARNINGS', 'true').lower() == 'true',
            fetch_dividend_calendar=os.getenv('EVENTS_DIVIDENDS', 'true').lower() == 'true',
            fetch_splits=os.getenv('EVENTS_SPLITS', 'true').lower() == 'true',
            fetch_insider_trades=os.getenv('EVENTS_INSIDER', 'true').lower() == 'true',
            fetch_sec_filings=os.getenv('EVENTS_SEC', 'false').lower() == 'true',
        )


# =============================================================================
# SELECTION & RANKING CONFIG
# =============================================================================

@dataclass
class SelectionConfig:
    """
    Configuration for stock selection and ranking.
    """
    # Selection method
    method: SelectionMethod = SelectionMethod.MULTI_FACTOR
    num_stocks_select: int = 20               # Final portfolio size

    # === FACTOR WEIGHTS (for multi-factor) ===
    momentum_weight: float = 0.25
    value_weight: float = 0.25
    quality_weight: float = 0.25
    growth_weight: float = 0.15
    low_vol_weight: float = 0.10

    # === MOMENTUM PARAMETERS ===
    momentum_lookback_short: int = 20         # 1 month
    momentum_lookback_medium: int = 60        # 3 months
    momentum_lookback_long: int = 252         # 12 months
    momentum_skip_recent: int = 5             # Skip most recent days (mean reversion)

    # === VALUE PARAMETERS ===
    value_metrics: List[str] = field(default_factory=lambda: [
        'pe_ratio', 'pb_ratio', 'ev_ebitda', 'fcf_yield'
    ])

    # === QUALITY PARAMETERS ===
    quality_metrics: List[str] = field(default_factory=lambda: [
        'roe', 'roic', 'gross_margin', 'debt_to_equity', 'interest_coverage'
    ])

    # === FILTERING BEFORE RANKING ===
    exclude_recent_earnings: int = 5          # Days to exclude around earnings
    exclude_if_no_fundamentals: bool = True
    min_analyst_coverage: int = 0             # Minimum analysts covering

    # === SECTOR CONSTRAINTS ===
    max_sector_weight: float = 0.30           # Max 30% in single sector
    min_sectors: int = 3                      # Minimum number of sectors

    @classmethod
    def from_env(cls) -> 'SelectionConfig':
        """Load from environment variables."""
        method_str = os.getenv('SELECTION_METHOD', 'multi_factor').lower()
        try:
            method = SelectionMethod(method_str)
        except ValueError:
            method = SelectionMethod.MULTI_FACTOR

        return cls(
            method=method,
            num_stocks_select=int(os.getenv('NUM_STOCKS_SELECT', '20')),
            momentum_weight=float(os.getenv('FACTOR_MOMENTUM_WEIGHT', '0.25')),
            value_weight=float(os.getenv('FACTOR_VALUE_WEIGHT', '0.25')),
            quality_weight=float(os.getenv('FACTOR_QUALITY_WEIGHT', '0.25')),
            growth_weight=float(os.getenv('FACTOR_GROWTH_WEIGHT', '0.15')),
            low_vol_weight=float(os.getenv('FACTOR_LOWVOL_WEIGHT', '0.10')),
            momentum_lookback_short=int(os.getenv('MOMENTUM_SHORT', '20')),
            momentum_lookback_medium=int(os.getenv('MOMENTUM_MEDIUM', '60')),
            momentum_lookback_long=int(os.getenv('MOMENTUM_LONG', '252')),
            max_sector_weight=float(os.getenv('MAX_SECTOR_WEIGHT', '0.30')),
            min_sectors=int(os.getenv('MIN_SECTORS', '3')),
        )


# =============================================================================
# DATA PROVIDER CONFIG
# =============================================================================

@dataclass
class DataProviderConfig:
    """
    Configuration for data providers and caching.
    """
    # Primary provider
    primary_provider: str = 'yfinance'        # yfinance, alpaca, polygon
    fallback_providers: List[str] = field(default_factory=lambda: ['alpaca', 'alphavantage'])

    # Live quotes
    enable_live_quotes: bool = True
    live_quote_provider: str = 'alpaca'

    # Caching
    cache_enabled: bool = True
    cache_hours: int = 4                      # TTL for cached data
    cache_dir: str = 'data/cache'

    # Rate limiting
    rate_limit_requests_per_min: int = 100
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    # Timeouts
    request_timeout_seconds: int = 30

    @classmethod
    def from_env(cls) -> 'DataProviderConfig':
        """Load from environment variables."""
        fallback_str = os.getenv('DATA_FALLBACK_PROVIDERS', 'alpaca,alphavantage')
        fallback_providers = [p.strip() for p in fallback_str.split(',')]

        return cls(
            primary_provider=os.getenv('DATA_PRIMARY_PROVIDER', 'yfinance'),
            fallback_providers=fallback_providers,
            enable_live_quotes=os.getenv('ENABLE_LIVE_QUOTES', 'true').lower() == 'true',
            live_quote_provider=os.getenv('LIVE_QUOTE_PROVIDER', 'alpaca'),
            cache_enabled=os.getenv('DATA_CACHE_ENABLED', 'true').lower() == 'true',
            cache_hours=int(os.getenv('DATA_CACHE_HOURS', '4')),
            cache_dir=os.getenv('DATA_CACHE_DIR', 'data/cache'),
            rate_limit_requests_per_min=int(os.getenv('DATA_RATE_LIMIT', '100')),
            request_timeout_seconds=int(os.getenv('DATA_TIMEOUT', '30')),
        )


# =============================================================================
# MASTER CONFIG - COMBINES ALL
# =============================================================================

@dataclass
class SecurityDataConfig:
    """
    Master configuration for all security data fetching.

    This is the main entry point for data configuration.
    Combines all sub-configs into a single comprehensive config object.
    """
    # Sub-configurations
    universe: UniverseScreeningConfig = field(default_factory=UniverseScreeningConfig)
    price: PriceDataConfig = field(default_factory=PriceDataConfig)
    fundamentals: FundamentalDataConfig = field(default_factory=FundamentalDataConfig)
    technical: TechnicalIndicatorsConfig = field(default_factory=TechnicalIndicatorsConfig)
    risk: RiskMetricsConfig = field(default_factory=RiskMetricsConfig)
    microstructure: MarketMicrostructureConfig = field(default_factory=MarketMicrostructureConfig)
    alternative: AlternativeDataConfig = field(default_factory=AlternativeDataConfig)
    events: CorporateEventsConfig = field(default_factory=CorporateEventsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    providers: DataProviderConfig = field(default_factory=DataProviderConfig)

    @classmethod
    def from_env(cls) -> 'SecurityDataConfig':
        """Load complete configuration from environment variables."""
        return cls(
            universe=UniverseScreeningConfig.from_env(),
            price=PriceDataConfig.from_env(),
            fundamentals=FundamentalDataConfig.from_env(),
            technical=TechnicalIndicatorsConfig.from_env(),
            risk=RiskMetricsConfig.from_env(),
            microstructure=MarketMicrostructureConfig.from_env(),
            alternative=AlternativeDataConfig.from_env(),
            events=CorporateEventsConfig.from_env(),
            selection=SelectionConfig.from_env(),
            providers=DataProviderConfig.from_env(),
        )

    def get_all_metrics(self) -> Dict[str, List[str]]:
        """Get all metrics that will be fetched, organized by category."""
        metrics = {}

        if self.fundamentals.enabled:
            metrics['valuation'] = self.fundamentals.valuation_metrics
            metrics['profitability'] = self.fundamentals.profitability_metrics
            metrics['growth'] = self.fundamentals.growth_metrics
            metrics['financial_health'] = self.fundamentals.health_metrics
            metrics['cash_flow'] = self.fundamentals.cash_flow_metrics
            metrics['dividends'] = self.fundamentals.dividend_metrics
            metrics['earnings'] = self.fundamentals.earnings_metrics

        if self.microstructure.enabled:
            metrics['short_interest'] = self.microstructure.short_metrics
            metrics['institutional'] = self.microstructure.institutional_metrics
            metrics['share_stats'] = self.microstructure.share_metrics

        if self.alternative.enabled:
            metrics['analyst'] = self.alternative.analyst_metrics
            if self.alternative.fetch_esg:
                metrics['esg'] = self.alternative.esg_metrics
            if self.alternative.fetch_options_flow:
                metrics['options'] = self.alternative.options_metrics

        if self.events.enabled and self.events.fetch_insider_trades:
            metrics['insider'] = self.events.insider_metrics

        return metrics

    def to_dict(self) -> Dict:
        """Export configuration to dictionary."""
        return {
            'universe': {
                'type': self.universe.universe_type.value,
                'max_size': self.universe.max_universe_size,
                'min_market_cap': self.universe.min_market_cap,
                'min_avg_volume': self.universe.min_avg_volume,
            },
            'selection': {
                'method': self.selection.method.value,
                'num_stocks': self.selection.num_stocks_select,
            },
            'data_categories': {
                'fundamentals': self.fundamentals.enabled,
                'technical': self.technical.enabled,
                'risk_metrics': self.risk.enabled,
                'microstructure': self.microstructure.enabled,
                'alternative': self.alternative.enabled,
                'events': self.events.enabled,
            },
            'providers': {
                'primary': self.providers.primary_provider,
                'live_quotes': self.providers.enable_live_quotes,
                'cache_hours': self.providers.cache_hours,
            },
        }


def load_security_data_config() -> SecurityDataConfig:
    """
    Load security data configuration from environment.

    This is the main entry point for loading data configuration.
    """
    return SecurityDataConfig.from_env()


if __name__ == "__main__":
    # Test configuration loading
    from dotenv import load_dotenv
    load_dotenv()

    config = load_security_data_config()

    print("=" * 60)
    print("SECURITY DATA CONFIGURATION")
    print("=" * 60)

    print(f"\n=== Universe Screening ===")
    print(f"  Type: {config.universe.universe_type.value}")
    print(f"  Max size: {config.universe.max_universe_size}")
    print(f"  Min market cap: ${config.universe.min_market_cap:,.0f}")
    print(f"  Min avg volume: {config.universe.min_avg_volume:,}")

    print(f"\n=== Selection ===")
    print(f"  Method: {config.selection.method.value}")
    print(f"  Stocks to select: {config.selection.num_stocks_select}")

    print(f"\n=== Data Categories Enabled ===")
    print(f"  Fundamentals: {config.fundamentals.enabled}")
    print(f"  Technical: {config.technical.enabled}")
    print(f"  Risk metrics: {config.risk.enabled}")
    print(f"  Microstructure: {config.microstructure.enabled}")
    print(f"  Alternative data: {config.alternative.enabled}")
    print(f"  Corporate events: {config.events.enabled}")

    print(f"\n=== All Metrics ({sum(len(v) for v in config.get_all_metrics().values())} total) ===")
    for category, metrics in config.get_all_metrics().items():
        print(f"  {category}: {len(metrics)} metrics")
