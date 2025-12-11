"""
Standardized Data Types for Security Analysis
==============================================
All data structures are independent of data source and analysis method.
These can be serialized, cached, and passed to any analysis/selection algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json


@dataclass
class PriceData:
    """
    Historical price data for a security.

    Attributes:
        symbol: Ticker symbol
        prices: DataFrame with OHLCV columns (Open, High, Low, Close, Volume, Adj Close)
        currency: Price currency (USD, EUR, etc.)
        source: Data provider (yahoo, alpaca, polygon)
        fetched_at: Timestamp when data was fetched
    """
    symbol: str
    prices: pd.DataFrame  # Index: Date, Columns: Open, High, Low, Close, Volume, Adj Close
    currency: str = "USD"
    source: str = "unknown"
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def returns(self) -> pd.Series:
        """Daily returns from adjusted close."""
        if 'Adj Close' in self.prices.columns:
            return self.prices['Adj Close'].pct_change().dropna()
        return self.prices['Close'].pct_change().dropna()

    @property
    def log_returns(self) -> pd.Series:
        """Log returns from adjusted close."""
        if 'Adj Close' in self.prices.columns:
            return np.log(self.prices['Adj Close'] / self.prices['Adj Close'].shift(1)).dropna()
        return np.log(self.prices['Close'] / self.prices['Close'].shift(1)).dropna()

    @property
    def latest_price(self) -> float:
        """Most recent closing price."""
        if 'Adj Close' in self.prices.columns:
            return float(self.prices['Adj Close'].iloc[-1])
        return float(self.prices['Close'].iloc[-1])

    @property
    def days_available(self) -> int:
        """Number of trading days available."""
        return len(self.prices)

    def to_dict(self) -> Dict:
        """Convert to serializable dict (excluding DataFrame)."""
        return {
            'symbol': self.symbol,
            'currency': self.currency,
            'source': self.source,
            'fetched_at': self.fetched_at.isoformat(),
            'days_available': self.days_available,
            'latest_price': self.latest_price,
            'date_range': {
                'start': str(self.prices.index.min()),
                'end': str(self.prices.index.max()),
            }
        }


@dataclass
class FundamentalData:
    """
    Fundamental data for a security.

    All fields are optional as availability varies by source and security type.
    """
    symbol: str

    # Company Info
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    market_cap: Optional[float] = None  # In USD
    enterprise_value: Optional[float] = None

    # Valuation Metrics
    pe_ratio: Optional[float] = None  # Trailing P/E
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None  # Price/Book
    ps_ratio: Optional[float] = None  # Price/Sales
    ev_ebitda: Optional[float] = None
    ev_revenue: Optional[float] = None

    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    roic: Optional[float] = None  # Return on Invested Capital

    # Growth Metrics
    revenue_growth: Optional[float] = None  # YoY
    earnings_growth: Optional[float] = None  # YoY
    revenue_growth_quarterly: Optional[float] = None
    earnings_growth_quarterly: Optional[float] = None

    # Financial Health
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    total_debt: Optional[float] = None
    total_cash: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None

    # Per Share Data
    eps_ttm: Optional[float] = None  # Trailing 12 months
    eps_forward: Optional[float] = None
    book_value_per_share: Optional[float] = None
    revenue_per_share: Optional[float] = None

    # Dividend Info
    dividend_yield: Optional[float] = None
    dividend_rate: Optional[float] = None
    payout_ratio: Optional[float] = None
    ex_dividend_date: Optional[str] = None

    # Analyst Info
    target_mean_price: Optional[float] = None
    target_high_price: Optional[float] = None
    target_low_price: Optional[float] = None
    num_analysts: Optional[int] = None
    recommendation: Optional[str] = None  # buy, hold, sell

    # Shares Info
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    shares_short: Optional[float] = None
    short_ratio: Optional[float] = None
    short_percent_of_float: Optional[float] = None

    # Institutional
    institutional_ownership: Optional[float] = None
    insider_ownership: Optional[float] = None

    # Metadata
    source: str = "unknown"
    fetched_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['fetched_at'] = self.fetched_at.isoformat()
        return result

    def get_valuation_score(self) -> Optional[float]:
        """
        Simple valuation score (lower = cheaper).
        Normalizes P/E, P/B, EV/EBITDA into a composite score.
        """
        scores = []

        if self.pe_ratio and 0 < self.pe_ratio < 100:
            # Normalize P/E (0-50 range maps to 0-1)
            scores.append(min(self.pe_ratio / 50, 2))

        if self.pb_ratio and 0 < self.pb_ratio < 20:
            scores.append(min(self.pb_ratio / 5, 2))

        if self.ev_ebitda and 0 < self.ev_ebitda < 50:
            scores.append(min(self.ev_ebitda / 15, 2))

        return np.mean(scores) if scores else None

    def get_quality_score(self) -> Optional[float]:
        """
        Quality score based on profitability and financial health.
        Higher = better quality.
        """
        scores = []

        if self.roe is not None and -1 < self.roe < 1:
            scores.append(max(0, min(self.roe * 2, 1)))  # ROE scaled

        if self.profit_margin is not None and -1 < self.profit_margin < 1:
            scores.append(max(0, min(self.profit_margin * 3, 1)))

        if self.current_ratio is not None and 0 < self.current_ratio < 10:
            # Current ratio around 1.5-2 is ideal
            scores.append(max(0, 1 - abs(self.current_ratio - 1.5) / 3))

        if self.debt_to_equity is not None and self.debt_to_equity >= 0:
            # Lower D/E is better (cap at 2)
            scores.append(max(0, 1 - self.debt_to_equity / 3))

        return np.mean(scores) if scores else None


@dataclass
class TechnicalIndicators:
    """
    Technical indicators computed from price data.

    All indicators are pre-computed and stored for analysis.
    """
    symbol: str
    as_of_date: date

    # Price Momentum
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    return_60d: Optional[float] = None
    return_252d: Optional[float] = None  # 1 year

    # Moving Averages (as % deviation from current price)
    sma_10_dev: Optional[float] = None  # (price - SMA10) / SMA10
    sma_20_dev: Optional[float] = None
    sma_50_dev: Optional[float] = None
    sma_200_dev: Optional[float] = None
    ema_12_dev: Optional[float] = None
    ema_26_dev: Optional[float] = None

    # Volatility
    volatility_20d: Optional[float] = None  # Annualized
    volatility_60d: Optional[float] = None
    volatility_252d: Optional[float] = None
    atr_14: Optional[float] = None  # Average True Range (% of price)

    # Momentum Oscillators
    rsi_14: Optional[float] = None  # 0-100
    rsi_7: Optional[float] = None
    macd: Optional[float] = None  # MACD line value
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stochastic_k: Optional[float] = None  # %K
    stochastic_d: Optional[float] = None  # %D

    # Volume Indicators
    volume_sma_20_ratio: Optional[float] = None  # Current vol / SMA(20)
    obv_trend: Optional[float] = None  # OBV change direction

    # Trend Indicators
    adx_14: Optional[float] = None  # Average Directional Index
    plus_di: Optional[float] = None  # +DI
    minus_di: Optional[float] = None  # -DI

    # Bollinger Bands
    bb_position: Optional[float] = None  # 0=lower, 0.5=middle, 1=upper
    bb_width: Optional[float] = None  # (upper-lower)/middle

    # Support/Resistance
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    pct_from_52w_low: Optional[float] = None

    # Risk-Adjusted
    sharpe_60d: Optional[float] = None  # 60-day Sharpe (assuming Rf=0)
    sortino_60d: Optional[float] = None  # 60-day Sortino

    source: str = "computed"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['as_of_date'] = str(self.as_of_date)
        return result

    def to_feature_vector(self) -> Dict[str, float]:
        """
        Convert to flat feature dict for ML models.
        Only includes non-None numeric values.
        """
        features = {}
        for key, value in asdict(self).items():
            if key in ('symbol', 'as_of_date', 'source'):
                continue
            if value is not None and isinstance(value, (int, float)):
                features[key] = float(value)
        return features


@dataclass
class RiskMetrics:
    """
    Risk metrics for a security.
    """
    symbol: str
    as_of_date: date

    # Volatility Metrics
    volatility_ann: Optional[float] = None  # Annualized realized vol
    downside_volatility: Optional[float] = None  # Semi-deviation

    # Value at Risk
    var_95_1d: Optional[float] = None  # 95% VaR, 1-day horizon
    var_99_1d: Optional[float] = None
    cvar_95_1d: Optional[float] = None  # Conditional VaR (Expected Shortfall)

    # Tail Risk
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    max_drawdown: Optional[float] = None  # Maximum drawdown over period

    # Beta & Correlation (vs market, e.g., SPY)
    beta: Optional[float] = None
    correlation_spy: Optional[float] = None
    r_squared: Optional[float] = None  # % variance explained by market

    # Idiosyncratic Risk
    idiosyncratic_vol: Optional[float] = None  # Residual volatility

    source: str = "computed"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['as_of_date'] = str(self.as_of_date)
        return result


@dataclass
class SectorInfo:
    """
    Sector and industry classification.
    """
    symbol: str

    # GICS Classification
    sector: Optional[str] = None
    industry_group: Optional[str] = None
    industry: Optional[str] = None
    sub_industry: Optional[str] = None

    # Sector ETF proxy (for relative analysis)
    sector_etf: Optional[str] = None

    # Peer Group
    peers: List[str] = field(default_factory=list)

    source: str = "unknown"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LiveQuote:
    """
    Real-time quote data (from Alpaca or similar).
    """
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    volume: int
    timestamp: datetime
    source: str = "alpaca"

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid."""
        return self.spread / self.mid if self.mid > 0 else 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'mid': self.mid,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'volume': self.volume,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
        }
