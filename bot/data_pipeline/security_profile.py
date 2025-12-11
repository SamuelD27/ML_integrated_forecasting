"""
Security Profile Fetcher
=========================
Fetches comprehensive data profiles for securities from multiple sources.

Combines data from:
- Yahoo Finance (fundamentals, historical prices, options)
- Alpaca (real-time quotes, intraday data)
- Computed metrics (technicals, risk)

Architecture:
    SecurityProfileFetcher
        ├── YahooDataFetcher (fundamentals, daily prices)
        ├── AlpacaDataFetcher (real-time, intraday)
        └── TechnicalCalculator (derived metrics)
"""

from __future__ import annotations

import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .data_types import (
    PriceData,
    FundamentalData,
    TechnicalIndicators,
    RiskMetrics,
    SectorInfo,
    LiveQuote,
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityProfile:
    """
    Complete data profile for a security.

    Aggregates all available data from multiple sources into a single,
    standardized structure that can be used by any analysis method.
    """
    symbol: str
    fetched_at: datetime = field(default_factory=datetime.now)

    # Core Data (always populated)
    price_data: Optional[PriceData] = None

    # Fundamental Data (if available)
    fundamentals: Optional[FundamentalData] = None

    # Technical Indicators (computed from price data)
    technicals: Optional[TechnicalIndicators] = None

    # Risk Metrics (computed)
    risk_metrics: Optional[RiskMetrics] = None

    # Sector Info
    sector_info: Optional[SectorInfo] = None

    # Real-time Quote (if available)
    live_quote: Optional[LiveQuote] = None

    # Data Quality Flags
    has_sufficient_history: bool = False
    has_fundamentals: bool = False
    has_live_data: bool = False
    data_sources: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if profile has minimum required data."""
        return self.price_data is not None and self.has_sufficient_history

    def get_latest_price(self) -> Optional[float]:
        """Get most recent price (live if available, else historical)."""
        if self.live_quote:
            return self.live_quote.mid
        if self.price_data:
            return self.price_data.latest_price
        return None

    def to_feature_dict(self) -> Dict[str, float]:
        """
        Convert profile to flat feature dictionary for ML models.

        Returns:
            Dict with all numeric features
        """
        features = {'symbol': self.symbol}

        # Add technical features
        if self.technicals:
            features.update(self.technicals.to_feature_vector())

        # Add fundamental features (if available)
        if self.fundamentals:
            fund_dict = self.fundamentals.to_dict()
            for key, value in fund_dict.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    features[f'fund_{key}'] = float(value)

        # Add risk features
        if self.risk_metrics:
            risk_dict = self.risk_metrics.to_dict()
            for key, value in risk_dict.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    features[f'risk_{key}'] = float(value)

        return features

    def to_dict(self) -> Dict:
        """Convert entire profile to dictionary."""
        return {
            'symbol': self.symbol,
            'fetched_at': self.fetched_at.isoformat(),
            'price_data': self.price_data.to_dict() if self.price_data else None,
            'fundamentals': self.fundamentals.to_dict() if self.fundamentals else None,
            'technicals': self.technicals.to_dict() if self.technicals else None,
            'risk_metrics': self.risk_metrics.to_dict() if self.risk_metrics else None,
            'sector_info': self.sector_info.to_dict() if self.sector_info else None,
            'live_quote': self.live_quote.to_dict() if self.live_quote else None,
            'has_sufficient_history': self.has_sufficient_history,
            'has_fundamentals': self.has_fundamentals,
            'has_live_data': self.has_live_data,
            'data_sources': self.data_sources,
            'errors': self.errors,
        }


class SecurityProfileFetcher:
    """
    Fetches comprehensive security profiles from multiple data sources.

    Usage:
        fetcher = SecurityProfileFetcher()
        profile = fetcher.fetch_profile('AAPL')
        profiles = fetcher.fetch_profiles(['AAPL', 'MSFT', 'GOOGL'])
    """

    # Minimum days of history required for full analysis
    MIN_HISTORY_DAYS = 60
    PREFERRED_HISTORY_DAYS = 252

    def __init__(
        self,
        enable_alpaca: bool = True,
        enable_fundamentals: bool = True,
        enable_technicals: bool = True,
        enable_risk_metrics: bool = True,
        lookback_days: int = 252,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 1,
    ):
        """
        Initialize security profile fetcher.

        Args:
            enable_alpaca: Fetch real-time data from Alpaca
            enable_fundamentals: Fetch fundamental data
            enable_technicals: Compute technical indicators
            enable_risk_metrics: Compute risk metrics
            lookback_days: Days of historical data to fetch
            cache_enabled: Enable caching
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.enable_alpaca = enable_alpaca
        self.enable_fundamentals = enable_fundamentals
        self.enable_technicals = enable_technicals
        self.enable_risk_metrics = enable_risk_metrics
        self.lookback_days = lookback_days
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = cache_ttl_hours

        # Cache directory
        self.cache_dir = Path("data/cache/profiles")
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Alpaca client (if available and enabled)
        self._alpaca_client = None
        if enable_alpaca:
            self._init_alpaca()

    def _init_alpaca(self) -> None:
        """Initialize Alpaca client."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')

            if api_key and api_secret:
                self._alpaca_client = StockHistoricalDataClient(
                    api_key=api_key,
                    secret_key=api_secret
                )
                logger.info("Alpaca client initialized")
            else:
                logger.warning("Alpaca credentials not found, live data disabled")

        except ImportError:
            logger.warning("alpaca-py not installed, live data disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}")

    def fetch_profile(self, symbol: str) -> SecurityProfile:
        """
        Fetch complete security profile for a single symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            SecurityProfile with all available data
        """
        logger.debug(f"Fetching profile for {symbol}")

        profile = SecurityProfile(symbol=symbol)

        # Fetch from multiple sources in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            # Submit price data fetch (always)
            futures['prices'] = executor.submit(self._fetch_price_data, symbol)

            # Submit fundamentals fetch
            if self.enable_fundamentals:
                futures['fundamentals'] = executor.submit(self._fetch_fundamentals, symbol)

            # Submit live quote
            if self.enable_alpaca and self._alpaca_client:
                futures['live'] = executor.submit(self._fetch_live_quote, symbol)

            # Collect results
            for key, future in futures.items():
                try:
                    result = future.result(timeout=30)

                    if key == 'prices' and result:
                        profile.price_data = result
                        profile.data_sources.append('yahoo')
                        profile.has_sufficient_history = result.days_available >= self.MIN_HISTORY_DAYS

                    elif key == 'fundamentals' and result:
                        profile.fundamentals = result
                        profile.has_fundamentals = True

                    elif key == 'live' and result:
                        profile.live_quote = result
                        profile.has_live_data = True
                        profile.data_sources.append('alpaca')

                except Exception as e:
                    error_msg = f"Failed to fetch {key} for {symbol}: {e}"
                    logger.warning(error_msg)
                    profile.errors.append(error_msg)

        # Compute derived metrics (requires price data)
        if profile.price_data and profile.has_sufficient_history:
            if self.enable_technicals:
                try:
                    profile.technicals = self._compute_technicals(profile.price_data)
                except Exception as e:
                    profile.errors.append(f"Technical computation failed: {e}")

            if self.enable_risk_metrics:
                try:
                    profile.risk_metrics = self._compute_risk_metrics(profile.price_data)
                except Exception as e:
                    profile.errors.append(f"Risk metrics computation failed: {e}")

        # Extract sector info
        if profile.fundamentals:
            profile.sector_info = SectorInfo(
                symbol=symbol,
                sector=profile.fundamentals.sector,
                industry=profile.fundamentals.industry,
                source='yahoo',
            )

        return profile

    def fetch_profiles(
        self,
        symbols: List[str],
        max_workers: int = 10,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, SecurityProfile]:
        """
        Fetch profiles for multiple symbols in parallel.

        Args:
            symbols: List of ticker symbols
            max_workers: Maximum parallel fetches
            progress_callback: Optional callback(completed, total)

        Returns:
            Dict mapping symbol -> SecurityProfile
        """
        logger.info(f"Fetching profiles for {len(symbols)} symbols...")

        profiles = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_profile, sym): sym for sym in symbols}

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    profile = future.result(timeout=60)
                    profiles[symbol] = profile
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
                    profiles[symbol] = SecurityProfile(
                        symbol=symbol,
                        errors=[str(e)],
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(symbols))

        logger.info(f"Fetched {len(profiles)} profiles")
        return profiles

    def _fetch_price_data(self, symbol: str) -> Optional[PriceData]:
        """Fetch historical price data from Yahoo Finance."""
        try:
            import yfinance as yf

            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 30)

            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                auto_adjust=False,  # Keep raw OHLC and Adj Close
            )

            if df.empty:
                logger.warning(f"No price data for {symbol}")
                return None

            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing columns for {symbol}")
                return None

            # Add Adj Close if not present
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']

            # Clean up
            df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].dropna()

            return PriceData(
                symbol=symbol,
                prices=df,
                currency="USD",
                source="yahoo",
                fetched_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Price fetch failed for {symbol}: {e}")
            return None

    def _fetch_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch fundamental data from Yahoo Finance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            return FundamentalData(
                symbol=symbol,
                company_name=info.get('shortName') or info.get('longName'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                country=info.get('country'),
                exchange=info.get('exchange'),
                market_cap=info.get('marketCap'),
                enterprise_value=info.get('enterpriseValue'),

                # Valuation
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                pb_ratio=info.get('priceToBook'),
                ps_ratio=info.get('priceToSalesTrailing12Months'),
                ev_ebitda=info.get('enterpriseToEbitda'),
                ev_revenue=info.get('enterpriseToRevenue'),

                # Profitability
                profit_margin=info.get('profitMargins'),
                operating_margin=info.get('operatingMargins'),
                gross_margin=info.get('grossMargins'),
                roe=info.get('returnOnEquity'),
                roa=info.get('returnOnAssets'),

                # Growth
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),
                revenue_growth_quarterly=info.get('revenueQuarterlyGrowth'),
                earnings_growth_quarterly=info.get('earningsQuarterlyGrowth'),

                # Financial Health
                current_ratio=info.get('currentRatio'),
                quick_ratio=info.get('quickRatio'),
                debt_to_equity=info.get('debtToEquity'),
                total_debt=info.get('totalDebt'),
                total_cash=info.get('totalCash'),
                free_cash_flow=info.get('freeCashflow'),
                operating_cash_flow=info.get('operatingCashflow'),

                # Per Share
                eps_ttm=info.get('trailingEps'),
                eps_forward=info.get('forwardEps'),
                book_value_per_share=info.get('bookValue'),
                revenue_per_share=info.get('revenuePerShare'),

                # Dividend
                dividend_yield=info.get('dividendYield'),
                dividend_rate=info.get('dividendRate'),
                payout_ratio=info.get('payoutRatio'),
                ex_dividend_date=info.get('exDividendDate'),

                # Analyst
                target_mean_price=info.get('targetMeanPrice'),
                target_high_price=info.get('targetHighPrice'),
                target_low_price=info.get('targetLowPrice'),
                num_analysts=info.get('numberOfAnalystOpinions'),
                recommendation=info.get('recommendationKey'),

                # Shares
                shares_outstanding=info.get('sharesOutstanding'),
                float_shares=info.get('floatShares'),
                shares_short=info.get('sharesShort'),
                short_ratio=info.get('shortRatio'),
                short_percent_of_float=info.get('shortPercentOfFloat'),

                # Institutional
                institutional_ownership=info.get('heldPercentInstitutions'),
                insider_ownership=info.get('heldPercentInsiders'),

                source="yahoo",
                fetched_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Fundamentals fetch failed for {symbol}: {e}")
            return None

    def _fetch_live_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Fetch live quote from Alpaca."""
        if not self._alpaca_client:
            return None

        try:
            from alpaca.data.requests import StockLatestQuoteRequest

            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._alpaca_client.get_stock_latest_quote(request)

            if symbol not in quotes:
                return None

            quote = quotes[symbol]

            return LiveQuote(
                symbol=symbol,
                bid=float(quote.bid_price),
                ask=float(quote.ask_price),
                last=float(quote.ask_price),  # Alpaca quotes don't have 'last'
                bid_size=int(quote.bid_size),
                ask_size=int(quote.ask_size),
                volume=0,  # Would need separate trades request
                timestamp=datetime.now(),
                source="alpaca",
            )

        except Exception as e:
            logger.debug(f"Live quote failed for {symbol}: {e}")
            return None

    def _compute_technicals(self, price_data: PriceData) -> TechnicalIndicators:
        """Compute technical indicators from price data."""
        prices = price_data.prices
        close = prices['Adj Close'] if 'Adj Close' in prices.columns else prices['Close']
        high = prices['High']
        low = prices['Low']
        volume = prices['Volume']

        returns = close.pct_change()

        today = date.today()

        indicators = TechnicalIndicators(
            symbol=price_data.symbol,
            as_of_date=today,
        )

        # Returns
        if len(returns) >= 1:
            indicators.return_1d = float(returns.iloc[-1]) if not pd.isna(returns.iloc[-1]) else None
        if len(returns) >= 5:
            indicators.return_5d = float((close.iloc[-1] / close.iloc[-5] - 1))
        if len(returns) >= 10:
            indicators.return_10d = float((close.iloc[-1] / close.iloc[-10] - 1))
        if len(returns) >= 20:
            indicators.return_20d = float((close.iloc[-1] / close.iloc[-20] - 1))
        if len(returns) >= 60:
            indicators.return_60d = float((close.iloc[-1] / close.iloc[-60] - 1))
        if len(returns) >= 252:
            indicators.return_252d = float((close.iloc[-1] / close.iloc[-252] - 1))

        # Moving Averages (deviation from current price)
        current_price = close.iloc[-1]

        if len(close) >= 10:
            sma_10 = close.iloc[-10:].mean()
            indicators.sma_10_dev = float((current_price - sma_10) / sma_10)

        if len(close) >= 20:
            sma_20 = close.iloc[-20:].mean()
            indicators.sma_20_dev = float((current_price - sma_20) / sma_20)

        if len(close) >= 50:
            sma_50 = close.iloc[-50:].mean()
            indicators.sma_50_dev = float((current_price - sma_50) / sma_50)

        if len(close) >= 200:
            sma_200 = close.iloc[-200:].mean()
            indicators.sma_200_dev = float((current_price - sma_200) / sma_200)

        # EMAs
        if len(close) >= 12:
            ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
            indicators.ema_12_dev = float((current_price - ema_12) / ema_12)

        if len(close) >= 26:
            ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
            indicators.ema_26_dev = float((current_price - ema_26) / ema_26)

        # Volatility (annualized)
        if len(returns) >= 20:
            indicators.volatility_20d = float(returns.iloc[-20:].std() * np.sqrt(252))
        if len(returns) >= 60:
            indicators.volatility_60d = float(returns.iloc[-60:].std() * np.sqrt(252))
        if len(returns) >= 252:
            indicators.volatility_252d = float(returns.iloc[-252:].std() * np.sqrt(252))

        # ATR (Average True Range)
        if len(close) >= 14:
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)
            atr = tr.iloc[-14:].mean()
            indicators.atr_14 = float(atr / current_price)  # As % of price

        # RSI
        if len(returns) >= 14:
            indicators.rsi_14 = self._compute_rsi(returns, 14)
        if len(returns) >= 7:
            indicators.rsi_7 = self._compute_rsi(returns, 7)

        # MACD
        if len(close) >= 26:
            ema_12_series = close.ewm(span=12, adjust=False).mean()
            ema_26_series = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12_series - ema_26_series
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            indicators.macd = float(macd_line.iloc[-1])
            indicators.macd_signal = float(signal_line.iloc[-1])
            indicators.macd_histogram = float(macd_line.iloc[-1] - signal_line.iloc[-1])

        # Stochastic
        if len(close) >= 14:
            low_14 = low.iloc[-14:].min()
            high_14 = high.iloc[-14:].max()
            k = 100 * (current_price - low_14) / (high_14 - low_14) if high_14 != low_14 else 50
            indicators.stochastic_k = float(k)

        # Volume
        if len(volume) >= 20:
            vol_sma = volume.iloc[-20:].mean()
            if vol_sma > 0:
                indicators.volume_sma_20_ratio = float(volume.iloc[-1] / vol_sma)

        # Bollinger Bands
        if len(close) >= 20:
            sma = close.iloc[-20:].mean()
            std = close.iloc[-20:].std()
            upper = sma + 2 * std
            lower = sma - 2 * std

            if upper != lower:
                indicators.bb_position = float((current_price - lower) / (upper - lower))
            indicators.bb_width = float((upper - lower) / sma)

        # 52-week high/low
        if len(close) >= 252:
            high_52w = close.iloc[-252:].max()
            low_52w = close.iloc[-252:].min()
            indicators.high_52w = float(high_52w)
            indicators.low_52w = float(low_52w)
            indicators.pct_from_52w_high = float((current_price - high_52w) / high_52w)
            indicators.pct_from_52w_low = float((current_price - low_52w) / low_52w)

        # Sharpe (60-day, assuming Rf=0)
        if len(returns) >= 60:
            ret_60 = returns.iloc[-60:]
            mean_ret = ret_60.mean() * 252
            vol = ret_60.std() * np.sqrt(252)
            if vol > 0:
                indicators.sharpe_60d = float(mean_ret / vol)

        # Sortino (60-day)
        if len(returns) >= 60:
            ret_60 = returns.iloc[-60:]
            mean_ret = ret_60.mean() * 252
            downside_returns = ret_60[ret_60 < 0]
            if len(downside_returns) > 0:
                downside_vol = downside_returns.std() * np.sqrt(252)
                if downside_vol > 0:
                    indicators.sortino_60d = float(mean_ret / downside_vol)

        return indicators

    def _compute_rsi(self, returns: pd.Series, period: int) -> float:
        """Compute RSI from returns."""
        gains = returns.clip(lower=0)
        losses = (-returns).clip(lower=0)

        avg_gain = gains.iloc[-period:].mean()
        avg_loss = losses.iloc[-period:].mean()

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _compute_risk_metrics(self, price_data: PriceData) -> RiskMetrics:
        """Compute risk metrics from price data."""
        returns = price_data.returns
        today = date.today()

        metrics = RiskMetrics(
            symbol=price_data.symbol,
            as_of_date=today,
        )

        if len(returns) < 60:
            return metrics

        # Volatility
        metrics.volatility_ann = float(returns.std() * np.sqrt(252))

        # Downside volatility (semi-deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            metrics.downside_volatility = float(downside_returns.std() * np.sqrt(252))

        # VaR (Historical)
        metrics.var_95_1d = float(-np.percentile(returns, 5))
        metrics.var_99_1d = float(-np.percentile(returns, 1))

        # CVaR (Expected Shortfall)
        var_95_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_95_threshold]
        if len(tail_returns) > 0:
            metrics.cvar_95_1d = float(-tail_returns.mean())

        # Higher moments
        metrics.skewness = float(returns.skew())
        metrics.kurtosis = float(returns.kurtosis())

        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = float(drawdown.min())

        # Beta vs SPY (if we can fetch it)
        try:
            spy_data = self._fetch_price_data('SPY')
            if spy_data and len(spy_data.returns) >= 60:
                # Align dates
                aligned = pd.DataFrame({
                    'stock': returns,
                    'market': spy_data.returns,
                }).dropna()

                if len(aligned) >= 60:
                    cov = aligned.cov()
                    market_var = aligned['market'].var()

                    if market_var > 0:
                        metrics.beta = float(cov.loc['stock', 'market'] / market_var)
                        metrics.correlation_spy = float(aligned.corr().loc['stock', 'market'])

                        # R-squared
                        metrics.r_squared = float(metrics.correlation_spy ** 2)

                        # Idiosyncratic volatility
                        if metrics.beta is not None:
                            residual_returns = aligned['stock'] - metrics.beta * aligned['market']
                            metrics.idiosyncratic_vol = float(residual_returns.std() * np.sqrt(252))

        except Exception as e:
            logger.debug(f"Beta calculation failed: {e}")

        return metrics


# Convenience function
def fetch_security_profiles(
    symbols: List[str],
    **kwargs
) -> Dict[str, SecurityProfile]:
    """
    Convenience function to fetch multiple security profiles.

    Args:
        symbols: List of ticker symbols
        **kwargs: Additional arguments for SecurityProfileFetcher

    Returns:
        Dict mapping symbol -> SecurityProfile
    """
    fetcher = SecurityProfileFetcher(**kwargs)
    return fetcher.fetch_profiles(symbols)


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO)

    fetcher = SecurityProfileFetcher(
        enable_alpaca=True,
        enable_fundamentals=True,
        lookback_days=252,
    )

    # Test single profile
    print("Fetching AAPL profile...")
    profile = fetcher.fetch_profile('AAPL')

    print(f"\n=== {profile.symbol} Profile ===")
    print(f"Complete: {profile.is_complete()}")
    print(f"Data sources: {profile.data_sources}")
    print(f"Errors: {profile.errors}")

    if profile.price_data:
        print(f"\nPrice Data:")
        print(f"  Days available: {profile.price_data.days_available}")
        print(f"  Latest price: ${profile.price_data.latest_price:.2f}")

    if profile.fundamentals:
        print(f"\nFundamentals:")
        print(f"  Sector: {profile.fundamentals.sector}")
        print(f"  Market Cap: ${profile.fundamentals.market_cap:,.0f}" if profile.fundamentals.market_cap else "  Market Cap: N/A")
        print(f"  P/E Ratio: {profile.fundamentals.pe_ratio:.2f}" if profile.fundamentals.pe_ratio else "  P/E Ratio: N/A")

    if profile.technicals:
        print(f"\nTechnicals:")
        print(f"  RSI(14): {profile.technicals.rsi_14:.1f}" if profile.technicals.rsi_14 else "  RSI(14): N/A")
        print(f"  20-day return: {profile.technicals.return_20d:.2%}" if profile.technicals.return_20d else "  20-day return: N/A")
        print(f"  Volatility: {profile.technicals.volatility_60d:.2%}" if profile.technicals.volatility_60d else "  Volatility: N/A")

    if profile.risk_metrics:
        print(f"\nRisk Metrics:")
        print(f"  Beta: {profile.risk_metrics.beta:.2f}" if profile.risk_metrics.beta else "  Beta: N/A")
        print(f"  Max Drawdown: {profile.risk_metrics.max_drawdown:.2%}" if profile.risk_metrics.max_drawdown else "  Max Drawdown: N/A")

    if profile.live_quote:
        print(f"\nLive Quote:")
        print(f"  Bid: ${profile.live_quote.bid:.2f}")
        print(f"  Ask: ${profile.live_quote.ask:.2f}")
        print(f"  Spread: {profile.live_quote.spread_pct:.3%}")

    # Test multiple profiles
    print("\n\n=== Fetching Multiple Profiles ===")
    symbols = ['MSFT', 'GOOGL', 'NVDA']
    profiles = fetcher.fetch_profiles(symbols)

    for sym, prof in profiles.items():
        price = prof.get_latest_price()
        print(f"{sym}: ${price:.2f}" if price else f"{sym}: No price")
