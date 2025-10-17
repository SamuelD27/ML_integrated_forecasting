"""
Complete ML-Enhanced Portfolio Optimization Example
====================================================
Demonstrates full integration of all upgraded components:
1. ML Ensemble Forecasting
2. Black-Litterman Optimization
3. Professional Dashboard Integration

Run: python examples/ml_portfolio_integration.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from ml_models.practical_ensemble import StockEnsemble, generate_trading_signal
from portfolio.black_litterman import BlackLittermanOptimizer, integrate_ml_forecasts
from portfolio.hrp_optimizer import HRPOptimizer


def fetch_data(tickers: list, period: str = '3y') -> pd.DataFrame:
    """Fetch historical data for multiple tickers."""
    print(f"📥 Fetching data for {len(tickers)} tickers...")

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                data[ticker] = df['Close']
        except Exception as e:
            print(f"⚠️  Failed to fetch {ticker}: {e}")

    prices_df = pd.DataFrame(data)
    prices_df = prices_df.dropna()

    print(f"✅ Fetched {len(prices_df)} days of data for {len(prices_df.columns)} tickers\n")
    return prices_df


def generate_ml_forecasts(prices_df: pd.DataFrame) -> tuple:
    """Generate ML forecasts for all tickers using ensemble."""
    print("🤖 GENERATING ML FORECASTS")
    print("=" * 60)

    ml_forecasts = {}
    ml_confidence = {}
    signals = {}

    for ticker in prices_df.columns:
        try:
            print(f"\n{ticker}:")

            # Create and train ensemble
            ensemble = StockEnsemble()
            prices = prices_df[ticker]

            # Train
            train_results = ensemble.fit(prices, verbose=False)

            # Predict
            forecast = ensemble.predict(prices)

            # Store results
            ml_forecasts[ticker] = forecast['forecast_return']
            ml_confidence[ticker] = forecast['confidence']
            signals[ticker] = generate_trading_signal(forecast)

            # Display
            print(f"  Forecast Return: {forecast['forecast_return']:+.2%}")
            print(f"  Confidence: {forecast['confidence']:.1%}")
            print(f"  Signal: {signals[ticker]}")
            print(f"  Directional Accuracy: {train_results['lgb_dir_acc']:.1%}")

        except Exception as e:
            print(f"  ⚠️ Forecast failed: {e}")
            # Use neutral forecast on error
            ml_forecasts[ticker] = 0.0
            ml_confidence[ticker] = 0.0
            signals[ticker] = "HOLD (Error)"

    print("\n" + "=" * 60)
    return ml_forecasts, ml_confidence, signals


def optimize_traditional(prices_df: pd.DataFrame):
    """Traditional HRP optimization (no ML)."""
    print("\n📊 TRADITIONAL OPTIMIZATION (HRP)")
    print("=" * 60)

    returns = prices_df.pct_change().dropna()

    hrp = HRPOptimizer()
    weights = hrp.allocate(returns)

    print("\nWeights:")
    for ticker, weight in weights.items():
        print(f"  {ticker}: {weight:.1%}")

    # Calculate expected metrics
    expected_return = (returns.mean() * 252 * weights).sum()
    cov = returns.cov() * 252
    portfolio_var = weights.values @ cov.values @ weights.values
    portfolio_vol = np.sqrt(portfolio_var)
    sharpe = expected_return / portfolio_vol if portfolio_vol > 0 else 0

    print(f"\nExpected Annual Return: {expected_return:.2%}")
    print(f"Expected Volatility: {portfolio_vol:.2%}")
    print(f"Expected Sharpe Ratio: {sharpe:.2f}")

    return weights


def optimize_ml_enhanced(
    prices_df: pd.DataFrame,
    ml_forecasts: dict,
    ml_confidence: dict
):
    """ML-Enhanced optimization using Black-Litterman."""
    print("\n🚀 ML-ENHANCED OPTIMIZATION (Black-Litterman)")
    print("=" * 60)

    returns = prices_df.pct_change().dropna()

    # Use Black-Litterman with ML views
    weights, stats = integrate_ml_forecasts(
        returns,
        ml_forecasts,
        ml_confidence,
        risk_aversion=2.5
    )

    print("\nWeights:")
    for ticker, weight in weights.items():
        print(f"  {ticker}: {weight:.1%}")

    print(f"\nExpected Annual Return: {stats['expected_return']:.2%}")
    print(f"Expected Volatility: {stats['volatility']:.2%}")
    print(f"Expected Sharpe Ratio: {stats['sharpe_ratio']:.2f}")

    return weights, stats


def compare_portfolios(
    traditional_weights: pd.Series,
    ml_weights: pd.Series
):
    """Compare traditional vs ML-enhanced portfolios."""
    print("\n🔍 PORTFOLIO COMPARISON")
    print("=" * 60)

    comparison_df = pd.DataFrame({
        'Traditional (HRP)': traditional_weights,
        'ML-Enhanced (BL)': ml_weights,
        'Difference': ml_weights - traditional_weights
    })

    comparison_df = comparison_df.sort_values('Difference', ascending=False)

    print("\n" + comparison_df.to_string())

    # Highlight biggest changes
    biggest_increase = comparison_df['Difference'].idxmax()
    biggest_decrease = comparison_df['Difference'].idxmin()

    print(f"\n💡 Insights:")
    print(f"  Biggest ML increase: {biggest_increase} "
          f"({comparison_df.loc[biggest_increase, 'Difference']:+.1%})")
    print(f"  Biggest ML decrease: {biggest_decrease} "
          f"({comparison_df.loc[biggest_decrease, 'Difference']:+.1%})")


def main():
    """Run complete ML portfolio integration example."""
    print("\n" + "=" * 60)
    print(" ML-ENHANCED PORTFOLIO OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM']
    PERIOD = '3y'

    print(f"\n📋 Configuration:")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Historical Period: {PERIOD}")
    print(f"  ML Forecast Horizon: 20 days")

    # Step 1: Fetch Data
    prices_df = fetch_data(TICKERS, PERIOD)

    if prices_df.empty:
        print("❌ No data fetched. Exiting.")
        return

    # Step 2: Generate ML Forecasts
    ml_forecasts, ml_confidence, signals = generate_ml_forecasts(prices_df)

    # Step 3: Traditional Optimization
    traditional_weights = optimize_traditional(prices_df)

    # Step 4: ML-Enhanced Optimization
    ml_weights, ml_stats = optimize_ml_enhanced(
        prices_df,
        ml_forecasts,
        ml_confidence
    )

    # Step 5: Compare
    compare_portfolios(traditional_weights, ml_weights)

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    print("\n✅ Completed full ML integration workflow:")
    print("  1. ✓ Fetched historical data")
    print("  2. ✓ Trained ML ensemble models (LightGBM + Ridge + Momentum)")
    print("  3. ✓ Generated 20-day forecasts with confidence")
    print("  4. ✓ Optimized traditional portfolio (HRP)")
    print("  5. ✓ Optimized ML-enhanced portfolio (Black-Litterman)")
    print("  6. ✓ Compared results")

    print(f"\n📊 Trading Signals:")
    for ticker, signal in signals.items():
        if "BUY" in signal:
            emoji = "🟢"
        elif "SELL" in signal:
            emoji = "🔴"
        else:
            emoji = "⚪"
        print(f"  {emoji} {ticker}: {signal}")

    print("\n🎯 Next Steps:")
    print("  1. Review the ML-enhanced weights vs traditional")
    print("  2. Check forecast confidence levels")
    print("  3. Use dashboard for detailed analysis:")
    print("     streamlit run dashboard/app.py")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
