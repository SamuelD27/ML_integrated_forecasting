"""
Single Stock Analysis Page - FIXED VERSION
==========================================
Complete quantitative analysis with multi-signal long/short recommendation.

Improvements:
- Robust Fama-French factor analysis with error handling
- Multi-signal trading decision (valuation + momentum + mean reversion)
- No neutral positions - always Long or Short
- Better data alignment for factor regression
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
import io

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from portfolio.security_valuation import DCFInputs, DCFValuation
from portfolio.factor_models import FamaFrenchFactorModel
from ml_models.practical_ensemble import StockEnsemble, generate_trading_signal
from dashboard.utils.stock_search import compact_stock_search
from dashboard.utils.theme_terminal import apply_terminal_theme, COLORS, FONTS, FONT_SIZES, SPACING, RADIUS


@st.cache_data(ttl=3600)
def get_ml_forecast(ticker: str, hist: pd.DataFrame) -> dict:
    """
    Generate ML forecast for stock using ensemble.

    Args:
        ticker: Stock ticker
        hist: Historical price data

    Returns:
        Dictionary with forecast results or error
    """
    try:
        ensemble = StockEnsemble()

        # Train on historical data
        prices = hist['Close']
        if len(prices) < 100:
            return {
                'success': False,
                'error': 'Insufficient data (need at least 100 days)'
            }

        train_results = ensemble.fit(prices, verbose=False)

        # Generate forecast
        forecast = ensemble.predict(prices)

        return {
            'success': True,
            'forecast': forecast,
            'train_results': train_results
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def calculate_risk_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Calculate comprehensive risk metrics."""
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

    # Sortino
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # VaR and CVaR (95%)
    var_95 = -np.percentile(returns, 5)
    cvar_95 = -returns[returns <= -var_95].mean() if len(returns[returns <= -var_95]) > 0 else var_95

    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95
    }


def prepare_export_data(ticker: str, hist: pd.DataFrame, analysis_results: dict) -> tuple:
    """
    Prepare comprehensive data for CSV export.

    Returns:
        Tuple of (price_data_df, indicators_df, summary_df, signals_df)
    """
    # 1. Price data with all technical indicators
    price_df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Add returns
    price_df['Daily_Return'] = hist['Close'].pct_change()
    price_df['Cumulative_Return'] = (1 + price_df['Daily_Return']).cumprod() - 1

    # Add moving averages
    price_df['SMA_20'] = hist['Close'].rolling(20).mean()
    price_df['SMA_50'] = hist['Close'].rolling(50).mean()
    price_df['SMA_200'] = hist['Close'].rolling(200).mean()

    # Add volatility
    price_df['Volatility_20'] = price_df['Daily_Return'].rolling(20).std() * np.sqrt(252)

    # Add RSI
    delta = price_df['Daily_Return']
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    price_df['RSI_14'] = 100 - (100 / (1 + rs))

    # Add Bollinger Bands
    bb_ma = hist['Close'].rolling(20).mean()
    bb_std = hist['Close'].rolling(20).std()
    price_df['BB_Upper'] = bb_ma + (2 * bb_std)
    price_df['BB_Lower'] = bb_ma - (2 * bb_std)
    price_df['BB_Width'] = (price_df['BB_Upper'] - price_df['BB_Lower']) / bb_ma

    # Add drawdown
    cumulative = (1 + price_df['Daily_Return']).cumprod()
    running_max = cumulative.expanding().max()
    price_df['Drawdown'] = (cumulative - running_max) / running_max

    # 2. Summary metrics
    summary_data = {
        'Ticker': ticker,
        'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Period_Start': hist.index[0].strftime('%Y-%m-%d'),
        'Period_End': hist.index[-1].strftime('%Y-%m-%d'),
        'Days_Analyzed': len(hist),
        'Current_Price': analysis_results.get('current_price', np.nan),
        'Period_Return': analysis_results.get('period_return', np.nan),
        'Annual_Return': analysis_results.get('annual_return', np.nan),
        'Annual_Volatility': analysis_results.get('annual_volatility', np.nan),
        'Sharpe_Ratio': analysis_results.get('sharpe_ratio', np.nan),
        'Sortino_Ratio': analysis_results.get('sortino_ratio', np.nan),
        'Max_Drawdown': analysis_results.get('max_drawdown', np.nan),
        'VaR_95': analysis_results.get('var_95', np.nan),
        'CVaR_95': analysis_results.get('cvar_95', np.nan),
        'DCF_Intrinsic_Value': analysis_results.get('intrinsic_value', np.nan),
        'DCF_Upside_Pct': analysis_results.get('dcf_upside', np.nan),
        'Current_RSI': analysis_results.get('current_rsi', np.nan),
        'SMA20_Distance_Pct': analysis_results.get('sma20_distance', np.nan),
        'SMA50_Distance_Pct': analysis_results.get('sma50_distance', np.nan),
    }

    # Add Fama-French results if available
    if analysis_results.get('ff_success', False):
        ff = analysis_results.get('ff_results', {})
        summary_data.update({
            'FF_Alpha_Annual': ff.get('alpha', 0) * 252,
            'FF_Alpha_PValue': ff.get('alpha_pvalue', np.nan),
            'FF_Beta_Market': ff.get('beta_MKT', np.nan),
            'FF_Beta_SMB': ff.get('beta_SMB', np.nan),
            'FF_Beta_HML': ff.get('beta_HML', np.nan),
            'FF_Beta_RMW': ff.get('beta_RMW', np.nan),
            'FF_Beta_CMA': ff.get('beta_CMA', np.nan),
            'FF_R_Squared': ff.get('r_squared', np.nan),
        })

    # Add ML forecast if available
    if analysis_results.get('ml_success', False):
        ml = analysis_results.get('ml_forecast', {})
        summary_data.update({
            'ML_Forecast_Price': ml.get('forecast_price', np.nan),
            'ML_Forecast_Return_Pct': ml.get('forecast_return', 0) * 100,
            'ML_Confidence_Pct': ml.get('confidence', 0) * 100,
            'ML_Lower_Bound': ml.get('lower_bound', np.nan),
            'ML_Upper_Bound': ml.get('upper_bound', np.nan),
        })

    # Add trading signals
    summary_data.update({
        'Signal_Valuation': analysis_results.get('valuation_signal', np.nan),
        'Signal_Momentum': analysis_results.get('momentum_signal', np.nan),
        'Signal_Mean_Reversion': analysis_results.get('mean_reversion_signal', np.nan),
        'Signal_RSI': analysis_results.get('rsi_signal', np.nan),
        'Signal_Alpha': analysis_results.get('alpha_signal', np.nan),
        'Signal_Combined': analysis_results.get('combined_signal', np.nan),
        'Recommendation': analysis_results.get('decision', 'N/A'),
        'Confidence': analysis_results.get('confidence', 'N/A'),
    })

    summary_df = pd.DataFrame([summary_data]).T
    summary_df.columns = ['Value']

    # 3. Signals breakdown
    signals_list = []
    for name, (sig, weight) in analysis_results.get('signals', {}).items():
        if weight > 0:
            signals_list.append({
                'Signal_Name': name,
                'Signal_Value': sig,
                'Weight_Pct': weight * 100,
                'Weighted_Contribution': sig * weight
            })
    signals_df = pd.DataFrame(signals_list) if signals_list else pd.DataFrame()

    # 4. Technical indicators summary (recent values)
    recent_indicators = price_df.tail(1).T
    recent_indicators.columns = ['Current_Value']

    return price_df, recent_indicators, summary_df, signals_df


def show():
    """Main function for single stock analysis page."""
    apply_terminal_theme()

    st.markdown(f"""
    <div style="font-size: {FONT_SIZES['lg']}; font-weight: 600; color: {COLORS['primary']}; margin-bottom: {SPACING['sm']};">
        Single Stock Analysis
    </div>
    <div style="font-size: {FONT_SIZES['sm']}; color: {COLORS['text_muted']}; margin-bottom: {SPACING['md']};">
        Complete quantitative analysis with institutional-grade metrics
    </div>
    """, unsafe_allow_html=True)

    # Input section
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = compact_stock_search(key="single_stock_search", default="AAPL")

        with col2:
            period = st.selectbox(
                "Analysis Period",
                ["1y", "2y", "3y", "5y"],
                index=2,  # Default to 3y for better factor regression
                help="Historical period for analysis"
            )

        with col3:
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                value=5.0,
                min_value=0.0,
                max_value=20.0,
                help="Used for Sharpe ratio and CAPM"
            ) / 100

    # Run analysis button
    if st.button("Run Complete Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Initialize results dictionary for export
                analysis_results = {}

                # Fetch data
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                info = stock.info

                if hist.empty:
                    st.error(f"No data found for ticker {ticker}. Please check the symbol.")
                    return

                current_price = hist['Close'].iloc[-1]
                returns = hist['Close'].pct_change().dropna()

                # Store basic info
                analysis_results['current_price'] = current_price
                analysis_results['period_return'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)

                # === 1. VALUATION ===
                st.header("1. Valuation Analysis")

                fcf = info.get('freeCashflow', 0)
                valuation_signal = 0  # -1 to 1 scale

                if fcf > 0:
                    # DCF Valuation
                    try:
                        dcf_inputs = DCFInputs(
                            fcf_current=fcf,
                            growth_rate_stage1=0.15,  # 15% high growth
                            growth_rate_stage2=0.03,  # 3% stable growth
                            wacc=0.10,  # 10% WACC
                            shares_outstanding=info.get('sharesOutstanding', 1)
                        )
                        dcf = DCFValuation(dcf_inputs)
                        valuation = dcf.calculate_intrinsic_value()

                        intrinsic = valuation['intrinsic_value']
                        upside = (intrinsic - current_price) / current_price

                        # Valuation signal: -1 (overvalued) to +1 (undervalued)
                        valuation_signal = np.clip(upside / 0.3, -1, 1)  # ±30% = max signal

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Current Price",
                            f"${current_price:.2f}"
                        )

                        col2.metric(
                            "Intrinsic Value (DCF)",
                            f"${intrinsic:.2f}",
                            delta=f"{upside:+.1%}"
                        )

                        col3.metric(
                            "Valuation Signal",
                            f"{valuation_signal:+.2f}",
                            help="±1.0 scale: +1 = very undervalued, -1 = very overvalued"
                        )

                        # Store for export
                        analysis_results['intrinsic_value'] = intrinsic
                        analysis_results['dcf_upside'] = upside

                    except Exception as e:
                        st.warning(f"DCF valuation unavailable: {e}")
                        intrinsic = current_price
                        upside = 0
                else:
                    st.info("DCF valuation not available (negative or missing free cash flow)")
                    intrinsic = current_price
                    upside = 0

                # === 1.5 ML FORECAST ===
                st.header("2. ML Ensemble Forecast (20-Day Ahead)")

                with st.spinner("Training ML models..."):
                    ml_result = get_ml_forecast(ticker, hist)

                ml_signal = 0  # Default neutral

                if ml_result['success']:
                    forecast = ml_result['forecast']
                    train_results = ml_result['train_results']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"${forecast['current_price']:.2f}",
                            help="Latest closing price"
                        )

                    with col2:
                        ret_pct = forecast['forecast_return'] * 100
                        st.metric(
                            "Forecast Price",
                            f"${forecast['forecast_price']:.2f}",
                            f"{ret_pct:+.1f}%",
                            help="ML ensemble prediction for 20 days ahead"
                        )

                    with col3:
                        conf_pct = forecast['confidence'] * 100
                        st.metric(
                            "Model Confidence",
                            f"{conf_pct:.0f}%",
                            help="Higher = models agree, lower = models disagree"
                        )

                    with col4:
                        range_str = f"${forecast['lower_bound']:.0f}-${forecast['upper_bound']:.0f}"
                        st.metric(
                            "95% CI Range",
                            range_str,
                            help="95% probability price falls in this range"
                        )

                    # Generate ML trading signal
                    signal_text = generate_trading_signal(forecast)

                    # ML signal for final recommendation
                    ml_signal = np.clip(forecast['forecast_return'] / 0.1, -1, 1)  # ±10% = max signal

                    # Display signal with color
                    if "STRONG BUY" in signal_text:
                        st.success(f"### ML Signal: {signal_text}")
                    elif "BUY" in signal_text:
                        st.success(f"### ML Signal: {signal_text}")
                    elif "STRONG SELL" in signal_text:
                        st.error(f"### ML Signal: {signal_text}")
                    elif "SELL" in signal_text:
                        st.error(f"### ML Signal: {signal_text}")
                    else:
                        st.info(f"### ML Signal: {signal_text}")

                    # Model breakdown
                    with st.expander("Model Breakdown & Performance"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Individual Model Predictions:**")
                            model_df = pd.DataFrame({
                                'Model': list(forecast['model_predictions'].keys()),
                                'Predicted Return': [f"{v*100:+.2f}%" for v in forecast['model_predictions'].values()],
                                'Weight': [f"{train_results['weights'][k]:.1%}" for k in forecast['model_predictions'].keys()]
                            })
                            st.dataframe(model_df, use_container_width=True, hide_index=True)

                        with col2:
                            st.markdown("**Validation Performance:**")
                            perf_df = pd.DataFrame({
                                'Model': ['LightGBM', 'Ridge', 'Momentum'],
                                'RMSE': [
                                    f"{train_results['lgb_rmse']:.4f}",
                                    f"{train_results['ridge_rmse']:.4f}",
                                    f"{train_results['momentum_rmse']:.4f}"
                                ],
                                'Dir Accuracy': [
                                    f"{train_results['lgb_dir_acc']:.1%}",
                                    f"{train_results['ridge_dir_acc']:.1%}",
                                    f"{train_results['momentum_dir_acc']:.1%}"
                                ]
                            })
                            st.dataframe(perf_df, use_container_width=True, hide_index=True)

                        st.caption(f"Trained on {train_results['n_train']} samples, validated on {train_results['n_val']} samples")
                        st.caption("Directional Accuracy: % of times model predicted correct direction (50% = random, >55% = good)")

                    # Store ML results
                    analysis_results['ml_success'] = True
                    analysis_results['ml_forecast'] = forecast

                else:
                    st.warning(f"Warning: ML forecast unavailable: {ml_result.get('error', 'Unknown error')}")
                    st.info("Using traditional valuation and technical analysis only")
                    analysis_results['ml_success'] = False

                # === 2. TECHNICAL SIGNALS ===
                st.header("3. Technical Analysis")

                # Calculate technical indicators
                sma_20 = hist['Close'].rolling(20).mean()
                sma_50 = hist['Close'].rolling(50).mean()

                # Momentum signal
                momentum_20 = (current_price / sma_20.iloc[-1] - 1) if not sma_20.empty else 0
                momentum_50 = (current_price / sma_50.iloc[-1] - 1) if not sma_50.empty else 0
                momentum_signal = np.clip((momentum_20 + momentum_50) / 0.2, -1, 1)  # ±10% avg = max

                # Mean reversion signal (inverse)
                vol_20 = returns.tail(20).std()
                z_score = (current_price - sma_20.iloc[-1]) / (vol_20 * sma_20.iloc[-1]) if vol_20 > 0 and not sma_20.empty else 0
                mean_reversion_signal = -np.clip(z_score / 2, -1, 1)  # Inverse: oversold = buy

                # RSI
                delta = returns
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50

                # RSI signal: 30 = +1 (oversold/buy), 70 = -1 (overbought/sell)
                rsi_signal = np.clip((50 - current_rsi) / 20, -1, 1)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "20-Day Momentum",
                    f"{momentum_20:+.2%}",
                    help="Price vs 20-day SMA"
                )

                col2.metric(
                    "50-Day Momentum",
                    f"{momentum_50:+.2%}",
                    help="Price vs 50-day SMA"
                )

                col3.metric(
                    "RSI (14)",
                    f"{current_rsi:.1f}",
                    help="<30 oversold, >70 overbought"
                )

                col4.metric(
                    "Technical Signal",
                    f"{momentum_signal:+.2f}",
                    help="Combined momentum + RSI signal"
                )

                # Store technical indicators
                analysis_results['current_rsi'] = current_rsi
                analysis_results['sma20_distance'] = momentum_20
                analysis_results['sma50_distance'] = momentum_50
                analysis_results['momentum_signal'] = momentum_signal
                analysis_results['mean_reversion_signal'] = mean_reversion_signal
                analysis_results['rsi_signal'] = rsi_signal
                analysis_results['valuation_signal'] = valuation_signal

                # === 3. RISK METRICS ===
                st.header("4. Risk Analysis")

                risk = calculate_risk_metrics(returns, risk_free_rate)

                # Store risk metrics
                analysis_results.update(risk)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Sharpe Ratio",
                    f"{risk['sharpe_ratio']:.2f}",
                    help="Risk-adjusted return (>1 good, >2 excellent)"
                )

                col2.metric(
                    "Sortino Ratio",
                    f"{risk['sortino_ratio']:.2f}",
                    help="Downside risk-adjusted return"
                )

                col3.metric(
                    "Max Drawdown",
                    f"{risk['max_drawdown']:.2%}",
                    help="Largest peak-to-trough decline"
                )

                col4.metric(
                    "95% VaR",
                    f"{risk['var_95']:.2%}",
                    help="Maximum 1-day loss (95% confidence)"
                )

                # === 4. FACTOR ANALYSIS (IMPROVED) ===
                st.header("5. Fama-French Factor Exposure")

                alpha_signal = 0
                ff_success = False

                try:
                    # Ensure we have enough data
                    if len(returns) < 252:
                        st.warning(f"Warning: Only {len(returns)} days of data. Factor analysis works best with 1+ years.")

                    ff = FamaFrenchFactorModel(model='5-factor')
                    ff_results = ff.regress_returns(ticker, returns, frequency='daily')

                    # Check if we got valid results
                    if ff_results and not np.isnan(ff_results.get('alpha', np.nan)):
                        ff_success = True

                        # Alpha signal: positive alpha = buy signal
                        alpha_annual = ff_results['alpha'] * 252
                        alpha_signal = np.clip(alpha_annual / 0.1, -1, 1)  # ±10% annual = max

                        # Display alpha and model fit
                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Alpha (annualized)",
                            f"{alpha_annual:.2%}",
                            delta="Significant ✓" if ff_results.get('alpha_pvalue', 1) < 0.05 else "Not sig.",
                            help=f"Skill-based return (p-value: {ff_results.get('alpha_pvalue', 1):.4f})"
                        )

                        col2.metric(
                            "Market Beta",
                            f"{ff_results.get('beta_MKT', np.nan):.2f}",
                            help="Sensitivity to market factor"
                        )

                        col3.metric(
                            "R-squared",
                            f"{ff_results.get('r_squared', 0):.1%}",
                            help="Variance explained by factors"
                        )

                        # Factor betas chart
                        betas_df = pd.DataFrame({
                            'Factor': ['Market\n(Mkt-RF)', 'Size\n(SMB)', 'Value\n(HML)', 'Profitability\n(RMW)', 'Investment\n(CMA)'],
                            'Beta': [
                                ff_results.get('beta_MKT', 0),
                                ff_results.get('beta_SMB', 0),
                                ff_results.get('beta_HML', 0),
                                ff_results.get('beta_RMW', 0),
                                ff_results.get('beta_CMA', 0)
                            ]
                        })

                        # Color bars based on magnitude
                        colors = ['green' if b > 0 else 'red' for b in betas_df['Beta']]

                        fig = go.Figure(data=go.Bar(
                            x=betas_df['Factor'],
                            y=betas_df['Beta'],
                            marker_color=colors,
                            text=betas_df['Beta'].round(2),
                            textposition='outside'
                        ))

                        fig.update_layout(
                            title="Factor Exposures (5-Factor Model)",
                            xaxis_title="Factor",
                            yaxis_title="Beta",
                            height=400,
                            showlegend=False
                        )

                        fig.add_hline(y=0, line_dash="dash", line_color="gray")

                        st.plotly_chart(fig, use_container_width=True)

                        # Store FF results
                        analysis_results['ff_success'] = True
                        analysis_results['ff_results'] = ff_results

                except Exception as e:
                    st.warning(f"Warning: Factor analysis unavailable: {str(e)[:200]}")
                    st.info("Factor data requires pandas-datareader and may fail for recent IPOs or non-US stocks.")
                    analysis_results['ff_success'] = False

                # === 5. TRADING DECISION (MULTI-SIGNAL) ===
                st.header("6. Trading Recommendation")

                # Combine signals with weights
                signals = {
                    'Valuation': (valuation_signal, 0.30),
                    'Momentum': (momentum_signal, 0.25),
                    'Mean Reversion': (mean_reversion_signal, 0.15),
                    'RSI': (rsi_signal, 0.15),
                    'Alpha': (alpha_signal, 0.15) if ff_success else (0, 0)
                }

                # Calculate weighted average
                total_weight = sum(w for _, w in signals.values() if w > 0)
                if total_weight > 0:
                    combined_signal = sum(s * w for s, w in signals.values()) / total_weight
                else:
                    combined_signal = 0

                # Decision logic: NO NEUTRAL - always Long or Short
                if combined_signal > 0.4:
                    decision = "STRONG LONG "
                    decision_color = "green"
                    reasoning = "Multiple strong bullish signals detected"
                    confidence = "High"
                elif combined_signal > 0:
                    decision = "LONG "
                    decision_color = "green"
                    reasoning = "Net bullish signals, moderate conviction"
                    confidence = "Medium"
                elif combined_signal < -0.4:
                    decision = "STRONG SHORT 📉"
                    decision_color = "red"
                    reasoning = "Multiple strong bearish signals detected"
                    confidence = "High"
                else:  # combined_signal <= 0
                    decision = "SHORT 🔻"
                    decision_color = "red"
                    reasoning = "Net bearish signals, moderate conviction"
                    confidence = "Medium"

                # Display decision prominently
                if decision_color == "green":
                    st.success(f"**{decision}**")
                elif decision_color == "red":
                    st.error(f"**{decision}**")

                st.info(f"**Reasoning**: {reasoning}")
                st.caption(f"**Combined Signal**: {combined_signal:+.2f} (±1.0 scale)")

                # Store final signals and recommendation
                analysis_results['signals'] = signals
                analysis_results['combined_signal'] = combined_signal
                analysis_results['decision'] = decision
                analysis_results['confidence'] = confidence
                analysis_results['alpha_signal'] = alpha_signal

                # Signal breakdown
                with st.expander("Signal Breakdown"):
                    signal_df = pd.DataFrame([
                        {'Signal': name, 'Value': f"{sig:+.2f}", 'Weight': f"{weight:.0%}"}
                        for name, (sig, weight) in signals.items()
                        if weight > 0
                    ])
                    st.dataframe(signal_df, use_container_width=True, hide_index=True)

                # === 6. PRICE CHART ===
                st.header("7. Price History & Technical Indicators")

                # Create subplot figure
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price & Moving Averages', 'RSI'),
                    row_heights=[0.7, 0.3]
                )

                # Price and SMAs
                fig.add_trace(
                    go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=hist.index, y=sma_20, name='SMA 20', line=dict(color='orange', dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=hist.index, y=sma_50, name='SMA 50', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )

                # RSI
                fig.add_trace(
                    go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)

                fig.update_layout(height=700, showlegend=True)

                st.plotly_chart(fig, use_container_width=True)

                st.success("Analysis complete! Review the recommendation and metrics above before making investment decisions.")
                st.caption("Warning: This is for educational purposes only. Not financial advice.")

                # === EXPORT SECTION ===
                st.header("8. Export Analysis Data")

                st.markdown("""
                Export all calculated indicators, metrics, and signals to CSV for further analysis in Excel, Python, or other tools.
                The export includes:
                - **Price Data**: Daily OHLCV with all technical indicators (SMA, RSI, Bollinger Bands, etc.)
                - **Summary Metrics**: Risk metrics, valuation, factor exposures, ML forecasts
                - **Trading Signals**: Individual and combined signals with weights
                - **Current Indicators**: Most recent values for all technical indicators
                """)

                # Prepare export data
                try:
                    price_data, indicators, summary, signals_export = prepare_export_data(
                        ticker, hist, analysis_results
                    )

                    # Create multi-sheet CSV (using separators)
                    output = io.StringIO()

                    # Write summary first
                    output.write("=" * 80 + "\n")
                    output.write(f"SINGLE STOCK ANALYSIS - {ticker}\n")
                    output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    output.write("=" * 80 + "\n\n")

                    output.write("SUMMARY METRICS\n")
                    output.write("-" * 80 + "\n")
                    summary.to_csv(output, header=True)
                    output.write("\n\n")

                    # Write signals
                    if not signals_export.empty:
                        output.write("TRADING SIGNALS BREAKDOWN\n")
                        output.write("-" * 80 + "\n")
                        signals_export.to_csv(output, index=False)
                        output.write("\n\n")

                    # Write current indicators
                    output.write("CURRENT TECHNICAL INDICATORS\n")
                    output.write("-" * 80 + "\n")
                    indicators.to_csv(output, header=True)
                    output.write("\n\n")

                    # Write full price data
                    output.write("HISTORICAL PRICE DATA WITH INDICATORS\n")
                    output.write("-" * 80 + "\n")
                    price_data.to_csv(output)

                    csv_data = output.getvalue()

                    # Download button
                    filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                    st.download_button(
                        label="Download Complete Analysis (CSV)",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                        help=f"Download all analysis data for {ticker} including price history, indicators, and signals"
                    )

                    # Show preview
                    with st.expander("Preview Export Data"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Summary Metrics")
                            st.dataframe(summary, use_container_width=True)

                            if not signals_export.empty:
                                st.subheader("Trading Signals")
                                st.dataframe(signals_export, use_container_width=True, hide_index=True)

                        with col2:
                            st.subheader("Recent Price Data (Last 10 Days)")
                            preview_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI_14', 'Volatility_20']
                            available_cols = [col for col in preview_cols if col in price_data.columns]
                            st.dataframe(price_data[available_cols].tail(10), use_container_width=True)

                        st.info(f"Total rows in price data: {len(price_data)} | Total columns: {len(price_data.columns)}")

                except Exception as e:
                    st.error(f"Error preparing export: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"Error: Error analyzing {ticker}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    show()
