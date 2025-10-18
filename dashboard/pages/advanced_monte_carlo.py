"""
Advanced Monte Carlo Simulation
================================
Advanced price simulation with multiple models.

Features:
- Jump diffusion (Merton model)
- Fat tail distributions
- Regime switching
- Variance reduction techniques
- Risk metrics from simulations
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict
import logging

# Import backend modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from portfolio.advanced_monte_carlo import (
    JumpDiffusionSimulator,
    JumpDiffusionParams,
    FatTailSimulator,
    RegimeSwitchingMC,
    VarianceReductionMC,
    calculate_path_statistics,
    calculate_risk_metrics
)
from dashboard.utils.stock_search import compact_stock_search
from dashboard.utils.theme import apply_vscode_theme
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


def show():
    """Main function for advanced Monte Carlo simulation page."""
    apply_vscode_theme()

    st.title("Advanced Monte Carlo Simulation")
    st.markdown("""
    Simulate future price paths using advanced models.
    Visualize potential outcomes, risk metrics, and probability distributions.
    """)

    # Sidebar inputs
    st.sidebar.header("Configuration")

    ticker = compact_stock_search(key="monte_carlo_search", default="AAPL")

    simulation_model = st.sidebar.selectbox(
        "Simulation Model",
        [
            "Jump Diffusion (Merton)",
            "Fat Tails (Student-t)",
            "Regime Switching (Bull/Bear)",
            "Standard GBM (Variance Reduction)"
        ]
    )

    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")

    time_horizon = st.sidebar.slider("Time Horizon (years)", 0.25, 5.0, 1.0, 0.25)
    n_paths = st.sidebar.selectbox("Number of Paths", [1000, 5000, 10000, 50000], index=1)
    steps_per_year = st.sidebar.selectbox("Steps per Year", [52, 252], index=1)

    steps = int(time_horizon * steps_per_year)

    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                # 1. Fetch historical data
                st.subheader("1. Historical Data Analysis")

                stock = yf.Ticker(ticker)
                hist_data = stock.history(period="2y")

                if hist_data.empty:
                    st.error(f"Error: No data found for {ticker}")
                    return

                current_price = hist_data['Close'].iloc[-1]
                returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1)).dropna()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    annual_return = returns.mean() * 252
                    st.metric("Historical Return", f"{annual_return:.2%}")
                with col3:
                    annual_vol = returns.std() * np.sqrt(252)
                    st.metric("Historical Volatility", f"{annual_vol:.2%}")
                with col4:
                    st.metric("Data Points", len(returns))

                # 2. Run simulation based on selected model
                st.subheader(f"2. Running {simulation_model}")

                mu = annual_return
                sigma = annual_vol
                S0 = current_price

                if simulation_model == "Jump Diffusion (Merton)":
                    # Estimate jump parameters from data
                    simulator = JumpDiffusionSimulator(
                        JumpDiffusionParams(
                            mu=mu,
                            sigma=sigma,
                            jump_lambda=2.0,  # ~2 jumps per year
                            jump_mean=-0.05,  # Negative jumps (crashes)
                            jump_std=0.10
                        )
                    )

                    # User can adjust jump parameters
                    with st.expander("🔧 Adjust Jump Parameters"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            jump_lambda = st.slider("Jump Intensity (per year)", 0.5, 10.0, 2.0)
                        with col2:
                            jump_mean = st.slider("Mean Jump Size", -0.20, 0.20, -0.05)
                        with col3:
                            jump_std = st.slider("Jump Volatility", 0.01, 0.30, 0.10)

                        simulator.params.jump_lambda = jump_lambda
                        simulator.params.jump_mean = jump_mean
                        simulator.params.jump_std = jump_std

                    paths = simulator.simulate(S0, time_horizon, steps, n_paths)

                elif simulation_model == "Fat Tails (Student-t)":
                    # Fat tail simulation
                    df = st.sidebar.slider("Degrees of Freedom", 3, 10, 5)

                    simulator = FatTailSimulator(mu, sigma, df)
                    paths = simulator.simulate(S0, time_horizon, steps, n_paths)

                    st.info(f"Using Student-t distribution with {df} degrees of freedom (lower = fatter tails)")

                elif simulation_model == "Regime Switching (Bull/Bear)":
                    # Regime switching
                    regime_params = [
                        (0.15, 0.15),  # Bull: high return, low vol
                        (-0.10, 0.30)  # Bear: negative return, high vol
                    ]

                    # Transition matrix: rows=from, cols=to
                    transition_probs = np.array([
                        [0.95, 0.05],  # Bull stays bull 95%, switches to bear 5%
                        [0.10, 0.90]   # Bear stays bear 90%, switches to bull 10%
                    ])

                    simulator = RegimeSwitchingMC(regime_params, transition_probs)
                    paths, regimes = simulator.simulate(S0, time_horizon, steps, initial_regime=0, n_paths=n_paths)

                    st.info("Simulating with 2 regimes: Bull (high return/low vol) and Bear (low return/high vol)")

                else:  # Standard GBM with variance reduction
                    paths = VarianceReductionMC.antithetic_variates(
                        S0, mu, sigma, time_horizon, steps, n_paths
                    )

                    st.info("Using antithetic variates for variance reduction (2x efficiency)")

                st.success(f"Generated {n_paths} price paths over {time_horizon} years")

                # 3. Visualize paths
                st.subheader("3. Simulated Price Paths")

                # Plot subset of paths
                n_display = min(100, n_paths)
                time_index = np.linspace(0, time_horizon, steps + 1)

                fig_paths = go.Figure()

                # Individual paths (sample)
                sample_indices = np.random.choice(n_paths, n_display, replace=False)
                for idx in sample_indices:
                    fig_paths.add_trace(go.Scatter(
                        x=time_index,
                        y=paths[idx],
                        mode='lines',
                        line=dict(width=0.5, color='lightblue'),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Mean path
                mean_path = paths.mean(axis=0)
                fig_paths.add_trace(go.Scatter(
                    x=time_index,
                    y=mean_path,
                    mode='lines',
                    name='Mean Path',
                    line=dict(color='blue', width=3)
                ))

                # Current price line
                fig_paths.add_hline(
                    y=S0,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="Current Price"
                )

                fig_paths.update_layout(
                    title=f'{ticker} Simulated Price Paths ({n_display} of {n_paths} shown)',
                    xaxis_title='Time (years)',
                    yaxis_title='Price ($)',
                    height=500,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_paths, use_container_width=True)

                # 4. Statistics
                st.subheader("4. Simulation Statistics")

                stats = calculate_path_statistics(paths)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Mean Final Price", f"${stats['mean']:.2f}")
                with col2:
                    st.metric("Median Final Price", f"${stats['median']:.2f}")
                with col3:
                    st.metric("5th Percentile", f"${stats['percentile_5']:.2f}")
                with col4:
                    st.metric("95th Percentile", f"${stats['percentile_95']:.2f}")

                # 5. Risk metrics
                st.subheader("5. Risk Metrics")

                risk_metrics = calculate_risk_metrics(paths, S0, confidence_level=0.95)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Expected Return", f"{risk_metrics['expected_return']:.2%}")
                with col2:
                    st.metric("95% VaR", f"{risk_metrics['var']:.2%}")
                with col3:
                    st.metric("95% CVaR", f"{risk_metrics['cvar']:.2%}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Probability of Loss", f"{risk_metrics['prob_loss']:.1%}")
                with col2:
                    st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
                with col3:
                    st.metric("Volatility", f"{risk_metrics['volatility']:.2%}")

                st.markdown(f"""
                **Interpretation**:
                - With 95% confidence, losses will not exceed **{risk_metrics['var']:.2%}**
                - If losses exceed VaR, expected loss is **{risk_metrics['cvar']:.2%}** (CVaR)
                - Probability of losing money: **{risk_metrics['prob_loss']:.1%}**
                """)

                # 6. Distribution of final prices
                st.subheader("6. Final Price Distribution")

                final_prices = paths[:, -1]

                fig_dist = go.Figure()

                # Histogram
                fig_dist.add_trace(go.Histogram(
                    x=final_prices,
                    nbinsx=50,
                    name='Final Prices',
                    opacity=0.7
                ))

                # Add percentile lines
                fig_dist.add_vline(x=stats['percentile_5'], line_dash="dash", line_color="red",
                                  annotation_text="5th %ile")
                fig_dist.add_vline(x=stats['median'], line_dash="dash", line_color="green",
                                  annotation_text="Median")
                fig_dist.add_vline(x=stats['percentile_95'], line_dash="dash", line_color="red",
                                  annotation_text="95th %ile")

                fig_dist.update_layout(
                    title=f'Distribution of {ticker} Price in {time_horizon} Year(s)',
                    xaxis_title='Price ($)',
                    yaxis_title='Frequency',
                    height=400
                )

                st.plotly_chart(fig_dist, use_container_width=True)

                # 7. Confidence intervals over time
                st.subheader("7. Confidence Intervals")

                # Calculate percentiles at each time step
                percentiles = {
                    5: np.percentile(paths, 5, axis=0),
                    25: np.percentile(paths, 25, axis=0),
                    50: np.percentile(paths, 50, axis=0),
                    75: np.percentile(paths, 75, axis=0),
                    95: np.percentile(paths, 95, axis=0)
                }

                fig_ci = go.Figure()

                # 90% confidence band
                fig_ci.add_trace(go.Scatter(
                    x=time_index,
                    y=percentiles[95],
                    mode='lines',
                    name='95th Percentile',
                    line=dict(width=0),
                    showlegend=False
                ))

                fig_ci.add_trace(go.Scatter(
                    x=time_index,
                    y=percentiles[5],
                    mode='lines',
                    name='5th-95th Percentile',
                    fill='tonexty',
                    fillcolor='rgba(0,100,200,0.2)',
                    line=dict(width=0)
                ))

                # 50% confidence band
                fig_ci.add_trace(go.Scatter(
                    x=time_index,
                    y=percentiles[75],
                    mode='lines',
                    name='75th Percentile',
                    line=dict(width=0),
                    showlegend=False
                ))

                fig_ci.add_trace(go.Scatter(
                    x=time_index,
                    y=percentiles[25],
                    mode='lines',
                    name='25th-75th Percentile',
                    fill='tonexty',
                    fillcolor='rgba(0,100,200,0.4)',
                    line=dict(width=0)
                ))

                # Median
                fig_ci.add_trace(go.Scatter(
                    x=time_index,
                    y=percentiles[50],
                    mode='lines',
                    name='Median',
                    line=dict(color='blue', width=2)
                ))

                fig_ci.update_layout(
                    title='Price Confidence Intervals Over Time',
                    xaxis_title='Time (years)',
                    yaxis_title='Price ($)',
                    height=500
                )

                st.plotly_chart(fig_ci, use_container_width=True)

                # 8. Export results
                st.subheader("8. Export Results")

                # Prepare summary data
                summary_data = {
                    'Metric': [
                        'Current Price',
                        'Time Horizon (years)',
                        'Number of Paths',
                        'Mean Final Price',
                        'Median Final Price',
                        'Std Dev Final Price',
                        '5th Percentile',
                        '95th Percentile',
                        'Expected Return',
                        'VaR (95%)',
                        'CVaR (95%)',
                        'Probability of Loss',
                        'Max Drawdown'
                    ],
                    'Value': [
                        f"${S0:.2f}",
                        f"{time_horizon}",
                        f"{n_paths}",
                        f"${stats['mean']:.2f}",
                        f"${stats['median']:.2f}",
                        f"${stats['std']:.2f}",
                        f"${stats['percentile_5']:.2f}",
                        f"${stats['percentile_95']:.2f}",
                        f"{risk_metrics['expected_return']:.2%}",
                        f"{risk_metrics['var']:.2%}",
                        f"{risk_metrics['cvar']:.2%}",
                        f"{risk_metrics['prob_loss']:.1%}",
                        f"{risk_metrics['max_drawdown']:.2%}"
                    ]
                }

                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=csv,
                    file_name=f"monte_carlo_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error: Error running simulation: {str(e)}")
                logger.error(f"Monte Carlo simulation error: {e}", exc_info=True)


if __name__ == "__main__":
    show()
