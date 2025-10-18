"""
Security Pricing
================
Advanced pricing models for different security types.

Features:
- Equity valuation (DCF, DDM, Relative)
- Bond pricing with duration
- Options pricing (Black-Scholes)
- Automatic model selection
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

from portfolio.security_valuation import (
    DCFValuation,
    DCFInputs,
    DDMValuation,
    DDMInputs,
    BondPricing,
    BondInputs,
    RelativeValuation,
    SecurityType,
    AutoValuationSelector
)
from portfolio.options_pricing import BlackScholesModel, OptionParameters
from dashboard.utils.stock_search import compact_stock_search
from dashboard.utils.theme import apply_vscode_theme
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


def show():
    """Main function for security pricing page."""
    apply_vscode_theme()

    st.title("Security Pricing Models")
    st.markdown("""
    Price different types of securities using appropriate valuation models.
    Models automatically adapt based on security type and available data.
    """)

    # Sidebar inputs
    st.sidebar.header("Configuration")

    security_type = st.sidebar.selectbox(
        "Security Type",
        ["Equity (Stock)", "Bond (Fixed Income)", "Option (Derivative)"]
    )

    if security_type == "Equity (Stock)":
        show_equity_pricing()
    elif security_type == "Bond (Fixed Income)":
        show_bond_pricing()
    else:  # Option
        show_option_pricing()


def show_equity_pricing():
    """Display equity pricing interface."""
    st.subheader("Equity Valuation")

    # Sidebar inputs
    with st.sidebar:
        st.subheader("Stock Selection")
        ticker = compact_stock_search(key="security_pricing_search", default="AAPL")

        st.subheader("Valuation Method")
        valuation_method = st.selectbox(
            "Select Method",
            ["DCF (Discounted Cash Flow)", "DDM (Dividend Discount)", "Relative Valuation", "Auto-Select"],
            label_visibility="collapsed"
        )

    if valuation_method in ["DCF (Discounted Cash Flow)", "Auto-Select"]:
        st.markdown("### DCF Valuation Inputs")

        col1, col2 = st.columns(2)

        with col1:
            fcf = st.number_input("Free Cash Flow (Latest Year, $M)", value=100000.0, step=1000.0)
            growth_high = st.slider("High Growth Rate (%)", 0, 50, 15) / 100.0
            growth_stable = st.slider("Stable Growth Rate (%)", 0, 10, 3) / 100.0

        with col2:
            wacc = st.slider("WACC - Cost of Capital (%)", 5, 20, 10) / 100.0
            years_high = st.slider("High Growth Period (years)", 3, 10, 5)
            shares = st.number_input("Shares Outstanding (M)", value=16000.0, step=100.0)

        if st.button("Calculate DCF Value", type="primary"):
            try:
                inputs = DCFInputs(
                    fcf_current=fcf * 1e6,  # Convert to dollars
                    growth_rate_stage1=growth_high,
                    growth_rate_stage2=growth_stable,
                    wacc=wacc,
                    years_stage1=years_high,
                    shares_outstanding=shares * 1e6
                )

                dcf = DCFValuation(inputs)
                result = dcf.calculate_intrinsic_value()

                # Display results
                st.success("DCF Valuation Complete")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Intrinsic Value per Share", f"${result['intrinsic_value']:.2f}")
                with col2:
                    st.metric("Enterprise Value", f"${result['enterprise_value']/1e9:.2f}B")
                with col3:
                    st.metric("Stage 1 PV", f"${result['stage1_pv']/1e9:.2f}B")
                with col4:
                    st.metric("Terminal PV", f"${result['terminal_pv']/1e9:.2f}B")

                # Get current price for comparison
                try:
                    stock = yf.Ticker(ticker)
                    current_price = stock.history(period="1d")['Close'].iloc[-1]

                    upside = (result['intrinsic_value'] - current_price) / current_price

                    st.markdown("### Valuation vs Market Price")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Market Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Intrinsic Value", f"${result['intrinsic_value']:.2f}")
                    with col3:
                        st.metric(
                            "Upside/Downside",
                            f"{upside:.1%}",
                            delta="Undervalued" if upside > 0 else "Overvalued"
                        )

                    # Recommendation
                    if upside > 0.20:
                        st.success("**Strong Buy**: Stock is significantly undervalued (>20% upside)")
                    elif upside > 0.10:
                        st.info("**Buy**: Stock appears undervalued (10-20% upside)")
                    elif upside > -0.10:
                        st.warning("➡️ **Hold**: Stock is fairly valued (±10%)")
                    else:
                        st.error("📉 **Sell**: Stock appears overvalued (>10% downside)")

                except Exception as e:
                    logger.warning(f"Could not fetch current price: {e}")

                # Sensitivity analysis
                st.markdown("### Sensitivity Analysis")

                sensitivity_df = dcf.sensitivity_analysis(
                    wacc_range=(wacc - 0.03, wacc + 0.03),
                    growth_range=(growth_stable - 0.02, growth_stable + 0.02),
                    steps=5
                )

                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=sensitivity_df.values,
                    x=sensitivity_df.columns,
                    y=sensitivity_df.index,
                    colorscale='RdYlGn',
                    text=sensitivity_df.values,
                    texttemplate='$%{text:.0f}',
                    textfont={"size": 10}
                ))

                fig_heatmap.update_layout(
                    title='Intrinsic Value Sensitivity (WACC vs Terminal Growth)',
                    xaxis_title='WACC',
                    yaxis_title='Terminal Growth Rate',
                    height=400
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

            except Exception as e:
                st.error(f"Error: Error calculating DCF: {str(e)}")
                logger.error(f"DCF calculation error: {e}", exc_info=True)

    elif valuation_method == "DDM (Dividend Discount)":
        st.markdown("### DDM (Gordon Growth Model) Inputs")

        col1, col2 = st.columns(2)

        with col1:
            dividend = st.number_input("Current Annual Dividend per Share ($)", value=1.0, step=0.1)
            div_growth = st.slider("Dividend Growth Rate (%)", 0, 15, 5) / 100.0

        with col2:
            required_return = st.slider("Required Return (%)", 5, 20, 10) / 100.0

        if st.button("Calculate DDM Value", type="primary"):
            try:
                if required_return <= div_growth:
                    st.error("Error: Required return must be greater than growth rate!")
                    return

                inputs = DDMInputs(
                    dividend_current=dividend,
                    growth_rate=div_growth,
                    required_return=required_return
                )

                ddm = DDMValuation(inputs)
                result = ddm.calculate_intrinsic_value()

                st.success("DDM Valuation Complete")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Intrinsic Value", f"${result['intrinsic_value']:.2f}")
                with col2:
                    st.metric("Next Year Dividend", f"${result['next_dividend']:.2f}")
                with col3:
                    st.metric("Dividend Yield", f"{result['dividend_yield']:.2%}")

            except Exception as e:
                st.error(f"Error: Error calculating DDM: {str(e)}")

    else:  # Relative Valuation
        st.markdown("### Relative Valuation")

        multiple_type = st.selectbox("Valuation Multiple", ["P/E Ratio", "P/B Ratio", "EV/EBITDA"])

        if multiple_type == "P/E Ratio":
            col1, col2 = st.columns(2)
            with col1:
                eps = st.number_input("Earnings per Share ($)", value=5.0, step=0.1)
            with col2:
                peer_pe = st.number_input("Peer Average P/E", value=20.0, step=0.5)

            if st.button("Calculate Value"):
                fair_value = RelativeValuation.pe_valuation(eps, peer_pe)
                st.metric("Fair Value (P/E Method)", f"${fair_value:.2f}")

        elif multiple_type == "P/B Ratio":
            col1, col2 = st.columns(2)
            with col1:
                book_value = st.number_input("Book Value per Share ($)", value=10.0, step=0.5)
            with col2:
                peer_pb = st.number_input("Peer Average P/B", value=3.0, step=0.1)

            if st.button("Calculate Value"):
                fair_value = RelativeValuation.pb_valuation(book_value, peer_pb)
                st.metric("Fair Value (P/B Method)", f"${fair_value:.2f}")

        else:  # EV/EBITDA
            col1, col2 = st.columns(2)
            with col1:
                ebitda = st.number_input("EBITDA ($M)", value=10000.0, step=100.0)
                net_debt = st.number_input("Net Debt ($M)", value=5000.0, step=100.0)
            with col2:
                peer_ev_ebitda = st.number_input("Peer Average EV/EBITDA", value=12.0, step=0.5)
                shares = st.number_input("Shares Outstanding (M)", value=1000.0, step=10.0)

            if st.button("Calculate Value"):
                fair_value = RelativeValuation.ev_ebitda_valuation(
                    ebitda * 1e6,
                    peer_ev_ebitda,
                    net_debt * 1e6,
                    shares * 1e6
                )
                st.metric("Fair Value per Share (EV/EBITDA Method)", f"${fair_value:.2f}")


def show_bond_pricing():
    """Display bond pricing interface."""
    st.subheader("🏦 Bond Pricing")

    st.markdown("### Bond Parameters")

    col1, col2 = st.columns(2)

    with col1:
        face_value = st.number_input("Face Value ($)", value=1000.0, step=100.0)
        coupon_rate = st.slider("Annual Coupon Rate (%)", 0.0, 15.0, 5.0) / 100.0
        ytm = st.slider("Yield to Maturity (%)", 0.0, 15.0, 5.0) / 100.0

    with col2:
        years_to_maturity = st.number_input("Years to Maturity", value=10.0, step=0.5)
        frequency = st.selectbox("Coupon Frequency", [1, 2, 4, 12], index=1,
                                format_func=lambda x: f"{x}x per year ({'Annual' if x==1 else 'Semi-annual' if x==2 else 'Quarterly' if x==4 else 'Monthly'})")

    if st.button("Calculate Bond Price", type="primary"):
        try:
            inputs = BondInputs(
                face_value=face_value,
                coupon_rate=coupon_rate,
                years_to_maturity=years_to_maturity,
                ytm=ytm,
                frequency=frequency
            )

            bond = BondPricing(inputs)
            result = bond.calculate_price()
            duration = bond.calculate_duration()

            st.success("Bond Pricing Complete")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Bond Price", f"${result['price']:.2f}")
            with col2:
                st.metric("Current Yield", f"{result['current_yield']:.2%}")
            with col3:
                st.metric("Duration (years)", f"{duration:.2f}")
            with col4:
                premium_discount = result['price'] - face_value
                st.metric(
                    "Premium/Discount",
                    f"${premium_discount:.2f}",
                    delta="Premium" if premium_discount > 0 else "Discount" if premium_discount < 0 else "At Par"
                )

            # Breakdown
            st.markdown("### Price Components")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("PV of Coupons", f"${result['pv_coupons']:.2f}")
            with col2:
                st.metric("PV of Face Value", f"${result['pv_face_value']:.2f}")

            # Price vs YTM chart
            st.markdown("### Bond Price vs Yield Curve")

            ytm_range = np.linspace(max(0.001, ytm - 0.05), ytm + 0.05, 50)
            prices = []

            for y in ytm_range:
                temp_inputs = BondInputs(
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    years_to_maturity=years_to_maturity,
                    ytm=y,
                    frequency=frequency
                )
                temp_bond = BondPricing(temp_inputs)
                temp_result = temp_bond.calculate_price()
                prices.append(temp_result['price'])

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=ytm_range * 100,
                y=prices,
                mode='lines',
                name='Bond Price',
                line=dict(color='blue', width=2)
            ))

            fig.add_vline(x=ytm * 100, line_dash="dash", line_color="red",
                         annotation_text=f"Current YTM: {ytm:.2%}")

            fig.update_layout(
                title='Bond Price vs Yield to Maturity',
                xaxis_title='Yield to Maturity (%)',
                yaxis_title='Bond Price ($)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: Error calculating bond price: {str(e)}")
            logger.error(f"Bond pricing error: {e}", exc_info=True)


def show_option_pricing():
    """Display option pricing interface."""
    st.subheader("Options Pricing (Black-Scholes)")

    st.markdown("### Option Parameters")

    col1, col2 = st.columns(2)

    with col1:
        S = st.number_input("Underlying Price ($)", value=100.0, step=1.0)
        K = st.number_input("Strike Price ($)", value=100.0, step=1.0)
        T = st.number_input("Time to Expiration (years)", value=1.0, step=0.1)

    with col2:
        r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100.0
        sigma = st.slider("Volatility (%)", 5.0, 100.0, 20.0) / 100.0
        q = st.slider("Dividend Yield (%)", 0.0, 10.0, 0.0) / 100.0

    option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)

    if st.button("Calculate Option Price", type="primary"):
        try:
            params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
            bs_model = BlackScholesModel(params)

            if option_type == "Call":
                price = bs_model.call_price()
            else:
                price = bs_model.put_price()

            greeks = bs_model.greeks(option_type.lower())

            st.success("Option Pricing Complete")

            # Price
            st.markdown("### Option Price")
            st.metric(f"{option_type} Option Price", f"${price:.2f}")

            # Greeks
            st.markdown("### The Greeks")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Delta", f"{greeks['delta']:.4f}")
                st.caption("Price sensitivity to underlying")
            with col2:
                st.metric("Gamma", f"{greeks['gamma']:.4f}")
                st.caption("Delta sensitivity")
            with col3:
                st.metric("Vega", f"{greeks['vega']:.4f}")
                st.caption("Volatility sensitivity")
            with col4:
                st.metric("Theta", f"{greeks['theta']:.4f}")
                st.caption("Time decay (per day)")
            with col5:
                st.metric("Rho", f"{greeks['rho']:.4f}")
                st.caption("Interest rate sensitivity")

            # Payoff diagram
            st.markdown("### Payoff Diagram")

            spot_range = np.linspace(S * 0.7, S * 1.3, 100)
            intrinsic_values = []
            option_values = []

            for spot in spot_range:
                temp_params = OptionParameters(S=spot, K=K, T=T, r=r, sigma=sigma, q=q)
                temp_bs = BlackScholesModel(temp_params)

                if option_type == "Call":
                    intrinsic = max(spot - K, 0)
                    option_value = temp_bs.call_price()
                else:
                    intrinsic = max(K - spot, 0)
                    option_value = temp_bs.put_price()

                intrinsic_values.append(intrinsic)
                option_values.append(option_value)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=spot_range,
                y=intrinsic_values,
                mode='lines',
                name='Intrinsic Value',
                line=dict(color='gray', dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=spot_range,
                y=option_values,
                mode='lines',
                name='Option Value',
                line=dict(color='blue', width=2)
            ))

            fig.add_vline(x=K, line_dash="dash", line_color="red",
                         annotation_text=f"Strike: ${K}")
            fig.add_vline(x=S, line_dash="dash", line_color="green",
                         annotation_text=f"Current: ${S}")

            fig.update_layout(
                title=f'{option_type} Option Value vs Underlying Price',
                xaxis_title='Underlying Price ($)',
                yaxis_title='Option Value ($)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: Error calculating option price: {str(e)}")
            logger.error(f"Option pricing error: {e}", exc_info=True)


if __name__ == "__main__":
    show()
