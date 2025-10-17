"""
Advanced Security Valuation
===========================
Comprehensive valuation models for different security types.

Implements:
- DCF (Discounted Cash Flow) for equities
- DDM (Dividend Discount Model)
- Bond pricing (fixed income)
- Relative valuation (P/E, P/B, EV/EBITDA)
- Automatic model selection based on security type
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityType(Enum):
    """Types of securities."""
    EQUITY = "equity"
    BOND = "bond"
    PREFERRED_STOCK = "preferred"
    REIT = "reit"
    ETF = "etf"


@dataclass
class DCFInputs:
    """Inputs for Discounted Cash Flow model."""
    fcf_current: float  # Free cash flow (most recent year)
    growth_rate_stage1: float  # High growth rate (years 1-5)
    growth_rate_stage2: float  # Stable growth rate (terminal)
    wacc: float  # Weighted average cost of capital
    years_stage1: int = 5  # Years of high growth
    shares_outstanding: float = 1.0  # Shares outstanding


@dataclass
class DDMInputs:
    """Inputs for Dividend Discount Model."""
    dividend_current: float  # Current annual dividend
    growth_rate: float  # Dividend growth rate
    required_return: float  # Required rate of return


@dataclass
class BondInputs:
    """Inputs for bond pricing."""
    face_value: float  # Par value
    coupon_rate: float  # Annual coupon rate
    years_to_maturity: float  # Years to maturity
    ytm: float  # Yield to maturity
    frequency: int = 2  # Coupon frequency (2=semi-annual)


class DCFValuation:
    """
    Discounted Cash Flow valuation for equities.

    Two-stage model: high growth period + stable growth terminal value.
    """

    def __init__(self, inputs: DCFInputs):
        """
        Initialize DCF model.

        Args:
            inputs: DCF model inputs
        """
        self.inputs = inputs

    def calculate_intrinsic_value(self) -> Dict[str, float]:
        """
        Calculate intrinsic value using two-stage DCF.

        Returns:
            Dictionary with valuation results
        """
        fcf = self.inputs.fcf_current
        g1 = self.inputs.growth_rate_stage1
        g2 = self.inputs.growth_rate_stage2
        wacc = self.inputs.wacc
        years = self.inputs.years_stage1

        # Stage 1: High growth period
        stage1_pv = 0
        for year in range(1, years + 1):
            fcf_year = fcf * (1 + g1) ** year
            pv = fcf_year / (1 + wacc) ** year
            stage1_pv += pv

        # Stage 2: Terminal value (stable growth)
        fcf_terminal = fcf * (1 + g1) ** years * (1 + g2)
        terminal_value = fcf_terminal / (wacc - g2)
        terminal_pv = terminal_value / (1 + wacc) ** years

        # Total enterprise value
        enterprise_value = stage1_pv + terminal_pv

        # Equity value per share
        equity_value_per_share = enterprise_value / self.inputs.shares_outstanding

        return {
            'intrinsic_value': equity_value_per_share,
            'stage1_pv': stage1_pv,
            'terminal_pv': terminal_pv,
            'enterprise_value': enterprise_value,
            'terminal_value': terminal_value
        }

    def sensitivity_analysis(self, wacc_range: Tuple[float, float],
                            growth_range: Tuple[float, float],
                            steps: int = 10) -> pd.DataFrame:
        """
        Perform sensitivity analysis on WACC and terminal growth rate.

        Args:
            wacc_range: (min_wacc, max_wacc)
            growth_range: (min_growth, max_growth)
            steps: Number of steps in each dimension

        Returns:
            DataFrame with intrinsic values for different parameter combinations
        """
        wacc_values = np.linspace(wacc_range[0], wacc_range[1], steps)
        growth_values = np.linspace(growth_range[0], growth_range[1], steps)

        results = np.zeros((len(growth_values), len(wacc_values)))

        original_wacc = self.inputs.wacc
        original_growth = self.inputs.growth_rate_stage2

        for i, growth in enumerate(growth_values):
            for j, wacc in enumerate(wacc_values):
                self.inputs.wacc = wacc
                self.inputs.growth_rate_stage2 = growth

                valuation = self.calculate_intrinsic_value()
                results[i, j] = valuation['intrinsic_value']

        # Restore original values
        self.inputs.wacc = original_wacc
        self.inputs.growth_rate_stage2 = original_growth

        # Create DataFrame
        df = pd.DataFrame(
            results,
            index=[f'{g:.1%}' for g in growth_values],
            columns=[f'{w:.1%}' for w in wacc_values]
        )

        return df


class DDMValuation:
    """
    Dividend Discount Model (Gordon Growth Model).

    Best for mature companies with stable dividends.
    """

    def __init__(self, inputs: DDMInputs):
        """
        Initialize DDM model.

        Args:
            inputs: DDM model inputs
        """
        self.inputs = inputs

    def calculate_intrinsic_value(self) -> Dict[str, float]:
        """
        Calculate intrinsic value using Gordon Growth Model.

        Formula: V = D1 / (r - g)
        where D1 = next year's dividend

        Returns:
            Dictionary with valuation results
        """
        D0 = self.inputs.dividend_current
        g = self.inputs.growth_rate
        r = self.inputs.required_return

        if r <= g:
            raise ValueError(
                f"Required return ({r:.2%}) must be greater than growth rate ({g:.2%})"
            )

        # Next year's dividend
        D1 = D0 * (1 + g)

        # Intrinsic value
        intrinsic_value = D1 / (r - g)

        # Dividend yield
        dividend_yield = D1 / intrinsic_value

        return {
            'intrinsic_value': intrinsic_value,
            'next_dividend': D1,
            'dividend_yield': dividend_yield,
            'growth_rate': g,
            'required_return': r
        }


class BondPricing:
    """
    Bond pricing and yield calculations.
    """

    def __init__(self, inputs: BondInputs):
        """
        Initialize bond pricing model.

        Args:
            inputs: Bond pricing inputs
        """
        self.inputs = inputs

    def calculate_price(self) -> Dict[str, float]:
        """
        Calculate bond price.

        Returns:
            Dictionary with pricing results
        """
        FV = self.inputs.face_value
        c = self.inputs.coupon_rate
        T = self.inputs.years_to_maturity
        y = self.inputs.ytm
        freq = self.inputs.frequency

        # Number of periods
        n_periods = int(T * freq)

        # Periodic coupon payment
        coupon_payment = (FV * c) / freq

        # Periodic yield
        periodic_yield = y / freq

        # Present value of coupons
        if periodic_yield > 0:
            pv_coupons = coupon_payment * (
                (1 - (1 + periodic_yield) ** (-n_periods)) / periodic_yield
            )
        else:
            pv_coupons = coupon_payment * n_periods

        # Present value of face value
        pv_face = FV / (1 + periodic_yield) ** n_periods

        # Total bond price
        bond_price = pv_coupons + pv_face

        # Current yield
        current_yield = (FV * c) / bond_price

        return {
            'price': bond_price,
            'pv_coupons': pv_coupons,
            'pv_face_value': pv_face,
            'current_yield': current_yield,
            'ytm': y
        }

    def calculate_duration(self) -> float:
        """
        Calculate Macaulay duration.

        Returns:
            Duration in years
        """
        FV = self.inputs.face_value
        c = self.inputs.coupon_rate
        T = self.inputs.years_to_maturity
        y = self.inputs.ytm
        freq = self.inputs.frequency

        n_periods = int(T * freq)
        coupon_payment = (FV * c) / freq
        periodic_yield = y / freq

        # Calculate price
        bond_price = self.calculate_price()['price']

        # Weighted cash flows
        weighted_cf = 0
        for t in range(1, n_periods + 1):
            time_years = t / freq
            if t < n_periods:
                cf = coupon_payment
            else:
                cf = coupon_payment + FV

            pv = cf / (1 + periodic_yield) ** t
            weighted_cf += time_years * pv

        duration = weighted_cf / bond_price

        return duration


class RelativeValuation:
    """
    Relative valuation using multiples.
    """

    @staticmethod
    def pe_valuation(earnings_per_share: float, peer_average_pe: float) -> float:
        """
        Value stock using P/E multiple.

        Args:
            earnings_per_share: Company's EPS
            peer_average_pe: Average P/E of peer group

        Returns:
            Estimated fair value
        """
        return earnings_per_share * peer_average_pe

    @staticmethod
    def pb_valuation(book_value_per_share: float, peer_average_pb: float) -> float:
        """
        Value stock using P/B multiple.

        Args:
            book_value_per_share: Company's book value per share
            peer_average_pb: Average P/B of peer group

        Returns:
            Estimated fair value
        """
        return book_value_per_share * peer_average_pb

    @staticmethod
    def ev_ebitda_valuation(ebitda: float, peer_average_ev_ebitda: float,
                           net_debt: float, shares_outstanding: float) -> float:
        """
        Value stock using EV/EBITDA multiple.

        Args:
            ebitda: Company's EBITDA
            peer_average_ev_ebitda: Average EV/EBITDA of peer group
            net_debt: Net debt (debt - cash)
            shares_outstanding: Number of shares

        Returns:
            Estimated fair value per share
        """
        enterprise_value = ebitda * peer_average_ev_ebitda
        equity_value = enterprise_value - net_debt
        value_per_share = equity_value / shares_outstanding

        return value_per_share


class AutoValuationSelector:
    """
    Automatically select and apply appropriate valuation model.
    """

    @staticmethod
    def select_model(security_type: SecurityType,
                    has_dividends: bool = False,
                    has_fcf: bool = False) -> str:
        """
        Select appropriate valuation model based on security characteristics.

        Args:
            security_type: Type of security
            has_dividends: Whether security pays dividends
            has_fcf: Whether free cash flow data is available

        Returns:
            Recommended model name
        """
        if security_type == SecurityType.BOND:
            return "bond_pricing"

        elif security_type == SecurityType.EQUITY:
            if has_fcf:
                return "dcf"
            elif has_dividends:
                return "ddm"
            else:
                return "relative_valuation"

        elif security_type == SecurityType.REIT:
            return "ddm"  # REITs pay high dividends

        elif security_type == SecurityType.PREFERRED_STOCK:
            return "ddm"  # Preferred stock is dividend-focused

        else:
            return "relative_valuation"

    @staticmethod
    def value_security(security_type: SecurityType, data: Dict) -> Dict[str, float]:
        """
        Value security using automatically selected model.

        Args:
            security_type: Type of security
            data: Dictionary with necessary inputs

        Returns:
            Valuation results
        """
        model = AutoValuationSelector.select_model(
            security_type,
            has_dividends=data.get('dividend', 0) > 0,
            has_fcf=data.get('fcf', 0) != 0
        )

        logger.info(f"Using {model} for {security_type.value} valuation")

        if model == "dcf" and 'fcf' in data:
            inputs = DCFInputs(
                fcf_current=data['fcf'],
                growth_rate_stage1=data.get('growth_high', 0.15),
                growth_rate_stage2=data.get('growth_stable', 0.03),
                wacc=data.get('wacc', 0.10),
                shares_outstanding=data.get('shares', 1.0)
            )
            dcf = DCFValuation(inputs)
            return dcf.calculate_intrinsic_value()

        elif model == "ddm" and 'dividend' in data:
            inputs = DDMInputs(
                dividend_current=data['dividend'],
                growth_rate=data.get('div_growth', 0.05),
                required_return=data.get('required_return', 0.10)
            )
            ddm = DDMValuation(inputs)
            return ddm.calculate_intrinsic_value()

        elif model == "bond_pricing":
            inputs = BondInputs(
                face_value=data.get('face_value', 1000),
                coupon_rate=data.get('coupon_rate', 0.05),
                years_to_maturity=data.get('maturity', 10),
                ytm=data.get('ytm', 0.05)
            )
            bond = BondPricing(inputs)
            return bond.calculate_price()

        else:
            # Relative valuation
            if 'eps' in data and 'peer_pe' in data:
                fair_value = RelativeValuation.pe_valuation(
                    data['eps'], data['peer_pe']
                )
                return {'intrinsic_value': fair_value, 'method': 'P/E'}

            elif 'book_value' in data and 'peer_pb' in data:
                fair_value = RelativeValuation.pb_valuation(
                    data['book_value'], data['peer_pb']
                )
                return {'intrinsic_value': fair_value, 'method': 'P/B'}

            else:
                return {'intrinsic_value': None, 'method': 'insufficient_data'}
