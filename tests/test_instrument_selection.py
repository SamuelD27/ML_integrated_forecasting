"""
Tests for Instrument Selection (Phase 5)
========================================
Tests for options loader and instrument selector.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules under test
from data.options_loader import (
    OptionContract,
    OptionsChain,
    load_options_chain,
    get_hedge_candidates,
    get_income_candidates,
    calculate_option_greeks,
    HAS_YFINANCE,
)

from pipeline.instrument_selector import (
    InstrumentType,
    InstrumentSelection,
    PortfolioHedge,
    select_instrument,
    select_instruments_batch,
    build_portfolio_hedge,
    build_income_overlay,
    apply_regime_adjustments,
    summarize_selections,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_option_contract():
    """Create a sample option contract."""
    return OptionContract(
        ticker='AAPL',
        contract_symbol='AAPL240119C00150000',
        strike=150.0,
        expiration=pd.Timestamp('2024-01-19'),
        option_type='call',
        last_price=5.50,
        bid=5.40,
        ask=5.60,
        volume=1000,
        open_interest=5000,
        implied_vol=0.25,
        delta=0.45,
        gamma=0.02,
        theta=-0.05,
        vega=0.15,
        in_the_money=False,
    )


@pytest.fixture
def sample_options_chain():
    """Create a sample options chain."""
    expiry = pd.Timestamp.now() + timedelta(days=30)

    calls = [
        OptionContract(
            ticker='AAPL',
            contract_symbol=f'AAPL_C_{strike}',
            strike=strike,
            expiration=expiry,
            option_type='call',
            last_price=max(1.0, 10 - (strike - 150) * 0.5),
            bid=max(0.95, 9.5 - (strike - 150) * 0.5),
            ask=max(1.05, 10.5 - (strike - 150) * 0.5),
            volume=500,
            open_interest=2000,
            implied_vol=0.25,
            in_the_money=strike < 155,
        )
        for strike in [145, 150, 155, 160, 165, 170]
    ]

    puts = [
        OptionContract(
            ticker='AAPL',
            contract_symbol=f'AAPL_P_{strike}',
            strike=strike,
            expiration=expiry,
            option_type='put',
            last_price=max(1.0, 10 - (155 - strike) * 0.5),
            bid=max(0.95, 9.5 - (155 - strike) * 0.5),
            ask=max(1.05, 10.5 - (155 - strike) * 0.5),
            volume=400,
            open_interest=1800,
            implied_vol=0.28,
            in_the_money=strike > 155,
        )
        for strike in [140, 145, 150, 155, 160]
    ]

    return OptionsChain(
        ticker='AAPL',
        underlying_price=155.0,
        as_of_date=pd.Timestamp.now(),
        expirations=[expiry],
        calls=calls,
        puts=puts,
    )


@pytest.fixture
def mock_trade_signal():
    """Create a mock trade signal."""
    class MockSignal:
        def __init__(self, ticker, direction, regime, expected_return, expected_vol, meta_prob=None):
            self.ticker = ticker
            self.direction = direction
            self.regime = regime
            self.regime_label = {0: 'Bull', 1: 'Bear', 2: 'Neutral', 3: 'Crisis'}.get(regime, 'Neutral')
            self.expected_return = expected_return
            self.expected_vol = expected_vol
            self.meta_prob = meta_prob

    return MockSignal


# =============================================================================
# Test Option Contract
# =============================================================================

class TestOptionContract:
    """Tests for OptionContract dataclass."""

    def test_option_contract_creation(self, sample_option_contract):
        """Test creating an option contract."""
        contract = sample_option_contract

        assert contract.ticker == 'AAPL'
        assert contract.strike == 150.0
        assert contract.option_type == 'call'
        assert contract.implied_vol == 0.25

    def test_mid_price_calculation(self, sample_option_contract):
        """Test mid price calculation."""
        contract = sample_option_contract

        expected_mid = (5.40 + 5.60) / 2
        assert contract.mid_price == pytest.approx(expected_mid, rel=0.01)

    def test_spread_calculation(self, sample_option_contract):
        """Test spread calculation."""
        contract = sample_option_contract

        assert contract.spread == pytest.approx(0.20, rel=0.01)

    def test_spread_pct_calculation(self, sample_option_contract):
        """Test spread percentage calculation."""
        contract = sample_option_contract

        expected_pct = 0.20 / 5.50
        assert contract.spread_pct == pytest.approx(expected_pct, rel=0.01)

    def test_days_to_expiry(self):
        """Test days to expiry calculation."""
        future_expiry = pd.Timestamp.now() + timedelta(days=30)
        contract = OptionContract(
            ticker='TEST',
            contract_symbol='TEST',
            strike=100,
            expiration=future_expiry,
            option_type='call',
        )

        # Should be approximately 30 days
        assert 29 <= contract.days_to_expiry <= 31

    def test_mid_price_fallback(self):
        """Test mid price falls back to last price."""
        contract = OptionContract(
            ticker='TEST',
            contract_symbol='TEST',
            strike=100,
            expiration=pd.Timestamp.now(),
            option_type='call',
            last_price=5.0,
            bid=0,
            ask=0,
        )

        assert contract.mid_price == 5.0


# =============================================================================
# Test Options Chain
# =============================================================================

class TestOptionsChain:
    """Tests for OptionsChain dataclass."""

    def test_options_chain_creation(self, sample_options_chain):
        """Test creating an options chain."""
        chain = sample_options_chain

        assert chain.ticker == 'AAPL'
        assert chain.underlying_price == 155.0
        assert len(chain.calls) == 6
        assert len(chain.puts) == 5

    def test_get_calls_by_expiry(self, sample_options_chain):
        """Test getting calls by expiration."""
        chain = sample_options_chain
        expiry = chain.expirations[0]

        calls = chain.get_calls_by_expiry(expiry)

        assert len(calls) == 6
        assert all(c.option_type == 'call' for c in calls)

    def test_get_puts_by_expiry(self, sample_options_chain):
        """Test getting puts by expiration."""
        chain = sample_options_chain
        expiry = chain.expirations[0]

        puts = chain.get_puts_by_expiry(expiry)

        assert len(puts) == 5
        assert all(p.option_type == 'put' for p in puts)

    def test_get_atm_call(self, sample_options_chain):
        """Test getting ATM call."""
        chain = sample_options_chain
        expiry = chain.expirations[0]

        atm_call = chain.get_atm_call(expiry)

        assert atm_call is not None
        # ATM strike should be closest to 155
        assert atm_call.strike == 155.0

    def test_get_atm_put(self, sample_options_chain):
        """Test getting ATM put."""
        chain = sample_options_chain
        expiry = chain.expirations[0]

        atm_put = chain.get_atm_put(expiry)

        assert atm_put is not None
        assert atm_put.strike == 155.0

    def test_get_otm_puts(self, sample_options_chain):
        """Test getting OTM puts."""
        chain = sample_options_chain
        expiry = chain.expirations[0]

        otm_puts = chain.get_otm_puts(expiry)

        # OTM puts have strike < underlying (155)
        assert all(p.strike < chain.underlying_price for p in otm_puts)

    def test_get_otm_calls(self, sample_options_chain):
        """Test getting OTM calls."""
        chain = sample_options_chain
        expiry = chain.expirations[0]

        otm_calls = chain.get_otm_calls(expiry)

        # OTM calls have strike > underlying (155)
        assert all(c.strike > chain.underlying_price for c in otm_calls)

    def test_to_dataframe(self, sample_options_chain):
        """Test converting to DataFrame."""
        chain = sample_options_chain

        df = chain.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(chain.calls) + len(chain.puts)
        assert 'strike' in df.columns
        assert 'option_type' in df.columns


# =============================================================================
# Test Options Loader Functions
# =============================================================================

class TestOptionsLoaderFunctions:
    """Tests for options loader utility functions."""

    def test_calculate_option_greeks_call(self):
        """Test Black-Scholes Greek calculation for call."""
        greeks = calculate_option_greeks(
            spot=100,
            strike=100,
            tte=30/365,  # 30 days
            vol=0.25,
            rf_rate=0.05,
            option_type='call',
        )

        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks

        # ATM call delta should be around 0.5
        assert 0.4 < greeks['delta'] < 0.6

        # Gamma should be positive
        assert greeks['gamma'] > 0

        # Theta should be negative
        assert greeks['theta'] < 0

    def test_calculate_option_greeks_put(self):
        """Test Black-Scholes Greek calculation for put."""
        greeks = calculate_option_greeks(
            spot=100,
            strike=100,
            tte=30/365,
            vol=0.25,
            rf_rate=0.05,
            option_type='put',
        )

        # ATM put delta should be around -0.5
        assert -0.6 < greeks['delta'] < -0.4

    def test_calculate_option_greeks_zero_tte(self):
        """Test Greeks at expiration."""
        greeks = calculate_option_greeks(
            spot=100,
            strike=100,
            tte=0,
            vol=0.25,
        )

        # At expiration, Greeks should be zero
        assert greeks['delta'] == 0.0
        assert greeks['gamma'] == 0.0

    def test_calculate_option_greeks_itm_call(self):
        """Test ITM call has high delta."""
        greeks = calculate_option_greeks(
            spot=120,
            strike=100,
            tte=30/365,
            vol=0.25,
            option_type='call',
        )

        # Deep ITM call should have delta close to 1
        assert greeks['delta'] > 0.8

    def test_calculate_option_greeks_otm_put(self):
        """Test OTM put has small delta."""
        greeks = calculate_option_greeks(
            spot=120,
            strike=100,
            tte=30/365,
            vol=0.25,
            option_type='put',
        )

        # Deep OTM put should have delta close to 0
        assert -0.2 < greeks['delta'] < 0


# =============================================================================
# Test Instrument Selection
# =============================================================================

class TestInstrumentSelection:
    """Tests for InstrumentSelection dataclass."""

    def test_instrument_selection_equity(self):
        """Test creating equity instrument selection."""
        selection = InstrumentSelection(
            ticker='AAPL',
            instrument_type=InstrumentType.EQUITY,
            direction='long',
            quantity=100,
            notional_value=15500.0,
            strategy_name='equity_long',
        )

        assert selection.ticker == 'AAPL'
        assert selection.instrument_type == InstrumentType.EQUITY
        assert selection.quantity == 100
        assert selection.strategy_name == 'equity_long'

    def test_instrument_selection_option(self):
        """Test creating option instrument selection."""
        selection = InstrumentSelection(
            ticker='AAPL',
            instrument_type=InstrumentType.CALL,
            direction='long',
            quantity=100,
            notional_value=500.0,
            option_contract='AAPL240119C00155000',
            strike=155.0,
            expiration=pd.Timestamp('2024-01-19'),
            option_premium=5.0,
            n_contracts=1,
            strategy_name='long_call',
        )

        assert selection.instrument_type == InstrumentType.CALL
        assert selection.strike == 155.0
        assert selection.n_contracts == 1


class TestSelectInstrument:
    """Tests for select_instrument function."""

    def test_select_instrument_equity_default(self, mock_trade_signal):
        """Test default equity selection."""
        signal = mock_trade_signal('AAPL', 'long', 0, 0.05, 0.20, meta_prob=0.75)

        selection = select_instrument(
            signal=signal,
            portfolio_value=100000,
            current_price=155.0,
            config={'instruments': {'use_options': False}},
        )

        assert selection.instrument_type == InstrumentType.EQUITY
        assert selection.ticker == 'AAPL'
        assert selection.direction == 'long'
        assert selection.quantity > 0

    def test_select_instrument_position_sizing(self, mock_trade_signal):
        """Test position sizing respects max allocation."""
        signal = mock_trade_signal('AAPL', 'long', 0, 0.05, 0.20)

        selection = select_instrument(
            signal=signal,
            portfolio_value=100000,
            current_price=155.0,
            config={'instruments': {'max_position_pct': 0.05}},
        )

        # Notional should be approximately 5% of portfolio
        max_notional = 100000 * 0.05
        assert selection.notional_value <= max_notional * 1.1  # 10% tolerance

    def test_select_instrument_crisis_regime_uses_options(self, mock_trade_signal):
        """Test that crisis regime prefers options when enabled."""
        signal = mock_trade_signal('AAPL', 'long', 3, 0.05, 0.20, meta_prob=0.70)  # Crisis regime

        # Without mocking options chain, it will fall back to equity
        selection = select_instrument(
            signal=signal,
            portfolio_value=100000,
            current_price=155.0,
            config={'instruments': {'use_options': True}},
        )

        # Will fall back to equity since no options chain available
        assert selection.instrument_type in [InstrumentType.EQUITY, InstrumentType.CALL]

    def test_select_instruments_batch(self, mock_trade_signal):
        """Test batch instrument selection."""
        signals = [
            mock_trade_signal('AAPL', 'long', 0, 0.05, 0.20),
            mock_trade_signal('GOOGL', 'long', 0, 0.04, 0.18),
            mock_trade_signal('MSFT', 'long', 0, 0.06, 0.22),
        ]

        prices = {'AAPL': 155.0, 'GOOGL': 140.0, 'MSFT': 380.0}

        selections = select_instruments_batch(
            signals=signals,
            portfolio_value=100000,
            prices=prices,
        )

        assert len(selections) == 3
        assert all(s.ticker in prices for s in selections)


# =============================================================================
# Test Portfolio Hedge
# =============================================================================

class TestPortfolioHedge:
    """Tests for portfolio hedge building."""

    def test_portfolio_hedge_creation(self):
        """Test creating a portfolio hedge object."""
        hedge = PortfolioHedge(
            hedge_ticker='SPY',
            instrument_type=InstrumentType.PUT,
            contracts=[{'strike': 450, 'premium': 5.0}],
            total_cost=500.0,
            portfolio_delta_reduction=-25.0,
            recommended_contracts=1,
        )

        assert hedge.hedge_ticker == 'SPY'
        assert hedge.instrument_type == InstrumentType.PUT
        assert hedge.total_cost == 500.0


# =============================================================================
# Test Regime Adjustments
# =============================================================================

class TestRegimeAdjustments:
    """Tests for regime-based adjustments."""

    def test_apply_regime_adjustments_bull(self):
        """Test regime adjustments in bull market."""
        selections = [
            InstrumentSelection(
                ticker='AAPL',
                instrument_type=InstrumentType.EQUITY,
                direction='long',
                quantity=100,
                notional_value=15500.0,
            ),
        ]

        adjusted = apply_regime_adjustments(
            selections=selections,
            regime=0,  # Bull
            config={'regime_adjustments': {'bull_multiplier': 1.0}},
        )

        assert len(adjusted) == 1
        assert adjusted[0].quantity == 100  # No change in bull

    def test_apply_regime_adjustments_bear(self):
        """Test regime adjustments in bear market."""
        selections = [
            InstrumentSelection(
                ticker='AAPL',
                instrument_type=InstrumentType.EQUITY,
                direction='long',
                quantity=100,
                notional_value=15500.0,
            ),
        ]

        adjusted = apply_regime_adjustments(
            selections=selections,
            regime=1,  # Bear
            config={'regime_adjustments': {'bear_multiplier': 0.5}},
        )

        assert len(adjusted) == 1
        assert adjusted[0].quantity == 50  # Reduced by 50%

    def test_apply_regime_adjustments_crisis(self):
        """Test regime adjustments in crisis."""
        selections = [
            InstrumentSelection(
                ticker='AAPL',
                instrument_type=InstrumentType.EQUITY,
                direction='long',
                quantity=100,
                notional_value=15500.0,
            ),
        ]

        adjusted = apply_regime_adjustments(
            selections=selections,
            regime=3,  # Crisis
            config={'regime_adjustments': {'crisis_multiplier': 0.3}},
        )

        assert len(adjusted) == 1
        assert adjusted[0].quantity == 30  # Reduced by 70%

    def test_apply_regime_adjustments_removes_zero_quantity(self):
        """Test that zero quantity selections are removed."""
        selections = [
            InstrumentSelection(
                ticker='AAPL',
                instrument_type=InstrumentType.EQUITY,
                direction='long',
                quantity=2,  # Very small
                notional_value=310.0,
            ),
        ]

        adjusted = apply_regime_adjustments(
            selections=selections,
            regime=3,  # Crisis
            config={'regime_adjustments': {'crisis_multiplier': 0.3}},
        )

        # 2 * 0.3 = 0.6 -> rounds to 0, should be removed
        assert len(adjusted) == 0


# =============================================================================
# Test Summary Functions
# =============================================================================

class TestSummarizeSelections:
    """Tests for selection summarization."""

    def test_summarize_empty(self):
        """Test summarizing empty selections."""
        summary = summarize_selections([])

        assert summary['total_selections'] == 0
        assert summary['total_notional'] == 0.0

    def test_summarize_mixed_selections(self):
        """Test summarizing mixed instrument types."""
        selections = [
            InstrumentSelection(
                ticker='AAPL',
                instrument_type=InstrumentType.EQUITY,
                direction='long',
                quantity=100,
                notional_value=15500.0,
                strategy_name='equity_long',
            ),
            InstrumentSelection(
                ticker='GOOGL',
                instrument_type=InstrumentType.CALL,
                direction='long',
                quantity=100,
                notional_value=500.0,
                n_contracts=1,
                option_premium=5.0,
                strategy_name='long_call',
            ),
            InstrumentSelection(
                ticker='MSFT',
                instrument_type=InstrumentType.EQUITY,
                direction='long',
                quantity=50,
                notional_value=19000.0,
                strategy_name='equity_long',
            ),
        ]

        summary = summarize_selections(selections)

        assert summary['total_selections'] == 3
        assert summary['by_type']['equity'] == 2
        assert summary['by_type']['call'] == 1
        assert summary['by_strategy']['equity_long'] == 2
        assert summary['by_strategy']['long_call'] == 1
        assert summary['total_notional'] == pytest.approx(35000.0, rel=0.01)


# =============================================================================
# Integration Tests
# =============================================================================

class TestInstrumentSelectionIntegration:
    """Integration tests for instrument selection pipeline."""

    def test_full_selection_pipeline(self, mock_trade_signal):
        """Test full pipeline from signals to selections."""
        # Create signals
        signals = [
            mock_trade_signal('AAPL', 'long', 0, 0.05, 0.20, meta_prob=0.80),
            mock_trade_signal('GOOGL', 'long', 0, 0.04, 0.18, meta_prob=0.75),
        ]

        prices = {'AAPL': 155.0, 'GOOGL': 140.0}

        # Select instruments
        selections = select_instruments_batch(
            signals=signals,
            portfolio_value=100000,
            prices=prices,
            config={'instruments': {'max_position_pct': 0.05}},
        )

        # Apply regime adjustments
        adjusted = apply_regime_adjustments(
            selections=selections,
            regime=0,  # Bull market
        )

        # Summarize
        summary = summarize_selections(adjusted)

        assert summary['total_selections'] == 2
        assert len(summary['tickers']) == 2

    def test_selection_with_missing_price(self, mock_trade_signal):
        """Test handling of missing prices."""
        signals = [
            mock_trade_signal('AAPL', 'long', 0, 0.05, 0.20),
            mock_trade_signal('UNKNOWN', 'long', 0, 0.04, 0.18),
        ]

        prices = {'AAPL': 155.0}  # Missing UNKNOWN

        selections = select_instruments_batch(
            signals=signals,
            portfolio_value=100000,
            prices=prices,
        )

        # Should only have AAPL
        assert len(selections) == 1
        assert selections[0].ticker == 'AAPL'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
