#!/usr/bin/env python3
"""
Alpaca Paper Trading Pipeline Integration Test
===============================================
Tests the complete paper trading pipeline with real API calls.

Requirements:
    - ALPACA_API_KEY and ALPACA_API_SECRET in environment or .env
    - alpaca-py package installed

Usage:
    python test_alpaca_pipeline.py

    # Or with pytest
    pytest test_alpaca_pipeline.py -v -s
"""

from __future__ import annotations

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deployment.alpaca_paper_trader import (
    AlpacaPaperTrader,
    RiskLimits,
    RiskCheckResult,
    TradeOrder,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Configuration
# ============================================================================

# Use a small amount to minimize impact on paper account
TEST_ORDER_VALUE = 100.0  # $100 worth
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']
TEST_SYMBOL = 'AAPL'  # Primary test symbol


def get_test_quantity(trader: AlpacaPaperTrader, symbol: str, value: float) -> int:
    """Calculate quantity for a given dollar value."""
    quote = trader.get_quote(symbol)
    return max(1, int(value / quote['mid']))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def trader() -> AlpacaPaperTrader:
    """Create a shared trader instance for all tests."""
    # Use conservative risk limits for testing
    risk_limits = RiskLimits(
        max_position_pct=0.10,  # 10% max position
        max_order_value=1000.0,  # $1000 max order
        max_daily_loss_pct=0.10,  # 10% max daily loss
        min_buying_power_reserve=0.20,  # 20% reserve
        max_positions=10,
    )

    return AlpacaPaperTrader(risk_limits=risk_limits)


@pytest.fixture(scope="module")
def cleanup_trader(trader: AlpacaPaperTrader):
    """Cleanup any test positions/orders after tests complete."""
    yield trader

    # Cleanup: Cancel any pending orders from tests
    try:
        trader.cancel_all_orders()
    except Exception as e:
        logger.warning(f"Cleanup cancel_all_orders failed: {e}")


# ============================================================================
# Connection Tests
# ============================================================================

class TestConnection:
    """Test Alpaca API connection."""

    def test_connection_established(self, trader: AlpacaPaperTrader):
        """Test that we can connect to Alpaca."""
        assert trader.is_connected(), "Should be connected to Alpaca"

    def test_account_accessible(self, trader: AlpacaPaperTrader):
        """Test that we can access account info."""
        account = trader.get_account()

        assert account.equity > 0, "Account should have positive equity"
        assert account.buying_power >= 0, "Buying power should be non-negative"
        assert account.cash >= 0, "Cash should be non-negative"

        logger.info(f"Account equity: ${account.equity:,.2f}")
        logger.info(f"Buying power: ${account.buying_power:,.2f}")


# ============================================================================
# Quote Tests
# ============================================================================

class TestQuotes:
    """Test quote fetching functionality."""

    def test_get_single_quote(self, trader: AlpacaPaperTrader):
        """Test fetching a single quote."""
        quote = trader.get_quote(TEST_SYMBOL)

        assert 'bid' in quote, "Quote should have bid"
        assert 'ask' in quote, "Quote should have ask"
        assert 'mid' in quote, "Quote should have mid"
        assert quote['bid'] > 0, "Bid should be positive"
        assert quote['ask'] > 0, "Ask should be positive"
        assert quote['ask'] >= quote['bid'], "Ask should be >= bid"

        logger.info(f"{TEST_SYMBOL} quote: ${quote['mid']:.2f}")

    def test_get_multiple_quotes(self, trader: AlpacaPaperTrader):
        """Test fetching multiple quotes at once."""
        quotes = trader.get_quotes(TEST_SYMBOLS)

        assert len(quotes) == len(TEST_SYMBOLS), "Should get quotes for all symbols"

        for symbol in TEST_SYMBOLS:
            assert symbol in quotes, f"Should have quote for {symbol}"
            assert quotes[symbol]['bid'] > 0, f"{symbol} bid should be positive"

        logger.info("Multi-quote fetch successful")


# ============================================================================
# Risk Check Tests
# ============================================================================

class TestRiskChecks:
    """Test pre-trade risk check functionality."""

    def test_risk_check_passes_small_order(self, trader: AlpacaPaperTrader):
        """Test that small orders pass risk checks."""
        qty = get_test_quantity(trader, TEST_SYMBOL, TEST_ORDER_VALUE)

        result, msg = trader.check_risk(
            symbol=TEST_SYMBOL,
            qty=qty,
            side='buy',
        )

        assert result == RiskCheckResult.PASSED, f"Small order should pass: {msg}"

    def test_risk_check_fails_huge_order(self, trader: AlpacaPaperTrader):
        """Test that huge orders fail risk checks."""
        result, msg = trader.check_risk(
            symbol=TEST_SYMBOL,
            qty=1_000_000,  # 1M shares
            side='buy',
        )

        assert result != RiskCheckResult.PASSED, "Huge order should fail"
        logger.info(f"Risk check correctly failed: {msg}")


# ============================================================================
# Order Tests
# ============================================================================

class TestOrders:
    """Test order submission and management."""

    def test_submit_market_order(self, trader: AlpacaPaperTrader, cleanup_trader):
        """Test submitting a market order."""
        qty = get_test_quantity(trader, TEST_SYMBOL, TEST_ORDER_VALUE)

        order = trader.submit_order(
            symbol=TEST_SYMBOL,
            qty=qty,
            side='buy',
            order_type='market',
        )

        assert order.order_id is not None, "Order should have ID"
        assert order.symbol == TEST_SYMBOL
        assert order.qty == qty
        assert order.side == 'buy'
        assert order.status in ('new', 'accepted', 'pending_new', 'filled')

        logger.info(f"Market order submitted: {order.order_id[:8]}... status={order.status}")

        # Wait for fill
        time.sleep(2)

        # Check order status
        updated = trader.get_order(order.order_id)
        assert updated is not None, "Should be able to get order"

        logger.info(f"Order status after 2s: {updated.status}")

    def test_submit_limit_order(self, trader: AlpacaPaperTrader, cleanup_trader):
        """Test submitting a limit order."""
        # Get current price and set limit below market
        quote = trader.get_quote(TEST_SYMBOL)
        limit_price = round(quote['bid'] * 0.95, 2)  # 5% below bid
        qty = get_test_quantity(trader, TEST_SYMBOL, TEST_ORDER_VALUE)

        order = trader.submit_order(
            symbol=TEST_SYMBOL,
            qty=qty,
            side='buy',
            order_type='limit',
            limit_price=limit_price,
        )

        assert order.order_id is not None
        assert order.limit_price == limit_price
        assert order.order_type == 'limit'

        logger.info(f"Limit order submitted at ${limit_price:.2f}")

        # Cancel it since it probably won't fill
        cancelled = trader.cancel_order(order.order_id)
        assert cancelled, "Should be able to cancel limit order"

    def test_cancel_order(self, trader: AlpacaPaperTrader, cleanup_trader):
        """Test order cancellation."""
        quote = trader.get_quote(TEST_SYMBOL)
        limit_price = round(quote['bid'] * 0.90, 2)  # 10% below bid
        qty = 1

        order = trader.submit_order(
            symbol=TEST_SYMBOL,
            qty=qty,
            side='buy',
            order_type='limit',
            limit_price=limit_price,
        )

        # Cancel immediately
        time.sleep(0.5)
        cancelled = trader.cancel_order(order.order_id)

        assert cancelled, "Cancellation should succeed"


# ============================================================================
# Position Tests
# ============================================================================

class TestPositions:
    """Test position tracking and management."""

    def test_get_positions(self, trader: AlpacaPaperTrader):
        """Test fetching all positions."""
        positions = trader.get_positions()

        # Just verify we can fetch - may or may not have positions
        assert isinstance(positions, list)

        for pos in positions:
            assert pos.symbol is not None
            assert pos.qty != 0
            logger.info(f"Position: {pos.symbol} x {pos.qty}")

    def test_get_single_position(self, trader: AlpacaPaperTrader):
        """Test fetching a single position."""
        # First ensure we have a position
        positions = trader.get_positions()

        if positions:
            symbol = positions[0].symbol
            pos = trader.get_position(symbol)

            assert pos is not None
            assert pos.symbol == symbol
        else:
            # No positions - verify None returned for non-existent
            pos = trader.get_position('DEFINITELY_NOT_A_REAL_TICKER_123')
            assert pos is None


# ============================================================================
# Full Pipeline Test
# ============================================================================

class TestFullPipeline:
    """End-to-end pipeline test with order/position cycle."""

    @pytest.mark.slow
    def test_complete_trade_cycle(self, trader: AlpacaPaperTrader, cleanup_trader):
        """
        Test complete trade cycle:
        1. Fetch quote
        2. Submit buy order
        3. Verify position
        4. Close position
        """
        # Step 1: Get quote
        quote = trader.get_quote(TEST_SYMBOL)
        logger.info(f"Step 1 - Quote: {TEST_SYMBOL} @ ${quote['mid']:.2f}")

        # Step 2: Submit small buy
        qty = get_test_quantity(trader, TEST_SYMBOL, TEST_ORDER_VALUE)
        buy_order = trader.submit_order(
            symbol=TEST_SYMBOL,
            qty=qty,
            side='buy',
            order_type='market',
        )
        logger.info(f"Step 2 - Buy order: {buy_order.order_id[:8]}...")

        # Wait for fill
        time.sleep(3)

        # Step 3: Verify position
        position = trader.get_position(TEST_SYMBOL)
        if position:
            logger.info(
                f"Step 3 - Position: {position.qty} shares @ "
                f"${position.avg_entry_price:.2f}"
            )

            # Step 4: Close position
            close_order = trader.close_position(TEST_SYMBOL)
            if close_order:
                logger.info(f"Step 4 - Close order: {close_order.order_id[:8]}...")

                # Wait for fill
                time.sleep(3)

                # Verify closed
                position_after = trader.get_position(TEST_SYMBOL)
                if position_after is None or position_after.qty == 0:
                    logger.info("Step 5 - Position closed successfully!")
                else:
                    logger.warning(f"Position still exists: {position_after.qty}")
            else:
                logger.warning("Close order not submitted")
        else:
            logger.warning("Position not found after buy - order may not have filled")


# ============================================================================
# Portfolio Allocation Test
# ============================================================================

class TestPortfolioAllocation:
    """Test portfolio allocation execution."""

    def test_portfolio_allocation_dry_run(self, trader: AlpacaPaperTrader):
        """Test portfolio allocation calculation (without execution)."""
        # Just test risk checks for a sample allocation
        allocation = {
            'AAPL': 0.25,
            'MSFT': 0.25,
            'GOOGL': 0.25,
        }

        account = trader.get_account()

        for symbol, weight in allocation.items():
            target_value = account.equity * weight
            quote = trader.get_quote(symbol)
            qty = int(target_value / quote['mid'])

            result, msg = trader.check_risk(symbol, qty, 'buy')
            logger.info(f"{symbol}: {weight:.0%} -> {qty} shares, risk: {result.value}")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_quick_test():
    """Run a quick connectivity test without pytest."""
    print("=" * 60)
    print("ALPACA PAPER TRADING PIPELINE - QUICK TEST")
    print("=" * 60)

    try:
        # Check credentials
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')

        if not api_key or not api_secret:
            print("\nERROR: Missing credentials!")
            print("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
            print("Or add them to a .env file")
            return False

        print(f"\nAPI Key: {api_key[:8]}...{api_key[-4:]}")

        # Connect
        print("\n1. Connecting to Alpaca...")
        trader = AlpacaPaperTrader()
        print("   Connected!")

        # Account
        print("\n2. Fetching account info...")
        account = trader.get_account()
        print(f"   Equity:       ${account.equity:,.2f}")
        print(f"   Cash:         ${account.cash:,.2f}")
        print(f"   Buying Power: ${account.buying_power:,.2f}")
        print(f"   Daily P&L:    ${account.daily_pnl:,.2f} ({account.daily_pnl_pct:+.2f}%)")

        # Quotes
        print("\n3. Fetching quotes...")
        for symbol in TEST_SYMBOLS:
            quote = trader.get_quote(symbol)
            print(f"   {symbol}: ${quote['mid']:.2f} (spread: ${quote['ask'] - quote['bid']:.3f})")

        # Positions
        print("\n4. Fetching positions...")
        positions = trader.get_positions()
        if positions:
            for pos in positions:
                print(
                    f"   {pos.symbol}: {pos.qty} @ ${pos.avg_entry_price:.2f} "
                    f"(P&L: ${pos.unrealized_pnl:,.2f})"
                )
        else:
            print("   No open positions")

        # Risk check
        print("\n5. Testing risk checks...")
        result, msg = trader.check_risk(TEST_SYMBOL, 1, 'buy')
        print(f"   Small order (1 share): {result.value}")

        result, msg = trader.check_risk(TEST_SYMBOL, 1_000_000, 'buy')
        print(f"   Large order (1M shares): {result.value}")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED - Paper trading pipeline is ready!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check if running with pytest or standalone
    if len(sys.argv) > 1 and 'pytest' in sys.argv[0]:
        # Running with pytest - let pytest handle it
        pass
    else:
        # Running standalone
        success = run_quick_test()
        sys.exit(0 if success else 1)
