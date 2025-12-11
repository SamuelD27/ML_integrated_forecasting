"""
Tests for Execution Module (Phase 9)
====================================
Tests for paper broker and order execution.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.paper_broker import (
    PaperBroker,
    Order,
    Position,
    OrderStatus,
    OrderType,
    create_broker,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def broker():
    """Create a paper broker with default settings."""
    return PaperBroker(initial_capital=100000, use_alpaca=False)


@pytest.fixture
def broker_with_prices(broker):
    """Broker with pre-set prices."""
    broker.set_price('AAPL', 150.0)
    broker.set_price('GOOGL', 140.0)
    broker.set_price('MSFT', 380.0)
    return broker


# =============================================================================
# Test Order
# =============================================================================

class TestOrder:
    """Tests for Order dataclass."""

    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            order_id='ORD_001',
            ticker='AAPL',
            side='buy',
            quantity=100,
        )

        assert order.ticker == 'AAPL'
        assert order.side == 'buy'
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING

    def test_order_is_filled(self):
        """Test is_filled property."""
        order = Order(
            order_id='ORD_001',
            ticker='AAPL',
            side='buy',
            quantity=100,
            status=OrderStatus.FILLED,
        )

        assert order.is_filled == True

    def test_order_fill_value(self):
        """Test fill_value property."""
        order = Order(
            order_id='ORD_001',
            ticker='AAPL',
            side='buy',
            quantity=100,
            filled_quantity=100,
            filled_avg_price=150.0,
        )

        assert order.fill_value == 15000.0


# =============================================================================
# Test Position
# =============================================================================

class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            ticker='AAPL',
            quantity=100,
            avg_cost=150.0,
        )

        assert pos.ticker == 'AAPL'
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0

    def test_position_update_price(self):
        """Test position price update."""
        pos = Position(
            ticker='AAPL',
            quantity=100,
            avg_cost=150.0,
        )

        pos.update_price(160.0)

        assert pos.current_price == 160.0
        assert pos.market_value == 16000.0
        assert pos.unrealized_pnl == 1000.0  # 100 * (160 - 150)
        assert pos.unrealized_pnl_pct == pytest.approx(0.0667, rel=0.01)

    def test_position_negative_pnl(self):
        """Test position with negative P&L."""
        pos = Position(
            ticker='AAPL',
            quantity=100,
            avg_cost=150.0,
        )

        pos.update_price(140.0)

        assert pos.unrealized_pnl == -1000.0
        assert pos.unrealized_pnl_pct < 0


# =============================================================================
# Test Paper Broker
# =============================================================================

class TestPaperBroker:
    """Tests for PaperBroker."""

    def test_broker_initialization(self, broker):
        """Test broker initialization."""
        assert broker.initial_capital == 100000
        assert broker.cash == 100000
        assert len(broker.positions) == 0

    def test_submit_buy_order(self, broker_with_prices):
        """Test submitting a buy order."""
        broker = broker_with_prices

        order = broker.submit_order('AAPL', 'buy', 100)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.filled_avg_price > 0

        # Check position created
        assert 'AAPL' in broker.positions
        assert broker.positions['AAPL'].quantity == 100

        # Check cash reduced
        assert broker.cash < 100000

    def test_submit_sell_order(self, broker_with_prices):
        """Test submitting a sell order."""
        broker = broker_with_prices

        # First buy
        broker.submit_order('AAPL', 'buy', 100)

        # Then sell
        order = broker.submit_order('AAPL', 'sell', 50)

        assert order.status == OrderStatus.FILLED
        assert broker.positions['AAPL'].quantity == 50

    def test_sell_entire_position(self, broker_with_prices):
        """Test selling entire position."""
        broker = broker_with_prices

        # Buy
        broker.submit_order('AAPL', 'buy', 100)

        # Sell all
        broker.submit_order('AAPL', 'sell', 100)

        # Position should be closed
        assert 'AAPL' not in broker.positions

    def test_insufficient_cash(self, broker_with_prices):
        """Test order with insufficient cash."""
        broker = broker_with_prices
        broker.cash = 1000  # Very little cash

        # Try to buy expensive position
        order = broker.submit_order('GOOGL', 'buy', 100)  # $14,000+ needed

        # Should either reject or fill partial
        if order.status == OrderStatus.FILLED:
            assert order.filled_quantity < 100
        else:
            assert order.status == OrderStatus.REJECTED

    def test_get_portfolio_value(self, broker_with_prices):
        """Test portfolio value calculation."""
        broker = broker_with_prices

        # Buy some stock
        broker.submit_order('AAPL', 'buy', 100)

        # Update price
        broker.update_prices({'AAPL': 160.0})

        value = broker.get_portfolio_value()

        # Should be cash + position value
        expected = broker.cash + 100 * 160.0
        assert value == pytest.approx(expected, rel=0.01)

    def test_get_account_summary(self, broker_with_prices):
        """Test account summary."""
        broker = broker_with_prices

        broker.submit_order('AAPL', 'buy', 100)
        broker.update_prices({'AAPL': 160.0})

        summary = broker.get_account_summary()

        assert 'portfolio_value' in summary
        assert 'cash' in summary
        assert 'unrealized_pnl' in summary
        assert 'n_positions' in summary
        assert summary['n_positions'] == 1

    def test_close_position(self, broker_with_prices):
        """Test close_position method."""
        broker = broker_with_prices

        broker.submit_order('AAPL', 'buy', 100)

        order = broker.close_position('AAPL')

        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert 'AAPL' not in broker.positions

    def test_close_all_positions(self, broker_with_prices):
        """Test close_all_positions method."""
        broker = broker_with_prices

        # Buy multiple
        broker.submit_order('AAPL', 'buy', 50)
        broker.submit_order('GOOGL', 'buy', 30)

        orders = broker.close_all_positions()

        assert len(orders) == 2
        assert len(broker.positions) == 0

    def test_reset_broker(self, broker_with_prices):
        """Test broker reset."""
        broker = broker_with_prices

        broker.submit_order('AAPL', 'buy', 100)
        broker.reset()

        assert broker.cash == 100000
        assert len(broker.positions) == 0
        assert len(broker.orders) == 0

    def test_order_rejected_no_price(self, broker):
        """Test order rejected when no price available."""
        order = broker.submit_order('UNKNOWN', 'buy', 100)

        assert order.status == OrderStatus.REJECTED
        assert 'error' in order.metadata

    def test_multiple_buys_average_cost(self, broker_with_prices):
        """Test average cost calculation on multiple buys."""
        broker = broker_with_prices

        # First buy at $150
        broker.set_price('AAPL', 150.0)
        broker.submit_order('AAPL', 'buy', 100)

        # Second buy at $160
        broker.set_price('AAPL', 160.0)
        broker.submit_order('AAPL', 'buy', 100)

        pos = broker.positions['AAPL']
        assert pos.quantity == 200
        # Average cost should be around $155 (with slippage)
        assert pos.avg_cost > 150.0
        assert pos.avg_cost < 165.0


# =============================================================================
# Test Factory Function
# =============================================================================

class TestCreateBroker:
    """Tests for create_broker factory function."""

    def test_create_simulated_broker(self):
        """Test creating simulated broker."""
        broker = create_broker(use_alpaca=False, initial_capital=50000)

        assert isinstance(broker, PaperBroker)
        assert broker.initial_capital == 50000
        assert not broker.use_alpaca


# =============================================================================
# Integration Tests
# =============================================================================

class TestBrokerIntegration:
    """Integration tests for paper broker."""

    def test_full_trading_cycle(self, broker_with_prices):
        """Test full buy-hold-sell cycle."""
        broker = broker_with_prices
        initial_cash = broker.cash

        # Buy
        buy_order = broker.submit_order('AAPL', 'buy', 100)
        assert buy_order.is_filled

        # Hold - price goes up
        broker.set_price('AAPL', 170.0)
        broker.update_prices({'AAPL': 170.0})

        # Check unrealized P&L
        pos = broker.positions['AAPL']
        assert pos.unrealized_pnl > 0

        # Sell
        sell_order = broker.close_position('AAPL')
        assert sell_order.is_filled

        # Check realized P&L
        final_cash = broker.cash
        assert final_cash > initial_cash  # Made profit

    def test_multiple_positions(self, broker_with_prices):
        """Test managing multiple positions."""
        broker = broker_with_prices

        # Build portfolio
        broker.submit_order('AAPL', 'buy', 50)
        broker.submit_order('GOOGL', 'buy', 30)
        broker.submit_order('MSFT', 'buy', 20)

        assert len(broker.positions) == 3

        # Update all prices
        broker.update_prices({
            'AAPL': 160.0,
            'GOOGL': 150.0,
            'MSFT': 400.0,
        })

        # Check portfolio
        summary = broker.get_account_summary()
        assert summary['n_positions'] == 3
        assert summary['portfolio_value'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
