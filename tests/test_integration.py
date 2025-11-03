"""
Integration tests for end-to-end trading pipeline.
"""
import pytest
import asyncio
from datetime import datetime
from crypto_trading.main import Application


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline():
    """Test full pipeline from WebSocket to trade execution (mock)"""
    # This is a placeholder for integration testing
    # In practice, you'd mock the WebSocket to avoid live testing
    app = Application()

    # Verify components initialized
    assert app.db is not None
    assert app.signal_engine is not None
    assert app.risk_manager is not None
    assert app.paper_trader is not None

    # Verify initial state
    assert app.paper_trader.get_balance() == app.config['trading']['initial_balance']
    assert len(app.paper_trader.get_positions()) == 0


def test_configuration_loading():
    """Test configuration loads correctly"""
    app = Application()

    # Check config structure
    assert 'kraken' in app.config
    assert 'trading' in app.config
    assert 'strategy' in app.config
    assert 'risk' in app.config
    assert 'execution' in app.config
    assert 'database' in app.config

    # Check key values
    assert app.config['strategy']['ema_fast_period'] == 20
    assert app.config['strategy']['ema_slow_period'] == 50
    assert app.config['trading']['initial_balance'] == 100
