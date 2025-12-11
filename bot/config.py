"""
Centralized Configuration for Trading Bot
==========================================
All configuration is read from environment variables.
No secrets are hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from bot.data_config import SecurityDataConfig


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"

    @classmethod
    def from_env(cls) -> 'AlpacaConfig':
        """Load from environment variables."""
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not api_key or not api_secret:
            raise ValueError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        return cls(api_key=api_key, api_secret=api_secret, base_url=base_url)


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str
    chat_id: str
    enabled: bool = True

    @classmethod
    def from_env(cls) -> Optional['TelegramConfig']:
        """Load from environment variables. Returns None if not configured."""
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        enabled = os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'

        if not bot_token or not chat_id:
            return None

        return cls(bot_token=bot_token, chat_id=chat_id, enabled=enabled)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 0.25  # Max 25% of portfolio in single position
    max_order_value: float = 50000.0  # Max $50k per order
    max_daily_loss_pct: float = 0.05  # Max 5% daily drawdown
    min_buying_power_reserve: float = 0.10  # Keep 10% cash reserve
    max_positions: int = 20  # Max concurrent positions

    # Kill switch settings
    kill_switch_daily_loss_pct: float = 0.03  # 3% daily loss triggers kill switch
    kill_switch_max_drawdown_pct: float = 0.10  # 10% drawdown from peak triggers kill switch
    kill_switch_enabled: bool = True  # Master enable for kill switch
    position_reconcile_on_start: bool = True  # Sync positions with Alpaca on startup

    @classmethod
    def from_env(cls) -> 'RiskConfig':
        """Load from environment variables with defaults."""
        return cls(
            max_position_pct=float(os.getenv('RISK_MAX_POSITION_PCT', '0.25')),
            max_order_value=float(os.getenv('RISK_MAX_ORDER_VALUE', '50000')),
            max_daily_loss_pct=float(os.getenv('RISK_MAX_DAILY_LOSS_PCT', '0.05')),
            min_buying_power_reserve=float(os.getenv('RISK_MIN_RESERVE', '0.10')),
            max_positions=int(os.getenv('RISK_MAX_POSITIONS', '20')),
            kill_switch_daily_loss_pct=float(os.getenv('RISK_KILL_DAILY_LOSS_PCT', '0.03')),
            kill_switch_max_drawdown_pct=float(os.getenv('RISK_KILL_MAX_DRAWDOWN_PCT', '0.10')),
            kill_switch_enabled=os.getenv('RISK_KILL_SWITCH_ENABLED', 'true').lower() == 'true',
            position_reconcile_on_start=os.getenv('RISK_RECONCILE_ON_START', 'true').lower() == 'true',
        )


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    symbols: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL'])
    timeframe: str = '1D'  # Daily bars
    lookback_days: int = 252  # ~1 year of trading days
    rebalance_threshold: float = 0.05  # 5% deviation triggers rebalance

    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Load from environment variables with defaults."""
        symbols_str = os.getenv('TRADING_SYMBOLS', 'AAPL,MSFT,GOOGL')
        symbols = [s.strip().upper() for s in symbols_str.split(',')]

        return cls(
            symbols=symbols,
            timeframe=os.getenv('TRADING_TIMEFRAME', '1D'),
            lookback_days=int(os.getenv('TRADING_LOOKBACK_DAYS', '252')),
            rebalance_threshold=float(os.getenv('TRADING_REBALANCE_THRESHOLD', '0.05')),
        )


@dataclass
class BotConfig:
    """Complete bot configuration."""
    alpaca: AlpacaConfig
    telegram: Optional[TelegramConfig]
    risk: RiskConfig
    trading: TradingConfig

    # Paths
    data_dir: Path = field(default_factory=lambda: Path('data'))
    logs_dir: Path = field(default_factory=lambda: Path('logs'))
    trades_db: Path = field(default_factory=lambda: Path('data/trades.db'))

    # Bot settings
    log_level: str = 'INFO'
    dry_run: bool = False  # If True, no real orders are submitted

    # Security data configuration (lazy loaded)
    _security_data_config: Optional['SecurityDataConfig'] = field(default=None, repr=False)

    @property
    def security_data(self) -> 'SecurityDataConfig':
        """Get security data configuration (lazy loaded)."""
        if self._security_data_config is None:
            try:
                from bot.data_config import SecurityDataConfig
            except ModuleNotFoundError:
                # Fallback for running as standalone script
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from bot.data_config import SecurityDataConfig
            self._security_data_config = SecurityDataConfig.from_env()
        return self._security_data_config

    @classmethod
    def from_env(cls) -> 'BotConfig':
        """Load complete configuration from environment."""
        base_dir = Path(os.getenv('BOT_BASE_DIR', '.'))

        return cls(
            alpaca=AlpacaConfig.from_env(),
            telegram=TelegramConfig.from_env(),
            risk=RiskConfig.from_env(),
            trading=TradingConfig.from_env(),
            data_dir=base_dir / os.getenv('BOT_DATA_DIR', 'data'),
            logs_dir=base_dir / os.getenv('BOT_LOGS_DIR', 'logs'),
            trades_db=base_dir / os.getenv('BOT_TRADES_DB', 'data/trades.db'),
            log_level=os.getenv('BOT_LOG_LEVEL', 'INFO'),
            dry_run=os.getenv('BOT_DRY_RUN', 'false').lower() == 'true',
        )

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.trades_db.parent.mkdir(parents=True, exist_ok=True)


def load_config() -> BotConfig:
    """
    Load configuration from environment variables.

    This is the main entry point for configuration.
    Call this at bot startup after loading .env file.

    Returns:
        BotConfig: Complete configuration object

    Raises:
        ValueError: If required configuration is missing
    """
    config = BotConfig.from_env()
    config.ensure_directories()
    return config


if __name__ == "__main__":
    # Test configuration loading
    from dotenv import load_dotenv
    load_dotenv()

    try:
        config = load_config()
        print("=" * 60)
        print("BOT CONFIGURATION")
        print("=" * 60)
        print(f"\n=== Alpaca ===")
        print(f"  API Key: {config.alpaca.api_key[:8]}...")
        print(f"  Base URL: {config.alpaca.base_url}")
        print(f"\n=== Telegram ===")
        print(f"  Status: {'Configured' if config.telegram else 'Not configured'}")
        print(f"\n=== Trading ===")
        print(f"  Symbols: {config.trading.symbols}")
        print(f"  Lookback: {config.trading.lookback_days} days")
        print(f"\n=== Risk ===")
        print(f"  Max position: {config.risk.max_position_pct:.0%}")
        print(f"  Kill switch: {config.risk.kill_switch_enabled}")
        print(f"\n=== Bot Settings ===")
        print(f"  Log level: {config.log_level}")
        print(f"  Dry run: {config.dry_run}")

        # Test security data config
        print(f"\n=== Security Data Config ===")
        sd = config.security_data
        print(f"  Universe: {sd.universe.universe_type.value} (max {sd.universe.max_universe_size})")
        print(f"  Selection: {sd.selection.method.value} -> {sd.selection.num_stocks_select} stocks")
        print(f"  Data categories enabled:")
        print(f"    - Fundamentals: {sd.fundamentals.enabled}")
        print(f"    - Technical: {sd.technical.enabled}")
        print(f"    - Risk metrics: {sd.risk.enabled}")
        print(f"    - Microstructure: {sd.microstructure.enabled}")
        print(f"    - Alternative: {sd.alternative.enabled}")
        print(f"    - Events: {sd.events.enabled}")
        print(f"  Total metrics: {sum(len(v) for v in sd.get_all_metrics().values())}")

    except ValueError as e:
        print(f"Configuration error: {e}")
