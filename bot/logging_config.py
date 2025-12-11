"""
Logging Configuration for Trading Bot
======================================
Provides structured logging with rotating file handlers.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


# Global logger cache
_loggers: dict = {}


class TradingFormatter(logging.Formatter):
    """Custom formatter for trading bot logs."""

    FORMATS = {
        logging.DEBUG: "%(asctime)s | DEBUG    | %(name)s | %(message)s",
        logging.INFO: "%(asctime)s | INFO     | %(name)s | %(message)s",
        logging.WARNING: "%(asctime)s | WARNING  | %(name)s | %(message)s",
        logging.ERROR: "%(asctime)s | ERROR    | %(name)s | %(message)s",
        logging.CRITICAL: "%(asctime)s | CRITICAL | %(name)s | %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logging(
    log_dir: Path = Path('logs'),
    log_level: str = 'INFO',
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up logging with rotating file handlers.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Also log to console
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Root logger configured for the trading bot
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    formatter = TradingFormatter()

    # File handler - rotating by size
    log_file = log_dir / 'trading_bot.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Error file handler - separate file for errors
    error_file = log_dir / 'trading_bot_errors.log'
    error_handler = RotatingFileHandler(
        error_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Log startup
    root_logger.info(f"Logging initialized: level={log_level}, dir={log_dir}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


class TradeLogger:
    """
    Specialized logger for trade events.

    Provides structured logging for trades with consistent format.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger('trades')

    def log_order_submitted(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        order_id: str,
        price: Optional[float] = None,
    ) -> None:
        """Log order submission."""
        price_str = f" @ ${price:.2f}" if price else ""
        self.logger.info(
            f"ORDER SUBMITTED | {side.upper()} {qty} {symbol} {order_type.upper()}"
            f"{price_str} | ID: {order_id[:12]}..."
        )

    def log_order_filled(
        self,
        symbol: str,
        side: str,
        qty: float,
        avg_price: float,
        order_id: str,
    ) -> None:
        """Log order fill."""
        self.logger.info(
            f"ORDER FILLED | {side.upper()} {qty} {symbol} @ ${avg_price:.2f} | "
            f"ID: {order_id[:12]}..."
        )

    def log_order_cancelled(self, symbol: str, order_id: str, reason: str = "") -> None:
        """Log order cancellation."""
        reason_str = f" | Reason: {reason}" if reason else ""
        self.logger.info(f"ORDER CANCELLED | {symbol} | ID: {order_id[:12]}...{reason_str}")

    def log_position_opened(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        position_value: float,
    ) -> None:
        """Log new position."""
        self.logger.info(
            f"POSITION OPENED | {symbol} | {qty} shares @ ${entry_price:.2f} | "
            f"Value: ${position_value:,.2f}"
        )

    def log_position_closed(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
    ) -> None:
        """Log position closure."""
        self.logger.info(
            f"POSITION CLOSED | {symbol} | {qty} shares | "
            f"Entry: ${entry_price:.2f} -> Exit: ${exit_price:.2f} | "
            f"PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)"
        )

    def log_risk_check_failed(
        self,
        symbol: str,
        reason: str,
    ) -> None:
        """Log failed risk check."""
        self.logger.warning(f"RISK CHECK FAILED | {symbol} | {reason}")

    def log_connection_status(self, connected: bool, details: str = "") -> None:
        """Log connection status."""
        status = "CONNECTED" if connected else "DISCONNECTED"
        details_str = f" | {details}" if details else ""
        level = logging.INFO if connected else logging.WARNING
        self.logger.log(level, f"CONNECTION | {status}{details_str}")

    def log_account_snapshot(
        self,
        equity: float,
        cash: float,
        daily_pnl: float,
        daily_pnl_pct: float,
    ) -> None:
        """Log account snapshot."""
        self.logger.info(
            f"ACCOUNT | Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | "
            f"Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)"
        )


if __name__ == "__main__":
    # Test logging setup
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        setup_logging(Path(tmpdir), 'DEBUG')

        logger = get_logger('test')
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        trade_logger = TradeLogger()
        trade_logger.log_order_submitted('AAPL', 'buy', 10, 'market', 'abc123def456')
        trade_logger.log_order_filled('AAPL', 'buy', 10, 150.50, 'abc123def456')
        trade_logger.log_position_closed('AAPL', 10, 150.00, 155.00, 50.00, 3.33)

        print(f"\nLog files created in {tmpdir}")
