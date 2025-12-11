"""
Central Logger Configuration
=============================
Standardized logging for the trading pipeline.

All pipeline modules should use this logger for consistent:
- Start/end of main functions
- Number of items processed
- Summary statistics of outputs
- Exceptions with full trace

Usage:
    >>> from utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting universe build...")
    >>> logger.info(f"Processed {n_stocks} stocks")
    >>> logger.exception("Failed to fetch data")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


# Default configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_DIR = Path('logs')
DEFAULT_LOG_FILE = 'pipeline.log'
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# Module-level flag to track if logging is configured
_logging_configured = False


def configure_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None,
    console: bool = True,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> None:
    """
    Configure root logger with file and console handlers.

    Should be called once at application startup.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Log file name (default: pipeline.log)
        log_dir: Directory for log files (default: logs/)
        console: Whether to log to console
        format_string: Custom format string
        date_format: Custom date format

    Example:
        >>> from utils.logger import configure_logging
        >>> configure_logging(level=logging.DEBUG, console=True)
    """
    global _logging_configured

    if _logging_configured:
        return

    # Use defaults if not specified
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_file = log_file or DEFAULT_LOG_FILE
    format_string = format_string or DEFAULT_LOG_FORMAT
    date_format = date_format or DEFAULT_DATE_FORMAT

    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add file handler with rotation
    file_path = log_dir / log_file
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    _logging_configured = True

    # Log startup
    root_logger.info(f"Logging configured: level={logging.getLevelName(level)}, file={file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Automatically configures logging if not already done.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Details here")
        >>> logger.warning("Something looks off")
        >>> logger.error("Operation failed")
        >>> logger.exception("Exception occurred")  # Includes traceback
    """
    # Auto-configure if not done
    if not _logging_configured:
        configure_logging()

    return logging.getLogger(name)


class PipelineLogger:
    """
    Enhanced logger for pipeline phases with structured logging.

    Provides convenience methods for common logging patterns:
    - Phase start/end with timing
    - Progress tracking
    - Summary statistics

    Example:
        >>> logger = PipelineLogger('universe_builder')
        >>> logger.phase_start("Building universe")
        >>> logger.progress(50, 100, "Processing stocks")
        >>> logger.stats({'processed': 80, 'filtered': 20, 'final': 60})
        >>> logger.phase_end("Universe built")
    """

    def __init__(self, name: str):
        """Initialize with module name."""
        self._logger = get_logger(name)
        self._phase_start_time: Optional[datetime] = None
        self._phase_name: Optional[str] = None

    def info(self, msg: str, *args, **kwargs):
        """Log info level message."""
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug level message."""
        self._logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning level message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error level message."""
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Log exception with full traceback."""
        self._logger.exception(msg, *args, **kwargs)

    def phase_start(self, phase_name: str, **context) -> None:
        """
        Log phase start with context.

        Args:
            phase_name: Name of the phase (e.g., "Universe Building")
            **context: Additional context to log
        """
        self._phase_start_time = datetime.now()
        self._phase_name = phase_name

        context_str = ', '.join(f"{k}={v}" for k, v in context.items())
        if context_str:
            self._logger.info(f"[START] {phase_name} | {context_str}")
        else:
            self._logger.info(f"[START] {phase_name}")

    def phase_end(self, message: Optional[str] = None, **stats) -> None:
        """
        Log phase end with timing and statistics.

        Args:
            message: Optional end message
            **stats: Statistics to include in log
        """
        duration = None
        if self._phase_start_time:
            duration = (datetime.now() - self._phase_start_time).total_seconds()

        phase_name = self._phase_name or "Phase"
        msg = message or f"{phase_name} complete"

        parts = [f"[END] {msg}"]
        if duration is not None:
            parts.append(f"duration={duration:.2f}s")
        for key, value in stats.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")

        self._logger.info(" | ".join(parts))

        self._phase_start_time = None
        self._phase_name = None

    def progress(self, current: int, total: int, message: str = "") -> None:
        """
        Log progress update.

        Args:
            current: Current item number
            total: Total number of items
            message: Optional message
        """
        pct = (current / total * 100) if total > 0 else 0
        if message:
            self._logger.info(f"[PROGRESS] {message}: {current}/{total} ({pct:.1f}%)")
        else:
            self._logger.info(f"[PROGRESS] {current}/{total} ({pct:.1f}%)")

    def stats(self, statistics: dict) -> None:
        """
        Log summary statistics.

        Args:
            statistics: Dict of statistic name -> value
        """
        stats_str = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in statistics.items()
        )
        self._logger.info(f"[STATS] {stats_str}")

    def items_processed(
        self,
        n_input: int,
        n_output: int,
        item_type: str = "items"
    ) -> None:
        """
        Log items processed count.

        Args:
            n_input: Number of input items
            n_output: Number of output items
            item_type: Type of items (e.g., "stocks", "signals")
        """
        filter_rate = (1 - n_output / n_input) * 100 if n_input > 0 else 0
        self._logger.info(
            f"[PROCESSED] {item_type}: {n_input} input -> {n_output} output "
            f"(filtered {filter_rate:.1f}%)"
        )


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function entry and exit.

    Args:
        logger: Logger to use (default: get from function module)

    Example:
        >>> @log_function_call()
        ... def my_function(arg1, arg2):
        ...     return arg1 + arg2
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"[CALL] {func_name}(args={len(args)}, kwargs={list(kwargs.keys())})")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"[RETURN] {func_name} -> {type(result).__name__}")
                return result
            except Exception as e:
                logger.exception(f"[ERROR] {func_name} raised {type(e).__name__}: {e}")
                raise

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# Convenience function for quick setup
def setup_pipeline_logging(
    level: str = 'INFO',
    log_file: str = 'pipeline.log',
    console: bool = True
) -> None:
    """
    Quick setup for pipeline logging.

    Args:
        level: Log level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Log file name
        console: Whether to log to console
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    numeric_level = level_map.get(level.upper(), logging.INFO)
    configure_logging(level=numeric_level, log_file=log_file, console=console)


if __name__ == "__main__":
    # Test the logger
    setup_pipeline_logging(level='DEBUG')

    logger = get_logger(__name__)
    logger.info("Testing logger...")
    logger.debug("Debug message")
    logger.warning("Warning message")

    # Test PipelineLogger
    pl = PipelineLogger('test_phase')
    pl.phase_start("Test Phase", as_of_date="2024-01-15")
    pl.progress(50, 100, "Processing items")
    pl.stats({'processed': 100, 'filtered': 20, 'avg_score': 0.75})
    pl.items_processed(100, 80, "stocks")
    pl.phase_end(n_final=80, duration_sec=1.5)

    print("\nLogger test complete!")
