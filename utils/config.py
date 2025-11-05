"""
Configuration Management
=========================
Centralized configuration loading and management for the ML trading system.

Usage:
    >>> from utils.config import load_config, get_config
    >>>
    >>> # Load all configs
    >>> config = load_config()
    >>> print(config.data.market_data.providers)
    >>>
    >>> # Access specific sections
    >>> batch_size = config.training.training.batch_size
    >>> risk_aversion = config.portfolio.optimization.black_litterman.risk_aversion
    >>>
    >>> # Or use singleton pattern
    >>> cfg = get_config()
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """
    Dictionary that allows attribute access.

    Example:
        >>> d = ConfigDict({'a': 1, 'b': {'c': 2}})
        >>> d.a  # 1
        >>> d.b.c  # 2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


@dataclass
class Config:
    """
    Application configuration container.

    Attributes:
        data: Data configuration (fetching, caching, features)
        model: Model architecture configuration
        training: Training hyperparameters
        portfolio: Portfolio optimization configuration
    """
    data: ConfigDict = field(default_factory=ConfigDict)
    model: ConfigDict = field(default_factory=ConfigDict)
    training: ConfigDict = field(default_factory=ConfigDict)
    portfolio: ConfigDict = field(default_factory=ConfigDict)

    @classmethod
    def load(
        cls,
        config_dir: Optional[Path] = None,
        override: Optional[Dict[str, Any]] = None
    ) -> "Config":
        """
        Load all configuration files.

        Args:
            config_dir: Directory containing config files (default: configs/)
            override: Dictionary to override specific config values

        Returns:
            Config object with all configurations loaded

        Example:
            >>> config = Config.load()
            >>> config = Config.load(override={'training.batch_size': 128})
        """
        if config_dir is None:
            # Default to configs/ in project root
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "configs"

        config_dir = Path(config_dir)

        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        # Load each config file
        configs = {}
        config_files = ["data.yaml", "model.yaml", "training.yaml", "portfolio.yaml"]

        for config_file in config_files:
            config_path = config_dir / config_file
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                continue

            try:
                with open(config_path) as f:
                    content = f.read()

                    # Substitute environment variables (${VAR_NAME})
                    content = _substitute_env_vars(content)

                    # Parse YAML
                    config_data = yaml.safe_load(content)

                # Store with key matching filename (e.g., "data" for data.yaml)
                key = config_file.replace(".yaml", "")
                configs[key] = ConfigDict(config_data) if config_data else ConfigDict()

                logger.debug(f"Loaded config: {config_file}")

            except Exception as e:
                logger.error(f"Error loading {config_file}: {e}")
                configs[key] = ConfigDict()

        # Create Config object
        config = cls(
            data=configs.get("data", ConfigDict()),
            model=configs.get("model", ConfigDict()),
            training=configs.get("training", ConfigDict()),
            portfolio=configs.get("portfolio", ConfigDict())
        )

        # Apply overrides
        if override:
            config = _apply_overrides(config, override)

        return config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path.

        Args:
            key_path: Dot-separated path (e.g., "training.batch_size")
            default: Default value if key not found

        Returns:
            Config value or default

        Example:
            >>> config.get("training.training.batch_size", 128)
        """
        keys = key_path.split(".")
        value = self

        for key in keys:
            if isinstance(value, (Config, ConfigDict, dict)):
                if isinstance(value, Config):
                    value = getattr(value, key, None)
                else:
                    value = value.get(key, None)

                if value is None:
                    return default
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "data": dict(self.data),
            "model": dict(self.model),
            "training": dict(self.training),
            "portfolio": dict(self.portfolio)
        }


def _substitute_env_vars(content: str) -> str:
    """
    Substitute environment variables in config content.

    Supports syntax: ${VAR_NAME} or ${VAR_NAME:default_value}

    Args:
        content: YAML content string

    Returns:
        Content with environment variables substituted
    """
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replace(match):
        var_name = match.group(1)
        default_value = match.group(2)

        value = os.getenv(var_name)

        if value is None:
            if default_value is not None:
                return default_value
            else:
                logger.warning(
                    f"Environment variable {var_name} not set and no default provided"
                )
                return match.group(0)  # Keep original

        return value

    return re.sub(pattern, replace, content)


def _apply_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    """
    Apply override dictionary to config.

    Args:
        config: Base config
        overrides: Dictionary with dot-separated keys

    Returns:
        Config with overrides applied

    Example:
        >>> overrides = {"training.batch_size": 256}
        >>> config = _apply_overrides(config, overrides)
    """
    for key_path, value in overrides.items():
        keys = key_path.split(".")

        # Navigate to parent
        obj = config
        for key in keys[:-1]:
            if isinstance(obj, Config):
                obj = getattr(obj, key)
            else:
                obj = obj[key]

        # Set final value
        final_key = keys[-1]
        if isinstance(obj, Config):
            setattr(obj, final_key, value)
        else:
            obj[final_key] = value

        logger.info(f"Override applied: {key_path} = {value}")

    return config


# Singleton instance
_global_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance (singleton pattern).

    Args:
        reload: Force reload from files

    Returns:
        Global Config instance

    Example:
        >>> from utils.config import get_config
        >>> config = get_config()
        >>> batch_size = config.training.training.batch_size
    """
    global _global_config

    if _global_config is None or reload:
        _global_config = Config.load()

    return _global_config


def load_config(
    config_dir: Optional[Path] = None,
    override: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration from files.

    Convenience wrapper around Config.load().

    Args:
        config_dir: Directory containing config files
        override: Dictionary to override specific config values

    Returns:
        Config object

    Example:
        >>> from utils.config import load_config
        >>> config = load_config()
        >>> config = load_config(override={'training.batch_size': 128})
    """
    return Config.load(config_dir=config_dir, override=override)


def save_config(config: Config, output_path: Path):
    """
    Save configuration to YAML files.

    Args:
        config: Config object to save
        output_path: Directory to save config files

    Example:
        >>> save_config(config, Path("configs_backup"))
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()

    for section, data in config_dict.items():
        output_file = output_path / f"{section}.yaml"

        with open(output_file, 'w') as f:
            yaml.dump(dict(data), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config: {output_file}")


# Convenience functions for common access patterns
def get_data_config() -> ConfigDict:
    """Get data configuration."""
    return get_config().data


def get_model_config(model_type: str = "ensemble") -> ConfigDict:
    """
    Get model configuration for specific model type.

    Args:
        model_type: Model type ("ensemble", "tft", "hybrid")

    Returns:
        Model config dictionary
    """
    return get_config().model.get(model_type, ConfigDict())


def get_training_config() -> ConfigDict:
    """Get training configuration."""
    return get_config().training


def get_portfolio_config() -> ConfigDict:
    """Get portfolio configuration."""
    return get_config().portfolio
