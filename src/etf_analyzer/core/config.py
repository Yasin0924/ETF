"""YAML configuration loader with dot-notation access."""

from pathlib import Path
from typing import Any

import yaml

_SENTINEL = object()


def load_config(path: str) -> dict:
    """Load YAML config file. Raises FileNotFoundError if missing."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_setting(config: dict, key: str, default: Any = _SENTINEL) -> Any:
    """Access nested config with dot notation: 'backtest.initial_capital'.

    Args:
        config: Loaded config dict.
        key: Dot-separated key path.
        default: Default value if key missing. Raises KeyError if not provided.
    """
    parts = key.split(".")
    current = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif default is not _SENTINEL:
            return default
        else:
            raise KeyError(f"Config key not found: {key}")
    return current
