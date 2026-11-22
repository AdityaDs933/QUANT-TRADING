"""
Configuration Parser Module
Loads and validates settings from config.yaml with typed dictionary access.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigParser:
    """Singleton-style or instantiable config loader with fallback validation."""

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads and parses the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path.resolve()}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                cfg = yaml.safe_load(f)
                return cfg if cfg is not None else {}
            except yaml.YAMLError as exc:
                raise ValueError(f"Error parsing YAML config: {exc}") from exc

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Safely retrieves nested dictionary values.
        Example: config.get('backtest', 'initial_capital', default=100000.0)
        """
        curr = self.config
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr

    def __repr__(self) -> str:
        return f"ConfigParser(config_path='{self.config_path}', sections={list(self.config.keys())})"
