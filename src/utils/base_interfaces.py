"""
Abstract Base Classes for System Pipeline
Enforces clean, modular contracts for Data Loaders, Models, and Strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import pandas as pd


class BaseDataLoader(ABC):
    """Abstract interface for historical and synthetic market data loaders."""

    @abstractmethod
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch or generate raw OHLCV market data."""
        pass

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validates that OHLCV columns exist and timestamps are sorted and unique."""
        pass


class BaseQuantModel(ABC):
    """Abstract interface for predictive machine learning models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseQuantModel":
        """Fits model parameters on training features and targets."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Outputs directional classifications or regression return forecasts."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Outputs calibrated class probabilities."""
        pass


class BaseStrategy(ABC):
    """Abstract interface for quantitative trading strategies."""

    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame, predictions: pd.Series) -> pd.Series:
        """Translates market observations and ML predictions into discrete positions (-1, 0, 1)."""
        pass

    @abstractmethod
    def calculate_positions(self, signals: pd.Series, **kwargs: Any) -> pd.Series:
        """Computes sized capital allocations or share quantities."""
        pass
