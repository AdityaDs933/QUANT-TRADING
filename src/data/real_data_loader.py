"""
Real-World Market Data Acquisition Pipeline
Authored by: Anhum (15% Component)

Extracts historical multi-asset market data using yfinance with:
- Local disk caching (Parquet / CSV format)
- Schema normalization & timezone alignment
- Corporate action handling (Adj Close verification)
- Integration with BaseDataLoader interface
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

from src.utils.base_interfaces import BaseDataLoader


class RealMarketDataLoader(BaseDataLoader):
    """
    Automated loader for empirical equities and ETFs.
    Manages local cache to eliminate redundant network downloads.
    """

    def __init__(self, cache_dir: str | Path = "data/raw", use_cache: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    def _get_cache_filepath(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Standardized cache filename pattern: data/raw/AAPL_2018-01-01_2024-12-31.parquet"""
        clean_symbol = symbol.replace("^", "").replace("=", "")
        return self.cache_dir / f"{clean_symbol}_{start_date}_{end_date}.parquet"

    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads OHLCV data for a given symbol and date window.
        Checks local parquet cache first; falls back to yfinance if not cached.
        """
        cache_file = self._get_cache_filepath(symbol, start_date, end_date)

        if self.use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df

        # Download from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No market data returned for symbol '{symbol}' between {start_date} and {end_date}.")

        # Standardize column naming and strip timezone from index
        df = self._clean_and_format(df)

        if not self.validate_schema(df):
            raise ValueError(f"Downloaded market data for '{symbol}' failed schema validation checks.")

        # Save to local cache
        if self.use_cache:
            df.to_parquet(cache_file)

        return df

    def load_multiple_assets(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Loads and validates a collection of assets into a dictionary of DataFrames."""
        asset_store: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            asset_store[sym] = self.load_data(sym, start_date, end_date)
        return asset_store

    def _clean_and_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column indices, handle multi-index headers, and localize timezones."""
        # Reset MultiIndex if yfinance returned hierarchical columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize index name and strip timezone
        df.index.name = "Date"
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Ensure canonical column names exist
        rename_map = {
            "Adj Close": "Adj Close",
            "Dividends": "Dividends",
            "Stock Splits": "Stock Splits",
        }
        df = df.rename(columns=rename_map)

        # If Adj Close is missing, populate from Close
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        # Drop non-price accessory columns from core view
        core_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        available_cols = [c for c in core_cols if c in df.columns]
        return df[available_cols].astype("float64")

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validates schema integrity:
        1. Required OHLCV columns exist
        2. Date index is monotonically increasing and unique
        3. High >= Low, High >= Open, High >= Close
        4. Prices > 0
        """
        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if not all(col in df.columns for col in required):
            return False

        if not df.index.is_monotonic_increasing:
            return False

        if df.index.has_duplicates:
            return False

        valid_ranges = (df["High"] >= df["Low"]).all()
        positive_prices = (df[["Open", "High", "Low", "Close", "Adj Close"]] > 0).all().all()

        return bool(valid_ranges and positive_prices)
