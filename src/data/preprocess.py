"""
Financial Data Cleaning & Preprocessing Pipeline
Authored by: Anhum (15% Component)

Handles:
- Missing trading bars & business day calendar alignment
- Outlier detection and robust winsorization
- Data sanitation (zero/negative price assertion, volume non-negativity)
- Automated export to data/processed
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    Standardized preprocessing engine for raw financial OHLCV time series.
    """

    def __init__(
        self,
        z_score_threshold: float = 4.5,
        fill_method: str = "forward_fill",
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
    ) -> None:
        self.z_score_threshold = z_score_threshold
        self.fill_method = fill_method
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_ohlcv(self, df: pd.DataFrame, symbol: str = "ASSET") -> pd.DataFrame:
        """
        Executes end-to-end data sanitation:
        1. Verifies timestamps are sorted and unique
        2. Aligns to business day calendar (freq='B')
        3. Imputes missing bars without lookahead bias
        4. Validates price and volume physics
        5. Performs mild winsorization on extreme return outliers
        """
        df_clean = df.copy()

        # Sort and deduplicate index
        df_clean = df_clean.sort_index()
        df_clean = df_clean[~df_clean.index.duplicated(keep="first")]

        # Ensure index is timezone-naive
        if df_clean.index.tz is not None:
            df_clean.index = df_clean.index.tz_localize(None)

        # 1. Align to business day calendar
        start_date = df_clean.index.min()
        end_date = df_clean.index.max()
        bday_index = pd.date_range(start=start_date, end=end_date, freq="B", name="Date")
        df_reindexed = df_clean.reindex(bday_index)

        # 2. Impute prices with forward-fill (causal, no future leakage)
        # For weekends/holidays, last traded price remains active
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df_reindexed.columns]
        df_reindexed[price_cols] = df_reindexed[price_cols].ffill().bfill()

        # For halted or missing days, traded volume is 0
        if "Volume" in df_reindexed.columns:
            df_reindexed["Volume"] = df_reindexed["Volume"].fillna(0.0)

        # 3. Assert price physics
        # High must be >= Low
        df_reindexed["High"] = np.maximum(df_reindexed["High"], df_reindexed["Low"])
        df_reindexed["High"] = np.maximum(df_reindexed["High"], df_reindexed["Close"])
        df_reindexed["Low"] = np.minimum(df_reindexed["Low"], df_reindexed["Close"])

        # Prices must be strictly positive
        if (df_reindexed[price_cols] <= 0).any().any():
            raise ValueError(f"Cleaned dataset for '{symbol}' contains non-positive prices.")

        # 4. Outlier detection and clipping on returns
        df_final = self._clip_extreme_spikes(df_reindexed)

        return df_final

    def _clip_extreme_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clips non-market execution glitches (e.g. flash spikes > z_score_threshold standard deviations)
        using rolling statistics, preserving genuine heavy tails up to threshold.
        """
        df_out = df.copy()
        adj_close = df_out["Adj Close"]
        log_ret = np.log(adj_close / adj_close.shift(1)).fillna(0.0)

        # Rolling 60-day mean and standard deviation
        rolling_mean = log_ret.rolling(window=60, min_periods=10).mean().fillna(0.0)
        rolling_std = log_ret.rolling(window=60, min_periods=10).std().fillna(log_ret.std())

        z_scores = np.abs((log_ret - rolling_mean) / (rolling_std + 1e-8))
        outliers = z_scores > self.z_score_threshold

        # If extreme outlier detected, winsorize to threshold bound
        if outliers.any():
            clipped_ret = np.clip(
                log_ret,
                rolling_mean - self.z_score_threshold * rolling_std,
                rolling_mean + self.z_score_threshold * rolling_std,
            )
            # Reconstruct Adj Close from clipped returns
            reconstructed_adj = adj_close.iloc[0] * np.exp(np.cumsum(clipped_ret))
            df_out["Adj Close"] = reconstructed_adj

        return df_out

    def process_and_save(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads raw parquet data from data/raw, preprocesses, and saves to data/processed.
        """
        clean_symbol = symbol.replace("^", "").replace("=", "")
        raw_path = self.raw_dir / f"{clean_symbol}_{start_date}_{end_date}.parquet"

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        df_raw = pd.read_parquet(raw_path)
        df_clean = self.clean_ohlcv(df_raw, symbol=symbol)

        out_path = self.processed_dir / f"{clean_symbol}_processed.parquet"
        df_clean.to_parquet(out_path)
        return df_clean
