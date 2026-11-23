"""
Financial Time-Series Inspection and Statistical Summary Utilities
Authored by: Tanveer (15% Component)

Verifies data continuity, detects missing trading days / holidays,
and computes stylized statistical comparisons between real and synthetic assets.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.finance_math import calculate_log_returns


class DataInspector:
    """
    Diagnostic tool for auditing financial datasets and comparing statistical distributions.
    """

    @staticmethod
    def audit_continuity(df: pd.DataFrame) -> Dict[str, object]:
        """
        Checks for gaps in business day continuity, missing values, and duplicate timestamps.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a pandas DatetimeIndex.")

        total_rows = len(df)
        null_counts = df.isnull().sum().to_dict()
        duplicates = df.index.duplicated().sum()

        # Check calendar continuity against business day calendar (freq='B')
        full_bday_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
        missing_bdays = full_bday_range.difference(df.index)

        return {
            "total_bars": total_rows,
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "null_values": null_counts,
            "duplicate_timestamps": int(duplicates),
            "expected_business_days": len(full_bday_range),
            "calendar_holiday_gaps": len(missing_bdays),
        }

    @staticmethod
    def compute_stylized_facts(price_series: pd.Series) -> Dict[str, float]:
        """
        Computes financial econometrics properties of the return series:
        - Mean daily return
        - Daily & annualized volatility
        - Skewness (asymmetry)
        - Kurtosis (fat tails / leptokurtosis, normal distribution = 3.0)
        - Jarque-Bera test p-value for normality
        """
        returns = calculate_log_returns(price_series)

        mean_daily = float(returns.mean())
        vol_daily = float(returns.std())
        vol_annual = float(vol_daily * np.sqrt(252))
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns, fisher=False))  # Pearson definition: Normal = 3.0
        jb_stat, jb_pval = stats.jarque_bera(returns)

        return {
            "observations": len(returns),
            "mean_daily_return": mean_daily,
            "annualized_volatility": vol_annual,
            "skewness": skew,
            "kurtosis": kurt,
            "is_leptokurtic": bool(kurt > 3.0),
            "jarque_bera_pval": float(jb_pval),
            "rejects_normality": bool(jb_pval < 0.05),
        }

    @staticmethod
    def compare_real_vs_synthetic(
        real_series: pd.Series, gbm_series: pd.Series, merton_series: pd.Series
    ) -> pd.DataFrame:
        """
        Constructs a comparative summary table demonstrating how Merton Jump Diffusion
        captures heavy tails and non-normality better than pure GBM.
        """
        stats_real = DataInspector.compute_stylized_facts(real_series)
        stats_gbm = DataInspector.compute_stylized_facts(gbm_series)
        stats_merton = DataInspector.compute_stylized_facts(merton_series)

        df_comparison = pd.DataFrame(
            [stats_real, stats_gbm, stats_merton],
            index=["Real Market Asset", "Synthetic GBM (Continuous)", "Synthetic Merton Jump (Discontinuous)"],
        )
        return df_comparison
