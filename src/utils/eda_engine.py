"""
Quantitative EDA Engine — Statistical Analysis of Market Dynamics
Authored by: Aditya (70% Component)

Implements formal statistical tests and rolling-window diagnostics:
1. Normality Tests: Jarque-Bera, Shapiro-Wilk
2. Volatility Clustering: Engle ARCH LM Test
3. Rolling Market Statistics: Beta, Sharpe, Drawdown
4. Cross-Asset Correlation and Covariance Dynamics
5. Drawdown anatomy (depth, duration, underwater curve)
"""

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.utils.finance_math import calculate_log_returns, annualize_return, annualize_volatility


class DistributionAnalyzer:
    """
    Formal statistical hypothesis testing for return distribution properties.
    """

    @staticmethod
    def describe_returns(returns: pd.Series, label: str = "Asset") -> Dict:
        """
        Computes a comprehensive distributional summary of a return series:
        mean, vol, skewness, kurtosis, Jarque-Bera, percentiles.
        """
        clean = returns.dropna()
        n = len(clean)

        mean_d = float(clean.mean())
        std_d = float(clean.std())
        skew = float(stats.skew(clean))
        kurt = float(stats.kurtosis(clean, fisher=False))  # Pearson (Normal=3)

        jb_stat, jb_pval = stats.jarque_bera(clean)

        # Sharpe-like ratio of daily returns (not annualized yet)
        daily_sharpe = mean_d / std_d if std_d > 0 else 0.0

        return {
            "label": label,
            "n_obs": n,
            "mean_daily": mean_d,
            "std_daily": std_d,
            "ann_return": annualize_return(mean_d, 252, method="compound"),
            "ann_vol": annualize_volatility(std_d, 252),
            "skewness": skew,
            "kurtosis": kurt,
            "is_leptokurtic": kurt > 3.0,
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pval": float(jb_pval),
            "rejects_normality": jb_pval < 0.05,
            "pct_1": float(np.percentile(clean, 1)),
            "pct_5": float(np.percentile(clean, 5)),
            "pct_95": float(np.percentile(clean, 95)),
            "pct_99": float(np.percentile(clean, 99)),
        }

    @staticmethod
    def arch_test(returns: pd.Series, nlags: int = 10) -> Dict:
        """
        Engle's ARCH Lagrange Multiplier (LM) Test for conditional heteroskedasticity.

        Null Hypothesis (H0): No ARCH effects (homoskedastic residuals).
        Alternative (H1): ARCH(q) effects present (volatility clustering).

        Volatility clustering means that large shocks (of either sign) tend to be
        followed by more large shocks — the hallmark of financial time series.
        """
        clean = returns.dropna()
        lm_stat, lm_pval, f_stat, f_pval = het_arch(clean, nlags=nlags)
        return {
            "lm_stat": float(lm_stat),
            "lm_pval": float(lm_pval),
            "f_stat": float(f_stat),
            "f_pval": float(f_pval),
            "has_arch_effects": lm_pval < 0.05,
        }


class RollingMarketMetrics:
    """
    Computes time-varying rolling-window statistics capturing changing market regimes.
    """

    @staticmethod
    def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
        """Annualized rolling standard deviation of daily log returns."""
        return returns.rolling(window=window, min_periods=max(5, window // 2)).std() * np.sqrt(252)

    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 63, risk_free: float = 0.04) -> pd.Series:
        """
        Rolling annualized Sharpe Ratio:
            S_t = (mu_t - r_f) / sigma_t
        """
        rf_daily = risk_free / 252.0
        excess = returns - rf_daily
        rolling_mean = excess.rolling(window=window, min_periods=max(10, window // 3)).mean() * 252
        rolling_std = returns.rolling(window=window, min_periods=max(10, window // 3)).std() * np.sqrt(252)
        return rolling_mean / rolling_std.replace(0, np.nan)

    @staticmethod
    def rolling_beta(
        asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63
    ) -> pd.Series:
        """
        Rolling OLS beta against benchmark:
            beta_t = Cov(R_asset, R_benchmark)_t / Var(R_benchmark)_t
        """
        aligned = pd.DataFrame({"asset": asset_returns, "bench": benchmark_returns}).dropna()
        betas = {}
        for i in range(window, len(aligned) + 1):
            window_df = aligned.iloc[i - window : i]
            cov_matrix = np.cov(window_df["asset"], window_df["bench"])
            cov = cov_matrix[0, 1]
            var_bench = cov_matrix[1, 1]
            beta = cov / var_bench if var_bench > 1e-10 else np.nan
            betas[aligned.index[i - 1]] = beta
        return pd.Series(betas, name=f"Rolling_Beta_{window}d")

    @staticmethod
    def rolling_correlation(
        s1: pd.Series, s2: pd.Series, window: int = 63
    ) -> pd.Series:
        """Rolling Pearson correlation coefficient between two return series."""
        df = pd.DataFrame({"s1": s1, "s2": s2}).dropna()
        return df["s1"].rolling(window=window, min_periods=max(10, window // 3)).corr(df["s2"])


class DrawdownAnalyzer:
    """
    Computes maximum drawdown, drawdown duration, and the underwater equity curve.
    """

    @staticmethod
    def compute_drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
        """
        Returns the percentage drawdown at each point in time:
            DD_t = (W_t - Peak_t) / Peak_t
        where Peak_t = max(W_s, s <= t)
        """
        rolling_peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_peak) / rolling_peak
        return drawdown

    @staticmethod
    def max_drawdown(cumulative_returns: pd.Series) -> float:
        """Maximum peak-to-trough decline as a negative fraction."""
        return float(DrawdownAnalyzer.compute_drawdown_series(cumulative_returns).min())

    @staticmethod
    def max_drawdown_duration(cumulative_returns: pd.Series) -> int:
        """
        Maximum number of consecutive days the strategy remained underwater (below prior peak).
        """
        dd_series = DrawdownAnalyzer.compute_drawdown_series(cumulative_returns)
        underwater = dd_series < 0

        max_duration = 0
        current_duration = 0
        for is_under in underwater:
            if is_under:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return max_duration

    @staticmethod
    def full_drawdown_profile(arithmetic_returns: pd.Series) -> Dict:
        """
        Returns a complete drawdown profile dictionary:
        - Wealth index, drawdown series, max drawdown, max duration.
        """
        wealth = (1 + arithmetic_returns).cumprod()
        dd_series = DrawdownAnalyzer.compute_drawdown_series(wealth)
        mdd = float(dd_series.min())
        mdd_dur = DrawdownAnalyzer.max_drawdown_duration(wealth)
        return {
            "wealth_index": wealth,
            "drawdown_series": dd_series,
            "max_drawdown": mdd,
            "max_drawdown_pct": f"{mdd * 100:.2f}%",
            "max_drawdown_duration_days": mdd_dur,
        }


class CrossAssetAnalyzer:
    """
    Multi-asset correlation and covariance matrix analysis.
    """

    @staticmethod
    def correlation_matrix(returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Computes the full Pearson correlation matrix across assets.
        Uses intersection of available dates to handle different trading calendars.
        """
        df = pd.DataFrame(returns_dict).dropna()
        return df.corr()

    @staticmethod
    def average_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
        """
        Returns the mean of all off-diagonal correlation pairs,
        a proxy for portfolio diversification potential.
        """
        n = corr_matrix.shape[0]
        corr_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                corr_vals.append(corr_matrix.iloc[i, j])
        return float(np.mean(corr_vals)) if corr_vals else 0.0
