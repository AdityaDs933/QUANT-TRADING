"""
Stationarity Testing and Fractional Differentiation Engine
Authored by: Aditya (70% Component)

Implements:
1. Augmented Dickey-Fuller (ADF) Unit Root Test
2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Trend Stationarity Test
3. Joint ADF-KPSS Diagnostic Matrix
4. Fractional Differentiation (Marcos López de Prado, AFML Chapter 5):
   Solves the Memory vs. Stationarity dilemma by computing (1 - B)^d
   to find the minimum d that achieves stationarity while preserving memory.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

# Suppress statsmodels p-value interpolation and future return-type warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SeriesOrArray = Union[pd.Series, np.ndarray]


class StationarityTester:
    """
    Econometric testing suite for identifying unit roots and verifying time-series stationarity.
    """

    @staticmethod
    def adf_test(series: SeriesOrArray, maxlag: Optional[int] = None) -> Dict[str, Union[float, int, bool, Dict]]:
        """
        Runs the Augmented Dickey-Fuller (ADF) test:
            Delta y_t = alpha + beta*t + gamma * y_{t-1} + sum(delta_i * Delta y_{t-i}) + eps_t

        Null Hypothesis (H0): gamma = 0 (Series has a unit root / is non-stationary).
        Alternative (H1): gamma < 0 (Series is stationary).
        """
        clean_series = pd.Series(series).dropna()
        if len(clean_series) < 20:
            raise ValueError("Series length must be at least 20 observations for reliable ADF testing.")

        result = adfuller(clean_series, maxlag=maxlag, autolag="AIC")
        adf_stat, p_val, lags_used, n_obs, crit_values, icbest = result

        return {
            "test_stat": float(adf_stat),
            "p_value": float(p_val),
            "lags_used": int(lags_used),
            "n_obs": int(n_obs),
            "critical_values": {k: float(v) for k, v in crit_values.items()},
            "is_stationary": bool(p_val < 0.05),
        }

    @staticmethod
    def kpss_test(series: SeriesOrArray, regression: str = "c") -> Dict[str, Union[float, int, bool, Dict]]:
        """
        Runs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test:
        Null Hypothesis (H0): Series is stationary around a constant (or trend).
        Alternative (H1): Series has a unit root.
        """
        clean_series = pd.Series(series).dropna()
        if len(clean_series) < 20:
            raise ValueError("Series length must be at least 20 observations for reliable KPSS testing.")

        # Filter warnings regarding p-value truncation
        stat, p_val, lags, crit_values = kpss(clean_series, regression=regression, nlags="auto")

        return {
            "test_stat": float(stat),
            "p_value": float(p_val),
            "lags_used": int(lags),
            "critical_values": {k: float(v) for k, v in crit_values.items()},
            # In KPSS: p >= 0.05 means we FAIL to reject H0 -> stationary!
            "is_stationary": bool(p_val >= 0.05),
        }

    @staticmethod
    def diagnose_stationarity(series: SeriesOrArray, name: str = "Series") -> Dict[str, str]:
        """
        Performs dual ADF-KPSS classification matrix:
        - Case 1: ADF stationary & KPSS stationary -> Strictly Stationary
        - Case 2: ADF non-stationary & KPSS non-stationary -> Unit Root / Non-Stationary
        - Case 3: ADF non-stationary & KPSS stationary -> Trend Stationary
        - Case 4: ADF stationary & KPSS non-stationary -> Difference Stationary
        """
        adf_res = StationarityTester.adf_test(series)
        kpss_res = StationarityTester.kpss_test(series)

        adf_stat = adf_res["is_stationary"]
        kpss_stat = kpss_res["is_stationary"]

        if adf_stat and kpss_stat:
            diagnosis = "Strictly Stationary (I(0))"
        elif not adf_stat and not kpss_stat:
            diagnosis = "Non-Stationary / Unit Root (I(1))"
        elif not adf_stat and kpss_stat:
            diagnosis = "Trend Stationary"
        else:
            diagnosis = "Difference Stationary (Weak Evidence)"

        return {
            "name": name,
            "adf_stat": f"{adf_res['test_stat']:.4f}",
            "adf_pval": f"{adf_res['p_value']:.4e}",
            "adf_verdict": "Reject H0 (Stationary)" if adf_stat else "Fail to Reject H0 (Non-Stationary)",
            "kpss_stat": f"{kpss_res['test_stat']:.4f}",
            "kpss_pval": f"{kpss_res['p_value']:.4e}",
            "kpss_verdict": "Fail to Reject H0 (Stationary)" if kpss_stat else "Reject H0 (Non-Stationary)",
            "final_diagnosis": diagnosis,
        }


class FractionalDifferentiator:
    """
    Implements Marcos López de Prado's Fractional Differentiation algorithm.
    Expands the lag operator (1 - B)^d via infinite binomial series:
        (1 - B)^d = sum_{k=0}^infty omega_k * B^k
        omega_0 = 1
        omega_k = -omega_{k-1} * (d - k + 1) / k
    """

    @staticmethod
    def get_weights(d: float, size: int) -> np.ndarray:
        """
        Generates iterative weights for fractional differentiation.
        """
        weights = [1.0]
        for k in range(1, size):
            w_k = -weights[-1] / k * (d - k + 1)
            weights.append(w_k)
        return np.array(weights[::-1])

    @staticmethod
    def get_weights_threshold(d: float, threshold: float = 1e-4) -> np.ndarray:
        """
        Generates weights dropping those with absolute magnitude < threshold.
        """
        weights = [1.0]
        k = 1
        while True:
            w_k = -weights[-1] / k * (d - k + 1)
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1
        return np.array(weights[::-1])

    @staticmethod
    def fractional_diff_fixed_window(
        series: pd.Series, d: float, threshold: float = 1e-4
    ) -> pd.Series:
        """
        Fractionally differentiates a series using a fixed-width window of weights.
        Prevents lookahead bias and ensures constant lag window memory.
        """
        weights = FractionalDifferentiator.get_weights_threshold(d, threshold)
        width = len(weights)

        res = {}
        values = series.values
        indices = series.index

        for i in range(width - 1, len(series)):
            window = values[i - width + 1 : i + 1]
            res[indices[i]] = np.dot(weights, window)

        out_series = pd.Series(res, name=f"{series.name}_frac_d{d:.2f}")
        return out_series

    @staticmethod
    def find_min_d_for_stationarity(
        series: pd.Series,
        d_range: Optional[np.ndarray] = None,
        threshold: float = 1e-4,
        p_val_threshold: float = 0.05,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Sweeps d across [0.0, 1.0] to find the minimum d that achieves stationarity (ADF p-val < 0.05).
        Also measures the Pearson correlation between the original and differentiated series
        to quantify retained memory.
        """
        if d_range is None:
            d_range = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, 0.10, ..., 1.00

        records = []
        min_d = 1.0
        found_stationary = False

        for d in d_range:
            if d == 0.0:
                diff_series = series.copy()
            elif d == 1.0:
                diff_series = series.diff().dropna()
            else:
                diff_series = FractionalDifferentiator.fractional_diff_fixed_window(
                    series, d=d, threshold=threshold
                )

            # Align original and differentiated series for correlation
            common_idx = series.index.intersection(diff_series.index)
            corr = np.corrcoef(series.loc[common_idx], diff_series.loc[common_idx])[0, 1]

            adf_res = StationarityTester.adf_test(diff_series)
            p_val = adf_res["p_value"]
            is_stat = p_val < p_val_threshold

            if is_stat and not found_stationary:
                min_d = d
                found_stationary = True

            records.append(
                {
                    "d": round(d, 2),
                    "adf_stat": adf_res["test_stat"],
                    "p_val": p_val,
                    "correlation_memory": corr,
                    "is_stationary": is_stat,
                }
            )

        df_results = pd.DataFrame(records)
        return min_d, df_results
