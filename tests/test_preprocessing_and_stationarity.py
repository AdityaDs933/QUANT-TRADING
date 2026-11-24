"""
Unit Tests for Preprocessing and Stationarity Testing
Authored by: Tanveer (15% Component)

Verifies:
1. StationarityTester on known stationary (white noise) vs non-stationary (random walk)
2. Joint ADF-KPSS classification matrix
3. Fractional differentiation weight binomial formula and threshold truncation
4. Memory retention vs stationarity trade-off
5. DataPreprocessor calendar alignment, forward fill, and price physics
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.stationarity import StationarityTester, FractionalDifferentiator
from src.data.preprocess import DataPreprocessor


@pytest.fixture
def stationary_series():
    """Generates pure i.i.d. Gaussian white noise (strictly stationary)."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 1, size=500), name="WhiteNoise")


@pytest.fixture
def non_stationary_series():
    """Generates a random walk with drift (unit root / non-stationary)."""
    np.random.seed(42)
    shocks = np.random.normal(0.001, 0.02, size=500)
    prices = 100.0 * np.exp(np.cumsum(shocks))
    return pd.Series(prices, name="RandomWalk")


def test_stationarity_white_noise(stationary_series):
    """White noise should pass ADF (p < 0.05) and pass KPSS (p >= 0.05)."""
    diag = StationarityTester.diagnose_stationarity(stationary_series, "WhiteNoise")
    assert "Strictly Stationary" in diag["final_diagnosis"]


def test_stationarity_random_walk(non_stationary_series):
    """Random walk should fail ADF (p >= 0.05) and fail KPSS (p < 0.05)."""
    diag = StationarityTester.diagnose_stationarity(non_stationary_series, "RandomWalk")
    assert "Non-Stationary" in diag["final_diagnosis"] or "Unit Root" in diag["final_diagnosis"]


def test_fractional_diff_weights_formula():
    """Verifies binomial expansion weights: omega_0 = 1, omega_1 = -d."""
    d = 0.4
    weights = FractionalDifferentiator.get_weights(d=d, size=5)
    # Weights array is in reverse order (oldest to newest), so last element is omega_0
    assert pytest.approx(weights[-1], rel=1e-5) == 1.0
    # Second to last element is omega_1 = -d
    assert pytest.approx(weights[-2], rel=1e-5) == -d


def test_fractional_diff_memory_retention(non_stationary_series):
    """
    Verifies that fractional differentiation at d=0.4 preserves higher correlation
    with price series than integer return differentiation (d=1.0).
    """
    d_frac = FractionalDifferentiator.fractional_diff_fixed_window(non_stationary_series, d=0.35)
    d_int = non_stationary_series.diff().dropna()

    common_frac = non_stationary_series.loc[d_frac.index]
    common_int = non_stationary_series.loc[d_int.index]

    corr_frac = np.corrcoef(common_frac, d_frac)[0, 1]
    corr_int = np.corrcoef(common_int, d_int)[0, 1]

    assert corr_frac > corr_int
    assert corr_frac > 0.55  # Fractional retains over 55% correlation while integer diff retains almost 0


def test_data_preprocessor_calendar_and_physics():
    """Verifies calendar reindexing, forward-filling, and price bounds."""
    # Create irregular series with weekend gap and missing price
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])  # Jan 4 missing
    df_raw = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 105.0],
            "High": [103.0, 104.0, 106.0],
            "Low": [99.0, 101.0, 103.0],
            "Close": [102.0, 103.0, 104.0],
            "Adj Close": [102.0, 103.0, 104.0],
            "Volume": [1000, 1200, 1500],
        },
        index=dates,
    )
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_ohlcv(df_raw, symbol="TEST")

    # Business day Jan 4 should be imputed via forward fill
    assert pd.to_datetime("2024-01-04") in df_clean.index
    assert df_clean.loc["2024-01-04", "Close"] == 103.0  # forward-filled from Jan 3
    assert df_clean.loc["2024-01-04", "Volume"] == 0.0   # missing day volume is 0
    assert (df_clean["High"] >= df_clean["Low"]).all()
    assert (df_clean["Low"] > 0).all()
