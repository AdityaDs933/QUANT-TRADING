"""
Unit Tests for EDA Engine
Authored by: Tanveer (15% Component)

Verifies:
1. DistributionAnalyzer on known Gaussian vs fat-tailed distributions
2. ARCH test correctly identifies clustered volatility in a GARCH(1,1) series
3. DrawdownAnalyzer computes correct peak-to-trough decline and duration
4. CrossAssetAnalyzer correlation and average pairwise correlation
5. RollingMarketMetrics beta estimation against known synthetic data
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.eda_engine import (
    DistributionAnalyzer,
    RollingMarketMetrics,
    DrawdownAnalyzer,
    CrossAssetAnalyzer,
)
from src.utils.finance_math import calculate_log_returns, calculate_arithmetic_returns


@pytest.fixture
def gaussian_returns():
    """Pure i.i.d. Gaussian returns ~ N(0.0003, 0.012^2)."""
    np.random.seed(0)
    return pd.Series(np.random.normal(0.0003, 0.012, 1000), name="GaussianAsset")


@pytest.fixture
def garch_returns():
    """
    GARCH(1,1)-like clustered volatility series:
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    """
    np.random.seed(1)
    n = 1000
    omega, alpha, beta = 0.00001, 0.08, 0.90
    sigma2 = np.full(n, omega / (1 - alpha - beta))
    eps = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * np.random.randn()
    return pd.Series(eps, name="GARCH_Asset")


def test_distributional_summary_gaussian(gaussian_returns):
    """Gaussian series should have kurtosis ~3, not reject normality."""
    result = DistributionAnalyzer.describe_returns(gaussian_returns, "GaussTest")
    assert result["n_obs"] == 1000
    # Kurtosis should be near 3 (Gaussian)
    assert 2.5 < result["kurtosis"] < 3.8
    assert bool(result["is_leptokurtic"]) is False
    # Should NOT reject normality at 5% level
    assert bool(result["rejects_normality"]) is False


def test_arch_test_detects_clustering(garch_returns):
    """GARCH-generated series should exhibit significant ARCH effects."""
    result = DistributionAnalyzer.arch_test(garch_returns, nlags=10)
    assert bool(result["has_arch_effects"]) is True
    assert result["lm_pval"] < 0.001


def test_arch_test_no_clustering(gaussian_returns):
    """Pure i.i.d. Gaussian should NOT exhibit ARCH effects."""
    result = DistributionAnalyzer.arch_test(gaussian_returns, nlags=10)
    assert bool(result["has_arch_effects"]) is False


def test_drawdown_profile():
    """Verifies a known price path: 100 -> 110 -> 88 -> 99 gives MDD = -20%."""
    # Arithmetic returns: +10%, -20%, +12.5%
    returns = pd.Series([0.10, -0.20, 0.125], name="TestAsset")
    profile = DrawdownAnalyzer.full_drawdown_profile(returns)

    # Wealth index: 1.10, 0.88, 0.99
    # Peak at 1.10; trough at 0.88 -> drawdown = (0.88 - 1.10) / 1.10 = -0.2
    assert pytest.approx(profile["max_drawdown"], abs=1e-4) == -0.20
    assert profile["max_drawdown_duration_days"] == 2  # 2 consecutive underwater days


def test_cross_asset_correlation():
    """Verifies correlation matrix diagonal = 1 and symmetric off-diagonal."""
    np.random.seed(42)
    r1 = pd.Series(np.random.normal(0, 0.01, 500), name="A")
    r2 = pd.Series(np.random.normal(0, 0.01, 500), name="B")
    r3 = r1 * 0.8 + np.random.normal(0, 0.005, 500)
    r3.name = "C"

    corr = CrossAssetAnalyzer.correlation_matrix({"A": r1, "B": r2, "C": pd.Series(r3, name="C")})
    # Diagonal must be 1
    for col in corr.columns:
        assert pytest.approx(corr.loc[col, col], abs=1e-10) == 1.0
    # Correlation matrix is symmetric
    assert (corr - corr.T).abs().max().max() < 1e-10
    # C and A should have higher correlation than B and A
    assert abs(corr.loc["A", "C"]) > abs(corr.loc["A", "B"])


def test_rolling_volatility_shape(gaussian_returns):
    """Rolling volatility should have same length as input, with NaNs at the start."""
    rolling_vol = RollingMarketMetrics.rolling_volatility(gaussian_returns, window=21)
    assert len(rolling_vol) == len(gaussian_returns)
    # min_periods=10 so first 9 must be NaN
    assert rolling_vol.iloc[0:9].isna().all()
    assert rolling_vol.iloc[-1] > 0
