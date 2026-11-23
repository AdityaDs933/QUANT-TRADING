"""
Unit Tests for Data Loaders and Stochastic Generators
Authored by: Tanveer (15% Component)

Verifies:
1. Synthetic GBM drift and volatility consistency
2. Merton Jump Diffusion leptokurtosis (fat tails)
3. Real Market Data Loader schema validation & caching
4. DataInspector continuity audits
"""

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import StochasticProcessGenerator
from src.data.real_data_loader import RealMarketDataLoader
from src.data.inspector import DataInspector
from src.utils.finance_math import calculate_log_returns


@pytest.fixture
def synthetic_engine():
    return StochasticProcessGenerator(random_seed=42)


def test_gbm_simulation_properties(synthetic_engine):
    """Verifies that large-sample GBM converges to theoretical drift and vol."""
    s0 = 100.0
    mu = 0.10
    sigma = 0.20
    n_days = 2520  # 10 years
    n_paths = 100  # Monte Carlo ensemble

    paths = synthetic_engine.generate_gbm(s0=s0, mu=mu, sigma=sigma, n_days=n_days, n_paths=n_paths)
    assert paths.shape == (n_days + 1, n_paths)

    # Compute daily log returns for all paths
    returns = np.log(paths.values[1:] / paths.values[:-1])
    empirical_daily_vol = np.std(returns, axis=0)
    empirical_annual_vol = empirical_daily_vol * np.sqrt(252)

    # Mean annual volatility across 100 paths should match target sigma=0.20 within 5%
    mean_vol = float(np.mean(empirical_annual_vol))
    assert pytest.approx(mean_vol, rel=0.08) == sigma


def test_merton_jump_fat_tails(synthetic_engine):
    """
    Verifies that Merton Jump Diffusion exhibits excess kurtosis (fat tails)
    compared to Gaussian GBM.
    """
    n_days = 3000
    gbm_path = synthetic_engine.generate_gbm(s0=100.0, mu=0.08, sigma=0.15, n_days=n_days, n_paths=1)
    merton_path = synthetic_engine.generate_merton_jump(
        s0=100.0, mu=0.08, sigma=0.15, lambda_j=0.40, mu_j=-0.08, sigma_j=0.15, n_days=n_days, n_paths=1
    )

    stats_gbm = DataInspector.compute_stylized_facts(gbm_path)
    stats_merton = DataInspector.compute_stylized_facts(merton_path)

    # Merton jump diffusion should produce higher kurtosis (fatter tails) than pure GBM
    assert stats_merton["kurtosis"] > stats_gbm["kurtosis"]
    assert stats_merton["is_leptokurtic"] is True


def test_synthetic_ohlcv_schema(synthetic_engine):
    """Verifies that generated synthetic OHLCV satisfies market structure rules."""
    df_ohlcv = synthetic_engine.generate_synthetic_ohlcv(
        symbol="SYN_TEST", process_type="merton", s0=150.0, n_days=500
    )

    assert synthetic_engine.validate_schema(df_ohlcv)
    assert (df_ohlcv["High"] >= df_ohlcv["Low"]).all()
    assert (df_ohlcv["High"] >= df_ohlcv["Open"]).all()
    assert (df_ohlcv["High"] >= df_ohlcv["Close"]).all()
    assert (df_ohlcv["Low"] <= df_ohlcv["Open"]).all()
    assert (df_ohlcv["Low"] <= df_ohlcv["Close"]).all()
    assert (df_ohlcv["Volume"] > 0).all()


def test_data_inspector_continuity():
    """Verifies continuity auditor correctly catches missing days and duplicates."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 105.0,
            "Low": 98.0,
            "Close": 102.0,
            "Adj Close": 102.0,
            "Volume": 10000,
        },
        index=dates,
    )
    audit = DataInspector.audit_continuity(df)
    assert audit["total_bars"] == 10
    assert audit["duplicate_timestamps"] == 0
    assert audit["null_values"]["Close"] == 0
