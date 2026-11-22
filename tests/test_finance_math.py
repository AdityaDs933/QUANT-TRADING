"""
Unit Tests for Financial Mathematics Library
Authored by: Tanveer (15% Component)
Verifies mathematical correctness, vectorization, and edge-case handling.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.finance_math import (
    calculate_arithmetic_returns,
    calculate_log_returns,
    log_to_arithmetic,
    arithmetic_to_log,
    compound_returns,
    calculate_portfolio_returns,
    annualize_return,
    annualize_volatility,
    calculate_bid_ask_spread,
)


@pytest.fixture
def sample_price_series():
    """Generates a deterministic price series: 100 -> 110 -> 99 -> 108.9"""
    return pd.Series([100.0, 110.0, 99.0, 108.9], name="Close")


@pytest.fixture
def sample_multi_asset_returns():
    """Generates 3 periods of returns for 2 assets."""
    data = {
        "Asset_A": [0.01, 0.02, -0.01],
        "Asset_B": [0.03, -0.01, 0.04],
    }
    return pd.DataFrame(data)


def test_arithmetic_returns_values(sample_price_series):
    returns = calculate_arithmetic_returns(sample_price_series)
    assert len(returns) == 3
    # 100 -> 110 is +10%
    assert pytest.approx(returns.iloc[0], rel=1e-5) == 0.10
    # 110 -> 99 is -10%
    assert pytest.approx(returns.iloc[1], rel=1e-5) == -0.10
    # 99 -> 108.9 is +10%
    assert pytest.approx(returns.iloc[2], rel=1e-5) == 0.10


def test_log_returns_values(sample_price_series):
    log_rets = calculate_log_returns(sample_price_series)
    assert len(log_rets) == 3
    assert pytest.approx(log_rets.iloc[0], rel=1e-5) == np.log(1.10)
    assert pytest.approx(log_rets.iloc[1], rel=1e-5) == np.log(0.90)


def test_log_arithmetic_conversions():
    r_arith = pd.Series([0.05, -0.03, 0.12, -0.08])
    r_log = arithmetic_to_log(r_arith)
    r_recovered = log_to_arithmetic(r_log)
    pd.testing.assert_series_equal(r_arith, r_recovered, check_exact=False, rtol=1e-6)


def test_compound_returns():
    returns = pd.Series([0.10, -0.10, 0.10])
    wealth = compound_returns(returns)
    # Day 1: 1.10
    # Day 2: 1.10 * 0.90 = 0.99
    # Day 3: 0.99 * 1.10 = 1.089
    assert pytest.approx(wealth.iloc[0], rel=1e-5) == 1.10
    assert pytest.approx(wealth.iloc[1], rel=1e-5) == 0.99
    assert pytest.approx(wealth.iloc[2], rel=1e-5) == 1.089


def test_portfolio_returns_weighting(sample_multi_asset_returns):
    weights = [0.60, 0.40]
    p_rets = calculate_portfolio_returns(sample_multi_asset_returns, weights)
    # Row 0: 0.60 * 0.01 + 0.40 * 0.03 = 0.006 + 0.012 = 0.018
    assert pytest.approx(p_rets.iloc[0], rel=1e-5) == 0.018


def test_portfolio_weights_validation(sample_multi_asset_returns):
    with pytest.raises(ValueError, match="Portfolio weights must sum to 1.0"):
        calculate_portfolio_returns(sample_multi_asset_returns, [0.5, 0.2])


def test_annualization():
    daily_r = 0.0005  # ~13.4% compound annual
    annual_compound = annualize_return(daily_r, periods_per_year=252, method="compound")
    assert annual_compound > 0.13

    daily_vol = 0.015  # 1.5% daily vol
    annual_vol = annualize_volatility(daily_vol, periods_per_year=252)
    # 0.015 * sqrt(252) ~= 0.2381 (23.8%)
    assert pytest.approx(annual_vol, rel=1e-3) == 0.2381


def test_bid_ask_spread():
    bid = pd.Series([99.95, 149.90])
    ask = pd.Series([100.05, 150.10])
    rel_spread = calculate_bid_ask_spread(bid, ask, relative=True)
    # Mid for first is 100.00, diff is 0.10 -> 0.10 / 100 = 0.0010 (10 bps)
    assert pytest.approx(rel_spread.iloc[0], rel=1e-5) == 0.0010


def test_negative_price_exception():
    bad_prices = pd.Series([100.0, -50.0, 80.0])
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_arithmetic_returns(bad_prices)
