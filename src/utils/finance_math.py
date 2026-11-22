"""
Financial Mathematics Foundations
Core mathematical functions for price transformations, returns, compounding,
annualization, and microstructure friction calculations.
"""

import numpy as np
import pandas as pd
from typing import Union

SeriesOrDataFrame = Union[pd.Series, pd.DataFrame]


def calculate_arithmetic_returns(prices: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """
    Computes simple (arithmetic) returns:
        R_t = (P_t - P_{t-1}) / P_{t-1} = (P_t / P_{t-1}) - 1

    Properties:
    - Cross-sectional additivity across portfolio constituents.
    - Bound from below at -1.0 (-100%).
    """
    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("Prices must be strictly positive to calculate arithmetic returns.")
    return prices.pct_change().dropna()


def calculate_log_returns(prices: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """
    Computes continuously compounded (logarithmic) returns:
        r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Properties:
    - Intertemporal additivity across multiple holding periods: r_{0, T} = sum(r_t).
    - Unbounded from below (-inf as P_t -> 0), suitable for Gaussian-like modeling.
    """
    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("Prices must be strictly positive to calculate log returns.")
    return np.log(prices / prices.shift(1)).dropna()


def log_to_arithmetic(log_returns: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """Converts log returns to arithmetic returns: R_t = exp(r_t) - 1"""
    return np.exp(log_returns) - 1.0


def arithmetic_to_log(arithmetic_returns: SeriesOrDataFrame) -> SeriesOrDataFrame:
    """Converts arithmetic returns to log returns: r_t = ln(1 + R_t)"""
    if (arithmetic_returns <= -1.0).any().any() if isinstance(arithmetic_returns, pd.DataFrame) else (arithmetic_returns <= -1.0).any():
        raise ValueError("Arithmetic return cannot be <= -1.0 (-100%) for log conversion.")
    return np.log(1.0 + arithmetic_returns)


def compound_returns(returns: pd.Series) -> pd.Series:
    """
    Computes the cumulative compounding wealth index starting at 1.0:
        Wealth_t = prod_{s=1}^t (1 + R_s)
    """
    return (1.0 + returns).cumprod()


def calculate_portfolio_returns(returns_df: pd.DataFrame, weights: Union[np.ndarray, list]) -> pd.Series:
    """
    Calculates weighted portfolio arithmetic returns:
        R_{p, t} = sum_{i=1}^N w_i * R_{i, t}
    """
    weights_arr = np.asarray(weights, dtype=float)
    if not np.isclose(np.sum(weights_arr), 1.0, atol=1e-4):
        raise ValueError(f"Portfolio weights must sum to 1.0 (current sum: {np.sum(weights_arr):.4f})")
    if returns_df.shape[1] != len(weights_arr):
        raise ValueError(f"Number of assets ({returns_df.shape[1]}) does not match weights length ({len(weights_arr)})")
    
    return returns_df.dot(weights_arr)


def annualize_return(daily_return: float, periods_per_year: int = 252, method: str = "compound") -> float:
    """
    Annualizes a single-period (daily) return.
    - 'compound': (1 + R)^252 - 1
    - 'linear': R * 252 (for log returns)
    """
    if method == "compound":
        return float((1.0 + daily_return) ** periods_per_year - 1.0)
    elif method == "linear":
        return float(daily_return * periods_per_year)
    else:
        raise ValueError("Method must be either 'compound' or 'linear'.")


def annualize_volatility(daily_vol: float, periods_per_year: int = 252) -> float:
    """
    Annualizes daily standard deviation (volatility) using square-root-of-time rule:
        sigma_{annual} = sigma_{daily} * sqrt(252)
    """
    return float(daily_vol * np.sqrt(periods_per_year))


def calculate_bid_ask_spread(bid: pd.Series, ask: pd.Series, relative: bool = True) -> pd.Series:
    """
    Computes bid-ask spread:
    - Quoted Spread: Ask - Bid
    - Relative (Proportional) Spread: (Ask - Bid) / Mid, where Mid = (Ask + Bid) / 2
    """
    quoted_spread = ask - bid
    if (quoted_spread < 0).any():
        raise ValueError("Ask price cannot be strictly lower than Bid price in a normal market.")
    
    if not relative:
        return quoted_spread
    
    mid_price = (ask + bid) / 2.0
    return quoted_spread / mid_price
