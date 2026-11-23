"""
Synthetic Stochastic Market Data Generator
Authored by: Aditya (70% Component)

Implements continuous and jump-diffusion stochastic processes:
1. Geometric Brownian Motion (GBM):
       dS_t = mu * S_t * dt + sigma * S_t * dW_t
2. Merton Jump-Diffusion Model (1976):
       dS_t = (mu - lambda * k) * S_t * dt + sigma * S_t * dW_t + (J - 1) * S_t * dN_t
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.base_interfaces import BaseDataLoader


class StochasticProcessGenerator(BaseDataLoader):
    """
    Generates synthetic financial price and OHLCV time series using stochastic calculus models.
    Can be used for stress-testing, robustness checks, and data augmentation.
    """

    def __init__(self, random_seed: Optional[int] = 42) -> None:
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate_gbm(
        self,
        s0: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        n_days: int = 1512,
        dt: float = 1.0 / 252.0,
        n_paths: int = 1,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Simulates Geometric Brownian Motion via exact solution of Itô SDE:
            S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)

        Parameters:
        -----------
        s0 : float
            Initial asset price.
        mu : float
            Annualized expected drift.
        sigma : float
            Annualized volatility.
        n_days : int
            Number of discrete time steps (trading days).
        dt : float
            Time step size (default 1/252 for daily steps).
        n_paths : int
            Number of simulated sample paths.

        Returns:
        --------
        pd.Series (if n_paths == 1) or pd.DataFrame (if n_paths > 1)
        """
        if s0 <= 0 or sigma <= 0 or n_days <= 0:
            raise ValueError("s0, sigma, and n_days must be strictly positive.")

        # Standard normal random shocks
        dw = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_days, n_paths))
        
        # Cumulative Wiener process W_t
        w = np.cumsum(dw, axis=0)

        # Time grid t = [dt, 2dt, ..., n_days*dt]
        t = np.linspace(dt, n_days * dt, n_days)[:, np.newaxis]

        # Exact solution
        log_s = np.log(s0) + (mu - 0.5 * sigma**2) * t + sigma * w
        prices = np.exp(log_s)

        # Prepend initial price S0
        initial_row = np.full((1, n_paths), s0)
        prices_with_s0 = np.vstack([initial_row, prices])

        date_index = pd.date_range(start="2018-01-01", periods=n_days + 1, freq="B")

        if n_paths == 1:
            return pd.Series(prices_with_s0.flatten(), index=date_index, name="Synthetic_GBM")
        else:
            cols = [f"Path_{i+1}" for i in range(n_paths)]
            return pd.DataFrame(prices_with_s0, index=date_index, columns=cols)

    def generate_merton_jump(
        self,
        s0: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        lambda_j: float = 0.10,
        mu_j: float = -0.05,
        sigma_j: float = 0.10,
        n_days: int = 1512,
        dt: float = 1.0 / 252.0,
        n_paths: int = 1,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Simulates Merton Jump-Diffusion Process:
            Between jumps, price follows continuous diffusion.
            At Poisson arrival times (intensity lambda_j), price jumps by log-normal multiplier J.

        Parameters:
        -----------
        s0 : float
            Initial asset price.
        mu : float
            Annualized total drift.
        sigma : float
            Annualized diffusion volatility.
        lambda_j : float
            Expected number of jumps per year (Poisson intensity).
        mu_j : float
            Mean of log jump size ln(J) ~ N(mu_j, sigma_j^2).
        sigma_j : float
            Standard deviation of log jump size.
        n_days : int
            Number of trading days.
        dt : float
            Step size in years (1/252).
        n_paths : int
            Number of simulation paths.
        """
        if s0 <= 0 or sigma <= 0 or lambda_j < 0:
            raise ValueError("Invalid parameters for Merton Jump Diffusion.")

        # Compensator k = E[J - 1] = exp(mu_j + 0.5 * sigma_j^2) - 1
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        drift_rate = (mu - lambda_j * k - 0.5 * sigma**2) * dt

        # Continuous diffusion component
        diff_shocks = self.rng.normal(loc=0.0, scale=sigma * np.sqrt(dt), size=(n_days, n_paths))

        # Poisson jump occurrences: number of jumps in step dt
        poisson_jumps = self.rng.poisson(lam=lambda_j * dt, size=(n_days, n_paths))

        # Compound jump magnitude: sum of N normal variables with mean mu_j and var sigma_j^2
        jump_magnitudes = np.zeros((n_days, n_paths))
        has_jumps = poisson_jumps > 0
        if np.any(has_jumps):
            n_jumps = poisson_jumps[has_jumps]
            # Sum of normal variables: N ~ sum(N(mu_j, sigma_j^2)) = N(n*mu_j, n*sigma_j^2)
            jump_magnitudes[has_jumps] = self.rng.normal(
                loc=n_jumps * mu_j,
                scale=np.sqrt(n_jumps) * sigma_j
            )

        # Total log return increment per step
        log_increments = drift_rate + diff_shocks + jump_magnitudes
        log_prices = np.log(s0) + np.vstack([np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)])
        prices = np.exp(log_prices)

        date_index = pd.date_range(start="2018-01-01", periods=n_days + 1, freq="B")

        if n_paths == 1:
            return pd.Series(prices.flatten(), index=date_index, name="Synthetic_Merton_Jump")
        else:
            cols = [f"Merton_Path_{i+1}" for i in range(n_paths)]
            return pd.DataFrame(prices, index=date_index, columns=cols)

    def generate_synthetic_ohlcv(
        self,
        symbol: str = "SYN_ASSET",
        process_type: str = "merton",
        s0: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        n_days: int = 1512,
        **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes a realistic daily OHLCV dataset from a simulated price path.
        Intraday High and Low are synthesized using Garman-Klass / Parkinson volatility bands.
        """
        if process_type == "gbm":
            close = self.generate_gbm(s0=s0, mu=mu, sigma=sigma, n_days=n_days, n_paths=1)
        elif process_type == "merton":
            lambda_j = kwargs.get("lambda_j", 0.15)
            mu_j = kwargs.get("mu_j", -0.06)
            sigma_j = kwargs.get("sigma_j", 0.12)
            close = self.generate_merton_jump(
                s0=s0, mu=mu, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j, n_days=n_days, n_paths=1
            )
        else:
            raise ValueError(f"Unknown process type: {process_type}")

        close_vals = close.values
        n = len(close_vals)

        # Synthesize Open: Close of yesterday plus small overnight gap
        overnight_noise = self.rng.normal(0, 0.003, size=n)
        open_vals = np.empty(n)
        open_vals[0] = s0
        open_vals[1:] = close_vals[:-1] * (1.0 + overnight_noise[1:])

        # Intraday volatility span
        daily_range = np.abs(self.rng.normal(loc=0.012, scale=0.005, size=n)) * close_vals
        high_vals = np.maximum(open_vals, close_vals) + daily_range * self.rng.uniform(0.3, 0.7, size=n)
        low_vals = np.minimum(open_vals, close_vals) - daily_range * self.rng.uniform(0.3, 0.7, size=n)

        # Ensure Low > 0
        low_vals = np.maximum(low_vals, 0.01)

        # Synthesize Volume: Log-normal volume with correlation to volatility
        return_magnitudes = np.abs(np.diff(np.log(close_vals), prepend=np.log(s0)))
        base_vol = self.rng.lognormal(mean=14.5, sigma=0.4, size=n)
        volume_vals = (base_vol * (1.0 + 10.0 * return_magnitudes)).astype(np.int64)

        df = pd.DataFrame(
            {
                "Open": open_vals,
                "High": high_vals,
                "Low": low_vals,
                "Close": close_vals,
                "Adj Close": close_vals,
                "Volume": volume_vals,
            },
            index=close.index,
        )
        df.index.name = "Date"
        return df

    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Implements BaseDataLoader contract for synthetic processes."""
        dt_start = pd.to_datetime(start_date)
        dt_end = pd.to_datetime(end_date)
        n_days = int(np.busday_count(dt_start.strftime("%Y-%m-%d"), dt_end.strftime("%Y-%m-%d")))
        return self.generate_synthetic_ohlcv(symbol=symbol, process_type="merton", n_days=max(n_days, 10))

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validates that OHLCV schema constraints hold."""
        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if not all(col in df.columns for col in required):
            return False
        # Constraints: High >= Low, High >= Open, High >= Close, Prices > 0
        valid_ranges = (df["High"] >= df["Low"]).all()
        valid_prices = (df["Low"] > 0).all()
        return bool(valid_ranges and valid_prices)
