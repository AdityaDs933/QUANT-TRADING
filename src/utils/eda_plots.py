"""
EDA Visualization Utilities — Styled Charts for Market Dynamics
Authored by: Anhum (15% Component)

Provides reusable, publication-quality Matplotlib/Seaborn chart functions:
- Multi-asset price & normalized wealth comparison
- Volatility clustering (rolling vol + return bars)
- Return distribution histograms with Gaussian overlay
- Correlation heatmap
- Underwater drawdown curves
"""

import warnings
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# Consistent styling
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_normalized_prices(
    price_dict: Dict[str, pd.Series],
    title: str = "Normalized Price Trajectories (Base=100)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots each asset's price normalized to 100 at the start date,
    enabling fair visual comparison regardless of absolute price scale.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (label, prices) in enumerate(price_dict.items()):
        normalized = prices / prices.iloc[0] * 100.0
        ax.plot(normalized.index, normalized.values, label=label,
                color=PALETTE[idx % len(PALETTE)], lw=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalized Price (Base = 100)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(FIGURE_DIR / save_path, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {save_path}")
    plt.close(fig)


def plot_volatility_clustering(
    returns: pd.Series,
    symbol: str = "Asset",
    window: int = 21,
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel chart:
    Top: Daily return bars (grey) with rolling volatility overlay.
    Bottom: Rolling 21-day annualized volatility.
    Visually demonstrates ARCH volatility clustering.
    """
    rolling_vol = returns.rolling(window=window, min_periods=5).std() * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].bar(returns.index, returns.values, color="grey", alpha=0.5, label="Daily Returns", width=1)
    axes[0].set_title(f"{symbol}: Return Series & Rolling Volatility (Clustering Evidence)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Daily Log Return")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].legend(frameon=True, fontsize=9)

    axes[1].fill_between(rolling_vol.index, rolling_vol.values, alpha=0.4, color="#d62728")
    axes[1].plot(rolling_vol.index, rolling_vol.values, color="#d62728", lw=1.5, label=f"{window}d Rolling Ann. Vol")
    axes[1].set_ylabel("Annualized Volatility")
    axes[1].set_xlabel("Date")
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        fig.savefig(FIGURE_DIR / save_path, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {save_path}")
    plt.close(fig)


def plot_return_distribution(
    returns_dict: Dict[str, pd.Series],
    save_path: Optional[str] = None,
) -> None:
    """
    Plots return histograms for multiple assets overlaid with the fitted Gaussian.
    Shows how real returns exhibit fat tails vs. the theoretical bell curve.
    """
    n = len(returns_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for idx, (label, ret) in enumerate(returns_dict.items()):
        ax = axes[idx]
        clean = ret.dropna()
        mu, sigma = clean.mean(), clean.std()

        ax.hist(clean, bins=80, density=True, color=PALETTE[idx % len(PALETTE)],
                alpha=0.65, label="Empirical Returns")

        # Gaussian overlay
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 300)
        gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, gaussian, color="black", lw=2, ls="--", label="Normal Fit")

        from scipy.stats import kurtosis as kurt_fn
        k = float(kurt_fn(clean, fisher=False))
        ax.set_title(f"{label}\nKurtosis = {k:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Log Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle("Return Distributions: Fat Tails vs. Gaussian Assumption", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(FIGURE_DIR / save_path, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {save_path}")
    plt.close(fig)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Cross-Asset Return Correlation Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Renders an annotated seaborn heatmap of the correlation matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(FIGURE_DIR / save_path, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {save_path}")
    plt.close(fig)


def plot_underwater_drawdown(
    drawdown_dict: Dict[str, pd.Series],
    title: str = "Underwater Drawdown Curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots the underwater (below high-water-mark) drawdown curve for each asset.
    Shows magnitude and persistence of losses relative to portfolio peaks.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (label, dd_series) in enumerate(drawdown_dict.items()):
        ax.fill_between(dd_series.index, dd_series.values * 100, 0,
                        alpha=0.3, color=PALETTE[idx % len(PALETTE)])
        ax.plot(dd_series.index, dd_series.values * 100, label=label,
                color=PALETTE[idx % len(PALETTE)], lw=1.2)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(FIGURE_DIR / save_path, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {save_path}")
    plt.close(fig)
