"""
Batch EDA Pipeline Runner
Authored by: Aditya (70%) + Tanveer (15%)

Runs the complete Day 4 EDA pipeline on all processed assets:
1. Distributional statistics summary table (mean, vol, skew, kurt, Jarque-Bera)
2. ARCH volatility clustering tests (Engle LM test)
3. Cross-asset correlation matrix
4. Drawdown profiles
5. Saves all visualizations to reports/figures/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.utils.config_parser import ConfigParser
from src.utils.finance_math import calculate_log_returns, calculate_arithmetic_returns
from src.utils.eda_engine import DistributionAnalyzer, RollingMarketMetrics, DrawdownAnalyzer, CrossAssetAnalyzer
from src.utils.eda_plots import (
    plot_normalized_prices,
    plot_volatility_clustering,
    plot_return_distribution,
    plot_correlation_heatmap,
    plot_underwater_drawdown,
)


def load_all_assets(symbols: list, processed_dir: str = "data/processed") -> dict:
    out = {}
    for sym in symbols:
        path = Path(processed_dir) / f"{sym}_processed.parquet"
        if path.exists():
            out[sym] = pd.read_parquet(path)
    return out


def main() -> None:
    print("=" * 70)
    print(" DAY 4: EXPLORATORY DATA ANALYSIS & MARKET DYNAMICS")
    print("=" * 70)

    cfg = ConfigParser("config/config.yaml")
    equities = cfg.get("assets", "equities", default=["AAPL", "MSFT", "NVDA"])
    benchmarks = cfg.get("assets", "benchmarks", default=["SPY", "QQQ"])
    rf_rate = cfg.get("backtest", "risk_free_rate", default=0.04)

    all_symbols = equities + benchmarks
    dfs = load_all_assets(all_symbols)

    # Build return series dict
    log_rets = {sym: calculate_log_returns(dfs[sym]["Adj Close"]) for sym in dfs}
    arith_rets = {sym: calculate_arithmetic_returns(dfs[sym]["Adj Close"]) for sym in dfs}
    prices = {sym: dfs[sym]["Adj Close"] for sym in dfs}

    # ─────────────────────────────────────────────────────────────────
    print("\n--- 1. DISTRIBUTIONAL STATISTICS SUMMARY ---")
    dist_records = []
    for sym, ret in log_rets.items():
        summary = DistributionAnalyzer.describe_returns(ret, label=sym)
        dist_records.append(summary)
    dist_df = pd.DataFrame(dist_records).set_index("label")
    print(dist_df[[
        "n_obs", "ann_return", "ann_vol", "skewness", "kurtosis",
        "is_leptokurtic", "jarque_bera_pval", "rejects_normality",
        "pct_1", "pct_5"
    ]].round(4).to_string())

    # ─────────────────────────────────────────────────────────────────
    print("\n--- 2. ARCH VOLATILITY CLUSTERING TESTS ---")
    for sym, ret in log_rets.items():
        arch = DistributionAnalyzer.arch_test(ret, nlags=10)
        flag = "[ARCH EFFECTS DETECTED]" if arch["has_arch_effects"] else "[No ARCH effects]"
        print(f"  {sym:<6}  LM Stat={arch['lm_stat']:.3f}  p-val={arch['lm_pval']:.4e}  {flag}")

    # ─────────────────────────────────────────────────────────────────
    print("\n--- 3. DRAWDOWN PROFILES ---")
    for sym in equities:
        profile = DrawdownAnalyzer.full_drawdown_profile(arith_rets[sym])
        print(f"  {sym:<6}  Max Drawdown={profile['max_drawdown_pct']}  "
              f"Duration={profile['max_drawdown_duration_days']} days")

    # ─────────────────────────────────────────────────────────────────
    print("\n--- 4. CROSS-ASSET CORRELATION MATRIX ---")
    corr_matrix = CrossAssetAnalyzer.correlation_matrix(log_rets)
    print(corr_matrix.round(3).to_string())
    avg_corr = CrossAssetAnalyzer.average_pairwise_correlation(corr_matrix)
    print(f"\n  Average Pairwise Correlation: {avg_corr:.3f}")
    if avg_corr > 0.70:
        print("  [NOTE] High mean correlation — moderate diversification benefit across this universe.")

    # ─────────────────────────────────────────────────────────────────
    print("\n--- 5. GENERATING & SAVING VISUALIZATIONS ---")

    # 5a. Normalized price trajectories
    plot_normalized_prices(prices, save_path="normalized_prices.png")

    # 5b. Volatility clustering chart for AAPL and NVDA
    plot_volatility_clustering(log_rets["AAPL"], symbol="AAPL", window=21, save_path="aapl_vol_clustering.png")
    plot_volatility_clustering(log_rets["NVDA"], symbol="NVDA", window=21, save_path="nvda_vol_clustering.png")

    # 5c. Return distribution comparison
    plot_return_distribution(
        {sym: log_rets[sym] for sym in ["AAPL", "SPY"]},
        save_path="return_distributions.png"
    )

    # 5d. Full correlation heatmap
    plot_correlation_heatmap(corr_matrix, save_path="correlation_heatmap.png")

    # 5e. Underwater drawdown curves (equity portfolio view)
    dd_series = {}
    for sym in all_symbols:
        wealth = (1 + arith_rets[sym]).cumprod()
        dd_series[sym] = DrawdownAnalyzer.compute_drawdown_series(wealth)
    plot_underwater_drawdown(dd_series, save_path="underwater_drawdown.png")

    print("\n" + "=" * 70)
    print(" DAY 4 EDA COMPLETE — All charts saved to reports/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
