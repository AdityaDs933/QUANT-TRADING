"""
Batch Preprocessing & Stationarity Analysis CLI
Authored by: Anhum (15%) & Aditya (70%)

Processes all raw datasets and outputs:
1. Cleaned datasets in data/processed
2. Detailed stationarity diagnostic report (ADF, KPSS, Fractional Diff)
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils.config_parser import ConfigParser
from src.data.preprocess import DataPreprocessor
from src.utils.stationarity import StationarityTester, FractionalDifferentiator
from src.utils.finance_math import calculate_log_returns


def main() -> None:
    print("=" * 70)
    print(" DAY 3: DATA PREPROCESSING & STATIONARITY ANALYSIS PIPELINE")
    print("=" * 70)

    cfg = ConfigParser("config/config.yaml")
    equities = cfg.get("assets", "equities", default=["AAPL", "MSFT", "NVDA"])
    benchmarks = cfg.get("assets", "benchmarks", default=["SPY", "QQQ"])
    start_date = cfg.get("data", "start_date", default="2018-01-01")
    end_date = cfg.get("data", "end_date", default="2024-12-31")

    symbols = equities + benchmarks + ["SYN_GBM", "SYN_MERTON"]
    preprocessor = DataPreprocessor()

    print(f"\n--- 1. Cleaning & Imputing {len(symbols)} Datasets ---")
    processed_dfs = {}
    for sym in symbols:
        try:
            df_proc = preprocessor.process_and_save(sym, start_date, end_date)
            processed_dfs[sym] = df_proc
            print(f"  [PROCESSED] {sym:<12} -> {len(df_proc)} clean daily bars saved to data/processed/")
        except Exception as e:
            print(f"  [ERROR]     {sym:<12} -> {e}")

    print("\n" + "=" * 70)
    print(" 2. ECONOMETRIC STATIONARITY AUDIT (RAW PRICE vs LOG RETURNS)")
    print("=" * 70)

    diag_rows = []
    for sym in ["AAPL", "SPY", "SYN_MERTON"]:
        df = processed_dfs[sym]
        price_series = df["Adj Close"]
        ret_series = calculate_log_returns(price_series)

        # Raw Price diagnosis
        diag_price = StationarityTester.diagnose_stationarity(price_series, name=f"{sym} (Raw Price)")
        # Log Returns diagnosis
        diag_ret = StationarityTester.diagnose_stationarity(ret_series, name=f"{sym} (Log Returns)")

        diag_rows.extend([diag_price, diag_ret])

    diag_df = pd.DataFrame(diag_rows)
    print(diag_df[["name", "adf_stat", "adf_pval", "kpss_stat", "kpss_pval", "final_diagnosis"]].to_string(index=False))

    print("\n" + "=" * 70)
    print(" 3. FRACTIONAL DIFFERENTIATION: OPTIMAL MEMORY-PRESERVING d*")
    print("=" * 70)
    print("Searching for min d in [0.0, 1.0] where ADF rejects unit root at 95% confidence:\n")

    for sym in ["AAPL", "SPY"]:
        series = processed_dfs[sym]["Adj Close"]
        min_d, sweep_df = FractionalDifferentiator.find_min_d_for_stationarity(series)
        row_at_min_d = sweep_df[sweep_df["d"] == round(min_d, 2)].iloc[0]
        row_at_d1 = sweep_df[sweep_df["d"] == 1.0].iloc[0]

        print(f"Asset: {sym}")
        print(f"  Optimal d* for stationarity:      d = {min_d:.2f}")
        print(f"  Correlation Memory at d*:         {row_at_min_d['correlation_memory']:.4f}  (Retains {row_at_min_d['correlation_memory']*100:.1f}% memory!)")
        print(f"  Correlation Memory at standard d=1: {row_at_d1['correlation_memory']:.4f}  (Standard returns lose nearly ALL price memory!)")
        print(f"  ADF Stat at d*:                   {row_at_min_d['adf_stat']:.4f} (p-val: {row_at_min_d['p_val']:.4e})")
        print()

    print("=" * 70)
    print(" DAY 3 PREPROCESSING & STATIONARITY PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
