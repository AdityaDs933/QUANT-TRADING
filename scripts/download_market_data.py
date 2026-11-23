"""
Market Data Download and Cache CLI Script
Authored by: Anhum (15% Component)

Reads target assets from config.yaml and caches them to data/raw.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_parser import ConfigParser
from src.data.real_data_loader import RealMarketDataLoader
from src.data.synthetic_generator import StochasticProcessGenerator


def main() -> None:
    print("=" * 60)
    print(" INGESTING MARKET DATA (REAL & SYNTHETIC)")
    print("=" * 60)

    cfg = ConfigParser("config/config.yaml")
    equities = cfg.get("assets", "equities", default=["AAPL", "MSFT", "NVDA"])
    benchmarks = cfg.get("assets", "benchmarks", default=["SPY", "QQQ"])
    start_date = cfg.get("data", "start_date", default="2018-01-01")
    end_date = cfg.get("data", "end_date", default="2024-12-31")

    all_symbols = equities + benchmarks
    print(f"Target Symbols: {all_symbols}")
    print(f"Date Horizon:   {start_date} to {end_date}\n")

    loader = RealMarketDataLoader(cache_dir="data/raw", use_cache=True)

    print("--- 1. Downloading / Caching Real-World Assets ---")
    for sym in all_symbols:
        try:
            df = loader.load_data(sym, start_date, end_date)
            print(f"  [OK] {sym:<6} - {len(df)} bars cached from {df.index.min().date()} to {df.index.max().date()}")
        except Exception as e:
            print(f"  [FAIL] {sym:<6} - Error: {e}")

    print("\n--- 2. Generating Synthetic Benchmark Datasets ---")
    synth = StochasticProcessGenerator(random_seed=42)

    # Generate Synthetic GBM
    df_gbm = synth.generate_synthetic_ohlcv(
        symbol="SYN_GBM", process_type="gbm", s0=100.0, mu=0.08, sigma=0.20, n_days=1512
    )
    gbm_path = Path("data/raw/SYN_GBM_2018-01-01_2024-12-31.parquet")
    df_gbm.to_parquet(gbm_path)
    print(f"  [OK] SYN_GBM    - {len(df_gbm)} synthetic bars saved to {gbm_path.name}")

    # Generate Synthetic Merton Jump Diffusion
    df_merton = synth.generate_synthetic_ohlcv(
        symbol="SYN_MERTON",
        process_type="merton",
        s0=100.0,
        mu=0.08,
        sigma=0.20,
        lambda_j=0.15,
        mu_j=-0.06,
        sigma_j=0.12,
        n_days=1512,
    )
    merton_path = Path("data/raw/SYN_MERTON_2018-01-01_2024-12-31.parquet")
    df_merton.to_parquet(merton_path)
    print(f"  [OK] SYN_MERTON - {len(df_merton)} synthetic bars saved to {merton_path.name}")

    print("\n" + "=" * 60)
    print(" DATA INGESTION AND STOCHASTIC SIMULATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
