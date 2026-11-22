# System Architecture & Pipeline Design
**Architected by: Aditya (70% Component)**
**Documented by: Anhum (15% Component)**

---

## 1. High-Level Modular Design

The quantitative trading system follows a strict unidirectional data-flow architecture designed to make data snooping and lookahead bias mathematically impossible:

```
┌──────────────────┐
│  Data Ingestion  │ ──► Real Market Data (yfinance) + Synthetic Generators (GBM / Merton)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Preprocessing   │ ──► Missing value handling, split adjustments, stationarity tests (ADF/KPSS)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Feature Engine   │ ──► Multi-timeframe momentum, mean-reversion, volatility, and volume factors
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Leakage Shield   │ ──► Purged & Embargoed Walk-Forward Time-Series Splitter + VIF Pruning
└────────┬─────────┘
         ▼
┌──────────────────┐
│ ML & Regimes     │ ──► Supervised Classifiers/Regressors conditioned on HMM/GMM Regimes
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Strategy Engine  │ ──► Signal Generation + Dynamic Position Sizing (Kelly / Volatility Targeting)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Backtest & Risk  │ ──► Friction-aware accounting (Slippage, Commission, Lag t+1) + VaR / Sharpe
└──────────────────┘
```

---

## 2. Directory Layout & Module Responsibilities

* `config/config.yaml`: Centralized configuration repository for assets, dates, model hyperparameters, and backtest constraints.
* `src/utils/finance_math.py`: Pure, tested mathematical primitives for return calculation, compounding, annualization, and spreads.
* `src/utils/config_parser.py`: Safe YAML parser with schema verification and typed lookups.
* `src/utils/base_interfaces.py`: Abstract Base Classes (`BaseDataLoader`, `BaseQuantModel`, `BaseStrategy`) enforcing modular design patterns.
* `scripts/verify_env.py`: Automated sanity check verifying environment dependencies, directory write permissions, and config integrity.
