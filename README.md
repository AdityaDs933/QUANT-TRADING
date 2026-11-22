# Machine Learning-Based Quantitative Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **An institutional-grade quantitative trading system combining ML alpha models (XGBoost/LightGBM), unsupervised regime detection (HMM), purged walk-forward cross-validation, and a friction-aware backtesting engine.**

---

## 📌 Project Overview

This repository houses an end-to-end, industry-oriented quantitative trading research and execution framework developed for **Applied Predictive Analysis**. Rather than applying machine learning as a naive curve-fitting exercise, the system addresses core financial econometrics challenges:
* **Leakage-Free Validation**: Implementation of purged and embargoed walk-forward cross-validation to prevent lookahead and overlapping label bias.
* **Stationarity & Memory**: Stationarity transformations preserving predictive memory for noisy financial time series.
* **Market Regime Awareness**: Unsupervised clustering (GMM) and Hidden Markov Models (HMM) to dynamically toggle strategy behaviors across trending and volatile market states.
* **Microstructure Friction**: Realistic event-aware backtesting accounting for bid-ask spreads, transaction commissions, execution latency ($t+1$), and linear slippage.
* **Risk-Adjusted Sizing**: Mathematical position sizing via fractional Kelly Criterion and volatility targeting, evaluated via Sharpe, Sortino, Calmar, and Value at Risk (VaR/CVaR).

---

## 👥 Team Structure & Role Distribution

| Team Member | Role & Workload | Core Responsibilities |
| :--- | :---: | :--- |
| **Aditya (Lead)** | **70%** | Quantitative Research, Core ML Architecture, Walk-Forward Splitter, Strategy Engine, Backtester, Risk Attribution |
| **Anhum** | **15%** | Data Ingestion Pipelines (Real & Synthetic), Preprocessing, Technical Indicator Utilities, Documentation |
| **Tanveer** | **15%** | Exploratory Data Analysis (EDA), Statistical Testing, Performance Metrics Compilation, Visual Tear Sheets, Slide Deck |

---

## 🗓️ 14-Day Incremental Roadmap

```
Phase 1: Foundations & Ingestion (Days 1–4)
  ├── Day 01: System Architecture, Environment Setup & Financial Foundations
  ├── Day 02: Market Data Acquisition (Real-World & Synthetic Generators)
  ├── Day 03: Data Quality, Preprocessing & Stationarity Analysis
  └── Day 04: Exploratory Data Analysis (EDA) & Market Dynamics

Phase 2: Features & Baseline Models (Days 5–7)
  ├── Day 05: Quantitative Feature Engineering & Alpha Factor Design
  ├── Day 06: Feature Selection, Multicollinearity & Data Leakage Prevention
  └── Day 07: Baseline Quantitative Models (Linear, Logistic & Regularized)

Phase 3: Advanced ML & Regime Detection (Days 8–10)
  ├── Day 08: Non-Linear Models (Decision Trees & Random Forests)
  ├── Day 09: Advanced Gradient Boosting (XGBoost/LightGBM) & SHAP Interpretability
  └── Day 10: Unsupervised Learning & Market Regime Detection (GMM & HMM)

Phase 4: Strategy, Backtesting & Delivery (Days 11–14)
  ├── Day 11: Trading Strategy Engine & Dynamic Position Sizing
  ├── Day 12: Realistic Backtesting Framework (Friction, Slippage & Execution)
  ├── Day 13: Quantitative Risk Management, Attribution & Stress Testing
  └── Day 14: System Integration, Interactive Tear Sheet & Final Presentation
```

---

## 🏗️ Repository Architecture

```text
QUANT PROJECT/
├── config.yaml                    # Global strategy & model configurations
├── requirements.txt               # Reproducible environment dependencies
├── main.py                        # Master pipeline CLI runner
├── data/
│   ├── raw/                       # Cached market & synthetic datasets
│   └── processed/                 # Engineered features with zero lookahead bias
├── src/
│   ├── data/                      # Ingestion & stochastic generators (GBM, Merton)
│   ├── features/                  # Technical indicators, targets & selection
│   ├── models/                    # Baselines, Ensembles, XGBoost, HMM Regimes
│   ├── strategy/                  # Signal generation & dynamic position sizing
│   ├── backtest/                  # Friction-aware backtesting & risk engine
│   └── utils/                     # Walk-forward validation, stationarity, SHAP
├── notebooks/                     # Step-by-step analytical walkthroughs
├── reports/
│   ├── figures/                   # Tear sheets, drawdowns, and SHAP plots
│   └── final_report.md            # In-depth technical research report
└── presentation/                  # Slide deck for academic defense
```

---

## 🚀 Quickstart & Setup

*(Will be continuously populated as Day 1 through Day 14 are executed)*

```bash
# Clone the repository
git clone https://github.com/<your-username>/quant-trading-ml.git
cd quant-trading-ml

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
