# Quantitative Trading Financial Data Dictionary
**Documented by: Anhum (15% Component)**

---

## 1. Raw Market Data Schema (OHLCV)
All empirical equities and ETFs follow the standard daily trading bar schema:

| Column Name | Data Type | Physical Meaning | Econometric Role | Constraints |
| :--- | :---: | :--- | :--- | :--- |
| `Date` / `Timestamp` | `datetime64[ns]` | Trading session close timestamp (UTC/EST) | Time index; monotonically increasing | Unique, no duplicates, sorted |
| `Open` | `float64` | Price of the first transaction of the day | Session opening price | $\text{Open} > 0$ |
| `High` | `float64` | Highest transaction price during session | Intraday price ceiling | $\text{High} \ge \max(\text{Open}, \text{Close})$ |
| `Low` | `float64` | Lowest transaction price during session | Intraday price floor | $0 < \text{Low} \le \min(\text{Open}, \text{Close})$ |
| `Close` | `float64` | Unadjusted final transaction price of session | Base for historical settlement | $\text{Close} > 0$ |
| `Adj Close` | `float64` | Price adjusted for stock splits and dividend distributions | **Primary price used for returns and ML modeling** | $\text{Adj Close} > 0$ |
| `Volume` | `int64` / `float64` | Total number of shares transacted | Liquidity proxy; weight for VWAP | $\text{Volume} \ge 0$ |

> **Crucial Rule on Corporate Actions**: We use `Adj Close` rather than `Close` for returns calculation to prevent artificial price drops caused by stock splits (e.g. Apple's 4-for-1 split in 2020) and dividend distribution.

---

## 2. Mathematical Return Types

| Metric Name | Mathematical Definition | Key Property | When to Use |
| :--- | :--- | :--- | :--- |
| **Arithmetic Return** ($R_t$) | $R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$ | Cross-sectional portfolio linearity: $R_p = \sum w_i R_i$ | Real money accounting, capital allocation, PnL calculations |
| **Logarithmic Return** ($r_t$) | $r_t = \ln(P_t / P_{t-1}) = \ln(P_t) - \ln(P_{t-1})$ | Intertemporal additivity: $r_{0, T} = \sum_{t=1}^T r_t$ | Statistical modeling, ML targets, volatility estimation, hypothesis testing |

---

## 3. Order Book & Market Microstructure Friction

| Friction Parameter | Symbol | Industry Standard Default | Impact on Strategy |
| :--- | :---: | :---: | :--- |
| **Bid Price** | $P_{\text{bid}}$ | Market observation | Price received when selling via market order |
| **Ask Price** | $P_{\text{ask}}$ | Market observation | Price paid when buying via market order |
| **Quoted Spread** | $S_{\text{quoted}}$ | $P_{\text{ask}} - P_{\text{bid}}$ | Absolute transaction penalty |
| **Relative Spread** | $S_{\text{rel}}$ | $\frac{P_{\text{ask}} - P_{\text{bid}}}{P_{\text{mid}}}$ | Proportional transaction friction |
| **Brokerage Commission** | $c_{\text{bps}}$ | 10 bps ($0.0010$) | Exchange clearing and broker fee per turnover |
| **Execution Slippage** | $s_{\text{bps}}$ | 5 bps ($0.0005$) | Market impact penalty during aggressive execution |
