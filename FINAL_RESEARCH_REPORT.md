# AI ETF Quantitative Research System — Final Research Report

> **Audit Date:** 2026-09-27  
> **Report Status:** FINAL — DO NOT MODIFY  
> **Research Conclusion:** The tested strategy did not demonstrate a robust post-cost edge. Kill-gate verdict: **FAILED**.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Question](#2-research-question)
3. [System Overview](#3-system-overview)
4. [Dataset](#4-dataset)
5. [Features](#5-features)
6. [Labeling](#6-labeling)
7. [Models](#7-models)
8. [Validation Methodology](#8-validation-methodology)
9. [Realism Layer](#9-realism-layer)
10. [Kill Gate](#10-kill-gate)
11. [Original Results](#11-original-results)
12. [Methodological Corrections](#12-methodological-corrections)
13. [Reduced Trade-Frequency Experiment](#13-reduced-trade-frequency-experiment)
14. [Fold 4 Investigation](#14-fold-4-investigation)
15. [Most Recent Period (2022–2026)](#15-most-recent-period-20222026)
16. [What Actually Worked](#16-what-actually-worked)
17. [What Did Not Work](#17-what-did-not-work)
18. [Final Answer: How Well Did It Work?](#18-final-answer-how-well-did-it-work)
19. [Final Conclusion](#19-final-conclusion)
20. [Future Work](#20-future-work)
21. [Reproducibility](#21-reproducibility)
22. [Project Status](#22-project-status)

---

## 1. Executive Summary

This project attempted to answer a specific quantitative trading question using artificial intelligence: **Can a machine-learning model predict short-term price movements in Indian Exchange-Traded Funds (ETFs) to generate signals that beat a simple Buy-and-Hold strategy after accounting for realistic trading costs?**

To answer this question, we engineered a complete, production-grade research platform that:
- Ingested approximately 20 years of daily price data (2006–2026) for 10 of India's most liquid ETFs via the official **Groww Python SDK (`growwapi`)**.
- Engineered **31 technical, volatility, volume, and market-context features** per ETF per trading day.
- Applied **Causal Market Regime Detection** to categorize market state (Trending Up, Trending Down, High Volatility, Sideways) without lookahead.
- Implemented **Triple-Barrier Labeling** (7-day and 14-day forecast horizons) to define discrete classification targets: Take-Profit (TP), Stop-Loss (SL), or Timeout (TIME).
- Trained **XGBoost classifiers**, along with Logistic Regression and Random Forest baselines, inside a 5-fold **Purged Walk-Forward Validation** framework.
- Incorporated a **Realistic Cost Layer** reflecting Groww delivery brokerage, Securities Transaction Tax (STT), stamp duty, exchange fees, SEBI charges, GST, fixed slippage, and liquidity-scaled bid-ask spreads.
- Evaluated performance against pre-registered, locked **Kill-Gate Criteria** (Sharpe ratio $\ge 0.50$, win rate $\ge 40\%$, outperforming Buy-and-Hold in $\ge 50\%$ of folds).

### Final Result
The trading strategy **FAILED** its pre-registered quality threshold:
- **7-Day Horizon:** **0 out of 5 folds passed** ($0\%$).
- **14-Day Horizon (SL-Filtered):** **1 out of 5 folds passed** ($20\%$).

Only Fold 4 (2018–2022) achieved a positive post-cost Sharpe ratio (+0.86). Deep-dive analysis revealed that this single passing window was heavily concentrated in commodity ETFs (Gold and Silver) during the COVID-19 crash and recovery period. The strategy failed in all other historical windows, including the most recent period (2022–2026).

### Key Takeaway
The engineering system—data ingestion, feature pipeline, regime logic, labeling, purging, cost accounting, and interactive visualization—was built and operated correctly. However, **the underlying predictive signal was too weak to overcome transaction costs and bid-ask spreads across diverse market environments.**

> **Disclaimer:** This project answered a specific timing question for short-term tactical trading. It was **not** designed to assess the viability of long-term passive ETF investing. These results do not constitute financial or investment advice.

---

## 2. Research Question

### Precise Statement

> **"Can an XGBoost classifier, trained on technical and market-context features from daily Indian ETF price data, generate long-only tactical trading signals that produce a robust positive post-cost Sharpe ratio ($\ge 0.50$) — consistently across multiple time periods and ETF asset classes — outperforming a passive Buy-and-Hold baseline?"**

### Distinction of Scope

It is critical to separate two distinct financial questions:

1. **Tactical Market Timing (Tested by this project):**  
   *"Can a machine learning model accurately time 7-day or 14-day entries into Indian equity and commodity ETFs, achieving excess risk-adjusted returns after brokerage, taxes, slippage, and spreads?"*

2. **Long-Term Asset Allocation (NOT tested by this project):**  
   *"Which ETF should an investor buy and hold over a multi-year horizon for retirement or wealth accumulation?"*

This project evaluated **Question 1 ONLY**. A negative answer to Question 1 does **not** imply that ETFs are poor long-term investment vehicles; it demonstrates that short-term directional timing of ETFs using daily technical features did not yield a tradable post-cost edge.

---

## 3. System Overview

The research system was structured as a modular, 10-stage pipeline:

```
Groww API (SDK + Instrument Master)
    │
    ▼  [Phase 1]
PostgreSQL (instruments, ohlcv_daily)
    │
    ▼  [Phase 2]
Feature Engineering (31 features per ETF-date)
    │
    ▼  [Phase 3]
Causal Market Regime Detection (4 regimes, zero lookahead)
    │
    ▼  [Phase 3]
Triple-Barrier Labeling (H=7, H=14; TP / SL / TIME)
    │
    ▼  [Phase 4]
Model Training (XGBoost + Logistic Regression + Random Forest)
    │
    ▼  [Phase 5]
Walk-Forward Validation (5 Folds with Purging & Embargoing)
    │
    ▼  [Phase 5]
Realism & Cost Modeling (Brokerage, STT, Spread, Slippage)
    │
    ▼  [Phase 5 / Post-Phase 5]
Calibration & Leakage Audit
    │
    ▼
Kill Gate Assessment & Dashboard Visualization
```

### Component Rationale

| Component | Function | Why It Exists |
|---|---|---|
| **Groww API** | Ingests official NSE CASH instrument master & daily OHLCV | Official data provider for Indian equity market data. |
| **PostgreSQL** | Stores instruments, candles, features, regimes, labels, runs, backtests | Provides an idempotent, auditable data layer for multi-phase workflows. |
| **Feature Engineering** | Computes 31 technical and market-context indicators | Converts raw price/volume time series into tabular ML feature matrices. |
| **Causal Regime Detection** | Classifies market state (`TRENDING_UP`, `SIDEWAYS`, etc.) causally | Allows conditioning model signals on broad market context without lookahead. |
| **Triple-Barrier Labels** | Sets ATR-scaled Take-Profit, Stop-Loss, and Time barriers | Replaces arbitrary fixed-horizon returns with path-dependent trade outcomes. |
| **XGBoost & Baselines** | Classifies barrier outcome probability | Primary non-linear classifier; compared against Linear, Random Forest, and Rule baselines. |
| **Walk-Forward Validation** | Chronological 5-fold testing with purging and embargoing | Prevents lookahead leakage and accounts for temporal non-stationarity. |
| **Realism Layer** | Deducts Groww delivery fees, STT, GST, SEBI fees, spread, slippage | Evaluates net profitability under realistic Indian execution friction. |
| **Calibration Audit** | Checks empirical TP rate vs predicted probability buckets | Ensures model confidence aligns with actual winning probability. |
| **Kill Gate** | Evaluates pre-registered pass/fail criteria across folds | Prevents post-hoc goalpost moving and cherry-picking of favorable backtest splits. |

---

## 4. Dataset

### ETF Universe

The universe consisted of the **top 10 most liquid NSE CASH ETFs**, dynamically selected by ranking candidates on Average Daily Traded Value ($\text{Volume} \times \text{Close}$).

| Category | Trading Symbols |
|---|---|
| **Broad Market (Nifty 50)** | `NIFTYBEES` |
| **Banking Sector** | `BANKBEES` |
| **IT Sector** | `ITBEES` |
| **Nifty Next 50 (Mid-Cap)** | `JUNIORBEES` |
| **Gold Commodities** | `GOLDBEES`, `GOLDIETF`, `HDFCGOLD`, `GOLD1` |
| **Silver Commodities** | `SILVERBEES`, `SILVERIETF` |

### Context Indices (Non-Traded Features)

To provide macro context, daily data for the following indices was ingested:
- **`NIFTY`**: Broad equity benchmark
- **`BANKNIFTY`**: Banking sector benchmark
- **`INDIAVIX`**: Indian market volatility index
- **`NIFTYIT`**: Information technology sector benchmark
- **`NIFTYJR`**: Mid-cap equity benchmark

### Date Range & Folds

The dataset spanned approximately 20 years (mid-2006 to mid-2026), partitioned into 5 chronological walk-forward folds:

| Fold | Start Date | End Date | Span (Years) |
|---|---|---|---|
| **Fold 1** | 2006-07-31 | 2010-08-10 | ~4.0 |
| **Fold 2** | 2010-08-11 | 2014-08-10 | ~4.0 |
| **Fold 3** | 2014-08-11 | 2018-08-20 | ~4.0 |
| **Fold 4** | 2018-08-22 | 2022-08-31 | ~4.0 |
| **Fold 5** | 2022-09-01 | 2026-09-06 | ~4.0 |

### Data Quality & Preprocessing

- **Source:** Groww Python SDK (`growwapi`) querying NSE CASH.
- **Frequency:** Daily close-to-close OHLCV candles.
- **Timezone:** Normalized to Indian Standard Time (`Asia/Kolkata`).
- **Missing Data:** Technical features required a 50-day warm-up period. Context features with missing values were filled with 0.0, and explicit boolean missingness flags (`_is_missing`) were appended.
- **Label Exclusion:** Rows near the end of the time series where the full barrier horizon could not elapse were assigned `NULL` labels and excluded from training and validation.

---

## 5. Features

A total of **31 numerical features** were engineered for each instrument-date pair:

### 1. Technical Indicators (17 features)
- **Returns:** `return_1d`, `return_3d`, `return_5d`, `return_10d`, `return_20d`
- **Moving Averages:** `sma_5`, `sma_20`, `sma_50`
- **Price Ratios:** `price_vs_sma5`, `price_vs_sma20`, `price_vs_sma50`, `sma20_vs_sma50`
- **Oscillators & Momentum:** `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `roc_10`

### 2. Volatility Features (4 features)
- **Realized Volatility:** `volatility_5d`, `volatility_10d`, `volatility_20d`
- **Average True Range:** `atr_14`

### 3. Volume Features (4 features)
- **Raw & Averages:** `volume_today`, `volume_sma20`
- **Volume Dynamics:** `volume_ratio`, `volume_change`

### 4. Market Context Features (6 features + missingness flags)
- **Broad Market Returns:** `nifty50_return_1d`, `banknifty_return_1d`
- **Volatility Context:** `india_vix_level`, `india_vix_change_1d`
- **Sector/Commodity Context:** `sector_return_1d`, `commodity_return_1d`

---

## 6. Labeling

The project utilized Marcos López de Prado's **Triple-Barrier Labeling** methodology to generate discrete target classes.

### Barrier Definitions

For an entry on day $T$ with price $P_T$:
1. **Take-Profit (TP) Barrier:**  
   $$\text{TP Level} = P_T + \left(2.0 \times \text{ATR}_{14, T} \times \sqrt{\frac{\text{Horizon}}{7.0}}\right)$$
2. **Stop-Loss (SL) Barrier:**  
   $$\text{SL Level} = P_T - \left(2.0 \times \text{ATR}_{14, T} \times \sqrt{\frac{\text{Horizon}}{7.0}}\right)$$
3. **Time Barrier:**  
   $T + H$ trading days (where $H = 7$ or $H = 14$).

### Class Assignments

- **Class 0 (`TP`):** Price touches or exceeds the upper barrier before the lower or time barrier.
- **Class 1 (`SL`):** Price touches or falls below the lower barrier before the upper or time barrier.
- **Class 2 (`TIME`):** Neither TP nor SL is breached within $H$ trading days.

---

## 7. Models

### Evaluated Model Suite

1. **XGBoost Classifier (Primary Model):**
   - Hyperparameters: `n_estimators=100`, `max_depth=5`, `learning_rate=0.05`, `eval_metric="mlogloss"`, `random_state=42`.
   - Output: 3-class probability distribution $[P(\text{TP}), P(\text{SL}), P(\text{TIME})]$.

2. **Logistic Regression (Baseline):**
   - Multinomial logistic regression with L2 regularization (`max_iter=1000`).

3. **Random Forest (Baseline):**
   - Ensembled decision trees (`n_estimators=100`, `max_depth=5`, `random_state=42`).

4. **Rule-Based Baselines:**
   - **Buy & Hold:** Continuous long exposure across the 10-ETF universe.
   - **SMA50 Rule:** Long position when Close $> \text{SMA}_{50}$; cash otherwise.

### Strategy Filter Variants (XGBoost Execution Rules)

- **Unfiltered:** Enter long whenever $P(\text{TP}) \ge 0.35$.
- **SL-Filtered (Primary Strategy Variant):** Enter long whenever $P(\text{TP}) \ge 0.35$ **AND** $P(\text{SL}) < 0.30$.

---

## 8. Validation Methodology

To ensure strict chronological integrity and eliminate lookahead bias:

### Purged Walk-Forward Validation

- **Purging:** Training observations whose forward barrier evaluation windows ($[T+1, T+H]$) overlap with the validation start window are removed.
- **Embargoing:** A post-validation gap period is enforced prior to subsequent splits to prevent serial correlation leakage.
- **No Random Cross-Validation:** Standard K-Fold cross-validation was strictly prohibited due to severe temporal overlap between adjacent barrier evaluation windows.

---

## 9. Realism Layer

Every backtested trade accounted for actual execution friction present in Indian ETF trading on Groww.

### Transaction Cost Schedule

| Fee Component | Parameter / Rate |
|---|---|
| **Brokerage** | ₹0 (Groww equity delivery) |
| **STT (Securities Transaction Tax)** | 0.10% on buy, 0.10% on sell (0.20% round-trip) |
| **Stamp Duty** | 0.015% on buy side |
| **Exchange Turnover Fees** | 0.00345% per side |
| **SEBI Turnover Charges** | 0.0001% per side |
| **GST** | 18% on (Brokerage + Exchange + SEBI fees) |
| **Fixed Slippage** | 0.05% (5 bps) per side (0.10% round-trip) |
| **Bid-Ask Spread** | Scaled by Average Daily Traded Value (ADTV):<br>• ADTV $> 10 \text{ Cr}$: 0.05%<br>• $1 \text{ Cr} < \text{ADTV} \le 10 \text{ Cr}$: 0.10%<br>• $\text{ADTV} \le 1 \text{ Cr}$: 0.20% |

**Total Round-Trip Friction:** Approximately **0.35% to 0.50%** per trade depending on ETF liquidity.

---

## 10. Kill Gate

The project established pre-registered quality gates locked prior to backtest execution.

### Criteria Definitions

1. **Post-Cost Sharpe Ratio:** $\ge 0.50$ per fold.
2. **Post-Cost Win Rate:** $\ge 40\%$ per fold.
3. **Outperform Baseline:** Must beat Buy-and-Hold Sharpe in the fold.
4. **Fold Pass Ratio:** $\ge 50\%$ of folds ($\ge 3 / 5$ folds) must satisfy criteria 1–3.

### Evaluation Summary

| Horizon / Strategy | Folds Passing | Pass Ratio | Verdict |
|---|---|---|---|
| **7-Day (SL-Filtered)** | 0 / 5 | 0% | **FAILED** |
| **14-Day (SL-Filtered)** | 1 / 5 | 20% | **FAILED** |

---

## 11. Original Results

### 14-Day Horizon — Fold-by-Fold Performance (SL-Filtered Strategy)

| Fold | Date Range | Trades | Pre-Cost Win% | Post-Cost Win% | Pre-Cost Sharpe | Post-Cost Sharpe | Max DD | Profit Factor | Kill Gate |
|---|---|---|---|---|---|---|---|---|---|
| **Fold 1** | 2006-07-31 to 2010-08-10 | 1,482 | 41.2% | 34.1% | -0.12 | -0.68 | -42.1% | 0.81 | **FAIL** |
| **Fold 2** | 2010-08-11 to 2014-08-10 | 1,210 | 38.5% | 31.0% | -0.85 | -1.42 | -38.6% | 0.72 | **FAIL** |
| **Fold 3** | 2014-08-11 to 2018-08-20 | 1,845 | 39.1% | 32.4% | -0.61 | -1.15 | -35.2% | 0.76 | **FAIL** |
| **Fold 4** | 2018-08-22 to 2022-08-31 | 1,604 | 52.8% | 45.2% | +1.41 | **+0.86** | -18.4% | 1.34 | **PASS** |
| **Fold 5** | 2022-09-01 to 2026-09-06 | 1,512 | 40.0% | 32.8% | -0.42 | -0.98 | -29.8% | 0.79 | **FAIL** |

### 7-Day Horizon — Fold-by-Fold Performance (SL-Filtered Strategy)

| Fold | Date Range | Trades | Post-Cost Win% | Post-Cost Sharpe | Max DD | Kill Gate |
|---|---|---|---|---|---|---|
| **Fold 1** | 2006-07-31 to 2010-08-10 | 2,140 | 31.2% | -1.12 | -48.2% | **FAIL** |
| **Fold 2** | 2010-08-11 to 2014-08-10 | 1,890 | 29.5% | -1.65 | -44.1% | **FAIL** |
| **Fold 3** | 2014-08-11 to 2018-08-20 | 2,410 | 30.8% | -1.38 | -41.0% | **FAIL** |
| **Fold 4** | 2018-08-22 to 2022-08-31 | 2,250 | 38.4% | +0.12 | -26.5% | **FAIL** |
| **Fold 5** | 2022-09-01 to 2026-09-06 | 2,180 | 31.0% | -1.22 | -36.4% | **FAIL** |

### Baseline Comparison (14-Day Horizon, Average Across All Folds)

| Strategy Variant | Mean Post-Cost Sharpe | Mean Post-Cost Win Rate | Total Trades | Mean Max DD |
|---|---|---|---|---|
| **XGBoost Unfiltered** | -0.88 | 33.1% | 11,240 | -38.2% |
| **XGBoost SL-Filtered** | -0.67 | 35.1% | 7,653 | -32.8% |
| **Buy & Hold (Baseline)** | **+0.42** | N/A | 10 | -18.5% |
| **SMA50 Filter (Rule)** | +0.15 | 44.2% | 420 | -21.4% |

---

## 12. Methodological Corrections

During auditing, two important methodological issues were discovered, analyzed, and documented:

### 1. Sharpe Ratio Annualization Limitation
- **Issue:** Trade-level returns were annualized using $\sqrt{252}$, treating each multi-day holding period as a single day.
- **Audit Findings:** Because trade frequency across 10 ETFs exceeded 252 trades per year, $\sqrt{252}$ underestimated the trade count, making the reported Sharpe figures **conservative** (understated).
- **Impact on Kill Gate:** Failing folds missed the $\ge 0.50$ threshold by large negative margins (-0.68 to -1.42). Correcting annualization does not change any fail verdict.

### 2. Phase 4 Calibration Data Leakage
- **Issue:** An initial calibration check used a pooled XGBoost model trained on the *full* dataset (including validation folds).
- **Audit Findings:** This pooled model introduced lookahead leakage, producing artificially steep probability calibration curves.
- **Correction:** When re-evaluated using per-fold retrained models on strictly purged training sets, the empirical probability curve flattened out, revealing that higher predicted probabilities ($P(\text{TP}) > 0.45$) offered negligible precision gain over base rates.

---

## 13. Reduced Trade-Frequency Experiment

To test whether raising entry thresholds could eliminate uncalibrated low-confidence trades and salvage post-cost performance, a secondary experiment (`experiment_freq_reduction_v2.py`) was executed.

### Pre-Registered Experiment Thresholds
- **Threshold A:** $P(\text{TP}) \ge 0.45$, $P(\text{SL}) < 0.30$
- **Threshold B:** $P(\text{TP}) \ge 0.50$, $P(\text{SL}) < 0.30$

### Experiment Results (14-Day Horizon)

| Strategy Threshold | Avg Trades/Yr | Post-Cost Win Rate | Post-Cost Sharpe | Gold/Silver Profit Share | Folds Passed | Verdict |
|---|---|---|---|---|---|---|
| **Baseline ($P \ge 0.35$)** | ~380 | 35.1% | -0.67 | 78.4% | 1 / 5 | **FAILED** |
| **Threshold A ($P \ge 0.45$)** | ~150 | 37.8% | -0.41 | 82.1% | 1 / 5 | **FAILED** |
| **Threshold B ($P \ge 0.50$)** | ~80 | 39.2% | -0.28 | 86.5% | 1 / 5 | **FAILED** |

### Summary
While higher confidence thresholds reduced total trade volume and trade costs, performance in Folds 1, 2, 3, and 5 remained negative. The strategy **FAILED** to achieve a robust edge under all tested threshold variations.

---

## 14. Fold 4 Investigation

Fold 4 (2018–2022) was the **only** fold to achieve positive net performance (+0.86 Sharpe). A deep-dive audit (`fold4_analysis.py`) was conducted to evaluate whether this represented genuine predictive skill or market-regime coincidence.

### Findings

1. **Asset Concentration:** Gold and Silver ETFs (`GOLDBEES`, `SILVERBEES`, `GOLDIETF`) generated **$84.2\%$ of total Fold 4 net profits** while accounting for $42.1\%$ of executed trades.
2. **Equity ETF Failure:** Equity-based ETFs (`NIFTYBEES`, `BANKBEES`, `ITBEES`) produced near-zero or negative post-cost returns during Fold 4.
3. **Macro Context:** Fold 4 spanned the March 2020 COVID-19 market crash and subsequent global central bank easing, triggering a massive macro bull rally in precious metals.
4. **Conclusion:** Fold 4 performance was driven by an exogenous commodity bull run combined with broad market volatility, rather than generalizable model signal across ETF asset classes.

---

## 15. Most Recent Period (2022–2026)

Fold 5 evaluated model performance on the most recent market data (September 2022 to September 2026).

### Performance Metrics

- **Post-Cost Sharpe Ratio:** `-0.98` (SL-Filtered baseline)
- **Post-Cost Win Rate:** `32.8%`
- **Max Drawdown:** `-29.8%`
- **Buy & Hold Baseline Sharpe:** `+0.72`

### Key Observations
- The strategy suffered severe underperformance during 2022–2026 compared to passive Buy & Hold.
- Increased market efficiency and sideways/choppy market regimes caused frequent stop-loss triggers and transaction cost drag.
- Raising confidence thresholds (Thresholds A & B) failed to make Fold 5 profitable.

---

## 16. What Actually Worked

It is vital to distinguish software engineering success from financial trading success:

### Engineering & Infrastructure Successes
- **Data Pipeline:** Reliable, idempotent ingestion of Groww API historical data into PostgreSQL.
- **Feature Pipeline:** 31 clean, zero-lookahead technical indicators and context features.
- **Regime & Labeling Modules:** Robust causal market regime detection and ATR-based triple-barrier labeling.
- **Purging & Validation:** Implemented Marcos López de Prado's purging/embargoing to ensure leakage-free walk-forward backtesting.
- **Cost Realism Layer:** Accurately incorporated all Indian market transaction costs, taxes, and spreads.
- **Audit & Leakage Detection:** Successfully identified and corrected calibration data leakage in Phase 4.

---

## 17. What Did Not Work

### Financial & Quantitative Strategy Failures
- **Absence of Post-Cost Edge:** Signal precision was insufficient to overcome 0.35%–0.50% round-trip trading costs.
- **High Turnover Friction:** Frequent trading eroded gross gains.
- **Flat Calibration:** Higher predicted probabilities failed to yield higher empirical win rates in out-of-sample testing.
- **Asset Class Concentration:** Gains were isolated to Gold/Silver during extraordinary macro conditions (Fold 4).
- **Recent Period Failure:** Strategy failed completely in the 2022–2026 out-of-sample period.

---

## 18. Final Answer: How Well Did It Work?

| Evaluation Aspect | Verdict | Summary Assessment |
|---|---|---|
| **Software Engineering** | **WORKED** | Complete, production-grade, reproducible research architecture built successfully. |
| **Research Methodology** | **WORKED** | Rigorous, leakage-free walk-forward testing framework executed correctly. |
| **Predictive Signal** | **FAILED** | Model probabilities demonstrated weak out-of-sample discriminative power. |
| **Trading Performance** | **FAILED** | Post-cost returns failed to beat simple Buy & Hold or achieve Sharpe $\ge 0.50$. |
| **Generalization** | **FAILED** | Performance failed to hold across time periods and non-commodity asset classes. |
| **Strategy Status** | **REJECTED** | Strategy is **NOT VALIDATED** for live or real-money trading. |

---

## 19. Final Conclusion

> **"The tested tactical ETF timing strategy did not demonstrate a robust post-cost edge, failing the pre-registered kill gate across both 7-day and 14-day horizons. Raising entry confidence thresholds reduced trading frequency but failed to establish profitability in 4 out of 5 historical folds, including the most recent 2022–2026 period. The single passing fold (Fold 4) was heavily concentrated in precious metals during COVID-19 macro volatility and did not generalize to broad equity ETFs. Based on available empirical evidence, no further model tuning or live deployment is justified."**

---

## 20. Future Work

The following items represent potential academic or future research hypotheses only. **None should be implemented without starting a new pre-registered research cycle:**

1. **Intraday or Multi-Month Horizons:** Test execution at tick/minute frequencies or multi-month macroeconomic rebalancing horizons.
2. **Alternative Feature Sets:** Incorporate order flow imbalance, ETF creation/redemption basket metrics, and global macro variables.
3. **Regime-Conditional Models:** Train dedicated sub-models filtered strictly for high-trending market regimes.
4. **Portfolio Position Sizing:** Implement risk-parity or volatility-targeted allocation instead of equal-weighted tactical entries.

---

## 21. Reproducibility

To reproduce the full research environment and inspect backtest outputs using the existing project codebase:

### 1. Prerequisites & Environment Setup
```bash
# Clone and enter directory
cd ai-etf-research

# Create and activate Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Services
```bash
# Ensure Docker Desktop is running, then start PostgreSQL container
docker compose up -d
```

### 3. Pipeline Execution Scripts
```bash
# Run data ingestion pipeline (Phase 1-3)
python -m src.main

# Inspect Phase 4 validation metrics
python -m scripts.report_phase4

# Inspect Phase 5 backtest results
python -m scripts.report_phase5

# Run Streamlit dashboard (Read-Only)
streamlit run app.py
```

Results are stored in PostgreSQL tables: `instruments`, `ohlcv_daily`, `features`, `market_regimes`, `triple_barrier_labels`, `model_runs`, `backtest_results`, and `experiment_freq_reduction_results_v2`.

---

## 22. Project Status

| Component | Status | Notes |
|---|---|---|
| **Data Ingestion Pipeline** | **COMPLETED** | Groww API integration & idempotent PostgreSQL storage. |
| **Feature Engineering** | **COMPLETED** | 31 technical and market context indicators. |
| **Regime Detection** | **COMPLETED** | 4-class causal market regime classifier. |
| **Triple-Barrier Labeling** | **COMPLETED** | ATR-scaled barriers for 7D and 14D horizons. |
| **Models & Training** | **COMPLETED** | XGBoost, Logistic Regression, Random Forest. |
| **Walk-Forward Validation** | **COMPLETED** | 5 purged chronological folds (Lopez de Prado). |
| **Realism & Cost Modeling** | **COMPLETED** | Brokerage, STT, fees, slippage, and spread. |
| **Calibration Audit** | **COMPLETED** | Leakage identified and corrected in v2 audit. |
| **Original Kill Gate** | **COMPLETED** | **FAILED** (0/5 folds passed 7D; 1/5 passed 14D). |
| **Reduced-Frequency Experiment**| **COMPLETED** | **FAILED** (v2 purged per-fold retraining). |
| **Final Research Report** | **COMPLETED** | Documented in `FINAL_RESEARCH_REPORT.md`. |
| **Real-Money Trading Validation**| **NOT VALIDATED** | **DO NOT DEPLOY FOR REAL TRADING**. |

---
*End of Final Research Report.*
