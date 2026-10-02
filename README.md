# AI ETF Quantitative Research System

> **Research conclusion: The current strategy did not demonstrate a robust edge over the baseline after costs.**
> Kill-gate verdict: **FAILED** (1/5 folds passed on the 14-day strategy; 0/5 on the 7-day strategy).

---

## 📊 Streamlit Research Dashboard

The dashboard visualizes all locked Phase 5 results. It is **read-only** — it does not retrain models, write to the database, or generate live trading signals.

### Run command (one command)

```bash
# 1. Start PostgreSQL (Docker must be running)
docker compose up -d

# 2. Launch the dashboard
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** in your browser.

### Pages

| Page | What it shows |
|---|---|
| **Overview** | Project description, 10 ETFs analyzed, model status (H=7 / H=14), kill-gate FAIL verdict, DB statistics (candles / features / regimes / labels / folds) |
| **ETF Explorer** | Dropdown per ETF: price history (line), volume, daily returns, RSI-14, 20d volatility; latest historical triple-barrier label; current regime label |
| **Backtest Results** | Per-fold + aggregated post-cost Sharpe, win rate, drawdown, trades for all variants (XGBoost Unfiltered, SL-Filtered, Buy & Hold, SMA50); kill-gate bar chart; baseline comparison; Phase 4 model calibration table; frequency-reduction experiment clearly labeled "EXPERIMENT — NOT PART OF LOCKED RESULTS" with the calibration leakage bug noted |
| **Research / Methodology** | Visual pipeline diagram (Groww API → PostgreSQL → Features → Regimes → Labels → XGBoost → Walk-Forward → Costs → Calibration → Kill Gate); kill-gate criteria table; Sharpe annualization caveat; design principles; database table lineage |

### Prerequisites

- Python 3.11+
- Docker Desktop running (for PostgreSQL)
- All dependencies: `pip install -r requirements.txt`

---

## 📌 Project Purpose
Phase 1 of the AI ETF Quantitative Research System builds a robust, auditable, and production-grade data ingestion layer using the official **Groww Python SDK (`growwapi`)** and **PostgreSQL**.

The goal of Phase 1 is to:
1. Dynamically discover all NSE CASH instruments from Groww's official instrument master without hardcoding tickers.
2. Filter candidate ETFs and rank them based on recent traded liquidity (Average Daily Traded Value = Volume × Close).
3. Select the top ~10 most liquid ETFs and store the selected universe mapping in PostgreSQL.
4. Download historical daily OHLCV candles, normalize timestamps to IST (`Asia/Kolkata`), and validate data math and structure.
5. Idempotently store raw OHLCV candles into PostgreSQL without creating duplicate records on repeated runs.

---

## 🏗️ Architecture

```
Groww API (SDK / Master CSV)
    │
    ▼
Instrument Discovery & Filtering (NSE CASH → ETF Candidates)
    │
    ▼
Liquidity-Based Universe Selection (Ranked by Avg Daily Traded Value)
    │
    ▼
Historical Daily OHLCV Collection (Normalized to IST)
    │
    ▼
Data Validation (Positive OHLC, High/Low Bounds, No Nulls/Duplicates)
    │
    ▼
PostgreSQL Database Storage (Idempotent ON CONFLICT IGNORE)
```

---

## 🛠️ Tech Stack & Prerequisites

* **Python 3.11+**
* **growwapi** official Python SDK
* **PostgreSQL 16** (via Docker Compose)
* **SQLAlchemy 2.0+** & **psycopg 3**
* **pandas**
* **python-dotenv**
* **pytest**
* **Docker & Docker Compose**

---

## 🚀 Environment Setup & Installation

### 1. Clone & Navigate to Project Directory
```bash
cd ai-etf-research
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your Groww API credentials:
```env
GROWW_API_KEY=your_groww_api_key_here
GROWW_API_SECRET=your_groww_api_secret_here
POSTGRES_USER=etf_user
POSTGRES_PASSWORD=etf_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=etf_research
DATABASE_URL=postgresql+psycopg://etf_user:etf_password@localhost:5433/etf_research
HISTORICAL_DAYS=60
TOP_ETF_COUNT=10
TIMEZONE=Asia/Kolkata
```

---

## 🐳 PostgreSQL Setup via Docker Compose

To start the PostgreSQL database container:
```bash
docker compose up -d
```
To stop the PostgreSQL container:
```bash
docker compose down
```

---

## 🏃 Running the Pipeline & Collector

To run the complete Phase 1 ingestion pipeline:
```bash
python -m src.main
```

### Sample Output Log:
```text
2026-09-27 13:18:27 [INFO] src.data.collector - Instruments Discovered (NSE CASH) : 4299
2026-09-27 13:18:27 [INFO] src.data.collector - ETF Candidates Identified        : 244
2026-09-27 13:18:27 [INFO] src.data.collector - ETFs Selected in Universe       : 10
2026-09-27 13:18:27 [INFO] src.data.collector - Date Range Requested             : 2026-07-29 to 2026-09-27
2026-09-27 13:18:27 [INFO] src.data.collector - Total Candles Downloaded         : 430
2026-09-27 13:18:27 [INFO] src.data.collector - Total DB Rows Inserted           : 430
2026-09-27 13:18:27 [INFO] src.data.collector - Total Duplicates Skipped (Idempotent): 0
2026-09-27 13:18:27 [INFO] src.data.collector - Validation Failures Count        : 0
```

Running the pipeline again demonstrates complete **Idempotency**:
```text
2026-09-27 13:18:45 [INFO] src.data.collector - Total DB Rows Inserted           : 0
2026-09-27 13:18:45 [INFO] src.data.collector - Total Duplicates Skipped (Idempotent): 430
```

---

## 🧪 Running Unit Tests

Run the full pytest test suite:
```bash
python -m pytest -v
```

All tests verify data validation, duplicate handling, instrument filtering, liquidity selection logic, and malformed API response handling.

---

## 🗄️ Database Schema

### `instruments`
Stores discovered exchange instruments and universe selection metadata.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | SERIAL PRIMARY KEY | Internal unique ID |
| `exchange` | VARCHAR(10) | Exchange name (e.g. `NSE`) |
| `segment` | VARCHAR(10) | Segment (e.g. `CASH`) |
| `trading_symbol` | VARCHAR(50) | Exchange trading symbol (e.g. `NIFTYBEES`) |
| `groww_symbol` | VARCHAR(100) UNIQUE | Groww internal symbol (e.g. `NSE-NIFTYBEES`) |
| `instrument_type` | VARCHAR(20) | Type (`ETF`, `EQ`, etc.) |
| `name` | VARCHAR(255) | Full security name |
| `is_selected_universe` | BOOLEAN | `TRUE` if selected in top liquid universe |
| `avg_daily_volume` | NUMERIC(18, 2) | Traded liquidity metric (Avg Daily Traded Value) |
| `created_at` | TIMESTAMPTZ | Record creation timestamp |

### `ohlcv_daily`
Stores raw daily OHLCV candles idempotently.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | SERIAL PRIMARY KEY | Record unique ID |
| `instrument_id` | INTEGER FK | Foreign key to `instruments.id` |
| `timestamp` | TIMESTAMPTZ | Candle timestamp localized to IST (`Asia/Kolkata`) |
| `open` | NUMERIC(14, 4) | Daily opening price |
| `high` | NUMERIC(14, 4) | Daily highest price |
| `low` | NUMERIC(14, 4) | Daily lowest price |
| `close` | NUMERIC(14, 4) | Daily closing price |
| `volume` | BIGINT | Daily traded volume |
| `created_at` | TIMESTAMPTZ | Record creation timestamp |

**Uniqueness Constraint:** `UNIQUE (instrument_id, timestamp)` prevents duplicate candles.

---

---

## 📊 Phase 3 — Causal Regime Detection & Triple-Barrier Labeling

### 1. Causal Market Regime Detection
Market regimes are calculated **strictly causally** using data available up to and including day $T$ (zero lookahead):
* **`HIGH_VOLATILITY`**: 20-day realized volatility exceeds the trailing/expanding 252-day 80th percentile threshold (causal expanding window for early history $T < 252$, trailing 252-day window thereafter).
* **`TRENDING_UP`**: $\text{Close}_T > \text{SMA}_{20, T}$ AND $\text{SMA}_{20, T} > \text{SMA}_{50, T}$ AND 5-day slope of $\text{SMA}_{20} > +0.2\%$.
* **`TRENDING_DOWN`**: $\text{Close}_T < \text{SMA}_{20, T}$ AND $\text{SMA}_{20, T} < \text{SMA}_{50, T}$ AND 5-day slope of $\text{SMA}_{20} < -0.2\%$.
* **`SIDEWAYS`**: Default baseline regime when volatility is normal and trend slope is neutral.

### 2. Triple-Barrier Labeling
Generates event-driven labels (`TP`, `SL`, or `TIME`) across 7-day and 14-day horizons:
* **Barrier Width**: $\text{Width}_T = 2.0 \times \text{ATR}_{14, T} \times \sqrt{\text{Horizon} / 7.0}$.
* **End-of-History Handling**: Rows near the end of available history where the full horizon cannot elapse and neither TP nor SL is breached are explicitly left unlabeled (`NULL`).

### ⚠️ IMPORTANT NOTE FOR PHASE 4/5: Label Overlap & Sample Dependency
> **CRITICAL ML DESIGN REQUIREMENT**:
> Triple-barrier labels generated for adjacent entry dates ($T, T+1, T+2, \dots$) look forward across overlapping future windows ($[T+1, T+H]$ and $[T+2, T+H+1]$) and are **not independent and identically distributed (i.i.d.) samples**.
> 
> When training XGBoost and baseline models in Phase 4/5:
> 1. **Random cross-validation splits MUST NOT be used** — doing so would cause severe data leakage across train and test folds.
> 2. **Walk-Forward Splits MUST enforce Purging and Embargoing** (Marcos Lopez de Prado):
>    - **Purging**: Remove training samples whose forward evaluation window overlaps with the test set evaluation window.
>    - **Embargoing**: Enforce a post-test gap period before starting the next training window to eliminate auto-correlated serial leakage.
> 3. **Sample Weighting**: Utilize sample-uniqueness weighting based on concurrent overlapping barrier lifetimes during ML model training.

---

## ⚠️ Known Limitations (Phase 5 Backtest)

### Sharpe Ratio Annualization Approximation

The model-driven strategy Sharpe ratios reported throughout Phase 5 are computed on **individual trade-level returns** (each observation = one 7–14 day holding period), then annualized using `sqrt(252)` — a formula that assumes each return represents **a single trading day**.

This is an approximation, not an exact calculation, for two reasons:

1. **Wrong annualization unit**: With ~333–1,018 trades per year across the 10-ETF universe, a more technically precise annualization would use `sqrt(N_trades_per_year)` rather than `sqrt(252)`. Since trades-per-year > 252 in most folds, `sqrt(252)` *understates* the annualization factor — meaning reported Sharpe magnitudes are conservative, not inflated.

2. **Non-independence of overlapping windows**: Triple-barrier labels look forward 7–14 calendar/trading days. Multiple concurrent open positions exist at any given time, meaning trade returns are not independent samples. A fully rigorous treatment would account for this serial correlation (e.g., using Newey-West standard errors or a return-series approach), but this would require restructuring the entire trade simulation to produce a synchronized daily P&L series.

**Implications for the kill-gate verdict**: Neither correction changes the Phase 5 FAIL verdict:
- The four failing folds (7d: all; 14d: Folds 2, 3, 5) fail by Sharpe margins of -1 to -4, which no annualization correction could flip.
- Fold 4 (14d SL-Filtered, Sharpe +0.86) is directionally real, but the exact magnitude should be treated as **indicative rather than precise**.
- The Buy & Hold baseline was separately corrected to use a proper daily return series (see Phase 5 B&H bug fix), so it is not subject to this caveat.

**Recommended future fix**: If the system progresses to paper trading, replace trade-level Sharpe with a daily portfolio P&L series (mark all open positions to market each day) before computing the final production Sharpe ratio.
