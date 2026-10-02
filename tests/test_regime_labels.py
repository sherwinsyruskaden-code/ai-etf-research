import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from src.features.calculator import compute_technical_features
from src.regime_labels.regime import detect_causal_regime
from src.regime_labels.labeler import generate_triple_barrier_labels
from src.regime_labels.repository import RegimeLabelRepository


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    close = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, n))
    open_p = close + np.random.normal(0, 0.2, n)
    high_p = np.maximum(close, open_p) + np.abs(np.random.normal(0.5, 0.2, n))
    low_p = np.minimum(close, open_p) - np.abs(np.random.normal(0.5, 0.2, n))
    volume = np.random.randint(50000, 200000, n)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close,
        "volume": volume
    })
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE TABLE instruments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange VARCHAR(10) NOT NULL,
                segment VARCHAR(10) NOT NULL,
                trading_symbol VARCHAR(50) NOT NULL,
                groww_symbol VARCHAR(100) NOT NULL UNIQUE,
                instrument_type VARCHAR(20) NOT NULL,
                name VARCHAR(255),
                is_selected_universe BOOLEAN DEFAULT 0,
                avg_daily_volume REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_exchange_symbol UNIQUE (exchange, trading_symbol)
            );
        """)
        conn.exec_driver_sql("""
            CREATE TABLE market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
                timestamp TIMESTAMP NOT NULL,
                instrument_regime VARCHAR(30) NOT NULL,
                market_regime VARCHAR(30) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_regimes_inst_ts UNIQUE (instrument_id, timestamp)
            );
        """)
        conn.exec_driver_sql("""
            CREATE TABLE triple_barrier_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
                timestamp TIMESTAMP NOT NULL,
                horizon INTEGER NOT NULL,
                barrier_hit VARCHAR(10),
                days_to_hit INTEGER,
                tp_level REAL,
                sl_level REAL,
                realized_return REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_labels_inst_ts_hz UNIQUE (instrument_id, timestamp, horizon)
            );
        """)
    return engine


def test_regime_causality(sample_ohlcv):
    """Verify that regime detection is strictly causal (no future data leakage)."""
    df_feat_full = compute_technical_features(sample_ohlcv)

    # Compute regimes on first 70 rows
    df_ohlcv_sub = sample_ohlcv.iloc[:70].copy()
    df_feat_sub = df_feat_full.iloc[:70].copy()
    regimes_sub = detect_causal_regime(df_ohlcv_sub, df_feat_sub)

    # Compute regimes on full 100 rows
    regimes_full = detect_causal_regime(sample_ohlcv, df_feat_full)

    # Regime at row 69 must be identical
    assert regimes_sub.iloc[69] == regimes_full.iloc[69]


def test_triple_barrier_forward_looking(sample_ohlcv):
    """Verify that triple-barrier labels ARE forward-looking (depend on future price path)."""
    df_feat = compute_technical_features(sample_ohlcv)

    # Path A: Original future
    df_path_a = sample_ohlcv.copy()
    labels_a = generate_triple_barrier_labels(df_path_a, df_feat, horizons=[7])

    # Path B: Modify future price at T+2 to surge high (trigger TP)
    df_path_b = sample_ohlcv.copy()
    entry_idx = 50
    df_path_b.loc[entry_idx + 2, "high"] = 500.0  # Huge spike to breach TP
    df_path_b.loc[entry_idx + 2, "close"] = 500.0

    labels_b = generate_triple_barrier_labels(df_path_b, df_feat, horizons=[7])

    lbl_a_row = labels_a[labels_a["timestamp"] == sample_ohlcv.loc[entry_idx, "timestamp"]].iloc[0]
    lbl_b_row = labels_b[labels_b["timestamp"] == sample_ohlcv.loc[entry_idx, "timestamp"]].iloc[0]

    # Modifying future path at T+2 must change label for entry day T
    assert lbl_b_row["barrier_hit"] == "TP"
    assert lbl_b_row["days_to_hit"] == 2


def test_triple_barrier_synthetic_correctness():
    """Test TP, SL, and TIME hit outcomes on a hand-crafted synthetic price series."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="Asia/Kolkata")

    # Entry at T=0 (Close = 100, ATR = 2.0). Barrier width for H=7 is 2 * 2.0 * 1.0 = 4.0.
    # TP_Level = 104.0, SL_Level = 96.0.
    # Price surge to 105.0 on T+3 (index 3) -> TP hit on day 3.
    df_ohlcv = pd.DataFrame({
        "timestamp": dates,
        "open": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "high": [101, 101, 101, 105, 101, 101, 101, 101, 101, 101],
        "low": [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
        "close": [100, 100, 100, 105, 100, 100, 100, 100, 100, 100],
        "volume": [1000] * 10
    })

    df_feat = pd.DataFrame({"atr_14": [2.0] * 10})

    labels = generate_triple_barrier_labels(df_ohlcv, df_feat, horizons=[7], atr_multiplier=2.0)
    lbl_entry = labels.iloc[0]

    assert lbl_entry["barrier_hit"] == "TP"
    assert lbl_entry["days_to_hit"] == 3
    assert np.isclose(lbl_entry["tp_level"], 104.0)
    assert np.isclose(lbl_entry["sl_level"], 96.0)


def test_end_of_history_null_labeling(sample_ohlcv):
    """Verify that rows near end-of-history without sufficient future bars are labeled NULL."""
    df_feat = compute_technical_features(sample_ohlcv)
    labels_h14 = generate_triple_barrier_labels(sample_ohlcv, df_feat, horizons=[14])

    sub_14 = labels_h14[labels_h14["horizon"] == 14]

    # The last 14 rows should be NULL for barrier_hit unless TP/SL was hit before end
    last_row = sub_14.iloc[-1]
    assert last_row["barrier_hit"] is None or pd.isna(last_row["barrier_hit"])
    assert last_row["realized_return"] is None or pd.isna(last_row["realized_return"])


def test_regime_and_label_idempotency(sqlite_engine, sample_ohlcv):
    """Verify idempotency of regime and label database saving."""
    repo = RegimeLabelRepository(sqlite_engine)
    df_feat = compute_technical_features(sample_ohlcv)

    regimes = detect_causal_regime(sample_ohlcv, df_feat)
    df_regimes = pd.DataFrame({
        "timestamp": sample_ohlcv["timestamp"],
        "instrument_regime": regimes,
        "market_regime": "SIDEWAYS"
    })

    df_labels = generate_triple_barrier_labels(sample_ohlcv, df_feat, horizons=[7, 14])

    inst_id = 1

    # First save
    res_reg1 = repo.save_regimes_batch(inst_id, df_regimes)
    res_lbl1 = repo.save_labels_batch(inst_id, df_labels)

    assert res_reg1["inserted"] == len(sample_ohlcv)
    assert res_lbl1["inserted"] == len(sample_ohlcv) * 2  # 2 horizons

    # Second save (must update 0 new inserts)
    res_reg2 = repo.save_regimes_batch(inst_id, df_regimes)
    res_lbl2 = repo.save_labels_batch(inst_id, df_labels)

    assert res_reg2["inserted"] == 0
    assert res_lbl2["inserted"] == 0
    assert res_reg2["updated"] == len(sample_ohlcv)
    assert res_lbl2["updated"] == len(sample_ohlcv) * 2
