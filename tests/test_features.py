import pytest
import pandas as pd
import numpy as np
import ta
from sqlalchemy import create_engine
from src.features.calculator import compute_technical_features
from src.features.repository import FeatureRepository
from src.features.builder import FeatureBuilder


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

    return pd.DataFrame({
        "timestamp": dates,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close,
        "volume": volume
    })


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
            CREATE TABLE ohlcv_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
                timestamp TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_instrument_timestamp UNIQUE (instrument_id, timestamp)
            );
        """)
        conn.exec_driver_sql("""
            CREATE TABLE features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
                timestamp TIMESTAMP NOT NULL,
                return_1d REAL, return_3d REAL, return_5d REAL, return_10d REAL, return_20d REAL,
                sma_5 REAL, sma_20 REAL, sma_50 REAL,
                price_vs_sma5 REAL, price_vs_sma20 REAL, price_vs_sma50 REAL, sma20_vs_sma50 REAL,
                rsi_14 REAL, macd REAL, macd_signal REAL, macd_hist REAL, roc_10 REAL,
                volatility_5d REAL, volatility_10d REAL, volatility_20d REAL, atr_14 REAL,
                volume_today INTEGER, volume_sma20 REAL, volume_ratio REAL, volume_change REAL,
                nifty50_return_1d REAL, banknifty_return_1d REAL, india_vix_level REAL, india_vix_change_1d REAL,
                sector_return_1d REAL, commodity_return_1d REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_features_inst_ts UNIQUE (instrument_id, timestamp)
            );
        """)
    return engine


def test_feature_correctness(sample_ohlcv):
    df_feat = compute_technical_features(sample_ohlcv)

    assert not df_feat.empty
    assert len(df_feat) == len(sample_ohlcv)

    # 1. Test 1-day Return
    expected_ret1 = sample_ohlcv["close"].pct_change(1).iloc[10]
    assert np.isclose(df_feat["return_1d"].iloc[10], expected_ret1, rtol=1e-5)

    # 2. Test 5-day SMA
    expected_sma5 = sample_ohlcv["close"].iloc[6:11].mean()
    assert np.isclose(df_feat["sma_5"].iloc[10], expected_sma5, rtol=1e-5)

    # 3. Test RSI 14 against ta reference
    expected_rsi14 = ta.momentum.RSIIndicator(close=sample_ohlcv["close"], window=14).rsi().iloc[25]
    assert np.isclose(df_feat["rsi_14"].iloc[25], expected_rsi14, rtol=1e-5)


def test_no_lookahead_leakage(sample_ohlcv):
    # Features computed on history up to index 70
    df_history = sample_ohlcv.iloc[:70].copy()
    feat_history = compute_technical_features(df_history)

    # Features computed on history up to index 100 (appending 30 future rows)
    feat_full = compute_technical_features(sample_ohlcv.copy())

    # Row 69 features must be identical regardless of appended future data
    row_hist = feat_history.iloc[69].drop("timestamp").astype(float)
    row_full = feat_full.iloc[69].drop("timestamp").astype(float)

    diff = (row_hist - row_full).abs().max()
    assert diff < 1e-10, f"Lookahead leakage detected! Max feature diff: {diff}"


def test_insufficient_history_handling(sample_ohlcv):
    df_feat = compute_technical_features(sample_ohlcv)

    # SMA 50 must be NaN for the first 49 rows (0 to 48)
    assert df_feat["sma_50"].iloc[:48].isna().all()
    assert pd.notna(df_feat["sma_50"].iloc[49])

    # Volatility 20d must be NaN for the first 19 rows
    assert df_feat["volatility_20d"].iloc[:19].isna().all()
    assert pd.notna(df_feat["volatility_20d"].iloc[20])

    # Confirm zero infinite values
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    assert not np.isinf(df_feat[numeric_cols]).any().any()


def test_feature_idempotency(sqlite_engine, sample_ohlcv):
    repo = FeatureRepository(sqlite_engine)
    df_feat = compute_technical_features(sample_ohlcv)

    inst_id = 1

    # First save
    res1 = repo.save_features_batch(inst_id, df_feat)
    assert res1["inserted"] == len(sample_ohlcv)

    # Query count
    df_db1 = repo.get_features(inst_id)
    assert len(df_db1) == len(sample_ohlcv)

    # Second save (idempotency check)
    res2 = repo.save_features_batch(inst_id, df_feat)
    assert res2["inserted"] == 0
    assert res2["updated"] == len(sample_ohlcv)

    df_db2 = repo.get_features(inst_id)
    assert len(df_db2) == len(sample_ohlcv)  # No duplicates created
