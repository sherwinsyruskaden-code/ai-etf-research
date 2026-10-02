import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from src.data.database import init_db, InstrumentRepository, OHLCVRepository
from src.data.groww_client import GrowwClient
from src.data.collector import ETFDataCollector


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
    return engine


def test_idempotent_collection(sqlite_engine):
    groww_client = MagicMock()

    # Mock instruments dataframe
    mock_inst_df = pd.DataFrame([
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "NIFTYBEES", "groww_symbol": "NSE-NIFTYBEES", "instrument_type": "ETF", "name": "Nifty BeES"},
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "BANKBEES", "groww_symbol": "NSE-BANKBEES", "instrument_type": "ETF", "name": "Bank BeES"},
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "GOLDBEES", "groww_symbol": "NSE-GOLDBEES", "instrument_type": "ETF", "name": "Gold BeES"},
    ])

    dates = pd.date_range("2026-01-01", periods=5, freq="D", tz="Asia/Kolkata")
    mock_candles = pd.DataFrame({
        "timestamp": dates,
        "open": 250.0,
        "high": 255.0,
        "low": 248.0,
        "close": 252.0,
        "volume": 100000
    })

    groww_client.fetch_all_instruments.return_value = mock_inst_df
    groww_client.fetch_historical_daily_candles.return_value = mock_candles

    collector = ETFDataCollector(groww_client=groww_client, db_engine=sqlite_engine)

    # First pipeline execution
    summary1 = collector.run_pipeline(historical_days=30, top_n=3)
    assert summary1["instruments_discovered"] == 3
    assert summary1["selected_etf_count"] == 3
    assert summary1["rows_inserted_total"] > 0
    assert summary1["duplicates_skipped_total"] == 0

    # Second pipeline execution (must be idempotent - skip duplicates)
    summary2 = collector.run_pipeline(historical_days=30, top_n=3)
    assert summary2["rows_inserted_total"] == 0
    assert summary2["duplicates_skipped_total"] > 0
