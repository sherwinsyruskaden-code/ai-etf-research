import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, Session
from src.config.settings import DATABASE_URL, SQL_DIR

logger = logging.getLogger(__name__)


def get_db_engine(db_url: str = DATABASE_URL) -> Engine:
    """Create and return a SQLAlchemy database engine."""
    return create_engine(db_url, echo=False, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Initialize database tables using schema.sql."""
    schema_path = SQL_DIR / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with engine.begin() as conn:
        conn.execute(text(schema_sql))
    logger.info("Database schema initialized successfully.")


class InstrumentRepository:
    """Repository for storing and querying instruments in PostgreSQL."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def save_instruments(self, df_instruments: pd.DataFrame) -> int:
        """
        Upsert instruments into PostgreSQL.
        Returns total inserted/updated count.
        """
        if df_instruments.empty:
            return 0

        upsert_query = text("""
            INSERT INTO instruments (exchange, segment, trading_symbol, groww_symbol, instrument_type, name)
            VALUES (:exchange, :segment, :trading_symbol, :groww_symbol, :instrument_type, :name)
            ON CONFLICT (groww_symbol) DO UPDATE SET
                trading_symbol = EXCLUDED.trading_symbol,
                instrument_type = EXCLUDED.instrument_type,
                name = EXCLUDED.name
            RETURNING id;
        """)

        records = []
        for idx, row in df_instruments.iterrows():
            name_val = str(row.get("name", "")) if pd.notna(row.get("name")) else ""
            records.append({
                "exchange": str(row.get("exchange", "NSE")),
                "segment": str(row.get("segment", "CASH")),
                "trading_symbol": str(row.get("trading_symbol")),
                "groww_symbol": str(row.get("groww_symbol")),
                "instrument_type": str(row.get("instrument_type", "ETF")),
                "name": name_val,
            })

        count = 0
        with self.engine.begin() as conn:
            for rec in records:
                conn.execute(upsert_query, rec)
                count += 1

        logger.info(f"Saved/Updated {count} instruments in PostgreSQL.")
        return count

    def mark_selected_universe(self, selected_groww_symbols: List[str], candidate_metrics: Dict[str, float]) -> None:
        """Mark selected ETF universe and save average daily volume metric in PostgreSQL."""
        with self.engine.begin() as conn:
            # Reset existing selected universe status
            conn.execute(text("UPDATE instruments SET is_selected_universe = FALSE;"))

            # Update metrics and selected status
            for symbol, avg_vol in candidate_metrics.items():
                is_selected = symbol in selected_groww_symbols
                conn.execute(
                    text("""
                        UPDATE instruments 
                        SET is_selected_universe = :is_selected,
                            avg_daily_volume = :avg_vol
                        WHERE groww_symbol = :symbol;
                    """),
                    {"is_selected": is_selected, "avg_vol": float(avg_vol), "symbol": symbol}
                )

    def get_selected_universe(self) -> pd.DataFrame:
        """Get instruments selected for the universe."""
        query = text("""
            SELECT id, exchange, segment, trading_symbol, groww_symbol, instrument_type, name, avg_daily_volume
            FROM instruments
            WHERE is_selected_universe = TRUE
            ORDER BY avg_daily_volume DESC;
        """)
        return pd.read_sql(query, self.engine)

    def get_all_instruments(self) -> pd.DataFrame:
        """Retrieve all instruments stored in DB."""
        return pd.read_sql("SELECT * FROM instruments;", self.engine)


class OHLCVRepository:
    """Repository for storing and querying daily OHLCV data idempotently."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def save_ohlcv_batch(self, instrument_id: int, df_ohlcv: pd.DataFrame) -> Dict[str, int]:
        """
        Idempotently insert OHLCV candles into PostgreSQL.
        Ignores duplicates based on (instrument_id, timestamp) unique constraint.
        Returns dict with rows_inserted and duplicates_skipped.
        """
        if df_ohlcv.empty:
            return {"inserted": 0, "skipped": 0}

        insert_sql = text("""
            INSERT INTO ohlcv_daily (instrument_id, timestamp, open, high, low, close, volume)
            VALUES (:instrument_id, :timestamp, :open, :high, :low, :close, :volume)
            ON CONFLICT (instrument_id, timestamp) DO NOTHING;
        """)

        inserted = 0
        skipped = 0

        with self.engine.begin() as conn:
            for _, row in df_ohlcv.iterrows():
                ts_val = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
                res = conn.execute(
                    insert_sql,
                    {
                        "instrument_id": instrument_id,
                        "timestamp": ts_val,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                )
                if res.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1

        return {"inserted": inserted, "skipped": skipped}

    def get_ohlcv(self, instrument_id: int) -> pd.DataFrame:
        """Fetch chronological daily OHLCV for an instrument."""
        query = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE instrument_id = :instrument_id
            ORDER BY timestamp ASC;
        """)
        return pd.read_sql(query, self.engine, params={"instrument_id": instrument_id})

    def clear_all_ohlcv(self) -> int:
        """Wipe all existing OHLCV records from PostgreSQL."""
        with self.engine.begin() as conn:
            res = conn.execute(text("DELETE FROM ohlcv_daily;"))
            count = res.rowcount
        logger.info(f"Wiped {count} old OHLCV records from PostgreSQL database.")
        return count

    def get_total_candle_count(self) -> int:
        """Return total row count in ohlcv_daily."""
        with self.engine.connect() as conn:
            res = conn.execute(text("SELECT COUNT(*) FROM ohlcv_daily;")).scalar()
            return res or 0
