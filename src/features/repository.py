import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy import text, Engine

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "sma_5", "sma_20", "sma_50",
    "price_vs_sma5", "price_vs_sma20", "price_vs_sma50", "sma20_vs_sma50",
    "rsi_14", "macd", "macd_signal", "macd_hist", "roc_10",
    "volatility_5d", "volatility_10d", "volatility_20d", "atr_14",
    "volume_today", "volume_sma20", "volume_ratio", "volume_change",
    "nifty50_return_1d", "banknifty_return_1d",
    "india_vix_level", "india_vix_change_1d",
    "sector_return_1d", "commodity_return_1d"
]


class FeatureRepository:
    """Repository for managing features table in PostgreSQL."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def save_features_batch(self, instrument_id: int, df_features: pd.DataFrame) -> Dict[str, int]:
        """
        Upsert features into database idempotently.
        Updates existing values on (instrument_id, timestamp) conflict.
        """
        if df_features.empty:
            return {"inserted": 0, "updated": 0}

        check_sql = text("""
            SELECT 1 FROM features WHERE instrument_id = :instrument_id AND timestamp = :timestamp;
        """)

        cols_str = ", ".join(["instrument_id", "timestamp"] + FEATURE_COLUMNS)
        vals_str = ", ".join([":instrument_id", ":timestamp"] + [f":{col}" for col in FEATURE_COLUMNS])

        insert_sql = text(f"""
            INSERT INTO features ({cols_str})
            VALUES ({vals_str});
        """)

        update_assignments = [f"{col} = :{col}" for col in FEATURE_COLUMNS]
        update_sql = text(f"""
            UPDATE features
            SET {', '.join(update_assignments)}
            WHERE instrument_id = :instrument_id AND timestamp = :timestamp;
        """)

        inserted_count = 0
        updated_count = 0

        records = []
        for _, row in df_features.iterrows():
            ts_val = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
            rec = {
                "instrument_id": instrument_id,
                "timestamp": ts_val
            }
            for col in FEATURE_COLUMNS:
                val = row.get(col)
                if pd.isna(val):
                    rec[col] = None
                elif col in ["volume_today"]:
                    rec[col] = int(val)
                else:
                    rec[col] = float(val)
            records.append(rec)

        with self.engine.begin() as conn:
            for rec in records:
                exists = conn.execute(check_sql, {"instrument_id": instrument_id, "timestamp": rec["timestamp"]}).scalar()
                if exists:
                    conn.execute(update_sql, rec)
                    updated_count += 1
                else:
                    conn.execute(insert_sql, rec)
                    inserted_count += 1

        return {"inserted": inserted_count, "updated": updated_count}

    def get_features(self, instrument_id: int) -> pd.DataFrame:
        """Fetch chronological features for an instrument."""
        query = text("""
            SELECT *
            FROM features
            WHERE instrument_id = :instrument_id
            ORDER BY timestamp ASC;
        """)
        return pd.read_sql(query, self.engine, params={"instrument_id": instrument_id})

    def clear_all_features(self) -> int:
        """Wipe all records from features table."""
        with self.engine.begin() as conn:
            res = conn.execute(text("DELETE FROM features;"))
            count = res.rowcount
        logger.info(f"Wiped {count} records from features table.")
        return count

    def get_feature_counts_per_instrument(self) -> pd.DataFrame:
        """Return row counts in features table per instrument."""
        query = text("""
            SELECT i.id AS instrument_id, i.trading_symbol, i.groww_symbol, COUNT(f.id) AS feature_count
            FROM instruments i
            LEFT JOIN features f ON i.id = f.instrument_id
            WHERE i.is_selected_universe = TRUE
            GROUP BY i.id, i.trading_symbol, i.groww_symbol
            ORDER BY feature_count DESC;
        """)
        return pd.read_sql(query, self.engine)
