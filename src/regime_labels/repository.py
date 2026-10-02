import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from sqlalchemy import text, Engine

logger = logging.getLogger(__name__)


class RegimeLabelRepository:
    """Repository for managing market_regimes and triple_barrier_labels in PostgreSQL."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def save_regimes_batch(self, instrument_id: int, df_regimes: pd.DataFrame) -> Dict[str, int]:
        """
        Upsert regime labels into market_regimes idempotently via ON CONFLICT.
        """
        if df_regimes.empty:
            return {"inserted": 0, "updated": 0}

        upsert_sql = text("""
            INSERT INTO market_regimes (instrument_id, timestamp, instrument_regime, market_regime)
            VALUES (:instrument_id, :timestamp, :instrument_regime, :market_regime)
            ON CONFLICT (instrument_id, timestamp) DO UPDATE SET
                instrument_regime = EXCLUDED.instrument_regime,
                market_regime = EXCLUDED.market_regime;
        """)

        records = []
        for _, row in df_regimes.iterrows():
            ts_val = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
            records.append({
                "instrument_id": instrument_id,
                "timestamp": ts_val,
                "instrument_regime": str(row["instrument_regime"]),
                "market_regime": str(row["market_regime"])
            })

        count_sql = text("SELECT COUNT(*) FROM market_regimes WHERE instrument_id = :instrument_id;")

        with self.engine.begin() as conn:
            c_before = conn.execute(count_sql, {"instrument_id": instrument_id}).scalar() or 0
            conn.execute(upsert_sql, records)
            c_after = conn.execute(count_sql, {"instrument_id": instrument_id}).scalar() or 0

        inserted = max(0, c_after - c_before)
        updated = len(records) - inserted
        return {"inserted": inserted, "updated": updated}

    def save_labels_batch(self, instrument_id: int, df_labels: pd.DataFrame) -> Dict[str, int]:
        """
        Upsert triple-barrier labels into triple_barrier_labels idempotently via ON CONFLICT.
        """
        if df_labels.empty:
            return {"inserted": 0, "updated": 0}

        upsert_sql = text("""
            INSERT INTO triple_barrier_labels (instrument_id, timestamp, horizon, barrier_hit, days_to_hit, tp_level, sl_level, realized_return)
            VALUES (:instrument_id, :timestamp, :horizon, :barrier_hit, :days_to_hit, :tp_level, :sl_level, :realized_return)
            ON CONFLICT (instrument_id, timestamp, horizon) DO UPDATE SET
                barrier_hit = EXCLUDED.barrier_hit,
                days_to_hit = EXCLUDED.days_to_hit,
                tp_level = EXCLUDED.tp_level,
                sl_level = EXCLUDED.sl_level,
                realized_return = EXCLUDED.realized_return;
        """)

        records = []
        for _, row in df_labels.iterrows():
            ts_val = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
            rec = {
                "instrument_id": instrument_id,
                "timestamp": ts_val,
                "horizon": int(row["horizon"]),
                "barrier_hit": str(row["barrier_hit"]) if pd.notna(row["barrier_hit"]) and row["barrier_hit"] is not None else None,
                "days_to_hit": int(row["days_to_hit"]) if pd.notna(row["days_to_hit"]) and row["days_to_hit"] is not None else None,
                "tp_level": float(row["tp_level"]) if pd.notna(row["tp_level"]) and row["tp_level"] is not None else None,
                "sl_level": float(row["sl_level"]) if pd.notna(row["sl_level"]) and row["sl_level"] is not None else None,
                "realized_return": float(row["realized_return"]) if pd.notna(row["realized_return"]) and row["realized_return"] is not None else None,
            }
            records.append(rec)

        count_sql = text("SELECT COUNT(*) FROM triple_barrier_labels WHERE instrument_id = :instrument_id;")

        with self.engine.begin() as conn:
            c_before = conn.execute(count_sql, {"instrument_id": instrument_id}).scalar() or 0
            conn.execute(upsert_sql, records)
            c_after = conn.execute(count_sql, {"instrument_id": instrument_id}).scalar() or 0

        inserted = max(0, c_after - c_before)
        updated = len(records) - inserted
        return {"inserted": inserted, "updated": updated}

    def get_regimes(self, instrument_id: int) -> pd.DataFrame:
        """Fetch chronological regimes for an instrument."""
        query = text("""
            SELECT * FROM market_regimes
            WHERE instrument_id = :instrument_id
            ORDER BY timestamp ASC;
        """)
        return pd.read_sql(query, self.engine, params={"instrument_id": instrument_id})

    def get_labels(self, instrument_id: int, horizon: Optional[int] = None) -> pd.DataFrame:
        """Fetch chronological triple-barrier labels for an instrument."""
        if horizon:
            query = text("""
                SELECT * FROM triple_barrier_labels
                WHERE instrument_id = :instrument_id AND horizon = :horizon
                ORDER BY timestamp ASC;
            """)
            return pd.read_sql(query, self.engine, params={"instrument_id": instrument_id, "horizon": horizon})
        else:
            query = text("""
                SELECT * FROM triple_barrier_labels
                WHERE instrument_id = :instrument_id
                ORDER BY horizon ASC, timestamp ASC;
            """)
            return pd.read_sql(query, self.engine, params={"instrument_id": instrument_id})

    def clear_all_regimes_and_labels(self) -> Tuple[int, int]:
        """Wipe all records from market_regimes and triple_barrier_labels tables."""
        with self.engine.begin() as conn:
            res_reg = conn.execute(text("DELETE FROM market_regimes;"))
            res_lbl = conn.execute(text("DELETE FROM triple_barrier_labels;"))
            wiped_reg = res_reg.rowcount
            wiped_lbl = res_lbl.rowcount
        logger.info(f"Wiped {wiped_reg} regime records and {wiped_lbl} label records from DB.")
        return wiped_reg, wiped_lbl
