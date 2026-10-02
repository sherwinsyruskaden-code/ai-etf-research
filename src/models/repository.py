import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import joblib
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class ModelRepository:
    """
    Repository for persisting Phase 4 model evaluation runs into PostgreSQL
    and saving model artifacts (.joblib) to disk.
    """

    def __init__(self, engine: Engine, models_dir: Path = MODELS_DIR):
        self.engine = engine
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model_run_metrics(
        self,
        model_name: str,
        horizon: int,
        train_date_range: str,
        val_date_range: str,
        accuracy: float,
        per_class_metrics: Dict[str, Any],
        confusion_matrix: List[List[int]]
    ) -> int:
        """
        Insert model evaluation metrics into PostgreSQL model_runs table.
        """
        insert_sql = text("""
            INSERT INTO model_runs 
            (model_name, horizon, train_date_range, val_date_range, accuracy, per_class_metrics, confusion_matrix)
            VALUES 
            (:model_name, :horizon, :train_date_range, :val_date_range, :accuracy, :per_class_metrics, :confusion_matrix)
            RETURNING id;
        """)

        metrics_json = json.dumps(per_class_metrics)
        cm_json = json.dumps(confusion_matrix)

        with self.engine.begin() as conn:
            run_id = conn.execute(
                insert_sql,
                {
                    "model_name": model_name,
                    "horizon": horizon,
                    "train_date_range": train_date_range,
                    "val_date_range": val_date_range,
                    "accuracy": float(accuracy),
                    "per_class_metrics": metrics_json,
                    "confusion_matrix": cm_json,
                }
            ).scalar()

        logger.info(f"Saved model run metrics for [{model_name} (H={horizon})] into DB with ID: {run_id}")
        return run_id

    def save_model_artifact(
        self,
        model_obj: Any,
        filename: str
    ) -> str:
        """
        Save trained model artifact (.joblib) to models/ directory.
        """
        filepath = self.models_dir / filename
        joblib.dump(model_obj, filepath)
        logger.info(f"Saved model artifact to: {filepath.absolute()}")
        return str(filepath.absolute())

    def get_model_runs(self, horizon: Optional[int] = None) -> pd.DataFrame:
        """Fetch recorded model runs from PostgreSQL."""
        if horizon:
            query = text("SELECT * FROM model_runs WHERE horizon = :horizon ORDER BY id DESC;")
            return pd.read_sql(query, self.engine, params={"horizon": horizon})
        else:
            query = text("SELECT * FROM model_runs ORDER BY id DESC;")
            return pd.read_sql(query, self.engine)

    def clear_all_model_runs(self) -> int:
        """Wipe all records from model_runs table."""
        with self.engine.begin() as conn:
            res = conn.execute(text("DELETE FROM model_runs;"))
            count = res.rowcount
        logger.info(f"Wiped {count} old model run records from database.")
        return count
