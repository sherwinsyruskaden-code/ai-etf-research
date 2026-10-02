import json
import pandas as pd
from sqlalchemy import text
from src.config.settings import DATABASE_URL
from src.data.database import get_db_engine
from src.models.repository import ModelRepository


def report_phase4_metrics():
    engine = get_db_engine(DATABASE_URL)
    repo = ModelRepository(engine)

    df_runs = repo.get_model_runs()

    print("=" * 100)
    print("PHASE 4 MODEL TRAINING & EVALUATION REPORT (HELD-OUT PURGED VALIDATION SET)")
    print("=" * 100)

    if df_runs.empty:
        print("No model runs found in database.")
        return

    for h in [7, 14]:
        print(f"\n" + "=" * 100)
        print(f"FORECAST HORIZON: {h} DAYS")
        print("=" * 100)

        sub_runs = df_runs[df_runs["horizon"] == h]

        for _, row in sub_runs.iterrows():
            m_name = row["model_name"]
            acc = float(row["accuracy"])
            train_rng = row["train_date_range"]
            val_rng = row["val_date_range"]

            per_class = json.loads(row["per_class_metrics"]) if isinstance(row["per_class_metrics"], str) else row["per_class_metrics"]
            cm = json.loads(row["confusion_matrix"]) if isinstance(row["confusion_matrix"], str) else row["confusion_matrix"]

            print(f"\n>>> MODEL: {m_name} (H={h}d)")
            print(f"    Train Period : {train_rng}")
            print(f"    Val Period   : {val_rng}")
            print(f"    Val Accuracy : {acc:.4f} ({acc:.2%})")
            print(f"    Macro F1     : {per_class['macro avg']['f1-score']:.4f}")
            print(f"    Per-Class Metrics:")
            print(f"      - TP   (Take-Profit) : Precision={per_class['TP']['precision']:.4f}, Recall={per_class['TP']['recall']:.4f}, F1={per_class['TP']['f1-score']:.4f} (Support={per_class['TP']['support']})")
            print(f"      - SL   (Stop-Loss)   : Precision={per_class['SL']['precision']:.4f}, Recall={per_class['SL']['recall']:.4f}, F1={per_class['SL']['f1-score']:.4f} (Support={per_class['SL']['support']})")
            print(f"      - TIME (Timeout)     : Precision={per_class['TIME']['precision']:.4f}, Recall={per_class['TIME']['recall']:.4f}, F1={per_class['TIME']['f1-score']:.4f} (Support={per_class['TIME']['support']})")
            print(f"    Confusion Matrix [Predicted TP, SL, TIME]:")
            print(f"      Actual TP   : {cm[0]}")
            print(f"      Actual SL   : {cm[1]}")
            print(f"      Actual TIME : {cm[2]}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    report_phase4_metrics()
