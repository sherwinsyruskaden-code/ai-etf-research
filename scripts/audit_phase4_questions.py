import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sqlalchemy import text
from src.config.settings import DATABASE_URL
from src.data.database import get_db_engine
from src.models.data_loader import ModelDataLoader, CLASS_NAMES
from src.models.splitter import PurgedTimeSeriesSplit
from src.models.repository import ModelRepository


def audit_phase4():
    engine = get_db_engine(DATABASE_URL)
    loader = ModelDataLoader(engine)
    splitter = PurgedTimeSeriesSplit(val_pct=0.20)
    repo = ModelRepository(engine)

    print("=" * 100)
    print("PHASE 4 AUDIT: EDGE-OVER-BASE-RATE & MARKET CONTEXT FEATURE ANALYSIS")
    print("=" * 100)

    # ---------------------------------------------------------
    # PART 1: EDGE-OVER-BASE-RATE CHECK
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("1. EDGE-OVER-BASE-RATE CHECK (Validation Set)")
    print("=" * 100)

    for h in [7, 14]:
        df_full, feature_names = loader.load_dataset_for_horizon(h)
        df_train, df_val, info = splitter.split(df_full, horizon=h)

        total_val = len(df_val)
        y_val = df_val["target"].values

        # Class counts in validation set: 0: TP, 1: SL, 2: TIME
        tp_base_count = np.sum(y_val == 0)
        sl_base_count = np.sum(y_val == 1)
        time_base_count = np.sum(y_val == 2)

        tp_base_rate = tp_base_count / total_val
        sl_base_rate = sl_base_count / total_val
        time_base_rate = time_base_count / total_val

        print(f"\n>>> HORIZON {h} DAYS VALIDATION SET BASE RATES (N = {total_val}):")
        print(f"    - TP   Base Rate : {tp_base_count:5d} / {total_val} = {tp_base_rate:.4f} ({tp_base_rate:.2%})")
        print(f"    - SL   Base Rate : {sl_base_count:5d} / {total_val} = {sl_base_rate:.4f} ({sl_base_rate:.2%})")
        print(f"    - TIME Base Rate : {time_base_count:5d} / {total_val} = {time_base_rate:.4f} ({time_base_rate:.2%})")

        print(f"\n    Model Edge Comparison (Precision vs Base Rate):")
        print(f"    " + "-" * 105)
        print(f"    {'Model':20s} | {'Class':6s} | {'Base Rate':10s} | {'N_pred':8s} | {'Precision':10s} | {'Edge (pp)':12s} | {'Caveat / Status':20s}")
        print(f"    " + "-" * 105)

        sub_runs = repo.get_model_runs(horizon=h)
        for _, row in sub_runs.iterrows():
            m_name = row["model_name"]
            metrics = json.loads(row["per_class_metrics"]) if isinstance(row["per_class_metrics"], str) else row["per_class_metrics"]
            cm = json.loads(row["confusion_matrix"]) if isinstance(row["confusion_matrix"], str) else row["confusion_matrix"]

            # Predicted counts = column sums of confusion matrix
            n_pred_tp = int(cm[0][0] + cm[1][0] + cm[2][0])
            n_pred_sl = int(cm[0][1] + cm[1][1] + cm[2][1])

            tp_prec = metrics["TP"]["precision"]
            sl_prec = metrics["SL"]["precision"]

            tp_edge = (tp_prec - tp_base_rate) * 100.0
            sl_edge = (sl_prec - sl_base_rate) * 100.0

            tp_status = f"[UNRELIABLE: N={n_pred_tp}<50]" if n_pred_tp < 50 else "Reliable"
            sl_status = f"[UNRELIABLE: N={n_pred_sl}<50]" if n_pred_sl < 50 else "Reliable"

            print(f"    {m_name:20s} | {'TP':6s} | {tp_base_rate:9.2%} | {n_pred_tp:8d} | {tp_prec:9.2%} | {tp_edge:+11.2f}% pts | {tp_status:20s}")
            print(f"    {m_name:20s} | {'SL':6s} | {sl_base_rate:9.2%} | {n_pred_sl:8d} | {sl_prec:9.2%} | {sl_edge:+11.2f}% pts | {sl_status:20s}")
            print(f"    " + "-" * 105)

    # ---------------------------------------------------------
    # PART 2: MARKET-CONTEXT FEATURE AUDIT
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("2. MARKET-CONTEXT FEATURE AUDIT")
    print("=" * 100)

    df_full_7, feat_names_7 = loader.load_dataset_for_horizon(7)
    print(f"\nFull List of Features in Training Matrix ({len(feat_names_7)} total):")
    for idx, fcol in enumerate(feat_names_7, start=1):
        print(f"  {idx:2d}. {fcol}")

    # Market context feature columns
    ctx_cols = [
        "nifty50_return_1d", "banknifty_return_1d", "india_vix_level",
        "india_vix_change_1d", "sector_return_1d", "commodity_return_1d"
    ]

    print("\nMarket Context Column Null / Zero Value Audit in Feature Matrix:")
    print("-" * 80)
    for c in ctx_cols:
        if c in df_full_7.columns:
            series = df_full_7[c]
            null_cnt = series.isna().sum()
            zero_cnt = (series == 0.0).sum()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()
            print(f"  {c:22s} | Nulls: {null_cnt:5d} ({null_cnt/len(df_full_7):.1%}) | Zeros: {zero_cnt:5d} ({zero_cnt/len(df_full_7):.1%}) | Std: {std_val:.6f} | Range: [{min_val:.4f}, {max_val:.4f}]")
        else:
            print(f"  {c:22s} | MISSING FROM DATAFRAME!")

    # Load saved XGBoost models and inspect feature importance ranks
    print("\nFeature Importance Ranks for Market Context Features:")
    print("-" * 80)

    for h in [7, 14]:
        model_path = Path("models") / f"xgboost_{h}d_v1.joblib"
        if model_path.exists():
            art = joblib.load(model_path)
            xgb_model = art["model"]
            f_names = art["feature_names"]
            imps = xgb_model.feature_importances_

            df_imp = pd.DataFrame({"feature": f_names, "importance": imps})
            df_imp["rank"] = df_imp["importance"].rank(ascending=False, method="min").astype(int)
            df_imp = df_imp.sort_values("rank")

            print(f"\n>>> XGBoost (H={h}d) Market Context Feature Ranks (out of {len(f_names)} features):")
            for c in ctx_cols:
                row = df_imp[df_imp["feature"] == c]
                if not row.empty:
                    rk = row.iloc[0]["rank"]
                    val = row.iloc[0]["importance"]
                    print(f"    - {c:22s} : Rank #{rk:2d} / {len(f_names)} (Importance = {val:.6f})")
        else:
            print(f"Model file not found: {model_path}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    audit_phase4()
