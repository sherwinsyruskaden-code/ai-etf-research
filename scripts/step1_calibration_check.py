"""
STEP 1 — Calibration data retrieval (READ ONLY, no experiment run yet).
"""
import os, warnings
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://etf_user:etf_password@localhost:5433/etf_research"
)
engine = create_engine(DB_URL)

HORIZON = 14
PROB_THRESHOLD = 0.35
SL_AVOID_THRESHOLD = 0.30

FOLD_WINDOWS = {
    1: ("2006-07-31", "2010-08-10"),
    2: ("2010-08-11", "2014-08-10"),
    3: ("2014-08-11", "2018-08-20"),
    4: ("2018-08-22", "2022-08-31"),
    5: ("2022-09-01", "2026-09-06"),
}

SEP = "=" * 80

with engine.connect() as conn:
    inst_df = pd.read_sql(
        text("SELECT id, trading_symbol, avg_daily_volume FROM instruments WHERE is_selected_universe = TRUE ORDER BY id"),
        conn
    )
inst_map = dict(zip(inst_df["id"], inst_df["trading_symbol"]))
inst_liq  = dict(zip(inst_df["id"], inst_df["avg_daily_volume"]))
print("Universe instruments:", list(inst_map.values()))

QUERY = text("""
    SELECT f.instrument_id, f.timestamp,
           o.open  AS entry_open_next, o.close AS close_curr,
           f.return_1d, f.return_3d, f.return_5d, f.return_10d, f.return_20d,
           f.sma_5, f.sma_20, f.sma_50,
           f.price_vs_sma5, f.price_vs_sma20, f.price_vs_sma50, f.sma20_vs_sma50,
           f.rsi_14, f.macd, f.macd_signal, f.macd_hist, f.roc_10,
           f.volatility_5d, f.volatility_10d, f.volatility_20d, f.atr_14,
           f.volume_today, f.volume_sma20, f.volume_ratio, f.volume_change,
           f.nifty50_return_1d, f.banknifty_return_1d, f.india_vix_level, f.india_vix_change_1d,
           f.sector_return_1d, f.commodity_return_1d,
           r.instrument_regime, r.market_regime,
           l.barrier_hit, l.days_to_hit, l.tp_level, l.sl_level, l.realized_return
    FROM features f
    JOIN ohlcv_daily o      ON f.instrument_id = o.instrument_id AND f.timestamp = o.timestamp
    JOIN market_regimes r   ON f.instrument_id = r.instrument_id AND f.timestamp = r.timestamp
    JOIN triple_barrier_labels l
                            ON f.instrument_id = l.instrument_id AND f.timestamp = l.timestamp
                           AND l.horizon = :horizon
    JOIN instruments i      ON f.instrument_id = i.id
    WHERE i.is_selected_universe = TRUE AND l.barrier_hit IS NOT NULL
    ORDER BY f.timestamp ASC, f.instrument_id ASC;
""")

with engine.connect() as conn:
    df_full = pd.read_sql(QUERY, conn, params={"horizon": HORIZON})

print(f"Total rows loaded: {len(df_full)}")
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"]).dt.tz_localize(None)
df_full["target"] = df_full["barrier_hit"].map({"TP": 0, "SL": 1, "TIME": 2})

regime_dummies = pd.get_dummies(
    df_full[["instrument_regime", "market_regime"]],
    prefix=["inst_regime", "mkt_regime"], dtype=float
)
df_clean = df_full.drop(columns=["instrument_regime", "market_regime"])
df_full2 = pd.concat([df_clean, regime_dummies], axis=1)

META_COLS = {
    "instrument_id","timestamp","target","barrier_hit","days_to_hit",
    "tp_level","sl_level","realized_return","entry_open_next","close_curr"
}
CONTEXT_COLS = [
    "nifty50_return_1d","banknifty_return_1d","india_vix_level",
    "india_vix_change_1d","sector_return_1d","commodity_return_1d"
]
for c in CONTEXT_COLS:
    if c in df_full2.columns:
        df_full2[f"{c}_is_missing"] = df_full2[c].isna().astype(float)

FEATURE_NAMES = [c for c in df_full2.columns if c not in META_COLS]
non_context = [c for c in FEATURE_NAMES if c not in CONTEXT_COLS]
df_full2[non_context] = df_full2[non_context].fillna(0.0)

def fold_slice(fold_id):
    s, e = FOLD_WINDOWS[fold_id]
    mask = (df_full2["timestamp"] >= s) & (df_full2["timestamp"] <= e)
    return df_full2[mask].copy()

folds_data = {fid: fold_slice(fid) for fid in FOLD_WINDOWS}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_14d_v1.joblib")
artifact    = joblib.load(MODEL_PATH)
xgb_model   = artifact["model"]
FEATURE_NAMES_MODEL = artifact.get("feature_names", FEATURE_NAMES)
FEATURE_NAMES_MODEL = [f for f in FEATURE_NAMES_MODEL if f in df_full2.columns]

for fid_key, df_slice in folds_data.items():
    for col in FEATURE_NAMES_MODEL:
        if col not in df_slice.columns:
            df_slice[col] = 0.0

for fid, df_f in folds_data.items():
    X = df_f[FEATURE_NAMES_MODEL].fillna(0.0)
    probs = xgb_model.predict_proba(X)
    df_f["prob_tp"]   = probs[:, 0]
    df_f["prob_sl"]   = probs[:, 1]
    df_f["prob_time"] = probs[:, 2]

# Collect signal rows at baseline filter
all_signal_list = []
for fid, df_f in folds_data.items():
    mask = (df_f["prob_tp"] >= PROB_THRESHOLD) & (df_f["prob_sl"] < SL_AVOID_THRESHOLD)
    sigs = df_f[mask].copy()
    if not sigs.empty:
        sigs["fold_id"] = fid
        sigs["instrument"] = sigs["instrument_id"].map(inst_map)
        all_signal_list.append(sigs)

all_signals = pd.concat(all_signal_list, ignore_index=True) if all_signal_list else pd.DataFrame()
print(f"\nTotal signal observations at baseline filter: {len(all_signals)}")

if not all_signals.empty:
    all_signals["actual_tp"] = (all_signals["barrier_hit"] == "TP").astype(int)
    base_rate = all_signals["actual_tp"].mean()
    print(f"\nBase TP rate among all filtered signals: {base_rate:.4f} ({base_rate:.1%})")

    bins   = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ["0.35-0.40","0.40-0.45","0.45-0.50","0.50-0.55","0.55-0.60","0.60-0.65","0.65-0.70","0.70+"]

    all_signals["bucket"] = pd.cut(all_signals["prob_tp"], bins=bins, labels=labels, right=False)
    grp = all_signals.groupby("bucket", observed=True).agg(
        count=("actual_tp","count"),
        actual_tp_rate=("actual_tp","mean")
    )

    print()
    print(SEP)
    print("  CALIBRATION TABLE -- ALL FOLDS COMBINED (prob_tp bucket -> actual TP rate)")
    print(SEP)
    print(f"  {'prob_tp bucket':<16} {'N':>6} {'actual_TP_rate':>15}   {'vs_base':>9}   {'lift':>7}")
    print("-" * 65)
    for bucket, row in grp.iterrows():
        lift = row["actual_tp_rate"] / base_rate if base_rate > 0 else 1.0
        above = row["actual_tp_rate"] - base_rate
        marker = " <<< ABOVE BASE" if row["actual_tp_rate"] > base_rate + 0.03 else ""
        print(f"  {str(bucket):<16} {int(row['count']):>6} {row['actual_tp_rate']:>14.1%}   "
              f"{above:>+8.1%}   {lift:>5.2f}x{marker}")

    print()
    print(SEP)
    print("  PER-FOLD CALIBRATION")
    print(SEP)
    for fid in FOLD_WINDOWS:
        sub = all_signals[all_signals["fold_id"] == fid].copy()
        if sub.empty:
            print(f"  Fold {fid}: no signals")
            continue
        sub["bucket"] = pd.cut(sub["prob_tp"], bins=bins, labels=labels, right=False)
        grp2 = sub.groupby("bucket", observed=True).agg(
            count=("actual_tp","count"),
            actual_tp_rate=("actual_tp","mean")
        )
        fold_base = sub["actual_tp"].mean()
        n_total = len(sub)
        print(f"\n  Fold {fid} | base_TP={fold_base:.1%} | n_signals={n_total}")
        for b, r in grp2.iterrows():
            if r["count"] > 0:
                lift2 = r["actual_tp_rate"] / fold_base if fold_base > 0 else 1.0
                print(f"    {str(b):<15} N={int(r['count']):>5}  actual_tp={r['actual_tp_rate']:.1%}  lift={lift2:.2f}x")

    print()
    print(SEP)
    print("  SIGNAL COUNT AT VARIOUS THRESHOLDS (prob_tp >= X, prob_sl < 0.30)")
    print(SEP)
    for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        sub = all_signals[all_signals["prob_tp"] >= thr]
        n = len(sub)
        tp_rate = sub["actual_tp"].mean() if not sub.empty else 0.0
        pct = n / len(all_signals) * 100 if len(all_signals) > 0 else 0
        print(f"  prob_tp >= {thr:.2f}: n={n:>6}  actual_tp_rate={tp_rate:.1%}  ({pct:.1f}% of baseline signals)")

print()
print(SEP)
print("  CALIBRATION DATA RETRIEVAL COMPLETE")
print(SEP)
