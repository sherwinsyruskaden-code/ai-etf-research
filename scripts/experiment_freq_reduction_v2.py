"""
EXPERIMENT — NOT PART OF LOCKED PHASE 5 RESULTS
=================================================
Frequency Reduction via Confidence Threshold — CORRECTED VERSION

CRITICAL CORRECTION from v1:
  v1 used the Phase 4 POOLED model (xgboost_14d_v1.joblib) which was trained
  on ALL data including validation windows -> DATA LEAKAGE -> invalid results.
  
  This corrected version replicates Phase 5 engine.py exactly: XGBoost is 
  RETRAINED on each fold's purged training set (same as Phase 5). Only the 
  entry threshold changes. All other parameters identical.

Pre-registered thresholds (locked before any experiment):
  Threshold A: prob_tp >= 0.45, prob_sl < 0.30
  Threshold B: prob_tp >= 0.50, prob_sl < 0.30

No other changes. Results stored in experiment_freq_reduction_results_v2.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://etf_user:etf_password@localhost:5433/etf_research"
)
engine = create_engine(DB_URL)

HORIZON        = 14
BASELINE_PT    = 0.35       # Original strategy threshold
BASELINE_SL_T  = 0.30       # SL-avoid threshold — UNCHANGED
THRESHOLD_A    = 0.45       # PRE-REGISTERED moderate
THRESHOLD_B    = 0.50       # PRE-REGISTERED strict
SEED           = 42

GOLD_SILVER_ETFS = {
    "SILVERBEES", "GOLDBEES", "GOLDIETF", "SILVERIETF", "HDFCGOLD", "GOLD1"
}

FOLD_WINDOWS = {
    1: ("2006-07-31", "2010-08-10"),
    2: ("2010-08-11", "2014-08-10"),
    3: ("2014-08-11", "2018-08-20"),
    4: ("2018-08-22", "2022-08-31"),
    5: ("2022-09-01", "2026-09-06"),
}

PRE_REGISTERED_KILL_GATE = {
    "min_post_cost_sharpe":   0.50,
    "min_post_cost_win_rate": 0.40,
    "required_fold_pass_ratio": 0.60,   # >= 3/5 folds
}

SEP  = "=" * 90
SEP2 = "-" * 70

# -----------------------------------------------------------------------
# 0. Instruments
# -----------------------------------------------------------------------
with engine.connect() as conn:
    inst_df = pd.read_sql(
        text("SELECT id, trading_symbol, avg_daily_volume FROM instruments WHERE is_selected_universe = TRUE ORDER BY id"),
        conn
    )
inst_map = dict(zip(inst_df["id"], inst_df["trading_symbol"]))
inst_liq  = dict(zip(inst_df["id"], inst_df["avg_daily_volume"]))
print("Universe:", list(inst_map.values()))

# -----------------------------------------------------------------------
# 1. Load full dataset — IDENTICAL to Phase 5 engine._load_backtest_dataset
# -----------------------------------------------------------------------
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
    df_raw = pd.read_sql(QUERY, conn, params={"horizon": HORIZON})

print(f"Total rows loaded: {len(df_raw)}")
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"]).dt.tz_localize(None)
df_raw["target"] = df_raw["barrier_hit"].map({"TP": 0, "SL": 1, "TIME": 2})

# One-hot encode regimes (same as Phase 5)
regime_dummies = pd.get_dummies(
    df_raw[["instrument_regime", "market_regime"]],
    prefix=["inst_regime", "mkt_regime"], dtype=float
)
df_clean = df_raw.drop(columns=["instrument_regime", "market_regime"])
df_full  = pd.concat([df_clean, regime_dummies], axis=1)

META_COLS = {
    "instrument_id", "timestamp", "target", "barrier_hit", "days_to_hit",
    "tp_level", "sl_level", "realized_return", "entry_open_next", "close_curr"
}
CONTEXT_COLS = [
    "nifty50_return_1d", "banknifty_return_1d", "india_vix_level",
    "india_vix_change_1d", "sector_return_1d", "commodity_return_1d"
]
for c in CONTEXT_COLS:
    if c in df_full.columns:
        df_full[f"{c}_is_missing"] = df_full[c].isna().astype(float)

feature_names = [c for c in df_full.columns if c not in META_COLS]
non_context   = [c for c in feature_names if c not in CONTEXT_COLS]
df_full[non_context] = df_full[non_context].fillna(0.0)

# -----------------------------------------------------------------------
# 2. Build walk-forward folds — IDENTICAL to Phase 5 engine._create_walk_forward_folds
# -----------------------------------------------------------------------
unique_dates = df_full["timestamp"].drop_duplicates().sort_values().tolist()
n_dates = len(unique_dates)
n_folds = 5
val_size = n_dates // (n_folds + 1)

folds = []
for f in range(n_folds):
    val_start_idx = (f + 1) * val_size
    val_end_idx   = min(n_dates, (f + 2) * val_size if f < n_folds - 1 else n_dates)
    val_start_date = unique_dates[val_start_idx]
    val_end_date   = unique_dates[val_end_idx - 1]

    val_mask   = (df_full["timestamp"] >= val_start_date) & (df_full["timestamp"] <= val_end_date)
    df_val     = df_full[val_mask].copy()

    # PURGING: same as Phase 5
    purge_cutoff_idx  = max(0, val_start_idx - HORIZON)
    purge_cutoff_date = unique_dates[purge_cutoff_idx]
    train_mask = df_full["timestamp"] < purge_cutoff_date
    df_train   = df_full[train_mask].copy()

    val_start_str = pd.Timestamp(val_start_date).strftime("%Y-%m-%d")
    val_end_str   = pd.Timestamp(val_end_date).strftime("%Y-%m-%d")
    folds.append((df_train, df_val, f + 1, f"{val_start_str} to {val_end_str}"))

print(f"Folds created: {len(folds)}")
for _, _, fid, drange in folds:
    print(f"  Fold {fid}: {drange}")

# -----------------------------------------------------------------------
# 3. Cost model — IDENTICAL to Phase 5
# -----------------------------------------------------------------------
def compute_cost(adtv):
    spread = 0.0005 if adtv > 100_000_000 else (0.0010 if adtv > 10_000_000 else 0.0020)
    slip   = 0.0005
    stt    = 0.0020
    stamp  = 0.00015
    exch   = 0.0000345 * 2
    sebi   = 0.000001  * 2
    gst    = (0.0005 + exch + sebi) * 0.18 * 2
    return slip * 2 + spread + stt + stamp + exch + sebi + gst

# -----------------------------------------------------------------------
# 4. Simulate trades — IDENTICAL to Phase 5 engine._simulate_trades
# -----------------------------------------------------------------------
def simulate_trades(df_val_preds, prob_threshold, sl_threshold=0.30):
    trades = []
    df_sorted = df_val_preds.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
    for i in range(len(df_sorted) - 1):
        cur = df_sorted.iloc[i]
        nxt = df_sorted.iloc[i + 1]
        if cur["instrument_id"] != nxt["instrument_id"]:
            continue
        p_tp = float(cur["prob_tp"])
        p_sl = float(cur["prob_sl"])
        if p_tp >= prob_threshold and p_sl < sl_threshold:
            entry   = float(nxt["entry_open_next"])
            inst_id = int(cur["instrument_id"])
            adtv    = float(inst_liq.get(inst_id, 50_000_000))
            hit     = cur["barrier_hit"]
            c_entry = float(cur["close_curr"])
            if hit == "TP" and pd.notna(cur["tp_level"]):
                raw_exit = float(cur["tp_level"])
            elif hit == "SL" and pd.notna(cur["sl_level"]):
                raw_exit = float(cur["sl_level"])
            else:
                rel = float(cur["realized_return"]) if pd.notna(cur["realized_return"]) else 0.0
                raw_exit = c_entry * (1.0 + rel)
            raw_ret   = (raw_exit - entry) / entry if entry > 0 else 0.0
            total_cost = compute_cost(adtv)
            trades.append({
                "timestamp":   cur["timestamp"],
                "instrument_id": inst_id,
                "instrument":  inst_map.get(inst_id, str(inst_id)),
                "barrier_hit": hit,
                "days_to_hit": float(cur["days_to_hit"]) if pd.notna(cur["days_to_hit"]) else float(HORIZON),
                "prob_tp":     p_tp,
                "raw_return":  raw_ret,
                "net_return":  raw_ret - total_cost,
                "cost_drag":   total_cost,
            })
    return trades

# -----------------------------------------------------------------------
# 5. Buy-and-hold benchmark — IDENTICAL to Phase 5
# -----------------------------------------------------------------------
def simulate_buy_and_hold(df_val_preds):
    daily_series = []
    for inst_id, group in df_val_preds.groupby("instrument_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2:
            continue
        closes    = g["close_curr"].values.astype(float)
        daily_raw = np.diff(closes) / closes[:-1]
        adtv      = float(inst_liq.get(int(inst_id), 50_000_000))
        tc        = compute_cost(adtv)
        n_days    = len(daily_raw)
        daily_cost = tc / n_days if n_days > 0 else 0.0
        daily_series.append(daily_raw - daily_cost)
    if not daily_series:
        return {"post_cost_sharpe": 0.0}
    max_len = max(len(s) for s in daily_series)
    padded  = np.array([np.pad(s, (0, max_len - len(s)), constant_values=0.0) for s in daily_series])
    port    = padded.mean(axis=0)
    mu, sig = float(np.mean(port)), float(np.std(port))
    sh = float((mu / sig) * np.sqrt(252)) if sig > 0 else 0.0
    return {"post_cost_sharpe": sh}

# -----------------------------------------------------------------------
# 6. Metrics
# -----------------------------------------------------------------------
def calc_metrics(trades):
    if not trades:
        return {"num_trades": 0, "win_rate_pre": 0.0, "win_rate_post": 0.0,
                "avg_raw": 0.0, "avg_net": 0.0, "pre_cost_sharpe": 0.0,
                "post_cost_sharpe": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0}
    df_t = pd.DataFrame(trades)
    raw, net = df_t["raw_return"].values, df_t["net_return"].values
    n = len(df_t)
    wr_pre, wr_post = float(np.mean(raw > 0)), float(np.mean(net > 0))
    avg_raw, avg_net = float(np.mean(raw)), float(np.mean(net))
    std_raw, std_net = np.std(raw), np.std(net)
    sh_pre  = float((avg_raw / std_raw) * np.sqrt(252)) if std_raw > 0 else 0.0
    sh_post = float((avg_net / std_net) * np.sqrt(252)) if std_net > 0 else 0.0
    cum_eq  = np.cumprod(1.0 + net)
    peak    = np.maximum.accumulate(cum_eq)
    max_dd  = float(np.min((cum_eq - peak) / peak)) if len(cum_eq) > 0 else 0.0
    gains   = net[net > 0].sum()
    losses  = np.abs(net[net < 0].sum())
    pf      = float(gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
    return {"num_trades": n, "win_rate_pre": wr_pre, "win_rate_post": wr_post,
            "avg_raw": avg_raw, "avg_net": avg_net, "pre_cost_sharpe": sh_pre,
            "post_cost_sharpe": sh_post, "max_drawdown": max_dd, "profit_factor": pf}

def concentration_stats(trades):
    if not trades:
        return {"gs_trade_pct": 0.0, "gs_profit_pct": 0.0, "gs_trades": 0, "total_trades": 0}
    df_t = pd.DataFrame(trades)
    df_t["is_gs"] = df_t["instrument"].isin(GOLD_SILVER_ETFS)
    gs_n = df_t["is_gs"].sum()
    total = len(df_t)
    total_pft = df_t["net_return"].sum()
    gs_pft = df_t[df_t["is_gs"]]["net_return"].sum()
    return {
        "gs_trade_pct":  gs_n / total * 100 if total > 0 else 0.0,
        "gs_profit_pct": gs_pft / total_pft * 100 if total_pft != 0 else 0.0,
        "gs_trades":     int(gs_n),
        "total_trades":  int(total),
    }

def trades_per_year(trades, fold_id):
    if not trades:
        return 0.0
    s, e = FOLD_WINDOWS[fold_id]
    yrs = (pd.Timestamp(e) - pd.Timestamp(s)).days / 365.25
    return len(trades) / yrs if yrs > 0 else 0.0

def calibration_within(trades):
    if not trades:
        return pd.DataFrame()
    df_t = pd.DataFrame(trades)
    df_t["actual_tp"] = (df_t["barrier_hit"] == "TP").astype(int)
    bins   = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ["0.35-0.40","0.40-0.45","0.45-0.50","0.50-0.55",
              "0.55-0.60","0.60-0.65","0.65-0.70","0.70+"]
    df_t["bucket"] = pd.cut(df_t["prob_tp"], bins=bins, labels=labels, right=False)
    return df_t.groupby("bucket", observed=True).agg(
        count=("actual_tp","count"), actual_tp_rate=("actual_tp","mean")
    )

# -----------------------------------------------------------------------
# 7. MAIN LOOP — per fold, retrain XGBoost, simulate 3 variants + B&H
# -----------------------------------------------------------------------
print()
print(SEP)
print("  Running walk-forward experiment (retraining per fold, same as Phase 5)...")
print(SEP)

fold_results = {}   # fold_id -> {baseline, thr_a, thr_b, bnh}

for df_train, df_val, fid, date_range in folds:
    print(f"\n--- FOLD {fid} ({date_range}) ---")
    X_train = df_train[feature_names].fillna(0.0)
    y_train = df_train["target"].values
    X_val   = df_val[feature_names].fillna(0.0)

    # RETRAIN — identical hyperparameters to Phase 5 engine
    xgb = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        eval_metric="mlogloss", random_state=SEED, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    print(f"  Model trained on {len(df_train)} rows.")

    probs = xgb.predict_proba(X_val)
    df_val_preds = df_val.copy()
    df_val_preds["prob_tp"]   = probs[:, 0]
    df_val_preds["prob_sl"]   = probs[:, 1]
    df_val_preds["prob_time"] = probs[:, 2]

    trades_base = simulate_trades(df_val_preds, BASELINE_PT, BASELINE_SL_T)
    trades_a    = simulate_trades(df_val_preds, THRESHOLD_A, BASELINE_SL_T)
    trades_b    = simulate_trades(df_val_preds, THRESHOLD_B, BASELINE_SL_T)
    bnh_m       = simulate_buy_and_hold(df_val_preds)

    fold_results[fid] = {
        "date_range": date_range,
        "baseline": {"trades": trades_base, "metrics": calc_metrics(trades_base)},
        "thr_a":    {"trades": trades_a,    "metrics": calc_metrics(trades_a)},
        "thr_b":    {"trades": trades_b,    "metrics": calc_metrics(trades_b)},
        "bnh":      {"metrics": bnh_m},
    }
    print(f"  Trades: baseline={len(trades_base)} | thr_a(0.45)={len(trades_a)} | thr_b(0.50)={len(trades_b)}")
    print(f"  B&H Sharpe: {bnh_m['post_cost_sharpe']:.2f}")

# -----------------------------------------------------------------------
# 8. SIDE-BY-SIDE COMPARISON REPORT
# -----------------------------------------------------------------------
print()
print(SEP)
print("  EXPERIMENT -- NOT PART OF LOCKED PHASE 5 RESULTS")
print("  Frequency Reduction via Confidence Threshold (Corrected v2)")
print("  Per-fold retrained XGBoost (identical to Phase 5). Only threshold changed.")
print("  Pre-registered thresholds: A=0.45, B=0.50 | SL-avoid=0.30 (unchanged)")
print(SEP)

VARIANTS = [
    ("baseline", "ORIGINAL (thr=0.35)"),
    ("thr_a",    "THRESHOLD A (thr=0.45)"),
    ("thr_b",    "THRESHOLD B (thr=0.50)"),
]

verdict_summary = {}

for v_key, v_label in VARIANTS:
    print()
    print(f">>> {v_label}")
    print(SEP2)
    hdr = (f"{'Fold':<5} {'Period':<28} {'Trd/yr':>7} {'WR_pre':>7} {'WR_post':>8} "
           f"{'Sh_pre':>8} {'Sh_post':>8} {'MaxDD':>8} {'PF':>6} {'GS_tr%':>8} {'GS_pft%':>8} {'KillGate':>10}")
    print(hdr)
    print("-" * len(hdr))

    fold_verdicts = []
    agg = {"tpy": [], "wr_pre": [], "wr_post": [], "sh_pre": [], "sh_post": [], "dd": []}

    for fid in FOLD_WINDOWS:
        m    = fold_results[fid][v_key]["metrics"]
        bm   = fold_results[fid]["bnh"]["metrics"]
        trd  = fold_results[fid][v_key]["trades"]
        s, e = FOLD_WINDOWS[fid]
        tpy  = trades_per_year(trd, fid)
        conc = concentration_stats(trd)
        kg   = (
            m["post_cost_sharpe"] >= PRE_REGISTERED_KILL_GATE["min_post_cost_sharpe"] and
            m["win_rate_post"]    >= PRE_REGISTERED_KILL_GATE["min_post_cost_win_rate"] and
            m["post_cost_sharpe"] > bm["post_cost_sharpe"]
        )
        fold_verdicts.append(kg)
        for k, v in zip(["tpy","wr_pre","wr_post","sh_pre","sh_post","dd"],
                         [tpy, m["win_rate_pre"], m["win_rate_post"],
                          m["pre_cost_sharpe"], m["post_cost_sharpe"], m["max_drawdown"]]):
            agg[k].append(v)
        gs_tr, gs_pft = conc["gs_trade_pct"], conc["gs_profit_pct"]
        print(
            f"  {fid:<3} {s} to {e}  {tpy:>7.0f} "
            f"{m['win_rate_pre']:>7.1%} {m['win_rate_post']:>7.1%} "
            f"{m['pre_cost_sharpe']:>8.2f} {m['post_cost_sharpe']:>8.2f} "
            f"{m['max_drawdown']:>8.1%} {m['profit_factor']:>6.2f} "
            f"{gs_tr:>7.1f}% {gs_pft:>8.1f}%  {'PASS' if kg else 'FAIL':>8}"
        )

    passes  = sum(fold_verdicts)
    n_folds = len(FOLD_WINDOWS)
    is_pass = (passes / n_folds) >= PRE_REGISTERED_KILL_GATE["required_fold_pass_ratio"]
    verdict_summary[v_key] = {
        "verdicts": fold_verdicts, "passes": passes, "is_pass": is_pass,
        "agg": agg
    }

    print(SEP2)
    print(f"  AGGREGATED | Trd/yr={np.mean(agg['tpy']):.0f} | WR_pre={np.mean(agg['wr_pre']):.1%} | "
          f"WR_post={np.mean(agg['wr_post']):.1%} | Sh_pre={np.mean(agg['sh_pre']):.2f} | "
          f"Sh_post={np.mean(agg['sh_post']):.2f} (std={np.std(agg['sh_post']):.2f}) | MaxDD={np.mean(agg['dd']):.1%}")
    print(f"  KILL GATE: {passes}/{n_folds} folds passed ({passes/n_folds:.0%}) -- "
          f"Required >=60% -- VERDICT: {'PASSED' if is_pass else 'FAILED'}")

# -----------------------------------------------------------------------
# 9. CONCENTRATION REPORT
# -----------------------------------------------------------------------
print()
print(SEP)
print("  CONCENTRATION CHECK (pre-registered: gold/silver ETFs)")
print("  Gold/Silver ETFs:", sorted(GOLD_SILVER_ETFS))
print(SEP)
print(f"  {'Fold':<5} {'Variant':<22} {'GS Trades':>10} {'Total Trd':>10} {'GS_trd%':>9} {'GS_pft%':>9}")
print("-" * 75)
for fid in FOLD_WINDOWS:
    for v_key, v_label in VARIANTS:
        trd  = fold_results[fid][v_key]["trades"]
        conc = concentration_stats(trd)
        print(f"  {fid:<5} {v_label:<22} {conc['gs_trades']:>10} {conc['total_trades']:>10} "
              f"{conc['gs_trade_pct']:>8.1f}% {conc['gs_profit_pct']:>8.1f}%")
    print()

# -----------------------------------------------------------------------
# 10. CALIBRATION WITHIN FILTERED TRADES
# -----------------------------------------------------------------------
print()
print(SEP)
print("  CALIBRATION WITHIN FILTERED TRADE SET — ALL FOLDS COMBINED")
print(SEP)
for v_key, v_label in VARIANTS:
    all_trd = []
    for fid in FOLD_WINDOWS:
        all_trd.extend(fold_results[fid][v_key]["trades"])
    if not all_trd:
        continue
    calib = calibration_within(all_trd)
    tp_rate = np.mean([(t["barrier_hit"] == "TP") for t in all_trd])
    print(f"\n  {v_label} | overall_TP={tp_rate:.1%} | n_trades={len(all_trd)}")
    print(f"  {'bucket':<15} {'N':>6} {'actual_TP':>10}")
    print("  " + "-" * 35)
    for bucket, row in calib.iterrows():
        if row["count"] > 0:
            print(f"  {str(bucket):<15} {int(row['count']):>6} {row['actual_tp_rate']:>9.1%}")

# -----------------------------------------------------------------------
# 11. FINAL VERDICT SUMMARY
# -----------------------------------------------------------------------
print()
print(SEP)
print("  FINAL VERDICT SUMMARY — EXPERIMENT vs ORIGINAL KILL GATE")
print("  (Kill gate unchanged: Sharpe>=0.50, WR>=40%, beats B&H, >=3/5 folds)")
print(SEP)

# Also compare directly against Phase 5 stored baseline
print()
print("  Phase 5 Locked Results (sl_filtered, 14d) — FOR COMPARISON:")
with engine.connect() as conn:
    phase5_df = pd.read_sql(
        text("SELECT fold_id, num_trades, win_rate, pre_cost_sharpe, post_cost_sharpe FROM backtest_results WHERE horizon=14 AND strategy_variant='sl_filtered' ORDER BY fold_id"),
        conn
    )
print(f"  {'Fold':<5} {'Trades':>7} {'WinRate':>8} {'PreSharpe':>10} {'PostSharpe':>11}")
for _, r in phase5_df.iterrows():
    print(f"  {int(r['fold_id']):<5} {int(r['num_trades']):>7} {float(r['win_rate']):>8.1%} "
          f"{float(r['pre_cost_sharpe']):>10.2f} {float(r['post_cost_sharpe']):>11.2f}")
p5_passes = sum([
    float(phase5_df[phase5_df.fold_id == fid]["post_cost_sharpe"].iloc[0]) >= 0.50 and
    float(phase5_df[phase5_df.fold_id == fid]["win_rate"].iloc[0]) >= 0.40
    for fid in range(1, 6)
])
print(f"  Phase 5 Kill Gate: {p5_passes}/5 folds passed (threshold check only, B&H comparison requires bnh_sharpe)")

print()
for v_key, v_label in VARIANTS:
    vs = verdict_summary[v_key]
    fold_str = " | ".join([f"F{fid}={'PASS' if p else 'FAIL'}"
                           for fid, p in zip(FOLD_WINDOWS, vs["verdicts"])])
    all_trd = []
    for fid in FOLD_WINDOWS:
        all_trd.extend(fold_results[fid][v_key]["trades"])
    conc = concentration_stats(all_trd)
    gs_label = ("GOLD-CONCENTRATED"
                if conc["gs_trade_pct"] > 50 or conc["gs_profit_pct"] > 60
                else "BROAD-BASED (not gold-dominated)")

    print(f"  {v_label}")
    print(f"    Per-fold: {fold_str}")
    print(f"    Overall: {vs['passes']}/{len(FOLD_WINDOWS)} folds PASSED -- VERDICT: {'PASSED' if vs['is_pass'] else 'FAILED'}")
    print(f"    Concentration: {conc['gs_trade_pct']:.1f}% of trades, {conc['gs_profit_pct']:.1f}% of profit from GS ETFs -> {gs_label}")
    print()

# -----------------------------------------------------------------------
# 12. Save to NEW table (v2) — backtest_results NOT touched
# -----------------------------------------------------------------------
CREATE_SQL = text("""
    CREATE TABLE IF NOT EXISTS experiment_freq_reduction_results_v2 (
        id SERIAL PRIMARY KEY,
        experiment_label VARCHAR(80) NOT NULL DEFAULT 'freq_reduction_corrected_v2',
        fold_id INTEGER NOT NULL,
        horizon INTEGER NOT NULL,
        date_range VARCHAR(100) NOT NULL,
        strategy_variant VARCHAR(50) NOT NULL,
        prob_threshold NUMERIC(6,4) NOT NULL,
        sl_avoid_threshold NUMERIC(6,4) NOT NULL,
        num_trades INTEGER NOT NULL,
        trades_per_year NUMERIC(10,2),
        win_rate_pre NUMERIC(8,4) NOT NULL,
        win_rate_post NUMERIC(8,4) NOT NULL,
        avg_return_pre NUMERIC(10,6) NOT NULL,
        avg_return_post NUMERIC(10,6) NOT NULL,
        pre_cost_sharpe NUMERIC(8,4) NOT NULL,
        post_cost_sharpe NUMERIC(8,4) NOT NULL,
        max_drawdown NUMERIC(8,4) NOT NULL,
        profit_factor NUMERIC(10,4) NOT NULL,
        gs_trade_pct NUMERIC(8,4),
        gs_profit_pct NUMERIC(8,4),
        kill_gate_pass BOOLEAN NOT NULL,
        bnh_sharpe NUMERIC(8,4),
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
""")

INSERT_SQL = text("""
    INSERT INTO experiment_freq_reduction_results_v2
    (experiment_label, fold_id, horizon, date_range, strategy_variant,
     prob_threshold, sl_avoid_threshold, num_trades, trades_per_year,
     win_rate_pre, win_rate_post, avg_return_pre, avg_return_post,
     pre_cost_sharpe, post_cost_sharpe, max_drawdown, profit_factor,
     gs_trade_pct, gs_profit_pct, kill_gate_pass, bnh_sharpe)
    VALUES
    (:experiment_label, :fold_id, :horizon, :date_range, :strategy_variant,
     :prob_threshold, :sl_avoid_threshold, :num_trades, :trades_per_year,
     :win_rate_pre, :win_rate_post, :avg_return_pre, :avg_return_post,
     :pre_cost_sharpe, :post_cost_sharpe, :max_drawdown, :profit_factor,
     :gs_trade_pct, :gs_profit_pct, :kill_gate_pass, :bnh_sharpe);
""")

PT_MAP = {"baseline": BASELINE_PT, "thr_a": THRESHOLD_A, "thr_b": THRESHOLD_B}
records = []
for fid in FOLD_WINDOWS:
    s, e = FOLD_WINDOWS[fid]
    date_rng = f"{s} to {e}"
    for v_key, v_label in VARIANTS:
        m    = fold_results[fid][v_key]["metrics"]
        trd  = fold_results[fid][v_key]["trades"]
        bm   = fold_results[fid]["bnh"]["metrics"]
        conc = concentration_stats(trd)
        tpy  = trades_per_year(trd, fid)
        kg   = (
            m["post_cost_sharpe"] >= PRE_REGISTERED_KILL_GATE["min_post_cost_sharpe"] and
            m["win_rate_post"]    >= PRE_REGISTERED_KILL_GATE["min_post_cost_win_rate"] and
            m["post_cost_sharpe"] > bm["post_cost_sharpe"]
        )
        records.append({
            "experiment_label":   "freq_reduction_corrected_v2",
            "fold_id":            fid,
            "horizon":            HORIZON,
            "date_range":         date_rng,
            "strategy_variant":   v_key,
            "prob_threshold":     PT_MAP[v_key],
            "sl_avoid_threshold": BASELINE_SL_T,
            "num_trades":         int(m["num_trades"]),
            "trades_per_year":    float(tpy),
            "win_rate_pre":       float(m["win_rate_pre"]),
            "win_rate_post":      float(m["win_rate_post"]),
            "avg_return_pre":     float(m["avg_raw"]),
            "avg_return_post":    float(m["avg_net"]),
            "pre_cost_sharpe":    float(m["pre_cost_sharpe"]),
            "post_cost_sharpe":   float(m["post_cost_sharpe"]),
            "max_drawdown":       float(m["max_drawdown"]),
            "profit_factor":      float(m["profit_factor"]),
            "gs_trade_pct":       float(conc["gs_trade_pct"]),
            "gs_profit_pct":      float(conc["gs_profit_pct"]),
            "kill_gate_pass":     kg,
            "bnh_sharpe":         float(bm["post_cost_sharpe"]),
        })

with engine.begin() as conn:
    conn.execute(CREATE_SQL)
    conn.execute(INSERT_SQL, records)

print()
print(SEP)
print(f"  Saved {len(records)} records to experiment_freq_reduction_results_v2.")
print("  backtest_results table NOT touched.")
print(SEP)
print()
print("  EXPERIMENT COMPLETE. STOP.")
print(SEP)
