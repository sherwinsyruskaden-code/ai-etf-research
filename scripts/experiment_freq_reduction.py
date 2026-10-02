"""
EXPERIMENT — NOT PART OF LOCKED PHASE 5 RESULTS
=================================================
Frequency Reduction via Confidence Threshold — Walk-Forward Rerun

Pre-registered thresholds (chosen from calibration, before this script ran):
  Threshold A: prob_tp >= 0.45, prob_sl < 0.30
  Threshold B: prob_tp >= 0.50, prob_sl < 0.30

No model retraining. Same folds, same data, same costs, same slippage.
Results stored in `experiment_freq_reduction_results` (never in backtest_results).

Kill gate evaluated: Sharpe >= 0.50, win rate >= 40%, beats B&H, on >= 3/5 folds.
"""

import os
import sys
import warnings
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
BASELINE_PT   = 0.35          # Original strategy threshold (for comparison)
BASELINE_SL_T = 0.30          # SL-avoid threshold — UNCHANGED across all variants
THRESHOLD_A   = 0.45          # PRE-REGISTERED moderate threshold
THRESHOLD_B   = 0.50          # PRE-REGISTERED strict threshold

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
    "min_post_cost_sharpe": 0.50,
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
# 1. Load full dataset
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
    "instrument_id", "timestamp", "target", "barrier_hit", "days_to_hit",
    "tp_level", "sl_level", "realized_return", "entry_open_next", "close_curr"
}
CONTEXT_COLS = [
    "nifty50_return_1d", "banknifty_return_1d", "india_vix_level",
    "india_vix_change_1d", "sector_return_1d", "commodity_return_1d"
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

# -----------------------------------------------------------------------
# 2. Load existing Phase 4 model — NO RETRAINING
# -----------------------------------------------------------------------
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

print("Model predictions applied to all folds.")

# -----------------------------------------------------------------------
# 3. Cost model (identical to Phase 5)
# -----------------------------------------------------------------------
def compute_cost(adtv):
    if adtv > 100_000_000:
        spread = 0.0005
    elif adtv > 10_000_000:
        spread = 0.0010
    else:
        spread = 0.0020
    slip  = 0.0005
    stt   = 0.0020
    stamp = 0.00015
    exch  = 0.0000345 * 2
    sebi  = 0.000001 * 2
    gst   = (0.0005 + exch + sebi) * 0.18 * 2
    return slip * 2 + spread + stt + stamp + exch + sebi + gst


def simulate_trades(df_val_raw, prob_threshold, sl_threshold=0.30):
    """
    Simulate trades at given prob_tp threshold.
    Entry: next bar's open (T+1 open). Same as Phase 5 engine.
    Returns list of trade dicts with raw_return, net_return, instrument symbol.
    """
    trades = []
    df_sorted = df_val_raw.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
    for i in range(len(df_sorted) - 1):
        cur = df_sorted.iloc[i]
        nxt = df_sorted.iloc[i + 1]
        if cur["instrument_id"] != nxt["instrument_id"]:
            continue
        p_tp = float(cur["prob_tp"])
        p_sl = float(cur["prob_sl"])
        if p_tp >= prob_threshold and p_sl < sl_threshold:
            entry = float(nxt["entry_open_next"])
            inst_id = int(cur["instrument_id"])
            adtv = float(inst_liq.get(inst_id, 50_000_000))
            hit   = cur["barrier_hit"]
            c_entry = float(cur["close_curr"])
            if hit == "TP" and pd.notna(cur["tp_level"]):
                raw_exit = float(cur["tp_level"])
            elif hit == "SL" and pd.notna(cur["sl_level"]):
                raw_exit = float(cur["sl_level"])
            else:
                rel = float(cur["realized_return"]) if pd.notna(cur["realized_return"]) else 0.0
                raw_exit = c_entry * (1.0 + rel)
            raw_ret = (raw_exit - entry) / entry if entry > 0 else 0.0
            total_cost = compute_cost(adtv)
            net_ret = raw_ret - total_cost
            trades.append({
                "timestamp":     cur["timestamp"],
                "instrument_id": inst_id,
                "instrument":    inst_map.get(inst_id, str(inst_id)),
                "barrier_hit":   hit,
                "days_to_hit":   float(cur["days_to_hit"]) if pd.notna(cur["days_to_hit"]) else float(HORIZON),
                "prob_tp":       p_tp,
                "raw_return":    raw_ret,
                "net_return":    net_ret,
                "cost_drag":     total_cost,
            })
    return trades


def calc_metrics(trades):
    """Calculate per-fold metrics dict."""
    if not trades:
        return {
            "num_trades": 0, "win_rate_pre": 0.0, "win_rate_post": 0.0,
            "avg_raw": 0.0, "avg_net": 0.0,
            "pre_cost_sharpe": 0.0, "post_cost_sharpe": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0,
        }
    df_t = pd.DataFrame(trades)
    raw  = df_t["raw_return"].values
    net  = df_t["net_return"].values
    n    = len(df_t)
    wr_pre  = float(np.mean(raw > 0))
    wr_post = float(np.mean(net > 0))
    avg_raw = float(np.mean(raw))
    avg_net = float(np.mean(net))
    std_raw = np.std(raw)
    std_net = np.std(net)
    sh_pre  = float((avg_raw / std_raw) * np.sqrt(252)) if std_raw > 0 else 0.0
    sh_post = float((avg_net / std_net) * np.sqrt(252)) if std_net > 0 else 0.0
    cum_eq  = np.cumprod(1.0 + net)
    peak    = np.maximum.accumulate(cum_eq)
    dd      = (cum_eq - peak) / peak
    max_dd  = float(np.min(dd)) if len(dd) > 0 else 0.0
    gains   = net[net > 0].sum()
    losses  = np.abs(net[net < 0].sum())
    pf      = float(gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
    return {
        "num_trades": n, "win_rate_pre": wr_pre, "win_rate_post": wr_post,
        "avg_raw": avg_raw, "avg_net": avg_net,
        "pre_cost_sharpe": sh_pre, "post_cost_sharpe": sh_post,
        "max_drawdown": max_dd, "profit_factor": pf,
    }


def simulate_buy_and_hold(df_val_raw):
    """Buy-and-hold benchmark — same as Phase 5 engine (daily series, corrected)."""
    daily_series_list = []
    total_raw_list = []
    total_net_list = []
    for inst_id, group in df_val_raw.groupby("instrument_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2:
            continue
        closes = g["close_curr"].values.astype(float)
        daily_raw = np.diff(closes) / closes[:-1]
        entry_open = float(g.iloc[0]["entry_open_next"])
        exit_close = float(g.iloc[-1]["close_curr"])
        adtv = float(inst_liq.get(int(inst_id), 50_000_000))
        total_cost = compute_cost(adtv)
        n_days = len(daily_raw)
        daily_cost = total_cost / n_days if n_days > 0 else 0.0
        daily_series_list.append(daily_raw - daily_cost)
        raw_ret = (exit_close - entry_open) / entry_open if entry_open > 0 else 0.0
        net_ret = raw_ret - total_cost
        total_raw_list.append(raw_ret)
        total_net_list.append(net_ret)
    if not daily_series_list:
        return {"post_cost_sharpe": 0.0, "num_trades": 0, "win_rate_post": 0.0,
                "avg_net": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0,
                "pre_cost_sharpe": 0.0, "win_rate_pre": 0.0, "avg_raw": 0.0}
    max_len = max(len(s) for s in daily_series_list)
    padded  = np.array([np.pad(s, (0, max_len - len(s)), constant_values=0.0) for s in daily_series_list])
    port    = padded.mean(axis=0)
    mu  = float(np.mean(port))
    sig = float(np.std(port))
    sh  = float((mu / sig) * np.sqrt(252)) if sig > 0 else 0.0
    cum_eq = np.cumprod(1.0 + port)
    peak   = np.maximum.accumulate(cum_eq)
    dd     = (cum_eq - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
    net_arr = np.array(total_net_list)
    raw_arr = np.array(total_raw_list)
    gains   = net_arr[net_arr > 0].sum()
    losses  = np.abs(net_arr[net_arr < 0].sum())
    pf      = float(gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
    return {
        "num_trades": len(net_arr), "win_rate_pre": float(np.mean(raw_arr > 0)),
        "win_rate_post": float(np.mean(net_arr > 0)), "avg_raw": float(np.mean(raw_arr)),
        "avg_net": float(np.mean(net_arr)), "pre_cost_sharpe": sh,
        "post_cost_sharpe": sh, "max_drawdown": max_dd, "profit_factor": pf,
    }


def concentration_stats(trades):
    """Return gold/silver concentration stats."""
    if not trades:
        return {"gs_trade_pct": 0.0, "gs_profit_pct": 0.0, "gs_trades": 0, "total_trades": 0}
    df_t = pd.DataFrame(trades)
    df_t["is_gs"] = df_t["instrument"].isin(GOLD_SILVER_ETFS)
    gs_n   = df_t["is_gs"].sum()
    total  = len(df_t)
    total_profit = df_t["net_return"].sum()
    gs_profit    = df_t[df_t["is_gs"]]["net_return"].sum()
    pct_trades  = gs_n / total * 100 if total > 0 else 0.0
    pct_profit  = gs_profit / total_profit * 100 if total_profit != 0 else 0.0
    return {
        "gs_trade_pct":  pct_trades,
        "gs_profit_pct": pct_profit,
        "gs_trades":     int(gs_n),
        "total_trades":  int(total),
    }


def calibration_within_filtered(trades):
    """Calibration of prob_tp vs actual TP rate within the filtered trade set."""
    if not trades:
        return {}
    df_t = pd.DataFrame(trades)
    df_t["actual_tp"] = (df_t["barrier_hit"] == "TP").astype(int)
    bins   = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ["0.35-0.40","0.40-0.45","0.45-0.50","0.50-0.55","0.55-0.60","0.60-0.65","0.65-0.70","0.70+"]
    df_t["bucket"] = pd.cut(df_t["prob_tp"], bins=bins, labels=labels, right=False)
    grp = df_t.groupby("bucket", observed=True).agg(
        count=("actual_tp","count"),
        actual_tp_rate=("actual_tp","mean")
    )
    return grp


def trades_per_year(trades, fold_id):
    """Annualised trade rate for a fold."""
    if not trades:
        return 0.0
    s, e = FOLD_WINDOWS[fold_id]
    period_yrs = (pd.Timestamp(e) - pd.Timestamp(s)).days / 365.25
    return len(trades) / period_yrs if period_yrs > 0 else 0.0


# -----------------------------------------------------------------------
# 4. Run experiment across all folds for ALL three variants
# -----------------------------------------------------------------------
print("\nRunning experiment across all folds...")

results = {}  # fold_id -> {baseline, thr_a, thr_b, bnh}

for fid, df_f in folds_data.items():
    print(f"  Fold {fid} ...", end=" ")
    trades_base = simulate_trades(df_f, BASELINE_PT, BASELINE_SL_T)
    trades_a    = simulate_trades(df_f, THRESHOLD_A, BASELINE_SL_T)
    trades_b    = simulate_trades(df_f, THRESHOLD_B, BASELINE_SL_T)
    bnh_metrics = simulate_buy_and_hold(df_f)
    results[fid] = {
        "baseline":  {"trades": trades_base, "metrics": calc_metrics(trades_base)},
        "thr_a":     {"trades": trades_a,    "metrics": calc_metrics(trades_a)},
        "thr_b":     {"trades": trades_b,    "metrics": calc_metrics(trades_b)},
        "bnh":       {"metrics": bnh_metrics},
    }
    print(f"baseline={len(trades_base)} | thr_a={len(trades_a)} | thr_b={len(trades_b)}")

# -----------------------------------------------------------------------
# 5. SIDE-BY-SIDE COMPARISON REPORT
# -----------------------------------------------------------------------
print()
print(SEP)
print("  EXPERIMENT -- NOT PART OF LOCKED PHASE 5 RESULTS")
print("  Frequency Reduction via Confidence Threshold")
print("  Pre-registered thresholds: A=0.45, B=0.50 | SL-avoid=0.30 (unchanged)")
print(SEP)

VARIANTS = [
    ("baseline",  f"ORIGINAL (thr=0.35)"),
    ("thr_a",     f"THRESHOLD A (thr=0.45)"),
    ("thr_b",     f"THRESHOLD B (thr=0.50)"),
]

for v_key, v_label in VARIANTS:
    print()
    print(f">>> {v_label}")
    print(SEP2)
    hdr = (f"{'Fold':<5} {'Period':<28} {'Trd/yr':>7} {'WR_pre':>7} {'WR_post':>8} "
           f"{'Sh_pre':>8} {'Sh_post':>8} {'MaxDD':>8} {'PF':>6} {'GS_tr%':>8} {'GS_pft%':>8} {'KillGate':>10}")
    print(hdr)
    print("-" * len(hdr))

    fold_verdicts = []
    agg_trades_yr = []
    agg_wr_pre    = []
    agg_wr_post   = []
    agg_sh_pre    = []
    agg_sh_post   = []
    agg_dd        = []

    for fid in FOLD_WINDOWS:
        m    = results[fid][v_key]["metrics"]
        bm   = results[fid]["bnh"]["metrics"]
        trd  = results[fid][v_key]["trades"]
        s, e = FOLD_WINDOWS[fid]

        tpy  = trades_per_year(trd, fid)
        conc = concentration_stats(trd)

        # Kill gate: Sharpe >= 0.50, WR >= 40%, beats B&H
        kg_pass = (
            m["post_cost_sharpe"] >= PRE_REGISTERED_KILL_GATE["min_post_cost_sharpe"] and
            m["win_rate_post"]    >= PRE_REGISTERED_KILL_GATE["min_post_cost_win_rate"] and
            m["post_cost_sharpe"] > bm["post_cost_sharpe"]
        )
        fold_verdicts.append(kg_pass)
        agg_trades_yr.append(tpy)
        agg_wr_pre.append(m["win_rate_pre"])
        agg_wr_post.append(m["win_rate_post"])
        agg_sh_pre.append(m["pre_cost_sharpe"])
        agg_sh_post.append(m["post_cost_sharpe"])
        agg_dd.append(m["max_drawdown"])

        kg_str = "PASS" if kg_pass else "FAIL"
        gs_tr  = conc["gs_trade_pct"]
        gs_pft = conc["gs_profit_pct"]
        print(
            f"  {fid:<3} {s} to {e}  {tpy:>7.0f} "
            f"{m['win_rate_pre']:>7.1%} {m['win_rate_post']:>7.1%} "
            f"{m['pre_cost_sharpe']:>8.2f} {m['post_cost_sharpe']:>8.2f} "
            f"{m['max_drawdown']:>8.1%} {m['profit_factor']:>6.2f} "
            f"{gs_tr:>7.1f}% {gs_pft:>8.1f}%  {kg_str:>8}"
        )

    passes = sum(fold_verdicts)
    n_folds = len(FOLD_WINDOWS)
    pass_ratio = passes / n_folds
    is_pass = pass_ratio >= PRE_REGISTERED_KILL_GATE["required_fold_pass_ratio"]

    print(SEP2)
    print(f"  AGGREGATED | Trd/yr={np.mean(agg_trades_yr):.0f} | WR_pre={np.mean(agg_wr_pre):.1%} | "
          f"WR_post={np.mean(agg_wr_post):.1%} | Sh_pre={np.mean(agg_sh_pre):.2f} | "
          f"Sh_post={np.mean(agg_sh_post):.2f} (std={np.std(agg_sh_post):.2f}) | MaxDD={np.mean(agg_dd):.1%}")
    print(f"  KILL GATE: {passes}/{n_folds} folds passed ({pass_ratio:.0%}) -- "
          f"Required >=60% -- VERDICT: {'PASSED' if is_pass else 'FAILED'}")

# -----------------------------------------------------------------------
# 6. CONCENTRATION REPORT — full detail
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
        trd  = results[fid][v_key]["trades"]
        conc = concentration_stats(trd)
        print(
            f"  {fid:<5} {v_label:<22} {conc['gs_trades']:>10} {conc['total_trades']:>10} "
            f"{conc['gs_trade_pct']:>8.1f}% {conc['gs_profit_pct']:>8.1f}%"
        )
    print()

# -----------------------------------------------------------------------
# 7. CALIBRATION WITHIN FILTERED TRADES (per variant, all folds)
# -----------------------------------------------------------------------
print()
print(SEP)
print("  CALIBRATION WITHIN FILTERED TRADE SET — ALL FOLDS COMBINED")
print(SEP)

for v_key, v_label in VARIANTS:
    all_variant_trades = []
    for fid in FOLD_WINDOWS:
        all_variant_trades.extend(results[fid][v_key]["trades"])
    if not all_variant_trades:
        continue
    calib = calibration_within_filtered(all_variant_trades)
    base_tp = np.mean([(t["barrier_hit"] == "TP") for t in all_variant_trades])
    print(f"\n  {v_label} | overall_TP={base_tp:.1%} | n_trades={len(all_variant_trades)}")
    print(f"  {'bucket':<15} {'N':>6} {'actual_TP':>10}")
    print("  " + "-" * 35)
    for bucket, row in calib.iterrows():
        if row["count"] > 0:
            print(f"  {str(bucket):<15} {int(row['count']):>6} {row['actual_tp_rate']:>9.1%}")

# -----------------------------------------------------------------------
# 8. FINAL VERDICTS SUMMARY
# -----------------------------------------------------------------------
print()
print(SEP)
print("  FINAL VERDICT SUMMARY — EXPERIMENT vs ORIGINAL KILL GATE")
print("  (Kill gate unchanged: Sharpe>=0.50, WR>=40%, beats B&H, >=3/5 folds)")
print(SEP)

for v_key, v_label in VARIANTS:
    fold_verdicts = []
    for fid in FOLD_WINDOWS:
        m  = results[fid][v_key]["metrics"]
        bm = results[fid]["bnh"]["metrics"]
        kg = (
            m["post_cost_sharpe"] >= PRE_REGISTERED_KILL_GATE["min_post_cost_sharpe"] and
            m["win_rate_post"]    >= PRE_REGISTERED_KILL_GATE["min_post_cost_win_rate"] and
            m["post_cost_sharpe"] > bm["post_cost_sharpe"]
        )
        fold_verdicts.append(kg)
    passes    = sum(fold_verdicts)
    n_folds   = len(FOLD_WINDOWS)
    is_pass   = (passes / n_folds) >= PRE_REGISTERED_KILL_GATE["required_fold_pass_ratio"]
    fold_str  = " | ".join([f"F{fid}={'PASS' if p else 'FAIL'}" for fid, p in zip(FOLD_WINDOWS, fold_verdicts)])
    print(f"\n  {v_label}")
    print(f"    Per-fold: {fold_str}")
    print(f"    Overall: {passes}/{n_folds} folds PASSED -- VERDICT: {'PASSED' if is_pass else 'FAILED'}")

    # Concentration (aggregate)
    all_trd = []
    for fid in FOLD_WINDOWS:
        all_trd.extend(results[fid][v_key]["trades"])
    conc = concentration_stats(all_trd)
    gs_label = "GOLD-CONCENTRATED" if conc["gs_trade_pct"] > 50 or conc["gs_profit_pct"] > 60 else "BROAD-BASED (not gold-dominated)"
    print(f"    Concentration: {conc['gs_trade_pct']:.1f}% of trades, {conc['gs_profit_pct']:.1f}% of profit from GS ETFs -> {gs_label}")

# -----------------------------------------------------------------------
# 9. Save results to experiment_freq_reduction_results (NEW table)
# -----------------------------------------------------------------------
CREATE_TABLE_SQL = text("""
    CREATE TABLE IF NOT EXISTS experiment_freq_reduction_results (
        id SERIAL PRIMARY KEY,
        experiment_label VARCHAR(50) NOT NULL DEFAULT 'freq_reduction_threshold',
        fold_id INTEGER NOT NULL,
        horizon INTEGER NOT NULL,
        date_range VARCHAR(100) NOT NULL,
        strategy_variant VARCHAR(50) NOT NULL,
        prob_threshold NUMERIC(6, 4) NOT NULL,
        sl_avoid_threshold NUMERIC(6, 4) NOT NULL,
        num_trades INTEGER NOT NULL,
        trades_per_year NUMERIC(10, 2),
        win_rate_pre NUMERIC(8, 4) NOT NULL,
        win_rate_post NUMERIC(8, 4) NOT NULL,
        avg_return_pre NUMERIC(10, 6) NOT NULL,
        avg_return_post NUMERIC(10, 6) NOT NULL,
        pre_cost_sharpe NUMERIC(8, 4) NOT NULL,
        post_cost_sharpe NUMERIC(8, 4) NOT NULL,
        max_drawdown NUMERIC(8, 4) NOT NULL,
        profit_factor NUMERIC(10, 4) NOT NULL,
        gs_trade_pct NUMERIC(8, 4),
        gs_profit_pct NUMERIC(8, 4),
        kill_gate_pass BOOLEAN NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
""")

INSERT_SQL = text("""
    INSERT INTO experiment_freq_reduction_results
    (experiment_label, fold_id, horizon, date_range, strategy_variant,
     prob_threshold, sl_avoid_threshold, num_trades, trades_per_year,
     win_rate_pre, win_rate_post, avg_return_pre, avg_return_post,
     pre_cost_sharpe, post_cost_sharpe, max_drawdown, profit_factor,
     gs_trade_pct, gs_profit_pct, kill_gate_pass)
    VALUES
    (:experiment_label, :fold_id, :horizon, :date_range, :strategy_variant,
     :prob_threshold, :sl_avoid_threshold, :num_trades, :trades_per_year,
     :win_rate_pre, :win_rate_post, :avg_return_pre, :avg_return_post,
     :pre_cost_sharpe, :post_cost_sharpe, :max_drawdown, :profit_factor,
     :gs_trade_pct, :gs_profit_pct, :kill_gate_pass);
""")

VARIANT_META = {
    "baseline": ("freq_reduction_threshold", BASELINE_PT),
    "thr_a":    ("freq_reduction_threshold", THRESHOLD_A),
    "thr_b":    ("freq_reduction_threshold", THRESHOLD_B),
}

records = []
for fid in FOLD_WINDOWS:
    s, e = FOLD_WINDOWS[fid]
    date_rng = f"{s} to {e}"
    for v_key, v_label in VARIANTS:
        m    = results[fid][v_key]["metrics"]
        trd  = results[fid][v_key]["trades"]
        bm   = results[fid]["bnh"]["metrics"]
        conc = concentration_stats(trd)
        tpy  = trades_per_year(trd, fid)
        _, pt = VARIANT_META[v_key]
        kg = (
            m["post_cost_sharpe"] >= PRE_REGISTERED_KILL_GATE["min_post_cost_sharpe"] and
            m["win_rate_post"]    >= PRE_REGISTERED_KILL_GATE["min_post_cost_win_rate"] and
            m["post_cost_sharpe"] > bm["post_cost_sharpe"]
        )
        records.append({
            "experiment_label":   "freq_reduction_threshold",
            "fold_id":            fid,
            "horizon":            HORIZON,
            "date_range":         date_rng,
            "strategy_variant":   v_key,
            "prob_threshold":     pt,
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
        })

with engine.begin() as conn:
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(INSERT_SQL, records)

print()
print(SEP)
print(f"  Saved {len(records)} records to experiment_freq_reduction_results table.")
print("  (backtest_results table was NOT touched)")
print(SEP)
print()
print("  EXPERIMENT COMPLETE. STOP.")
print(SEP)
