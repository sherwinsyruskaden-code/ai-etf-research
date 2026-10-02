"""
Page 3 — BACKTEST RESULTS
Shows Phase 5 walk-forward results (locked) from the backtest_results table
and the frequency-reduction experiment results from experiment_freq_reduction_results_v2
(clearly labeled as EXPERIMENT — NOT PART OF LOCKED RESULTS).
Read-only.
"""
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from dashboard.db import get_engine

# Pre-registered kill gate (mirrors engine.py)
KILL_GATE = {
    "min_post_cost_sharpe": 0.50,
    "min_post_cost_win_rate": 0.40,
    "required_fold_pass_ratio": 0.50,
}


def _load_backtest_results(engine) -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT * FROM backtest_results ORDER BY horizon, fold_id, strategy_variant;"),
        engine,
    )


def _load_model_runs(engine) -> pd.DataFrame:
    """Load model validation runs with per-class accuracy metrics."""
    return pd.read_sql(
        text("SELECT * FROM model_runs ORDER BY horizon, id;"),
        engine,
    )


def _load_experiment(engine) -> pd.DataFrame:
    try:
        return pd.read_sql(
            text(
                "SELECT * FROM experiment_freq_reduction_results_v2 "
                "ORDER BY fold_id, strategy_variant;"
            ),
            engine,
        )
    except Exception:
        return pd.DataFrame()


def _sharpe_color(v):
    if pd.isnull(v):
        return ""
    if v >= 0.5:
        return "background-color: #c8e6c9"
    if v >= 0.0:
        return "background-color: #fff9c4"
    return "background-color: #ffcdd2"


def render():
    engine = get_engine()

    st.title("📉 Backtest Results")
    st.caption(
        "All results are **historical backtested outputs** from the locked "
        "Phase 5 walk-forward experiment. No live trading signals."
    )

    # ── Load data ───────────────────────────────────────────────────────────
    try:
        df_bt = _load_backtest_results(engine)
    except Exception as e:
        st.error(f"Cannot load backtest_results table: {e}")
        return

    if df_bt.empty:
        st.warning(
            "No backtest results found in the database. "
            "Run the Phase 5 backtest pipeline first."
        )
        return

    # ── Horizon selector ────────────────────────────────────────────────────
    horizons = sorted(df_bt["horizon"].unique().tolist())
    horizon = st.radio("Select Horizon", horizons, format_func=lambda h: f"{h}-Day", horizontal=True)

    df_h = df_bt[df_bt["horizon"] == horizon].copy()

    # ── Strategy variants ───────────────────────────────────────────────────
    st.subheader(f"📋 Per-Fold Results — {horizon}-Day Horizon")
    st.caption("Showing **post-cost** metrics for all strategy variants.")

    VARIANT_LABELS = {
        "unfiltered": "XGBoost (Unfiltered)",
        "sl_filtered": "XGBoost (SL-Filtered) ✦",
        "buy_and_hold": "Buy & Hold (Baseline)",
        "rule_sma50": "SMA50 (Baseline)",
    }

    for variant, label in VARIANT_LABELS.items():
        df_v = df_h[df_h["strategy_variant"] == variant].copy()
        if df_v.empty:
            continue

        is_primary = variant == "sl_filtered"
        with st.expander(f"{'🔷 ' if is_primary else ''}{label}", expanded=is_primary):
            display_cols = {
                "fold_id": "Fold",
                "date_range": "Date Range",
                "win_rate": "Win Rate",
                "avg_return": "Avg Return",
                "pre_cost_sharpe": "Pre-Cost Sharpe",
                "post_cost_sharpe": "Post-Cost Sharpe ▸",
                "max_drawdown": "Max Drawdown",
                "num_trades": "Trades",
            }
            df_disp = df_v[list(display_cols.keys())].copy()
            df_disp.rename(columns=display_cols, inplace=True)
            df_disp["Win Rate"] = df_disp["Win Rate"].apply(lambda x: f"{float(x):.1%}")
            df_disp["Avg Return"] = df_disp["Avg Return"].apply(lambda x: f"{float(x):.4f}")
            df_disp["Pre-Cost Sharpe"] = df_disp["Pre-Cost Sharpe"].apply(lambda x: f"{float(x):.3f}")
            df_disp["Post-Cost Sharpe ▸"] = df_disp["Post-Cost Sharpe ▸"].apply(lambda x: f"{float(x):.3f}")
            df_disp["Max Drawdown"] = df_disp["Max Drawdown"].apply(lambda x: f"{float(x):.2%}")

            st.dataframe(df_disp.set_index("Fold"), use_container_width=True)

            # Aggregated row
            agg_sharpe = df_v["post_cost_sharpe"].mean()
            agg_wr = df_v["win_rate"].mean()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Post-Cost Sharpe", f"{agg_sharpe:.3f}")
            c2.metric("Mean Win Rate", f"{agg_wr:.1%}")
            c3.metric("Total Trades", int(df_v["num_trades"].sum()))
            c4.metric("Mean Max Drawdown", f"{df_v['max_drawdown'].mean():.2%}")

    # ── Kill-gate verdict bar chart ─────────────────────────────────────────
    st.subheader(f"🚧 Kill-Gate Assessment — {horizon}-Day SL-Filtered")
    df_sl = df_h[df_h["strategy_variant"] == "sl_filtered"].copy()
    df_bh = df_h[df_h["strategy_variant"] == "buy_and_hold"].copy()

    if not df_sl.empty and not df_bh.empty:
        fold_ids = sorted(df_sl["fold_id"].tolist())
        sl_sharpes = df_sl.set_index("fold_id")["post_cost_sharpe"].to_dict()
        bh_sharpes = df_bh.set_index("fold_id")["post_cost_sharpe"].to_dict()

        pass_flags = []
        for fid in fold_ids:
            s = float(sl_sharpes.get(fid, -999))
            w = float(df_sl[df_sl["fold_id"] == fid]["win_rate"].values[0])
            b = float(bh_sharpes.get(fid, 999))
            passed = (
                s >= KILL_GATE["min_post_cost_sharpe"]
                and w >= KILL_GATE["min_post_cost_win_rate"]
                and s > b
            )
            pass_flags.append("PASS ✅" if passed else "FAIL ❌")

        fig_kg = go.Figure()
        colors = ["#4CAF50" if "PASS" in f else "#F44336" for f in pass_flags]
        fig_kg.add_trace(
            go.Bar(
                x=[f"Fold {fid}" for fid in fold_ids],
                y=[float(sl_sharpes.get(fid, 0)) for fid in fold_ids],
                marker_color=colors,
                name="Post-Cost Sharpe (SL-Filtered)",
                text=[f"{pass_flags[i]}" for i in range(len(fold_ids))],
                textposition="outside",
            )
        )
        fig_kg.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="orange",
            annotation_text="Kill-Gate Min Sharpe (0.50)",
        )
        fig_kg.add_hline(y=0, line_color="grey", line_width=0.5)
        fig_kg.update_layout(
            height=340,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="Post-Cost Sharpe Ratio",
            hovermode="x unified",
        )
        st.plotly_chart(fig_kg, use_container_width=True)

        n_pass = sum(1 for f in pass_flags if "PASS" in f)
        if n_pass >= 3:
            st.success(f"Kill gate: {n_pass}/5 folds PASSED")
        else:
            st.error(
                f"Kill gate: **{n_pass}/5** folds PASSED — "
                f"**FAILED** (required ≥ 50% = 3/5)"
            )

    # ── Baseline comparison chart ────────────────────────────────────────────
    st.subheader("📊 Baseline Comparison — Post-Cost Sharpe by Fold")
    compare_variants = ["sl_filtered", "buy_and_hold", "rule_sma50"]
    compare_labels = {
        "sl_filtered": "XGBoost SL-Filtered",
        "buy_and_hold": "Buy & Hold",
        "rule_sma50": "SMA50",
    }
    colors_map = {"sl_filtered": "#2196F3", "buy_and_hold": "#4CAF50", "rule_sma50": "#FF9800"}

    fig_cmp = go.Figure()
    for v in compare_variants:
        df_v = df_h[df_h["strategy_variant"] == v].sort_values("fold_id")
        if df_v.empty:
            continue
        fig_cmp.add_trace(
            go.Scatter(
                x=[f"Fold {fid}" for fid in df_v["fold_id"]],
                y=df_v["post_cost_sharpe"].astype(float),
                mode="lines+markers",
                name=compare_labels[v],
                line=dict(color=colors_map[v]),
            )
        )
    fig_cmp.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_cmp.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis_title="Post-Cost Sharpe Ratio",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Model validation accuracy (calibration) ──────────────────────────────
    st.subheader("🎯 Model Validation Accuracy (Phase 4 Calibration)")
    try:
        df_mr = _load_model_runs(engine)
        if df_mr.empty:
            st.info("No model_runs records in the database.")
        else:
            df_mr_disp = df_mr[["id", "model_name", "horizon", "train_date_range",
                                  "val_date_range", "accuracy"]].copy()
            df_mr_disp["accuracy"] = df_mr_disp["accuracy"].apply(lambda x: f"{float(x):.2%}")
            st.dataframe(df_mr_disp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load model_runs: {e}")

    # ────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🧪 EXPERIMENT — NOT PART OF LOCKED RESULTS")
    st.warning(
        "**Frequency-Reduction Experiment (Corrected v2)**  \n"
        "This experiment tested whether raising the XGBoost entry confidence "
        "threshold (0.45 / 0.50) would reduce gold/silver concentration and "
        "improve performance on the 14-day strategy.  \n\n"
        "**⚠️ Calibration Leakage Bug:** v1 of this experiment used the Phase 4 "
        "pooled model (trained on ALL data including validation windows), "
        "introducing data leakage. Results from v1 are invalid.  \n\n"
        "**v2 correction:** XGBoost is retrained on each fold's purged training "
        "set (identical to Phase 5). Only the entry threshold changes.  \n\n"
        "**FINAL VERDICT: FAILED** — Most recent period (2022–2026) loses money "
        "under every tested threshold. Strategy remains gold-concentrated.",
        icon="⚗️",
    )

    df_exp = _load_experiment(engine)
    if df_exp.empty:
        st.info(
            "No experiment_freq_reduction_results_v2 table found in the database.  \n"
            "Run `scripts/experiment_freq_reduction_v2.py` to populate it."
        )
    else:
        st.caption(
            "Columns: fold_id, strategy_variant, prob_threshold, post_cost_sharpe, "
            "win_rate_post, max_drawdown, kill_gate_pass"
        )
        key_cols = [
            "fold_id", "strategy_variant", "prob_threshold",
            "post_cost_sharpe", "win_rate_post", "max_drawdown",
            "gs_trade_pct", "kill_gate_pass", "bnh_sharpe",
        ]
        available = [c for c in key_cols if c in df_exp.columns]
        df_exp_disp = df_exp[available].copy()

        if "post_cost_sharpe" in df_exp_disp.columns:
            df_exp_disp["post_cost_sharpe"] = df_exp_disp["post_cost_sharpe"].apply(
                lambda x: f"{float(x):.3f}"
            )
        if "win_rate_post" in df_exp_disp.columns:
            df_exp_disp["win_rate_post"] = df_exp_disp["win_rate_post"].apply(
                lambda x: f"{float(x):.1%}"
            )
        if "max_drawdown" in df_exp_disp.columns:
            df_exp_disp["max_drawdown"] = df_exp_disp["max_drawdown"].apply(
                lambda x: f"{float(x):.2%}"
            )
        if "gs_trade_pct" in df_exp_disp.columns:
            df_exp_disp["gs_trade_pct"] = df_exp_disp["gs_trade_pct"].apply(
                lambda x: f"{float(x):.1%}"
            )

        st.dataframe(df_exp_disp, use_container_width=True)

        # Experiment fold chart
        if "post_cost_sharpe" in df_exp.columns and "strategy_variant" in df_exp.columns:
            fig_exp = px.bar(
                df_exp,
                x="fold_id",
                y="post_cost_sharpe",
                color="strategy_variant",
                barmode="group",
                title="Experiment: Post-Cost Sharpe by Fold & Threshold",
                labels={"post_cost_sharpe": "Post-Cost Sharpe", "fold_id": "Fold"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_exp.add_hline(y=0.5, line_dash="dash", line_color="orange",
                              annotation_text="Kill-Gate Min (0.50)")
            fig_exp.add_hline(y=0, line_color="grey", line_width=0.5)
            fig_exp.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_exp, use_container_width=True)
