"""
Page 1 — OVERVIEW
Displays project description, ETF universe, model status, kill-gate verdict,
and high-level DB statistics. Read-only.
"""
import pandas as pd
import streamlit as st
from sqlalchemy import text

from dashboard.db import get_engine


def render():
    engine = get_engine()

    st.title("🔬 AI ETF Research — Overview")

    # ── Project description ─────────────────────────────────────────────────
    st.markdown(
        """
        This research project investigated whether a machine-learning–driven,
        long-only trading strategy on Indian equity ETFs (NSE / Groww) could
        generate a statistically robust edge over passive baselines after
        realistic transaction costs (Groww delivery brokerage, STT, stamp duty,
        SEBI charges, fixed slippage, and liquidity-scaled bid-ask spreads).
        The pipeline ingested ~20 years of daily OHLCV data via the Groww API,
        engineered 31 technical/contextual features, detected causal market
        regimes, generated triple-barrier labels (7-day and 14-day horizons),
        trained XGBoost classifiers inside purged walk-forward folds, and
        evaluated the resulting strategy against pre-registered kill-gate
        criteria.
        """
    )

    # ── Kill-gate verdict ───────────────────────────────────────────────────
    st.error(
        "**KILL-GATE VERDICT: FAILED**  \n"
        "Only 1 of 5 walk-forward folds passed on the 14-day strategy.  \n"
        "0 of 5 folds passed on the 7-day strategy.  \n\n"
        "**Research conclusion: The current strategy did not demonstrate a "
        "robust edge over the baseline after costs.**",
        icon="🚫",
    )

    # ── ETF Universe ────────────────────────────────────────────────────────
    st.subheader("📋 ETF Universe (10 Selected ETFs)")
    try:
        df_inst = pd.read_sql(
            text(
                "SELECT trading_symbol, name, avg_daily_volume "
                "FROM instruments WHERE is_selected_universe = TRUE "
                "ORDER BY avg_daily_volume DESC;"
            ),
            engine,
        )
        if df_inst.empty:
            st.warning("No instruments found in the database yet.")
        else:
            df_inst.columns = ["Symbol", "Name", "Avg Daily Traded Value (₹)"]
            df_inst["Avg Daily Traded Value (₹)"] = (
                df_inst["Avg Daily Traded Value (₹)"]
                .apply(lambda x: f"₹{x:,.0f}" if pd.notnull(x) else "—")
            )
            st.dataframe(df_inst, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load ETF universe: {e}")

    # ── Model status ────────────────────────────────────────────────────────
    st.subheader("🤖 Model Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**7-Day Horizon (XGBoost)**")
        try:
            cnt_7 = pd.read_sql(
                text("SELECT COUNT(*) AS n FROM model_runs WHERE horizon = 7;"),
                engine,
            )["n"].iloc[0]
            if cnt_7 > 0:
                st.success(f"✅ {cnt_7} validation run(s) recorded")
            else:
                st.warning("No model runs recorded for H=7")
        except Exception as e:
            st.error(str(e))

    with col2:
        st.markdown("**14-Day Horizon (XGBoost)**")
        try:
            cnt_14 = pd.read_sql(
                text("SELECT COUNT(*) AS n FROM model_runs WHERE horizon = 14;"),
                engine,
            )["n"].iloc[0]
            if cnt_14 > 0:
                st.success(f"✅ {cnt_14} validation run(s) recorded")
            else:
                st.warning("No model runs recorded for H=14")
        except Exception as e:
            st.error(str(e))

    # ── Key metrics ─────────────────────────────────────────────────────────
    st.subheader("📊 Database Statistics")
    try:
        total_candles = pd.read_sql(
            text("SELECT COUNT(*) AS n FROM ohlcv_daily;"), engine
        )["n"].iloc[0]
        total_features = pd.read_sql(
            text("SELECT COUNT(*) AS n FROM features;"), engine
        )["n"].iloc[0]
        total_regimes = pd.read_sql(
            text("SELECT COUNT(*) AS n FROM market_regimes;"), engine
        )["n"].iloc[0]
        total_labels = pd.read_sql(
            text("SELECT COUNT(*) AS n FROM triple_barrier_labels;"), engine
        )["n"].iloc[0]
        bt_folds = pd.read_sql(
            text(
                "SELECT COUNT(DISTINCT fold_id) AS n FROM backtest_results;"
            ),
            engine,
        )["n"].iloc[0]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("OHLCV Candles", f"{total_candles:,}")
        m2.metric("Feature Rows", f"{total_features:,}")
        m3.metric("Regime Labels", f"{total_regimes:,}")
        m4.metric("Triple-Barrier Labels", f"{total_labels:,}")
        m5.metric("Backtest Folds", f"{bt_folds}")
    except Exception as e:
        st.error(f"Could not load DB statistics: {e}")

    # ── Kill-gate criteria ──────────────────────────────────────────────────
    st.subheader("🚧 Pre-Registered Kill-Gate Criteria")
    st.table(
        pd.DataFrame(
            {
                "Criterion": [
                    "Min post-cost Sharpe ratio",
                    "Min post-cost win rate",
                    "Must beat Buy & Hold Sharpe",
                    "Required fold pass ratio",
                ],
                "Threshold": ["≥ 0.50", "≥ 40%", "Yes", "≥ 50% of folds"],
                "Result (14-day SL-Filtered)": [
                    "FAILED (4 folds below)",
                    "FAILED (most folds)",
                    "FAILED (most folds)",
                    "FAILED (1/5 = 20%)",
                ],
            }
        )
    )
