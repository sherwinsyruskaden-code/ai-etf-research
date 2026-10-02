"""
Page 4 -- RESEARCH / METHODOLOGY
Visual pipeline diagram and key methodological notes.
Read-only -- no writes, no live model execution.
"""
import streamlit as st
import pandas as pd


def render():
    st.title("Research and Methodology")

    # Prominent research conclusion
    st.error(
        "**Research conclusion: The current strategy did not demonstrate a "
        "robust edge over the baseline after costs.**",
        icon="🚫",
    )

    # Pipeline diagram
    st.subheader("Research Pipeline")
    st.markdown(
        """
        The full pipeline ran from raw market data through to a pre-registered
        kill-gate assessment. Each stage below fed the next; no shortcuts were
        taken between data collection and final verdict.
        """
    )

    pipeline_stages = [
        ("📡 Groww API", "NSE instrument master + daily OHLCV candles (~20 years)", "#1565C0"),
        ("🗄️ PostgreSQL", "Idempotent storage: instruments, ohlcv_daily, features, regimes, labels", "#0277BD"),
        ("⚙️ Feature Engineering", "31 technical / contextual features (momentum, trend, volatility, volume, market context)", "#00695C"),
        ("🌐 Causal Regime Detection", "4 regimes (TRENDING_UP, TRENDING_DOWN, HIGH_VOLATILITY, SIDEWAYS) -- strictly causal, no lookahead", "#2E7D32"),
        ("🏷️ Triple-Barrier Labeling", "TP / SL / TIME labels at 7-day and 14-day horizons, ATR-scaled barriers", "#558B2F"),
        ("🤖 XGBoost + Baselines", "Trained on purged walk-forward folds; baselines: Buy-and-Hold, SMA50", "#E65100"),
        ("🔄 Walk-Forward Testing (5 folds)", "Purging + embargoing per Lopez de Prado; no random CV splits", "#BF360C"),
        ("💰 Costs / Slippage / Spread", "Groww delivery brokerage, STT, stamp duty, SEBI charges, fixed slippage, liquidity-scaled spread", "#4A148C"),
        ("📐 Calibration", "Per-class accuracy on held-out folds; leakage audit after Phase 4 pooled-model bug was found and corrected", "#880E4F"),
        ("🚧 Kill Gate", "Pre-registered: Sharpe >= 0.50, win rate >= 40%, beat Buy-and-Hold, >= 50% folds pass", "#B71C1C"),
    ]

    for i, (title, detail, color) in enumerate(pipeline_stages):
        if i > 0:
            st.markdown(
                "<div style='text-align:center;font-size:22px;color:#888;'>&#8595;</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
            <div style='background-color:{color}18;border-left:4px solid {color};
                        border-radius:6px;padding:10px 14px;margin-bottom:2px;'>
                <strong style='color:{color};'>{title}</strong><br/>
                <span style='font-size:0.88rem;color:#444;'>{detail}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # Kill-gate criteria
    st.subheader("Pre-Registered Kill-Gate (Phase 5)")
    st.markdown(
        "All criteria below were locked **before** Phase 5 ran. "
        "A strategy needed to pass **all three** in **>= 50% of folds**."
    )
    st.table(
        pd.DataFrame({
            "Criterion": [
                "Post-cost Sharpe ratio",
                "Post-cost win rate",
                "Beat Buy-and-Hold Sharpe",
                "Fold pass ratio",
            ],
            "Threshold": [">= 0.50", ">= 40%", "Yes (per fold)", ">= 50% (>= 3/5)"],
            "7-Day result": ["FAILED (0/5)", "FAILED", "FAILED", "FAILED (0/5)"],
            "14-Day SL-Filtered": ["FAILED (1/5)", "FAILED", "FAILED", "FAILED (1/5 = 20%)"],
        })
    )

    # Sharpe caveat
    st.subheader("Known Limitation: Sharpe Annualization")
    st.info(
        "Trade-level Sharpe ratios are annualized using sqrt(252), approximating each trade as "
        "one day. Trades-per-year typically exceed 252, so this understates (is conservative on) "
        "the annualization factor. The failing folds miss 0.50 by -1 to -4 Sharpe units -- "
        "no correction flips the FAIL verdict.",
        icon="ℹ️",
    )

    # Design principles
    st.subheader("Design Principles")
    principles = [
        ("Zero lookahead",
         "All regime labels, features, and walk-forward splits are computed with data "
         "available *up to and including* the signal date only."),
        ("Purging and Embargoing",
         "Walk-forward folds follow Lopez de Prado: training samples whose forward evaluation "
         "window overlaps the test window are removed (purging), and a post-test gap is "
         "enforced (embargoing)."),
        ("Pre-registration",
         "Kill-gate thresholds were fixed before Phase 5 ran; they were not tuned post-hoc."),
        ("Cost realism",
         "Every trade deducts Groww delivery brokerage, STT (0.1%), stamp duty, SEBI charges, "
         "a fixed 0.05% slippage, and a liquidity-scaled bid-ask spread (0.05%--0.30%)."),
        ("Read-only dashboard",
         "This dashboard queries existing results only. It does not retrain models, run the "
         "strategy live, or generate actionable trading signals."),
    ]
    for title, body in principles:
        with st.expander(f"**{title}**"):
            st.markdown(body)

    # Data lineage
    st.subheader("Database Tables Used")
    st.markdown(
        """
        | Table | Contents |
        |---|---|
        | `instruments` | ETF universe (10 selected) + liquidity metrics |
        | `ohlcv_daily` | Raw daily OHLCV candles |
        | `features` | 31 engineered features per instrument-date |
        | `market_regimes` | Causal regime label per instrument-date |
        | `triple_barrier_labels` | TP / SL / TIME outcome labels (H=7, H=14) |
        | `model_runs` | Phase 4 validation accuracy records |
        | `backtest_results` | Phase 5 walk-forward fold metrics (locked) |
        | `experiment_freq_reduction_results_v2` | Freq-reduction experiment (corrected v2) |
        """
    )
