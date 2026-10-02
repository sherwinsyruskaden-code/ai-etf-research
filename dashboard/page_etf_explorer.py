"""
Page 2 — ETF EXPLORER
Per-ETF dropdown: price chart (line), volume, returns, RSI, volatility.
Latest model prediction shown only if it exists in the DB — never generated live.
Regime label from market_regimes table.
Read-only.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from dashboard.db import get_engine


def _load_instruments(engine):
    return pd.read_sql(
        text(
            "SELECT id, trading_symbol, name FROM instruments "
            "WHERE is_selected_universe = TRUE ORDER BY trading_symbol;"
        ),
        engine,
    )


def _load_ohlcv(engine, instrument_id: int) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            "SELECT timestamp, open, high, low, close, volume "
            "FROM ohlcv_daily WHERE instrument_id = :iid ORDER BY timestamp ASC;"
        ),
        engine,
        params={"iid": instrument_id},
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df


def _load_features(engine, instrument_id: int) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            "SELECT timestamp, rsi_14, volatility_20d, return_1d, return_5d, return_20d "
            "FROM features WHERE instrument_id = :iid ORDER BY timestamp ASC;"
        ),
        engine,
        params={"iid": instrument_id},
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df


def _load_latest_regime(engine, instrument_id: int) -> dict:
    row = pd.read_sql(
        text(
            "SELECT timestamp, instrument_regime, market_regime "
            "FROM market_regimes WHERE instrument_id = :iid "
            "ORDER BY timestamp DESC LIMIT 1;"
        ),
        engine,
        params={"iid": instrument_id},
    )
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _load_latest_prediction(engine, instrument_id: int) -> dict:
    """
    Checks triple_barrier_labels for the most recent *historical* label.
    The dashboard does NOT run the model live. If no row exists, returns {}.
    """
    row = pd.read_sql(
        text(
            "SELECT timestamp, barrier_hit, realized_return "
            "FROM triple_barrier_labels WHERE instrument_id = :iid "
            "ORDER BY timestamp DESC LIMIT 1;"
        ),
        engine,
        params={"iid": instrument_id},
    )
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def render():
    engine = get_engine()

    st.title("📈 ETF Explorer")
    st.caption(
        "All charts and metrics are **historical/backtested data only**. "
        "This dashboard does not run the model live and does not generate "
        "actionable trading signals."
    )

    # ── Instrument selector ─────────────────────────────────────────────────
    try:
        df_inst = _load_instruments(engine)
    except Exception as e:
        st.error(f"Cannot load instruments: {e}")
        return

    if df_inst.empty:
        st.warning("No instruments in the database. Run the data pipeline first.")
        return

    options = {
        f"{row['trading_symbol']} — {row['name']}": int(row["id"])
        for _, row in df_inst.iterrows()
    }
    selected_label = st.selectbox("Select ETF", list(options.keys()))
    inst_id = options[selected_label]
    symbol = selected_label.split(" — ")[0]

    # ── Load data ───────────────────────────────────────────────────────────
    df_ohlcv = _load_ohlcv(engine, inst_id)
    df_feat = _load_features(engine, inst_id)
    regime = _load_latest_regime(engine, inst_id)
    pred = _load_latest_prediction(engine, inst_id)

    if df_ohlcv.empty:
        st.warning(f"No OHLCV data found for {symbol}.")
        return

    # ── Summary row ─────────────────────────────────────────────────────────
    latest = df_ohlcv.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"₹{latest['close']:,.2f}")
    c2.metric(
        "Latest Volume",
        f"{latest['volume']:,}",
    )
    c3.metric(
        "Current Regime",
        regime.get("instrument_regime", "N/A"),
    )
    c4.metric(
        "Market Regime",
        regime.get("market_regime", "N/A"),
    )

    # ── Historical prediction note ───────────────────────────────────────────
    st.subheader("📋 Latest Historical Model Output")
    if pred:
        ts = pd.to_datetime(pred.get("timestamp", "")).strftime("%Y-%m-%d") if pred.get("timestamp") else "N/A"

        rr = pred.get("realized_return")
        rr_str = f"{float(rr):.2%}" if rr is not None else "Not yet resolved (insufficient forward history)"

        bh = pred.get("barrier_hit")
        bh_str = str(bh) if bh is not None else "Unlabeled (end-of-history)"

        dth = pred.get("days_to_hit")
        dth_str = str(int(dth)) if dth is not None else "N/A"

        st.info(
            f"**⚠️ HISTORICAL BACKTESTED OUTPUT — NOT A LIVE SIGNAL**  \n"
            f"Most recent triple-barrier label in DB  \n"
            f"Date: `{ts}` | Outcome: `{bh_str}` | Days to hit: `{dth_str}` | "
            f"Realized Return: `{rr_str}`"
        )
    else:
        st.info("ℹ️ No historical model output available in the database for this ETF.")

    # ── Price chart ─────────────────────────────────────────────────────────
    st.subheader("💹 Price History (Close)")
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=df_ohlcv["timestamp"],
            y=df_ohlcv["close"],
            mode="lines",
            name="Close",
            line=dict(color="#2196F3", width=1.5),
        )
    )
    fig_price.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Volume chart ────────────────────────────────────────────────────────
    st.subheader("📊 Volume")
    fig_vol = go.Figure(
        go.Bar(
            x=df_ohlcv["timestamp"],
            y=df_ohlcv["volume"],
            name="Volume",
            marker_color="#90CAF9",
        )
    )
    fig_vol.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode="x unified",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    if df_feat.empty:
        st.warning("No feature data found for this ETF.")
        return

    # ── Returns ─────────────────────────────────────────────────────────────
    st.subheader("📉 Daily Returns (1-day)")
    fig_ret = go.Figure(
        go.Scatter(
            x=df_feat["timestamp"],
            y=df_feat["return_1d"],
            mode="lines",
            name="1-Day Return",
            line=dict(color="#4CAF50", width=1),
        )
    )
    fig_ret.add_hline(y=0, line_dash="dash", line_color="grey", line_width=0.8)
    fig_ret.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Return",
        hovermode="x unified",
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── RSI ─────────────────────────────────────────────────────────────────
    st.subheader("📐 RSI (14-day)")
    fig_rsi = go.Figure(
        go.Scatter(
            x=df_feat["timestamp"],
            y=df_feat["rsi_14"],
            mode="lines",
            name="RSI-14",
            line=dict(color="#FF9800", width=1.5),
        )
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig_rsi.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 100]),
        yaxis_title="RSI",
        hovermode="x unified",
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

    # ── Volatility ──────────────────────────────────────────────────────────
    st.subheader("📊 20-Day Realized Volatility")
    fig_vlt = go.Figure(
        go.Scatter(
            x=df_feat["timestamp"],
            y=df_feat["volatility_20d"],
            mode="lines",
            name="Volatility 20d",
            line=dict(color="#9C27B0", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.1)",
        )
    )
    fig_vlt.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Volatility",
        hovermode="x unified",
    )
    st.plotly_chart(fig_vlt, use_container_width=True)
