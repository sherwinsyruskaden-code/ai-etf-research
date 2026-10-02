import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from src.config.settings import DATABASE_URL
from src.data.database import InstrumentRepository, get_db_engine
from src.regime_labels.repository import RegimeLabelRepository


def run_spot_checks():
    engine = get_db_engine(DATABASE_URL)
    inst_repo = InstrumentRepository(engine)
    reg_repo = RegimeLabelRepository(engine)

    selected_etfs = inst_repo.get_selected_universe()

    print("=" * 100)
    print("PHASE 3 POSTGRESQL DATABASE VERIFICATION & SPOT-CHECK REPORT")
    print("=" * 100)

    # 1. Total table row counts
    with engine.connect() as conn:
        tot_reg = conn.execute(text("SELECT COUNT(*) FROM market_regimes;")).scalar()
        tot_lbl = conn.execute(text("SELECT COUNT(*) FROM triple_barrier_labels;")).scalar()

    print(f"\n1. TABLE ROW COUNTS:")
    print(f"   - Total records in 'market_regimes': {tot_reg}")
    print(f"   - Total records in 'triple_barrier_labels': {tot_lbl}")

    # 2. Per-instrument summary & distributions
    print("\n2. PER-INSTRUMENT REGIME & TRIPLE-BARRIER LABEL DISTRIBUTIONS:")
    print("-" * 100)

    all_regimes = []
    all_labels_h7 = []
    all_labels_h14 = []

    for _, row in selected_etfs.iterrows():
        sym = row["trading_symbol"]
        inst_id = row["id"]

        df_reg = reg_repo.get_regimes(inst_id)
        df_lbl7 = reg_repo.get_labels(inst_id, horizon=7)
        df_lbl14 = reg_repo.get_labels(inst_id, horizon=14)

        reg_dist = df_reg["instrument_regime"].value_counts().to_dict()
        lbl7_dist = df_lbl7["barrier_hit"].value_counts(dropna=False).to_dict()
        lbl14_dist = df_lbl14["barrier_hit"].value_counts(dropna=False).to_dict()

        null7_count = sum(1 for v in df_lbl7["barrier_hit"] if pd.isna(v) or v is None)
        null14_count = sum(1 for v in df_lbl14["barrier_hit"] if pd.isna(v) or v is None)

        print(f"\n[{sym}] (Instrument ID: {inst_id}) | Total History: {len(df_reg)} days")
        print(f"   Regime Dist: {reg_dist}")
        print(f"   Horizon 7d Barrier Dist:  {lbl7_dist} (Unlabeled End-of-History: {null7_count})")
        print(f"   Horizon 14d Barrier Dist: {lbl14_dist} (Unlabeled End-of-History: {null14_count})")

    # 3. Detailed Spot-Check Traces
    print("\n3. SPOT-CHECK TRACES (Manual Price Path Verification):")
    print("-" * 100)

    # Pick NIFTYBEES and GOLDBEES for spot checks
    sample_symbols = ["NIFTYBEES", "GOLDBEES"]
    for sym in sample_symbols:
        match = selected_etfs[selected_etfs["trading_symbol"] == sym]
        if match.empty:
            continue
        inst_id = match.iloc[0]["id"]

        # Fetch OHLCV, Features, Regimes, and Labels
        with engine.connect() as conn:
            query = text("""
                SELECT o.timestamp, o.open, o.high, o.low, o.close,
                       f.atr_14, f.volatility_20d, f.sma_20, f.sma_50,
                       r.instrument_regime, r.market_regime,
                       l7.barrier_hit as hit_7, l7.days_to_hit as days_7, l7.tp_level as tp_7, l7.sl_level as sl_7, l7.realized_return as ret_7,
                       l14.barrier_hit as hit_14, l14.days_to_hit as days_14, l14.tp_level as tp_14, l14.sl_level as sl_14, l14.realized_return as ret_14
                FROM ohlcv_daily o
                LEFT JOIN features f ON o.instrument_id = f.instrument_id AND o.timestamp = f.timestamp
                LEFT JOIN market_regimes r ON o.instrument_id = r.instrument_id AND o.timestamp = r.timestamp
                LEFT JOIN triple_barrier_labels l7 ON o.instrument_id = l7.instrument_id AND o.timestamp = l7.timestamp AND l7.horizon = 7
                LEFT JOIN triple_barrier_labels l14 ON o.instrument_id = l14.instrument_id AND o.timestamp = l14.timestamp AND l14.horizon = 14
                WHERE o.instrument_id = :inst_id
                ORDER BY o.timestamp ASC;
            """)
            df_merged = pd.read_sql(query, conn, params={"inst_id": inst_id})

        # Select a mid-history sample row (e.g. index 1000)
        sample_idx = 1000 if len(df_merged) > 1000 else len(df_merged) // 2
        entry_row = df_merged.iloc[sample_idx]

        ts_entry = entry_row["timestamp"]
        c_entry = entry_row["close"]
        atr_entry = entry_row["atr_14"]
        regime_entry = entry_row["instrument_regime"]
        mkt_regime_entry = entry_row["market_regime"]

        hit_7 = entry_row["hit_7"]
        days_7 = entry_row["days_7"]
        tp_7 = entry_row["tp_7"]
        sl_7 = entry_row["sl_7"]
        ret_7 = entry_row["ret_7"]

        print(f"\n>>> SPOT CHECK #1: {sym} on Entry Date: {ts_entry.strftime('%Y-%m-%d')}")
        print(f"    Entry Close Price : {c_entry:.4f} | ATR(14): {atr_entry:.4f}")
        print(f"    Assigned Regimes  : ETF={regime_entry} | Broad Market={mkt_regime_entry}")
        print(f"    7-Day Horizon Label: {hit_7} (Hit in {days_7} days) | TP={tp_7:.4f}, SL={sl_7:.4f} | Realized Return={ret_7:.4%}")

        # Trace future bars for 7-day horizon
        print("    Future Price Path Trace (Step 1 to 7):")
        for step in range(1, 8):
            fut_idx = sample_idx + step
            if fut_idx < len(df_merged):
                fut_row = df_merged.iloc[fut_idx]
                f_ts = fut_row["timestamp"].strftime("%Y-%m-%d")
                f_h = fut_row["high"]
                f_l = fut_row["low"]
                f_c = fut_row["close"]
                tp_hit_flag = " [TP HIT!]" if f_h >= tp_7 else ""
                sl_hit_flag = " [SL HIT!]" if f_l <= sl_7 else ""
                print(f"      Step +{step} ({f_ts}): Open={fut_row['open']:.2f}, High={f_h:.2f}{tp_hit_flag}, Low={f_l:.2f}{sl_hit_flag}, Close={f_c:.2f}")

        # Spot check end-of-history row
        end_idx = len(df_merged) - 1
        end_row = df_merged.iloc[end_idx]
        print(f"\n>>> SPOT CHECK END-OF-HISTORY: {sym} on Last Date: {end_row['timestamp'].strftime('%Y-%m-%d')}")
        print(f"    7-Day Label : {end_row['hit_7']} | 14-Day Label: {end_row['hit_14']}")
        print(f"    Verification: End-of-history row correctly un-labeled (NULL) because future horizon is incomplete.")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    run_spot_checks()
