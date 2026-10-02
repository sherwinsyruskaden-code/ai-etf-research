import pandas as pd
import numpy as np
from sqlalchemy import text
from src.config.settings import DATABASE_URL
from src.data.database import get_db_engine
from src.backtest.engine import PRE_REGISTERED_KILL_GATE


def report_phase5():
    engine = get_db_engine(DATABASE_URL)

    with engine.connect() as conn:
        df_res = pd.read_sql("SELECT * FROM backtest_results ORDER BY horizon ASC, fold_id ASC, id ASC;", conn)

    print("=" * 100)
    print("PHASE 5 WALK-FORWARD BACKTESTING REPORT (REALISTIC COSTS, SLIPPAGE, & SPREAD)")
    print("=" * 100)
    print(f"PRE-REGISTERED KILL GATE CRITERIA (written BEFORE backtest execution):")
    print(f"  1. Minimum Post-Cost Sharpe Ratio      : >= {PRE_REGISTERED_KILL_GATE['min_post_cost_sharpe']}")
    print(f"  2. Minimum Post-Cost Win Rate         : >= {PRE_REGISTERED_KILL_GATE['min_post_cost_win_rate']:.0%}")
    print(f"  3. Must Outperform Buy & Hold Sharpe   : {PRE_REGISTERED_KILL_GATE['must_beat_buy_and_hold_sharpe']}")
    print(f"  4. Required Fold Pass Ratio            : >= {PRE_REGISTERED_KILL_GATE['required_fold_pass_ratio']:.0%} of folds")
    print("=" * 100)

    for h in [7, 14]:
        print(f"\n" + "=" * 100)
        print(f"FORECAST HORIZON: {h} DAYS")
        print("=" * 100)

        sub_df = df_res[df_res["horizon"] == h]

        for variant in ["unfiltered", "sl_filtered", "buy_and_hold", "rule_sma50"]:
            var_df = sub_df[sub_df["strategy_variant"] == variant]
            if var_df.empty:
                continue

            print(f"\n>>> STRATEGY VARIANT: {variant.upper()}")
            print("-" * 100)
            print(f"{'Fold':5s} | {'Date Range':25s} | {'Trades':6s} | {'Pre Win%':9s} | {'Post Win%':9s} | {'Pre Sharpe':10s} | {'Post Sharpe':11s} | {'Max DD':9s} | {'PF':6s}")
            print("-" * 100)

            for _, row in var_df.iterrows():
                f_id = row["fold_id"]
                d_rng = row["date_range"]
                trades = row["num_trades"]
                pre_win = row.get("pre_cost_win_rate", row["win_rate"])
                if pd.isna(pre_win):
                    pre_win = row["win_rate"]
                post_win = row["win_rate"]
                pre_s = row.get("pre_cost_sharpe", row["sharpe_ratio"])
                if pd.isna(pre_s):
                    pre_s = row["sharpe_ratio"]
                post_s = row["sharpe_ratio"]
                dd = row["max_drawdown"]
                pf = row["profit_factor"]

                print(f"Fold {f_id:1d} | {d_rng:25s} | {trades:6d} | {pre_win:8.2%} | {post_win:8.2%} | {pre_s:10.2f} | {post_s:11.2f} | {dd:8.2%} | {pf:6.2f}")

            # Aggregated metrics for this variant
            avg_trades = var_df["num_trades"].mean()
            avg_pre_win = var_df["pre_cost_win_rate"].mean() if "pre_cost_win_rate" in var_df.columns else var_df["win_rate"].mean()
            avg_post_win = var_df["win_rate"].mean()
            avg_pre_s = var_df["pre_cost_sharpe"].mean() if "pre_cost_sharpe" in var_df.columns else var_df["sharpe_ratio"].mean()
            avg_post_s = var_df["sharpe_ratio"].mean()
            std_post_s = var_df["sharpe_ratio"].std()

            print("-" * 100)
            print(f"AGGREGATED {variant.upper()}: Avg Trades={avg_trades:.1f} | Pre Win={avg_pre_win:.2%} | Post Win={avg_post_win:.2%} | Pre Sharpe={avg_pre_s:.2f} | Post Sharpe={avg_post_s:.2f} (Std={std_post_s:.2f})")

        # Evaluate Kill Gate Verdict for this horizon
        sl_df = sub_df[sub_df["strategy_variant"] == "sl_filtered"]
        bnh_df = sub_df[sub_df["strategy_variant"] == "buy_and_hold"]

        pass_count = 0
        n_folds = len(sl_df)

        for f_id in range(1, n_folds + 1):
            sl_row = sl_df[sl_df["fold_id"] == f_id].iloc[0]
            bnh_row = bnh_df[bnh_df["fold_id"] == f_id].iloc[0]

            s_sharpe = sl_row["sharpe_ratio"]
            s_win = sl_row["win_rate"]
            b_sharpe = bnh_row["sharpe_ratio"]

            if (s_sharpe >= PRE_REGISTERED_KILL_GATE["min_post_cost_sharpe"] and
                s_win >= PRE_REGISTERED_KILL_GATE["min_post_cost_win_rate"] and
                s_sharpe > b_sharpe):
                pass_count += 1

        pass_ratio = pass_count / n_folds if n_folds > 0 else 0.0
        is_pass = pass_ratio >= PRE_REGISTERED_KILL_GATE["required_fold_pass_ratio"]

        print("\n" + "=" * 100)
        print(f"PRE-REGISTERED KILL GATE VERDICT (HORIZON {h} DAYS):")
        print(f"  Folds Passing Criteria : {pass_count} / {n_folds} ({pass_ratio:.0%})")
        print(f"  Required Pass Ratio   : >= {PRE_REGISTERED_KILL_GATE['required_fold_pass_ratio']:.0%}")
        print(f"  VERDICT: [{'PASSED' if is_pass else 'FAILED'}]")
        print("=" * 100)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    report_phase5()
