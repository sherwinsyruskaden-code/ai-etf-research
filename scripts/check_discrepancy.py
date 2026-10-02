"""Check model discrepancy between Phase 5 engine and experiment."""
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg://etf_user:etf_password@localhost:5433/etf_research')

print('CRITICAL FINDING:')
print('Phase 5 engine.py (lines 102-110) RETRAINS XGBoost on each fold training set.')
print('The experiment uses the SAVED Phase 4 pooled model (trained on ALL data).')
print()
print('Phase 5 sl_filtered (14d, all folds):')
with engine.connect() as conn:
    df = pd.read_sql(
        text("SELECT fold_id, num_trades, win_rate, pre_cost_sharpe, post_cost_sharpe FROM backtest_results WHERE horizon=14 AND strategy_variant='sl_filtered' ORDER BY fold_id"),
        conn
    )
print(df.to_string())

print()
print('Experiment baseline (pooled model):')
with engine.connect() as conn:
    df2 = pd.read_sql(
        text("SELECT fold_id, num_trades, win_rate_post, pre_cost_sharpe, post_cost_sharpe FROM experiment_freq_reduction_results WHERE strategy_variant='baseline' ORDER BY fold_id"),
        conn
    )
print(df2.to_string())
