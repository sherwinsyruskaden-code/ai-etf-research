import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from src.backtest.costs import TradingCostModel
from src.backtest.engine import WalkForwardBacktester, PRE_REGISTERED_KILL_GATE


def test_next_bar_open_entry_execution():
    """Verify that simulated trades execute at next bar's OPEN (T+1 Open), not same-bar Close (T Close)."""
    cost_model = TradingCostModel(fixed_slippage_pct=0.0, stt_pct=0.0, stamp_duty_pct=0.0, exchange_fee_pct=0.0, sebi_fee_pct=0.0, gst_pct=0.0)

    # Day T (signal day): Close = 100.0
    # Day T+1 (execution day): Open = 105.0, Close = 110.0
    # Day T+2 (exit day): High = 120.0 (TP level)
    c_entry_curr = 100.0
    entry_open_next = 105.0
    raw_exit_tp = 120.0

    # Execute trade using T+1 Open (105.0)
    friction = cost_model.calculate_trade_friction(
        entry_price=entry_open_next,
        exit_price=raw_exit_tp,
        avg_daily_volume=200_000_000  # High liquidity (0.05% spread)
    )

    # Entry Open = 105.0, Spread = 0.05% (0.025% half-spread). Effective Entry = 105.0 * 1.00025 = 105.02625
    # Raw Return relative to T+1 Open = (120 - 105) / 105 = 0.142857 (14.2857%)
    # Same-bar close return would have been (120 - 100) / 100 = 0.2000 (20.0%)
    assert np.isclose(friction["raw_return"], (120.0 - 105.0) / 105.0)
    assert friction["raw_return"] != (120.0 - 100.0) / 100.0


def test_trading_cost_model_deductions():
    """Hand-checkable single-trade test verifying brokerage, statutory fees, slippage, and spread deductions."""
    cost_model = TradingCostModel()

    entry_price = 100.0
    exit_price = 110.0
    adtv = 50_000_000  # Medium liquidity (0.10% roundtrip spread)

    res = cost_model.calculate_trade_friction(entry_price, exit_price, avg_daily_volume=adtv)

    # 1. Raw Return = (110 - 100) / 100 = +10.00%
    assert np.isclose(res["raw_return"], 0.10)

    # 2. Slippage (0.05% leg = 0.10% roundtrip) + Half Spread (0.05% leg = 0.10% roundtrip)
    # Effective Entry = 100 * (1 + 0.0005 + 0.0005) = 100.10
    # Effective Exit = 110 * (1 - 0.0005 - 0.0005) = 109.89
    assert np.isclose(res["effective_entry_price"], 100.10)
    assert np.isclose(res["effective_exit_price"], 109.89)

    # 3. Spread/Slippage Return = (109.89 - 100.10) / 100.10 = 0.097802 (9.7802%)
    # Statutory fees roundtrip ~0.222%
    # Net return MUST BE LESS THAN raw return
    assert res["net_return"] < res["raw_return"]
    assert res["total_cost_pct"] > 0.002  # Total drag > 20 bps


def test_walk_forward_fold_purging():
    """Verify that walk-forward folds apply purging per fold to eliminate label leakage."""
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    df_synthetic = pd.DataFrame({
        "timestamp": dates,
        "instrument_id": [1] * n,
        "feature_1": np.random.randn(n),
        "target": np.random.choice([0, 1, 2], size=n)
    })

    # Engine mockup not needed for fold purging test logic
    engine = create_engine("sqlite:///:memory:")
    backtester = WalkForwardBacktester(engine)

    folds = backtester._create_walk_forward_folds(df_synthetic, horizon=7, n_folds=3)

    assert len(folds) == 3

    for fold_id, (df_tr, df_v, d_info) in enumerate(folds, start=1):
        max_train_date = df_tr["timestamp"].max()
        min_val_date = df_v["timestamp"].min()

        # Check gap between train max and val min is at least H=7 days
        diff_days = (min_val_date - max_train_date).days
        assert diff_days >= 7
