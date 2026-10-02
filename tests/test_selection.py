import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.data.instruments import InstrumentManager


def test_liquidity_based_universe_selection():
    mock_client = MagicMock()

    # Candidate ETFs
    candidates = pd.DataFrame([
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "LOW_LIQ_ETF", "groww_symbol": "NSE-LOW_LIQ_ETF", "instrument_type": "ETF"},
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "HIGH_LIQ_ETF", "groww_symbol": "NSE-HIGH_LIQ_ETF", "instrument_type": "ETF"},
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "MED_LIQ_ETF", "groww_symbol": "NSE-MED_LIQ_ETF", "instrument_type": "ETF"},
    ])

    dates = pd.date_range("2026-01-01", periods=5, freq="D", tz="Asia/Kolkata")

    # Mock candle responses for candidates
    candles_high = pd.DataFrame({"timestamp": dates, "open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000000})
    candles_med = pd.DataFrame({"timestamp": dates, "open": 50, "high": 52, "low": 48, "close": 50, "volume": 100000})
    candles_low = pd.DataFrame({"timestamp": dates, "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1000})

    def mock_fetch(groww_symbol, **kwargs):
        if "HIGH" in groww_symbol:
            return candles_high
        elif "MED" in groww_symbol:
            return candles_med
        else:
            return candles_low

    mock_client.fetch_historical_daily_candles.side_effect = mock_fetch

    mgr = InstrumentManager(mock_client)
    selected_df, ranked_df, metrics_dict = mgr.select_top_etf_universe(candidates, historical_days=30, top_n=2)

    assert len(selected_df) == 2
    assert ranked_df.iloc[0]["trading_symbol"] == "HIGH_LIQ_ETF"
    assert ranked_df.iloc[1]["trading_symbol"] == "MED_LIQ_ETF"
    assert ranked_df.iloc[2]["trading_symbol"] == "LOW_LIQ_ETF"
    assert not bool(ranked_df.iloc[2]["selected"])
