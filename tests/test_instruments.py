import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.data.instruments import InstrumentManager


@pytest.fixture
def mock_groww_client():
    client = MagicMock()
    return client


@pytest.fixture
def sample_instruments_df():
    return pd.DataFrame([
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "RELIANCE", "groww_symbol": "NSE-RELIANCE", "instrument_type": "EQ", "name": "Reliance Industries Ltd"},
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "NIFTYBEES", "groww_symbol": "NSE-NIFTYBEES", "instrument_type": "ETF", "name": "Nippon India ETF Nifty BeES"},
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "BANKBEES", "groww_symbol": "NSE-BANKBEES", "instrument_type": "ETF", "name": "Nippon India ETF Bank BeES"},
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "LIQUIDBEES", "groww_symbol": "NSE-LIQUIDBEES", "instrument_type": "ETF", "name": "Nippon India ETF Liquid BeES"},
        {"exchange": "NSE", "segment": "CASH", "trading_symbol": "LIQUIDETF", "groww_symbol": "NSE-LIQUIDETF", "instrument_type": "ETF", "name": "DSP Liquid ETF"},
        {"exchange": "BSE", "segment": "CASH", "trading_symbol": "NIFTYBEES", "groww_symbol": "BSE-NIFTYBEES", "instrument_type": "ETF", "name": "Nippon India ETF Nifty BeES"},
        {"exchange": "NSE", "segment": "FNO", "trading_symbol": "NIFTY26MARFUT", "groww_symbol": "NSE-NIFTY26MARFUT", "instrument_type": "FUT", "name": "Nifty March Future"},
    ])


def test_discover_nse_cash_instruments(mock_groww_client, sample_instruments_df):
    mgr = InstrumentManager(mock_groww_client)
    df_nse = mgr.discover_nse_cash_instruments(sample_instruments_df)
    assert len(df_nse) == 5
    assert all(df_nse["exchange"] == "NSE")
    assert all(df_nse["segment"] == "CASH")


def test_filter_etf_candidates_excludes_liquid_funds(mock_groww_client, sample_instruments_df):
    mgr = InstrumentManager(mock_groww_client)
    df_nse = mgr.discover_nse_cash_instruments(sample_instruments_df)
    df_etfs = mgr.filter_etf_candidates(df_nse)
    
    symbols = df_etfs["trading_symbol"].tolist()
    assert "NIFTYBEES" in symbols
    assert "BANKBEES" in symbols
    
    # Verify liquid/cash funds are excluded
    assert "LIQUIDBEES" not in symbols
    assert "LIQUIDETF" not in symbols
    assert "RELIANCE" not in symbols
