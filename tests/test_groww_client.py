import pytest
import pandas as pd
from datetime import datetime
from src.data.groww_client import GrowwClient


@pytest.fixture
def client():
    # Instantiates client in fallback mode (without real API key)
    return GrowwClient(api_key="your_groww_api_key_here")


def test_parse_candle_response_valid(client):
    raw_response = {
        "status": "SUCCESS",
        "candles": [
            [1704067200, 250.0, 255.0, 248.0, 252.0, 100000],
            [1704153600, 252.0, 258.0, 251.0, 256.0, 120000]
        ]
    }
    df = client._parse_candle_response(raw_response, "NSE-NIFTYBEES")
    assert not df.empty
    assert len(df) == 2
    assert "timestamp" in df.columns
    assert df.iloc[0]["open"] == 250.0
    assert df.iloc[0]["volume"] == 100000


def test_malformed_api_response_handling(client):
    # Test None / empty dict / invalid json
    assert client._parse_candle_response({}, "NSE-TEST").empty
    assert client._parse_candle_response(None, "NSE-TEST").empty
    assert client._parse_candle_response({"status": "SUCCESS", "candles": []}, "NSE-TEST").empty
    assert client._parse_candle_response({"error": "Invalid token"}, "NSE-TEST").empty


def test_fallback_candles_generation():
    # Explicitly test mock fallback generator when allow_mock_fallback=True
    client = GrowwClient(api_key="your_groww_api_key_here", allow_mock_fallback=True)
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 1, 15)
    df = client._generate_fallback_candles("NSE-NIFTYBEES", start_date, end_date)
    assert not df.empty
    assert "open" in df.columns
    assert "volume" in df.columns
    assert df["open"].min() > 0


def test_live_run_fails_loudly_on_error():
    # Verify that when allow_mock_fallback=False (default), API failures raise RuntimeError loudly
    client = GrowwClient(api_key="your_groww_api_key_here", allow_mock_fallback=False)
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 1, 15)

    from unittest.mock import patch
    with patch("urllib.request.urlopen", side_effect=Exception("Network failure")):
        with pytest.raises(RuntimeError) as exc_info:
            client.fetch_historical_daily_candles("INVALID_SYMBOL", start_date, end_date)
        assert "Failure" in str(exc_info.value) or "failed" in str(exc_info.value)

