import pytest
import pandas as pd
from datetime import datetime
from src.data.validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


@pytest.fixture
def sample_valid_ohlcv():
    dates = pd.date_range("2026-01-01", periods=5, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame({
        "timestamp": dates,
        "open": [250.0, 252.0, 251.0, 253.0, 255.0],
        "high": [255.0, 256.0, 254.0, 258.0, 260.0],
        "low": [249.0, 250.0, 248.0, 251.0, 254.0],
        "close": [252.0, 251.0, 253.0, 255.0, 258.0],
        "volume": [100000, 150000, 120000, 180000, 200000]
    })


def test_valid_ohlcv_data(validator, sample_valid_ohlcv):
    is_valid, errors, summary = validator.validate_ohlcv(sample_valid_ohlcv, symbol="TESTBEES")
    assert is_valid is True
    assert len(errors) == 0
    assert summary["valid_rows"] == 5


def test_invalid_positive_ohlcv(validator, sample_valid_ohlcv):
    df_invalid = sample_valid_ohlcv.copy()
    df_invalid.loc[2, "close"] = -10.0
    is_valid, errors, summary = validator.validate_ohlcv(df_invalid, symbol="TESTBEES")
    assert is_valid is False
    assert any("non-positive" in err for err in errors)


def test_high_low_boundary_violations(validator, sample_valid_ohlcv):
    df_invalid = sample_valid_ohlcv.copy()
    df_invalid.loc[1, "high"] = 240.0  # Less than open/close
    is_valid, errors, summary = validator.validate_ohlcv(df_invalid, symbol="TESTBEES")
    assert is_valid is False
    assert any("High >= Open" in err for err in errors)


def test_duplicate_timestamps(validator, sample_valid_ohlcv):
    df_dup = pd.concat([sample_valid_ohlcv, sample_valid_ohlcv.iloc[[0]]], ignore_index=True)
    is_valid, errors, summary = validator.validate_ohlcv(df_dup, symbol="TESTBEES")
    assert is_valid is False
    assert summary["duplicate_count"] > 0
    assert any("Duplicate timestamps" in err for err in errors)


def test_negative_volume(validator, sample_valid_ohlcv):
    df_invalid = sample_valid_ohlcv.copy()
    df_invalid.loc[0, "volume"] = -500
    is_valid, errors, summary = validator.validate_ohlcv(df_invalid, symbol="TESTBEES")
    assert is_valid is False
    assert any("negative volume" in err for err in errors)
