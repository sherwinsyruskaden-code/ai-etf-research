import logging
import pandas as pd
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates daily OHLCV market data according to strict quantitative rules:
    - No duplicate timestamps per instrument
    - Positive OHLC values (> 0)
    - High >= Open and High >= Close
    - Low <= Open and Low <= Close
    - Volume >= 0
    - Timestamps are valid and localized to IST timezone
    - Data sorted chronologically
    - Identifies and reports missing trading dates without fabricating data
    """

    def validate_ohlcv(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate OHLCV dataframe for a single instrument.

        Returns:
            Tuple of (is_valid: bool, errors: List[str], summary: Dict[str, Any])
        """
        errors = []
        summary = {
            "symbol": symbol,
            "total_rows": len(df),
            "valid_rows": 0,
            "duplicate_count": 0,
            "missing_date_gaps": [],
            "timezone_valid": True,
            "sorted_chronologically": True,
        }

        if df.empty:
            errors.append(f"[{symbol}] DataFrame is empty. No candles to validate.")
            return False, errors, summary

        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            errors.append(f"[{symbol}] Missing required columns: {missing_cols}")
            return False, errors, summary

        # 1. Check for null values
        null_count = df[["open", "high", "low", "close", "volume"]].isnull().sum().sum()
        if null_count > 0:
            errors.append(f"[{symbol}] Contains {null_count} null values in OHLCV columns.")

        # 2. Check for duplicate timestamps
        duplicates = df.duplicated(subset=["timestamp"], keep=False)
        duplicate_count = int(duplicates.sum())
        summary["duplicate_count"] = duplicate_count
        if duplicate_count > 0:
            dup_dates = df.loc[duplicates, "timestamp"].tolist()
            errors.append(f"[{symbol}] Duplicate timestamps detected ({duplicate_count} instances): {dup_dates[:3]}")

        # 3. Check chronological ordering
        if not df["timestamp"].is_monotonic_increasing:
            summary["sorted_chronologically"] = False
            errors.append(f"[{symbol}] Timestamps are not sorted in strict ascending chronological order.")

        # 4. Check timezone compliance (IST)
        ts_sample = df["timestamp"].iloc[0]
        if hasattr(ts_sample, "tz") and ts_sample.tz is not None:
            tz_str = str(ts_sample.tz)
            if tz_str not in ["Asia/Kolkata", "IST", "+05:30"]:
                summary["timezone_valid"] = False
                errors.append(f"[{symbol}] Timestamp timezone '{tz_str}' is not IST / Asia/Kolkata.")

        # 5. OHLC mathematical boundaries validation
        invalid_positive = df[(df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)]
        if len(invalid_positive) > 0:
            errors.append(f"[{symbol}] {len(invalid_positive)} rows have non-positive OHLC values.")

        invalid_high = df[(df["high"] < df["open"]) | (df["high"] < df["close"])]
        if len(invalid_high) > 0:
            errors.append(f"[{symbol}] {len(invalid_high)} rows fail High >= Open and High >= Close constraint.")

        invalid_low = df[(df["low"] > df["open"]) | (df["low"] > df["close"])]
        if len(invalid_low) > 0:
            errors.append(f"[{symbol}] {len(invalid_low)} rows fail Low <= Open and Low <= Close constraint.")

        invalid_volume = df[df["volume"] < 0]
        if len(invalid_volume) > 0:
            errors.append(f"[{symbol}] {len(invalid_volume)} rows have negative volume values.")

        # 6. Detect and report missing dates / calendar gaps (> 4 consecutive days, accounting for weekends)
        if len(df) > 1:
            dates = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            diffs = (dates.diff().dt.total_seconds() / (24 * 3600)).fillna(1)
            gaps = df[diffs > 4]
            if len(gaps) > 0:
                for idx, row in gaps.iterrows():
                    prev_date = df.loc[idx - 1, "timestamp"] if idx > 0 else "N/A"
                    curr_date = row["timestamp"]
                    gap_msg = f"Gap detected between {prev_date} and {curr_date}"
                    summary["missing_date_gaps"].append(gap_msg)
                logger.info(f"[{symbol}] Reported {len(gaps)} potential market holiday/date gaps.")

        is_valid = len(errors) == 0
        summary["valid_rows"] = len(df) if is_valid else 0

        if not is_valid:
            for err in errors:
                logger.error(err)
        else:
            logger.info(f"[{symbol}] Data validation PASSED cleanly for {len(df)} candles.")

        return is_valid, errors, summary
