import logging
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PurgedTimeSeriesSplit:
    """
    Purged & Embargoed Time Series Train/Validation Splitter.
    (Marcos Lopez de Prado - Advances in Financial Machine Learning)

    Prevents future data leakage in triple-barrier labels across train and validation splits:
    1. Validation set is selected from chronological tail (e.g. most recent ~20% of data).
    2. PURGING: Removes any training sample whose forward-looking label window [T+1, T+horizon]
       overlaps with the validation set start date.
    3. EMBARGO: Applies a gap of at least `horizon` trading days after validation period if needed.
    """

    def __init__(self, val_pct: float = 0.20, embargo_pct: float = 0.0):
        self.val_pct = val_pct
        self.embargo_pct = embargo_pct

    def split(
        self,
        df: pd.DataFrame,
        horizon: int,
        timestamp_col: str = "timestamp"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Perform purged and embargoed train/val split on DataFrame `df`.

        Args:
            df (pd.DataFrame): Input DataFrame containing `timestamp_col`.
            horizon (int): Triple-barrier forecast horizon in trading days (7 or 14).
            timestamp_col (str): Column name for timestamp.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
                - df_train: Purged training DataFrame.
                - df_val: Validation DataFrame.
                - split_info: Dictionary containing dates, counts, and purged row metrics.
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        # Ensure sorted chronologically by timestamp
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        unique_dates = df_sorted[timestamp_col].drop_duplicates().sort_values().tolist()
        n_dates = len(unique_dates)

        if n_dates < 50:
            raise ValueError(f"Insufficient unique trading dates ({n_dates}) for purged split.")

        val_cutoff_idx = int(np.floor(n_dates * (1.0 - self.val_pct)))
        val_start_date = unique_dates[val_cutoff_idx]

        # Validation set: timestamps >= val_start_date
        val_mask = df_sorted[timestamp_col] >= val_start_date
        df_val = df_sorted[val_mask].copy()

        # Unpurged training candidate set: timestamps < val_start_date
        train_cand_mask = df_sorted[timestamp_col] < val_start_date
        df_train_cand = df_sorted[train_cand_mask].copy()

        # PURGING: Identify for each trading date the timestamp after H trading days
        # For an instrument, if a row at date T has date(T + H days) >= val_start_date,
        # its forward-looking label window overlaps the validation set, so it MUST BE PURGED.
        purge_cutoff_idx = max(0, val_cutoff_idx - horizon)
        purge_cutoff_date = unique_dates[purge_cutoff_idx]

        # Retain only training rows with timestamp < purge_cutoff_date
        train_purged_mask = df_sorted[timestamp_col] < purge_cutoff_date
        df_train = df_sorted[train_purged_mask].copy()

        purged_count = len(df_train_cand) - len(df_train)

        train_min_date = df_train[timestamp_col].min().strftime("%Y-%m-%d")
        train_max_date = df_train[timestamp_col].max().strftime("%Y-%m-%d")
        val_min_date = df_val[timestamp_col].min().strftime("%Y-%m-%d")
        val_max_date = df_val[timestamp_col].max().strftime("%Y-%m-%d")

        split_info = {
            "horizon": horizon,
            "total_rows": len(df_sorted),
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "purged_rows": purged_count,
            "train_date_range": f"{train_min_date} to {train_max_date}",
            "val_date_range": f"{val_min_date} to {val_max_date}",
            "val_start_date": str(val_start_date),
            "purge_cutoff_date": str(purge_cutoff_date),
        }

        logger.info(
            f"[PurgedTimeSeriesSplit (H={horizon})] Total: {len(df_sorted)} | "
            f"Train: {len(df_train)} ({train_min_date} to {train_max_date}) | "
            f"Purged: {purged_count} rows | "
            f"Val: {len(df_val)} ({val_min_date} to {val_max_date})"
        )

        return df_train, df_val, split_info
