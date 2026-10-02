import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_triple_barrier_labels(
    df_ohlcv: pd.DataFrame,
    df_features: pd.DataFrame,
    horizons: List[int] = [7, 14],
    atr_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Generate Triple-Barrier labels for each trading day T across specified horizons.
    Calculates Take-Profit (TP), Stop-Loss (SL), or Time (TIME) exit events.

    Barrier Width Formula:
      Width_T = atr_multiplier * ATR_14_T * sqrt(Horizon / 7.0)
      TP_Level = Close_T + Width_T
      SL_Level = Close_T - Width_T

    Rationale for multiplier choice (2.0 * ATR * sqrt(H/7)):
      A 2.0x ATR width represents a 2-standard-deviation move over 7 trading days.
      Scaling by sqrt(H/7) expands barriers for 14-day horizons proportionally to
      volatility diffusion over time, preventing longer-horizon barriers from triggering prematurely.

    End-of-History Handling:
      If fewer than H future bars remain in the dataset and neither TP nor SL is hit
      before dataset ends, the row is explicitly assigned NULL (None) for barrier_hit,
      days_to_hit, and realized_return.

    Args:
        df_ohlcv (pd.DataFrame): Dataframe containing ['timestamp', 'open', 'high', 'low', 'close'].
        df_features (pd.DataFrame): Dataframe containing ['atr_14'].
        horizons (List[int]): List of forecast horizons in trading days (default: [7, 14]).
        atr_multiplier (float): Multiplier for ATR barrier width (default: 2.0).

    Returns:
        pd.DataFrame: Dataframe with columns:
          ['timestamp', 'horizon', 'barrier_hit', 'days_to_hit', 'tp_level', 'sl_level', 'realized_return'].
    """
    if df_ohlcv.empty or df_features.empty:
        return pd.DataFrame()

    df = df_ohlcv.sort_values("timestamp").copy().reset_index(drop=True)
    atr = df_features["atr_14"].astype(float).values
    close = df["close"].astype(float).values
    open_p = df["open"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    timestamps = df["timestamp"].tolist()

    n_rows = len(df)
    results = []

    for h_days in horizons:
        scaling_factor = np.sqrt(h_days / 7.0)

        for i in range(n_rows):
            ts = timestamps[i]
            c_entry = close[i]
            atr_entry = atr[i]

            # If ATR is missing (warmup period), barrier cannot be computed
            if pd.isna(atr_entry) or atr_entry <= 0:
                results.append({
                    "timestamp": ts,
                    "horizon": h_days,
                    "barrier_hit": None,
                    "days_to_hit": None,
                    "tp_level": None,
                    "sl_level": None,
                    "realized_return": None
                })
                continue

            width = atr_multiplier * atr_entry * scaling_factor
            tp_level = c_entry + width
            sl_level = c_entry - width

            barrier_hit = None
            days_to_hit = None
            realized_return = None

            # Look forward up to h_days
            max_future_idx = min(n_rows - 1, i + h_days)

            for step in range(1, h_days + 1):
                future_idx = i + step

                if future_idx >= n_rows:
                    # Reached end of dataset before horizon completed
                    break

                h_fut = high[future_idx]
                l_fut = low[future_idx]
                c_fut = close[future_idx]
                o_fut = open_p[future_idx]

                tp_breached = h_fut >= tp_level
                sl_breached = l_fut <= sl_level

                if tp_breached and sl_breached:
                    # Intraday conflict (both hit on same bar): use candle close direction
                    if c_fut >= o_fut:
                        barrier_hit = "TP"
                        days_to_hit = step
                        realized_return = (tp_level - c_entry) / c_entry
                    else:
                        barrier_hit = "SL"
                        days_to_hit = step
                        realized_return = (sl_level - c_entry) / c_entry
                    break
                elif tp_breached:
                    barrier_hit = "TP"
                    days_to_hit = step
                    realized_return = (tp_level - c_entry) / c_entry
                    break
                elif sl_breached:
                    barrier_hit = "SL"
                    days_to_hit = step
                    realized_return = (sl_level - c_entry) / c_entry
                    break

            # If neither TP nor SL was hit:
            if barrier_hit is None:
                target_exit_idx = i + h_days
                if target_exit_idx < n_rows:
                    # Horizon complete: TIME exit
                    barrier_hit = "TIME"
                    days_to_hit = h_days
                    c_exit = close[target_exit_idx]
                    realized_return = (c_exit - c_entry) / c_entry
                else:
                    # End of history: leave explicit NULL
                    barrier_hit = None
                    days_to_hit = None
                    realized_return = None

            results.append({
                "timestamp": ts,
                "horizon": h_days,
                "barrier_hit": barrier_hit,
                "days_to_hit": days_to_hit,
                "tp_level": float(tp_level),
                "sl_level": float(sl_level),
                "realized_return": float(realized_return) if realized_return is not None else None
            })

    df_labels = pd.DataFrame(results)
    return df_labels
