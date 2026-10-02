import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_causal_regime(df_ohlcv: pd.DataFrame, df_features: pd.DataFrame) -> pd.Series:
    """
    Detect market regime causally for a single price series.
    Uses ONLY historical data up to and including day T (zero lookahead).

    Rule Logic:
    1. HIGH_VOLATILITY:
       If 20-day realized volatility (volatility_20d) on day T exceeds the trailing 252-day
       80th percentile volatility threshold calculated causally as of day T.

    2. TRENDING_UP:
       If Close_T > SMA_20_T AND SMA_20_T > SMA_50_T AND 5-day slope of SMA_20 > +0.2% (+0.002).

    3. TRENDING_DOWN:
       If Close_T < SMA_20_T AND SMA_20_T < SMA_50_T AND 5-day slope of SMA_20 < -0.2% (-0.002).

    4. SIDEWAYS:
       Default regime when volatility is normal and price/MAs exhibit no strong trend slope.

    Args:
        df_ohlcv (pd.DataFrame): Dataframe containing ['timestamp', 'close'].
        df_features (pd.DataFrame): Dataframe containing ['volatility_20d', 'sma_20', 'sma_50'].

    Returns:
        pd.Series: Series of regime string labels indexed by df_ohlcv.index.
    """
    if df_ohlcv.empty or df_features.empty:
        return pd.Series(dtype=str)

    close = df_ohlcv["close"].astype(float)
    vol20 = df_features["volatility_20d"].astype(float)
    sma20 = df_features["sma_20"].astype(float)
    sma50 = df_features["sma_50"].astype(float)

    # 1. Trailing/Expanding 252-day 80th percentile volatility threshold (causal window)
    # Uses expanding window for days 20..251 (all past available days up to T) and trailing 252-day window thereafter.
    vol_p80 = vol20.rolling(window=252, min_periods=1).quantile(0.80)

    # 2. 5-day slope of 20-day SMA
    sma20_slope = (sma20 - sma20.shift(5)) / sma20.shift(5)

    regimes = []
    for i in range(len(df_ohlcv)):
        v_now = vol20.iloc[i]
        v_thresh = vol_p80.iloc[i]
        c_now = close.iloc[i]
        s20 = sma20.iloc[i]
        s50 = sma50.iloc[i]
        slope = sma20_slope.iloc[i]

        if pd.isna(c_now) or pd.isna(s20) or pd.isna(s50):
            regimes.append("SIDEWAYS")
        elif pd.notna(v_now) and pd.notna(v_thresh) and v_now > v_thresh:
            regimes.append("HIGH_VOLATILITY")
        elif c_now > s20 and s20 > s50 and pd.notna(slope) and slope > 0.002:
            regimes.append("TRENDING_UP")
        elif c_now < s20 and s20 < s50 and pd.notna(slope) and slope < -0.002:
            regimes.append("TRENDING_DOWN")
        else:
            regimes.append("SIDEWAYS")

    return pd.Series(regimes, index=df_ohlcv.index)
