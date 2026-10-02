import logging
import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


def compute_technical_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily technical indicators for a single instrument.
    Guarantees strict zero-lookahead (only data available up to day T is used).

    Args:
        df_ohlcv (pd.DataFrame): Dataframe containing ['timestamp', 'open', 'high', 'low', 'close', 'volume'].

    Returns:
        pd.DataFrame: Dataframe with timestamp and computed technical feature columns.
    """
    if df_ohlcv.empty:
        return pd.DataFrame()

    df = df_ohlcv.sort_values("timestamp").copy()
    df.reset_index(drop=True, inplace=True)

    features = pd.DataFrame({"timestamp": df["timestamp"]})

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # 1. Momentum / Returns
    features["return_1d"] = close.pct_change(1)
    features["return_3d"] = close.pct_change(3)
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_20d"] = close.pct_change(20)

    # 2. Trend
    features["sma_5"] = ta.trend.SMAIndicator(close=close, window=5).sma_indicator()
    features["sma_20"] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    features["sma_50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

    features["price_vs_sma5"] = close / features["sma_5"]
    features["price_vs_sma20"] = close / features["sma_20"]
    features["price_vs_sma50"] = close / features["sma_50"]
    features["sma20_vs_sma50"] = features["sma_20"] / features["sma_50"]

    # 3. Momentum Indicators
    features["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    features["macd"] = macd_obj.macd()
    features["macd_signal"] = macd_obj.macd_signal()
    features["macd_hist"] = macd_obj.macd_diff()

    features["roc_10"] = ta.momentum.ROCIndicator(close=close, window=10).roc()

    # 4. Volatility
    features["volatility_5d"] = features["return_1d"].rolling(window=5).std()
    features["volatility_10d"] = features["return_1d"].rolling(window=10).std()
    features["volatility_20d"] = features["return_1d"].rolling(window=20).std()

    features["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # 5. Volume
    features["volume_today"] = df["volume"].astype(int)
    features["volume_sma20"] = volume.rolling(window=20).mean()
    features["volume_ratio"] = volume / features["volume_sma20"]
    features["volume_change"] = volume.pct_change(1)

    # Replace any infinite values from division by zero with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    return features
