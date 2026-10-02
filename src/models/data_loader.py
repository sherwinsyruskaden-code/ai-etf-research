import logging
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy import Engine, text

from src.data.database import InstrumentRepository
from src.features.repository import FeatureRepository
from src.regime_labels.repository import RegimeLabelRepository

logger = logging.getLogger(__name__)

TARGET_MAP = {"TP": 0, "SL": 1, "TIME": 2}
REVERSE_TARGET_MAP = {0: "TP", 1: "SL", 2: "TIME"}
CLASS_NAMES = ["TP", "SL", "TIME"]

# Columns from triple_barrier_labels that MUST BE EXCLUDED to prevent leakage
PROHIBITED_LABEL_COLUMNS = {
    "barrier_hit", "days_to_hit", "tp_level", "sl_level", "realized_return"
}


class ModelDataLoader:
    """
    Data Loader for Phase 4 Model Training:
    - Queries features, market regimes, and triple barrier labels from PostgreSQL.
    - Joins across all 10 universe ETFs into a pooled dataset.
    - Filters out NULL barrier labels (start ATR warmup & end-of-history incomplete rows).
    - Encodes 3-class target y (0: TP, 1: SL, 2: TIME).
    - Encodes regime categoricals via One-Hot Encoding.
    - Enforces zero-leakage guarantee by strictly excluding future label metadata from features.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.inst_repo = InstrumentRepository(engine)

    def load_dataset_for_horizon(
        self,
        horizon: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load pooled dataset for a given horizon across selected universe ETFs.

        Args:
            horizon (int): Target horizon (7 or 14).

        Returns:
            Tuple[pd.DataFrame, List[str]]:
                - df_dataset: Joined dataset containing ['timestamp', 'instrument_id', 'target', <features...>]
                - feature_names: List of feature column names.
        """
        query = text("""
            SELECT 
                f.instrument_id,
                f.timestamp,
                -- Features from Phase 2
                f.return_1d, f.return_3d, f.return_5d, f.return_10d, f.return_20d,
                f.sma_5, f.sma_20, f.sma_50,
                f.price_vs_sma5, f.price_vs_sma20, f.price_vs_sma50, f.sma20_vs_sma50,
                f.rsi_14, f.macd, f.macd_signal, f.macd_hist, f.roc_10,
                f.volatility_5d, f.volatility_10d, f.volatility_20d, f.atr_14,
                f.volume_today, f.volume_sma20, f.volume_ratio, f.volume_change,
                f.nifty50_return_1d, f.banknifty_return_1d, f.india_vix_level, f.india_vix_change_1d,
                f.sector_return_1d, f.commodity_return_1d,
                -- Causal Regimes from Phase 3
                r.instrument_regime,
                r.market_regime,
                -- Target from Phase 3
                l.barrier_hit
            FROM features f
            JOIN market_regimes r ON f.instrument_id = r.instrument_id AND f.timestamp = r.timestamp
            JOIN triple_barrier_labels l ON f.instrument_id = l.instrument_id AND f.timestamp = l.timestamp AND l.horizon = :horizon
            JOIN instruments i ON f.instrument_id = i.id
            WHERE i.is_selected_universe = TRUE
              AND l.barrier_hit IS NOT NULL
            ORDER BY f.timestamp ASC, f.instrument_id ASC;
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"horizon": horizon})

        if df.empty:
            logger.warning(f"No usable dataset rows found for horizon {horizon}.")
            return pd.DataFrame(), []

        # 1. Target Encoding (TP -> 0, SL -> 1, TIME -> 2)
        df["target"] = df["barrier_hit"].map(TARGET_MAP)
        df = df[df["target"].notna()].copy()
        df["target"] = df["target"].astype(int)

        # 2. One-Hot Encode Regime Categoricals
        regime_dummies = pd.get_dummies(
            df[["instrument_regime", "market_regime"]],
            prefix=["inst_regime", "mkt_regime"],
            dtype=float
        )

        # Drop original raw categorical string columns & raw label column
        df_clean = df.drop(columns=["barrier_hit", "instrument_regime", "market_regime"])
        df_full = pd.concat([df_clean, regime_dummies], axis=1)

        # 3. Identify Feature Column Names (Strict Leakage Audit)
        non_feature_cols = {"instrument_id", "timestamp", "target"}
        feature_names = [c for c in df_full.columns if c not in non_feature_cols]

        # Audit against prohibited label leakage
        leaked = set(feature_names).intersection(PROHIBITED_LABEL_COLUMNS)
        if leaked:
            raise RuntimeError(f"CRITICAL LEAKAGE DETECTED! Prohibited label columns found in feature set: {leaked}")

        # 4. Explicit NULL Handling for Market Context Features
        # DO NOT fill pre-launch market context features with fake 0.0 values!
        # Leave them as explicit NULLs (np.nan) so XGBoost handles them natively without artificial zero-fill proxy bias.
        # Add explicit _is_missing indicator columns for LogisticRegression and RandomForest.
        context_cols = [
            "nifty50_return_1d", "banknifty_return_1d", "india_vix_level",
            "india_vix_change_1d", "sector_return_1d", "commodity_return_1d"
        ]

        for col in context_cols:
            if col in df_full.columns:
                df_full[f"{col}_is_missing"] = df_full[col].isna().astype(float)

        non_context_features = [c for c in feature_names if c not in context_cols]
        # Fill only non-context technical features (warmup days) with 0.0
        df_full[non_context_features] = df_full[non_context_features].fillna(0.0)

        # Refresh feature_names list to include new missing indicator columns
        feature_names = [c for c in df_full.columns if c not in non_feature_cols]

        logger.info(
            f"[DataLoader (H={horizon})] Loaded {len(df_full)} pooled samples across selected universe ETFs. "
            f"Feature count: {len(feature_names)}. Target distribution: {df_full['target'].value_counts().to_dict()}"
        )

        return df_full, feature_names
