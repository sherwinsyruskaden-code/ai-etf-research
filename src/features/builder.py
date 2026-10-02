import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import Engine

from src.data.database import InstrumentRepository, OHLCVRepository
from src.features.calculator import compute_technical_features
from src.features.repository import FeatureRepository, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Orchestrates Phase 2 Feature Engineering:
    - Reads daily OHLCV from PostgreSQL for the 10 selected ETFs and context indices.
    - Computes technical indicators, returns, trend, momentum, volatility, volume, and market context.
    - Enforces zero-lookahead discipline.
    - Identifies and tracks rows with insufficient history.
    - Stores feature records idempotently in PostgreSQL features table.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.inst_repo = InstrumentRepository(engine)
        self.ohlcv_repo = OHLCVRepository(engine)
        self.feature_repo = FeatureRepository(engine)

    def build_all_features(self, wipe_existing: bool = False) -> Dict[str, Any]:
        """
        Execute full feature engineering pipeline across all selected ETFs.

        Returns:
            Dict[str, Any]: Summary dictionary of feature computation results.
        """
        logger.info("=" * 85)
        logger.info("STARTING PHASE 2 FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 85)

        if wipe_existing:
            wiped = self.feature_repo.clear_all_features()
            logger.info(f"Wiped {wiped} existing feature rows before recomputing.")

        # 1. Fetch Selected Universe & Context Instruments
        db_instruments = self.inst_repo.get_all_instruments()
        symbol_to_id = {row["trading_symbol"]: row["id"] for _, row in db_instruments.iterrows()}

        selected_etfs = self.inst_repo.get_selected_universe()
        if selected_etfs.empty:
            logger.error("No selected universe ETFs found in database. Run Phase 1 collector first.")
            return {}

        # 2. Build Context Return DataFrames
        context_returns = self._build_context_dataframes(symbol_to_id)

        summary = {
            "total_etfs": len(selected_etfs),
            "total_feature_rows_saved": 0,
            "etf_reports": []
        }

        # 3. Process Each ETF
        for _, etf_row in selected_etfs.iterrows():
            trading_sym = str(etf_row["trading_symbol"])
            inst_id = int(etf_row["id"])

            df_ohlcv = self.ohlcv_repo.get_ohlcv(inst_id)
            if df_ohlcv.empty:
                logger.warning(f"[{trading_sym}] No OHLCV data found in DB. Skipping.")
                continue

            # Compute technical features
            df_feat = compute_technical_features(df_ohlcv)

            # Merge Macro Market Context (Nifty 50, Bank Nifty, VIX)
            df_feat = self._merge_market_context(df_feat, context_returns)

            # Merge Specific Sector Context
            df_feat["sector_return_1d"] = self._compute_sector_return(trading_sym, df_feat, context_returns)

            # Merge Specific Commodity Context
            df_feat["commodity_return_1d"] = self._compute_commodity_return(trading_sym, df_feat, context_returns)

            # Evaluate Insufficient History Flag (rows where 50-day SMA or 20-day volatility is NaN)
            insufficient_mask = df_feat["sma_50"].isna() | df_feat["volatility_20d"].isna()
            insufficient_count = int(insufficient_mask.sum())
            complete_count = len(df_feat) - insufficient_count

            # Check for invalid inf/nan values in computed columns
            invalid_inf = np.isinf(df_feat[FEATURE_COLUMNS].select_dtypes(include=[np.number])).sum().sum()
            if invalid_inf > 0:
                logger.error(f"[{trading_sym}] Found {invalid_inf} infinite values in feature dataframe!")

            # Save batch to database
            save_res = self.feature_repo.save_features_batch(inst_id, df_feat)
            saved_total = save_res["inserted"] + save_res["updated"]
            summary["total_feature_rows_saved"] += saved_total

            etf_report = {
                "trading_symbol": trading_sym,
                "instrument_id": inst_id,
                "total_rows": len(df_feat),
                "complete_rows": complete_count,
                "insufficient_history_rows": insufficient_count,
                "saved_rows": saved_total,
                "inserted": save_res["inserted"],
                "updated": save_res["updated"],
                "min_date": df_feat["timestamp"].min().strftime("%Y-%m-%d"),
                "max_date": df_feat["timestamp"].max().strftime("%Y-%m-%d")
            }
            summary["etf_reports"].append(etf_report)

            logger.info(
                f"[{trading_sym:12s}] Total Rows: {len(df_feat):5d} | Complete: {complete_count:5d} | "
                f"Insufficient History (Flagged): {insufficient_count:3d} | Saved: {saved_total:5d}"
            )

        self._log_summary_report(summary)
        return summary

    def _build_context_dataframes(self, symbol_to_id: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Build reference return DataFrames for context indices and commodity benchmarks."""
        ctx_data = {}

        # 1. Nifty 50 Return
        if "NIFTY" in symbol_to_id:
            df_nifty = self.ohlcv_repo.get_ohlcv(symbol_to_id["NIFTY"])
            df_nifty["nifty50_return_1d"] = df_nifty["close"].pct_change(1)
            ctx_data["NIFTY"] = df_nifty[["timestamp", "nifty50_return_1d"]]

        # 2. Bank Nifty Return
        if "BANKNIFTY" in symbol_to_id:
            df_bank = self.ohlcv_repo.get_ohlcv(symbol_to_id["BANKNIFTY"])
            df_bank["banknifty_return_1d"] = df_bank["close"].pct_change(1)
            ctx_data["BANKNIFTY"] = df_bank[["timestamp", "banknifty_return_1d"]]

        # 3. India VIX Level and 1-day Change
        if "INDIAVIX" in symbol_to_id:
            df_vix = self.ohlcv_repo.get_ohlcv(symbol_to_id["INDIAVIX"])
            df_vix["india_vix_level"] = df_vix["close"]
            df_vix["india_vix_change_1d"] = df_vix["close"].diff(1)
            ctx_data["INDIAVIX"] = df_vix[["timestamp", "india_vix_level", "india_vix_change_1d"]]

        # 4. Nifty IT Return
        if "NIFTYIT" in symbol_to_id:
            df_it = self.ohlcv_repo.get_ohlcv(symbol_to_id["NIFTYIT"])
            df_it["niftyit_return_1d"] = df_it["close"].pct_change(1)
            ctx_data["NIFTYIT"] = df_it[["timestamp", "niftyit_return_1d"]]

        # 5. Nifty Next 50 Return
        if "NIFTYJR" in symbol_to_id:
            df_jr = self.ohlcv_repo.get_ohlcv(symbol_to_id["NIFTYJR"])
            df_jr["niftyjr_return_1d"] = df_jr["close"].pct_change(1)
            ctx_data["NIFTYJR"] = df_jr[["timestamp", "niftyjr_return_1d"]]

        # 6. Gold Commodity Benchmark Return (derived from GOLDBEES)
        if "GOLDBEES" in symbol_to_id:
            df_gold = self.ohlcv_repo.get_ohlcv(symbol_to_id["GOLDBEES"])
            df_gold["gold_return_1d"] = df_gold["close"].pct_change(1)
            ctx_data["GOLD"] = df_gold[["timestamp", "gold_return_1d"]]

        # 7. Silver Commodity Benchmark Return (derived from SILVERBEES)
        if "SILVERBEES" in symbol_to_id:
            df_silver = self.ohlcv_repo.get_ohlcv(symbol_to_id["SILVERBEES"])
            df_silver["silver_return_1d"] = df_silver["close"].pct_change(1)
            ctx_data["SILVER"] = df_silver[["timestamp", "silver_return_1d"]]

        return ctx_data

    def _merge_market_context(self, df_feat: pd.DataFrame, context_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge macro context columns (Nifty 50, Bank Nifty, India VIX)."""
        df_merged = df_feat.copy()

        if "NIFTY" in context_data:
            df_merged = pd.merge(df_merged, context_data["NIFTY"], on="timestamp", how="left")
        else:
            df_merged["nifty50_return_1d"] = np.nan

        if "BANKNIFTY" in context_data:
            df_merged = pd.merge(df_merged, context_data["BANKNIFTY"], on="timestamp", how="left")
        else:
            df_merged["banknifty_return_1d"] = np.nan

        if "INDIAVIX" in context_data:
            df_merged = pd.merge(df_merged, context_data["INDIAVIX"], on="timestamp", how="left")
        else:
            df_merged["india_vix_level"] = np.nan
            df_merged["india_vix_change_1d"] = np.nan

        return df_merged

    def _compute_sector_return(
        self,
        trading_sym: str,
        df_feat: pd.DataFrame,
        context_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Return sector return series matching the ETF's sector."""
        if trading_sym == "BANKBEES" and "BANKNIFTY" in context_data:
            df_m = pd.merge(df_feat[["timestamp"]], context_data["BANKNIFTY"], on="timestamp", how="left")
            return df_m["banknifty_return_1d"]
        elif trading_sym == "ITBEES" and "NIFTYIT" in context_data:
            df_m = pd.merge(df_feat[["timestamp"]], context_data["NIFTYIT"], on="timestamp", how="left")
            return df_m["niftyit_return_1d"]
        elif trading_sym == "JUNIORBEES" and "NIFTYJR" in context_data:
            df_m = pd.merge(df_feat[["timestamp"]], context_data["NIFTYJR"], on="timestamp", how="left")
            return df_m["niftyjr_return_1d"]
        elif trading_sym == "NIFTYBEES" and "NIFTY" in context_data:
            df_m = pd.merge(df_feat[["timestamp"]], context_data["NIFTY"], on="timestamp", how="left")
            return df_m["nifty50_return_1d"]
        else:
            return pd.Series(index=df_feat.index, dtype=float)

    def _compute_commodity_return(
        self,
        trading_sym: str,
        df_feat: pd.DataFrame,
        context_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Return commodity benchmark return series for Gold/Silver ETFs."""
        if trading_sym in ["GOLDBEES", "GOLDIETF", "HDFCGOLD", "GOLD1"] and "GOLD" in context_data:
            df_m = pd.merge(df_feat[["timestamp"]], context_data["GOLD"], on="timestamp", how="left")
            return df_m["gold_return_1d"]
        elif trading_sym in ["SILVERBEES", "SILVERIETF"] and "SILVER" in context_data:
            df_m = pd.merge(df_feat[["timestamp"]], context_data["SILVER"], on="timestamp", how="left")
            return df_m["silver_return_1d"]
        else:
            return pd.Series(index=df_feat.index, dtype=float)

    def _log_summary_report(self, summary: Dict[str, Any]) -> None:
        """Log summary metrics report for Phase 2 feature builder."""
        logger.info("=" * 85)
        logger.info("PHASE 2 FEATURE ENGINEERING EXECUTION SUMMARY")
        logger.info("=" * 85)
        logger.info(f"Total ETFs Processed           : {summary['total_etfs']}")
        logger.info(f"Total Feature Rows Saved to DB : {summary['total_feature_rows_saved']}")
        logger.info("-" * 85)
        for rep in summary["etf_reports"]:
            logger.info(
                f"ETF: {rep['trading_symbol']:12s} | Date Range: {rep['min_date']} to {rep['max_date']} | "
                f"Total Rows: {rep['total_rows']:5d} | Complete Features: {rep['complete_rows']:5d} | "
                f"Insufficient History (Flagged): {rep['insufficient_history_rows']:3d}"
            )
        logger.info("=" * 85)
