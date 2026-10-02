import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy import Engine

from src.data.database import InstrumentRepository, OHLCVRepository
from src.features.repository import FeatureRepository
from src.regime_labels.regime import detect_causal_regime
from src.regime_labels.labeler import generate_triple_barrier_labels
from src.regime_labels.repository import RegimeLabelRepository

logger = logging.getLogger(__name__)


class RegimeLabelBuilder:
    """
    Orchestrates Phase 3 Causal Regime Detection & Triple-Barrier Labeling.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.inst_repo = InstrumentRepository(engine)
        self.ohlcv_repo = OHLCVRepository(engine)
        self.feat_repo = FeatureRepository(engine)
        self.repo = RegimeLabelRepository(engine)

    def build_all_regimes_and_labels(
        self,
        horizons: List[int] = [7, 14],
        atr_multiplier: float = 2.0,
        wipe_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Execute full Phase 3 Pipeline across all 10 selected ETFs.

        Returns:
            Dict[str, Any]: Detailed summary metrics and distributions dictionary.
        """
        logger.info("=" * 85)
        logger.info("STARTING PHASE 3 CAUSAL REGIME DETECTION & TRIPLE-BARRIER LABELING PIPELINE")
        logger.info("=" * 85)

        if wipe_existing:
            w_reg, w_lbl = self.repo.clear_all_regimes_and_labels()
            logger.info(f"Wiped {w_reg} old regime rows and {w_lbl} old label rows before computing.")

        # 1. Fetch Selected Universe & Context Instruments
        db_instruments = self.inst_repo.get_all_instruments()
        symbol_to_id = {row["trading_symbol"]: row["id"] for _, row in db_instruments.iterrows()}

        selected_etfs = self.inst_repo.get_selected_universe()
        if selected_etfs.empty:
            logger.error("No selected universe ETFs found in database. Run Phase 1 collector first.")
            return {}

        # 2. Compute Broad Market Regime (Nifty 50)
        nifty_regime_series = pd.Series(dtype=str)
        nifty_sym = "NIFTYBEES" if "NIFTYBEES" in symbol_to_id else ("NIFTY" if "NIFTY" in symbol_to_id else None)
        if nifty_sym:
            nifty_id = symbol_to_id[nifty_sym]
            df_nifty_ohlcv = self.ohlcv_repo.get_ohlcv(nifty_id)
            df_nifty_feat = self.feat_repo.get_features(nifty_id)

            if not df_nifty_ohlcv.empty and not df_nifty_feat.empty:
                nifty_regime_labels = detect_causal_regime(df_nifty_ohlcv, df_nifty_feat)
                nifty_regime_series = pd.Series(
                    nifty_regime_labels.values,
                    index=df_nifty_ohlcv["timestamp"]
                )

        summary = {
            "total_etfs": len(selected_etfs),
            "horizons": horizons,
            "atr_multiplier": atr_multiplier,
            "etf_reports": [],
            "overall_regime_dist": {},
            "overall_label_dist": {}
        }

        all_regimes = []
        all_label_hits = []

        # 3. Process Each ETF
        for _, etf_row in selected_etfs.iterrows():
            trading_sym = str(etf_row["trading_symbol"])
            inst_id = int(etf_row["id"])

            df_ohlcv = self.ohlcv_repo.get_ohlcv(inst_id)
            df_feat = self.feat_repo.get_features(inst_id)

            if df_ohlcv.empty or df_feat.empty:
                logger.warning(f"[{trading_sym}] Incomplete OHLCV or Feature data. Skipping.")
                continue

            # A. Detect Instrument Causal Regime
            inst_regimes = detect_causal_regime(df_ohlcv, df_feat)

            # Map Broad Market Regime onto ETF Timestamps
            df_regimes = pd.DataFrame({
                "timestamp": df_ohlcv["timestamp"],
                "instrument_regime": inst_regimes
            })

            if not nifty_regime_series.empty:
                df_regimes = pd.merge(
                    df_regimes,
                    pd.DataFrame({"timestamp": nifty_regime_series.index, "market_regime": nifty_regime_series.values}),
                    on="timestamp",
                    how="left"
                )
                df_regimes["market_regime"] = df_regimes["market_regime"].fillna("SIDEWAYS")
            else:
                df_regimes["market_regime"] = "SIDEWAYS"

            reg_save = self.repo.save_regimes_batch(inst_id, df_regimes)

            # B. Generate Triple-Barrier Labels
            df_labels = generate_triple_barrier_labels(
                df_ohlcv=df_ohlcv,
                df_features=df_feat,
                horizons=horizons,
                atr_multiplier=atr_multiplier
            )

            lbl_save = self.repo.save_labels_batch(inst_id, df_labels)

            # C. Collect ETF Metrics Report
            reg_counts = df_regimes["instrument_regime"].value_counts().to_dict()
            all_regimes.extend(df_regimes["instrument_regime"].tolist())

            label_counts_by_h = {}
            unlabeled_by_h = {}

            for h in horizons:
                sub_lbl = df_labels[df_labels["horizon"] == h]
                hit_counts = sub_lbl["barrier_hit"].value_counts(dropna=False).to_dict()
                # Format counts including NaN/None as 'NULL'
                hit_counts_formatted = {
                    ("NULL" if pd.isna(k) or k is None else str(k)): v for k, v in hit_counts.items()
                }
                label_counts_by_h[h] = hit_counts_formatted
                unlabeled_by_h[h] = hit_counts_formatted.get("NULL", 0)

                all_label_hits.extend(sub_lbl["barrier_hit"].map(lambda x: "NULL" if pd.isna(x) or x is None else str(x)).tolist())

            etf_report = {
                "trading_symbol": trading_sym,
                "instrument_id": inst_id,
                "total_rows": len(df_ohlcv),
                "regimes_saved": reg_save["inserted"] + reg_save["updated"],
                "labels_saved": lbl_save["inserted"] + lbl_save["updated"],
                "regime_distribution": reg_counts,
                "label_distribution_by_horizon": label_counts_by_h,
                "unlabeled_end_of_history": unlabeled_by_h
            }
            summary["etf_reports"].append(etf_report)

            logger.info(
                f"[{trading_sym:12s}] Rows: {len(df_ohlcv):5d} | "
                f"Regimes (UP/DN/SW/HV): {reg_counts.get('TRENDING_UP',0)}/{reg_counts.get('TRENDING_DOWN',0)}/{reg_counts.get('SIDEWAYS',0)}/{reg_counts.get('HIGH_VOLATILITY',0)} | "
                f"Labels H7 (TP/SL/TIME/NULL): {label_counts_by_h.get(7,{}).get('TP',0)}/{label_counts_by_h.get(7,{}).get('SL',0)}/{label_counts_by_h.get(7,{}).get('TIME',0)}/{label_counts_by_h.get(7,{}).get('NULL',0)} | "
                f"Labels H14: {label_counts_by_h.get(14,{}).get('TP',0)}/{label_counts_by_h.get(14,{}).get('SL',0)}/{label_counts_by_h.get(14,{}).get('TIME',0)}/{label_counts_by_h.get(14,{}).get('NULL',0)}"
            )

        # Overall distributions
        if all_regimes:
            tot_r = len(all_regimes)
            s_r = pd.Series(all_regimes).value_counts()
            summary["overall_regime_dist"] = {k: f"{v/tot_r:.2%}" for k, v in s_r.items()}

        if all_label_hits:
            tot_l = len(all_label_hits)
            s_l = pd.Series(all_label_hits).value_counts()
            summary["overall_label_dist"] = {k: f"{v/tot_l:.2%}" for k, v in s_l.items()}

        self._log_summary_report(summary)
        return summary

    def _log_summary_report(self, summary: Dict[str, Any]) -> None:
        """Log summary metrics report for Phase 3 pipeline."""
        logger.info("=" * 85)
        logger.info("PHASE 3 REGIME DETECTION & TRIPLE-BARRIER LABELING SUMMARY REPORT")
        logger.info("=" * 85)
        logger.info(f"Total ETFs Processed : {summary['total_etfs']}")
        logger.info(f"Overall Regime Dist : {summary.get('overall_regime_dist', {})}")
        logger.info(f"Overall Label Dist  : {summary.get('overall_label_dist', {})}")
        logger.info("=" * 85)
