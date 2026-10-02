import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pandas as pd
from sqlalchemy import Engine

from src.data.groww_client import GrowwClient
from src.data.instruments import InstrumentManager
from src.data.validator import DataValidator
from src.data.database import InstrumentRepository, OHLCVRepository

logger = logging.getLogger(__name__)


class ETFDataCollector:
    """
    Core orchestrator for Phase 1 ETF Data Ingestion:
    - Instrument Discovery
    - Liquid / Overnight Fund Exclusion & Equity Universe Selection
    - Historical Multi-Year Daily OHLCV Candle Collection
    - Data Validation
    - Idempotent PostgreSQL Storage with IST Trading Date Alignment
    """

    def __init__(
        self,
        groww_client: GrowwClient,
        db_engine: Engine,
        validator: Optional[DataValidator] = None
    ):
        self.client = groww_client
        self.engine = db_engine
        self.instrument_mgr = InstrumentManager(groww_client)
        self.validator = validator or DataValidator()
        self.inst_repo = InstrumentRepository(db_engine)
        self.ohlcv_repo = OHLCVRepository(db_engine)

    def run_pipeline(
        self,
        historical_days: int = 3650,
        top_n: int = 10,
        wipe_existing_ohlcv: bool = False
    ) -> Dict[str, Any]:
        """
        Execute full ingestion pipeline.
        
        Args:
            historical_days (int): Number of days of historical history.
            top_n (int): Number of top liquid ETFs to select.
            wipe_existing_ohlcv (bool): If True, clears existing OHLCV records before inserting fresh data.

        Returns:
            Dict[str, Any]: Detailed summary metrics dictionary.
        """
        exec_mode = getattr(self.client, "execution_mode", "UNKNOWN")

        logger.info("=" * 85)
        logger.info(f"STARTING PHASE 1 ETF DATA INGESTION PIPELINE")
        logger.info(f"EXECUTION MODE: [{exec_mode}]")
        logger.info(f"HISTORICAL COVERAGE: Up to {historical_days} days (Full Multi-Year History)")
        logger.info("=" * 85)

        pipeline_summary = {
            "start_time": datetime.now().isoformat(),
            "execution_mode": exec_mode,
            "instruments_discovered": 0,
            "etf_candidates_found": 0,
            "selected_etf_count": 0,
            "date_range_requested": "",
            "candles_downloaded_total": 0,
            "rows_inserted_total": 0,
            "duplicates_skipped_total": 0,
            "validation_failures": 0,
            "wiped_old_rows": 0,
            "etf_details": [],
        }

        # Step 0: Wipe old OHLCV records if requested
        if wipe_existing_ohlcv:
            wiped_count = self.ohlcv_repo.clear_all_ohlcv()
            pipeline_summary["wiped_old_rows"] = wiped_count

        # Step 1: Instrument Master Discovery
        logger.info("Step 1: Downloading official Groww instrument master...")
        df_all = self.client.fetch_all_instruments()
        df_nse_cash = self.instrument_mgr.discover_nse_cash_instruments(df_all)
        pipeline_summary["instruments_discovered"] = len(df_nse_cash)

        if not df_nse_cash.empty:
            self.inst_repo.save_instruments(df_nse_cash)

        # Step 2: Filter Genuine ETF Candidates (with Liquid Fund Exclusion)
        logger.info("Step 2: Identifying genuine Equity/Index/Sector/Commodity ETF candidates...")
        df_etf_candidates = self.instrument_mgr.filter_etf_candidates(df_nse_cash)
        pipeline_summary["etf_candidates_found"] = len(df_etf_candidates)

        if df_etf_candidates.empty:
            logger.error("No genuine ETF candidates discovered. Aborting ingestion pipeline.")
            return pipeline_summary

        # Step 3: Data-driven Liquidity Universe Selection
        logger.info("Step 3: Evaluating candidate liquidity and selecting top ETF universe...")
        df_selected, df_ranked, metrics_dict = self.instrument_mgr.select_top_etf_universe(
            df_candidates=df_etf_candidates,
            historical_days=historical_days,
            top_n=top_n
        )
        pipeline_summary["selected_etf_count"] = len(df_selected)

        selected_symbols = df_selected["groww_symbol"].tolist() if not df_selected.empty else []
        self.inst_repo.mark_selected_universe(selected_symbols, metrics_dict)

        # Step 4: Historical Multi-Year Daily OHLCV Candle Collection
        end_date = datetime.now()
        start_date = datetime(2000, 1, 1)
        date_range_str = f"All Available Multi-Year History (2000 to {end_date.strftime('%Y-%m-%d')})"
        pipeline_summary["date_range_requested"] = date_range_str
        logger.info(f"Step 4: Fetching historical daily OHLCV candles [{date_range_str}]...")

        db_instruments = self.inst_repo.get_selected_universe()
        symbol_to_id = {row["groww_symbol"]: row["id"] for _, row in db_instruments.iterrows()}

        for _, etf_row in df_selected.iterrows():
            groww_sym = str(etf_row["groww_symbol"])
            trading_sym = str(etf_row["trading_symbol"])
            inst_id = symbol_to_id.get(groww_sym)

            if not inst_id:
                logger.warning(f"Instrument ID for {groww_sym} not found in DB. Skipping.")
                continue

            logger.info(f"Downloading multi-year historical candles for {trading_sym} ({groww_sym})...")
            df_candles = self.client.fetch_historical_daily_candles(
                groww_symbol=groww_sym,
                start_date=start_date,
                end_date=end_date,
                exchange=str(etf_row.get("exchange", "NSE")),
                segment=str(etf_row.get("segment", "CASH"))
            )

            downloaded_count = len(df_candles)
            pipeline_summary["candles_downloaded_total"] += downloaded_count

            if df_candles.empty:
                logger.warning(f"No candles returned for {trading_sym}.")
                continue

            # Validate data
            is_valid, val_errors, val_summary = self.validator.validate_ohlcv(df_candles, symbol=trading_sym)

            if not is_valid:
                pipeline_summary["validation_failures"] += 1
                logger.error(f"Validation failed for {trading_sym}: {val_errors}")
                continue

            # Idempotent storage in PostgreSQL
            db_res = self.ohlcv_repo.save_ohlcv_batch(inst_id, df_candles)
            inserted = db_res["inserted"]
            skipped = db_res["skipped"]

            pipeline_summary["rows_inserted_total"] += inserted
            pipeline_summary["duplicates_skipped_total"] += skipped

            min_date = df_candles["timestamp"].min().strftime("%Y-%m-%d")
            max_date = df_candles["timestamp"].max().strftime("%Y-%m-%d")

            etf_detail = {
                "trading_symbol": trading_sym,
                "groww_symbol": groww_sym,
                "downloaded": downloaded_count,
                "inserted": inserted,
                "duplicates_skipped": skipped,
                "date_range": f"{min_date} to {max_date}",
                "valid": is_valid,
            }
            pipeline_summary["etf_details"].append(etf_detail)

            logger.info(
                f"[{trading_sym}] Range: {min_date} to {max_date} | Candles: {downloaded_count} | "
                f"Inserted: {inserted} | Duplicates Skipped: {skipped} | Valid: {is_valid}"
            )

        # Step 5: Ingest Market Context Indices (Nifty 50, Bank Nifty, India VIX, Nifty IT, Nifty Next 50)
        logger.info("Step 5: Fetching and storing market context index candles...")
        self.ingest_market_context_indices(start_date=start_date, end_date=end_date)

        pipeline_summary["execution_mode"] = getattr(self.client, "execution_mode", "UNKNOWN")
        pipeline_summary["end_time"] = datetime.now().isoformat()
        self._log_pipeline_summary(pipeline_summary)
        return pipeline_summary

    def ingest_market_context_indices(self, start_date: datetime, end_date: datetime) -> None:
        """Ingest market context index candles (Nifty 50, Bank Nifty, VIX, Nifty IT, Nifty Next 50)."""
        context_tuples = [
            ("NSE", "CASH", "NIFTY", "NSE-NIFTY", "IDX", "NIFTY 50"),
            ("NSE", "CASH", "BANKNIFTY", "NSE-BANKNIFTY", "IDX", "NIFTY Bank"),
            ("NSE", "CASH", "INDIAVIX", "NSE-INDIAVIX", "IDX", "India Vix"),
            ("NSE", "CASH", "NIFTYIT", "NSE-NIFTYIT", "IDX", "NIFTY IT"),
            ("NSE", "CASH", "NIFTYJR", "NSE-NIFTYJR", "IDX", "Nifty Next 50"),
        ]
        df_ctx = pd.DataFrame(
            context_tuples,
            columns=["exchange", "segment", "trading_symbol", "groww_symbol", "instrument_type", "name"]
        )
        self.inst_repo.save_instruments(df_ctx)

        all_insts = self.inst_repo.get_all_instruments()
        sym_to_id = {row["groww_symbol"]: row["id"] for _, row in all_insts.iterrows()}

        for _, row in df_ctx.iterrows():
            gsym = row["groww_symbol"]
            tsym = row["trading_symbol"]
            inst_id = sym_to_id.get(gsym)
            if not inst_id:
                continue

            df_candles = self.client.fetch_historical_daily_candles(
                groww_symbol=gsym,
                start_date=start_date,
                end_date=end_date,
                exchange="NSE",
                segment="CASH"
            )
            if not df_candles.empty:
                is_valid, errs, _ = self.validator.validate_ohlcv(df_candles, symbol=tsym)
                if is_valid:
                    self.ohlcv_repo.save_ohlcv_batch(inst_id, df_candles)


    def _log_pipeline_summary(self, summary: Dict[str, Any]) -> None:
        """Log pipeline metrics summary report."""
        logger.info("=" * 85)
        logger.info("PIPELINE EXECUTION SUMMARY REPORT")
        logger.info("=" * 85)
        logger.info(f"Execution Mode                   : [{summary['execution_mode']}]")
        logger.info(f"Instruments Discovered (NSE CASH) : {summary['instruments_discovered']}")
        logger.info(f"Genuine ETF Candidates Identified: {summary['etf_candidates_found']}")
        logger.info(f"ETFs Selected in Universe       : {summary['selected_etf_count']}")
        logger.info(f"Date Range Requested             : {summary['date_range_requested']}")
        logger.info(f"Total Candles Downloaded         : {summary['candles_downloaded_total']}")
        logger.info(f"Total DB Rows Inserted           : {summary['rows_inserted_total']}")
        logger.info(f"Total Duplicates Skipped         : {summary['duplicates_skipped_total']}")
        logger.info(f"Validation Failures Count        : {summary['validation_failures']}")
        if summary.get("wiped_old_rows", 0) > 0:
            logger.info(f"Wiped Mislabeled Old Rows        : {summary['wiped_old_rows']}")
        logger.info("=" * 85)
