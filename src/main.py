import logging
import sys
import os
import argparse
from sqlalchemy import create_engine, text
from src.config.settings import DATABASE_URL, HISTORICAL_DAYS, TOP_ETF_COUNT
from src.data.database import get_db_engine, init_db, InstrumentRepository, OHLCVRepository
from src.data.groww_client import GrowwClient
from src.data.collector import ETFDataCollector

from src.features.builder import FeatureBuilder
from src.features.repository import FeatureRepository
from src.regime_labels.builder import RegimeLabelBuilder
from src.regime_labels.repository import RegimeLabelRepository

from src.models.trainer import ModelTrainer
from src.backtest.engine import WalkForwardBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("main")


def main() -> None:
    """Main entry point for AI ETF Quantitative Research System (Phases 1, 2, 3, 4, & 5)."""
    parser = argparse.ArgumentParser(description="AI ETF Quantitative Research System CLI")
    parser.add_argument("--ingest", action="store_true", help="Force re-running Phase 1 data ingestion pipeline")
    parser.add_argument("--wipe", action="store_true", help="Wipe existing database records before ingestion")
    parser.add_argument("--compute-features", action="store_true", help="Compute and store features for selected ETFs")
    parser.add_argument("--wipe-features", action="store_true", help="Wipe existing features table before recomputing")
    parser.add_argument("--compute-phase3", action="store_true", help="Compute Phase 3 causal regimes and triple-barrier labels")
    parser.add_argument("--wipe-phase3", action="store_true", help="Wipe existing regime and triple-barrier label tables before recomputing")
    parser.add_argument("--train-models", action="store_true", help="Train Phase 4 models (Logistic Regression, Random Forest, XGBoost)")
    parser.add_argument("--wipe-models", action="store_true", help="Wipe existing model_runs table records before training")
    parser.add_argument("--run-backtest", action="store_true", help="Run Phase 5 Walk-Forward Backtest with realistic costs & spread")
    parser.add_argument("--wipe-backtest", action="store_true", help="Wipe existing backtest_results table records before running backtest")
    args = parser.parse_args()

    logger.info("Initializing AI ETF Quantitative Research System - Phase 1 Ingestion, Phase 2 Features, Phase 3 Regimes/Labels, Phase 4 Models, & Phase 5 Walk-Forward Backtesting")

    # 1. Connect to database
    logger.info(f"Connecting to database via URL: {DATABASE_URL}")
    engine = get_db_engine(DATABASE_URL)
    init_db(engine)

    # Check database state
    inst_repo = InstrumentRepository(engine)
    ohlcv_repo = OHLCVRepository(engine)
    feat_repo = FeatureRepository(engine)
    reg_repo = RegimeLabelRepository(engine)

    selected_universe = inst_repo.get_selected_universe()
    has_ohlcv = not selected_universe.empty and ohlcv_repo.get_total_candle_count() > 0

    # 2. Phase 1 Data Ingestion Pipeline
    if args.ingest or args.wipe or not has_ohlcv:
        logger.info("Running Phase 1 Ingestion Pipeline...")
        groww_client = GrowwClient()
        collector = ETFDataCollector(groww_client=groww_client, db_engine=engine)
        summary_ingest = collector.run_pipeline(
            historical_days=HISTORICAL_DAYS,
            top_n=TOP_ETF_COUNT,
            wipe_existing_ohlcv=args.wipe
        )
    else:
        logger.info("Phase 1 OHLCV data already populated in PostgreSQL. Skipping network re-ingestion.")

    # 3. Phase 2 Feature Engineering Pipeline
    with engine.connect() as conn:
        feat_count = conn.execute(text("SELECT count(*) FROM features")).scalar() or 0

    if args.compute_features or args.wipe or args.wipe_features or feat_count == 0:
        logger.info("Running Phase 2 Feature Engineering Pipeline...")
        feature_builder = FeatureBuilder(engine=engine)
        summary_features = feature_builder.build_all_features(wipe_existing=args.wipe_features or args.wipe)
        logger.info("Phase 2 Feature Engineering Pipeline Execution Completed Successfully.")
    else:
        logger.info("Phase 2 Features already populated in PostgreSQL. Skipping feature re-computation.")

    # 4. Phase 3 Causal Regime Detection & Triple-Barrier Labeling Pipeline
    with engine.connect() as conn:
        reg_count = conn.execute(text("SELECT count(*) FROM market_regimes")).scalar() or 0

    if args.compute_phase3 or args.wipe or args.wipe_phase3 or reg_count == 0:
        logger.info("Running Phase 3 Causal Regime Detection & Triple-Barrier Labeling Pipeline...")
        regime_label_builder = RegimeLabelBuilder(engine=engine)
        summary_p3 = regime_label_builder.build_all_regimes_and_labels(
            horizons=[7, 14],
            atr_multiplier=2.0,
            wipe_existing=args.wipe_phase3 or args.wipe
        )
        logger.info("Phase 3 Regime Detection & Triple-Barrier Labeling Execution Completed Successfully.")
    else:
        logger.info("Phase 3 Regimes & Labels already populated in PostgreSQL.")

    # 5. Phase 4 Model Training & Evaluation Pipeline
    with engine.connect() as conn:
        model_runs_count = conn.execute(text("SELECT count(*) FROM model_runs")).scalar() or 0

    if args.train_models or args.wipe or args.wipe_models or model_runs_count == 0:
        logger.info("Running Phase 4 Model Training Pipeline (Logistic Regression, Random Forest, XGBoost)...")
        trainer = ModelTrainer(engine=engine)
        summary_p4 = trainer.train_and_evaluate_all(
            horizons=[7, 14],
            wipe_existing=args.wipe_models or args.wipe
        )
        logger.info("Phase 4 Model Training & Evaluation Pipeline Completed Successfully.")
    else:
        logger.info("Phase 4 Trained Model Runs already logged in PostgreSQL.")

    # 6. Phase 5 Walk-Forward Backtesting Pipeline
    with engine.connect() as conn:
        backtest_count = conn.execute(text("SELECT count(*) FROM backtest_results")).scalar() or 0

    if args.run_backtest or args.wipe or args.wipe_backtest or backtest_count == 0:
        logger.info("Running Phase 5 Walk-Forward Backtesting Pipeline...")
        if args.wipe_backtest or args.wipe:
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM backtest_results;"))
                logger.info("Wiped old backtest_results records before running backtest.")

        backtester = WalkForwardBacktester(engine=engine)
        summary_p5 = backtester.run_full_walk_forward_backtest(
            horizons=[7, 14],
            n_folds=5,
            prob_threshold=0.35,
            sl_avoid_threshold=0.30
        )
        logger.info("Phase 5 Walk-Forward Backtesting Pipeline Completed Successfully.")
    else:
        logger.info("Phase 5 Backtest Results already logged in PostgreSQL.")

    logger.info("System Execution Completed Successfully.")


if __name__ == "__main__":
    main()




