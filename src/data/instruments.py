import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Any
from src.data.groww_client import GrowwClient

logger = logging.getLogger(__name__)


class InstrumentManager:
    """
    Manages instrument discovery, filtering, and data-driven liquidity-based ETF universe selection.
    Includes explicit exclusion of liquid/overnight/money-market cash equivalent funds.
    """

    def __init__(self, groww_client: GrowwClient):
        self.groww_client = groww_client

    def discover_nse_cash_instruments(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """
        Filter instrument master list for NSE CASH instruments.
        """
        if df_all.empty:
            logger.warning("Input DataFrame for instrument discovery is empty.")
            return pd.DataFrame()

        exchange_col = df_all["exchange"].astype(str).str.upper() if "exchange" in df_all else pd.Series()
        segment_col = df_all["segment"].astype(str).str.upper() if "segment" in df_all else pd.Series()

        df_nse_cash = df_all[
            (exchange_col == "NSE") & (segment_col == "CASH")
        ].copy()

        logger.info(f"Discovered {len(df_nse_cash)} NSE CASH instruments out of {len(df_all)} total instruments.")
        return df_nse_cash

    def filter_etf_candidates(self, df_nse_cash: pd.DataFrame) -> pd.DataFrame:
        """
        Identify genuine Equity / Index / Sector / Commodity ETF candidates dynamically.
        Applies explicit exclusion for liquid/overnight/money-market cash equivalent funds.
        """
        if df_nse_cash.empty:
            return pd.DataFrame()

        inst_type = df_nse_cash.get("instrument_type", pd.Series()).astype(str).str.upper()
        trading_sym = df_nse_cash.get("trading_symbol", pd.Series()).astype(str).str.upper()
        groww_sym = df_nse_cash.get("groww_symbol", pd.Series()).astype(str).str.upper()
        name_col = df_nse_cash.get("name", pd.Series()).astype(str).str.upper()

        # Step 1: Broad ETF / Index / Sector Candidate Matching
        is_explicit_etf = inst_type == "ETF"
        is_etf_pattern = (
            trading_sym.str.contains(r"BEES|ETF$", regex=True, na=False) |
            groww_sym.str.contains(r"BEES|ETF$", regex=True, na=False) |
            name_col.str.contains(r"EXCHANGE TRADED|ETF| BEES", regex=True, na=False)
        )
        
        # Exclude non-ETF corporate stocks matching substring patterns
        not_individual_stock = (
            ~name_col.str.contains(r"LIMITED|LTD|MINES|INTNL|DIAMONDS|SOLUTIONS|PRODUCTS|MILLS", regex=True, na=False) |
            is_explicit_etf
        )
        is_not_fno = ~inst_type.isin(["CE", "PE", "FUT", "OPT"])

        candidates = df_nse_cash[(is_explicit_etf | (is_etf_pattern & not_individual_stock)) & is_not_fno].copy()

        # Step 2: EXPLICIT EXCLUSION FILTER FOR LIQUID / OVERNIGHT / MONEY MARKET FUNDS
        liquid_pattern = r"LIQUID|OVERNIGHT|CASH|TREPS|MONEYMKT|MONEY MARKET|1D RATE|ARBITRAGE|LIQID|NAV"
        is_liquid_fund = (
            candidates["trading_symbol"].astype(str).str.contains(liquid_pattern, regex=True, na=False) |
            candidates["groww_symbol"].astype(str).str.contains(liquid_pattern, regex=True, na=False) |
            candidates["name"].astype(str).str.contains(liquid_pattern, regex=True, na=False)
        )

        excluded_liquid = candidates[is_liquid_fund]
        df_genuine_etfs = candidates[~is_liquid_fund].copy()

        if "groww_symbol" in df_genuine_etfs.columns:
            df_genuine_etfs.drop_duplicates(subset=["groww_symbol"], inplace=True)

        logger.info(f"Identified {len(candidates)} raw ETF candidates.")
        logger.info(f"Excluded {len(excluded_liquid)} liquid/overnight/money-market cash-equivalent funds.")
        logger.info(f"Retained {len(df_genuine_etfs)} genuine Equity/Index/Sector/Commodity ETF candidates for liquidity ranking.")

        return df_genuine_etfs

    def select_top_etf_universe(
        self,
        df_candidates: pd.DataFrame,
        historical_days: int = 60,
        top_n: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Data-driven selection process:
        1. For each genuine ETF candidate, fetch recent daily candles.
        2. Compute Average Daily Traded Value = Average(volume * close).
        3. Rank candidates by liquidity metric.
        4. Select top_n candidates and log detailed auditable reasoning table.

        Returns:
            Tuple of (selected_etfs_df, full_ranked_candidates_df, liquidity_metrics_dict)
        """
        if df_candidates.empty:
            logger.warning("No candidate ETFs provided for universe selection.")
            return pd.DataFrame(), pd.DataFrame(), {}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Use recent 90-day window for liquidity ranking

        rankings = []
        metrics_dict = {}

        logger.info(f"Evaluating traded liquidity for {len(df_candidates)} genuine ETF candidates...")

        for idx, row in df_candidates.iterrows():
            groww_sym = str(row["groww_symbol"])
            trading_sym = str(row["trading_symbol"])
            exchange = str(row.get("exchange", "NSE"))
            segment = str(row.get("segment", "CASH"))

            df_candles = self.groww_client.fetch_historical_daily_candles(
                groww_symbol=groww_sym,
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
                segment=segment
            )

            if df_candles.empty:
                avg_daily_volume = 0.0
                avg_daily_value = 0.0
                candle_count = 0
            else:
                candle_count = len(df_candles)
                avg_daily_volume = float(df_candles["volume"].mean())
                avg_daily_value = float((df_candles["volume"] * df_candles["close"]).mean())

            metrics_dict[groww_sym] = avg_daily_value

            rankings.append({
                "groww_symbol": groww_sym,
                "trading_symbol": trading_sym,
                "name": str(row.get("name", trading_sym)),
                "candle_count": candle_count,
                "avg_daily_volume": avg_daily_volume,
                "avg_daily_value": avg_daily_value,
                "exchange": exchange,
                "segment": segment,
                "instrument_type": str(row.get("instrument_type", "ETF"))
            })

        df_ranked = pd.DataFrame(rankings)
        if df_ranked.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        df_ranked.sort_values(by="avg_daily_value", ascending=False, inplace=True)
        df_ranked.reset_index(drop=True, inplace=True)

        df_ranked["rank"] = df_ranked.index + 1
        df_ranked["selected"] = df_ranked["rank"] <= top_n
        df_ranked["status"] = df_ranked["selected"].map({True: "SELECTED", False: "REJECTED"})

        selected_df = df_ranked[df_ranked["selected"]].copy()

        logger.info("=" * 95)
        logger.info(f"GENUINE EQUITY/INDEX/SECTOR/COMMODITY ETF LIQUIDITY SELECTION RANKING")
        logger.info("Metric: Average Daily Traded Value (INR) = Mean(Volume * Close)")
        logger.info("=" * 95)
        for _, r in df_ranked.head(30).iterrows():
            status_flag = "[SELECTED]" if r["selected"] else "[REJECTED]"
            logger.info(
                f"Rank {r['rank']:2d} | {status_flag:10s} | {r['trading_symbol']:15s} | "
                f"Avg Vol: {r['avg_daily_volume']:12,.0f} | "
                f"Avg Value (INR): {r['avg_daily_value']:15,.2f} | "
                f"Candles: {r['candle_count']:3d}"
            )
        if len(df_ranked) > 30:
            logger.info(f"... and {len(df_ranked) - 30} additional rejected lower-liquidity candidate ETFs logged.")
        logger.info("=" * 95)
        logger.info(f"Selected Top {len(selected_df)} genuine ETFs for universe out of {len(df_ranked)} candidates.")

        return selected_df, df_ranked, metrics_dict
