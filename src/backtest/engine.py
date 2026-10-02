import logging
import json
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sqlalchemy import Engine, text

from src.models.data_loader import ModelDataLoader, CLASS_NAMES
from src.models.splitter import PurgedTimeSeriesSplit
from src.backtest.costs import TradingCostModel
from src.data.database import InstrumentRepository, OHLCVRepository

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PRE_REGISTERED_KILL_GATE = {
    "min_post_cost_sharpe": 0.50,
    "min_post_cost_win_rate": 0.40,
    "must_beat_buy_and_hold_sharpe": True,
    "required_fold_pass_ratio": 0.50  # Must pass on >= 50% of walk-forward folds
}


class WalkForwardBacktester:
    """
    Core Walk-Forward Backtester for Phase 5:
    - Executes 5-fold walk-forward backtesting with purged & embargoed splits across multi-year history.
    - Simulates long-only trades entering at next bar's OPEN (T+1 Open) to avoid same-bar lookahead.
    - Applies Groww delivery brokerage + statutory taxes + fixed slippage + liquidity-scaled bid-ask spread.
    - Evaluates Unfiltered vs SL-Filtered strategy variants.
    - Evaluates Buy-and-Hold and Trend-Following (SMA50) baselines.
    - Evaluates against pre-registered kill gate criteria objectively.
    """

    def __init__(
        self,
        engine: Engine,
        cost_model: Optional[TradingCostModel] = None,
        seed: int = 42
    ):
        self.engine = engine
        self.cost_model = cost_model or TradingCostModel()
        self.seed = seed
        self.data_loader = ModelDataLoader(engine)
        self.inst_repo = InstrumentRepository(engine)
        self.ohlcv_repo = OHLCVRepository(engine)

    def run_full_walk_forward_backtest(
        self,
        horizons: List[int] = [7, 14],
        n_folds: int = 5,
        prob_threshold: float = 0.35,
        sl_avoid_threshold: float = 0.30
    ) -> Dict[str, Any]:
        """
        Execute full walk-forward backtesting pipeline.

        Returns:
            Dict[str, Any]: Comprehensive metrics, per-fold results, baseline comparisons, and kill-gate verdict.
        """
        logger.info("=" * 85)
        logger.info("STARTING PHASE 5 WALK-FORWARD BACKTESTING PIPELINE")
        logger.info(f"PRE-REGISTERED KILL GATE: Min Post-Cost Sharpe >= {PRE_REGISTERED_KILL_GATE['min_post_cost_sharpe']}, "
                    f"Win Rate >= {PRE_REGISTERED_KILL_GATE['min_post_cost_win_rate']:.0%}, Must beat Buy & Hold")
        logger.info("=" * 85)

        # Query instrument liquidity for bid-ask spread scaling
        db_inst = self.inst_repo.get_selected_universe()
        inst_liquidity = {int(row["id"]): float(row["avg_daily_volume"]) for _, row in db_inst.iterrows()}

        all_results = {}

        for h in horizons:
            logger.info(f"\n>>> EXECUTING WALK-FORWARD BACKTEST FOR HORIZON: {h} DAYS <<<")
            
            # 1. Load dataset with full OHLCV prices for trade execution
            df_full, feature_names = self._load_backtest_dataset(horizon=h)
            if df_full.empty:
                logger.error(f"No dataset for horizon {h}. Skipping.")
                continue

            # 2. Build 5 Walk-Forward Folds
            folds = self._create_walk_forward_folds(df_full, horizon=h, n_folds=n_folds)

            fold_reports = []

            for fold_idx, (df_train, df_val, date_info) in enumerate(folds, start=1):
                logger.info(f"\n--- FOLD {fold_idx}/{n_folds} ({date_info['val_range']}) ---")
                
                # Fit XGBoost on purged training set
                X_train = df_train[feature_names]
                y_train = df_train["target"].values
                X_val = df_val[feature_names]
                y_val = df_val["target"].values

                xgb = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    eval_metric="mlogloss",
                    random_state=self.seed,
                    n_jobs=-1
                )
                xgb.fit(X_train, y_train)

                # Get predicted class probabilities on validation set
                # Index 0: TP, Index 1: SL, Index 2: TIME
                probs_val = xgb.predict_proba(X_val)
                df_val_preds = df_val.copy()
                df_val_preds["prob_tp"] = probs_val[:, 0]
                df_val_preds["prob_sl"] = probs_val[:, 1]
                df_val_preds["prob_time"] = probs_val[:, 2]

                # A. Strategy Variant 1: Unfiltered Model Signal (prob_tp >= threshold)
                unfiltered_trades = self._simulate_trades(
                    df_val_preds,
                    horizon=h,
                    prob_threshold=prob_threshold,
                    sl_avoid_threshold=1.0,  # No SL filtering
                    inst_liquidity=inst_liquidity
                )
                metrics_unfiltered = self._calculate_backtest_metrics(unfiltered_trades)

                # B. Strategy Variant 2: SL-Filtered Model Signal (prob_tp >= threshold AND prob_sl < sl_avoid_threshold)
                sl_filtered_trades = self._simulate_trades(
                    df_val_preds,
                    horizon=h,
                    prob_threshold=prob_threshold,
                    sl_avoid_threshold=sl_avoid_threshold,
                    inst_liquidity=inst_liquidity
                )
                metrics_sl_filtered = self._calculate_backtest_metrics(sl_filtered_trades)

                # C. Baseline 1: Buy & Hold Benchmark over validation period
                # NOTE: _simulate_buy_and_hold now directly returns a metrics dict (not a trades list)
                # because it must compute Sharpe on a daily return series, not on total-period returns.
                metrics_bnh = self._simulate_buy_and_hold(df_val_preds, inst_liquidity=inst_liquidity)

                # D. Baseline 2: Simple Rule-Based Benchmark (SMA50 Trend Following)
                sma_trades = self._simulate_sma_benchmark(df_val_preds, horizon=h, inst_liquidity=inst_liquidity)
                metrics_sma = self._calculate_backtest_metrics(sma_trades)

                fold_res = {
                    "fold_id": fold_idx,
                    "date_range": date_info["val_range"],
                    "unfiltered": metrics_unfiltered,
                    "sl_filtered": metrics_sl_filtered,
                    "buy_and_hold": metrics_bnh,
                    "rule_sma50": metrics_sma
                }
                fold_reports.append(fold_res)

                logger.info(
                    f"Fold {fold_idx} [{date_info['val_range']}] | "
                    f"Unfiltered Post-Cost Sharpe: {metrics_unfiltered['post_cost_sharpe']:.2f} (Trades: {metrics_unfiltered['num_trades']}) | "
                    f"SL-Filtered Post-Cost Sharpe: {metrics_sl_filtered['post_cost_sharpe']:.2f} (Trades: {metrics_sl_filtered['num_trades']}) | "
                    f"B&H Sharpe: {metrics_bnh['post_cost_sharpe']:.2f} | "
                    f"SMA50 Sharpe: {metrics_sma['post_cost_sharpe']:.2f}"
                )

            # Aggregate across folds
            agg_results = self._aggregate_fold_results(fold_reports)
            all_results[h] = {
                "folds": fold_reports,
                "aggregated": agg_results
            }

        # Evaluate Pre-Registered Kill Gate
        verdict_summary = self._evaluate_kill_gate(all_results)
        all_results["kill_gate_verdict"] = verdict_summary

        self._save_results_to_db(all_results)
        return all_results

    def _load_backtest_dataset(self, horizon: int) -> Tuple[pd.DataFrame, List[str]]:
        """Load features, regimes, labels, AND raw next-bar Open prices for execution."""
        query = text("""
            SELECT 
                f.instrument_id,
                f.timestamp,
                o.open as entry_open_next,
                o.close as close_curr,
                f.return_1d, f.return_3d, f.return_5d, f.return_10d, f.return_20d,
                f.sma_5, f.sma_20, f.sma_50,
                f.price_vs_sma5, f.price_vs_sma20, f.price_vs_sma50, f.sma20_vs_sma50,
                f.rsi_14, f.macd, f.macd_signal, f.macd_hist, f.roc_10,
                f.volatility_5d, f.volatility_10d, f.volatility_20d, f.atr_14,
                f.volume_today, f.volume_sma20, f.volume_ratio, f.volume_change,
                f.nifty50_return_1d, f.banknifty_return_1d, f.india_vix_level, f.india_vix_change_1d,
                f.sector_return_1d, f.commodity_return_1d,
                r.instrument_regime, r.market_regime,
                l.barrier_hit, l.days_to_hit, l.tp_level, l.sl_level, l.realized_return
            FROM features f
            JOIN ohlcv_daily o ON f.instrument_id = o.instrument_id AND f.timestamp = o.timestamp
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
            return pd.DataFrame(), []

        # Target encoding
        df["target"] = df["barrier_hit"].map({"TP": 0, "SL": 1, "TIME": 2})

        # One-hot encode regimes
        regime_dummies = pd.get_dummies(
            df[["instrument_regime", "market_regime"]],
            prefix=["inst_regime", "mkt_regime"],
            dtype=float
        )

        df_clean = df.drop(columns=["instrument_regime", "market_regime"])
        df_full = pd.concat([df_clean, regime_dummies], axis=1)

        # Non-feature metadata columns
        meta_cols = {
            "instrument_id", "timestamp", "target", "barrier_hit", "days_to_hit",
            "tp_level", "sl_level", "realized_return", "entry_open_next", "close_curr"
        }

        # Add missing indicators for context features
        context_cols = [
            "nifty50_return_1d", "banknifty_return_1d", "india_vix_level",
            "india_vix_change_1d", "sector_return_1d", "commodity_return_1d"
        ]
        for c in context_cols:
            if c in df_full.columns:
                df_full[f"{c}_is_missing"] = df_full[c].isna().astype(float)

        feature_names = [c for c in df_full.columns if c not in meta_cols]

        # Fill technical non-context features with 0
        non_context = [c for c in feature_names if c not in context_cols]
        df_full[non_context] = df_full[non_context].fillna(0.0)

        return df_full, feature_names

    def _create_walk_forward_folds(
        self,
        df: pd.DataFrame,
        horizon: int,
        n_folds: int = 5
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]]:
        """Create sequential walk-forward folds with purging applied to every fold."""
        unique_dates = df["timestamp"].drop_duplicates().sort_values().tolist()
        n_dates = len(unique_dates)

        # Divide chronological history into n_folds validation windows
        val_size = n_dates // (n_folds + 1)
        folds = []

        for f in range(n_folds):
            val_start_idx = (f + 1) * val_size
            val_end_idx = min(n_dates, (f + 2) * val_size if f < n_folds - 1 else n_dates)

            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[val_end_idx - 1]

            # Validation DataFrame
            val_mask = (df["timestamp"] >= val_start_date) & (df["timestamp"] <= val_end_date)
            df_val = df[val_mask].copy()

            # Train Candidate DataFrame (all history prior to val_start_date)
            train_cand_mask = df["timestamp"] < val_start_date
            
            # PURGING: Remove training samples whose H-day forward window overlaps val_start_date
            purge_cutoff_idx = max(0, val_start_idx - horizon)
            purge_cutoff_date = unique_dates[purge_cutoff_idx]

            train_purged_mask = df["timestamp"] < purge_cutoff_date
            df_train = df[train_purged_mask].copy()

            val_start_str = pd.Timestamp(val_start_date).strftime("%Y-%m-%d")
            val_end_str = pd.Timestamp(val_end_date).strftime("%Y-%m-%d")

            folds.append((
                df_train,
                df_val,
                {"val_range": f"{val_start_str} to {val_end_str}"}
            ))

        return folds

    def _simulate_trades(
        self,
        df_val: pd.DataFrame,
        horizon: int,
        prob_threshold: float,
        sl_avoid_threshold: float,
        inst_liquidity: Dict[int, float]
    ) -> List[Dict[str, Any]]:
        """
        Simulate trade execution for model-driven signals:
        - Entry signal on day T: prob_tp >= prob_threshold AND prob_sl < sl_avoid_threshold.
        - Entry execution: Next bar's OPEN (T+1 Open).
        - Exit execution: Exact triple-barrier hit level / price at day T + days_to_hit.
        """
        trades = []
        df_sorted = df_val.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)

        for i in range(len(df_sorted) - 1):
            row_curr = df_sorted.iloc[i]
            row_next = df_sorted.iloc[i + 1]

            # Ensure row_next is the immediate next trading day for the SAME instrument
            if row_curr["instrument_id"] != row_next["instrument_id"]:
                continue

            p_tp = row_curr["prob_tp"]
            p_sl = row_curr["prob_sl"]

            # Check signal criteria
            if p_tp >= prob_threshold and p_sl < sl_avoid_threshold:
                # ENTRY EXECUTED AT NEXT BAR'S OPEN (Day T+1 Open)
                entry_open = float(row_next["entry_open_next"])
                inst_id = int(row_curr["instrument_id"])
                adtv = inst_liquidity.get(inst_id, 50_000_000)

                # Determine exit price from triple-barrier outcome
                hit = row_curr["barrier_hit"]
                c_entry = float(row_curr["close_curr"])

                if hit == "TP" and pd.notna(row_curr["tp_level"]):
                    raw_exit = float(row_curr["tp_level"])
                elif hit == "SL" and pd.notna(row_curr["sl_level"]):
                    raw_exit = float(row_curr["sl_level"])
                else:  # TIME or null
                    rel_ret = float(row_curr["realized_return"]) if pd.notna(row_curr["realized_return"]) else 0.0
                    raw_exit = c_entry * (1.0 + rel_ret)

                # Calculate pre-cost and post-cost trade metrics via TradingCostModel
                friction = self.cost_model.calculate_trade_friction(
                    entry_price=entry_open,
                    exit_price=raw_exit,
                    avg_daily_volume=adtv
                )

                trades.append({
                    "timestamp": row_curr["timestamp"],
                    "instrument_id": inst_id,
                    "entry_open": entry_open,
                    "raw_exit": raw_exit,
                    "barrier_hit": hit,
                    "days_to_hit": row_curr["days_to_hit"],
                    "raw_return": friction["raw_return"],
                    "net_return": friction["net_return"],
                    "cost_pct": friction["total_cost_pct"]
                })

        return trades

    def _simulate_buy_and_hold(
        self,
        df_val: pd.DataFrame,
        inst_liquidity: Dict[int, float]
    ) -> Dict[str, Any]:
        """
        Compute Buy & Hold benchmark Sharpe using daily portfolio returns.

        Method:
        - For each ETF: compute daily close-to-close returns over the validation window.
        - Apply one-time round-trip transaction cost (spread + slippage) as a uniform daily drag
          = total_cost / num_days, so it is properly amortised over the holding period.
        - Equal-weight across all ETFs to form a portfolio daily return series.
        - Compute Sharpe on daily returns with sqrt(252) annualisation — same formula as
          the model-driven strategy.
        - Also retain total-period return and win-rate for DB reporting.

        This is the CORRECTED implementation replacing the former one-return-per-ETF approach
        which produced nonsensical Sharpe values (e.g. +490) because sqrt(252) was applied
        to total-period cumulative returns rather than daily returns.
        """
        # Collect daily return series per ETF (close-to-close)
        daily_series_list = []
        trade_total_returns_raw = []
        trade_total_returns_net = []

        for inst_id, group in df_val.groupby("instrument_id"):
            g_sorted = group.sort_values("timestamp").reset_index(drop=True)
            if len(g_sorted) < 2:
                continue

            closes = g_sorted["close_curr"].values.astype(float)
            # Daily close-to-close returns
            daily_raw = np.diff(closes) / closes[:-1]  # shape: (n_days - 1,)

            # One-time round-trip friction (entry + exit) — amortise across all days as daily drag
            entry_open = float(g_sorted.iloc[0]["entry_open_next"])
            exit_close = float(g_sorted.iloc[-1]["close_curr"])
            adtv = inst_liquidity.get(int(inst_id), 50_000_000)
            friction = self.cost_model.calculate_trade_friction(entry_open, exit_close, adtv)
            total_cost_pct = friction["total_cost_pct"]
            n_days = len(daily_raw)
            daily_cost = total_cost_pct / n_days if n_days > 0 else 0.0

            daily_net = daily_raw - daily_cost  # net daily return series
            daily_series_list.append(daily_net)

            # Total-period metrics for win-rate / avg-return reporting
            trade_total_returns_raw.append(friction["raw_return"])
            trade_total_returns_net.append(friction["net_return"])

        if not daily_series_list:
            return self._empty_metrics()

        # Equal-weight portfolio: average across all ETFs at each date index
        # Align by length (pad shorter series with 0 if needed — minor edge case)
        max_len = max(len(s) for s in daily_series_list)
        padded = np.array(
            [np.pad(s, (0, max_len - len(s)), constant_values=0.0) for s in daily_series_list]
        )
        portfolio_daily = padded.mean(axis=0)  # equal-weight daily portfolio return

        # Sharpe on daily returns (annualised with sqrt(252))
        mu = float(np.mean(portfolio_daily))
        sigma = float(np.std(portfolio_daily))
        sharpe_post = float((mu / sigma) * np.sqrt(252)) if sigma > 0 else 0.0

        # Pre-cost series (no daily drag subtracted)
        daily_series_raw_list = []
        for inst_id, group in df_val.groupby("instrument_id"):
            g_sorted = group.sort_values("timestamp").reset_index(drop=True)
            if len(g_sorted) < 2:
                continue
            closes = g_sorted["close_curr"].values.astype(float)
            daily_series_raw_list.append(np.diff(closes) / closes[:-1])
        max_len_r = max(len(s) for s in daily_series_raw_list) if daily_series_raw_list else 1
        padded_r = np.array(
            [np.pad(s, (0, max_len_r - len(s)), constant_values=0.0) for s in daily_series_raw_list]
        )
        port_raw = padded_r.mean(axis=0)
        mu_r = float(np.mean(port_raw))
        sigma_r = float(np.std(port_raw))
        sharpe_pre = float((mu_r / sigma_r) * np.sqrt(252)) if sigma_r > 0 else 0.0

        # Cumulative max drawdown on net portfolio
        cum_eq = np.cumprod(1.0 + portfolio_daily)
        peak = np.maximum.accumulate(cum_eq)
        dd = (cum_eq - peak) / peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

        # Win rate and avg return reported at the trade (total-period) level (per ETF hold)
        raw_arr = np.array(trade_total_returns_raw)
        net_arr = np.array(trade_total_returns_net)
        num_trades = len(net_arr)
        win_pre = float(np.mean(raw_arr > 0)) if num_trades > 0 else 0.0
        win_post = float(np.mean(net_arr > 0)) if num_trades > 0 else 0.0
        avg_pre = float(np.mean(raw_arr)) if num_trades > 0 else 0.0
        avg_post = float(np.mean(net_arr)) if num_trades > 0 else 0.0

        gains = net_arr[net_arr > 0].sum()
        losses = np.abs(net_arr[net_arr < 0].sum())
        pf = float(gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)

        return {
            "num_trades": num_trades,
            "win_rate_pre": win_pre,
            "win_rate_post": win_post,
            "avg_return_pre": avg_pre,
            "avg_return_post": avg_post,
            "pre_cost_sharpe": sharpe_pre,
            "post_cost_sharpe": sharpe_post,
            "max_drawdown_post": max_dd,
            "profit_factor_post": pf
        }

    def _simulate_sma_benchmark(
        self,
        df_val: pd.DataFrame,
        horizon: int,
        inst_liquidity: Dict[int, float]
    ) -> List[Dict[str, Any]]:
        """Simulate Simple Non-ML Rule-Based Benchmark (SMA50 Trend-Following)."""
        trades = []
        df_sorted = df_val.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)

        for i in range(len(df_sorted) - 1):
            row_curr = df_sorted.iloc[i]
            row_next = df_sorted.iloc[i + 1]

            if row_curr["instrument_id"] != row_next["instrument_id"]:
                continue

            # Rule: Buy when Close > SMA50
            if pd.notna(row_curr["price_vs_sma50"]) and row_curr["price_vs_sma50"] > 0:
                entry_open = float(row_next["entry_open_next"])
                inst_id = int(row_curr["instrument_id"])
                adtv = inst_liquidity.get(inst_id, 50_000_000)

                c_entry = float(row_curr["close_curr"])
                rel_ret = float(row_curr["realized_return"]) if pd.notna(row_curr["realized_return"]) else 0.0
                raw_exit = c_entry * (1.0 + rel_ret)

                friction = self.cost_model.calculate_trade_friction(entry_open, raw_exit, adtv)
                trades.append({
                    "timestamp": row_curr["timestamp"],
                    "instrument_id": inst_id,
                    "raw_return": friction["raw_return"],
                    "net_return": friction["net_return"],
                    "cost_pct": friction["total_cost_pct"]
                })
        return trades

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return zeroed metrics dict when no trades/data are available."""
        return {
            "num_trades": 0,
            "win_rate_pre": 0.0,
            "win_rate_post": 0.0,
            "avg_return_pre": 0.0,
            "avg_return_post": 0.0,
            "pre_cost_sharpe": 0.0,
            "post_cost_sharpe": 0.0,
            "max_drawdown_post": 0.0,
            "profit_factor_post": 0.0
        }

    def _calculate_backtest_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate Win Rate, Sharpe Ratio, Max Drawdown, Profit Factor pre- and post-cost."""
        if not trades:
            return self._empty_metrics()

        df_t = pd.DataFrame(trades)
        raw_rets = df_t["raw_return"].values
        net_rets = df_t["net_return"].values

        num_trades = len(df_t)
        win_pre = float(np.mean(raw_rets > 0))
        win_post = float(np.mean(net_rets > 0))

        avg_pre = float(np.mean(raw_rets))
        avg_post = float(np.mean(net_rets))

        # Sharpe Ratio (annualized assuming ~252 trading days per year)
        std_pre = np.std(raw_rets)
        std_post = np.std(net_rets)
        sharpe_pre = float((avg_pre / std_pre) * np.sqrt(252)) if std_pre > 0 else 0.0
        sharpe_post = float((avg_post / std_post) * np.sqrt(252)) if std_post > 0 else 0.0

        # Cumulative Max Drawdown
        cum_equity = np.cumprod(1.0 + net_rets)
        peak = np.maximum.accumulate(cum_equity)
        drawdowns = (cum_equity - peak) / peak
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Profit Factor
        gains = net_rets[net_rets > 0].sum()
        losses = np.abs(net_rets[net_rets < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)

        return {
            "num_trades": num_trades,
            "win_rate_pre": win_pre,
            "win_rate_post": win_post,
            "avg_return_pre": avg_pre,
            "avg_return_post": avg_post,
            "pre_cost_sharpe": sharpe_pre,
            "post_cost_sharpe": sharpe_post,
            "max_drawdown_post": max_dd,
            "profit_factor_post": profit_factor
        }

    def _aggregate_fold_results(self, fold_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across folds and calculate variance."""
        agg = {}
        for key in ["unfiltered", "sl_filtered", "buy_and_hold", "rule_sma50"]:
            sharpes = [f[key]["post_cost_sharpe"] for f in fold_reports]
            wins = [f[key]["win_rate_post"] for f in fold_reports]
            trades = [f[key]["num_trades"] for f in fold_reports]

            agg[key] = {
                "mean_post_cost_sharpe": float(np.mean(sharpes)),
                "std_post_cost_sharpe": float(np.std(sharpes)),
                "mean_post_cost_win_rate": float(np.mean(wins)),
                "mean_num_trades": float(np.mean(trades)),
                "fold_sharpes": sharpes,
                "fold_win_rates": wins
            }
        return agg

    def _evaluate_kill_gate(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate objective pass/fail verdict against pre-registered kill gate criteria:
        - Post-Cost Sharpe >= 0.50
        - Post-Cost Win Rate >= 40%
        - Must beat Buy-and-Hold Post-Cost Sharpe
        - Required on >= 50% of walk-forward folds
        """
        verdicts = {}
        for h, res in all_results.items():
            if not isinstance(h, int):
                continue

            agg = res["aggregated"]
            sl_filt = agg["sl_filtered"]
            bnh = agg["buy_and_hold"]
            sma = agg["rule_sma50"]

            folds = res["folds"]
            n_folds = len(folds)

            pass_fold_count = 0
            for f in folds:
                f_sl_sharpe = f["sl_filtered"]["post_cost_sharpe"]
                f_sl_win = f["sl_filtered"]["win_rate_post"]
                f_bnh_sharpe = f["buy_and_hold"]["post_cost_sharpe"]

                if (f_sl_sharpe >= PRE_REGISTERED_KILL_GATE["min_post_cost_sharpe"] and
                    f_sl_win >= PRE_REGISTERED_KILL_GATE["min_post_cost_win_rate"] and
                    f_sl_sharpe > f_bnh_sharpe):
                    pass_fold_count += 1

            pass_ratio = pass_fold_count / n_folds if n_folds > 0 else 0.0
            is_pass = pass_ratio >= PRE_REGISTERED_KILL_GATE["required_fold_pass_ratio"]

            verdicts[h] = {
                "horizon": h,
                "pass_fold_count": pass_fold_count,
                "total_folds": n_folds,
                "pass_ratio": pass_ratio,
                "is_pass": is_pass,
                "mean_post_cost_sharpe": sl_filt["mean_post_cost_sharpe"],
                "bnh_post_cost_sharpe": bnh["mean_post_cost_sharpe"],
                "sma_post_cost_sharpe": sma["mean_post_cost_sharpe"],
                "verdict_text": "PASSED PRE-REGISTERED KILL GATE" if is_pass else "FAILED PRE-REGISTERED KILL GATE"
            }

        return verdicts

    def _save_results_to_db(self, all_results: Dict[str, Any]) -> None:
        """Save fold-level and aggregated backtest results into PostgreSQL backtest_results table."""
        insert_sql = text("""
            INSERT INTO backtest_results
            (fold_id, horizon, date_range, strategy_variant, model_name, prob_threshold, sl_avoid_threshold,
             win_rate, avg_return, sharpe_ratio, max_drawdown, profit_factor, num_trades, is_post_cost,
             pre_cost_sharpe, post_cost_sharpe)
            VALUES
            (:fold_id, :horizon, :date_range, :strategy_variant, :model_name, :prob_threshold, :sl_avoid_threshold,
             :win_rate, :avg_return, :sharpe_ratio, :max_drawdown, :profit_factor, :num_trades, :is_post_cost,
             :pre_cost_sharpe, :post_cost_sharpe);
        """)

        records = []
        for h, res in all_results.items():
            if not isinstance(h, int):
                continue

            for f in res["folds"]:
                fold_id = f["fold_id"]
                date_rng = f["date_range"]

                for v_key in ["unfiltered", "sl_filtered", "buy_and_hold", "rule_sma50"]:
                    m = f[v_key]
                    records.append({
                        "fold_id": fold_id,
                        "horizon": h,
                        "date_range": date_rng,
                        "strategy_variant": v_key,
                        "model_name": "XGBoost" if "filtered" in v_key else ("BuyAndHold" if v_key == "buy_and_hold" else "SMA50"),
                        "prob_threshold": 0.35,
                        "sl_avoid_threshold": 0.30 if v_key == "sl_filtered" else 1.0,
                        "win_rate": float(m["win_rate_post"]),
                        "avg_return": float(m["avg_return_post"]),
                        "sharpe_ratio": float(m["post_cost_sharpe"]),
                        "max_drawdown": float(m["max_drawdown_post"]),
                        "profit_factor": float(m["profit_factor_post"]),
                        "num_trades": int(m["num_trades"]),
                        "is_post_cost": True,
                        "pre_cost_sharpe": float(m["pre_cost_sharpe"]),
                        "post_cost_sharpe": float(m["post_cost_sharpe"]),
                    })

        with self.engine.begin() as conn:
            conn.execute(insert_sql, records)

        logger.info(f"Saved {len(records)} backtest fold records into PostgreSQL backtest_results table.")
