-- Schema for AI ETF Quantitative Research System (Phase 1)

CREATE TABLE IF NOT EXISTS instruments (
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(10) NOT NULL,
    segment VARCHAR(10) NOT NULL,
    trading_symbol VARCHAR(50) NOT NULL,
    groww_symbol VARCHAR(100) NOT NULL UNIQUE,
    instrument_type VARCHAR(20) NOT NULL,
    name VARCHAR(255),
    is_selected_universe BOOLEAN DEFAULT FALSE,
    avg_daily_volume NUMERIC(18, 2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_exchange_symbol UNIQUE (exchange, trading_symbol)
);

CREATE TABLE IF NOT EXISTS ohlcv_daily (
    id SERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(14, 4) NOT NULL,
    high NUMERIC(14, 4) NOT NULL,
    low NUMERIC(14, 4) NOT NULL,
    close NUMERIC(14, 4) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_instrument_timestamp UNIQUE (instrument_id, timestamp)
);

CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,

    -- Momentum / Returns
    return_1d NUMERIC(12, 6),
    return_3d NUMERIC(12, 6),
    return_5d NUMERIC(12, 6),
    return_10d NUMERIC(12, 6),
    return_20d NUMERIC(12, 6),

    -- Trend
    sma_5 NUMERIC(14, 4),
    sma_20 NUMERIC(14, 4),
    sma_50 NUMERIC(14, 4),
    price_vs_sma5 NUMERIC(12, 6),
    price_vs_sma20 NUMERIC(12, 6),
    price_vs_sma50 NUMERIC(12, 6),
    sma20_vs_sma50 NUMERIC(12, 6),

    -- Momentum Indicators
    rsi_14 NUMERIC(10, 4),
    macd NUMERIC(12, 6),
    macd_signal NUMERIC(12, 6),
    macd_hist NUMERIC(12, 6),
    roc_10 NUMERIC(12, 6),

    -- Volatility
    volatility_5d NUMERIC(12, 6),
    volatility_10d NUMERIC(12, 6),
    volatility_20d NUMERIC(12, 6),
    atr_14 NUMERIC(14, 4),

    -- Volume
    volume_today BIGINT,
    volume_sma20 NUMERIC(18, 2),
    volume_ratio NUMERIC(12, 6),
    volume_change NUMERIC(12, 6),

    -- Market Context
    nifty50_return_1d NUMERIC(12, 6),
    banknifty_return_1d NUMERIC(12, 6),
    india_vix_level NUMERIC(10, 4),
    india_vix_change_1d NUMERIC(12, 6),
    sector_return_1d NUMERIC(12, 6),
    commodity_return_1d NUMERIC(12, 6),

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_features_inst_ts UNIQUE (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_features_inst_ts ON features(instrument_id, timestamp DESC);

-- Phase 3 Tables: Market Regimes and Triple Barrier Labels

CREATE TABLE IF NOT EXISTS market_regimes (
    id SERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    instrument_regime VARCHAR(30) NOT NULL,
    market_regime VARCHAR(30) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_regimes_inst_ts UNIQUE (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_regimes_inst_ts ON market_regimes(instrument_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS triple_barrier_labels (
    id SERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    horizon INTEGER NOT NULL,
    barrier_hit VARCHAR(10),
    days_to_hit INTEGER,
    tp_level NUMERIC(14, 4),
    sl_level NUMERIC(14, 4),
    realized_return NUMERIC(12, 6),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_labels_inst_ts_hz UNIQUE (instrument_id, timestamp, horizon)
);

CREATE INDEX IF NOT EXISTS idx_labels_inst_ts_hz ON triple_barrier_labels(instrument_id, timestamp DESC, horizon);

-- Phase 4 Table: Model Validation Runs
CREATE TABLE IF NOT EXISTS model_runs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    horizon INTEGER NOT NULL,
    train_date_range VARCHAR(100) NOT NULL,
    val_date_range VARCHAR(100) NOT NULL,
    accuracy NUMERIC(8, 4) NOT NULL,
    per_class_metrics JSONB NOT NULL,
    confusion_matrix JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_runs_name_hz ON model_runs(model_name, horizon);

-- Phase 5 Table: Walk-Forward Backtest Results
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    fold_id INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    date_range VARCHAR(100) NOT NULL,
    strategy_variant VARCHAR(50) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prob_threshold NUMERIC(6, 4) NOT NULL,
    sl_avoid_threshold NUMERIC(6, 4),
    win_rate NUMERIC(8, 4) NOT NULL,
    avg_return NUMERIC(10, 6) NOT NULL,
    sharpe_ratio NUMERIC(8, 4) NOT NULL,
    max_drawdown NUMERIC(8, 4) NOT NULL,
    profit_factor NUMERIC(10, 4) NOT NULL,
    num_trades INTEGER NOT NULL,
    is_post_cost BOOLEAN NOT NULL,
    pre_cost_sharpe NUMERIC(8, 4),
    post_cost_sharpe NUMERIC(8, 4),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_backtest_results_variant ON backtest_results(strategy_variant, horizon, fold_id);




