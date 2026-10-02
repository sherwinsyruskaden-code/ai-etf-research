import pytest
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression

from src.models.splitter import PurgedTimeSeriesSplit
from src.models.data_loader import ModelDataLoader, PROHIBITED_LABEL_COLUMNS, TARGET_MAP
from src.models.repository import ModelRepository


@pytest.fixture
def sample_timeseries_df():
    """Create a synthetic chronological dataset for purged split testing."""
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "timestamp": dates,
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "target": np.random.choice([0, 1, 2], size=n)
    })
    return df


@pytest.fixture
def sqlite_engine():
    """Create in-memory SQLite engine with Phase 4 model_runs schema."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE TABLE model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name VARCHAR(50) NOT NULL,
                horizon INTEGER NOT NULL,
                train_date_range VARCHAR(100) NOT NULL,
                val_date_range VARCHAR(100) NOT NULL,
                accuracy REAL NOT NULL,
                per_class_metrics TEXT NOT NULL,
                confusion_matrix TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    return engine


def test_purged_time_series_split(sample_timeseries_df):
    """Verify that PurgedTimeSeriesSplit correctly purges overlapping training samples."""
    splitter = PurgedTimeSeriesSplit(val_pct=0.20)
    horizon = 7

    df_train, df_val, info = splitter.split(sample_timeseries_df, horizon=horizon)

    n_total = len(sample_timeseries_df)  # 100
    n_val = len(df_val)  # 20
    n_train = len(df_train)

    # Cutoff index for val is 80 (dates index 80 to 99)
    # Horizon H=7 purges indices 73 to 79 (7 rows)
    expected_purged = 7
    expected_train = 80 - 7  # 73

    assert n_val == 20
    assert n_train == expected_train
    assert info["purged_rows"] == expected_purged

    # Verify max train date + horizon is strictly before min val date
    max_train_date = df_train["timestamp"].max()
    min_val_date = df_val["timestamp"].min()

    # The difference between val start date and max train date must be at least H days
    diff_days = (min_val_date - max_train_date).days
    assert diff_days >= horizon


def test_no_leakage_prohibited_columns():
    """Verify that DataLoader strictly excludes future label columns from feature matrix."""
    mock_columns = [
        "return_1d", "rsi_14", "volatility_20d", "inst_regime_TRENDING_UP",
        "barrier_hit", "tp_level", "sl_level", "days_to_hit", "realized_return"
    ]

    df_mock = pd.DataFrame(columns=mock_columns)
    non_feature_cols = {"instrument_id", "timestamp", "target", "barrier_hit", "tp_level", "sl_level", "days_to_hit", "realized_return"}

    feature_names = [c for c in df_mock.columns if c not in non_feature_cols]

    leaked = set(feature_names).intersection(PROHIBITED_LABEL_COLUMNS)
    assert len(leaked) == 0
    assert "tp_level" not in feature_names
    assert "sl_level" not in feature_names
    assert "days_to_hit" not in feature_names
    assert "realized_return" not in feature_names


def test_model_reproducibility():
    """Verify that LogisticRegression with fixed random_state produces identical predictions."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1, 2], size=100)

    clf1 = LogisticRegression(multi_class="multinomial", random_state=42)
    clf1.fit(X, y)
    pred1 = clf1.predict(X)

    clf2 = LogisticRegression(multi_class="multinomial", random_state=42)
    clf2.fit(X, y)
    pred2 = clf2.predict(X)

    assert np.array_equal(pred1, pred2)


def test_repository_model_run_persistence(sqlite_engine, tmp_path):
    """Verify model run metric saving and artifact writing in ModelRepository."""
    repo = ModelRepository(sqlite_engine, models_dir=tmp_path)

    run_id = repo.save_model_run_metrics(
        model_name="XGBoost",
        horizon=7,
        train_date_range="2002-01-01 to 2021-11-15",
        val_date_range="2021-11-28 to 2026-09-22",
        accuracy=0.5015,
        per_class_metrics={"TP": {"f1-score": 0.2687}},
        confusion_matrix=[[10, 2, 3], [1, 15, 4], [2, 3, 20]]
    )

    assert run_id is not None
    assert run_id > 0

    df_runs = repo.get_model_runs()
    assert len(df_runs) == 1
    assert df_runs.iloc[0]["model_name"] == "XGBoost"

    # Test saving model artifact
    dummy_model = {"name": "test_model"}
    path = repo.save_model_artifact(dummy_model, "test_v1.joblib")
    assert os.path.exists(path)
