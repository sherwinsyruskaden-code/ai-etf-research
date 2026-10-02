import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sqlalchemy import Engine

from src.models.data_loader import ModelDataLoader, TARGET_MAP, CLASS_NAMES
from src.models.splitter import PurgedTimeSeriesSplit
from src.models.repository import ModelRepository

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates Phase 4 Model Training across Logistic Regression, Random Forest, and XGBoost.
    - Uses identical Purged & Embargoed Train/Validation splits per horizon.
    - Evaluates performance on held-out purged validation sets.
    - Evaluates 3-class target outcomes (TP / SL / TIME).
    - Extracts gain-based feature importances and coefficient magnitudes.
    - Persists model artifacts (.joblib) and database execution logs in PostgreSQL (model_runs).
    """

    def __init__(self, engine: Engine, seed: int = 42):
        self.engine = engine
        self.seed = seed
        self.data_loader = ModelDataLoader(engine)
        self.splitter = PurgedTimeSeriesSplit(val_pct=0.20)
        self.repo = ModelRepository(engine)

    def train_and_evaluate_all(
        self,
        horizons: List[int] = [7, 14],
        wipe_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Train and evaluate all 3 models across all horizons.

        Returns:
            Dict[str, Any]: Comprehensive summary of evaluation metrics and rankings.
        """
        logger.info("=" * 85)
        logger.info("STARTING PHASE 4 MODEL TRAINING & EVALUATION PIPELINE")
        logger.info("=" * 85)

        if wipe_existing:
            wiped = self.repo.clear_all_model_runs()
            logger.info(f"Wiped {wiped} old model run records before training.")

        summary = {
            "horizons": horizons,
            "models_trained": [],
            "horizon_results": {}
        }

        for h in horizons:
            logger.info(f"\n>>> PROCESSING FORECAST HORIZON: {h} DAYS <<<")
            
            # 1. Load Pooled Dataset
            df_full, feature_names = self.data_loader.load_dataset_for_horizon(horizon=h)
            if df_full.empty:
                logger.error(f"Dataset for horizon {h} is empty. Skipping.")
                continue

            # 2. Perform Purged & Embargoed Split
            df_train, df_val, split_info = self.splitter.split(df_full, horizon=h)

            X_train = df_train[feature_names]
            y_train = df_train["target"].values
            X_val = df_val[feature_names]
            y_val = df_val["target"].values

            # 3. Define Model Registry with SimpleImputer for baselines (handles explicit NULLs without fake zero-fill)
            models_to_train = {
                "LogisticRegression": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        multi_class="multinomial",
                        max_iter=1000,
                        random_state=self.seed
                    ))
                ]),
                "RandomForest": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=self.seed,
                        n_jobs=-1
                    ))
                ]),
                "XGBoost": XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    eval_metric="mlogloss",
                    random_state=self.seed,
                    n_jobs=-1
                )
            }

            horizon_res = {
                "split_info": split_info,
                "models": {}
            }

            for model_name, model_obj in models_to_train.items():
                logger.info(f"Training [{model_name}] on {len(X_train)} purged samples...")
                
                # Fit model
                model_obj.fit(X_train, y_train)

                # Predict on held-out validation set
                y_pred = model_obj.predict(X_val)

                # Compute Metrics
                acc = accuracy_score(y_val, y_pred)
                clf_report = classification_report(
                    y_val,
                    y_pred,
                    target_names=CLASS_NAMES,
                    output_dict=True,
                    zero_division=0
                )
                cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2]).tolist()

                # Extract Feature Importances
                feat_importances = self._extract_feature_importances(model_obj, model_name, feature_names)

                # Save Model Artifact (.joblib)
                artifact_filename = f"{model_name.lower()}_{h}d_v1.joblib"
                artifact_dict = {
                    "model": model_obj,
                    "model_name": model_name,
                    "horizon": h,
                    "feature_names": feature_names,
                    "target_map": TARGET_MAP,
                    "split_info": split_info
                }
                artifact_path = self.repo.save_model_artifact(artifact_dict, artifact_filename)

                # Save Metrics in PostgreSQL DB
                db_run_id = self.repo.save_model_run_metrics(
                    model_name=model_name,
                    horizon=h,
                    train_date_range=split_info["train_date_range"],
                    val_date_range=split_info["val_date_range"],
                    accuracy=acc,
                    per_class_metrics=clf_report,
                    confusion_matrix=cm
                )

                model_eval = {
                    "db_run_id": db_run_id,
                    "model_name": model_name,
                    "accuracy": acc,
                    "macro_f1": clf_report["macro avg"]["f1-score"],
                    "per_class_metrics": clf_report,
                    "confusion_matrix": cm,
                    "top_features": feat_importances[:10],
                    "artifact_path": artifact_path
                }
                horizon_res["models"][model_name] = model_eval
                summary["models_trained"].append(f"{model_name}_{h}d")

                logger.info(
                    f"[{model_name:18s} (H={h}d)] Val Accuracy: {acc:.4f} | "
                    f"Macro F1: {clf_report['macro avg']['f1-score']:.4f} | "
                    f"TP F1: {clf_report['TP']['f1-score']:.4f}, "
                    f"SL F1: {clf_report['SL']['f1-score']:.4f}, "
                    f"TIME F1: {clf_report['TIME']['f1-score']:.4f}"
                )

            summary["horizon_results"][h] = horizon_res

        self._log_summary_report(summary)
        return summary

    def _extract_feature_importances(
        self,
        model_obj: Any,
        model_name: str,
        feature_names: List[str]
    ) -> List[Tuple[str, float]]:
        """Extract sorted feature importances / coefficient magnitudes."""
        importances = []
        if model_name == "LogisticRegression":
            clf = model_obj.named_steps["clf"]
            # Average absolute coefficient across 3 classes
            avg_coef = np.mean(np.abs(clf.coef_), axis=0)
            for f_name, imp in zip(feature_names, avg_coef):
                importances.append((f_name, float(imp)))
        elif model_name == "RandomForest":
            clf = model_obj.named_steps["clf"]
            raw_imp = clf.feature_importances_
            for f_name, imp in zip(feature_names, raw_imp):
                importances.append((f_name, float(imp)))
        elif model_name == "XGBoost":
            raw_imp = model_obj.feature_importances_
            for f_name, imp in zip(feature_names, raw_imp):
                importances.append((f_name, float(imp)))
        else:
            for f_name in feature_names:
                importances.append((f_name, 0.0))

        importances.sort(key=lambda x: x[1], reverse=True)
        return importances

    def _log_summary_report(self, summary: Dict[str, Any]) -> None:
        """Log summary comparison across trained models."""
        logger.info("=" * 85)
        logger.info("PHASE 4 MODEL EVALUATION SUMMARY REPORT")
        logger.info("=" * 85)
        for h, res in summary["horizon_results"].items():
            logger.info(f"\n--- HORIZON {h} DAYS RESULTS ---")
            models = res["models"]
            for m_name, m_eval in models.items():
                logger.info(
                    f"  Model: {m_name:18s} | Accuracy: {m_eval['accuracy']:.4f} | "
                    f"Macro F1: {m_eval['macro_f1']:.4f} | "
                    f"Top Feature: {m_eval['top_features'][0][0]} ({m_eval['top_features'][0][1]:.4f})"
                )
        logger.info("=" * 85)
