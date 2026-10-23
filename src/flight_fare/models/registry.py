"""Model registry loader and runtime inference orchestrator."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flight_fare.core.config import AppConfig, load_config
from flight_fare.core.exceptions import InferenceError, ModelArtifactError
from flight_fare.core.logger import get_logger, log_event
from flight_fare.data.ingestion import DataIngestion
from flight_fare.data.preprocessing import DataTransformer
from flight_fare.data.validation import validate_dataframe
from flight_fare.models.trainer import ModelTrainer

logger = get_logger(__name__)


def run_training_pipeline(config: AppConfig | None = None) -> dict[str, Any]:
    """Execute end-to-end data ingestion, preprocessing, and model training."""
    cfg = config or load_config()
    ingestion = DataIngestion(cfg)
    train_path, test_path = ingestion.run()

    transformer = DataTransformer(cfg)
    x_train, y_train, x_test, y_test = transformer.fit_transform_splits(train_path, test_path)

    trainer = ModelTrainer(cfg)
    return trainer.train_and_evaluate(x_train, y_train, x_test, y_test)


class ModelRegistry:
    """Loads persisted model artifacts and executes validated predictions."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.preprocessor: Any = None
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self.reference_stats: dict[str, Any] = {}

    def ensure_loaded(self, auto_train: bool = True) -> None:
        """Load model, preprocessor, metadata, and reference stats from disk."""
        model_cfg = self.config.model
        required_files = [
            model_cfg.preprocessor_path,
            model_cfg.model_path,
            model_cfg.metadata_path,
            model_cfg.reference_stats_path,
        ]
        if not all(path.exists() for path in required_files):
            if not auto_train:
                raise ModelArtifactError("Model artifacts are missing and auto_train is disabled.")
            log_event(logger, "bootstrapping_missing_model_artifacts")
            run_training_pipeline(self.config)

        try:
            self.preprocessor = joblib.load(model_cfg.preprocessor_path)
            self.model = joblib.load(model_cfg.model_path)
            self.metadata = json.loads(model_cfg.metadata_path.read_text(encoding="utf-8"))
            self.reference_stats = json.loads(
                model_cfg.reference_stats_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ModelArtifactError(f"Failed to load model artifacts: {exc}") from exc

    def predict_single(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate a single flight itinerary and return fare prediction with confidence bounds."""
        if self.model is None or self.preprocessor is None:
            self.ensure_loaded(auto_train=True)

        try:
            input_df = pd.DataFrame([payload])
            validated_df = validate_dataframe(
                input_df,
                self.config.features,
                require_target=False,
            )
            features = self.preprocessor.transform(validated_df)
            raw_pred = float(self.model.predict(features)[0])
            predicted_fare = max(500.0, round(raw_pred, 2))

            margin = float(self.metadata.get("residual_q90", predicted_fare * 0.12))
            lower_bound = max(500.0, round(predicted_fare - margin, 2))
            upper_bound = round(predicted_fare + margin, 2)

            if hasattr(self.model, "estimators_"):
                tree_preds = np.array(
                    [ float(est.predict(features)[0]) for est in self.model.estimators_[:30] ]
                    if isinstance(self.model.estimators_[0], (list, np.ndarray)) is False
                    else [predicted_fare]
                )
                cv = float(np.std(tree_preds) / max(predicted_fare, 1.0))
                confidence = round(float(np.clip(1.0 - cv, 0.65, 0.98)), 4)
            else:
                r2_base = float(self.metadata.get("metrics", {}).get("r2", 0.85))
                confidence = round(float(np.clip(r2_base, 0.65, 0.98)), 4)

            return {
                "predicted_price_inr": predicted_fare,
                "prediction_interval_inr": {"lower": lower_bound, "upper": upper_bound},
                "confidence_score": confidence,
                "model_name": self.metadata.get("model_name", "unknown"),
                "model_version": self.metadata.get("model_version", self.config.version),
            }
        except Exception as exc:
            if isinstance(exc, (InferenceError, ModelArtifactError)):
                raise
            raise InferenceError(f"Inference failed: {exc}") from exc
