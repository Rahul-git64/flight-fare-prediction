"""Unit tests for model training, evaluation, and registry inference."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from flight_fare.core.config import load_config
from flight_fare.models.registry import ModelRegistry, run_training_pipeline


def test_training_pipeline_and_registry_prediction(tmp_path: Path) -> None:
    base_cfg = load_config()
    test_cfg = replace(
        base_cfg,
        data=replace(
            base_cfg.data,
            raw_data_path=tmp_path / "data" / "raw.csv",
            train_data_path=tmp_path / "data" / "train.csv",
            test_data_path=tmp_path / "data" / "test.csv",
            synthetic_rows=320,
        ),
        model=replace(
            base_cfg.model,
            artifacts_dir=tmp_path / "models",
            r2_threshold=0.70,
        ),
    )

    metadata = run_training_pipeline(test_cfg)
    assert metadata["metrics"]["r2"] >= 0.70
    assert test_cfg.model.model_path.exists()
    assert test_cfg.model.metadata_path.exists()

    registry = ModelRegistry(test_cfg)
    registry.ensure_loaded(auto_train=False)

    result = registry.predict_single(
        {
            "airline": "Vistara",
            "source_city": "Delhi",
            "departure_time": "Morning",
            "stops": "zero",
            "arrival_time": "Afternoon",
            "destination_city": "Mumbai",
            "travel_class": "Business",
            "duration": 2.15,
            "days_left": 14,
        }
    )
    assert result["predicted_price_inr"] > 1000.0
    assert 0.5 <= result["confidence_score"] <= 1.0
    assert (
        result["prediction_interval_inr"]["lower"]
        <= result["predicted_price_inr"]
        <= result["prediction_interval_inr"]["upper"]
    )
