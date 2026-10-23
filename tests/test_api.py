"""Integration tests for FastAPI `/health`, `/predict`, and `/monitor/drift` endpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flight_fare.api.main import create_app
from flight_fare.core.config import load_config


@pytest.fixture()
def api_client(tmp_path: Path) -> TestClient:
    base_cfg = load_config()
    test_cfg = replace(
        base_cfg,
        data=replace(
            base_cfg.data,
            raw_data_path=tmp_path / "data" / "raw.csv",
            train_data_path=tmp_path / "data" / "train.csv",
            test_data_path=tmp_path / "data" / "test.csv",
            synthetic_rows=260,
        ),
        model=replace(
            base_cfg.model,
            artifacts_dir=tmp_path / "models",
            r2_threshold=0.70,
        ),
    )
    app = create_app(test_cfg)
    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "flight-fare-prediction"
    assert body["active_model"] != "unknown"
    assert "r2" in body["model_metrics"]


def test_predict_endpoint_and_legacy_alias(api_client: TestClient) -> None:
    payload = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "zero",
        "arrival_time": "Afternoon",
        "destination_city": "Mumbai",
        "classs": "Economy",
        "duration": 2.15,
        "days_left": 20,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price_inr"] > 1000.0
    assert body["latency_ms"] >= 0.0
    assert 0.5 <= body["confidence_score"] <= 1.0


def test_predict_rejects_identical_cities(api_client: TestClient) -> None:
    payload = {
        "airline": "Indigo",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "zero",
        "arrival_time": "Afternoon",
        "destination_city": "Delhi",
        "travel_class": "Economy",
        "duration": 2.0,
        "days_left": 10,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_drift_monitoring_progression(api_client: TestClient) -> None:
    initial_drift = api_client.get("/monitor/drift").json()
    assert initial_drift["status"] == "insufficient_data"

    sample_requests = [
        {
            "airline": "Indigo",
            "source_city": "Mumbai",
            "departure_time": "Evening",
            "stops": "zero",
            "arrival_time": "Night",
            "destination_city": "Bangalore",
            "travel_class": "Economy",
            "duration": 1.85 + 0.1 * idx,
            "days_left": 10 + idx,
        }
        for idx in range(6)
    ]
    for req in sample_requests:
        res = api_client.post("/predict", json=req)
        assert res.status_code == 200

    drift_after = api_client.get("/monitor/drift")
    assert drift_after.status_code == 200
    report = drift_after.json()
    assert report["sample_size"] == 6
    assert "duration" in report["numerical_features"]
    assert "airline" in report["categorical_features"]
