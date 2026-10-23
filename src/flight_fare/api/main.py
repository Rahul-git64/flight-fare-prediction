"""FastAPI application with inference and data drift supervision endpoints."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from flight_fare.api.monitoring import DriftSupervisor
from flight_fare.api.schemas import (
    DriftReportResponse,
    FlightPredictionRequest,
    FlightPredictionResponse,
    HealthResponse,
)
from flight_fare.core.config import AppConfig, load_config
from flight_fare.core.exceptions import FlightFareError
from flight_fare.core.logger import get_logger, log_event
from flight_fare.models.registry import ModelRegistry

logger = get_logger(__name__)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI inference and supervision application."""
    cfg = config or load_config()
    registry = ModelRegistry(cfg)
    supervisor = DriftSupervisor(cfg)
    start_time = time.monotonic()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        log_event(logger, "service_startup_initiated", version=cfg.version)
        registry.ensure_loaded(auto_train=True)
        log_event(
            logger,
            "service_startup_complete",
            active_model=registry.metadata.get("model_name", "unknown"),
        )
        yield

    app = FastAPI(
        title="Flight Fare Prediction Service",
        description="Production API for domestic flight fare estimation with statistical drift supervision.",
        version=cfg.version,
        lifespan=lifespan,
    )

    app.state.config = cfg
    app.state.registry = registry
    app.state.supervisor = supervisor
    app.state.start_time = start_time

    @app.get("/health", response_model=HealthResponse, tags=["Supervision"])
    def health_check() -> HealthResponse:
        """Return service health, uptime, and active model metadata."""
        app.state.registry.ensure_loaded(auto_train=True)
        uptime = round(time.monotonic() - app.state.start_time, 2)
        meta = app.state.registry.metadata
        metrics = {
            k: float(v)
            for k, v in meta.get("metrics", {}).items()
            if isinstance(v, (int, float))
        }
        return HealthResponse(
            status="healthy",
            service=cfg.project_name,
            version=cfg.version,
            uptime_seconds=uptime,
            active_model=str(meta.get("model_name", "unknown")),
            model_metrics=metrics,
            buffered_requests=len(app.state.supervisor.buffer),
        )

    @app.post("/predict", response_model=FlightPredictionResponse, tags=["Inference"])
    def predict_fare(request: FlightPredictionRequest) -> FlightPredictionResponse:
        """Validate flight itinerary, execute inference, log latency, and update drift window."""
        t0 = time.perf_counter()
        payload = {
            "airline": request.airline,
            "source_city": request.source_city,
            "departure_time": request.departure_time,
            "stops": request.stops,
            "arrival_time": request.arrival_time,
            "destination_city": request.destination_city,
            "travel_class": request.travel_class,
            "duration": request.duration,
            "days_left": request.days_left,
        }
        try:
            prediction = app.state.registry.predict_single(payload)
        except FlightFareError as exc:
            log_event(logger, "inference_failure", error=exc.message)
            raise HTTPException(status_code=422, detail=exc.message) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        app.state.supervisor.record_request(payload)

        log_event(
            logger,
            "prediction_completed",
            airline=request.airline,
            source_city=request.source_city,
            destination_city=request.destination_city,
            travel_class=request.travel_class,
            predicted_price_inr=prediction["predicted_price_inr"],
            confidence_score=prediction["confidence_score"],
            latency_ms=latency_ms,
        )

        return FlightPredictionResponse(
            predicted_price_inr=prediction["predicted_price_inr"],
            prediction_interval_inr=prediction["prediction_interval_inr"],
            confidence_score=prediction["confidence_score"],
            model_name=prediction["model_name"],
            model_version=prediction["model_version"],
            latency_ms=latency_ms,
        )

    @app.get("/monitor/drift", response_model=DriftReportResponse, tags=["Supervision"])
    def monitor_drift() -> DriftReportResponse:
        """Evaluate feature distribution drift on recent inference requests."""
        app.state.registry.ensure_loaded(auto_train=True)
        report = app.state.supervisor.evaluate(app.state.registry.reference_stats)
        log_event(
            logger,
            "drift_evaluation_executed",
            status=report["status"],
            sample_size=report["sample_size"],
            overall_drift_detected=report["overall_drift_detected"],
        )
        return DriftReportResponse(**report)

    return app


app = create_app()
