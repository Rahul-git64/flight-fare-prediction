"""Pydantic v2 schemas for flight fare inference and drift supervision."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FlightPredictionRequest(BaseModel):
    """Input schema for single-itinerary flight fare prediction."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    airline: Literal["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"] = Field(
        ...,
        description="Operating domestic carrier.",
    )
    source_city: Literal["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"] = Field(
        ...,
        description="Origin airport city.",
    )
    departure_time: Literal[
        "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
    ] = Field(
        ...,
        description="Scheduled departure time window.",
    )
    stops: Literal["zero", "one", "two_or_more"] = Field(
        ...,
        description="Number of layover stops.",
    )
    arrival_time: Literal[
        "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
    ] = Field(
        ...,
        description="Scheduled arrival time window.",
    )
    destination_city: Literal[
        "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai", "Delhi"
    ] = Field(
        ...,
        description="Destination airport city.",
    )
    travel_class: Literal["Economy", "Business"] = Field(
        ...,
        alias="classs",
        description="Cabin class (accepts 'travel_class' or legacy 'classs').",
    )
    duration: float = Field(
        ...,
        gt=0.4,
        le=48.0,
        description="Total flight duration in hours.",
    )
    days_left: int = Field(
        ...,
        ge=1,
        le=365,
        description="Days remaining before departure date.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_travel_class_key(cls, data: object) -> object:
        if isinstance(data, dict):
            if "travel_class" in data and "classs" not in data:
                data = {**data, "classs": data["travel_class"]}
            elif "class" in data and "classs" not in data and "travel_class" not in data:
                data = {**data, "classs": data["class"]}
        return data

    @model_validator(mode="after")
    def check_distinct_cities(self) -> FlightPredictionRequest:
        if self.source_city == self.destination_city:
            raise ValueError("source_city and destination_city must be different.")
        return self


class PredictionInterval(BaseModel):
    lower: float
    upper: float


class FlightPredictionResponse(BaseModel):
    """Response schema for `/predict` endpoint."""

    predicted_price_inr: float
    prediction_interval_inr: PredictionInterval
    confidence_score: float
    model_name: str
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Response schema for `/health` endpoint."""

    status: str
    service: str
    version: str
    uptime_seconds: float
    active_model: str
    model_metrics: dict[str, float]
    buffered_requests: int


class NumericalDriftMetric(BaseModel):
    ks_statistic: float
    ks_p_value: float
    psi: float
    mean_reference: float
    mean_current: float
    drift_detected: bool


class CategoricalDriftMetric(BaseModel):
    l1_divergence: float
    drift_detected: bool


class DriftReportResponse(BaseModel):
    """Response schema for `/monitor/drift` supervision endpoint."""

    status: str
    overall_drift_detected: bool
    sample_size: int
    reference_size: int
    numerical_features: dict[str, NumericalDriftMetric]
    categorical_features: dict[str, CategoricalDriftMetric]
