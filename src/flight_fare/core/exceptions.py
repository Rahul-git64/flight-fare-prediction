"""Custom exception hierarchy for the flight fare prediction service."""

from __future__ import annotations


class FlightFareError(Exception):
    """Base exception for all domain errors in flight-fare-prediction."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(FlightFareError):
    """Raised when configuration files are missing or invalid."""


class DataValidationError(FlightFareError):
    """Raised when input data violates expected schema or domain constraints."""


class ModelArtifactError(FlightFareError):
    """Raised when model or preprocessor artifacts cannot be loaded or trained."""


class InferenceError(FlightFareError):
    """Raised when prediction execution fails at runtime."""
