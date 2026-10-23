"""Data ingestion and deterministic synthetic flight dataset generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from flight_fare.core.config import AppConfig
from flight_fare.core.exceptions import DataValidationError
from flight_fare.core.logger import get_logger, log_event
from flight_fare.data.validation import validate_dataframe

logger = get_logger(__name__)

AIRLINE_BASE_FARE: dict[str, float] = {
    "SpiceJet": 3600.0,
    "AirAsia": 3300.0,
    "GO_FIRST": 3500.0,
    "Indigo": 4100.0,
    "Air_India": 5400.0,
    "Vistara": 6200.0,
}

STOPS_MULTIPLIER: dict[str, float] = {
    "zero": 1.0,
    "one": 1.24,
    "two_or_more": 1.42,
}

CITY_PAIR_FACTOR: dict[str, float] = {
    "Delhi": 1.05,
    "Mumbai": 1.08,
    "Bangalore": 1.06,
    "Kolkata": 1.02,
    "Hyderabad": 0.98,
    "Chennai": 1.00,
}


def generate_synthetic_flights(
    features_cfg: AppConfig,
    n_rows: int = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic flight fare dataset matching the Indian domestic aviation schema."""
    rng = np.random.default_rng(random_state)
    cats = features_cfg.features.categories

    airlines = rng.choice(cats["airline"], size=n_rows, p=[0.12, 0.10, 0.26, 0.10, 0.24, 0.18])
    source_cities = rng.choice(cats["source_city"], size=n_rows)
    destination_cities = []
    for src in source_cities:
        choices = [c for c in cats["destination_city"] if c != src]
        destination_cities.append(rng.choice(choices))
    destination_cities_arr = np.array(destination_cities)

    departure_times = rng.choice(cats["departure_time"], size=n_rows)
    arrival_times = rng.choice(cats["arrival_time"], size=n_rows)
    stops = rng.choice(cats["stops"], size=n_rows, p=[0.28, 0.58, 0.14])
    travel_classes = rng.choice(cats["travel_class"], size=n_rows, p=[0.74, 0.26])

    base_durations = np.where(
        stops == "zero",
        rng.uniform(1.25, 3.2, size=n_rows),
        np.where(
            stops == "one",
            rng.uniform(4.0, 12.5, size=n_rows),
            rng.uniform(10.0, 24.5, size=n_rows),
        ),
    )
    durations = np.round(base_durations, 2)
    days_left = rng.integers(1, 50, size=n_rows)

    prices = np.zeros(n_rows, dtype=float)
    for idx in range(n_rows):
        base = AIRLINE_BASE_FARE[airlines[idx]]
        stop_mult = STOPS_MULTIPLIER[stops[idx]]
        city_mult = 0.5 * (
            CITY_PAIR_FACTOR[source_cities[idx]] + CITY_PAIR_FACTOR[destination_cities_arr[idx]]
        )
        class_mult = 4.65 if travel_classes[idx] == "Business" else 1.0
        urgency_premium = 1.0 + 0.85 * np.exp(-days_left[idx] / 10.5)
        duration_component = durations[idx] * 215.0
        noise = rng.normal(0.0, 260.0)
        fare = (base * stop_mult * city_mult + duration_component) * class_mult * urgency_premium + noise
        prices[idx] = max(1450.0, round(float(fare), 2))

    return pd.DataFrame(
        {
            "airline": airlines,
            "source_city": source_cities,
            "departure_time": departure_times,
            "stops": stops,
            "arrival_time": arrival_times,
            "destination_city": destination_cities_arr,
            "travel_class": travel_classes,
            "duration": durations,
            "days_left": days_left,
            "price": prices,
        }
    )


class DataIngestion:
    """Handles dataset loading, schema validation, and train/test splitting."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self, source_csv: Path | None = None) -> tuple[Path, Path]:
        """Ingest raw flight data (or synthesize if unavailable) and write train/test splits."""
        raw_path = source_csv or self.config.data.raw_data_path
        if raw_path.exists():
            log_event(logger, "loading_existing_dataset", path=str(raw_path))
            raw_df = pd.read_csv(raw_path)
        else:
            log_event(
                logger,
                "generating_synthetic_dataset",
                rows=self.config.data.synthetic_rows,
                target_path=str(self.config.data.raw_data_path),
            )
            raw_df = generate_synthetic_flights(
                self.config,
                n_rows=self.config.data.synthetic_rows,
                random_state=self.config.data.random_state,
            )
            self.config.data.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(self.config.data.raw_data_path, index=False)

        validated_df = validate_dataframe(raw_df, self.config.features, require_target=True)
        if len(validated_df) < 10:
            raise DataValidationError("Dataset must contain at least 10 rows for splitting.")

        train_df, test_df = train_test_split(
            validated_df,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
        )

        self.config.data.train_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.data.test_data_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self.config.data.train_data_path, index=False)
        test_df.to_csv(self.config.data.test_data_path, index=False)

        log_event(
            logger,
            "data_ingestion_complete",
            train_rows=len(train_df),
            test_rows=len(test_df),
        )
        return self.config.data.train_data_path, self.config.data.test_data_path
