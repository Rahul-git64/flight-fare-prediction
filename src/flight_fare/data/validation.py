"""Dataset schema and domain constraint validation."""

from __future__ import annotations

import pandas as pd

from flight_fare.core.config import FeaturesConfig
from flight_fare.core.exceptions import DataValidationError

LEGACY_COLUMN_ALIASES: dict[str, str] = {
    "classs": "travel_class",
    "class": "travel_class",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy column names and drop index artifacts."""
    cleaned = df.copy()
    drop_candidates = [col for col in ("Unnamed: 0", "flight") if col in cleaned.columns]
    if drop_candidates:
        cleaned = cleaned.drop(columns=drop_candidates)

    rename_map = {
        col: LEGACY_COLUMN_ALIASES[col]
        for col in cleaned.columns
        if col in LEGACY_COLUMN_ALIASES and LEGACY_COLUMN_ALIASES[col] not in cleaned.columns
    }
    if rename_map:
        cleaned = cleaned.rename(columns=rename_map)
    return cleaned


def validate_dataframe(
    df: pd.DataFrame,
    features_cfg: FeaturesConfig,
    require_target: bool = True,
) -> pd.DataFrame:
    """Validate column presence, types, and numerical bounds."""
    if df.empty:
        raise DataValidationError("Input dataframe is empty.")

    normalized = normalize_columns(df)
    expected_cols = list(features_cfg.numerical_columns) + list(features_cfg.categorical_columns)
    if require_target:
        expected_cols.append(features_cfg.target_column)

    missing_cols = [col for col in expected_cols if col not in normalized.columns]
    if missing_cols:
        raise DataValidationError(
            f"Missing required columns: {missing_cols}",
            details={"missing_columns": missing_cols},
        )

    for num_col in features_cfg.numerical_columns:
        coerced = pd.to_numeric(normalized[num_col], errors="coerce")
        if coerced.isna().all():
            raise DataValidationError(f"Numerical column '{num_col}' contains no valid numbers.")
        normalized[num_col] = coerced

    if (normalized["duration"].dropna() <= 0).any():
        raise DataValidationError("Flight duration must be strictly positive.")

    if (normalized["days_left"].dropna() < 1).any():
        raise DataValidationError("days_left must be at least 1.")

    if require_target:
        target = pd.to_numeric(normalized[features_cfg.target_column], errors="coerce")
        if target.isna().any() or (target <= 0).any():
            raise DataValidationError("Target column 'price' must contain positive numeric values.")
        normalized[features_cfg.target_column] = target

    return normalized[expected_cols]
