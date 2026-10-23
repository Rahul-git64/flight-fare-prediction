"""Feature transformation pipeline and reference distribution calculator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from flight_fare.core.config import AppConfig
from flight_fare.core.logger import get_logger, log_event

logger = get_logger(__name__)


def build_preprocessor(config: AppConfig) -> ColumnTransformer:
    """Create the scikit-learn ColumnTransformer for numerical and categorical features."""
    num_cols = config.features.numerical_columns
    cat_cols = config.features.categorical_columns
    ordered_categories = [config.features.categories[col] for col in cat_cols]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal_encoder", OrdinalEncoder(categories=ordered_categories)),
            ("scaler", MinMaxScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )


def compute_reference_statistics(
    df: pd.DataFrame,
    config: AppConfig,
    max_sample_size: int = 500,
) -> dict[str, Any]:
    """Compute reference baseline distributions for numerical and categorical features."""
    stats: dict[str, Any] = {"numerical": {}, "categorical": {}, "n_reference": len(df)}

    for col in config.features.numerical_columns:
        series = df[col].dropna().astype(float)
        sampled = series.sample(
            n=min(len(series), max_sample_size),
            random_state=config.data.random_state,
        )
        quantiles = np.quantile(series, np.linspace(0.0, 1.0, 11)).tolist()
        stats["numerical"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1) if len(series) > 1 else 1.0),
            "min": float(series.min()),
            "max": float(series.max()),
            "quantiles": [float(q) for q in quantiles],
            "sample": [float(val) for val in sampled.tolist()],
        }

    for col in config.features.categorical_columns:
        counts = df[col].astype(str).value_counts(normalize=True).to_dict()
        expected_cats = config.features.categories[col]
        stats["categorical"][col] = {
            cat: float(counts.get(cat, 0.0)) for cat in expected_cats
        }

    return stats


class DataTransformer:
    """Fits the feature preprocessor, transforms datasets, and persists artifacts."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def fit_transform_splits(
        self,
        train_path: Path,
        test_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit preprocessor on train split, transform both splits, and save artifacts."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        target_col = self.config.features.target_column
        feature_cols = (
            self.config.features.numerical_columns + self.config.features.categorical_columns
        )

        x_train_df = train_df[feature_cols]
        y_train = train_df[target_col].to_numpy(dtype=float)

        x_test_df = test_df[feature_cols]
        y_test = test_df[target_col].to_numpy(dtype=float)

        preprocessor = build_preprocessor(self.config)
        x_train = preprocessor.fit_transform(x_train_df)
        x_test = preprocessor.transform(x_test_df)

        self.config.model.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, self.config.model.preprocessor_path)

        ref_stats = compute_reference_statistics(x_train_df, self.config)
        self.config.model.reference_stats_path.write_text(
            json.dumps(ref_stats, indent=2),
            encoding="utf-8",
        )

        log_event(
            logger,
            "feature_transformation_complete",
            train_shape=list(x_train.shape),
            test_shape=list(x_test.shape),
            preprocessor_path=str(self.config.model.preprocessor_path),
        )
        return x_train, y_train, x_test, y_test
