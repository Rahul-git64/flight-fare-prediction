"""Unit tests for configuration, data validation, ingestion, and preprocessing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_fare.core.config import get_project_root, load_config
from flight_fare.core.exceptions import DataValidationError
from flight_fare.data.ingestion import DataIngestion
from flight_fare.data.preprocessing import DataTransformer, build_preprocessor
from flight_fare.data.validation import normalize_columns, validate_dataframe


@pytest.fixture()
def sample_fixture_path() -> Path:
    return get_project_root() / "tests" / "fixtures" / "sample_flights.csv"


def test_load_config_resolves_paths() -> None:
    cfg = load_config()
    assert cfg.project_name == "flight-fare-prediction"
    assert "duration" in cfg.features.numerical_columns
    assert "travel_class" in cfg.features.categorical_columns


def test_normalize_legacy_classs_column(sample_fixture_path: Path) -> None:
    raw = pd.read_csv(sample_fixture_path).rename(columns={"travel_class": "classs"})
    raw["Unnamed: 0"] = range(len(raw))
    cleaned = normalize_columns(raw)
    assert "travel_class" in cleaned.columns
    assert "classs" not in cleaned.columns
    assert "Unnamed: 0" not in cleaned.columns


def test_validate_dataframe_rejects_invalid_duration(sample_fixture_path: Path) -> None:
    cfg = load_config()
    df = pd.read_csv(sample_fixture_path)
    df.loc[0, "duration"] = -1.5
    with pytest.raises(DataValidationError, match="strictly positive"):
        validate_dataframe(df, cfg.features, require_target=True)


def test_preprocessor_handles_unseen_category(sample_fixture_path: Path) -> None:
    cfg = load_config()
    df = pd.read_csv(sample_fixture_path)
    feature_cols = cfg.features.numerical_columns + cfg.features.categorical_columns
    preprocessor = build_preprocessor(cfg)
    preprocessor.fit(df[feature_cols])

    unseen = df[feature_cols].iloc[:2].copy()
    unseen.loc[0, "airline"] = "AkasaAir_Unseen"
    transformed = preprocessor.transform(unseen)
    assert transformed.shape == (2, len(feature_cols))
    assert not np.isnan(transformed).any()


def test_end_to_end_ingestion_and_transformation(
    tmp_path: Path,
    sample_fixture_path: Path,
) -> None:
    base_cfg = load_config()
    test_cfg = replace(
        base_cfg,
        data=replace(
            base_cfg.data,
            raw_data_path=tmp_path / "raw.csv",
            train_data_path=tmp_path / "train.csv",
            test_data_path=tmp_path / "test.csv",
        ),
        model=replace(
            base_cfg.model,
            artifacts_dir=tmp_path / "models",
        ),
    )

    ingestion = DataIngestion(test_cfg)
    train_path, test_path = ingestion.run(source_csv=sample_fixture_path)
    assert train_path.exists()
    assert test_path.exists()

    transformer = DataTransformer(test_cfg)
    x_train, y_train, x_test, y_test = transformer.fit_transform_splits(train_path, test_path)
    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)
    assert test_cfg.model.preprocessor_path.exists()
    assert test_cfg.model.reference_stats_path.exists()
