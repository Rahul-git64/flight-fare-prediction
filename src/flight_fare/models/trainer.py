"""Model training, cross-evaluation, and artifact persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from flight_fare.core.config import AppConfig
from flight_fare.core.exceptions import ModelArtifactError
from flight_fare.core.logger import get_logger, log_event

logger = get_logger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate RMSE, MAE, and R2 regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def get_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Return candidate regression estimators for model selection."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=1.0, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=random_state),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=12, random_state=random_state),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=140,
            learning_rate=0.08,
            max_depth=5,
            random_state=random_state,
        ),
    }


class ModelTrainer:
    """Trains candidate regressors, selects the best model, and saves artifacts."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def train_and_evaluate(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Train all candidate models, evaluate on holdout set, and persist the winner."""
        candidates = get_candidate_models(self.config.data.random_state)
        leaderboard: dict[str, dict[str, float]] = {}
        trained_estimators: dict[str, Any] = {}

        for name, estimator in candidates.items():
            estimator.fit(x_train, y_train)
            preds = estimator.predict(x_test)
            metrics = compute_regression_metrics(y_test, preds)
            leaderboard[name] = metrics
            trained_estimators[name] = estimator
            log_event(logger, "candidate_model_evaluated", model_name=name, **metrics)

        best_name = max(leaderboard, key=lambda k: leaderboard[k]["r2"])
        best_metrics = leaderboard[best_name]
        best_estimator = trained_estimators[best_name]

        if best_metrics["r2"] < self.config.model.r2_threshold:
            raise ModelArtifactError(
                f"Best model '{best_name}' R2 ({best_metrics['r2']}) is below threshold "
                f"({self.config.model.r2_threshold}).",
                details={"leaderboard": leaderboard},
            )

        residuals = np.abs(y_test - best_estimator.predict(x_test))
        residual_q90 = float(np.quantile(residuals, 0.90))

        self.config.model.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_estimator, self.config.model.model_path)

        metadata: dict[str, Any] = {
            "model_name": best_name,
            "model_version": self.config.version,
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": best_metrics,
            "residual_q90": round(residual_q90, 2),
            "train_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "leaderboard": leaderboard,
        }
        self.config.model.metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        log_event(
            logger,
            "best_model_persisted",
            best_model=best_name,
            r2=best_metrics["r2"],
            rmse=best_metrics["rmse"],
            model_path=str(self.config.model.model_path),
        )
        return metadata
