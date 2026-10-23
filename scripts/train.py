"""CLI entrypoint to run the training pipeline and export artifacts."""

from __future__ import annotations

import json

from flight_fare.core.config import load_config
from flight_fare.models.registry import run_training_pipeline


def main() -> None:
    config = load_config()
    metadata = run_training_pipeline(config)
    print(
        json.dumps(
            {
                "status": "completed",
                "model_name": metadata["model_name"],
                "metrics": metadata["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
