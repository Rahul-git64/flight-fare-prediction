"""Structured JSON and console logger for training and inference."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        extra_fields = getattr(record, "structured_data", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "flight_fare", level: int = logging.INFO) -> logging.Logger:
    """Return a configured structured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, **kwargs: Any) -> None:
    """Emit a structured log record with key-value metadata."""
    logger.info(message, extra={"structured_data": kwargs})
