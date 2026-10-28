"""Logging configuration for entry-point scripts.

Library modules only create a logger with ``logging.getLogger(__name__)``.
Scripts call :func:`setup_logging` once, at start-up, to decide the format,
the minimum level and where log records go.
"""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configures the root logger to write to the console and, optionally, a file.

    Args:
        level (int | str, optional): Minimum level to show, e.g. ``logging.DEBUG`` or "DEBUG".
            Defaults to ``logging.INFO``.
        log_file (str | Path | None, optional): If set, records are also appended to this file.
            Defaults to None.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers, force=True
    )
