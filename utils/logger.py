from __future__ import annotations

import logging

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None  # type: ignore[assignment]


LOGGER_NAME = "glant"


def get_logger() -> logging.Logger:
    """Return the shared project logger configured with Rich."""
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    if RichHandler is None:
        handler: logging.Handler = logging.StreamHandler()
    else:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger


logger = get_logger()
