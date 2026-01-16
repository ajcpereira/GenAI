"""Logging setup.

Enterprise goals:
  - Log rotation (size/time/none)
  - Correlation ID propagation into every log line
  - Single entrypoint to configure root handlers
"""

import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from .utils.log_context import get_correlation_id


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


def _build_file_handler(
    *,
    log_file: str,
    level: int,
    rotation_type: str = "size",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    when: str = "midnight",
    interval: int = 1,
):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    rotation_type = (rotation_type or "size").lower()
    if rotation_type == "none":
        # Fall back to a non-rotating FileHandler.
        fh = logging.FileHandler(log_file, encoding="utf-8")
    elif rotation_type == "time":
        fh = TimedRotatingFileHandler(
            log_file,
            when=when,
            interval=int(interval),
            backupCount=int(backup_count),
            encoding="utf-8",
            utc=True,
        )
    else:
        fh = RotatingFileHandler(
            log_file,
            maxBytes=int(max_bytes),
            backupCount=int(backup_count),
            encoding="utf-8",
        )

    fh.setLevel(level)
    return fh


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = "./logs/core.log",
    rotation: Optional[dict] = None,
) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s cid=%(correlation_id)s %(message)s"
    )

    handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    ch.addFilter(CorrelationIdFilter())
    handlers.append(ch)

    if log_file:
        rotation = rotation or {}
        fh = _build_file_handler(
            log_file=log_file,
            level=lvl,
            rotation_type=rotation.get("type", "size"),
            max_bytes=rotation.get("max_bytes", 10 * 1024 * 1024),
            backup_count=rotation.get("backup_count", 5),
            when=rotation.get("when", "midnight"),
            interval=rotation.get("interval", 1),
        )
        fh.setFormatter(fmt)
        fh.addFilter(CorrelationIdFilter())
        handlers.append(fh)

    root = logging.getLogger()
    root.setLevel(lvl)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
