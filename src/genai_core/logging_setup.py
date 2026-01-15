import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = "./logs/core.log") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    handlers.append(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        handlers.append(fh)

    root = logging.getLogger()
    root.setLevel(lvl)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
