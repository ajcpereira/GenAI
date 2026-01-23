import os
import logging
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger


def configure_mcp_logging(level: str) -> None:
    os.makedirs("logs", exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level.upper())

    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)

    fh = RotatingFileHandler("logs/mcp-host.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logging.getLogger("uvicorn").setLevel(level.upper())
    logging.getLogger("uvicorn.error").setLevel(level.upper())
    logging.getLogger("uvicorn.access").setLevel(level.upper())
