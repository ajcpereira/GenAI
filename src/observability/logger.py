import json
import logging
import uuid
from typing import Any

logging.basicConfig(level=logging.INFO)

def new_correlation_id() -> str:
    return str(uuid.uuid4())

def log(event: str, correlation_id: str | None = None, **data: Any) -> None:
    payload = {"event": event, **data}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    logging.info(json.dumps(payload, ensure_ascii=False))
