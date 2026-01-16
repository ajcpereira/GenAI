from __future__ import annotations

import contextvars


correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")


def set_correlation_id(cid: str) -> None:
    correlation_id_var.set(cid or "")


def get_correlation_id() -> str:
    return correlation_id_var.get()
