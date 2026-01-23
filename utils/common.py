import os
import uuid
import json
import logging
import contextvars
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import yaml
from pythonjsonlogger import jsonlogger
from jsonschema import Draft202012Validator, RefResolver


# ---------------------------
# Request-scoped context
# ---------------------------
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("session_id", default=None)
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("span_id", default=None)


def set_request_context(
    *,
    request_id: Optional[str],
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> Tuple[contextvars.Token, contextvars.Token, contextvars.Token, contextvars.Token]:
    """Set request-scoped context for logging correlation."""
    t1 = _request_id_var.set(request_id)
    t2 = _session_id_var.set(session_id)
    t3 = _trace_id_var.set(trace_id)
    t4 = _span_id_var.set(span_id)
    return t1, t2, t3, t4


def reset_request_context(tokens: Tuple[contextvars.Token, contextvars.Token, contextvars.Token, contextvars.Token]) -> None:
    t1, t2, t3, t4 = tokens
    _request_id_var.reset(t1)
    _session_id_var.reset(t2)
    _trace_id_var.reset(t3)
    _span_id_var.reset(t4)


class RequestContextFilter(logging.Filter):
    """Inject request_id/session_id into log records when missing."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id") or record.request_id is None:
            record.request_id = _request_id_var.get()
        if not hasattr(record, "session_id") or record.session_id is None:
            record.session_id = _session_id_var.get()
        if not hasattr(record, "trace_id") or record.trace_id is None:
            record.trace_id = _trace_id_var.get()
        if not hasattr(record, "span_id") or record.span_id is None:
            record.span_id = _span_id_var.get()
        return True


# ---------------------------
# Helpers
# ---------------------------
def new_request_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Logging
    log_level = os.getenv("LOG_LEVEL")
    if log_level:
        cfg.setdefault("logging", {})["level"] = log_level
        cfg.setdefault("app", {})["log_level"] = log_level  # backward compatibility

    # Shared vLLM key
    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        cfg.setdefault("planner", {})["api_key"] = api_key
        cfg.setdefault("responder", {})["api_key"] = api_key

    # Endpoints (optional)
    vllm_base_url = os.getenv("VLLM_BASE_URL")
    if vllm_base_url:
        cfg.setdefault("planner", {})["vllm_base_url"] = vllm_base_url
        cfg.setdefault("responder", {})["vllm_base_url"] = vllm_base_url

    mcp_base_url = os.getenv("MCP_BASE_URL")
    if mcp_base_url:
        cfg.setdefault("mcp", {})["base_url"] = mcp_base_url

    return cfg


# ---------------------------
# Logging
# ---------------------------
def configure_logging(cfg: Dict[str, Any]) -> None:
    """Configure JSON logging according to config.yaml.

    Supported config keys:
      app.log_level (legacy)
      logging.level
      logging.stdout
      logging.log_dir
      logging.files.{orchestrator,api,planner,validator,executor,responder}
      logging.rotation.{max_bytes,backup_count,use_timed_rotation,when,interval,timed_backup_count}
    """
    app_cfg = cfg.get("app", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}

    level = str(log_cfg.get("level") or app_cfg.get("log_level") or "INFO").upper()
    log_dir = str(log_cfg.get("log_dir") or "logs")
    stdout_enabled = bool(log_cfg.get("stdout", True))

    files_cfg = log_cfg.get("files", {}) or {}
    rotation_cfg = log_cfg.get("rotation", {}) or {}
    max_bytes = int(rotation_cfg.get("max_bytes", 10 * 1024 * 1024))
    backup_count = int(rotation_cfg.get("backup_count", 5))

    use_timed = bool(rotation_cfg.get("use_timed_rotation", False))
    when = str(rotation_cfg.get("when", "D"))
    interval = int(rotation_cfg.get("interval", 1))
    timed_backup_count = int(rotation_cfg.get("timed_backup_count", 7))

    _ensure_dir(log_dir)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing root handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(session_id)s %(trace_id)s %(span_id)s"
    )

    ctx_filter = RequestContextFilter()

    # Console handler
    if stdout_enabled:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.addFilter(ctx_filter)
        root.addHandler(sh)

    def _build_file_handler(path: str) -> logging.Handler:
        if use_timed:
            fh = TimedRotatingFileHandler(
                path,
                when=when,
                interval=interval,
                backupCount=timed_backup_count,
                encoding="utf-8",
                utc=True,
            )
        else:
            fh = RotatingFileHandler(
                path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        fh.setFormatter(formatter)
        fh.addFilter(ctx_filter)
        return fh

    def _clear_logger_handlers(logger_name: str) -> None:
        lg = logging.getLogger(logger_name)
        for h in list(lg.handlers):
            lg.removeHandler(h)

    def add_file_handler(logger_name: str, filename: str) -> None:
        lg = logging.getLogger(logger_name)
        lg.setLevel(level)

        _clear_logger_handlers(logger_name)

        filepath = os.path.join(log_dir, filename)
        lg.addHandler(_build_file_handler(filepath))
        lg.propagate = True

    # Map component loggers -> config keys
    add_file_handler("genai.main", files_cfg.get("orchestrator", "orchestrator.log"))
    add_file_handler("genai.orchestrator", files_cfg.get("orchestrator", "orchestrator.log"))

    add_file_handler("genai.api", files_cfg.get("api", "api.log"))
    add_file_handler("genai.planner", files_cfg.get("planner", "planner.log"))
    add_file_handler("genai.validator", files_cfg.get("validator", "validator.log"))
    add_file_handler("genai.executor", files_cfg.get("executor", "executor.log"))
    add_file_handler("genai.responder", files_cfg.get("responder", "responder.log"))

    # Quiet noisy libs (still configurable via level)
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("httpx").setLevel(level)


# ---------------------------
# Contracts / schema validation
# ---------------------------
def load_contract_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_resolver(bundle: Dict[str, Any]) -> RefResolver:
    store = {}
    schemas = bundle.get("schemas", {})
    for _, schema in schemas.items():
        sid = schema.get("$id")
        if sid:
            store[sid] = schema
    return RefResolver.from_schema(schemas.get("Envelope", {}), store=store)


def validate_json(schema: Dict[str, Any], instance: Any, bundle: Optional[Dict[str, Any]] = None) -> None:
    resolver = _make_resolver(bundle or {}) if bundle else None
    v = Draft202012Validator(schema, resolver=resolver)
    errors = sorted(v.iter_errors(instance), key=lambda e: e.path)
    if errors:
        e0 = errors[0]
        raise ValueError(f"schema_validation_failed: {e0.message}")


def to_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")
