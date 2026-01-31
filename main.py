# GenAIv2/main.py
import logging
import time
import uuid
from typing import Dict, Any, Optional, List

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils.common import (
    configure_logging,
    load_yaml,
    apply_env_overrides,
    load_contract_bundle,
    new_request_id,
    set_request_context,
    reset_request_context,
)

from planner.planner import Planner
from orchestrator import Orchestrator
from executor.executor import Executor
from responder.responder import Responder

from orchestrator.session_store import PostgresSessionStore
from orchestrator.context_manager import ContextManager
from orchestrator.context_policy_classifier import ContextPolicyClassifier


from api.api import make_router
from api.ui import make_ui_router  # "/" -> "/ui/index.html" redirect

logger = logging.getLogger("genai.main")


def _new_session_id() -> str:
    return str(uuid.uuid4())


async def _check_http_any(base_url: str, paths: List[str], timeout_s: float) -> bool:
    base = str(base_url).rstrip("/")
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for p in paths:
            try:
                url = f"{base}{p}"
                r = await client.get(url)
                if r.status_code < 500:
                    return True
            except Exception:
                continue
    return False


async def _discover_vllm_max_context_tokens(*, base_url: str, model_id: str, timeout_s: float) -> Optional[int]:
    """Best-effort discovery of model context length from vLLM's /v1/models.

    vLLM typically includes fields such as max_model_len. We also check a few common alternatives
    to stay robust across versions.
    """
    base = str(base_url).rstrip("/")
    url = f"{base}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json() or {}
    except Exception:
        return None

    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None

    # Prefer exact match, otherwise fall back to first item.
    chosen: Optional[Dict[str, Any]] = None
    for m in models:
        if isinstance(m, dict) and str(m.get("id")) == str(model_id):
            chosen = m
            break
    if chosen is None:
        chosen = models[0] if models and isinstance(models[0], dict) else None
    if not isinstance(chosen, dict):
        return None

    for k in ("max_model_len", "max_context_len", "context_length", "max_seq_len", "n_ctx"):
        v = chosen.get(k)
        try:
            if v is not None and int(v) > 0:
                return int(v)
        except Exception:
            continue
    return None


def create_app() -> FastAPI:
    cfg: Dict[str, Any] = load_yaml("config/config.yaml")
    cfg = apply_env_overrides(cfg)

    configure_logging(cfg)

    bundle = load_contract_bundle(cfg["contracts"]["bundle_path"])

    # Core components (DI explícito, compatível com o teu design atual)
    planner = Planner(cfg["planner"], bundle)

    context_policy_classifier = ContextPolicyClassifier(
        base_url=cfg["planner"]["vllm_base_url"],
        chat_path=cfg["planner"]["chat_path"],
        model=cfg["planner"]["model"],
        api_key=cfg["planner"].get("api_key"),
)

    executor = Executor(
        mcp_base_url=str(cfg["mcp"]["base_url"]),
        mcp_timeout_s=float(cfg["mcp"].get("timeout_s", 10.0)),
        caller_id=str(cfg["mcp"].get("caller_id", "orchestrator")),
        max_steps=int(cfg.get("executor", {}).get("max_steps", cfg.get("orchestrator", {}).get("max_steps", 40))),
        max_input_bytes_per_step=int(cfg.get("executor", {}).get("max_input_bytes_per_step", 64000)),
        allow_optional_tool_calls=bool(cfg.get("executor", {}).get("allow_optional_tool_calls", False)),
    )

    responder = Responder(cfg.get("responder", {}), bundle)

    # Storage (Postgres is REQUIRED for this major release)
    storage_cfg = (cfg.get("storage", {}) or {}).get("postgres", {}) or {}
    if not storage_cfg.get("dsn"):
        raise RuntimeError("storage.postgres.dsn must be configured for major release")

    pg_store = PostgresSessionStore(
        dsn=str(storage_cfg["dsn"]),
        pool_min=int(storage_cfg.get("pool_min", 1)),
        pool_max=int(storage_cfg.get("pool_max", 5)),
        connect_timeout_s=float(storage_cfg.get("connect_timeout_s", 5.0)),
        statement_timeout_ms=int(storage_cfg.get("statement_timeout_ms", 20000)),
    )

    context_cfg = cfg.get("context", {}) or {}
    context_manager = ContextManager(store=pg_store, cfg=context_cfg)

    orchestrator = Orchestrator(
        planner=planner,
        executor=executor,
        responder=responder,
        contract_bundle=bundle,
        confidence_threshold=float(cfg.get("orchestrator", {}).get("confidence_threshold", 0.8)),
        max_steps=int(cfg.get("orchestrator", {}).get("max_steps", 40)),
        max_tool_calls=int(cfg.get("orchestrator", {}).get("max_tool_calls", 15)),
        reject_unknown_enabled_tools=bool(cfg.get("orchestrator", {}).get("reject_unknown_enabled_tools", True)),
        max_replans=int(cfg.get("orchestrator", {}).get("max_replans", 2)),
        session_store=pg_store,
        storage_cfg=storage_cfg,
        context_manager=context_manager,
        context_policy_classifier=context_policy_classifier,
    )

    app = FastAPI(title=cfg["app"]["name"])

    @app.on_event("startup")
    async def _startup() -> None:
        # Postgres is required: if not reachable, fail startup -> app unavailable.
        await pg_store.start()
        await pg_store.ensure_schema()

        # Validate vLLM availability (planner + responder).
        # If vLLM is down, we consider the app unavailable.
        vllm_paths = list((cfg.get("planner", {}) or {}).get("health_paths") or ["/v1/models", "/health"])
        timeout_s = float((cfg.get("planner", {}) or {}).get("health_timeout_s", 2.0))
        planner_ok = await _check_http_any(cfg["planner"]["vllm_base_url"], vllm_paths, timeout_s=timeout_s)
        responder_ok = await _check_http_any(cfg["responder"]["vllm_base_url"], vllm_paths, timeout_s=timeout_s)
        if not (planner_ok and responder_ok):
            raise RuntimeError("vLLM health check failed (planner/responder). App unavailable.")

        # Discover model context length at runtime (source of truth: vLLM).
        # We intentionally do not treat max_context_tokens as a static config parameter.
        ctx_timeout_s = max(1.0, float((cfg.get("planner", {}) or {}).get("health_timeout_s", 2.0)))
        model_id = str((cfg.get("planner", {}) or {}).get("model") or "")
        discovered = await _discover_vllm_max_context_tokens(
            base_url=str(cfg["planner"]["vllm_base_url"]),
            model_id=model_id,
            timeout_s=ctx_timeout_s,
        )
        if discovered:
            context_manager.set_max_context_tokens(discovered)
        else:
            # Keep existing budget as fallback (e.g., older vLLM builds that don't expose max_model_len).
            logger.warning(
                "context_budget_discovery_failed",
                extra={"vllm_base_url": str(cfg["planner"]["vllm_base_url"]), "model": model_id},
            )

        logger.info("startup_ok", extra={"postgres": "ok", "vllm_planner": planner_ok, "vllm_responder": responder_ok})

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await pg_store.stop()

    # CORS (opcional)
    cors_cfg = cfg.get("cors", {}) or {}
    if bool(cors_cfg.get("enabled", False)):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_cfg.get("allow_origins", ["*"]),
            allow_credentials=bool(cors_cfg.get("allow_credentials", True)),
            allow_methods=cors_cfg.get("allow_methods", ["*"]),
            allow_headers=cors_cfg.get("allow_headers", ["*"]),
            expose_headers=["x-request-id", "x-session-id", "x-latency-ms"],
        )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        start = time.time()

        # request_id
        req_id = request.headers.get("x-request-id") or new_request_id()
        request.state.request_id = req_id

        # trace ids (opcional)
        request.state.trace_id = request.headers.get("x-trace-id")
        request.state.span_id = request.headers.get("x-span-id")

        # session_id: header -> cookie -> gerar novo
        sess_id: Optional[str] = request.headers.get("x-session-id")
        if not sess_id:
            sess_id = request.cookies.get("genai_session_id")
        if not sess_id:
            sess_id = _new_session_id()
        request.state.session_id = sess_id

        # bind request context for log correlation across components
        _ctx_tokens = set_request_context(
            request_id=req_id,
            session_id=sess_id,
            trace_id=request.state.trace_id,
            span_id=request.state.span_id,
        )

        try:
            response: Response = await call_next(request)
        finally:
            reset_request_context(_ctx_tokens)

        elapsed_ms = int((time.time() - start) * 1000)

        response.headers["x-request-id"] = req_id
        response.headers["x-session-id"] = sess_id
        response.headers["x-latency-ms"] = str(elapsed_ms)

        # cookie para UI
        ui_cfg = cfg.get("ui", {}) or {}
        response.set_cookie(
            key="genai_session_id",
            value=sess_id,
            # UI does not need JavaScript access to the session cookie; keep it HttpOnly.
            httponly=True,
            samesite="lax",
            secure=bool(ui_cfg.get("cookie_secure", False)),
            max_age=int(ui_cfg.get("cookie_max_age_s", 60 * 60 * 24 * 30)),
        )

        logger.info(
            "http_request",
            extra={
                "request_id": req_id,
                "session_id": sess_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": elapsed_ms,
            },
        )
        return response

    # API
    app.include_router(
        make_router(orchestrator=orchestrator, contract_bundle=bundle),
        prefix="/api",
)




    # Root redirect "/" -> "/ui/index.html"
    app.include_router(make_ui_router(), include_in_schema=False)

    # Static UI files
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

    return app


app = create_app()

if __name__ == "__main__":
    cfg = load_yaml("config/config.yaml")
    cfg = apply_env_overrides(cfg)

    uvicorn.run(
        app,
        host=cfg["app"].get("host", "0.0.0.0"),
        port=int(cfg["app"].get("port", 8000)),
        log_level=str(cfg["app"].get("log_level", "info")).lower(),
        reload=False,
    )
