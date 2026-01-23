# GenAIv2/main.py
import logging
import time
import uuid
from typing import Dict, Any, Optional

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
from orchestrator.orchestrator import Orchestrator
from executor.executor import Executor
from responder.responder import Responder

from api.api import make_router
from api.ui import make_ui_router  # "/" -> "/ui/index.html" redirect

logger = logging.getLogger("genai.main")


def _new_session_id() -> str:
    return str(uuid.uuid4())


def create_app() -> FastAPI:
    cfg: Dict[str, Any] = load_yaml("config/config.yaml")
    cfg = apply_env_overrides(cfg)

    configure_logging(cfg)

    bundle = load_contract_bundle(cfg["contracts"]["bundle_path"])

    # Core components (DI explícito, compatível com o teu design atual)
    planner = Planner(cfg["planner"], bundle)

    executor = Executor(
        mcp_base_url=str(cfg["mcp"]["base_url"]),
        mcp_timeout_s=float(cfg["mcp"].get("timeout_s", 10.0)),
        caller_id=str(cfg["mcp"].get("caller_id", "orchestrator")),
        max_steps=int(cfg.get("executor", {}).get("max_steps", cfg.get("orchestrator", {}).get("max_steps", 40))),
        max_input_bytes_per_step=int(cfg.get("executor", {}).get("max_input_bytes_per_step", 64000)),
        allow_optional_tool_calls=bool(cfg.get("executor", {}).get("allow_optional_tool_calls", False)),
    )

    responder = Responder(cfg.get("responder", {}), bundle)

    orchestrator = Orchestrator(
        planner=planner,
        executor=executor,
        responder=responder,
        contract_bundle=bundle,
        confidence_threshold=float(cfg.get("orchestrator", {}).get("confidence_threshold", 0.8)),
        max_steps=int(cfg.get("orchestrator", {}).get("max_steps", 40)),
        max_tool_calls=int(cfg.get("orchestrator", {}).get("max_tool_calls", 15)),
        # NEW: para UI, responde TOOL_NOT_AVAILABLE se pedirem tools inválidas
        reject_unknown_enabled_tools=bool(cfg.get("orchestrator", {}).get("reject_unknown_enabled_tools", True)),
        max_replans=int(cfg.get("orchestrator", {}).get("max_replans", 2)),
    )

    app = FastAPI(title=cfg["app"]["name"])

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
            # make sure context is reset even if downstream raises
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
            httponly=False,         # UI pode ler se precisares; torna True quando passares a storage server-side
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
    app.include_router(make_router(orchestrator, bundle), prefix="/api")

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
