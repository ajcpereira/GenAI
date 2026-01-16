import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .orchestrator.agent import OrchestratorAgent
from .tools.mcp_client import MCPClient
from .runtime.state import RuntimeState


log = logging.getLogger("genai_core.api")


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


def create_app(cfg: dict, runtime: RuntimeState) -> FastAPI:
    app = FastAPI(title="GenAI Core Phase 1")

    mcp = MCPClient.from_config(cfg.get("tools", {}).get("mcp", {}))
    orchestrator = OrchestratorAgent(cfg=cfg, runtime=runtime, mcp=mcp)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        trace_id = str(uuid.uuid4())
        log.exception("Unhandled error trace_id=%s path=%s", trace_id, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "trace_id": trace_id, "message": str(exc)},
        )

    # Liveness: core process is up
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "ready": runtime.ready,
            "ready_reason": runtime.ready_reason,
            "vllm": runtime.vllm_health,
            "model": runtime.model_info.model_name if runtime.model_info else None,
            "limits": runtime.model_info.model_limits if runtime.model_info else None,
        }

    # Readiness: model is ready to serve chat requests
    @app.get("/ready")
    async def ready():
        status = 200 if runtime.ready else 503
        return JSONResponse(
            status_code=status,
            content={
                "ready": runtime.ready,
                "reason": runtime.ready_reason,
                "vllm": runtime.vllm_health,
                "model": runtime.model_info.model_name if runtime.model_info else None,
            },
        )

    @app.post("/chat")
    async def chat(req: ChatRequest):
        if not runtime.ready:
            # Fail-fast during warmup: do not block clients with long timeouts
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "3"},
                content={
                    "error": "model_not_ready",
                    "ready": False,
                    "reason": runtime.ready_reason,
                    "vllm": runtime.vllm_health,
                    "message": "Model is warming up. Retry shortly.",
                },
            )

        resp = await orchestrator.chat(
            user_id=req.user_id,
            session_id=req.session_id,
            message=req.message,
        )
        return resp

    return app
