import logging
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .orchestrator.agent import OrchestratorAgent
from .runtime.state import RuntimeState
from .tools.mcp_client import MCPClient

log = logging.getLogger("genai_core.api")


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


def _normalize_chat_response(resp: Any, correlation_id: str) -> Dict[str, Any]:
    if not isinstance(resp, dict):
        return {
            "answer": "Ocorreu um erro interno ao processar o pedido.",
            "error": "invalid_orchestrator_response",
            "correlation_id": correlation_id,
        }

    answer = resp.get("answer", None)
    if answer is None:
        resp["answer"] = "Ocorreu um erro interno ao gerar a resposta."
        resp["error"] = resp.get("error") or "null_answer"
    elif not isinstance(answer, str):
        resp["answer"] = str(answer)
    else:
        resp["answer"] = answer.strip() or "Nao consegui gerar uma resposta util. Podes reformular a pergunta?"

    resp["correlation_id"] = correlation_id
    return resp


def create_app(cfg: dict, runtime: RuntimeState) -> FastAPI:
    app = FastAPI(title="GenAI Core Phase 1")

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = cid
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    mcp = MCPClient.from_config(cfg.get("tools", {}).get("mcp", {}))
    orchestrator = OrchestratorAgent(cfg=cfg, runtime=runtime, mcp=mcp)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        trace_id = str(uuid.uuid4())
        cid = getattr(request.state, "correlation_id", "")
        log.exception("Unhandled error trace_id=%s correlation_id=%s path=%s", trace_id, cid, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "trace_id": trace_id,
                "correlation_id": cid,
                "message": str(exc),
                "answer": "Ocorreu um erro interno.",
            },
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "ready": runtime.ready,
            "ready_reason": runtime.ready_reason,
            "vllm": runtime.vllm_health,
            "mcp": runtime.mcp_health,
            "model": runtime.model_info.model_name if runtime.model_info else None,
            "limits": runtime.model_info.model_limits if runtime.model_info else None,
        }

    @app.get("/ready")
    async def ready():
        status = 200 if runtime.ready else 503
        return JSONResponse(
            status_code=status,
            content={
                "ready": runtime.ready,
                "reason": runtime.ready_reason,
                "vllm": runtime.vllm_health,
                "mcp": runtime.mcp_health,
                "model": runtime.model_info.model_name if runtime.model_info else None,
            },
        )

    @app.post("/chat")
    async def chat(req: ChatRequest, request: Request):
        cid = getattr(request.state, "correlation_id", str(uuid.uuid4()))

        if not runtime.ready:
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "3", "X-Correlation-ID": cid},
                content={
                    "error": "model_not_ready",
                    "ready": False,
                    "reason": runtime.ready_reason,
                    "vllm": runtime.vllm_health,
                    "mcp": runtime.mcp_health,
                    "message": "Model is warming up. Retry shortly.",
                    "answer": "O modelo ainda nao esta pronto. Tenta novamente em instantes.",
                    "correlation_id": cid,
                },
            )

        try:
            resp = await orchestrator.chat(
                user_id=req.user_id,
                session_id=req.session_id,
                message=req.message,
                correlation_id=cid,
            )
        except Exception as e:
            log.exception("orchestrator.chat failed correlation_id=%s", cid)
            return JSONResponse(
                status_code=500,
                headers={"X-Correlation-ID": cid},
                content={
                    "error": "orchestrator_failure",
                    "message": str(e),
                    "answer": "Ocorreu um erro interno ao processar o pedido.",
                    "correlation_id": cid,
                },
            )

        return _normalize_chat_response(resp, cid)

    return app
