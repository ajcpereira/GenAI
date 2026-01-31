# api/api.py
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from utils.common import now_iso

logger = logging.getLogger("genai.api")


def _error_response_envelope(
    *,
    request_id: str,
    session_id: Optional[str],
    user_id: Optional[str],
    trace_id: Optional[str],
    span_id: Optional[str],
    code: str,
    message: str,
    detail: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "metadata": {
            "schema_version": "1.2",
            "message_type": "error",
            "request_id": request_id,
            "timestamp": now_iso(),
            "source": "api",
            "user_id": user_id,
            "session_id": session_id,
            "trace": {"trace_id": trace_id, "span_id": span_id},
            "timings_ms": {},
        },
        "payload": {"error": {"code": str(code), "message": str(message), "detail": detail}, "debug": {"stage": "api"}},
    }


def _user_request_envelope(
    *,
    request_id: str,
    session_id: Optional[str],
    user_id: Optional[str],
    trace_id: Optional[str],
    span_id: Optional[str],
    message: str,
    enabled_tools: list,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "schema_version": "1.2",
            "message_type": "user_request",
            "request_id": request_id,
            "timestamp": now_iso(),
            "source": "api",
            "user_id": user_id,
            "session_id": session_id,
            "trace": {"trace_id": trace_id, "span_id": span_id},
            "timings_ms": {},
        },
        "payload": {
            "message": message,
            "enabled_tools": enabled_tools,
        },
    }


def make_router(*, orchestrator, contract_bundle) -> APIRouter:
    """API Layer routes (prefix applied in main.py).

    - GET  /health
    - GET  /tools
    - POST /chat
    - GET  /sessions
    - GET  /sessions/{session_id}/messages
    - DELETE /sessions/{session_id}
    - GET  /requests/{request_id}/envelopes
    """

    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"status": "ok"}

    @router.get("/tools")
    async def tools():
        """List tools from the already-initialized MCP client (no outbound HTTP)."""
        mcp = getattr(getattr(orchestrator, "executor", None), "mcp", None)
        if not mcp:
            return {"tools": []}
        try:
            tools_list = await mcp.list_tools()
            return {"tools": tools_list}
        except Exception as e:
            logger.exception("tools_list_failed")
            raise HTTPException(status_code=502, detail=f"Tools indisponíveis: {e}")

    @router.post("/chat")
    async def chat(
        request: Request,
        x_session_id: Optional[str] = Header(default=None, alias="x-session-id"),
        x_user_id: Optional[str] = Header(default=None, alias="x-user-id"),
        x_request_id: Optional[str] = Header(default=None, alias="x-request-id"),
        trace_id: Optional[str] = Header(default=None, alias="x-trace-id"),
        span_id: Optional[str] = Header(default=None, alias="x-span-id"),
    ):
        request_id = x_request_id or str(uuid.uuid4())

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        message = (body or {}).get("message")
        enabled_tools = (body or {}).get("enabled_tools") or []

        if not isinstance(message, str) or not message.strip():
            raise HTTPException(status_code=400, detail="Field 'message' is required")
        if not isinstance(enabled_tools, list):
            enabled_tools = []

        # Ensure we always have a session id for persistence/UI.
        session_id = x_session_id or str(uuid.uuid4())

        req_env = _user_request_envelope(
            request_id=request_id,
            session_id=session_id,
            user_id=x_user_id,
            trace_id=trace_id,
            span_id=span_id,
            message=message.strip(),
            enabled_tools=enabled_tools,
        )

        try:
            resp_env = await orchestrator.handle_envelope(req_env)
        except ValueError as e:
            # Contract / schema errors must be treated as client errors.
            msg = str(e)
            if msg.startswith("schema_validation_failed:"):
                raise HTTPException(status_code=422, detail=msg)

            logger.exception("chat_failed", extra={"request_id": request_id, "session_id": session_id})
            err_env = _error_response_envelope(
                request_id=request_id,
                session_id=session_id,
                user_id=x_user_id,
                trace_id=trace_id,
                span_id=span_id,
                code="INTERNAL_ERROR",
                message=msg,
                detail=None,
            )
            resp_env = err_env
        except Exception as e:
            # Never crash the API: convert unexpected failures into a controlled error envelope.
            logger.exception("chat_failed_unexpected", extra={"request_id": request_id, "session_id": session_id})
            resp_env = _error_response_envelope(
                request_id=request_id,
                session_id=session_id,
                user_id=x_user_id,
                trace_id=trace_id,
                span_id=span_id,
                code="INTERNAL_ERROR",
                message=str(e),
                detail=None,
            )
        out = {"request": req_env, "response": resp_env}
        return JSONResponse(status_code=200, content=out, headers={"x-session-id": session_id})

    # -------------------------
    # Sessions API (for UI)
    # -------------------------

    @router.get("/sessions")
    async def list_sessions(limit: int = 50, offset: int = 0):
        store = getattr(orchestrator, "session_store", None)
        if store is None:
            raise HTTPException(status_code=501, detail="Session store not configured")

        rows = await store.list_sessions(limit=int(limit), offset=int(offset))
        return {
            "sessions": [
                {
                    "session_id": r.session_id,
                    "user_id": r.user_id,
                    "created_at": r.created_at,
                    "last_seen_at": r.last_seen_at,
                    "message_count": r.message_count,
                    "meta": r.meta,
                }
                for r in rows
            ]
        }

    @router.get("/sessions/{session_id}/messages")
    async def get_session_messages(session_id: str, limit: int = 500):
        store = getattr(orchestrator, "session_store", None)
        if store is None:
            raise HTTPException(status_code=501, detail="Session store not configured")

        rows = await store.get_messages_for_session(session_id=str(session_id), limit=int(limit))
        return {
            "session_id": session_id,
            "messages": [
                {
                    "seq": r.seq,
                    "role": r.role,
                    "content": r.content,
                    "request_id": r.request_id,
                }
                for r in rows
            ],
        }

    @router.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        store = getattr(orchestrator, "session_store", None)
        if store is None:
            raise HTTPException(status_code=501, detail="Session store not configured")
        await store.delete_session(session_id=str(session_id))
        return {"ok": True}

    @router.get("/requests/{request_id}/envelopes")
    async def list_request_envelopes(request_id: str, limit: int = 200):
        store = getattr(orchestrator, "session_store", None)
        if store is None or not hasattr(store, "list_envelopes_for_request"):
            raise HTTPException(status_code=501, detail="Envelope tracing not supported")
        envs = await store.list_envelopes_for_request(request_id=str(request_id), limit=int(limit))
        return {"request_id": request_id, "envelopes": envs}

    return router
