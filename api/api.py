# api/api.py
import logging
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request, Header
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from utils.common import now_iso, validate_json
from orchestrator.orchestrator import Orchestrator

logger = logging.getLogger("genai.api")
router = APIRouter()


def make_router(
    orchestrator: Orchestrator,
    contract_bundle: Dict[str, Any],
) -> APIRouter:
    schemas = contract_bundle["schemas"]
    envelope_schema = schemas["Envelope"]
    metadata_schema = schemas["Metadata"]
    user_request_schema = schemas["UserRequestPayload"]
    error_payload_schema = schemas["ErrorPayload"]

    def _metadata(
        *,
        request_id: str,
        message_type: str,
        source: str,
        x_user_id: Optional[str],
        x_session_id: Optional[str],
        trace_id: Optional[str],
        span_id: Optional[str],
    ) -> Dict[str, Any]:
        md: Dict[str, Any] = {
            "schema_version": "1.1",
            "message_type": message_type,
            "request_id": request_id,
            "timestamp": now_iso(),
            "source": source,
            "user_id": x_user_id,
            "session_id": x_session_id,
            # NOTE: trace is optional in schema, but if present must contain trace_id/span_id.
            "trace": {"trace_id": trace_id, "span_id": span_id},
        }
        return md

    def _make_error_envelope(
        *,
        request_id: str,
        code: str,
        message: str,
        detail: Optional[Dict[str, Any]],
        x_user_id: Optional[str],
        x_session_id: Optional[str],
        trace_id: Optional[str],
        span_id: Optional[str],
        source: str = "api",
    ) -> Dict[str, Any]:
        payload = {"error": {"code": code, "message": message, "detail": detail}}
        env = {
            "metadata": _metadata(
                request_id=request_id,
                message_type="error",
                source=source,
                x_user_id=x_user_id,
                x_session_id=x_session_id,
                trace_id=trace_id,
                span_id=span_id,
            ),
            "payload": payload,
        }
        validate_json(metadata_schema, env["metadata"], bundle=contract_bundle)
        validate_json(error_payload_schema, env["payload"], bundle=contract_bundle)
        validate_json(envelope_schema, env, bundle=contract_bundle)
        return env

    @router.get("/health")
    async def health():
        return {"status": "ok"}

    @router.get("/tools")
    async def tools(request: Request):
        try:
            mcp = getattr(orchestrator.executor, "mcp", None)
            if not mcp:
                return {"tools": []}
            tools = await mcp.list_tools()
            return {"tools": tools}
        except Exception as e:
            logger.warning("tools_list_failed", extra={"error": str(e)})
            return {"tools": []}

    @router.post("/chat")
    async def chat(
        body: Dict[str, Any],
        request: Request,
        x_request_id: Optional[str] = Header(default=None, alias="x-request-id"),
        x_user_id: Optional[str] = Header(default=None, alias="x-user-id"),
        x_session_id: Optional[str] = Header(default=None, alias="x-session-id"),
    ):
        request_id = x_request_id or getattr(request.state, "request_id", None) or "unknown"
        trace_id = getattr(request.state, "trace_id", None)
        span_id = getattr(request.state, "span_id", None)

        try:
            # Accept either envelope request or legacy request
            if isinstance(body, dict) and "metadata" in body and "payload" in body:
                validate_json(envelope_schema, body, bundle=contract_bundle)
                validate_json(metadata_schema, body["metadata"], bundle=contract_bundle)
                validate_json(user_request_schema, body["payload"], bundle=contract_bundle)
                req_env = body
            else:
                msg = str((body or {}).get("message", "")).strip()
                req_env = {
                    "metadata": _metadata(
                        request_id=request_id,
                        message_type="request",
                        source="api",
                        x_user_id=x_user_id,
                        x_session_id=x_session_id,
                        trace_id=trace_id,
                        span_id=span_id,
                    ),
                    "payload": {
                        "message": msg,
                        "enabled_tools": list((body or {}).get("enabled_tools") or []),
                    },
                }
                validate_json(metadata_schema, req_env["metadata"], bundle=contract_bundle)
                validate_json(user_request_schema, req_env["payload"], bundle=contract_bundle)
                validate_json(envelope_schema, req_env, bundle=contract_bundle)

            out_env = await orchestrator.handle_envelope(req_env)

            validate_json(envelope_schema, out_env, bundle=contract_bundle)
            validate_json(metadata_schema, out_env["metadata"], bundle=contract_bundle)

            status = 200
            if "error" in (out_env.get("payload") or {}):
                status = 422

            return JSONResponse(status_code=status, content=out_env)

        except JsonSchemaValidationError as e:
            err_env = _make_error_envelope(
                request_id=request_id,
                code="CONTRACT_VALIDATION_ERROR",
                message=str(e),
                detail=None,
                x_user_id=x_user_id,
                x_session_id=x_session_id,
                trace_id=trace_id,
                span_id=span_id,
            )
            return JSONResponse(status_code=422, content=err_env)

        except ValueError as e:
            # Raised by validate_json(...) helper: schema_validation_failed: ...
            err_env = _make_error_envelope(
                request_id=request_id,
                code="CONTRACT_VALIDATION_ERROR",
                message=str(e),
                detail=None,
                x_user_id=x_user_id,
                x_session_id=x_session_id,
                trace_id=trace_id,
                span_id=span_id,
            )
            return JSONResponse(status_code=422, content=err_env)

        except Exception as e:
            err_env = _make_error_envelope(
                request_id=request_id,
                code="INTERNAL_ERROR",
                message=str(e),
                detail=None,
                x_user_id=x_user_id,
                x_session_id=x_session_id,
                trace_id=trace_id,
                span_id=span_id,
            )
            return JSONResponse(status_code=500, content=err_env)

    @router.post("/chat/stream")
    async def chat_stream(
        body: Dict[str, Any],
        request: Request,
        x_request_id: Optional[str] = Header(default=None, alias="x-request-id"),
        x_user_id: Optional[str] = Header(default=None, alias="x-user-id"),
        x_session_id: Optional[str] = Header(default=None, alias="x-session-id"),
    ):
        request_id = x_request_id or getattr(request.state, "request_id", None) or "unknown"
        trace_id = getattr(request.state, "trace_id", None)
        span_id = getattr(request.state, "span_id", None)

        async def event_gen():
            try:
                msg = str((body or {}).get("message", "")).strip()
                req_env = {
                    "metadata": _metadata(
                        request_id=request_id,
                        message_type="request",
                        source="api",
                        x_user_id=x_user_id,
                        x_session_id=x_session_id,
                        trace_id=trace_id,
                        span_id=span_id,
                    ),
                    "payload": {
                        "message": msg,
                        "enabled_tools": list((body or {}).get("enabled_tools") or []),
                    },
                }
                validate_json(envelope_schema, req_env, bundle=contract_bundle)

                out_env = await orchestrator.handle_envelope(req_env)

                answer = ""
                if isinstance(out_env.get("payload"), dict):
                    answer = str(out_env["payload"].get("answer", ""))

                yield f"event: delta\ndata: {json.dumps({'text': answer})}\n\n"
                yield f"event: final\ndata: {json.dumps(out_env)}\n\n"

            except ValueError as e:
                err_env = _make_error_envelope(
                    request_id=request_id,
                    code="CONTRACT_VALIDATION_ERROR",
                    message=str(e),
                    detail=None,
                    x_user_id=x_user_id,
                    x_session_id=x_session_id,
                    trace_id=trace_id,
                    span_id=span_id,
                )
                yield f"event: error\ndata: {json.dumps(err_env)}\n\n"

            except Exception as e:
                err_env = _make_error_envelope(
                    request_id=request_id,
                    code="INTERNAL_ERROR",
                    message=str(e),
                    detail=None,
                    x_user_id=x_user_id,
                    x_session_id=x_session_id,
                    trace_id=trace_id,
                    span_id=span_id,
                )
                yield f"event: error\ndata: {json.dumps(err_env)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return router
