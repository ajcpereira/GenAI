# mcp_host_service/api.py
import logging
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from mcp_host_service.registry import ToolRegistry

logger = logging.getLogger("mcp.api")
router = APIRouter()


class RunRequest(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)


# Policy simples: UI não pode chamar tools; orchestrator pode.
# "admin" pode listar/usar para debug.
ALLOWED_CALLERS: Dict[str, Set[str]] = {
    "time.now": {"orchestrator", "admin"},
}


def _caller_allowed(tool_name: str, caller: str) -> bool:
    # Default policy: allow orchestrator+admin unless explicitly specified.
    allowed = ALLOWED_CALLERS.get(tool_name, {"orchestrator", "admin"})
    return caller in allowed


def make_mcp_router(registry: ToolRegistry) -> APIRouter:
    @router.get("/health")
    async def health():
        return {"status": "ok"}

    @router.get("/mcp/tools")
    async def list_tools(x_caller: str = Header(default="unknown", alias="x-caller")):
        tools = registry.list_specs()

        # Filter tools by caller policy
        filtered = [t for t in tools if _caller_allowed(t["name"], x_caller)]

        logger.info(
            "tools_list",
            extra={
                "caller": x_caller,
                "tools_total": len(tools),
                "tools_returned": len(filtered),
            },
        )

        # Helpful debug metadata (safe): shows counts, not hidden tool details.
        return {
            "tools": filtered,
            "meta": {
                "caller": x_caller,
                "tools_total": len(tools),
                "tools_returned": len(filtered),
            },
        }

    @router.post("/mcp/tools/{tool_name}:run")
    async def run_tool(
        tool_name: str,
        req: RunRequest,
        x_caller: str = Header(default="unknown", alias="x-caller"),
        x_request_id: Optional[str] = Header(default=None, alias="x-request-id"),
    ):
        if not _caller_allowed(tool_name, x_caller):
            logger.warning(
                "tool_denied",
                extra={"tool": tool_name, "caller": x_caller, "request_id": x_request_id},
            )
            raise HTTPException(status_code=403, detail="Tool not allowed for caller")

        try:
            out = await registry.run(tool_name, req.inputs)
            logger.info(
                "tool_run",
                extra={"tool": tool_name, "caller": x_caller, "request_id": x_request_id},
            )
            return out
        except JSONSchemaValidationError as e:
            # Contract violation (invalid inputs or tool output). Treat as client error.
            raise HTTPException(status_code=422, detail=f"schema_validation_failed: {e.message}")
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown tool")
        except ValueError as e:
            # Tool-level deterministic error (e.g., safe math parser). Treat as bad request.
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(
                "tool_run_failed",
                extra={"tool": tool_name, "caller": x_caller, "request_id": x_request_id},
            )
            raise HTTPException(status_code=500, detail=str(e))

    return router
