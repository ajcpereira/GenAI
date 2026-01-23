import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from mcp_host_service.registry import ToolRegistry

logger = logging.getLogger("mcp.api")
router = APIRouter()


class RunRequest(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)


# Policy simples: UI não pode chamar tools; orchestrator pode.
# Mais tarde podes ligar isto a config/policy.
ALLOWED_CALLERS = {
    "time.now": {"orchestrator", "admin"},
}


def _caller_allowed(tool_name: str, caller: str) -> bool:
    allowed = ALLOWED_CALLERS.get(tool_name, {"orchestrator", "admin"})
    return caller in allowed


def make_mcp_router(registry: ToolRegistry) -> APIRouter:
    @router.get("/health")
    async def health():
        return {"status": "ok"}

    @router.get("/mcp/tools")
    async def list_tools(x_caller: str = Header(default="unknown", alias="x-caller")):
        tools = registry.list_specs()
        # Filtra tools por caller
        filtered = [t for t in tools if _caller_allowed(t["name"], x_caller)]
        return {"tools": filtered}

    @router.post("/mcp/tools/{tool_name}:run")
    async def run_tool(
        tool_name: str,
        req: RunRequest,
        x_caller: str = Header(default="unknown", alias="x-caller"),
        x_request_id: Optional[str] = Header(default=None, alias="x-request-id"),
    ):
        if not _caller_allowed(tool_name, x_caller):
            raise HTTPException(status_code=403, detail="Tool not allowed for caller")

        try:
            out = await registry.run(tool_name, req.inputs)
            logger.info("tool_run", extra={"tool": tool_name, "caller": x_caller, "request_id": x_request_id})
            return out
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown tool")
        except Exception as e:
            logger.exception("tool_run_failed", extra={"tool": tool_name, "caller": x_caller, "request_id": x_request_id})
            raise HTTPException(status_code=500, detail=str(e))

    return router
