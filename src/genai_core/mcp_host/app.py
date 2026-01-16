from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from pydantic import BaseModel

from .registry import ToolRegistry, ToolSpec
from .web_search import web_search
from .influxdb_tool import influxdb_query

log = logging.getLogger("genai_core.mcp_host")


class CallRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


def create_mcp_host(registry: ToolRegistry) -> FastAPI:
    """Create a minimal MCP HTTP server.

    Wire format:
      POST /call {"tool": "name", "args": {...}}
      -> {"results": [...], "error": ""}
    """
    app = FastAPI(title="GenAI MCP Host (Python)")

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        try:
            log.info("request %s %s", request.method, request.url.path)
        except Exception:
            pass
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", **registry.describe()}

    @app.post("/call")
    async def call(req: CallRequest):
        spec = registry.get(req.tool)
        if not spec:
            return {"results": [], "error": f"unknown_tool: {req.tool}"}
        try:
            return await spec.handler(req.args or {})
        except Exception as e:
            log.exception("tool_failed tool=%s args=%s", req.tool, req.args)
            return {"results": [], "error": str(e)}

    return app


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        ToolSpec(
            name="web_search",
            handler=web_search,
            description="Retrieve fresh facts from the internet (DuckDuckGo Lite/HTML Phase 1).",
        )
    )

    reg.register(
        ToolSpec(
            name="influxdb_query",
            handler=influxdb_query,
            description="Query time-series data from InfluxDB v2 (Flux).",
        )
    )

    return reg


def create_app() -> FastAPI:
    return create_mcp_host(default_registry())
