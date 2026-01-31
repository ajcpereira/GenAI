
import os
import sys

# Allow running as a script: python mcp_host_service/main.py
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
import uvicorn

from .registry import ToolRegistry
from .loader import load_tools
from mcp_host_service.api import make_mcp_router
from mcp_host_service.logging_setup import configure_mcp_logging

logger = logging.getLogger("mcp.main")


def create_app() -> FastAPI:
    configure_mcp_logging("INFO")

    registry = ToolRegistry()
    load_tools(registry)

    app = FastAPI(title="mcp-host")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        start = time.time()
        response: Response = await call_next(request)
        elapsed_ms = int((time.time() - start) * 1000)
        response.headers["x-latency-ms"] = str(elapsed_ms)
        return response

    app.include_router(make_mcp_router(registry))
    return app


app = create_app()

if __name__ == "__main__":
    # IMPORTANT: use fully-qualified module path. Using "main:app" can accidentally
    # resolve the orchestrator's main.py when executed from the project root.
    uvicorn.run("mcp_host_service.main:app", host="0.0.0.0", port=8010, log_level="info", reload=False)
