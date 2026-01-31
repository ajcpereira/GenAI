# ui_main.py - standalone UI server (runs separately from the main API)
import os
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from utils.common import configure_logging, load_yaml, apply_env_overrides

logger = logging.getLogger("genai.ui")

def build_app(orchestrator_base_url: str) -> FastAPI:
    app = FastAPI(title="GenAIv2 UI", version="1.0")

    client = httpx.AsyncClient(timeout=30.0)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # Avoid leaking sockets in long-lived UI processes.
        try:
            await client.aclose()
        except Exception:
            pass

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def proxy_api(path: str, request: Request) -> Response:
        # Proxy API calls to the orchestrator, preserving method, headers, and body.
        upstream = orchestrator_base_url.rstrip("/") + "/api/" + path.lstrip("/")
        method = request.method.upper()
        body = await request.body()

        # Forward headers but drop hop-by-hop and host
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        try:
            upstream_resp = await client.request(method, upstream, content=body, headers=headers, params=dict(request.query_params))
        except Exception as e:
            logger.exception("ui_proxy_failed", extra={"upstream": upstream})
            return Response(status_code=502, content=f"Upstream API unavailable: {e}")

        resp_headers = dict(upstream_resp.headers)
        # Avoid invalid hop-by-hop headers
        for h in ["content-encoding", "transfer-encoding", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade"]:
            resp_headers.pop(h, None)

        return Response(status_code=upstream_resp.status_code, content=upstream_resp.content, headers=resp_headers, media_type=upstream_resp.headers.get("content-type"))

    # Serve static UI assets.
    # IMPORTANT: mount AFTER the /api proxy route, otherwise StaticFiles will catch /api/*
    # and POSTs will return 405 Method Not Allowed.
    ui_dir = os.path.join(os.path.dirname(__file__), "ui")
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    return app

def main() -> None:
    cfg_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    cfg = load_yaml(cfg_path)
    cfg = apply_env_overrides(cfg)

    configure_logging(cfg)

    host = os.environ.get("UI_HOST", "0.0.0.0")
    port = int(os.environ.get("UI_PORT", "8003"))
    orchestrator_base_url = os.environ.get("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")

    app = build_app(orchestrator_base_url)

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
