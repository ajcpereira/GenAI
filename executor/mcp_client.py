#executor/mcp_client.py
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("genai.executor")


class MCPClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0, caller_id: str = "orchestrator"):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.caller_id = caller_id

    def _headers(self) -> Dict[str, str]:
        return {"x-caller": self.caller_id, "Content-Type": "application/json"}

    async def list_tools(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/mcp/tools"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(url, headers=self._headers())
            r.raise_for_status()
            data = r.json()
        return list(data.get("tools") or [])

    async def run_tool(self, tool_name: str, inputs: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/mcp/tools/{tool_name}:run"
        headers = self._headers()
        if request_id:
            headers["x-request-id"] = request_id

        payload = {"inputs": inputs}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()

    async def run_tool_with_trace(
        self, tool_name: str, inputs: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a tool call and return a structured trace suitable for persistence.

        Contract:
          { ok: bool, response: object|null, error: string|null, trace: object }

        Notes:
          - This does NOT raise for non-2xx; it captures status/error and returns ok=False.
          - The goal is deterministic auditability (raw request + raw response).
        """
        url = f"{self.base_url}/mcp/tools/{tool_name}:run"
        headers = self._headers()
        if request_id:
            headers["x-request-id"] = request_id

        payload = {"inputs": inputs}
        t0 = time.perf_counter()
        trace: Dict[str, Any] = {
            "method": "POST",
            "url": url,
            "request": {"headers": dict(headers), "json": payload},
            "response": {"status_code": None, "headers": None, "json": None, "text": None},
            "elapsed_ms": None,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(url, headers=headers, json=payload)
                trace["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
                trace["response"]["status_code"] = int(r.status_code)
                # Headers can contain non-serializable values; force to plain dict[str,str]
                try:
                    trace["response"]["headers"] = {str(k): str(v) for k, v in (r.headers or {}).items()}
                except Exception:
                    trace["response"]["headers"] = None

                # Capture body (best-effort)
                body_text: Optional[str] = None
                body_json: Optional[Dict[str, Any]] = None
                try:
                    body_json = r.json()
                except Exception:
                    try:
                        body_text = r.text
                    except Exception:
                        body_text = None

                trace["response"]["json"] = body_json
                if body_text is not None:
                    # Keep it bounded for persistence safety
                    trace["response"]["text"] = body_text[:8_000]

                if r.is_success:
                    return {"ok": True, "response": body_json, "error": None, "trace": trace}

                err = f"tool_http_error: status={r.status_code}"
                return {"ok": False, "response": body_json, "error": err, "trace": trace}

        except Exception as e:
            trace["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
            return {
                "ok": False,
                "response": None,
                "error": f"tool_transport_error: {type(e).__name__}: {str(e)}",
                "trace": trace,
            }
