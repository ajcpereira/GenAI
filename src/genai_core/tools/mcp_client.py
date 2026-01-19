from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

log = logging.getLogger("genai_core.mcp")


class MCPClient:
    """
    Enterprise MCP client:
    - Generic tool invocation via POST /call {"tool": "...", "args": {...}}
    - Deterministic timeout
    - No secrets in args (enforced in Orchestrator; tool side should also enforce)
    """

    def __init__(self, enabled: bool, base_url: str, timeout_s: int = 30):
        self.enabled = bool(enabled)
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_s = int(timeout_s)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MCPClient":
        enabled = bool(cfg.get("enabled", True))
        base_url = str(cfg.get("base_url", "")).strip()
        timeout_s = int(cfg.get("timeout_s", 30))
        return cls(enabled=enabled, base_url=base_url, timeout_s=timeout_s)

    async def call_tool(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"results": [], "error": "mcp_disabled"}
        if not self.base_url:
            return {"results": [], "error": "mcp_missing_base_url"}

        payload = {"tool": str(tool), "args": (args or {})}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(f"{self.base_url}/call", json=payload)
                r.raise_for_status()
                data: Dict[str, Any] = r.json()
        except Exception as e:
            return {"results": [], "error": f"mcp_call_failed:{type(e).__name__}"}

        # Stable contract expected from MCP host: {"results":[...], "error":""}
        if not isinstance(data, dict):
            return {"results": [], "error": "mcp_invalid_response"}

        results = data.get("results")
        err = data.get("error") or ""
        if not isinstance(results, list):
            results = []
        if not isinstance(err, str):
            err = str(err)

        return {"results": results, "error": err}

    # Backward compatibility (optional)
    async def web_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        return await self.call_tool("web_search", {"query": query, "top_k": int(k)})
