from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("genai_core.mcp")


class MCPClient:
    def __init__(
        self,
        enabled: bool,
        base_url: str,
        timeout_s: int = 30,
        healthcheck: bool = True,
    ):
        self.enabled = bool(enabled)
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_s = int(timeout_s)
        self.healthcheck = bool(healthcheck)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MCPClient":
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            base_url=str(cfg.get("base_url", "")).strip(),
            timeout_s=int(cfg.get("timeout_s", 30)),
            healthcheck=bool(cfg.get("healthcheck", False)),  # default false to avoid extra latency
        )

    async def _check_health(self, client: httpx.AsyncClient) -> None:
        if not self.healthcheck:
            return
        r = await client.get(f"{self.base_url}/health")
        r.raise_for_status()

    async def call_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generic MCP call:
          POST {base_url}/call {"tool": "<tool_name>", "args": {...}}
        Returns: list of result dicts (tool-specific schema).
        Raises on transport errors or tool error string.
        """
        if not self.enabled:
            return []
        if not self.base_url:
            raise RuntimeError("MCP base_url not configured")

        tool = (tool_name or "").strip()
        if not tool:
            raise RuntimeError("MCP tool_name is empty")

        payload = {"tool": tool, "args": args or {}}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                await self._check_health(client)
                r = await client.post(f"{self.base_url}/call", json=payload)
                r.raise_for_status()
                data: Dict[str, Any] = r.json()
        except Exception as e:
            raise RuntimeError(f"MCP request failed to {self.base_url}/call: {e}") from e

        tool_error = (data.get("error") or "").strip()
        if tool_error:
            raise RuntimeError(f"MCP tool error ({tool}): {tool_error}")

        raw = data.get("results") or []
        if not isinstance(raw, list):
            raise RuntimeError(f"MCP invalid results type for {tool}: {type(raw).__name__}")

        return raw

    # Convenience wrapper (keeps your existing naming)
    async def web_search(self, query: str, k: Optional[int] = None) -> List[Dict[str, str]]:
        top_k = int(k) if k is not None else 5
        res = await self.call_tool("web_search", {"query": query, "top_k": top_k, "k": top_k})
        out: List[Dict[str, str]] = []
        for item in res:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": str(item.get("title", "") or "").strip(),
                    "url": str(item.get("url", "") or "").strip(),
                    "snippet": str(item.get("snippet", "") or "").strip(),
                }
            )
        return out
