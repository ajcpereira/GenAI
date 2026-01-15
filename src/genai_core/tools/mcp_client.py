from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import httpx


@dataclass
class MCPConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8765"
    tool_name_web_search: str = "web_search"
    top_k: int = 5
    timeout_s: int = 10


class MCPClient:
    """Very small MCP-over-HTTP client (Phase 1).

    Expected wire format:
      POST {base_url}/call
      { "tool": "web_search", "args": {"query":"...", "top_k": 5} }

    Response:
      { "results": [{"title":..,"url":..,"snippet":..}, ...] }
    """

    def __init__(self, cfg: MCPConfig):
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MCPClient":
        if not cfg:
            return cls(MCPConfig(enabled=False))
        return cls(MCPConfig(**cfg))

    async def web_search(self, query: str) -> List[Dict[str, str]]:
        if not self.enabled:
            return []
        payload = {
            "tool": self.cfg.tool_name_web_search,
            "args": {"query": query, "top_k": int(self.cfg.top_k)},
        }
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s) as client:
            r = await client.post(f"{self.cfg.base_url.rstrip('/')}/call", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("results", [])
