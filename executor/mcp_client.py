#executor/mcp_client.py
import logging
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
