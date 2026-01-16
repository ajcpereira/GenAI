from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolDef:
    key: str                 # key in YAML under tools.<key>
    tool_name: str           # MCP tool name (what MCP Host registry uses)
    description: str
    tags: List[str]
    intents: List[str]
    enabled: bool
    config: Dict[str, Any]   # full tool config (connection/limits/etc.), excluding mcp transport


class ToolCatalog:
    """
    Reads tools from cfg["tools"] in a config-driven way.
    We keep tools.mcp as the transport config; other entries are logical tools.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self._cfg = cfg or {}
        self._tools_cfg = (self._cfg.get("tools") or {}) if isinstance(self._cfg, dict) else {}

    def list_enabled_tools(self) -> List[ToolDef]:
        out: List[ToolDef] = []
        for key, tcfg in self._tools_cfg.items():
            if key == "mcp":
                continue
            if not isinstance(tcfg, dict):
                continue

            enabled = bool(tcfg.get("enabled", False))
            tool_name = str(tcfg.get("tool_name") or key).strip()
            desc = str(tcfg.get("description") or "").strip()

            routing = tcfg.get("routing") or {}
            tags = [str(x).strip() for x in (routing.get("tags") or []) if str(x).strip()]
            intents = [str(x).strip() for x in (routing.get("intents") or []) if str(x).strip()]

            out.append(
                ToolDef(
                    key=key,
                    tool_name=tool_name,
                    description=desc,
                    tags=tags,
                    intents=intents,
                    enabled=enabled,
                    config=tcfg,
                )
            )

        return [t for t in out if t.enabled]

    def get_tool(self, key: str) -> Optional[ToolDef]:
        key = (key or "").strip()
        if not key:
            return None
        for t in self.list_enabled_tools():
            if t.key == key:
                return t
        return None

    def render_for_router_prompt(self) -> str:
        """
        A compact tool list for the LLM router, derived from YAML.
        """
        tools = self.list_enabled_tools()
        if not tools:
            return "(nenhuma tool MCP habilitada)"

        lines: List[str] = []
        for t in tools:
            lines.append(
                f"- key={t.key} tool_name={t.tool_name}\n"
                f"  description={t.description}\n"
                f"  tags={t.tags}\n"
                f"  intents={t.intents}"
            )
        return "\n".join(lines)
