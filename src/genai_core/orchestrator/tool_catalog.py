from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolDef:
    key: str
    tool_name: str
    enabled: bool
    description: str
    tags: List[str]
    intents: List[str]
    config: Dict[str, Any]


class ToolCatalog:
    """
    Config-driven tool catalog (new format only).

    tools:
      mcp: {...}
      web_search:
        enabled: true
        tool_name: "web_search"
        routing:
          tags: ["sources", "freshness"]
          intents: ["latest_version", "news"]
        limits:
          top_k: 5
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self._tools: Dict[str, ToolDef] = {}
        self._load()

    def _load(self) -> None:
        tools_cfg = self.cfg.get("tools")
        if not isinstance(tools_cfg, dict):
            return

        for key, spec in tools_cfg.items():
            if key == "mcp":
                continue
            if not isinstance(spec, dict):
                continue

            enabled = bool(spec.get("enabled", True))
            tool_name = str(spec.get("tool_name") or key).strip()
            description = str(spec.get("description") or "").strip()

            routing = spec.get("routing") if isinstance(spec.get("routing"), dict) else {}
            tags = routing.get("tags") if isinstance(routing.get("tags"), list) else []
            intents = routing.get("intents") if isinstance(routing.get("intents"), list) else []

            self._tools[key] = ToolDef(
                key=key,
                tool_name=tool_name,
                enabled=enabled,
                description=description,
                tags=[str(x) for x in tags],
                intents=[str(x) for x in intents],
                config=spec,
            )

    def list_enabled_tools(self) -> List[ToolDef]:
        return [t for t in self._tools.values() if t.enabled]

    def get_tool(self, key: str) -> Optional[ToolDef]:
        t = self._tools.get(key)
        return t if (t and t.enabled) else None

    def pick_by_tag(self, tag: str) -> Optional[ToolDef]:
        tag_l = (tag or "").strip().lower()
        for t in self.list_enabled_tools():
            if tag_l in [x.lower() for x in (t.tags or [])]:
                return t
        return None

    def render_for_router_prompt(self) -> str:
        lines: List[str] = []
        for t in self.list_enabled_tools():
            limits = t.config.get("limits") if isinstance(t.config.get("limits"), dict) else {}
            lines.append(
                f"- key={t.key} tool_name={t.tool_name} tags={t.tags} intents={t.intents} limits={limits} desc={t.description[:160]}"
            )
        return "\n".join(lines).strip()
