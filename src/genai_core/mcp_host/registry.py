from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


ToolFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class ToolSpec:
    name: str
    handler: ToolFn
    description: str = ""


class ToolRegistry:
    """In-process tool registry for the Python MCP host.

    New tools/agents can be added by registering additional ToolSpecs.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def describe(self) -> Dict[str, Any]:
        return {
            "tools": [
                {"name": t.name, "description": t.description}
                for t in self._tools.values()
            ]
        }
