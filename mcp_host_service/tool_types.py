# mcp_host_service/tool_types.py
from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class Tool(Protocol):
    spec: ToolSpec

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...
