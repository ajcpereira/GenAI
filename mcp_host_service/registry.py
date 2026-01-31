# mcp_host_service/registry.py
import logging
from typing import Any, Dict, List

from jsonschema import Draft202012Validator

from mcp_host_service.tool_types import Tool

logger = logging.getLogger("mcp.registry")


def _validate(schema: Dict[str, Any], instance: Any) -> None:
    Draft202012Validator(schema).validate(instance)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool
        logger.info("tool_registered", extra={"tool": name})

    def list_specs(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in self._tools.values():
            out.append(
                {
                    "name": t.spec.name,
                    "description": t.spec.description,
                    "input_schema": t.spec.input_schema,
                    "output_schema": t.spec.output_schema,
                }
            )
        return out

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    async def run(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.get(name)
        _validate(tool.spec.input_schema, inputs)
        out = await tool.run(inputs)
        _validate(tool.spec.output_schema, out)
        return out
