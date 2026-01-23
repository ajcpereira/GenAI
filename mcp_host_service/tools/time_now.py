# mcp_host_service/tools/time_now.py
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict

from mcp_host_service.tool_types import ToolSpec


class TimeNowTool:
    spec = ToolSpec(
        name="time.now",
        description="Return current date/time for a requested timezone (defaults to system timezone).",
        input_schema={
            "type": "object",
            "properties": {"timezone": {"type": ["string", "null"], "minLength": 1}},
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "required": ["iso", "timezone"],
            "properties": {
                "iso": {"type": "string", "minLength": 1},
                "timezone": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
    )

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tz_name = inputs.get("timezone")
        if tz_name:
            tz = ZoneInfo(str(tz_name))
            now = datetime.now(tz)
            return {"iso": now.isoformat(), "timezone": str(tz_name)}

        now = datetime.now().astimezone()
        tz = now.tzinfo
        tz_label = getattr(tz, "key", None) or str(tz) or "system"
        return {"iso": now.isoformat(), "timezone": tz_label}
