# validator/validator.py
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("genai.validator")


class ValidationFailedError(RuntimeError):
    """
    Raised when the plan fails validation and execution must stop.
    Keeping this class at module top-level allows stable imports.
    """

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.detail = detail or {}


def _tool_allowed(tool_name: str, tool_policy: Optional[Dict[str, Any]]) -> bool:
    if not tool_policy:
        return True

    mode = str(tool_policy.get("mode") or "denylist").lower()
    allow = set(tool_policy.get("allow") or [])
    deny = set(tool_policy.get("deny") or [])

    if mode == "allowlist":
        return tool_name in allow
    return tool_name not in deny


def _detect_cycle(step_deps: Dict[str, List[str]]) -> Optional[List[str]]:
    """
    Returns a cycle path if a cycle exists, otherwise None.
    """
    visiting: Set[str] = set()
    visited: Set[str] = set()
    stack: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        visiting.add(node)
        stack.append(node)
        for dep in step_deps.get(node, []):
            if dep not in step_deps:
                continue
            if dep in visiting:
                # extract cycle
                if dep in stack:
                    idx = stack.index(dep)
                    return stack[idx:] + [dep]
                return [node, dep, node]
            if dep not in visited:
                cyc = dfs(dep)
                if cyc:
                    return cyc
        visiting.remove(node)
        visited.add(node)
        stack.pop()
        return None

    for n in step_deps.keys():
        if n not in visited:
            cyc = dfs(n)
            if cyc:
                return cyc
    return None


class PlanValidator:
    """
    Produces ValidatorOutput payload as per internal-json.json.
    Expects ValidatorInput payload:
      { planner_output, tool_policy, discovered_tools? }
    """

    def __init__(
        self,
        mcp_client: Optional[Any] = None,
        *,
        max_steps: int = 40,
        max_tool_calls: int = 15,
        allow_optional_tool_calls: bool = False,
    ):
        self.mcp = mcp_client
        self.max_steps = int(max_steps)
        self.max_tool_calls = int(max_tool_calls)
        self.allow_optional_tool_calls = bool(allow_optional_tool_calls)

    async def validate(self, validator_input: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        missing_steps: List[str] = []

        planner_output = validator_input.get("planner_output") or {}
        tool_policy = validator_input.get("tool_policy") or {}

        plan = planner_output.get("plan") or {}
        steps = list(plan.get("steps") or [])

        if not steps:
            errors.append("Plan has no steps.")
            payload = {"validation": {"is_valid": False, "errors": errors, "warnings": warnings, "missing_steps": missing_steps}, "plan": {"steps_ready": []}}
            raise ValidationFailedError("Plan validation failed", detail=payload)

        # Hard caps (fail fast): do not accept unbounded plans
        if len(steps) > self.max_steps:
            errors.append(f"Plan exceeds max_steps={self.max_steps} (got {len(steps)}).")

        tool_calls = [s for s in steps if s.get("type") == "tool_call"]
        if len(tool_calls) > self.max_tool_calls:
            errors.append(f"Plan exceeds max_tool_calls={self.max_tool_calls} (got {len(tool_calls)}).")

        # Unique IDs
        ids: Set[str] = set()
        for s in steps:
            sid = s.get("id")
            if not sid:
                errors.append("Step without 'id'.")
                continue
            sid = str(sid)
            if sid in ids:
                errors.append(f"Duplicate step id: {sid}")
            ids.add(sid)

        # Dependencies + tool_call checks
        step_deps: Dict[str, List[str]] = {}
        for s in steps:
            sid = str(s.get("id") or "")
            deps = [str(d) for d in (s.get("dependencies") or [])]
            step_deps[sid] = deps
            for d in deps:
                if d not in ids:
                    errors.append(f"Step '{sid}' depends on missing step '{d}'.")
                if d == sid:
                    errors.append(f"Step '{sid}' depends on itself.")

            if s.get("type") == "tool_call":
                cap = str(s.get("capability") or "").strip()
                if not cap:
                    errors.append(f"tool_call step '{sid}' missing capability.")
                else:
                    if not _tool_allowed(cap, tool_policy):
                        errors.append(f"tool_call step '{sid}' uses disabled tool '{cap}' by tool_policy.")
                if (not self.allow_optional_tool_calls) and bool(s.get("optional", False)):
                    errors.append(f"tool_call step '{sid}' cannot be optional under current policy.")

        # Cycle detection
        cyc = _detect_cycle(step_deps)
        if cyc:
            errors.append(f"Plan dependency cycle detected: {' -> '.join(cyc)}")

        # Optional MCP discovery (defense-in-depth)
        # Prefer discovered_tools from input; fall back to MCP call if absent.
        tool_names: Optional[Set[str]] = None
        discovered = validator_input.get("discovered_tools")
        if isinstance(discovered, list):
            tool_names = {str(t.get("name")) for t in discovered if isinstance(t, dict) and t.get("name")}
        elif self.mcp is not None:
            try:
                tools = await self.mcp.list_tools()
                tool_names = {str(t.get("name")) for t in tools if isinstance(t, dict) and t.get("name")}
            except Exception as e:
                warnings.append(f"Could not verify MCP tools (discovery failed): {e}")

        if tool_names is not None:
            for s in steps:
                if s.get("type") == "tool_call":
                    cap = str(s.get("capability") or "").strip()
                    if cap and cap not in tool_names:
                        errors.append(f"tool_call step '{s.get('id')}' references unknown tool '{cap}' (not in MCP).")

        is_valid = len(errors) == 0

        payload = {
            "validation": {"is_valid": is_valid, "errors": errors, "warnings": warnings, "missing_steps": missing_steps},
            # Contract: steps_ready is a list of step IDs (strings), not Step objects.
            "plan": {"steps_ready": [str(s.get("id")) for s in steps if s.get("id")] if is_valid else []},
        }

        if not is_valid:
            raise ValidationFailedError("Plan validation failed", detail=payload)

        return payload
