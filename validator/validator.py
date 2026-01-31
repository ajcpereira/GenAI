# validator/validator.py
import logging
from typing import Any, Dict, List, Optional, Set

from utils.common import validate_json

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
    """Schema-first tool authorization check.

    v1.2 contract uses ToolPolicy:
      - enabled_tools: explicit allowlist
      - deny_tools: explicit denylist (takes precedence)
    Security default: if enabled_tools is empty, no tools are permitted.
    """
    if not tool_policy:
        return False

    enabled = set(tool_policy.get("enabled_tools") or [])
    deny = set(tool_policy.get("deny_tools") or [])

    if tool_name in deny:
        return False
    if not enabled:
        return False
    return tool_name in enabled



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
        rules: Optional[Dict[str, Any]] = None,
    ):
        self.mcp = mcp_client
        self.max_steps = int(max_steps)
        self.max_tool_calls = int(max_tool_calls)
        self.allow_optional_tool_calls = bool(allow_optional_tool_calls)

        # Validator rule catalog (contractual, declared in contract bundle).
        # Output schema expects strings, so we prefix messages with stable rule IDs.
        self._rules = rules or {}
        self._error_prefix = str(self._rules.get("error_prefix") or "PV").strip() or "PV"
        self._warning_prefix = str(self._rules.get("warning_prefix") or "PVW").strip() or "PVW"
        self._templates = self._rules.get("templates") or {}
        self._warning_templates = self._rules.get("warning_templates") or {}

    def _err(self, rule_id: str, default: str, **fmt: Any) -> str:
        tmpl = self._templates.get(rule_id) or default
        try:
            msg = str(tmpl).format(**fmt)
        except Exception:
            msg = default
        return f"[{self._error_prefix}{rule_id}] {msg}"

    def _warn(self, rule_id: str, default: str, **fmt: Any) -> str:
        tmpl = self._warning_templates.get(rule_id) or self._templates.get(rule_id) or default
        try:
            msg = str(tmpl).format(**fmt)
        except Exception:
            msg = default
        return f"[{self._warning_prefix}{rule_id}] {msg}"

    async def validate(self, validator_input: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        missing_steps: List[str] = []

        planner_output = validator_input.get("planner_output") or {}
        tool_policy = validator_input.get("tool_policy") or {}

        plan = planner_output.get("plan") or {}
        steps = list(plan.get("steps") or [])

        if not steps:
            errors.append(self._err("001", "Plan has no steps."))
            payload = {"validation": {"is_valid": False, "errors": errors, "warnings": warnings, "missing_steps": missing_steps}, "plan": {"steps_ready": []}}
            raise ValidationFailedError("Plan validation failed", detail=payload)

        # Hard caps (fail fast): do not accept unbounded plans
        if len(steps) > self.max_steps:
            errors.append(self._err("002", "Plan exceeds max_steps={max_steps} (got {got}).", max_steps=self.max_steps, got=len(steps)))

        tool_calls = [s for s in steps if s.get("type") == "tool_call"]
        if len(tool_calls) > self.max_tool_calls:
            errors.append(self._err("003", "Plan exceeds max_tool_calls={max_tool_calls} (got {got}).", max_tool_calls=self.max_tool_calls, got=len(tool_calls)))

        # Unique IDs
        ids: Set[str] = set()
        for s in steps:
            sid = s.get("id")
            if not sid:
                errors.append(self._err("004", "Step without 'id'."))
                continue
            sid = str(sid)
            if sid in ids:
                errors.append(self._err("005", "Duplicate step id: {id}", id=sid))
            ids.add(sid)

        # Dependencies + tool_call checks
        step_deps: Dict[str, List[str]] = {}
        for s in steps:
            sid = str(s.get("id") or "")
            deps = [str(d) for d in (s.get("dependencies") or [])]
            step_deps[sid] = deps
            for d in deps:
                if d not in ids:
                    errors.append(self._err("006", "Step '{id}' depends on missing step '{dep}'.", id=sid, dep=d))
                if d == sid:
                    errors.append(self._err("007", "Step '{id}' depends on itself.", id=sid))

            if s.get("type") == "tool_call":
                cap = str(s.get("capability") or "").strip()
                if not cap:
                    errors.append(self._err("008", "tool_call step '{id}' missing capability.", id=sid))
                else:
                    if not _tool_allowed(cap, tool_policy):
                        errors.append(self._err("009", "tool_call step '{id}' uses disabled tool '{tool}' by tool_policy.", id=sid, tool=cap))
                if (not self.allow_optional_tool_calls) and bool(s.get("optional", False)):
                    errors.append(self._err("010", "tool_call step '{id}' cannot be optional under current policy.", id=sid))

        # Cycle detection
        cyc = _detect_cycle(step_deps)
        if cyc:
            errors.append(self._err("011", "Plan dependency cycle detected: {cycle}", cycle=" -> ".join(cyc)))

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
                warnings.append(self._warn("001", "Could not verify MCP tools (discovery failed): {error}", error=str(e)))

        if tool_names is not None:
            for s in steps:
                if s.get("type") == "tool_call":
                    cap = str(s.get("capability") or "").strip()
                    if cap and cap not in tool_names:
                        errors.append(self._err("012", "tool_call step '{id}' references unknown tool '{tool}' (not in MCP).", id=str(s.get("id") or ""), tool=cap))

        # Validate tool_call inputs against the discovered tool input_schema (best-effort, defense-in-depth).
        # This prevents avoidable 400s from the MCP host when the planner emits malformed tool inputs.
        tool_schema_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(discovered, list):
            for t in discovered:
                if isinstance(t, dict) and t.get("name") and isinstance(t.get("input_schema"), dict):
                    tool_schema_map[str(t["name"])] = dict(t["input_schema"])

        if tool_schema_map:
            for s in steps:
                if s.get("type") != "tool_call":
                    continue
                cap = str(s.get("capability") or "").strip()
                if not cap:
                    continue
                schema = tool_schema_map.get(cap)
                if not schema:
                    continue
                inputs = s.get("inputs") if isinstance(s, dict) else None
                if inputs is None:
                    inputs = {}
                if not isinstance(inputs, dict):
                    errors.append(self._err("013", "tool_call step '{id}' has non-object inputs.", id=str(s.get("id") or "")))
                    continue
                try:
                    validate_json(schema, inputs, bundle=None)
                except Exception as e:
                    errors.append(
                        self._err(
                            "014",
                            "tool_call step '{id}' inputs do not match tool input_schema for '{tool}': {error}",
                            id=str(s.get("id") or ""),
                            tool=cap,
                            error=str(e),
                        )
                    )

        is_valid = len(errors) == 0

        payload = {
            "validation": {"is_valid": is_valid, "errors": errors, "warnings": warnings, "missing_steps": missing_steps},
            # Contract: steps_ready is a list of step IDs (strings), not Step objects.
            "plan": {"steps_ready": [str(s.get("id")) for s in steps if s.get("id")] if is_valid else []},
        }

        if not is_valid:
            raise ValidationFailedError("Plan validation failed", detail=payload)

        return payload
