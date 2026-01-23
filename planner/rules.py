import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger("genai.planner.rules")


@dataclass
class PlannerViolations:
    code: str
    message: str
    detail: Optional[Dict[str, Any]] = None


class PlannerRules:
    """Deterministic guardrails for the planner.

    Important: Configuration must live in config.yaml (project convention). We therefore
    build rules directly from the `planner:` section (and optional nested `planner.rules:`).
    """

    def __init__(self, planner_cfg: Dict[str, Any]):
        planner_cfg = planner_cfg or {}
        # Optional nested structure `planner.rules` to keep config tidy, but defaults
        # also accept top-level `planner.max_steps`, etc.
        nested = planner_cfg.get("rules") if isinstance(planner_cfg.get("rules"), dict) else {}

        def _get(name: str, default: Any) -> Any:
            return nested.get(name, planner_cfg.get(name, default))

        self.max_steps = int(_get("max_steps", 40))
        self.max_tool_calls = int(_get("max_tool_calls", 15))
        self.max_dependency_depth = int(_get("max_dependency_depth", 15))
        self.allow_optional_tool_calls = bool(_get("allow_optional_tool_calls", False))
        self.default_locale = str(_get("default_locale", "pt"))


class PlannerRuleEngine:
    def __init__(self, rules: PlannerRules):
        self.rules = rules

    def apply(self, planner_output: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PlannerViolations]]:
        """
        Apply deterministic invariants and caps to a planner_output dictionary.
        Returns the possibly-modified planner_output and a list of violations.
        """
        violations: List[PlannerViolations] = []
        po = planner_output or {}

        plan = po.get("plan") or {}
        steps: List[Dict[str, Any]] = list(plan.get("steps") or [])

        # Cap total steps
        if len(steps) > self.rules.max_steps:
            violations.append(
                PlannerViolations(
                    code="CAP_MAX_STEPS",
                    message=f"Planner steps capped from {len(steps)} to {self.rules.max_steps}.",
                    detail={"original": len(steps), "capped_to": self.rules.max_steps},
                )
            )
            steps = steps[: self.rules.max_steps]

        # Cap tool_calls
        tool_call_ids = [s.get("id") for s in steps if s.get("type") == "tool_call"]
        if len(tool_call_ids) > self.rules.max_tool_calls:
            # Keep first N tool_calls and drop the remainder by converting to compose? safer: remove extra tool_calls.
            keep = set(tool_call_ids[: self.rules.max_tool_calls])
            new_steps: List[Dict[str, Any]] = []
            dropped = 0
            for s in steps:
                if s.get("type") == "tool_call" and s.get("id") not in keep:
                    dropped += 1
                    continue
                new_steps.append(s)
            if dropped:
                violations.append(
                    PlannerViolations(
                        code="CAP_MAX_TOOL_CALLS",
                        message=f"Dropped {dropped} tool_call steps over the configured cap.",
                        detail={"dropped": dropped, "cap": self.rules.max_tool_calls},
                    )
                )
            steps = new_steps

        # Enforce optional tool_call policy (defense-in-depth)
        if not self.rules.allow_optional_tool_calls:
            for s in steps:
                if s.get("type") == "tool_call" and bool(s.get("optional", False)):
                    s["optional"] = False
                    violations.append(
                        PlannerViolations(
                            code="OPTIONAL_TOOL_CALL_FORBIDDEN",
                            message="tool_call steps cannot be optional under current policy; forced optional=false.",
                            detail={"step_id": s.get("id")},
                        )
                    )

        plan["steps"] = steps
        po["plan"] = plan
        return po, violations
