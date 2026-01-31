# executor/executor.py
import json
import logging
from typing import Any, Dict, List, Optional, Set

from executor.mcp_client import MCPClient

logger = logging.getLogger("genai.executor")


class Executor:
    def __init__(
        self,
        mcp_base_url: str = "http://127.0.0.1:8010",
        mcp_timeout_s: float = 10.0,
        caller_id: str = "orchestrator",
        *,
        max_steps: int = 40,
        max_input_bytes_per_step: int = 64_000,
        allow_optional_tool_calls: bool = False,
    ):
        self.mcp_base_url = mcp_base_url
        self.max_steps = int(max_steps)
        self.max_input_bytes_per_step = int(max_input_bytes_per_step)
        self.allow_optional_tool_calls = bool(allow_optional_tool_calls)

        self.mcp: Optional[MCPClient] = None
        try:
            self.mcp = MCPClient(base_url=mcp_base_url, timeout_s=mcp_timeout_s, caller_id=caller_id)
        except Exception as e:
            logger.warning("mcp_client_init_failed", extra={"error": str(e), "mcp_base_url": mcp_base_url})
            self.mcp = None

    @staticmethod
    def _toposort_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Kahn's algorithm
        by_id: Dict[str, Dict[str, Any]] = {}
        indeg: Dict[str, int] = {}
        out_edges: Dict[str, List[str]] = {}

        for s in steps:
            sid = str(s.get("id") or "")
            if not sid:
                continue
            by_id[sid] = s
            indeg[sid] = 0
            out_edges[sid] = []

        for s in steps:
            sid = str(s.get("id") or "")
            if not sid:
                continue
            deps = s.get("dependencies") or []
            for d in deps:
                d = str(d)
                if d in by_id:
                    indeg[sid] += 1
                    out_edges[d].append(sid)

        q = [sid for sid, deg in indeg.items() if deg == 0]
        ordered: List[str] = []
        while q:
            n = q.pop(0)
            ordered.append(n)
            for m in out_edges.get(n, []):
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)

        if len(ordered) != len(by_id):
            remaining = [sid for sid, deg in indeg.items() if deg > 0]
            raise RuntimeError(f"Plan has cyclic dependencies or invalid graph. Remaining: {remaining}")

        return [by_id[sid] for sid in ordered]

    @staticmethod
    def _deps_ok(step: Dict[str, Any], status_by_id: Dict[str, str]) -> bool:
        deps = step.get("dependencies") or []
        for d in deps:
            if status_by_id.get(str(d)) != "success":
                return False
        return True

    @staticmethod
    def _compose_context(user_message: str, executed: List[Dict[str, Any]], planner_intent: Optional[Dict[str, Any]] = None) -> str:
        intent_summary = ""
        locale = "pt"
        if isinstance(planner_intent, dict):
            intent_summary = str(planner_intent.get("summary") or "")
            locale = str(planner_intent.get("locale") or "pt")

        lines: List[str] = []
        if intent_summary:
            lines.append(("INTENÇÃO: " if locale == "pt" else "INTENT: ") + intent_summary)

        lines.append("RESULTADOS DE EXECUÇÃO:" if locale == "pt" else "EXECUTION RESULTS:")
        for s in executed:
            sid = s.get("id")
            st = s.get("status")
            if st == "success":
                lines.append(f"- {sid}: success => {s.get('output')}")
            elif st == "failed":
                lines.append(f"- {sid}: failed => {s.get('error')}")
            else:
                lines.append(f"- {sid}: {st}")

        lines.append("")
        lines.append("PERGUNTA DO UTILIZADOR:" if locale == "pt" else "USER QUESTION:")
        lines.append(user_message)
        return "\n".join(lines)

    def _inputs_within_limits(self, step: Dict[str, Any]) -> bool:
        inputs = step.get("inputs")
        if inputs is None:
            return True
        try:
            b = json.dumps(inputs, ensure_ascii=False).encode("utf-8")
            return len(b) <= self.max_input_bytes_per_step
        except Exception:
            # If inputs cannot be serialized, treat as invalid/too large
            return False

    async def execute(self, executor_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects ExecutorInput payload:
          { planner_output, request_context: {request_id, user_message} }
        """
        planner_output = executor_input.get("planner_output") or {}
        rc = executor_input.get("request_context") or {}
        request_id = str(rc.get("request_id") or "unknown")
        user_message = str(rc.get("user_message") or "")

        intent_summary = str((planner_output.get("user_intent") or {}).get("summary") or "unknown")
        steps = list((planner_output.get("plan") or {}).get("steps") or [])

        # Defense in depth
        if len(steps) > self.max_steps:
            msg = f"Executor guardrail: plan has {len(steps)} steps, exceeds max_steps={self.max_steps}."
            logger.error("executor_guardrail", extra={"request_id": request_id, "error": msg})
            return {"intent": intent_summary, "steps_executed": [{"id": "executor_guardrail", "status": "failed", "output": None, "error": msg}]}

        ordered_steps = self._toposort_steps(steps) if steps else []

        executed: List[Dict[str, Any]] = []
        status_by_id: Dict[str, str] = {}

        for step in ordered_steps:
            sid = str(step.get("id") or "unknown_step")
            stype = step.get("type")
            optional = bool(step.get("optional", False))

            # Enforce policy: tool_call cannot be optional unless explicitly allowed
            if stype == "tool_call" and optional and not self.allow_optional_tool_calls:
                optional = False

            if not self._inputs_within_limits(step):
                rec = {"id": sid, "status": "failed", "output": None, "error": "Step inputs exceed size limits or are invalid"}
                executed.append(rec)
                status_by_id[sid] = "failed"
                continue

            if not self._deps_ok(step, status_by_id):
                if optional:
                    rec = {"id": sid, "status": "skipped", "output": None, "error": None}
                    executed.append(rec)
                    status_by_id[sid] = "skipped"
                    continue
                rec = {"id": sid, "status": "failed", "output": None, "error": "Dependency failed"}
                executed.append(rec)
                status_by_id[sid] = "failed"
                continue

            try:
                if stype == "tool_call":
                    cap = str(step.get("capability") or "").strip()
                    inputs = step.get("inputs") or {}

                    if not self.mcp:
                        rec = {"id": sid, "status": "failed", "output": None, "error": "MCP client not available"}
                        executed.append(rec)
                        status_by_id[sid] = "failed"
                        continue

                    # Persist raw tool I/O deterministically for auditability.
                    # Output schema allows arbitrary object for "output".
                    tool_call_obj = {"capability": cap, "inputs": inputs}

                    if hasattr(self.mcp, "run_tool_with_trace"):
                        traced = await self.mcp.run_tool_with_trace(cap, inputs, request_id=request_id)  # type: ignore[attr-defined]
                        out = traced.get("response")
                        http_trace = traced.get("trace")
                        if traced.get("ok"):
                            rec = {
                                "id": sid,
                                "status": "success",
                                "output": {"tool_call": tool_call_obj, "http": http_trace, "data": out},
                                "error": None,
                            }
                        else:
                            rec = {
                                "id": sid,
                                "status": "failed",
                                "output": {"tool_call": tool_call_obj, "http": http_trace, "data": out},
                                "error": str(traced.get("error") or "tool_call_failed"),
                            }
                    else:
                        out = await self.mcp.run_tool(cap, inputs, request_id=request_id)
                        rec = {
                            "id": sid,
                            "status": "success",
                            "output": {"tool_call": tool_call_obj, "http": None, "data": out},
                            "error": None,
                        }
                    executed.append(rec)
                    status_by_id[sid] = str(rec.get("status"))

                elif stype == "compose":
                    context = self._compose_context(
                        user_message=str(user_message or (step.get("inputs") or {}).get("message") or ""),
                        executed=executed,
                        planner_intent=(planner_output.get("user_intent") or {}),
                    )
                    rec = {"id": sid, "status": "success", "output": {"context": context}, "error": None}
                    executed.append(rec)
                    status_by_id[sid] = "success"

                else:
                    rec = {"id": sid, "status": "skipped", "output": None, "error": None}
                    executed.append(rec)
                    status_by_id[sid] = "skipped"

            except Exception as e:
                rec = {"id": sid, "status": "failed", "output": None, "error": str(e)}
                executed.append(rec)
                status_by_id[sid] = "failed"

        return {"intent": intent_summary, "steps_executed": executed}
