# orchestrator/orchestrator.py
import logging
import json
import time
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from planner.planner import Planner
from executor.executor import Executor
from responder.responder import Responder
from validator.validator import PlanValidator
from validator.output_validator import OutputValidator
from utils.common import now_iso, validate_json
from utils.lang import detect_language

logger = logging.getLogger("genai.orchestrator")


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schema_version": meta.get("schema_version", "1.1"),
        "message_type": meta.get("message_type", "internal"),
        "request_id": meta.get("request_id", "unknown"),
        "timestamp": meta.get("timestamp", now_iso()),
        "source": meta.get("source", "orchestrator"),
        "user_id": meta.get("user_id", None),
        "session_id": meta.get("session_id", None),
        "timings_ms": meta.get("timings_ms", {}) or {},
    }
    if "trace" in meta and meta["trace"] is not None:
        tr = meta["trace"] or {}
        out["trace"] = {"trace_id": tr.get("trace_id", None), "span_id": tr.get("span_id", None)}
    return out


def _normalize_tool_name(name: str) -> str:
    """
    Normalização tolerante apenas para matching de enabled_tools (UI/API boundary).
    Internamente, a tool canónica continua a ser tool.name do discovery.
    """
    return str(name or "").strip().lower().replace("_", ".")


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        responder: Responder,
        contract_bundle: Dict[str, Any],
        confidence_threshold: float = 0.8,
        max_steps: int = 40,
        max_tool_calls: int = 15,
        reject_unknown_enabled_tools: bool = True,
        max_replans: int = 2,
    ):
        self.planner = planner
        self.executor = executor
        self.responder = responder

        self.confidence_threshold = float(confidence_threshold)
        self.max_steps = int(max_steps)
        self.max_tool_calls = int(max_tool_calls)
        self.reject_unknown_enabled_tools = bool(reject_unknown_enabled_tools)
        self.max_replans = int(max_replans)

        self.bundle = contract_bundle
        self.schemas = contract_bundle["schemas"]

        self.envelope_schema = self.schemas["Envelope"]
        self.metadata_schema = self.schemas["Metadata"]
        self.user_request_schema = self.schemas["UserRequestPayload"]

        self.planner_input_schema = self.schemas.get("PlannerInput")
        self.planner_output_schema = self.schemas["PlannerOutput"]
        self.validator_input_schema = self.schemas.get("ValidatorInput")
        self.validator_output_schema = self.schemas["ValidatorOutput"]
        self.executor_input_schema = self.schemas.get("ExecutorInput")
        self.executor_result_schema = self.schemas["ExecutorResult"]
        self.final_llm_input_schema = self.schemas["FinalLLMInput"]
        self.answer_payload_schema = self.schemas["AnswerPayload"]
        self.error_payload_schema = self.schemas["ErrorPayload"]

        self.plan_validator = PlanValidator(
            getattr(self.executor, "mcp", None),
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            allow_optional_tool_calls=False,
        )
        self.output_validator = OutputValidator(contract_bundle)

    def _make_envelope(
        self,
        base_metadata: Dict[str, Any],
        message_type: str,
        source: str,
        payload: Dict[str, Any],
        payload_schema: Dict[str, Any],
        timings_ms: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        md = _sanitize_metadata(base_metadata)
        md["message_type"] = message_type
        md["source"] = source
        md["timestamp"] = now_iso()
        if timings_ms is not None:
            md["timings_ms"] = timings_ms

        env = {"metadata": md, "payload": payload}
        validate_json(self.metadata_schema, env["metadata"], bundle=self.bundle)
        validate_json(payload_schema, env["payload"], bundle=self.bundle)
        validate_json(self.envelope_schema, env, bundle=self.bundle)
        return env

    def _log_envelope(self, stage: str, env: Dict[str, Any]) -> None:
        md = env.get("metadata") or {}
        logger.info(
            "envelope_stage",
            extra={
                "stage": stage,
                "request_id": md.get("request_id"),
                "message_type": md.get("message_type"),
                "source": md.get("source"),
                "timings_ms": md.get("timings_ms"),
            },
        )

    async def _discover_tools(self, request_id: str) -> List[Dict[str, Any]]:
        if not getattr(self.executor, "mcp", None):
            return []
        try:
            tools = await self.executor.mcp.list_tools()  # type: ignore[union-attr]
            return list(tools or [])
        except Exception as e:
            logger.warning("mcp_discovery_failed", extra={"request_id": request_id, "error": str(e)})
            return []

    @staticmethod
    def _available_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
        names: List[str] = []
        for t in tools or []:
            n = str(t.get("name") or "").strip()
            if n:
                names.append(n)
        # dedupe stable
        seen = set()
        out: List[str] = []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out

    @staticmethod
    def _validate_enabled_tools(enabled: List[str], discovered_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        enabled = list(enabled or [])
        available = Orchestrator._available_tool_names(discovered_tools)

        available_norm_map = {_normalize_tool_name(n): n for n in available}  # norm -> canonical
        enabled_norm = [_normalize_tool_name(x) for x in enabled]

        matched: List[str] = []
        unknown: List[str] = []

        for raw, norm in zip(enabled, enabled_norm):
            canonical = available_norm_map.get(norm)
            if canonical:
                matched.append(canonical)
            else:
                unknown.append(raw)

        # suggestions (UI-friendly)
        suggestions: Dict[str, List[str]] = {}
        avail_norms = list(available_norm_map.keys())
        for raw in unknown:
            norm = _normalize_tool_name(raw)
            close_norms = get_close_matches(norm, avail_norms, n=3, cutoff=0.6)
            suggestions[raw] = [available_norm_map[x] for x in close_norms if x in available_norm_map]

        return {
            "is_valid": len(unknown) == 0,
            "requested": enabled,
            "matched": matched,         # canonical matches
            "unknown": unknown,
            "available": available,     # canonical list
            "suggestions": suggestions,
        }

    @staticmethod
    def _filter_tools_by_enabled(tools: List[Dict[str, Any]], enabled: List[str]) -> List[Dict[str, Any]]:
        enabled_norm = {_normalize_tool_name(x) for x in (enabled or [])}
        out: List[Dict[str, Any]] = []
        for t in tools or []:
            name = str(t.get("name") or "").strip()
            if not name:
                continue
            if _normalize_tool_name(name) in enabled_norm:
                out.append(t)
        return out

    @staticmethod
    def _plan_has_tool_calls(planner_payload: Dict[str, Any]) -> bool:
        steps = (planner_payload.get("plan") or {}).get("steps") or []
        return any(s.get("type") == "tool_call" for s in steps)

    @staticmethod
    def _first_failed_tool_step(planner_payload: Dict[str, Any], exec_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        plan_steps = {str(s.get("id")): s for s in ((planner_payload.get("plan") or {}).get("steps") or [])}
        for s in (exec_payload.get("steps_executed") or []):
            sid = str(s.get("id"))
            if s.get("status") == "failed" and plan_steps.get(sid, {}).get("type") == "tool_call":
                return s
        return None

    @staticmethod
    def _cap_plan_safety(planner_payload: Dict[str, Any], max_steps: int, max_tool_calls: int) -> Dict[str, Any]:
        """
        Defense-in-depth: cap steps/tool_calls at orchestrator boundary even if planner emits more.
        """
        po = dict(planner_payload or {})
        plan = po.get("plan") or {}
        steps = list(plan.get("steps") or [])
        if len(steps) > max_steps:
            steps = steps[:max_steps]
        tool_ids = [s.get("id") for s in steps if s.get("type") == "tool_call"]
        if len(tool_ids) > max_tool_calls:
            keep = set(tool_ids[:max_tool_calls])
            steps = [s for s in steps if not (s.get("type") == "tool_call" and s.get("id") not in keep)]
        plan["steps"] = steps
        po["plan"] = plan
        return po

    async def handle_envelope(self, request_envelope: Dict[str, Any]) -> Dict[str, Any]:
        t_request0 = time.perf_counter()

        validate_json(self.envelope_schema, request_envelope, bundle=self.bundle)
        validate_json(self.metadata_schema, request_envelope["metadata"], bundle=self.bundle)
        validate_json(self.user_request_schema, request_envelope["payload"], bundle=self.bundle)

        base_metadata = request_envelope["metadata"]
        request_id = str(base_metadata.get("request_id") or "unknown")

        self._log_envelope("request", request_envelope)

        payload = request_envelope["payload"] or {}
        user_message = str(payload.get("message") or "").strip()
        enabled_tools = list(payload.get("enabled_tools") or [])

        locale = detect_language(user_message, default="pt").code

        # Discover tools
        t0 = time.perf_counter()
        discovered_tools = await self._discover_tools(request_id=request_id)
        t_discovery = int((time.perf_counter() - t0) * 1000)

        timings: Dict[str, int] = {"mcp_discovery_ms": t_discovery}

        # Validate enabled_tools for UI (fail-fast)
        enabled_validation = self._validate_enabled_tools(enabled_tools, discovered_tools)
        if self.reject_unknown_enabled_tools and not enabled_validation["is_valid"]:
            err_payload = {
                "error": {
                    "code": "TOOL_NOT_AVAILABLE",
                    "message": "One or more requested tools are not available.",
                    "detail": enabled_validation,
                },
                "debug": {"stage": "orchestrator", "timings_ms": timings},
            }
            err_env = self._make_envelope(base_metadata, "error", "orchestrator", err_payload, self.error_payload_schema, timings_ms=timings)
            self._log_envelope("error", err_env)
            return err_env

        allowed_tools = self._filter_tools_by_enabled(discovered_tools, enabled_tools)

        # Build tool policy (runtime allowlist) to constrain planner + validator
        # ToolPolicy.allow must be a list of strings (capability names), not tool objects.
        tool_policy = {
            "mode": "allowlist",
            "allow": [t["name"] for t in allowed_tools],
        }

        # Planner input payload/envelope
        planner_input_payload: Dict[str, Any] = {
            "user_message": user_message,
            "allowed_tools": allowed_tools,
            "enabled_tools": enabled_tools,
            "locale": locale,
        }
        if self.planner_input_schema:
            validate_json(self.planner_input_schema, planner_input_payload, bundle=self.bundle)
        planner_in_env = self._make_envelope(
            base_metadata, "planner_input", "orchestrator", planner_input_payload,
            self.planner_input_schema or self.planner_output_schema, timings_ms=timings,
        )
        self._log_envelope("planner_input", planner_in_env)


        # Planner + Validator (with replan loop)
        last_validation_detail: Optional[Dict[str, Any]] = None
        planner_payload: Dict[str, Any] = {}
        validator_payload: Dict[str, Any] = {}

        for attempt in range(1, self.max_replans + 2):
            # Planner
            t0 = time.perf_counter()
            if attempt > 1 and last_validation_detail:
                # Provide compact feedback to the planner (schema-approved via PlannerInput.replan_feedback)
                errs = (last_validation_detail.get("validation") or {}).get("errors") or []
                warns = (last_validation_detail.get("validation") or {}).get("warnings") or []
                feedback = {
                    "attempt": attempt,
                    "errors": errs,
                    "warnings": warns,
                    "rule": "tool_call steps MUST include non-empty capability equal to an allowed tool name",
                }
                planner_in_env["payload"]["replan_feedback"] = json.dumps(feedback, ensure_ascii=False)
            else:
                planner_in_env["payload"].pop("replan_feedback", None)

            planner_payload = await self.planner.build_plan(planner_in_env["payload"])
            timings["planner_ms"] = int((time.perf_counter() - t0) * 1000)

            planner_payload = self._cap_plan_safety(planner_payload, self.max_steps, self.max_tool_calls)

            planner_env = self._make_envelope(
                base_metadata, "planner_output", "planner", planner_payload, self.planner_output_schema, timings_ms=timings
            )
            self._log_envelope("planner_output", planner_env)

            # Validator input envelope
            validator_input_payload = {
                "planner_output": planner_payload,
                "tool_policy": tool_policy,
                "discovered_tools": discovered_tools,
            }
            if self.validator_input_schema:
                validate_json(self.validator_input_schema, validator_input_payload, bundle=self.bundle)
            validator_in_env = self._make_envelope(
                base_metadata,
                "validator_input",
                "orchestrator",
                validator_input_payload,
                self.validator_input_schema or self.validator_output_schema,
                timings_ms=timings,
            )
            self._log_envelope("validator_input", validator_in_env)

            # Validator
            t0 = time.perf_counter()
            try:
                validator_payload = await self.plan_validator.validate(validator_in_env["payload"])
                timings["validator_ms"] = int((time.perf_counter() - t0) * 1000)
                validator_env = self._make_envelope(
                    base_metadata, "validator_output", "validator", validator_payload, self.validator_output_schema, timings_ms=timings
                )
                self._log_envelope("validator_output", validator_env)
                last_validation_detail = None
                break  # valid plan
            except Exception as e:
                timings["validator_ms"] = int((time.perf_counter() - t0) * 1000)
                detail = getattr(e, "detail", None) or {}
                last_validation_detail = detail

                logger.warning(
                    "replan_required",
                    extra={
                        "request_id": base_metadata.get("request_id"),
                        "attempt": attempt,
                        "max_replans": self.max_replans,
                        "validation_errors": (detail.get("validation") or {}).get("errors") or [],
                    },
                )

                if attempt <= self.max_replans:
                    continue

                err_payload = {
                    "error": {"code": "INVALID_PLAN", "message": str(e), "detail": detail},
                    "debug": {"stage": "validator", "timings_ms": timings},
                }
                err_env = self._make_envelope(base_metadata, "error", "validator", err_payload, self.error_payload_schema, timings_ms=timings)
                self._log_envelope("error", err_env)
                return err_env
# ExecutorInput
        executor_input_payload = {"planner_output": planner_env["payload"], "request_context": {"request_id": request_id, "user_message": user_message}}
        if self.executor_input_schema:
            validate_json(self.executor_input_schema, executor_input_payload, bundle=self.bundle)
        executor_in_env = self._make_envelope(
            base_metadata, "executor_input", "orchestrator", executor_input_payload, self.executor_input_schema or self.executor_result_schema, timings_ms=timings
        )
        self._log_envelope("executor_input", executor_in_env)

        # Executor
        t0 = time.perf_counter()
        exec_payload = await self.executor.execute(executor_in_env["payload"])
        timings["executor_ms"] = int((time.perf_counter() - t0) * 1000)

        exec_env = self._make_envelope(base_metadata, "executor_result", "executor", exec_payload, self.executor_result_schema, timings_ms=timings)
        self._log_envelope("executor_result", exec_env)

        self.output_validator.validate(exec_env["payload"])

        # Hard-failure policy: if a planned tool_call failed, respond with failure
        failed_tool = self._first_failed_tool_step(planner_env["payload"], exec_env["payload"])
        if failed_tool is not None:
            err_msg = str(failed_tool.get("error") or "Tool execution failed")
            answer_payload = {
                "answer": f"Falha ao executar uma ferramenta necessária (step '{failed_tool.get('id')}'): {err_msg}",
                "final_context": {"intent": "tool_execution_failed", "steps_executed": exec_env["payload"]["steps_executed"]},
                "timings_ms": timings,
            }
            validate_json(self.answer_payload_schema, answer_payload, bundle=self.bundle)
            response_env = self._make_envelope(base_metadata, "response", "orchestrator", answer_payload, self.answer_payload_schema, timings_ms=timings)
            self._log_envelope("response", response_env)
            return response_env

        # Derive composed context from compose step output (if present)
        composed_context = None
        for s in exec_env["payload"]["steps_executed"]:
            if (s.get("status") == "success") and isinstance(s.get("output"), dict) and "context" in s["output"]:
                composed_context = s["output"]["context"]

        intent_summary = str((planner_env["payload"].get("user_intent") or {}).get("summary") or "")
        final_intent = composed_context or intent_summary or "final_response"

        final_llm_input = {
            "final_context": {"intent": final_intent, "steps_executed": exec_env["payload"]["steps_executed"]},
            "final_output_spec": {"format": "text", "tone": "enterprise", "constraints": ["concise", "accurate", f"language:{locale}"]},
        }
        validate_json(self.final_llm_input_schema, final_llm_input, bundle=self.bundle)

        final_llm_env = self._make_envelope(base_metadata, "final_llm_input", "orchestrator", final_llm_input, self.final_llm_input_schema, timings_ms=timings)
        self._log_envelope("final_llm_input", final_llm_env)

        # Responder
        t0 = time.perf_counter()
        answer_text = await self.responder.answer(final_llm_env["payload"])
        timings["responder_ms"] = int((time.perf_counter() - t0) * 1000)
        timings["total_ms"] = int((time.perf_counter() - t_request0) * 1000)

        answer_payload = {
            "answer": answer_text,
            "final_context": final_llm_input["final_context"],
            "timings_ms": timings,
        }
        validate_json(self.answer_payload_schema, answer_payload, bundle=self.bundle)

        response_env = self._make_envelope(base_metadata, "response", "orchestrator", answer_payload, self.answer_payload_schema, timings_ms=timings)
        self._log_envelope("response", response_env)
        return response_env