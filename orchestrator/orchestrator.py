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

from orchestrator.session_store import SessionStore
from orchestrator.context_manager import ContextManager
from orchestrator.context_policy_classifier import ContextPolicyClassifier, ContextPolicyDecision

logger = logging.getLogger("genai.orchestrator")

SAFE_FALLBACK_PT = "Não tenho informação segura e precisa para responder"
SAFE_FALLBACK_EN = "I don't have reliable, precise information to answer"


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schema_version": meta.get("schema_version", "1.2"),
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
        # NEW (major): persistence + context
        session_store: Optional[SessionStore] = None,
        storage_cfg: Optional[Dict[str, Any]] = None,
        context_manager: Optional[ContextManager] = None,
        context_policy_classifier: Optional[ContextPolicyClassifier] = None,
    ):
        self.planner = planner
        self.executor = executor
        self.responder = responder

        self.confidence_threshold = float(confidence_threshold)
        self.max_steps = int(max_steps)
        self.max_tool_calls = int(max_tool_calls)
        self.reject_unknown_enabled_tools = bool(reject_unknown_enabled_tools)
        self.max_replans = int(max_replans)

        self.session_store = session_store
        self.storage_cfg = storage_cfg or {}
        self.context_manager = context_manager
        self.context_policy_classifier = context_policy_classifier

        self.persist_stages = set(self.storage_cfg.get("persist_stages") or [])
        self.max_envelope_bytes = int(self.storage_cfg.get("max_envelope_bytes", 1048576))

        self.bundle = contract_bundle
        self.schemas = contract_bundle["schemas"]

        # Fail-fast: required schemas must exist in the contract bundle
        required_schema_names = [
            "Envelope",
            "Metadata",
            "UserRequestPayload",
            "PlannerInput",
            "PlannerOutput",
            "ValidatorInput",
            "ValidatorOutput",
            "ExecutorInput",
            "ExecutorResult",
            "FinalLLMInput",
            "AnswerPayload",
            "ErrorPayload",
        ]
        missing = [n for n in required_schema_names if n not in self.schemas]
        if missing:
            raise RuntimeError(f"contract_bundle_missing_schemas: {missing}")

        self.envelope_schema = self.schemas["Envelope"]
        self.metadata_schema = self.schemas["Metadata"]
        self.user_request_schema = self.schemas["UserRequestPayload"]

        self.planner_input_schema = self.schemas["PlannerInput"]
        self.planner_output_schema = self.schemas["PlannerOutput"]
        self.validator_input_schema = self.schemas["ValidatorInput"]
        self.validator_output_schema = self.schemas["ValidatorOutput"]
        self.executor_input_schema = self.schemas["ExecutorInput"]
        self.executor_result_schema = self.schemas["ExecutorResult"]
        self.final_llm_input_schema = self.schemas["FinalLLMInput"]
        self.answer_payload_schema = self.schemas["AnswerPayload"]
        self.error_payload_schema = self.schemas["ErrorPayload"]

        # Optional: schema for context policy observability
        self.context_policy_output_schema = self.schemas.get("ContextPolicyOutput")
        if self.persist_stages:
            # If stage allow-list is enabled, include this new stage to avoid silent drops.
            self.persist_stages.add("context_policy_output")

        self.plan_validator = PlanValidator(
            getattr(self.executor, "mcp", None),
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            allow_optional_tool_calls=False,
            rules=(contract_bundle.get("validator_rules") or {}).get("plan_validator") or {},
        )
        self.output_validator = OutputValidator(contract_bundle)

    @staticmethod
    def _build_validation_failure_answer(*, locale: str, errors: List[str], enabled_tools: List[str]) -> str:
        """Deterministic user-facing message for plan validation failures.

        Security/UX policy: never reveal tools, permissions, internal codes, or validation details.
        """
        loc_pt = str(locale).lower().startswith("pt")
        return SAFE_FALLBACK_PT if loc_pt else SAFE_FALLBACK_EN

    async def _persist(self, stage: str, env: Dict[str, Any]) -> None:
        if not self.session_store:
            return
        if self.persist_stages and stage not in self.persist_stages:
            return
        await self.session_store.persist_envelope(stage=stage, envelope=env, max_bytes=self.max_envelope_bytes)

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
            "matched": matched,  # canonical matches
            "unknown": unknown,
            "available": available,  # canonical list
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



    async def handle_envelope(self, request_envelope: Dict[str, Any]) -> Dict[str, Any]:
        t_request0 = time.perf_counter()

        # Request-scoped timings propagated across envelopes.
        # Always initialise deterministically to avoid NameError regressions.
        timings: Dict[str, int] = dict((request_envelope.get("metadata") or {}).get("timings_ms") or {})

        validate_json(self.envelope_schema, request_envelope, bundle=self.bundle)
        validate_json(self.metadata_schema, request_envelope["metadata"], bundle=self.bundle)
        validate_json(self.user_request_schema, request_envelope["payload"], bundle=self.bundle)

        base_metadata = request_envelope["metadata"]
        request_id = str(base_metadata.get("request_id") or "unknown")
        session_id = str(base_metadata.get("session_id") or "")
        user_id = base_metadata.get("user_id")

        self._log_envelope("request", request_envelope)
        await self._persist("request", request_envelope)

        payload = request_envelope["payload"] or {}
        user_message = str(payload.get("message") or "").strip()
        enabled_tools = list(payload.get("enabled_tools") or [])

        locale = detect_language(user_message, default="pt").code

        # Persistence: ensure session exists + append current user message.
        current_user_seq = 0
        if self.session_store and session_id:
            await self.session_store.touch_session(session_id=session_id, user_id=user_id, meta={"locale": locale})
            current_user_seq = await self.session_store.append_message(
                session_id=session_id, user_id=user_id, request_id=request_id, role="user", content=user_message
            )


        # Build conversation context (major feature)
        # Decide whether to include recent conversation context (standalone vs recent)
        decision: ContextPolicyDecision | None = None
        ctx_error: Optional[str] = None
        ctx_debug = "CONTEXT_POLICY: standalone"
        conversation_context = ""

        if self.context_policy_classifier is not None:
            try:
                decision = await self.context_policy_classifier.classify(user_message)
            except Exception as e:
                decision = None
                ctx_error = f"classifier_error: {type(e).__name__}"

        if decision is None:
            # If classifier is missing or failed, default to standalone deterministically.
            ctx_debug = "CONTEXT_POLICY: fallback(standalone)"
            if ctx_error is None:
                ctx_error = "classifier_unavailable"
            decision = ContextPolicyDecision(mode="standalone", recent_turns=0, confidence=0.0)
        elif decision.mode == "recent":
            if self.context_manager is None:
                # Deterministic guardrail: don't crash if context manager isn't wired.
                ctx_debug = "CONTEXT_POLICY: recent_requested_but_context_manager_missing -> fallback(standalone)"
                ctx_error = "context_manager_missing"
                decision = ContextPolicyDecision(mode="standalone", recent_turns=0, confidence=decision.confidence)
            else:
                try:
                    conversation_context, ctx_debug = await self.context_manager.build_context(
                        session_id=session_id,
                        max_turns=decision.recent_turns,
                    )
                except Exception as e:
                    conversation_context = ""
                    ctx_debug = "CONTEXT_POLICY: build_context_failed -> fallback(standalone)"
                    ctx_error = f"build_context_error: {type(e).__name__}"
                    decision = ContextPolicyDecision(mode="standalone", recent_turns=0, confidence=decision.confidence)
        else:
            conversation_context = ""
            ctx_debug = "CONTEXT_POLICY: standalone"

        # Persist decision envelope for traceability (schema-driven; never breaks request flow)
        if self.context_policy_output_schema is not None:
            try:
                ctx_out_env = self._make_envelope(
                    base_metadata,
                    "context_policy_output",
                    "orchestrator",
                    {
                        "current_user_message": user_message,
                        "decision": {
                            "mode": decision.mode,
                            "recent_turns": decision.recent_turns,
                            "confidence": decision.confidence,
                        },
                        "error": ctx_error,
                    },
                    self.context_policy_output_schema,
                )
                self._log_envelope("context_policy_output", ctx_out_env)
                await self._persist("context_policy_output", ctx_out_env)
            except Exception:
                # Never crash request flow because of observability.
                pass

        # ----------------
        # Tool discovery + tool policy
        # ----------------
        discovered_tools: List[Dict[str, Any]] = await self._discover_tools(request_id=request_id)
        enabled_validation = self._validate_enabled_tools(enabled_tools, discovered_tools)

        if self.reject_unknown_enabled_tools and (not enabled_validation.get("is_valid", True)):
            # Controlled, schema-valid error response. Never crash.
            err_payload = {
                "error": {
                    "code": "INVALID_ENABLED_TOOLS",
                    "message": "One or more enabled_tools are not available.",
                    "detail": {
                        "unknown": enabled_validation.get("unknown") or [],
                        "available": enabled_validation.get("available") or [],
                        "suggestions": enabled_validation.get("suggestions") or {},
                    },
                },
                "debug": {"stage": "orchestrator"},
            }
            error_env = self._make_envelope(base_metadata, "error", "orchestrator", err_payload, self.error_payload_schema)
            self._log_envelope("response", error_env)
            await self._persist("response", error_env)
            return error_env

        # Canonical tool allow-list (exact names from discovery).
        enabled_canonical: List[str] = list(enabled_validation.get("matched") or [])
        allowed_tools: List[Dict[str, Any]] = self._filter_tools_by_enabled(discovered_tools, enabled_canonical)

        tool_policy: Dict[str, Any] = {"enabled_tools": enabled_canonical, "deny_tools": []}

        # ----------------
        # Planner input payload (schema-driven)
        # ----------------
        recent_user_messages: List[Dict[str, Any]] = []
        if self.session_store and session_id and current_user_seq:
            try:
                rows = await self.session_store.get_recent_messages(
                    session_id=session_id,
                    before_seq=current_user_seq,
                    limit=12,
                )
                # Keep only user messages; oldest -> newest.
                rows = [r for r in rows if getattr(r, "role", "") == "user"]
                rows = list(reversed(rows))
                for r in rows:
                    item: Dict[str, Any] = {"id": str(getattr(r, "seq")), "text": str(getattr(r, "content"))}
                    rid = getattr(r, "request_id", None)
                    if rid:
                        item["request_id"] = str(rid)
                    recent_user_messages.append(item)
            except Exception:
                # Context is optional; schema still requires the array key.
                recent_user_messages = []

        # Constraints should come from config-driven rule engine; no heuristics.
        max_dependency_depth = int(getattr(getattr(self.planner, "rules", None), "max_dependency_depth", 15))

        planner_input_payload: Dict[str, Any] = {
            "schema_version": str(base_metadata.get("schema_version") or "1.2"),
            "request": {
                "request_id": request_id,
                "session_id": session_id or "unknown_session",
                "timestamp": now_iso(),
                "locale": str(locale),
                "current_user_message": user_message,
            },
            "context": {
                "recent_user_messages": recent_user_messages,
                "memory_facts": [],
            },
            "tools": {
                "catalog": allowed_tools,
                "policy": tool_policy,
            },
            "constraints": {
                "max_steps": self.max_steps,
                "max_tool_calls": self.max_tool_calls,
                "max_dependency_depth": max_dependency_depth,
            },
        }

        planner_in_env = self._make_envelope(
            base_metadata,
            "planner_input",
            "orchestrator",
            planner_input_payload,
            self.planner_input_schema or self.planner_output_schema,
            timings_ms=timings,
        )
        self._log_envelope("planner_input", planner_in_env)
        await self._persist("planner_input", planner_in_env)

        # Planner -> Validator -> Executor: single unified replan loop.
        # We replan deterministically when:
        #   - plan validation fails (schema/tool_policy/tool input_schema)
        #   - a required tool execution fails (tool backend error / invalid tool inputs)
        # This keeps the orchestrator free of heuristics: it never changes the plan itself;
        # it only asks the Planner to replan with explicit feedback.

        last_feedback: Optional[Dict[str, Any]] = None
        planner_env: Optional[Dict[str, Any]] = None
        exec_env: Optional[Dict[str, Any]] = None
        final_context_obj: Optional[Dict[str, Any]] = None

        for attempt in range(1, self.max_replans + 2):
            # ----------------
            # Planner
            # ----------------
            t0 = time.perf_counter()
            if last_feedback:
                planner_in_env["payload"]["replan_feedback"] = last_feedback
            else:
                planner_in_env["payload"].pop("replan_feedback", None)

            planner_payload = await self.planner.build_plan(planner_in_env["payload"])
            timings["planner_ms"] = int((time.perf_counter() - t0) * 1000)

            planner_env = self._make_envelope(
                base_metadata, "planner_output", "planner", planner_payload, self.planner_output_schema, timings_ms=timings
            )
            self._log_envelope("planner_output", planner_env)
            await self._persist("planner_output", planner_env)

            # ----------------
            # Validator
            # ----------------
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
            await self._persist("validator_input", validator_in_env)

            t0 = time.perf_counter()
            try:
                validator_payload = await self.plan_validator.validate(validator_in_env["payload"])
                timings["validator_ms"] = int((time.perf_counter() - t0) * 1000)
                validator_env = self._make_envelope(
                    base_metadata, "validator_output", "validator", validator_payload, self.validator_output_schema, timings_ms=timings
                )
                self._log_envelope("validator_output", validator_env)
                await self._persist("validator_output", validator_env)
            except Exception as e:
                timings["validator_ms"] = int((time.perf_counter() - t0) * 1000)
                detail = getattr(e, "detail", None) or {}
                errors = (detail.get("validation") or {}).get("errors") or []
                warnings = (detail.get("validation") or {}).get("warnings") or []

                logger.warning(
                    "replan_required",
                    extra={
                        "request_id": base_metadata.get("request_id"),
                        "attempt": attempt,
                        "max_replans": self.max_replans,
                        "validation_errors": errors,
                    },
                )

                # If we are out of replans, do not crash the user with a validator error.
                # Industry-grade UX: provide a deterministic, actionable response.
                if attempt > self.max_replans:
                    answer_text = self._build_validation_failure_answer(locale=locale, errors=errors, enabled_tools=enabled_tools)
                    answer_payload = {
                        "answer": answer_text,
                        "final_context": {
                            "intent": "invalid_plan",
                            "steps_executed": [],
                            **({"conversation_context": conversation_context} if conversation_context else {}),
                        },
                        "timings_ms": timings,
                    }
                    validate_json(self.answer_payload_schema, answer_payload, bundle=self.bundle)
                    response_env = self._make_envelope(
                        base_metadata, "response", "orchestrator", answer_payload, self.answer_payload_schema, timings_ms=timings
                    )
                    self._log_envelope("response", response_env)
                    await self._persist("response", response_env)
                    if self.session_store and session_id and current_user_seq:
                        await self.session_store.append_message(
                            session_id=session_id,
                            user_id=user_id,
                            request_id=request_id,
                            role="assistant",
                            content=answer_payload["answer"],
                        )
                    return response_env

                last_feedback = {
                    "attempt": attempt + 1,
                    "stage": "validator",
                    "errors": errors,
                    "warnings": warnings,
                    "instruction": "Return a corrected PlannerOutput plan that passes validation. If tools are not allowed/available, use a single compose step.",
                }
                continue

            # ----------------
            # Confidence gating (no-tools path)
            # ----------------
            try:
                conf = float((planner_payload.get("user_intent") or {}).get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            has_tool_calls = self._plan_has_tool_calls(planner_payload)
            if (not has_tool_calls) and (conf < self.confidence_threshold):
                msg = SAFE_FALLBACK_PT if str(locale).lower().startswith("pt") else SAFE_FALLBACK_EN
                answer_payload = {
                    "answer": msg,
                    "final_context": {
                        "intent": "insufficient_information",
                        "steps_executed": [],
                        **({"conversation_context": conversation_context} if conversation_context else {}),
                    },
                    "timings_ms": timings,
                }
                validate_json(self.answer_payload_schema, answer_payload, bundle=self.bundle)
                response_env = self._make_envelope(
                    base_metadata, "response", "orchestrator", answer_payload, self.answer_payload_schema, timings_ms=timings
                )
                self._log_envelope("response", response_env)
                await self._persist("response", response_env)
                if self.session_store and session_id and current_user_seq:
                    await self.session_store.append_message(
                        session_id=session_id,
                        user_id=user_id,
                        request_id=request_id,
                        role="assistant",
                        content=answer_payload["answer"],
                    )
                return response_env

            # ----------------
            # Executor
            # ----------------
            executor_input_payload = {
                "planner_output": planner_payload,
                "request_context": {"request_id": request_id, "user_message": user_message},
            }
            if self.executor_input_schema:
                validate_json(self.executor_input_schema, executor_input_payload, bundle=self.bundle)
            executor_in_env = self._make_envelope(
                base_metadata,
                "executor_input",
                "orchestrator",
                executor_input_payload,
                self.executor_input_schema or self.executor_result_schema,
                timings_ms=timings,
            )
            self._log_envelope("executor_input", executor_in_env)
            await self._persist("executor_input", executor_in_env)

            t0 = time.perf_counter()
            exec_payload = await self.executor.execute(executor_in_env["payload"])
            timings["executor_ms"] = int((time.perf_counter() - t0) * 1000)

            exec_env = self._make_envelope(
                base_metadata, "executor_result", "executor", exec_payload, self.executor_result_schema, timings_ms=timings
            )
            self._log_envelope("executor_result", exec_env)
            await self._persist("executor_result", exec_env)

            self.output_validator.validate(exec_env["payload"])

            failed_tool = self._first_failed_tool_step(planner_payload, exec_env["payload"])
            if failed_tool is not None:
                # Ask the planner to replan with execution feedback (industry standard: automatic recovery path).
                if attempt <= self.max_replans:
                    last_feedback = {
                        "attempt": attempt + 1,
                        "stage": "executor",
                        "failed_step": failed_tool.get("id"),
                        "tool": (failed_tool.get("capability") or ""),
                        "error": (failed_tool.get("error") or ""),
                        "instruction": "Replan. If the request can be answered without tools, produce a single compose step. If a tool is required, keep tool_call but fix inputs to match the tool input_schema exactly.",
                    }
                    continue

                err_msg = str(failed_tool.get("error") or "Tool execution failed")
                answer_payload = {
                    "answer": SAFE_FALLBACK_PT if str(locale).lower().startswith("pt") else SAFE_FALLBACK_EN,
                    "final_context": {
                        "intent": "tool_execution_failed",
                        "steps_executed": exec_env["payload"]["steps_executed"],
                        **({"conversation_context": conversation_context} if conversation_context else {}),
                    },
                    "timings_ms": timings,
                }
                validate_json(self.answer_payload_schema, answer_payload, bundle=self.bundle)
                response_env = self._make_envelope(
                    base_metadata, "response", "orchestrator", answer_payload, self.answer_payload_schema, timings_ms=timings
                )
                self._log_envelope("response", response_env)
                await self._persist("response", response_env)
                if self.session_store and session_id and current_user_seq:
                    await self.session_store.append_message(
                        session_id=session_id,
                        user_id=user_id,
                        request_id=request_id,
                        role="assistant",
                        content=answer_payload["answer"],
                    )
                return response_env

            # Success path: break out and proceed to final LLM responder.
            break

        # At this point, planner_env and exec_env MUST be available.
        if planner_env is None or exec_env is None:
            raise RuntimeError("orchestrator_invariant_failed: missing planner_env/exec_env")

        # Derive composed context from compose step output (if present)
        composed_context = None
        for s in exec_env["payload"]["steps_executed"]:
            if (s.get("status") == "success") and isinstance(s.get("output"), dict) and "context" in s["output"]:
                composed_context = s["output"]["context"]

        intent_summary = str((planner_env["payload"].get("user_intent") or {}).get("summary") or "")
        final_intent = composed_context or intent_summary or "final_response"

        final_context_obj: Dict[str, Any] = {"intent": final_intent, "steps_executed": exec_env["payload"]["steps_executed"]}
        if conversation_context:
            final_context_obj["conversation_context"] = conversation_context

        final_llm_input = {
            "final_context": final_context_obj,
            "final_output_spec": {"format": "text", "tone": "enterprise", "constraints": ["concise", "accurate", f"language:{locale}"]},
        }
        validate_json(self.final_llm_input_schema, final_llm_input, bundle=self.bundle)

        final_llm_env = self._make_envelope(base_metadata, "final_llm_input", "orchestrator", final_llm_input, self.final_llm_input_schema, timings_ms=timings)
        self._log_envelope("final_llm_input", final_llm_env)
        await self._persist("final_llm_input", final_llm_env)

        # Responder
        t0 = time.perf_counter()
        answer_text = await self.responder.answer(final_llm_env["payload"])
        timings["responder_ms"] = int((time.perf_counter() - t0) * 1000)
        timings["total_ms"] = int((time.perf_counter() - t_request0) * 1000)

        # Final confidence gate: enforce uniform safe fallback when confidence is below threshold.
        try:
            est = getattr(self.responder, "estimate_confidence", None)
            final_conf = float(await est(final_llm_env["payload"], answer_text)) if est else 1.0
        except Exception:
            final_conf = 0.0

        if final_conf < float(self.confidence_threshold):
            answer_text = SAFE_FALLBACK_PT if str(locale).lower().startswith("pt") else SAFE_FALLBACK_EN

        answer_payload = {
            "answer": answer_text,
            "final_context": final_context_obj,
            "timings_ms": timings,
        }
        validate_json(self.answer_payload_schema, answer_payload, bundle=self.bundle)

        response_env = self._make_envelope(base_metadata, "response", "orchestrator", answer_payload, self.answer_payload_schema, timings_ms=timings)
        self._log_envelope("response", response_env)
        await self._persist("response", response_env)

        # Persist assistant message
        if self.session_store and session_id and current_user_seq:
            await self.session_store.append_message(
                session_id=session_id, user_id=user_id, request_id=request_id, role="assistant", content=answer_text
            )

        return response_env