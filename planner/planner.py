# planner/planner.py
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from utils.common import validate_json
from utils.lang import detect_language
from planner.rules import PlannerRules, PlannerRuleEngine

logger = logging.getLogger("genai.planner")


class Planner:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, bundle: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.bundle = bundle or {}
        self.schemas = (bundle or {}).get("schemas") or {}
        self.planner_input_schema = self.schemas.get("PlannerInput")
        self.planner_output_schema = self.schemas.get("PlannerOutput")

        self.base_url = str(self.cfg["vllm_base_url"]).rstrip("/")
        self.model = str(self.cfg.get("model", ""))
        self.api_key = str(self.cfg.get("api_key", "EMPTY"))
        self.timeout_s = float(self.cfg.get("timeout_s", 20.0))
        self.temperature = float(self.cfg.get("temperature", 0.0))
        self.max_tokens = int(self.cfg.get("max_tokens", 1024))
        self.stop = list(self.cfg.get("stop") or [])
        self.max_raw_log_chars = int(self.cfg.get("max_raw_log_chars", 8000))
        self.max_replan_feedback_chars = int(self.cfg.get("max_replan_feedback_chars", 4000))
        # Prefer structured JSON outputs when supported by the backend.
        self.use_structured_outputs = bool(self.cfg.get("use_structured_outputs", False)) or bool(
            self.cfg.get("strict_json", False)
        )
        self.chat_path = str(self.cfg.get("chat_path", "/v1/chat/completions"))
        self.retries = int(self.cfg.get("retries", 2))

        # Some backends (notably certain DeepSeek-R1 deployments) may down-weight/ignore role='system'.
        # Allow forcing the system prompt to be injected as 'user' for deterministic behavior.
        self.system_prompt_role = str(self.cfg.get("system_prompt_role", "system")).strip().lower() or "system"
        if self.system_prompt_role not in ("system", "user"):
            self.system_prompt_role = "system"

        # Rules-as-config (deterministic guardrails) - sourced from config.yaml only.
        self.rules = PlannerRules(self.cfg)
        self.engine = PlannerRuleEngine(self.rules)

        # Base instruction; tool catalog is injected dynamically per request.
        self.base_system_prompt = str(
            self.cfg.get(
                "system_prompt",
                (
                    "You are a planning component in an enterprise orchestration system.\n"
                    "You MUST output ONLY a single JSON object that matches the PlannerOutput schema.\n"
                    "No markdown. No commentary. No extra keys. No trailing text.\n"
                ),
            )
        )

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _parse_first_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        t = (text or "").strip()
        if not t:
            return None, "empty"

        # remove common code fences
        t = t.replace("```json", "```").strip()
        if t.startswith("```"):
            # best-effort: drop first fence line and last fence
            t = t.strip("`").strip()

        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, "no_json_object"

        candidate = t[start : end + 1]
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, str(e)

    def _build_system_prompt(self, allowed_tools: List[Dict[str, Any]], locale: str) -> str:
        # Provide a compact tool catalog with schemas to reduce hallucinations.
        tool_lines: List[str] = []
        for t in allowed_tools:
            name = str(t.get("name") or "").strip()
            if not name:
                continue
            desc = str(t.get("description") or "").strip()
            inp = t.get("input_schema")
            tool_lines.append(
                json.dumps(
                    {"name": name, "description": desc, "input_schema": inp},
                    ensure_ascii=False,
                )
            )

        catalog = "\n".join(tool_lines) if tool_lines else "(none)"

        no_tools_note = "\n- IMPORTANT: TOOL CATALOG is empty. You MUST NOT output any tool_call steps.\n" if catalog == "(none)" else "\n"

        rule_lines = [
            "Rules:",
            "- Output MUST be a single JSON object that matches the PlannerOutput schema (no markdown, no extra keys).",
            "- Top-level keys MUST be exactly: user_intent, plan, violations.",
            "- Never invent capabilities.",
"- You may ONLY use capabilities that appear in the TOOL CATALOG.",
"- The plan MUST be based ONLY on the CURRENT_USER_MESSAGE. Ignore prior messages unless the user explicitly references them.",
"- Tools are OPTIONAL. Do NOT use a tool unless it is REQUIRED to answer the CURRENT_USER_MESSAGE.",
"- Never use tools for tasks solvable by reasoning alone (e.g., Fibonacci, combinations, code explanation).",
"- Use math.eval ONLY to evaluate an explicit mathematical expression when it is needed; never for unrelated tasks.",
        ]
        if catalog == "(none)":
            rule_lines.append("- IMPORTANT: TOOL CATALOG is empty. You MUST NOT output any tool_call steps.")
        rule_lines.extend(
            [
                "- Default to a single 'compose' step when the user request can be answered without external data/tools.",
                "- Use 'tool_call' only when a tool is clearly required.",
                "- SCHEMA INVARIANT: If a step has type='tool_call', it MUST include a non-empty 'capability' string equal to one of the TOOL CATALOG names.",
                "- SCHEMA INVARIANT: If a step has type='compose', set capability to null (or omit it).",
                "- Do NOT output tool execution results (no fields like action/args/result). You are only planning.",
                "- If tools are required but none are available, keep a single 'compose' step and set confidence low (<0.2).",
                "- Set user_intent.confidence in [0,1].",
                f"- Set user_intent.locale to '{locale}'.",
                f"- Produce user_intent.summary and all textual fields primarily in '{locale}'.",
                "- The only valid capability strings are the exact values of name in the TOOL CATALOG JSON objects.",
            ]
        )
        rules = "\n" + "\n".join(rule_lines) + "\n"

        def _example_tool_call_block() -> str:
            """Build a tool_call example using ONLY the provided catalog.

            Critical: do not leak capability names when the catalog is empty.
            """

            if not allowed_tools:
                return ""

            first = allowed_tools[0] or {}
            tool_name = str(first.get("name") or "").strip()
            schema = first.get("input_schema") or {}

            # Create a minimal inputs object that satisfies required fields.
            # If there are no required fields, an empty object is valid.
            inputs: Dict[str, Any] = {}
            if isinstance(schema, dict):
                req = schema.get("required")
                props = schema.get("properties")
                if isinstance(req, list) and isinstance(props, dict):
                    for k in req:
                        if isinstance(k, str):
                            inputs[k] = None

            if not tool_name:
                # Defensive: if the catalog entry is malformed, omit the tool_call example entirely.
                return ""

            return (
                "\nEXAMPLE OUTPUT (tool_call):\n"
                "{\n"
                f"  \"user_intent\": {{\"summary\": \"...\", \"type\": \"informational\", \"confidence\": 0.9, \"locale\": \"{locale}\"}},\n"
                "  \"plan\": {\n"
                "    \"strategy\": \"sequential\",\n"
                "    \"steps\": [\n"
                "      {\n"
                "        \"id\": \"1\",\n"
                "        \"type\": \"tool_call\",\n"
                f"        \"capability\": \"{tool_name}\",\n"
                "        \"description\": \"...\",\n"
                f"        \"inputs\": {json.dumps(inputs, ensure_ascii=False)},\n"
                "        \"dependencies\": []\n"
                "      }\n"
                "    ]\n"
                "  },\n"
                "  \"violations\": []\n"
                "}\n"
            )

        examples = (
            "\nEXAMPLE OUTPUT (compose):\n"
            "{\n"
            f"  \"user_intent\": {{\"summary\": \"...\", \"type\": \"informational\", \"confidence\": 0.5, \"locale\": \"{locale}\"}},\n"
            "  \"plan\": {\n"
            "    \"strategy\": \"sequential\",\n"
            "    \"steps\": [\n"
            "      {\n"
            "        \"id\": \"compose_1\",\n"
            "        \"type\": \"compose\",\n"
            "        \"description\": \"Responder sem tools.\",\n"
            "        \"capability\": null,\n"
            f"        \"inputs\": {{\"message\": \"...\", \"locale\": \"{locale}\"}},\n"
            "        \"dependencies\": []\n"
            "      }\n"
            "    ]\n"
            "  },\n"
            "  \"violations\": []\n"
            "}\n"
            + _example_tool_call_block()
        )

        return (
            self.base_system_prompt.rstrip()
            + f"\n\nUSER_LOCALE: {locale}\n"
            + "\nTOOL CATALOG (the ONLY allowed capabilities):\n"
            + catalog
            + "\n"
            + rules
            + examples
        )

    async def _call_vllm(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}{self.chat_path}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": float(self.cfg.get("top_p", 1.0)),
            "max_tokens": self.max_tokens,
            **({"stop": self.stop} if self.stop else {}),
        }
        # If the backend supports it, force JSON object output to reduce parser failures.
        if self.use_structured_outputs:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        return "" if content is None else str(content)

    async def build_plan(self, planner_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        planner_input must conform to PlannerInput schema (validated by orchestrator).
        Returns PlannerOutput payload.
        """
        if self.planner_input_schema:
            validate_json(self.planner_input_schema, planner_input, bundle=self.bundle)

        req = planner_input.get("request") or {}
        ctx = planner_input.get("context") or {}
        tools = planner_input.get("tools") or {}
        policy = tools.get("policy") or {}

        msg = str(req.get("current_user_message") or "").strip()
        if not msg:
            msg = "Pedido vazio"

        locale = str(req.get("locale") or "").strip() or self.rules.default_locale
        if not locale:
            locale = detect_language(msg, default=self.rules.default_locale).code

        allowed_tools = list(tools.get("catalog") or [])
        allowed_tool_names = [
            str(t.get("name") or "").strip()
            for t in allowed_tools
            if str(t.get("name") or "").strip()
        ]
        enabled_tool_names = list(policy.get("enabled_tools") or [])

        # Build a compact user-only transcript for the planner (reduces anchoring).
        recent_user_messages = list(ctx.get("recent_user_messages") or [])
        transcript_lines: List[str] = []
        for m in recent_user_messages:
            mid = str(m.get("id") or "")
            text = str(m.get("text") or "")
            if text:
                transcript_lines.append(f"[{mid}] USER: {text}")

        conversation_context = "\n".join(transcript_lines).strip()
        system_prompt = self._build_system_prompt(allowed_tools, locale=locale)
        logger.debug(
            "planner_input",
            extra={
                "enabled_tools": enabled_tool_names,
                "allowed_tool_names": allowed_tool_names,
                "locale": locale,
            },
        )

        last_err: Optional[str] = None
        raw_last: str = ""
        for attempt in range(1, self.retries + 2):
            try:
                replan_fb_obj = planner_input.get("replan_feedback")
                replan_fb = ""
                if isinstance(replan_fb_obj, dict):
                    reason = str(replan_fb_obj.get("reason") or "")
                    errs = list(replan_fb_obj.get("errors") or [])
                    err_lines: List[str] = []
                    for e in errs:
                        if isinstance(e, dict):
                            code = str(e.get("code") or "")
                            msg_e = str(e.get("message") or "")
                            if code or msg_e:
                                err_lines.append(f"- {code}: {msg_e}".strip())
                    if reason or err_lines:
                        replan_fb = "REPLAN_FEEDBACK:\n" + (f"reason: {reason}\n" if reason else "")
                        if err_lines:
                            replan_fb += "errors:\n" + "\n".join(err_lines) + "\n"
                if replan_fb:
                    replan_fb = replan_fb[: self.max_replan_feedback_chars]

# System prompt role can be forced to 'user' for certain chat templates.
                messages = [{"role": self.system_prompt_role, "content": system_prompt}]
                if replan_fb:
                    messages.append({"role": self.system_prompt_role, "content": f"VALIDATION_FEEDBACK:\n{replan_fb}"})

                if last_err:
                    # Deterministic self-correction: feed previous schema/contract failure back to the model.
                    messages.append(
                        {
                            "role": self.system_prompt_role,
                            "content": (
                                "VALIDATION_FEEDBACK:\n"
                                f"Previous attempt failed schema validation: {last_err}\n"
                                "Return a corrected PlannerOutput JSON.\n"
                                "Constraints reminder:\n"
                                "1) Top-level keys: user_intent, plan, violations only.\n"
                                "2) tool_call steps MUST include capability (exact tool name from TOOL CATALOG).\n"
                                "3) Do not output action/args/result; you are not executing tools."
                            ),
                        }
                    )

                conv_ctx = conversation_context
                if conv_ctx:
                    messages.append({"role": self.system_prompt_role, "content": f"CONVERSATION_CONTEXT:\n{conv_ctx}"})

                messages.append({"role": "user", "content": msg})
                raw = await self._call_vllm(messages)
                raw_last = raw
                logger.debug("planner_raw_completion", extra={"raw_completion": raw[: self.max_raw_log_chars]})

                plan_obj, perr = self._parse_first_json_object(raw)
                if perr:
                    raise ValueError(f"planner_json_parse_failed: {perr}")

                plan = dict(plan_obj or {})
                logger.debug("planner_parsed_output", extra={"planner_output": plan})

                # Ensure locale is present (defense in depth)
                ui = plan.get("user_intent") or {}
                if isinstance(ui, dict):
                    ui.setdefault("locale", locale)
                    plan["user_intent"] = ui

                # Apply deterministic rules/caps
                plan, violations = self.engine.apply(plan)
                plan["violations"] = [{"code": v.code, "message": v.message, "detail": v.detail} for v in violations]

                # Validate against PlannerOutput schema (contract bundle)
                if self.planner_output_schema:
                    validate_json(self.planner_output_schema, plan, bundle=self.bundle)

                logger.info(
                    "plan_built",
                    extra={"steps": len((plan.get("plan") or {}).get("steps") or []), "attempt": attempt, "locale": locale},
                )
                return plan

            except Exception as e:
                last_err = str(e)
                logger.warning("plan_build_failed", extra={"attempt": attempt, "error": last_err})

        # Final fallback: minimal compliant plan with low confidence
        plan = {
            "user_intent": {"summary": msg, "type": "informational", "confidence": 0.1, "locale": locale or self.rules.default_locale},
            "plan": {
                "strategy": "sequential",
                "steps": [
                    {
                        "id": "compose_1",
                        "type": "compose",
                        "description": "Gerar resposta com base apenas no pedido do utilizador (sem tools).",
                        "capability": None,
                        "agent": "planner_fallback",
                        "inputs": {"message": msg, "locale": locale},
                        "expected_output": "Contexto final em texto.",
                        "dependencies": [],
                        "optional": False,
                    }
                ],
            },
            "violations": [{"code": "PLANNER_FALLBACK", "message": "Planner fallback used due to repeated failures.", "detail": {"error": last_err}}],
        }
        logger.error("plan_build_fallback", extra={"error": last_err})
        # Validate fallback if schema available
        if self.planner_output_schema:
            validate_json(self.planner_output_schema, plan, bundle=self.bundle)
        return plan
