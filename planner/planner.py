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

        rules = (
            "\nRules:\n"
            "- Output MUST be a single JSON object that matches the PlannerOutput schema (no markdown, no extra keys).\n"
            "- Top-level keys MUST be exactly: user_intent, plan, violations.\n"
            "- Never invent capabilities.\n"
            "- You may ONLY use capabilities that appear in the TOOL CATALOG.\n"
            "- Default to a single 'compose' step when the user request can be answered without external data/tools.\n"
            "- Use 'tool_call' only when a tool is clearly required.\n"
            "- SCHEMA INVARIANT: If a step has type='tool_call', it MUST include a non-empty 'capability' string equal to one of the TOOL CATALOG names.\n"
            "- SCHEMA INVARIANT: If a step has type='compose', set capability to null (or omit it).\n"
            "- Do NOT output tool execution results (no fields like action/args/result). You are only planning.\n"
            "- If tools are required but none are available, keep a single 'compose' step and set confidence low (<0.2).\n"
            "- Set user_intent.confidence in [0,1].\n"
            f"- Set user_intent.locale to '{locale}'.\n"
            f"- Produce user_intent.summary and all textual fields primarily in '{locale}'.\n"
            "- The only valid capability strings are the exact values of name in the TOOL CATALOG JSON objects."
        )

        examples = """

EXAMPLE OUTPUT (tool_call):
{
  "user_intent": {"summary": "...", "type": "informational", "confidence": 0.9, "locale": "pt"},
  "plan": {
    "strategy": "sequential",
    "steps": [
      {
        "id": "1",
        "type": "tool_call",
        "capability": "time.now",
        "description": "Obter a data e hora atuais.",
        "inputs": {"timezone": null},
        "dependencies": []
      }
    ]
  },
  "violations": []
}

EXAMPLE OUTPUT (compose):
{
  "user_intent": {"summary": "...", "type": "informational", "confidence": 0.5, "locale": "pt"},
  "plan": {
    "strategy": "sequential",
    "steps": [
      {
        "id": "compose_1",
        "type": "compose",
        "description": "Responder sem tools.",
        "capability": null,
        "inputs": {"message": "...", "locale": "pt"},
        "dependencies": []
      }
    ]
  },
  "violations": []
}
"""

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

        msg = str(planner_input.get("user_message") or "").strip()
        if not msg:
            msg = "Pedido vazio"

        allowed_tools = list(planner_input.get("allowed_tools") or [])

        locale = str(planner_input.get("locale") or "").strip() or self.rules.default_locale
        if not locale:
            locale = detect_language(msg, default=self.rules.default_locale).code

        system_prompt = self._build_system_prompt(allowed_tools, locale=locale)
        logger.debug(
            "planner_input",
            extra={
                "enabled_tools": list(planner_input.get("enabled_tools") or []),
                "allowed_tool_names": [str(t.get("name") or "") for t in allowed_tools],
                "locale": locale,
            },
        )

        last_err: Optional[str] = None
        raw_last: str = ""
        for attempt in range(1, self.retries + 2):
            try:
                replan_fb = str(planner_input.get("replan_feedback") or "").strip()
                if replan_fb:
                    replan_fb = replan_fb[: self.max_replan_feedback_chars]
                messages = [
                    {"role": "system", "content": system_prompt},
                ]
                if replan_fb:
                    messages.append({"role": "system", "content": f"VALIDATION_FEEDBACK:\n{replan_fb}"})

                if last_err:
                    # Deterministic self-correction: feed previous schema/contract failure back to the model.
                    messages.append(
                        {
                            "role": "system",
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
