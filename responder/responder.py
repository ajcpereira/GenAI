import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from utils.common import validate_json

logger = logging.getLogger("genai.responder")

SAFE_FALLBACK_PT = "Não tenho informação segura e precisa para responder"
SAFE_FALLBACK_EN = "I don't have reliable, precise information to answer"


class Responder:
    """Final response generator (last stage).

    Contract:
      - input must validate against schema FinalLLMInput (when schema is present)
      - output MUST be a plain string (answer text)
    """

    def __init__(self, cfg: Dict[str, Any], bundle: Dict[str, Any]):
        self.cfg = cfg or {}
        self.bundle = bundle or {}
        self.schemas = (bundle or {}).get("schemas") or {}
        self.final_llm_input_schema = self.schemas.get("FinalLLMInput")

        self.base_url = str(self.cfg["vllm_base_url"]).rstrip("/")
        self.model = str(self.cfg["model"])
        self.api_key = str(self.cfg.get("api_key", "EMPTY"))
        self.timeout_s = float(self.cfg.get("timeout_s", 20.0))
        self.temperature = float(self.cfg.get("temperature", 0.2))
        self.max_tokens = int(self.cfg.get("max_tokens", 512))
        self.use_structured_outputs = bool(self.cfg.get("use_structured_outputs", True))

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # -------------------------
    # Deterministic helpers
    # -------------------------

    @staticmethod
    def _extract_language(constraints: List[str]) -> str:
        for c in constraints or []:
            if isinstance(c, str) and c.lower().startswith("language:"):
                lang = c.split(":", 1)[1].strip()
                return lang or "pt"
        return "pt"

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove reasoning tags like <think>...</think>."""
        if not text:
            return ""
        t = str(text)
        t = re.sub(r"(?is)<think>.*?</think>\s*", "", t).strip()
        t = t.replace("</think>", "").strip()
        return t

    @staticmethod
    def _maybe_strip_code_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            # Remove the first fence line and the last fence if present.
            t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
            t = t.strip()
        return t

    @staticmethod
    def _extract_text_from_json_obj(obj: Dict[str, Any]) -> Optional[str]:
        """Extract text deterministically from a JSON object."""
        for k in ("answer", "output", "text", "message"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (int, float, bool)):
                return str(v)

        for _, v in obj.items():  # deterministic by insertion order
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (int, float, bool)):
                return str(v)

        return None

    @staticmethod
    def _parse_first_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort parse of the first embedded JSON object."""
        t = Responder._maybe_strip_code_fences(text)
        start = t.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = t[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            return None
        return None

    @staticmethod
    def _extract_text_from_partial_json(text: str) -> Optional[str]:
        """Extract answer-like fields from JSON-like text even if it's truncated.

        This is deterministic contract enforcement for format=text. We only look for a stable
        set of keys and return the first match in a fixed order.
        """
        t = Responder._maybe_strip_code_fences(text)
        if not t:
            return None

        # Only attempt when it looks like JSON.
        lt = t.lstrip()
        if not (lt.startswith("{") or lt.startswith("[")):
            return None

        # Fixed preference order.
        keys = ("answer", "output", "text", "message")
        for k in keys:
            # Match: "key" : "value"
            m = re.search(rf'"{re.escape(k)}"\s*:\s*"((?:\\.|[^"\\])*)"', t)
            if m:
                raw = m.group(1)
                try:
                    return bytes(raw, "utf-8").decode("unicode_escape").strip()
                except Exception:
                    return raw.strip()

            # Match: "key" : 123 / true / false
            m2 = re.search(rf'"{re.escape(k)}"\s*:\s*(true|false|null|-?\d+(?:\.\d+)?)', t, re.IGNORECASE)
            if m2:
                return m2.group(1).strip()

        return None

    @staticmethod
    def _deterministic_answer_from_steps(lang: str, steps_executed: List[Dict[str, Any]]) -> Optional[str]:
        """Render a deterministic answer from simple tool outputs."""
        if not steps_executed:
            return None

        successes = [
            s for s in steps_executed
            if s.get("status") == "success" and isinstance(s.get("output"), dict)
        ]
        if not successes:
            return None

        if len(successes) != 1:
            return None

        out = successes[0].get("output") or {}
        data = out.get("data") if isinstance(out.get("data"), dict) else None
        if not isinstance(data, dict):
            return None

        # math.eval style: data.value scalar
        if "value" in data and isinstance(data.get("value"), (int, float, bool, str)):
            v = str(data.get("value")).strip()
            return v or None

        # time.now style: data.iso + data.timezone
        if isinstance(data.get("iso"), str) and data.get("iso").strip():
            iso = data["iso"].strip()
            tz = str(data.get("timezone") or "UTC")
            try:
                dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                t_str = dt.strftime("%H:%M:%S")
            except Exception:
                if lang.lower().startswith("pt"):
                    return f"A hora atual no fuso horário {tz} é {iso}."
                return f"Current time in {tz} is {iso}."

            if lang.lower().startswith("pt"):
                return f"A hora atual no fuso horário {tz} é {t_str}."
            return f"The current time in {tz} is {t_str}."

        return None

    # -------------------------
    # Public API
    # -------------------------

    async def answer(self, final_llm_input: Dict[str, Any]) -> str:
        if self.final_llm_input_schema:
            validate_json(self.final_llm_input_schema, final_llm_input, bundle=self.bundle)

        fc = final_llm_input["final_context"]
        spec = final_llm_input["final_output_spec"]

        intent = str(fc.get("intent") or "")
        conversation_context = str(fc.get("conversation_context") or "").strip()
        steps_executed = list(fc.get("steps_executed") or [])

        fmt = str(spec.get("format") or "text")
        tone = str(spec.get("tone") or "enterprise")
        constraints = list(spec.get("constraints") or [])
        lang = self._extract_language(constraints)

        # Deterministic fast-path for simple tool results (format=text only).
        if fmt.lower() == "text":
            det = self._deterministic_answer_from_steps(lang, steps_executed)
            if det is not None:
                return det

        system = (
            "You are the final answer generator in an enterprise orchestration system.\n"
            f"Output language: {lang}.\n"
            f"Output format: {fmt}.\n"
            f"Tone: {tone}.\n"
            "If response_format is json_object, output MUST be a JSON object with key 'answer'.\n"
            "Use only the provided context and execution results.\n"
            "Do NOT reveal chain-of-thought.\n"
            "Do NOT output <think>...</think> or any scratchpad.\n"
            "If data is insufficient, say so explicitly.\n"
        )

        context_lines: List[str] = []
        if conversation_context:
            context_lines.append(f"CONVERSATION_CONTEXT:\n{conversation_context}\n")
        context_lines.append(f"INTENT_CONTEXT:\n{intent}\n")
        context_lines.append("EXECUTION_RESULTS:")
        for s in steps_executed:
            sid = s.get("id")
            status = s.get("status")
            if status == "success":
                context_lines.append(f"- {sid}: success => {s.get('output')}")
            elif status == "failed":
                context_lines.append(f"- {sid}: failed => {s.get('error')}")
            else:
                context_lines.append(f"- {sid}: skipped")

        user = "CONTEXT:\n" + "\n".join(context_lines)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.use_structured_outputs:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()

        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        txt = self._strip_think(str(content or ""))

        # 1) If structured outputs are enabled, first try strict json.loads.
        extracted: Optional[str] = None
        if self.use_structured_outputs:
            t = self._maybe_strip_code_fences(txt)
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    extracted = self._extract_text_from_json_obj(obj)
            except Exception:
                extracted = None

        # 2) Best-effort parse first embedded JSON object.
        if extracted is None and fmt.lower() == "text":
            obj2 = self._parse_first_json_object(txt)
            if isinstance(obj2, dict):
                extracted = self._extract_text_from_json_obj(obj2)

        # 3) Truncated/invalid JSON: extract "answer"/"output"/... via deterministic regex.
        if extracted is None and fmt.lower() == "text":
            extracted = self._extract_text_from_partial_json(txt)

        if extracted is not None and extracted.strip():
            return extracted.strip()

        # 4) Enforce contract: if still JSON-like, return deterministic fallback.
        if fmt.lower() == "text":
            det = self._deterministic_answer_from_steps(lang, steps_executed)
            if det is not None:
                return det
            return SAFE_FALLBACK_PT if lang.lower().startswith("pt") else SAFE_FALLBACK_EN

        # Non-text formats: return raw output (still stripped of think tags).
        return txt.strip()

