import logging
import json
import re
from typing import Any, Dict, List

import httpx

from utils.common import validate_json

logger = logging.getLogger("genai.responder")


class Responder:
    """
    Final response LLM (always the last stage).
    Expects FinalLLMInput payload and produces a natural-language answer string.
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
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _extract_language(constraints: List[str]) -> str:
        for c in constraints or []:
            if isinstance(c, str) and c.lower().startswith("language:"):
                return c.split(":", 1)[1].strip() or "pt"
        return "pt"

    @staticmethod
    def _strip_think(text: str) -> str:
        """
        Remove conteúdo de raciocínio de modelos que usam tags tipo <think>...</think>.
        """
        if not text:
            return ""
        txt = str(text)
        txt = re.sub(r"(?is)<think>.*?</think>\s*", "", txt).strip()
        txt = txt.replace("</think>", "").strip()
        return txt

    async def answer(self, final_llm_input: Dict[str, Any]) -> str:
        if self.final_llm_input_schema:
            validate_json(self.final_llm_input_schema, final_llm_input, bundle=self.bundle)

        fc = final_llm_input["final_context"]
        spec = final_llm_input["final_output_spec"]

        intent = str(fc.get("intent") or "")
        steps_executed = list(fc.get("steps_executed") or [])

        fmt = str(spec.get("format") or "text")
        tone = str(spec.get("tone") or "enterprise")
        constraints = list(spec.get("constraints") or [])

        lang = self._extract_language(constraints)

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

        context_lines = [f"INTENT_CONTEXT:\n{intent}\n", "EXECUTION_RESULTS:"]
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

        payload = {
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
        if content is None:
            return ""
        txt = self._strip_think(str(content))

        if self.use_structured_outputs:
            # Expect {"answer": "..."} and return only the answer field.
            try:
                obj = json.loads(txt)
                ans = obj.get("answer")
                if isinstance(ans, str):
                    return ans.strip()
            except Exception:
                # Fall back to raw text if backend/model didn't honor json_object.
                pass

        return txt
