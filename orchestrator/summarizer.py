import json
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("genai.summarizer")


class VLLMSummarizer:
    """
    Deterministic summarizer for conversation compaction.
    Uses vLLM OpenAI-compatible /v1/chat/completions.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.base_url = str(self.cfg["vllm_base_url"]).rstrip("/")
        self.model = str(self.cfg["model"])
        self.api_key = str(self.cfg.get("api_key", "EMPTY"))
        self.timeout_s = float(self.cfg.get("timeout_s", 20.0))
        self.temperature = float(self.cfg.get("temperature", 0.0))
        self.max_tokens = int(self.cfg.get("max_tokens", 512))
        self.use_structured_outputs = bool(self.cfg.get("use_structured_outputs", True))

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def summarize(self, *, existing_summary: Optional[str], transcript: str, locale: str) -> str:
        existing_summary = (existing_summary or "").strip()
        transcript = (transcript or "").strip()

        system = (
            "You are a deterministic summarization component in an enterprise system.\n"
            "Your task is to compress conversation history without losing critical facts, constraints, and decisions.\n"
            "You MUST NOT invent facts.\n"
            "You MUST NOT include chain-of-thought.\n"
            "Write in the requested language.\n"
            "If response_format is json_object, output MUST be a JSON object with key 'summary'.\n"
        )

        user_parts = [f"LANGUAGE: {locale}"]
        if existing_summary:
            user_parts.append("EXISTING_SUMMARY:\n" + existing_summary)
        user_parts.append("NEW_TRANSCRIPT_TO_MERGE:\n" + transcript)
        user_parts.append(
            "OUTPUT_REQUIREMENTS:\n"
            "- Produce a concise summary capturing: user goals, constraints, entities, decisions, and open questions.\n"
            "- Keep it factual and enterprise-oriented.\n"
        )
        user = "\n\n".join(user_parts)

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

        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        txt = str(content).strip()

        if self.use_structured_outputs:
            try:
                obj = json.loads(txt)
                s = obj.get("summary")
                if isinstance(s, str) and s.strip():
                    return s.strip()
            except Exception:
                pass

        # fallback: treat as plain text
        return txt.strip()
