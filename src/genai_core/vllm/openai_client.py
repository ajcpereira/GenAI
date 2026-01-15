from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx


log = logging.getLogger("genai_core.vllm_client")


class VLLMOpenAIClient:
    """Minimal OpenAI-compatible client for vLLM (OpenAI API server)."""

    def __init__(self, base_url: str, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def chat_completion(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if extra:
            payload.update(extra)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.error("vLLM request failed: %s", str(e))
            raise
