from typing import Any, Dict, List
import requests
from llm.base import BaseLLM

class VLLMAdapter(BaseLLM):
    def __init__(self, config: dict):
        self.cfg = config["llm"]

    def load(self) -> None:
        return

    def generate(self, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        endpoint = self.cfg["runtime"].get("endpoint", "")
        if endpoint.startswith("unix://"):
            raise NotImplementedError("unix:// endpoints require requests-unixsocket; prefer http:// for Phase 1.")
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        model_name = self.cfg["model"]["name"]

        messages = []
        for b in context:
            role = "user" if b.get("type") == "user" else "system"
            messages.append({"role": role, "content": b.get("content", "")})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": float(self.cfg["runtime"].get("temperature", 0.2)),
            "max_tokens": int(self.cfg["runtime"].get("max_new_tokens", 512)),
            "stream": False
        }

        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return {"response": text}

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def healthcheck(self) -> Dict[str, Any]:
        return {"status": "ok", "provider": "vllm"}
