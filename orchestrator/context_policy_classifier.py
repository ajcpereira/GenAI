from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


CLASSIFIER_SYSTEM_PROMPT = (
    "You are a deterministic classifier for conversational dependency.\n\n"
    "Goal:\n"
    "Classify whether CURRENT_USER_MESSAGE needs recent prior turns to be correctly interpreted.\n\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    "{\"mode\":\"standalone\"|\"recent\",\"recent_turns\":0|6,\"confidence\":number}\n\n"
    "Definitions:\n"
    "- standalone: self-contained; meaning is clear without any prior turns.\n"
    "- recent: depends on prior turns for meaning (anaphora/ellipsis/deixis).\n\n"
    "Hard rules (if any true => mode=recent, recent_turns=6):\n"
    "- Contains deictic/anaphoric references: \"isso\", \"isto\", \"aquilo\", \"aquele\", \"essa\", \"dessa\", \"tal\", \"o mesmo\", \"assim\", \"deste\", \"daquele\".\n"
    "- Explicitly refers to previous content: \"o que disseste\", \"como falámos\", \"como acima\", \"naquele exemplo\", \"continua\", \"agora faz\".\n"
    "- Follow-up structure with omitted subject: starts with \"E ...?\", \"E isso...?\", \"E então...?\", \"Também...?\".\n\n"
    "Default:\n"
    "- If none of the hard rules match, choose standalone.\n\n"
    "Confidence:\n"
    "- 0.9-1.0 when a hard rule matches strongly.\n"
    "- 0.6-0.8 when borderline.\n\n"
    "Examples:\n"
    "- \"1+1\" => standalone\n"
    "- \"Que horas são?\" => standalone\n"
    "- \"E isso funciona em produção?\" => recent\n"
    "- \"Faz o mesmo para o outro\" => recent"
)


@dataclass
class ContextPolicyDecision:
    mode: str  # "standalone" | "recent"
    recent_turns: int  # 0 | 6
    confidence: float


class ContextPolicyClassifier:
    """Small deterministic classifier that decides if we should include recent conversation context.

    It MUST not mention or infer tools. Output is strict JSON with keys: mode, recent_turns, confidence.
    """

    def __init__(
        self,
        *,
        base_url: str,
        chat_path: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_s: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
        self.model = model
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_s)

    async def classify(self, current_user_message: str) -> ContextPolicyDecision:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 120,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": current_user_message},
            ],
        }

        url = f"{self.base_url}{self.chat_path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content) if isinstance(content, str) else content

        mode = obj.get("mode", "standalone")
        recent_turns = int(obj.get("recent_turns", 0))
        confidence = float(obj.get("confidence", 0.0))

        # Final defensive normalization (never throw; default to standalone)
        if mode not in ("standalone", "recent"):
            mode = "standalone"
        if mode == "recent":
            recent_turns = 6
        else:
            recent_turns = 0
        if not (0.0 <= confidence <= 1.0):
            confidence = 0.0

        return ContextPolicyDecision(mode=mode, recent_turns=recent_turns, confidence=confidence)
