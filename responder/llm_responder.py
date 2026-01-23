# responder/llm_responder.py
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("genai.responder")


class LLMResponder:
    """
    Interface expected by main.py:
        responder = LLMResponder(cfg.get('responder', {}))

    Interface expected by orchestrator:
        await responder.generate(user_message=..., intent=..., steps_executed=...) -> AnswerPayload
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}

        # v1 stable: responder is "local" (no upstream call). Keep placeholders for future.
        self.model_name = self.cfg.get("model", None)
        self.base_url = self.cfg.get("base_url", None)
        self.api_key = self.cfg.get("api_key", None)

        logger.info(
            "responder_initialized",
            extra={
                "model": self.model_name,
                "base_url": self.base_url,
                "has_api_key": bool(self.api_key),
            },
        )

    async def generate(
        self,
        *,
        user_message: str,
        intent: Dict[str, Any],
        steps_executed: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Returns AnswerPayload per internal-json.json:
          {
            "answer": "...",
            "final_context": {
              "intent": "<string>",
              "steps_executed": [StepExecution...]
            }
          }
        """
        intent_summary = str(intent.get("summary") or "").strip()
        if not intent_summary:
            intent_summary = str(user_message or "").strip() or "unknown"

        # v1 stable answer (no vLLM call yet)
        answer = (
            f"Pedido: {user_message}\n"
            f"Intenção: {intent_summary}\n"
            f"Passos executados: {len(steps_executed)}"
        )

        return {
            "answer": answer,
            "final_context": {
                "intent": intent_summary,
                "steps_executed": steps_executed,
            },
        }
