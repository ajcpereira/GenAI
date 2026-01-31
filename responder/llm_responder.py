import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("genai.responder")


class LLMResponder:
    """Legacy responder interface (kept for compatibility).

    Current system uses responder.Responder as the final LLM stage, but some
    external scripts may still import this class.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.model_name = self.cfg.get("model")
        self.base_url = self.cfg.get("base_url")
        self.api_key = self.cfg.get("api_key")

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
        intent_summary = str(intent.get("summary") or "").strip() or str(user_message or "").strip() or "unknown"
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
