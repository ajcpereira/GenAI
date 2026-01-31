import logging
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.session_store import Checkpoint, MessageRow, SessionStore
from orchestrator.summarizer import VLLMSummarizer
from orchestrator.token_estimator import TokenBudget, TokenEstimator

logger = logging.getLogger("genai.context_manager")


class ContextManager:
    """
    Builds conversation_context for planner + responder using:
      - latest summary checkpoint (persisted)
      - recent raw messages after checkpoint
    Performs deterministic compaction when exceeding budget.
    """

    def __init__(self, *, store: SessionStore, cfg: Dict[str, Any]):
        self.store = store
        self.cfg = cfg or {}

        # Budget is initialized from cfg but is expected to be overridden at runtime
        # with the model's declared context length (vLLM /v1/models) during app startup.
        self._apply_budget(max_context_tokens=int(self.cfg.get("max_context_tokens", 8192)))

        self.max_recent_messages = int(self.cfg.get("max_recent_messages", 20))
        self.keep_recent_messages = int(self.cfg.get("keep_recent_messages", 12))
        self.checkpoint_max_tokens = int(self.cfg.get("checkpoint_max_tokens", 512))

        summ_cfg = dict(self.cfg.get("summarizer") or {})
        if "max_tokens" not in summ_cfg:
            summ_cfg["max_tokens"] = self.checkpoint_max_tokens
        self.summarizer = VLLMSummarizer(summ_cfg)

        self.enabled = bool(self.cfg.get("enabled", True))

        # Planner reliability: assistant messages in the planning context frequently cause anchoring
        # and propagation of earlier wrong answers. Default to user-only transcripts.
        self.include_assistant_messages = bool(self.cfg.get("include_assistant_messages", False))

    def _apply_budget(self, *, max_context_tokens: int) -> None:
        reserved_output_tokens = int(self.cfg.get("reserved_output_tokens", 512))
        target_prompt_tokens = int(self.cfg.get("target_prompt_tokens", 0) or 0)
        chars_per_token = float(self.cfg.get("chars_per_token", 4.0))
        safety_factor = float(self.cfg.get("safety_factor", 1.2))

        budget = TokenBudget(
            max_context_tokens=int(max_context_tokens),
            reserved_output_tokens=reserved_output_tokens,
            target_prompt_tokens=target_prompt_tokens,
            chars_per_token=chars_per_token,
            safety_factor=safety_factor,
        )
        if budget.target_prompt_tokens <= 0:
            budget = TokenBudget(
                max_context_tokens=budget.max_context_tokens,
                reserved_output_tokens=budget.reserved_output_tokens,
                target_prompt_tokens=max(1, budget.max_context_tokens - budget.reserved_output_tokens),
                chars_per_token=budget.chars_per_token,
                safety_factor=budget.safety_factor,
            )

        self.budget = budget
        self.estimator = TokenEstimator(budget)

    def set_max_context_tokens(self, max_context_tokens: int) -> None:
        """Override context budget at runtime (e.g., derived from vLLM model metadata)."""
        try:
            m = int(max_context_tokens)
        except Exception:
            return
        if m <= 0:
            return
        prev = getattr(self, "budget", None)
        prev_m = int(prev.max_context_tokens) if prev else None
        self._apply_budget(max_context_tokens=m)
        logger.info(
            "context_budget_set",
            extra={"max_context_tokens": m, "prev_max_context_tokens": prev_m, "target_prompt_tokens": self.budget.target_prompt_tokens},
        )

    @staticmethod
    def _format_transcript(messages: List[MessageRow], *, include_assistant: bool) -> str:
        lines: List[str] = []
        for m in messages:
            role = m.role.lower()
            if (not include_assistant) and role == "assistant":
                continue
            if role not in ("user", "assistant", "system"):
                role = "other"
            lines.append(f"[{m.seq}] {role.upper()}: {m.content}")
        return "\n".join(lines).strip()

    def _format_context(self, checkpoint: Optional[Checkpoint], recent_messages: List[MessageRow]) -> str:
        parts: List[str] = []
        if checkpoint is not None and checkpoint.summary.strip():
            parts.append(f"SUMMARY_UP_TO_SEQ {checkpoint.covers_seq_end}:\n{checkpoint.summary.strip()}")
        if recent_messages:
            parts.append("RECENT_MESSAGES:\n" + ContextManager._format_transcript(recent_messages, include_assistant=self.include_assistant_messages))
        return "\n\n".join(parts).strip()

    async def build_context(
        self,
        *,
        session_id: str,
        user_id: Optional[str],
        locale: str,
        current_request_id: str,
        current_user_seq: int,
        current_user_message: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (conversation_context, debug_meta). Context excludes the current user message text
        (it is provided separately via PlannerInput.user_message / request message).
        """
        if not self.enabled:
            return "", {"enabled": False}

        checkpoint = await self.store.get_latest_checkpoint(session_id)
        cp_end = checkpoint.covers_seq_end if checkpoint else 0

        # recent messages are those after checkpoint and before current user message seq
        # fetch last max_recent_messages for safety (deterministic window)
        recent = await self.store.get_recent_messages(session_id=session_id, before_seq=current_user_seq, limit=self.max_recent_messages)

        # If we have a checkpoint, we only want messages after cp_end
        if cp_end > 0:
            recent = [m for m in recent if m.seq > cp_end]

        context = self._format_context(checkpoint, recent)
        # Include current user message in estimation (but not in context payload)
        estimated_prompt = f"{context}\n\nCURRENT_USER_MESSAGE:\n{current_user_message}"
        est_tokens = self.estimator.estimate_tokens(estimated_prompt)

        debug: Dict[str, Any] = {
            "enabled": True,
            "checkpoint_end": cp_end,
            "recent_count": len(recent),
            "estimated_prompt_tokens": est_tokens,
            "target_prompt_tokens": self.budget.target_prompt_tokens,
            "compacted": False,
        }

        if est_tokens <= self.budget.target_prompt_tokens:
            return context, debug

        # Compaction loop: create/extend checkpoint to keep only last keep_recent_messages verbatim.
        # Deterministic policy: keep last K messages (excluding current user message) as raw.
        latest_seq_before_current = current_user_seq - 1
        new_cp_end = max(cp_end, max(0, latest_seq_before_current - self.keep_recent_messages))
        if new_cp_end <= cp_end:
            # Can't extend checkpoint; reduce recent messages deterministically until it fits.
            trimmed = list(recent)
            while trimmed and not self.estimator.fits(f"{self._format_context(checkpoint, trimmed)}\n\nCURRENT_USER_MESSAGE:\n{current_user_message}"):
                trimmed.pop(0)  # drop oldest
            context2 = self._format_context(checkpoint, trimmed)
            debug["recent_count"] = len(trimmed)
            debug["estimated_prompt_tokens"] = self.estimator.estimate_tokens(f"{context2}\n\nCURRENT_USER_MESSAGE:\n{current_user_message}")
            debug["compacted"] = True
            debug["compaction_mode"] = "trim_recent_only"
            return context2, debug

        # Build transcript to summarize: messages (cp_end+1 .. new_cp_end)
        to_summarize = await self.store.get_messages_range(session_id=session_id, seq_start=cp_end + 1, seq_end=new_cp_end)
        transcript = self._format_transcript(to_summarize)

        try:
            merged_summary = await self.summarizer.summarize(
                existing_summary=(checkpoint.summary if checkpoint else None),
                transcript=transcript,
                locale=locale,
            )
        except Exception as e:
            # If summarization fails, fall back to trimming recent only (never crash).
            logger.warning("summarization_failed", extra={"session_id": session_id, "error": str(e)})
            trimmed = list(recent)
            while trimmed and not self.estimator.fits(f"{self._format_context(checkpoint, trimmed)}\n\nCURRENT_USER_MESSAGE:\n{current_user_message}"):
                trimmed.pop(0)
            context2 = self._format_context(checkpoint, trimmed)
            debug["recent_count"] = len(trimmed)
            debug["estimated_prompt_tokens"] = self.estimator.estimate_tokens(f"{context2}\n\nCURRENT_USER_MESSAGE:\n{current_user_message}")
            debug["compacted"] = True
            debug["compaction_mode"] = "trim_recent_fallback"
            return context2, debug

        meta = {
            "locale": locale,
            "prev_checkpoint_end": cp_end,
            "new_checkpoint_end": new_cp_end,
            "summarized_message_count": len(to_summarize),
            "strategy": "merge_checkpoint",
        }
        await self.store.upsert_checkpoint(
            session_id=session_id,
            user_id=user_id,
            covers_seq_end=new_cp_end,
            summary=merged_summary,
            meta=meta,
        )

        # Rebuild context with new checkpoint and post-checkpoint recent messages
        checkpoint2 = Checkpoint(covers_seq_end=new_cp_end, summary=merged_summary, meta=meta)
        recent2 = await self.store.get_recent_messages(session_id=session_id, before_seq=current_user_seq, limit=self.max_recent_messages)
        recent2 = [m for m in recent2 if m.seq > new_cp_end]
        context2 = self._format_context(checkpoint2, recent2)
        est2 = self.estimator.estimate_tokens(f"{context2}\n\nCURRENT_USER_MESSAGE:\n{current_user_message}")

        # If still too large, deterministically trim oldest recent messages.
        if est2 > self.budget.target_prompt_tokens and recent2:
            trimmed = list(recent2)
            while trimmed and not self.estimator.fits(f"{self._format_context(checkpoint2, trimmed)}\n\nCURRENT_USER_MESSAGE:\n{current_user_message}"):
                trimmed.pop(0)
            context2 = self._format_context(checkpoint2, trimmed)
            est2 = self.estimator.estimate_tokens(f"{context2}\n\nCURRENT_USER_MESSAGE:\n{current_user_message}")
            recent2 = trimmed

        debug.update(
            {
                "checkpoint_end": new_cp_end,
                "recent_count": len(recent2),
                "estimated_prompt_tokens": est2,
                "compacted": True,
                "compaction_mode": "checkpoint",
            }
        )
        return context2, debug
