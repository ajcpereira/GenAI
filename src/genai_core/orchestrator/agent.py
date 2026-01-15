from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..runtime.state import RuntimeState
from ..tools.mcp_client import MCPClient
from ..tools.token_counter import TokenCounter
from ..vllm.openai_client import VLLMOpenAIClient


@dataclass
class ToolDecision:
    use_web_search: bool = False
    use_rag: bool = False
    web_query: Optional[str] = None


class OrchestratorAgent:
    """Orchestrator uses the vLLM model for decisions and response generation."""

    def __init__(self, cfg: dict, runtime: RuntimeState, mcp: MCPClient):
        self.cfg = cfg
        self.runtime = runtime
        self.mcp = mcp

        orch_cfg = cfg.get("orchestrator", {})
        self.vllm_base_url = orch_cfg.get("vllm_base_url", "http://127.0.0.1:8001")
        self.model = orch_cfg.get("model", runtime.model_info.model_name if runtime.model_info else "local-model")
        self.request_timeout_s = int(orch_cfg.get("request_timeout_s", 120))
        self.reserved_output_tokens = int(orch_cfg.get("reserved_output_tokens", 512))

        self.llm = VLLMOpenAIClient(base_url=self.vllm_base_url, timeout_s=self.request_timeout_s)
        self.tokens = TokenCounter(tokenizer_path=(runtime.model_info.tokenizer_name_or_path if runtime.model_info else None))

    async def chat(self, user_id: str, session_id: str, message: str) -> Dict[str, Any]:
        limits = self.runtime.model_info.model_limits if self.runtime.model_info else {"max_context_tokens": 4096, "max_new_tokens_default": 512}
        max_ctx = limits["max_context_tokens"]
        max_new = min(limits["max_new_tokens_default"], self.reserved_output_tokens)

        decision = await self._decide_tools(message=message, max_new=max_new)
        sources: List[Dict[str, str]] = []
        external_blocks: List[str] = []

        if decision.use_web_search and self.mcp.enabled:
            results = await self.mcp.web_search(decision.web_query or message)
            for r in results:
                sources.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")})
            external_blocks.append(self._format_web_results(results))

        if decision.use_rag:
            external_blocks.append("[RAG placeholder]")

        system = (
            "You are a helpful assistant. If external sources are provided, use them and be explicit about them. "
            "Never invent citations."
        )

        external_context = "\n\n".join(external_blocks).strip()
        if external_context:
            final_context = await self._fit_context_with_summaries(
                context=external_context,
                question=message,
                system=system,
                max_ctx=max_ctx,
                max_new=max_new,
            )
        else:
            final_context = ""

        user_prompt = message
        if final_context:
            user_prompt = f"""Question:
{message}

External context:
{final_context}
"""

        answer = await self.llm.chat_completion(
            model=self.model,
            system=system,
            user=user_prompt,
            max_tokens=max_new,
            temperature=0.2,
        )

        final_answer = answer.strip()
        if sources:
            final_answer += "\n\nSources (internet/RAG):\n"
            for i, s in enumerate(sources, 1):
                title = s.get("title") or "source"
                url = s.get("url") or ""
                snippet = s.get("snippet") or ""
                final_answer += f"[{i}] {title} – {url}\n"
                if snippet:
                    final_answer += f"    {snippet}\n"

        return {
            "user_id": user_id,
            "session_id": session_id,
            "decision": decision.__dict__,
            "answer": final_answer,
        }

    async def _decide_tools(self, message: str, max_new: int) -> ToolDecision:
        tools_desc = {
            "web_search": "Use to retrieve fresh facts from the internet via MCP.",
            "rag_search": "Use to retrieve internal documents (not implemented in Phase 1).",
        }
        system = (
            "You are an orchestrator. Decide whether tools are required. "
            "Return ONLY valid JSON (no markdown) with keys: use_web_search, use_rag, web_query."
        )
        user = f"""User message:
{message}

Available tools:
{json.dumps(tools_desc, ensure_ascii=False)}

Policy:
- Use web_search if the user asks for up-to-date facts, recent changes, current events, or specific external references.
- Otherwise do not use it.
"""
        raw = await self.llm.chat_completion(
            model=self.model,
            system=system,
            user=user,
            max_tokens=min(256, max_new),
            temperature=0.0,
        )

        decision = ToolDecision()
        try:
            data = json.loads(self._extract_json(raw))
            decision.use_web_search = bool(data.get("use_web_search", False))
            decision.use_rag = bool(data.get("use_rag", False))
            decision.web_query = data.get("web_query") or None
        except Exception:
            pass
        return decision

    def _extract_json(self, s: str) -> str:
        s = s.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1]
        return s

    def _format_web_results(self, results: List[Dict[str, str]]) -> str:
        lines = ["Web search results:"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"[{i}] {title} ({url})\n{snippet}".strip())
        return "\n\n".join(lines)

    async def _fit_context_with_summaries(self, context: str, question: str, system: str, max_ctx: int, max_new: int) -> str:
        budget = max_ctx - max_new - 512
        if budget < 512:
            budget = max(256, max_ctx // 2)

        if self.tokens.count(context) <= budget:
            return context

        chunk_budget = max(256, budget // 2)
        chunks = self.tokens.chunk_text(context, chunk_budget)

        summaries: List[str] = []
        for ch in chunks:
            prompt = f"""Summarize the following text for answering the question: {question}
Text:
{ch}
"""
            s = await self.llm.chat_completion(
                model=self.model,
                system=system,
                user=prompt,
                max_tokens=min(256, max_new),
                temperature=0.2,
            )
            summaries.append(s.strip())

        merged = "\n\n".join(f"Summary {i+1}: {s}" for i, s in enumerate(summaries))
        if self.tokens.count(merged) > budget:
            prompt = f"""Compress these summaries into a single concise brief to answer: {question}
Summaries:
{merged}
"""
            merged = (await self.llm.chat_completion(
                model=self.model,
                system=system,
                user=prompt,
                max_tokens=min(256, max_new),
                temperature=0.2,
            )).strip()

        return merged
