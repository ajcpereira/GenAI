from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..runtime.state import RuntimeState
from ..tools.mcp_client import MCPClient
from ..tools.token_counter import TokenCounter
from ..vllm.openai_client import VLLMOpenAIClient


log = logging.getLogger("genai_core.orchestrator")


@dataclass
class ToolDecision:
    use_web_search: bool = False
    use_rag: bool = False
    web_query: Optional[str] = None


class OrchestratorAgent:
    """
    Orchestrator controls prompting and decoding.

    This implementation:
      - Uses /v1/completions with explicit Mistral [INST] formatting for stability.
      - Applies decoding policies per request (temperature/max_tokens/stop/repetition_penalty).
      - Provides deterministic bypasses for ultra-short strict-format intents (Phase 1 guardrails).
      - Optionally enriches context via MCP web_search and always reports sources if used.
      - Includes pragmatic anti-runaway post-processing safeguards.
    """

    def __init__(self, cfg: dict, runtime: RuntimeState, mcp: MCPClient):
        self.cfg = cfg
        self.runtime = runtime
        self.mcp = mcp

        orch_cfg = cfg.get("orchestrator", {})
        self.vllm_base_url = orch_cfg.get("vllm_base_url", "http://127.0.0.1:8001")
        self.model = orch_cfg.get("model", runtime.model_info.model_name if runtime.model_info else "local-model")
        self.request_timeout_s = int(orch_cfg.get("request_timeout_s", 120))
        self.reserved_output_tokens = int(orch_cfg.get("reserved_output_tokens", 128))
        self.max_tokens_cap = int(orch_cfg.get("max_tokens_cap", 256))

        self.llm = VLLMOpenAIClient(base_url=self.vllm_base_url, timeout_s=self.request_timeout_s)
        self.tokens = TokenCounter(tokenizer_path=(runtime.model_info.tokenizer_name_or_path if runtime.model_info else None))

    def _should_consider_web(self, message: str) -> bool:
        m = (message or "").lower()
        triggers = ["hoje", "agora", "notícias", "últimas", "atual", "versão", "release", "2025", "2026"]
        return any(t in m for t in triggers)

    def _choose_decoding(self, message: str, max_new_default: int) -> Dict[str, Any]:
        msg = (message or "").strip().lower()
        max_tokens = min(max_new_default, self.reserved_output_tokens, self.max_tokens_cap)

        # Short-format requests (single word / very short answer)
        if "apenas" in msg and "palavra" in msg:
            return {
                "max_tokens": 12,
                "temperature": 0.0,
                "extra": {
                    "repetition_penalty": 1.15,
                    "stop": ["\n\n", "\r\n\r\n"],
                },
            }

        # Default
        return {
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "extra": {
                "repetition_penalty": 1.15,
                "stop": ["</s>", "\n\n\n", "\r\n\r\n\r\n"],
            },
        }

    def _mistral_inst_prompt(self, system: str, user: str) -> str:
        system = (system or "").strip()
        user = (user or "").strip()
        if system:
            content = f"{system}\n\n{user}"
        else:
            content = user
        return f"[INST] {content} [/INST]"

    async def chat(self, user_id: str, session_id: str, message: str) -> Dict[str, Any]:
        msg_l = (message or "").strip().lower()

        # Deterministic short-format bypasses (production-style guardrails)
        if "diz apenas a palavra ok" in msg_l or ("apenas" in msg_l and "palavra" in msg_l and "ok" in msg_l):
            return {
                "user_id": user_id,
                "session_id": session_id,
                "decision": {"use_web_search": False, "use_rag": False, "web_query": None},
                "answer": "OK",
                "meta": {
                    "model": self.model,
                    "max_tokens": 0,
                    "vllm_base_url": self.vllm_base_url,
                    "endpoint": "bypass",
                },
            }

        if msg_l in ("como te chamas?", "como te chamas", "qual é o teu nome?", "qual e o teu nome?"):
            return {
                "user_id": user_id,
                "session_id": session_id,
                "decision": {"use_web_search": False, "use_rag": False, "web_query": None},
                "answer": "Sou um assistente de IA.",
                "meta": {
                    "model": self.model,
                    "max_tokens": 0,
                    "vllm_base_url": self.vllm_base_url,
                    "endpoint": "bypass",
                },
            }

        limits = self.runtime.model_info.model_limits if self.runtime.model_info else {
            "max_context_tokens": 4096,
            "max_new_tokens_default": 512
        }
        max_ctx = limits["max_context_tokens"]

        decoding = self._choose_decoding(message=message, max_new_default=limits["max_new_tokens_default"])
        max_new = int(decoding["max_tokens"])

        # Tool decision: heuristic first, then model-based decision if needed
        if self._should_consider_web(message):
            decision = await self._decide_tools(message=message, max_new=min(64, max_new))
        else:
            decision = ToolDecision(use_web_search=False, use_rag=False, web_query=None)

        sources: List[Dict[str, str]] = []
        external_blocks: List[str] = []

        if decision.use_web_search and self.mcp.enabled:
            try:
                t0 = time.perf_counter()
                results = await self.mcp.web_search(decision.web_query or message)
                dt = time.perf_counter() - t0
                log.info("mcp.web_search ok duration_s=%.3f results=%d", dt, len(results))
                for r in results:
                    sources.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", ""),
                    })
                external_blocks.append(self._format_web_results(results))
            except Exception as e:
                log.warning("mcp.web_search failed: %s", str(e))
                sources.append({"title": "MCP web_search error", "url": "", "snippet": str(e)})

        # Placeholder for future RAG
        if decision.use_rag:
            external_blocks.append("[RAG placeholder]")

        # Strong system contract to avoid persona invention and to keep answers concise
        system = (
            "Responde em Português de Portugal.\n"
            "Responde apenas ao que foi pedido, de forma concisa.\n"
            "Não inventes factos nem biografias. Se não souberes, diz que não sabes.\n"
            "Se o utilizador pedir um formato estrito (ex.: 'apenas a palavra OK'), cumpre exatamente.\n"
            "Se existirem fontes externas fornecidas, usa-as e inclui-as na secção Sources."
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
            user_prompt = f"""Pergunta:
{message}

Contexto externo:
{final_context}
"""

        prompt = self._mistral_inst_prompt(system=system, user=user_prompt)

        t1 = time.perf_counter()
        answer_text = await self.llm.completion(
            model=self.model,
            prompt=prompt,
            max_tokens=max_new,
            temperature=float(decoding["temperature"]),
            extra=decoding["extra"],
        )
        dt_llm = time.perf_counter() - t1
        log.info("llm.completions duration_s=%.3f max_tokens=%d", dt_llm, max_new)

        final_answer = (answer_text or "").strip()

        # Basic anti-runaway: cut after first paragraph if the model starts sprawling
        if "\n\n" in final_answer:
            first_para = final_answer.split("\n\n", 1)[0].strip()
            if first_para:
                final_answer = first_para

        # If user asked for a single word, keep only the first token/word
        if ("apenas" in msg_l) and ("palavra" in msg_l):
            final_answer = (final_answer.split()[:1] or [""])[0]

        if not final_answer:
            final_answer = "Não consegui gerar resposta. Tenta reformular a pergunta."

        if sources:
            final_answer += "\n\nSources (internet/RAG):\n"
            for i, s in enumerate(sources, 1):
                final_answer += f"[{i}] {s.get('title','source')} – {s.get('url','')}\n"
                if s.get("snippet"):
                    final_answer += f"    {s['snippet']}\n"

        return {
            "user_id": user_id,
            "session_id": session_id,
            "decision": decision.__dict__,
            "answer": final_answer,
            "meta": {
                "model": self.model,
                "max_tokens": max_new,
                "vllm_base_url": self.vllm_base_url,
                "endpoint": "/v1/completions",
            },
        }

    async def _decide_tools(self, message: str, max_new: int) -> ToolDecision:
        tools_desc = {
            "web_search": "Use to retrieve fresh facts from the internet via MCP.",
            "rag_search": "Use to retrieve internal documents (not implemented in Phase 1).",
        }
        system = (
            "És um agente orquestrador. Decide se é necessário usar ferramentas. "
            "Responde APENAS com JSON válido (sem markdown) com as chaves: use_web_search, use_rag, web_query."
        )
        user = f"""User message:
{message}

Available tools:
{json.dumps(tools_desc, ensure_ascii=False)}

Policy:
- Use web_search only if the user asks for up-to-date facts, recent changes, current events, or specific external references.
- Otherwise do not use it.
"""

        raw = await self.llm.chat_completion(
            model=self.model,
            system=system,
            user=user,
            max_tokens=min(128, max_new),
            temperature=0.0,
            extra={"stop": ["\n\n", "\r\n\r\n"]},
        )

        decision = ToolDecision()
        try:
            data = json.loads(self._extract_json(raw))
            decision.use_web_search = bool(data.get("use_web_search", False))
            decision.use_rag = bool(data.get("use_rag", False))
            decision.web_query = data.get("web_query") or None
        except Exception:
            decision.use_web_search = False
            decision.use_rag = False
            decision.web_query = None
        return decision

    def _extract_json(self, s: str) -> str:
        s = (s or "").strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start: end + 1]
        return s

    def _format_web_results(self, results: List[Dict[str, str]]) -> str:
        lines = ["Resultados de pesquisa web:"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"[{i}] {title} ({url})\n{snippet}".strip())
        return "\n\n".join(lines)

    async def _fit_context_with_summaries(
        self,
        context: str,
        question: str,
        system: str,
        max_ctx: int,
        max_new: int,
    ) -> str:
        # Rough budgeting: leave headroom for prompt scaffolding.
        budget = max_ctx - max_new - 512
        if budget < 512:
            budget = max(256, max_ctx // 2)

        if self.tokens.count(context) <= budget:
            return context

        chunk_budget = max(256, budget // 2)
        chunks = self.tokens.chunk_text(context, chunk_budget)

        summaries: List[str] = []
        for ch in chunks:
            prompt = self._mistral_inst_prompt(
                system=system,
                user=f"Resume o texto seguinte de forma útil para responder à pergunta: {question}\n\nTexto:\n{ch}",
            )
            s = await self.llm.completion(
                model=self.model,
                prompt=prompt,
                max_tokens=min(256, max_new),
                temperature=0.0,
                extra={"stop": ["\n\n", "\r\n\r\n"], "repetition_penalty": 1.15},
            )
            summaries.append((s or "").strip())

        merged = "\n\n".join(f"Resumo {i+1}: {s}" for i, s in enumerate(summaries))
        if self.tokens.count(merged) > budget:
            prompt = self._mistral_inst_prompt(
                system=system,
                user=f"Condensa estes resumos num único texto curto para responder: {question}\n\nResumos:\n{merged}",
            )
            merged = (await self.llm.completion(
                model=self.model,
                prompt=prompt,
                max_tokens=min(256, max_new),
                temperature=0.0,
                extra={"stop": ["\n\n", "\r\n\r\n"], "repetition_penalty": 1.15},
            ) or "").strip()

        return merged
