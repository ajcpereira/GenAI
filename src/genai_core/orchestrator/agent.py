from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from ..runtime.state import RuntimeState
from ..tools.mcp_client import MCPClient
from ..vllm.openai_client import VLLMOpenAIClient

from .tool_catalog import ToolCatalog
from .prompts import system_answer_pt
from .decisioning import Decision, route_with_llm_router
from .grounding.classify import classify_question
from .grounding.plan import build_plan
from .grounding.query_refiner import build_query_ladder, infer_domain_allowlist
from .grounding.engine import GroundingEngine, RetrievalStats
from .grounding.resolve import resolve_from_evidence
from .grounding.evidence import EvidencePack, EvidenceItem

log = logging.getLogger("genai_core.orchestrator")


class OrchestratorAgent:
    """
    Enterprise-grade Orchestrator (policy-first grounding).

    Key rules:
      - Volatile factual questions require grounding (web_search) unless clarified first.
      - If evidence is insufficient -> ALWAYS inconclusive (per product decision 1-A).
      - If question is ambiguous -> ask ONE clarification (per product decision 2-A).
      - LLM router is only used for non-volatile / low-stakes questions.
    """

    def __init__(self, cfg: dict, runtime: RuntimeState, mcp: MCPClient):
        self.cfg = cfg or {}
        self.runtime = runtime
        self.mcp = mcp
        self.catalog = ToolCatalog(self.cfg)

        orch_cfg = self.cfg.get("orchestrator", {}) if isinstance(self.cfg.get("orchestrator"), dict) else {}

        self.vllm_base_url = str(orch_cfg.get("vllm_base_url", "http://127.0.0.1:8001")).rstrip("/")
        self.model = orch_cfg.get("model") or (
            runtime.model_info.model_name if getattr(runtime, "model_info", None) else "mistral-7b-instruct"
        )
        self.request_timeout_s = int(orch_cfg.get("request_timeout_s", 120))

        self.max_answer_tokens = int(orch_cfg.get("max_answer_tokens", 512))
        self.answer_temperature = float(orch_cfg.get("answer_temperature", 0.1))

        self.router_max_tokens = int(orch_cfg.get("router_max_tokens", 256))
        self.router_temperature = float(orch_cfg.get("router_temperature", 0.0))

        self.tool_budget_per_session = int(orch_cfg.get("tool_budget_per_session", 3))
        self._session_tool_uses: Dict[str, int] = {}

        self.llm = VLLMOpenAIClient(base_url=self.vllm_base_url, timeout_s=self.request_timeout_s)
        self.engine = GroundingEngine(self.mcp)

    # ---------------- budgets ----------------
    def _session_key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    def _tool_budget_ok(self, user_id: str, session_id: str) -> bool:
        used = self._session_tool_uses.get(self._session_key(user_id, session_id), 0)
        return used < self.tool_budget_per_session

    def _inc_tool_budget(self, user_id: str, session_id: str) -> None:
        k = self._session_key(user_id, session_id)
        self._session_tool_uses[k] = self._session_tool_uses.get(k, 0) + 1

    # ---------------- tool selection ----------------
    def _pick_web_tool(self) -> Optional[tuple[str, str]]:
        t = self.catalog.pick_by_tag("sources") or self.catalog.pick_by_tag("freshness") or self.catalog.pick_by_tag("search")
        if not t:
            t = self.catalog.get_tool("web_search")
        if not t:
            return None
        return t.key, t.tool_name

    # ---------------- main ----------------
    async def chat(self, user_id: str, session_id: str, message: str, correlation_id: str = "") -> Dict[str, Any]:
        t0 = time.perf_counter()
        msg = (message or "").strip()

        # 1) Classify (deterministic)
        spec = classify_question(msg)

        # 2) Clarification path (2-A)
        if spec.ambiguous and spec.clarification:
            return self._response(
                user_id=user_id,
                session_id=session_id,
                correlation_id=correlation_id,
                decision=Decision("llm", None, None, {}, "clarification_required", 0.99),
                answer=spec.clarification,
                tool_invoked=False,
                evidence=None,
                retrieval=None,
                total_ms=int((time.perf_counter() - t0) * 1000),
            )

        # 3) Policy-first grounding for volatile question types
        need_grounding = bool(spec.volatile)

        decision: Decision
        evidence: Optional[EvidencePack] = None
        retrieval: Optional[RetrievalStats] = None
        tool_invoked = False

        if need_grounding:
            picked = self._pick_web_tool()
            if not picked:
                # No tool available -> safe inconclusive (1-A)
                return self._response(
                    user_id, session_id, correlation_id,
                    Decision("llm", None, None, {}, "no_web_tool_available", 0.80),
                    "Não consigo confirmar com segurança porque não tenho uma ferramenta de pesquisa configurada.",
                    False, None, None, int((time.perf_counter() - t0) * 1000),
                )

            tool_key, tool_name = picked
            decision = Decision("mcp_tool", tool_key, tool_name, {"query": msg, "top_k": 5}, "policy.grounding_required", 0.95)

            if not self.mcp.enabled:
                return self._response(
                    user_id, session_id, correlation_id,
                    Decision("llm", None, None, {}, "mcp_disabled", 0.80),
                    "Não consigo confirmar com segurança porque a ferramenta de pesquisa está desativada.",
                    False, None, None, int((time.perf_counter() - t0) * 1000),
                )

            if not self._tool_budget_ok(user_id, session_id):
                return self._response(
                    user_id, session_id, correlation_id,
                    Decision("llm", None, None, {}, "tool_budget_exceeded", 0.80),
                    "Não consigo confirmar com segurança porque esgotei o orçamento de pesquisa nesta sessão.",
                    False, None, None, int((time.perf_counter() - t0) * 1000),
                )

            self._inc_tool_budget(user_id, session_id)
            tool_invoked = True

            # 3.1 Build plan (query ladder + allowlists)
            query_ladder = build_query_ladder(spec.question_type, spec.raw, spec.entity_hint)
            allowlist = infer_domain_allowlist(spec.question_type, spec.entity_hint)

            plan = build_plan(
                tool_key=tool_key,
                tool_name=tool_name,
                question_type=spec.question_type,
                raw_question=spec.raw,
                entity_hint=spec.entity_hint,
                query_ladder=query_ladder,
            )

            # 3.2 Run retrieval loop
            evidence, retrieval = await self.engine.run_web(
                tool_name=plan.tool_name,
                query_ladder=plan.query_ladder,
                topk_schedule=plan.topk_schedule,
                max_iterations=plan.max_iterations,
                timeout_budget_ms=plan.timeout_budget_ms,
                question_type=spec.question_type,
                entity_hint=spec.entity_hint,
                min_relevance_ratio=plan.min_relevance_ratio,
                min_distinct_domains=plan.min_distinct_domains,
                min_sources=plan.min_sources,
                allowlist_domains=allowlist,
            )

            # 3.3 Deterministic resolution gate (versions/dates)
            resolution = resolve_from_evidence(
                question_type=spec.question_type,
                entity_hint=spec.entity_hint,
                evidence=evidence,
                require_consensus=plan.require_consensus,
                min_consensus=plan.min_consensus,
            )

            # If resolution not ok -> ALWAYS inconclusive (1-A)
            if not resolution.ok:
                ans = "Não consigo confirmar com segurança com a evidência recolhida neste momento."
                ans = self._append_sources(ans, evidence)
                return self._response(
                    user_id, session_id, correlation_id,
                    decision,
                    "Nota: usei uma tool via MCP para enriquecer esta resposta.\n\n" + ans,
                    True,
                    evidence,
                    retrieval,
                    int((time.perf_counter() - t0) * 1000),
                )

            # If deterministic answer_line exists, return it (still with sources)
            if resolution.answer_line:
                ans = resolution.answer_line
                ans = self._append_sources(ans, evidence, prefer=resolution.supporting_urls)
                return self._response(
                    user_id, session_id, correlation_id,
                    decision,
                    "Nota: usei uma tool via MCP para enriquecer esta resposta.\n\n" + ans,
                    True,
                    evidence,
                    retrieval,
                    int((time.perf_counter() - t0) * 1000),
                )

            # Otherwise synthesize with LLM from evidence only
            answer = await self._synthesize_with_llm(msg, evidence)
            answer = self._append_sources(answer, evidence)
            return self._response(
                user_id, session_id, correlation_id,
                decision,
                "Nota: usei uma tool via MCP para enriquecer esta resposta.\n\n" + answer,
                True,
                evidence,
                retrieval,
                int((time.perf_counter() - t0) * 1000),
            )

        # 4) Non-volatile: allow LLM router fallback
        decision = await route_with_llm_router(
            llm=self.llm,
            model=self.model,
            catalog=self.catalog,
            message=msg,
            max_tokens=self.router_max_tokens,
            temperature=self.router_temperature,
        )

        # If router chooses tool, run tool once (governed) but still safe
        if decision.route == "mcp_tool":
            if not self.mcp.enabled or not self._tool_budget_ok(user_id, session_id):
                decision = Decision("llm", None, None, {}, "tool_unavailable_fallback", 0.70)
            else:
                self._inc_tool_budget(user_id, session_id)
                tool_invoked = True
                # best-effort single-shot
                tool_name = decision.tool_name or ""
                resp = await self.mcp.call_tool(tool_name, decision.tool_args or {})
                # normalize
                evidence = EvidencePack(kind="tool", query=str((decision.tool_args or {}).get("query") or ""), items=[], errors=[], meta={"tool": tool_name})

                if isinstance(resp, dict):
                    err = str(resp.get("error") or "").strip()
                    if err:
                        evidence.errors.append(err)

                    results = resp.get("results") if isinstance(resp.get("results"), list) else []
                    for r in results[:8]:
                        if isinstance(r, dict):
                            evidence.items.append(
                                EvidenceItem(
                                    kind="tool_row",
                                    title=str(r.get("title") or ""),
                                    url=str(r.get("url") or ""),
                                    text=str(r.get("snippet") or ""),
                                    data=r,
                                    source_tool=tool_name,
                                    score=1.0,
                                )
                            )
                        else:
                            evidence.items.append(
                                EvidenceItem(
                                    kind="tool_row",
                                    title="",
                                    url="",
                                    text=str(r),
                                    data={"value": r},
                                    source_tool=tool_name,
                                    score=1.0,
                                )
                            )


        # LLM direct answer
        answer = await self._synthesize_with_llm(msg, evidence if tool_invoked else None)
        if tool_invoked:
            answer = "Nota: usei uma tool via MCP para enriquecer esta resposta.\n\n" + self._append_sources(answer, evidence)
        return self._response(
            user_id, session_id, correlation_id,
            decision,
            answer,
            tool_invoked,
            evidence if tool_invoked else None,
            retrieval,
            int((time.perf_counter() - t0) * 1000),
        )

    async def _synthesize_with_llm(self, msg: str, evidence: Optional[EvidencePack]) -> str:
        user_parts = [f"Pergunta:\n{msg}"]
        if evidence and evidence.items:
            lines = []
            for i, it in enumerate(evidence.items[:8], 1):
                if it.url:
                    lines.append(f"[{i}] {it.title}\n{it.url}\n{it.text}".strip())
                else:
                    lines.append(f"[{i}] {json.dumps(it.data or {}, ensure_ascii=False)}")
            user_parts.append("Evidência:\n" + "\n\n".join(lines))

        user_prompt = "\n\n".join(user_parts).strip()

        raw = await self.llm.chat_completion(
            model=self.model,
            system=system_answer_pt(evidence_present=bool(evidence and evidence.items)),
            user=user_prompt,
            max_tokens=self.max_answer_tokens,
            temperature=self.answer_temperature,
            extra={"stop": ["</s>"], "repetition_penalty": 1.15},
        )
        return (raw or "").strip() or "Não consigo confirmar com segurança."

    def _append_sources(self, answer: str, evidence: Optional[EvidencePack], prefer: Optional[list[str]] = None) -> str:
        urls = []
        if evidence:
            urls = evidence.top_urls(3)
        prefer = prefer or []
        for u in reversed(prefer):
            if u and u not in urls:
                urls.insert(0, u)
            elif u in urls:
                urls.remove(u)
                urls.insert(0, u)

        urls = urls[:3]
        if not urls:
            return answer + "\n\nFontes:\n- (nenhuma URL disponível)"
        out = answer + "\n\nFontes:"
        for u in urls:
            out += f"\n- {u}"
        return out

    def _response(
        self,
        user_id: str,
        session_id: str,
        correlation_id: str,
        decision: Decision,
        answer: str,
        tool_invoked: bool,
        evidence: Optional[EvidencePack],
        retrieval: Optional[RetrievalStats],
        total_ms: int,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        resp: Dict[str, Any] = {
            "user_id": user_id,
            "session_id": session_id,
            "decision": {
                "route": decision.route,
                "tool_key": decision.tool_key,
                "tool_name": decision.tool_name,
                "tool_args": decision.tool_args,
                "reason": decision.reason,
                "confidence": float(decision.confidence),
            },
            "answer": answer,
            "meta": {
                "model": self.model,
                "vllm_base_url": self.vllm_base_url,
                "endpoint": getattr(self.llm, "last_endpoint", "") or "",
                "tool_invoked": bool(tool_invoked),
                "correlation_id": correlation_id,
                "latency_ms": total_ms,
                "grounding": {
                    "evidence_kind": evidence.kind if evidence else None,
                    "evidence_items": len(evidence.items) if evidence else 0,
                    "errors": evidence.errors if (evidence and evidence.errors) else [],
                    "retrieval": asdict(retrieval) if retrieval else None,
                },
            },
            "correlation_id": correlation_id,
        }
        if error:
            resp["error"] = error
        log.info(json.dumps({"event": "chat.done", "cid": correlation_id, "route": decision.route, "latency_ms": total_ms}, ensure_ascii=False))
        return resp
