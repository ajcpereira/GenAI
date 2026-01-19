from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..grounding.evidence import EvidenceItem, EvidencePack
from ..grounding.grading import Grade, grade_evidence
from ...tools.mcp_client import MCPClient


@dataclass
class RetrievalStats:
    iterations: int = 0
    tool_calls: int = 0
    stop_reason: str = "none"
    duration_ms: int = 0
    relevance_ratio: float = 0.0
    distinct_domains: int = 0
    authority_score: float = 0.0
    usable_items: int = 0


class GroundingEngine:
    """
    Governed retrieval loop:
      retrieve -> normalize -> grade -> refine/stop
    """

    def __init__(self, mcp: MCPClient):
        self.mcp = mcp

    async def run_web(
        self,
        *,
        tool_name: str,
        query_ladder: List[str],
        topk_schedule: List[int],
        max_iterations: int,
        timeout_budget_ms: int,
        question_type: str,
        entity_hint: str,
        min_relevance_ratio: float,
        min_distinct_domains: int,
        min_sources: int,
        allowlist_domains: Optional[List[str]] = None,
    ) -> Tuple[EvidencePack, RetrievalStats]:
        t0 = time.perf_counter()
        stats = RetrievalStats()

        merged: List[EvidenceItem] = []
        seen_urls = set()

        deadline = time.perf_counter() + (timeout_budget_ms / 1000.0)

        def add_items(results: List[Dict[str, Any]]) -> int:
            added = 0
            allow = [d.lower() for d in allowlist_domains] if allowlist_domains else None

            for r in results:
                if not isinstance(r, dict):
                    continue

                url = str(r.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue

                title = str(r.get("title") or "").strip()
                snippet = str(r.get("snippet") or "").strip()

                it = EvidenceItem(
                    kind="web_result",
                    title=title,
                    url=url,
                    text=snippet,
                    source_tool=tool_name,
                    score=1.0,
                )

                if allow and it.domain and it.domain not in allow:
                    continue

                merged.append(it)
                seen_urls.add(url)
                added += 1

            return added

        stop_reason = "max_iterations"
        last_grade: Optional[Grade] = None

        for i in range(max_iterations):
            if time.perf_counter() > deadline:
                stop_reason = "deadline_exceeded"
                break

            q = query_ladder[min(i, len(query_ladder) - 1)] if query_ladder else ""
            k = int(topk_schedule[min(i, len(topk_schedule) - 1)]) if topk_schedule else 5

            stats.iterations += 1
            stats.tool_calls += 1

            resp = await self.mcp.call_tool(tool_name, {"query": q, "top_k": k})
            err = str(resp.get("error") or "").strip()

            if err:
                # Fail-fast on deterministic misconfiguration
                if err.startswith("misconfigured:"):
                    stop_reason = f"misconfigured:{err}"
                    break
                stop_reason = f"upstream_error:{err[:120]}"
                continue

            results = resp.get("results") if isinstance(resp.get("results"), list) else []
            added = add_items(results)

            # Grade the candidate set (bounded slice for determinism)
            slice_items = merged[: max(5, min(15, len(merged)))]
            last_grade = grade_evidence(
                question_type=question_type,
                entity_hint=entity_hint,
                items=slice_items,
                min_relevance_ratio=min_relevance_ratio,
                min_distinct_domains=min_distinct_domains,
                min_sources=min_sources,
            )

            stats.relevance_ratio = last_grade.relevance_ratio
            stats.distinct_domains = last_grade.distinct_domains
            stats.authority_score = last_grade.authority_score
            stats.usable_items = last_grade.usable_items

            if last_grade.stop_reason == "sufficient_candidate_set":
                stop_reason = "evidence_sufficient"
                break

            if added == 0 and i >= 1:
                stop_reason = "no_new_results"
                break

            stop_reason = last_grade.stop_reason

        stats.stop_reason = stop_reason
        stats.duration_ms = int((time.perf_counter() - t0) * 1000)

        # Deterministic per-item scoring (lightweight)
        hint = (entity_hint or "").strip().lower()
        for it in merged:
            it.score = 1.0
            if hint and hint in f"{it.title}\n{it.text}\n{it.url}".lower():
                it.score += 0.5

        pack = EvidencePack(
            kind="web",
            query=query_ladder[0] if query_ladder else "",
            items=sorted(merged, key=lambda x: x.score, reverse=True)[:10],
            errors=[],
            meta={
                "stop_reason": stop_reason,
                "graded": last_grade.__dict__ if last_grade else None,
                "queries": query_ladder,
            },
        )
        return pack, stats
