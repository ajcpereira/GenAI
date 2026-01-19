from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class GroundingPlan:
    tool_key: str
    tool_name: str
    max_iterations: int
    topk_schedule: List[int]
    timeout_budget_ms: int

    # Evidence requirements
    min_distinct_domains: int
    min_relevance_ratio: float
    min_sources: int  # number of items considered "usable"
    require_consensus: bool  # for version/date-like extraction
    min_consensus: int  # e.g., 2-of-3

    # Query shaping
    query_ladder: List[str]
    allowlist_domains: Optional[List[str]] = None


def build_plan(
    *,
    tool_key: str,
    tool_name: str,
    question_type: str,
    raw_question: str,
    entity_hint: str,
    query_ladder: List[str],
) -> GroundingPlan:
    # Defaults: safe and bounded
    max_iterations = 3
    topk_schedule = [5, 8, 10]
    timeout_budget_ms = 5000

    # Conservative requirements
    min_distinct_domains = 2
    min_relevance_ratio = 0.50
    min_sources = 2
    require_consensus = False
    min_consensus = 2

    allowlist = None

    # Type-specific policies
    if question_type == "date":
        # Date can be answered with fewer sources, but we keep it conservative.
        min_distinct_domains = 1
        min_relevance_ratio = 0.40
        min_sources = 1
        require_consensus = True
        min_consensus = 1

    if question_type == "latest_version":
        require_consensus = True
        min_consensus = 2

    if question_type == "ownership":
        # After clarification, this can be strict, but for now we keep it similar.
        require_consensus = False

    if question_type == "acquisition" or question_type == "news":
        require_consensus = False
        min_sources = 2
        min_distinct_domains = 2

    return GroundingPlan(
        tool_key=tool_key,
        tool_name=tool_name,
        max_iterations=max_iterations,
        topk_schedule=topk_schedule,
        timeout_budget_ms=timeout_budget_ms,
        min_distinct_domains=min_distinct_domains,
        min_relevance_ratio=min_relevance_ratio,
        min_sources=min_sources,
        require_consensus=require_consensus,
        min_consensus=min_consensus,
        query_ladder=query_ladder or [raw_question],
        allowlist_domains=allowlist,
    )
