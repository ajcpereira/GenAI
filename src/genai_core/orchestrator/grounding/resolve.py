from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .evidence import EvidencePack
from .grading import consensus, extract_date_candidates, extract_version_candidates


@dataclass(frozen=True)
class Resolution:
    ok: bool
    answer_line: str
    supporting_urls: List[str]
    reason: str


def resolve_from_evidence(
    *,
    question_type: str,
    entity_hint: str,
    evidence: EvidencePack,
    require_consensus: bool,
    min_consensus: int,
) -> Resolution:
    """
    Deterministic resolution when possible.
    If not possible, return ok=False for safe inconclusive response (1-A).
    """
    if not evidence or not evidence.items:
        return Resolution(False, "", [], "no_evidence")

    urls = evidence.top_urls(3)

    if question_type == "date":
        cands = extract_date_candidates(evidence.items[:3])
        con = consensus(cands, min_consensus=min_consensus)
        if not con:
            return Resolution(False, "", urls, "date_no_consensus")
        val, u = con
        supporting = (u or urls)[:3]
        return Resolution(True, f"Hoje é {val}.", supporting, "date_consensus")

    if question_type == "latest_version":
        cands = extract_version_candidates(evidence.items[:3])
        con = consensus(cands, min_consensus=min_consensus if require_consensus else 1)
        if not con:
            return Resolution(False, "", urls, "version_no_consensus")
        val, u = con
        supporting = (u or urls)[:3]
        # Do NOT claim “definitive” beyond evidence.
        return Resolution(True, f"A versão mais recente (com base na evidência recolhida) é {val}.", supporting, "version_consensus")

    # For ownership/acquisition/news, we do not do deterministic extraction here.
    # The LLM will summarize, but ONLY from evidence; Orchestrator enforces inconclusive if evidence insufficient.
    return Resolution(True, "", urls, "llm_synthesis_required")
