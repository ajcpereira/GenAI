from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .evidence import EvidenceItem, EvidencePack


# -----------------------------
# Date extraction regexes
# -----------------------------

# ISO: 2026-01-19
_DATE_RE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Slash formats: 19/01/2026 or 01/19/2026
_DATE_RE_SLASH = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")

# English textual dates:
# Monday, January 19, 2026
# January 19, 2026
_DATE_RE_EN = re.compile(
    r"\b("
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s*"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2},\s*\d{4}"
    r"|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2},\s*\d{4}"
    r")\b",
    flags=re.IGNORECASE,
)

# Portuguese textual dates:
# 19 de janeiro de 2026
_DATE_RE_PT = re.compile(
    r"\b(\d{1,2}\s+de\s+[a-zç]+(?:\s+de\s+\d{4})?)\b",
    flags=re.IGNORECASE,
)

@dataclass
class Grade:
    relevance_ratio: float
    distinct_domains: int
    authority_score: float
    usable_items: int
    stop_reason: str


def _tokenize_entity_hint(entity_hint: str) -> List[str]:
    e = (entity_hint or "").strip().lower()
    if not e:
        return []
    return [t for t in re.split(r"[\s\-_]+", e) if t]


def compute_relevance_ratio(items: List[EvidenceItem], entity_hint: str) -> float:
    tokens = _tokenize_entity_hint(entity_hint)
    if not items:
        return 0.0
    if not tokens:
        # If we don't know the entity, we can't compute relevance safely.
        return 0.5

    hit = 0
    for it in items:
        hay = f"{it.title}\n{it.url}\n{it.text}".lower()
        if all(tok in hay for tok in tokens):
            hit += 1
    return hit / max(1, len(items))


def authority_score(items: List[EvidenceItem], question_type: str) -> float:
    """
    Deterministic authority scoring by domains and context.
    Keep it conservative; this is not “truth”, only a prioritization signal.
    """
    score = 0.0
    for it in items:
        d = it.domain
        if not d:
            continue
        if question_type == "ownership":
            if d.endswith("sec.gov"):
                score += 3.0
        if question_type in ("acquisition", "news"):
            if "investor" in d or d.endswith(".com"):
                score += 0.3
        if question_type == "latest_version":
            if d in ("pypi.org", "github.com"):
                score += 1.0
            if "docs" in d:
                score += 0.5
    return score


def grade_evidence(
    *,
    question_type: str,
    entity_hint: str,
    items: List[EvidenceItem],
    min_relevance_ratio: float,
    min_distinct_domains: int,
    min_sources: int,
) -> Grade:
    rr = compute_relevance_ratio(items, entity_hint)
    domains = {it.domain for it in items if it.domain}
    auth = authority_score(items, question_type)
    usable = len([it for it in items if it.url])

    stop_reason = "insufficient"
    if usable < min_sources:
        stop_reason = "insufficient_sources"
    elif rr < min_relevance_ratio:
        stop_reason = "low_relevance"
    elif len(domains) < min_distinct_domains:
        stop_reason = "low_domain_diversity"
    else:
        stop_reason = "sufficient_candidate_set"

    return Grade(
        relevance_ratio=rr,
        distinct_domains=len(domains),
        authority_score=auth,
        usable_items=usable,
        stop_reason=stop_reason,
    )


_VERSION_RE = re.compile(r"\b(v?\d+\.\d+(?:\.\d+)?(?:[a-z0-9\.\-_]+)?)\b", flags=re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}\s+de\s+[a-zç]+(?:\s+de\s+\d{4})?)\b", flags=re.IGNORECASE)


def extract_version_candidates(items: List[EvidenceItem]) -> List[Tuple[str, str]]:
    """
    Returns list of (candidate, url) from titles/snippets.
    This is best-effort until web_fetch exists.
    """
    out: List[Tuple[str, str]] = []
    for it in items:
        text = f"{it.title}\n{it.text}"
        m = _VERSION_RE.search(text)
        if m:
            out.append((m.group(1), it.url))
    return out


def extract_date_candidates(items: List[EvidenceItem]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for it in items:
        text = f"{it.title}\n{it.text}"
        # Try multiple patterns
        m = _DATE_RE_ISO.search(text)
        if m:
            out.append((m.group(1), it.url))
            continue
        m = _DATE_RE_EN.search(text)
        if m:
            out.append((m.group(1), it.url))
            continue
        m = _DATE_RE_PT.search(text)
        if m:
            out.append((m.group(1), it.url))
            continue
        m = _DATE_RE_SLASH.search(text)
        if m:
            out.append((m.group(1), it.url))
            continue
    return out



def consensus(candidates: List[Tuple[str, str]], min_consensus: int) -> Optional[Tuple[str, List[str]]]:
    """
    Simple majority/threshold consensus over extracted candidates.
    Distinct URLs count as independent votes.
    """
    if not candidates:
        return None

    votes = {}
    urls_by = {}
    for val, url in candidates:
        v = (val or "").strip()
        if not v:
            continue
        votes[v] = votes.get(v, 0) + 1
        urls_by.setdefault(v, [])
        if url and url not in urls_by[v]:
            urls_by[v].append(url)

    best_val = None
    best_votes = 0
    for v, n in votes.items():
        if n > best_votes:
            best_val = v
            best_votes = n

    if not best_val:
        return None
    if best_votes < min_consensus:
        return None

    return best_val, urls_by.get(best_val, [])
