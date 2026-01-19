from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class EvidenceItem:
    kind: str
    title: str = ""
    url: str = ""
    text: str = ""
    data: Optional[Dict[str, Any]] = None
    source_tool: str = ""
    score: float = 0.0

    @property
    def domain(self) -> str:
        try:
            return (urlparse(self.url).netloc or "").lower()
        except Exception:
            return ""


@dataclass
class EvidencePack:
    """
    Normalized evidence container.
    The Orchestrator treats this as the only acceptable input for factual claims.
    """
    kind: str  # "web" | "metrics" | "tool"
    query: str
    items: List[EvidenceItem] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def top_urls(self, n: int = 3) -> List[str]:
        out: List[str] = []
        seen = set()
        for it in sorted(self.items, key=lambda x: x.score, reverse=True):
            u = (it.url or "").strip()
            if not u or u in seen:
                continue
            out.append(u)
            seen.add(u)
            if len(out) >= n:
                break
        return out

    def domains(self) -> List[str]:
        seen = set()
        out = []
        for it in self.items:
            d = it.domain
            if d and d not in seen:
                out.append(d)
                seen.add(d)
        return out
