from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QuestionSpec:
    raw: str
    question_type: str  # date|latest_version|ownership|acquisition|news|generic_fact|explanation
    volatile: bool
    ambiguous: bool
    clarification: Optional[str]
    entity_hint: str


def _norm(s: str) -> str:
    return (s or "").strip()


def _lower(s: str) -> str:
    return _norm(s).lower()


def classify_question(message: str) -> QuestionSpec:
    """
    Lightweight deterministic classifier.
    (You can later add an optional LLM-assisted classifier, but keep this as the default guardrail.)
    """
    raw = _norm(message)
    s = _lower(raw)

    # Date/time queries ("hoje", "agora", "que dia é")
    if any(p in s for p in ["que dia é hoje", "que dia e hoje", "hoje é que dia", "qual é a data", "qual e a data", "que horas são", "que horas sao", "agora"]):
        return QuestionSpec(
            raw=raw,
            question_type="date",
            volatile=True,
            ambiguous=False,
            clarification=None,
            entity_hint="",
        )

    # Ownership / shareholders (often ambiguous)
    if any(p in s for p in ["principal accionista", "principal acionista", "maior acionista", "largest shareholder", "major shareholder", "shareholder"]):
        # Require clarification (2-A)
        clarification = (
            "Quando dizes “principal acionista”, queres dizer:\n"
            "1) maior acionista institucional,\n"
            "2) maior acionista individual/insider, ou\n"
            "3) maior detentor total (institucional + insiders)?\n"
            "Diz o número (1/2/3) para eu confirmar com fontes."
        )
        return QuestionSpec(
            raw=raw,
            question_type="ownership",
            volatile=True,
            ambiguous=True,
            clarification=clarification,
            entity_hint=_extract_entity_hint(raw),
        )

    # M&A / acquisitions / “última aquisição”
    if any(p in s for p in ["última aquisição", "ultima aquisicao", "acquisition", "acquired", "comprou", "compra de", "aquisição da", "aquisicao da"]):
        return QuestionSpec(
            raw=raw,
            question_type="acquisition",
            volatile=True,
            ambiguous=False,
            clarification=None,
            entity_hint=_extract_entity_hint(raw),
        )

    # Latest version / releases / “mais recente”
    if any(p in s for p in ["versão mais recente", "versao mais recente", "última versão", "ultima versao", "latest version", "latest release", "última release", "ultima release", "release mais recente"]):
        return QuestionSpec(
            raw=raw,
            question_type="latest_version",
            volatile=True,
            ambiguous=False,
            clarification=None,
            entity_hint=_extract_entity_hint(raw),
        )

    # News-like volatility
    if any(p in s for p in ["últimas notícias", "ultimas noticias", "notícias", "noticias", "breaking", "today", "esta semana", "este mês", "este mes"]):
        return QuestionSpec(
            raw=raw,
            question_type="news",
            volatile=True,
            ambiguous=False,
            clarification=None,
            entity_hint=_extract_entity_hint(raw),
        )

    # Default: explanation vs generic fact
    if any(p in s for p in ["explica", "como funciona", "arquitetura", "design", "porque", "porquê", "comparar", "melhor abordagem"]):
        return QuestionSpec(
            raw=raw,
            question_type="explanation",
            volatile=False,
            ambiguous=False,
            clarification=None,
            entity_hint=_extract_entity_hint(raw),
        )

    return QuestionSpec(
        raw=raw,
        question_type="generic_fact",
        volatile=True,  # conservative: treat unknown facts as volatile, so web grounding is allowed
        ambiguous=False,
        clarification=None,
        entity_hint=_extract_entity_hint(raw),
    )


def _extract_entity_hint(text: str) -> str:
    """
    Cheap heuristic: pick a salient capitalized token/phrase or known keywords.
    This is NOT NER; it’s a hint used for relevance gating and query refinement.
    """
    t = _norm(text)
    low = t.lower()

    # Common entities you used in examples
    for k in ["microsoft", "nvidia", "ubuntu", "vllm", "openai", "google", "amazon", "meta"]:
        if k in low:
            return k

    # Capture a capitalized word sequence (best-effort)
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", t)
    if m:
        return m.group(1).strip().lower()

    return ""
