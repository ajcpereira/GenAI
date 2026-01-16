from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class Language:
    """Language descriptor.

    This module intentionally avoids heavy external dependencies.
    In Phase 1 we use a pragmatic heuristic-based detector to:
      - choose the language of non-LLM error messages (e.g., model not ready)
      - provide a stable "answer in the same language" hint to the LLM
    """

    code: str  # e.g. "en", "pt", "es", "fr"
    name: str  # e.g. "English", "Portuguese"


_LANGS: Dict[str, Language] = {
    "en": Language("en", "English"),
    "pt": Language("pt", "Portuguese"),
    "es": Language("es", "Spanish"),
    "fr": Language("fr", "French"),
}


_STOPWORDS: Dict[str, Iterable[str]] = {
    # Very small, high-signal sets.
    "pt": ("que", "hoje", "versão", "versao", "porque", "para", "nao", "não", "uma", "como", "onde"),
    "es": ("que", "hoy", "version", "porque", "para", "no", "una", "como", "donde"),
    "fr": ("que", "aujourd", "version", "pourquoi", "pour", "ne", "pas", "une", "comment", "où"),
    "en": ("the", "today", "version", "because", "for", "not", "a", "how", "where"),
}


_RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']{2,}")


def detect_language(text: str, default: str = "en") -> Language:
    """Best-effort language detection.

    This is not intended to be perfect. It is designed to be stable and
    safe under typical enterprise prompts. The LLM is still instructed to
    respond in the user's language.
    """

    t = (text or "").strip().lower()
    if not t:
        return _LANGS.get(default, _LANGS["en"])

    # Fast path: Portuguese-specific diacritics / contractions.
    if any(ch in t for ch in ("ã", "õ", "ç")):
        return _LANGS["pt"]

    words = _RE_WORD.findall(t)
    if not words:
        return _LANGS.get(default, _LANGS["en"])

    scores = {k: 0 for k in _LANGS.keys()}
    for lang, sw in _STOPWORDS.items():
        s = 0
        for w in words:
            if w in sw:
                s += 1
        scores[lang] = s

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return _LANGS.get(default, _LANGS["en"])
    return _LANGS[best]


def disclosure_text(lang: Language) -> str:
    """Disclosure string when web-search was performed."""
    if lang.code == "pt":
        return "Nota: pesquisei na internet (via MCP web_search) para enriquecer esta resposta."
    if lang.code == "es":
        return "Nota: busque en Internet (via MCP web_search) para enriquecer esta respuesta."
    if lang.code == "fr":
        return "Note : j'ai effectue une recherche sur Internet (via MCP web_search) pour enrichir cette reponse."
    return "Note: I searched the internet (via MCP web_search) to enrich this answer."


def not_ready_text(lang: Language) -> str:
    if lang.code == "pt":
        return "O modelo ainda nao esta pronto. Tenta novamente dentro de momentos."
    if lang.code == "es":
        return "El modelo aun no esta listo. Vuelve a intentarlo en unos instantes."
    if lang.code == "fr":
        return "Le modele n'est pas encore pret. Reessayez dans un instant."
    return "The model is not ready yet. Please retry shortly."


def strip_answer_prefixes(text: str) -> str:
    """Remove common 'Answer:' prefixes produced by instruct models."""
    t = (text or "").lstrip()
    for prefix in ("Answer:", "Resposta:", "Respuesta:", "Reponse:"):
        if t.lower().startswith(prefix.lower()):
            return t[len(prefix):].lstrip()
    return t
