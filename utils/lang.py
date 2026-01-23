import re
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Language:
    code: str  # e.g., "pt", "en"

_PT_HINTS = {
    "não","sim","que","para","com","uma","um","por","porque","isto","isso","aqui","agora",
    "também","mais","menos","onde","quando","como","qual","quais","obrigado","preciso",
    "equipa","projeto","configuração","ferramenta","resposta","pedido"
}

def detect_language(text: str, default: str = "en") -> Language:
    """
    Lightweight PT/EN detector (no external dependency).
    If unsure, returns default.
    """
    t = (text or "").strip().lower()
    if not t:
        return Language(default)

    # Strong signals: diacritics commonly used in Portuguese
    if re.search(r"[ãõçáéíóúâêôà]", t):
        return Language("pt")

    # Token-based heuristic
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", t)
    if not tokens:
        return Language(default)

    hits = sum(1 for tok in tokens[:200] if tok in _PT_HINTS)
    if hits >= 2:
        return Language("pt")

    return Language(default)
