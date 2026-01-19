from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from .prompts import system_router_pt
from .tool_catalog import ToolCatalog
from ..vllm.openai_client import VLLMOpenAIClient


Route = Literal["llm", "mcp_tool"]


@dataclass
class Decision:
    route: Route
    tool_key: Optional[str]
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    reason: str
    confidence: float = 0.75


def _extract_json(raw: str) -> str:
    s = (raw or "").strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else s


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except Exception:
        return 0.75
    return max(0.0, min(1.0, f))


def _reject_secretish_args(tool_args: Dict[str, Any]) -> None:
    bad = {"token", "password", "secret", "api_key", "apikey", "authorization", "cookie", "bearer"}
    for k in list(tool_args.keys()):
        if k.lower() in bad:
            tool_args.pop(k, None)


async def route_with_llm_router(
    *,
    llm: VLLMOpenAIClient,
    model: str,
    catalog: ToolCatalog,
    message: str,
    max_tokens: int,
    temperature: float,
) -> Decision:
    tools_text = catalog.render_for_router_prompt()
    user = (
        f"Mensagem do utilizador:\n{message}\n\n"
        f"Tools disponíveis:\n{tools_text}\n\n"
        "Escolhe 'mcp_tool' se alguma tool ajudar; caso contrário escolhe 'llm'.\n"
    )

    raw = await llm.chat_completion(
        model=model,
        system=system_router_pt(),
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
        extra={"stop": ["</s>"], "repetition_penalty": 1.10},
    )

    data = None
    try:
        data = json.loads(_extract_json(raw))
    except Exception:
        return Decision("llm", None, None, {}, "router_parse_failed", 0.55)

    route = data.get("route", "llm")
    if route not in ("llm", "mcp_tool"):
        route = "llm"

    if route == "llm":
        return Decision("llm", None, None, {}, str(data.get("reason") or "ok")[:160], _clamp01(data.get("confidence", 0.75)))

    tool_key = (data.get("tool_key") or "").strip()
    tool_args = data.get("tool_args") if isinstance(data.get("tool_args"), dict) else {}

    t = catalog.get_tool(tool_key)
    if not t:
        return Decision("llm", None, None, {}, "router_invalid_tool_fallback", 0.50)

    _reject_secretish_args(tool_args)

    return Decision(
        "mcp_tool",
        tool_key=t.key,
        tool_name=t.tool_name,
        tool_args=tool_args,
        reason=str(data.get("reason") or "router_ok")[:160],
        confidence=_clamp01(data.get("confidence", 0.75)),
    )
