from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined


@dataclass
class ChatMessage:
    role: str
    content: str


class PromptRenderer:
    """Renders chat messages using a configurable Jinja chat-template.

    Keeps Orchestrator provider/model agnostic.
    """

    def __init__(self, template_path: Optional[str]):
        self.template_path = template_path

    def render(self, messages: List[Dict[str, str]]) -> str:
        if not self.template_path:
            # Fallback: minimal concatenation.
            return "\n\n".join(m.get("content", "") for m in messages)

        p = Path(self.template_path)
        search_dir = str(p.parent) if p.parent else "."
        env = Environment(
            loader=FileSystemLoader(search_dir),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        tpl = env.get_template(p.name)
        return tpl.render(messages=messages)
