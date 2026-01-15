from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelInfo:
    model_name: str
    model_limits: Dict[str, int]  # keys: max_context_tokens, max_new_tokens_default
    tokenizer_name_or_path: str


@dataclass
class RuntimeState:
    # Updated by VLLMLauncher
    vllm_health: Dict[str, Any] = field(default_factory=dict)
    model_info: Optional[ModelInfo] = None
