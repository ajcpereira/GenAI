from __future__ import annotations

from typing import Dict

from transformers import AutoConfig, AutoTokenizer


def derive_model_limits(model_path: str) -> Dict[str, int]:
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True, trust_remote_code=True)

    max_context = None
    if hasattr(tok, "model_max_length") and isinstance(tok.model_max_length, int):
        if tok.model_max_length and tok.model_max_length < 10**9:
            max_context = tok.model_max_length

    if max_context is None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
            if hasattr(cfg, attr):
                v = getattr(cfg, attr)
                if isinstance(v, int) and v > 0:
                    max_context = v
                    break

    if max_context is None:
        max_context = 4096

    max_new_default = min(1024, max(128, max_context // 8))

    return {
        "max_context_tokens": int(max_context),
        "max_new_tokens_default": int(max_new_default),
    }
