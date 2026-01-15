from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer


@dataclass
class TokenCounter:
    tokenizer_path: Optional[str] = None

    def __post_init__(self):
        if self.tokenizer_path:
            self.tok = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                local_files_only=True,
                use_fast=True,
                trust_remote_code=True,
            )
        else:
            self.tok = None

    def count(self, text: str) -> int:
        if not self.tok:
            return max(1, len(text) // 4)
        return len(self.tok.encode(text))

    def chunk_text(self, text: str, max_tokens: int) -> List[str]:
        if max_tokens <= 0:
            return [text]
        if not self.tok:
            size = max(200, max_tokens * 4)
            return [text[i : i + size] for i in range(0, len(text), size)]

        ids = self.tok.encode(text)
        chunks = []
        for i in range(0, len(ids), max_tokens):
            chunk_ids = ids[i : i + max_tokens]
            chunks.append(self.tok.decode(chunk_ids))
        return chunks
