from typing import Any, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.base import BaseLLM

class HuggingFaceAdapter(BaseLLM):
    def __init__(self, config: dict):
        self.cfg = config["llm"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        path = self.cfg["model"]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(self, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = "\n".join(b.get("content", "") for b in context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(self.cfg["runtime"]["max_new_tokens"]),
            temperature=float(self.cfg["runtime"].get("temperature", 0.2)),
            do_sample=False,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"response": text}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if self.tokenizer else len(text.split())

    def healthcheck(self) -> Dict[str, Any]:
        return {"status": "ok", "provider": "huggingface", "device": self.device}
