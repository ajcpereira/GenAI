import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM

class HuggingFaceAdapter(BaseLLM):
    def __init__(self, config):
        self.cfg = config["llm"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"]["path"])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model"]["path"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate(self, context):
        prompt = "\n".join(b["content"] for b in context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=self.cfg["runtime"]["max_new_tokens"])
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"response": text}

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))
