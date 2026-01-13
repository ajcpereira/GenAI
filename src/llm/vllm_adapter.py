from .base import BaseLLM

class VLLMAdapter(BaseLLM):
    def __init__(self, config):
        self.config = config

    def load(self): pass
    def generate(self, context):
        return {"response": "mock vLLM response"}
    def count_tokens(self, text):
        return len(text.split())
