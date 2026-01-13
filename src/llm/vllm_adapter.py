from .base import BaseLLM

class VLLMAdapter(BaseLLM):
    def __init__(self, config):
        self.config = config

    def generate(self, context):
        return {"response": "mock response"}

    def count_tokens(self, text):
        return len(text.split())
