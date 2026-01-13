from utils.config_loader import load_config
from core.orchestrator.orchestrator import Orchestrator
from llm.vllm_adapter import VLLMAdapter
from llm.huggingface_adapter import HuggingFaceAdapter

class ExecutionPipeline:
    def __init__(self):
        self.config = load_config("config/default.yaml", "config/schema.yaml")
        self.orchestrator = Orchestrator(self.config)

        provider = self.config["llm"]["provider"]
        self.llm = HuggingFaceAdapter(self.config) if provider == "huggingface" else VLLMAdapter(self.config)
        self.llm.load()

    def run(self, request):
        plan = self.orchestrator.plan(request["prompt"])
        if plan["use_llm"]:
            context = [{"type": "user", "content": request["prompt"]}]
            return self.llm.generate(context)
        return {"response": ""}
