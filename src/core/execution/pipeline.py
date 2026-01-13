from utils.config_loader import load_config
from core.orchestrator.orchestrator import Orchestrator
from llm.vllm_adapter import VLLMAdapter

class ExecutionPipeline:
    def __init__(self):
        self.config = load_config("config/default.yaml", "config/schema.yaml")
        self.orchestrator = Orchestrator(self.config)
        self.llm = VLLMAdapter(self.config)

    def run(self, request):
        plan = self.orchestrator.plan(request["prompt"])
        if plan["use_llm"]:
            return self.llm.generate([{"type": "user", "content": request["prompt"]}])
        return {"response": ""}
