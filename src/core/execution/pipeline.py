from typing import Any, Dict
from utils.config_loader import load_config
from observability.logger import log, new_correlation_id
from security.auth_stub import authorize
from core.orchestrator.orchestrator import Orchestrator
from llm.huggingface_adapter import HuggingFaceAdapter
from llm.vllm_adapter import VLLMAdapter

class ExecutionPipeline:
    def __init__(self):
        self.config = load_config("config/default.yaml", "config/schema.yaml")
        self.orchestrator = Orchestrator(self.config)
        provider = self.config["llm"]["provider"]
        self.llm = HuggingFaceAdapter(self.config) if provider == "huggingface" else VLLMAdapter(self.config)
        self.llm.load()

    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        correlation_id = new_correlation_id()
        log("request.received", correlation_id=correlation_id)

        if not authorize(request):
            log("request.denied", correlation_id=correlation_id)
            return {"response": "Unauthorized"}

        prompt = request.get("prompt", "")
        context_override = request.get("context")

        decision = self.orchestrator.plan(prompt, correlation_id=correlation_id, context_override=context_override)

        if not decision.within_context or not decision.execution_plan.use_llm:
            msg = decision.out_of_scope_reason or "Request is out of scope for this deployment."
            log("request.out_of_scope", correlation_id=correlation_id, message=msg)
            return {"response": msg}

        blocks = [{"type": "user", "content": prompt}]
        out = self.llm.generate(blocks)
        log("response.generated", correlation_id=correlation_id)
        return out
