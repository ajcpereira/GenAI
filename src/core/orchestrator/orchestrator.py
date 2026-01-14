from typing import Any, Dict, Optional
from observability.logger import log
from core.orchestrator.reasoning_model import ReasoningModel
from core.orchestrator.models import ReasoningDecision

class Orchestrator:
    def __init__(self, config: dict):
        self.reasoner = ReasoningModel(config)

    def plan(self, prompt: str, correlation_id: str, context_override: Optional[Dict[str, Any]] = None) -> ReasoningDecision:
        decision = self.reasoner.decide(prompt, context_override=context_override)
        log("orchestrator.decision", correlation_id=correlation_id, within_context=decision.within_context,
            out_of_scope_reason=decision.out_of_scope_reason)
        log("execution_plan", correlation_id=correlation_id, plan={
            "use_llm": decision.execution_plan.use_llm,
            "use_rag": decision.execution_plan.use_rag,
            "use_mcp": decision.execution_plan.use_mcp,
            "tools": decision.execution_plan.tools,
        })
        return decision
