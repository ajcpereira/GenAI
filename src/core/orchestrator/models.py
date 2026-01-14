from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class ExecutionPlan:
    use_llm: bool
    use_rag: bool
    use_mcp: bool
    tools: List[Dict[str, Any]]

@dataclass(frozen=True)
class ReasoningDecision:
    within_context: bool
    out_of_scope_reason: str
    execution_plan: ExecutionPlan
