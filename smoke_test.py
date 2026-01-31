from pathlib import Path
import asyncio
from typing import Any, Dict, Optional, List

from utils.common import load_contract_bundle
from orchestrator.orchestrator import Orchestrator, SAFE_FALLBACK_PT


class StubPlanner:
    def __init__(self, plan_payload: Dict[str, Any]):
        self._plan_payload = plan_payload

    async def build_plan(self, planner_input: Dict[str, Any]) -> Dict[str, Any]:
        return self._plan_payload


class StubMCP:
    def __init__(self, tools: List[Dict[str, Any]]):
        self._tools = tools

    async def list_tools(self) -> List[Dict[str, Any]]:
        return list(self._tools)


class StubExecutor:
    def __init__(self, tools: List[Dict[str, Any]], exec_payload: Dict[str, Any]):
        self.mcp = StubMCP(tools)
        self._exec_payload = exec_payload

    async def execute(self, executor_input: Dict[str, Any]) -> Dict[str, Any]:
        return self._exec_payload


class StubResponder:
    def __init__(self, answer: str, confidence: float = 1.0):
        self._answer = answer
        self._confidence = confidence

    async def answer(self, final_llm_input: Dict[str, Any]) -> str:
        return self._answer

    async def estimate_confidence(self, final_llm_input: Dict[str, Any], answer_text: str) -> float:
        return float(self._confidence)


def make_request_envelope(message: str, enabled_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "metadata": {
            "schema_version": "1.2",
            "message_type": "user_request",
            "request_id": "08c1ffed-1708-4f1e-8e1a-f2a0b4bcd2f3",
            "timestamp": "2026-01-30T00:00:00+00:00",
            "source": "api",
            "user_id": None,
            "session_id": "3c6f34d2-2701-46dc-b6dc-5b0f25223f89",
            "trace": {"trace_id": None, "span_id": None},
            "timings_ms": {},
        },
        "payload": {
            "message": message,
            "enabled_tools": enabled_tools or [],
        },
    }


async def main() -> None:
    bundle = load_contract_bundle(str((Path(__file__).resolve().parent / 'config' / 'internal-json.json')))

    # Scenario A: time question but no tools enabled; planner insists on tool_call -> should return SAFE_FALLBACK_PT
    planner_payload_time = {
        "user_intent": {"summary": "ask time", "type": "informational", "confidence": 0.9, "locale": "pt"},
        "plan": {
            "strategy": "sequential",
            "steps": [
                {"id": "1", "type": "tool_call", "capability": "time.now", "description": "now", "inputs": {"timezone": None}, "dependencies": []}
            ],
        },
        "violations": [],
    }

    # Executor payload won't be reached for invalid plan, but must validate schema if accidentally used.
    exec_payload_dummy = {"intent": "dummy", "steps_executed": []}

    orch = Orchestrator(
        planner=StubPlanner(planner_payload_time),
        executor=StubExecutor(tools=[{"name": "time.now", "description": "", "input_schema": {"type": "object"}}], exec_payload=exec_payload_dummy),
        responder=StubResponder(answer='SHOULD_NOT_APPEAR', confidence=1.0),
        contract_bundle=bundle,
        confidence_threshold=0.8,
        max_replans=0,
        reject_unknown_enabled_tools=True,
        session_store=None,
        context_manager=None,
    )

    env = make_request_envelope('Que horas são?', enabled_tools=[])
    out = await orch.handle_envelope(env)
    ans = out['payload']['answer']
    print('Scenario A answer:', ans)
    assert ans == SAFE_FALLBACK_PT

    # Scenario B: time question with tool enabled; plan valid -> responder answer returned (confidence high)
    exec_payload_time = {
        "intent": "ask time",
        "steps_executed": [
            {"id": "1", "status": "success", "output": {"data": {"iso": "2026-01-30T16:00:00+00:00", "timezone": "UTC"}}, "error": None}
        ],
    }
    orch2 = Orchestrator(
        planner=StubPlanner(planner_payload_time),
        executor=StubExecutor(tools=[{"name": "time.now", "description": "", "input_schema": {"type": "object"}}], exec_payload=exec_payload_time),
        responder=StubResponder(answer='São 16:00 UTC', confidence=1.0),
        contract_bundle=bundle,
        confidence_threshold=0.8,
        max_replans=0,
        reject_unknown_enabled_tools=False,
        session_store=None,
        context_manager=None,
    )
    env2 = make_request_envelope('Que horas são?', enabled_tools=['time.now'])
    out2 = await orch2.handle_envelope(env2)
    ans2 = out2['payload']['answer']
    print('Scenario B answer:', ans2)
    assert ans2 == 'São 16:00 UTC'


if __name__ == '__main__':
    asyncio.run(main())
