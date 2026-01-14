import os
os.environ["ORCH_DISABLE_MODEL"] = "1"

from utils.config_loader import load_config
from core.orchestrator.orchestrator import Orchestrator
from observability.logger import new_correlation_id

def test_orchestrator_blocks_politics():
    cfg = load_config("config/default.yaml", "config/schema.yaml")
    orch = Orchestrator(cfg)
    decision = orch.plan("Quem vai ganhar as próximas eleições?", correlation_id=new_correlation_id())
    assert decision.within_context is False
    assert decision.execution_plan.use_llm is False
