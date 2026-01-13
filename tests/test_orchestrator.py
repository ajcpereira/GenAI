from core.orchestrator.orchestrator import Orchestrator
from utils.config_loader import load_config

def test_orchestrator_plan():
    cfg = load_config("config/default.yaml", "config/schema.yaml")
    orch = Orchestrator(cfg)
    plan = orch.plan("hi")
    assert plan["use_llm"] is True
