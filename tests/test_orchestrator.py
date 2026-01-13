from utils.config_loader import load_config
from core.orchestrator.orchestrator import Orchestrator

def test_orchestrator():
    cfg = load_config("config/default.yaml", "config/schema.yaml")
    orch = Orchestrator(cfg)
    plan = orch.plan("hello")
    assert plan["use_llm"]
