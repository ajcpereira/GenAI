from llm.vllm_adapter import VLLMAdapter
from utils.config_loader import load_config

def test_llm():
    cfg = load_config("config/default.yaml", "config/schema.yaml")
    llm = VLLMAdapter(cfg)
    assert llm.count_tokens("a b c") == 3
