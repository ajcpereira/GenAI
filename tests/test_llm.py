from llm.huggingface_adapter import HuggingFaceAdapter
from utils.config_loader import load_config

def test_llm_tokens():
    cfg = load_config("config/default.yaml", "config/schema.yaml")
    llm = HuggingFaceAdapter(cfg)
    llm.load()
    assert llm.count_tokens("a b c") >= 3
