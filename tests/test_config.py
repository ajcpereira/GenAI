from utils.config_loader import load_config

def test_config_load():
    cfg = load_config("config/default.yaml", "config/schema.yaml")
    assert cfg["llm"]["provider"] in ("huggingface", "vllm")
