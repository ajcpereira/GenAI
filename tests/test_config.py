from utils.config_loader import load_config

def test_config():
    cfg = load_config("config/default.yaml", "config/schema.yaml")
    assert "llm" in cfg
