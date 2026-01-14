import yaml
import jsonschema

def load_config(path: str, schema_path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    jsonschema.validate(cfg, schema)
    return cfg
