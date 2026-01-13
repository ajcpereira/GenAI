import yaml, jsonschema

def load_config(path, schema_path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    jsonschema.validate(cfg, schema)
    return cfg
