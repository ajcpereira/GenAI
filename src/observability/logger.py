import json, logging

logging.basicConfig(level=logging.INFO)

def log(event, **data):
    logging.info(json.dumps({"event": event, **data}))
