# GenAI Core – Phase 1 (Reasoning Orchestrator + HF/VLLM)

Enterprise-grade, on-prem, API-first GenAI core with:
- Strict separation of concerns
- Config-driven behavior (YAML + schema validation)
- Orchestrator Agent as the only decision authority
- Real policy gate using Phi-3 Mini (Hugging Face) to enforce allowed context (e.g., block politics)
- Execution LLM via Hugging Face (in-process) or vLLM (external server; OpenAI-compatible)

## API contract (Phase 1)
POST `/chat/`

Request:
```json
{
  "prompt": "string",
  "params": {},
  "context": {
    "allowed_topics": ["string"],
    "denied_topics": ["string"]
  }
}
```

Response:
```json
{ "response": "string" }
```

`context` is optional; if omitted, defaults come from `config/default.yaml`.

## Model placement (server instructions)

### Recommended on-disk layout
```
/models/
  mistral-7b-instruct/
    config.json
    model*.safetensors
    tokenizer.json
    ...
  phi-3-mini/
    config.json
    model*.safetensors
    tokenizer.json
    ...
```

### Download (Hugging Face CLI)
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download <ORG>/<MISTRAL_REPO> --local-dir /models/mistral-7b-instruct --local-dir-use-symlinks False
huggingface-cli download <ORG>/<PHI3_REPO>    --local-dir /models/phi-3-mini          --local-dir-use-symlinks False
```

## GPU (Nvidia L4) vs CPU
The code auto-selects GPU if CUDA is available; otherwise CPU is used.

Recommended (CUDA 12.1 example):
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install -U transformers accelerate safetensors
```

## vLLM serving (prepared to switch model)
Start vLLM (OpenAI-compatible):
```bash
pip install -U vllm
vllm serve /models/mistral-7b-instruct \
  --host 0.0.0.0 --port 8001 \
  --max-model-len 8192
```

Switch config to vLLM:
```yaml
llm:
  provider: vllm
  runtime:
    endpoint: http://localhost:8001
  model:
    name: mistral-7b-instruct
```

## Run
```bash
pip install -e .
python src/main.py
```

## Tests
Tests default to no-model mode (CI-friendly):
```bash
pytest
```

To run with real Phi-3 Mini reasoning:
- Set `ORCH_DISABLE_MODEL=0`
- Ensure `orchestrator.model.path` exists
