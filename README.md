# GenAI Core – Phase 1 (vLLM + Orchestrator + MCP-ready)

This repository is a Phase 1, **API-first** core that:
- runs **only** with **vLLM** (OpenAI-compatible server),
- starts vLLM from `main.py` (inside your Python virtual environment),
- validates that vLLM started correctly (health check),
- derives model limits offline from local model/tokenizer files,
- runs an **Orchestrator Agent** that controls prompting and decoding per request,
- supports an **MCP web_search** tool (pluggable) and is RAG-ready (placeholder),
- always includes external info (internet/RAG) in the final answer as **Sources**,
- chunks + summarizes external context if it would exceed the model window,
- includes logging for both Core and vLLM subprocess.

## 1) Requirements / Assumptions
- On-prem deployment; you run inside a `venv`.
- `vllm` is installed in that environment.
- Local model path is available on disk (HF format).
- Optional MCP server (if `tools.mcp.enabled=true`).

## 2) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Configure
Edit `config/config.yaml`:
- `vllm.model_path`: path to the local model
- `vllm.served_model_name`: model id exposed by vLLM
- `orchestrator.model`: must match `served_model_name`
- `tools.mcp.*`: MCP endpoint and timeouts
- `logging.*`: log file paths and level

### Deterministic interpreter (optional)
To force which Python interpreter starts vLLM (recommended for deterministic ops):
```yaml
vllm:
  python_bin: "/path/to/venv/bin/python"
```

## 4) Run
```bash
python main.py
```

## 5) Test (curl)

### Health
```bash
curl -s http://127.0.0.1:8000/health | jq
curl -s http://127.0.0.1:8001/health | jq
```

### Chat via Core
```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1","message":"Como te chamas?"}' | jq
```

### Direct vLLM check
```bash
curl -s -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b-instruct","messages":[{"role":"user","content":"Diz apenas a palavra OK."}],"max_tokens":16,"temperature":0}' | jq
```

## 6) Orchestrator behavior (industry guardrails)

### On-the-fly decoding
The orchestrator adjusts generation parameters per request:
- `temperature`: default 0.0 (Phase 1 predictability)
- `max_tokens`: bounded by `reserved_output_tokens` and `max_tokens_cap`
- for short-format requests (e.g. “apenas a palavra OK”) it uses very small `max_tokens` and stop sequences

### Tool routing
To avoid “hallucinated tool use”, Phase 1 first applies a heuristic: only messages with freshness triggers
(e.g., “hoje”, “últimas”, “release”, “2026”) can call the LLM tool-decision step.

### Tool failures: fail-open (fixes 500)
If MCP is enabled but unreachable, chat does not crash. The response continues, and a tool error entry is included under `Sources`.

## 7) Logging & troubleshooting
- Core logs: `./logs/core.log`
- vLLM logs: `./logs/vllm.log` (subprocess stdout/stderr)

If vLLM fails to start, `VLLMLauncher` raises with the last log lines and the log file path.

---
Copyright (c) 2026
