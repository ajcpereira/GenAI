# GenAI Core – Phase 1 (vLLM + Orchestrator + MCP-ready)

This repository is a Phase 1, **API-first** core that:
- runs **only** with **vLLM** (OpenAI-compatible server),
- starts vLLM from `main.py` (inside your Python virtual environment),
- validates that vLLM started correctly (health check),
- runs an **Orchestrator Agent** that uses the same vLLM model to decide when to call tools,
- supports an **MCP-based “internet search” tool** (pluggable),
- includes a **RAG placeholder** (not implemented yet),
- ensures that **any external information (internet/RAG)** is **always** included in the final answer,
- chunks/merges context if it would exceed the model context window.

## 1) Requirements / Assumptions

- You are running on-prem, inside a `venv`.
- `vllm` is installed in that environment.
- You have a **local** model path (Hugging Face format) accessible on disk.
- If you want internet search, you must run an MCP server that exposes a `web_search` tool (example described below).

## 2) Quickstart

### 2.1 Create and activate a venv (example)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Configure
Edit `config/config.yaml`:
- `vllm.model_path` must point to your local model directory
- `vllm.host` / `vllm.port` define where the OpenAI-compatible server will listen
- `orchestrator.vllm_base_url` can be changed later to point to a **remote** vLLM server

### 2.3 Run
```bash
python main.py
```
What this does:
1. launches the vLLM OpenAI server as a subprocess
2. checks `GET /health` until vLLM responds
3. loads model/tokenizer metadata locally (no internet required) and stores it in runtime
4. starts a minimal API (`/chat`) for demonstration (FastAPI)
5. routes prompts to the orchestrator which may call tools and then vLLM

## 3) Runtime Model Limits (max_context_tokens, max_new_tokens)

Phase 1 does **not** keep these in `config.yaml`.

Instead, `main.py` derives runtime limits from the model/tokenizer:
- `max_context_tokens` is derived from tokenizer/config (`model_max_length` or `max_position_embeddings`)
- `max_new_tokens_default` is derived conservatively from the context window
  (models rarely declare a reliable max-new-tokens; we compute a safe default)

These values are available throughout execution via `RuntimeState`.

## 4) Orchestrator Behavior

The orchestrator:
- uses the same vLLM model for **tool selection**
- supports tools:
  - `web_search` via MCP (implemented as a generic MCP HTTP client)
  - `rag_search` (placeholder)
- always includes any tool/RAG content in the **final answer** under **Sources**
- if context would exceed the model window, it:
  1) chunks external content
  2) summarizes each chunk via vLLM
  3) synthesizes a final answer

## 5) MCP Internet Search

### 5.1 What is expected
You provide an MCP server reachable via HTTP which can execute a tool like:

- `web_search(query: str, top_k: int) -> list[{"title": str, "url": str, "snippet": str}]`

In `config/config.yaml` set:
- `tools.mcp.base_url`
- `tools.mcp.tool_name_web_search` (default: `web_search`)

### 5.2 Wire format expectation (Phase 1 client)
This Phase 1 client expects:
- `POST {base_url}/call`
- body:
```json
{"tool":"web_search","args":{"query":"...","top_k":5}}
```
- response:
```json
{"results":[{"title":"...","url":"...","snippet":"..."}]}
```

If your MCP server uses a different format, adapt `src/genai_core/tools/mcp_client.py`.

## 6) Endpoints (Demo API)

- `GET /health` – health of the **core**
- `POST /chat` – sends a message `{ "user_id": "...", "session_id": "...", "message": "..." }`

## 7) Notes / Suggested Next Steps

- Add persistence for sessions (Redis/Postgres) + authn/authz
- Implement RAG module (vector store + ACL by user/group)
- Add structured “tool spec” discovery via MCP listing
- Harden JSON parsing and add evaluation for tool-selection quality

---
Copyright (c) 2026
