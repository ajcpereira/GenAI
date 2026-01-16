# GenAI Core (Phase 1)

An **enterprise-grade**, **on-premises**, **API-first** GenAI core designed to be **secure by default**, **observable**, **configuration-driven**, and **extensible by design**.

This repository provides:

- **Core API** (FastAPI)
- **Orchestrator Agent** (the only component allowed to make control decisions)
- **Model serving** via **vLLM** (OpenAI-compatible endpoints)
- **Tool integration** via **MCP** (Model Context Protocol), prepared for multiple tools/agents

---

## Mens legis (binding intent)

The objective of this project is to build an enterprise-grade GenAI platform that is on-premises, API-first, secure by default, observable, config-driven, and extensible by design.

### Non-negotiable principles

- Strict separation of responsibilities between **API**, **orchestration**, **execution**, **models**, and **tools**.
- LLMs are **replaceable runtimes**, never autonomous entities; **vLLM** is used for model serving.
- All control decisions (RAG, MCP/tools, context policy) are taken **exclusively** by the **Orchestrator Agent**.
- All operational configuration (paths, models, endpoints, providers, flags) lives in **schema-validated YAML** files.

---

## Runtime language policy

- The **codebase and documentation** are written in **English**.
- At runtime, the system will **answer in the same language as the user's prompt**.
  - Example: Portuguese prompt → Portuguese answer; English prompt → English answer.
- If web search is used, the orchestrator will include a **mandatory disclosure** in the same language.

---

## Architecture overview

1. **API** (`src/genai_core/api.py`)
   - Exposes `/chat`, `/health`, `/ready`
   - Enforces a strict response contract: always returns a non-empty `answer`
   - Assigns and propagates a `X-Correlation-ID` for end-to-end observability

2. **Orchestrator** (`src/genai_core/orchestrator/agent.py`)
   - The only control-plane component
   - Performs a **two-step pipeline**:
     1) **Route decision**: `llm` vs `mcp_web` (structured JSON, schema-validated)
     2) **Answer generation**: optionally enriched with MCP web results
   - Enforces **tool-truth**: the model cannot claim web usage unless MCP was actually invoked
   - Maintains lightweight per-session memory (configurable)

3. **Tools / MCP** (`src/genai_core/tools/mcp_client.py`)
   - A generic `call_tool()` interface (tool-agnostic orchestrator)
   - Enterprise safeguards: retries + exponential backoff + simple circuit breaker
   - Normalizes tool output for deterministic downstream processing

4. **Model execution via vLLM** (`src/genai_core/vllm/*`)
   - Uses OpenAI-compatible endpoints
   - This Phase 1 orchestrator uses `/v1/completions` with a configurable chat template for stability (Mistral `[INST]`).

5. **MCP Host (Python)** (`src/genai_core/mcp_host/*`)
   - A lightweight scaffold to host MCP tools/agents in Python
   - Designed for future expansion beyond internet search

---

## Configuration

Configuration is stored in `config/config.yaml` and validated against a Pydantic schema in `src/genai_core/config/schema.py`.

Notable settings:

- `logging.rotation`: size/time/none rotation
- `orchestrator.*`: router and answer generation controls
  - `router_*`: routing JSON step (tool decision)
  - `answer_temperature`, token caps
  - `web_budget_per_session`: cost/risk control
- `tools.mcp.*`: MCP endpoint + robustness
  - retries, backoff, circuit breaker thresholds

---

## Running

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start

```bash
python main.py --config config/config.yaml
```

### Test

```bash
curl -s -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1","message":"Say only the word OK."}' | jq -r '.answer'
```

---

## Security and safety notes

- The orchestrator is the only component permitted to call tools.
- Tool usage is disclosed in the final answer when invoked.
- The model is not allowed to claim external verification unless provided tool context.

---

## Roadmap (Phase 2+)

- First-class tool registry with per-tool authz and policy rules
- Dedicated time/date tool (no internet required)
- RAG integration behind the orchestrator control plane
- Structured logging (JSON) and OpenTelemetry export
