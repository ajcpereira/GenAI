# GenAIv3 — On‑Prem Contract‑First GenAI Orchestrator (FastAPI + vLLM + MCP)

This repository implements an on‑prem orchestration service that:

- Accepts chat-style user requests via an HTTP API
- Discovers and executes tools from an **MCP Host**
- Plans tool usage via a **Planner LLM** (OpenAI‑compatible `/v1/chat/completions`)
- Validates plans and envelopes strictly against **schema contracts**
- Persists **all envelopes** to Postgres for auditability and debugging

The design goal is **deterministic, contract‑first orchestration**:

- Contracts live in `config/internal-json.json` (source of truth)
- Each stage validates inputs/outputs against the contracts
- No orchestrator heuristics: the Planner decides when to call tools
- Failures are returned as controlled **error envelopes** (never an internal crash)

---

## High‑level Architecture

### Services

- **orchestrator-service** (this repo): API + planning + validation + execution + response assembly
- **llm-service**: vLLM exposing an OpenAI-compatible API (planner + responder models)
- **mcp-host**: tool discovery (`GET /mcp/tools`) + tool execution (`POST /mcp/tools/{name}:run`)
- **Postgres** (required): persistence of envelopes + session/message history

### Core pipeline

1. **API** receives a user request and wraps it into a `user_request` envelope.
2. **Context policy classifier** decides whether the current message is `standalone` or needs `recent` context.
3. **Planner** produces a structured plan (`compose` steps + optional `tool_call` steps).
4. **Validator** enforces schema + tool allowlist + caps + dependency rules.
5. **Executor** runs steps and persists per-step results. For tool calls, it persists:
   - tool request (capability + inputs)
   - raw HTTP request payload
   - raw HTTP response (status + body)
6. **Responder** produces the final user answer using intent + execution results.

---

## HTTP API

The API router is mounted under `/api`.

### `POST /api/chat`

Request body:

- `message` (string, required)
- `enabled_tools` (string[], optional) — per-request tool allowlist

Response body:

```json
{
  "request": {"metadata": {...}, "payload": {...}},
  "response": {"metadata": {...}, "payload": {...}}
}
```

The `response` envelope contains either:

- `payload.answer` on success, or
- `payload.error` on controlled failure (e.g., schema validation error, invalid plan, tool failure)

### Other useful endpoints

- `GET /api/health` — health check
- `GET /api/tools` — tools known by the orchestrator’s initialized MCP client
- `GET /api/sessions` / `GET /api/sessions/{id}/messages` — session browsing for the UI
- `GET /api/requests/{request_id}/envelopes` — request trace inspection

---

## Configuration

Runtime configuration is in `config/config.yaml` (with environment overrides).

Important:

- Contracts bundle path is configured under `contracts.bundle_path`.
- Postgres DSN is required: `storage.postgres.dsn`.
- vLLM base URLs/models are configured under `planner.*` and `responder.*`.
- MCP base URL + caller id are configured under `mcp.*`.

---

## Flowchart

This diagram is GitHub‑Mermaid compatible (no HTML tags in labels).

```mermaid
flowchart TD
  A["Client\nPOST /api/chat"] --> B["API Layer\nValidate UserRequestPayload\nCreate user_request envelope"]
  B --> C["Orchestrator.handle_envelope\nrequest_id + metadata"]

  C --> CP["Context Policy Classifier\nstandalone vs recent"]
  CP --> D["MCP Discovery\nGET /mcp/tools"]
  D --> E["Tool policy\nFilter by enabled_tools + internal rules"]

  E --> F["PlannerInput\nmessage + allowed_tools\n(optional) recent context\n(optional) replan_feedback"]
  F --> G["Planner LLM (vLLM)\n/v1/chat/completions\nresponse_format=json_object"]

  G --> H["PlannerOutput\nplan.steps[] + confidence"]
  H --> I["PlanValidator\nschema + caps + allowlist + deps"]

  I -->|"invalid + replans left"| R["Build replan_feedback\nvalidation errors"]
  R --> F

  I -->|"invalid + no replans"| Z["Error envelope\nINVALID_PLAN"]

  I -->|"valid"| N["ExecutorInput\nplan + tool policy"]
  N --> O["Executor\nrun steps (DAG order)"]

  O --> P{"Step type"}
  P -->|"tool_call"| S["MCP Host\nPOST /mcp/tools/{capability}:run"]
  P -->|"compose"| Q["Compose step\ncontext assembly"]

  S --> T["Persist tool I/O\nrequest + raw response"]
  Q --> V["Persist compose output"]
  T --> O
  V --> O

  O --> W["ExecutorResult\nsteps_executed[]"]
  W --> AA["FinalLLMInput\nintent + steps_executed"]

  AA --> AB["Responder LLM (vLLM)\n/v1/chat/completions\nresponse_format=json_object"]
  AB --> AC["Response envelope\nanswer"]

  Z --> OUT["HTTP 200\nrequest+response envelopes"]
  AC --> OUT
```

---

## Developer tooling

### Trace viewer

`scripts/trace_viewer.py` reads envelopes from Postgres and prints a per-request trace.

```bash
export DATABASE_URL=postgresql://genai:genai123@127.0.0.1:5432/genai_db
python scripts/trace_viewer.py <session_id>
```

---

## Notes

- This project is intentionally **schema-driven**. If you add a tool or change payload shapes, update the schema bundle first.
- The orchestrator must never rely on keyword heuristics to force tool usage; only the Planner chooses tools.
