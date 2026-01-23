# GenAIv2 — On‑Prem Orchestrated GenAI (FastAPI + vLLM + MCP)

This repository implements an on‑prem orchestration service that:
- Accepts chat-style user requests (`POST /api/chat`)
- Discovers tools dynamically from an MCP Host (`GET /mcp/tools`)
- Plans tool usage via a **Planner LLM**
- Validates the plan with deterministic rules (schema + caps + allowlist)
- Executes tool calls as a DAG (topological order)
- Produces a final answer via a **Responder LLM**

The design goal is **contract-first orchestration**:
- Contracts live in `config/internal-json.json`
- Each stage validates inputs/outputs against the contracts
- The system favors **reliability** (fail fast, replan, bounded execution)

## Components

- **gateway-service** (optional; not in this repo): reverse proxy / API entry
- **orchestrator-service** (this repo): planning + validation + execution + response assembly
- **llm-service**: vLLM exposing an OpenAI-compatible API (planner + responder)
- **mcp-host**: exposes tool discovery and tool execution endpoints
- **session store** (optional): Postgres/Redis-backed conversation state (future extension)

## API

### `POST /api/chat`

Request body (UserRequestPayload):
- `message` (string)
- `enabled_tools` (string[]) — optional per-request allowlist at the UI/API boundary

Response:
- Envelope with `payload.answer` (string) on success
- Envelope with `payload.error` (structured) on failure

## Decision logic (high-level)

1. **Tool discovery**
   - Orchestrator calls MCP Host: `GET /mcp/tools`
   - Filters by `enabled_tools` (request) and internal allow/deny policies
   - Passes the resulting `allowed_tools` (name + description + schemas) to the Planner

2. **Planning**
   - Planner emits `PlannerOutput` (strict JSON)
   - If it proposes tool calls, each `tool_call` step MUST include a non-empty `capability` matching an allowed tool name

3. **Validation**
   - Validator enforces:
     - schema correctness
     - hard caps (`max_steps`, `max_tool_calls`)
     - dependencies (no cycles, no missing steps)
     - tool allowlist/policy constraints

4. **Replan loop**
   - If validation fails, Orchestrator provides compact validation feedback to Planner and retries planning (`orchestrator.max_replans`)
   - If replans are exhausted, returns `INVALID_PLAN`

5. **Execution**
   - Executor runs steps (DAG order)
   - For each `tool_call`, it calls MCP Host `/mcp/tools/{name}:run`
   - Execution results are recorded as `steps_executed[]`

6. **Final answer**
   - Responder receives a `FinalLLMInput` containing intent + execution results
   - Responder returns a JSON object `{"answer":"..."}` when `responder.use_structured_outputs=true`
   - Orchestrator returns only the `answer` string to the user

## Flowchart

```mermaid
flowchart TD
    A[Client<br/>POST /api/chat] --> B[API Layer<br/>Validate UserRequestPayload<br/>Create Envelope(request)]
    B --> C[Orchestrator.handle_message<br/>request_id + metadata]

    C --> D[MCP Discovery<br/>HTTP GET MCP Host /mcp/tools]
    D --> E[Filter tools by enabled_tools + policy<br/>Build ToolPolicy allowlist]

    E --> F[PlannerInput<br/>user_message + allowed_tools + locale + optional replan_feedback]
    F --> G[Planner LLM (vLLM)<br/>/v1/chat/completions<br/>response_format=json_object]

    G --> H[PlannerOutput<br/>plan.steps[] + confidence]
    H --> I[PlanValidator.validate<br/>schema + caps + allowlist + deps]

    I -->|invalid and replans left| R[Build replan_feedback<br/>validation.errors + warnings]
    R --> F

    I -->|invalid and no replans left| Z[Envelope(error)<br/>INVALID_PLAN]

    I -->|valid| N[ExecutorInput<br/>plan + tool_policy + discovered_tools]
    N --> O[Executor.execute_plan<br/>Topological sort]
    O --> P{Step type}
    P -->|tool_call| S[HTTP POST MCP Host<br/>/mcp/tools/{capability}:run]
    P -->|compose| Q[Compose context step]
    S --> T[Record StepExecution(output/error)]
    Q --> V[Record StepExecution(context)]
    T --> O
    V --> O

    O --> W[ExecutorResult<br/>steps_executed[]]
    W --> X[OutputValidator.validate]

    X -->|invalid| Y[Envelope(error)<br/>INVALID_OUTPUT]
    X -->|ok| AA[FinalLLMInput<br/>intent + steps_executed]

    AA --> AB[Responder LLM (vLLM)<br/>/v1/chat/completions<br/>response_format=json_object]
    AB --> AC[Envelope(response)<br/>AnswerPayload.answer]

    Z --> AE[HTTP error response]
    Y --> AE
    AC --> AF[HTTP success response]
```

## Configuration

All runtime configuration is in `config/config.yaml`.

Key tuning knobs:
- `logging.*` — directory, stdout, per-component files, rotation policy
- `planner.*` — vLLM URL/model, JSON mode, timeouts, retries, stop tokens
- `orchestrator.*` — confidence threshold, replans, caps
- `validator.*` — caps on steps/tool calls
- `executor.*` — caps and payload size limits
- `responder.*` — vLLM URL/model, structured outputs to prevent reasoning leakage
- `mcp.*` — MCP Host base URL and timeouts

See the `commentary:` section inside `config/config.yaml` for a complete list of tunables.

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ensure vLLM is running (OpenAI-compatible) and MCP Host is available
python main.py
```

## Notes on logging and disk safety

- File logs use rotation configured in `logging.rotation`.
- Container stdout logs (Docker/Podman) must also be rotated via the runtime’s log driver; file rotation does not control stdout volume.
