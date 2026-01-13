# GenAI Core – Phase 1

Enterprise-grade, on‑prem, API‑first GenAI core with strict orchestration, config-driven behavior, and extensible architecture.

## Key Guarantees
- Orchestrator Agent is the sole authority for decisions
- LLMs are pure runtimes
- YAML configuration is the single source of truth
- No implicit tool or RAG execution
- Contracts are stable and explicit

## Running
```bash
pip install -e .
python src/main.py
```

## Testing
```bash
pytest
```
