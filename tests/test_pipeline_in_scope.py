import os
os.environ["ORCH_DISABLE_MODEL"] = "1"

from core.execution.pipeline import ExecutionPipeline

def test_pipeline_in_scope_returns_response():
    p = ExecutionPipeline()
    out = p.run({"prompt": "Explica como integrar vLLM neste projeto.", "params": {}})
    assert "response" in out
