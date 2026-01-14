import os
os.environ["ORCH_DISABLE_MODEL"] = "1"

from core.execution.pipeline import ExecutionPipeline

def test_pipeline_out_of_scope():
    p = ExecutionPipeline()
    out = p.run({"prompt": "Fala-me do presidente e das eleições.", "params": {}})
    assert ("out of scope" in out["response"].lower()) or ("fora do" in out["response"].lower())
