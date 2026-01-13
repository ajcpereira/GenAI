from core.execution.pipeline import ExecutionPipeline

def test_pipeline():
    p = ExecutionPipeline()
    out = p.run({"prompt": "hello", "params": {}})
    assert "response" in out
