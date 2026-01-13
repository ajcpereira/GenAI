from fastapi import APIRouter
from core.execution.pipeline import ExecutionPipeline

router = APIRouter()

@router.post("/")
def chat(req: dict):
    pipeline = ExecutionPipeline()
    return pipeline.run(req)
