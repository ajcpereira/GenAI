from fastapi import FastAPI
from pydantic import BaseModel

from .orchestrator.agent import OrchestratorAgent
from .tools.mcp_client import MCPClient
from .runtime.state import RuntimeState


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


def create_app(cfg: dict, runtime: RuntimeState) -> FastAPI:
    app = FastAPI(title="GenAI Core Phase 1")

    mcp = MCPClient.from_config(cfg.get("tools", {}).get("mcp", {}))
    orchestrator = OrchestratorAgent(cfg=cfg, runtime=runtime, mcp=mcp)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "vllm": runtime.vllm_health,
            "model": runtime.model_info.model_name if runtime.model_info else None,
            "limits": runtime.model_info.model_limits if runtime.model_info else None,
        }

    @app.post("/chat")
    async def chat(req: ChatRequest):
        resp = await orchestrator.chat(
            user_id=req.user_id,
            session_id=req.session_id,
            message=req.message,
        )
        return resp

    return app
