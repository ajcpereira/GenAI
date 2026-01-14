from fastapi import FastAPI
from api.routes import chat, health

def create_app() -> FastAPI:
    app = FastAPI(title="GenAI Core", version="0.2.0")
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    return app
