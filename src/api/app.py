from fastapi import FastAPI
from api.routes import chat, health

def create_app():
    app = FastAPI()
    app.include_router(chat.router, prefix="/chat")
    app.include_router(health.router, prefix="/health")
    return app
