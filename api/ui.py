# ui/ui.py
from fastapi import APIRouter
from fastapi.responses import RedirectResponse


def make_ui_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse(url="/ui/index.html")

    return router
