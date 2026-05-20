from fastapi.routing import APIRouter

from AiRecAgent.web.api import docs, echo, monitoring
from AiRecAgent.web.api.recruiting import views as recruiting_views

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(docs.router)
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
api_router.include_router(
    recruiting_views.router,
    prefix="/v1",
    tags=["recruiting"],
)
