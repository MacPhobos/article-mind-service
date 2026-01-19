"""FastAPI routers for API endpoints."""

from .articles import router as articles_router
from .chat import router as chat_router
from .health import router as health_router
from .search import router as search_router
from .sessions import router as sessions_router

__all__ = [
    "articles_router",
    "chat_router",
    "health_router",
    "search_router",
    "sessions_router",
]
