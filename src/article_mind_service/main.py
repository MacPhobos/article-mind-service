"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import settings
from .database import engine
from .logging_config import configure_logging
from .routers import (
    admin_router,
    articles_router,
    chat_router,
    health_router,
    search_router,
    sessions_router,
    settings_router,
)

# Configure structured logging at application startup
configure_logging(log_level=settings.log_level, json_logs=False)

# Configure SQLAlchemy logging level (suppress INFO logs like "BEGIN", "COMMIT")
logging.getLogger("sqlalchemy.engine").setLevel(
    getattr(logging, settings.sqlalchemy_log_level.upper())
)
logging.getLogger("sqlalchemy.pool").setLevel(
    getattr(logging, settings.sqlalchemy_log_level.upper())
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")

    # Test database connection (non-blocking - service can start without DB)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("✅ Database connection verified")
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("⚠️  Service will start but /health will report degraded status")

    yield

    # Shutdown
    print("Shutting down application")
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
# When CORS_ALLOW_ALL=true, allows all origins (["*"])
# Otherwise, uses comma-separated CORS_ORIGINS list
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Health check router (no prefix - as per API contract)
app.include_router(health_router)

# Sessions CRUD API
app.include_router(sessions_router)

# Articles CRUD API (nested under sessions)
app.include_router(articles_router)

# Search API (knowledge query)
app.include_router(search_router)

# Chat API (Q&A with RAG)
app.include_router(chat_router)

# Admin API (system operations)
app.include_router(admin_router)

# Settings API (provider configuration)
app.include_router(settings_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Article Mind Service API"}
