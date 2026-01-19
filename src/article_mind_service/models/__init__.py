"""SQLAlchemy models for database schema."""

# Import all models here to ensure they are registered with Base.metadata
# This is required for Alembic autogenerate to detect models

from .article import Article, ArticleType, ExtractionStatus
from .session import ResearchSession, SessionStatus

__all__ = [
    "Article",
    "ArticleType",
    "ExtractionStatus",
    "ResearchSession",
    "SessionStatus",
]
