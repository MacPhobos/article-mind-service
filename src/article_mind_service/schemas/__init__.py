"""Pydantic schemas for API request/response validation."""

from .admin import AdminReindexRequest, AdminReindexResponse, TaskStatusResponse
from .article import (
    AddUrlRequest,
    ArticleContentResponse,
    ArticleListResponse,
    ArticleResponse,
    ArticleType,
    ExtractionStatus,
    UploadFileResponse,
)
from .health import HealthResponse
from .search import SearchMode, SearchRequest, SearchResponse, SearchResult
from .session import (
    ChangeStatusRequest,
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
    SessionStatus,
    UpdateSessionRequest,
)

__all__ = [
    # Health
    "HealthResponse",
    # Session
    "ChangeStatusRequest",
    "CreateSessionRequest",
    "SessionListResponse",
    "SessionResponse",
    "SessionStatus",
    "UpdateSessionRequest",
    # Article
    "AddUrlRequest",
    "ArticleContentResponse",
    "ArticleListResponse",
    "ArticleResponse",
    "ArticleType",
    "ExtractionStatus",
    "UploadFileResponse",
    # Search
    "SearchMode",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    # Admin
    "AdminReindexRequest",
    "AdminReindexResponse",
    "TaskStatusResponse",
]
