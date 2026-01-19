"""Article request/response schemas for API contract."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

# =============================================================================
# Enums (as Literal types for better type safety)
# =============================================================================

ArticleType = Literal["url", "file"]
ExtractionStatus = Literal["pending", "processing", "completed", "failed"]


# =============================================================================
# Article Schemas
# =============================================================================


class AddUrlRequest(BaseModel):
    """Request schema for adding an article via URL.

    Design Decision: URL Validation
    --------------------------------
    Using Pydantic's HttpUrl type for validation because:
    - Automatic scheme validation (http/https)
    - Well-tested URL parsing
    - Clear error messages for invalid URLs

    The title field is optional and will be extracted from the page if not provided.
    """

    url: HttpUrl = Field(
        ...,
        description="URL of the article to add",
        examples=["https://arxiv.org/abs/2301.00001"],
    )
    title: str | None = Field(
        default=None,
        max_length=512,
        description="Optional title (auto-extracted if not provided)",
        examples=["Attention Is All You Need"],
    )

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: HttpUrl) -> HttpUrl:
        """Ensure URL uses http or https scheme."""
        if v.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https scheme")
        return v


class UploadFileResponse(BaseModel):
    """Response schema after file upload.

    Returned immediately after upload, before content extraction.
    """

    id: int = Field(..., description="Article ID")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    extraction_status: ExtractionStatus = Field(
        "pending",
        description="Content extraction status",
    )
    created_at: datetime = Field(..., description="Upload timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 42,
                    "filename": "research_paper.pdf",
                    "size_bytes": 1048576,
                    "extraction_status": "pending",
                    "created_at": "2026-01-19T10:00:00Z",
                }
            ]
        },
    }


class ArticleResponse(BaseModel):
    """Response schema for a single article.

    Used for both URL and file articles with appropriate fields populated.
    """

    id: int = Field(..., description="Unique article identifier")
    session_id: int = Field(..., description="Parent session ID")
    type: ArticleType = Field(..., description="Article type (url or file)")
    original_url: str | None = Field(None, description="Source URL (for url type)")
    original_filename: str | None = Field(
        None,
        description="Original filename (for file type)",
    )
    title: str | None = Field(None, description="Article title")
    extraction_status: ExtractionStatus = Field(
        ...,
        description="Content extraction status",
    )
    has_content: bool = Field(
        False,
        description="Whether extracted content is available",
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "session_id": 1,
                    "type": "url",
                    "original_url": "https://arxiv.org/abs/2301.00001",
                    "original_filename": None,
                    "title": "Attention Is All You Need",
                    "extraction_status": "completed",
                    "has_content": True,
                    "created_at": "2026-01-19T10:00:00Z",
                    "updated_at": "2026-01-19T10:30:00Z",
                }
            ]
        },
    }


class ArticleListResponse(BaseModel):
    """Response schema for listing articles in a session."""

    items: list[ArticleResponse] = Field(..., description="List of articles")
    total: int = Field(..., description="Total number of articles in session")
    session_id: int = Field(..., description="Session ID")


class ArticleContentResponse(BaseModel):
    """Response schema for article extracted content.

    Returns the extracted text content from the article.
    Only available when extraction_status is 'completed'.
    """

    id: int = Field(..., description="Article ID")
    title: str | None = Field(None, description="Article title")
    content_text: str = Field(..., description="Extracted text content")
    extraction_status: ExtractionStatus = Field(..., description="Extraction status")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "title": "Attention Is All You Need",
                    "content_text": "Abstract: The dominant sequence transduction models...",
                    "extraction_status": "completed",
                }
            ]
        },
    }
