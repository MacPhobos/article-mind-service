"""Session request/response schemas for API contract."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Session status type for type safety
SessionStatus = Literal["draft", "active", "completed", "archived"]


class CreateSessionRequest(BaseModel):
    """Request schema for creating a new session.

    All new sessions start in 'draft' status.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Session display name",
        examples=["My Research Project"],
    )
    description: str | None = Field(
        default=None,
        max_length=5000,
        description="Optional session description",
        examples=["Research on machine learning algorithms"],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Strip whitespace from name."""
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Strip whitespace from description."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None


class UpdateSessionRequest(BaseModel):
    """Request schema for updating a session.

    All fields are optional - only provided fields are updated.

    To distinguish between "field not sent" and "field explicitly set to empty",
    we use model_config to exclude unset fields from model_dump().
    """

    model_config = {"extra": "forbid"}  # Reject unknown fields

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New session name",
        examples=["Updated Project Name"],
    )
    description: str | None = Field(
        default=None,
        max_length=5000,
        description="New session description (empty string to clear)",
        examples=["Updated description"],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Strip whitespace from name."""
        if v is None:
            return None
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Strip whitespace from description and convert empty to None."""
        if v is None:
            return None
        stripped = v.strip()
        # Empty string explicitly converts to None (this is intentional)
        # We can distinguish between "not sent" (field not in model dump exclude_unset)
        # and "sent as empty" (field in model dump with value None)
        return stripped if stripped else None


class ChangeStatusRequest(BaseModel):
    """Request schema for changing session status.

    Valid transitions:
    - draft -> active, archived
    - active -> completed, archived
    - completed -> archived
    - archived -> (none - cannot be changed)
    """

    status: SessionStatus = Field(
        ...,
        description="New status for the session",
        examples=["active"],
    )


class SessionResponse(BaseModel):
    """Response schema for a single session.

    This is the canonical session representation in the API.
    Used for all endpoints that return session data.
    """

    id: int = Field(
        ...,
        description="Unique session identifier",
        examples=[1],
    )
    name: str = Field(
        ...,
        description="Session display name",
        examples=["My Research Project"],
    )
    description: str | None = Field(
        default=None,
        description="Session description",
        examples=["Research on machine learning algorithms"],
    )
    status: SessionStatus = Field(
        ...,
        description="Current session status",
        examples=["active"],
    )
    article_count: int = Field(
        default=0,
        ge=0,
        description="Number of articles in this session",
        examples=[12],
    )
    created_at: datetime = Field(
        ...,
        description="When the session was created",
        examples=["2026-01-15T10:30:00Z"],
    )
    updated_at: datetime = Field(
        ...,
        description="When the session was last updated",
        examples=["2026-01-19T14:45:00Z"],
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "name": "Machine Learning Research",
                    "description": "Collecting papers on deep learning",
                    "status": "active",
                    "article_count": 15,
                    "created_at": "2026-01-15T10:30:00Z",
                    "updated_at": "2026-01-19T14:45:00Z",
                }
            ]
        },
    }


class SessionListResponse(BaseModel):
    """Response schema for listing sessions.

    Includes pagination metadata for future expansion.
    """

    sessions: list[SessionResponse] = Field(
        ...,
        description="List of sessions",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of sessions (excluding deleted)",
        examples=[25],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sessions": [
                        {
                            "id": 1,
                            "name": "Research Project A",
                            "description": "First project",
                            "status": "active",
                            "article_count": 10,
                            "created_at": "2026-01-15T10:30:00Z",
                            "updated_at": "2026-01-19T14:45:00Z",
                        }
                    ],
                    "total": 1,
                }
            ]
        },
    }


class ReindexResponse(BaseModel):
    """Response schema for session reindex operation.

    Returned when manually triggering reindexing of articles in a session.
    This operation queues articles for embedding that were previously extracted
    but failed to be indexed (e.g., before auto-embedding was enabled).
    """

    session_id: int = Field(
        ...,
        description="ID of the session that was reindexed",
        examples=[179],
    )
    articles_queued: int = Field(
        ...,
        ge=0,
        description="Number of articles queued for reindexing",
        examples=[3],
    )
    article_ids: list[int] = Field(
        ...,
        description="List of article IDs queued for reindexing",
        examples=[[39, 40, 41]],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": 179,
                    "articles_queued": 3,
                    "article_ids": [39, 40, 41],
                }
            ]
        },
    }
