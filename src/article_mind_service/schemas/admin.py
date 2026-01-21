"""Admin panel schemas for task management and operations.

Design Decisions:

1. AdminReindexRequest:
   - session_ids: Optional list to enable selective reindexing
   - force: Boolean flag to force reindex even if already completed
   - Rationale: Flexibility for different admin scenarios
   - Trade-off: More complex API vs. single "reindex all" button

2. AdminReindexResponse:
   - Returns task_id for SSE progress tracking
   - Includes total counts for UI display
   - Includes progress_url for convenience (client can construct it)
   - Rationale: All info needed to start tracking in one response
   - Trade-off: Slightly larger payload vs. multiple requests

3. TaskStatusResponse:
   - Snapshot of current task state (not streaming)
   - Includes progress_percent for convenience (client could calculate)
   - Rationale: Polling-friendly for clients without SSE support
   - Trade-off: Redundant with SSE stream vs. fallback support

4. Literal Types:
   - Using Literal for status/task_type enums
   - Rationale: Type-safe, auto-generates OpenAPI constraints
   - Matches existing pattern in health.py
"""

from pydantic import BaseModel, Field


class AdminReindexRequest(BaseModel):
    """Request to start admin reindex operation.

    Attributes:
        session_ids: Optional list of session IDs to reindex.
                    If None, reindexes all sessions.
        force: If True, reindexes even articles with status='completed'.
              If False, only reindexes pending/failed embeddings.

    Example:
        Reindex all sessions:
        >>> AdminReindexRequest(session_ids=None, force=True)

        Reindex specific sessions:
        >>> AdminReindexRequest(session_ids=[1, 2, 3], force=False)
    """

    session_ids: list[int] | None = Field(
        default=None,
        description="Session IDs to reindex (None = all sessions)",
        examples=[[1, 2, 3], None],
    )
    force: bool = Field(
        default=False,
        description="Force reindex even if status is 'completed'",
    )


class AdminReindexResponse(BaseModel):
    """Response after starting admin reindex.

    Contains all information needed to track reindex progress via SSE.

    Attributes:
        task_id: Unique task identifier (UUID)
        total_sessions: Number of sessions being reindexed
        total_articles: Number of articles being reindexed
        progress_url: Full URL to SSE progress stream endpoint

    Example:
        >>> AdminReindexResponse(
        ...     task_id="550e8400-e29b-41d4-a716-446655440000",
        ...     total_sessions=5,
        ...     total_articles=150,
        ...     progress_url="/api/v1/admin/reindex/550e8400-.../progress"
        ... )
    """

    task_id: str = Field(description="Unique task identifier")
    total_sessions: int = Field(description="Number of sessions being reindexed", ge=0)
    total_articles: int = Field(description="Number of articles being reindexed", ge=0)
    progress_url: str = Field(description="SSE progress stream endpoint URL")


class TaskStatusResponse(BaseModel):
    """Current status of a background task.

    This is a snapshot of task state for polling clients.
    For real-time updates, use SSE endpoint instead.

    Attributes:
        task_id: Unique task identifier
        task_type: Type of task (reindex, export, etc.)
        status: Current task status
        total_items: Total number of items to process
        processed_items: Number of items processed so far
        failed_items: Number of items that failed
        progress_percent: Progress percentage (0-100)
        message: Human-readable status message
        errors: List of errors encountered

    Example:
        >>> TaskStatusResponse(
        ...     task_id="550e8400-...",
        ...     task_type="reindex",
        ...     status="running",
        ...     total_items=100,
        ...     processed_items=50,
        ...     failed_items=2,
        ...     progress_percent=50,
        ...     message="Processing article 50 of 100",
        ...     errors=[{"item_id": "123", "error": "timeout"}]
        ... )
    """

    task_id: str = Field(description="Unique task identifier")
    task_type: str = Field(description="Type of task (reindex, export, etc.)")
    status: str = Field(description="Current task status (pending, running, completed, failed)")
    total_items: int = Field(description="Total number of items to process", ge=0)
    processed_items: int = Field(description="Number of items processed so far", ge=0)
    failed_items: int = Field(description="Number of items that failed", ge=0)
    progress_percent: int = Field(description="Progress percentage (0-100)", ge=0, le=100)
    message: str | None = Field(default=None, description="Human-readable status message")
    errors: list[dict[str, str]] = Field(
        default_factory=list, description="List of errors encountered"
    )
