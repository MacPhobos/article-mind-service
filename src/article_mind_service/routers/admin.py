"""Admin panel endpoints for task management and system operations.

This router provides administrative endpoints for managing background tasks
like reindexing embeddings across all sessions with real-time progress tracking.

Design Decisions:

1. Endpoint Structure:
   - POST /api/v1/admin/reindex - Start reindex task
   - GET /api/v1/admin/reindex/{task_id}/progress - SSE stream
   - GET /api/v1/admin/reindex/{task_id} - Polling endpoint
   - POST /api/v1/admin/reindex/{task_id}/cancel - Cancel task

   Rationale: RESTful design with clear separation of concerns
   Trade-off: More endpoints vs. simpler single endpoint

2. SSE vs Polling:
   - Provides both SSE (real-time) and polling (snapshot) endpoints
   - Rationale: SSE for modern browsers, polling as fallback
   - Trade-off: More complex API surface vs. flexibility

3. Error Handling:
   - Returns 404 if task_id not found
   - Returns 404 if no articles to reindex (not 400)
   - Rationale: Consistent error responses, avoids client confusion
   - Trade-off: 404 may be misleading vs. more specific status codes

4. Background Task Management:
   - Uses FastAPI BackgroundTasks for async execution
   - Task runs in background, endpoint returns immediately
   - Rationale: Non-blocking API, user gets instant task_id
   - Trade-off: No direct error handling vs. simpler API

5. Database Session Factory:
   - Passes AsyncSessionLocal (sessionmaker) to background task
   - Rationale: Background task creates its own session lifecycle
   - Trade-off: More complex vs. shared session (which would timeout)
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from article_mind_service.database import AsyncSessionLocal, get_db
from article_mind_service.models.article import Article
from article_mind_service.schemas.admin import (
    AdminReindexRequest,
    AdminReindexResponse,
    TaskStatusResponse,
)
from article_mind_service.tasks import reindex_all_articles, task_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/reindex", response_model=AdminReindexResponse)
async def start_admin_reindex(
    request: AdminReindexRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> AdminReindexResponse:
    """Start reindexing all or selected sessions.

    This endpoint starts a background task to regenerate embeddings for
    articles across sessions. Progress can be tracked via SSE endpoint.

    Query Strategy:
    - If session_ids provided: Only reindex those sessions
    - If session_ids is None: Reindex ALL sessions
    - If force=True: Reindex even if embedding_status="completed"
    - If force=False: Only reindex pending/failed embeddings

    Args:
        request: Reindex request with session_ids and force flags
        background_tasks: FastAPI background task manager
        db: Database session (dependency injected)

    Returns:
        AdminReindexResponse with task_id and progress_url

    Raises:
        HTTPException: 404 if no articles found to reindex

    Example:
        POST /api/v1/admin/reindex
        {
          "session_ids": [1, 2, 3],
          "force": false
        }

        Response:
        {
          "task_id": "550e8400-e29b-41d4-a716-446655440000",
          "total_sessions": 3,
          "total_articles": 150,
          "progress_url": "/api/v1/admin/reindex/550e8400-.../progress"
        }
    """
    # Build query to count articles that will be reindexed
    query = select(Article).where(Article.deleted_at.is_(None))

    # Filter by session_ids if provided
    if request.session_ids is not None:
        query = query.where(Article.session_id.in_(request.session_ids))

    # Filter by embedding status (unless force=True)
    if request.force:
        # Reindex all articles with completed extraction
        query = query.where(Article.extraction_status == "completed")
    else:
        # Only reindex articles with pending or failed embeddings
        query = query.where(
            and_(
                Article.extraction_status == "completed",
                or_(
                    Article.embedding_status == "pending",
                    Article.embedding_status == "failed",
                ),
            )
        )

    # Execute query to get articles
    result = await db.execute(query)
    articles = result.scalars().all()

    if not articles:
        raise HTTPException(
            status_code=404,
            detail="No articles found to reindex with the specified criteria",
        )

    # Count unique sessions
    session_ids = set(a.session_id for a in articles)

    # Create task in registry
    task_id = task_registry.create_task(
        task_type="reindex",
        total_items=len(articles),
    )

    logger.info(
        f"Admin reindex started: task_id={task_id}, "
        f"sessions={len(session_ids)}, articles={len(articles)}, "
        f"force={request.force}"
    )

    # Queue background task
    # Note: We pass AsyncSessionLocal (sessionmaker), not the db session
    # The background task will create its own session lifecycle
    background_tasks.add_task(
        reindex_all_articles,
        task_id=task_id,
        session_ids=list(session_ids) if request.session_ids else None,
        force=request.force,
        task_registry=task_registry,
        db_session_factory=AsyncSessionLocal,
    )

    return AdminReindexResponse(
        task_id=task_id,
        total_sessions=len(session_ids),
        total_articles=len(articles),
        progress_url=f"/api/v1/admin/reindex/{task_id}/progress",
    )


@router.get("/reindex/{task_id}/progress")
async def stream_reindex_progress(task_id: str) -> EventSourceResponse:
    """Stream reindex progress via Server-Sent Events (SSE).

    This endpoint provides real-time progress updates for a reindex task.
    Events are streamed as Server-Sent Events (SSE) which clients can
    consume using EventSource API.

    SSE Event Format:
    - event: "progress" | "complete"
    - data: JSON-serialized TaskProgress model

    Args:
        task_id: Task ID from start_admin_reindex response

    Returns:
        EventSourceResponse streaming TaskProgress updates

    Raises:
        HTTPException: 404 if task_id not found

    Example:
        GET /api/v1/admin/reindex/550e8400-.../progress

        Event Stream:
        event: progress
        data: {"task_id": "550e8400-...", "status": "running", "processed_items": 50, ...}

        event: complete
        data: {"task_id": "550e8400-...", "status": "completed", "processed_items": 100, ...}

    Client Usage (JavaScript):
        const eventSource = new EventSource('/api/v1/admin/reindex/task-id/progress');
        eventSource.addEventListener('progress', (e) => {
          const data = JSON.parse(e.data);
          updateProgressBar(data.processed_items / data.total_items);
        });
        eventSource.addEventListener('complete', (e) => {
          const data = JSON.parse(e.data);
          showCompletionMessage(data);
          eventSource.close();
        });
    """
    task = task_registry.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    async def event_generator():
        """Generate SSE events from task registry progress stream."""
        async for progress in task_registry.stream_progress(task_id):
            # Determine event type based on status
            if progress.status == "completed":
                event_type = "complete"
            elif progress.status == "failed":
                event_type = "error"
            elif progress.status == "cancelled":
                event_type = "cancelled"
            else:
                event_type = "progress"

            # Yield SSE event
            yield {
                "event": event_type,
                "data": progress.model_dump_json(),
            }

    return EventSourceResponse(event_generator())


@router.get("/reindex/{task_id}", response_model=TaskStatusResponse)
async def get_reindex_status(task_id: str) -> TaskStatusResponse:
    """Get current status of a reindex task (polling endpoint).

    This endpoint provides a snapshot of task state for clients that
    cannot use SSE. For real-time updates, use the SSE endpoint instead.

    Args:
        task_id: Task ID from start_admin_reindex response

    Returns:
        TaskStatusResponse with current task state

    Raises:
        HTTPException: 404 if task_id not found

    Example:
        GET /api/v1/admin/reindex/550e8400-.../

        Response:
        {
          "task_id": "550e8400-...",
          "task_type": "reindex",
          "status": "running",
          "total_items": 100,
          "processed_items": 50,
          "failed_items": 2,
          "progress_percent": 50,
          "message": "Processing article 50 of 100",
          "errors": [{"item_id": "123", "error": "timeout"}]
        }
    """
    task = task_registry.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Calculate progress percentage
    percent = task.progress_percent()

    return TaskStatusResponse(
        task_id=task.task_id,
        task_type=task.task_type,
        status=task.status,
        total_items=task.total_items,
        processed_items=task.processed_items,
        failed_items=task.failed_items,
        progress_percent=percent,
        message=task.message,
        errors=task.errors,
    )


@router.post("/reindex/{task_id}/cancel")
async def cancel_reindex(task_id: str) -> dict[str, str]:
    """Request cancellation of a running reindex task.

    This endpoint sets a cancellation flag that the background task will
    check periodically. Cancellation is cooperative - the task must check
    the flag and gracefully stop.

    Cancellation Behavior:
    - Ongoing article processing will complete
    - No new articles will be processed
    - Already-processed articles remain reindexed
    - Task status will be updated to "cancelled"

    Args:
        task_id: Task ID from start_admin_reindex response

    Returns:
        Status message indicating cancellation requested

    Raises:
        HTTPException: 404 if task_id not found or already completed

    Example:
        POST /api/v1/admin/reindex/550e8400-.../cancel

        Response:
        {
          "status": "cancellation_requested",
          "task_id": "550e8400-..."
        }
    """
    if not task_registry.cancel_task(task_id):
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found or already completed",
        )

    logger.info(f"Cancellation requested for task {task_id}")

    return {
        "status": "cancellation_requested",
        "task_id": task_id,
    }
