"""Session CRUD API endpoints."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import get_db
from article_mind_service.models.article import Article
from article_mind_service.models.session import ResearchSession

if TYPE_CHECKING:
    from article_mind_service.embeddings import EmbeddingPipeline
from article_mind_service.schemas.session import (
    ChangeStatusRequest,
    CreateSessionRequest,
    ReindexResponse,
    SessionListResponse,
    SessionResponse,
    SessionStatus,
    UpdateSessionRequest,
)

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["sessions"],
)


def session_to_response(
    session: ResearchSession, article_count: int | None = None
) -> SessionResponse:
    """Convert SQLAlchemy model to Pydantic response.

    Args:
        session: Database model instance
        article_count: Optional pre-computed article count. If None, computes from relationship.

    Returns:
        Pydantic response schema
    """
    if article_count is None:
        # Compute from relationship (filters out soft-deleted articles)
        article_count = len([a for a in session.articles if a.deleted_at is None])

    return SessionResponse(
        id=session.id,
        name=session.name,
        description=session.description,
        status=session.status,
        article_count=article_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


async def get_session_or_404(
    session_id: int,
    db: AsyncSession,
) -> ResearchSession:
    """Fetch a session by ID or raise 404.

    Args:
        session_id: Session ID to fetch
        db: Database session

    Returns:
        ResearchSession model

    Raises:
        HTTPException: 404 if session not found or deleted
    """
    result = await db.execute(
        select(ResearchSession).where(
            ResearchSession.id == session_id,
            ResearchSession.deleted_at.is_(None),  # Exclude soft-deleted
        )
    )
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id {session_id} not found",
        )

    return session


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
    description="Create a new research session. Sessions start in 'draft' status.",
    responses={
        201: {"description": "Session created successfully"},
        400: {"description": "Validation error"},
    },
)
async def create_session(
    data: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Create a new research session.

    Args:
        data: Session creation data (name, description)
        db: Database session

    Returns:
        Created session data
    """
    session = ResearchSession(
        name=data.name,
        description=data.description,
        status="draft",
    )

    db.add(session)
    await db.flush()  # Get the ID without committing
    await db.refresh(session)  # Refresh to get server defaults

    return session_to_response(session)


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List all sessions",
    description="Get a list of all non-deleted research sessions with optional filtering.",
    responses={
        200: {"description": "List of sessions"},
    },
)
async def list_sessions(
    status_filter: SessionStatus | None = Query(
        default=None,
        alias="status",
        description="Filter by session status",
    ),
    db: AsyncSession = Depends(get_db),
) -> SessionListResponse:
    """List all research sessions.

    Args:
        status_filter: Optional status to filter by
        db: Database session

    Returns:
        List of sessions with total count
    """
    # Base query - exclude soft-deleted sessions
    query = select(ResearchSession).where(ResearchSession.deleted_at.is_(None))

    # Apply status filter if provided
    if status_filter:
        query = query.where(ResearchSession.status == status_filter)

    # Order by updated_at descending (most recent first)
    query = query.order_by(ResearchSession.updated_at.desc())

    # Execute query
    result = await db.execute(query)
    sessions = result.scalars().all()

    # Get total count (for pagination metadata)
    count_query = select(func.count(ResearchSession.id)).where(ResearchSession.deleted_at.is_(None))
    if status_filter:
        count_query = count_query.where(ResearchSession.status == status_filter)

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return SessionListResponse(
        sessions=[session_to_response(s) for s in sessions],
        total=total,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get a session by ID",
    description="Retrieve a single research session by its unique identifier.",
    responses={
        200: {"description": "Session found"},
        404: {"description": "Session not found"},
    },
)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Get a single research session.

    Args:
        session_id: Unique session identifier
        db: Database session

    Returns:
        Session data

    Raises:
        HTTPException: 404 if not found
    """
    session = await get_session_or_404(session_id, db)
    return session_to_response(session)


@router.patch(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Update a session",
    description="Update session name and/or description. Only provided fields are updated.",
    responses={
        200: {"description": "Session updated"},
        404: {"description": "Session not found"},
        400: {"description": "Validation error"},
    },
)
async def update_session(
    session_id: int,
    data: UpdateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Update a research session.

    Args:
        session_id: Session to update
        data: Fields to update (name and/or description)
        db: Database session

    Returns:
        Updated session data

    Raises:
        HTTPException: 404 if not found
    """
    session = await get_session_or_404(session_id, db)

    # Get only fields that were explicitly set in the request
    # This distinguishes between "field not sent" and "field set to None/empty"
    updated_fields = data.model_dump(exclude_unset=True)

    # Get article count before modifying (to avoid lazy load after flush)
    # We need to access this before flush() to avoid triggering lazy loads
    article_count = len([a for a in session.articles if a.deleted_at is None])

    # Update only provided fields
    if "name" in updated_fields:
        session.name = data.name

    if "description" in updated_fields:
        # data.description will be None if empty string was sent (validator converts it)
        # This is correct - we want to clear the description in this case
        session.description = data.description

    # Flush changes to trigger database-level updates (like updated_at)
    await db.flush()

    # Refresh to get server-generated values (like updated_at from onupdate trigger)
    # This is safe after flush() - it will see the flushed changes
    await db.refresh(session, attribute_names=["updated_at"])

    return session_to_response(session, article_count=article_count)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session (soft delete)",
    description="Soft delete a session. The session can be recovered by an administrator.",
    responses={
        204: {"description": "Session deleted"},
        404: {"description": "Session not found"},
    },
)
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft delete a research session.

    Args:
        session_id: Session to delete
        db: Database session

    Raises:
        HTTPException: 404 if not found
    """
    session = await get_session_or_404(session_id, db)

    # Soft delete by setting deleted_at
    session.deleted_at = datetime.now(UTC)

    await db.flush()


@router.post(
    "/{session_id}/status",
    response_model=SessionResponse,
    summary="Change session status",
    description=(
        "Transition session to a new status. "
        "Valid transitions: draft->active, draft->archived, "
        "active->completed, active->archived, completed->archived."
    ),
    responses={
        200: {"description": "Status changed"},
        400: {"description": "Invalid status transition"},
        404: {"description": "Session not found"},
    },
)
async def change_session_status(
    session_id: int,
    data: ChangeStatusRequest,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Change the status of a research session.

    Args:
        session_id: Session to update
        data: New status
        db: Database session

    Returns:
        Updated session data

    Raises:
        HTTPException: 404 if not found, 400 if invalid transition
    """
    session = await get_session_or_404(session_id, db)

    # Validate transition
    if not session.can_transition_to(data.status):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot transition from '{session.status}' to '{data.status}'",
        )

    # Get article count before modifying (to avoid lazy load after flush)
    article_count = len([a for a in session.articles if a.deleted_at is None])

    session.status = data.status

    # Flush changes to trigger database-level updates (like updated_at)
    await db.flush()

    # Refresh to get server-generated values (like updated_at from onupdate trigger)
    await db.refresh(session, attribute_names=["updated_at"])

    return session_to_response(session, article_count=article_count)


@router.post(
    "/{session_id}/reindex",
    response_model=ReindexResponse,
    summary="Reindex articles in a session",
    description=(
        "Manually trigger reindexing of articles in a session. "
        "This queues articles that have completed extraction but failed or "
        "are pending embedding. Useful for recovering articles extracted "
        "before auto-embedding was enabled."
    ),
    responses={
        200: {"description": "Articles queued for reindexing"},
        404: {"description": "Session not found"},
    },
)
async def reindex_session(
    session_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ReindexResponse:
    """Reindex articles in a session that need embedding.

    This endpoint finds all articles in the session with:
    - extraction_status="completed" (successfully extracted)
    - embedding_status="pending" OR "failed" (not yet embedded or failed)

    For each matching article, it triggers the embedding pipeline in the background.

    Design Decisions:

    1. Background Processing:
       - Uses FastAPI BackgroundTasks for async processing
       - Returns immediately with count of queued articles
       - Trade-off: Fast response vs. no progress tracking

    2. Article Selection Criteria:
       - MUST have completed extraction (has content_text)
       - MUST have pending or failed embedding status
       - Ignores soft-deleted articles
       - Trade-off: Conservative (only completed) vs. Aggressive (retry all)

    3. Idempotent Operation:
       - Safe to call multiple times on same session
       - Already-indexed articles (embedding_status="completed") are skipped
       - No-op if no articles need reindexing

    Args:
        session_id: Session containing articles to reindex
        background_tasks: FastAPI background task manager
        db: Database session

    Returns:
        Response with count and IDs of articles queued

    Raises:
        HTTPException: 404 if session not found

    Performance:
        - Query time: O(n) where n = articles in session
        - Response time: <100ms (queues background tasks)
        - Actual embedding: 2-10 seconds per article (async)
    """
    # Verify session exists (reuses existing helper)
    session = await get_session_or_404(session_id, db)

    # Find articles needing reindex
    # Criteria:
    # 1. Belongs to this session
    # 2. Extraction completed (has content)
    # 3. Embedding pending or failed (needs indexing)
    # 4. Not soft-deleted
    result = await db.execute(
        select(Article).where(
            and_(
                Article.session_id == session_id,
                Article.extraction_status == "completed",
                or_(
                    Article.embedding_status == "pending",
                    Article.embedding_status == "failed",
                ),
                Article.deleted_at.is_(None),  # Exclude soft-deleted
            )
        )
    )
    articles = result.scalars().all()

    # Queue embedding for each article
    # Import here to avoid circular dependency and keep module loading fast
    from article_mind_service.embeddings import get_embedding_pipeline

    # Pass db session to use database provider settings
    pipeline = await get_embedding_pipeline(db=db)

    for article in articles:
        # Add background task to process this article
        # Each task runs independently and updates article status
        background_tasks.add_task(
            _reindex_article,
            article_id=article.id,
            session_id=str(session.id),
            text=article.content_text,
            source_url=article.original_url or article.canonical_url or "",
            pipeline=pipeline,
            db=db,
        )

    return ReindexResponse(
        session_id=session_id,
        articles_queued=len(articles),
        article_ids=[a.id for a in articles],
    )


async def _reindex_article(
    article_id: int,
    session_id: str,
    text: str | None,
    source_url: str,
    pipeline: "EmbeddingPipeline",
    db: AsyncSession,
) -> None:
    """Background task to reindex a single article.

    This is a simple wrapper around the embedding pipeline that handles
    errors gracefully and logs progress.

    Args:
        article_id: Article to reindex
        session_id: Session ID as string (for ChromaDB collection)
        text: Extracted article text
        source_url: Original article URL
        pipeline: Configured embedding pipeline
        db: Database session

    Error Handling:
        - Catches all exceptions to prevent one failure from blocking others
        - Logs errors for debugging
        - Updates article embedding_status to "failed" on error
    """
    import logging

    logger = logging.getLogger(__name__)

    if not text:
        logger.warning(f"Article {article_id} has no content text, skipping reindex")
        return

    try:
        logger.info(f"Reindexing article {article_id} in session {session_id}")

        chunk_count = await pipeline.process_article(
            article_id=article_id,
            session_id=session_id,
            text=text,
            source_url=source_url,
            db=db,
        )

        logger.info(f"Reindexing completed for article {article_id}: {chunk_count} chunks")

    except Exception as e:
        logger.error(f"Reindexing failed for article {article_id}: {e}")
        # Pipeline already updates status to "failed" on error
