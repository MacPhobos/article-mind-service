"""Session CRUD API endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import get_db
from article_mind_service.models.session import ResearchSession
from article_mind_service.schemas.session import (
    ChangeStatusRequest,
    CreateSessionRequest,
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
