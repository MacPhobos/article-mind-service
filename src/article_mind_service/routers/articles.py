"""Article CRUD API endpoints."""

from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings
from article_mind_service.database import get_db
from article_mind_service.models import Article, ResearchSession
from article_mind_service.schemas import (
    AddUrlRequest,
    ArticleContentResponse,
    ArticleListResponse,
    ArticleResponse,
    UploadFileResponse,
)
from article_mind_service.tasks import extract_article_content

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}/articles",
    tags=["articles"],
)


# =============================================================================
# Helper Functions
# =============================================================================


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
            ResearchSession.deleted_at.is_(None),
        )
    )
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return session


async def get_article_or_404(
    session_id: int,
    article_id: int,
    db: AsyncSession,
) -> Article:
    """Fetch an article by ID or raise 404.

    Args:
        session_id: Parent session ID
        article_id: Article ID to fetch
        db: Database session

    Returns:
        Article model

    Raises:
        HTTPException: 404 if article not found or deleted
    """
    result = await db.execute(
        select(Article).where(
            Article.id == article_id,
            Article.session_id == session_id,
            Article.deleted_at.is_(None),
        )
    )
    article = result.scalar_one_or_none()

    if article is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found in session {session_id}",
        )

    return article


def article_to_response(article: Article) -> ArticleResponse:
    """Convert SQLAlchemy model to Pydantic response.

    Args:
        article: Database model instance

    Returns:
        Pydantic response schema
    """
    return ArticleResponse(
        id=article.id,
        session_id=article.session_id,
        type=article.type,
        original_url=article.original_url,
        original_filename=article.original_filename,
        title=article.title,
        extraction_status=article.extraction_status,
        has_content=article.content_text is not None and len(article.content_text) > 0,
        created_at=article.created_at,
        updated_at=article.updated_at,
    )


def get_upload_dir(session_id: int, article_id: int) -> Path:
    """Get upload directory for an article.

    Args:
        session_id: Parent session ID
        article_id: Article ID

    Returns:
        Path to upload directory
    """
    base_path = Path(settings.upload_base_path)
    return base_path / str(session_id) / str(article_id)


def ensure_upload_dir(session_id: int, article_id: int) -> Path:
    """Create and return upload directory.

    Args:
        session_id: Parent session ID
        article_id: Article ID

    Returns:
        Path to upload directory (created if it doesn't exist)
    """
    upload_dir = get_upload_dir(session_id, article_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/url",
    response_model=ArticleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add article from URL",
    description="Add an article to the session by providing its URL. Content extraction happens asynchronously.",
    responses={
        201: {"description": "Article added successfully"},
        404: {"description": "Session not found"},
        400: {"description": "Validation error"},
    },
)
async def add_url_article(
    session_id: int,
    data: AddUrlRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ArticleResponse:
    """Add an article from a URL.

    Args:
        session_id: Parent session ID
        data: URL and optional title
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Created article data

    Raises:
        HTTPException: 404 if session not found
    """
    # Verify session exists
    await get_session_or_404(session_id, db)

    # Create article
    article = Article(
        session_id=session_id,
        type="url",
        original_url=str(data.url),
        title=data.title,
        extraction_status="pending",
    )
    db.add(article)
    await db.commit()
    await db.refresh(article)

    # Trigger background extraction
    background_tasks.add_task(extract_article_content, article.id, db)

    return article_to_response(article)


@router.post(
    "/upload",
    response_model=UploadFileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload article file",
    description="Upload a file (PDF, DOCX, TXT, etc.) as an article. Content extraction happens asynchronously.",
    responses={
        201: {"description": "File uploaded successfully"},
        404: {"description": "Session not found"},
        400: {"description": "Validation error"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    },
)
async def upload_article_file(
    session_id: int,
    file: UploadFile = File(..., description="File to upload"),
    db: AsyncSession = Depends(get_db),
) -> UploadFileResponse:
    """Upload a file as an article.

    Design Decision: File Storage Strategy
    --------------------------------------
    Files are stored in: data/uploads/{session_id}/{article_id}/original.{ext}

    This structure enables:
    - Easy cleanup when session is deleted
    - Isolation between sessions
    - Simple file serving without DB lookup
    - Future: multiple files per article (e.g., attachments)

    The original file extension is preserved for content-type detection
    during extraction.

    Args:
        session_id: Parent session ID
        file: Uploaded file
        db: Database session

    Returns:
        Upload response with article ID and metadata

    Raises:
        HTTPException: 404 if session not found, 400 for validation, 413 for size, 415 for type
    """
    # Verify session exists
    await get_session_or_404(session_id, db)

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    # Check file size (read size from file object)
    file_content = await file.read()
    file_size = len(file_content)

    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB",
        )

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Create article record first to get ID
    article = Article(
        session_id=session_id,
        type="file",
        original_filename=file.filename,
        extraction_status="pending",
    )
    db.add(article)
    await db.flush()
    await db.refresh(article)

    # Save file to filesystem
    try:
        upload_dir = ensure_upload_dir(session_id, article.id)
        file_path = upload_dir / f"original{file_ext}"

        with open(file_path, "wb") as f:
            f.write(file_content)

        # Update article with storage path (relative to base path)
        article.storage_path = f"{session_id}/{article.id}/original{file_ext}"
        await db.flush()

    except OSError as e:
        # Rollback article creation on file save failure
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        ) from e

    return UploadFileResponse(
        id=article.id,
        filename=file.filename,
        size_bytes=file_size,
        extraction_status=article.extraction_status,
        created_at=article.created_at,
    )


@router.get(
    "",
    response_model=ArticleListResponse,
    summary="List articles in session",
    description="Get all active articles in a research session.",
    responses={
        200: {"description": "List of articles"},
        404: {"description": "Session not found"},
    },
)
async def list_articles(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> ArticleListResponse:
    """List all articles in a session.

    Args:
        session_id: Parent session ID
        db: Database session

    Returns:
        List of articles with total count

    Raises:
        HTTPException: 404 if session not found
    """
    # Verify session exists
    await get_session_or_404(session_id, db)

    # Query articles
    result = await db.execute(
        select(Article)
        .where(
            Article.session_id == session_id,
            Article.deleted_at.is_(None),
        )
        .order_by(Article.created_at.desc())
    )
    articles = result.scalars().all()

    return ArticleListResponse(
        items=[article_to_response(a) for a in articles],
        total=len(articles),
        session_id=session_id,
    )


@router.get(
    "/{article_id}",
    response_model=ArticleResponse,
    summary="Get article details",
    description="Get detailed information about a specific article.",
    responses={
        200: {"description": "Article found"},
        404: {"description": "Article not found"},
    },
)
async def get_article(
    session_id: int,
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> ArticleResponse:
    """Get a specific article.

    Args:
        session_id: Parent session ID
        article_id: Article ID
        db: Database session

    Returns:
        Article data

    Raises:
        HTTPException: 404 if article not found
    """
    article = await get_article_or_404(session_id, article_id, db)
    return article_to_response(article)


@router.delete(
    "/{article_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete article",
    description="Soft delete an article from the session.",
    responses={
        204: {"description": "Article deleted"},
        404: {"description": "Article not found"},
    },
)
async def delete_article(
    session_id: int,
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft delete an article.

    Args:
        session_id: Parent session ID
        article_id: Article ID
        db: Database session

    Raises:
        HTTPException: 404 if article not found
    """
    article = await get_article_or_404(session_id, article_id, db)

    # Soft delete
    article.deleted_at = datetime.now(UTC)
    await db.flush()

    # Note: We don't delete files on soft delete
    # A cleanup job can remove files for hard-deleted articles later


@router.get(
    "/{article_id}/content",
    response_model=ArticleContentResponse,
    summary="Get extracted content",
    description="Get the extracted text content of an article. Only available when extraction is completed.",
    responses={
        200: {"description": "Content retrieved"},
        404: {"description": "Article not found"},
        400: {"description": "Content not available (extraction not completed)"},
    },
)
async def get_article_content(
    session_id: int,
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> ArticleContentResponse:
    """Get extracted content for an article.

    Args:
        session_id: Parent session ID
        article_id: Article ID
        db: Database session

    Returns:
        Article content data

    Raises:
        HTTPException: 404 if article not found, 400 if content not available
    """
    article = await get_article_or_404(session_id, article_id, db)

    if article.extraction_status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Content not available. Extraction status: {article.extraction_status}",
        )

    if not article.content_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No content available for this article",
        )

    return ArticleContentResponse(
        id=article.id,
        title=article.title,
        content_text=article.content_text,
        extraction_status=article.extraction_status,
    )
