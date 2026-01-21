"""Reindex task implementation for admin panel.

This module provides shared reindex logic used by both session-level
and admin-level reindex endpoints. It handles embedding regeneration
for articles with progress tracking via TaskRegistry.

Design Decisions:

1. Extracted from sessions.py:
   - Rationale: Avoid code duplication between session and admin reindex
   - Session reindex uses FastAPI BackgroundTasks (simple, no progress)
   - Admin reindex uses TaskRegistry (complex, real-time progress)
   - Shared logic: reindex_article function

2. Database Session Factory:
   - reindex_all_articles takes a callable that creates new sessions
   - Rationale: Long-running task needs dedicated session, not shared one
   - Avoids session lifecycle issues (timeout, closed connections)
   - Trade-off: More complex API vs. safer session management

3. Error Handling Strategy:
   - Continue on individual article failures (don't fail entire batch)
   - Record errors in TaskRegistry for visibility
   - Rationale: Partial success better than all-or-nothing
   - Trade-off: More complex error tracking vs. simpler fail-fast

4. Progress Granularity:
   - Update progress after each article (real-time feedback)
   - Rationale: Better UX, allows cancellation mid-operation
   - Trade-off: More database writes vs. coarser progress updates
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.models.article import Article

if TYPE_CHECKING:
    from article_mind_service.embeddings import EmbeddingPipeline
    from article_mind_service.tasks.registry import TaskRegistry

logger = logging.getLogger(__name__)


async def reindex_article(
    article_id: int,
    session_id: str,
    text: str | None,
    source_url: str,
    pipeline: "EmbeddingPipeline",
    db: AsyncSession,
) -> bool:
    """Reindex a single article. Returns True if successful.

    This function is shared between session reindex (BackgroundTasks)
    and admin reindex (TaskRegistry-tracked).

    Args:
        article_id: Article to reindex
        session_id: Session ID as string (for ChromaDB collection)
        text: Extracted article text
        source_url: Original article URL
        pipeline: Configured embedding pipeline
        db: Database session

    Returns:
        True if reindex succeeded, False otherwise

    Error Handling:
        - Catches all exceptions to prevent one failure from blocking others
        - Logs errors for debugging
        - Updates article embedding_status to "failed" on error
        - Returns False on failure (caller decides whether to continue)

    Performance:
        - Time: 2-10 seconds per article (depends on content length, model)
        - Database: 2 writes (status update, chunk insertion)
        - ChromaDB: N writes where N = number of chunks (typically 1-20)
    """
    if not text:
        logger.warning(f"Article {article_id} has no content text, skipping reindex")
        return False

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
        return True

    except Exception as e:
        logger.error(f"Reindexing failed for article {article_id}: {e}")
        # Pipeline already updates status to "failed" on error
        return False


async def reindex_all_articles(
    task_id: str,
    session_ids: list[int] | None,
    force: bool,
    task_registry: "TaskRegistry",
    db_session_factory: Callable[[], AsyncSession],
) -> None:
    """Background task to reindex all articles across sessions.

    This is the main admin reindex implementation. It queries all articles
    needing reindex, processes them with progress tracking, and handles
    errors gracefully.

    Design Decisions:

    1. Query Strategy:
       - If session_ids provided: filter to those sessions
       - If session_ids is None: reindex ALL sessions
       - If force=True: reindex even if embedding_status="completed"
       - If force=False: only reindex pending/failed embeddings

    2. Progress Tracking:
       - Update TaskRegistry after each article
       - Allows real-time progress via SSE
       - Allows mid-operation cancellation

    3. Concurrency:
       - Sequential processing (one article at a time)
       - Rationale: Avoid overwhelming embedding API rate limits
       - Future: Could add configurable concurrency with asyncio.Semaphore

    Args:
        task_id: TaskRegistry task ID for progress tracking
        session_ids: List of session IDs to reindex (None = all sessions)
        force: If True, reindex even completed articles
        task_registry: TaskRegistry instance for progress updates
        db_session_factory: Factory function that creates AsyncSession instances

    Error Handling:
        - Continues on individual article failures
        - Records errors in TaskRegistry
        - Marks task as completed even if some articles failed
        - Only marks task as failed if query or initialization fails

    Cancellation:
        - Checks task_registry.is_cancelled() before each article
        - Marks task as cancelled and exits gracefully
        - Already-processed articles remain reindexed

    Example:
        >>> async def session_factory():
        ...     return get_db()
        >>> await reindex_all_articles(
        ...     task_id="550e8400-...",
        ...     session_ids=[1, 2, 3],
        ...     force=False,
        ...     task_registry=task_registry,
        ...     db_session_factory=session_factory
        ... )
    """
    db: AsyncSession | None = None

    try:
        # Create dedicated database session for this task
        db = db_session_factory()

        # Build query to find articles needing reindex
        query = select(Article).where(Article.deleted_at.is_(None))

        # Filter by session_ids if provided
        if session_ids is not None:
            query = query.where(Article.session_id.in_(session_ids))

        # Filter by embedding status (unless force=True)
        if force:
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

        total_articles = len(articles)
        logger.info(
            f"Task {task_id}: Found {total_articles} articles to reindex "
            f"(session_ids={session_ids}, force={force})"
        )

        # Initialize embedding pipeline
        from article_mind_service.embeddings import get_embedding_pipeline

        pipeline = get_embedding_pipeline()

        # Process each article with progress tracking
        processed = 0
        failed = 0

        for article in articles:
            # Check for cancellation before each article
            if task_registry.is_cancelled(task_id):
                logger.info(f"Task {task_id}: Cancellation requested, stopping")
                # Note: We don't call mark_complete here - let the cancellation
                # handler update the status
                return

            # Reindex article
            success = await reindex_article(
                article_id=article.id,
                session_id=str(article.session_id),
                text=article.content_text,
                source_url=article.original_url or article.canonical_url or "",
                pipeline=pipeline,
                db=db,
            )

            processed += 1

            if not success:
                failed += 1
                # Record error in task registry
                task_registry.record_error(
                    task_id,
                    item_id=str(article.id),
                    error=f"Failed to reindex article {article.id}",
                )

            # Update progress
            await task_registry.update_progress(
                task_id,
                processed=processed,
                current_item=f"Article {article.id}",
                message=f"Processed {processed} of {total_articles} articles",
            )

        # Mark task as completed
        await task_registry.mark_complete(task_id, failed_count=failed)

        logger.info(
            f"Task {task_id}: Reindex completed - "
            f"{processed} processed, {failed} failed"
        )

    except Exception as e:
        logger.exception(f"Task {task_id}: Reindex failed with error: {e}")
        await task_registry.mark_failed(task_id, error=str(e))

    finally:
        # Clean up database session
        if db is not None:
            await db.close()
