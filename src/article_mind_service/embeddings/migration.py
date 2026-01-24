"""Migration utilities for transitioning chunking strategies.

Design Decision: Migration Strategy for Semantic Chunking
==========================================================

Problem: Existing articles are chunked with fixed-size strategy. How do we
migrate them to semantic chunking without breaking existing functionality?

Solution: Provide explicit migration utility instead of auto-migration.

Rationale:
- Explicit control: Admin decides when to migrate
- Performance: Semantic chunking is 3-5x slower, don't auto-migrate all content
- Testing: Can test migration on subset before full rollout
- Rollback: Can revert to fixed-size if semantic doesn't work well

Alternative Approaches Rejected:
1. Auto-migration on service startup
   - Rejected: Too slow for large deployments (could take hours)
2. Lazy migration (re-chunk on access)
   - Rejected: Inconsistent performance, complex state management
3. Background queue migration
   - Rejected: Over-engineering for initial implementation

Trade-offs:
- ✅ Explicit control over migration timing
- ✅ Can migrate incrementally (session by session)
- ✅ Simple implementation (no background workers)
- ❌ Manual process (requires admin action)
- ❌ No automatic rollback (must re-migrate manually)

Migration Process:
1. Change configuration: chunking_strategy = "semantic"
2. Run migration utility for specific sessions or all sessions
3. Monitor migration progress and errors
4. Verify search quality improvement

Rollback Process:
1. Change configuration: chunking_strategy = "fixed"
2. Run migration utility again (will re-chunk with fixed-size)
3. Verify rollback completed successfully
"""

import logging
from typing import Any, AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.models.article import Article
from article_mind_service.models.session import ResearchSession

from .pipeline import EmbeddingPipeline

logger = logging.getLogger(__name__)


class ChunkingMigration:
    """Handles migration from fixed to semantic chunking (or vice versa).

    Example:
        # Migrate single session
        migration = ChunkingMigration(db, embedding_pipeline)
        stats = await migration.migrate_session(session_id=123)
        print(f"Migrated {stats['articles_processed']} articles")

        # Migrate all sessions
        async for stats in migration.migrate_all():
            print(f"Session {stats['session_id']}: {stats['articles_processed']} articles")
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_pipeline: EmbeddingPipeline,
    ):
        """Initialize migration utility.

        Args:
            db: Database session for querying articles.
            embedding_pipeline: Embedding pipeline with new chunking strategy.

        Design Decision: Pass pipeline instead of creating internally
        ==============================================================

        Rationale: Caller controls pipeline configuration.
        - Strategy selection: Pipeline already configured with desired strategy
        - Testability: Can inject mock pipeline
        - Flexibility: Can use different pipelines for different sessions
        """
        self.db = db
        self.pipeline = embedding_pipeline

    async def migrate_session(
        self,
        session_id: int,
        force: bool = False,
    ) -> dict[str, Any]:
        """Migrate a single session to new chunking strategy.

        Process:
        1. Query all articles in session
        2. Re-process each article with new chunking strategy
        3. Collect statistics and errors
        4. Return migration summary

        Args:
            session_id: The session to migrate.
            force: If True, re-chunk even if already using same strategy.
                   Useful for fixing corrupted chunks.

        Returns:
            Migration statistics dictionary with keys:
            - session_id: Session ID migrated
            - articles_total: Total articles in session
            - articles_processed: Articles successfully migrated
            - articles_failed: Articles that failed migration
            - chunks_before: Total chunks before migration (estimated)
            - chunks_after: Total chunks after migration
            - errors: List of error dicts with article_id and error message

        Performance:
            - Fixed-size: 100 articles ~1-2 minutes
            - Semantic: 100 articles ~5-15 minutes (3-5x slower)

        Note:
            This operation is expensive. Consider running during off-peak hours.
            Progress is logged to help monitor long-running migrations.

        Example:
            stats = await migration.migrate_session(session_id=123)
            print(f"Migrated {stats['articles_processed']}/{stats['articles_total']} articles")
            if stats['errors']:
                print(f"Errors: {stats['errors']}")
        """
        stats: dict[str, Any] = {
            "session_id": session_id,
            "articles_total": 0,
            "articles_processed": 0,
            "articles_failed": 0,
            "chunks_before": 0,
            "chunks_after": 0,
            "errors": [],
        }

        # Get session articles
        result = await self.db.execute(
            select(Article).where(Article.session_id == session_id)
        )
        articles = result.scalars().all()
        stats["articles_total"] = len(articles)

        if not articles:
            logger.info(f"No articles found in session {session_id}")
            return stats

        logger.info(
            f"Starting migration for session {session_id} ({len(articles)} articles)"
        )

        for i, article in enumerate(articles, 1):
            try:
                # Track chunks before migration
                chunks_before = article.chunk_count or 0
                stats["chunks_before"] += chunks_before

                # Re-process article with new chunking strategy
                # This will delete old chunks and create new ones
                chunk_count = await self.pipeline.process_article(
                    article_id=article.id,
                    session_id=str(article.session_id),
                    text=article.content or "",
                    source_url=article.url,
                    db=self.db,
                )

                stats["articles_processed"] += 1
                stats["chunks_after"] += chunk_count

                # Log progress every 10 articles
                if i % 10 == 0:
                    logger.info(
                        f"Progress: {i}/{len(articles)} articles migrated "
                        f"({stats['articles_processed']} successful, {stats['articles_failed']} failed)"
                    )

            except Exception as e:
                stats["articles_failed"] += 1
                stats["errors"].append(
                    {
                        "article_id": article.id,
                        "url": article.url,
                        "error": str(e),
                    }
                )
                logger.error(f"Failed to migrate article {article.id} ({article.url}): {e}")

        logger.info(
            f"Completed migration for session {session_id}: "
            f"{stats['articles_processed']} successful, {stats['articles_failed']} failed, "
            f"{stats['chunks_before']} -> {stats['chunks_after']} chunks"
        )

        return stats

    async def migrate_all(
        self,
        batch_size: int = 10,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Migrate all sessions in batches.

        Yields migration stats for each session as it completes.
        This allows monitoring progress in real-time for large deployments.

        Args:
            batch_size: Number of articles to process per batch (not used currently,
                       but reserved for future batch optimization).

        Yields:
            Migration statistics dict for each session.

        Performance:
            - Fixed-size: 1000 articles ~10-20 minutes
            - Semantic: 1000 articles ~50-150 minutes (3-5x slower)

        Example:
            async for stats in migration.migrate_all():
                print(f"Session {stats['session_id']}: {stats['articles_processed']} articles")
                if stats['errors']:
                    print(f"  Errors: {len(stats['errors'])}")

        Note:
            This is a long-running operation. Consider:
            - Running during off-peak hours
            - Using a background task queue (Celery, etc.)
            - Implementing pause/resume functionality
        """
        # Get all session IDs
        result = await self.db.execute(select(ResearchSession.id))
        session_ids = [row[0] for row in result.fetchall()]

        total_sessions = len(session_ids)
        logger.info(f"Starting migration for {total_sessions} sessions")

        for i, session_id in enumerate(session_ids, 1):
            logger.info(f"Migrating session {session_id} ({i}/{total_sessions})")
            stats = await self.migrate_session(session_id)
            yield stats

        logger.info(f"Completed migration for all {total_sessions} sessions")


# Convenience function for CLI or admin endpoints
async def migrate_to_semantic(
    db: AsyncSession,
    embedding_pipeline: EmbeddingPipeline,
    session_id: int | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Convenience function to migrate to semantic chunking.

    Args:
        db: Database session.
        embedding_pipeline: Pipeline with semantic chunking strategy.
        session_id: Optional session ID to migrate. If None, migrates all sessions.

    Returns:
        Migration statistics (single dict or list of dicts).

    Example:
        # Migrate single session
        stats = await migrate_to_semantic(db, pipeline, session_id=123)

        # Migrate all sessions
        all_stats = await migrate_to_semantic(db, pipeline)
    """
    migration = ChunkingMigration(db, embedding_pipeline)

    if session_id is not None:
        return await migration.migrate_session(session_id)
    else:
        results = []
        async for stats in migration.migrate_all():
            results.append(stats)
        return results
