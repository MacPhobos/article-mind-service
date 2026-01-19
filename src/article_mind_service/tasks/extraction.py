"""Background task handlers for content extraction."""

import hashlib
import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings
from article_mind_service.extraction import (
    ExtractionError,
    ExtractionPipeline,
    NetworkError,
    PipelineConfig,
)
from article_mind_service.models.article import Article

logger = logging.getLogger(__name__)


async def extract_article_content(
    article_id: int,
    db: AsyncSession,
    config: PipelineConfig | None = None,
) -> None:
    """Background task to extract content from an article's URL.

    Updates the article record with extraction results.

    Design Decision: Direct database session passing instead of creating new session
    - Reuses existing session from endpoint context
    - Avoids session lifecycle complexity
    - Trade-off: Caller must manage session

    Args:
        article_id: ID of article to extract
        db: Database session
        config: Optional pipeline configuration
    """
    # Load article
    article = await db.get(Article, article_id)
    if not article:
        logger.error(f"Article {article_id} not found")
        return

    # Update status to processing
    article.extraction_status = "processing"
    article.extraction_error = None
    await db.commit()

    # Initialize pipeline
    pipeline_config = config or PipelineConfig(
        timeout_seconds=settings.extraction_timeout_seconds,
        max_retries=settings.extraction_max_retries,
        user_agent=settings.extraction_user_agent,
        playwright_headless=settings.playwright_headless,
        max_content_size_mb=settings.extraction_max_content_size_mb,
    )
    pipeline = ExtractionPipeline(pipeline_config)

    try:
        # Determine URL to extract
        url = article.original_url or article.canonical_url
        if not url:
            raise ValueError("Article has no URL to extract")

        # Run extraction
        logger.info(f"Starting extraction for article {article_id}: {url}")
        result = await pipeline.extract(url)

        # Update article with results
        article.extraction_status = "completed"
        article.title = result.title or article.title  # Keep existing if not found
        article.content_text = result.content
        article.content_hash = _compute_hash(result.content)
        article.author = result.author
        article.published_date = result.published_date
        article.language = result.language
        article.word_count = result.word_count
        article.reading_time_minutes = _estimate_reading_time(result.word_count)
        article.extraction_metadata = result.metadata
        article.extraction_method = result.extraction_method
        article.extracted_at = datetime.utcnow()

        if result.warnings:
            article.extraction_metadata = article.extraction_metadata or {}
            article.extraction_metadata["extraction_warnings"] = result.warnings

        logger.info(
            f"Extraction completed for article {article_id}: "
            f"{result.word_count} words, method={result.extraction_method}"
        )

    except NetworkError as e:
        logger.warning(f"Network error extracting article {article_id}: {e}")
        article.extraction_status = "failed"
        article.extraction_error = f"Network error: {e}"

    except ExtractionError as e:
        logger.warning(f"Extraction error for article {article_id}: {e}")
        article.extraction_status = "failed"
        article.extraction_error = f"Extraction error: {e}"

    except Exception as e:
        logger.exception(f"Unexpected error extracting article {article_id}")
        article.extraction_status = "failed"
        article.extraction_error = f"Unexpected error: {e}"

    finally:
        article.updated_at = datetime.utcnow()
        await db.commit()


def _compute_hash(content: str | None) -> str | None:
    """Compute SHA-256 hash of content for deduplication.

    Args:
        content: Text content to hash

    Returns:
        Hexadecimal hash string or None if content is empty
    """
    if not content:
        return None
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _estimate_reading_time(word_count: int | None, wpm: int = 200) -> int | None:
    """Estimate reading time in minutes.

    Args:
        word_count: Number of words in content
        wpm: Words per minute reading speed (default 200)

    Returns:
        Estimated reading time in minutes (minimum 1)
    """
    if not word_count:
        return None
    return max(1, round(word_count / wpm))
