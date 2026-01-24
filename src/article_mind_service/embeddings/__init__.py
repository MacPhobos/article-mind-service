"""Embedding module public exports."""

from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings

from .base import EmbeddingProvider
from .chromadb_store import ChromaDBStore
from .chunker import TextChunker
from .chunking_strategy import (
    ChunkingStrategy,
    ChunkResult,
    FixedSizeChunkingStrategy,
    SemanticChunkingStrategy,
)
from .client import get_chromadb_client
from .exceptions import EmbeddingError
from .migration import ChunkingMigration, migrate_to_semantic
from .ollama_provider import OllamaEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .pipeline import EmbeddingPipeline, get_chunking_strategy
from .semantic_chunker import SemanticChunk, SemanticChunker


async def get_embedding_provider(
    db: AsyncSession | None = None,
    provider_override: str | None = None,
) -> EmbeddingProvider:
    """Factory function to get configured embedding provider.

    Provider Resolution Priority:
    1. provider_override parameter (explicit override)
    2. Database settings (if db session provided)
    3. Environment variable settings (fallback)

    Design Decision: Three-level priority
    - Rationale: Explicit override > persistent settings > defaults
    - Allows testing without database
    - Allows admin panel to override via database
    - Falls back gracefully when database unavailable

    Args:
        db: Database session for reading settings (optional)
        provider_override: Explicit provider to use, bypasses database (optional)

    Returns:
        EmbeddingProvider instance based on priority order.

    Raises:
        ValueError: If provider is not configured correctly.

    Backward Compatibility:
        Works without database session (uses settings from .env)

    Example:
        # Use database settings
        async with get_db() as db:
            provider = await get_embedding_provider(db=db)

        # Use explicit override (for testing)
        provider = await get_embedding_provider(provider_override="ollama")

        # Use .env settings (no database)
        provider = await get_embedding_provider()
    """
    # Determine provider name using priority order
    if provider_override:
        provider_name = provider_override
    elif db:
        # Import here to avoid circular dependency
        from article_mind_service.services.settings_service import get_settings

        db_settings = await get_settings(db)
        provider_name = db_settings.embedding_provider
    else:
        # Fallback to .env settings
        provider_name = settings.embedding_provider

    # Instantiate provider
    if provider_name == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI provider")
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key)

    elif provider_name == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    else:
        raise ValueError(f"Unknown provider: {provider_name}")


async def get_embedding_pipeline(
    db: AsyncSession | None = None,
    provider_override: str | None = None,
) -> EmbeddingPipeline:
    """Factory function to create configured pipeline.

    Args:
        db: Database session for reading settings (optional)
        provider_override: Explicit provider to use (optional)

    Returns:
        EmbeddingPipeline ready for use.

    Example:
        # Use database settings
        async with get_db() as db:
            pipeline = await get_embedding_pipeline(db=db)

        # Use explicit override
        pipeline = await get_embedding_pipeline(provider_override="ollama")

        # Use .env settings (backward compatible)
        pipeline = await get_embedding_pipeline()

        # Use pipeline
        chunk_count = await pipeline.process_article(
            article_id=1,
            session_id="abc123",
            text="Article content...",
            source_url="https://example.com",
            db=db_session
        )
    """
    provider = await get_embedding_provider(db=db, provider_override=provider_override)
    store = ChromaDBStore(
        persist_path=settings.chromadb_path,
        embedding_provider=provider,
    )
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return EmbeddingPipeline(
        provider=provider,
        store=store,
        chunker=chunker,
    )


__all__ = [
    # Core providers and pipeline
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "EmbeddingPipeline",
    "ChromaDBStore",
    "get_chromadb_client",
    "get_embedding_provider",
    "get_embedding_pipeline",
    # Chunking (fixed-size)
    "TextChunker",
    # Chunking (semantic)
    "SemanticChunker",
    "SemanticChunk",
    # Chunking strategies
    "ChunkingStrategy",
    "ChunkResult",
    "FixedSizeChunkingStrategy",
    "SemanticChunkingStrategy",
    "get_chunking_strategy",
    # Migration utilities
    "ChunkingMigration",
    "migrate_to_semantic",
    # Exceptions
    "EmbeddingError",
]
