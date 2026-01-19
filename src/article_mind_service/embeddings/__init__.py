"""Embedding module public exports."""

from article_mind_service.config import settings

from .base import EmbeddingProvider
from .chromadb_store import ChromaDBStore
from .chunker import TextChunker
from .exceptions import EmbeddingError
from .ollama_provider import OllamaEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .pipeline import EmbeddingPipeline


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to get configured embedding provider.

    Returns:
        EmbeddingProvider instance based on EMBEDDING_PROVIDER setting.

    Raises:
        ValueError: If provider is not configured correctly.

    Example:
        provider = get_embedding_provider()
        embeddings = await provider.embed(["text1", "text2"])
    """
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI provider")
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key)

    elif settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    else:
        raise ValueError(f"Unknown provider: {settings.embedding_provider}")


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Factory function to create configured pipeline.

    Returns:
        EmbeddingPipeline ready for use.

    Example:
        pipeline = get_embedding_pipeline()
        chunk_count = await pipeline.process_article(
            article_id=1,
            session_id="abc123",
            text="Article content...",
            source_url="https://example.com",
            db=db_session
        )
    """
    provider = get_embedding_provider()
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
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "TextChunker",
    "ChromaDBStore",
    "EmbeddingPipeline",
    "EmbeddingError",
    "get_embedding_provider",
    "get_embedding_pipeline",
]
