"""Integration tests for chunk deduplication on re-index.

Test Coverage:
- Existing chunks detected and skipped on re-index
- Changed chunks re-embedded on re-index
- New chunks added on re-index
- Chunk IDs remain stable for unchanged content
- Deduplication saves embedding API calls
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.embeddings.pipeline import EmbeddingPipeline, generate_chunk_id
from article_mind_service.embeddings.chromadb_store import ChromaDBStore
from article_mind_service.embeddings.chunker import TextChunker
from article_mind_service.embeddings.base import EmbeddingProvider


@pytest.fixture
def mock_provider() -> EmbeddingProvider:
    """Create mock embedding provider."""
    provider = AsyncMock(spec=EmbeddingProvider)
    provider.dimensions = 1536
    provider.embed = AsyncMock(return_value=[[0.1] * 1536])
    return provider


@pytest.fixture
def mock_store() -> ChromaDBStore:
    """Create mock ChromaDB store."""
    store = MagicMock(spec=ChromaDBStore)

    # Mock collection
    mock_collection = MagicMock()
    store.get_or_create_collection = MagicMock(return_value=mock_collection)

    # Mock add/upsert methods
    store.add_embeddings = MagicMock()
    store.upsert_embeddings = MagicMock()

    # Mock get_existing_chunks (starts with empty collection)
    store.get_existing_chunks = MagicMock(return_value={})

    return store


@pytest.fixture
def chunker() -> TextChunker:
    """Create text chunker."""
    return TextChunker(chunk_size=512, chunk_overlap=50)


@pytest.fixture
def pipeline(
    mock_provider: EmbeddingProvider,
    mock_store: ChromaDBStore,
    chunker: TextChunker,
) -> EmbeddingPipeline:
    """Create embedding pipeline with mocks."""
    return EmbeddingPipeline(
        provider=mock_provider,
        store=mock_store,
        chunker=chunker,
    )


@pytest.fixture
def mock_db() -> AsyncSession:
    """Create mock database session."""
    db = AsyncMock(spec=AsyncSession)
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return db


@pytest.mark.asyncio
class TestChunkDeduplication:
    """Test chunk deduplication on re-index."""

    async def test_first_index_embeds_all_chunks(
        self,
        pipeline: EmbeddingPipeline,
        mock_provider: EmbeddingProvider,
        mock_store: ChromaDBStore,
        mock_db: AsyncSession,
    ) -> None:
        """First index embeds all chunks (no deduplication)."""
        # Mock BM25 index
        with patch("article_mind_service.embeddings.pipeline.BM25IndexCache"):
            await pipeline.process_article(
                article_id=1,
                session_id="1",  # Numeric string for BM25 compatibility
                text="First chunk. Second chunk. Third chunk.",
                source_url="https://example.com",
                db=mock_db,
            )

        # Should have called embed (no chunks to skip)
        assert mock_provider.embed.called

        # Should have called upsert_embeddings
        assert mock_store.upsert_embeddings.called

    async def test_reindex_unchanged_content_skips_embedding(
        self,
        pipeline: EmbeddingPipeline,
        mock_provider: EmbeddingProvider,
        mock_store: ChromaDBStore,
        mock_db: AsyncSession,
    ) -> None:
        """Re-index with unchanged content skips embedding API calls."""
        article_id = 1
        text = "Unchanged content for deduplication test."

        # Simulate existing chunks with matching content_hash
        import hashlib
        chunks = pipeline.chunker.chunk(text)
        existing_chunks = {}

        for i, chunk_text in enumerate(chunks):
            chunk_id = generate_chunk_id(article_id, chunk_text, i)
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:8]
            existing_chunks[chunk_id] = {
                "content_hash": content_hash,
                "article_id": article_id,
                "chunk_index": i,
            }

        # Mock get_existing_chunks to return matching chunks
        mock_store.get_existing_chunks = MagicMock(return_value=existing_chunks)

        # Reset embed call count
        mock_provider.embed.reset_mock()

        # Mock BM25 index
        with patch("article_mind_service.embeddings.pipeline.BM25IndexCache"):
            await pipeline.process_article(
                article_id=article_id,
                session_id="1",
                text=text,
                source_url="https://example.com",
                db=mock_db,
            )

        # Should NOT have called embed (all chunks unchanged)
        assert not mock_provider.embed.called, "Should skip embedding for unchanged chunks"

        # Should NOT have called upsert_embeddings (no new embeddings)
        assert not mock_store.upsert_embeddings.called

    async def test_reindex_changed_content_embeds_new_chunks(
        self,
        pipeline: EmbeddingPipeline,
        mock_provider: EmbeddingProvider,
        mock_store: ChromaDBStore,
        mock_db: AsyncSession,
    ) -> None:
        """Re-index with changed content embeds only changed chunks."""
        article_id = 1
        original_text = "Original chunk one. Original chunk two."
        updated_text = "Updated chunk one. Original chunk two."  # First chunk changed

        # Simulate existing chunks from original text
        import hashlib
        original_chunks = pipeline.chunker.chunk(original_text)
        existing_chunks = {}

        for i, chunk_text in enumerate(original_chunks):
            chunk_id = generate_chunk_id(article_id, chunk_text, i)
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:8]
            existing_chunks[chunk_id] = {
                "content_hash": content_hash,
                "article_id": article_id,
                "chunk_index": i,
            }

        # Mock get_existing_chunks
        mock_store.get_existing_chunks = MagicMock(return_value=existing_chunks)

        # Reset mock counters
        mock_provider.embed.reset_mock()
        mock_store.upsert_embeddings.reset_mock()

        # Mock BM25 index
        with patch("article_mind_service.embeddings.pipeline.BM25IndexCache"):
            await pipeline.process_article(
                article_id=article_id,
                session_id="1",
                text=updated_text,
                source_url="https://example.com",
                db=mock_db,
            )

        # Should have called embed (for changed chunk)
        assert mock_provider.embed.called, "Should embed changed chunks"

        # Should have called upsert_embeddings (for changed chunk)
        assert mock_store.upsert_embeddings.called

    async def test_chunk_id_stability_for_unchanged_content(
        self,
        pipeline: EmbeddingPipeline,
        mock_db: AsyncSession,
    ) -> None:
        """Chunk IDs remain stable for unchanged content across re-index."""
        article_id = 1
        text = "Stable content."

        # Generate chunk IDs from first index
        chunks_first = pipeline.chunker.chunk(text)
        ids_first = [
            generate_chunk_id(article_id, chunk, i)
            for i, chunk in enumerate(chunks_first)
        ]

        # Generate chunk IDs from second index (same content)
        chunks_second = pipeline.chunker.chunk(text)
        ids_second = [
            generate_chunk_id(article_id, chunk, i)
            for i, chunk in enumerate(chunks_second)
        ]

        # IDs should be identical (deterministic)
        assert ids_first == ids_second, "Chunk IDs should be stable for unchanged content"

    async def test_new_chunks_added_on_reindex(
        self,
        pipeline: EmbeddingPipeline,
        mock_provider: EmbeddingProvider,
        mock_store: ChromaDBStore,
        mock_db: AsyncSession,
    ) -> None:
        """Re-index with additional content embeds new chunks."""
        article_id = 1
        original_text = "Original content."
        expanded_text = "Original content. New additional content."

        # Simulate existing chunks from original text
        import hashlib
        original_chunks = pipeline.chunker.chunk(original_text)
        existing_chunks = {}

        for i, chunk_text in enumerate(original_chunks):
            chunk_id = generate_chunk_id(article_id, chunk_text, i)
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:8]
            existing_chunks[chunk_id] = {
                "content_hash": content_hash,
                "article_id": article_id,
                "chunk_index": i,
            }

        # Mock get_existing_chunks
        mock_store.get_existing_chunks = MagicMock(return_value=existing_chunks)

        # Reset mock counters
        mock_provider.embed.reset_mock()
        mock_store.upsert_embeddings.reset_mock()

        # Mock BM25 index
        with patch("article_mind_service.embeddings.pipeline.BM25IndexCache"):
            await pipeline.process_article(
                article_id=article_id,
                session_id="1",
                text=expanded_text,
                source_url="https://example.com",
                db=mock_db,
            )

        # Should have called embed (for new chunks)
        assert mock_provider.embed.called, "Should embed new chunks"

    async def test_content_hash_in_metadata(
        self,
        pipeline: EmbeddingPipeline,
        mock_store: ChromaDBStore,
        mock_db: AsyncSession,
    ) -> None:
        """Content hash is stored in chunk metadata."""
        # Mock BM25 index
        with patch("article_mind_service.embeddings.pipeline.BM25IndexCache"):
            await pipeline.process_article(
                article_id=1,
                session_id="1",
                text="Test content for metadata.",
                source_url="https://example.com",
                db=mock_db,
            )

        # Verify upsert_embeddings was called
        assert mock_store.upsert_embeddings.called

        # Get the metadatas argument
        call_args = mock_store.upsert_embeddings.call_args
        metadatas = call_args.kwargs["metadatas"]

        # Verify content_hash is in metadata
        for metadata in metadatas:
            assert "content_hash" in metadata, "content_hash should be in metadata"
            assert len(metadata["content_hash"]) == 8, "content_hash should be 8 hex chars"


@pytest.mark.asyncio
class TestDeduplicationPerformance:
    """Test deduplication performance benefits."""

    async def test_deduplication_reduces_embedding_calls(
        self,
        pipeline: EmbeddingPipeline,
        mock_provider: EmbeddingProvider,
        mock_store: ChromaDBStore,
        mock_db: AsyncSession,
    ) -> None:
        """Deduplication significantly reduces embedding API calls."""
        article_id = 1
        # Create text with multiple chunks
        text = ". ".join([f"Chunk {i}" for i in range(10)])

        # First index: all chunks embedded
        with patch("article_mind_service.embeddings.pipeline.BM25IndexCache"):
            await pipeline.process_article(
                article_id=article_id,
                session_id="1",
                text=text,
                source_url="https://example.com",
                db=mock_db,
            )

        first_embed_calls = mock_provider.embed.call_count

        # Simulate all chunks exist with correct hashes
        import hashlib
        chunks = pipeline.chunker.chunk(text)
        existing_chunks = {}

        for i, chunk_text in enumerate(chunks):
            chunk_id = generate_chunk_id(article_id, chunk_text, i)
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:8]
            existing_chunks[chunk_id] = {
                "content_hash": content_hash,
                "article_id": article_id,
                "chunk_index": i,
            }

        mock_store.get_existing_chunks = MagicMock(return_value=existing_chunks)
        mock_provider.embed.reset_mock()

        # Re-index: no chunks should be embedded
        with patch("article_mind_service.embeddings.pipeline.BM25IndexCache"):
            await pipeline.process_article(
                article_id=article_id,
                session_id="1",
                text=text,
                source_url="https://example.com",
                db=mock_db,
            )

        second_embed_calls = mock_provider.embed.call_count

        # Verify significant reduction
        assert first_embed_calls > 0, "First index should embed chunks"
        assert second_embed_calls == 0, "Re-index should skip all unchanged chunks"
