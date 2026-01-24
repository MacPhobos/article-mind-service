"""Unit tests for chunking strategies."""

import pytest

from article_mind_service.embeddings.base import EmbeddingProvider
from article_mind_service.embeddings.chunker import TextChunker
from article_mind_service.embeddings.chunking_strategy import (
    ChunkResult,
    FixedSizeChunkingStrategy,
    SemanticChunkingStrategy,
)
from article_mind_service.embeddings.semantic_chunker import SemanticChunker


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[1.0, 0.0, 0.0] for _ in texts]

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def max_tokens(self) -> int:
        return 8192


class TestFixedSizeChunkingStrategy:
    """Tests for FixedSizeChunkingStrategy."""

    @pytest.fixture
    def text_chunker(self) -> TextChunker:
        """Create text chunker for testing."""
        return TextChunker(chunk_size=100, chunk_overlap=10)

    @pytest.fixture
    def strategy(self, text_chunker: TextChunker) -> FixedSizeChunkingStrategy:
        """Create fixed-size chunking strategy."""
        return FixedSizeChunkingStrategy(text_chunker)

    @pytest.mark.asyncio
    async def test_chunk_empty_text(self, strategy: FixedSizeChunkingStrategy) -> None:
        """Test chunking empty text returns empty list."""
        result = await strategy.chunk("")
        assert result == []

    @pytest.mark.asyncio
    async def test_chunk_returns_chunk_results(
        self, strategy: FixedSizeChunkingStrategy
    ) -> None:
        """Test chunk returns ChunkResult objects."""
        text = "This is a test sentence."
        result = await strategy.chunk(text)

        assert len(result) == 1
        assert isinstance(result[0], ChunkResult)
        assert result[0].text == text
        assert result[0].chunk_index == 0
        assert isinstance(result[0].metadata, dict)

    @pytest.mark.asyncio
    async def test_chunk_with_metadata(self, strategy: FixedSizeChunkingStrategy) -> None:
        """Test chunking includes provided metadata."""
        text = "This is a test sentence."
        metadata = {"article_id": 123, "source_url": "https://example.com"}

        result = await strategy.chunk(text, metadata)

        assert len(result) == 1
        assert result[0].metadata["article_id"] == 123
        assert result[0].metadata["source_url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_chunk_long_text_creates_multiple_chunks(
        self, strategy: FixedSizeChunkingStrategy
    ) -> None:
        """Test long text creates multiple chunks."""
        # Generate text with ~300 tokens (3x chunk_size)
        text = " ".join(["word"] * 300)
        result = await strategy.chunk(text)

        assert len(result) > 1
        # Verify chunk indices are sequential
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    @pytest.mark.asyncio
    async def test_chunk_preserves_text_content(
        self, strategy: FixedSizeChunkingStrategy
    ) -> None:
        """Test all chunks combined contain original text."""
        text = "First sentence. Second sentence. Third sentence."
        result = await strategy.chunk(text)

        # Combine all chunk texts (with overlap, may have duplicates)
        combined = " ".join(c.text for c in result)

        # Original words should appear in combined text
        for word in text.split():
            assert word in combined


class TestSemanticChunkingStrategy:
    """Tests for SemanticChunkingStrategy."""

    @pytest.fixture
    def mock_provider(self) -> MockEmbeddingProvider:
        """Create mock embedding provider."""
        return MockEmbeddingProvider()

    @pytest.fixture
    def semantic_chunker(
        self, mock_provider: MockEmbeddingProvider
    ) -> SemanticChunker:
        """Create semantic chunker for testing."""
        return SemanticChunker(
            embedding_provider=mock_provider,
            breakpoint_percentile=90,
            min_chunk_size=10,
            max_chunk_size=200,
        )

    @pytest.fixture
    def strategy(
        self, semantic_chunker: SemanticChunker
    ) -> SemanticChunkingStrategy:
        """Create semantic chunking strategy."""
        return SemanticChunkingStrategy(semantic_chunker)

    @pytest.mark.asyncio
    async def test_chunk_empty_text(self, strategy: SemanticChunkingStrategy) -> None:
        """Test chunking empty text returns empty list."""
        result = await strategy.chunk("")
        assert result == []

    @pytest.mark.asyncio
    async def test_chunk_returns_chunk_results(
        self, strategy: SemanticChunkingStrategy
    ) -> None:
        """Test chunk returns ChunkResult objects."""
        text = "First sentence. Second sentence. Third sentence."
        result = await strategy.chunk(text)

        assert len(result) > 0
        for chunk in result:
            assert isinstance(chunk, ChunkResult)
            assert isinstance(chunk.text, str)
            assert isinstance(chunk.chunk_index, int)
            assert isinstance(chunk.metadata, dict)

    @pytest.mark.asyncio
    async def test_chunk_with_metadata(self, strategy: SemanticChunkingStrategy) -> None:
        """Test chunking includes provided metadata."""
        text = "First sentence. Second sentence. Third sentence."
        metadata = {"article_id": 123, "source_url": "https://example.com"}

        result = await strategy.chunk(text, metadata)

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["article_id"] == 123
            assert chunk.metadata["source_url"] == "https://example.com"
            assert "chunk_index" in chunk.metadata

    @pytest.mark.asyncio
    async def test_chunk_indices_are_sequential(
        self, strategy: SemanticChunkingStrategy
    ) -> None:
        """Test chunk indices are sequential starting from 0."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = await strategy.chunk(text)

        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    @pytest.mark.asyncio
    async def test_chunk_metadata_includes_chunk_index(
        self, strategy: SemanticChunkingStrategy
    ) -> None:
        """Test chunk metadata includes chunk_index field."""
        text = "First sentence. Second sentence."
        result = await strategy.chunk(text)

        for chunk in result:
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunk_index"] == chunk.chunk_index


class TestChunkResultDataclass:
    """Tests for ChunkResult dataclass."""

    def test_chunk_result_creation(self) -> None:
        """Test ChunkResult can be created with all fields."""
        chunk = ChunkResult(
            text="Sample text",
            chunk_index=0,
            metadata={"article_id": 123},
        )

        assert chunk.text == "Sample text"
        assert chunk.chunk_index == 0
        assert chunk.metadata == {"article_id": 123}

    def test_chunk_result_immutability(self) -> None:
        """Test ChunkResult is a dataclass (mutable by default)."""
        chunk = ChunkResult(
            text="Sample text",
            chunk_index=0,
            metadata={"article_id": 123},
        )

        # Should allow modification (dataclass is mutable by default)
        chunk.metadata["new_key"] = "new_value"
        assert chunk.metadata["new_key"] == "new_value"
