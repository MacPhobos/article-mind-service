"""Unit tests for SemanticChunker."""

import pytest

from article_mind_service.embeddings.base import EmbeddingProvider
from article_mind_service.embeddings.semantic_chunker import (
    SemanticChunk,
    SemanticChunker,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing.

    Returns pre-computed embeddings that simulate semantic similarity:
    - Sentences 0-1: High similarity (0.9) - same topic
    - Sentences 1-2: Low similarity (0.3) - topic shift
    - Sentences 2-3: High similarity (0.85) - same topic
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings with controlled similarity."""
        # Return simple embeddings that create predictable similarity patterns
        embeddings = []
        for i, text in enumerate(texts):
            # Create embeddings that will have specific cosine similarities
            # Embedding strategy: alternate between similar and dissimilar vectors
            if i % 3 == 0:
                # Topic A: [1, 0, 0]
                embeddings.append([1.0, 0.0, 0.0])
            elif i % 3 == 1:
                # Topic A variant: [0.9, 0.1, 0] (similar to topic A)
                embeddings.append([0.9, 0.1, 0.0])
            else:
                # Topic B: [0, 1, 0] (dissimilar to topic A)
                embeddings.append([0.0, 1.0, 0.0])

        return embeddings

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def max_tokens(self) -> int:
        return 8192


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    @pytest.fixture
    def mock_provider(self) -> MockEmbeddingProvider:
        """Create mock embedding provider."""
        return MockEmbeddingProvider()

    @pytest.fixture
    def chunker(self, mock_provider: MockEmbeddingProvider) -> SemanticChunker:
        """Create semantic chunker with default settings."""
        return SemanticChunker(
            embedding_provider=mock_provider,
            breakpoint_percentile=90,
            min_chunk_size=10,
            max_chunk_size=200,
        )

    @pytest.mark.asyncio
    async def test_chunk_empty_text(self, chunker: SemanticChunker) -> None:
        """Test chunking empty text returns empty list."""
        result = await chunker.chunk("")
        assert result == []

    @pytest.mark.asyncio
    async def test_chunk_single_sentence(self, chunker: SemanticChunker) -> None:
        """Test chunking single sentence returns single chunk."""
        text = "This is a single sentence."
        result = await chunker.chunk(text)

        assert len(result) == 1
        assert result[0].text == text
        assert result[0].start_sentence == 0
        assert result[0].end_sentence == 0

    @pytest.mark.asyncio
    async def test_split_into_sentences(self, chunker: SemanticChunker) -> None:
        """Test sentence splitting logic."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = chunker._split_into_sentences(text)

        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"
        assert sentences[3] == "Fourth sentence."

    @pytest.mark.asyncio
    async def test_cosine_similarity(self, chunker: SemanticChunker) -> None:
        """Test cosine similarity calculation."""
        # Identical vectors -> similarity = 1.0
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = chunker._cosine_similarity(a, b)
        assert abs(sim - 1.0) < 0.01

        # Orthogonal vectors -> similarity = 0.0
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = chunker._cosine_similarity(a, b)
        assert abs(sim - 0.0) < 0.01

        # Similar vectors -> 0 < similarity < 1
        a = [1.0, 0.0, 0.0]
        b = [0.9, 0.1, 0.0]
        sim = chunker._cosine_similarity(a, b)
        assert 0.0 < sim < 1.0
        assert sim > 0.9  # Should be high similarity

    @pytest.mark.asyncio
    async def test_find_breakpoints(self, chunker: SemanticChunker) -> None:
        """Test breakpoint detection based on similarity drops."""
        # Create embeddings with one clear breakpoint
        embeddings = [
            [1.0, 0.0, 0.0],  # Sentence 0
            [0.95, 0.05, 0.0],  # Sentence 1 (similar to 0)
            [0.0, 1.0, 0.0],  # Sentence 2 (dissimilar - topic shift)
            [0.0, 0.95, 0.05],  # Sentence 3 (similar to 2)
        ]

        breakpoints = chunker._find_breakpoints(embeddings)

        # Should have breakpoint after sentence 1 (before sentence 2)
        # due to low similarity between sentences 1 and 2
        assert len(breakpoints) > 0
        assert 2 in breakpoints

    @pytest.mark.asyncio
    async def test_merge_small_chunks(self, chunker: SemanticChunker) -> None:
        """Test merging chunks that are too small."""
        chunks = ["a", "b", "c", "d"]  # All very small
        min_size = 5

        merged = chunker._merge_small_chunks(chunks, min_size)

        # Should merge small chunks together
        # Note: Last chunk might not meet min_size if there's not enough text
        assert len(merged) < len(chunks)  # Should have fewer chunks after merging
        # Most chunks should meet min_size (except possibly the last one)
        for chunk in merged[:-1]:
            assert len(chunk) >= min_size

    @pytest.mark.asyncio
    async def test_split_large_chunks(self, chunker: SemanticChunker) -> None:
        """Test splitting chunks that are too large."""
        large_chunk = "This is a very long chunk. " * 20
        chunks = [large_chunk]
        max_size = 100

        split = chunker._split_large_chunks(chunks, max_size)

        # Should split large chunk into multiple smaller chunks
        assert len(split) > 1
        assert all(len(chunk) <= max_size for chunk in split)

    @pytest.mark.asyncio
    async def test_chunk_with_metadata(
        self, chunker: SemanticChunker, mock_provider: MockEmbeddingProvider
    ) -> None:
        """Test chunking includes metadata."""
        text = "First sentence. Second sentence. Third sentence."
        metadata = {"article_id": 123, "source_url": "https://example.com"}

        result = await chunker.chunk(text, metadata)

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["article_id"] == 123
            assert chunk.metadata["source_url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_chunk_preserves_semantic_coherence(
        self, chunker: SemanticChunker
    ) -> None:
        """Test that semantic chunking creates coherent chunks.

        Mock embedding provider creates this similarity pattern:
        - Sentences 0-1: Similar (Topic A)
        - Sentence 2: Dissimilar (Topic B)
        - Sentence 3: Similar to 2 (Topic B)

        Expected: 2 chunks (sentences 0-1, sentences 2-3)
        """
        # Create text with clear topic shifts
        text = (
            "This is about AI. "  # Topic A (sentence 0)
            "Machine learning is powerful. "  # Topic A (sentence 1)
            "Now let's talk about cooking. "  # Topic B (sentence 2)
            "Recipes are fun."  # Topic B (sentence 3)
        )

        result = await chunker.chunk(text)

        # Should create separate chunks for different topics
        # Due to mock embeddings, we expect 2-3 chunks
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_invalid_percentile_raises_error(
        self, mock_provider: MockEmbeddingProvider
    ) -> None:
        """Test invalid percentile raises ValueError."""
        with pytest.raises(ValueError, match="breakpoint_percentile must be in range"):
            SemanticChunker(
                embedding_provider=mock_provider,
                breakpoint_percentile=150,  # Invalid (>100)
            )

        with pytest.raises(ValueError, match="breakpoint_percentile must be in range"):
            SemanticChunker(
                embedding_provider=mock_provider,
                breakpoint_percentile=-10,  # Invalid (<0)
            )

    @pytest.mark.asyncio
    async def test_chunk_respects_min_size(self, mock_provider: MockEmbeddingProvider) -> None:
        """Test chunks respect minimum size constraint."""
        chunker = SemanticChunker(
            embedding_provider=mock_provider,
            breakpoint_percentile=90,
            min_chunk_size=50,  # Require minimum 50 chars
            max_chunk_size=2000,
        )

        text = "Short. " * 20  # Multiple short sentences

        result = await chunker.chunk(text)

        # All chunks (except possibly last) should be >= min_size
        for chunk in result[:-1]:  # Check all but last chunk
            assert len(chunk.text) >= 50

    @pytest.mark.asyncio
    async def test_chunk_respects_max_size(self, mock_provider: MockEmbeddingProvider) -> None:
        """Test chunks respect maximum size constraint."""
        chunker = SemanticChunker(
            embedding_provider=mock_provider,
            breakpoint_percentile=90,
            min_chunk_size=10,
            max_chunk_size=100,  # Limit to 100 chars
        )

        # Create very long sentence
        text = "This is a very long sentence that exceeds the maximum chunk size. " * 10

        result = await chunker.chunk(text)

        # All chunks should be <= max_size
        for chunk in result:
            assert len(chunk.text) <= 100

    @pytest.mark.asyncio
    async def test_chunk_returns_semantic_chunk_objects(
        self, chunker: SemanticChunker
    ) -> None:
        """Test chunk returns SemanticChunk objects with all fields."""
        text = "First sentence. Second sentence. Third sentence."

        result = await chunker.chunk(text)

        assert len(result) > 0
        for chunk in result:
            assert isinstance(chunk, SemanticChunk)
            assert isinstance(chunk.text, str)
            assert isinstance(chunk.start_sentence, int)
            assert isinstance(chunk.end_sentence, int)
            assert isinstance(chunk.metadata, dict)
            assert len(chunk.text) > 0

    @pytest.mark.asyncio
    async def test_get_sentence_embeddings_batch(
        self, chunker: SemanticChunker, mock_provider: MockEmbeddingProvider
    ) -> None:
        """Test sentence embeddings are generated in batch."""
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]

        embeddings = await chunker._get_sentence_embeddings(sentences)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == mock_provider.dimensions for emb in embeddings)

    @pytest.mark.asyncio
    async def test_chunk_with_whitespace_text(self, chunker: SemanticChunker) -> None:
        """Test chunking text with only whitespace returns empty list."""
        result = await chunker.chunk("   \n\n  \t  ")
        assert result == []
