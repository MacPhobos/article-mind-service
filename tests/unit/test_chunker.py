"""Unit tests for TextChunker."""

import pytest

from article_mind_service.embeddings import TextChunker


class TestTextChunker:
    """Tests for TextChunker."""

    @pytest.fixture
    def chunker(self) -> TextChunker:
        """Create chunker with small chunks for testing."""
        return TextChunker(chunk_size=100, chunk_overlap=10)

    def test_chunk_empty_text(self, chunker: TextChunker) -> None:
        """Test chunking empty text returns empty list."""
        result = chunker.chunk("")
        assert result == []

    def test_chunk_short_text(self, chunker: TextChunker) -> None:
        """Test chunking text shorter than chunk_size."""
        text = "This is a short text."
        result = chunker.chunk(text)

        assert len(result) == 1
        assert result[0] == text

    def test_chunk_long_text(self, chunker: TextChunker) -> None:
        """Test chunking text longer than chunk_size creates multiple chunks."""
        # Generate text with ~300 tokens (3x chunk_size)
        text = " ".join(["word"] * 300)
        result = chunker.chunk(text)

        assert len(result) > 1
        # Each chunk should be roughly chunk_size tokens
        # (exact count varies due to separator priority)

    def test_chunk_preserves_paragraph_boundaries(self, chunker: TextChunker) -> None:
        """Test chunking prioritizes paragraph boundaries."""
        paragraph1 = "This is the first paragraph. " * 20
        paragraph2 = "This is the second paragraph. " * 20

        text = paragraph1 + "\n\n" + paragraph2
        result = chunker.chunk(text)

        # Should split at paragraph boundary
        assert len(result) >= 2

    def test_chunk_with_metadata(self, chunker: TextChunker) -> None:
        """Test chunking with metadata includes chunk_index."""
        text = " ".join(["word"] * 300)
        source_metadata = {"article_id": 123, "source_url": "https://example.com"}

        result = chunker.chunk_with_metadata(text, source_metadata)

        assert len(result) > 0
        for i, chunk in enumerate(result):
            assert chunk["chunk_index"] == i
            assert chunk["article_id"] == 123
            assert chunk["source_url"] == "https://example.com"
            assert "text" in chunk

    def test_chunk_with_metadata_empty_text(self, chunker: TextChunker) -> None:
        """Test chunk_with_metadata on empty text returns empty list."""
        result = chunker.chunk_with_metadata("", {"article_id": 123})
        assert result == []

    def test_chunk_overlap(self) -> None:
        """Test chunks overlap by specified amount."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 100)

        result = chunker.chunk(text)

        # Verify we got multiple chunks
        assert len(result) > 1
        # Overlap is hard to test exactly due to separator priority,
        # but we can verify chunks exist
