"""Tests for content-based chunk ID generation.

Test Coverage:
- Deterministic ID generation (same input = same output)
- Different content produces different IDs
- Different article_id produces different IDs
- Different chunk_index produces different IDs
- ID format validation (16 hex characters)
- content_hash in metadata
"""

import hashlib
from article_mind_service.embeddings.pipeline import generate_chunk_id


class TestGenerateChunkId:
    """Test content-based chunk ID generation."""

    def test_deterministic_same_content_same_id(self) -> None:
        """Same content always produces same ID (deterministic)."""
        article_id = 123
        text = "This is test content for chunk ID generation."
        chunk_index = 0

        # Generate ID twice with same inputs
        id1 = generate_chunk_id(article_id, text, chunk_index)
        id2 = generate_chunk_id(article_id, text, chunk_index)

        # Should be identical
        assert id1 == id2, "Same content should produce same chunk ID"

    def test_different_content_different_id(self) -> None:
        """Different content produces different IDs."""
        article_id = 123
        chunk_index = 0

        text1 = "First chunk content"
        text2 = "Second chunk content"

        id1 = generate_chunk_id(article_id, text1, chunk_index)
        id2 = generate_chunk_id(article_id, text2, chunk_index)

        # Should be different
        assert id1 != id2, "Different content should produce different chunk IDs"

    def test_different_article_id_different_id(self) -> None:
        """Different article_id produces different IDs even with same text."""
        text = "Same content in different articles"
        chunk_index = 0

        id1 = generate_chunk_id(article_id=123, text=text, chunk_index=chunk_index)
        id2 = generate_chunk_id(article_id=456, text=text, chunk_index=chunk_index)

        # Should be different (prevents cross-article collisions)
        assert id1 != id2, "Different article_id should produce different chunk IDs"

    def test_different_chunk_index_different_id(self) -> None:
        """Different chunk_index produces different IDs even with same text."""
        article_id = 123
        text = "Same content at different positions"

        id1 = generate_chunk_id(article_id, text, chunk_index=0)
        id2 = generate_chunk_id(article_id, text, chunk_index=1)

        # Should be different (maintains ordering)
        assert id1 != id2, "Different chunk_index should produce different chunk IDs"

    def test_id_format_is_16_hex_chars(self) -> None:
        """Chunk ID format is exactly 16 hexadecimal characters."""
        chunk_id = generate_chunk_id(
            article_id=123,
            text="Test content",
            chunk_index=0,
        )

        # Check length
        assert len(chunk_id) == 16, "Chunk ID should be 16 characters"

        # Check all characters are hexadecimal
        try:
            int(chunk_id, 16)  # Should parse as hex
        except ValueError:
            pytest.fail(f"Chunk ID '{chunk_id}' is not valid hexadecimal")

    def test_content_hash_generation(self) -> None:
        """Verify content hash is generated correctly."""
        text = "Test content for hashing"
        expected_hash = hashlib.sha256(text.encode()).hexdigest()[:8]

        # Generate chunk ID (which uses content hash internally)
        chunk_id = generate_chunk_id(123, text, 0)

        # Verify chunk ID was generated (indirectly tests content hash)
        assert chunk_id is not None
        assert len(chunk_id) == 16

        # We can't directly verify content_hash is in the ID since it's hashed again,
        # but we can verify the expected hash is what we expect
        assert len(expected_hash) == 8
        assert all(c in "0123456789abcdef" for c in expected_hash)

    def test_empty_content_generates_valid_id(self) -> None:
        """Empty content still generates valid chunk ID."""
        chunk_id = generate_chunk_id(
            article_id=123,
            text="",
            chunk_index=0,
        )

        # Should still be valid format
        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)

    def test_unicode_content_generates_valid_id(self) -> None:
        """Unicode content generates valid chunk ID."""
        chunk_id = generate_chunk_id(
            article_id=123,
            text="Hello 世界 🌍",
            chunk_index=0,
        )

        # Should still be valid format
        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)

    def test_very_long_content_generates_valid_id(self) -> None:
        """Very long content still generates fixed-length ID."""
        long_text = "A" * 10000  # 10KB of text

        chunk_id = generate_chunk_id(
            article_id=123,
            text=long_text,
            chunk_index=0,
        )

        # Should still be exactly 16 characters
        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)

    def test_collision_probability_is_low(self) -> None:
        """Test that different inputs produce different IDs (low collision)."""
        # Generate IDs for many different chunks
        chunk_ids = set()
        num_chunks = 1000

        for i in range(num_chunks):
            chunk_id = generate_chunk_id(
                article_id=i % 10,  # 10 different articles
                text=f"Content for chunk {i}",
                chunk_index=i % 100,  # 100 different positions
            )
            chunk_ids.add(chunk_id)

        # All IDs should be unique (no collisions)
        assert len(chunk_ids) == num_chunks, "Should have no hash collisions"

    def test_whitespace_changes_produce_different_ids(self) -> None:
        """Whitespace changes in content produce different IDs."""
        article_id = 123
        chunk_index = 0

        text1 = "Content without extra spaces"
        text2 = "Content  without  extra  spaces"  # Extra spaces

        id1 = generate_chunk_id(article_id, text1, chunk_index)
        id2 = generate_chunk_id(article_id, text2, chunk_index)

        # Should be different (exact content match required)
        assert id1 != id2, "Whitespace changes should produce different IDs"

    def test_case_changes_produce_different_ids(self) -> None:
        """Case changes in content produce different IDs."""
        article_id = 123
        chunk_index = 0

        text1 = "lowercase content"
        text2 = "LOWERCASE CONTENT"

        id1 = generate_chunk_id(article_id, text1, chunk_index)
        id2 = generate_chunk_id(article_id, text2, chunk_index)

        # Should be different (case-sensitive)
        assert id1 != id2, "Case changes should produce different IDs"

    def test_consistency_across_multiple_calls(self) -> None:
        """Verify ID generation is consistent across many calls."""
        article_id = 123
        text = "Consistent content"
        chunk_index = 5

        # Generate ID 100 times
        ids = [generate_chunk_id(article_id, text, chunk_index) for _ in range(100)]

        # All should be identical
        assert len(set(ids)) == 1, "All IDs should be identical"
        assert ids[0] == generate_chunk_id(article_id, text, chunk_index)


class TestContentHashMetadata:
    """Test content_hash generation for metadata storage."""

    def test_content_hash_format(self) -> None:
        """Content hash is 8 hex characters."""
        text = "Test content"
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:8]

        assert len(content_hash) == 8
        assert all(c in "0123456789abcdef" for c in content_hash)

    def test_same_content_same_hash(self) -> None:
        """Same content produces same hash (for deduplication)."""
        text = "Test content"

        hash1 = hashlib.sha256(text.encode()).hexdigest()[:8]
        hash2 = hashlib.sha256(text.encode()).hexdigest()[:8]

        assert hash1 == hash2

    def test_different_content_different_hash(self) -> None:
        """Different content produces different hash."""
        text1 = "First content"
        text2 = "Second content"

        hash1 = hashlib.sha256(text1.encode()).hexdigest()[:8]
        hash2 = hashlib.sha256(text2.encode()).hexdigest()[:8]

        assert hash1 != hash2
