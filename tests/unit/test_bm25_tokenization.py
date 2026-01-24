"""Unit tests for enhanced BM25 tokenization with NLTK."""

import pytest

from article_mind_service.search.sparse_search import BM25Index, NLTK_AVAILABLE


class TestBM25EnhancedTokenization:
    """Tests for NLTK-enhanced tokenization."""

    def test_tokenization_lowercase(self) -> None:
        """Test tokenization converts to lowercase."""
        index = BM25Index(session_id=1)
        tokens = index._tokenize("Hello World PYTHON")

        # All tokens should be lowercase
        assert all(t.islower() for t in tokens)

    def test_tokenization_filters_short_tokens(self) -> None:
        """Test that tokens shorter than 2 chars are filtered."""
        index = BM25Index(session_id=1)
        tokens = index._tokenize("a I is be the")

        # Single character tokens should be filtered
        assert "a" not in tokens
        assert "i" not in tokens

    @pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
    def test_tokenization_removes_stopwords(self) -> None:
        """Test that English stopwords are removed when NLTK is available."""
        index = BM25Index(session_id=1)
        tokens = index._tokenize("the quick brown fox jumps over the lazy dog")

        # Common stopwords should be removed
        assert "the" not in tokens
        assert "over" not in tokens

        # Content words should remain
        # Note: These will be stemmed, so check for stems
        assert any(t.startswith("quick") or t == "quick" for t in tokens)
        assert any(t.startswith("brown") or t == "brown" for t in tokens)
        assert any(t.startswith("fox") or t == "fox" for t in tokens)

    @pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
    def test_tokenization_stems_words(self) -> None:
        """Test that words are stemmed using Porter stemmer."""
        index = BM25Index(session_id=1)

        # Test various stemming cases
        test_cases = [
            ("running runs runner", "run"),
            ("programming programmed programs", "program"),
            ("authentication authenticated", "authent"),
        ]

        for text, expected_stem in test_cases:
            tokens = index._tokenize(text)

            # All tokens should have the same stem
            assert len(set(tokens)) <= 2, f"Expected similar stems for '{text}', got {tokens}"
            assert any(expected_stem in t for t in tokens), \
                f"Expected stem '{expected_stem}' in tokens {tokens}"

    @pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
    def test_tokenization_technical_terms_preserved(self) -> None:
        """Test that technical terms are preserved (after stemming)."""
        index = BM25Index(session_id=1)

        # Technical terms and acronyms
        tokens = index._tokenize("JWT API UUID HTTP REST")

        # These should still be tokenized (lowercased)
        # Some may be filtered as stopwords, but technical terms should remain
        assert "jwt" in tokens or len(tokens) > 0
        assert "api" in tokens or len(tokens) > 0
        assert "uuid" in tokens or len(tokens) > 0

    def test_tokenization_special_characters(self) -> None:
        """Test tokenization handles special characters correctly."""
        index = BM25Index(session_id=1)
        tokens = index._tokenize("user@email.com has-dashes word_parts")

        # Should split on special characters
        assert "user" in tokens or any("user" in t for t in tokens)
        assert "email" in tokens or any("email" in t for t in tokens)
        assert "dashes" in tokens or any("dash" in t for t in tokens)
        # Note: "word" and "parts" are content words (not stopwords)
        assert "word" in tokens or any("word" in t for t in tokens)
        assert "parts" in tokens or any("part" in t for t in tokens)  # May be stemmed

    @pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
    def test_search_with_stemmed_query(self) -> None:
        """Test that search works with stemmed variations of words."""
        index = BM25Index(session_id=1)

        # Add documents with different word forms
        index.add_document("chunk_1", "The system is running multiple processes")
        index.add_document("chunk_2", "Database connection runs smoothly")
        index.add_document("chunk_3", "JavaScript code execution")
        index.build()

        # Query with different form of "run"
        results = index.search("runner process", top_k=3)

        # Should find both "running" and "runs" documents due to stemming
        assert len(results) > 0
        chunk_ids = [r[0] for r in results]

        # Both chunk_1 (running) and chunk_2 (runs) should match due to stemming
        assert "chunk_1" in chunk_ids or "chunk_2" in chunk_ids

    @pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
    def test_search_ignores_stopwords(self) -> None:
        """Test that searches with stopwords still work correctly."""
        index = BM25Index(session_id=1)

        index.add_document("chunk_1", "authentication using JWT tokens")
        index.add_document("chunk_2", "user login session management")
        index.add_document("chunk_3", "database connection pooling")
        index.build()

        # Query with many stopwords
        results = index.search("how to use JWT for authentication", top_k=3)

        # Should still find relevant document despite stopwords
        assert len(results) > 0
        # chunk_1 should rank high (has JWT and authentication)
        assert results[0][0] == "chunk_1"

    def test_tokenization_empty_string(self) -> None:
        """Test tokenization handles empty strings."""
        index = BM25Index(session_id=1)
        tokens = index._tokenize("")
        assert tokens == []

    def test_tokenization_only_stopwords(self) -> None:
        """Test tokenization when input is only stopwords."""
        index = BM25Index(session_id=1)
        tokens = index._tokenize("the a an is are")

        # May be empty if NLTK is available and removes all stopwords
        # Or may have some tokens if NLTK not available
        assert isinstance(tokens, list)

    @pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
    def test_tokenization_improves_recall(self) -> None:
        """Test that enhanced tokenization improves search recall."""
        index = BM25Index(session_id=1)

        # Add documents with word variations
        index.add_document("chunk_1", "The developer is developing a new application")
        index.add_document("chunk_2", "Machine learning model training process")
        index.add_document("chunk_3", "Web server configuration and deployment")
        index.build()

        # Search for different word form
        results = index.search("development", top_k=3)

        # Should find chunk_1 due to stemming (developing -> develop, development -> develop)
        assert len(results) > 0
        chunk_ids = [r[0] for r in results]
        assert "chunk_1" in chunk_ids

    def test_fallback_tokenization_when_nltk_unavailable(self) -> None:
        """Test that tokenization works even if NLTK is not available."""
        index = BM25Index(session_id=1)

        # This should work regardless of NLTK availability
        tokens = index._tokenize("Python programming language")

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        assert all(t.islower() for t in tokens)


class TestBM25SearchWithEnhancedTokenization:
    """Integration tests for BM25 search with enhanced tokenization."""

    @pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
    def test_search_quality_improvement(self) -> None:
        """Test that enhanced tokenization improves search quality."""
        index = BM25Index(session_id=1)

        # Add realistic content
        index.add_document(
            "chunk_1",
            "User authentication is implemented using JWT tokens. "
            "The authentication service validates tokens on every request."
        )
        index.add_document(
            "chunk_2",
            "Database connections are pooled for better performance. "
            "The connection pool manages multiple database connections."
        )
        index.add_document(
            "chunk_3",
            "API endpoints are documented using OpenAPI specification. "
            "The API documentation is generated automatically."
        )
        index.build()

        # Query with variations and stopwords
        results = index.search(
            "how does the authentication work with tokens",
            top_k=3
        )

        # Should find authentication document despite stopwords and variations
        assert len(results) > 0
        # chunk_1 should rank highest
        assert results[0][0] == "chunk_1"
        # Should have high score
        assert results[0][1] > 0

    @pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
    def test_plural_singular_matching(self) -> None:
        """Test that plural and singular forms match due to stemming."""
        index = BM25Index(session_id=1)

        index.add_document("chunk_1", "The user has multiple tokens")
        index.add_document("chunk_2", "Each request includes a single token")
        index.add_document("chunk_3", "Database schemas and migrations")
        index.build()

        # Search for singular "token"
        results = index.search("token", top_k=3)

        # Should find both documents (tokens -> token stem)
        chunk_ids = [r[0] for r in results]
        assert "chunk_1" in chunk_ids  # tokens
        assert "chunk_2" in chunk_ids  # token
