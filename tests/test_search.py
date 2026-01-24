"""Tests for hybrid search functionality."""

import pytest
from httpx import AsyncClient

from article_mind_service.search import BM25Index, BM25IndexCache
from article_mind_service.search.hybrid_search import reciprocal_rank_fusion

# ============================================================================
# BM25 Index Tests
# ============================================================================


class TestBM25Index:
    """Tests for BM25 index functionality."""

    def test_create_empty_index(self) -> None:
        """Test creating an empty BM25 index."""
        index = BM25Index(session_id=1)
        assert len(index) == 0
        assert index.search("test") == []

    def test_add_document(self) -> None:
        """Test adding a document to the index."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "This is a test document about Python programming.")

        assert len(index) == 1
        assert "chunk_1" in index.chunk_ids

    def test_search_finds_relevant_document(self) -> None:
        """Test that search finds relevant documents."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "Python is a programming language.")
        index.add_document("chunk_2", "JavaScript is used for web development.")
        index.add_document("chunk_3", "Python can be used for machine learning.")
        index.build()

        results = index.search("Python programming", top_k=2)

        assert len(results) > 0
        # Python chunks should rank higher
        chunk_ids = [r[0] for r in results]
        assert "chunk_1" in chunk_ids or "chunk_3" in chunk_ids

    def test_search_exact_match(self) -> None:
        """Test that exact keyword matches score high."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "Authentication using JWT tokens.")
        index.add_document("chunk_2", "User login and session management.")
        index.add_document("chunk_3", "Database connection pooling.")
        index.build()

        results = index.search("JWT authentication", top_k=3)

        # JWT chunk should be first
        assert results[0][0] == "chunk_1"

    def test_remove_document(self) -> None:
        """Test removing a document from the index."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "First document")
        index.add_document("chunk_2", "Second document")

        assert len(index) == 2

        removed = index.remove_document("chunk_1")

        assert removed is True
        assert len(index) == 1
        assert "chunk_1" not in index.chunk_ids

    def test_tokenization(self) -> None:
        """Test tokenization handles various inputs."""
        index = BM25Index(session_id=1)

        # Test basic tokenization
        tokens = index._tokenize("Hello World")
        # Note: With NLTK stemming, tokens may be stemmed
        assert any("hello" in t or "world" in t for t in tokens)

        # Test with special characters
        tokens = index._tokenize("user@email.com has-dashes")
        # Check for base words or their stems
        assert any("user" in t for t in tokens)
        assert any("email" in t for t in tokens)
        assert any("dash" in t for t in tokens)  # May be stemmed

        # Test filtering short tokens
        tokens = index._tokenize("a the is be")
        assert "a" not in tokens  # Single char filtered
        # With NLTK, stopwords like "the", "is" may also be filtered

    def test_get_content(self) -> None:
        """Test retrieving content by chunk ID."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "Test content")

        content = index.get_content("chunk_1")
        assert content == "Test content"

        # Non-existent chunk
        content = index.get_content("chunk_999")
        assert content is None


class TestBM25IndexCache:
    """Tests for BM25 index caching."""

    def test_get_or_create(self) -> None:
        """Test get_or_create returns same instance."""
        index1 = BM25IndexCache.get_or_create(session_id=1)
        index2 = BM25IndexCache.get_or_create(session_id=1)

        assert index1 is index2

    def test_different_sessions_get_different_indexes(self) -> None:
        """Test different sessions have separate indexes."""
        index1 = BM25IndexCache.get_or_create(session_id=1)
        index2 = BM25IndexCache.get_or_create(session_id=2)

        assert index1 is not index2
        assert index1.session_id == 1
        assert index2.session_id == 2

    def test_invalidate_removes_index(self) -> None:
        """Test invalidating a session's index."""
        BM25IndexCache.get_or_create(session_id=1)

        BM25IndexCache.invalidate(session_id=1)

        assert BM25IndexCache.get(session_id=1) is None

    def test_populate_from_chunks(self) -> None:
        """Test populating index from chunk list."""
        chunks = [
            ("chunk_1", "First document content"),
            ("chunk_2", "Second document content"),
        ]

        index = BM25IndexCache.populate_from_chunks(session_id=1, chunks=chunks)

        assert len(index) == 2
        assert index.get_content("chunk_1") == "First document content"


# ============================================================================
# RRF Tests
# ============================================================================


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def test_empty_results(self) -> None:
        """Test RRF with empty inputs."""
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_dense_only(self) -> None:
        """Test RRF with only dense results."""
        dense_results = [
            ("chunk_1", 0.9),
            ("chunk_2", 0.8),
        ]

        results = reciprocal_rank_fusion(dense_results, [])

        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[0].dense_rank == 1
        assert results[0].sparse_rank is None

    def test_sparse_only(self) -> None:
        """Test RRF with only sparse results."""
        sparse_results = [
            ("chunk_1", 5.5),
            ("chunk_2", 4.2),
        ]

        results = reciprocal_rank_fusion([], sparse_results)

        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[0].sparse_rank == 1
        assert results[0].dense_rank is None

    def test_fusion_combines_results(self) -> None:
        """Test RRF properly fuses dense and sparse results."""
        dense_results = [
            ("chunk_1", 0.9),
            ("chunk_2", 0.8),
            ("chunk_3", 0.7),
        ]
        sparse_results = [
            ("chunk_2", 5.5),  # chunk_2 ranks higher in sparse
            ("chunk_1", 4.2),
            ("chunk_4", 3.0),  # unique to sparse
        ]

        results = reciprocal_rank_fusion(dense_results, sparse_results)

        # Should have 4 unique chunks
        assert len(results) == 4

        # chunk_1 and chunk_2 appear in both, should rank high
        top_ids = [r.chunk_id for r in results[:2]]
        assert "chunk_1" in top_ids or "chunk_2" in top_ids

    def test_weights_affect_ranking(self) -> None:
        """Test that weights affect final ranking."""
        dense_results = [("chunk_1", 0.9)]
        sparse_results = [("chunk_2", 5.5)]

        # High dense weight
        results_dense_heavy = reciprocal_rank_fusion(
            dense_results, sparse_results, dense_weight=0.9, sparse_weight=0.1
        )

        # High sparse weight
        results_sparse_heavy = reciprocal_rank_fusion(
            dense_results, sparse_results, dense_weight=0.1, sparse_weight=0.9
        )

        # Dense-heavy should favor chunk_1
        assert results_dense_heavy[0].chunk_id == "chunk_1"
        # Sparse-heavy should favor chunk_2
        assert results_sparse_heavy[0].chunk_id == "chunk_2"


# ============================================================================
# API Tests
# ============================================================================


@pytest.mark.asyncio
class TestSearchAPI:
    """Tests for search API endpoint."""

    async def test_search_empty_session(self, async_client: AsyncClient) -> None:
        """Test search on session with no indexed content."""
        response = await async_client.post(
            "/api/v1/sessions/999/search",
            json={"query": "test query"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total_chunks_searched"] == 0

    async def test_search_request_validation(self, async_client: AsyncClient) -> None:
        """Test search request validation."""
        # Empty query
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": ""},
        )
        assert response.status_code == 422

        # Query too long
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "x" * 2000},
        )
        assert response.status_code == 422

        # Invalid top_k
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422

    async def test_search_sparse_mode(self, async_client: AsyncClient) -> None:
        """Test sparse-only search mode."""
        # First populate the index (need at least 3 documents for BM25 to work)
        BM25IndexCache.populate_from_chunks(
            session_id=1,
            chunks=[
                ("chunk_1", "Python programming language for data science"),
                ("chunk_2", "JavaScript web development frameworks"),
                ("chunk_3", "Python machine learning and AI"),
            ],
        )

        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "Python",
                "search_mode": "sparse",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "sparse"
        assert len(data["results"]) > 0

    async def test_search_response_structure(self, async_client: AsyncClient) -> None:
        """Test search response has correct structure."""
        BM25IndexCache.populate_from_chunks(
            session_id=1,
            chunks=[("chunk_1", "Test content")],
        )

        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "test"},
        )

        data = response.json()

        # Required fields
        assert "query" in data
        assert "results" in data
        assert "total_chunks_searched" in data
        assert "search_mode" in data
        assert "timing_ms" in data

        # Result structure
        if data["results"]:
            result = data["results"][0]
            assert "chunk_id" in result
            assert "score" in result

    async def test_search_stats_endpoint(self, async_client: AsyncClient) -> None:
        """Test search stats endpoint."""
        response = await async_client.get("/api/v1/sessions/1/search/stats")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "bm25_index_exists" in data
        assert "total_chunks" in data


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
class TestSearchIntegration:
    """Integration tests for end-to-end search flow."""

    async def test_full_search_flow(self, async_client: AsyncClient) -> None:
        """Test complete search flow from indexing to results."""
        # Simulate indexed content (would come from R4/R5 in real flow)
        chunks = [
            ("doc_1:chunk_1", "Authentication in web applications uses tokens like JWT."),
            ("doc_1:chunk_2", "JWTs contain encoded claims about the user."),
            ("doc_2:chunk_1", "Database connections should use connection pooling."),
            ("doc_2:chunk_2", "PostgreSQL supports advanced features like JSONB."),
        ]

        BM25IndexCache.populate_from_chunks(session_id=1, chunks=chunks)

        # Search for authentication
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "How does JWT authentication work?",
                "top_k": 3,
                "search_mode": "sparse",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should find JWT-related chunks
        results = data["results"]
        assert len(results) > 0

        # Top results should be about JWT/auth
        top_chunk_ids = [r["chunk_id"] for r in results[:2]]
        assert any("doc_1" in cid for cid in top_chunk_ids)

    async def test_search_relevance_ordering(self, async_client: AsyncClient) -> None:
        """Test that results are ordered by relevance.

        Note: BM25 returns zero scores when term appears in exactly 50% of docs,
        so we use 5 documents to avoid this edge case.
        """
        # Use unique session ID to avoid interference
        session_id = 999
        chunks = [
            ("chunk_1", "Python programming language introduction"),
            ("chunk_2", "Java enterprise application development"),
            ("chunk_3", "Python Python Python - comprehensive Python tutorial"),
            ("chunk_4", "JavaScript web development frameworks"),
            ("chunk_5", "Ruby on Rails web framework"),
        ]

        BM25IndexCache.populate_from_chunks(session_id=session_id, chunks=chunks)

        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/search",
            json={"query": "Python", "search_mode": "sparse"},
        )

        results = response.json()["results"]

        # Chunk with most Python mentions should score highest
        assert len(results) > 0, "Expected search results but got none"
        # chunk_3 has most Python mentions (4 times) so it should rank first
        assert results[0]["chunk_id"] == "chunk_3", f"Expected chunk_3 first, got {results[0]['chunk_id']}"

    async def test_search_include_content_flag(self, async_client: AsyncClient) -> None:
        """Test include_content flag works correctly."""
        chunks = [("chunk_1", "Test content for search")]

        BM25IndexCache.populate_from_chunks(session_id=1, chunks=chunks)

        # With content
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "test", "include_content": True, "search_mode": "sparse"},
        )

        data = response.json()
        if data["results"]:
            assert data["results"][0]["content"] is not None

        # Without content
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "test", "include_content": False, "search_mode": "sparse"},
        )

        data = response.json()
        if data["results"]:
            assert data["results"][0]["content"] is None
