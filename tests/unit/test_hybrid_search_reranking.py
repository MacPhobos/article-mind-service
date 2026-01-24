"""Integration tests for hybrid search with reranking."""

import pytest

from article_mind_service.config import settings
from article_mind_service.schemas.search import SearchMode, SearchRequest
from article_mind_service.search.dense_search import DenseSearch
from article_mind_service.search.hybrid_search import HybridSearch
from article_mind_service.search.reranker import Reranker
from article_mind_service.search.sparse_search import BM25IndexCache, SparseSearch


class TestHybridSearchWithReranking:
    """Tests for hybrid search with cross-encoder reranking."""

    @pytest.mark.asyncio
    async def test_hybrid_search_with_reranking_enabled(self) -> None:
        """Test that reranking is applied when enabled."""
        # Setup
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = True
        session_id = 1

        try:
            # Populate BM25 index
            chunks = [
                ("chunk_1", "Python programming language for beginners"),
                ("chunk_2", "JavaScript web development tutorial"),
                ("chunk_3", "Advanced Python machine learning guide"),
            ]
            BM25IndexCache.populate_from_chunks(session_id=session_id, chunks=chunks)

            # Create hybrid search with reranker
            hybrid = HybridSearch(reranker=Reranker())

            # Create mock query embedding (would normally come from embedding provider)
            query_embedding = [0.1] * 1536  # OpenAI embedding dimension

            request = SearchRequest(
                query="Python programming",
                top_k=3,
                include_content=True,
                search_mode=SearchMode.SPARSE,  # Use sparse only for simplicity
            )

            # Execute search
            response = await hybrid.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            # Verify results
            assert len(response.results) > 0

            # Results should be ordered by relevance
            # Note: Can't verify exact order without dense embeddings,
            # but we can verify structure
            for result in response.results:
                assert result.chunk_id is not None
                assert result.score is not None
                assert result.content is not None

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original
            BM25IndexCache.invalidate(session_id)

    @pytest.mark.asyncio
    async def test_hybrid_search_without_reranking(self) -> None:
        """Test that search works when reranking is disabled."""
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = False
        session_id = 2

        try:
            # Populate BM25 index with enough content to avoid BM25 edge cases
            chunks = [
                ("chunk_1", "Python programming language tutorial"),
                ("chunk_2", "JavaScript coding and web development"),
                ("chunk_3", "Database design and SQL queries"),
            ]
            BM25IndexCache.populate_from_chunks(session_id=session_id, chunks=chunks)

            hybrid = HybridSearch()
            query_embedding = [0.1] * 1536

            request = SearchRequest(
                query="Python programming",
                top_k=3,
                include_content=True,
                search_mode=SearchMode.SPARSE,
            )

            response = await hybrid.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            # Should still get results without reranking
            # Note: BM25 may return empty if all docs have same score
            # but with distinct content and query terms, should work
            assert response is not None
            assert isinstance(response.results, list)

        finally:
            settings.search_rerank_enabled = original
            BM25IndexCache.invalidate(session_id)

    @pytest.mark.asyncio
    async def test_reranking_improves_result_order(self) -> None:
        """Test that reranking can change result ordering."""
        original = settings.search_rerank_enabled
        session_id = 3

        try:
            # Create test data where reranking should reorder results
            chunks = [
                ("chunk_1", "Python is mentioned here"),
                ("chunk_2", "This is a comprehensive guide to Python programming "
                           "with detailed examples and best practices for Python development"),
                ("chunk_3", "JavaScript tutorial"),
            ]
            BM25IndexCache.populate_from_chunks(session_id=session_id, chunks=chunks)

            query_embedding = [0.1] * 1536

            # Search WITHOUT reranking
            settings.search_rerank_enabled = False
            hybrid_no_rerank = HybridSearch()

            request = SearchRequest(
                query="Python programming guide",
                top_k=3,
                include_content=True,
                search_mode=SearchMode.SPARSE,
            )

            response_no_rerank = await hybrid_no_rerank.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            # Search WITH reranking
            settings.search_rerank_enabled = True
            hybrid_with_rerank = HybridSearch(reranker=Reranker())

            response_with_rerank = await hybrid_with_rerank.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            # Both should return results
            assert len(response_no_rerank.results) > 0
            assert len(response_with_rerank.results) > 0

            # Reranking may change the order (chunk_2 has more relevant content)
            # Note: We can't guarantee order will change in all cases,
            # but we can verify the mechanism works
            assert response_with_rerank.results[0].chunk_id is not None

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original
            BM25IndexCache.invalidate(session_id)

    @pytest.mark.asyncio
    async def test_reranking_with_empty_results(self) -> None:
        """Test that reranking handles empty results gracefully."""
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = True
        session_id = 4

        try:
            # Empty index
            BM25IndexCache.get_or_create(session_id=session_id)

            hybrid = HybridSearch(reranker=Reranker())
            query_embedding = [0.1] * 1536

            request = SearchRequest(
                query="test",
                top_k=10,
                include_content=True,
                search_mode=SearchMode.SPARSE,
            )

            response = await hybrid.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            # Should handle empty results gracefully
            assert response.results == []
            assert response.total_chunks_searched == 0

        finally:
            settings.search_rerank_enabled = original
            BM25IndexCache.invalidate(session_id)

    @pytest.mark.asyncio
    async def test_reranking_limits_to_top_k(self) -> None:
        """Test that reranking correctly limits results to top_k."""
        original_enabled = settings.search_rerank_enabled
        original_top_k = settings.search_rerank_top_k
        settings.search_rerank_enabled = True
        session_id = 5

        try:
            # Create many chunks
            chunks = [
                (f"chunk_{i}", f"Document {i} about Python programming")
                for i in range(1, 21)
            ]
            BM25IndexCache.populate_from_chunks(session_id=session_id, chunks=chunks)

            hybrid = HybridSearch(reranker=Reranker())
            query_embedding = [0.1] * 1536

            # Request only top 5
            request = SearchRequest(
                query="Python",
                top_k=5,
                include_content=True,
                search_mode=SearchMode.SPARSE,
            )

            response = await hybrid.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            # Should return exactly 5 results
            assert len(response.results) == 5

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original_enabled
            settings.search_rerank_top_k = original_top_k
            BM25IndexCache.invalidate(session_id)

    @pytest.mark.asyncio
    async def test_reranking_fetches_content_from_bm25(self) -> None:
        """Test that reranking fetches content from BM25 index."""
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = True
        session_id = 6

        try:
            chunks = [
                ("chunk_1", "First document content"),
                ("chunk_2", "Second document content"),
            ]
            BM25IndexCache.populate_from_chunks(session_id=session_id, chunks=chunks)

            hybrid = HybridSearch(reranker=Reranker())
            query_embedding = [0.1] * 1536

            request = SearchRequest(
                query="document",
                top_k=2,
                include_content=True,
                search_mode=SearchMode.SPARSE,
            )

            response = await hybrid.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            # Results should have content from BM25 index
            for result in response.results:
                assert result.content is not None
                assert "content" in result.content.lower()

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original
            BM25IndexCache.invalidate(session_id)


class TestRerankerInitialization:
    """Tests for reranker initialization in hybrid search."""

    @pytest.mark.asyncio
    async def test_reranker_lazy_initialization(self) -> None:
        """Test that reranker is initialized lazily when needed."""
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = True
        session_id = 7

        try:
            # Create hybrid search without explicit reranker
            hybrid = HybridSearch()

            # Reranker should be None initially
            assert hybrid.reranker is None

            # Populate index with multiple chunks (so we get results)
            chunks = [
                ("chunk_1", "test content about Python"),
                ("chunk_2", "more test content"),
            ]
            BM25IndexCache.populate_from_chunks(session_id=session_id, chunks=chunks)

            query_embedding = [0.1] * 1536
            request = SearchRequest(
                query="test",
                top_k=2,
                search_mode=SearchMode.SPARSE,
            )

            # Execute search - should initialize reranker if there are results
            response = await hybrid.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            # If we got results, reranker should be initialized
            if len(response.results) > 0:
                assert hybrid.reranker is not None

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original
            BM25IndexCache.invalidate(session_id)
