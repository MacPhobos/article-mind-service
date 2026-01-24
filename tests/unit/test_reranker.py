"""Unit tests for cross-encoder reranking."""

import pytest

from article_mind_service.config import settings
from article_mind_service.search.reranker import Reranker


class TestReranker:
    """Tests for Reranker class."""

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self) -> None:
        """Test reranking with empty document list."""
        reranker = Reranker()
        scores = await reranker.rerank(query="test", documents=[])
        assert scores == []

    @pytest.mark.asyncio
    async def test_rerank_disabled_returns_uniform_scores(self) -> None:
        """Test that disabled reranking returns uniform scores."""
        # Temporarily disable reranking
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = False

        try:
            reranker = Reranker()
            documents = ["doc1", "doc2", "doc3"]
            scores = await reranker.rerank(query="test", documents=documents)

            assert len(scores) == 3
            assert all(s == 0.5 for s in scores)
        finally:
            settings.search_rerank_enabled = original

    @pytest.mark.asyncio
    async def test_rerank_enabled_loads_model(self) -> None:
        """Test that reranking loads the cross-encoder model when enabled."""
        # Ensure reranking is enabled
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = True

        try:
            reranker = Reranker()
            documents = [
                "Python is a programming language",
                "JavaScript is used for web development",
                "Machine learning with Python",
            ]

            scores = await reranker.rerank(
                query="Python programming",
                documents=documents,
            )

            # Should return scores for all documents
            assert len(scores) == 3

            # Scores should be floats
            assert all(isinstance(s, float) for s in scores)

            # Scores should vary (not all the same dummy value)
            # Cross-encoder should rank Python docs higher
            assert scores != [0.5, 0.5, 0.5]

            # Python-related docs should score higher than JavaScript
            assert scores[0] > scores[1] or scores[2] > scores[1]

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original

    @pytest.mark.asyncio
    async def test_rerank_with_ids(self) -> None:
        """Test reranking with document IDs."""
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = True

        try:
            reranker = Reranker()
            documents = [
                ("chunk_1", "Python programming language"),
                ("chunk_2", "JavaScript web development"),
                ("chunk_3", "Python machine learning"),
            ]

            results = await reranker.rerank_with_ids(
                query="Python",
                documents=documents,
            )

            # Should return tuples of (id, score)
            assert len(results) == 3
            assert all(isinstance(r, tuple) for r in results)
            assert all(len(r) == 2 for r in results)

            # Results should be sorted by score descending
            scores = [r[1] for r in results]
            assert scores == sorted(scores, reverse=True)

            # Python chunks should rank higher
            top_ids = [r[0] for r in results[:2]]
            assert "chunk_1" in top_ids or "chunk_3" in top_ids

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original

    @pytest.mark.asyncio
    async def test_rerank_model_lazy_loading(self) -> None:
        """Test that model is loaded lazily on first use."""
        original = settings.search_rerank_enabled
        settings.search_rerank_enabled = True

        try:
            reranker = Reranker()

            # Model should not be loaded yet
            assert reranker._model is None

            # First rerank call should load model
            await reranker.rerank(query="test", documents=["doc1"])

            # Model should now be loaded
            assert reranker._model is not None

            # Second call should reuse the same model
            model_ref = reranker._model
            await reranker.rerank(query="test2", documents=["doc2"])
            assert reranker._model is model_ref

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original

    @pytest.mark.asyncio
    async def test_rerank_custom_model(self) -> None:
        """Test using a custom cross-encoder model."""
        original_enabled = settings.search_rerank_enabled
        settings.search_rerank_enabled = True

        try:
            # Use a smaller model for faster testing
            reranker = Reranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2")

            documents = ["Python programming", "JavaScript coding"]
            scores = await reranker.rerank(query="Python", documents=documents)

            assert len(scores) == 2
            assert all(isinstance(s, float) for s in scores)

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not installed")
            raise
        finally:
            settings.search_rerank_enabled = original_enabled
