"""Test RAG pipeline metadata capture and persistence."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from article_mind_service.chat.rag_pipeline import RAGPipeline, RAGResponse
from article_mind_service.chat.llm_providers import LLMResponse


@pytest.mark.asyncio
async def test_rag_pipeline_captures_retrieval_metadata():
    """Test that RAG pipeline captures retrieval metadata."""
    # Mock LLM provider
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Test answer [1]",
            provider="openai",
            model="gpt-4o-mini",
            tokens_input=50,
            tokens_output=20,
        )
    )

    # Mock search client
    mock_search = AsyncMock()
    mock_search.search = AsyncMock(
        return_value={
            "results": [
                {
                    "content": "Test chunk content",
                    "article_id": 1,
                    "chunk_id": "chunk_1",
                    "article": {"title": "Test Article", "url": "https://example.com"},
                }
            ]
        }
    )

    # Create pipeline and inject mocks
    pipeline = RAGPipeline()
    pipeline._llm_provider = mock_llm

    # Execute query
    db_mock = AsyncMock()
    response = await pipeline.query(
        session_id=1,
        question="Test question?",
        db=db_mock,
        search_client=mock_search,
    )

    # Verify retrieval_metadata is captured
    assert isinstance(response, RAGResponse)
    assert response.retrieval_metadata is not None
    assert "search_mode" in response.retrieval_metadata
    assert response.retrieval_metadata["search_mode"] == "hybrid"
    assert "chunks_retrieved" in response.retrieval_metadata
    assert response.retrieval_metadata["chunks_retrieved"] == 1
    assert "chunks_cited" in response.retrieval_metadata
    assert "search_timing_ms" in response.retrieval_metadata
    assert response.retrieval_metadata["search_timing_ms"] >= 0  # Can be 0 for very fast operations


@pytest.mark.asyncio
async def test_rag_pipeline_captures_context_chunks():
    """Test that RAG pipeline captures full context chunks."""
    # Mock LLM provider
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Test answer [1]",
            provider="openai",
            model="gpt-4o-mini",
            tokens_input=50,
            tokens_output=20,
        )
    )

    # Mock search client with multiple chunks
    mock_search = AsyncMock()
    mock_search.search = AsyncMock(
        return_value={
            "results": [
                {
                    "content": "First chunk content",
                    "article_id": 1,
                    "chunk_id": "chunk_1",
                    "article": {"title": "Article 1", "url": "https://example.com/1"},
                },
                {
                    "content": "Second chunk content",
                    "article_id": 2,
                    "chunk_id": "chunk_2",
                    "article": {"title": "Article 2", "url": "https://example.com/2"},
                },
            ]
        }
    )

    # Create pipeline and inject mocks
    pipeline = RAGPipeline()
    pipeline._llm_provider = mock_llm

    # Execute query
    db_mock = AsyncMock()
    response = await pipeline.query(
        session_id=1,
        question="Test question?",
        db=db_mock,
        search_client=mock_search,
    )

    # Verify context_chunks is captured
    assert response.context_chunks is not None
    assert len(response.context_chunks) == 2

    # Verify chunk structure
    chunk = response.context_chunks[0]
    assert "chunk_id" in chunk
    assert "article_id" in chunk
    assert "content" in chunk
    assert "cited" in chunk
    assert chunk["chunk_id"] == "chunk_1"
    assert chunk["content"] == "First chunk content"


@pytest.mark.asyncio
async def test_rag_pipeline_marks_cited_chunks():
    """Test that RAG pipeline correctly marks which chunks were cited."""
    # Mock LLM provider that cites first chunk
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Test answer based on first chunk [1]",
            provider="openai",
            model="gpt-4o-mini",
            tokens_input=50,
            tokens_output=20,
        )
    )

    # Mock search client with multiple chunks
    mock_search = AsyncMock()
    mock_search.search = AsyncMock(
        return_value={
            "results": [
                {
                    "content": "First chunk content",
                    "article_id": 1,
                    "chunk_id": "chunk_1",
                    "article": {"title": "Article 1", "url": "https://example.com/1"},
                },
                {
                    "content": "Second chunk content not cited",
                    "article_id": 2,
                    "chunk_id": "chunk_2",
                    "article": {"title": "Article 2", "url": "https://example.com/2"},
                },
            ]
        }
    )

    # Create pipeline and inject mocks
    pipeline = RAGPipeline()
    pipeline._llm_provider = mock_llm

    # Execute query
    db_mock = AsyncMock()
    response = await pipeline.query(
        session_id=1,
        question="Test question?",
        db=db_mock,
        search_client=mock_search,
    )

    # Verify cited flags
    assert response.context_chunks is not None
    assert len(response.context_chunks) == 2

    # First chunk should be cited
    assert response.context_chunks[0]["cited"] is True

    # Second chunk should not be cited
    assert response.context_chunks[1]["cited"] is False
